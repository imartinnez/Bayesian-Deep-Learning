from pathlib import Path
import sys
import argparse
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.download import load_config
from src.data.classification_dataset import ClassificationWindowDataset
from src.models.Bayesian_LSTM_classifier import BayesianLSTMClassifier
from src.models.LSTM_classifier import LSTMClassifier
from src.training.Bayesian_LSTM_classifier_trainer import predict as predict_bayesian
from src.training.LSTM_classifier_trainer import predict as predict_baseline
from src.evaluation.metrics_classification import compute_classification_metrics, compute_uncertainty_metrics
from src.evaluation.calibration_classification import (
    calibrate_temperature_classification,
    apply_temperature,
    compute_ece,
    compute_reliability_data,
    plot_reliability_diagram,
)
from src.evaluation.reporting import (
    print_header,
    print_kv,
    print_artifacts,
    print_metric_block,
)


CLASSIFICATION_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "brier_score",
    "precision_LOW",
    "recall_LOW",
    "f1_LOW",
    "precision_MED",
    "recall_MED",
    "f1_MED",
    "precision_HIGH",
    "recall_HIGH",
    "f1_HIGH",
    "roc_auc_ovr",
]

UNCERTAINTY_METRIC_KEYS = [
    "H_total_mean",
    "H_total_std",
    "H_total_cv",
    "corr_H_error",
    "auroc_H_as_error_det",
    "acc_low_uncertainty_q1",
    "acc_high_uncertainty_q4",
]

SANITY_THRESHOLDS = {
    "H_total_cv": 0.20,
    "corr_H_error": 0.25,
    "auroc_H_as_error_det": 0.60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and Bayesian LSTM classifiers.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file, relative to project root or absolute.",
    )
    return parser.parse_args()


def make_loader(dataset: ClassificationWindowDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_bayesian_clf_model(checkpoint_path: Path, device: torch.device) -> tuple[BayesianLSTMClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BayesianLSTMClassifier(
        n_features=checkpoint["n_features"],
        hidden=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dense=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
        n_classes=checkpoint["n_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_baseline_clf_model(checkpoint_path: Path, device: torch.device) -> tuple[LSTMClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = LSTMClassifier(
        n_features=checkpoint["n_features"],
        hidden=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dense=checkpoint["dense_size"],
        n_classes=checkpoint["n_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_sanity_report(unc_val: dict) -> tuple[dict, bool]:
    report = {}
    all_pass = True

    for key, threshold in SANITY_THRESHOLDS.items():
        value = float(unc_val.get(key, float("nan")))
        passed = not np.isnan(value) and value >= threshold

        report[key] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }

        if not passed:
            all_pass = False

    return report, all_pass


def print_sanity_block(title: str, sanity_report: dict, passed: bool) -> None:
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Check':<28}{'Value':>12}{'Threshold':>14}{'Status':>10}")
    for key in ("H_total_cv", "corr_H_error", "auroc_H_as_error_det"):
        row = sanity_report[key]
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{key:<28}"
            f"{row['value']:>12.4f}"
            f"{row['threshold']:>14.4f}"
            f"{status:>10}"
        )
    print(f"\nOverall status: {'PASS' if passed else 'FAIL'}")


def print_confusion_matrix(title: str, matrix: list[list[int]]) -> None:
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'':<12}{'Pred LOW':>10}{'Pred MED':>10}{'Pred HIGH':>12}")
    labels = ("True LOW", "True MED", "True HIGH")
    for label, row in zip(labels, matrix):
        print(f"{label:<12}{row[0]:>10d}{row[1]:>10d}{row[2]:>12d}")


def _make_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


if __name__ == "__main__":
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = ROOT_DIR / cfg["paths"]["splits"]
    models_dir = ROOT_DIR / cfg["paths"]["clf_models"]
    figures_dir = ROOT_DIR / cfg["paths"]["clf_figures"]
    results_dir = ROOT_DIR / cfg["paths"]["clf_results"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg["classifier_training"]["batch_size"]
    num_workers = cfg["classifier_training"]["num_workers"]
    mc_samples = cfg["classifier_inference"]["mc_samples"]
    n_classes = cfg["classification"]["n_classes"]

    val_dataset = ClassificationWindowDataset.from_npz(
        splits_dir / cfg["paths"]["val_clf_windows_filename"]
    )
    test_dataset = ClassificationWindowDataset.from_npz(
        splits_dir / cfg["paths"]["test_clf_windows_filename"]
    )

    val_loader = make_loader(val_dataset, batch_size, num_workers)
    test_loader = make_loader(test_dataset, batch_size, num_workers)

    bayesian_checkpoint_path = models_dir / cfg["paths"]["bayesian_clf_checkpoint"]
    baseline_checkpoint_path = models_dir / cfg["paths"]["baseline_clf_checkpoint"]

    bayesian_reliability_path = figures_dir / cfg["paths"].get(
        "bayesian_clf_reliability_diagram_filename",
        "bayesian_clf_reliability_diagram.png",
    )
    baseline_reliability_path = figures_dir / cfg["paths"].get(
        "baseline_clf_reliability_diagram_filename",
        "baseline_clf_reliability_diagram.png",
    )
    evaluation_results_path = results_dir / cfg["paths"].get(
        "evaluation_results_classifier_filename",
        "evaluation_results_classifier.json",
    )

    bayes_model, bayes_ckpt = load_bayesian_clf_model(bayesian_checkpoint_path, device)

    bayes_val = predict_bayesian(bayes_model, val_loader, device, T=mc_samples)
    bayes_test = predict_bayesian(bayes_model, test_loader, device, T=mc_samples)

    unc_val = compute_uncertainty_metrics(
        bayes_val["y_true"],
        bayes_val["y_pred"],
        bayes_val["total_entropy"],
    )
    sanity_report, sanity_ok = build_sanity_report(unc_val)

    tau_bayes = calibrate_temperature_classification(
        bayes_val["y_true"],
        bayes_val["p_mean"],
    )
    bayes_p_cal = apply_temperature(bayes_test["p_mean"], tau_bayes)
    bayes_pred_cal = np.argmax(bayes_p_cal, axis=-1)

    bayes_metrics = compute_classification_metrics(
        bayes_test["y_true"],
        bayes_pred_cal,
        bayes_p_cal,
        n_classes,
    )
    bayes_unc = compute_uncertainty_metrics(
        bayes_test["y_true"],
        bayes_pred_cal,
        bayes_test["total_entropy"],
    )
    bayes_ece = compute_ece(
        bayes_test["y_true"],
        bayes_pred_cal,
        bayes_p_cal,
    )

    plot_reliability_diagram(
        compute_reliability_data(bayes_test["y_true"], bayes_pred_cal, bayes_p_cal),
        ece=bayes_ece,
        title="Bayesian classifier reliability diagram",
        save_path=bayesian_reliability_path,
    )

    baseline_model, baseline_ckpt = load_baseline_clf_model(baseline_checkpoint_path, device)

    baseline_val = predict_baseline(baseline_model, val_loader, device)
    baseline_test = predict_baseline(baseline_model, test_loader, device)

    tau_baseline = calibrate_temperature_classification(
        baseline_val["y_true"],
        baseline_val["p_mean"],
    )
    baseline_p_cal = apply_temperature(baseline_test["p_mean"], tau_baseline)
    baseline_pred_cal = np.argmax(baseline_p_cal, axis=-1)

    baseline_metrics = compute_classification_metrics(
        baseline_test["y_true"],
        baseline_pred_cal,
        baseline_p_cal,
        n_classes,
    )
    baseline_ece = compute_ece(
        baseline_test["y_true"],
        baseline_pred_cal,
        baseline_p_cal,
    )

    plot_reliability_diagram(
        compute_reliability_data(baseline_test["y_true"], baseline_pred_cal, baseline_p_cal),
        ece=baseline_ece,
        title="Baseline classifier reliability diagram",
        save_path=baseline_reliability_path,
    )

    test_counts = dict(zip(*np.unique(test_dataset.y_label, return_counts=True)))

    results = {
        "device": str(device),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "test_distribution": {
            "LOW": int(test_counts.get(0, 0)),
            "MED": int(test_counts.get(1, 0)),
            "HIGH": int(test_counts.get(2, 0)),
        },
        "sanity_checks_passed": sanity_ok,
        "sanity_report": _make_serializable(sanity_report),
        "bayesian": {
            "best_epoch": int(bayes_ckpt["best_epoch"]),
            "best_val_ce": float(bayes_ckpt["best_val_loss"]),
            "temperature_tau": float(tau_bayes),
            "ece": float(bayes_ece),
            "checkpoint_path": str(bayesian_checkpoint_path),
            "reliability_diagram_path": str(bayesian_reliability_path),
            "global_metrics": _make_serializable(bayes_metrics),
            "uncertainty_metrics": _make_serializable(bayes_unc),
        },
        "baseline": {
            "best_epoch": int(baseline_ckpt["best_epoch"]),
            "best_val_ce": float(baseline_ckpt["best_val_loss"]),
            "temperature_tau": float(tau_baseline),
            "ece": float(baseline_ece),
            "checkpoint_path": str(baseline_checkpoint_path),
            "reliability_diagram_path": str(baseline_reliability_path),
            "global_metrics": _make_serializable(baseline_metrics),
        },
    }

    with open(evaluation_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print_header("RUN")
    print_kv("Script", "Classification evaluation")
    print_kv("Device", str(device))
    print_kv("Validation samples", len(val_dataset))
    print_kv("Test samples", len(test_dataset))
    print_kv(
        "Test distribution",
        f"LOW={test_counts.get(0, 0)} | MED={test_counts.get(1, 0)} | HIGH={test_counts.get(2, 0)}",
    )

    print_header("MODEL SETUP")
    print_kv("Bayesian checkpoint", str(bayesian_checkpoint_path))
    print_kv("Baseline checkpoint", str(baseline_checkpoint_path))
    print_kv("Bayesian tau", f"{tau_bayes:.4f}")
    print_kv("Baseline tau", f"{tau_baseline:.4f}")
    print_kv("MC samples", mc_samples)
    print_kv("Classes", n_classes)

    print_header("MAIN RESULTS")
    print_metric_block("Bayesian | classification metrics", bayes_metrics, CLASSIFICATION_METRIC_KEYS)
    print_kv("Bayesian ECE", f"{bayes_ece:.4f}")
    print_metric_block("Baseline | classification metrics", baseline_metrics, CLASSIFICATION_METRIC_KEYS)
    print_kv("Baseline ECE", f"{baseline_ece:.4f}")

    print_header("DIAGNOSTICS")
    print_sanity_block("Validation gate", sanity_report, sanity_ok)
    print_metric_block("Bayesian | uncertainty metrics", bayes_unc, UNCERTAINTY_METRIC_KEYS)
    print_confusion_matrix("Bayesian | confusion matrix", bayes_metrics["confusion_matrix"])
    print_confusion_matrix("Baseline | confusion matrix", baseline_metrics["confusion_matrix"])

    print_header("ARTIFACTS")
    print_artifacts({
        "Bayesian reliability diagram": str(bayesian_reliability_path),
        "Baseline reliability diagram": str(baseline_reliability_path),
        "Results JSON": str(evaluation_results_path),
    })