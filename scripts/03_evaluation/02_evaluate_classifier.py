from pathlib import Path
import sys
import json
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


SANITY_THRESHOLDS = {
    "H_total_cv":           0.20,
    "corr_H_error":         0.25,
    "auroc_H_as_error_det": 0.60,
}


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_bayesian_clf_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BayesianLSTMClassifier(
        n_features=ckpt["n_features"],
        hidden=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dense=ckpt["dense_size"],
        dropout=ckpt["dropout"],
        n_classes=ckpt["n_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_baseline_clf_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = LSTMClassifier(
        n_features=ckpt["n_features"],
        hidden=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dense=ckpt["dense_size"],
        n_classes=ckpt["n_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def run_sanity_checks(unc_val: dict) -> bool:
    print("\n--- Cortafuegos: sanity checks sobre val (Bayesiano) ---")
    passed = True
    for key, threshold in SANITY_THRESHOLDS.items():
        value = unc_val.get(key, float("nan"))
        ok = not np.isnan(value) and value >= threshold
        print(f"  {key}: {value:.4f} >= {threshold} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            passed = False
    if not passed:
        print("  ADVERTENCIA: hay checks fallidos. Revisar antes de confiar en resultados de test.")
    else:
        print("  Todos los checks superados. Procediendo a test.")
    return passed


def _make_serializable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.integer):   out[k] = int(v)
        elif isinstance(v, np.floating): out[k] = float(v)
        elif isinstance(v, np.ndarray):  out[k] = v.tolist()
        else:                            out[k] = v
    return out


def _print_metrics(title: str, metrics: dict, unc: dict | None = None, ece: float | None = None) -> None:
    print(f"\n{title}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}  <- metrica principal")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Log-loss:          {metrics['log_loss']:.4f}")
    print(f"  Brier score:       {metrics['brier_score']:.4f}")
    if ece is not None:
        print(f"  ECE:               {ece:.4f}  (umbral < 0.08)")
    print(f"  Recall LOW:        {metrics['recall_LOW']:.4f}")
    print(f"  Recall MED:        {metrics['recall_MED']:.4f}")
    print(f"  Recall HIGH:       {metrics['recall_HIGH']:.4f}  <- mas importante semaforo")
    print(f"  ROC-AUC ovr:       {metrics['roc_auc_ovr']:.4f}")
    print(f"  Confusion matrix (filas=real, cols=pred):")
    for i, row in enumerate(metrics["confusion_matrix"]):
        label = {0: "LOW ", 1: "MED ", 2: "HIGH"}[i]
        print(f"    {label}: {row}")
    if unc is not None:
        print(f"  H_total CV:        {unc['H_total_cv']:.4f}  (umbral > 0.20)")
        print(f"  corr(H, error):    {unc['corr_H_error']:.4f}  (umbral > 0.25)")
        print(f"  AUROC H->error:    {unc['auroc_H_as_error_det']:.4f}  (umbral > 0.60)")
        print(f"  Acc Q1 (baja inc): {unc['acc_low_uncertainty_q1']:.4f}")
        print(f"  Acc Q4 (alta inc): {unc['acc_high_uncertainty_q4']:.4f}")


if __name__ == "__main__":
    cfg         = load_config(ROOT_DIR / "config" / "config.yaml")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir  = ROOT_DIR / cfg["paths"]["splits"]
    models_dir  = ROOT_DIR / cfg["paths"]["clf_models"]
    figures_dir = ROOT_DIR / cfg["paths"]["clf_figures"]
    results_dir = ROOT_DIR / cfg["paths"]["clf_results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    batch_size  = cfg["classifier_training"]["batch_size"]
    num_workers = cfg["classifier_training"]["num_workers"]
    mc_samples  = cfg["classifier_inference"]["mc_samples"]
    n_classes   = cfg["classification"]["n_classes"]

    val_dataset  = ClassificationWindowDataset.from_npz(splits_dir / cfg["paths"]["val_clf_windows_filename"])
    test_dataset = ClassificationWindowDataset.from_npz(splits_dir / cfg["paths"]["test_clf_windows_filename"])
    val_loader   = make_loader(val_dataset,  batch_size, num_workers)
    test_loader  = make_loader(test_dataset, batch_size, num_workers)

    # ── BAYESIAN ─────────────────────────────────────────────────────────────────
    bayes_model, bayes_ckpt = load_bayesian_clf_model(
        models_dir / cfg["paths"]["bayesian_clf_checkpoint"], device
    )

    bayes_val  = predict_bayesian(bayes_model, val_loader,  device, T=mc_samples)
    bayes_test = predict_bayesian(bayes_model, test_loader, device, T=mc_samples)

    unc_val    = compute_uncertainty_metrics(bayes_val["y_true"], bayes_val["y_pred"], bayes_val["total_entropy"])
    sanity_ok  = run_sanity_checks(unc_val)

    tau_bayes  = calibrate_temperature_classification(bayes_val["y_true"], bayes_val["p_mean"])
    print(f"\nBayesian tau: {tau_bayes:.4f}")
    bayes_p_cal   = apply_temperature(bayes_test["p_mean"], tau_bayes)
    bayes_pred_cal = np.argmax(bayes_p_cal, axis=-1)

    bayes_metrics = compute_classification_metrics(bayes_test["y_true"], bayes_pred_cal, bayes_p_cal, n_classes)
    bayes_unc     = compute_uncertainty_metrics(bayes_test["y_true"], bayes_pred_cal, bayes_test["total_entropy"])
    bayes_ece     = compute_ece(bayes_test["y_true"], bayes_pred_cal, bayes_p_cal)

    plot_reliability_diagram(
        compute_reliability_data(bayes_test["y_true"], bayes_pred_cal, bayes_p_cal),
        ece=bayes_ece,
        title="Bayesian classifier — Reliability diagram",
        save_path=figures_dir / "bayesian_clf_reliability_diagram.png",
    )

    # ── BASELINE ─────────────────────────────────────────────────────────────────
    base_model, base_ckpt = load_baseline_clf_model(
        models_dir / cfg["paths"]["baseline_clf_checkpoint"], device
    )

    base_val  = predict_baseline(base_model, val_loader,  device)
    base_test = predict_baseline(base_model, test_loader, device)

    tau_base  = calibrate_temperature_classification(base_val["y_true"], base_val["p_mean"])
    print(f"Baseline tau:  {tau_base:.4f}")
    base_p_cal    = apply_temperature(base_test["p_mean"], tau_base)
    base_pred_cal = np.argmax(base_p_cal, axis=-1)

    base_metrics  = compute_classification_metrics(base_test["y_true"], base_pred_cal, base_p_cal, n_classes)
    base_ece      = compute_ece(base_test["y_true"], base_pred_cal, base_p_cal)

    plot_reliability_diagram(
        compute_reliability_data(base_test["y_true"], base_pred_cal, base_p_cal),
        ece=base_ece,
        title="Baseline classifier — Reliability diagram",
        save_path=figures_dir / "baseline_clf_reliability_diagram.png",
    )

    # ── PRINT + SAVE ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Device: {device} | Test samples: {len(test_dataset)}")
    counts = dict(zip(*np.unique(test_dataset.y_label, return_counts=True)))
    print(f"Distribucion test: LOW={counts.get(0,0)} MED={counts.get(1,0)} HIGH={counts.get(2,0)}")

    _print_metrics("BAYESIAN classifier:", bayes_metrics, unc=bayes_unc, ece=bayes_ece)
    _print_metrics("BASELINE classifier:", base_metrics, ece=base_ece)

    results = {
        "device":               str(device),
        "test_samples":         len(test_dataset),
        "sanity_checks_passed": sanity_ok,
        "bayesian": {
            "best_epoch":        bayes_ckpt["best_epoch"],
            "best_val_ce":       float(bayes_ckpt["best_val_loss"]),
            "temperature_tau":   tau_bayes,
            "ece":               bayes_ece,
            "global_metrics":    _make_serializable(bayes_metrics),
            "uncertainty_metrics": _make_serializable(bayes_unc),
        },
        "baseline": {
            "best_epoch":      base_ckpt["best_epoch"],
            "best_val_ce":     float(base_ckpt["best_val_loss"]),
            "temperature_tau": tau_base,
            "ece":             base_ece,
            "global_metrics":  _make_serializable(base_metrics),
        },
    }

    results_path = results_dir / "evaluation_results_classifier.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados guardados en: {results_path}")