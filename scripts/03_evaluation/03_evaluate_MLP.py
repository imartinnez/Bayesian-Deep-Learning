from pathlib import Path
import sys
import argparse
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.data.download import load_config
from src.data.har_dataset import HARDataset
from src.models.MLP import MLP
from src.models.Bayesian_MLP import BayesianMLP
from src.training.MLP_trainer import predict as predict_baseline
from src.training.Bayesian_MLP_trainer import predict as predict_bayesian
from src.evaluation.metrics import (
    compute_probabilistic_metrics,
    build_gaussian_interval,
    compute_coverage,
    compute_sharpness,
)
from src.evaluation.calibration import (
    compute_calibration_data,
    plot_calibration,
    calibrate_temperature,
)
from src.evaluation.regimes import classify_regimes, evaluate_by_regime
from src.evaluation.semaphore import build_semaphore_risk, evaluate_spr_states
from src.evaluation.reporting import (
    print_header,
    print_kv,
    print_artifacts,
    print_metric_block,
    print_regime_block,
)


GLOBAL_METRIC_KEYS = [
    "rmse",
    "mae",
    "nll",
    "crps",
    "coverage_80",
    "coverage_90",
    "coverage_95",
    "sharpness_90",
]

SELECTIVE_COLUMNS = [
    ("coverage", "Keep"),
    ("n_obs", "n"),
    ("rmse", "RMSE"),
    ("nll", "NLL"),
    ("crps", "CRPS"),
    ("coverage_90", "Cov90"),
    ("rmse_gain_pct", "RMSE gain"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and Bayesian MLP models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file, relative to project root or absolute.",
    )
    return parser.parse_args()


def make_loader(dataset: HARDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_mlp_model(checkpoint_path: Path, device: torch.device) -> tuple[MLP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MLP(
        n_features=checkpoint["n_features"],
        hidden_size=checkpoint["hidden_size"],
        dense_size=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_bayesian_mlp_model(checkpoint_path: Path, device: torch.device) -> tuple[BayesianMLP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BayesianMLP(
        n_features=checkpoint["n_features"],
        hidden_size=checkpoint["hidden_size"],
        dense_size=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def add_interval_metrics(metrics: dict, y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> dict:
    for level in (0.80, 0.90, 0.95):
        lower, upper = build_gaussian_interval(y_mean=y_mean, y_std=y_std, level=level)
        pct = int(level * 100)
        metrics[f"coverage_{pct}"] = compute_coverage(y_true=y_true, lower=lower, upper=upper)
        metrics[f"sharpness_{pct}"] = compute_sharpness(lower=lower, upper=upper)
    return metrics


def validation_gate(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, sigma_constant: float) -> dict:
    rmse_model = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
    rmse_naive = float(np.std(y_true, ddof=0))
    check1_pass = rmse_model < rmse_naive
    check1_ratio = rmse_model / rmse_naive

    nll_model = float(np.mean(0.5 * ((y_true - y_mean) ** 2 / y_std**2 + np.log(2 * np.pi * y_std**2))))
    nll_constant = float(
        np.mean(0.5 * ((y_true - y_mean) ** 2 / sigma_constant**2 + np.log(2 * np.pi * sigma_constant**2)))
    )
    check2_pass = nll_model < nll_constant

    cv_std = float(np.std(y_std) / (np.mean(y_std) + 1e-8))
    check3_pass = cv_std > 0.10

    abs_err = np.abs(y_true - y_mean)
    spearman_corr, _ = spearmanr(y_std, abs_err)
    check4_pass = float(spearman_corr) > 0.10

    all_pass = check1_pass and check2_pass and check3_pass and check4_pass

    return {
        "check1_rmse_ratio": check1_ratio,
        "check1_pass": check1_pass,
        "check2_nll_model": nll_model,
        "check2_nll_constant": nll_constant,
        "check2_pass": check2_pass,
        "check3_cv_std": cv_std,
        "check3_pass": check3_pass,
        "check4_spearman": float(spearman_corr),
        "check4_pass": check4_pass,
        "all_pass": all_pass,
    }


def print_gate_block(title: str, gate: dict) -> None:
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Check':<24}{'Value':>12}{'Status':>10}")

    rows = [
        ("RMSE ratio", gate["check1_rmse_ratio"], gate["check1_pass"]),
        ("NLL vs constant", gate["check2_nll_model"], gate["check2_pass"]),
        ("CV(std)", gate["check3_cv_std"], gate["check3_pass"]),
        ("Spearman(std,error)", gate["check4_spearman"], gate["check4_pass"]),
    ]

    for label, value, ok in rows:
        status = "PASS" if ok else "FAIL"
        print(f"{label:<24}{value:>12.4f}{status:>10}")

    overall = "PASS" if gate["all_pass"] else "FAIL"
    print(f"\nOverall status: {overall}")


def run_backtest(realized_rv: np.ndarray, signals: np.ndarray) -> dict:
    realized_rv = np.asarray(realized_rv, dtype=float).reshape(-1)
    signals = np.asarray(signals, dtype=object).reshape(-1)

    if realized_rv.shape != signals.shape:
        raise ValueError("realized_rv and signals must share the same shape.")

    signal_vol_stats = {}
    for state in ("NORMAL", "ALERTA", "ESTRES"):
        mask = signals == state
        n_obs = int(np.sum(mask))
        if n_obs > 0:
            signal_vol_stats[state] = {
                "n": n_obs,
                "mean_rv": float(np.mean(realized_rv[mask])),
                "std_rv": float(np.std(realized_rv[mask])),
            }

    weights_map = {"NORMAL": 1.0, "ALERTA": 0.5, "ESTRES": 0.0}
    weights = np.array([weights_map[state] for state in signals])

    vol_z = (realized_rv - realized_rv.mean()) / (realized_rv.std() + 1e-8)
    daily_returns = -vol_z * weights

    cumulative = np.cumsum(daily_returns)
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252))
    peak = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.min(cumulative - peak))

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "signal_vol_stats": signal_vol_stats,
        "signal_counts": {state: values["n"] for state, values in signal_vol_stats.items()},
    }


def compute_selective_prediction_curve(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    uncertainty_score: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_mean = np.asarray(y_mean, dtype=float).reshape(-1)
    y_std = np.asarray(y_std, dtype=float).reshape(-1)
    uncertainty_score = np.asarray(uncertainty_score, dtype=float).reshape(-1)

    if not (y_true.shape == y_mean.shape == y_std.shape == uncertainty_score.shape):
        raise ValueError("All selective-prediction inputs must share the same shape.")

    if len(y_true) == 0:
        raise ValueError("Selective prediction requires at least one observation.")

    order = np.argsort(uncertainty_score, kind="mergesort")
    n_obs = len(y_true)

    full_metrics = compute_probabilistic_metrics(
        y_true=y_true,
        y_mean=y_mean,
        y_std=y_std,
        levels=(0.80, 0.90, 0.95),
    )

    rows = []
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError(f"coverage must be in (0, 1]. Received: {coverage}")

        n_keep = max(1, int(np.floor(n_obs * coverage)))
        idx = order[:n_keep]

        metrics = compute_probabilistic_metrics(
            y_true=y_true[idx],
            y_mean=y_mean[idx],
            y_std=y_std[idx],
            levels=(0.80, 0.90, 0.95),
        )

        rows.append({
            "coverage": float(n_keep / n_obs),
            "n_obs": int(n_keep),
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "nll": float(metrics["nll"]),
            "crps": float(metrics["crps"]),
            "coverage_90": float(metrics["coverage_90"]),
            "sharpness_90": float(
                compute_sharpness(*build_gaussian_interval(y_mean[idx], y_std[idx], level=0.90))
            ),
            "rmse_gain_pct": float(100.0 * (full_metrics["rmse"] - metrics["rmse"]) / (full_metrics["rmse"] + 1e-8)),
            "nll_gain_pct": float(100.0 * (full_metrics["nll"] - metrics["nll"]) / (abs(full_metrics["nll"]) + 1e-8)),
            "crps_gain_pct": float(100.0 * (full_metrics["crps"] - metrics["crps"]) / (full_metrics["crps"] + 1e-8)),
            "mean_uncertainty": float(np.mean(uncertainty_score[idx])),
        })

    return rows


def summarise_selective_curve(curve: list[dict], key: str) -> float:
    coverages = np.array([row["coverage"] for row in curve], dtype=float)
    values = np.array([row[key] for row in curve], dtype=float)

    if len(coverages) < 2:
        return float(values[0])

    order = np.argsort(coverages)
    x = coverages[order]
    y = values[order]
    widths = np.diff(x)
    heights = 0.5 * (y[:-1] + y[1:])
    area = np.sum(widths * heights)

    return float(area / (x.max() - x.min()))


def print_semaphore_block(title: str, semaphore_meta: dict, semaphore_eval: dict) -> None:
    print(f"\n{title}")
    print("-" * 80)

    print_kv(
        "Thresholds",
        (
            f"alerta={semaphore_meta['umbral_alerta']:.4f} | "
            f"estres={semaphore_meta['umbral_estres']:.4f} | "
            f"epist={semaphore_meta['umbral_epist']:.4f}"
        ),
        indent=2,
    )
    print_kv(
        "Decision rule",
        (
            f"P(alerta)>={semaphore_meta['prob_alerta_thr']:.2f} | "
            f"P(estres)>={semaphore_meta['prob_estres_thr']:.2f} | "
            f"epist>={semaphore_meta['umbral_epist_decision']:.4f} | "
            f"persist={semaphore_meta['stress_persistence']} | "
            f"highP>={semaphore_meta['stress_high_prob_thr']:.2f}"
        ),
        indent=2,
    )

    if "tuning" in semaphore_meta:
        tuning = semaphore_meta["tuning"]
        print_kv(
            "Validation tuning",
            (
                f"score={tuning['validation_score']:.4f} | "
                f"macro_f1={tuning['validation_metrics']['macro_f1']:.4f} | "
                f"alert_f1={tuning['validation_metrics']['alert_metrics']['f1']:.4f} | "
                f"stress_f1={tuning['validation_metrics']['stress_metrics']['f1']:.4f}"
            ),
            indent=2,
        )

    print_kv(
        "Test quality",
        (
            f"macro_f1={semaphore_eval['macro_f1']:.4f} | "
            f"alert_precision={semaphore_eval['alert_metrics']['precision']:.4f} | "
            f"alert_recall={semaphore_eval['alert_metrics']['recall']:.4f} | "
            f"stress_precision={semaphore_eval['stress_metrics']['precision']:.4f} | "
            f"stress_recall={semaphore_eval['stress_metrics']['recall']:.4f}"
        ),
        indent=2,
    )
    print_kv(
        "State rates",
        (
            f"alert actual/pred={semaphore_eval['actual_alert_rate']:.3f}/{semaphore_eval['predicted_alert_rate']:.3f} | "
            f"stress actual/pred={semaphore_eval['actual_stress_rate']:.3f}/{semaphore_eval['predicted_stress_rate']:.3f}"
        ),
        indent=2,
    )

    for state in ("NORMAL", "ALERTA", "ESTRES"):
        stats = semaphore_eval["signal_rv_stats"].get(state, {})
        n_obs = stats.get("n", 0)
        mean_rv = stats.get("mean_rv", float("nan"))
        print_kv(f"{state} state", f"n={n_obs} | mean RV={mean_rv:.4f}", indent=2)


def print_backtest_block(title: str, backtest: dict) -> None:
    print(f"\n{title}")
    print("-" * 80)
    print_kv("Sharpe", f"{backtest['sharpe']:.4f}", indent=2)
    print_kv("Max drawdown", f"{backtest['max_drawdown']:.4f}", indent=2)


def print_selective_block(title: str, selective_summary: dict) -> None:
    print(f"\n{title}")
    print("-" * 80)

    for label, block in (
        ("Total uncertainty", selective_summary["total_uncertainty"]),
        ("Epistemic uncertainty", selective_summary["epistemic_uncertainty"]),
    ):
        print_kv(
            label,
            f"AURC-RMSE={block['aurc_rmse']:.4f} | AURC-NLL={block['aurc_nll']:.4f} | AURC-CRPS={block['aurc_crps']:.4f}",
            indent=2,
        )
        print(f"{'Keep':>8}{'n':>8}{'RMSE':>10}{'NLL':>10}{'CRPS':>10}{'Cov90':>10}{'RMSE gain':>12}")
        for row in block["curve"]:
            print(
                f"{row['coverage']:>8.0%}"
                f"{row['n_obs']:>8d}"
                f"{row['rmse']:>10.4f}"
                f"{row['nll']:>10.4f}"
                f"{row['crps']:>10.4f}"
                f"{row['coverage_90']:>10.4f}"
                f"{row['rmse_gain_pct']:>11.1f}%"
            )
        print()


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


if __name__ == "__main__":
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    har_dir = ROOT_DIR / cfg["paths"]["har_dir"]
    models_dir = ROOT_DIR / cfg["paths"]["har_models"]
    figures_dir = ROOT_DIR / cfg["paths"]["har_figures"]
    results_dir = ROOT_DIR / cfg["paths"]["har_results"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg["har_training"]["batch_size"]
    num_workers = cfg["har_training"]["num_workers"]
    mc_samples = cfg["har_inference"]["mc_samples"]

    baseline_calibration_plot_path = figures_dir / cfg["paths"]["mlp_baseline_calibration_plot"]
    bayesian_calibration_plot_path = figures_dir / cfg["paths"]["bayesian_mlp_calibration_plot"]
    evaluation_results_path = results_dir / cfg["paths"]["har_evaluation_results"]

    train_dataset = HARDataset.from_npz(har_dir / cfg["paths"]["har_train_filename"])
    val_dataset = HARDataset.from_npz(har_dir / cfg["paths"]["har_val_filename"])
    test_dataset = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])

    train_loader = make_loader(train_dataset, batch_size, num_workers)
    val_loader = make_loader(val_dataset, batch_size, num_workers)
    test_loader = make_loader(test_dataset, batch_size, num_workers)

    baseline_model, baseline_ckpt = load_mlp_model(
        models_dir / cfg["paths"]["mlp_baseline_checkpoint"],
        device,
    )
    pred_val_baseline, y_val = predict_baseline(baseline_model, val_loader, device)
    pred_test_baseline, y_test = predict_baseline(baseline_model, test_loader, device)

    sigma_constant = float(np.std(y_val - pred_val_baseline, ddof=0))
    sigma_baseline_test = np.full_like(pred_test_baseline, fill_value=sigma_constant)

    baseline_metrics = compute_probabilistic_metrics(
        y_true=y_test,
        y_mean=pred_test_baseline,
        y_std=sigma_baseline_test,
        levels=(0.80, 0.90, 0.95),
    )
    baseline_metrics = add_interval_metrics(
        baseline_metrics,
        y_test,
        pred_test_baseline,
        sigma_baseline_test,
    )
    baseline_metrics["sigma_constant"] = sigma_constant

    baseline_calibration_data = compute_calibration_data(
        y_true=y_test,
        predictive_mean=pred_test_baseline,
        predictive_std=sigma_baseline_test,
    )
    plot_calibration(baseline_calibration_data, save_path=baseline_calibration_plot_path)

    baseline_regimes = classify_regimes(rv_series=y_test, train_rv=train_dataset.y)
    baseline_regime_metrics = evaluate_by_regime(
        y_true=y_test,
        mu=pred_test_baseline,
        sigma=sigma_baseline_test,
        regimes=baseline_regimes,
    )

    bayesian_model, bayesian_ckpt = load_bayesian_mlp_model(
        models_dir / cfg["paths"]["bayesian_mlp_checkpoint"],
        device,
    )
    bayes_train = predict_bayesian(bayesian_model, train_loader, device, T=mc_samples)
    bayes_val = predict_bayesian(bayesian_model, val_loader, device, T=mc_samples)
    bayes_test = predict_bayesian(bayesian_model, test_loader, device, T=mc_samples)

    tau = calibrate_temperature(
        y_true=bayes_val["y_true"],
        predictive_mean=bayes_val["predictive_mean"],
        predictive_std=np.sqrt(bayes_val["total_uncertainty"]),
    )

    sigma_bayes_val = np.sqrt(bayes_val["total_uncertainty"]) / tau
    sigma_bayes_test = np.sqrt(bayes_test["total_uncertainty"]) / tau
    sigma_epist_val = np.sqrt(np.clip(bayes_val["epistemic_uncertainty"], 0.0, None))
    sigma_epist_test = np.sqrt(np.clip(bayes_test["epistemic_uncertainty"], 0.0, None))

    gate = validation_gate(
        y_true=bayes_val["y_true"],
        y_mean=bayes_val["predictive_mean"],
        y_std=sigma_bayes_val,
        sigma_constant=sigma_constant,
    )

    bayesian_metrics = compute_probabilistic_metrics(
        y_true=bayes_test["y_true"],
        y_mean=bayes_test["predictive_mean"],
        y_std=sigma_bayes_test,
        levels=(0.80, 0.90, 0.95),
    )
    bayesian_metrics = add_interval_metrics(
        bayesian_metrics,
        bayes_test["y_true"],
        bayes_test["predictive_mean"],
        sigma_bayes_test,
    )

    bayesian_calibration_data = compute_calibration_data(
        y_true=bayes_test["y_true"],
        predictive_mean=bayes_test["predictive_mean"],
        predictive_std=sigma_bayes_test,
    )
    plot_calibration(bayesian_calibration_data, save_path=bayesian_calibration_plot_path)

    bayesian_regimes = classify_regimes(rv_series=bayes_test["y_true"], train_rv=train_dataset.y)
    bayesian_regime_metrics = evaluate_by_regime(
        y_true=bayes_test["y_true"],
        mu=bayes_test["predictive_mean"],
        sigma=sigma_bayes_test,
        regimes=bayesian_regimes,
    )

    signals, semaphore_meta = build_semaphore_risk(
        mu_pred=bayes_test["predictive_mean"],
        sigma_total=sigma_bayes_test,
        sigma_epist=sigma_epist_test,
        train_rv=np.exp(bayes_train["y_true"]),
        sigma_epist_train=np.sqrt(np.clip(bayes_train["epistemic_uncertainty"], 0.0, None)),
        mu_val=bayes_val["predictive_mean"],
        sigma_total_val=sigma_bayes_val,
        sigma_epist_val=sigma_epist_val,
        val_rv=np.exp(bayes_val["y_true"]),
    )
    semaphore_eval = evaluate_spr_states(
        actual_rv=np.exp(bayes_test["y_true"]),
        predicted_states=signals,
        umbral_alerta=semaphore_meta["umbral_alerta"],
        umbral_estres=semaphore_meta["umbral_estres"],
    )

    backtest = run_backtest(realized_rv=np.exp(bayes_test["y_true"]), signals=signals)

    selective_total = compute_selective_prediction_curve(
        y_true=bayes_test["y_true"],
        y_mean=bayes_test["predictive_mean"],
        y_std=sigma_bayes_test,
        uncertainty_score=sigma_bayes_test,
    )
    selective_epist = compute_selective_prediction_curve(
        y_true=bayes_test["y_true"],
        y_mean=bayes_test["predictive_mean"],
        y_std=sigma_bayes_test,
        uncertainty_score=sigma_epist_test,
    )

    selective_summary = {
        "total_uncertainty": {
            "aurc_rmse": summarise_selective_curve(selective_total, "rmse"),
            "aurc_nll": summarise_selective_curve(selective_total, "nll"),
            "aurc_crps": summarise_selective_curve(selective_total, "crps"),
            "curve": selective_total,
        },
        "epistemic_uncertainty": {
            "aurc_rmse": summarise_selective_curve(selective_epist, "rmse"),
            "aurc_nll": summarise_selective_curve(selective_epist, "nll"),
            "aurc_crps": summarise_selective_curve(selective_epist, "crps"),
            "curve": selective_epist,
        },
    }

    summary = {
        "device": str(device),
        "temperature_scaling_tau": float(tau),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "validation_gate": gate,
        "baseline": {
            "best_epoch": int(baseline_ckpt["best_epoch"]),
            "best_val_loss": float(baseline_ckpt["best_val_loss"]),
            "sigma_constant": sigma_constant,
            "checkpoint_path": str(models_dir / cfg["paths"]["mlp_baseline_checkpoint"]),
            "calibration_plot_path": str(baseline_calibration_plot_path),
            "global_metrics": to_serializable(baseline_metrics),
            "regime_metrics": to_serializable(baseline_regime_metrics),
        },
        "bayesian": {
            "best_epoch": int(bayesian_ckpt["best_epoch"]),
            "best_val_nll": float(bayesian_ckpt["best_val_loss"]),
            "checkpoint_path": str(models_dir / cfg["paths"]["bayesian_mlp_checkpoint"]),
            "calibration_plot_path": str(bayesian_calibration_plot_path),
            "global_metrics": to_serializable(bayesian_metrics),
            "regime_metrics": to_serializable(bayesian_regime_metrics),
            "selective_prediction": to_serializable(selective_summary),
        },
        "semaphore": {
            "thresholds": {
                "umbral_alerta": semaphore_meta["umbral_alerta"],
                "umbral_estres": semaphore_meta["umbral_estres"],
                "umbral_epist": semaphore_meta["umbral_epist"],
            },
            "decision_rule": {
                "prob_alerta_thr": semaphore_meta["prob_alerta_thr"],
                "prob_estres_thr": semaphore_meta["prob_estres_thr"],
                "epist_scale": semaphore_meta["epist_scale"],
                "umbral_epist_decision": semaphore_meta["umbral_epist_decision"],
                "stress_persistence": semaphore_meta["stress_persistence"],
                "stress_high_prob_scale": semaphore_meta["stress_high_prob_scale"],
                "stress_high_prob_thr": semaphore_meta["stress_high_prob_thr"],
            },
            "evaluation": to_serializable(semaphore_eval),
            "validation_tuning": to_serializable(semaphore_meta.get("tuning", {})),
            "signal_counts": {
                label: semaphore_eval["signal_rv_stats"].get(label, {}).get("n", 0)
                for label in ("NORMAL", "ALERTA", "ESTRES")
            },
            "signal_rv_stats": to_serializable(semaphore_eval["signal_rv_stats"]),
        },
        "backtest": {
            "sharpe": backtest["sharpe"],
            "max_drawdown": backtest["max_drawdown"],
        },
    }

    with open(evaluation_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print_header("RUN")
    print_kv("Script", "HAR / MLP evaluation")
    print_kv("Device", str(device))
    print_kv("Train samples", len(train_dataset))
    print_kv("Validation samples", len(val_dataset))
    print_kv("Test samples", len(test_dataset))

    print_header("MODEL SETUP")
    print_kv("Baseline checkpoint", str(models_dir / cfg["paths"]["mlp_baseline_checkpoint"]))
    print_kv("Bayesian checkpoint", str(models_dir / cfg["paths"]["bayesian_mlp_checkpoint"]))
    print_kv("Temperature tau", f"{tau:.4f}")
    print_kv("Baseline sigma", f"{sigma_constant:.4f}")
    print_kv("MC samples", mc_samples)

    print_header("MAIN RESULTS")
    print_metric_block("Baseline | global metrics", baseline_metrics, GLOBAL_METRIC_KEYS)
    print_metric_block("Bayesian | global metrics", bayesian_metrics, GLOBAL_METRIC_KEYS)

    print_header("DIAGNOSTICS")
    print_gate_block("Validation gate", gate)
    print_regime_block("Baseline | by regime", baseline_regime_metrics)
    print_regime_block("Bayesian | by regime", bayesian_regime_metrics)
    print_semaphore_block("Semaphore diagnostics", semaphore_meta, semaphore_eval)
    print_backtest_block("Backtest", backtest)
    print_selective_block("Selective prediction", selective_summary)

    print_header("ARTIFACTS")
    print_artifacts({
        "Baseline calibration plot": str(baseline_calibration_plot_path),
        "Bayesian calibration plot": str(bayesian_calibration_plot_path),
        "Results JSON": str(evaluation_results_path),
    })