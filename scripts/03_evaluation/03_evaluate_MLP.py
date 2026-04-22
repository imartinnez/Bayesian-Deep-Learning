from pathlib import Path
import sys
import json
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
from src.evaluation.metrics import compute_probabilistic_metrics, build_gaussian_interval, compute_coverage, compute_sharpness
from src.evaluation.calibration import compute_calibration_data, plot_calibration, calibrate_temperature
from src.evaluation.regimes import classify_regimes, evaluate_by_regime

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def make_loader(dataset: HARDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_mlp_model(checkpoint_path: Path, device: torch.device) -> MLP:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MLP(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        dense_size=ckpt["dense_size"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_bayesian_mlp_model(checkpoint_path: Path, device: torch.device) -> BayesianMLP:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = BayesianMLP(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        dense_size=ckpt["dense_size"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def add_interval_metrics(metrics: dict, y_true, y_mean, y_std) -> dict:
    for level in (0.80, 0.90, 0.95):
        lower, upper = build_gaussian_interval(y_mean=y_mean, y_std=y_std, level=level)
        pct = int(level * 100)
        metrics[f"coverage_{pct}"] = compute_coverage(y_true=y_true, lower=lower, upper=upper)
        metrics[f"sharpness_{pct}"] = compute_sharpness(lower=lower, upper=upper)
    return metrics

def validation_gate(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray,
                    sigma_constant: float) -> dict:
    rmse_model = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
    rmse_naive = float(np.std(y_true, ddof=0))
    check1_pass = rmse_model < rmse_naive
    check1_ratio = rmse_model / rmse_naive

    nll_model = float(np.mean(0.5 * ((y_true - y_mean) ** 2 / y_std ** 2 + np.log(2 * np.pi * y_std ** 2))))
    nll_constant = float(np.mean(0.5 * ((y_true - y_mean) ** 2 / sigma_constant ** 2 + np.log(2 * np.pi * sigma_constant ** 2))))
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


def print_gate(gate: dict) -> None:
    status = "PASS" if gate["all_pass"] else "FAIL"
    print(f"\n=== Validation Gate [{status}] ===")
    p = "✓" if gate["check1_pass"] else "✗"
    print(f"  {p} Check 1 RMSE ratio:    {gate['check1_rmse_ratio']:.4f} (< 1.0 to pass)")
    p = "✓" if gate["check2_pass"] else "✗"
    print(f"  {p} Check 2 NLL model:     {gate['check2_nll_model']:.4f} vs constant {gate['check2_nll_constant']:.4f}")
    p = "✓" if gate["check3_pass"] else "✗"
    print(f"  {p} Check 3 CV(std):       {gate['check3_cv_std']:.4f} (> 0.10 to pass)")
    p = "✓" if gate["check4_pass"] else "✗"
    print(f"  {p} Check 4 Spearman:      {gate['check4_spearman']:.4f} (> 0.10 to pass)")


def build_semaphore_risk(
    mu: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    mu_val: np.ndarray,
    epistemic_uncertainty_val: np.ndarray,
) -> tuple[np.ndarray, dict]:
    epistemic_std     = np.sqrt(epistemic_uncertainty)
    epistemic_std_val = np.sqrt(epistemic_uncertainty_val)

    # Normalise both components using validation statistics so they contribute
    # on the same scale before combining. mu is negative log-vol (higher = more
    # volatile), epistemic_std is positive (higher = more uncertainty).
    mu_mean  = float(mu_val.mean());  mu_s  = float(mu_val.std())  + 1e-8
    epi_mean = float(epistemic_std_val.mean()); epi_s = float(epistemic_std_val.std()) + 1e-8

    mu_z_test  = (mu          - mu_mean)  / mu_s
    epi_z_test = (epistemic_std - epi_mean) / epi_s
    mu_z_val   = (mu_val        - mu_mean)  / mu_s
    epi_z_val  = (epistemic_std_val - epi_mean) / epi_s

    # 70 % weight on predicted vol level, 30 % on epistemic uncertainty
    risk_score_test = 0.7 * mu_z_test + 0.3 * epi_z_test
    risk_score_val  = 0.7 * mu_z_val  + 0.3 * epi_z_val

    # Thresholds from validation — no leakage from test
    q33 = float(np.percentile(risk_score_val, 33))
    q67 = float(np.percentile(risk_score_val, 67))

    signals = np.where(
        risk_score_test <= q33, "VERDE",
        np.where(risk_score_test <= q67, "AMARILLO", "ROJO")
    )
    return signals, {"q33": q33, "q67": q67, "risk_scores": risk_score_test}


def run_backtest(y_true: np.ndarray, signals: np.ndarray) -> dict:
    # Key validation: does the signal discriminate actual realized volatility?
    signal_vol_stats = {}
    for s in ["VERDE", "AMARILLO", "ROJO"]:
        mask = signals == s
        n = int(np.sum(mask))
        if n > 0:
            signal_vol_stats[s] = {
                "n": n,
                "mean_logvol": float(np.mean(y_true[mask])),
                "std_logvol":  float(np.std(y_true[mask])),
            }

    # Vol-timing strategy: full position on VERDE, half on AMARILLO, out on ROJO
    peso = {"VERDE": 1.0, "AMARILLO": 0.5, "ROJO": 0.0}
    weights = np.array([peso[s] for s in signals])

    # Proxy return: z-score of (-realized_vol) × position weight
    # Being invested on low-vol days is "profit"; high-vol days avoided = avoided loss
    vol_z = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
    daily_returns = -vol_z * weights

    cumret = np.cumsum(daily_returns)
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252))
    peak   = np.maximum.accumulate(cumret)
    max_dd = float(np.min(cumret - peak))

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "signal_vol_stats": signal_vol_stats,
        "signal_counts": {s: d["n"] for s, d in signal_vol_stats.items()},
    }


if __name__ == "__main__":
    cfg = load_config(ROOT_DIR / "config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    har_dir    = ROOT_DIR / cfg["paths"]["har_dir"]
    models_dir = ROOT_DIR / cfg["paths"]["har_models"]
    figures_dir = ROOT_DIR / cfg["paths"]["har_figures"]
    results_dir = ROOT_DIR / cfg["paths"]["har_results"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size  = cfg["har_training"]["batch_size"]
    num_workers = cfg["har_training"]["num_workers"]
    mc_samples  = cfg["har_inference"]["mc_samples"]

    # ── BLOCK A: load data ────────────────────────────────────────────────────
    train_dataset = HARDataset.from_npz(har_dir / cfg["paths"]["har_train_filename"])
    val_dataset   = HARDataset.from_npz(har_dir / cfg["paths"]["har_val_filename"])
    test_dataset  = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])

    val_loader  = make_loader(val_dataset,  batch_size, num_workers)
    test_loader = make_loader(test_dataset, batch_size, num_workers)

    with open(har_dir / cfg["paths"]["har_thresholds_filename"]) as f:
        thresholds = json.load(f)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── BLOCK B: inference ────────────────────────────────────────────────────
    mlp_model, mlp_ckpt = load_mlp_model(models_dir / cfg["paths"]["mlp_baseline_checkpoint"], device)
    pred_val_mlp, y_val = predict_baseline(mlp_model, val_loader, device)
    pred_test_mlp, y_test = predict_baseline(mlp_model, test_loader, device)

    sigma_constant = float(np.std(y_val - pred_val_mlp, ddof=0))
    sigma_baseline_test = np.full_like(pred_test_mlp, fill_value=sigma_constant)

    bayes_model, bayes_ckpt = load_bayesian_mlp_model(models_dir / cfg["paths"]["bayesian_mlp_checkpoint"], device)
    bayes_val = predict_bayesian(bayes_model, val_loader, device, T=mc_samples)
    bayes_test = predict_bayesian(bayes_model, test_loader, device, T=mc_samples)

    tau = calibrate_temperature(
    y_true=bayes_val["y_true"],
    predictive_mean=bayes_val["predictive_mean"],
    predictive_std=np.sqrt(bayes_val["total_uncertainty"]),
    )

    print(f"\nTemperature scaling tau: {tau:.4f}")
    sigma_bayes_test = np.sqrt(bayes_test["total_uncertainty"]) / tau


    # ── BLOCK C: validation gate ──────────────────────────────────────────────
    gate = validation_gate(
        y_true=bayes_test["y_true"],
        y_mean=bayes_test["predictive_mean"],
        y_std=sigma_bayes_test,
        sigma_constant=sigma_constant,
    )
    print_gate(gate)

    if not gate["all_pass"]:
        print("\nWARNING: Bayesian MLP did not pass all validation checks.")
        print("Proceeding with evaluation anyway for diagnostic purposes.\n")

    # ── BLOCK D: global metrics ───────────────────────────────────────────────
    baseline_metrics = compute_probabilistic_metrics(
        y_true=y_test, y_mean=pred_test_mlp, y_std=sigma_baseline_test, levels=(0.80, 0.90, 0.95),
    )
    baseline_metrics = add_interval_metrics(baseline_metrics, y_test, pred_test_mlp, sigma_baseline_test)
    baseline_metrics["sigma_constant"] = sigma_constant

    bayesian_metrics = compute_probabilistic_metrics(
        y_true=bayes_test["y_true"], y_mean=bayes_test["predictive_mean"],
        y_std=sigma_bayes_test, levels=(0.80, 0.90, 0.95),
    )
    bayesian_metrics = add_interval_metrics(bayesian_metrics, bayes_test["y_true"],
                                            bayes_test["predictive_mean"], sigma_bayes_test)

    baseline_calib = compute_calibration_data(y_true=y_test, predictive_mean=pred_test_mlp,
                                              predictive_std=sigma_baseline_test)
    plot_calibration(baseline_calib, save_path=figures_dir / cfg["paths"]["mlp_baseline_calibration_plot"])

    bayesian_calib = compute_calibration_data(y_true=bayes_test["y_true"],
                                              predictive_mean=bayes_test["predictive_mean"],
                                              predictive_std=sigma_bayes_test)
    plot_calibration(bayesian_calib, save_path=figures_dir / cfg["paths"]["bayesian_mlp_calibration_plot"])

    # ── BLOCK E: regimes + semaphore ──────────────────────────────────────────
    bayesian_regimes = classify_regimes(rv_series=bayes_test["y_true"], train_rv=train_dataset.y)
    bayesian_regime_metrics = evaluate_by_regime(
        y_true=bayes_test["y_true"], mu=bayes_test["predictive_mean"],
        sigma=sigma_bayes_test, regimes=bayesian_regimes,
    )

    baseline_regimes = classify_regimes(rv_series=y_test, train_rv=train_dataset.y)
    baseline_regime_metrics = evaluate_by_regime(
        y_true=y_test, mu=pred_test_mlp, sigma=sigma_baseline_test, regimes=baseline_regimes,
    )

    signals, semaphore_meta = build_semaphore_risk(
        mu=bayes_test["predictive_mean"],
        epistemic_uncertainty=bayes_test["epistemic_uncertainty"],
        mu_val=bayes_val["predictive_mean"],
        epistemic_uncertainty_val=bayes_val["epistemic_uncertainty"],
    )

    # ── BLOCK F: backtest ─────────────────────────────────────────────────────
    backtest = run_backtest(y_true=bayes_test["y_true"], signals=signals)

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n=== MLP Baseline global metrics ===")
    for k in ("rmse", "mae", "nll", "crps", "coverage_80", "coverage_90", "coverage_95", "sharpness_90"):
        print(f"  {k:<20} {baseline_metrics.get(k, float('nan')):.6f}")

    print("\n=== Bayesian MLP global metrics ===")
    for k in ("rmse", "mae", "nll", "crps", "coverage_80", "coverage_90", "coverage_95", "sharpness_90"):
        print(f"  {k:<20} {bayesian_metrics.get(k, float('nan')):.6f}")

    print("\n=== Bayesian by regime ===")
    for regime in ("LOW", "MED", "HIGH"):
        v = bayesian_regime_metrics[regime]
        print(f"  {regime} n={v['n_obs']} | cov90={v['coverage_90']:.4f} | crps={v['crps']:.4f} | nll={v['nll']:.4f}")

    print("\n=== Semaphore distribution ===")
    print(f"  Risk thresholds (val): q33={semaphore_meta['q33']:.4f} | q67={semaphore_meta['q67']:.4f}")
    for s in ("VERDE", "AMARILLO", "ROJO"):
        stats = backtest["signal_vol_stats"].get(s, {})
        n = stats.get("n", 0)
        mean_v = stats.get("mean_logvol", float("nan"))
        print(f"  {s:<10} n={n:<4} | mean actual log-vol = {mean_v:.4f}")

    print("\n=== Backtest (vol-timing strategy) ===")
    print(f"  Sharpe:       {backtest['sharpe']:.4f}")
    print(f"  Max Drawdown: {backtest['max_drawdown']:.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    def to_serializable(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.generic,)):
                out[k] = v.item()
            elif isinstance(v, dict):
                out[k] = to_serializable(v)
            else:
                out[k] = v
        return out

    summary = {
        "device": str(device),
        "temperature_scaling_tau": float(tau),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "validation_gate": gate,
        "baseline": {
            "best_epoch": int(mlp_ckpt["best_epoch"]),
            "best_val_loss": float(mlp_ckpt["best_val_loss"]),
            "sigma_constant": sigma_constant,
            "global_metrics": to_serializable(baseline_metrics),
            "regime_metrics": to_serializable(baseline_regime_metrics),
        },
        "bayesian": {
            "best_epoch": int(bayes_ckpt["best_epoch"]),
            "best_val_nll": float(bayes_ckpt["best_val_loss"]),
            "global_metrics": to_serializable(bayesian_metrics),
            "regime_metrics": to_serializable(bayesian_regime_metrics),
        },
        "semaphore": {
            "thresholds": {"q33": semaphore_meta["q33"], "q67": semaphore_meta["q67"]},
            "signal_counts": backtest["signal_counts"],
            "signal_vol_stats": backtest["signal_vol_stats"],
        },
        "backtest": {"sharpe": backtest["sharpe"], "max_drawdown": backtest["max_drawdown"]},
    }

    out_path = results_dir / cfg["paths"]["har_evaluation_results"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {out_path}")