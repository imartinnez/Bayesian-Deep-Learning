from pathlib import Path
import sys
import argparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.download import load_config
from src.data.dataset import WindowedTimeSeriesDataset
from src.models.LSTM import BaselineLSTM
from src.models.Bayesian_LSTM import BayesianLSTM
from src.training.LSTM_trainer import predict as predict_baseline
from src.training.Bayesian_LSTM_trainer import predict as predict_bayesian
from src.evaluation.metrics import (
    compute_probabilistic_metrics,
    build_gaussian_interval,
    compute_coverage,
    compute_sharpness,
)
from src.evaluation.calibration import compute_calibration_data, plot_calibration, calibrate_temperature
from src.evaluation.regimes import classify_regimes, evaluate_by_regime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and Bayesian LSTM models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file, relative to project root or absolute.",
    )
    return parser.parse_args()


def make_loader(dataset: WindowedTimeSeriesDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_bayesian_model(checkpoint_path: Path, device: torch.device) -> tuple[BayesianLSTM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = BayesianLSTM(
        n_features=checkpoint["n_features"],
        hidden=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dense=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_baseline_model(checkpoint_path: Path, device: torch.device) -> tuple[BaselineLSTM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = BaselineLSTM(
        n_features=checkpoint["n_features"],
        hidden=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dense=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def add_interval_90_metrics(metrics: dict, y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> dict:
    lower_90, upper_90 = build_gaussian_interval(y_mean=y_mean, y_std=y_std, level=0.90)
    metrics["coverage_90"] = compute_coverage(y_true=y_true, lower=lower_90, upper=upper_90)
    metrics["sharpness_90"] = compute_sharpness(lower=lower_90, upper=upper_90)
    return metrics


def print_global_metrics(title: str, metrics: dict) -> None:
    print(title)
    print(f"RMSE:         {metrics['rmse']:.6f}")
    print(f"MAE:          {metrics['mae']:.6f}")
    print(f"NLL:          {metrics['nll']:.6f}")
    print(f"CRPS:         {metrics['crps']:.6f}")
    print(f"Coverage 80%: {metrics['coverage_80']:.6f}")
    print(f"Coverage 90%: {metrics['coverage_90']:.6f}")
    print(f"Coverage 95%: {metrics['coverage_95']:.6f}")
    print(f"Sharpness 90%:{metrics['sharpness_90']:.6f}")


def print_regime_metrics(title: str, regime_results: dict) -> None:
    print(title)
    for regime_name in ("LOW", "MED", "HIGH"):
        values = regime_results[regime_name]
        print(
            f"{regime_name:<4} | n={values['n_obs']:<4d} "
            f"| coverage_90={values['coverage_90']:.6f} "
            f"| crps={values['crps']:.6f} "
            f"| sharpness_90={values['sharpness_90']:.6f} "
            f"| nll={values['nll']:.6f}"
        )


if __name__ == "__main__":
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = ROOT_DIR / cfg["paths"]["splits"]
    models_dir = ROOT_DIR / cfg["paths"]["models"]
    figures_dir = ROOT_DIR / cfg["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_windows_path = splits_dir / cfg["paths"]["train_windows_filename"]
    val_windows_path = splits_dir / cfg["paths"]["val_windows_filename"]
    test_windows_path = splits_dir / cfg["paths"]["test_windows_filename"]

    baseline_checkpoint_path = models_dir / cfg["paths"]["baseline_checkpoint"]
    bayesian_checkpoint_path = models_dir / cfg["paths"]["bayesian_checkpoint"]

    bayesian_calibration_plot_path = figures_dir / cfg["paths"].get(
        "bayesian_calibration_plot_filename",
        "bayesian_calibration_plot.png",
    )
    baseline_calibration_plot_path = figures_dir / cfg["paths"].get(
        "baseline_calibration_plot_filename",
        "baseline_calibration_plot.png",
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    mc_samples = cfg["inference"]["mc_samples"]

    train_dataset = WindowedTimeSeriesDataset.from_npz(train_windows_path)
    val_dataset = WindowedTimeSeriesDataset.from_npz(val_windows_path)
    test_dataset = WindowedTimeSeriesDataset.from_npz(test_windows_path)

    val_loader = make_loader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = make_loader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    bayesian_model, bayesian_checkpoint = load_bayesian_model(bayesian_checkpoint_path, device)
    bayes_results = predict_bayesian(
        model=bayesian_model,
        loader=test_loader,
        device=device,
        T=mc_samples,
    )

    y_test = bayes_results["y_true"]
    mu_bayes = bayes_results["predictive_mean"]
    sigma_bayes = bayes_results["predictive_std"]

    bayes_val_results = predict_bayesian(
        model=bayesian_model,
        loader=val_loader,
        device=device,
        T=mc_samples,
    )

    tau = calibrate_temperature(
        y_true=bayes_val_results["y_true"],
        predictive_mean=bayes_val_results["predictive_mean"],
        predictive_std=bayes_val_results["predictive_std"],
    )
    
    print(f"Temperature scaling tau: {tau:.4f}")
    sigma_bayes = sigma_bayes/tau

    bayesian_global_metrics = compute_probabilistic_metrics(
        y_true=y_test,
        y_mean=mu_bayes,
        y_std=sigma_bayes,
        levels=(0.80, 0.90, 0.95),
    )
    bayesian_global_metrics = add_interval_90_metrics(
        metrics=bayesian_global_metrics,
        y_true=y_test,
        y_mean=mu_bayes,
        y_std=sigma_bayes,
    )

    bayesian_calibration_data = compute_calibration_data(
        y_true=y_test,
        predictive_mean=mu_bayes,
        predictive_std=sigma_bayes,
    )
    plot_calibration(bayesian_calibration_data, save_path=bayesian_calibration_plot_path)

    bayesian_regimes = classify_regimes(rv_series=y_test, train_rv=train_dataset.y)
    bayesian_regime_metrics = evaluate_by_regime(
        y_true=y_test,
        mu=mu_bayes,
        sigma=sigma_bayes,
        regimes=bayesian_regimes,
    )

    baseline_model, baseline_checkpoint = load_baseline_model(baseline_checkpoint_path, device)

    pred_val_baseline, y_val = predict_baseline(baseline_model, val_loader, device)
    pred_test_baseline, y_test_baseline = predict_baseline(baseline_model, test_loader, device)

    residuals_val = y_val - pred_val_baseline
    sigma_baseline = float(np.std(residuals_val, ddof=0))
    sigma_baseline_test = np.full_like(pred_test_baseline, fill_value=sigma_baseline, dtype=float)

    baseline_global_metrics = compute_probabilistic_metrics(
        y_true=y_test_baseline,
        y_mean=pred_test_baseline,
        y_std=sigma_baseline_test,
        levels=(0.80, 0.90, 0.95),
    )
    baseline_global_metrics = add_interval_90_metrics(
        metrics=baseline_global_metrics,
        y_true=y_test_baseline,
        y_mean=pred_test_baseline,
        y_std=sigma_baseline_test,
    )
    baseline_global_metrics["sigma_constant"] = sigma_baseline

    baseline_calibration_data = compute_calibration_data(
        y_true=y_test_baseline,
        predictive_mean=pred_test_baseline,
        predictive_std=sigma_baseline_test,
    )
    plot_calibration(baseline_calibration_data, save_path=baseline_calibration_plot_path)

    baseline_regimes = classify_regimes(rv_series=y_test_baseline, train_rv=train_dataset.y)
    baseline_regime_metrics = evaluate_by_regime(
        y_true=y_test_baseline,
        mu=pred_test_baseline,
        sigma=sigma_baseline_test,
        regimes=baseline_regimes,
    )

    print("-" * 60)
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train window shape: {train_dataset.X.shape}")
    print(f"Validation window shape: {val_dataset.X.shape}")
    print(f"Test window shape: {test_dataset.X.shape}")
    print()

    print(f"Bayesian best epoch: {bayesian_checkpoint['best_epoch']}")
    print(f"Bayesian best validation NLL: {bayesian_checkpoint['best_val_loss']:.6f}")
    print(f"Bayesian checkpoint: {bayesian_checkpoint_path}")
    print(f"Bayesian calibration plot: {bayesian_calibration_plot_path}")
    print_global_metrics("Bayesian global metrics:", bayesian_global_metrics)
    print()
    print_regime_metrics("Bayesian by regime:", bayesian_regime_metrics)
    print()

    print(f"Baseline best epoch: {baseline_checkpoint['best_epoch']}")
    print(f"Baseline best validation loss: {baseline_checkpoint['best_val_loss']:.6f}")
    print(f"Baseline checkpoint: {baseline_checkpoint_path}")
    print(f"Baseline constant sigma from validation residuals: {sigma_baseline:.6f}")
    print(f"Baseline calibration plot: {baseline_calibration_plot_path}")
    print_global_metrics("Baseline global metrics:", baseline_global_metrics)
    print()
    print_regime_metrics("Baseline by regime:", baseline_regime_metrics)