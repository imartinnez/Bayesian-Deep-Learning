from pathlib import Path
import sys
import argparse
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.data.download import load_config
from src.data.target import add_future_vol_target
from src.models.benchmarks import (
    HAROLSVolatility,
    EWMAVolatility,
    GARCH11Volatility,
    GJRGARCH11Volatility,
)
from src.evaluation.metrics import (
    compute_probabilistic_metrics,
    build_gaussian_interval,
    compute_coverage,
    compute_sharpness,
)
from src.evaluation.reporting import (
    print_header,
    print_kv,
    print_artifacts,
)


BENCHMARK_METRIC_KEYS = [
    "rmse",
    "mae",
    "nll",
    "crps",
    "coverage_80",
    "coverage_90",
    "coverage_95",
    "sharpness_90",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classical volatility benchmarks on the HAR split.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file, relative to project root or absolute.",
    )
    return parser.parse_args()


def add_interval_metrics(metrics: dict, y_true, y_mean, y_std) -> dict:
    for level in (0.80, 0.90, 0.95):
        lower, upper = build_gaussian_interval(y_mean=y_mean, y_std=y_std, level=level)
        pct = int(level * 100)
        metrics[f"coverage_{pct}"] = compute_coverage(y_true=y_true, lower=lower, upper=upper)
        metrics[f"sharpness_{pct}"] = compute_sharpness(lower=lower, upper=upper)
    return metrics


def rolling_log_rv(log_returns: pd.Series, window: int, epsilon: float) -> pd.Series:
    realized_vol = np.sqrt(log_returns.pow(2).rolling(window=window, min_periods=window).mean())
    return np.log(realized_vol.clip(lower=epsilon))


def build_har_splits(parquet_path: Path, train_end: str, val_end: str, epsilon: float, horizon: int):
    df = pd.read_parquet(parquet_path)

    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    df["log_rv_1d_har"] = rolling_log_rv(df["log_return"], window=1, epsilon=epsilon)
    df["log_rv_5d_har"] = rolling_log_rv(df["log_return"], window=5, epsilon=epsilon)
    df["log_rv_22d_har"] = rolling_log_rv(df["log_return"], window=22, epsilon=epsilon)
    df = add_future_vol_target(df, horizon=horizon, target_col="target_har")
    df["target_har"] = df["target_har"].clip(lower=np.log(epsilon))

    needed = ["log_return", "log_rv_1d_har", "log_rv_5d_har", "log_rv_22d_har", "target_har"]
    df = df.dropna(subset=needed)

    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    return train_df, val_df, test_df


def eval_predictive(y_true, y_mean, y_std) -> dict:
    metrics = compute_probabilistic_metrics(
        y_true=y_true,
        y_mean=y_mean,
        y_std=y_std,
        levels=(0.80, 0.90, 0.95),
    )
    metrics = add_interval_metrics(metrics, y_true, y_mean, y_std)
    return metrics


def print_benchmark_table(title: str, rows: list[tuple[str, dict]], metric_keys: list[str]) -> None:
    print(f"\n{title}")
    print("-" * 132)

    model_width = 22
    col_width = 13

    header = f"{'Model':<{model_width}}" + "".join(f"{key:>{col_width}}" for key in metric_keys)
    print(header)
    print("-" * len(header))

    for model_name, metrics in rows:
        line = f"{model_name:<{model_width}}" + "".join(
            f"{metrics.get(key, float('nan')):>{col_width}.4f}" for key in metric_keys
        )
        print(line)


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

    processed_path = ROOT_DIR / cfg["paths"]["processed"] / cfg["paths"]["processed_filename"]
    results_dir = ROOT_DIR / cfg["paths"]["har_results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluation_results_path = results_dir / cfg["paths"].get(
        "evaluation_results_benchmarks_filename",
        "evaluation_results_benchmarks.json",
    )

    train_df, val_df, test_df = build_har_splits(
        parquet_path=processed_path,
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
        epsilon=cfg["har"]["epsilon_squared_return"],
        horizon=cfg["har"]["horizon"],
    )

    har_cols = ["log_rv_1d_har", "log_rv_5d_har", "log_rv_22d_har"]
    X_train_har = train_df[har_cols].values
    X_val_har = val_df[har_cols].values
    X_test_har = test_df[har_cols].values

    y_train = train_df["target_har"].values
    y_val = val_df["target_har"].values
    y_test = test_df["target_har"].values

    r_train = train_df["log_return"].values
    r_val = val_df["log_return"].values
    r_test = test_df["log_return"].values

    results: dict = {}

    har = HAROLSVolatility().fit(X_train_har, y_train)
    har.calibrate_sigma(X_val_har, y_val)
    mu_test_har, sigma_test_har = har.predict(X_test_har)
    results["har_ols"] = {
        "params": {
            "intercept": float(har.intercept_),
            "coef_1d": float(har.coef_[0]),
            "coef_5d": float(har.coef_[1]),
            "coef_22d": float(har.coef_[2]),
            "sigma_const": float(har.sigma_),
        },
        "metrics": eval_predictive(y_test, mu_test_har, sigma_test_har),
    }

    ewma = EWMAVolatility(lam=0.94).fit(r_train)
    sigma2_val_ewma = ewma.forecast_rolling(r_val)
    mu_val_ewma = 0.5 * np.log(sigma2_val_ewma + 1e-12)
    ewma.calibrate_sigma(y_val, mu_val_ewma)
    sigma2_test_ewma = ewma.forecast_rolling(r_test)
    mu_test_ewma = 0.5 * np.log(sigma2_test_ewma + 1e-12)
    sigma_test_ewma = np.full_like(mu_test_ewma, fill_value=ewma.sigma_)
    results["ewma_094"] = {
        "params": {
            "lambda": float(ewma.lam),
            "sigma_const": float(ewma.sigma_),
        },
        "metrics": eval_predictive(y_test, mu_test_ewma, sigma_test_ewma),
    }

    garch = GARCH11Volatility().fit(r_train)
    sigma2_val_garch = garch.forecast_rolling(r_val)
    mu_val_garch = 0.5 * np.log(sigma2_val_garch + 1e-12)
    garch.calibrate_sigma(y_val, mu_val_garch)
    sigma2_test_garch = garch.forecast_rolling(r_test)
    mu_test_garch = 0.5 * np.log(sigma2_test_garch + 1e-12)
    sigma_test_garch = np.full_like(mu_test_garch, fill_value=garch.sigma_)
    results["garch_11"] = {
        "params": {
            "omega": float(garch.omega),
            "alpha": float(garch.alpha),
            "beta": float(garch.beta),
            "alpha_plus_beta": float(garch.alpha + garch.beta),
            "converged": bool(garch._converged),
            "nll_train": float(garch._nll),
            "sigma_const": float(garch.sigma_),
        },
        "metrics": eval_predictive(y_test, mu_test_garch, sigma_test_garch),
    }

    gjr = GJRGARCH11Volatility().fit(r_train)
    sigma2_val_gjr = gjr.forecast_rolling(r_val)
    mu_val_gjr = 0.5 * np.log(sigma2_val_gjr + 1e-12)
    gjr.calibrate_sigma(y_val, mu_val_gjr)
    sigma2_test_gjr = gjr.forecast_rolling(r_test)
    mu_test_gjr = 0.5 * np.log(sigma2_test_gjr + 1e-12)
    sigma_test_gjr = np.full_like(mu_test_gjr, fill_value=gjr.sigma_)
    results["gjr_garch_11"] = {
        "params": {
            "omega": float(gjr.omega),
            "alpha": float(gjr.alpha),
            "gamma": float(gjr.gamma),
            "beta": float(gjr.beta),
            "stationarity": float(gjr.alpha + gjr.beta + 0.5 * gjr.gamma),
            "converged": bool(gjr._converged),
            "nll_train": float(gjr._nll),
            "sigma_const": float(gjr.sigma_),
        },
        "metrics": eval_predictive(y_test, mu_test_gjr, sigma_test_gjr),
    }

    har_eval_path = ROOT_DIR / cfg["paths"]["har_results"] / cfg["paths"]["har_evaluation_results"]
    with open(har_eval_path, "r", encoding="utf-8") as f:
        har_eval = json.load(f)

    mlp_metrics = har_eval["baseline"]["global_metrics"]
    bayesian_mlp_metrics = har_eval["bayesian"]["global_metrics"]

    rows = [
        ("HAR-OLS", results["har_ols"]["metrics"]),
        ("EWMA (0.94)", results["ewma_094"]["metrics"]),
        ("GARCH(1,1)", results["garch_11"]["metrics"]),
        ("GJR-GARCH(1,1)", results["gjr_garch_11"]["metrics"]),
        ("MLP baseline", mlp_metrics),
        ("Bayesian MLP", bayesian_mlp_metrics),
    ]

    output = {
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "processed_dataset_path": str(processed_path),
        "har_evaluation_path": str(har_eval_path),
        "benchmarks": to_serializable(results),
        "mlp_baseline_metrics": to_serializable(mlp_metrics),
        "bayesian_mlp_metrics": to_serializable(bayesian_mlp_metrics),
    }

    with open(evaluation_results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print_header("RUN")
    print_kv("Script", "Benchmark evaluation")
    print_kv("Train samples", len(train_df))
    print_kv("Validation samples", len(val_df))
    print_kv("Test samples", len(test_df))
    print_kv("Target", "target_har")
    print_kv("Processed dataset", str(processed_path))

    print_header("MODEL SETUP")
    h = results["har_ols"]["params"]
    print_kv(
        "HAR-OLS",
        (
            f"intercept={h['intercept']:.4f} | "
            f"b_1d={h['coef_1d']:.4f} | "
            f"b_5d={h['coef_5d']:.4f} | "
            f"b_22d={h['coef_22d']:.4f} | "
            f"sigma={h['sigma_const']:.4f}"
        ),
    )

    e = results["ewma_094"]["params"]
    print_kv(
        "EWMA(0.94)",
        f"lambda={e['lambda']:.2f} | sigma={e['sigma_const']:.4f}",
    )

    g = results["garch_11"]["params"]
    print_kv(
        "GARCH(1,1)",
        (
            f"omega={g['omega']:.6f} | "
            f"alpha={g['alpha']:.4f} | "
            f"beta={g['beta']:.4f} | "
            f"alpha+beta={g['alpha_plus_beta']:.4f} | "
            f"converged={g['converged']}"
        ),
    )

    j = results["gjr_garch_11"]["params"]
    print_kv(
        "GJR-GARCH(1,1)",
        (
            f"omega={j['omega']:.6f} | "
            f"alpha={j['alpha']:.4f} | "
            f"gamma={j['gamma']:.4f} | "
            f"beta={j['beta']:.4f} | "
            f"stationarity={j['stationarity']:.4f} | "
            f"converged={j['converged']}"
        ),
    )

    print_header("MAIN RESULTS")
    print_benchmark_table("Benchmark comparison", rows, BENCHMARK_METRIC_KEYS)

    print_header("DIAGNOSTICS")
    print_kv("Reference MLP results", str(har_eval_path))
    print_kv("Included models", "HAR-OLS | EWMA | GARCH | GJR-GARCH | MLP | Bayesian MLP")

    print_header("ARTIFACTS")
    print_artifacts({
        "HAR / MLP evaluation JSON": str(har_eval_path),
        "Benchmark results JSON": str(evaluation_results_path),
    })