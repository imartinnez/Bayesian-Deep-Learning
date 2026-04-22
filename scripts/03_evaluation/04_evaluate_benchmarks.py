"""Fit and evaluate classical volatility benchmarks on the same HAR split.

Benchmarks: HAR-OLS (Corsi 2009), EWMA (RiskMetrics, lambda=0.94),
            GARCH(1,1), GJR-GARCH(1,1).

All models produce a Gaussian predictive over target_har = log(r_{t+1}^2 + eps),
with sigma calibrated as the std of validation residuals — same convention as
the MLP baseline in script 12.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

from src.data.download import load_config
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def add_interval_metrics(metrics: dict, y_true, y_mean, y_std) -> dict:
    for level in (0.80, 0.90, 0.95):
        lower, upper = build_gaussian_interval(y_mean=y_mean, y_std=y_std, level=level)
        pct = int(level * 100)
        metrics[f"coverage_{pct}"] = compute_coverage(y_true=y_true, lower=lower, upper=upper)
        metrics[f"sharpness_{pct}"] = compute_sharpness(lower=lower, upper=upper)
    return metrics


def build_har_splits(parquet_path: Path, train_end: str, val_end: str,
                     epsilon: float, horizon: int):
    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    r_sq = df["log_return"].pow(2)
    df["log_rv_1d_har"]  = np.log(r_sq.rolling(1).mean())
    df["log_rv_5d_har"]  = np.log(r_sq.rolling(5).mean())
    df["log_rv_22d_har"] = np.log(r_sq.rolling(22).mean())
    df["target_har"]     = np.log(r_sq.shift(-horizon) + epsilon)

    needed = ["log_return", "log_rv_1d_har", "log_rv_5d_har", "log_rv_22d_har", "target_har"]
    df = df.dropna(subset=needed)

    train_df = df[df.index <= train_end]
    val_df   = df[(df.index > train_end) & (df.index <= val_end)]
    test_df  = df[df.index > val_end]
    return train_df, val_df, test_df


def eval_predictive(y_true, y_mean, y_std):
    m = compute_probabilistic_metrics(
        y_true=y_true, y_mean=y_mean, y_std=y_std, levels=(0.80, 0.90, 0.95)
    )
    m = add_interval_metrics(m, y_true, y_mean, y_std)
    return m


if __name__ == "__main__":
    cfg = load_config(ROOT_DIR / "config/config.yaml")

    parquet_path = ROOT_DIR / cfg["paths"]["processed"] / cfg["paths"]["processed_filename"]
    train_df, val_df, test_df = build_har_splits(
        parquet_path=parquet_path,
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
        epsilon=cfg["har"]["epsilon_squared_return"],
        horizon=cfg["har"]["horizon"],
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    har_cols = ["log_rv_1d_har", "log_rv_5d_har", "log_rv_22d_har"]
    X_train_har = train_df[har_cols].values
    X_val_har   = val_df[har_cols].values
    X_test_har  = test_df[har_cols].values

    y_train = train_df["target_har"].values
    y_val   = val_df["target_har"].values
    y_test  = test_df["target_har"].values

    r_train = train_df["log_return"].values
    r_val   = val_df["log_return"].values
    r_test  = test_df["log_return"].values

    results: dict = {}

    # ── HAR-OLS ────────────────────────────────────────────────────────────
    har = HAROLSVolatility().fit(X_train_har, y_train)
    har.calibrate_sigma(X_val_har, y_val)
    mu_test_har, sigma_test_har = har.predict(X_test_har)
    results["har_ols"] = {
        "params": {
            "intercept": har.intercept_,
            "coef_1d":   float(har.coef_[0]),
            "coef_5d":   float(har.coef_[1]),
            "coef_22d":  float(har.coef_[2]),
            "sigma_const": har.sigma_,
        },
        "metrics": eval_predictive(y_test, mu_test_har, sigma_test_har),
    }

    # ── EWMA (RiskMetrics, lambda=0.94) ────────────────────────────────────
    ewma = EWMAVolatility(lam=0.94).fit(r_train)
    sigma2_val_ewma  = ewma.forecast_rolling(r_val)
    mu_val_ewma      = np.log(sigma2_val_ewma + 1e-12)
    ewma.calibrate_sigma(y_val, mu_val_ewma)
    sigma2_test_ewma = ewma.forecast_rolling(r_test)
    mu_test_ewma, sigma_test_ewma = ewma.predict_from_variance(sigma2_test_ewma)
    results["ewma_094"] = {
        "params": {"lambda": ewma.lam, "sigma_const": ewma.sigma_},
        "metrics": eval_predictive(y_test, mu_test_ewma, sigma_test_ewma),
    }

    # ── GARCH(1,1) ─────────────────────────────────────────────────────────
    garch = GARCH11Volatility().fit(r_train)
    sigma2_val_garch  = garch.forecast_rolling(r_val)
    mu_val_garch      = np.log(sigma2_val_garch + 1e-12)
    garch.calibrate_sigma(y_val, mu_val_garch)
    sigma2_test_garch = garch.forecast_rolling(r_test)
    mu_test_garch, sigma_test_garch = garch.predict_from_variance(sigma2_test_garch)
    results["garch_11"] = {
        "params": {
            "omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta,
            "alpha_plus_beta": garch.alpha + garch.beta,
            "converged": garch._converged, "nll_train": garch._nll,
            "sigma_const": garch.sigma_,
        },
        "metrics": eval_predictive(y_test, mu_test_garch, sigma_test_garch),
    }

    # ── GJR-GARCH(1,1) ─────────────────────────────────────────────────────
    gjr = GJRGARCH11Volatility().fit(r_train)
    sigma2_val_gjr  = gjr.forecast_rolling(r_val)
    mu_val_gjr      = np.log(sigma2_val_gjr + 1e-12)
    gjr.calibrate_sigma(y_val, mu_val_gjr)
    sigma2_test_gjr = gjr.forecast_rolling(r_test)
    mu_test_gjr, sigma_test_gjr = gjr.predict_from_variance(sigma2_test_gjr)
    results["gjr_garch_11"] = {
        "params": {
            "omega": gjr.omega, "alpha": gjr.alpha, "gamma": gjr.gamma, "beta": gjr.beta,
            "stationarity": gjr.alpha + gjr.beta + 0.5 * gjr.gamma,
            "converged": gjr._converged, "nll_train": gjr._nll,
            "sigma_const": gjr.sigma_,
        },
        "metrics": eval_predictive(y_test, mu_test_gjr, sigma_test_gjr),
    }

    # ── Pull MLP/BMLP from existing HAR evaluation ────────────────────────
    har_eval_path = ROOT_DIR / cfg["paths"]["har_results"] / cfg["paths"]["har_evaluation_results"]
    with open(har_eval_path) as f:
        har_eval = json.load(f)
    mlp_m  = har_eval["baseline"]["global_metrics"]
    bmlp_m = har_eval["bayesian"]["global_metrics"]

    # ── Print comparison table ────────────────────────────────────────────
    rows = [
        ("HAR-OLS",          results["har_ols"]["metrics"]),
        ("EWMA (lam=0.94)",  results["ewma_094"]["metrics"]),
        ("GARCH(1,1)",       results["garch_11"]["metrics"]),
        ("GJR-GARCH(1,1)",   results["gjr_garch_11"]["metrics"]),
        ("MLP baseline",     mlp_m),
        ("Bayesian MLP",     bmlp_m),
    ]
    keys = ["rmse", "mae", "nll", "crps", "coverage_80", "coverage_90", "coverage_95", "sharpness_90"]

    print("\n=== GARCH / GJR estimation ===")
    g = results["garch_11"]["params"]
    print(f"  GARCH(1,1)    omega={g['omega']:.6f} alpha={g['alpha']:.4f} beta={g['beta']:.4f}  "
          f"a+b={g['alpha_plus_beta']:.4f}  converged={g['converged']}")
    j = results["gjr_garch_11"]["params"]
    print(f"  GJR-GARCH     omega={j['omega']:.6f} alpha={j['alpha']:.4f} gamma={j['gamma']:.4f} "
          f"beta={j['beta']:.4f}  stationarity={j['stationarity']:.4f}  converged={j['converged']}")

    print("\n=== HAR-OLS coefficients ===")
    h = results["har_ols"]["params"]
    print(f"  intercept={h['intercept']:.4f}  b_1d={h['coef_1d']:.4f}  "
          f"b_5d={h['coef_5d']:.4f}  b_22d={h['coef_22d']:.4f}  sigma_const={h['sigma_const']:.4f}")

    col_w = 13
    print("\n" + "=" * (22 + col_w * len(keys)))
    print(f"{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in keys))
    print("-" * (22 + col_w * len(keys)))
    for name, m in rows:
        print(f"{name:<22}" + "".join(f"{m.get(k, float('nan')):>{col_w}.4f}" for k in keys))
    print("=" * (22 + col_w * len(keys)))

    # ── Save ─────────────────────────────────────────────────────────────
    def _json_default(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")

    out = {
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "benchmarks": results,
        "mlp_baseline_metrics": mlp_m,
        "bayesian_mlp_metrics": bmlp_m,
    }
    out_path = ROOT_DIR / cfg["paths"]["har_results"] / "evaluation_results_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, default=_json_default)

    print(f"\nResults saved to: {out_path}")
