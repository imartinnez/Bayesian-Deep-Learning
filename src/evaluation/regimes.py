import numpy as np

from src.evaluation.metrics import build_gaussian_interval, compute_coverage, compute_crps
from src.evaluation.metrics import compute_sharpness, compute_nll


def classify_regimes(
    rv_series: np.ndarray,
    train_rv: np.ndarray,
    low_pct: float = 33,
    high_pct: float = 67,
) -> np.ndarray:
    rv_series = np.asarray(rv_series, dtype=float).reshape(-1)
    train_rv = np.asarray(train_rv, dtype=float).reshape(-1)

    if rv_series.size == 0:
        raise ValueError("rv_series cannot be empty.")
    if train_rv.size == 0:
        raise ValueError("train_rv cannot be empty.")
    if not 0 <= low_pct <= 100:
        raise ValueError(f"low_pct must be between 0 and 100. Received: {low_pct}")
    if not 0 <= high_pct <= 100:
        raise ValueError(f"high_pct must be between 0 and 100. Received: {high_pct}")
    if low_pct >= high_pct:
        raise ValueError(
            f"low_pct must be smaller than high_pct. Received: {low_pct} and {high_pct}"
        )

    low_threshold = np.percentile(train_rv, low_pct)
    high_threshold = np.percentile(train_rv, high_pct)

    regimes = np.full(rv_series.shape, "MED", dtype=object)
    regimes[rv_series <= low_threshold] = "LOW"
    regimes[rv_series >= high_threshold] = "HIGH"

    return regimes


def evaluate_by_regime(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    regimes: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    regimes = np.asarray(regimes).reshape(-1)

    shapes = [y_true.shape, mu.shape, sigma.shape, regimes.shape]
    if len(set(shapes)) != 1:
        raise ValueError(f"All inputs must have the same shape. Received: {shapes}")

    results = {}

    for regime_name in ("LOW", "MED", "HIGH"):
        mask = regimes == regime_name
        n_obs = int(mask.sum())

        if n_obs == 0:
            results[regime_name] = {
                "n_obs": 0,
                "coverage_90": float("nan"),
                "crps": float("nan"),
                "sharpness_90": float("nan"),
                "nll": float("nan"),
            }
            continue

        y_true_regime = y_true[mask]
        mu_regime = mu[mask]
        sigma_regime = sigma[mask]

        lower_90, upper_90 = build_gaussian_interval(
            y_mean=mu_regime,
            y_std=sigma_regime,
            level=0.90,
        )

        results[regime_name] = {
            "n_obs": n_obs,
            "coverage_90": compute_coverage(y_true_regime, lower_90, upper_90),
            "crps": compute_crps(y_true_regime, mu_regime, sigma_regime),
            "sharpness_90": compute_sharpness(lower_90, upper_90),
            "nll": compute_nll(y_true_regime, mu_regime, sigma_regime),
        }

    return results