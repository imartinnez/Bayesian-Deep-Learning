from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import build_gaussian_interval, compute_coverage, compute_nll


def _default_levels() -> tuple[float, ...]:
    return (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)


def compute_calibration_data(
    y_true: np.ndarray,
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    levels: tuple[float, ...] | list[float] | None = None,
) -> dict:
    if levels is None:
        levels = _default_levels()

    nominal = []
    empirical = []

    for level in levels:
        lower, upper = build_gaussian_interval(
            y_mean=predictive_mean,
            y_std=predictive_std,
            level=level,
        )
        coverage = compute_coverage(y_true=y_true, lower=lower, upper=upper)

        nominal.append(float(level))
        empirical.append(float(coverage))

    return {
        "nominal": nominal,
        "empirical": empirical,
    }

def calibrate_temperature(
        y_true: np.ndarray,
        predictive_mean: np.ndarray,
        predictive_std: np.ndarray,
        target_coverage: float = 0.90,
        bounds: tuple[float, float] = (0.1, 10),
) -> float:
    def coverage_error(tau):
        lower, upper = build_gaussian_interval(
            y_mean=predictive_mean,
            y_std=predictive_std / tau,
            level=target_coverage,
        )
        empirical = compute_coverage(y_true, lower, upper)
        return (empirical - target_coverage)**2

    result = minimize_scalar(coverage_error,
                             bounds=bounds,
                             method="Bounded")
    return float(result.x)

def plot_calibration(
    calibration_data: dict,
    save_path: str | Path | None = None,
) -> None:
    nominal = np.asarray(calibration_data["nominal"], dtype=float).reshape(-1)
    empirical = np.asarray(calibration_data["empirical"], dtype=float).reshape(-1)

    if nominal.size == 0 or empirical.size == 0:
        raise ValueError("calibration_data cannot be empty.")

    if nominal.shape != empirical.shape:
        raise ValueError(
            f"'nominal' and 'empirical' must have the same shape. "
            f"Received: {nominal.shape} and {empirical.shape}"
        )

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.plot(nominal, empirical, marker="o", linewidth=1.8, label="Model")

    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Calibration plot")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()