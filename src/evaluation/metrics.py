import numpy as np
from math import erf, pi, sqrt
from statistics import NormalDist


_EPS = 1e-8
_NORMAL = NormalDist()


def _to_1d_float_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"All arrays must have the same shape. Received: {shapes}")


def _clip_std(y_std: np.ndarray) -> np.ndarray:
    return np.clip(y_std, _EPS, None)


def _z_value(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1. Received: {level}")
    return _NORMAL.inv_cdf(0.5 + level / 2.0)


def _standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * x ** 2)


def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_1d_float_array(y_true, "y_true")
    y_pred = _to_1d_float_array(y_pred, "y_pred")
    _validate_same_shape(y_true, y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_1d_float_array(y_true, "y_true")
    y_pred = _to_1d_float_array(y_pred, "y_pred")
    _validate_same_shape(y_true, y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)


def build_gaussian_interval(
    y_mean: np.ndarray,
    y_std: np.ndarray,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    y_mean = _to_1d_float_array(y_mean, "y_mean")
    y_std = _clip_std(_to_1d_float_array(y_std, "y_std"))
    _validate_same_shape(y_mean, y_std)

    z = _z_value(level)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    return lower, upper


def compute_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    y_true = _to_1d_float_array(y_true, "y_true")
    lower = _to_1d_float_array(lower, "lower")
    upper = _to_1d_float_array(upper, "upper")
    _validate_same_shape(y_true, lower, upper)

    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = _to_1d_float_array(lower, "lower")
    upper = _to_1d_float_array(upper, "upper")
    _validate_same_shape(lower, upper)

    width = upper - lower
    return float(np.mean(width))


def compute_coverage_at_levels(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict:
    metrics = {}
    for level in levels:
        lower, upper = build_gaussian_interval(y_mean, y_std, level=level)
        key = f"coverage_{int(round(level * 100))}"
        metrics[key] = compute_coverage(y_true, lower, upper)
    return metrics


def compute_nll(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> float:
    y_true = _to_1d_float_array(y_true, "y_true")
    y_mean = _to_1d_float_array(y_mean, "y_mean")
    y_std = _clip_std(_to_1d_float_array(y_std, "y_std"))
    _validate_same_shape(y_true, y_mean, y_std)

    var = y_std ** 2
    nll = 0.5 * np.log(2.0 * pi * var) + ((y_true - y_mean) ** 2) / (2.0 * var)
    return float(np.mean(nll))


def compute_crps(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> float:
    y_true = _to_1d_float_array(y_true, "y_true")
    y_mean = _to_1d_float_array(y_mean, "y_mean")
    y_std = _clip_std(_to_1d_float_array(y_std, "y_std"))
    _validate_same_shape(y_true, y_mean, y_std)

    z = (y_true - y_mean) / y_std
    pdf = _standard_normal_pdf(z)
    cdf = _standard_normal_cdf(z)

    crps = y_std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / sqrt(pi))
    return float(np.mean(crps))


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    default_coverage_level: float = 0.95,
) -> dict:
    lower, upper = build_gaussian_interval(y_mean, y_std, level=default_coverage_level)

    metrics = {
        "rmse": compute_rmse(y_true, y_mean),
        "mae": compute_mae(y_true, y_mean),
        "coverage": compute_coverage(y_true, lower, upper),
        "nll": compute_nll(y_true, y_mean, y_std),
        "crps": compute_crps(y_true, y_mean, y_std),
        "sharpness": compute_sharpness(lower, upper),
    }

    metrics.update(compute_coverage_at_levels(y_true, y_mean, y_std, levels=levels))
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = _to_1d_float_array(y_true, "y_true")
    y_pred = _to_1d_float_array(y_pred, "y_pred")
    _validate_same_shape(y_true, y_pred)

    mse = np.mean((y_true - y_pred) ** 2)

    return {
        "mse": float(mse),
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
    }


def compute_mean_baseline_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    y_train = _to_1d_float_array(y_train, "y_train")
    y_test = _to_1d_float_array(y_test, "y_test")

    mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
    return compute_regression_metrics(y_test, mean_pred)