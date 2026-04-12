import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def compute_mean_baseline_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))
    return compute_regression_metrics(y_test, mean_pred)