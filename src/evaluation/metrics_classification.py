import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)
from scipy.stats import pearsonr


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_mean: np.ndarray,
    n_classes: int = 3,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss":          float(log_loss(y_true, p_mean, labels=list(range(n_classes)))),
        "brier_score":       float(_brier_multiclass(y_true, p_mean, n_classes)),
        "confusion_matrix":  confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }

    for c in range(n_classes):
        label = {0: "LOW", 1: "MED", 2: "HIGH"}[c]
        metrics[f"precision_{label}"] = float(precision_score(y_true, y_pred, labels=[c], average="micro", zero_division=0))
        metrics[f"recall_{label}"]    = float(recall_score(   y_true, y_pred, labels=[c], average="micro", zero_division=0))
        metrics[f"f1_{label}"]        = float(f1_score(       y_true, y_pred, labels=[c], average="micro", zero_division=0))

    try:
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, p_mean, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["roc_auc_ovr"] = float("nan")

    return metrics


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_entropy: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    H      = np.asarray(total_entropy, dtype=float)

    is_error = (y_pred != y_true).astype(float)

    corr, _ = pearsonr(H, is_error)

    # AUROC: H como score para detectar errores
    try:
        from sklearn.metrics import roc_auc_score as _auc
        auroc_error = float(_auc(is_error, H))
    except ValueError:
        auroc_error = float("nan")

    # Accuracy por cuartil de incertidumbre
    quartiles = np.percentile(H, [25, 50, 75])
    q1_mask = H <= quartiles[0]
    q4_mask = H > quartiles[2]

    acc_low_uncertainty  = float(accuracy_score(y_true[q1_mask], y_pred[q1_mask])) if q1_mask.any() else float("nan")
    acc_high_uncertainty = float(accuracy_score(y_true[q4_mask], y_pred[q4_mask])) if q4_mask.any() else float("nan")

    H_cv = float(np.std(H) / np.mean(H)) if np.mean(H) > 0 else float("nan")

    return {
        "H_total_mean":          float(np.mean(H)),
        "H_total_std":           float(np.std(H)),
        "H_total_cv":            H_cv,
        "corr_H_error":          float(corr),
        "auroc_H_as_error_det":  auroc_error,
        "acc_low_uncertainty_q1": acc_low_uncertainty,
        "acc_high_uncertainty_q4": acc_high_uncertainty,
    }


def _brier_multiclass(y_true: np.ndarray, p_mean: np.ndarray, n_classes: int) -> float:
    n = len(y_true)
    y_onehot = np.zeros((n, n_classes), dtype=float)
    y_onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((p_mean - y_onehot) ** 2, axis=-1)))