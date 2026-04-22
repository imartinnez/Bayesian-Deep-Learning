import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import softmax as scipy_softmax


def _log_p(p_mean: np.ndarray) -> np.ndarray:
    # log(p_mean) son pseudo-logits válidos para temperature scaling:
    # softmax(log(p)/T) es matemáticamente equivalente a softmax(logits/T)
    # porque softmax es invariante a constantes aditivas.
    return np.log(np.clip(p_mean, 1e-8, 1.0))


def apply_temperature(p_mean: np.ndarray, tau: float) -> np.ndarray:
    return scipy_softmax(_log_p(p_mean) / tau, axis=-1)


def calibrate_temperature_classification(
    y_true: np.ndarray,
    p_mean: np.ndarray,
    bounds: tuple[float, float] = (0.1, 10.0),
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    log_p  = _log_p(p_mean)

    def nll(tau: float) -> float:
        p_cal = scipy_softmax(log_p / tau, axis=-1)
        correct_probs = p_cal[np.arange(len(y_true)), y_true]
        return -float(np.mean(np.log(np.clip(correct_probs, 1e-8, 1.0))))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_mean: np.ndarray,
    n_bins: int = 15,
) -> float:
    y_true     = np.asarray(y_true, dtype=int)
    y_pred     = np.asarray(y_pred, dtype=int)
    confidence = np.max(p_mean, axis=-1)
    correct    = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        acc  = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def compute_reliability_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_mean: np.ndarray,
    n_bins: int = 15,
) -> dict:
    y_true     = np.asarray(y_true, dtype=int)
    y_pred     = np.asarray(y_pred, dtype=int)
    confidence = np.max(p_mean, axis=-1)
    correct    = (y_pred == y_true).astype(float)

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc     = []
    bin_conf    = []
    bin_counts  = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        bin_counts.append(int(mask.sum()))
        if mask.any():
            bin_acc.append(float(correct[mask].mean()))
            bin_conf.append(float(confidence[mask].mean()))
        else:
            bin_acc.append(float("nan"))
            bin_conf.append(float((lo + hi) / 2))

    return {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_counts": bin_counts}


def plot_reliability_diagram(
    reliability_data: dict,
    ece: float,
    title: str = "Reliability diagram",
    save_path: str | Path | None = None,
) -> None:
    bin_conf   = np.asarray(reliability_data["bin_conf"])
    bin_acc    = np.asarray(reliability_data["bin_acc"])
    bin_counts = np.asarray(reliability_data["bin_counts"])

    valid = ~np.isnan(bin_acc)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.bar(bin_conf[valid], bin_acc[valid], width=1.0 / len(bin_counts),
            alpha=0.6, label=f"Model (ECE={ece:.4f})", align="center")
    plt.xlabel("Confidence (max probability)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()