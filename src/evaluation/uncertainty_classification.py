import numpy as np

_EPS = 1e-8


def decompose_uncertainty_classification(p_samples: np.ndarray) -> dict:
    """
    Decompose predictive uncertainty for classification via MC Dropout.

    Args:
        p_samples: shape (T, N, C) — softmax probabilities from T MC passes,
                   N samples, C classes.

    Returns dict with:
        p_mean            (N, C)  mean predictive distribution
        y_pred            (N,)    predicted class — argmax of p_mean
        total_entropy     (N,)    H[p_mean]          — total uncertainty
        aleatoric_entropy (N,)    E_t[H[p_t]]        — aleatoric uncertainty
        mutual_information(N,)    MI = total-aleatoric — epistemic uncertainty
    """
    p_samples = np.asarray(p_samples, dtype=float)

    if p_samples.ndim != 3:
        raise ValueError(f"p_samples must have shape (T, N, C). Got: {p_samples.shape}")

    p_samples = np.clip(p_samples, _EPS, 1.0)

    p_mean = p_samples.mean(axis=0)          # (N, C)
    p_mean = np.clip(p_mean, _EPS, 1.0)

    y_pred = np.argmax(p_mean, axis=-1)      # (N,)

    # H[p_mean] — entropía de la distribución predictiva media
    total_entropy = -np.sum(p_mean * np.log(p_mean), axis=-1)          # (N,)

    # E_t[H[p_t]] — entropía esperada por cada pasada MC
    per_pass_entropy = -np.sum(p_samples * np.log(p_samples), axis=-1) # (T, N)
    aleatoric_entropy = per_pass_entropy.mean(axis=0)                   # (N,)

    # MI = H_total - H_aleatoric — siempre >= 0 por desigualdad de Jensen
    mutual_information = np.clip(total_entropy - aleatoric_entropy, 0.0, None)

    return {
        "p_mean": p_mean,
        "y_pred": y_pred,
        "total_entropy": total_entropy,
        "aleatoric_entropy": aleatoric_entropy,
        "mutual_information": mutual_information,
    }