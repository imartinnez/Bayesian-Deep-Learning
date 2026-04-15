import numpy as np


def decompose_uncertainty(mu_samples: np.ndarray, sigma2_samples: np.ndarray) -> dict:
    """
    Decompose predictive uncertainty into epistemic, aleatoric, and total parts.

    Args:
        mu_samples: Array of shape (T, batch)
        sigma2_samples: Array of shape (T, batch)

    Returns:
        Dictionary with:
            predictive_mean: mean of mu samples
            epistemic_uncertainty: variance of mu samples
            aleatoric_uncertainty: mean of predicted variances
            total_uncertainty: epistemic + aleatoric
            predictive_std: sqrt(total_uncertainty)
    """
    mu_samples = np.asarray(mu_samples, dtype=float)
    sigma2_samples = np.asarray(sigma2_samples, dtype=float)

    if mu_samples.ndim != 2:
        raise ValueError(f"mu_samples must have shape (T, batch). Received: {mu_samples.shape}")
    if sigma2_samples.ndim != 2:
        raise ValueError(f"sigma2_samples must have shape (T, batch). Received: {sigma2_samples.shape}")
    if mu_samples.shape != sigma2_samples.shape:
        raise ValueError(
            f"mu_samples and sigma2_samples must have the same shape. "
            f"Received: {mu_samples.shape} and {sigma2_samples.shape}"
        )

    predictive_mean = mu_samples.mean(axis=0)
    epistemic_uncertainty = mu_samples.var(axis=0)
    aleatoric_uncertainty = sigma2_samples.mean(axis=0)
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
    predictive_std = np.sqrt(total_uncertainty)

    return {
        "predictive_mean": predictive_mean,
        "epistemic_uncertainty": epistemic_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "total_uncertainty": total_uncertainty,
        "predictive_std": predictive_std,
    }