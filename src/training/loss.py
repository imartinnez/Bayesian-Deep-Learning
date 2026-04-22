import torch
import torch.nn.functional as F


def heteroscedastic_gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
    min_logvar: float = -6.0,
    max_logvar: float = 2.0,
    beta: float = 0.5,
) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=min_logvar, max=max_logvar)
    variance = torch.exp(logvar)

    loss = 0.5 * ((y - mu)**2 / variance + logvar)

    if beta > 0:
        loss = loss * (variance.detach()**beta)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    
    return loss

def cross_entropy_with_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weights, reduction=reduction)