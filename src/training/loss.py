import torch

def heteroscedastic_gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
    min_logvar: float = -10.0,
    max_logvar: float = 10.0,
) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=min_logvar, max=max_logvar)
    loss = 0.5 * (logvar + ((y - mu) ** 2) / torch.exp(logvar))

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")