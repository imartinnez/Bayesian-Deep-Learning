import torch

def heteroscedastic_gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
    min_logvar: float = -6.0,
    max_logvar: float = 2.0,
    logvar_reg: float = 0.01,
) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=min_logvar, max=max_logvar)
    loss = 0.5 * (logvar + ((y - mu) ** 2) / torch.exp(logvar))

    if logvar_reg > 0.0:
        loss = loss + logvar_reg * logvar.pow(2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")