"""Regulatory-style risk tests: Kupiec, Christoffersen, ES, Gneiting score.

All tests take as input:
    returns:    realized 1-step-ahead returns r_{t+1}
    var_alpha:  predicted VaR at confidence alpha (one-sided lower tail, POSITIVE magnitude)
    es_alpha:   predicted Expected Shortfall at alpha (POSITIVE magnitude)

An "exception" is r_{t+1} < -var_alpha (actual loss exceeded forecast VaR).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def kupiec_pof(exceptions: np.ndarray, alpha: float) -> dict:
    """Kupiec proportion-of-failures (unconditional coverage) test.

    H0: exception rate = 1 - alpha  (e.g., 1% for VaR_99)
    LR_UC ~ chi2(1) under H0.
    """
    x = np.asarray(exceptions, dtype=bool)
    n = len(x)
    n_ex = int(x.sum())
    p = 1.0 - alpha
    pi_hat = n_ex / n if n > 0 else 0.0

    if n_ex == 0 or n_ex == n:
        lr_uc = -2.0 * (n_ex * np.log(p) + (n - n_ex) * np.log(1.0 - p))
    else:
        ll_null = n_ex * np.log(p) + (n - n_ex) * np.log(1.0 - p)
        ll_alt  = n_ex * np.log(pi_hat) + (n - n_ex) * np.log(1.0 - pi_hat)
        lr_uc = -2.0 * (ll_null - ll_alt)

    pval = float(1.0 - chi2.cdf(lr_uc, df=1))
    return {
        "n": n,
        "exceptions": n_ex,
        "rate_empirical": float(pi_hat),
        "rate_expected": float(p),
        "LR_uc": float(lr_uc),
        "p_value": pval,
        "reject_at_5pct": bool(pval < 0.05),
    }


def christoffersen_independence(exceptions: np.ndarray) -> dict:
    """Christoffersen independence test on exception clustering.

    Build 2x2 transition counts (i -> j) where i,j in {0,1} (no-exc, exc).
    H0: pi_01 == pi_11 (exceptions are iid). LR_IND ~ chi2(1).
    """
    x = np.asarray(exceptions, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(x)):
        if x[t-1] == 0 and x[t] == 0: n00 += 1
        if x[t-1] == 0 and x[t] == 1: n01 += 1
        if x[t-1] == 1 and x[t] == 0: n10 += 1
        if x[t-1] == 1 and x[t] == 1: n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    pi_01 = n01 / n0 if n0 > 0 else 0.0
    pi_11 = n11 / n1 if n1 > 0 else 0.0
    pi    = (n01 + n11) / max(n0 + n1, 1)

    def _log_safe(v):
        return np.log(v) if v > 0 else 0.0

    ll_alt  = (n00 * _log_safe(1 - pi_01) + n01 * _log_safe(pi_01)
             + n10 * _log_safe(1 - pi_11) + n11 * _log_safe(pi_11))
    ll_null = ((n00 + n10) * _log_safe(1 - pi) + (n01 + n11) * _log_safe(pi))

    lr_ind = -2.0 * (ll_null - ll_alt)
    if np.isnan(lr_ind) or np.isinf(lr_ind):
        lr_ind = 0.0
    pval = float(1.0 - chi2.cdf(lr_ind, df=1))

    return {
        "n01": int(n01), "n11": int(n11),
        "pi_01": float(pi_01), "pi_11": float(pi_11),
        "LR_ind": float(lr_ind),
        "p_value": pval,
        "reject_at_5pct": bool(pval < 0.05),
    }


def christoffersen_cc(exceptions: np.ndarray, alpha: float) -> dict:
    """Christoffersen conditional-coverage test = Kupiec + Independence.

    LR_CC = LR_UC + LR_IND ~ chi2(2)
    """
    uc = kupiec_pof(exceptions, alpha)
    ind = christoffersen_independence(exceptions)
    lr_cc = uc["LR_uc"] + ind["LR_ind"]
    pval = float(1.0 - chi2.cdf(lr_cc, df=2))
    return {
        "LR_cc": float(lr_cc),
        "p_value": pval,
        "reject_at_5pct": bool(pval < 0.05),
        "uc": uc,
        "ind": ind,
    }


def expected_shortfall_empirical(returns: np.ndarray, var_alpha: np.ndarray) -> float:
    """Empirical ES: mean of realized returns on exception days."""
    mask = returns < -var_alpha
    if not np.any(mask):
        return float("nan")
    return float(-np.mean(returns[mask]))


def quantile_loss(returns: np.ndarray, var_alpha: np.ndarray, alpha: float) -> float:
    """Pinball / Gneiting-consistent quantile loss at level 1-alpha.

    q = -var_alpha (predicted lower tail quantile).
    L = mean( (q - r) * (alpha1 - I(r < q)) )  where alpha1 = 1 - alpha.
    """
    q = -var_alpha
    alpha1 = 1.0 - alpha
    diff = q - returns
    ind = (returns < q).astype(float)
    return float(np.mean(diff * (alpha1 - ind)))


def fz0_loss(returns: np.ndarray, var_alpha: np.ndarray, es_alpha: np.ndarray,
             alpha: float) -> float:
    """Fissler-Ziegel (2016) loss for joint (VaR, ES), homogeneity 0.

    L_t = (I(r_t < -V_t) / (alpha1*E_t)) * (E_t - V_t + (V_t - r_t)_+ / alpha1) + log(E_t) - 1
    where alpha1 = 1 - alpha. Lower is better. Requires ES > 0.
    """
    alpha1 = 1.0 - alpha
    ind = (returns < -var_alpha).astype(float)
    es_pos = np.clip(es_alpha, 1e-8, None)
    hinge = np.maximum(-var_alpha - returns, 0.0)
    term1 = (ind / (alpha1 * es_pos)) * (es_pos - var_alpha + hinge / alpha1)
    term2 = np.log(es_pos) - 1.0
    return float(np.mean(term1 + term2))


def conditional_coverage_by_decile(y_true: np.ndarray, y_mean: np.ndarray,
                                   y_std: np.ndarray, level: float = 0.90,
                                   n_bins: int = 10) -> dict:
    """Empirical coverage at the given level, stratified by decile of y_std.

    A well-calibrated conditional model should show flat coverage ~ level across
    all deciles. A model with constant sigma will show monotonic drift.
    """
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + level / 2.0)

    idx = np.argsort(y_std)
    n = len(y_std)
    bins = np.array_split(idx, n_bins)

    decile_cov = []
    decile_mean_sigma = []
    decile_empirical_abs_err = []
    for b in bins:
        s = y_std[b]
        m = y_mean[b]
        t = y_true[b]
        lower = m - z * s
        upper = m + z * s
        cov = float(np.mean((t >= lower) & (t <= upper)))
        decile_cov.append(cov)
        decile_mean_sigma.append(float(np.mean(s)))
        decile_empirical_abs_err.append(float(np.mean(np.abs(t - m))))

    return {
        "level": float(level),
        "n_bins": int(n_bins),
        "coverage_by_decile": decile_cov,
        "mean_sigma_by_decile": decile_mean_sigma,
        "mean_abs_err_by_decile": decile_empirical_abs_err,
        "cov_std_dev": float(np.std(decile_cov, ddof=0)),
        "cov_max_deviation": float(np.max(np.abs(np.array(decile_cov) - level))),
    }
