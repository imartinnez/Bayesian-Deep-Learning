# @author: Inigo Martinez Jimenez
# Thesis figures: per-model predictions, learning curves,
# uncertainty decomposition, and cross-model comparison.

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from src.visualization import palette as P
from src.visualization import primitives as pr
from src.visualization.baselines import BaselineForecast
from src.visualization.io import ensure_dir, save_figure
from src.visualization.predictions import BayesianPrediction, DeterministicPrediction
from src.visualization.theme import FIGURE_SIZES


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _clip_ylim(arrays: list[np.ndarray], lo: float = 4.0, hi: float = 99.0,
               pad: float = 0.3) -> tuple[float, float]:
    """Y-axis limits based on percentiles across all arrays, ignoring extreme outliers."""
    combined = np.concatenate([a[np.isfinite(a)] for a in arrays])
    return (float(np.percentile(combined, lo)) - pad,
            float(np.percentile(combined, hi)) + pad)


def _smooth(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Rolling mean to show underlying trend without the spike noise."""
    return pd.Series(arr).rolling(window, min_periods=1, center=True).mean().to_numpy()


def _legend_below(ax, ncols: int = 3) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=ncols,
              framealpha=0.0, fontsize=9, labelcolor=P.TEXT_SECONDARY,
              handlelength=2.2, columnspacing=1.6)


# ---------------------------------------------------------------------------
# 1. Deterministic model: actual vs predicted
#    Two layers: raw actual (faint) + smoothed actual (bright) + predicted
# ---------------------------------------------------------------------------


def det_vs_actual(pred: DeterministicPrediction, out_dir: Path) -> list[Path]:
    dates = pd.to_datetime(pred.dates)
    y_true = np.asarray(pred.y_true, dtype=float)
    y_pred = np.asarray(pred.y_pred, dtype=float)
    y_smooth = _smooth(y_true)
    color = P.MODEL_COLORS.get(pred.name, P.ACCENT_COOL)
    rmse = _rmse(y_true, y_pred)

    ylo, yhi = _clip_ylim([y_true, y_pred])

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    # Raw actual: very faint background reference
    ax.plot(dates, y_true, color=P.TEXT_PRIMARY, linewidth=0.6,
            alpha=0.20, zorder=2, label="_nolegend_")
    # Smoothed actual: main reference line
    ax.plot(dates, y_smooth, color=P.TEXT_PRIMARY, linewidth=1.8,
            alpha=0.85, zorder=5, label="Realized Vol (7d smooth)")
    # Prediction
    ax.plot(dates, y_pred, color=color, linewidth=2.0, linestyle="--",
            zorder=6, label=f"{pred.name} prediction")

    ax.set_ylim(ylo, yhi)
    ax.set_title(f"{pred.name}  ·  Actual vs Predicted  ·  RMSE = {rmse:.4f}  ·  test split",
                 fontsize=12, loc="left")
    pr.axis_label(ax, x="Date", y="Log Realized Volatility (h=5)")
    pr.format_date_axis(ax)
    _legend_below(ax, ncols=2)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, out_dir, f"pred_actual_{_slug(pred.name)}_{pred.split}")


# ---------------------------------------------------------------------------
# 2. Bayesian model: actual vs predicted + uncertainty band
# ---------------------------------------------------------------------------


def bay_vs_actual(pred: BayesianPrediction, out_dir: Path) -> list[Path]:
    dates = pd.to_datetime(pred.dates)
    y_true = np.asarray(pred.y_true, dtype=float)
    mean = np.asarray(pred.predictive_mean, dtype=float)
    sigma = np.asarray(pred.predictive_std, dtype=float)
    y_smooth = _smooth(y_true)
    color = P.MODEL_COLORS.get(pred.name, P.ACCENT_PRIMARY)
    rmse = _rmse(y_true, mean)

    ylo, yhi = _clip_ylim([y_true, mean])
    band_lo = np.clip(mean - 1.645 * sigma, ylo - 0.2, yhi + 0.2)
    band_hi = np.clip(mean + 1.645 * sigma, ylo - 0.2, yhi + 0.2)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    # Confidence band (clipped so it doesn't flood the plot)
    ax.fill_between(dates, band_lo, band_hi, color=color, alpha=0.18,
                    zorder=1, label="90% predictive interval")
    # Raw actual: faint
    ax.plot(dates, y_true, color=P.TEXT_PRIMARY, linewidth=0.6,
            alpha=0.20, zorder=2, label="_nolegend_")
    # Smoothed actual
    ax.plot(dates, y_smooth, color=P.TEXT_PRIMARY, linewidth=1.8,
            alpha=0.85, zorder=5, label="Realized Vol (7d smooth)")
    # Predicted mean
    ax.plot(dates, mean, color=color, linewidth=2.2,
            zorder=6, label=f"{pred.name} (mean)")

    ax.set_ylim(ylo, yhi)
    ax.set_title(f"{pred.name}  ·  Actual vs Predicted  ·  RMSE = {rmse:.4f}  ·  test split",
                 fontsize=12, loc="left")
    pr.axis_label(ax, x="Date", y="Log Realized Volatility (h=5)")
    pr.format_date_axis(ax)
    _legend_below(ax, ncols=3)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, out_dir, f"pred_actual_{_slug(pred.name)}_{pred.split}")


# ---------------------------------------------------------------------------
# 3. Bayesian MLP vs risk models — two-panel layout
#    Top:    Bayesian MLP mean + band vs actual
#    Bottom: risk models (HAR-RV, EWMA, Historical Vol) vs actual
# ---------------------------------------------------------------------------


def bayesian_vs_risk(
    pred: BayesianPrediction,
    baselines: dict[str, BaselineForecast],
    out_dir: Path,
) -> list[Path]:
    dates = pd.to_datetime(pred.dates)
    y_true = np.asarray(pred.y_true, dtype=float)
    mean = np.asarray(pred.predictive_mean, dtype=float)
    sigma = np.asarray(pred.predictive_std, dtype=float)
    y_smooth = _smooth(y_true)
    color = P.MODEL_COLORS.get(pred.name, P.ACCENT_PRIMARY)

    ylo, yhi = _clip_ylim([y_true, mean])
    band_lo = np.clip(mean - 1.645 * sigma, ylo - 0.2, yhi + 0.2)
    band_hi = np.clip(mean + 1.645 * sigma, ylo - 0.2, yhi + 0.2)

    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(2, 1, hspace=0.45)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ---- Top panel: Bayesian MLP ----
    ax_top.fill_between(dates, band_lo, band_hi, color=color,
                        alpha=0.20, zorder=1, label="90% interval")
    ax_top.plot(dates, y_true, color=P.TEXT_PRIMARY, linewidth=0.5,
                alpha=0.18, zorder=2)
    ax_top.plot(dates, y_smooth, color=P.TEXT_PRIMARY, linewidth=1.6,
                alpha=0.80, zorder=5, label="Realized Vol (7d smooth)")
    ax_top.plot(dates, mean, color=color, linewidth=2.0, zorder=6,
                label=f"{pred.name}")
    ax_top.set_ylim(ylo, yhi)
    ax_top.set_title(f"{pred.name}  ·  RMSE = {_rmse(y_true, mean):.4f}", fontsize=11, loc="left")
    pr.axis_label(ax_top, x="", y="Log RV (h=5)")
    pr.format_date_axis(ax_top)
    ax_top.legend(loc="upper right", framealpha=0.0, fontsize=9,
                  labelcolor=P.TEXT_SECONDARY, ncol=3)

    # ---- Bottom panel: risk models ----
    bl_styles = {
        "HAR-RV":              (P.MODEL_COLORS["HAR-RV"], "-", 1.8),
        "EWMA (RiskMetrics)":  (P.MODEL_COLORS["EWMA (RiskMetrics)"], "--", 1.5),
        "Historical Volatility": (P.MODEL_COLORS["Historical Volatility"], "-.", 1.4),
        "Mean Baseline":       (P.MODEL_COLORS["Mean Baseline"], ":", 1.2),
    }
    ax_bot.plot(dates, y_true, color=P.TEXT_PRIMARY, linewidth=0.5,
                alpha=0.18, zorder=2)
    ax_bot.plot(dates, y_smooth, color=P.TEXT_PRIMARY, linewidth=1.6,
                alpha=0.80, zorder=5, label="Realized Vol (7d smooth)")
    for name, bl in baselines.items():
        c, ls, lw = bl_styles.get(name, (P.TEXT_MUTED, "--", 1.2))
        ax_bot.plot(pd.to_datetime(bl.dates), np.asarray(bl.y_pred, float),
                    color=c, linewidth=lw, linestyle=ls, zorder=4,
                    label=f"{name}  (RMSE {_rmse(bl.y_true, bl.y_pred):.3f})")
    bl_preds = [np.asarray(bl.y_pred, float) for bl in baselines.values()]
    ylo_b, yhi_b = _clip_ylim([y_true] + bl_preds)
    ax_bot.set_ylim(ylo_b, yhi_b)
    ax_bot.set_title("Risk Model Baselines", fontsize=11, loc="left")
    pr.axis_label(ax_bot, x="Date", y="Log RV (h=5)")
    pr.format_date_axis(ax_bot)
    ax_bot.legend(loc="upper right", framealpha=0.0, fontsize=9,
                  labelcolor=P.TEXT_SECONDARY, ncol=2)

    fig.suptitle(f"{pred.name} vs Risk Models — Test Split",
                 x=0.02, ha="left", fontsize=13, color=P.TEXT_PRIMARY, fontweight="bold")
    fig.subplots_adjust(top=0.93)
    return save_figure(fig, out_dir, f"bay_vs_risk_{_slug(pred.name)}_{pred.split}")


# ---------------------------------------------------------------------------
# 4. Uncertainty decomposition: aleatoric + epistemic over time
# ---------------------------------------------------------------------------


def uncertainty_over_time(pred: BayesianPrediction, out_dir: Path) -> list[Path]:
    dates = pd.to_datetime(pred.dates)
    epi = np.asarray(pred.epistemic_uncertainty, dtype=float)
    ale = np.asarray(pred.aleatoric_uncertainty, dtype=float)
    total = ale + epi

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    ax.fill_between(dates, 0, ale,
                    color=P.UNCERTAINTY_ALEATORIC, alpha=0.75,
                    label=f"Aleatoric  (mean={np.nanmean(ale):.4f})")
    ax.fill_between(dates, ale, total,
                    color=P.UNCERTAINTY_EPISTEMIC, alpha=0.75,
                    label=f"Epistemic  (mean={np.nanmean(epi):.4f})")
    # Cap the y-axis so occasional spikes don't collapse the view
    ax.set_ylim(0, float(np.nanpercentile(total, 97)) * 1.15)
    ax.set_title(f"Uncertainty Decomposition — {pred.name}  ·  {pred.split} split",
                 fontsize=12, loc="left")
    pr.axis_label(ax, x="Date", y="Variance")
    pr.format_date_axis(ax)
    _legend_below(ax, ncols=2)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, out_dir, f"unc_time_{_slug(pred.name)}_{pred.split}")


# ---------------------------------------------------------------------------
# 5. Uncertainty vs absolute error
# ---------------------------------------------------------------------------


def uncertainty_vs_error(pred: BayesianPrediction, out_dir: Path) -> list[Path]:
    y_true = np.asarray(pred.y_true, dtype=float)
    y_pred = np.asarray(pred.predictive_mean, dtype=float)
    sigma = np.asarray(pred.total_uncertainty, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(sigma) & (sigma > 0)
    abs_err = np.abs(y_true[mask] - y_pred[mask])
    unc = sigma[mask]

    # Clip both axes to 98th percentile for readability
    unc_hi = float(np.percentile(unc, 98))
    err_hi = float(np.percentile(abs_err, 98))
    m = (unc <= unc_hi) & (abs_err <= err_hi)
    unc_c, err_c = unc[m], abs_err[m]

    slope, intercept, r, *_ = stats.linregress(unc_c, err_c)
    x_line = np.linspace(unc_c.min(), unc_c.max(), 200)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["square"])
    ax.scatter(unc_c, err_c, color=P.ACCENT_PRIMARY, alpha=0.25, s=14,
               linewidths=0, zorder=3, label=f"n={m.sum()} observations")
    ax.plot(x_line, intercept + slope * x_line, color=P.ACCENT_TEAL,
            linewidth=2.2, zorder=4, label=f"OLS trend  (r = {r:.2f})")
    ax.set_title(
        f"Uncertainty vs Error — {pred.name}  ·  {pred.split}\n"
        f"Positive r means uncertainty is informative",
        fontsize=11, loc="left")
    pr.axis_label(ax, x="Predictive uncertainty (σ)", y="|Error|")
    _legend_below(ax, ncols=2)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, out_dir, f"unc_vs_error_{_slug(pred.name)}_{pred.split}")


# ---------------------------------------------------------------------------
# 6. Scatter: predicted vs actual
# ---------------------------------------------------------------------------


def scatter_regression(
    pred: BayesianPrediction | DeterministicPrediction,
    out_dir: Path,
) -> list[Path]:
    y_pred = (np.asarray(pred.predictive_mean, float)
              if isinstance(pred, BayesianPrediction)
              else np.asarray(pred.y_pred, float))
    y_true = np.asarray(pred.y_true, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    r2 = float(np.corrcoef(yt, yp)[0, 1] ** 2)
    rmse = _rmse(yt, yp)
    color = P.MODEL_COLORS.get(pred.name, P.ACCENT_PRIMARY)

    lo, hi = _clip_ylim([yt, yp], lo=1, hi=99, pad=0.1)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["square"])
    ax.plot([lo, hi], [lo, hi], color=P.TEXT_SECONDARY, linewidth=1.2,
            linestyle="--", zorder=2, label="Perfect fit  (slope=1)")
    ax.scatter(yt, yp, color=color, alpha=0.30, s=14, linewidths=0,
               zorder=3, label="Observations")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(
        f"Scatter — {pred.name}  ·  {pred.split}\n"
        f"R² = {r2:.3f}   RMSE = {rmse:.4f}",
        fontsize=11, loc="left")
    pr.axis_label(ax, x="Actual log-RV", y="Predicted log-RV")
    _legend_below(ax, ncols=2)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, out_dir, f"scatter_{_slug(pred.name)}_{pred.split}")


# ---------------------------------------------------------------------------
# 7. Learning curves
# ---------------------------------------------------------------------------


def learning_curves(history: dict, name: str, out_dir: Path) -> list[Path]:
    train = np.asarray(history["train"], dtype=float)
    val = np.asarray(history["val"], dtype=float)
    if len(train) < 2:
        return []

    best = int(history.get("best_epoch", int(np.argmin(val))))
    color = P.MODEL_COLORS.get(name, P.ACCENT_PRIMARY)
    min_val = float(np.nanmin(val))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    ax.plot(train, color=color, linewidth=1.8, label=f"Train  (final={train[-1]:.4f})")
    ax.plot(val, color=color, linewidth=1.8, linestyle="--", alpha=0.8,
            label=f"Validation  (best={min_val:.4f})")
    ax.axvline(best, color=P.ACCENT_TEAL, linewidth=1.4, linestyle=":",
               zorder=5, label=f"Best checkpoint  (epoch {best})")
    ax.set_title(f"Learning Curves — {name}", fontsize=12, loc="left")
    pr.axis_label(ax, x="Epoch", y="Loss")
    _legend_below(ax, ncols=3)
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, out_dir, f"learning_curves_{_slug(name)}")


# ---------------------------------------------------------------------------
# 8. All-models comparison — two panels: neural (top) vs risk models (bottom)
# ---------------------------------------------------------------------------


def predictions_over_time(
    bayesian_pred: BayesianPrediction,
    deterministic_preds: list[DeterministicPrediction],
    baselines: dict[str, BaselineForecast],
    out_dir: Path,
) -> list[Path]:
    dates = pd.to_datetime(bayesian_pred.dates)
    y_true = np.asarray(bayesian_pred.y_true, dtype=float)
    y_smooth = _smooth(y_true)
    mean_bay = np.asarray(bayesian_pred.predictive_mean, dtype=float)
    color_bay = P.MODEL_COLORS.get(bayesian_pred.name, P.ACCENT_PRIMARY)

    neural_preds = [mean_bay] + [np.asarray(p.y_pred, float) for p in deterministic_preds]
    bl_preds = [np.asarray(bl.y_pred, float) for bl in baselines.values()]

    ylo_n, yhi_n = _clip_ylim([y_true] + neural_preds)
    ylo_b, yhi_b = _clip_ylim([y_true] + bl_preds)

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 1, hspace=0.45)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # ---- Top: neural models ----
    ax_top.plot(dates, y_true, color=P.TEXT_PRIMARY, linewidth=0.5,
                alpha=0.18, zorder=2)
    ax_top.plot(dates, y_smooth, color=P.TEXT_PRIMARY, linewidth=1.7,
                alpha=0.85, zorder=8, label="Realized Vol (7d smooth)")
    ax_top.plot(dates, mean_bay, color=color_bay, linewidth=2.2,
                zorder=7, label=f"{bayesian_pred.name}  (RMSE {_rmse(y_true, mean_bay):.3f})")
    for p in deterministic_preds:
        ax_top.plot(pd.to_datetime(p.dates), np.asarray(p.y_pred, float),
                    color=P.MODEL_COLORS.get(p.name, P.ACCENT_COOL),
                    linewidth=1.6, linestyle="--", zorder=5,
                    label=f"{p.name}  (RMSE {_rmse(p.y_true, p.y_pred):.3f})")
    ax_top.set_ylim(ylo_n, yhi_n)
    ax_top.set_title("Neural Models", fontsize=11, loc="left")
    pr.axis_label(ax_top, x="", y="Log RV (h=5)")
    pr.format_date_axis(ax_top)
    ax_top.legend(loc="upper right", framealpha=0.0, fontsize=9,
                  labelcolor=P.TEXT_SECONDARY, ncol=2)

    # ---- Bottom: risk models ----
    bl_styles = {
        "HAR-RV":              (P.MODEL_COLORS["HAR-RV"], "-", 1.8),
        "EWMA (RiskMetrics)":  (P.MODEL_COLORS["EWMA (RiskMetrics)"], "--", 1.6),
        "Historical Volatility": (P.MODEL_COLORS["Historical Volatility"], "-.", 1.5),
        "Mean Baseline":       (P.MODEL_COLORS["Mean Baseline"], ":", 1.3),
    }
    ax_bot.plot(dates, y_true, color=P.TEXT_PRIMARY, linewidth=0.5,
                alpha=0.18, zorder=2)
    ax_bot.plot(dates, y_smooth, color=P.TEXT_PRIMARY, linewidth=1.7,
                alpha=0.85, zorder=8, label="Realized Vol (7d smooth)")
    for name, bl in baselines.items():
        c, ls, lw = bl_styles.get(name, (P.TEXT_MUTED, "--", 1.2))
        ax_bot.plot(pd.to_datetime(bl.dates), np.asarray(bl.y_pred, float),
                    color=c, linewidth=lw, linestyle=ls, zorder=5,
                    label=f"{name}  (RMSE {_rmse(bl.y_true, bl.y_pred):.3f})")
    ax_bot.set_ylim(ylo_b, yhi_b)
    ax_bot.set_title("Risk Model Baselines", fontsize=11, loc="left")
    pr.axis_label(ax_bot, x="Date", y="Log RV (h=5)")
    pr.format_date_axis(ax_bot)
    ax_bot.legend(loc="upper right", framealpha=0.0, fontsize=9,
                  labelcolor=P.TEXT_SECONDARY, ncol=2)

    fig.suptitle("All Models vs Realized Volatility — Test Split",
                 x=0.02, ha="left", fontsize=13, color=P.TEXT_PRIMARY, fontweight="bold")
    fig.subplots_adjust(top=0.94)
    return save_figure(fig, out_dir, "predictions_over_time_test")


# ---------------------------------------------------------------------------
# 9. RMSE bar chart — restricted to common test dates across all models
# ---------------------------------------------------------------------------


def _common_rmse(
    dates_a: np.ndarray,
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    common_dates: pd.DatetimeIndex,
) -> tuple[float, int]:
    """RMSE computed only on dates that appear in common_dates."""
    idx = pd.DatetimeIndex(dates_a).isin(common_dates)
    yt, yp = y_true_a[idx], y_pred_a[idx]
    mask = np.isfinite(yt) & np.isfinite(yp)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2))), n


def comparison_rmse(
    bayesian_pred: BayesianPrediction,
    deterministic_preds: list[DeterministicPrediction],
    baselines: dict[str, BaselineForecast],
    out_dir: Path,
) -> list[Path]:
    # Build the intersection of test dates across all models so RMSE is
    # computed on the exact same observations.
    all_date_sets = [pd.DatetimeIndex(bayesian_pred.dates)]
    for p in deterministic_preds:
        all_date_sets.append(pd.DatetimeIndex(p.dates))
    for bl in baselines.values():
        all_date_sets.append(pd.DatetimeIndex(bl.dates))

    common = all_date_sets[0]
    for ds in all_date_sets[1:]:
        common = common.intersection(ds)

    names, rmses, ns, colors = [], [], [], []

    for pred in [bayesian_pred] + deterministic_preds:  # type: ignore[operator]
        y_p = (np.asarray(pred.predictive_mean, float)
               if isinstance(pred, BayesianPrediction) else np.asarray(pred.y_pred, float))
        r, n = _common_rmse(pred.dates, np.asarray(pred.y_true, float), y_p, common)
        names.append(pred.name)
        rmses.append(r)
        ns.append(n)
        colors.append(P.MODEL_COLORS.get(pred.name, P.ACCENT_PRIMARY))

    for name, bl in baselines.items():
        r, n = _common_rmse(bl.dates, np.asarray(bl.y_true, float),
                            np.asarray(bl.y_pred, float), common)
        names.append(name)
        rmses.append(r)
        ns.append(n)
        colors.append(P.MODEL_COLORS.get(name, P.TEXT_MUTED))

    order = np.argsort(rmses)
    names_s  = [names[i]  for i in order]
    rmses_s  = [rmses[i]  for i in order]
    colors_s = [colors[i] for i in order]
    ns_s     = [ns[i]     for i in order]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    for i, (name, val, c) in enumerate(zip(names_s, rmses_s, colors_s)):
        alpha = 1.0 if i == 0 else 0.65
        ax.barh(name, val, color=c, height=0.5, alpha=alpha)
        ax.text(val + max(rmses_s) * 0.006, i,
                f"{val:.4f}  (n={ns_s[i]})",
                va="center", ha="left", fontsize=9, color=P.TEXT_SECONDARY)

    ax.set_title(
        f"RMSE Comparison — Test Split  ·  lower is better\n"
        f"Evaluated on {len(common)} common dates across all models",
        fontsize=11, loc="left")
    pr.axis_label(ax, x="RMSE  (log realized volatility, h=5)", y="")
    ax.set_xlim(0, max(rmses_s) * 1.25)
    ax.invert_yaxis()
    fig.tight_layout()
    return save_figure(fig, out_dir, "comparison_rmse_test")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_figures(
    deterministic_preds: Iterable[DeterministicPrediction],
    bayesian_preds: Iterable[BayesianPrediction],
    training_histories: dict,
    baselines_by_split: dict,
    out_dir: Path,
) -> None:
    out_dir = ensure_dir(out_dir)
    det_list = list(deterministic_preds)
    bay_list = list(bayesian_preds)
    test_baselines = baselines_by_split.get("test", {})

    for pred in det_list:
        if pred.split == "test":
            det_vs_actual(pred, out_dir)
            scatter_regression(pred, out_dir)

    for pred in bay_list:
        if pred.split == "test":
            bay_vs_actual(pred, out_dir)
            scatter_regression(pred, out_dir)
            uncertainty_over_time(pred, out_dir)
            uncertainty_vs_error(pred, out_dir)

    for name, history in training_histories.items():
        if len(history.get("train", [])) > 1:
            learning_curves(history, name, out_dir)

    main_bay = next(
        (p for p in bay_list if p.split == "test" and "Bayesian MLP" in p.name),
        next((p for p in bay_list if p.split == "test"), None),
    )
    if main_bay is not None:
        test_det = [p for p in det_list if p.split == "test"]
        if test_baselines:
            bayesian_vs_risk(main_bay, test_baselines, out_dir)
        predictions_over_time(main_bay, test_det, test_baselines, out_dir)
        comparison_rmse(main_bay, test_det, test_baselines, out_dir)
