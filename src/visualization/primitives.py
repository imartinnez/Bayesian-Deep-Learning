# @author: Inigo Martinez Jimenez
# Small, composable plotting primitives used by every category-level figure
# module. Keeping them here means every chart shares the same line style,
# band gradient, label placement, and legend behavior.

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from src.visualization import palette as P
from src.visualization.theme import style_axes


# ---------------------------------------------------------------------------
# Core line / band plots
# ---------------------------------------------------------------------------


def time_series(
    ax,
    dates,
    values,
    *,
    color: str = P.ACCENT_PRIMARY,
    label: str | None = None,
    linewidth: float = 1.8,
    alpha: float = 1.0,
    zorder: int = 3,
    glow: bool = False,
    linestyle: str = "-",
) -> None:
    # Render a clean time-series line. Optional ``glow`` draws a translucent
    # wider stroke behind the main line so headline series stand out.
    dates = pd.to_datetime(dates)

    if glow:
        ax.plot(
            dates, values,
            color=color, alpha=0.18, linewidth=linewidth * 3.2,
            solid_capstyle="round", zorder=zorder - 1,
        )

    ax.plot(
        dates, values,
        color=color, alpha=alpha, linewidth=linewidth,
        label=label, zorder=zorder, linestyle=linestyle,
        solid_capstyle="round",
    )


def confidence_band(
    ax,
    dates,
    lower,
    upper,
    *,
    color: str = P.ACCENT_PRIMARY,
    alpha: float = 0.18,
    label: str | None = None,
    edge_alpha: float = 0.35,
    zorder: int = 2,
) -> None:
    # Draw a filled uncertainty band plus a subtle outline. The outline makes
    # the interval readable on both very dark and very bright backgrounds.
    dates = pd.to_datetime(dates)
    ax.fill_between(
        dates, lower, upper,
        color=color, alpha=alpha, linewidth=0,
        label=label, zorder=zorder,
    )
    ax.plot(dates, lower, color=color, alpha=edge_alpha, linewidth=0.6, zorder=zorder + 0.1)
    ax.plot(dates, upper, color=color, alpha=edge_alpha, linewidth=0.6, zorder=zorder + 0.1)


def stacked_band(
    ax,
    dates,
    baseline,
    components: Sequence[tuple[str, np.ndarray, str]],
    *,
    alpha: float = 0.55,
    zorder: int = 2,
) -> None:
    # Stack positive components on top of a baseline. ``components`` is a list
    # of ``(label, values, color)`` tuples. Used for uncertainty decomposition
    # where epistemic sits on top of aleatoric.
    dates = pd.to_datetime(dates)
    current = np.asarray(baseline, dtype=float)
    for label, values, color in components:
        values = np.asarray(values, dtype=float)
        top = current + values
        ax.fill_between(
            dates, current, top,
            color=color, alpha=alpha, linewidth=0,
            label=label, zorder=zorder,
        )
        current = top


def reference_line(
    ax,
    y: float,
    *,
    label: str | None = None,
    color: str = P.REFERENCE_COLOR,
    linestyle: str = (0, (6, 4)),
    linewidth: float = 1.2,
    alpha: float = 0.5,
) -> None:
    # Horizontal reference line for baselines, zero lines, long-run means...
    ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label, zorder=1.5)


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------


def scatter(
    ax,
    x,
    y,
    *,
    color: str = P.ACCENT_PRIMARY,
    size: float = 18.0,
    alpha: float = 0.7,
    edgecolor: str = "none",
    label: str | None = None,
    zorder: int = 3,
) -> None:
    ax.scatter(
        x, y,
        s=size, color=color, alpha=alpha, edgecolors=edgecolor,
        linewidths=0.4, label=label, zorder=zorder,
    )


def identity_line(ax, *, color: str = P.REFERENCE_COLOR, alpha: float = 0.55, label: str | None = None) -> None:
    # Draw a 1:1 identity line that spans the current axis limits. Useful for
    # actual-vs-predicted scatters and calibration plots.
    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color=color, linestyle=(0, (6, 4)), linewidth=1.2, alpha=alpha,
            label=label, zorder=1.5)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def histogram(
    ax,
    values,
    *,
    bins: int = 50,
    color: str = P.ACCENT_PRIMARY,
    alpha: float = 0.62,
    density: bool = True,
    label: str | None = None,
    edgecolor: str | None = None,
    zorder: int = 2,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges, patches = ax.hist(
        values, bins=bins, density=density,
        color=color, alpha=alpha,
        edgecolor=edgecolor or P.BACKGROUND_AXES,
        linewidth=0.35, label=label, zorder=zorder,
    )
    return counts, edges


def kde_line(
    ax,
    values,
    *,
    color: str = P.ACCENT_PRIMARY_GLOW,
    bandwidth: float | None = None,
    n_points: int = 512,
    linewidth: float = 1.8,
    alpha: float = 1.0,
    label: str | None = None,
    fill: bool = False,
    zorder: int = 3,
) -> None:
    # Kernel density estimate drawn as a smooth line with optional warm fill
    # underneath. Uses scipy.stats.gaussian_kde under the hood.
    from scipy.stats import gaussian_kde
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return

    x = np.linspace(values.min(), values.max(), n_points)
    kde = gaussian_kde(values, bw_method=bandwidth)
    y = kde(x)

    if fill:
        ax.fill_between(x, 0, y, color=color, alpha=0.25, linewidth=0, zorder=zorder - 1)
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)


def rug(ax, values, *, color: str = P.ACCENT_PRIMARY, alpha: float = 0.5, height: float = 0.02) -> None:
    # Tiny tick marks along the bottom of an axis for pointwise diagnostics.
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    y_low, y_high = ax.get_ylim()
    rug_height = (y_high - y_low) * height
    ax.vlines(values, y_low, y_low + rug_height, color=color, alpha=alpha, linewidth=0.5, zorder=2)


# ---------------------------------------------------------------------------
# Bars / heatmaps / ridge
# ---------------------------------------------------------------------------


def grouped_bars(
    ax,
    labels: Sequence[str],
    groups: dict[str, np.ndarray],
    *,
    colors: dict[str, str] | None = None,
    width: float = 0.8,
    value_format: str = "{:.3f}",
    annotate: bool = True,
) -> None:
    # Standard grouped bar chart where each group is keyed by a model name.
    # Colors fall back to the shared MODEL_COLORS palette.
    colors = colors or {}
    n_groups = len(groups)
    if n_groups == 0:
        return

    idx = np.arange(len(labels))
    bar_w = width / n_groups

    for i, (name, values) in enumerate(groups.items()):
        color = colors.get(name, P.MODEL_COLORS.get(name, P.ACCENT_PRIMARY))
        offset = (i - (n_groups - 1) / 2) * bar_w
        bars = ax.bar(idx + offset, values, width=bar_w * 0.92,
                      color=color, edgecolor="none", label=name, zorder=3)
        if annotate:
            for bar in bars:
                height = bar.get_height()
                if not np.isfinite(height):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    value_format.format(height),
                    ha="center", va="bottom",
                    color=P.TEXT_SECONDARY, fontsize=8.4,
                )

    ax.set_xticks(idx)
    ax.set_xticklabels(labels)


def heatmap(
    ax,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    cmap: str = "thesis_warm",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
    text_color: str = P.TEXT_PRIMARY,
    cbar: bool = True,
    cbar_label: str | None = None,
):
    matrix = np.asarray(matrix, dtype=float)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, annotate_fmt.format(v),
                        ha="center", va="center",
                        fontsize=8.2, color=text_color, alpha=0.95)

    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if cbar:
        cbar_obj = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar_obj.outline.set_edgecolor(P.SPINE_COLOR)
        cbar_obj.outline.set_linewidth(0.6)
        cbar_obj.ax.tick_params(colors=P.TEXT_SECONDARY, labelsize=9)
        if cbar_label:
            cbar_obj.set_label(cbar_label, color=P.TEXT_SECONDARY)
        return im, cbar_obj
    return im, None


def ridge_plot(
    ax,
    groups: dict[str, np.ndarray],
    *,
    colors: dict[str, str] | None = None,
    bandwidth: float | None = None,
    overlap: float = 1.8,
    alpha: float = 0.75,
) -> None:
    # Stacked KDE densities with a warm-to-cool ramp. Great for comparing
    # return distributions across splits or across regimes.
    from scipy.stats import gaussian_kde

    colors = colors or {}
    names = list(groups.keys())
    n = len(names)
    if n == 0:
        return

    # Shared x grid so every ridge is rendered on the same domain.
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in groups.values()])
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size < 2:
        return
    x = np.linspace(all_values.min(), all_values.max(), 512)

    y_positions = np.arange(n)[::-1]

    for y_pos, name in zip(y_positions, names):
        values = np.asarray(groups[name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        kde = gaussian_kde(values, bw_method=bandwidth)
        y = kde(x)
        y = y / y.max() * overlap
        color = colors.get(name, P.ACCENT_PRIMARY)

        ax.fill_between(x, y_pos, y_pos + y, color=color, alpha=alpha, linewidth=0, zorder=3)
        ax.plot(x, y_pos + y, color=color, linewidth=1.4, alpha=0.95, zorder=4)
        ax.hlines(y_pos, x.min(), x.max(), color=P.SPINE_COLOR, linewidth=0.6, alpha=0.7, zorder=2)
        ax.text(x.min(), y_pos + 0.1, name, color=P.TEXT_PRIMARY, fontsize=10, fontweight="semibold", zorder=5)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Labels, annotations, and formatting
# ---------------------------------------------------------------------------


def format_date_axis(ax, *, rotation: int = 0, ha: str = "center", minor: bool = False) -> None:
    # Apply the thesis date formatter so every timeline figure shares the
    # same tick density and label style.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator(),
        show_offset=False,
        formats=["%Y", "%b %Y", "%d %b", "%d %b", "%H:%M", "%H:%M"],
    ))
    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)
    if minor:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())


def style_spines(ax, *, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        ax.spines[side].set_color(P.SPINE_COLOR)


def plot_metric_callout(
    ax,
    x,
    y,
    text: str,
    *,
    color: str = P.ACCENT_PRIMARY,
    offset: tuple[float, float] = (10, 10),
    fontsize: float = 9.0,
) -> None:
    # A rounded callout box used to mark individual values (mode, mean...)
    # on distribution plots, mimicking the style of the reference images.
    bbox = dict(
        boxstyle="round,pad=0.38",
        facecolor=P.BACKGROUND_PANEL,
        edgecolor=color,
        alpha=0.95,
        linewidth=0.9,
    )
    ax.annotate(
        text, xy=(x, y), xytext=offset, textcoords="offset points",
        fontsize=fontsize, color=P.TEXT_PRIMARY, fontweight="medium",
        bbox=bbox, ha="left", va="bottom", zorder=10,
    )


def add_legend(
    ax,
    *,
    loc: str = "upper right",
    ncols: int = 1,
    title: str | None = None,
    frame: bool = True,
    columnspacing: float = 1.2,
    bbox_to_anchor=None,
) -> None:
    # Thin wrapper around ax.legend so every figure gets the same legend
    # typography and spacing.
    legend = ax.legend(
        loc=loc, ncols=ncols, title=title,
        frameon=frame, framealpha=0.85,
        facecolor=P.BACKGROUND_PANEL,
        edgecolor=P.SPINE_COLOR,
        labelcolor=P.TEXT_PRIMARY,
        bbox_to_anchor=bbox_to_anchor,
        columnspacing=columnspacing,
        fontsize=9.5, title_fontsize=10.0,
        borderpad=0.65, handlelength=2.0, handletextpad=0.7,
    )
    if legend is not None and legend.get_frame() is not None:
        legend.get_frame().set_linewidth(0.7)
    return legend


def axis_label(ax, *, x: str | None = None, y: str | None = None, title: str | None = None) -> None:
    # Central wrapper so we never forget color/weight overrides.
    if title is not None:
        ax.set_title(title, color=P.TEXT_PRIMARY, fontweight="bold", fontsize=12.5, loc="left", pad=12)
    if x is not None:
        ax.set_xlabel(x, color=P.TEXT_SECONDARY, fontsize=10.5)
    if y is not None:
        ax.set_ylabel(y, color=P.TEXT_SECONDARY, fontsize=10.5)


def place_panel_label(ax, letter: str) -> None:
    # Adds a bold "(a)" / "(b)" label in the top-left corner of an axis, used
    # on multi-panel thesis figures.
    ax.text(
        -0.02, 1.02, f"({letter})",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11, fontweight="bold",
        color=P.TEXT_PRIMARY,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def lighten(color, factor: float) -> tuple[float, float, float, float]:
    # Return a lighter version of the given color by blending toward white.
    r, g, b, a = to_rgba(color)
    return (
        r + (1 - r) * factor,
        g + (1 - g) * factor,
        b + (1 - b) * factor,
        a,
    )


def darken(color, factor: float) -> tuple[float, float, float, float]:
    # Return a darker version of the given color by blending toward black.
    r, g, b, a = to_rgba(color)
    return (r * (1 - factor), g * (1 - factor), b * (1 - factor), a)


def safe_nanmin(arr: Iterable, default: float = np.nan) -> float:
    arr = np.asarray(list(arr), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else default


def safe_nanmax(arr: Iterable, default: float = np.nan) -> float:
    arr = np.asarray(list(arr), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else default
