# @author: Inigo Martinez Jimenez
# Annotation helpers that overlay domain context (splits, regimes, stress
# events) on top of the core plotting primitives. Each function is side-effect
# only: it adds artists to the provided axis and returns nothing.

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.visualization import palette as P
from src.visualization.events import StressEvent


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def shade_splits(
    ax,
    *,
    train_range: tuple,
    val_range: tuple,
    test_range: tuple,
    label: bool = True,
    alpha: float = 0.06,
) -> None:
    # Overlay three soft color bands corresponding to the train/val/test
    # partitions. The bands are intentionally very subtle so the main lines
    # remain the primary focus.
    spans = {
        "Training": (train_range, P.ACCENT_COOL),
        "Validation": (val_range, P.ACCENT_LAVENDER),
        "Test": (test_range, P.ACCENT_PRIMARY),
    }

    y_min, y_max = ax.get_ylim()
    for name, ((start, end), color) in spans.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=color, alpha=alpha, linewidth=0, zorder=0)
        if label:
            x_mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(
                x_mid, y_max, name,
                ha="center", va="top",
                color=color, fontsize=9.5, fontweight="semibold", alpha=0.92,
                zorder=3,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=P.BACKGROUND_AXES, edgecolor="none",
                    alpha=0.55,
                ),
            )
    ax.set_ylim(y_min, y_max)


def draw_split_boundaries(
    ax,
    *,
    train_end,
    val_end,
    color: str = P.SPINE_COLOR,
    label: bool = True,
) -> None:
    # Vertical dividers between partitions. Called by diagrams that want the
    # partition borders without the colored shading.
    for date, name in [(train_end, "Train / Val"), (val_end, "Val / Test")]:
        ax.axvline(pd.Timestamp(date), color=color, linestyle=(0, (4, 3)),
                   linewidth=0.9, alpha=0.9, zorder=1.5)
        if label:
            y_min, y_max = ax.get_ylim()
            ax.text(
                pd.Timestamp(date), y_max,
                f" {name}",
                va="top", ha="left",
                color=P.TEXT_MUTED, fontsize=8.5, alpha=0.9, zorder=3,
            )
            ax.set_ylim(y_min, y_max)


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


def shade_regimes(
    ax,
    dates,
    regimes: Sequence[str],
    *,
    alpha: float = 0.12,
    label: bool = False,
) -> None:
    # Paint contiguous regime segments as vertical color bands underneath the
    # data. We collapse consecutive days that share the same regime into a
    # single axvspan to keep the number of artists small.
    # We coerce to a DatetimeIndex so positional (including negative) indexing
    # works regardless of whether the caller passes a Series, ndarray or list.
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    regimes = np.asarray(regimes)

    if len(dates) == 0:
        return

    start_idx = 0
    current = regimes[0]
    for i in range(1, len(regimes)):
        if regimes[i] != current:
            _shade_regime_span(ax, dates[start_idx], dates[i - 1], current, alpha=alpha)
            start_idx = i
            current = regimes[i]
    _shade_regime_span(ax, dates[start_idx], dates[-1], current, alpha=alpha)

    if label:
        _regime_legend(ax)


def _shade_regime_span(ax, start, end, regime: str, *, alpha: float) -> None:
    color = {
        "LOW": P.REGIME_LOW,
        "MED": P.REGIME_MED,
        "HIGH": P.REGIME_HIGH,
    }.get(str(regime).upper(), P.TEXT_MUTED)
    ax.axvspan(start, end + pd.Timedelta(days=1), color=color, alpha=alpha,
               linewidth=0, zorder=0)


def _regime_legend(ax) -> None:
    # Adds a compact inline legend explaining the regime colors.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=P.REGIME_LOW, edgecolor="none", alpha=0.55, label="Low Volatility"),
        Patch(facecolor=P.REGIME_MED, edgecolor="none", alpha=0.55, label="Medium Volatility"),
        Patch(facecolor=P.REGIME_HIGH, edgecolor="none", alpha=0.55, label="High Volatility"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True, framealpha=0.8,
        facecolor=P.BACKGROUND_PANEL, edgecolor=P.SPINE_COLOR,
        fontsize=8.5, ncols=3, columnspacing=1.2,
    )
    leg.get_frame().set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Stress events
# ---------------------------------------------------------------------------


def mark_events(
    ax,
    events: Iterable[StressEvent],
    *,
    alpha: float = 0.16,
    color: str = P.ACCENT_PRIMARY_DEEP,
    label: bool = True,
    label_position: str = "top",
    show_lines: bool = False,
) -> None:
    # Highlight stress events as warm vertical bands. Optionally place short
    # labels above the band to identify the crisis in context.
    events = list(events)
    if not events:
        return

    y_min, y_max = ax.get_ylim()
    for event in events:
        ax.axvspan(
            event.start_ts, event.end_ts,
            color=color, alpha=alpha, linewidth=0, zorder=0.5,
        )
        if show_lines:
            ax.axvline(event.start_ts, color=color, alpha=0.35, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
            ax.axvline(event.end_ts, color=color, alpha=0.35, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)

        if label:
            if label_position == "top":
                y_text = y_max - (y_max - y_min) * 0.04
                va = "top"
            else:
                y_text = y_min + (y_max - y_min) * 0.04
                va = "bottom"

            ax.text(
                event.mid_ts, y_text, event.label,
                ha="center", va=va,
                color=P.TEXT_PRIMARY, fontsize=8.6, fontweight="semibold", alpha=0.95,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor=P.BACKGROUND_PANEL,
                    edgecolor=color,
                    alpha=0.9, linewidth=0.7,
                ),
                zorder=5,
            )
    ax.set_ylim(y_min, y_max)


def mark_event_bar(
    ax,
    events: Iterable[StressEvent],
    *,
    color: str = P.ACCENT_PRIMARY,
    y: float = -0.09,
) -> None:
    # Render a compact horizontal event timeline below the axis, useful for
    # diagrams where we do not want to clutter the main plotting area.
    events = list(events)
    if not events:
        return

    for event in events:
        ax.axvspan(event.start_ts, event.end_ts, ymin=y - 0.02, ymax=y + 0.02,
                   color=color, alpha=0.8, linewidth=0, zorder=5,
                   transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(
            event.mid_ts, y + 0.04, event.label,
            ha="center", va="bottom", color=P.TEXT_SECONDARY,
            fontsize=8.2, fontweight="medium", alpha=0.95,
            transform=ax.get_xaxis_transform(), clip_on=False,
        )


# ---------------------------------------------------------------------------
# Generic annotations
# ---------------------------------------------------------------------------


def callout(
    ax,
    xy,
    text: str,
    *,
    offset=(12, 12),
    color: str = P.ACCENT_PRIMARY,
    fontsize: float = 9.5,
    arrow: bool = True,
) -> None:
    # Rounded label box with an optional arrow. Used to point at peaks,
    # extreme values, or notable runs in forecasts.
    bbox = dict(
        boxstyle="round,pad=0.38",
        facecolor=P.BACKGROUND_PANEL,
        edgecolor=color,
        linewidth=0.9, alpha=0.95,
    )
    arrowprops = None
    if arrow:
        arrowprops = dict(
            arrowstyle="->",
            color=color, alpha=0.9, linewidth=0.9,
            connectionstyle="arc3,rad=0.15",
        )
    ax.annotate(
        text, xy=xy, xytext=offset, textcoords="offset points",
        fontsize=fontsize, color=P.TEXT_PRIMARY, fontweight="medium",
        bbox=bbox, arrowprops=arrowprops,
        ha="left", va="bottom", zorder=10,
    )


def add_vertical_marker(
    ax,
    x,
    text: str,
    *,
    color: str = P.ACCENT_PRIMARY,
    linestyle: str = "-",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    y_text: float = 0.96,
) -> None:
    ax.axvline(pd.Timestamp(x) if not isinstance(x, (int, float)) else x,
               color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=2)
    ax.text(
        pd.Timestamp(x) if not isinstance(x, (int, float)) else x,
        y_text, text,
        transform=ax.get_xaxis_transform() if not isinstance(x, (int, float)) else None,
        ha="center", va="top",
        color=color, fontsize=9, fontweight="semibold", alpha=0.95,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor=P.BACKGROUND_PANEL, edgecolor=color,
                  alpha=0.92, linewidth=0.7),
        zorder=5,
    )
