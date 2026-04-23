# @author: Inigo Martinez Jimenez
# Animations for the thesis defense. Every animation is encoded as a GIF via
# matplotlib's Pillow writer — no ffmpeg is required. We keep the animations
# short (<= ~12s) and tuned for a presentation deck where each loop reveals
# one additional piece of information.

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.visualization import palette as P
from src.visualization import primitives as pr
from src.visualization.annotate import mark_events
from src.visualization.events import STRESS_EVENTS, StressEvent
from src.visualization.io import ensure_dir, save_animation
from src.visualization.predictions import BayesianPrediction
from src.visualization.theme import FIGURE_SIZES, add_title_block


MAIN_MODEL_NAME = "Bayesian MLP"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _model_color(name: str) -> str:
    return P.MODEL_COLORS.get(name, P.ACCENT_PRIMARY)


# ---------------------------------------------------------------------------
# 1. Rolling prediction animation
# ---------------------------------------------------------------------------


def animate_rolling_prediction(
    pred: BayesianPrediction,
    out_dir: Path,
    frames: int = 120,
    fps: int = 18,
) -> Path | None:
    dates = pd.to_datetime(pred.dates)
    if len(dates) < 20:
        return None

    step = max(1, len(dates) // frames)
    frame_indices = list(range(step, len(dates) + 1, step))
    color = _model_color(pred.name)

    fig = plt.figure(figsize=FIGURE_SIZES["wide"])
    ax = fig.add_subplot(111)

    mean = np.asarray(pred.predictive_mean, dtype=float)
    sigma = np.asarray(pred.predictive_std, dtype=float)
    y_true = np.asarray(pred.y_true, dtype=float)
    low = mean - 1.645 * sigma
    up = mean + 1.645 * sigma

    ymin = float(min(np.nanmin(low), np.nanmin(y_true)))
    ymax = float(max(np.nanmax(up), np.nanmax(y_true)))
    pad = 0.08 * (ymax - ymin)
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(ymin - pad, ymax + pad)

    line_true, = ax.plot([], [], color=P.TEXT_PRIMARY, linewidth=1.6, label="Realized Volatility")
    line_mean, = ax.plot([], [], color=color, linewidth=2.0, label=f"{pred.name} Predictive Mean")
    band = ax.fill_between(dates, 0, 0, color=color, alpha=0.0)

    mark_events(ax, STRESS_EVENTS, alpha=0.09, label=False)
    pr.axis_label(ax, x="Date", y="Log Realized Volatility (h=5)",
                  title=f"Rolling Prediction — {pred.name} ({pred.split})")
    pr.format_date_axis(ax)
    pr.add_legend(ax, loc="upper left")
    add_title_block(fig, f"Rolling Prediction Animation — {pred.name}",
                    "Reveals the predictive mean and 90% interval as time advances.")
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.06, right=0.98)

    def _update(k):
        nonlocal band
        idx = frame_indices[k]
        line_true.set_data(dates[:idx], y_true[:idx])
        line_mean.set_data(dates[:idx], mean[:idx])
        band.remove()
        band = ax.fill_between(dates[:idx], low[:idx], up[:idx],
                               color=color, alpha=0.22, linewidth=0)
        return line_true, line_mean, band

    anim = FuncAnimation(fig, _update, frames=len(frame_indices), blit=False, interval=1000 // fps)
    out_path = save_animation(anim, out_dir,
                              f"anim_rolling_{_slug(pred.name)}_{pred.split}", fps=fps, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Posterior spaghetti reveal
# ---------------------------------------------------------------------------


def animate_posterior_spaghetti(
    pred: BayesianPrediction,
    out_dir: Path,
    n_traces: int = 60,
    fps: int = 16,
) -> Path | None:
    if pred.mu_samples is None:
        return None

    mu_samples = np.asarray(pred.mu_samples, dtype=float)
    if mu_samples.ndim != 2 or mu_samples.shape[0] < 3:
        return None

    dates = pd.to_datetime(pred.dates)
    y_true = np.asarray(pred.y_true, dtype=float)
    n_traces = min(n_traces, mu_samples.shape[0])
    traces = mu_samples[:n_traces]
    color = _model_color(pred.name)

    fig = plt.figure(figsize=FIGURE_SIZES["wide"])
    ax = fig.add_subplot(111)
    ax.set_xlim(dates.min(), dates.max())
    ymin = float(min(np.nanmin(traces), np.nanmin(y_true)))
    ymax = float(max(np.nanmax(traces), np.nanmax(y_true)))
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    pr.time_series(ax, dates, y_true, color=P.TEXT_PRIMARY, linewidth=1.5,
                   alpha=0.95, label="Realized Volatility", glow=True, zorder=8)
    lines = [ax.plot([], [], color=color, alpha=0.14, linewidth=1.0, zorder=3)[0]
             for _ in range(n_traces)]
    mean_line, = ax.plot([], [], color=color, linewidth=2.1, label=f"{pred.name} Mean", zorder=7)

    pr.axis_label(ax, x="Date", y="Log Realized Volatility (h=5)",
                  title=f"MC Posterior Reveal — {pred.name}")
    pr.format_date_axis(ax)
    pr.add_legend(ax, loc="upper left")
    add_title_block(fig, f"Posterior Predictive Reveal — {pred.name}",
                    "Each frame adds one MC dropout trace; the warm line tracks the running mean.")
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.06, right=0.98)

    def _update(k):
        for i, ln in enumerate(lines):
            ln.set_data(dates, traces[i]) if i < k else ln.set_data([], [])
        mean_line.set_data(dates, traces[: max(1, k)].mean(axis=0))
        return lines + [mean_line]

    anim = FuncAnimation(fig, _update, frames=n_traces + 1, blit=False, interval=1000 // fps)
    out_path = save_animation(anim, out_dir,
                              f"anim_spaghetti_{_slug(pred.name)}_{pred.split}", fps=fps, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Training loss evolution
# ---------------------------------------------------------------------------


def animate_training_curves(
    histories: dict,
    out_dir: Path,
    fps: int = 18,
) -> Path | None:
    if not histories:
        return None

    max_epochs = max(len(h["train"]) for h in histories.values())
    if max_epochs < 2:
        return None

    fig = plt.figure(figsize=FIGURE_SIZES["wide"])
    ax = fig.add_subplot(111)

    y_values = np.concatenate([np.asarray(h["train"]) for h in histories.values()] +
                              [np.asarray(h["val"]) for h in histories.values()])
    y_values = y_values[np.isfinite(y_values)]
    ymin, ymax = float(y_values.min()), float(y_values.max())
    pad = 0.08 * (ymax - ymin)
    ax.set_xlim(0, max_epochs - 1)
    ax.set_ylim(ymin - pad, ymax + pad)

    lines = {}
    for name, h in histories.items():
        color = _model_color(name)
        ln_train, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.9,
                            label=f"{name} (train)")
        ln_val, = ax.plot([], [], color=color, linewidth=1.8, alpha=0.9,
                          linestyle="--", label=f"{name} (val)")
        lines[name] = (ln_train, ln_val)

    pr.axis_label(ax, x="Epoch", y="Loss", title="Training Loss Evolution")
    pr.add_legend(ax, loc="upper right", ncols=min(2, max(1, len(histories))))
    add_title_block(fig, "Training Curves",
                    "Training and validation loss revealed epoch by epoch.")
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.07, right=0.97)

    def _update(k):
        artists = []
        for name, (ln_train, ln_val) in lines.items():
            h = histories[name]
            ln_train.set_data(range(min(k + 1, len(h["train"]))), h["train"][: k + 1])
            ln_val.set_data(range(min(k + 1, len(h["val"]))), h["val"][: k + 1])
            artists += [ln_train, ln_val]
        return artists

    anim = FuncAnimation(fig, _update, frames=max_epochs, blit=False, interval=1000 // fps)
    out_path = save_animation(anim, out_dir, "anim_training_curves", fps=fps, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4. Uncertainty decomposition evolution
# ---------------------------------------------------------------------------


def animate_uncertainty_decomposition(
    pred: BayesianPrediction,
    out_dir: Path,
    fps: int = 18,
    frames: int = 140,
) -> Path | None:
    dates = pd.to_datetime(pred.dates)
    if len(dates) < 20:
        return None

    epi = np.asarray(pred.epistemic_uncertainty, dtype=float)
    ale = np.asarray(pred.aleatoric_uncertainty, dtype=float)
    step = max(1, len(dates) // frames)
    frame_indices = list(range(step, len(dates) + 1, step))

    fig = plt.figure(figsize=FIGURE_SIZES["wide"])
    ax = fig.add_subplot(111)
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, float(np.nanmax(epi + ale)) * 1.05)

    fill_ale = ax.fill_between(dates, 0, 0, color=P.UNCERTAINTY_ALEATORIC, alpha=0.55,
                               label="Aleatoric (data) variance")
    fill_epi = ax.fill_between(dates, 0, 0, color=P.UNCERTAINTY_EPISTEMIC, alpha=0.55,
                               label="Epistemic (model) variance")

    mark_events(ax, STRESS_EVENTS, alpha=0.07, label=False)
    pr.axis_label(ax, x="Date", y="Variance",
                  title=f"Uncertainty Decomposition — {pred.name}")
    pr.format_date_axis(ax)
    pr.add_legend(ax, loc="upper left")
    add_title_block(fig, f"Variance Decomposition — {pred.name}",
                    "Aleatoric (irreducible) and epistemic (model) uncertainty stack over time.")
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.07, right=0.97)

    def _update(k):
        nonlocal fill_ale, fill_epi
        idx = frame_indices[k]
        fill_ale.remove()
        fill_epi.remove()
        fill_ale = ax.fill_between(dates[:idx], 0, ale[:idx],
                                   color=P.UNCERTAINTY_ALEATORIC, alpha=0.55)
        fill_epi = ax.fill_between(dates[:idx], ale[:idx], ale[:idx] + epi[:idx],
                                   color=P.UNCERTAINTY_EPISTEMIC, alpha=0.55)
        return fill_ale, fill_epi

    anim = FuncAnimation(fig, _update, frames=len(frame_indices), blit=False, interval=1000 // fps)
    out_path = save_animation(anim, out_dir,
                              f"anim_uncertainty_{_slug(pred.name)}_{pred.split}", fps=fps, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. Event window zoom
# ---------------------------------------------------------------------------


def animate_event_zoom(
    processed: pd.DataFrame,
    pred: BayesianPrediction,
    event: StressEvent,
    out_dir: Path,
    pad_days: int = 30,
    fps: int = 16,
) -> Path | None:
    dates = pd.to_datetime(pred.dates)
    y_true = np.asarray(pred.y_true, dtype=float)
    mean = np.asarray(pred.predictive_mean, dtype=float)
    sigma = np.asarray(pred.predictive_std, dtype=float)

    window_start = event.start_ts - pd.Timedelta(days=pad_days)
    window_end = event.end_ts + pd.Timedelta(days=pad_days)
    mask = (dates >= window_start) & (dates <= window_end)
    if mask.sum() < 5:
        return None

    dates_w = dates[mask]
    y_w, m_w, s_w = y_true[mask], mean[mask], sigma[mask]
    lower, upper = m_w - 1.645 * s_w, m_w + 1.645 * s_w
    color = _model_color(pred.name)

    fig = plt.figure(figsize=FIGURE_SIZES["wide"])
    ax = fig.add_subplot(111)
    ax.set_xlim(dates_w.min(), dates_w.max())
    ymin = float(min(np.nanmin(lower), np.nanmin(y_w)))
    ymax = float(max(np.nanmax(upper), np.nanmax(y_w)))
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    mark_events(ax, [event], alpha=0.14, color=P.ACCENT_PRIMARY_DEEP,
                label=True, show_lines=True)
    line_true, = ax.plot([], [], color=P.TEXT_PRIMARY, linewidth=1.6, label="Realized Volatility")
    line_mean, = ax.plot([], [], color=color, linewidth=2.0, label=f"{pred.name} Mean")
    band = ax.fill_between(dates_w, 0, 0, color=color, alpha=0)

    pr.axis_label(ax, x="Date", y="Log Realized Volatility (h=5)",
                  title=f"{event.label} — Animated Reveal")
    pr.format_date_axis(ax)
    pr.add_legend(ax, loc="upper left")
    add_title_block(fig, f"{event.label} — Zoom Animation",
                    f"±{pad_days}-day window around {event.name}. {event.description}")
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.07, right=0.97)

    def _update(k):
        nonlocal band
        idx = max(1, k + 1)
        line_true.set_data(dates_w[:idx], y_w[:idx])
        line_mean.set_data(dates_w[:idx], m_w[:idx])
        band.remove()
        band = ax.fill_between(dates_w[:idx], lower[:idx], upper[:idx],
                               color=color, alpha=0.22, linewidth=0)
        return line_true, line_mean, band

    anim = FuncAnimation(fig, _update, frames=len(dates_w), blit=False, interval=1000 // fps)
    out_path = save_animation(anim, out_dir,
                              f"anim_event_{event.key}_{_slug(pred.name)}", fps=fps, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_animations(
    processed: pd.DataFrame,
    bayesian_preds: Iterable[BayesianPrediction],
    training_histories: dict | None,
    out_dir: Path,
) -> None:
    anim_dir = ensure_dir(out_dir)
    bay_list = list(bayesian_preds)

    for pred in bay_list:
        if pred.split != "test":
            continue
        animate_rolling_prediction(pred, anim_dir)
        animate_uncertainty_decomposition(pred, anim_dir)
        if pred.mu_samples is not None:
            animate_posterior_spaghetti(pred, anim_dir)

    if training_histories:
        animate_training_curves(training_histories, anim_dir)

    main_pred = next(
        (p for p in bay_list if p.name == MAIN_MODEL_NAME and p.split == "test"),
        next((p for p in bay_list if p.split == "test"), None),
    )
    if main_pred is not None:
        dates = pd.to_datetime(main_pred.dates)
        for event in STRESS_EVENTS:
            if ((dates >= event.start_ts) & (dates <= event.end_ts)).any():
                animate_event_zoom(processed, main_pred, event, anim_dir)
