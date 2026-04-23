# @author: Inigo Martinez Jimenez
# Minimal plotting helpers called by the training scripts.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_loss_curve(train_losses: list, val_losses: list, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_prediction_plot(dates, y_true, y_pred, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.asarray(y_true), label="Actual", linewidth=1.2)
    ax.plot(np.asarray(y_pred), label="Predicted", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log-RV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
