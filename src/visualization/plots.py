from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Baseline training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_plot(dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_dates = pd.to_datetime(dates)

    plt.figure(figsize=(12, 5))
    plt.plot(x_dates, y_true, label="Actual")
    plt.plot(x_dates, y_pred, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.title("Baseline predictions vs actual values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()