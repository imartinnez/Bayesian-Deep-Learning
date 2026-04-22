from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.download import load_config
from src.training.LSTM_trainer import run_LSTM_training

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    results = run_LSTM_training(cfg, ROOT_DIR)

    print("-" * 60)
    print(f"Using device: {results['device']}")
    print(f"Train samples: {results['train_samples']}")
    print(f"Validation samples: {results['val_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Train window shape: {results['train_shape']}")
    print(f"Validation window shape: {results['val_shape']}")
    print(f"Test window shape: {results['test_shape']}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")

    print("Test metrics:")
    print(f"Test MSE:  {results['test_metrics']['mse']:.6f}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")
    print(f"Test MAE:  {results['test_metrics']['mae']:.6f}")

    print("Mean baseline on test:")
    print(f"Mean baseline MSE:  {results['mean_baseline_metrics']['mse']:.6f}")
    print(f"Mean baseline RMSE: {results['mean_baseline_metrics']['rmse']:.6f}")
    print(f"Mean baseline MAE:  {results['mean_baseline_metrics']['mae']:.6f}")

    print(f"Best model saved to: {results['checkpoint_path']}")
    print(f"Loss curve saved to: {results['loss_curve_path']}")
    print(f"Prediction plot saved to: {results['pred_plot_path']}")