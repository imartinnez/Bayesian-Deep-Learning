from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.download import load_config
from src.training.Bayesian_LSTM_trainer import run_Bayesian_LSTM_training

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def build_train_summary(results: dict) -> dict:
    return {
        "device": results["device"],
        "train_samples": results["train_samples"],
        "validation_samples": results["val_samples"],
        "test_samples": results["test_samples"],
        "train_shape": list(results["train_shape"]),
        "validation_shape": list(results["val_shape"]),
        "test_shape": list(results["test_shape"]),
        "best_epoch": results["best_epoch"],
        "best_validation_nll": results["best_val_loss"],
        "checkpoint_path": str(results["checkpoint_path"]),
        "loss_curve_path": str(results["loss_curve_path"]),
        "prediction_plot_path": str(results["pred_plot_path"]),
        "test_metrics": results["test_metrics"],
    }

if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    results = run_Bayesian_LSTM_training(cfg, ROOT_DIR)

    results_dir = ROOT_DIR / cfg["paths"]["lstm_results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / cfg["paths"]["bayesian_train_results_filename"]

    summary = build_train_summary(results)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Training summary saved to: {results_path}\n\n\n")

    print("-" * 60)
    print(f"Using device: {results['device']}")
    print(f"Train samples: {results['train_samples']}")
    print(f"Validation samples: {results['val_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Train window shape: {results['train_shape']}")
    print(f"Validation window shape: {results['val_shape']}")
    print(f"Test window shape: {results['test_shape']}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation NLL: {results['best_val_loss']:.6f}")
    print(f"Checkpoint saved to: {results['checkpoint_path']}")
    print(f"Loss curve saved to: {results['loss_curve_path']}")
    print(f"Prediction plot saved to: {results['pred_plot_path']}")

    print("Point metrics using predictive mean:")
    print(f"Test MSE:  {results['test_metrics']['mse']:.6f}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")
    print(f"Test MAE:  {results['test_metrics']['mae']:.6f}")