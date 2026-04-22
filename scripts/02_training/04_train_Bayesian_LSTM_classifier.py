from pathlib import Path
import sys

from src.data.download import load_config
from src.training.Bayesian_LSTM_classifier_trainer import run_Bayesian_LSTM_classifier_training

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    results = run_Bayesian_LSTM_classifier_training(cfg, ROOT_DIR)

    print("-" * 60)
    print(f"Device:          {results['device']}")
    print(f"Best epoch:      {results['best_epoch']}")
    print(f"Best val CE:     {results['best_val_loss']:.6f}")
    print(f"Class weights:   {results['class_weights']}")
    print(f"Train samples:   {results['train_samples']}")
    print(f"Test samples:    {results['test_samples']}")
    print(f"Checkpoint:      {results['checkpoint_path']}")