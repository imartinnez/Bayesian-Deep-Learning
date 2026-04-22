from pathlib import Path
import sys

from src.data.download import load_config
from src.training.MLP_trainer import run_MLP_baseline_training
from src.visualization.plots import save_loss_curve, save_prediction_plot

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

if __name__ == "__main__":
    cfg = load_config(ROOT_DIR / "config/config.yaml")

    figures_dir = ROOT_DIR / cfg["paths"]["har_figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = run_MLP_baseline_training(cfg, ROOT_DIR)

    save_loss_curve(
        results["train_losses"],
        results["val_losses"],
        figures_dir / cfg["paths"]["mlp_baseline_loss_curve"],
    )
    save_prediction_plot(
        results["test_dates"],
        results["y_test"],
        results["pred_test"],
        figures_dir / cfg["paths"]["mlp_baseline_pred_plot"],
    )

    print(f"Best epoch:        {results['best_epoch']}")
    print(f"Best val MSE:      {results['best_val_loss']:.6f}")
    print(f"Sigma constant:    {results['sigma_constant']:.6f}")
    print(f"Checkpoint:        {results['checkpoint_path']}")