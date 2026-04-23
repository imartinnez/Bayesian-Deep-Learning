# @author: Inigo Martinez Jimenez

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.visualization.baselines import (
    build_feature_dataframe,
    calibrated_residual_std,
    ewma_forecast,
    fit_har_model,
    har_forecast,
    historical_vol_forecast,
    mean_baseline_forecast,
)
from src.visualization.data_loaders import (
    Paths,
    build_paths,
    load_all_splits,
    load_config,
    load_processed,
)
from src.visualization.figures import generate_figures
from src.visualization.io import ensure_dir
from src.visualization.predictions import (
    discover_bayesian_predictions,
    discover_deterministic_predictions,
    generate_all_predictions,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def _build_baselines(processed: pd.DataFrame, splits: dict, horizon: int) -> dict:
    features_df = build_feature_dataframe(processed, horizon=horizon)
    train_mask = features_df["date"].isin(splits["train"]["date"])
    har_model = fit_har_model(features_df, train_mask.to_numpy())

    val_dates = splits["val"]["date"].to_numpy()
    val_y = splits["val"]["target"].to_numpy()
    sigma_hist = calibrated_residual_std(features_df, historical_vol_forecast, val_dates, val_y)
    sigma_ewma = calibrated_residual_std(features_df, ewma_forecast, val_dates, val_y)
    sigma_har = calibrated_residual_std(
        features_df,
        lambda fdf, d, y, residual_std_val: har_forecast(fdf, har_model, d, y, residual_std_val),
        val_dates, val_y,
    )
    train_target = splits["train"]["target"].to_numpy()
    sigma_mean = float(np.std(train_target - np.mean(train_target), ddof=0))

    baselines_by_split: dict = {}
    for split in ("train", "val", "test"):
        df = splits[split]
        dates = df["date"].to_numpy()
        y = df["target"].to_numpy()
        baselines_by_split[split] = {
            "Mean Baseline": mean_baseline_forecast(train_target, dates, y, sigma_mean),
            "Historical Volatility": historical_vol_forecast(features_df, dates, y, sigma_hist),
            "EWMA (RiskMetrics)": ewma_forecast(features_df, dates, y, sigma_ewma),
            "HAR-RV": har_forecast(features_df, har_model, dates, y, sigma_har),
        }
    return baselines_by_split


def _load_training_histories(paths: Paths) -> dict:
    histories: dict = {}
    try:
        import torch
    except ImportError:
        return histories

    checkpoints = [
        ("LSTM", paths.lstm.baseline_checkpoint),
        ("Bayesian LSTM", paths.lstm.bayesian_checkpoint),
        ("MLP", paths.har.baseline_checkpoint),
        ("Bayesian MLP", paths.har.bayesian_checkpoint),
    ]
    for name, ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            hist = ckpt.get("history")
            if hist:
                histories[name] = {
                    "train": np.asarray(hist.get("train_losses", []), dtype=float),
                    "val": np.asarray(hist.get("val_losses", []), dtype=float),
                    "best_epoch": int(hist.get("best_epoch", 0)),
                }
        except Exception:
            continue
    return histories


if __name__ == "__main__":
    paths = build_paths(ROOT_DIR)
    cfg = load_config(paths.root)

    figures_dir = ensure_dir(paths.viz_figures_root)
    animations_dir = ensure_dir(paths.animations_root)
    processed = load_processed(paths)
    splits = load_all_splits(paths)

    # Predictions
    try:
        generate_all_predictions(
            paths, force=False,
            logger=lambda _: None, include_classifier=False,
        )
    except Exception as e:
        print(f"[warn] predictions: {e}")

    det_preds = list(discover_deterministic_predictions(paths.predictions_root).values())
    bay_preds = list(discover_bayesian_predictions(paths.predictions_root).values())
    histories = _load_training_histories(paths)

    # Baselines
    baselines_by_split: dict = {}
    if det_preds or bay_preds:
        try:
            baselines_by_split = _build_baselines(
                processed, splits,
                horizon=int(cfg["features"].get("horizon", 5)),
            )
        except Exception as e:
            print(f"[warn] baselines: {e}")

    # Static figures
    if det_preds or bay_preds:
        try:
            generate_figures(det_preds, bay_preds, histories, baselines_by_split, figures_dir)
        except Exception as e:
            print(f"[warn] figures: {e}")

    # Animations
    if bay_preds:
        try:
            from src.visualization.animations import generate_animations
            generate_animations(processed, bay_preds, histories, animations_dir)
        except Exception as e:
            print(f"[warn] animations: {e}")
