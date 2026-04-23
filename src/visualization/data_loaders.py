# @author: Inigo Martinez Jimenez
# Data loading helpers for the visualization layer. Every figure module pulls
# its inputs through these helpers so there is a single, audited way of
# reaching the raw, processed, windowed, and HAR datasets.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class LstmPaths:
    # Regression pipeline: windowed LSTM + Bayesian LSTM models.
    models_dir: Path
    figures_dir: Path
    results_dir: Path
    train_windows: Path
    val_windows: Path
    test_windows: Path
    baseline_checkpoint: Path     # LSTM.pt
    bayesian_checkpoint: Path     # Bayesian_LSTM.pt


@dataclass
class ClfPaths:
    # Classification pipeline: windowed LSTM classifier + Bayesian variant.
    models_dir: Path
    figures_dir: Path
    results_dir: Path
    train_windows: Path
    val_windows: Path
    test_windows: Path
    thresholds_path: Path
    baseline_checkpoint: Path     # LSTM_classifier.pt
    bayesian_checkpoint: Path     # Bayesian_LSTM_classifier.pt


@dataclass
class HarPaths:
    # HAR/MLP pipeline: flat (non-windowed) HAR feature inputs.
    models_dir: Path
    figures_dir: Path
    results_dir: Path
    har_dir: Path
    train_npz: Path
    val_npz: Path
    test_npz: Path
    feature_scaler: Path
    thresholds_path: Path
    baseline_checkpoint: Path     # MLP_baseline.pt
    bayesian_checkpoint: Path     # Bayesian_MLP.pt


@dataclass
class Paths:
    root: Path
    config_path: Path

    # Shared data
    raw_csv: Path
    processed_parquet: Path
    train_parquet: Path
    val_parquet: Path
    test_parquet: Path
    scaler_path: Path

    # Per-pipeline trees
    lstm: LstmPaths
    clf: ClfPaths
    har: HarPaths

    # Cross-pipeline outputs produced by the visualization layer itself.
    viz_figures_root: Path        # outputs/figures  (top level, we write under `common/`)
    animations_root: Path         # outputs/animations
    manifests_root: Path          # outputs/manifests
    predictions_root: Path        # outputs/predictions (our NPZ cache)


# ---------------------------------------------------------------------------
# Config loading + Paths factory
# ---------------------------------------------------------------------------


def load_config(root_dir: Path | str) -> dict:
    root_dir = Path(root_dir)
    config_path = root_dir / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_paths(root_dir: Path | str) -> Paths:
    root = Path(root_dir)
    cfg = load_config(root)
    pcfg = cfg["paths"]

    splits_dir = root / pcfg["splits"]

    lstm_models = root / pcfg["lstm_models"]
    lstm = LstmPaths(
        models_dir=lstm_models,
        figures_dir=root / pcfg["lstm_figures"],
        results_dir=root / pcfg["lstm_results"],
        train_windows=splits_dir / pcfg["train_windows_filename"],
        val_windows=splits_dir / pcfg["val_windows_filename"],
        test_windows=splits_dir / pcfg["test_windows_filename"],
        baseline_checkpoint=lstm_models / pcfg["baseline_checkpoint"],
        bayesian_checkpoint=lstm_models / pcfg["bayesian_checkpoint"],
    )

    clf_models = root / pcfg["clf_models"]
    clf = ClfPaths(
        models_dir=clf_models,
        figures_dir=root / pcfg["clf_figures"],
        results_dir=root / pcfg["clf_results"],
        train_windows=splits_dir / pcfg["train_clf_windows_filename"],
        val_windows=splits_dir / pcfg["val_clf_windows_filename"],
        test_windows=splits_dir / pcfg["test_clf_windows_filename"],
        thresholds_path=splits_dir / pcfg["clf_thresholds_filename"],
        baseline_checkpoint=clf_models / pcfg["baseline_clf_checkpoint"],
        bayesian_checkpoint=clf_models / pcfg["bayesian_clf_checkpoint"],
    )

    har_dir = root / pcfg["har_dir"]
    har_models = root / pcfg["har_models"]
    har = HarPaths(
        models_dir=har_models,
        figures_dir=root / pcfg["har_figures"],
        results_dir=root / pcfg["har_results"],
        har_dir=har_dir,
        train_npz=har_dir / pcfg["har_train_filename"],
        val_npz=har_dir / pcfg["har_val_filename"],
        test_npz=har_dir / pcfg["har_test_filename"],
        feature_scaler=har_dir / pcfg["har_feature_scaler_filename"],
        thresholds_path=har_dir / pcfg["har_thresholds_filename"],
        baseline_checkpoint=har_models / pcfg["mlp_baseline_checkpoint"],
        bayesian_checkpoint=har_models / pcfg["bayesian_mlp_checkpoint"],
    )

    return Paths(
        root=root,
        config_path=root / "config" / "config.yaml",
        raw_csv=root / pcfg["raw"] / pcfg["raw_filename"],
        processed_parquet=root / pcfg["processed"] / pcfg["processed_filename"],
        train_parquet=splits_dir / pcfg["train_filename"],
        val_parquet=splits_dir / pcfg["val_filename"],
        test_parquet=splits_dir / pcfg["test_filename"],
        scaler_path=splits_dir / pcfg["scaler_filename"],
        lstm=lstm,
        clf=clf,
        har=har,
        viz_figures_root=root / "outputs" / "figures" / "common",
        animations_root=root / "outputs" / "animations",
        manifests_root=root / "outputs" / "manifests",
        predictions_root=root / "outputs" / "predictions",
    )


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def load_raw(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.raw_csv, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_processed(paths: Paths) -> pd.DataFrame:
    df = pd.read_parquet(paths.processed_parquet)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_split(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_window(npz_path: Path) -> dict:
    # Regression window files carry ``X``, ``y``, ``dates``. Classifier window
    # files carry ``X``, ``y_label``, ``y_continuous``, ``dates``. We return
    # everything we find so downstream callers pick what they need.
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_all_splits(paths: Paths) -> dict:
    return {
        "train": load_split(paths.train_parquet),
        "val": load_split(paths.val_parquet),
        "test": load_split(paths.test_parquet),
    }


def load_all_windows(paths: Paths) -> dict:
    return {
        "train": load_window(paths.lstm.train_windows),
        "val": load_window(paths.lstm.val_windows),
        "test": load_window(paths.lstm.test_windows),
    }


def load_all_clf_windows(paths: Paths) -> dict:
    return {
        "train": load_window(paths.clf.train_windows),
        "val": load_window(paths.clf.val_windows),
        "test": load_window(paths.clf.test_windows),
    }


def load_all_har_windows(paths: Paths) -> dict:
    return {
        "train": load_window(paths.har.train_npz),
        "val": load_window(paths.har.val_npz),
        "test": load_window(paths.har.test_npz),
    }


def as_datetime_index(dates) -> pd.DatetimeIndex:
    # Small helper around pd.to_datetime so every downstream call has the
    # same date parsing semantics.
    return pd.to_datetime(dates)
