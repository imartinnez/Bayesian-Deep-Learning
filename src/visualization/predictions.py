# @author: Inigo Martinez Jimenez
# Inference + caching helpers for every thesis model.
#
# The visualization layer is expensive to run against raw checkpoints, so we
# run each model once per split, persist the outputs as compact NPZ files under
# ``outputs/predictions/``, and let every figure module read those files. This
# decouples plotting from torch and from the training-time config.
#
# Naming convention on disk:
#   outputs/predictions/lstm_{split}.npz              — DeterministicPrediction
#   outputs/predictions/bayesian_lstm_{split}.npz     — BayesianPrediction
#   outputs/predictions/mlp_{split}.npz               — DeterministicPrediction
#   outputs/predictions/bayesian_mlp_{split}.npz      — BayesianPrediction
#   outputs/predictions/lstm_classifier_{split}.npz   — ClassifierPrediction
#   outputs/predictions/bayesian_lstm_classifier_{split}.npz — ClassifierPrediction

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.visualization.data_loaders import Paths, load_config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DeterministicPrediction:
    name: str
    split: str
    dates: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_std: np.ndarray


@dataclass
class BayesianPrediction:
    name: str
    split: str
    dates: np.ndarray
    y_true: np.ndarray
    predictive_mean: np.ndarray
    predictive_std: np.ndarray
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    mu_samples: np.ndarray | None = None
    sigma2_samples: np.ndarray | None = None
    temperature: float | None = None


@dataclass
class ClassifierPrediction:
    name: str
    split: str
    dates: np.ndarray
    y_true: np.ndarray                       # int labels
    y_pred: np.ndarray                       # argmax
    probabilities: np.ndarray                # (N, C) mean probabilities
    # MC-dropout diagnostics (Bayesian classifier only). Optional.
    total_entropy: np.ndarray | None = None
    aleatoric_entropy: np.ndarray | None = None
    mutual_information: np.ndarray | None = None
    probability_samples: np.ndarray | None = None   # (T, N, C)


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_deterministic(pred: DeterministicPrediction, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array(pred.name),
        split=np.array(pred.split),
        dates=pred.dates,
        y_true=pred.y_true,
        y_pred=pred.y_pred,
        y_std=pred.y_std,
    )
    return path


def load_deterministic(path: Path) -> DeterministicPrediction:
    data = np.load(path, allow_pickle=False)
    return DeterministicPrediction(
        name=str(data["name"]),
        split=str(data["split"]),
        dates=data["dates"],
        y_true=data["y_true"],
        y_pred=data["y_pred"],
        y_std=data["y_std"],
    )


def save_bayesian(pred: BayesianPrediction, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": np.array(pred.name),
        "split": np.array(pred.split),
        "dates": pred.dates,
        "y_true": pred.y_true,
        "predictive_mean": pred.predictive_mean,
        "predictive_std": pred.predictive_std,
        "epistemic_uncertainty": pred.epistemic_uncertainty,
        "aleatoric_uncertainty": pred.aleatoric_uncertainty,
        "total_uncertainty": pred.total_uncertainty,
        "temperature": np.array(
            pred.temperature if pred.temperature is not None else float("nan")
        ),
    }
    if pred.mu_samples is not None:
        payload["mu_samples"] = pred.mu_samples
    if pred.sigma2_samples is not None:
        payload["sigma2_samples"] = pred.sigma2_samples
    np.savez_compressed(path, **payload)
    return path


def load_bayesian(path: Path) -> BayesianPrediction:
    data = np.load(path, allow_pickle=False)
    mu_samples = data["mu_samples"] if "mu_samples" in data.files else None
    sigma2_samples = data["sigma2_samples"] if "sigma2_samples" in data.files else None
    temperature = data["temperature"] if "temperature" in data.files else None
    if temperature is not None:
        t = float(temperature)
        temperature = None if np.isnan(t) else t
    return BayesianPrediction(
        name=str(data["name"]),
        split=str(data["split"]),
        dates=data["dates"],
        y_true=data["y_true"],
        predictive_mean=data["predictive_mean"],
        predictive_std=data["predictive_std"],
        epistemic_uncertainty=data["epistemic_uncertainty"],
        aleatoric_uncertainty=data["aleatoric_uncertainty"],
        total_uncertainty=data["total_uncertainty"],
        mu_samples=mu_samples,
        sigma2_samples=sigma2_samples,
        temperature=temperature,
    )


def save_classifier(pred: ClassifierPrediction, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": np.array(pred.name),
        "split": np.array(pred.split),
        "dates": pred.dates,
        "y_true": pred.y_true,
        "y_pred": pred.y_pred,
        "probabilities": pred.probabilities,
    }
    for field in ("total_entropy", "aleatoric_entropy", "mutual_information",
                  "probability_samples"):
        value = getattr(pred, field)
        if value is not None:
            payload[field] = value
    np.savez_compressed(path, **payload)
    return path


def load_classifier(path: Path) -> ClassifierPrediction:
    data = np.load(path, allow_pickle=False)
    return ClassifierPrediction(
        name=str(data["name"]),
        split=str(data["split"]),
        dates=data["dates"],
        y_true=data["y_true"],
        y_pred=data["y_pred"],
        probabilities=data["probabilities"],
        total_entropy=data["total_entropy"] if "total_entropy" in data.files else None,
        aleatoric_entropy=data["aleatoric_entropy"] if "aleatoric_entropy" in data.files else None,
        mutual_information=data["mutual_information"] if "mutual_information" in data.files else None,
        probability_samples=data["probability_samples"] if "probability_samples" in data.files else None,
    )


# ---------------------------------------------------------------------------
# Inference orchestration (windowed LSTM / LSTM classifier / flat MLP)
# ---------------------------------------------------------------------------


def _load_checkpoint(ckpt_path: Path, device) -> dict:
    import torch
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def _build_lstm(ckpt: dict, device, bayesian: bool):
    from src.models.LSTM import BaselineLSTM
    from src.models.Bayesian_LSTM import BayesianLSTM
    Cls = BayesianLSTM if bayesian else BaselineLSTM
    model = Cls(
        n_features=ckpt["n_features"],
        hidden=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dense=ckpt["dense_size"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _build_mlp(ckpt: dict, device, bayesian: bool):
    from src.models.MLP import MLP
    from src.models.Bayesian_MLP import BayesianMLP
    Cls = BayesianMLP if bayesian else MLP
    model = Cls(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        dense_size=ckpt["dense_size"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _build_lstm_classifier(ckpt: dict, device, bayesian: bool):
    from src.models.LSTM_classifier import LSTMClassifier
    from src.models.Bayesian_LSTM_classifier import BayesianLSTMClassifier
    if bayesian:
        model = BayesianLSTMClassifier(
            n_features=ckpt["n_features"],
            hidden=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dense=ckpt["dense_size"],
            dropout=ckpt["dropout"],
            n_classes=ckpt["n_classes"],
        ).to(device)
    else:
        model = LSTMClassifier(
            n_features=ckpt["n_features"],
            hidden=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dense=ckpt["dense_size"],
            n_classes=ckpt["n_classes"],
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-pipeline generators. Each returns the list of output paths it wrote.
# They are intentionally defensive: missing checkpoints just skip the pipeline.
# ---------------------------------------------------------------------------


def _ensure_loader(dataset, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def generate_lstm_predictions(paths: Paths, cfg: dict,
                              force: bool, logger) -> list[Path]:
    # Deterministic LSTM + Bayesian LSTM, windowed regression.
    import torch
    from src.data.dataset import WindowedTimeSeriesDataset
    from src.training.LSTM_trainer import predict as predict_baseline
    from src.training.Bayesian_LSTM_trainer import predict as predict_bayesian
    from src.evaluation.calibration import calibrate_temperature

    written: list[Path] = []
    if not paths.lstm.baseline_checkpoint.exists() and not paths.lstm.bayesian_checkpoint.exists():
        logger("LSTM pipeline: no checkpoints found, skipping.")
        return written

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    mc_samples = int(cfg["inference"]["mc_samples"])

    datasets = {
        "train": WindowedTimeSeriesDataset.from_npz(paths.lstm.train_windows),
        "val":   WindowedTimeSeriesDataset.from_npz(paths.lstm.val_windows),
        "test":  WindowedTimeSeriesDataset.from_npz(paths.lstm.test_windows),
    }
    loaders = {k: _ensure_loader(v, batch_size, num_workers) for k, v in datasets.items()}
    out_root = paths.predictions_root

    # Deterministic LSTM.
    if paths.lstm.baseline_checkpoint.exists():
        ckpt = _load_checkpoint(paths.lstm.baseline_checkpoint, device)
        lstm = _build_lstm(ckpt, device, bayesian=False)

        val_pred, val_true = predict_baseline(lstm, loaders["val"], device)
        sigma = float(np.std(val_true - val_pred, ddof=0))
        logger(f"LSTM residual sigma (val): {sigma:.6f}")

        for split in ("train", "val", "test"):
            out_path = out_root / f"lstm_{split}.npz"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            y_pred, y_true = predict_baseline(lstm, loaders[split], device)
            save_deterministic(
                DeterministicPrediction(
                    name="LSTM", split=split,
                    dates=datasets[split].dates,
                    y_true=y_true, y_pred=y_pred,
                    y_std=np.full_like(y_pred, sigma, dtype=float),
                ),
                out_path,
            )
            written.append(out_path)
            logger(f"  LSTM {split}: wrote {out_path.name}")

    # Bayesian LSTM.
    if paths.lstm.bayesian_checkpoint.exists():
        ckpt = _load_checkpoint(paths.lstm.bayesian_checkpoint, device)
        bayes = _build_lstm(ckpt, device, bayesian=True)

        # Temperature scaling from validation residuals.
        val_out = predict_bayesian(bayes, loaders["val"], device, T=mc_samples)
        tau = calibrate_temperature(
            y_true=val_out["y_true"],
            predictive_mean=val_out["predictive_mean"],
            predictive_std=val_out["predictive_std"],
        )
        logger(f"Bayesian LSTM temperature tau: {tau:.4f}")

        for split in ("train", "val", "test"):
            out_path = out_root / f"bayesian_lstm_{split}.npz"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            if split == "val":
                summary = val_out
            else:
                # For train we use fewer MC samples to keep it fast and small.
                T = mc_samples if split == "test" else min(mc_samples, 40)
                summary = predict_bayesian(bayes, loaders[split], device, T=T)

            save_bayesian(
                BayesianPrediction(
                    name="Bayesian LSTM", split=split,
                    dates=datasets[split].dates,
                    y_true=summary["y_true"],
                    predictive_mean=summary["predictive_mean"],
                    predictive_std=summary["predictive_std"] / tau,
                    epistemic_uncertainty=summary["epistemic_uncertainty"],
                    aleatoric_uncertainty=summary["aleatoric_uncertainty"],
                    total_uncertainty=summary["total_uncertainty"],
                    temperature=tau,
                ),
                out_path,
            )
            written.append(out_path)
            logger(f"  Bayesian LSTM {split}: wrote {out_path.name}")
    return written


def generate_mlp_predictions(paths: Paths, cfg: dict,
                             force: bool, logger) -> list[Path]:
    # Deterministic MLP + Bayesian MLP on flat HAR features. This is the
    # "main model" (Bayesian MLP) per the thesis.
    import torch
    from src.data.har_dataset import HARDataset
    from src.training.MLP_trainer import predict as predict_mlp
    from src.training.Bayesian_MLP_trainer import predict as predict_bayesian_mlp
    from src.evaluation.calibration import calibrate_temperature

    written: list[Path] = []
    if not paths.har.baseline_checkpoint.exists() and not paths.har.bayesian_checkpoint.exists():
        logger("MLP/HAR pipeline: no checkpoints found, skipping.")
        return written

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["har_training"]["batch_size"])
    num_workers = int(cfg["har_training"]["num_workers"])
    mc_samples = int(cfg["har_inference"]["mc_samples"])

    datasets = {
        "train": HARDataset.from_npz(paths.har.train_npz),
        "val":   HARDataset.from_npz(paths.har.val_npz),
        "test":  HARDataset.from_npz(paths.har.test_npz),
    }
    loaders = {k: _ensure_loader(v, batch_size, num_workers) for k, v in datasets.items()}
    out_root = paths.predictions_root

    # Deterministic MLP.
    if paths.har.baseline_checkpoint.exists():
        ckpt = _load_checkpoint(paths.har.baseline_checkpoint, device)
        mlp = _build_mlp(ckpt, device, bayesian=False)

        val_pred, val_true = predict_mlp(mlp, loaders["val"], device)
        sigma = float(np.std(val_true - val_pred, ddof=0))
        logger(f"MLP residual sigma (val): {sigma:.6f}")

        for split in ("train", "val", "test"):
            out_path = out_root / f"mlp_{split}.npz"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            y_pred, y_true = predict_mlp(mlp, loaders[split], device)
            save_deterministic(
                DeterministicPrediction(
                    name="MLP", split=split,
                    dates=datasets[split].dates,
                    y_true=y_true, y_pred=y_pred,
                    y_std=np.full_like(y_pred, sigma, dtype=float),
                ),
                out_path,
            )
            written.append(out_path)
            logger(f"  MLP {split}: wrote {out_path.name}")

    # Bayesian MLP (main model).
    if paths.har.bayesian_checkpoint.exists():
        ckpt = _load_checkpoint(paths.har.bayesian_checkpoint, device)
        bayes = _build_mlp(ckpt, device, bayesian=True)

        val_out = predict_bayesian_mlp(bayes, loaders["val"], device, T=mc_samples)
        tau = calibrate_temperature(
            y_true=val_out["y_true"],
            predictive_mean=val_out["predictive_mean"],
            predictive_std=val_out["predictive_std"],
        )
        logger(f"Bayesian MLP temperature tau: {tau:.4f}")

        for split in ("train", "val", "test"):
            out_path = out_root / f"bayesian_mlp_{split}.npz"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            if split == "val":
                summary = val_out
            else:
                T = mc_samples if split == "test" else min(mc_samples, 40)
                summary = predict_bayesian_mlp(bayes, loaders[split], device, T=T)
            save_bayesian(
                BayesianPrediction(
                    name="Bayesian MLP", split=split,
                    dates=datasets[split].dates,
                    y_true=summary["y_true"],
                    predictive_mean=summary["predictive_mean"],
                    predictive_std=summary["predictive_std"] / tau,
                    epistemic_uncertainty=summary["epistemic_uncertainty"],
                    aleatoric_uncertainty=summary["aleatoric_uncertainty"],
                    total_uncertainty=summary["total_uncertainty"],
                    temperature=tau,
                ),
                out_path,
            )
            written.append(out_path)
            logger(f"  Bayesian MLP {split}: wrote {out_path.name}")
    return written


def generate_classifier_predictions(paths: Paths, cfg: dict,
                                    force: bool, logger) -> list[Path]:
    # Deterministic LSTM classifier + Bayesian LSTM classifier.
    import torch
    from src.data.classification_dataset import ClassificationWindowDataset
    from src.training.LSTM_classifier_trainer import predict as predict_clf
    from src.training.Bayesian_LSTM_classifier_trainer import predict as predict_bayes_clf

    written: list[Path] = []
    if not paths.clf.baseline_checkpoint.exists() and not paths.clf.bayesian_checkpoint.exists():
        logger("LSTM classifier pipeline: no checkpoints found, skipping.")
        return written

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["classifier_training"]["batch_size"])
    num_workers = int(cfg["classifier_training"]["num_workers"])
    mc_samples = int(cfg["classifier_inference"]["mc_samples"])

    datasets = {
        "train": ClassificationWindowDataset.from_npz(paths.clf.train_windows),
        "val":   ClassificationWindowDataset.from_npz(paths.clf.val_windows),
        "test":  ClassificationWindowDataset.from_npz(paths.clf.test_windows),
    }
    loaders = {k: _ensure_loader(v, batch_size, num_workers) for k, v in datasets.items()}
    out_root = paths.predictions_root

    # Deterministic classifier.
    if paths.clf.baseline_checkpoint.exists():
        ckpt = _load_checkpoint(paths.clf.baseline_checkpoint, device)
        clf = _build_lstm_classifier(ckpt, device, bayesian=False)

        for split in ("train", "val", "test"):
            out_path = out_root / f"lstm_classifier_{split}.npz"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            summary = predict_clf(clf, loaders[split], device)
            save_classifier(
                ClassifierPrediction(
                    name="LSTM Classifier", split=split,
                    dates=datasets[split].dates,
                    y_true=summary["y_true"],
                    y_pred=summary["y_pred"],
                    probabilities=summary["p_mean"],
                ),
                out_path,
            )
            written.append(out_path)
            logger(f"  LSTM Classifier {split}: wrote {out_path.name}")

    # Bayesian classifier.
    if paths.clf.bayesian_checkpoint.exists():
        ckpt = _load_checkpoint(paths.clf.bayesian_checkpoint, device)
        bayes_clf = _build_lstm_classifier(ckpt, device, bayesian=True)

        for split in ("train", "val", "test"):
            out_path = out_root / f"bayesian_lstm_classifier_{split}.npz"
            if out_path.exists() and not force:
                written.append(out_path)
                continue
            T = mc_samples if split == "test" else min(mc_samples, 40)
            summary = predict_bayes_clf(bayes_clf, loaders[split], device, T=T)
            save_classifier(
                ClassifierPrediction(
                    name="Bayesian LSTM Classifier", split=split,
                    dates=datasets[split].dates,
                    y_true=summary["y_true"],
                    y_pred=summary["y_pred"],
                    probabilities=summary["p_mean"],
                    total_entropy=summary["total_entropy"],
                    aleatoric_entropy=summary["aleatoric_entropy"],
                    mutual_information=summary["mutual_information"],
                ),
                out_path,
            )
            written.append(out_path)
            logger(f"  Bayesian LSTM Classifier {split}: wrote {out_path.name}")
    return written


def generate_all_predictions(paths: Paths, force: bool = False,
                             logger: Callable[[str], None] | None = None,
                             include_lstm: bool = True,
                             include_mlp: bool = True,
                             include_classifier: bool = True) -> dict:
    """Run inference for every available checkpoint and cache results.

    Returns a dict mapping filename stem -> Path on disk. Missing checkpoints
    are silently skipped so partial training states still produce partial
    visualisations.
    """
    logger = logger or (lambda msg: None)
    paths.predictions_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(paths.root)
    written: list[Path] = []

    if include_lstm:
        written += generate_lstm_predictions(paths, cfg, force, logger)
    if include_mlp:
        written += generate_mlp_predictions(paths, cfg, force, logger)
    if include_classifier:
        written += generate_classifier_predictions(paths, cfg, force, logger)

    return {p.stem: p for p in written}


# ---------------------------------------------------------------------------
# Training-curve extraction
# ---------------------------------------------------------------------------


def training_curve_from_losses(train_losses: Sequence[float],
                               val_losses: Sequence[float],
                               best_epoch: int) -> dict:
    return {
        "train": np.asarray(train_losses, dtype=float),
        "val": np.asarray(val_losses, dtype=float),
        "best_epoch": int(best_epoch),
    }


# ---------------------------------------------------------------------------
# Discovery helpers (used by every comparison / hero / events figure module)
# ---------------------------------------------------------------------------


def discover_deterministic_predictions(predictions_root: Path) -> dict:
    predictions_root = Path(predictions_root)
    out: dict[str, DeterministicPrediction] = {}
    if not predictions_root.exists():
        return out
    for path in sorted(predictions_root.glob("*.npz")):
        try:
            data = np.load(path, allow_pickle=False)
            files = set(data.files)
            if {"y_pred", "y_std"}.issubset(files) and "predictive_mean" not in files \
                    and "probabilities" not in files:
                out[path.stem] = load_deterministic(path)
        except Exception:
            continue
    return out


def discover_bayesian_predictions(predictions_root: Path) -> dict:
    predictions_root = Path(predictions_root)
    out: dict[str, BayesianPrediction] = {}
    if not predictions_root.exists():
        return out
    for path in sorted(predictions_root.glob("*.npz")):
        try:
            data = np.load(path, allow_pickle=False)
            files = set(data.files)
            if "predictive_mean" in files and "epistemic_uncertainty" in files:
                out[path.stem] = load_bayesian(path)
        except Exception:
            continue
    return out


def discover_classifier_predictions(predictions_root: Path) -> dict:
    predictions_root = Path(predictions_root)
    out: dict[str, ClassifierPrediction] = {}
    if not predictions_root.exists():
        return out
    for path in sorted(predictions_root.glob("*.npz")):
        try:
            data = np.load(path, allow_pickle=False)
            files = set(data.files)
            if "probabilities" in files:
                out[path.stem] = load_classifier(path)
        except Exception:
            continue
    return out


def predictions_by_split(preds: dict) -> dict:
    out: dict[str, dict[str, object]] = {}
    for _key, pred in preds.items():
        out.setdefault(pred.split, {})[pred.name] = pred
    return out
