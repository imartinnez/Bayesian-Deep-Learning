from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.classification_dataset import ClassificationWindowDataset
from src.models.LSTM_classifier import LSTMClassifier
from src.training.loss import cross_entropy_with_weights
from src.visualization.plots import save_loss_curve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def compute_class_weights(y_label: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_label, minlength=n_classes).astype(float)
    weights = 1.0 / np.where(counts > 0, counts, 1.0)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = cross_entropy_with_weights(logits, y, class_weights=class_weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return running_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = cross_entropy_with_weights(logits, y, class_weights=class_weights)
        running_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return running_loss / total_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    p_list = []
    targets_list = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)
        p_list.append(probs.cpu().numpy())
        targets_list.append(y.cpu().numpy())

    p_mean = np.concatenate(p_list)       # (N, C)
    y_true = np.concatenate(targets_list) # (N,)
    y_pred = np.argmax(p_mean, axis=-1)   # (N,)

    return {
        "p_mean": p_mean,
        "y_pred": y_pred,
        "y_true": y_true,
    }


def run_LSTM_classifier_training(cfg: dict, root_dir: Path) -> dict:
    splits_dir  = root_dir / cfg["paths"]["splits"]
    models_dir  = root_dir / cfg["paths"]["clf_models"]
    figures_dir = root_dir / cfg["paths"]["clf_figures"]

    train_path = splits_dir / cfg["paths"]["train_clf_windows_filename"]
    val_path   = splits_dir / cfg["paths"]["val_clf_windows_filename"]
    test_path  = splits_dir / cfg["paths"]["test_clf_windows_filename"]

    checkpoint_path = models_dir  / cfg["paths"]["baseline_clf_checkpoint"]
    loss_curve_path = figures_dir / cfg["paths"]["baseline_clf_loss_curve_filename"]

    tcfg          = cfg["classifier_training"]
    batch_size    = tcfg["batch_size"]
    hidden_size   = tcfg["hidden_size"]
    num_layers    = tcfg["num_layers"]
    dense_size    = tcfg["dense_size"]
    learning_rate = tcfg["learning_rate"]
    weight_decay  = tcfg["weight_decay"]
    max_epochs    = tcfg["max_epochs"]
    patience      = tcfg["patience"]
    seed          = tcfg["seed"]
    num_workers   = tcfg["num_workers"]

    n_classes = cfg["classification"]["n_classes"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ClassificationWindowDataset.from_npz(train_path)
    val_dataset   = ClassificationWindowDataset.from_npz(val_path)
    test_dataset  = ClassificationWindowDataset.from_npz(test_path)

    train_loader = make_dataloader(train_dataset, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = make_dataloader(val_dataset,   batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = make_dataloader(test_dataset,  batch_size, shuffle=False, num_workers=num_workers)

    class_weights = compute_class_weights(train_dataset.y_label, n_classes=n_classes, device=device)

    model = LSTMClassifier(
        n_features=train_dataset.n_features,
        hidden=hidden_size,
        num_layers=num_layers,
        dense=dense_size,
        n_classes=n_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg["scheduler_factor"], patience=tcfg["scheduler_patience"]
    )

    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    train_losses     = []
    val_losses       = []

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, device)
        val_loss   = evaluate(model, val_loader, class_weights, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train CE: {train_loss:.6f} | Val CE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_features":       train_dataset.n_features,
                    "hidden_size":      hidden_size,
                    "num_layers":       num_layers,
                    "dense_size":       dense_size,
                    "dropout":          0.0,
                    "n_classes":        n_classes,
                    "best_epoch":       best_epoch,
                    "best_val_loss":    best_val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results = predict(model, test_loader, device)
    save_loss_curve(train_losses, val_losses, loss_curve_path)

    return {
        "device":          str(device),
        "best_epoch":      checkpoint["best_epoch"],
        "best_val_loss":   float(checkpoint["best_val_loss"]),
        "class_weights":   class_weights.cpu().numpy().tolist(),
        "checkpoint_path": checkpoint_path,
        "loss_curve_path": loss_curve_path,
        "train_samples":   len(train_dataset),
        "val_samples":     len(val_dataset),
        "test_samples":    len(test_dataset),
        **test_results,
    }