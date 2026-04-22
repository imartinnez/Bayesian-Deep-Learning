from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.har_dataset import HARDataset
from src.models.MLP import MLP


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        total += X.size(0)

    return running_loss / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        running_loss += loss.item() * X.size(0)
        total += X.size(0)

    return running_loss / total


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []

    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def compute_constant_sigma(y_val: np.ndarray, pred_val: np.ndarray) -> float:
    return float(np.std(y_val - pred_val, ddof=0))


def run_MLP_baseline_training(cfg: dict, root_dir: Path) -> dict:
    har_dir = root_dir / cfg["paths"]["har_dir"]
    models_dir = root_dir / cfg["paths"]["har_models"]
    figures_dir = root_dir / cfg["paths"]["har_figures"]

    train_dataset = HARDataset.from_npz(har_dir / cfg["paths"]["har_train_filename"])
    val_dataset   = HARDataset.from_npz(har_dir / cfg["paths"]["har_val_filename"])
    test_dataset  = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])

    hcfg = cfg["har_training"]
    batch_size    = hcfg["batch_size"]
    hidden_size   = hcfg["hidden_size"]
    dense_size    = hcfg["dense_size"]
    dropout       = hcfg["dropout"]
    lr            = float(hcfg["learning_rate"])
    weight_decay  = float(hcfg["weight_decay"])
    max_epochs    = hcfg["max_epochs"]
    patience      = hcfg["patience"]
    num_workers   = hcfg["num_workers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MLP(
        n_features=train_dataset.n_features,
        hidden_size=hidden_size,
        dense_size=dense_size,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=hcfg["scheduler_factor"],
        patience=hcfg["scheduler_patience"],
    )

    checkpoint_path = models_dir / cfg["paths"]["mlp_baseline_checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_features": train_dataset.n_features,
                "hidden_size": hidden_size,
                "dense_size": dense_size,
                "dropout": dropout,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_val, y_val   = predict(model, val_loader, device)
    pred_test, y_test = predict(model, test_loader, device)

    sigma_constant = compute_constant_sigma(y_val, pred_val)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": checkpoint_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "y_val": y_val,
        "y_test": y_test,
        "sigma_constant": sigma_constant,
        "test_dates": test_dataset.dates,
    }