from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.har_dataset import HARDataset
from src.models.Bayesian_MLP import BayesianMLP
from src.training.loss import heteroscedastic_gaussian_nll
from src.evaluation.uncertainty import decompose_uncertainty


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        mu, logvar = model(X)
        loss = heteroscedastic_gaussian_nll(mu, logvar, y, beta=0.0, max_logvar=4.0)
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
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        mu, logvar = model(X)
        loss = heteroscedastic_gaussian_nll(mu, logvar, y, beta=0.0, max_logvar=4.0)
        running_loss += loss.item() * X.size(0)
        total += X.size(0)

    return running_loss / total


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int,
) -> dict:
    model.eval()

    predictive_mean_list = []
    epistemic_list = []
    aleatoric_list = []
    total_list = []
    std_list = []
    targets_list = []

    for X, y in loader:
        X = X.to(device)
        model.eval()
        enable_dropout(model)

        mu_samples = []
        sigma2_samples = []

        for _ in range(T):
            mu, logvar = model(X)
            logvar = torch.clamp(logvar, min=-6.0, max=4.0)
            sigma2 = torch.exp(logvar)
            mu_samples.append(mu.cpu().numpy())
            sigma2_samples.append(sigma2.cpu().numpy())

        mu_samples    = np.stack(mu_samples, axis=0)
        sigma2_samples = np.stack(sigma2_samples, axis=0)

        summary = decompose_uncertainty(mu_samples, sigma2_samples)

        predictive_mean_list.append(summary["predictive_mean"])
        epistemic_list.append(summary["epistemic_uncertainty"])
        aleatoric_list.append(summary["aleatoric_uncertainty"])
        total_list.append(summary["total_uncertainty"])
        std_list.append(summary["predictive_std"])
        targets_list.append(y.numpy())

    return {
        "predictive_mean":       np.concatenate(predictive_mean_list),
        "epistemic_uncertainty": np.concatenate(epistemic_list),
        "aleatoric_uncertainty": np.concatenate(aleatoric_list),
        "total_uncertainty":     np.concatenate(total_list),
        "predictive_std":        np.concatenate(std_list),
        "y_true":                np.concatenate(targets_list),
    }


def run_Bayesian_MLP_training(cfg: dict, root_dir: Path) -> dict:
    har_dir   = root_dir / cfg["paths"]["har_dir"]
    models_dir = root_dir / cfg["paths"]["har_models"]

    train_dataset = HARDataset.from_npz(har_dir / cfg["paths"]["har_train_filename"])
    val_dataset   = HARDataset.from_npz(har_dir / cfg["paths"]["har_val_filename"])
    test_dataset  = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])

    hcfg = cfg["har_training"]
    batch_size   = hcfg["batch_size"]
    hidden_size  = hcfg["hidden_size"]
    dense_size   = hcfg["dense_size"]
    dropout      = hcfg["bayesian_dropout"]
    lr           = float(hcfg["learning_rate"])
    weight_decay = float(hcfg["weight_decay"])
    max_epochs   = hcfg["max_epochs"]
    patience     = hcfg["patience"]
    num_workers  = hcfg["num_workers"]
    mc_samples   = cfg["har_inference"]["mc_samples"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BayesianMLP(
        n_features=train_dataset.n_features,
        hidden_size=hidden_size,
        dense_size=dense_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=hcfg["scheduler_factor"],
        patience=hcfg["scheduler_patience"],
    )

    checkpoint_path = models_dir / cfg["paths"]["bayesian_mlp_checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}")

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

    mc_results = predict(model, test_loader, device, T=mc_samples)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": checkpoint_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "mc_results": mc_results,
        "test_dates": test_dataset.dates,
        "val_dataset": val_dataset,
        "val_loader": val_loader,
        "model": model,
        "device": device,
        "mc_samples": mc_samples,
    }