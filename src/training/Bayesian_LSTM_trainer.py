from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import WindowedTimeSeriesDataset
from src.models.Bayesian_LSTM import BayesianLSTM
from src.training.loss import heteroscedastic_gaussian_nll
from src.evaluation.uncertainty import decompose_uncertainty
from src.evaluation.metrics import compute_regression_metrics, compute_mean_baseline_metrics
from src.visualization.plots import save_loss_curve, save_prediction_plot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(dataset: WindowedTimeSeriesDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference while keeping the rest of the model
    in evaluation mode.

    This is the standard MC Dropout trick: BatchNorm and other layers stay in
    eval mode, but Dropout layers are switched to train mode so they keep
    sampling masks at each forward pass.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).squeeze(-1)

        optimizer.zero_grad()
        mu, logvar = model(X)
        mu = mu.squeeze(-1)
        logvar = logvar.squeeze(-1)
        
        loss = criterion(mu, logvar, y)

        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).squeeze(-1)

        mu, logvar = model(X)
        mu = mu.squeeze(-1)
        logvar = logvar.squeeze(-1)
        loss = criterion(mu, logvar, y)

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


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
            mu = mu.squeeze(-1)
            logvar = logvar.squeeze(-1)

            sigma2 = torch.exp(logvar)

            mu_samples.append(mu.cpu().numpy())
            sigma2_samples.append(sigma2.cpu().numpy())

        mu_samples = np.stack(mu_samples, axis=0)
        sigma2_samples = np.stack(sigma2_samples, axis=0)

        summary = decompose_uncertainty(mu_samples, sigma2_samples)

        predictive_mean_list.append(summary["predictive_mean"])
        epistemic_list.append(summary["epistemic_uncertainty"])
        aleatoric_list.append(summary["aleatoric_uncertainty"])
        total_list.append(summary["total_uncertainty"])
        std_list.append(summary["predictive_std"])
        targets_list.append(y.squeeze(-1).cpu().numpy())

    predictive_mean = np.concatenate(predictive_mean_list)
    epistemic_uncertainty = np.concatenate(epistemic_list)
    aleatoric_uncertainty = np.concatenate(aleatoric_list)
    total_uncertainty = np.concatenate(total_list)
    predictive_std = np.concatenate(std_list)
    y_true = np.concatenate(targets_list)

    return {
        "predictive_mean": predictive_mean,
        "epistemic_uncertainty": epistemic_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "total_uncertainty": total_uncertainty,
        "predictive_std": predictive_std,
        "y_true": y_true,
    }


def run_Bayesian_LSTM_training(cfg: dict, root_dir: Path) -> dict:
    splits_dir = root_dir / cfg["paths"]["splits"]
    models_dir = root_dir / cfg["paths"]["models"]
    figures_dir = root_dir / cfg["paths"]["figures"]

    train_windows_path = splits_dir / cfg["paths"]["train_windows_filename"]
    val_windows_path = splits_dir / cfg["paths"]["val_windows_filename"]
    test_windows_path = splits_dir / cfg["paths"]["test_windows_filename"]

    checkpoint_path = models_dir / cfg["paths"]["bayesian_checkpoint"]
    loss_curve_path = figures_dir / cfg["paths"]["bayesian_loss_curve_filename"]
    pred_plot_path = figures_dir / cfg["paths"]["bayesian_pred_plot_filename"]

    batch_size = cfg["training"]["batch_size"]
    hidden_size = cfg["training"]["hidden_size"]
    num_layers = cfg["training"]["num_layers"]
    dense_size = cfg["training"]["dense_size"]
    dropout = cfg["training"]["bayesian_dropout"]
    learning_rate = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["patience"]
    seed = cfg["training"]["seed"]
    num_workers = cfg["training"]["num_workers"]

    mc_samples = cfg["inference"]["mc_samples"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WindowedTimeSeriesDataset.from_npz(train_windows_path)
    val_dataset = WindowedTimeSeriesDataset.from_npz(val_windows_path)
    test_dataset = WindowedTimeSeriesDataset.from_npz(test_windows_path)

    train_loader = make_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BayesianLSTM(
        n_features=train_dataset.n_features,
        hidden=hidden_size,
        num_layers=num_layers,
        dense=dense_size,
        dropout=dropout,
    ).to(device)

    criterion = heteroscedastic_gaussian_nll
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg["training"]["scheduler_factor"],
        patience=cfg["training"]["scheduler_patience"]
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_features": train_dataset.n_features,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dense_size": dense_size,
                    "dropout": dropout,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    mc_results = predict(model, test_loader, device, T=mc_samples)

    y_test_mean = mc_results["predictive_mean"]
    y_test_true = mc_results["y_true"]

    test_metrics = compute_regression_metrics(y_test_true, y_test_mean)
    mean_baseline_metrics = compute_mean_baseline_metrics(train_dataset.y, y_test_true)

    save_loss_curve(train_losses, val_losses, loss_curve_path)
    save_prediction_plot(test_dataset.dates, y_test_true, y_test_mean, pred_plot_path)


    return {
        "device": str(device),
        "best_epoch": checkpoint["best_epoch"],
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "test_metrics": test_metrics,
        "mean_baseline_metrics": mean_baseline_metrics,
        "checkpoint_path": checkpoint_path,
        "loss_curve_path": loss_curve_path,
        "pred_plot_path": pred_plot_path,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_shape": train_dataset.X.shape,
        "val_shape": val_dataset.X.shape,
        "test_shape": test_dataset.X.shape,
        "test_predictive_mean": mc_results["predictive_mean"],
        "test_epistemic_uncertainty": mc_results["epistemic_uncertainty"],
        "test_aleatoric_uncertainty": mc_results["aleatoric_uncertainty"],
        "test_total_uncertainty": mc_results["total_uncertainty"],
        "test_predictive_std": mc_results["predictive_std"],
        "test_true": mc_results["y_true"],
        "mc_samples": mc_samples,
    }