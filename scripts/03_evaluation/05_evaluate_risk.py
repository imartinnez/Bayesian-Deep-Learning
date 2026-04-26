from pathlib import Path
import sys
import argparse
import json
import io

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data.download import load_config
from src.data.har_dataset import HARDataset
from src.data.target import add_future_vol_target
from src.models.MLP import MLP
from src.models.Bayesian_MLP import BayesianMLP
from src.models.benchmarks import (
    HAROLSVolatility,
    EWMAVolatility,
    GARCH11Volatility,
    GJRGARCH11Volatility,
)
from src.training.MLP_trainer import predict as predict_baseline
from src.training.Bayesian_MLP_trainer import predict as predict_bayesian
from src.evaluation.calibration import calibrate_temperature
from src.evaluation.risk_tests import (
    kupiec_pof,
    christoffersen_cc,
    expected_shortfall_empirical,
    quantile_loss,
    fz0_loss,
    conditional_coverage_by_decile,
)
from src.evaluation.reporting import (
    print_header,
    print_kv,
    print_artifacts,
)


MODEL_COLORS = {
    "HAR-OLS": "#1f77b4",
    "EWMA(0.94)": "#2ca02c",
    "GARCH(1,1)": "#9467bd",
    "GJR-GARCH(1,1)": "#8c564b",
    "MLP": "#ff7f0e",
    "BMLP": "#d62728",
}

BTC_EVENTS = [
    ("2023-03-10", "SVB / USDC depeg"),
    ("2023-06-06", "SEC vs Binance/Coinbase"),
    ("2024-01-11", "Spot BTC ETF approval"),
    ("2024-03-14", "BTC ATH ~$73k"),
    ("2024-04-20", "BTC halving"),
    ("2024-08-05", "Global vol selloff"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate market risk diagnostics across classical and neural models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file, relative to project root or absolute.",
    )
    return parser.parse_args()


def build_har_splits(
    parquet_path: Path,
    train_end: str,
    val_end: str,
    feature_cols: list[str],
    target_col: str,
    epsilon: float,
    horizon: int,
):
    df = pd.read_parquet(parquet_path)

    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    realized_vol = df["log_return"].pow(2)
    df["log_rv_1d_har"] = np.log(np.sqrt(realized_vol.rolling(window=1, min_periods=1).mean()).clip(lower=epsilon))
    df["log_rv_5d_har"] = np.log(np.sqrt(realized_vol.rolling(window=5, min_periods=5).mean()).clip(lower=epsilon))
    df["log_rv_22d_har"] = np.log(np.sqrt(realized_vol.rolling(window=22, min_periods=22).mean()).clip(lower=epsilon))
    df["log_return_5d"] = df["log_return"].rolling(5).sum()
    df = add_future_vol_target(df, horizon=horizon, target_col=target_col)
    df[target_col] = df[target_col].clip(lower=np.log(epsilon))

    future_returns = pd.concat(
        [df["log_return"].shift(-step).rename(f"r_t_plus_{step}") for step in range(1, horizon + 1)],
        axis=1,
    )
    df["r_horizon"] = future_returns.sum(axis=1)

    df = df.dropna(subset=feature_cols + [target_col, "r_horizon"])

    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    return train_df, val_df, test_df


def load_mlp_checkpoint(path: Path, device: torch.device) -> MLP:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLP(
        n_features=checkpoint["n_features"],
        hidden_size=checkpoint["hidden_size"],
        dense_size=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_bmlp_checkpoint(path: Path, device: torch.device) -> BayesianMLP:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = BayesianMLP(
        n_features=checkpoint["n_features"],
        hidden_size=checkpoint["hidden_size"],
        dense_size=checkpoint["dense_size"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def sigma_return_from_logvol_gaussian(mu: np.ndarray, logvol_std: np.ndarray, horizon: int = 1) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    logvol_std = np.clip(np.asarray(logvol_std, dtype=float), 0.0, None)
    sigma_daily = np.exp(mu + logvol_std**2 / 2)
    return np.sqrt(float(horizon)) * sigma_daily


def var_es_from_sigma(
    sigma_return: np.ndarray,
    alpha_var: float = 0.99,
    alpha_es: float = 0.975,
) -> tuple[np.ndarray, np.ndarray]:
    from statistics import NormalDist

    normal = NormalDist()
    z_var = normal.inv_cdf(alpha_var)
    z_es = normal.inv_cdf(alpha_es)
    phi_z_es = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z_es**2)

    var = sigma_return * z_var
    es = sigma_return * phi_z_es / (1.0 - alpha_es)
    return var, es


def print_block_title(title: str) -> None:
    print(f"\n{title}")
    print("-" * 100)


def print_block_b_table(block_b: dict) -> None:
    print_block_title("Block B | Conditional coverage by sigma decile")
    print(
        f"{'Model':<18}"
        + "".join(f"{f'D{i+1}':>7}" for i in range(10))
        + f"{'StdDev':>10}{'MaxDev':>10}"
    )
    for name, res in block_b.items():
        covs = res["coverage_by_decile"]
        print(
            f"{name:<18}"
            + "".join(f"{value:>7.2f}" for value in covs)
            + f"{res['cov_std_dev']:>10.3f}{res['cov_max_deviation']:>10.3f}"
        )


def print_block_c_table(block_c: dict) -> None:
    print_block_title("Block C | VaR 99% backtest")
    print(
        f"{'Model':<18}{'n_exc':>8}{'rate%':>9}{'Kupiec p':>12}{'Kupiec':>10}"
        f"{'CC p':>12}{'CC':>10}{'clust':>10}"
    )
    for name, values in block_c.items():
        uc = values["kupiec"]
        cc = values["christoffersen"]
        kupiec_status = "FAIL" if uc["reject_at_5pct"] else "PASS"
        cc_status = "FAIL" if cc["reject_at_5pct"] else "PASS"
        print(
            f"{name:<18}"
            f"{uc['exceptions']:>8d}"
            f"{100 * uc['rate_empirical']:>8.2f}%"
            f"{uc['p_value']:>12.4f}"
            f"{kupiec_status:>10}"
            f"{cc['p_value']:>12.4f}"
            f"{cc_status:>10}"
            f"{cc['ind']['pi_11']:>10.3f}"
        )


def print_block_d_table(block_d: dict) -> None:
    print_block_title("Block D | Expected Shortfall and scoring")
    print(f"{'Model':<18}{'Q-loss 99%':>14}{'FZ0':>12}{'ES emp':>12}{'ES pred':>12}{'Ratio':>10}")
    for name, values in block_d.items():
        print(
            f"{name:<18}"
            f"{values['quantile_loss_99']:>14.5f}"
            f"{values['fz0_loss']:>12.4f}"
            f"{values['es_empirical']:>12.4f}"
            f"{values['es_predicted_mean']:>12.4f}"
            f"{values['es_ratio']:>10.3f}"
        )


def print_event_summary(event_summary: list[dict], baseline_epistemic: float) -> None:
    print_block_title("Block A | Event summary")
    print_kv("Baseline epistemic mean", f"{baseline_epistemic:.4f}", indent=2)
    for row in event_summary:
        print_kv(
            row["date"],
            f"{row['label']} | epistemic={row['epistemic_mean']:.4f} | ratio={row['ratio_to_baseline']:.2f}x",
            indent=2,
        )

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


if __name__ == "__main__":
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    figures_dir = ROOT_DIR / cfg["paths"]["har_figures"]
    results_dir = ROOT_DIR / cfg["paths"]["har_results"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    processed_path = ROOT_DIR / cfg["paths"]["processed"] / cfg["paths"]["processed_filename"]
    risk_results_path = results_dir / cfg["paths"].get(
        "bayesian_risk_advantage_filename",
        "bayesian_risk_advantage.json",
    )

    train_df, val_df, test_df = build_har_splits(
        parquet_path=processed_path,
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
        feature_cols=cfg["har"]["features"],
        target_col=cfg["har"]["target_col"],
        epsilon=cfg["har"]["epsilon_squared_return"],
        horizon=cfg["har"]["horizon"],
    )

    y_train = train_df[cfg["har"]["target_col"]].values
    y_val = val_df[cfg["har"]["target_col"]].values
    y_test = test_df[cfg["har"]["target_col"]].values

    r_train = train_df["log_return"].values
    r_val = val_df["log_return"].values
    r_test = test_df["log_return"].values

    r_h_test = test_df["r_horizon"].values
    dates_test = pd.to_datetime(test_df.index).to_pydatetime()

    har_cols = ["log_rv_1d_har", "log_rv_5d_har", "log_rv_22d_har"]
    X_train_har = train_df[har_cols].values
    X_val_har = val_df[har_cols].values
    X_test_har = test_df[har_cols].values

    preds: dict = {}

    har = HAROLSVolatility().fit(X_train_har, y_train)
    har.calibrate_sigma(X_val_har, y_val)
    mu_har, s_har = har.predict(X_test_har)
    preds["HAR-OLS"] = {"mu": mu_har, "s": s_har, "epistemic": None}

    ewma = EWMAVolatility(lam=0.94).fit(r_train)
    sigma2_val_ew = ewma.forecast_rolling(r_val)
    mu_val_ew = 0.5 * np.log(sigma2_val_ew + 1e-12)
    ewma.calibrate_sigma(y_val, mu_val_ew)
    sigma2_test_ew = ewma.forecast_rolling(r_test)
    mu_ew = 0.5 * np.log(sigma2_test_ew + 1e-12)
    s_ew = np.full_like(mu_ew, fill_value=ewma.sigma_)
    preds["EWMA(0.94)"] = {"mu": mu_ew, "s": s_ew, "epistemic": None}

    garch = GARCH11Volatility().fit(r_train)
    sigma2_val_g = garch.forecast_rolling(r_val)
    mu_val_g = 0.5 * np.log(sigma2_val_g + 1e-12)
    garch.calibrate_sigma(y_val, mu_val_g)
    sigma2_test_g = garch.forecast_rolling(r_test)
    mu_g = 0.5 * np.log(sigma2_test_g + 1e-12)
    s_g = np.full_like(mu_g, fill_value=garch.sigma_)
    preds["GARCH(1,1)"] = {"mu": mu_g, "s": s_g, "epistemic": None}

    gjr = GJRGARCH11Volatility().fit(r_train)
    sigma2_val_j = gjr.forecast_rolling(r_val)
    mu_val_j = 0.5 * np.log(sigma2_val_j + 1e-12)
    gjr.calibrate_sigma(y_val, mu_val_j)
    sigma2_test_j = gjr.forecast_rolling(r_test)
    mu_j = 0.5 * np.log(sigma2_test_j + 1e-12)
    s_j = np.full_like(mu_j, fill_value=gjr.sigma_)
    preds["GJR-GARCH(1,1)"] = {"mu": mu_j, "s": s_j, "epistemic": None}

    har_dir = ROOT_DIR / cfg["paths"]["har_dir"]
    models_dir = ROOT_DIR / cfg["paths"]["har_models"]

    train_ds = HARDataset.from_npz(har_dir / cfg["paths"]["har_train_filename"])
    val_ds = HARDataset.from_npz(har_dir / cfg["paths"]["har_val_filename"])
    test_ds = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    mlp_checkpoint_path = models_dir / cfg["paths"]["mlp_baseline_checkpoint"]
    bayesian_mlp_checkpoint_path = models_dir / cfg["paths"]["bayesian_mlp_checkpoint"]

    mlp = load_mlp_checkpoint(mlp_checkpoint_path, device)
    mu_mlp_val, _ = predict_baseline(mlp, val_loader, device)
    mu_mlp_test, _ = predict_baseline(mlp, test_loader, device)
    sigma_mlp_const = float(np.std(val_ds.y - mu_mlp_val, ddof=0))
    s_mlp = np.full_like(mu_mlp_test, sigma_mlp_const)
    preds["MLP"] = {"mu": mu_mlp_test, "s": s_mlp, "epistemic": None}

    bmlp = load_bmlp_checkpoint(bayesian_mlp_checkpoint_path, device)
    mc_samples = cfg["har_inference"]["mc_samples"]
    bmlp_val = predict_bayesian(bmlp, val_loader, device, T=mc_samples)
    bmlp_test = predict_bayesian(bmlp, test_loader, device, T=mc_samples)

    tau = calibrate_temperature(
        y_true=bmlp_val["y_true"],
        predictive_mean=bmlp_val["predictive_mean"],
        predictive_std=np.sqrt(bmlp_val["total_uncertainty"]),
    )
    s_bmlp = np.sqrt(bmlp_test["total_uncertainty"]) / tau
    mu_bmlp = bmlp_test["predictive_mean"]
    epis_var = bmlp_test["epistemic_uncertainty"]
    epis_std = np.sqrt(epis_var)
    preds["BMLP"] = {
        "mu": mu_bmlp,
        "s": s_bmlp,
        "epistemic": epis_std,
        "epistemic_var": epis_var,
    }

    risk: dict = {}
    horizon = int(cfg["har"]["horizon"])

    for name, pred in preds.items():
        sigma_ret_daily = sigma_return_from_logvol_gaussian(pred["mu"], pred["s"], horizon=1)
        sigma_ret_h = sigma_return_from_logvol_gaussian(pred["mu"], pred["s"], horizon=horizon)
        var99, es975 = var_es_from_sigma(sigma_ret_h, alpha_var=0.99, alpha_es=0.975)

        risk[name] = {
            "sigma_return": sigma_ret_daily,
            "sigma_return_horizon": sigma_ret_h,
            "var_99": var99,
            "es_975": es975,
        }

    block_a_figure_path = figures_dir / "block_A_epistemic_timeline.png"

    fig, ax1 = plt.subplots(figsize=(14, 6))

    for name in ("HAR-OLS", "EWMA(0.94)", "GARCH(1,1)", "GJR-GARCH(1,1)"):
        ax1.plot(
            dates_test,
            risk[name]["sigma_return"],
            label=name,
            color=MODEL_COLORS[name],
            linewidth=1.0,
            alpha=0.7,
        )

    ax1.plot(
        dates_test,
        risk["MLP"]["sigma_return"],
        label="MLP",
        color=MODEL_COLORS["MLP"],
        linewidth=1.1,
        alpha=0.8,
    )
    ax1.plot(
        dates_test,
        risk["BMLP"]["sigma_return"],
        label="BMLP (total)",
        color=MODEL_COLORS["BMLP"],
        linewidth=1.3,
    )

    ax1.set_ylabel(r"$\sigma_{t+1|t}$ (return scale)")
    ax1.set_xlabel("Date")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(
        dates_test,
        0,
        preds["BMLP"]["epistemic"],
        color=MODEL_COLORS["BMLP"],
        alpha=0.15,
        label="BMLP epistemic std",
    )
    ax2.set_ylabel("Epistemic std", color=MODEL_COLORS["BMLP"])
    ax2.tick_params(axis="y", labelcolor=MODEL_COLORS["BMLP"])

    for date_str, label in BTC_EVENTS:
        event_date = pd.Timestamp(date_str)
        if dates_test[0] <= event_date <= dates_test[-1]:
            ax1.axvline(event_date, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
            ax1.text(
                event_date,
                ax1.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                alpha=0.8,
            )

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title("Volatility forecasts across models")
    plt.tight_layout()
    plt.savefig(block_a_figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    baseline_epistemic = float(np.mean(preds["BMLP"]["epistemic"]))
    event_summary = []

    for date_str, label in BTC_EVENTS:
        event_date = pd.Timestamp(date_str)
        mask = (
            (pd.Series(dates_test) >= event_date - pd.Timedelta(days=5))
            & (pd.Series(dates_test) <= event_date + pd.Timedelta(days=5))
        )
        if mask.any():
            local_mean = float(np.mean(preds["BMLP"]["epistemic"][mask.values]))
            event_summary.append({
                "date": date_str,
                "label": label,
                "epistemic_mean": local_mean,
                "ratio_to_baseline": float(local_mean / baseline_epistemic),
            })

    block_b_figure_path = figures_dir / "block_B_conditional_coverage.png"
    block_b: dict = {}

    for name, pred in preds.items():
        res = conditional_coverage_by_decile(
            y_true=y_test,
            y_mean=pred["mu"],
            y_std=pred["s"],
            level=0.90,
            n_bins=10,
        )
        block_b[name] = res

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(1, 11)

    for name, res in block_b.items():
        ax.plot(
            x,
            res["coverage_by_decile"],
            marker="o",
            label=name,
            color=MODEL_COLORS[name],
            linewidth=1.5,
        )

    ax.axhline(0.90, color="black", linestyle="--", alpha=0.5, label="Nominal 0.90")
    ax.set_xlabel("Decile of predicted sigma")
    ax.set_ylabel("Empirical coverage")
    ax.set_xticks(x)
    ax.set_ylim(0.55, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.set_title("Conditional coverage by predicted-sigma decile")
    plt.tight_layout()
    plt.savefig(block_b_figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    alpha_var = 0.99
    horizon = 5
    stride = horizon  # non-overlapping windows eliminate artificial clustering
    idx_nl = np.arange(0, len(r_h_test), stride)

    block_c: dict = {}

    for name in preds.keys():
        var99 = risk[name]["var_99"]
        exceptions = (r_h_test < -var99)[idx_nl]
        uc = kupiec_pof(exceptions, alpha_var)
        cc = christoffersen_cc(exceptions, alpha_var)
        block_c[name] = {
            "kupiec": uc,
            "christoffersen": cc,
        }

    block_d: dict = {}

    for name in preds.keys():
        var99 = risk[name]["var_99"]
        es975 = risk[name]["es_975"]
        qloss = quantile_loss(r_h_test, var99, alpha_var)
        fzl = fz0_loss(r_h_test, var99, es975, alpha_var)
        es_emp = expected_shortfall_empirical(r_h_test, var99)
        es_pred_mean = float(np.mean(es975))
        ratio = es_emp / es_pred_mean if es_pred_mean > 0 else float("nan")

        block_d[name] = {
            "quantile_loss_99": qloss,
            "fz0_loss": fzl,
            "es_empirical": es_emp,
            "es_predicted_mean": es_pred_mean,
            "es_ratio": ratio,
        }

    output = {
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "risk_horizon_days": horizon,
        "device": str(device),
        "tau_bmlp_total": float(tau),
        "processed_dataset_path": str(processed_path),
        "mlp_checkpoint_path": str(mlp_checkpoint_path),
        "bayesian_mlp_checkpoint_path": str(bayesian_mlp_checkpoint_path),
        "block_A_event_summary": to_serializable(event_summary),
        "block_A_figure_path": str(block_a_figure_path),
        "block_B_conditional_coverage": to_serializable(block_b),
        "block_B_figure_path": str(block_b_figure_path),
        "block_C_var99_backtest": to_serializable(block_c),
        "block_D_es_gneiting": to_serializable(block_d),
    }

    with open(risk_results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print_header("RUN")
    print_kv("Script", "Risk diagnostics")
    print_kv("Device", str(device))
    print_kv("Train samples", len(train_df))
    print_kv("Validation samples", len(val_df))
    print_kv("Test samples", len(test_df))
    print_kv("Risk horizon", f"{horizon} days")

    print_header("MODEL SETUP")
    print_kv("Processed dataset", str(processed_path))
    print_kv("MLP checkpoint", str(mlp_checkpoint_path))
    print_kv("Bayesian MLP checkpoint", str(bayesian_mlp_checkpoint_path))
    print_kv("Temperature tau", f"{tau:.4f}")
    print_kv("MC samples", mc_samples)
    print_kv("Compared models", "HAR-OLS | EWMA | GARCH | GJR-GARCH | MLP | BMLP")

    print_header("MAIN RESULTS")
    print_event_summary(event_summary, baseline_epistemic)

    print_header("DIAGNOSTICS")
    print_block_b_table(block_b)
    print_block_c_table(block_c)
    print_block_d_table(block_d)

    print_header("ARTIFACTS")
    print_artifacts({
        "Block A figure": str(block_a_figure_path),
        "Block B figure": str(block_b_figure_path),
        "Results JSON": str(risk_results_path),
    })