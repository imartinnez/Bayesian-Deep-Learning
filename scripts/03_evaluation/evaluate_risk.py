"""Demonstrate the Bayesian advantage for market risk management.

Four blocks:
    A. Epistemic uncertainty timeline vs classical sigma — what BNN sees that
       no deterministic volatility model can see.
    B. Conditional coverage by sigma decile — does the interval width track
       actual error? Only adaptive-sigma models can pass this.
    C. VaR_99 backtest with Kupiec (unconditional coverage) and Christoffersen
       (independence / conditional coverage) — regulatory standard.
    D. Expected Shortfall + Fissler-Ziegel joint loss — Basel IV / FRTB metric.

All six models evaluated on identical test split:
    HAR-OLS, EWMA(0.94), GARCH(1,1), GJR-GARCH(1,1), MLP baseline, Bayesian MLP.
"""

from pathlib import Path
import sys
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
from src.models.MLP import MLP
from src.models.Bayesian_MLP import BayesianMLP
from src.models.benchmarks import (
    HAROLSVolatility, EWMAVolatility, GARCH11Volatility, GJRGARCH11Volatility,
)
from src.training.MLP_trainer import predict as predict_baseline
from src.training.Bayesian_MLP_trainer import predict as predict_bayesian
from src.evaluation.calibration import calibrate_temperature
from src.evaluation.risk_tests import (
    kupiec_pof, christoffersen_cc,
    expected_shortfall_empirical, quantile_loss, fz0_loss,
    conditional_coverage_by_decile,
)


# ── helpers ──────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "HAR-OLS":         "#1f77b4",
    "EWMA(0.94)":      "#2ca02c",
    "GARCH(1,1)":      "#9467bd",
    "GJR-GARCH(1,1)":  "#8c564b",
    "MLP":             "#ff7f0e",
    "BMLP":            "#d62728",
}

# BTC events observable in the test window (2023-01-01 — 2024-12-30)
BTC_EVENTS = [
    ("2023-03-10", "SVB / USDC depeg"),
    ("2023-06-06", "SEC vs Binance/Coinbase"),
    ("2024-01-11", "Spot BTC ETF approval"),
    ("2024-03-14", "BTC ATH ~$73k"),
    ("2024-04-20", "BTC halving"),
    ("2024-08-05", "Global vol selloff"),
]


def build_har_splits(parquet_path: Path, train_end: str, val_end: str,
                     feature_cols: list, target_col: str,
                     epsilon: float, horizon: int):
    """Rebuild splits with same dropna as script 09 so every model aligns."""
    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    r_sq = df["log_return"].pow(2)
    df["log_rv_1d_har"]  = np.log(r_sq.rolling(1).mean())
    df["log_rv_5d_har"]  = np.log(r_sq.rolling(5).mean())
    df["log_rv_22d_har"] = np.log(r_sq.rolling(22).mean())
    df["log_return_5d"]  = df["log_return"].rolling(5).sum()
    df[target_col]       = np.log(r_sq.shift(-horizon) + epsilon)
    df["r_next"]         = df["log_return"].shift(-horizon)

    df = df.dropna(subset=feature_cols + [target_col])

    train_df = df[df.index <= train_end]
    val_df   = df[(df.index > train_end) & (df.index <= val_end)]
    test_df  = df[df.index > val_end]
    return train_df, val_df, test_df


def load_mlp_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m = MLP(n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
            dense_size=ckpt["dense_size"], dropout=ckpt["dropout"]).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m


def load_bmlp_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m = BayesianMLP(n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
                    dense_size=ckpt["dense_size"], dropout=ckpt["dropout"]).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m


# Jensen shift: if y = log(r^2) and r ~ N(0, sigma^2), then
#   E[y | sigma^2] = log(sigma^2) + E[log(chi^2_1)] = log(sigma^2) - 1.2704
# So to back out log(sigma^2) from a log(r^2) regression forecast: add 1.2704.
JENSEN_SHIFT = 1.2704


def sigma_return_point(mu: np.ndarray, is_logvar: bool) -> np.ndarray:
    """Point forecast of sigma_{t+1|t} in return-scale units.

    is_logvar=True   -> mu already encodes log(sigma^2) directly (EWMA/GARCH/GJR).
    is_logvar=False  -> mu encodes E[log(r^2)] = log(sigma^2) - 1.2704 (HAR/MLP/BMLP).
    """
    log_var = mu if is_logvar else mu + JENSEN_SHIFT
    return np.exp(0.5 * log_var)


def sigma_return_bayesian(mu: np.ndarray, epistemic_var: np.ndarray,
                          is_logvar: bool) -> np.ndarray:
    """Bayesian-marginal sigma using epistemic variance only.

    Aleatoric noise (log chi^2) is NOT model uncertainty — only epistemic
    variance on log(sigma^2) should inflate the predictive variance of r^2.
    """
    log_var = mu if is_logvar else mu + JENSEN_SHIFT
    return np.exp(0.5 * log_var + 0.25 * epistemic_var)


def var_es_from_sigma(sigma_return: np.ndarray, alpha_var: float = 0.99,
                      alpha_es: float = 0.975) -> tuple[np.ndarray, np.ndarray]:
    """Parametric VaR and ES under zero-mean Normal(0, sigma^2) on r_{t+1}."""
    from statistics import NormalDist
    N = NormalDist()
    z_var = N.inv_cdf(alpha_var)
    z_es  = N.inv_cdf(alpha_es)
    phi_z_es = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z_es ** 2)
    var = sigma_return * z_var
    es  = sigma_return * phi_z_es / (1.0 - alpha_es)
    return var, es


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config(ROOT_DIR / "config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    figures_dir = ROOT_DIR / cfg["paths"]["har_figures"]
    results_dir = ROOT_DIR / cfg["paths"]["har_results"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Section 1: build aligned data ────────────────────────────────────
    parquet = ROOT_DIR / cfg["paths"]["processed"] / cfg["paths"]["processed_filename"]
    train_df, val_df, test_df = build_har_splits(
        parquet_path=parquet,
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
        feature_cols=cfg["har"]["features"],
        target_col=cfg["har"]["target_col"],
        epsilon=cfg["har"]["epsilon_squared_return"],
        horizon=cfg["har"]["horizon"],
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    y_train = train_df[cfg["har"]["target_col"]].values
    y_val   = val_df[cfg["har"]["target_col"]].values
    y_test  = test_df[cfg["har"]["target_col"]].values

    r_train = train_df["log_return"].values
    r_val   = val_df["log_return"].values
    r_test  = test_df["log_return"].values

    r_next_test = test_df["r_next"].values  # realized r_{t+1} aligned with test features
    dates_test  = pd.to_datetime(test_df.index).to_pydatetime()

    har_cols = ["log_rv_1d_har", "log_rv_5d_har", "log_rv_22d_har"]
    X_train_har = train_df[har_cols].values
    X_val_har   = val_df[har_cols].values
    X_test_har  = test_df[har_cols].values

    # ── Section 2: classical benchmarks — predictive (mu, s) in log(r^2) space
    preds: dict = {}

    # HAR-OLS
    har = HAROLSVolatility().fit(X_train_har, y_train)
    har.calibrate_sigma(X_val_har, y_val)
    mu_har, s_har = har.predict(X_test_har)
    preds["HAR-OLS"] = {"mu": mu_har, "s": s_har, "epistemic": None}

    # EWMA
    ewma = EWMAVolatility(lam=0.94).fit(r_train)
    s2_val_ew = ewma.forecast_rolling(r_val)
    ewma.calibrate_sigma(y_val, np.log(s2_val_ew + 1e-12))
    s2_test_ew = ewma.forecast_rolling(r_test)
    mu_ew, s_ew = ewma.predict_from_variance(s2_test_ew)
    preds["EWMA(0.94)"] = {"mu": mu_ew, "s": s_ew, "epistemic": None}

    # GARCH(1,1)
    garch = GARCH11Volatility().fit(r_train)
    s2_val_g = garch.forecast_rolling(r_val)
    garch.calibrate_sigma(y_val, np.log(s2_val_g + 1e-12))
    s2_test_g = garch.forecast_rolling(r_test)
    mu_g, s_g = garch.predict_from_variance(s2_test_g)
    preds["GARCH(1,1)"] = {"mu": mu_g, "s": s_g, "epistemic": None}

    # GJR-GARCH(1,1)
    gjr = GJRGARCH11Volatility().fit(r_train)
    s2_val_j = gjr.forecast_rolling(r_val)
    gjr.calibrate_sigma(y_val, np.log(s2_val_j + 1e-12))
    s2_test_j = gjr.forecast_rolling(r_test)
    mu_j, s_j = gjr.predict_from_variance(s2_test_j)
    preds["GJR-GARCH(1,1)"] = {"mu": mu_j, "s": s_j, "epistemic": None}

    # ── Section 3: neural nets — run inference on saved checkpoints ──────
    har_dir  = ROOT_DIR / cfg["paths"]["har_dir"]
    models_d = ROOT_DIR / cfg["paths"]["har_models"]

    train_ds = HARDataset.from_npz(har_dir / cfg["paths"]["har_train_filename"])
    val_ds   = HARDataset.from_npz(har_dir / cfg["paths"]["har_val_filename"])
    test_ds  = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])
    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # MLP baseline (point, constant sigma from val residuals)
    mlp = load_mlp_checkpoint(models_d / cfg["paths"]["mlp_baseline_checkpoint"], device)
    mu_mlp_val,  _ = predict_baseline(mlp, val_loader,  device)
    mu_mlp_test, _ = predict_baseline(mlp, test_loader, device)
    sigma_mlp_const = float(np.std(val_ds.y - mu_mlp_val, ddof=0))
    s_mlp = np.full_like(mu_mlp_test, sigma_mlp_const)
    preds["MLP"] = {"mu": mu_mlp_test, "s": s_mlp, "epistemic": None}

    # Bayesian MLP (MC-Dropout): use total uncertainty + temperature calibration
    bmlp = load_bmlp_checkpoint(models_d / cfg["paths"]["bayesian_mlp_checkpoint"], device)
    mc_samples = cfg["har_inference"]["mc_samples"]
    bmlp_val  = predict_bayesian(bmlp, val_loader,  device, T=mc_samples)
    bmlp_test = predict_bayesian(bmlp, test_loader, device, T=mc_samples)

    # Calibrate tau on total predictive std (bug-fix from prior chapter)
    tau = calibrate_temperature(
        y_true=bmlp_val["y_true"],
        predictive_mean=bmlp_val["predictive_mean"],
        predictive_std=np.sqrt(bmlp_val["total_uncertainty"]),
    )
    s_bmlp   = np.sqrt(bmlp_test["total_uncertainty"]) / tau
    mu_bmlp  = bmlp_test["predictive_mean"]
    epis_var = bmlp_test["epistemic_uncertainty"]  # already a variance
    epis_std = np.sqrt(epis_var)
    preds["BMLP"] = {"mu": mu_bmlp, "s": s_bmlp,
                     "epistemic": epis_std, "epistemic_var": epis_var}
    print(f"Temperature scaling tau (BMLP, on total std) = {tau:.4f}")

    # ── Section 4: compute sigma_return, VaR, ES for each model ──────────
    # Classical variance models (EWMA/GARCH/GJR) forecast log(sigma^2) directly.
    # Log-r^2 regressors (HAR-OLS, MLP, BMLP) fit E[log(r^2)] = log(sigma^2) - 1.2704,
    # so we add JENSEN_SHIFT to their mu to back out log(sigma^2).
    IS_LOGVAR = {
        "HAR-OLS":         False,
        "EWMA(0.94)":      True,
        "GARCH(1,1)":      True,
        "GJR-GARCH(1,1)":  True,
        "MLP":             False,
        "BMLP":            False,
    }
    risk: dict = {}
    for name, p in preds.items():
        if name == "BMLP":
            sigma_ret = sigma_return_bayesian(
                p["mu"], p["epistemic_var"], is_logvar=IS_LOGVAR[name],
            )
        else:
            sigma_ret = sigma_return_point(p["mu"], is_logvar=IS_LOGVAR[name])
        var99, es975 = var_es_from_sigma(sigma_ret, alpha_var=0.99, alpha_es=0.975)
        risk[name] = {
            "sigma_return": sigma_ret,
            "var_99": var99,
            "es_975": es975,
        }

    # ─────────────────────────────────────────────────────────────────────
    # BLOCK A -Epistemic timeline + event annotations
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BLOCK A -Epistemic uncertainty timeline")
    print("=" * 70)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    for name in ("HAR-OLS", "EWMA(0.94)", "GARCH(1,1)", "GJR-GARCH(1,1)"):
        ax1.plot(dates_test, risk[name]["sigma_return"], label=name,
                 color=MODEL_COLORS[name], linewidth=1.0, alpha=0.7)
    ax1.plot(dates_test, risk["MLP"]["sigma_return"], label="MLP",
             color=MODEL_COLORS["MLP"], linewidth=1.1, alpha=0.8)
    ax1.plot(dates_test, risk["BMLP"]["sigma_return"], label="BMLP (total)",
             color=MODEL_COLORS["BMLP"], linewidth=1.3)

    ax1.set_ylabel(r"$\sigma_{t+1|t}$ (return scale)")
    ax1.set_xlabel("Date")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(dates_test, 0, preds["BMLP"]["epistemic"],
                     color=MODEL_COLORS["BMLP"], alpha=0.15,
                     label="BMLP epistemic std (log-variance)")
    ax2.set_ylabel("Epistemic std — BMLP only", color=MODEL_COLORS["BMLP"])
    ax2.tick_params(axis="y", labelcolor=MODEL_COLORS["BMLP"])

    for d_str, label in BTC_EVENTS:
        d = pd.Timestamp(d_str)
        if d >= dates_test[0] and d <= dates_test[-1]:
            ax1.axvline(d, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
            ax1.text(d, ax1.get_ylim()[1] * 0.98, label, rotation=90,
                     va="top", ha="right", fontsize=7, alpha=0.8)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title("Volatility forecasts across models — BMLP epistemic shaded")
    plt.tight_layout()
    fig_path = figures_dir / "block_A_epistemic_timeline.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  figure: {fig_path}")

    # Epistemic spike counts near events (+/- 5 days)
    print("\n  Epistemic std around BTC events (mean over +/-5 day window):")
    baseline_epi = float(np.mean(preds["BMLP"]["epistemic"]))
    print(f"    baseline (test-wide mean): {baseline_epi:.4f}")
    for d_str, label in BTC_EVENTS:
        d = pd.Timestamp(d_str)
        mask = (pd.Series(dates_test) >= d - pd.Timedelta(days=5)) & \
               (pd.Series(dates_test) <= d + pd.Timedelta(days=5))
        if mask.any():
            local_mean = float(np.mean(preds["BMLP"]["epistemic"][mask.values]))
            ratio = local_mean / baseline_epi
            mark = "  <--" if ratio > 1.2 else ""
            print(f"    {d_str} ({label:<28}) epistemic={local_mean:.4f}  "
                  f"ratio-to-baseline={ratio:.2f}x{mark}")

    # ─────────────────────────────────────────────────────────────────────
    # BLOCK B -Conditional coverage by sigma decile
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BLOCK B -Conditional coverage at 90% across sigma deciles")
    print("=" * 70)
    print("  (a well-calibrated conditional model → flat across deciles)")
    print(f"\n  {'Model':<18}" + "".join(f" D{i+1:<4}" for i in range(10))
          + "   stddev  maxdev")
    block_b: dict = {}
    for name, p in preds.items():
        res = conditional_coverage_by_decile(
            y_true=y_test, y_mean=p["mu"], y_std=p["s"],
            level=0.90, n_bins=10,
        )
        block_b[name] = res
        covs = res["coverage_by_decile"]
        print(f"  {name:<18}" + "".join(f" {c:>5.2f}" for c in covs)
              + f"   {res['cov_std_dev']:>5.3f}  {res['cov_max_deviation']:>5.3f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(1, 11)
    for name, res in block_b.items():
        ax.plot(x, res["coverage_by_decile"], marker="o",
                label=name, color=MODEL_COLORS[name], linewidth=1.5)
    ax.axhline(0.90, color="black", linestyle="--", alpha=0.5, label="nominal 0.90")
    ax.set_xlabel(r"Decile of predicted $\sigma$ (ascending)")
    ax.set_ylabel("Empirical coverage (90% nominal)")
    ax.set_xticks(x)
    ax.set_ylim(0.55, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.set_title("Conditional coverage by predicted-sigma decile")
    plt.tight_layout()
    fig_path = figures_dir / "block_B_conditional_coverage.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\n  figure: {fig_path}")

    # ─────────────────────────────────────────────────────────────────────
    # BLOCK C -VaR_99 backtest: Kupiec + Christoffersen
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BLOCK C -VaR_99 backtest (Kupiec + Christoffersen)")
    print("=" * 70)
    print("  Expected exception rate: 1.00%   -- reject-at-5% means model FAILS")
    print()
    alpha_var = 0.99
    print(f"  {'Model':<18} {'n_exc':>6} {'rate%':>7} "
          f"{'Kupiec p':>10} {'Kupiec':>7} {'CC p':>10} {'CC':>7} "
          f"{'clust':>8}")
    print("  " + "-" * 76)
    block_c: dict = {}
    for name in preds.keys():
        var99 = risk[name]["var_99"]
        exc = r_next_test < -var99
        uc  = kupiec_pof(exc, alpha_var)
        cc  = christoffersen_cc(exc, alpha_var)
        block_c[name] = {
            "kupiec": uc,
            "christoffersen": cc,
        }
        fail_k = "FAIL" if uc["reject_at_5pct"] else "pass"
        fail_c = "FAIL" if cc["reject_at_5pct"] else "pass"
        print(f"  {name:<18} {uc['exceptions']:>6d} {100*uc['rate_empirical']:>6.2f}% "
              f"{uc['p_value']:>10.4f} {fail_k:>7} "
              f"{cc['p_value']:>10.4f} {fail_c:>7} "
              f"{cc['ind']['pi_11']:>8.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # BLOCK D -Expected Shortfall + Gneiting losses
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BLOCK D -Expected Shortfall + Gneiting-consistent losses")
    print("=" * 70)
    print("  Quantile loss at 99%: lower = better")
    print("  FZ0 loss: lower = better (joint VaR/ES scoring)")
    print("  ES_975 emp / ES_975 pred: closer to 1 = better calibrated")
    print()
    print(f"  {'Model':<18} {'q-loss 99%':>12} {'FZ0':>10} "
          f"{'ES emp':>10} {'ES pred':>10} {'ratio':>8}")
    print("  " + "-" * 72)
    block_d: dict = {}
    for name in preds.keys():
        var99 = risk[name]["var_99"]
        es975 = risk[name]["es_975"]
        qloss = quantile_loss(r_next_test, var99, alpha_var)
        fzl   = fz0_loss(r_next_test, var99, es975, alpha_var)
        es_emp = expected_shortfall_empirical(r_next_test, var99)
        es_pred_mean = float(np.mean(es975))
        ratio = es_emp / es_pred_mean if es_pred_mean > 0 else float("nan")
        block_d[name] = {
            "quantile_loss_99": qloss,
            "fz0_loss": fzl,
            "es_empirical": es_emp,
            "es_predicted_mean": es_pred_mean,
            "es_ratio": ratio,
        }
        print(f"  {name:<18} {qloss:>12.5f} {fzl:>10.4f} "
              f"{es_emp:>10.4f} {es_pred_mean:>10.4f} {ratio:>8.3f}")

    # ── Save everything ──────────────────────────────────────────────────
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.generic,)): return obj.item()
        if isinstance(obj, dict): return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_to_serializable(v) for v in obj]
        return obj

    out = {
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "tau_bmlp_total": float(tau),
        "block_B_conditional_coverage": {k: _to_serializable(v) for k, v in block_b.items()},
        "block_C_var99_backtest":      {k: _to_serializable(v) for k, v in block_c.items()},
        "block_D_es_gneiting":          {k: _to_serializable(v) for k, v in block_d.items()},
    }
    out_path = results_dir / "bayesian_risk_advantage.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    print(f"\nResults saved to: {out_path}")
    print(f"Figures saved to: {figures_dir}")
