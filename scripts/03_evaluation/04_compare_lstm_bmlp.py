"""
Focused comparison: LSTM deterministic vs BMLP on the metrics where
heteroscedastic uncertainty matters.

BMLP metrics are loaded from the pre-computed evaluation JSON so the
results are identical to those reported in 03_evaluate_MLP.py.
LSTM metrics are computed fresh (no pre-existing evaluation JSON).

Computes four targeted comparisons:
  1. Validation gate  — CV(sigma) and Spearman(sigma, |error|)
  2. Coverage by regime — LOW / MED / HIGH
  3. Selective prediction — AURC-RMSE when abstaining on uncertain days
  4. Summary table
"""
from pathlib import Path
import sys
import io
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.data.download import load_config
from src.data.dataset import WindowedTimeSeriesDataset
from src.models.LSTM import BaselineLSTM
from src.training.LSTM_trainer import predict as predict_lstm
from src.evaluation.regimes import classify_regimes, evaluate_by_regime


# ── helpers ──────────────────────────────────────────────────────────────────

def cv(sigma: np.ndarray) -> float:
    mu = np.mean(sigma)
    return float(np.std(sigma) / mu) if mu > 0 else 0.0


def aurc_rmse(y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray,
              steps: int = 10) -> float:
    """Area under the selective RMSE curve (lower = better)."""
    order = np.argsort(uncertainty)
    fractions = np.linspace(0.5, 1.0, steps)
    rmses = []
    for f in fractions:
        n = max(1, int(f * len(order)))
        idx = order[:n]
        rmses.append(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
    widths = np.diff(fractions)
    heights = 0.5 * (np.array(rmses[:-1]) + np.array(rmses[1:]))
    return float(np.sum(widths * heights) / (fractions[-1] - fractions[0]))


def print_row(label: str, lstm_val, bmlp_val, fmt: str = ".4f", winner: str = "low") -> None:
    lv, bv = float(lstm_val), float(bmlp_val)
    l = "N/A" if np.isnan(lv) else f"{lv:{fmt}}"
    b = "N/A" if np.isnan(bv) else f"{bv:{fmt}}"
    if np.isnan(lv) or np.isnan(bv):
        mark = ""
    elif winner == "low":
        mark = "<- BMLP" if bv < lv else ""
    else:
        mark = "<- BMLP" if bv > lv else ""
    print(f"  {label:<35}{l:>12}{b:>12}  {mark}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    cfg = load_config(ROOT_DIR / "config" / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir  = ROOT_DIR / cfg["paths"]["splits"]
    lstm_models = ROOT_DIR / cfg["paths"]["lstm_models"]
    results_dir = ROOT_DIR / cfg["paths"]["har_results"]

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["har_training"]["num_workers"]

    # ── load BMLP results from JSON (stable, seed-independent) ───────────────
    bmlp_json_path = results_dir / cfg["paths"]["har_evaluation_results"]
    with open(bmlp_json_path) as f:
        bmlp_results = json.load(f)

    vg         = bmlp_results["validation_gate"]
    bmlp_cv    = vg["check3_cv_std"]
    bmlp_sp    = vg["check4_spearman"]
    bmlp_gate  = "PASS" if vg["all_pass"] else "FAIL"

    bmlp_reg   = bmlp_results["bayesian"]["regime_metrics"]
    bmlp_rmse  = bmlp_results["bayesian"]["global_metrics"]["rmse"]
    bmlp_aurc_total = bmlp_results["bayesian"]["selective_prediction"]["total_uncertainty"]["aurc_rmse"]
    bmlp_aurc_epist = bmlp_results["bayesian"]["selective_prediction"]["epistemic_uncertainty"]["aurc_rmse"]

    # ── load LSTM datasets ────────────────────────────────────────────────────
    lstm_train = WindowedTimeSeriesDataset.from_npz(splits_dir / cfg["paths"]["train_windows_filename"])
    lstm_val   = WindowedTimeSeriesDataset.from_npz(splits_dir / cfg["paths"]["val_windows_filename"])
    lstm_test  = WindowedTimeSeriesDataset.from_npz(splits_dir / cfg["paths"]["test_windows_filename"])

    # align LSTM test to the 729 HAR test dates (LSTM has one extra: 2024-12-30)
    har_dir   = ROOT_DIR / cfg["paths"]["har_dir"]
    from src.data.har_dataset import HARDataset
    har_test  = HARDataset.from_npz(har_dir / cfg["paths"]["har_test_filename"])
    har_dates = set(pd.to_datetime(har_test.dates))

    lstm_dates = pd.to_datetime(lstm_test.dates)
    common_idx = np.array([i for i, d in enumerate(lstm_dates) if d in har_dates])

    lstm_test_aligned = TensorDataset(
        torch.tensor(lstm_test.X[common_idx], dtype=torch.float32),
        torch.tensor(lstm_test.y[common_idx], dtype=torch.float32).unsqueeze(-1),
    )

    lstm_val_loader  = DataLoader(lstm_val,          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    lstm_test_loader = DataLoader(lstm_test_aligned,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── run LSTM inference ────────────────────────────────────────────────────
    lstm_ckpt = torch.load(lstm_models / cfg["paths"]["baseline_checkpoint"],
                           map_location=device, weights_only=True)
    lstm_model = BaselineLSTM(
        n_features=lstm_train.n_features,
        hidden=lstm_ckpt["hidden_size"],
        num_layers=lstm_ckpt["num_layers"],
        dense=lstm_ckpt["dense_size"],
        dropout=lstm_ckpt.get("dropout", 0.0),
    ).to(device)
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])

    mu_lstm_val,  y_val_lstm  = predict_lstm(lstm_model, lstm_val_loader,  device)
    mu_lstm_test, y_test_lstm = predict_lstm(lstm_model, lstm_test_loader, device)

    sigma_lstm_const = float(np.std(y_val_lstm - mu_lstm_val, ddof=0))
    sigma_lstm_test  = np.full_like(mu_lstm_test, sigma_lstm_const)

    # LSTM regime evaluation (aligned test set)
    regimes_lstm = classify_regimes(np.exp(y_test_lstm), train_rv=np.exp(lstm_train.y))
    reg_lstm     = evaluate_by_regime(y_test_lstm, mu_lstm_test, sigma_lstm_test, regimes_lstm)

    lstm_rmse  = float(np.sqrt(np.mean((y_test_lstm - mu_lstm_test) ** 2)))
    lstm_cv    = cv(sigma_lstm_test)                  # always 0 — constant sigma
    lstm_sp    = float("nan")                         # undefined for constant sigma
    lstm_gate  = "FAIL"                               # fails CV and Spearman by construction

    aurc_lstm = aurc_rmse(y_test_lstm, mu_lstm_test, sigma_lstm_test)

    # ── 1. Validation gate ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1 | Validation gate")
    print("=" * 70)
    print(f"  {'Metric':<35}{'LSTM':>12}{'BMLP':>12}")
    print("-" * 70)
    print_row("CV(sigma)  [need >= 0.10]",        lstm_cv, bmlp_cv, fmt=".4f", winner="high")
    print_row("Spearman(sigma,|err|) [need >= 0.20]", lstm_sp, bmlp_sp, fmt=".4f", winner="high")
    print(f"\n  Overall gate:  LSTM={lstm_gate}   BMLP={bmlp_gate}")

    # ── 2. Coverage by regime ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2 | Coverage by regime  (nominal 90%)")
    print("=" * 70)
    print(f"  {'Regime':<10}{'n(LSTM)':>9}{'n(BMLP)':>9}{'LSTM cov90':>12}{'BMLP cov90':>12}{'LSTM NLL':>10}{'BMLP NLL':>10}")
    print("-" * 70)

    for regime in ("LOW", "MED", "HIGH"):
        rl = reg_lstm[regime]
        rb = bmlp_reg[regime]
        mark = "<- BMLP" if abs(rb["coverage_90"] - 0.90) < abs(rl["coverage_90"] - 0.90) else ""
        print(f"  {regime:<10}{rl['n_obs']:>9}{rb['n_obs']:>9}"
              f"{rl['coverage_90']:>12.4f}{rb['coverage_90']:>12.4f}"
              f"{rl['nll']:>10.4f}{rb['nll']:>10.4f}  {mark}")

    # ── 3. Selective prediction ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3 | Selective prediction  (AURC-RMSE, lower = better)")
    print("=" * 70)
    print(f"  {'Model':<10}{'sigma source':<25}{'AURC-RMSE':>12}  Interpretation")
    print("-" * 70)
    print(f"  {'LSTM':<10}{'constant sigma':<25}{aurc_lstm:>12.4f}  sigma flat -> no abstention signal")
    print(f"  {'BMLP':<10}{'total sigma (heterosc.)':<25}{bmlp_aurc_total:>12.4f}  <- lower = better selective")
    print(f"  {'BMLP':<10}{'epistemic sigma':<25}{bmlp_aurc_epist:>12.4f}  uncertainty signal")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4 | Summary  (<- marks BMLP advantage)")
    print("=" * 70)
    print(f"  {'Property':<40}{'LSTM':>10}{'BMLP':>10}")
    print("-" * 70)
    print_row("RMSE (lower = better)",            lstm_rmse,  bmlp_rmse,       fmt=".4f", winner="low")
    print_row("CV(sigma) (higher = better)",       lstm_cv,    bmlp_cv,         fmt=".4f", winner="high")
    print_row("Spearman(sigma,|err|) (higher)",    lstm_sp,    bmlp_sp,         fmt=".4f", winner="high")
    print_row("Coverage HIGH (closer to 0.90)",
              reg_lstm["HIGH"]["coverage_90"],     bmlp_reg["HIGH"]["coverage_90"], fmt=".4f", winner="high")
    print_row("AURC-RMSE selective (lower)",       aurc_lstm,  bmlp_aurc_total, fmt=".4f", winner="low")
    print_row("Validation gate (PASS=1 / FAIL=0)",
              0.0 if lstm_gate == "FAIL" else 1.0,
              0.0 if bmlp_gate == "FAIL" else 1.0, fmt=".0f", winner="high")
    print()
