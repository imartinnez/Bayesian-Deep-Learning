# @author: Inigo Martinez Jimenez
# Traditional volatility forecasting baselines used for comparison against
# the deterministic LSTM and the Bayesian LSTM. Each baseline produces
# predictions on the same target date index as the deep-learning models so
# every downstream figure can compare apples-to-apples.
#
# Included baselines:
#     - Historical mean baseline   (constant = mean of train target)
#     - Historical rolling vol     (target proxy using past realized vol)
#     - EWMA (RiskMetrics lambda=0.94)
#     - HAR-RV (Corsi, 2009): daily / weekly / monthly realized volatility
#
# All forecasts are produced in the *log realized volatility* space to match
# the original regression target defined in src/data/target.py.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BaselineForecast:
    """Container for a single baseline forecast aligned to a set of dates."""
    name: str
    dates: np.ndarray           # target dates as strings "YYYY-MM-DD"
    y_true: np.ndarray          # realized log-RV target
    y_pred: np.ndarray          # point forecast of log-RV
    y_std: np.ndarray           # predictive sigma (constant for most baselines)
    family: str = "traditional"

    @property
    def residuals(self) -> np.ndarray:
        return self.y_true - self.y_pred


def _forward_realized_log_vol(log_returns: pd.Series, horizon: int) -> pd.Series:
    # Replicates the target definition in src/data/target.py so that baseline
    # forecasts align with what the neural models are trained to predict.
    squared = pd.concat(
        [log_returns.shift(-step).pow(2).rename(f"r2_{step}") for step in range(1, horizon + 1)],
        axis=1,
    )
    mean_sq = squared.mean(axis=1)
    rv = np.sqrt(mean_sq.where(mean_sq > 0))
    return np.log(rv)


def _backward_realized_variance(log_returns: pd.Series, window: int) -> pd.Series:
    # Rolling sum of squared log returns over the past `window` days, divided
    # by window to get a per-day variance. Matches the classical realized
    # variance definition.
    r2 = log_returns.pow(2)
    return r2.rolling(window=window, min_periods=window).sum() / window


def _var_to_log_rv(var: pd.Series) -> pd.Series:
    # Convert a per-day variance to our target units: log of the period
    # realized volatility (log of sqrt(variance)).
    return 0.5 * np.log(var.where(var > 0))


def _ewma_variance(log_returns: pd.Series, lam: float = 0.94) -> pd.Series:
    # RiskMetrics EWMA variance. We seed the recursion with the sample
    # variance of the first 30 observations and then propagate forward.
    r2 = log_returns.pow(2).to_numpy()
    n = len(r2)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=log_returns.index)

    seed = np.nanmean(r2[:30]) if n >= 30 else np.nanmean(r2)
    prev = seed
    out[0] = prev
    for t in range(1, n):
        if not np.isfinite(r2[t]):
            out[t] = prev
            continue
        prev = lam * prev + (1 - lam) * r2[t]
        out[t] = prev
    return pd.Series(out, index=log_returns.index)


def build_feature_dataframe(
    processed_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Attach forecast-compatible HAR / EWMA features aligned to each target date.

    The processed dataset already exposes log_return plus the realized
    volatility target. We rebuild the raw realized variance components here so
    that any baseline can be evaluated without reaching back to the raw CSV.
    """
    df = processed_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log_returns = df["log_return"]

    # Realized variances at the three HAR scales (daily, weekly, monthly)
    # ending at date t inclusive. We then take logs to keep the regression
    # on the log-volatility scale.
    df["rv_day"] = log_returns.pow(2)
    df["rv_week"] = _backward_realized_variance(log_returns, window=5)
    df["rv_month"] = _backward_realized_variance(log_returns, window=22)

    df["log_rv_day"] = _var_to_log_rv(df["rv_day"])
    df["log_rv_week"] = _var_to_log_rv(df["rv_week"])
    df["log_rv_month"] = _var_to_log_rv(df["rv_month"])

    # EWMA predicted volatility: at date t we use the current EWMA variance
    # as a constant forecast over the next `horizon` days. In log-RV units
    # that reduces to 0.5 * log(var) since the horizon mean equals the constant.
    df["ewma_var"] = _ewma_variance(log_returns, lam=0.94)
    df["ewma_log_rv"] = _var_to_log_rv(df["ewma_var"])

    # Historical rolling volatility forecast: use the most recent `horizon`
    # days of realized variance as the forecast for the next horizon days.
    df["hist_vol_log_rv"] = _var_to_log_rv(_backward_realized_variance(log_returns, window=horizon))

    # True forward target re-computed here for sanity checking; should match
    # the "target" column of the processed dataset almost exactly.
    df["target_recomputed"] = _forward_realized_log_vol(log_returns, horizon=horizon)

    return df


def _restrict_to_dates(df: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    # Select the rows of the engineered feature DataFrame that correspond to
    # the forecast dates we want to evaluate against.
    wanted = pd.to_datetime(pd.Series(dates))
    out = df.set_index("date").loc[wanted].reset_index()
    return out


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------


def mean_baseline_forecast(train_target: np.ndarray, target_dates: np.ndarray,
                           y_true: np.ndarray, residual_std: float) -> BaselineForecast:
    # Constant mean of the training target. Provides a lower-bar sanity check
    # that every trained model is expected to beat.
    mean_val = float(np.mean(train_target))
    y_pred = np.full_like(y_true, mean_val, dtype=float)
    y_std = np.full_like(y_true, residual_std, dtype=float)
    return BaselineForecast(
        name="Mean Baseline",
        dates=np.asarray(target_dates),
        y_true=y_true,
        y_pred=y_pred,
        y_std=y_std,
    )


def historical_vol_forecast(features_df: pd.DataFrame,
                            dates: np.ndarray,
                            y_true: np.ndarray,
                            residual_std_val: float) -> BaselineForecast:
    # Use the backward realized volatility over the horizon as the forecast:
    # "past N days of volatility is the best guess for the next N days".
    sub = _restrict_to_dates(features_df, dates)
    y_pred = sub["hist_vol_log_rv"].to_numpy(dtype=float)
    valid = np.isfinite(y_pred)

    y_pred_safe = np.where(valid, y_pred, np.nanmean(y_pred[valid]))
    y_std = np.full_like(y_pred_safe, residual_std_val, dtype=float)
    return BaselineForecast(
        name="Historical Volatility",
        dates=np.asarray(dates),
        y_true=y_true,
        y_pred=y_pred_safe,
        y_std=y_std,
    )


def ewma_forecast(features_df: pd.DataFrame,
                  dates: np.ndarray,
                  y_true: np.ndarray,
                  residual_std_val: float) -> BaselineForecast:
    # RiskMetrics EWMA variance applied in log-volatility space. We keep the
    # predictive sigma constant (the EWMA level at validation is a free
    # parameter that we set from validation residuals so calibration is
    # meaningful).
    sub = _restrict_to_dates(features_df, dates)
    y_pred = sub["ewma_log_rv"].to_numpy(dtype=float)
    y_pred_safe = np.where(np.isfinite(y_pred), y_pred, np.nanmean(y_pred[np.isfinite(y_pred)]))
    y_std = np.full_like(y_pred_safe, residual_std_val, dtype=float)
    return BaselineForecast(
        name="EWMA (RiskMetrics)",
        dates=np.asarray(dates),
        y_true=y_true,
        y_pred=y_pred_safe,
        y_std=y_std,
    )


def fit_har_model(features_df: pd.DataFrame,
                  train_mask: np.ndarray) -> dict:
    # Ordinary least squares fit of
    #   log_rv_target ~ const + beta_d*log_rv_day + beta_w*log_rv_week + beta_m*log_rv_month
    # where the target is the forward-looking log realized vol already stored
    # in the processed dataset.
    sub = features_df.loc[train_mask].copy()
    cols = ["log_rv_day", "log_rv_week", "log_rv_month", "target"]
    sub = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if sub.empty:
        raise ValueError("Cannot fit HAR: no valid training rows.")

    X = np.column_stack([
        np.ones(len(sub)),
        sub["log_rv_day"].to_numpy(),
        sub["log_rv_week"].to_numpy(),
        sub["log_rv_month"].to_numpy(),
    ])
    y = sub["target"].to_numpy()

    # Use the normal equation; for five features / a couple of thousand rows
    # this is numerically well behaved.
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    sigma_train = float(np.std(residuals, ddof=1))

    return {
        "beta": beta,
        "sigma_train": sigma_train,
        "n_train": int(len(sub)),
        "feature_names": ["const", "log_rv_day", "log_rv_week", "log_rv_month"],
    }


def har_forecast(features_df: pd.DataFrame,
                 har_model: dict,
                 dates: np.ndarray,
                 y_true: np.ndarray,
                 residual_std_val: float) -> BaselineForecast:
    sub = _restrict_to_dates(features_df, dates)
    X = np.column_stack([
        np.ones(len(sub)),
        sub["log_rv_day"].to_numpy(dtype=float),
        sub["log_rv_week"].to_numpy(dtype=float),
        sub["log_rv_month"].to_numpy(dtype=float),
    ])
    beta = har_model["beta"]

    valid = np.isfinite(X).all(axis=1)
    y_pred = np.full(len(sub), np.nan, dtype=float)
    y_pred[valid] = X[valid] @ beta

    if (~valid).any():
        # Backfill any leftover NaN with the fitted mean.
        y_pred[~valid] = float(np.nanmean(y_pred[valid]))

    y_std = np.full_like(y_pred, residual_std_val, dtype=float)

    return BaselineForecast(
        name="HAR-RV",
        dates=np.asarray(dates),
        y_true=y_true,
        y_pred=y_pred,
        y_std=y_std,
    )


# ---------------------------------------------------------------------------
# Calibration of predictive sigma
# ---------------------------------------------------------------------------


def calibrated_residual_std(
    features_df: pd.DataFrame,
    forecast_fn,
    val_dates: np.ndarray,
    val_y_true: np.ndarray,
    initial_sigma: float = 1.0,
) -> float:
    # Computes the standard deviation of validation residuals when applying a
    # given baseline forecaster. This is how the original `05_evaluate.py`
    # derives a constant sigma for the deterministic LSTM, and we reuse the
    # same philosophy here for HAR / EWMA / Historical so their predictive
    # distributions are at least calibrated to the validation period.
    fc = forecast_fn(features_df, val_dates, val_y_true, residual_std_val=initial_sigma)
    residuals = fc.y_true - fc.y_pred
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 2:
        return float(initial_sigma)
    return float(np.std(residuals, ddof=0))


def save_forecast_npz(forecast: BaselineForecast, path: Path) -> Path:
    # Persist a forecast so downstream figure scripts can read it back without
    # refitting.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dates=forecast.dates,
        y_true=forecast.y_true,
        y_pred=forecast.y_pred,
        y_std=forecast.y_std,
        name=np.array(forecast.name),
        family=np.array(forecast.family),
    )
    return path


def load_forecast_npz(path: Path) -> BaselineForecast:
    data = np.load(path, allow_pickle=False)
    return BaselineForecast(
        name=str(data["name"]),
        family=str(data["family"]) if "family" in data.files else "traditional",
        dates=data["dates"],
        y_true=data["y_true"],
        y_pred=data["y_pred"],
        y_std=data["y_std"],
    )
