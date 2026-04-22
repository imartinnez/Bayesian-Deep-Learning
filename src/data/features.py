# @author: Íñigo Martínez Jiménez
# This module builds the set of input features used by the model from the
# raw OHLCV data: log returns, high-low range, rolling realized volatility,
# and log-transformed volume.

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# We resolve the project root relative to this file so path construction
# works regardless of where the pipeline scripts are launched from.
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    # We open the YAML configuration file and return its contents as a plain
    # dictionary, keeping the rest of the codebase decoupled from yaml imports.
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    # We read the raw CSV and parse the date column immediately so that
    # downstream code works with datetime objects from the start. We then sort
    # by date, drop any duplicate timestamps, and reset the integer index to
    # guarantee a clean, ordered starting point for feature construction.
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    # We check for the presence of all required columns first, because a missing
    # column would produce cryptic KeyError messages further down the pipeline.
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The raw DataFrame is empty.")

    # We verify that price and volume columns are numeric because reading from
    # a malformed CSV can silently produce object-dtype columns.
    # Sort order and duplicate removal are already guaranteed by load_raw_data.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"The column {col} must be numeric.")


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    # We compute the daily log return as log(close_t / close_{t-1}).
    # Using the log ratio instead of the arithmetic return gives a symmetric,
    # additive measure of price changes that behaves better in regression models.
    # The first row will be NaN because no previous close is available.
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_hl_range(df: pd.DataFrame) -> pd.DataFrame:
    # We compute the log high-low range as a simple proxy for intraday volatility.
    # Taking the logarithm of the high/low ratio keeps the feature on the same
    # scale as log returns and makes it robust to absolute price levels.
    ratio = df["high"] / df["low"]
    df["hl_range"] = np.log(ratio)
    return df


def add_log_rv(df: pd.DataFrame, window: int, return_col: str = "log_return") -> pd.DataFrame:
    # We compute the rolling realized variance as the sum of squared log returns
    # over the past `window` days and then take the logarithm to compress the
    # scale and approximate a more Gaussian distribution. We use where(rv > 0)
    # to avoid passing zero or negative values to np.log on pathological periods.
    # The first (window - 1) rows will be NaN because the rolling window
    # has not yet accumulated enough observations.
    rv = df[return_col].pow(2).rolling(window=window, min_periods=window).sum()
    col_name = f"log_rv_{window}d"
    df[col_name] = np.log(rv.where(rv > 0))
    return df


def add_log_volume(df: pd.DataFrame) -> pd.DataFrame:
    # We apply a log transformation to trading volume to reduce its heavy right
    # skew. We mask zero-volume days with where(volume > 0) to avoid producing
    # -inf values that would propagate NaN through the rest of the pipeline.
    df["log_volume"] = np.log(df["volume"].where(df["volume"] > 0))
    return df

def add_log_rv_ratio(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    short_col = f"log_rv_{short_window}d"
    long_col = f"log_rv_{long_window}d"
    df["log_rv_ratio_5_20"] = df[short_col] - df[long_col]
    return df


def add_log_rv_change(df: pd.DataFrame, window: int = 5, lag: int = 5) -> pd.DataFrame:
    col = f"log_rv_{window}d"
    df["log_rv_5d_change"] = df[col] - df[col].shift(lag)
    return df


def build_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    # We work on a copy so that the raw DataFrame passed in by the caller
    # remains unchanged. Each feature is added only when it appears in the
    # configuration list, which keeps the pipeline flexible without hardcoding
    # the set of features.
    df = df.copy()

    if "log_return" in feature_names:
        df = add_log_return(df)

    if "hl_range" in feature_names:
        df = add_hl_range(df)

    # We dynamically detect rolling realized volatility features by matching
    # names of the form "log_rv_<N>d" and extracting the window length N from
    # the name itself, so no extra configuration keys are needed.
    rv_features = [name for name in feature_names if name.startswith("log_rv_") and name.endswith("d")]
    for feature in rv_features:
        window = int(feature.replace("log_rv_", "").replace("d", ""))
        df = add_log_rv(df, window=window)

    if "log_rv_ratio_5_20" in feature_names:
        df = add_log_rv_ratio(df, short_window=5, long_window=20)

    if "log_rv_5d_change" in feature_names:
        df = add_log_rv_change(df, window=5, lag=5)

    if "log_volume" in feature_names:
        df = add_log_volume(df)

    return df


def validate_features(df: pd.DataFrame, feature_names: list[str]) -> None:
    # We confirm that every expected feature column was successfully created
    # by build_features, catching any silent failure in the builder functions early.
    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns after construction: {missing_features}")

    # We check that hl_range, log_volume, and log_return contain only finite
    # values in their non-null entries. Division by zero or log of a negative
    # number would produce inf or NaN that corrupt the rest of the pipeline.
    if "hl_range" in df.columns:
        invalid_hl = ~np.isfinite(df["hl_range"].dropna())
        if invalid_hl.any():
            raise ValueError("The 'hl_range' column contains non-finite values.")

    if "log_volume" in df.columns:
        invalid_lv = ~np.isfinite(df["log_volume"].dropna())
        if invalid_lv.any():
            raise ValueError("The 'log_volume' column contains non-finite values.")

    if "log_return" in df.columns:
        invalid_lr = ~np.isfinite(df["log_return"].dropna())
        if invalid_lr.any():
            raise ValueError("The 'log_return' column contains non-finite values.")


def save_processed_data(df: pd.DataFrame, processed_path: Path) -> Path:
    # We create the parent directory if it does not yet exist and persist the
    # DataFrame in Parquet format, which preserves column dtypes and is
    # considerably faster to read back than CSV for large datasets.
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    return processed_path
