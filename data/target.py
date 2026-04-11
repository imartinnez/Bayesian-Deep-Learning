# @author: Íñigo Martínez Jiménez
# This module constructs the regression target variable: the log of realized
# volatility computed over the five trading days *following* each observation.
# Special care is taken to ensure that no future information leaks into the
# feature side of the dataset.

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# We resolve the project root relative to this file for consistent path handling.
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    # We open the YAML configuration file and return its contents as a plain
    # dictionary, keeping the rest of the codebase decoupled from yaml imports.
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_data(feature_path: Path) -> pd.DataFrame:
    # We read the Parquet file produced by the features module. If the date
    # column lost its datetime dtype during serialization we coerce it back here.
    # We then sort and deduplicate to guarantee a consistent integer index
    # before we start attaching the target column.
    df = pd.read_parquet(feature_path)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def validate_feature_input(df: pd.DataFrame, feature_names: list[str]) -> None:
    # We verify that both the date column and all feature columns are present,
    # because the target builder specifically relies on log_return being available.
    # Sort order and duplicate removal are already guaranteed by load_feature_data.
    required_cols = ["date", "log_return"] + feature_names
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The input feature DataFrame is empty.")


def add_future_vol_target(df: pd.DataFrame, horizon: int, target_col: str = "target") -> pd.DataFrame:
    # For each day t we build the target as log(RV_{t+1 : t+horizon}), where RV
    # is the root mean squared log return over the horizon window.
    # We work on a copy to leave the caller's DataFrame untouched.
    df = df.copy()

    # We construct each forward-shifted squared return series separately and
    # combine them into a wide DataFrame. Using shift(-step) places the return
    # of day t+step into the row of day t, so row t of the resulting frame
    # holds the squared returns for the entire future window starting at t+1.
    future_sq_returns = pd.concat(
        [df["log_return"].shift(-step).pow(2).rename(f"r2_t_plus_{step}") for step in range(1, horizon + 1)],
        axis=1,
    )

    # We average the squared returns across the horizon and apply a square root
    # to obtain the realized volatility for the period. We guard against zero
    # mean values with where(mean > 0) before taking the logarithm.
    mean_future_sq = future_sq_returns.mean(axis=1)
    realized_vol = np.sqrt(mean_future_sq.where(mean_future_sq > 0))
    df[target_col] = np.log(realized_vol)

    return df


def inspect_target_alignment(df: pd.DataFrame, date_str: str, horizon: int, target_col: str = "target") -> None:
    # We replicate the target calculation by hand for a single chosen date and
    # compare the result to the value stored in the target column. This is a
    # critical sanity check confirming that the target window starts at t+1
    # and never at t, which would introduce look-ahead bias into the entire model.
    ts = pd.Timestamp(date_str)
    matches = df.index[df["date"] == ts]

    if len(matches) == 0:
        raise ValueError(f"Date {date_str} was not found in the DataFrame.")

    idx = matches[0]

    if idx + horizon >= len(df):
        raise ValueError(f"Date {date_str} is too close to the end of the sample for a {horizon}-day check.")

    # We extract the actual future returns directly from DataFrame rows rather
    # than from the shifted columns, keeping the manual calculation fully
    # independent of the vectorized implementation.
    future_returns = df.loc[idx + 1 : idx + horizon, "log_return"].to_numpy()
    squared_returns = future_returns ** 2
    mean_squared = squared_returns.mean()
    realized_vol = np.sqrt(mean_squared)
    manual_target = np.log(realized_vol)
    stored_target = df.loc[idx, target_col]

    print(f"Alignment check for {date_str}")
    print(f"Future returns (t+1 to t+{horizon}): {future_returns}")
    print(f"Squared returns: {squared_returns}")
    print(f"Mean squared return: {mean_squared}")
    print(f"Realized volatility: {realized_vol}")
    print(f"Manual target value: {manual_target}")
    print(f"Stored target value: {stored_target}")
    print(f"Absolute difference: {abs(manual_target - stored_target)}")
    print("-" * 60)


def build_final_dataset(df: pd.DataFrame, feature_names: list[str], target_col: str = "target") -> pd.DataFrame:
    # We select only the columns needed by the model (date, features, target)
    # and drop every row that contains at least one NaN. NaNs are expected at
    # the beginning of the series (from rolling windows in features) and at the
    # end (because the forward-shifted target has no future data to look at).
    final_cols = ["date"] + feature_names + [target_col]
    final_df = df[final_cols].dropna().reset_index(drop=True)
    return final_df


def validate_final_dataset(df: pd.DataFrame, feature_names: list[str], target_col: str = "target") -> None:
    # We confirm the dataset is not empty after the NaN drop, since an empty
    # result would silently produce a trained model on zero samples.
    if df.empty:
        raise ValueError("The final DataFrame is empty after dropping NaNs.")

    # We assert that no NaN values survived the dropna call in build_final_dataset.
    if df.isna().any().any():
        raise ValueError("The final DataFrame still contains NaN values.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the final DataFrame.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("The final DataFrame is not sorted by date.")

    # We verify that the target column contains only real, finite values.
    # Non-finite targets (inf, -inf) would cause the loss function to diverge
    # immediately during training.
    if not np.isfinite(df[target_col]).all():
        raise ValueError("The target column contains non-finite values.")


def save_final_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    # We create the parent directory if it does not yet exist and persist the
    # cleaned dataset in Parquet format to preserve datetime types and allow
    # fast re-loading downstream.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path
