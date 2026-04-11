from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_data(feature_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(feature_path)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def validate_feature_input(df: pd.DataFrame, feature_names: list[str]) -> None:
    required_cols = ["date", "log_return"] + feature_names
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The input feature DataFrame is empty.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("The 'date' column is not sorted in ascending order.")


def add_future_vol_target(df: pd.DataFrame, horizon: int, target_col: str = "target") -> pd.DataFrame:
    df = df.copy()

    future_sq_returns = pd.concat(
        [df["log_return"].shift(-step).pow(2).rename(f"r2_t_plus_{step}") for step in range(1, horizon + 1)],
        axis=1,
    )

    mean_future_sq = future_sq_returns.mean(axis=1)
    realized_vol = np.sqrt(mean_future_sq.where(mean_future_sq > 0))
    df[target_col] = np.log(realized_vol)

    return df


def inspect_target_alignment(df: pd.DataFrame, date_str: str, horizon: int, target_col: str = "target") -> None:
    ts = pd.Timestamp(date_str)
    matches = df.index[df["date"] == ts]

    if len(matches) == 0:
        raise ValueError(f"Date {date_str} was not found in the DataFrame.")

    idx = matches[0]

    if idx + horizon >= len(df):
        raise ValueError(f"Date {date_str} is too close to the end of the sample for a {horizon}-day check.")

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
    final_cols = ["date"] + feature_names + [target_col]
    final_df = df[final_cols].dropna().reset_index(drop=True)
    return final_df


def validate_final_dataset(df: pd.DataFrame, feature_names: list[str], target_col: str = "target") -> None:
    if df.empty:
        raise ValueError("The final DataFrame is empty after dropping NaNs.")

    if df.isna().any().any():
        raise ValueError("The final DataFrame still contains NaN values.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the final DataFrame.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("The final DataFrame is not sorted by date.")

    required_cols = ["date"] + feature_names + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the final DataFrame: {missing_cols}")

    if not np.isfinite(df[target_col]).all():
        raise ValueError("The target column contains non-finite values.")


def save_final_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path