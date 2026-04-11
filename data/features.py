from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The raw DataFrame is empty.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the raw data.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("The 'date' column is not sorted in ascending order.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"The column {col} must be numeric.")


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_hl_range(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["high"] / df["low"]
    df["hl_range"] = np.log(ratio)
    return df


def add_log_rv(df: pd.DataFrame, window: int, return_col: str = "log_return") -> pd.DataFrame:
    rv = df[return_col].pow(2).rolling(window=window, min_periods=window).sum()
    col_name = f"log_rv_{window}d"
    df[col_name] = np.log(rv.where(rv > 0))
    return df


def add_log_volume(df: pd.DataFrame) -> pd.DataFrame:
    df["log_volume"] = np.log(df["volume"].where(df["volume"] > 0))
    return df


def build_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df = df.copy()

    if "log_return" in feature_names:
        df = add_log_return(df)

    if "hl_range" in feature_names:
        df = add_hl_range(df)

    rv_features = [name for name in feature_names if name.startswith("log_rv_") and name.endswith("d")]
    for feature in rv_features:
        window = int(feature.replace("log_rv_", "").replace("d", ""))
        df = add_log_rv(df, window=window)

    if "log_volume" in feature_names:
        df = add_log_volume(df)

    return df


def validate_features(df: pd.DataFrame, feature_names: list[str]) -> None:
    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns after construction: {missing_features}")

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
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    return processed_path