from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_data(processed_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(processed_path)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def validate_processed_data(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
    required_cols = ["date"] + feature_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The processed DataFrame is empty.")

    if df.isna().any().any():
        raise ValueError("The processed DataFrame contains NaN values.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the processed DataFrame.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("The 'date' column is not sorted in ascending order.")


def split_masks(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train_mask = df["date"] <= train_end_ts
    val_mask = (df["date"] > train_end_ts) & (df["date"] <= val_end_ts)
    test_mask = df["date"] > val_end_ts

    return train_mask, val_mask, test_mask


def validate_split_masks(train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> None:
    if not train_mask.any():
        raise ValueError("Train split is empty.")

    if not val_mask.any():
        raise ValueError("Validation split is empty.")

    if not test_mask.any():
        raise ValueError("Test split is empty.")

    overlap = (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)) != 1
    if overlap.any():
        raise ValueError("The split masks overlap or leave gaps.")


def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out


def save_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def save_scaler(scaler: StandardScaler, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    return output_path


def build_window_dataset(full_df: pd.DataFrame, target_indices: np.ndarray, feature_cols: list[str], target_col: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list = []
    y_list = []
    date_list = []

    for idx in target_indices:
        if idx < window - 1:
            continue

        start_idx = idx - window + 1
        x_window = full_df.loc[start_idx:idx, feature_cols].to_numpy(dtype=np.float32)
        y_value = np.float32(full_df.loc[idx, target_col])
        date_value = full_df.loc[idx, "date"].strftime("%Y-%m-%d")

        x_list.append(x_window)
        y_list.append(y_value)
        date_list.append(date_value)

    if not x_list:
        raise ValueError("No windows could be created for this split.")

    X = np.stack(x_list)
    y = np.array(y_list, dtype=np.float32)
    dates = np.array(date_list)

    return X, y, dates


def save_window_dataset(X: np.ndarray, y: np.ndarray, dates: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, dates=dates)
    return output_path


def describe_split(df: pd.DataFrame, split_name: str) -> None:
    print(f"{split_name} rows: {len(df)}")
    print(f"{split_name} date range: {df['date'].min().date()} to {df['date'].max().date()}")


def describe_windows(X: np.ndarray, y: np.ndarray, dates: np.ndarray, split_name: str, window: int) -> None:
    print(f"{split_name} window samples: {len(X)}")
    print(f"{split_name} X shape: {X.shape}")
    print(f"{split_name} y shape: {y.shape}")
    print(f"{split_name} first target date: {dates[0]}")
    print(f"{split_name} last target date: {dates[-1]}")
    print(f"{split_name} window length: {window}")