# @author: Íñigo Martínez Jiménez
# This module handles the chronological train/validation/test split,
# feature scaling (fitted exclusively on the training set), and the
# construction of sliding-window arrays ready for use with PyTorch DataLoaders.

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
import json
from sklearn.preprocessing import StandardScaler

# We resolve the project root relative to this file for consistent path handling.
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    # We open the YAML configuration file and return its contents as a plain
    # dictionary, keeping the rest of the codebase decoupled from yaml imports.
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_data(processed_path: Path) -> pd.DataFrame:
    # We load the cleaned Parquet file that contains both features and target.
    # If the date column lost its dtype during serialization we coerce it back.
    # We then sort and deduplicate as a defensive measure before partitioning.
    df = pd.read_parquet(processed_path)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def validate_processed_data(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
    # We confirm that every column the split and scaling functions will access
    # is present before any processing begins.
    required_cols = ["date"] + feature_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The processed DataFrame is empty.")

    # We check for NaN values because the NaN removal should already have been
    # performed by build_final_dataset in target.py. Any NaN at this stage would
    # indicate a broken upstream step.
    # Sort order and duplicate removal are already guaranteed by load_processed_data.
    if df.isna().any().any():
        raise ValueError("The processed DataFrame contains NaN values.")


def split_masks(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    # We define the three mutually exclusive temporal partitions using boolean
    # masks rather than subsetting the DataFrame directly. Keeping the masks
    # separate from the data lets us apply them later to both the raw and scaled
    # versions of the DataFrame without recomputing the date boundaries.
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train_mask = df["date"] <= train_end_ts
    val_mask = (df["date"] > train_end_ts) & (df["date"] <= val_end_ts)
    test_mask = df["date"] > val_end_ts

    return train_mask, val_mask, test_mask


def validate_split_masks(train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> None:
    # We verify that none of the three splits is empty, which would indicate
    # a misconfigured date boundary in the configuration file.
    if not train_mask.any():
        raise ValueError("Train split is empty.")

    if not val_mask.any():
        raise ValueError("Validation split is empty.")

    if not test_mask.any():
        raise ValueError("Test split is empty.")

    # We confirm that every row belongs to exactly one split by checking that
    # the element-wise sum of the three integer-cast masks equals 1 everywhere.
    # Any deviation means the masks overlap or leave a gap in the timeline.
    overlap = (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)) != 1
    if overlap.any():
        raise ValueError("The split masks overlap or leave gaps.")


def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    # We fit a StandardScaler exclusively on the training partition so that
    # validation and test data are standardized with statistics the model has
    # never seen. Fitting on the full dataset would leak distribution information
    # from future periods into the training process.
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]) -> pd.DataFrame:
    # We apply the pre-fitted scaler to the requested feature columns and return
    # a copy of the DataFrame with those columns replaced by their standardized
    # counterparts. The target column and date are left untouched.
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out


def save_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    # We create the output directory if it does not exist and serialize the
    # split DataFrame in Parquet format to preserve dtypes efficiently.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def save_scaler(scaler: StandardScaler, output_path: Path) -> Path:
    # We persist the fitted scaler with joblib so that it can be reloaded
    # during inference to apply the exact same standardization to new data.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    return output_path


def build_window_dataset(
    full_df: pd.DataFrame,
    target_indices: np.ndarray,
    feature_cols: list[str],
    target_col: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # We construct the sliding-window arrays by iterating over the target indices
    # of the current split. For each index we look back `window` rows into the
    # scaled full DataFrame to form the input matrix X, and read the target value
    # at that index as the label y.
    # Working against the full scaled DataFrame (not just the split subset) lets
    # us build complete windows at the start of validation and test splits by
    # drawing context rows from the previous partition. This is legitimate in
    # time-series modelling: using past observations as input is not data leakage.
    x_list = []
    y_list = []
    date_list = []

    for idx in target_indices:
        # We skip indices where not enough history exists to fill the window.
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

    # We stack the list of 2-D arrays into a 3-D array of shape
    # (n_samples, window, n_features) that can be consumed directly by PyTorch.
    X = np.stack(x_list)
    y = np.array(y_list, dtype=np.float32)
    dates = np.array(date_list)

    return X, y, dates


def save_window_dataset(X: np.ndarray, y: np.ndarray, dates: np.ndarray, output_path: Path) -> Path:
    # We save all three arrays together in a single compressed .npz archive so
    # that the Dataset class can reload them with a single np.load call.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, dates=dates)
    return output_path


def describe_split(df: pd.DataFrame, split_name: str) -> None:
    # We print a quick summary of the split size and date boundaries for a
    # visual sanity check after partitioning.
    print(f"{split_name} rows: {len(df)}")
    print(f"{split_name} date range: {df['date'].min().date()} to {df['date'].max().date()}")


def describe_windows(X: np.ndarray, y: np.ndarray, dates: np.ndarray, split_name: str, window: int) -> None:
    # We display the final array dimensions and the target date range so we
    # can visually confirm the window construction before training begins.
    print(f"{split_name} window samples: {len(X)}")
    print(f"{split_name} X shape: {X.shape}")
    print(f"{split_name} y shape: {y.shape}")
    print(f"{split_name} first target date: {dates[0]}")
    print(f"{split_name} last target date: {dates[-1]}")
    print(f"{split_name} window length: {window}")

def build_window_dataset_clf(
    full_df: pd.DataFrame,
    target_indices: np.ndarray,
    feature_cols: list[str],
    target_col: str,
    label_col: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_list = []
    y_label_list = []
    y_cont_list = []
    date_list = []

    for idx in target_indices:
        if idx < window - 1:
            continue

        start_idx = idx - window + 1
        x_window = full_df.loc[start_idx:idx, feature_cols].to_numpy(dtype=np.float32)
        y_label = int(full_df.loc[idx, label_col])
        y_cont = np.float32(full_df.loc[idx, target_col])
        date_value = full_df.loc[idx, "date"].strftime("%Y-%m-%d")

        x_list.append(x_window)
        y_label_list.append(y_label)
        y_cont_list.append(y_cont)
        date_list.append(date_value)

    if not x_list:
        raise ValueError("No windows could be created for this split.")

    X = np.stack(x_list)
    y_label = np.array(y_label_list, dtype=np.int64)
    y_continuous = np.array(y_cont_list, dtype=np.float32)
    dates = np.array(date_list)

    return X, y_label, y_continuous, dates


def save_window_dataset_clf(
    X: np.ndarray,
    y_label: np.ndarray,
    y_continuous: np.ndarray,
    dates: np.ndarray,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y_label=y_label, y_continuous=y_continuous, dates=dates)
    return output_path


def save_thresholds(thresholds: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    return output_path