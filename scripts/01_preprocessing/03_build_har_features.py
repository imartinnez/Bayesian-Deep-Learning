from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.data.download import load_config

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def add_log_rv_har(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    r_sq = df["log_return"].pow(2)
    df[col_name] = np.log(r_sq.rolling(window).mean())
    return df


def add_har_target(df: pd.DataFrame, horizon: int, epsilon: float) -> pd.DataFrame:
    r_sq = df["log_return"].pow(2)
    future_rv = r_sq.shift(-horizon)
    df["target_har"] = np.log(future_rv + epsilon)
    return df


def build_har_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    har_cfg = cfg["har"]
    epsilon = har_cfg["epsilon_squared_return"]
    horizon = har_cfg["horizon"]

    df = add_log_rv_har(df, window=1,  col_name="log_rv_1d_har")
    df = add_log_rv_har(df, window=5,  col_name="log_rv_5d_har")
    df = add_log_rv_har(df, window=22, col_name="log_rv_22d_har")

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["log_return_5d"] = df["log_return"].rolling(5).sum()

    if "hl_range" not in df.columns:
        df["hl_range"] = np.log(df["high"] / df["low"])

    if "log_volume" not in df.columns:
        df["log_volume"] = np.log(df["volume"].clip(lower=1e-8))

    df = add_har_target(df, horizon=horizon, epsilon=epsilon)

    return df


if __name__ == "__main__":
    cfg = load_config(ROOT_DIR / "config/config.yaml")

    processed_path = ROOT_DIR / cfg["paths"]["processed"] / cfg["paths"]["processed_filename"]
    df = pd.read_parquet(processed_path)

    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    df = build_har_features(df, cfg)

    feature_cols = cfg["har"]["features"]
    target_col = cfg["har"]["target_col"]

    df = df.dropna(subset=feature_cols + [target_col])

    train_end = cfg["splits"]["train_end"]
    val_end   = cfg["splits"]["val_end"]

    train_df = df[df.index <= train_end]
    val_df   = df[(df.index > train_end) & (df.index <= val_end)]
    test_df  = df[df.index > val_end]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val   = scaler.transform(val_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)

    y_train = train_df[target_col].values
    y_val   = val_df[target_col].values
    y_test  = test_df[target_col].values

    dates_train = train_df.index.values
    dates_val   = val_df.index.values
    dates_test  = test_df.index.values

    har_dir = ROOT_DIR / cfg["paths"]["har_dir"]
    har_dir.mkdir(parents=True, exist_ok=True)

    np.savez(har_dir / cfg["paths"]["har_train_filename"], X=X_train, y=y_train, dates=dates_train)
    np.savez(har_dir / cfg["paths"]["har_val_filename"],   X=X_val,   y=y_val,   dates=dates_val)
    np.savez(har_dir / cfg["paths"]["har_test_filename"],  X=X_test,  y=y_test,  dates=dates_test)

    scaler_path = har_dir / cfg["paths"]["har_feature_scaler_filename"]
    joblib.dump(scaler, scaler_path)

    thresholds_path = har_dir / cfg["paths"]["har_thresholds_filename"]
    p33 = float(np.percentile(y_train, 33))
    p67 = float(np.percentile(y_train, 67))

    with open(thresholds_path, "w") as f:
        json.dump({"low": p33, "high": p67, "n_train": len(y_train)}, f, indent=4)

    print(f"HAR data saved to: {har_dir}")
    print(f"Features: {feature_cols}")
    print(f"Target range train: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"Regime thresholds → p33={p33:.4f}, p67={p67:.4f}")