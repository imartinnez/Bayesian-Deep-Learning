# @author: Íñigo Martínez Jiménez
# Orchestration script for the second stage of the data pipeline.
# We build all input features and the regression target from the raw CSV,
# apply a chronological train/validation/test split, fit a StandardScaler
# on the training data only, and save the processed splits and sliding-window
# arrays that will be fed directly into the PyTorch training loop.

from pathlib import Path
import sys
import numpy as np

# We add the project root to sys.path so that the data package can be imported
# regardless of the working directory from which this script is executed.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data.download import load_config
from data.features import load_raw_data, build_features
from data.target import add_future_vol_target, build_final_dataset, save_final_dataset
from data.splits import split_masks, fit_feature_scaler, transform_features
from data.splits import save_parquet, save_scaler, build_window_dataset, save_window_dataset

CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


if __name__ == "__main__":
    # We load the configuration file and resolve all relevant path and parameter
    # values before touching any data file.
    cfg = load_config(CONFIG_PATH)

    raw_dir = ROOT_DIR / cfg["paths"]["raw"]
    processed_dir = ROOT_DIR / cfg["paths"]["processed"]
    splits_dir = ROOT_DIR / cfg["paths"]["splits"]

    raw_filename = cfg["paths"]["raw_filename"]
    processed_filename = cfg["paths"]["processed_filename"]

    train_filename = cfg["paths"]["train_filename"]
    val_filename = cfg["paths"]["val_filename"]
    test_filename = cfg["paths"]["test_filename"]

    scaler_filename = cfg["paths"]["scaler_filename"]
    train_windows_filename = cfg["paths"]["train_windows_filename"]
    val_windows_filename = cfg["paths"]["val_windows_filename"]
    test_windows_filename = cfg["paths"]["test_windows_filename"]

    feature_cols = cfg["features"]["columns"]
    horizon = cfg["features"]["horizon"]
    window = cfg["features"]["window"]
    target_col = "target"

    train_end = cfg["splits"]["train_end"]
    val_end = cfg["splits"]["val_end"]

    # We construct all output paths from the configuration so that renaming
    # a file in config.yaml propagates through the entire pipeline automatically.
    raw_path = raw_dir / raw_filename
    processed_path = processed_dir / processed_filename

    train_path = splits_dir / train_filename
    val_path = splits_dir / val_filename
    test_path = splits_dir / test_filename

    scaler_path = splits_dir / scaler_filename
    train_windows_path = splits_dir / train_windows_filename
    val_windows_path = splits_dir / val_windows_filename
    test_windows_path = splits_dir / test_windows_filename

    # --- Feature and target construction ---

    # We load the raw OHLCV data, add all configured input features, and attach
    # the forward-looking realized volatility target. The final dropna inside
    # build_final_dataset removes the leading NaNs from rolling windows and
    # the trailing NaNs produced by the forward-shifted target horizon.
    df = load_raw_data(raw_path)
    df = build_features(df, feature_cols)
    df = add_future_vol_target(df, horizon=horizon, target_col=target_col)

    final_df = build_final_dataset(df, feature_names=feature_cols, target_col=target_col)
    save_final_dataset(final_df, processed_path)

    # --- Chronological split ---

    # We compute boolean masks for each temporal partition and apply them to
    # both the unscaled and scaled versions of the DataFrame so we do not need
    # to recompute date boundaries after scaling.
    train_mask, val_mask, test_mask = split_masks(final_df, train_end=train_end, val_end=val_end)

    # We keep unscaled copies of each split so that we can fit the scaler
    # on training data only, which is the key anti-leakage guarantee.
    train_df_raw = final_df.loc[train_mask].copy()
    val_df_raw = final_df.loc[val_mask].copy()
    test_df_raw = final_df.loc[test_mask].copy()

    # --- Feature scaling ---

    # We fit the StandardScaler exclusively on the training partition and then
    # apply the same transformation to all three splits. Calling fit on the full
    # dataset would leak distribution statistics from future periods into the model.
    scaler = fit_feature_scaler(train_df_raw, feature_cols=feature_cols)
    scaled_df = transform_features(final_df, scaler=scaler, feature_cols=feature_cols)

    train_df = scaled_df.loc[train_mask].reset_index(drop=True)
    val_df = scaled_df.loc[val_mask].reset_index(drop=True)
    test_df = scaled_df.loc[test_mask].reset_index(drop=True)

    save_parquet(train_df, train_path)
    save_parquet(val_df, val_path)
    save_parquet(test_df, test_path)
    save_scaler(scaler, scaler_path)

    # --- Sliding-window array construction ---

    # We translate the boolean masks into integer index arrays pointing into the
    # scaled full DataFrame. Passing these global indices to build_window_dataset
    # allows each window to look back across split boundaries when needed, which
    # is legitimate: using past observations as context is not data leakage.
    train_indices = np.flatnonzero(train_mask.to_numpy())
    val_indices = np.flatnonzero(val_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())

    X_train, y_train, dates_train = build_window_dataset(
        full_df=scaled_df,
        target_indices=train_indices,
        feature_cols=feature_cols,
        target_col=target_col,
        window=window,
    )

    X_val, y_val, dates_val = build_window_dataset(
        full_df=scaled_df,
        target_indices=val_indices,
        feature_cols=feature_cols,
        target_col=target_col,
        window=window,
    )

    X_test, y_test, dates_test = build_window_dataset(
        full_df=scaled_df,
        target_indices=test_indices,
        feature_cols=feature_cols,
        target_col=target_col,
        window=window,
    )

    save_window_dataset(X_train, y_train, dates_train, train_windows_path)
    save_window_dataset(X_val, y_val, dates_val, val_windows_path)
    save_window_dataset(X_test, y_test, dates_test, test_windows_path)

    # We print a final summary covering paths, row counts, window shapes, and
    # date ranges so we can confirm the pipeline completed correctly at a glance.
    print(f"Processed dataset saved to: {processed_path}")
    print(f"Train split saved to: {train_path}")
    print(f"Validation split saved to: {val_path}")
    print(f"Test split saved to: {test_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Train windows saved to: {train_windows_path}")
    print(f"Validation windows saved to: {val_windows_path}")
    print(f"Test windows saved to: {test_windows_path}\n\n")

    print(f"Processed rows: {len(final_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}\n\n")

    print(f"Train window shape: {X_train.shape}")
    print(f"Validation window shape: {X_val.shape}")
    print(f"Test window shape: {X_test.shape}\n\n")

    print(f"Train date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Validation date range: {val_df['date'].min().date()} to {val_df['date'].max().date()}")
    print(f"Test date range: {test_df['date'].min().date()} to {test_df['date'].max().date()}\n\n")
