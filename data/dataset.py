# @author: Íñigo Martínez Jiménez
# This module defines the WindowedTimeSeriesDataset class, a PyTorch Dataset
# wrapper that serves pre-built sliding-window arrays to DataLoaders during
# model training and evaluation.

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedTimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset that wraps a set of fixed-length input windows and their
    corresponding scalar targets for time-series regression.

    Each sample consists of a 2-D feature matrix of shape (window_size, n_features)
    representing the past `window_size` trading days, and a scalar target value
    representing the log realized volatility over the next horizon days.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray | None = None):
        """
        Initialize the dataset by storing and validating the input arrays.

        Args:
            X (np.ndarray): Feature array of shape (n_samples, window_size, n_features).
            y (np.ndarray): Target array of shape (n_samples,).
            dates (np.ndarray | None): Optional array of date strings aligned with y.
        """
        # We cast both arrays to float32 explicitly so that they are immediately
        # ready for PyTorch without any additional conversion in __getitem__.
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.dates = None if dates is None else np.asarray(dates)

        # We validate the expected array shapes up front so that any mismatch
        # introduced by the upstream pipeline surfaces here rather than inside
        # the training loop, where the error message would be harder to trace.
        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (n_samples, window, n_features), got {self.X.shape}.")

        if self.y.ndim != 1:
            raise ValueError(f"y must have shape (n_samples,), got {self.y.shape}.")

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples.")

        if self.dates is not None and len(self.dates) != len(self.y):
            raise ValueError("dates and y must have the same number of samples.")

        # We reject non-finite values immediately because they would cause the
        # loss function to return NaN on the very first training step.
        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values.")

        if not np.isfinite(self.y).all():
            raise ValueError("y contains non-finite values.")

        # We store window size and feature count as attributes for quick
        # inspection without having to index into X.shape directly.
        self.window_size = self.X.shape[1]
        self.n_features = self.X.shape[2]

    @classmethod
    def from_npz(cls, file_path: str | Path) -> "WindowedTimeSeriesDataset":
        """
        Load a dataset that was previously saved with np.savez_compressed.

        Args:
            file_path (str | Path): Path to the .npz file produced by save_window_dataset.

        Returns:
            WindowedTimeSeriesDataset: A fully initialized dataset instance.
        """
        # We disable pickle when loading to prevent arbitrary code execution
        # in case the file has been tampered with.
        data = np.load(file_path, allow_pickle=False)
        X = data["X"]
        y = data["y"]
        dates = data["dates"] if "dates" in data.files else None
        return cls(X=X, y=y, dates=dates)

    def __len__(self) -> int:
        # We return the total number of windows so that DataLoader can infer
        # how many batches to produce per epoch.
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # We convert the NumPy slices to PyTorch tensors on the fly. The target
        # is wrapped in a 1-element tensor so that it can be concatenated cleanly
        # with batched model predictions during loss computation.
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.tensor([self.y[idx]], dtype=torch.float32)
        return X_item, y_item

    def get_date(self, idx: int) -> str | None:
        # We expose a convenience accessor to retrieve the target date for a
        # given sample index, useful for aligning predictions with a time axis
        # when plotting results.
        if self.dates is None:
            return None
        return str(self.dates[idx])

    def summary(self) -> None:
        # We print a compact overview of the dataset dimensions and date range
        # to allow a quick sanity check before starting a training run.
        print(f"Number of samples: {len(self)}")
        print(f"Window size: {self.window_size}")
        print(f"Number of features: {self.n_features}")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

        if self.dates is not None:
            print(f"First target date: {self.dates[0]}")
            print(f"Last target date: {self.dates[-1]}")
