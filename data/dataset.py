from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray | None = None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.dates = None if dates is None else np.asarray(dates)

        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (n_samples, window, n_features), got {self.X.shape}.")

        if self.y.ndim != 1:
            raise ValueError(f"y must have shape (n_samples,), got {self.y.shape}.")

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples.")

        if self.dates is not None and len(self.dates) != len(self.y):
            raise ValueError("dates and y must have the same number of samples.")

        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values.")

        if not np.isfinite(self.y).all():
            raise ValueError("y contains non-finite values.")

        self.window_size = self.X.shape[1]
        self.n_features = self.X.shape[2]

    @classmethod
    def from_npz(cls, file_path: str | Path) -> "WindowedTimeSeriesDataset":
        data = np.load(file_path, allow_pickle=False)
        X = data["X"]
        y = data["y"]
        dates = data["dates"] if "dates" in data.files else None
        return cls(X=X, y=y, dates=dates)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.tensor([self.y[idx]], dtype=torch.float32)
        return X_item, y_item

    def get_date(self, idx: int) -> str | None:
        if self.dates is None:
            return None
        return str(self.dates[idx])

    def summary(self) -> None:
        print(f"Number of samples: {len(self)}")
        print(f"Window size: {self.window_size}")
        print(f"Number of features: {self.n_features}")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

        if self.dates is not None:
            print(f"First target date: {self.dates[0]}")
            print(f"Last target date: {self.dates[-1]}")