from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationWindowDataset(Dataset):
    """
    PyTorch Dataset for classification. Wraps pre-built sliding-window arrays
    with integer class labels (0=LOW, 1=MED, 2=HIGH).
    """

    def __init__(
        self,
        X: np.ndarray,
        y_label: np.ndarray,
        y_continuous: np.ndarray,
        dates: np.ndarray | None = None,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y_label = np.asarray(y_label, dtype=np.int64)
        self.y_continuous = np.asarray(y_continuous, dtype=np.float32)
        self.dates = None if dates is None else np.asarray(dates)

        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (n_samples, window, n_features), got {self.X.shape}.")
        if self.y_label.ndim != 1:
            raise ValueError(f"y_label must have shape (n_samples,), got {self.y_label.shape}.")
        if len(self.X) != len(self.y_label):
            raise ValueError("X and y_label must have the same number of samples.")
        if not np.isin(self.y_label, [0, 1, 2]).all():
            raise ValueError("y_label must contain only values 0, 1 or 2.")
        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values.")

        self.window_size = self.X.shape[1]
        self.n_features = self.X.shape[2]

    @classmethod
    def from_npz(cls, file_path: str | Path) -> "ClassificationWindowDataset":
        data = np.load(file_path, allow_pickle=False)
        return cls(
            X=data["X"],
            y_label=data["y_label"],
            y_continuous=data["y_continuous"],
            dates=data["dates"] if "dates" in data.files else None,
        )

    def __len__(self) -> int:
        return len(self.y_label)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        label_item = torch.tensor(self.y_label[idx], dtype=torch.long)
        return X_item, label_item