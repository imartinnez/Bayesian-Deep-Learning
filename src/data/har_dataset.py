from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.dates = dates
        self.n_features = X.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    @classmethod
    def from_npz(cls, path: Path) -> "HARDataset":
        data = np.load(path, allow_pickle=True)
        return cls(X=data["X"], y=data["y"], dates=data["dates"])

    def to_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=self.X, y=self.y, dates=self.dates)