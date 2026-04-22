import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dense_size: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
