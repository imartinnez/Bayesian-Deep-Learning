import torch.nn as nn


class BayesianMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dense_size: int, dropout: float):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

        nn.init.constant_(self.fc_logvar[-1].bias, -1.2)

    def forward(self, x):
        h = self.shared(x)
        mu = self.fc_mu(h).squeeze(-1)
        logvar = self.fc_logvar(h).squeeze(-1)
        return mu, logvar