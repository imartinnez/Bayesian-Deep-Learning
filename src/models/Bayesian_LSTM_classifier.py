import torch.nn as nn


class BayesianLSTMClassifier(nn.Module):
    def __init__(
        self, n_features: int, hidden: int, num_layers: int, dense: int, dropout: float, n_classes: int):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc_hidden = nn.Linear(hidden, dense)
        self.fc_out = nn.Linear(dense, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out.mean(dim=1)          # (batch, hidden) — media sobre los 30 días
        h = self.dropout(h)
        h = self.relu(self.fc_hidden(h))
        h = self.dropout(h)
        logits = self.fc_out(h)
        return logits
