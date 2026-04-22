import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self, n_features: int, hidden: int, num_layers: int, dense: int, n_classes: int,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.relu = nn.ReLU()
        self.fc_hidden = nn.Linear(hidden, dense)
        self.fc_out = nn.Linear(dense, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out.mean(dim=1)          # (batch, hidden)
        h = self.relu(self.fc_hidden(h))
        logits = self.fc_out(h)
        return logits