import torch.nn as nn

class BayesianLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int, dense: int, dropout: float):
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

        self.fc1  = nn.Linear(hidden, dense)
        self.relu = nn.ReLU()

        self.fc_mu = nn.Linear(dense, 1)
        self.fc_logvar = nn.Linear(dense, 1)
        nn.init.constant_(self.fc_logvar.bias, -1.2)

    def forward(self, x):
        # h_n: estado oculto final de la LSTM
        # forma: (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]

        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)

        # salida final (batch, 1)
        return self.fc_mu(h), self.fc_logvar(h)