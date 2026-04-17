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
        self.relu = nn.ReLU()

        self.fc_mu_hidden = nn.Linear(hidden, dense)
        self.fc_mu = nn.Linear(dense, 1)

        self.fc_logvar_hidden = nn.Linear(hidden, dense)
        self.fc_logvar = nn.Linear(dense, 1)
        nn.init.constant_(self.fc_logvar.bias, -1.2)

    def forward(self, x):
        # h_n: estado oculto final de la LSTM
        # forma: (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        h = self.dropout(h)

        h_mu = self.relu(self.fc_mu_hidden(h))
        h_mu = self.dropout(h_mu)
        mu = self.fc_mu(h_mu)


        h_logvar = self.relu(self.fc_logvar_hidden(h))
        h_logvar = self.dropout(h_logvar)
        logvar = self.fc_logvar(h_logvar)

        # salida final (batch, 1)
        return mu, logvar