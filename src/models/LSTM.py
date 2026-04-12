import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int, dense: int, dropout: float = 0.0):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc1  = nn.Linear(hidden, dense)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(dense, 1)
    
    def forward(self, x):
        # h_n: estado oculto final de la LSTM
        # forma: (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)

        # nos quedamos con la última capa
        # forma: (batch, hidden)
        h = h_n[-1]

        # proyección a espacio más pequeño + no linealidad
        h = self.relu(self.fc1(h))
        
        # salida final (batch, 1)
        return self.fc2(h)