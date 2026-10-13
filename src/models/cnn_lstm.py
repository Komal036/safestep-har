import torch.nn as nn

from src.models.cnn import ConvBlock


class CNNLSTM(nn.Module):
    """Conv layers extract local motion patterns, LSTM models their order."""

    def __init__(self, n_channels, n_classes, filters=(64, 64), kernel=5,
                 hidden=64, lstm_layers=2, dropout=0.2):
        super().__init__()
        layers, c = [], n_channels
        for f in filters:
            layers.append(ConvBlock(c, f, kernel, dropout))
            c = f
        self.features = nn.Sequential(*layers)
        self.lstm = nn.LSTM(c, hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):            # x: (B, T, C)
        x = self.features(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
