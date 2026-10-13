import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out, k, dropout):
        super().__init__(
            nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )


class CNN1D(nn.Module):
    """Stacked 1D convolutions + global average pooling.

    Small and quantization friendly, the main candidate for on-device use.
    """

    def __init__(self, n_channels, n_classes, filters=(64, 64, 128), kernel=5, dropout=0.2):
        super().__init__()
        layers, c = [], n_channels
        for f in filters:
            layers.append(ConvBlock(c, f, kernel, dropout))
            c = f
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):            # x: (B, T, C)
        x = self.features(x.transpose(1, 2))
        return self.head(self.pool(x).squeeze(-1))
