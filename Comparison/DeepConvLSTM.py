import torch
import torch.nn as nn


# =============================================================================
# DeepConvLSTM
# =============================================================================
class DeepConvLSTM(nn.Module):

    def __init__(
        self,
        n_channels=9,
        n_classes=6,
        conv_channels=64,
        kernel_size=5,
        num_conv_layers=4,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.10,
    ):
        super().__init__()

        layers = []
        current = n_channels

        for _ in range(num_conv_layers):
            layers.extend(
                [
                    nn.Conv1d(current, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current = conv_channels

        self.convolutions = nn.Sequential(*layers)

        self.recurrent = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, n_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.convolutions(x).transpose(1, 2)
        h, _ = self.recurrent(h)
        return self.classifier(self.dropout(h[:, -1]))
