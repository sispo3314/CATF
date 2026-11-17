import torch
import torch.nn as nn


# =============================================================================
# Transformer Encoder
# =============================================================================
class TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_channels=9,
        n_classes=6,
        d_model=64,
        num_heads=4,
        dim_feedforward=128,
        num_layers=2,
        dropout=0.10,
        max_length=512,
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_channels, d_model)

        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, d_model) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

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
        h = x.transpose(1, 2)
        T = h.shape[1]
        h = self.input_projection(h) + self.position_embedding[:, :T]
        h = self.encoder(h)
        return self.classifier(self.norm(h).mean(dim=1))
