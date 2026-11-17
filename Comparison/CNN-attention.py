import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k, stride=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CNNAttentionBaseline(nn.Module):
    def __init__(self, in_channels=9, conv_width=72, num_heads=4, hidden_dim=124,
                 n_classes=6, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, conv_width, 7, dropout=dropout),
            ConvBNAct(conv_width, conv_width, 5, dropout=dropout),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=conv_width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(conv_width)
        self.head = nn.Sequential(
            nn.Linear(conv_width, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        x = self.norm(x + self.attention(x, x, x, need_weights=False)[0])
        x = x.mean(dim=1)
        return self.head(x)
