"""Architectures of the GRF baselines (paper GRF table, ForceBody rows).

  mlp_deep : MLP with LayerNorm and 2 residual blocks (hidden 1024)
  gln      : GroundLinkNet-style dilated 1D convolutions over the time window (hidden 384, 6 layers)

Each model takes a flattened window [B, in_dim] = [B, W * F_per_frame] and returns [B, out_dim].
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
class MLPDeep(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, n_blocks=2, dropout=0.1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.exit = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        h = self.entry(x)
        for b in self.blocks:
            h = b(h)
        return self.exit(h)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim); self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = torch.relu(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        return torch.relu(x + h)


# ------------------------------------------------------------------ #
class GroundLinkNet(nn.Module):
    """Larger conv-based architecture inspired by Han 2023's GroundLinkNet.

    Uses dilated 1D convolutions over the time window with deeper channels.
    """
    def __init__(self, in_dim, out_dim, window=21, hidden=384, n_layers=6,
                 dropout=0.1):
        super().__init__()
        if in_dim % window != 0:
            raise ValueError(f"in_dim {in_dim} not divisible by window {window}")
        self.window = window
        self.f_per_frame = in_dim // window
        layers = []
        ch_in = self.f_per_frame
        # 6 dilated conv layers with growing receptive field
        for i in range(n_layers):
            dilation = 2 ** (i % 3)
            layers += [
                nn.Conv1d(ch_in, hidden, kernel_size=5, padding=2 * dilation,
                          dilation=dilation),
                nn.BatchNorm1d(hidden), nn.ReLU(),
            ]
            if dropout > 0 and i < n_layers - 1:
                layers.append(nn.Dropout(dropout))
            ch_in = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.window, self.f_per_frame).transpose(1, 2).contiguous()
        h = self.conv(x)
        return self.head(h)


# ------------------------------------------------------------------ #
def build_model(arch: str, in_dim: int, out_dim: int, window: int = 21,
                dropout: float = 0.1) -> nn.Module:
    arch = arch.lower()
    if arch in ("mlp", "mlp_deep"):
        return MLPDeep(in_dim, out_dim, hidden=1024, n_blocks=2, dropout=dropout)
    if arch == "gln":
        return GroundLinkNet(in_dim, out_dim, window=window, hidden=384, n_layers=6, dropout=dropout)
    raise ValueError(f"unknown arch: {arch}")
