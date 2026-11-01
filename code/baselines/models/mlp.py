"""2-layer MLP baseline - AddBiomechanics (Werling et al. 2024) Table 4.

Architecture: Input -> Conv1d(512) -> ReLU -> Conv1d(512) -> ReLU -> Conv1d(output).
Per-frame model (no temporal context within the model itself).
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, output_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in] -> [B, D_out, T] -> [B, T, D_out]
        return self.net(x.transpose(1, 2)).transpose(1, 2)
