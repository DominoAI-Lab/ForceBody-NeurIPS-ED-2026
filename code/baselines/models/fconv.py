"""FConv (Fully Convolutional) - MInT (Schneider et al., NeurIPS 2024).

Source: github.com/simplexsigil/motion2muscle/models/m2m_convs2s.py
Architecture: Conv1d(D_in, 256) + 8x ResBlock(Conv1d(256,256,k=3)+ReLU+skip) + LN + Conv1d(256, D_out).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 width: int = 256, num_layers: int = 8):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, width, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=3, padding=1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output_proj = nn.Conv1d(width, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        h = F.relu(self.input_proj(x.transpose(1, 2)))  # [B, C, T]
        for layer in self.layers:
            residual = h
            h = F.relu(layer(h)) + residual
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)  # [B, C, T]
        return self.output_proj(h).transpose(1, 2)  # [B, T, D_out]
