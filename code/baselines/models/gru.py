"""4-layer unidirectional GRU baseline (paper 'GRU' row): d_model 512, 4 layers.

The class name GRUFallback is kept from the paper code.
"""
import torch
import torch.nn as nn


class GRUFallback(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 512, n_layer: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layer,
                          batch_first=True, bidirectional=False)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h, _ = self.gru(h)
        return self.output_proj(h)
