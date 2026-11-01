"""Bidirectional LSTM - MInT (Schneider et al., NeurIPS 2024).

Source: github.com/simplexsigil/motion2muscle/models/m2m_lstm.py
Architecture: Conv1d(D_in, 256) -> BiLSTM(hidden=256, layers=1) -> Linear(512->D_out).
"""
import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_proj = nn.Linear(hidden * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        h = self.input_proj(x.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden]
        h, _ = self.lstm(h)  # [B, T, hidden*2]
        return self.output_proj(h)  # [B, T, D_out]
