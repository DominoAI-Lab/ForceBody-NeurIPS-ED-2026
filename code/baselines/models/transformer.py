"""16-layer Transformer encoder - MInT (Schneider et al., NeurIPS 2024).

Source: github.com/simplexsigil/motion2muscle/models/m2m_transformer.py
CLI defaults from option_vq.py: --width=256, --transformer_layer=16.
Architecture: Conv1d(D_in, 256) + PosEnc -> 16x TransformerEncoderLayer(d=256, nhead=16, ff=512, dropout=0.1) -> Conv1d(256, D_out).
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 256, nhead: int = 16,
                 num_layers: int = 16, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Conv1d(d_model, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        h = self.input_proj(x.transpose(1, 2)).transpose(1, 2)  # [B, T, d_model]
        h = self.pos_enc(h)
        h = self.encoder(h)  # [B, T, d_model]
        return self.output_proj(h.transpose(1, 2)).transpose(1, 2)  # [B, T, D_out]
