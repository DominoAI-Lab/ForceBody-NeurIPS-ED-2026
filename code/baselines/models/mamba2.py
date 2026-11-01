"""Mamba2 sequence model - MInT (Schneider et al., NeurIPS 2024).

Source: github.com/simplexsigil/motion2muscle/models/m2m_mamba.py
Architecture: Linear(D_in, 512) -> 8x Mamba2Block(d_model=512) + RMSNorm -> Linear(512, D_out).

With mamba_ssm 2.3.1 + triton the Mamba2 kernels only work on the process's cuda:0
("Pointer argument cannot be accessed from Triton" otherwise): select the GPU with
CUDA_VISIBLE_DEVICES and pass --device cuda:0. A forward+backward test at the training
shape runs at import; if it fails, building the model raises an error.
"""
import os
import torch
import torch.nn as nn


def _smoke_test_mamba2(layer_cls, d_model: int = 512) -> bool:
    """Run forward+backward at production-like shape to catch the Triton
    autotuner crash that only triggers with larger batch."""
    if not torch.cuda.is_available():
        return False
    try:
        with torch.enable_grad():
            m = layer_cls(d_model=d_model).cuda()
            opt = torch.optim.Adam(m.parameters(), lr=1e-4)
            # production-ish shape - must crash here if it would crash in real training
            x = torch.randn(128, 256, d_model, device="cuda")
            opt.zero_grad()
            out = m(x)
            out.mean().backward()
            opt.step()
        del m, x, out, opt
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[models.mamba2] GPU test failed: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return False


# Decide once at import time whether the Mamba2 kernel works.
MAMBA2_OK = False
try:
    from mamba_ssm import Mamba2 as _Mamba2
    MAMBA2_OK = os.environ.get("MAMBA_SKIP_SMOKETEST") == "1" or _smoke_test_mamba2(_Mamba2)
except ImportError:
    pass
print(f"[models.mamba2] using SSM kernel: {'mamba2' if MAMBA2_OK else 'none'}")


class Mamba2Model(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 512, n_layer: int = 8):
        super().__init__()
        if not MAMBA2_OK:
            raise ImportError(
                "mamba_ssm Mamba2 is not importable or failed its GPU test. "
                "Install mamba_ssm 2.3.1 and run on the process's cuda:0 (CUDA_VISIBLE_DEVICES)."
            )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(_Mamba2(d_model=d_model))
            self.norms.append(nn.RMSNorm(d_model))
        self.final_norm = nn.RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        h = self.input_proj(x)  # [B, T, d_model]
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(layer(h) + residual)
        h = self.final_norm(h)
        return self.output_proj(h)  # [B, T, D_out]
