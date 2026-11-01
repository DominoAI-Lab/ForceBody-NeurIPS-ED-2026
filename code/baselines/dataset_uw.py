"""
Sequence dataset wrapper that ALSO returns per-frame, per-joint sigma_tau.

This extends the baseline SequenceDataset by including the
uncertainty tensor alongside (x, y, mask). The collate fn pads
sigma the same way as y.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from data.dataset import _build_input_features


class SequenceDatasetUW(Dataset):
    """Trial-level dataset returning (x, y, sigma, mask)."""

    def __init__(self, cache_path: str | Path, condition: str = "C",
                 norm_stats: dict | None = None,
                 seq_len: int = C.SEQ_LEN, training: bool = True):
        cache = torch.load(cache_path, weights_only=False)
        self.x_all     = _build_input_features(cache, condition)
        self.y_all     = cache["tau_gt"]
        self.sigma_all = cache["sigma_tau"]   # [N, 37], same shape as y
        self.mask_all  = cache["grf_valid"]
        self.offsets   = cache["trial_offsets"]
        self.seq_len   = seq_len
        self.training  = training
        self.n_trials  = len(self.offsets) - 1

        # Normalize x and y just like the baseline. sigma is in label units (N m/kg); when y is normalized by y_std,
        # the equivalent sigma in normalized space is sigma_norm = sigma / y_std.
        # We compute and store it once here for direct use in the loss.
        if norm_stats is not None:
            y_std = norm_stats["y_std"] + 1e-8
            self.x_all     = (self.x_all - norm_stats["x_mean"]) / (norm_stats["x_std"] + 1e-8)
            self.y_all     = (self.y_all - norm_stats["y_mean"]) / y_std
            self.sigma_all = self.sigma_all / y_std
        # Replace NaN / Inf in sigma by 0 so they cannot reach the weights.
        self.sigma_all = torch.nan_to_num(self.sigma_all, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        s, e = int(self.offsets[idx]), int(self.offsets[idx + 1])
        T = e - s
        if self.training and T > self.seq_len:
            start = np.random.randint(0, T - self.seq_len)
            sl = slice(s + start, s + start + self.seq_len)
        else:
            sl = slice(s, e)
        return self.x_all[sl], self.y_all[sl], self.sigma_all[sl], self.mask_all[sl]


def collate_seq_uw(batch):
    xs, ys, ss, ms = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    B = len(xs)
    D_in  = xs[0].shape[1]
    D_out = ys[0].shape[1]

    x_pad = torch.zeros(B, max_len, D_in)
    y_pad = torch.zeros(B, max_len, D_out)
    s_pad = torch.zeros(B, max_len, D_out)
    m_pad = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (x, y, s, m) in enumerate(zip(xs, ys, ss, ms)):
        T = x.shape[0]
        x_pad[i, :T] = x
        y_pad[i, :T] = y
        s_pad[i, :T] = s
        m_pad[i, :T] = m
    return x_pad, y_pad, s_pad, m_pad


def _worker_init_fn(worker_id: int) -> None:
    """Make each DataLoader worker deterministic given the parent seed.

    PyTorch sets `torch.initial_seed()` per worker (= base_seed + worker_id),
    but numpy and python's `random` are NOT seeded automatically. We seed
    them here so `np.random.randint` calls inside __getitem__ (the random
    crop) are reproducible across runs.
    """
    import numpy as _np
    import random as _random
    base = torch.initial_seed() % (2 ** 32)
    _np.random.seed(base)
    _random.seed(base)


def get_seq_loader_uw(cache_path, condition="C", norm_stats=None,
                       seq_len=C.SEQ_LEN, batch_size=32, shuffle=True,
                       training=True, num_workers=16, seed=0):
    ds = SequenceDatasetUW(cache_path, condition, norm_stats, seq_len, training)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_seq_uw, drop_last=training,
                      worker_init_fn=_worker_init_fn,
                      generator=g)
