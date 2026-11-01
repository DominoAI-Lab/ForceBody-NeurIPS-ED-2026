"""PyTorch Datasets and DataLoader factories."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def _build_input_features(cache: dict, condition: str) -> torch.Tensor:
    """Assemble the input tensor from the cache for input condition A / B / C (input ablation).

      A: poses                                                  46   (marker-style)
      B: A + betas + mass + height                              58   (parametric body)
      C: B + trans_vel + pose_vel + pose_acc                    153  (motion derivatives, main setting)
    """
    if condition not in ("A", "B", "C"):
        raise ValueError(f"unknown input condition {condition!r}; use A, B or C")
    poses = cache["poses"]  # [N, 46]
    parts = [poses]

    # B / C: + parametric body (betas, mass, height)
    if condition in ("B", "C"):
        parts.extend([
            cache["betas"],                  # [N, 10]
            cache["mass_kg"].unsqueeze(1),   # [N, 1]
            cache["height_m"].unsqueeze(1),  # [N, 1]
        ])

    # C: + motion derivatives (trans_vel, pose_vel, pose_acc)
    if condition == "C":
        # trans_vel is the benchmark input (m/s, numpy.gradient within the trial). pose_vel / pose_acc are
        # finite differences with dt = 1 frame (not divided by the frame rate), taken within each trial
        # (prep_caches.py), so no trial borrows a frame of its neighbour.
        parts.extend([cache["trans_vel"], cache["pose_vel"], cache["pose_acc"]])

    return torch.cat(parts, dim=1)


class FrameDataset(Dataset):
    """Per-frame dataset for MLP. Indexes into flat concatenated tensors."""

    def __init__(self, cache_path: str | Path, condition: str = "C",
                 norm_stats: dict | None = None):
        cache = torch.load(cache_path, weights_only=False)
        self.x = _build_input_features(cache, condition)
        self.y = cache["tau_gt"]
        self.mask = cache["grf_valid"]
        self.mass = cache["mass_kg"]
        self.height = cache["height_m"]

        if norm_stats is not None:
            self.x = (self.x - norm_stats["x_mean"]) / (norm_stats["x_std"] + 1e-8)
            self.y = (self.y - norm_stats["y_mean"]) / (norm_stats["y_std"] + 1e-8)

        valid_idx = torch.where(self.mask)[0]
        self.valid_idx = valid_idx

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        return self.x[i], self.y[i]


class SequenceDataset(Dataset):
    """Trial-level dataset for sequence models. Random crop at train, full trial at eval."""

    def __init__(self, cache_path: str | Path, condition: str = "C",
                 norm_stats: dict | None = None,
                 seq_len: int = C.SEQ_LEN, training: bool = True):
        cache = torch.load(cache_path, weights_only=False)
        self.x_all = _build_input_features(cache, condition)
        self.y_all = cache["tau_gt"]
        self.mask_all = cache["grf_valid"]
        self.offsets = cache["trial_offsets"]
        self.seq_len = seq_len
        self.training = training
        self.n_trials = len(self.offsets) - 1

        if norm_stats is not None:
            self.x_all = (self.x_all - norm_stats["x_mean"]) / (norm_stats["x_std"] + 1e-8)
            self.y_all = (self.y_all - norm_stats["y_mean"]) / (norm_stats["y_std"] + 1e-8)

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

        return self.x_all[sl], self.y_all[sl], self.mask_all[sl]


def collate_seq(batch):
    """Pad variable-length sequences to max length in batch."""
    xs, ys, ms = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    B = len(xs)
    D_in = xs[0].shape[1]
    D_out = ys[0].shape[1]

    x_pad = torch.zeros(B, max_len, D_in)
    y_pad = torch.zeros(B, max_len, D_out)
    m_pad = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (x, y, m) in enumerate(zip(xs, ys, ms)):
        T = x.shape[0]
        x_pad[i, :T] = x
        y_pad[i, :T] = y
        m_pad[i, :T] = m

    return x_pad, y_pad, m_pad


def get_frame_loader(cache_path, condition="C", norm_stats=None,
                     batch_size=4096, shuffle=True, num_workers=16):
    ds = FrameDataset(cache_path, condition, norm_stats)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def get_seq_loader(cache_path, condition="C", norm_stats=None,
                   seq_len=C.SEQ_LEN, batch_size=32, shuffle=True,
                   training=True, num_workers=16):
    ds = SequenceDataset(cache_path, condition, norm_stats, seq_len, training)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_seq, drop_last=training)
