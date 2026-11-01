#!/usr/bin/env python3
"""Predict torques (N m/kg) for every trial of a torque cache with one checkpoint (whole trial in one forward pass,
de-normalized with norm_stats_<condition>). Writes a benchmark/score_v12.py prediction npz
(one key per trial_uid, [T,37] in dof_names order).

Usage: python predict_tau.py --model transformer --ckpt <best.pt> --cache_dir <tau cache> --split test --out pred.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from data.dataset import _build_input_features  # noqa: E402
from models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--condition', default='C', choices=['A', 'B', 'C'])  # A, B = input ablation
    a = ap.parse_args()
    if not torch.cuda.is_available():
        a.device = 'cpu'
    cache = torch.load(Path(a.cache_dir) / f'{a.split}_cache.pt', weights_only=False)
    norm = torch.load(Path(a.cache_dir) / f'norm_stats_{a.condition}.pt', weights_only=False)
    x = (_build_input_features(cache, a.condition) - norm['x_mean']) / (norm['x_std'] + 1e-8)
    model = build_model(a.model, C.INPUT_DIMS[a.condition], C.OUTPUT_DIM).to(a.device)
    state = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    off = cache['trial_offsets']
    out = {}
    with torch.no_grad():
        for i, uid in enumerate(cache['trial_uid']):
            s, e = int(off[i]), int(off[i + 1])
            y = model(x[s:e].unsqueeze(0).to(a.device)).squeeze(0).cpu()
            out[uid] = (y * norm['y_std'] + norm['y_mean']).numpy().astype(np.float32)
    np.savez(a.out, **out)
    print(f'wrote {a.out} trials={len(out)} ckpt_epoch={state.get("epoch")} ckpt_val={state.get("val_loss")}')


if __name__ == '__main__':
    main()
