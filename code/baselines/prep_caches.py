#!/usr/bin/env python3
"""Turn the official benchmark caches (benchmark/build_bench_cache_v12.py, pose source pass0, extra input 'joints')
into the training caches of the baseline code.

Inputs come only from the official cache (benchmark allowlist). Side labels that the official cache does not carry are
read per trial with forcebody_io.load_trial(pass0): sigma_tau (loss weight of the uncertainty-weighted fine-tune) and
contact_torque_aligned (GRM label of the GRF baselines, unofficial). They are labels / loss weights, never model inputs.

Reads  <bench_dir>/bench_{train,val,test}_pass0.npz. If a file is missing it is built first from --release
       (same as: python benchmark/build_bench_cache_v12.py --release <release> --out <bench_dir> --extra_inputs joints).
Writes (<out_dir>):
  tau/{train,val,test}_cache.pt, tau/norm_stats_{A,B,C}.pt  torque caches (trials with >= 1 scored torque frame)
  grf/{train,val,test}_grf.npz, grf/norm_stats_grf.npz    GRF caches (trials with >= 1 scored GRF frame)
  prep_meta.json                                           counts per split

Usage: python prep_caches.py --release <ForceBody release dir> --out_dir <cache dir> [--bench_dir <dir>] [--workers 16]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
import build_bench_cache_v12 as B  # noqa: E402

G = B.G


def _side(task):
    path, uid = task
    d = B.fio.load_trial(path, pose_source='pass0')
    assert str(d['trial_uid']) == uid
    m = float(d['mass_kg_dynamics'])
    return (d['sigma_tau'].astype(np.float64) / m).astype(np.float32), \
           (d['contact_torque_aligned'][:, :2].astype(np.float64) / (m * G)).astype(np.float32)


def finite_diff(x):
    """Paper _finite_diff (dt=1, central inside, one-sided at the ends), applied within one trial."""
    dx = np.zeros_like(x)
    if len(x) < 2:
        return dx
    dx[1:-1] = (x[2:] - x[:-2]) / 2.0
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return dx


def central_diff_acc(x, dt):
    """Second derivative of the GRF baseline inputs (central difference with dt), within one trial."""
    a = np.zeros_like(x)
    if len(x) < 3:
        return a
    a[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt * dt)
    a[0] = a[1]
    a[-1] = a[-2]
    return a


def select(c, keep):
    """Frame index and new offsets of the kept trials."""
    off = c['trial_offsets']
    idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(keep)[0]])
    lens = np.diff(off)[keep]
    return idx, np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', required=True, help='ForceBody release root (holds manifest.csv)')
    ap.add_argument('--out_dir', required=True, help='output cache dir (tau/, grf/, prep_meta.json)')
    ap.add_argument('--bench_dir', default=None, help='dir of bench_*_pass0.npz (default: --out_dir)')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    release = Path(args.release)
    out_dir = Path(args.out_dir)
    bench_dir = Path(args.bench_dir) if args.bench_dir else out_dir
    out_dir.joinpath('tau').mkdir(parents=True, exist_ok=True)
    out_dir.joinpath('grf').mkdir(parents=True, exist_ok=True)
    man = {r['trial_uid']: r for r in B.read_manifest(release)}
    meta = {}
    for split in ('train', 'val', 'test'):
        bench_path = bench_dir / f'bench_{split}_pass0.npz'
        if not bench_path.exists():
            arr, bmeta = B.build_split(release, split, 'pass0', ('joints',), args.workers)
            B.save_cache(arr, bmeta, bench_dir)
        c = B.load_cache(bench_path)
        if 'joints' not in c:
            raise ValueError(f'{bench_path} has no extra input "joints"; rebuild it with --extra_inputs joints')
        uids = [str(u) for u in c['trial_uid']]
        off = c['trial_offsets']
        tasks = [(str(release / man[u]['npz_path']), u) for u in uids]
        with Pool(max(1, min(args.workers, len(tasks)))) as pool:
            side = pool.map(_side, tasks, chunksize=4)
        sigma = np.concatenate([s for s, _ in side])
        grm = np.concatenate([g for _, g in side])
        n = len(uids)
        pose_vel = np.zeros_like(c['poses'])
        pose_acc = np.zeros_like(c['poses'])
        qdd = np.zeros_like(c['poses'])
        for i in range(n):
            sl = slice(off[i], off[i + 1])
            pose_vel[sl] = finite_diff(c['poses'][sl])
            pose_acc[sl] = finite_diff(pose_vel[sl])
            qdd[sl] = central_diff_acc(c['poses'][sl].astype(np.float64), 1.0 / c['fps_true_grid'][i]).astype(np.float32)

        # ---------------- torque cache (paper format) ----------------
        tmask = c['tau_mask']
        keep = np.array([tmask[off[i]:off[i + 1]].any() for i in range(n)])
        idx, noff = select(c, keep)
        m = tmask[idx]
        tau = c['tau_nmkg'][idx].copy()
        tau[~m] = 0.0                                  # unscored frames carry no loss (mask) and must not be NaN
        sig = sigma[idx].copy()
        sig[~m] = 0.0
        tc = {
            'poses': torch.from_numpy(c['poses'][idx]), 'betas': torch.from_numpy(c['betas'][idx]),
            'mass_kg': torch.from_numpy(c['mass_kg'][idx]), 'height_m': torch.from_numpy(c['height_m'][idx]),
            'trans_vel': torch.from_numpy(c['trans_vel'][idx]),
            'pose_vel': torch.from_numpy(pose_vel[idx]), 'pose_acc': torch.from_numpy(pose_acc[idx]),
            'tau_gt': torch.from_numpy(np.nan_to_num(tau)), 'grf_valid': torch.from_numpy(m),
            'sigma_tau': torch.from_numpy(np.nan_to_num(sig)),
            'trial_offsets': torch.from_numpy(noff), 'trial_uid': [u for u, k in zip(uids, keep) if k],
        }
        torch.save(tc, out_dir / 'tau' / f'{split}_cache.pt')
        meta[f'tau_{split}'] = {'n_trials': int(keep.sum()), 'n_frames': int(len(idx)), 'n_scored_frames': int(m.sum()),
                                'n_trials_official_cache': n}
        if split == 'train':
            x = torch.cat([tc['poses'], tc['betas'], tc['mass_kg'][:, None], tc['height_m'][:, None],
                           tc['trans_vel'], tc['pose_vel'], tc['pose_acc']], 1)[tc['grf_valid']]
            y = tc['tau_gt'][tc['grf_valid']]
            torch.save({'x_mean': x.mean(0), 'x_std': x.std(0), 'y_mean': y.mean(0), 'y_std': y.std(0)},
                       out_dir / 'tau' / 'norm_stats_C.pt')
            # input ablation: A = poses (first 46 columns), B = A + betas + mass + height (first 58 columns)
            for cond, d in (('A', 46), ('B', 58)):
                xa = x[:, :d].contiguous()
                torch.save({'x_mean': xa.mean(0), 'x_std': xa.std(0), 'y_mean': y.mean(0), 'y_std': y.std(0)},
                           out_dir / 'tau' / f'norm_stats_{cond}.pt')
            meta['tau_y_std_min'] = float(y.std(0).min())

        # ---------------- GRF cache (original GRF code inputs / labels) ----------------
        gmask = c['grf_mask']
        keep = np.array([gmask[off[i]:off[i + 1]].any() for i in range(n)])
        idx, noff = select(c, keep)
        j = c['joints'][idx]
        grf_bw = np.nan_to_num(c['grf_bw'][idx]).reshape(-1, 6)
        grm_bw = np.nan_to_num(grm[idx]).reshape(-1, 6)
        com_acc = (grf_bw.reshape(-1, 2, 3).sum(1) * G + np.array([0.0, -G, 0.0])).astype(np.float32)  # sum F / m + g
        mass_dyn = np.repeat(c['mass_kg_dynamics'][keep], np.diff(noff)).astype(np.float32)
        gc = dict(q=c['poses'][idx], qdd=qdd[idx], jc_pelvis=(j - j[:, :1]).astype(np.float32),
                  grf_bw=grf_bw.astype(np.float32), grm_bw=grm_bw.astype(np.float32), com_acc=com_acc,
                  valid=gmask[idx], mass_dyn=mass_dyn, trial_offsets=noff,
                  trial_uid=np.array([u for u, k in zip(uids, keep) if k]))
        np.savez(out_dir / 'grf' / f'{split}_grf.npz', **gc)
        meta[f'grf_{split}'] = {'n_trials': int(keep.sum()), 'n_frames': int(len(idx)),
                                'n_scored_frames': int(gc['valid'].sum()), 'n_trials_official_cache': n}
        if split == 'train':   # GRF norm stats: all train frames, population std clamped at 1e-3
            ns = {}
            for k in ('q', 'qdd', 'jc_pelvis'):
                a = gc[k].astype(np.float64)
                ns[f'mean_{k}'] = a.mean(0).astype(np.float32)
                ns[f'std_{k}'] = np.maximum(a.std(0), 1e-3).astype(np.float32)
            np.savez(out_dir / 'grf' / 'norm_stats_grf.npz', **ns)
        print(split, meta[f'tau_{split}'], meta[f'grf_{split}'], flush=True)
    (out_dir / 'prep_meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
