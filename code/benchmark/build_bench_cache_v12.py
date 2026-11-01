#!/usr/bin/env python3
"""ForceBody v1.2 force benchmark: loader and cache builder (numpy only; see BENCHMARK.md and spec_v12.json).

Every trial is read with forcebody_io.load_trial(npz, pose_source), so a model sees exactly one pose set. The default
pose source is 'pass0' (kinematics pass, never saw the force plates). 'pass2' (dynamics pass) is allowed, but the cache
is then marked leaky_input=True and score_v12.py keeps its results out of the official tables. Asking for any field on
the leakage list (forcebody_io.LEAKY_INPUT_FIELDS, e.g. q_gt, grf_aligned, tau_selfconsistent) as a model input raises
ValueError.

Cache layout (one .npz per split and pose source; frames of all trials concatenated in manifest order):
  inputs   poses [N,46], trans_vel [N,3] (m/s), betas [N,10], mass_kg [N], height_m [N]   (T-tau and T-GRF inputs)
           plus the optional extra inputs asked for with --extra_inputs (q, qd, qdd, joints, addb_joints)
  labels   tau_nmkg [N,37] = tau_selfconsistent / mass_kg_dynamics                  (N m/kg; N/kg on pelvis translations)
           grf_bw   [N,2,3] = grf_aligned[:, :2] / (mass_kg_dynamics * G)          (body weights, right foot first)
  masks    tau_mask [N] = eval_tau_frame_ok & eval_torque_trial_ok
           grf_mask [N] = eval_grf_frame_ok & eval_grf_trial_ok
  trials   trial_offsets [n+1], trial_uid, person_id, study, activity, eval_stratum, split, mass_kg_dynamics,
           fps_true_grid, pose_usable, q_source
  meta     meta_json: spec version, pose_source, leaky_input, input fields, release path, counts

Usage:
  python build_bench_cache_v12.py --release <ForceBody_release_v1.3> --out <dir> [--splits train val test]
                                  [--pose_source pass0] [--extra_inputs q] [--workers 16]
  import build_bench_cache_v12 as B; c = B.load_cache('<dir>/bench_test_pass0.npz'); x = B.input_matrix(c)
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the code/ folder holds forcebody_io.py
import forcebody_io as fio  # noqa: E402

SPEC_PATH = HERE / 'spec_v12.json'
SPEC = json.loads(SPEC_PATH.read_text())
G = float(SPEC['constants']['G_mps2']['value'])
BASE_INPUTS = tuple(SPEC['tasks']['T-tau']['inputs']['required'])      # poses, trans_vel, betas, mass_kg, height_m
EXTRA_INPUTS = tuple(SPEC['tasks']['T-tau']['inputs']['optional'])     # q, qd, qdd, joints, addb_joints
SPLITS = ('train', 'val', 'test')


def check_inputs(names, pose_source):
    """Raise ValueError for a leaky or unknown model input; return the canonical pose source."""
    src = fio.canonical_pose_source(pose_source)
    for n in names:
        if fio.is_leaky_field(n):
            raise ValueError(f"input '{n}' is on the leakage list (forcebody_io.LEAKY_INPUT_FIELDS); a force benchmark "
                             f"model must not take it as input (BENCHMARK.md, Leakage)")
        if n not in BASE_INPUTS and n not in EXTRA_INPUTS:
            raise ValueError(f"input '{n}' is not a benchmark input; allowed: {BASE_INPUTS + EXTRA_INPUTS} "
                             f"(pose-set fields come from pose_source via load_trial common keys)")
    return src


def bench_split(split_addb, split_addb_val):
    """'train' | 'val' | 'test' from split_addb and split_addb_val (D26); raises on an inconsistent pair."""
    if split_addb == 'test' and split_addb_val == 'test':
        return 'test'
    if split_addb == 'train' and split_addb_val in ('train', 'val'):
        return split_addb_val
    raise ValueError(f'inconsistent split pair split_addb={split_addb!r} split_addb_val={split_addb_val!r}')


def load_bench_trial(npz_path, pose_source='pass0', extra_inputs=()):
    """One trial as benchmark arrays (inputs, labels, masks) through forcebody_io.load_trial."""
    src = check_inputs(tuple(extra_inputs), pose_source)
    d = fio.load_trial(npz_path, pose_source=src)
    T = int(d['poses'].shape[0])
    fps = float(d['fps_true_grid'])
    m_dyn = float(d['mass_kg_dynamics'])
    cb = [str(x) for x in d['contact_bodies'][:2]]
    if cb != ['calcn_r', 'calcn_l']:
        raise ValueError(f'{npz_path}: contact_bodies[:2] = {cb}, expected calcn_r, calcn_l')
    out = {
        'poses': d['poses'].astype(np.float32),
        'trans_vel': np.gradient(d['trans'].astype(np.float64), 1.0 / fps, axis=0).astype(np.float32),
        'betas': np.broadcast_to(d['betas'].astype(np.float32), (T, 10)).copy(),
        'mass_kg': np.full(T, float(d['mass_kg']), np.float32),
        'height_m': np.full(T, float(d['height_m']), np.float32),
        'tau_nmkg': (d['tau_selfconsistent'].astype(np.float64) / m_dyn).astype(np.float32),
        'grf_bw': (d['grf_aligned'][:, :2].astype(np.float64) / (m_dyn * G)).astype(np.float32),
        'tau_mask': np.asarray(d['eval_tau_frame_ok'], bool) & bool(d['eval_torque_trial_ok']),
        'grf_mask': np.asarray(d['eval_grf_frame_ok'], bool) & bool(d['eval_grf_trial_ok']),
    }
    for n in extra_inputs:
        out[n] = d[n].astype(np.float32)
    for k in ('tau_nmkg',):
        if not np.isfinite(out[k][out['tau_mask']]).all():
            raise ValueError(f'{npz_path}: non-finite {k} on scored frames')
    if not np.isfinite(out['grf_bw'][out['grf_mask']]).all():
        raise ValueError(f'{npz_path}: non-finite grf_bw on scored frames')
    for k in BASE_INPUTS + tuple(extra_inputs):
        if not np.isfinite(out[k]).all():
            raise ValueError(f'{npz_path}: non-finite input {k}')
    trial = {'trial_uid': str(d['trial_uid']), 'person_id': str(d['person_id']), 'study': str(d['study']),
             'activity': str(d['activity_recovered']), 'eval_stratum': str(d['eval_stratum']),
             'mass_kg_dynamics': m_dyn, 'fps_true_grid': fps, 'pose_usable': bool(d['pose_usable']),
             'q_source': str(d['q_source']), 'T': T, 'dof_names': tuple(str(x) for x in d['dof_names'])}
    return out, trial


def _worker(task):
    path, src, extra = task
    return load_bench_trial(path, src, extra)


def read_manifest(release):
    import csv
    with open(Path(release) / 'manifest.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['bench_split'] = bench_split(r['split_addb'], r['split_addb_val'])
    return rows


def build_split(release, split, pose_source='pass0', extra_inputs=(), workers=16):
    """Build one split in memory: returns (arrays dict, meta dict)."""
    src = check_inputs(tuple(extra_inputs), pose_source)
    rows = [r for r in read_manifest(release) if r['bench_split'] == split]
    if not rows:
        raise ValueError(f'no trials in split {split!r}')
    tasks = [(str(Path(release) / r['npz_path']), src, tuple(extra_inputs)) for r in rows]
    with Pool(max(1, min(workers, 16, len(tasks)))) as pool:
        res = list(pool.imap(_worker, tasks, chunksize=4))
    for r, (_, t) in zip(rows, res):
        if r['trial_uid'] != t['trial_uid']:
            raise AssertionError(f"manifest/npz trial_uid mismatch {r['trial_uid']} vs {t['trial_uid']}")
    if len({t['dof_names'] for _, t in res}) != 1:
        raise AssertionError('dof_names differ between trials')
    keys = list(res[0][0].keys())
    arr = {k: np.concatenate([a[k] for a, _ in res], axis=0) for k in keys}
    arr['trial_offsets'] = np.concatenate([[0], np.cumsum([t['T'] for _, t in res])]).astype(np.int64)
    for k in ('trial_uid', 'person_id', 'study', 'activity', 'eval_stratum', 'q_source'):
        arr[k] = np.array([t[k] for _, t in res])
    arr['split'] = np.array([split] * len(res))
    arr['dof_names'] = np.array(res[0][1]['dof_names'])
    arr['mass_kg_dynamics'] = np.array([t['mass_kg_dynamics'] for _, t in res], np.float64)
    arr['fps_true_grid'] = np.array([t['fps_true_grid'] for _, t in res], np.float64)
    arr['pose_usable'] = np.array([t['pose_usable'] for _, t in res], bool)
    man = Path(release) / 'manifest.csv'
    meta = {
        'spec_version': SPEC['spec_version'], 'split': split, 'pose_source': src,
        'leaky_input': src == 'pass2', 'official_table_eligible': src == 'pass0',
        'inputs': list(BASE_INPUTS) + list(extra_inputs), 'release': str(Path(release).resolve()),
        'manifest_md5': hashlib.md5(man.read_bytes()).hexdigest(), 'G_mps2': G,
        'n_trials': len(res), 'n_frames': int(arr['trial_offsets'][-1]),
        'n_tau_frames': int(arr['tau_mask'].sum()), 'n_grf_frames': int(arr['grf_mask'].sum()),
        'n_tau_trials': int(sum(a['tau_mask'].any() for a, _ in res)),
        'n_grf_trials': int(sum(a['grf_mask'].any() for a, _ in res)),
        'n_persons': len(set(arr['person_id'].tolist())),
        'n_pose_unusable_trials': int((~arr['pose_usable']).sum()),
    }
    if src == 'pass2':
        meta['leaky_note'] = ('pose_source pass2 = AddB dynamics pass, fitted with the force plates; results are '
                              'leaky_input=True and are excluded from the official BENCHMARK.md tables (D0prime)')
    return arr, meta


def save_cache(arr, meta, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench_{meta['split']}_{meta['pose_source']}.npz"
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, meta_json=np.array(json.dumps(meta)), **arr)
    os.replace(tmp, path)
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    return path


def load_cache(path):
    """Load a cache written by save_cache: dict of arrays plus 'meta' (dict)."""
    with np.load(path, allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    c['meta'] = json.loads(str(c.pop('meta_json')))
    return c


def input_matrix(cache, names=None):
    """Per-frame model input [N, D] from the cache (default: the task inputs in spec order)."""
    names = tuple(names or cache['meta']['inputs'])
    check_inputs(names, cache['meta']['pose_source'])
    return np.concatenate([cache[n].reshape(cache[n].shape[0], -1) for n in names], axis=1)


def iter_trials(cache):
    """Yield (trial index, trial_uid, frame slice) in cache order."""
    off = cache['trial_offsets']
    for i, uid in enumerate(cache['trial_uid']):
        yield i, str(uid), slice(int(off[i]), int(off[i + 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    ap.add_argument('--pose_source', default='pass0', help="pass0 (default, official) | kin | pass2 | dyn (leaky)")
    ap.add_argument('--extra_inputs', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    for s in a.splits:
        arr, meta = build_split(a.release, s, a.pose_source, a.extra_inputs, a.workers)
        p = save_cache(arr, meta, a.out)
        print(f"{p}  trials={meta['n_trials']} frames={meta['n_frames']} tau_frames={meta['n_tau_frames']} "
              f"grf_frames={meta['n_grf_frames']} leaky_input={meta['leaky_input']}")


if __name__ == '__main__':
    main()
