#!/usr/bin/env python3
"""ForceBody v1.2 force benchmark reference scorer (numpy only; spec_v12.json, BENCHMARK.md).

Scores predictions for T-tau (joint torques, N m/kg) or T-GRF (per-foot GRF, body weights) on one split:
per-channel RMSE / MAE / PCC over the scored frames, averaged over the channels of each group; aggregated as pooled
(headline, healthy stratum), study macro, per study, per activity, pathological stratum (separate) and all strata; with
person-level bootstrap 95% intervals for pooled and study macro.

The labels are read through build_bench_cache_v12 (forcebody_io.load_trial). The pose source that produced the
predictions must be declared: --pose_source (default pass0). pass2 / dyn results get leaky_input=true and
official_table_eligible=false.

Usage:
  python score_v12.py --task tau --cache <dir>/bench_test_pass0.npz --pred pred.npz --out result.json
  python score_v12.py --task grf --release <ForceBody_release_v1.3> --split test --pose_source pass0 --pred pred.npz
  python score_v12.py --mode paper --paper_inputs paper.npz    # paper Table 4 formula on arrays pred, gt, mass, valid
pred.npz: one key per trial_uid; T-tau [T,37] N m/kg (dof_names order), T-GRF [T,2,3] BW (skel_world, right foot first).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_bench_cache_v12 as B  # noqa: E402

SPEC = B.SPEC
N_BOOT = int(SPEC['metrics']['bootstrap']['n_boot'])
CI = tuple(SPEC['metrics']['bootstrap']['ci'])
SEED = int(SPEC['metrics']['bootstrap']['seed'])
TAU_GROUPS = {k: v['idx'] for k, v in SPEC['tasks']['T-tau']['dof_groups'].items()}
TAU_CHANNELS = TAU_GROUPS['all']                              # the 20 scored DOF columns
GRF_CHANNEL_NAMES = SPEC['tasks']['T-GRF']['label']['channels'] + ['sum_x', 'sum_y', 'sum_z']
GRF_GROUPS = {'all': [0, 1, 2, 3, 4, 5], 'vertical': [1, 4], 'horizontal': [0, 2, 3, 5], 'sum_feet': [6, 7, 8]}
assert GRF_GROUPS['vertical'] == SPEC['tasks']['T-GRF']['channel_groups']['vertical']['idx']
assert GRF_GROUPS['horizontal'] == SPEC['tasks']['T-GRF']['channel_groups']['horizontal']['idx']


# ---------------------------------------------------------------------------------------------- paper-style reference
def paper_style_metrics(pred_v, gt_v, mass_v, dof_idx):
    """Paper Table 4 formula (eval_table4.metrics_subset): N m errors per DOF over the given frames divided by the
    mean mass of those frames, PCC per DOF (0 when undefined), each averaged over the DOFs. Reference only."""
    p, g = pred_v[:, dof_idx], gt_v[:, dof_idx]
    diff = p - g
    mass_mean = mass_v.mean()
    rmse = (np.sqrt((diff ** 2).mean(axis=0)) / mass_mean).mean()
    mae = (np.abs(diff).mean(axis=0) / mass_mean).mean()
    p_c, g_c = p - p.mean(axis=0), g - g.mean(axis=0)
    num = (p_c * g_c).sum(axis=0)
    den = np.sqrt((p_c ** 2).sum(axis=0) * (g_c ** 2).sum(axis=0))
    pcc = np.where(den > 1e-8, num / np.maximum(den, 1e-12), 0.0).mean()
    return float(rmse), float(mae), float(pcc)


PAPER_GROUPS = {'upper': TAU_GROUPS['upper'], 'lower': TAU_GROUPS['lower'], 'all': TAU_GROUPS['all']}


def paper_style_table(pred, gt, mass, valid):
    """Paper Table 4 groups on flat arrays (valid = GRF-valid frames of the paper cache)."""
    v = np.asarray(valid, bool)
    pv, gv, mv = pred[v], gt[v], mass[v]
    return {grp: dict(zip(('rmse_nmkg', 'mae_nmkg', 'pcc'), paper_style_metrics(pv, gv, mv, idx)))
            for grp, idx in PAPER_GROUPS.items()}


# ---------------------------------------------------------------------------------------------- v1.2 metrics
# sufficient statistics per channel: n, sum e^2, sum |e|, sum p, sum g, sum p^2, sum g^2, sum p g
def _stats(p, g):
    p = p.astype(np.float64)
    g = g.astype(np.float64)
    e = p - g
    n = np.full(p.shape[1], p.shape[0], np.float64)
    return np.stack([n, (e * e).sum(0), np.abs(e).sum(0), p.sum(0), g.sum(0), (p * p).sum(0), (g * g).sum(0),
                     (p * g).sum(0)], axis=-1)


VAR_REL_TOL = float(SPEC['metrics']['pcc_zero_variance']['rel_tol'])


def _metrics(S):
    """S [..., C, 8] -> rmse, mae, pcc each [..., C].
    PCC rule (S19 F7): NaN (left out of the group mean, counted) only when the LABEL variance is 0; a constant prediction
    against a varying label scores 0. A variance counts as 0 when it is at most VAR_REL_TOL times the sum of squares
    (float64 cancellation of a constant series, OURS)."""
    n = S[..., 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        rmse = np.sqrt(S[..., 1] / n)
        mae = S[..., 2] / n
        vp = S[..., 5] - S[..., 3] ** 2 / n
        vg = S[..., 6] - S[..., 4] ** 2 / n
        cov = S[..., 7] - S[..., 3] * S[..., 4] / n
        p_zero = vp <= VAR_REL_TOL * S[..., 5]
        g_zero = vg <= VAR_REL_TOL * S[..., 6]
        den = np.sqrt(np.clip(vp, 0, None) * np.clip(vg, 0, None))
        pcc = np.where(g_zero, np.nan, np.where(p_zero, 0.0, cov / den))
    return rmse, mae, pcc


def _nanmean_last(x):
    with np.errstate(invalid='ignore'):
        cnt = np.isfinite(x).sum(-1)
        return np.where(cnt > 0, np.nansum(x, -1) / np.maximum(cnt, 1), np.nan)


def _group_values(S, groups):
    rmse, mae, pcc = _metrics(S)
    return {g: {'rmse': _nanmean_last(rmse[..., idx]), 'mae': _nanmean_last(mae[..., idx]),
                'pcc': _nanmean_last(pcc[..., idx]), 'n_channels_pcc_defined': np.isfinite(pcc[..., idx]).sum(-1)}
            for g, idx in groups.items()}


def _block(S, groups, channel_names, n_trials, n_persons):
    gv = _group_values(S, groups)
    rmse, mae, pcc = _metrics(S)
    return {'n_trials': int(n_trials), 'n_persons': int(n_persons), 'n_frames': int(S[0, 0]),
            'groups': {g: {k: (int(v) if k.startswith('n_') else float(v)) for k, v in d.items()} for g, d in gv.items()},
            'per_channel': {c: {'rmse': float(rmse[i]), 'mae': float(mae[i]), 'pcc': float(pcc[i])}
                            for i, c in enumerate(channel_names)}}


def _boot_pooled(P, groups, rng, n_boot):
    """P [n_persons, C, 8] -> {group: {metric: [lo, hi]}} by resampling persons."""
    W = rng.multinomial(P.shape[0], np.full(P.shape[0], 1.0 / P.shape[0]), size=n_boot).astype(np.float64)
    gv = _group_values(np.einsum('bp,pck->bck', W, P), groups)
    return gv


def _ci(values):
    lo, hi = np.nanpercentile(values, CI)
    return [float(lo), float(hi)]


def score(cache, preds, task, n_boot=N_BOOT, seed=SEED):
    """Score predictions (dict trial_uid -> array) against a benchmark cache (build_bench_cache_v12.load_cache)."""
    if task not in ('tau', 'grf'):
        raise ValueError("task must be 'tau' or 'grf'")
    meta = cache['meta']
    if task == 'tau':
        label, mask, groups = cache['tau_nmkg'], cache['tau_mask'], TAU_GROUPS
        names = [str(cache['dof_names'][c]) for c in TAU_CHANNELS]
        pos = {c: i for i, c in enumerate(TAU_CHANNELS)}
        groups = {g: [pos[c] for c in idx] for g, idx in groups.items()}
    else:
        label, mask, groups, names = cache['grf_bw'], cache['grf_mask'], GRF_GROUPS, GRF_CHANNEL_NAMES
    stats, rows = [], []
    for i, uid, sl in B.iter_trials(cache):
        m = mask[sl]
        if not m.any():
            continue
        if uid not in preds:
            raise KeyError(f'prediction missing for trial {uid}')
        p = np.asarray(preds[uid])
        T = sl.stop - sl.start
        want = (T, 37) if task == 'tau' else (T, 2, 3)
        if p.shape != want:
            raise ValueError(f'{uid}: prediction shape {p.shape}, expected {want}')
        g = label[sl][m]
        p = p[m]
        if task == 'tau':
            p, g = p[:, TAU_CHANNELS], g[:, TAU_CHANNELS]
        else:
            p = np.concatenate([p.reshape(-1, 6), p.sum(1)], 1)
            g = np.concatenate([g.reshape(-1, 6), g.sum(1)], 1)
        if not np.isfinite(p).all():
            raise ValueError(f'{uid}: non-finite prediction on scored frames')
        stats.append(_stats(p, g))
        rows.append({'uid': uid, 'person': str(cache['person_id'][i]), 'study': str(cache['study'][i]),
                     'activity': str(cache['activity'][i]), 'stratum': str(cache['eval_stratum'][i])})
    if not rows:
        raise ValueError('no scored frames in this cache')
    S = np.stack(stats)                                            # [n_trials, C, 8]
    rng = np.random.default_rng(seed)

    def sel(pred):
        return [k for k, r in enumerate(rows) if pred(r)]

    def block(idx):
        return _block(S[idx].sum(0), groups, names, len(idx), len({rows[k]['person'] for k in idx}))

    def persons(idx):
        pl = sorted({rows[k]['person'] for k in idx})
        return np.stack([S[[k for k in idx if rows[k]['person'] == p]].sum(0) for p in pl]), pl

    res = {}
    healthy = sel(lambda r: r['stratum'] == 'healthy')
    patho = sel(lambda r: r['stratum'] != 'healthy')
    if healthy:
        pooled = block(healthy)
        P, _ = persons(healthy)
        boot = _boot_pooled(P, groups, rng, n_boot)
        for g in groups:
            pooled['groups'][g]['ci95'] = {k: _ci(boot[g][k]) for k in ('rmse', 'mae', 'pcc')}
        res['pooled'] = pooled
        studies = sorted({rows[k]['study'] for k in healthy})
        per_study = {s: block([k for k in healthy if rows[k]['study'] == s]) for s in studies}
        res['per_study'] = per_study
        macro = {'n_studies': len(studies), 'groups': {}}
        boots = {s: _boot_pooled(persons([k for k in healthy if rows[k]['study'] == s])[0], groups, rng, n_boot)
                 for s in studies}
        for g in groups:
            macro['groups'][g] = {}
            for k in ('rmse', 'mae', 'pcc'):
                vals = np.array([per_study[s]['groups'][g][k] for s in studies])
                bm = _nanmean_last(np.stack([boots[s][g][k] for s in studies], -1))
                macro['groups'][g][k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
                macro['groups'][g].setdefault('ci95', {})[k] = _ci(bm)
        res['study_macro'] = macro
        acts = sorted({rows[k]['activity'] for k in healthy})
        res['per_activity'] = {a: block([k for k in healthy if rows[k]['activity'] == a]) for a in acts}
    if patho:
        res['pathological'] = {'pooled': block(patho),
                               'per_study': {s: block([k for k in patho if rows[k]['study'] == s])
                                             for s in sorted({rows[k]['study'] for k in patho})}}
    res['all_strata'] = block(list(range(len(rows))))
    src = meta['pose_source']
    leaky = bool(meta['leaky_input']) or src == 'pass2'
    return {
        'spec_version': SPEC['spec_version'], 'task': 'T-tau' if task == 'tau' else 'T-GRF', 'split': meta['split'],
        'pose_source': src, 'leaky_input': leaky,
        'official_table_eligible': (not leaky) and meta['split'] == 'test' and meta['spec_version'] == SPEC['spec_version'],
        'n_trials': len(rows), 'n_persons': len({r['person'] for r in rows}), 'n_frames': int(S[:, 0, 0].sum()),
        'n_trials_without_scored_frames': int(len(cache['trial_uid']) - len(rows)),
        'units': 'N m/kg' if task == 'tau' else 'BW',
        'channels': names,
        'bootstrap': {'unit': 'person', 'n_boot': n_boot, 'ci': list(CI), 'seed': seed},
        'cache_meta': meta, 'results': res,
    }


def table_row(result, model_name, group=None):
    """One markdown row for BENCHMARK.md (pooled R / M / P with CI, study macro P)."""
    if result['leaky_input'] or not result['official_table_eligible']:
        raise ValueError('leaky or non-test results do not go into the official tables')
    g = group or ('all' if result['task'] == 'T-tau' else 'all')
    p = result['results']['pooled']['groups'][g]
    m = result['results']['study_macro']['groups'][g]
    return (f"| {model_name} | {p['rmse']:.3f} [{p['ci95']['rmse'][0]:.3f}, {p['ci95']['rmse'][1]:.3f}] | "
            f"{p['mae']:.3f} | {p['pcc']:.3f} [{p['ci95']['pcc'][0]:.3f}, {p['ci95']['pcc'][1]:.3f}] | "
            f"{m['rmse']:.3f} | {m['pcc']:.3f} |")


def load_preds(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='v12', choices=['v12', 'paper'])
    ap.add_argument('--task', choices=['tau', 'grf'])
    ap.add_argument('--cache')
    ap.add_argument('--release')
    ap.add_argument('--split', default='test', choices=list(B.SPLITS))
    ap.add_argument('--pose_source', default=None, help='pose source of the predictions (default pass0)')
    ap.add_argument('--pred')
    ap.add_argument('--out')
    ap.add_argument('--n_boot', type=int, default=N_BOOT)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--paper_inputs', help='npz with pred [N,37], gt [N,37] (N m), mass [N], valid [N]')
    a = ap.parse_args()
    if a.mode == 'paper':
        with np.load(a.paper_inputs) as z:
            out = paper_style_table(z['pred'], z['gt'], z['mass'], z['valid'])
    else:
        if not a.task or not a.pred:
            ap.error('--task and --pred are required')
        if a.cache:
            cache = B.load_cache(a.cache)
            if a.pose_source is not None and B.fio.canonical_pose_source(a.pose_source) != cache['meta']['pose_source']:
                raise ValueError(f"--pose_source {a.pose_source} does not match the cache ({cache['meta']['pose_source']})")
        elif a.release:
            arr, meta = B.build_split(a.release, a.split, a.pose_source or 'pass0', (), a.workers)
            cache = dict(arr, meta=meta)
        else:
            ap.error('--cache or --release is required')
        out = score(cache, load_preds(a.pred), a.task, a.n_boot)
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
        print(f'saved {a.out}')
    else:
        print(text)


if __name__ == '__main__':
    main()
