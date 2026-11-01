#!/usr/bin/env python3
"""Score baseline predictions of one or more run folders (for example one per seed) and summarize them.

Torque: benchmark/score_v12.score on the official test cache (bench_test_pass0.npz). Reported per run:
  pooled healthy stratum (official headline) and all strata pooled, groups upper / lower / all (20 direct DOFs).
GRF: score_v12 (body weights) and the paper GRF formula: mean over scored frames of the sum over the two feet of
  |pred - label| (3D norm), in N/kg (= BW * 9.81). GRM (unofficial) uses the same formula on contact_torque_aligned.

A run folder holds any of:
  tau/pred_test_<model>.npz          (train.py + predict_tau.py)
  uw/pred_test_<model>_uw.npz        (train_uw.py + predict_tau.py)
  grf/<cell>/pred_test_grf.npz       (grf/train_grf.py, also writes pred_test_full15.npz)

Writes <out_dir>/scores/*.json, <out_dir>/summary.{json,csv} (per run values, mean +- std (ddof=1) over the runs,
UW gain = transformer_uw - transformer per run) and <out_dir>/paper_vs_release.csv (mean over runs next to the
numbers printed in the submitted paper, which were computed on release 1.0 and on all test strata).

Usage: python aggregate.py --cache_dir <cache> --runs seed1=<runs/seed1> seed2=<runs/seed2> --out_dir <results>
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
import build_bench_cache_v12 as B  # noqa: E402
import score_v12 as S  # noqa: E402

G = 9.81

# Numbers printed in the submitted paper (release 1.0): torque table (RMSE, MAE, PCC) per group, GRF table (GRF N/kg,
# GRM N m/kg). Keys are the prediction names used here.
PAPER_TAU = {
    'mlp': {'upper': (0.008, 0.004, 0.579), 'lower': (0.470, 0.245, 0.653), 'all': (0.285, 0.149, 0.623)},
    'fconv': {'upper': (0.008, 0.004, 0.575), 'lower': (0.471, 0.246, 0.644), 'all': (0.286, 0.149, 0.616)},
    'lstm': {'upper': (0.008, 0.004, 0.545), 'lower': (0.479, 0.258, 0.637), 'all': (0.291, 0.156, 0.600)},
    'gru': {'upper': (0.008, 0.004, 0.561), 'lower': (0.477, 0.251, 0.645), 'all': (0.289, 0.153, 0.611)},
    'mamba2': {'upper': (0.007, 0.004, 0.612), 'lower': (0.469, 0.236, 0.660), 'all': (0.284, 0.143, 0.641)},
    'transformer': {'upper': (0.009, 0.006, 0.461), 'lower': (0.538, 0.307, 0.572), 'all': (0.326, 0.186, 0.528)},
    'transformer_uw': {'upper': (0.008, 0.005, 0.536), 'lower': (0.500, 0.270, 0.605), 'all': (0.303, 0.164, 0.578)},
}
PAPER_GRF = {'G_mlp_deep': (1.522, 0.315), 'G_gln': (1.715, 0.258)}


def find_preds(run_dir):
    """(prediction name, task, path) of every prediction file in one run folder."""
    run_dir = Path(run_dir)
    out = [(p.stem[len('pred_test_'):], 'tau', p) for p in sorted((run_dir / 'tau').glob('pred_test_*.npz'))]
    out += [(p.stem[len('pred_test_'):], 'tau', p) for p in sorted((run_dir / 'uw').glob('pred_test_*_uw.npz'))]
    out += [(p.parent.name, 'grf', p) for p in sorted((run_dir / 'grf').glob('*/pred_test_grf.npz'))]
    return out


def grf_paper_metric(grf_cache, run_dir):
    z = np.load(grf_cache)
    pred = np.load(Path(run_dir) / 'pred_test_full15.npz')
    off, v = z['trial_offsets'], z['valid']
    p = np.concatenate([pred[str(u)] for u in z['trial_uid']])
    assert len(p) == off[-1]
    pv = p[v]
    grf = np.linalg.norm(((pv[:, :6] - z['grf_bw'][v]) * G).reshape(-1, 2, 3), axis=-1).sum(-1).mean()
    grm = np.linalg.norm(((pv[:, 6:12] - z['grm_bw'][v]) * G).reshape(-1, 2, 3), axis=-1).sum(-1).mean()
    return {'grf_n_per_kg': float(grf), 'grm_nm_per_kg_unofficial': float(grm), 'n_frames': int(v.sum())}


def _score(args):
    label, name, task, pred_path, cache_dir, score_dir = args
    cache = B.load_cache(Path(cache_dir) / 'bench_test_pass0.npz')
    r = S.score(cache, S.load_preds(pred_path), task)
    (Path(score_dir) / f'score_v12_{task}_{name}_{label}.json').write_text(json.dumps(r, indent=1))
    extra = grf_paper_metric(Path(cache_dir) / 'grf' / 'test_grf.npz', pred_path.parent) if task == 'grf' else None
    return label, name, task, r, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache_dir', required=True, help='prep_caches.py --out_dir (bench_test_pass0.npz, grf/)')
    ap.add_argument('--runs', nargs='+', required=True, help='label=run_dir, e.g. seed1=runs/seed1')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    score_dir = out_dir / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    labels = [i.split('=', 1)[0] for i in a.runs]
    tasks = []
    for item in a.runs:
        label, run_dir = item.split('=', 1)
        tasks += [(label, n, t, p, a.cache_dir, str(score_dir)) for n, t, p in find_preds(run_dir)]
    if not tasks:
        sys.exit('no prediction files found')
    with Pool(max(1, min(a.workers, len(tasks)))) as pool:
        scored = pool.map(_score, tasks)

    per = {}  # (task, name, metric) -> {label: value}
    for label, name, task, r, extra in scored:
        if task == 'tau':
            for stratum, key in (('all_strata', 'allstrata'), ('pooled', 'official')):
                g = r['results'][stratum]['groups']
                for grp in ('upper', 'lower', 'all'):
                    for m, mk in (('R', 'rmse'), ('M', 'mae'), ('P', 'pcc')):
                        per.setdefault(('tau', name, f'{key}_{grp}_{m}'), {})[label] = g[grp][mk]
        else:
            g = r['results']['pooled']['groups']['all']
            per.setdefault(('grf', name, 'GRF_Nkg'), {})[label] = extra['grf_n_per_kg']
            per.setdefault(('grf', name, 'GRM_Nmkg_unofficial'), {})[label] = extra['grm_nm_per_kg_unofficial']
            per.setdefault(('grf', name, 'official_all_R_BW'), {})[label] = g['rmse']
            per.setdefault(('grf', name, 'official_all_P'), {})[label] = g['pcc']
    for k in [k for k in per if k[1] == 'transformer_uw']:   # UW gain per run (same label)
        base = per.get(('tau', 'transformer', k[2]), {})
        per[('uw_gain', 'transformer_uw', k[2])] = {lb: v - base[lb] for lb, v in per[k].items() if lb in base}

    rows, js = [], {}
    for (t, n, m), d in sorted(per.items()):
        v = list(d.values())
        if not v:
            continue
        row = dict(task=t, model=n, metric=m, **{lb: d.get(lb) for lb in labels},
                   mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None, n_runs=len(v))
        rows.append(row)
        js.setdefault(t, {}).setdefault(n, {})[m] = row
    (out_dir / 'summary.json').write_text(json.dumps(js, indent=1))
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cmp_rows = []
    for n, groups in PAPER_TAU.items():
        for grp, (pr, pm, pp) in groups.items():
            got = [js.get('tau', {}).get(n, {}).get(f'allstrata_{grp}_{x}', {}).get('mean') for x in 'RMP']
            if got[0] is not None:
                cmp_rows.append(dict(task='tau', model=n, group=grp, paper_R=pr, paper_M=pm, paper_P=pp,
                                     release_allstrata_R=got[0], release_allstrata_M=got[1], release_allstrata_P=got[2]))
    for n, (pg, pgm) in PAPER_GRF.items():
        got = js.get('grf', {}).get(n, {})
        if got:
            cmp_rows.append(dict(task='grf', model=n, group='per foot', paper_GRF_Nkg=pg, paper_GRM_Nmkg=pgm,
                                 release_GRF_Nkg=got['GRF_Nkg']['mean'],
                                 release_GRM_Nmkg_unofficial=got['GRM_Nmkg_unofficial']['mean']))
    if cmp_rows:
        keys = list(dict.fromkeys(k for r in cmp_rows for k in r))
        with open(out_dir / 'paper_vs_release.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(cmp_rows)
    for r in rows:
        if r['task'] == 'grf' or r['metric'].startswith(('allstrata_all', 'official_all')):
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
    print(f'wrote {out_dir / "summary.csv"}')


if __name__ == '__main__':
    main()
