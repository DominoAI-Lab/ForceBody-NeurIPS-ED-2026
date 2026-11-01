#!/usr/bin/env python3
"""GRF baselines (paper GRF table, ForceBody rows MLP and GLN).

Inputs per frame: q (poses), qdd (central second difference with dt = 1 / fps), joint centres relative to the pelvis;
window 51 frames, stride 5 (train) / 1 (val), batch 32, AdamW lr 1e-4 wd 1e-5, grad clip 1.0, dropout 0.0,
loss = MSE on 15 outputs (GRF 6 + GRM 6 + CoM acceleration 3, labels in body weights / m s^-2), 10 epochs with
auto-extend +20 (at most 2 times) and patience 10, seed 20260427, checkpoint selected on the validation GRF metric.

Windows are gathered from the in-RAM cache with index arrays. Training and validation use only window centres with a
full window (trials shorter than the window are skipped). Frames scored = grf_mask (eval_grf_frame_ok &
eval_grf_trial_ok). After training, the best checkpoint predicts every frame of every test trial (edge frames use
edge-replicated windows) and writes prediction npz files for benchmark/score_v12.py.

Usage: python grf/train_grf.py --cell G_mlp_deep --arch mlp_deep --cache_dir <cache>/grf --out_dir <runs>/grf
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model  # noqa: E402

INPUTS = ('q', 'qdd', 'jc_pelvis')
G = 9.81


class WindowData:
    def __init__(self, path, norm, window):
        z = np.load(path)
        self.half = window // 2
        feats = []
        for k in INPUTS:
            a = z[k].reshape(len(z[k]), -1).astype(np.float32)
            a = (a - norm[f'mean_{k}'].reshape(-1)) / norm[f'std_{k}'].reshape(-1)
            feats.append(a)
        self.x = torch.from_numpy(np.concatenate(feats, 1))                       # [N, 164] per frame
        self.y = torch.from_numpy(np.concatenate([z['grf_bw'], z['grm_bw'], z['com_acc']], 1))   # [N, 15]
        self.valid = torch.from_numpy(z['valid'])
        self.off = z['trial_offsets']
        self.uid = [str(u) for u in z['trial_uid']]
        self.mass_dyn = z['mass_dyn']

    def centers(self, stride):
        """Global frame index of window centres: c in range(half, T - half, stride) per trial (paper index)."""
        out = []
        for i in range(len(self.off) - 1):
            s, e = int(self.off[i]), int(self.off[i + 1])
            if e - s < 2 * self.half + 1:
                continue
            out.append(np.arange(s + self.half, e - self.half, stride))
        return np.concatenate(out)

    def batch(self, c, device):
        rel = torch.arange(-self.half, self.half + 1)
        idx = torch.as_tensor(c)[:, None] + rel[None, :]
        x = self.x[idx].reshape(len(c), -1)
        return x.to(device, non_blocking=True), self.y[c].to(device), self.valid[c].to(device)


def quick_eval(model, data, centers, device, bs):
    """Validation metric: mean over valid samples of the sum over feet of |error| (N/kg = BW * 9.81)."""
    model.eval()
    sg = sm = sc = 0.0
    n = 0
    with torch.no_grad():
        for b in range(0, len(centers), bs):
            x, y, v = data.batch(centers[b:b + bs], device)
            if not v.any():
                continue
            p, t = model(x[v]), y[v]
            sg += ((p[:, :6] - t[:, :6]) * G).reshape(-1, 2, 3).norm(dim=-1).sum(-1).sum().item()
            sm += ((p[:, 6:12] - t[:, 6:12]) * G).reshape(-1, 2, 3).norm(dim=-1).sum(-1).sum().item()
            sc += (p[:, 12:15] - t[:, 12:15]).norm(dim=-1).sum().item()
            n += int(v.sum())
    return dict(n=n, grf_n_per_kg=sg / n, grm_nm_per_kg=sm / n, com_acc_m_s2=sc / n)


def predict_all(model, data, device, bs=4096):
    """Prediction for every frame of every trial; edge windows replicate the first / last frame."""
    model.eval()
    out = {}
    rel = torch.arange(-data.half, data.half + 1)
    with torch.no_grad():
        for i, u in enumerate(data.uid):
            s, e = int(data.off[i]), int(data.off[i + 1])
            fr = torch.arange(s, e)
            idx = (fr[:, None] + rel[None, :]).clamp(s, e - 1)
            preds = []
            for b in range(0, len(fr), bs):
                preds.append(model(data.x[idx[b:b + bs]].reshape(len(idx[b:b + bs]), -1).to(device)).cpu())
            out[u] = torch.cat(preds).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arch', required=True, choices=['mlp_deep', 'gln'])
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--window', type=int, default=51)
    ap.add_argument('--stride_train', type=int, default=5)
    ap.add_argument('--stride_eval', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--auto_extend', type=int, default=20)
    ap.add_argument('--max_extensions', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=20260427)
    ap.add_argument('--eval_batch', type=int, default=32,
                    help='validation batch (paper: same as --batch); only changes speed, not the metric')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache = Path(args.cache_dir)
    run = Path(args.out_dir) / args.cell
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2))
    logf = open(run / 'train.log', 'a')

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    norm = dict(np.load(cache / 'norm_stats_grf.npz'))
    tr = WindowData(cache / 'train_grf.npz', norm, args.window)
    va = WindowData(cache / 'val_grf.npz', norm, args.window)
    c_tr, c_va = tr.centers(args.stride_train), va.centers(args.stride_eval)
    in_dim = tr.x.shape[1] * args.window
    model = build_model(args.arch, in_dim, 15, window=args.window, dropout=args.dropout).to(device)
    log(f'=== {args.cell} arch={args.arch} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M '
        f'train_windows={len(c_tr)} val_windows={len(c_va)} in_dim={in_dim}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_ep, no_imp, hist, ep, ext, total = float('inf'), -1, 0, [], 0, 0, args.epochs
    g = torch.Generator().manual_seed(args.seed)
    while ep < total:
        ep += 1
        model.train()
        t0 = time.time()
        perm = c_tr[torch.randperm(len(c_tr), generator=g).numpy()]
        sl, nb = 0.0, 0
        for b in range(0, len(perm) - args.batch + 1, args.batch):       # drop_last=True
            x, y, v = tr.batch(perm[b:b + args.batch], device)
            if not v.any():
                continue
            loss = ((model(x[v]) - y[v]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sl += float(loss.item())
            nb += 1
        met = quick_eval(model, va, c_va, device, args.eval_batch)
        log(f"epoch {ep:3d}/{total} train={sl / max(nb, 1):.5f} val GRF={met['grf_n_per_kg']:.4f} "
            f"GRM={met['grm_nm_per_kg']:.4f} CoM={met['com_acc_m_s2']:.3f} dt={time.time() - t0:.0f}s")
        hist.append({'epoch': ep, 'train_loss': sl / max(nb, 1), 'val_main': met['grf_n_per_kg'], 'metrics': met})
        if met['grf_n_per_kg'] < best - 1e-5:
            best, best_ep, no_imp = met['grf_n_per_kg'], ep, 0
            torch.save({'model': model.state_dict(), 'epoch': ep, 'metrics': met, 'args': vars(args)}, run / 'best.pt')
        else:
            no_imp += 1
        if ep == total and no_imp == 0 and ext < args.max_extensions:
            ext += 1
            total += args.auto_extend
            log(f'Auto-extend +{args.auto_extend} ({ext}/{args.max_extensions})')
        if no_imp >= args.patience:
            log(f'Early stop: no val improvement for {args.patience} epochs')
            break
    (run / 'history.json').write_text(json.dumps(hist, indent=2))
    (run / 'best_metrics.json').write_text(json.dumps({'best_epoch': best_ep, 'best_val': best}, indent=2))
    log(f'DONE best_epoch={best_ep} best_val GRF={best:.4f}')

    # ---- test prediction (once, best checkpoint) ----
    model.load_state_dict(torch.load(run / 'best.pt', map_location=device)['model'])
    te = WindowData(cache / 'test_grf.npz', norm, args.window)
    pred = predict_all(model, te, device)
    np.savez(run / 'pred_test_grf.npz', **{u: p[:, :6].reshape(-1, 2, 3).astype(np.float32) for u, p in pred.items()})
    np.savez(run / 'pred_test_full15.npz', **{u: p.astype(np.float32) for u, p in pred.items()})
    log(f'wrote {run / "pred_test_grf.npz"} ({len(pred)} trials)')


if __name__ == '__main__':
    main()
