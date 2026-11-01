#!/usr/bin/env python3
"""Unified training script for joint torque prediction baselines.

Usage:
  python train.py --model mlp --cache_dir <cache>/tau --ckpt_dir <runs>/tau/ckpt --log_dir <runs>/tau/logs
  python train.py --model transformer --condition C --device cuda:0 --cache_dir <cache>/tau ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from data.dataset import get_frame_loader, get_seq_loader
from models import build_model


SEQUENCE_MODELS = {"fconv", "lstm", "gru", "mamba2", "transformer"}


def get_scheduler(optimizer, warmup: int, decay_at: int, decay_factor: float,
                  total_iters: int):
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if step >= decay_at:
            return decay_factor
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch_frames(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x.unsqueeze(1)).squeeze(1)  # MLP: [B,1,D]->[B,D]
        loss = nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_one_epoch_seq(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)  # [B, T, D_out]
        diff = (pred - y) ** 2  # [B, T, D_out]
        masked = diff * mask.unsqueeze(-1).float()
        n_valid = mask.sum().clamp(min=1)
        loss = masked.sum() / (n_valid * y.shape[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, is_seq: bool, eval_chunk: int = 0):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    if is_seq:
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            if eval_chunk > 0:
                # Split the padded batch along the batch axis only. Each row keeps the same padded
                # length, and rows do not interact, so outputs match the one-shot forward.
                pred = torch.cat([model(x[i:i + eval_chunk]) for i in range(0, x.shape[0], eval_chunk)], 0)
            else:
                pred = model(x)
            diff = (pred - y) ** 2
            masked = diff * mask.unsqueeze(-1).float()
            n_valid = mask.sum().clamp(min=1)
            loss = masked.sum() / (n_valid * y.shape[-1])
            total_loss += loss.item()
            n_batches += 1
    else:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(pred, y)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["mlp", "fconv", "lstm", "gru", "mamba2", "transformer"])
    parser.add_argument("--condition", default=C.DEFAULT_INPUT_CONDITION, choices=["A", "B", "C"],
                        help="Input features: C = main setting, A / B = input ablation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total_iters", type=int, default=C.TOTAL_ITERS)
    parser.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LR)
    parser.add_argument("--patience", type=int, default=C.PATIENCE,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--eval_chunk", type=int, default=0,
                        help="Validation forward in sub-batches of this many trials (0 = whole batch)")
    parser.add_argument("--cache_dir", type=str, default=str(C.CACHE_DIR),
                        help="Override cache directory (train/test_cache.pt + norm_stats)")
    parser.add_argument("--ckpt_dir", type=str, default=str(C.CKPT_DIR),
                        help="Override checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=str(C.RESULTS_DIR),
                        help="Override training log directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, seed python/numpy/torch/CUDA. Default None = unseeded (original behavior)")
    parser.add_argument("--cpu_budget", type=float, default=0,
                        help="If >0, save a resume state and exit (code 3) once this process used this many "
                             "CPU seconds (for nodes with a per-process CPU-time limit). Resume with --resume.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <ckpt_dir>/<model>_<cond>_state.pt if it exists")
    args = parser.parse_args()
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cache_dir = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir = Path(args.log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_seq = args.model in SEQUENCE_MODELS
    input_dim = C.INPUT_DIMS[args.condition]

    # --- Norm stats ---
    norm_path = cache_dir / f"norm_stats_{args.condition}.pt"
    if not norm_path.exists():
        sys.exit(f"Norm stats not found at {norm_path}. Run prep_caches.py first.")
    norm_stats = torch.load(norm_path, weights_only=False)

    # --- Data ---
    train_cache = cache_dir / "train_cache.pt"
    val_cache_path = cache_dir / "val_cache.pt"
    if not val_cache_path.exists():
        # Never select checkpoints on the test split.
        raise FileNotFoundError(f"{val_cache_path} not found; build the caches with prep_caches.py")
    val_cache = val_cache_path

    if is_seq:
        train_loader = get_seq_loader(train_cache, args.condition, norm_stats,
                                      batch_size=args.batch_size, training=True)
        val_loader = get_seq_loader(val_cache, args.condition, norm_stats,
                                    batch_size=args.batch_size, training=False)
    else:
        train_loader = get_frame_loader(train_cache, args.condition, norm_stats,
                                        batch_size=args.batch_size * 32)
        val_loader = get_frame_loader(val_cache, args.condition, norm_stats,
                                      batch_size=args.batch_size * 32, shuffle=False)

    # --- Model ---
    model = build_model(args.model, input_dim, C.OUTPUT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, params: {n_params:,}, input_dim: {input_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = get_scheduler(optimizer, C.WARMUP_ITERS, C.LR_DECAY_AT,
                              C.LR_DECAY_FACTOR, args.total_iters)

    # --- Training loop ---
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = len(train_loader)
    n_epochs = max(1, args.total_iters // steps_per_epoch)
    print(f"Steps/epoch: {steps_per_epoch}, epochs: {n_epochs}")

    best_val = float("inf")
    patience_counter = 0
    train_fn = train_one_epoch_seq if is_seq else train_one_epoch_frames
    log = []
    t0 = time.time()
    start_epoch = 0
    state_path = ckpt_dir / f"{args.model}_{args.condition}_state.pt"
    if args.resume and state_path.exists():
        import random, numpy as np
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        best_val, patience_counter, log = st["best_val"], st["patience_counter"], st["log"]
        start_epoch = st["epoch"] + 1
        t0 -= log[-1]["elapsed"]
        random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"])
        torch.set_rng_state(st["rng_torch"].cpu()); torch.cuda.set_rng_state_all([r.cpu() for r in st["rng_cuda"]])
        print(f"Resumed from {state_path} at epoch {start_epoch}, best={best_val:.6f}")

    for epoch in range(start_epoch, n_epochs):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device, is_seq, args.eval_chunk)

        elapsed = time.time() - t0
        step = (epoch + 1) * steps_per_epoch
        entry = {"epoch": epoch, "step": step, "train_loss": train_loss,
                 "val_loss": val_loss, "elapsed": elapsed,
                 "lr": optimizer.param_groups[0]["lr"]}
        log.append(entry)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / f"{args.model}_{args.condition}_best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, ckpt_path)
        else:
            patience_counter += 1

        if (epoch + 1) % max(1, n_epochs // 20) == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  step={step}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"best={best_val:.6f}  lr={entry['lr']:.2e}  "
                  f"patience={patience_counter}/{args.patience}  {elapsed:.0f}s")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

        if args.cpu_budget > 0:
            import resource, random, numpy as np
            ru = resource.getrusage(resource.RUSAGE_SELF)
            if ru.ru_utime + ru.ru_stime > args.cpu_budget:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val": best_val,
                            "patience_counter": patience_counter, "log": log,
                            "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
                           str(state_path) + ".tmp")
                Path(str(state_path) + ".tmp").replace(state_path)
                print(f"  CPU budget reached at epoch {epoch+1}; state saved to {state_path}", flush=True)
                sys.exit(3)

    # --- Save log ---
    log_path = log_dir / f"train_log_{args.model}_{args.condition}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best val: {best_val:.6f}")
    print(f"Checkpoint: {ckpt_dir / f'{args.model}_{args.condition}_best.pt'}")
    print(f"Log: {log_path}")

    # The test split is predicted once by predict_tau.py and scored by benchmark/score_v12.py, not here.


if __name__ == "__main__":
    main()
