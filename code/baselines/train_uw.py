"""Uncertainty-weighted (UW) fine-tune of a trained torque model (paper Transformer+UW row).

Warm-starts from a checkpoint of train.py and continues training with a weighted MSE (Zhang et al., ICCV 2023, Eq. 18):
  per training window b, s_b = mean over scored (frame x DOF) units of sigma_tau / y_std,
  w_b = 1 + softmax_B(s_b) over the windows of the batch,
  loss = sum(w * mask * (pred - y)^2) / sum(w * mask).
sigma_tau is a label-side Monte Carlo uncertainty, never a model input. Validation uses the plain masked MSE.

LR warm restart: when patience reaches --lr_restart_at, the best checkpoint is reloaded, the LR schedule restarts
(warm-up 500 steps) and patience is reset. Known quirk kept for parity with the paper run: the restart counter is read
from the log entry of the same epoch, so it is always 0 and --max_restarts never stops restarts. With
--lr_restart_at 100 < --patience 200, early stopping therefore never fires and the run ends at --total_iters.
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
from models import build_model
from dataset_uw import get_seq_loader_uw


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_uncertainty_weights(sigma: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-window weight w_b = 1 + softmax over the batch of the mean sigma of window b.

    sigma : [B, T, D] per-frame, per-DOF Monte Carlo std, already divided by y_std (normalized space).
    mask  : [B, T] boolean mask of scored frames.
    Returns weights of shape [B, T, D]; unscored entries are 0.
    """
    flat_mask = mask.unsqueeze(-1).expand_as(sigma)
    sigma_d = torch.nan_to_num(sigma.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    valid_f = flat_mask.float()
    s_sum = (sigma_d * valid_f).sum(dim=(1, 2))            # [B]
    n_b   = valid_f.sum(dim=(1, 2)).clamp(min=1)            # [B]
    per_sample = s_sum / n_b                                 # [B]
    per_sample = per_sample - per_sample.max()               # numerical
    sm_B = torch.softmax(per_sample, dim=0)                  # [B], mean = 1/B
    w_B  = 1.0 + sm_B                                         # [B]
    w = w_B.view(-1, 1, 1).expand_as(sigma).contiguous()
    w = torch.where(flat_mask, w, torch.zeros_like(w))
    return w


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                  weights: torch.Tensor, mask: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """Weighted MSE sum(w * sq) / sum(w): weights act as relative importance, the loss scale stays near plain MSE."""
    sq = (pred - target) ** 2                          # [B, T, D]
    masked_w = weights * mask.unsqueeze(-1).float()    # [B, T, D]
    denom = masked_w.sum().clamp(min=eps)
    return (masked_w * sq).sum() / denom


def plain_mse(pred, target, mask):
    sq = (pred - target) ** 2
    masked = sq * mask.unsqueeze(-1).float()
    n_valid = mask.sum().clamp(min=1)
    return masked.sum() / (n_valid * target.shape[-1])


# ---------------------------------------------------------------------------
# Train / eval epoch loops
# ---------------------------------------------------------------------------

def get_scheduler(optimizer, warmup, decay_at, decay_factor):
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if step >= decay_at:
            return decay_factor
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y, sigma, mask in loader:
        x, y, sigma, mask = x.to(device), y.to(device), sigma.to(device), mask.to(device)
        pred = model(x)
        w = compute_uncertainty_weights(sigma, mask)
        loss = weighted_mse(pred, y, w, mask)
        optimizer.zero_grad()
        if torch.isnan(loss) or torch.isinf(loss):
            # skip a non-finite batch (it still counts in the epoch mean denominator, as in the paper run)
            n_batches += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Validation: always plain MSE for fair model comparison."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y, sigma, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        total_loss += plain_mse(pred, y, mask).item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["mlp", "fconv", "lstm", "gru", "transformer", "mamba2"])
    parser.add_argument("--weight_mode", default="softmax_per_sample", choices=["softmax_per_sample"],
                        help="Only the paper weighting is provided")
    parser.add_argument("--condition", default=C.DEFAULT_INPUT_CONDITION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init_ckpt", required=True,
                        help="Checkpoint of train.py to warm-start from (<model>_<cond>_best.pt)")
    parser.add_argument("--total_iters", type=int, default=300_000,
                        help="Total iters of the fine-tune (paper run: 300000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Paper run: 32")
    parser.add_argument("--lr", type=float, default=1e-4, help="Paper run: 1e-4")
    parser.add_argument("--patience", type=int, default=200, help="Paper run: 200")
    parser.add_argument("--lr_restart_at", type=int, default=100,
                        help="If >0, when patience hits this value the best checkpoint is reloaded, the LR "
                             "schedule restarts and patience is zeroed (paper run: 100)")
    parser.add_argument("--max_restarts", type=int, default=2,
                        help="Intended maximum number of restarts. Not enforced (see module docstring); "
                             "kept for parity with the paper run.")
    parser.add_argument("--out_tag", default=None,
                        help="Tag appended to output filenames; defaults to weight_mode")
    parser.add_argument("--cache_dir", type=str, default=str(C.CACHE_DIR),
                        help="Override cache dir (train/test_cache + norm_stats)")
    parser.add_argument("--ckpt_out_dir", type=str, required=True, help="Output dir for checkpoints")
    parser.add_argument("--log_out_dir", type=str, required=True, help="Output dir for the training log")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default 0 = original hard-coded value)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_dim = C.INPUT_DIMS[args.condition]
    tag = args.out_tag or args.weight_mode
    cache_dir = Path(args.cache_dir)

    # Reproducible init val: fix all RNGs before model build / first eval.
    import numpy as np, random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Norm stats + data ----
    norm_path = cache_dir / f"norm_stats_{args.condition}.pt"
    norm_stats = torch.load(norm_path, weights_only=False)

    train_cache = cache_dir / "train_cache.pt"
    val_cache_path = cache_dir / "val_cache.pt"
    if not val_cache_path.exists():
        # Never select checkpoints on the test split.
        raise FileNotFoundError(f"{val_cache_path} not found; build the caches with prep_caches.py")
    val_cache = val_cache_path
    train_loader = get_seq_loader_uw(train_cache, args.condition, norm_stats,
                                       batch_size=args.batch_size,
                                       shuffle=True, training=True, seed=args.seed)
    # Deterministic validation: no shuffle so init val is reproducible
    # across runs that share the same checkpoint and dataset.
    val_loader   = get_seq_loader_uw(val_cache, args.condition, norm_stats,
                                       batch_size=args.batch_size,
                                       shuffle=False, training=False)

    # ---- Model + warm start ----
    init_ckpt = args.init_ckpt
    print(f"Loading warm-start from: {init_ckpt}")
    state = torch.load(init_ckpt, weights_only=False, map_location=device)
    model = build_model(args.model, input_dim, C.OUTPUT_DIM).to(device)

    model.load_state_dict(state["model"])
    print(f"  baseline val_loss: {state.get('val_loss', 'n/a')}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, params: {n_params:,}, weight_mode: {args.weight_mode}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = get_scheduler(optimizer, warmup=500,
                               decay_at=int(args.total_iters * 0.7),
                               decay_factor=0.1)

    # ---- Output paths ----
    CKPT_OUT = Path(args.ckpt_out_dir)
    LOG_OUT  = Path(args.log_out_dir)
    CKPT_OUT.mkdir(parents=True, exist_ok=True)
    LOG_OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_OUT / f"{args.model}_{args.condition}_{tag}_best.pt"
    log_path  = LOG_OUT  / f"{args.model}_{args.condition}_{tag}.json"

    # ---- Refinement loop ----
    steps_per_epoch = len(train_loader)
    n_epochs = max(1, args.total_iters // steps_per_epoch)
    print(f"Steps/epoch: {steps_per_epoch}, refinement epochs: {n_epochs}, "
          f"target iters: {args.total_iters}")

    # Sanity check: report baseline val_loss with the new dataset format
    init_val = evaluate(model, val_loader, device)
    print(f"Initial val MSE (warm-start, plain): {init_val:.6f}")

    best_val = init_val
    patience_counter = 0
    log = [{"epoch": -1, "step": 0, "train_loss": None, "val_loss": init_val,
            "lr": args.lr, "phase": "warm_start"}]
    t0 = time.time()

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        step = (epoch + 1) * steps_per_epoch
        log.append({"epoch": epoch, "step": step, "train_loss": train_loss,
                    "val_loss": val_loss, "elapsed": elapsed,
                    "lr": optimizer.param_groups[0]["lr"], "phase": "refinement"})

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "weight_mode": args.weight_mode,
                        "init_ckpt": init_ckpt}, ckpt_path)
        else:
            patience_counter += 1

        # LR warm restart on plateau. The counter below is read from the entry appended in this epoch,
        # so it is always 0 (paper behaviour, kept on purpose; see module docstring).
        if (args.lr_restart_at > 0
            and patience_counter >= args.lr_restart_at
            and patience_counter < args.patience):
            n_restarts = log[-1].get("n_restarts", 0)
            if n_restarts < args.max_restarts:
                # Reload best checkpoint and reset LR to initial.
                if ckpt_path.exists():
                    state = torch.load(ckpt_path, weights_only=False, map_location=device)
                    model.load_state_dict(state["model"])
                for g in optimizer.param_groups:
                    g["lr"] = args.lr
                # Reset scheduler so warmup happens again
                scheduler = get_scheduler(optimizer, warmup=500,
                                            decay_at=int(args.total_iters * 0.7),
                                            decay_factor=0.1)
                patience_counter = 0
                print(f"  [restart {n_restarts+1}/{args.max_restarts}] "
                      f"reloaded best ckpt + reset lr -> {args.lr:.1e} at epoch {epoch+1}")
                log[-1]["n_restarts"] = n_restarts + 1
                log[-1]["restart_event"] = True
            else:
                # already used all restarts, let early-stopping handle it
                pass

        if (epoch + 1) % max(1, n_epochs // 25) == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  step={step}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"best={best_val:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"patience={patience_counter}/{args.patience}  {elapsed:.0f}s",
                  flush=True)

        # Live JSON dump every epoch so the monitor can read progress
        # without waiting for the run to end / early-stop.
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ---- Save log ----
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # ---- Save FINAL checkpoint (state at training end, regardless of best)
    # Best ckpt is already saved on every val-loss improvement (above).
    # The final ckpt is for downstream analysis / re-loading the exact
    # state when training stopped (early-stop or max-iters).
    final_path = CKPT_OUT / f"{args.model}_{args.condition}_{tag}_final.pt"
    torch.save({"model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val": best_val,
                "weight_mode": args.weight_mode,
                "init_ckpt": init_ckpt,
                "tag": tag}, final_path)

    print(f"\nDone. Init val: {init_val:.6f}, Best val: {best_val:.6f}, "
          f"delta = {(best_val - init_val):.6f} ({(best_val - init_val)/init_val*100:+.2f}%)")
    print(f"Best ckpt:  {ckpt_path}")
    print(f"Final ckpt: {final_path}")
    print(f"Log:        {log_path}")


if __name__ == "__main__":
    main()
