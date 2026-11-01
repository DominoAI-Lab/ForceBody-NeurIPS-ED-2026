# ForceBody baselines

Training and evaluation code of the ForceBody force benchmark baselines:

- Joint torque prediction (T-tau): MLP, Conv1D (`fconv`), BiLSTM (`lstm`), GRU, Mamba2, Transformer.
- Uncertainty-weighted (UW) fine-tune of a torque model (paper row Transformer+UW).
- Ground reaction force prediction (T-GRF, with GRM and CoM acceleration as extra outputs): MLP (`mlp_deep`) and GLN (`gln`).

The code reuses the release loader `../forcebody_io.py` and the official benchmark code in `../benchmark/`
(cache builder `build_bench_cache_v12.py` and scorer `score_v12.py`). It needs only the ForceBody release files.
It does not need the SKEL body model or any fitting code.

## Install

```bash
pip install -r requirements.txt          # torch >= 2.4, numpy
# Mamba2 only (CUDA GPU):
pip install causal_conv1d==1.6.1 mamba_ssm==2.3.1 --no-build-isolation
```

Tested with Python 3.9, torch 2.8.0 (CUDA 12.8), numpy 1.26.4, mamba_ssm 2.3.1, causal_conv1d 1.6.1.

## Main steps

Run all commands from `code/baselines/`. `RELEASE` is the ForceBody release root (it holds `manifest.csv`).
`CACHE` and `RUNS` are any output folders. Use one `RUNS` folder per seed.

```text
1. cache build   prep_caches.py        -> CACHE/bench_*_pass0.npz, CACHE/tau, CACHE/grf
2. training      train.py              -> RUNS/tau/ckpt          (torque, 6 models)
                 grf/train_grf.py      -> RUNS/grf/<cell>        (GRF, 2 models; also writes test predictions)
3. UW fine-tune  train_uw.py           -> RUNS/uw/ckpt
4. inference     predict_tau.py        -> RUNS/{tau,uw}/pred_test_*.npz
5. scoring       ../benchmark/score_v12.py
```

### 1. Build the caches

```bash
python prep_caches.py --release $RELEASE --out_dir $CACHE --workers 16
```

This first builds the official benchmark caches `bench_{train,val,test}_pass0.npz` (pose source pass0, extra input
`joints`) if they are missing. It then writes the torque caches `tau/{train,val,test}_cache.pt`, the normalization
statistics `tau/norm_stats_{A,B,C}.pt` (train split, scored frames only), the GRF caches `grf/*_grf.npz` and
`grf/norm_stats_grf.npz`. Splits: train / val (`split_addb_val`) / test (`split_addb`).

### 2. Train

Torque (one model per call, `--model mlp | fconv | lstm | gru | mamba2 | transformer`):

```bash
python train.py --model transformer --condition C --device cuda:0 --patience 80 \
    --cache_dir $CACHE/tau --ckpt_dir $RUNS/tau/ckpt --log_dir $RUNS/tau/logs [--seed 2]
```

The checkpoint with the lowest validation loss is kept (`<model>_C_best.pt`). Early stopping uses `--patience 80`
epochs. `--eval_chunk N` splits the validation forward into sub-batches of N trials (same result, less GPU memory).
`--cpu_budget S` saves a resume state and exits with code 3 after S CPU seconds; restart with `--resume`.

GRF (`--arch mlp_deep | gln`):

```bash
python grf/train_grf.py --cell G_mlp_deep --arch mlp_deep --cache_dir $CACHE/grf --out_dir $RUNS/grf [--seed 20260427]
```

The script selects the checkpoint on the validation GRF error and then predicts the test split:
`RUNS/grf/<cell>/pred_test_grf.npz` ([T,2,3] body weights, for the scorer) and `pred_test_full15.npz` (all 15 outputs).

### 3. UW fine-tune

```bash
python train_uw.py --model transformer --weight_mode softmax_per_sample --condition C --device cuda:0 \
    --init_ckpt $RUNS/tau/ckpt/transformer_C_best.pt --cache_dir $CACHE/tau \
    --ckpt_out_dir $RUNS/uw/ckpt --log_out_dir $RUNS/uw/logs --out_tag softmax_per_sample \
    --patience 200 --lr 1e-4 --total_iters 300000 --lr_restart_at 100 --max_restarts 2 --batch_size 32 [--seed 0]
```

Weight per training window b: `w_b = 1 + softmax over the batch of mean(sigma_tau / y_std)` (Zhang et al., ICCV 2023,
Eq. 18). Loss: `sum(w * mask * (pred - y)^2) / sum(w * mask)`. `sigma_tau` is a label-side uncertainty, never an input.

### 4. Predict

```bash
python predict_tau.py --model transformer --ckpt $RUNS/tau/ckpt/transformer_C_best.pt \
    --cache_dir $CACHE/tau --split test --out $RUNS/tau/pred_test_transformer.npz
python predict_tau.py --model transformer --ckpt $RUNS/uw/ckpt/transformer_C_softmax_per_sample_best.pt \
    --cache_dir $CACHE/tau --split test --out $RUNS/uw/pred_test_transformer_uw.npz
```

Each trial is predicted in one forward pass and de-normalized to N m/kg. The output has one key per `trial_uid`,
[T,37] in `dof_names` order.

### 5. Score

```bash
python ../benchmark/score_v12.py --task tau --cache $CACHE/bench_test_pass0.npz \
    --pred $RUNS/tau/pred_test_transformer.npz --out $RUNS/score_tau_transformer.json
python ../benchmark/score_v12.py --task grf --cache $CACHE/bench_test_pass0.npz \
    --pred $RUNS/grf/G_mlp_deep/pred_test_grf.npz --out $RUNS/score_grf_mlp_deep.json
```

See `../benchmark/BENCHMARK.md` for the metrics and strata.

## Extra tools

| Tool | Purpose |
|---|---|
| `aggregate.py` | Scores every prediction file of one or more run folders, writes per-run values, mean +- std over runs (seeds), UW gain per run and `paper_vs_release.csv`. |
| `scripts/run_job.sh` | One job with the paper arguments: `JOB=tau|uw|grf|ablation`, `MODEL`, `CACHE`, `RUNS`, optional `SEED`, `COND`, `GPU`, `PY`. |
| `scripts/run_job.sbatch` | Slurm wrapper of `run_job.sh` (add your partition / account). |
| `../visualize.py` | Renders one trial (see `../VISUALIZATION.md`). Needs the external addb2skel render utilities and the SKEL model; the baselines do not. |

Seeds (the benchmark runs used three seeds, 1 to 3):

```bash
for S in 1 2 3; do
  for M in mlp fconv lstm gru mamba2 transformer; do JOB=tau MODEL=$M CACHE=$CACHE RUNS=runs/seed$S SEED=$S bash scripts/run_job.sh; done
  JOB=uw  MODEL=transformer CACHE=$CACHE RUNS=runs/seed$S SEED=$S bash scripts/run_job.sh
  for M in mlp_deep gln; do JOB=grf MODEL=$M CACHE=$CACHE RUNS=runs/seed$S SEED=$S bash scripts/run_job.sh; done
done
python aggregate.py --cache_dir $CACHE --runs seed1=runs/seed1 seed2=runs/seed2 seed3=runs/seed3 --out_dir results
```

The paper seed-1 runs were unseeded for the torque models (omit `SEED`), seed 0 for UW and seed 20260427 for GRF
(the script defaults). Seed-1 results therefore cannot be reproduced bit for bit.

Input ablation (Transformer; A = poses, B = A + betas + mass + height, C = B + velocities, the main setting):

```bash
for C in A B; do for S in 1 2 3; do
  JOB=ablation COND=$C MODEL=transformer CACHE=$CACHE RUNS=runs_ablation/cond$C/seed$S SEED=$S bash scripts/run_job.sh
done; done
python aggregate.py --cache_dir $CACHE --runs seed1=runs_ablation/condA/seed1 seed2=runs_ablation/condA/seed2 \
    seed3=runs_ablation/condA/seed3 --out_dir results_ablation_A
```

## Known quirks (kept for parity with the paper numbers)

- MLP batch: the MLP is trained per frame with `--batch_size 128` x 32 = 4,096 frames per step. The sequence models use
  128 windows of 256 frames.
- Pose velocity and acceleration use dt = 1 frame (not divided by the frame rate). Trials run at 61 to 250 Hz, so the
  same motion gives different `pose_vel` values at different rates. `trans_vel` is in m/s. The 256-frame crop is
  2.56 s only at 100 Hz.
- UW fine-tune: lr 1e-4, warm-up 500 steps, LR x 0.1 at 70% of `--total_iters`. When patience reaches
  `--lr_restart_at`, the best checkpoint is reloaded and the schedule restarts. The restart counter is read from the
  log entry of the same epoch, so it is always 0 and `--max_restarts` does not limit restarts. With
  `--lr_restart_at 100 < --patience 200`, early stopping never fires and the run ends at `--total_iters`.
- Validation loaders of the sequence models in `train.py` shuffle the trials. Batches pad trials with zeros and the
  models get no padding mask, so the validation loss depends on the batch composition. `predict_tau.py` runs each
  trial alone without padding.
- The validation loss is a mean of batch means. The MLP validation loader drops the last incomplete batch.
- The random crop uses `randint(0, T - 256)`, so the last frame of a trial longer than 256 frames is never in a crop.
- `train.py` without `--seed` is unseeded. Mamba2 and cuDNN kernels are not deterministic.
- 12 input features have a train standard deviation of 0 (normalization divides by 1e-8). They are constant in all splits.
- GRF: training and validation use only window centres with a full 51-frame window (shorter trials are skipped). Test
  prediction covers every frame with edge-replicated windows.
- Mamba2 kernels (mamba_ssm 2.3.1 + triton) work only on the process's `cuda:0`. Select the GPU with
  `CUDA_VISIBLE_DEVICES` (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`) and pass `--device cuda:0`. The model runs a GPU test at
  import (`MAMBA_SKIP_SMOKETEST=1` skips it).
- The Transformer validation forward on full trials needs about 30 GB of GPU memory at batch 128; use `--eval_chunk`
  on smaller GPUs.
