#!/bin/bash
# One baseline job with the paper arguments (local run: bash; cluster: see run_job.sbatch).
#
#   JOB   = tau | uw | grf | ablation
#   MODEL = mlp fconv lstm gru mamba2 transformer   (tau, uw, ablation)  |  mlp_deep gln   (grf)
#   CACHE = cache dir of prep_caches.py (holds tau/ and grf/)
#   RUNS  = output dir of this run, one per seed (for example runs/seed1); use a separate dir per ablation condition
#   SEED  = optional integer. Empty = unseeded tau run (as the paper seed-1 tau runs); UW then uses seed 0
#           and GRF seed 20260427 (their defaults).
#   COND  = A | B (ablation only; C is the main setting run by JOB=tau)
#   PY    = python executable (default: python)
#   GPU   = optional GPU index; sets CUDA_VISIBLE_DEVICES (Mamba2 kernels need the process's cuda:0)
#
# Example: JOB=tau MODEL=transformer CACHE=cache RUNS=runs/seed1 bash scripts/run_job.sh
set -euo pipefail
: "${JOB:?}" "${MODEL:?}" "${CACHE:?}" "${RUNS:?}"
PY=${PY:-python}
CACHE=$(realpath -m "$CACHE"); RUNS=$(realpath -m "$RUNS")
SEED_ARG=""
if [ -n "${SEED:-}" ]; then SEED_ARG="--seed $SEED"; fi
if [ -n "${GPU:-}" ]; then export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU; fi
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$JOB" in
  tau)
    $PY -u train.py --model $MODEL --condition C --device cuda:0 --patience 80 $SEED_ARG \
        --cache_dir $CACHE/tau --ckpt_dir $RUNS/tau/ckpt --log_dir $RUNS/tau/logs
    for s in val test; do
      $PY -u predict_tau.py --model $MODEL --ckpt $RUNS/tau/ckpt/${MODEL}_C_best.pt --cache_dir $CACHE/tau \
          --split $s --out $RUNS/tau/pred_${s}_${MODEL}.npz
    done ;;
  uw)
    $PY -u train_uw.py --model $MODEL --weight_mode softmax_per_sample --condition C --device cuda:0 $SEED_ARG \
        --init_ckpt $RUNS/tau/ckpt/${MODEL}_C_best.pt --cache_dir $CACHE/tau \
        --ckpt_out_dir $RUNS/uw/ckpt --log_out_dir $RUNS/uw/logs --out_tag softmax_per_sample \
        --patience 200 --lr 1e-4 --total_iters 300000 --lr_restart_at 100 --max_restarts 2 --batch_size 32
    for s in val test; do
      $PY -u predict_tau.py --model $MODEL --ckpt $RUNS/uw/ckpt/${MODEL}_C_softmax_per_sample_best.pt \
          --cache_dir $CACHE/tau --split $s --out $RUNS/uw/pred_${s}_${MODEL}_uw.npz
    done ;;
  grf)
    $PY -u grf/train_grf.py --cell G_${MODEL} --arch $MODEL $SEED_ARG --cache_dir $CACHE/grf --out_dir $RUNS/grf ;;
  ablation)
    : "${COND:?}"
    $PY -u train.py --model $MODEL --condition $COND --device cuda:0 --patience 80 $SEED_ARG \
        --cache_dir $CACHE/tau --ckpt_dir $RUNS/tau/ckpt --log_dir $RUNS/tau/logs
    for s in val test; do
      $PY -u predict_tau.py --model $MODEL --condition $COND --ckpt $RUNS/tau/ckpt/${MODEL}_${COND}_best.pt \
          --cache_dir $CACHE/tau --split $s --out $RUNS/tau/pred_${s}_${MODEL}.npz
    done ;;
  *) echo "unknown JOB=$JOB"; exit 1 ;;
esac
echo JOB_DONE
