#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log
# Slurm wrapper of run_job.sh. Add the partition / account lines of your cluster.
# The paper runs used one GPU with >= 40 GB for the Transformer (full-trial validation forward). The UW fine-tune
# runs to --total_iters (see README). It took about 8 to 10 hours per seed on our GPUs, and the 2-day limit leaves room for
# slower GPUs. The tau and grf jobs are shorter. Lower the limit for them with 'sbatch --time=...' if your queue
# favours short jobs.
# Example: sbatch --export=ALL,JOB=tau,MODEL=gru,CACHE=$PWD/cache,RUNS=$PWD/runs/seed2,SEED=2,PY=python \
#              scripts/run_job.sbatch
set -euo pipefail
nvidia-smi -L
bash "${SLURM_SUBMIT_DIR:-.}/scripts/run_job.sh"
