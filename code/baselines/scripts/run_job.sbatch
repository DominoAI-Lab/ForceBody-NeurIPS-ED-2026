#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log
# Slurm wrapper of run_job.sh. Add the partition / account lines of your cluster.
# The paper runs used one GPU with >= 40 GB for the Transformer (full-trial validation forward); the UW fine-tune
# runs to --total_iters (see README) and may need up to 2 days.
# Example: sbatch --export=ALL,JOB=tau,MODEL=gru,CACHE=$PWD/cache,RUNS=$PWD/runs/seed2,SEED=2,PY=python \
#              scripts/run_job.sbatch
set -euo pipefail
nvidia-smi -L
bash "${SLURM_SUBMIT_DIR:-.}/scripts/run_job.sh"
