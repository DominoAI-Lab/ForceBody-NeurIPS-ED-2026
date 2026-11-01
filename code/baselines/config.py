"""Settings of the paper torque baselines (the values used for the paper runs)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Torque cache written by prep_caches.py (<out_dir>/tau). Override with --cache_dir or FORCEBODY_TAU_CACHE.
CACHE_DIR = Path(os.environ.get("FORCEBODY_TAU_CACHE", ROOT / "cache" / "tau"))
RESULTS_DIR = ROOT / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"

# Optimization (MInT, option_vq.py)
LR = 2e-4
WARMUP_ITERS = 1000
LR_DECAY_FACTOR = 0.05
LR_DECAY_AT = 200_000
TOTAL_ITERS = 300_000
BATCH_SIZE = 128
SEQ_LEN = 256    # training crop in frames (2.56 s at 100 Hz)
PATIENCE = 80    # early-stopping patience in epochs; every paper run stopped 80 epochs after its best epoch

# Input features per frame. Table 4 uses C; the input ablation in the appendix compares A and B.
#   A: SKEL poses (46)
#   B: A + betas (10) + body mass + height                                             = 58
#   C: B + root translation velocity (3) + pose velocity (46) + pose acceleration (46)  = 153
INPUT_DIMS = {"A": 46, "B": 58, "C": 153}
DEFAULT_INPUT_CONDITION = "C"
OUTPUT_DIM = 37  # joint torques of the 37 AddBiomechanics (Rajagopal) DOFs
