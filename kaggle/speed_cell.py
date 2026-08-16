# =============================================================================
# BioNeural — SPEED PROBE · paste this whole file into ONE Kaggle cell
# =============================================================================
# Clones the repo, installs deps, then measures TRAIN tok/s for BioNeural and a
# standard transformer on the T4. No dataset needed (synthetic token ids).
#
# Read the numbers like this:
#   - first-seg time  = Triton JIT compile + warmup (expect a few seconds, once)
#   - bio tok/s       = steady-state throughput of the full organism pipeline
#   - std tok/s       = the reference (target is getting bio to ~2-3k)
# =============================================================================

import os
import random
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"
WINDOW = 64  # tokens per batched step (0 = legacy one-token path)

print("==> GPU check:")
import torch

print(
    f"    torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---- 1. clone / update ----
if not os.path.isdir(REPO_DIR):
    print("==> Cloning repo")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print("==> Pulling latest")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
os.chdir(REPO_DIR)

# ---- 2. install (slow, ~2-4 min) ----
print("==> Installing dependencies")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-build-isolation"], check=True
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", "psutil"], check=True)

# ---- 3. imports ----
from bioneural.config import BioNeuralConfig
from bioneural.eval.standard_model import StandardTransformer
from bioneural.quant import kernels
from bioneural.runtime.organism import BioNeural

print(f"    triton_kernel_path_armed={kernels.triton is not None}")

# ---- 4. synthetic token stream ----
cfg = BioNeuralConfig()
cfg.device = "cuda"
cfg.batch_window = WINDOW
rng = random.Random(0)
flat = [rng.randrange(cfg.vocab_size) for _ in range(4000)]

# ===========================================================================
# 5. BioNeural throughput
# ===========================================================================
org = BioNeural(cfg)
print(f"    BioNeural params={org.n_params():,} device={org.device}")

SEG = 64
WARM = 2
RUN = 8

_sync()
t0 = time.monotonic()
org.train_sequence(flat[:SEG])
_sync()
t_first = time.monotonic() - t0
print(f"    first seq ({SEG} tok) took {t_first:.2f}s  <- Triton JIT compile + warmup")

for _ in range(WARM - 1):
    org.train_sequence(flat[:SEG])

_sync()
t0 = time.monotonic()
for i in range(RUN):
    org.train_sequence(flat[i * SEG : (i + 1) * SEG])
_sync()
dt = time.monotonic() - t0
bio_tok = RUN * SEG
bio_tps = bio_tok / dt
print(f"    [bio] {bio_tok} tokens in {dt:.2f}s -> {bio_tps:.1f} tok/s")

# ===========================================================================
# 6. Standard transformer throughput (matched-vocab reference)
# ===========================================================================
std = StandardTransformer(
    vocab_size=cfg.vocab_size, dim=160, n_layer=2, n_head=4, max_len=128
)
print(f"    StandardTransformer params={std.n_params():,}")

BATCH = 16
SEQ = 128
std.fit(flat[:2000], seconds=2.0, batch=BATCH, seq=SEQ)  # warmup

_sync()
t0 = time.monotonic()
r = std.fit(flat, seconds=8.0, batch=BATCH, seq=SEQ)
_sync()
dt = time.monotonic() - t0
std_tps = r["steps"] * BATCH * SEQ / dt
print(f"    [std] {r['steps']} steps in {dt:.2f}s -> {std_tps:.0f} tok/s")

# ===========================================================================
# 7. summary
# ===========================================================================
print()
print("=" * 56)
print(f"  BioNeural  train tok/s: {bio_tps:10.1f}")
print(f"  Standard   train tok/s: {std_tps:10.0f}")
print(f"  ratio (bio/std)       : {bio_tps / std_tps:10.3f}")
print("=" * 56)
print("  If bio is >> 3.9 the kernel-fix worked. Target after batching: 2000-3000.")
