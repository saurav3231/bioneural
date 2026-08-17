# =============================================================================
# BioNeural — SPEED SWEEP · paste this whole file into ONE Kaggle cell
# =============================================================================
# Clones the repo, installs deps, then measures TRAIN tok/s across several
# (window, spike_ticks) configs on synthetic tokens (no dataset download).
# Pick the config that clears 10k tok/s and use it in learning_cell/benchmark.
# =============================================================================
# ruff: noqa: E402

import os
import random
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

CONFIGS = [  # (label, window, spike_ticks)
    ("w256_t3", 256, 3),
    ("w256_t2", 256, 2),
    ("w384_t3", 384, 3),
    ("w384_t2", 384, 2),
    ("w512_t3", 512, 3),
]
TOK_BUDGET = 400_000  # tokens per config (steady-state measurement)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

# ---- 2. install ----
print("==> Installing dependencies")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-build-isolation"], check=True
)

# ---- 3. imports ----
from bioneural.config import BioNeuralConfig
from bioneural.runtime.organism import BioNeural

rng = random.Random(0)
flat = [rng.randrange(1024) for _ in range(100_000)]

print()
print("=" * 64)
print(f"  {'config':>10} {'tok/s':>8}   ('first' = Triton JIT compile, one-time)")
print("=" * 64)
for label, window, ticks in CONFIGS:
    cfg = BioNeuralConfig()
    cfg.device = "cuda"
    cfg.vocab_size = 1024
    cfg.batch_window = window
    cfg.spike_ticks = ticks
    org = BioNeural(cfg)

    _sync()
    t0 = time.monotonic()
    org.train_sequence(flat[:1000])  # first call = JIT compile + warmup
    _sync()
    t_first = time.monotonic() - t0

    _sync()
    t0 = time.monotonic()
    n = 0
    i = 0
    while n < TOK_BUDGET:
        seg = flat[i : i + 100_000]
        org.train_sequence(seg)
        n += len(seg) - 1
        i = (i + len(seg)) % len(flat)
    _sync()
    dt = time.monotonic() - t0
    print(f"  {label:>10} {n / dt:>7.0f}    first={t_first:.2f}s", flush=True)

print("=" * 64)
print("  pick the fastest config that also keeps ppl dropping (check with learning_cell).")
