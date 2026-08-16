# =============================================================================
# BioNeural — PROFILER CELL · paste into ONE Kaggle cell (GPU T4)
# =============================================================================
# Runs a few tokens under torch.profiler and prints the top CUDA-time and
# CPU-time operations. Answer the question: where do the ~120ms/token go?

import os
import random
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", "psutil"], check=True)

# ---- 3. imports ----
from bioneural.config import BioNeuralConfig
from bioneural.runtime.organism import BioNeural

# ---- 4. build model + warmup (compiles triton, warms caches) ----
cfg = BioNeuralConfig()
cfg.device = "cuda"
rng = random.Random(0)
flat = [rng.randrange(cfg.vocab_size) for _ in range(1024)]

org = BioNeural(cfg)
print(f"    BioNeural params={org.n_params():,} device={org.device}")

for _ in range(2):
    org.train_sequence(flat[:32])
_sync()

# ---- 5. profile steady-state ----
from torch.profiler import ProfilerActivity, profile

SEG = 32
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    t0 = time.monotonic()
    for i in range(4):
        org.train_sequence(flat[i * SEG : (i + 1) * SEG])
    _sync()
dt = time.monotonic() - t0
print(f"\n    profiled {4 * SEG} tokens in {dt:.2f}s -> {4 * SEG / dt:.1f} tok/s\n")

total = prof.key_averages().total_average()


def _field(e, *names, default=0.0):
    for n in names:
        if hasattr(e, n):
            return getattr(e, n)
    return default


cu = _field(total, "self_cuda_time_total", "self_device_time_total", "cuda_time_total", "device_time_total")
cpu = _field(total, "self_cpu_time_total", "self_cpu_time_total", "cpu_time_total")
print(f"    self CUDA time total: {cu:.1f} ms  (of ~{dt * 1000:.0f} ms wall)")
print(f"    CPU time total       : {cpu:.1f} ms\n")


def _table(prof, *sort_keys, row_limit=18):
    for k in sort_keys:
        try:
            return prof.key_averages().table(sort_by=k, row_limit=row_limit)
        except Exception:
            continue
    return "(no sort key available for this torch version)"


print("=== TOP CUDA-TIME OPS ===")
print(_table(prof, "self_cuda_time_total", "cuda_time_total", "self_device_time_total"))
print("=== TOP CPU-TIME OPS ===")
print(_table(prof, "self_cpu_time_total", "cpu_time_total"))