# =============================================================================
# BioNeural — LEARNING CURVE over WALL-CLOCK (multi-epoch, like the benchmark)
# =============================================================================
# The corpus is only ~87k tokens, so a single pass can't move BPE-1024 ppl. The
# benchmark trains for MINUTES against a wall clock, looping over the corpus over
# and over (the std transformer gets ~19M token-passes in 15 min). This cell does
# the same for BioNeural at batch_window=256 (~5k tok/s) and plots ppl/acc vs time.
# =============================================================================
# ruff: noqa: E402

import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

WINDOW = 384  # best measured config: >10k tok/s with spike_ticks=2
TICKS = 2
MINUTES = 15             # wall-clock training budget (loop the corpus until this hits)
EVAL_TOK = 512
EVAL_EVERY = 250_000     # report a row every 250k token-passes
PROFILE = True           # per-phase window breakdown (columns/backbone/head/sdc-mem/rest)

# allocator hint from the earlier OOM (must be set before torch imports)
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", "psutil"], check=True)

# ---- 3. corpus + tokenizer ----
from bioneural.config import BioNeuralConfig
from bioneural.data.loader import load_dataset
from bioneural.io.tokenizer import build_tokenizer
from bioneural.runtime.organism import BioNeural

texts = load_dataset("tiny-stories", max_examples=400, seed=0)
train_texts, val_texts = texts[:340], texts[340:]
tok = build_tokenizer(train_texts[:200], 1024)
cfg_vocab = tok.vocab_size if hasattr(tok, "vocab_size") else tok.vocab
flat_train = [x for t in train_texts for x in tok.encode(t)]
flat_val = [x for t in val_texts for x in tok.encode(t)][:EVAL_TOK]
n = len(flat_train)
print(
    f"    tokenizer vocab={cfg_vocab} train_tokens={n} val_tokens={len(flat_val)} "
    f"(will loop the corpus ~{int(MINUTES * 60 * 5000 / max(n, 1))}x)"
)

# ---- 4. train against the wall clock, looping over the corpus ----
cfg = BioNeuralConfig()
cfg.device = "cuda"
cfg.vocab_size = cfg_vocab
cfg.batch_window = WINDOW
cfg.spike_ticks = TICKS
cfg.profile = PROFILE
org = BioNeural(cfg)

budget = MINUTES * 60.0
t0 = time.monotonic()
i = 0
last_eval = 0
print()
print("=" * 88)
print(f"  {'sec':>6} {'tokens':>9} {'tok/s':>7} {'ppl':>9} {'top1':>6} {'nll':>8} {'mem_gb':>7}")
print("=" * 88)
while time.monotonic() - t0 < budget:
    seg = flat_train[i : i + 100_000]
    org.train_sequence(seg)
    i = (i + len(seg)) % max(n, 1)
    torch.cuda.empty_cache()
    if org.total_tokens - last_eval >= EVAL_EVERY:
        last_eval = org.total_tokens
        _sync()
        elapsed = time.monotonic() - t0
        ev = org.evaluate_window(flat_val, window=WINDOW)
        mem = torch.cuda.memory_allocated() / 1e9
        print(
            f"  {elapsed:>6.0f} {org.total_tokens:>9} "
            f"{org.total_tokens / max(elapsed, 1e-9):>7.0f} "
            f"{ev['ppl']:>9.2f} {ev['acc']:>6.3f} {ev['nll']:>8.3f} {mem:>7.2f}",
            flush=True,
        )
elapsed = time.monotonic() - t0
ev = org.evaluate_window(flat_val, window=WINDOW)
mem = torch.cuda.memory_allocated() / 1e9
print(
    f"  {elapsed:>6.0f} {org.total_tokens:>9} {org.total_tokens / max(elapsed, 1e-9):>7.0f} "
    f"{ev['ppl']:>9.2f} {ev['acc']:>6.3f} {ev['nll']:>8.3f} {mem:>7.2f}",
    flush=True,
)
print("=" * 88)
print("  ppl 1024 = BPE random floor; mem_gb should stay ~flat after the autograd leak fix.")
