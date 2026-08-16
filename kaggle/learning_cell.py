# =============================================================================
# BioNeural — LEARNING CURVE (quality at scale, fast batched eval)
# =============================================================================
# The 2400-token sweep showed the batched path learns AS WELL as legacy per token
# (ppl/legacy ~= 1.0) and runs at ~3.2k tok/s at window 256, but ~1024 ppl is just
# the BPE-1024 random floor. This cell trains a LOT more tokens (~3 min at 256)
# and plots ppl/acc dropping below the floor with fast batched evals.
# =============================================================================
# ruff: noqa: E402

import math
import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

WINDOW = 256            # best config from the sweep
TRAIN_TOK = 600_000     # ~3 min at ~3.2k tok/s on a T4
EVAL_TOK = 512          # fast now (batched eval)
EVAL_EVERY = 100_000    # report a row every 100k tokens

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
print(
    f"    tokenizer vocab={cfg_vocab} train_tokens={len(flat_train)} "
    f"val_tokens={len(flat_val)}"
)

# ---- 4. train + periodic eval ----
cfg = BioNeuralConfig()
cfg.device = "cuda"
cfg.vocab_size = cfg_vocab
cfg.batch_window = WINDOW
org = BioNeural(cfg)

budget = min(TRAIN_TOK, len(flat_train) - 1)
last_eval = 0
t0 = time.monotonic()
print()
print("=" * 72)
print(f"  {'tokens':>9} {'tok/s':>8} {'ppl':>9} {'top1':>6} {'nll':>8} {'notes':>14}")
print("=" * 72)
while last_eval < budget:
    seg = flat_train[last_eval : last_eval + 100_000]
    org.train_sequence(seg)
    done = min(last_eval + len(seg) - 1, budget)
    _sync()
    tps = done / max(time.monotonic() - t0, 1e-9)
    ev = org.evaluate_window(flat_val, window=WINDOW)
    note = "random floor" if ev["ppl"] > 900 else ("learning!" if ev["ppl"] < 700 else "warming up")
    print(
        f"  {done:>9} {tps:>8.0f} {ev['ppl']:>9.2f} {ev['acc']:>6.3f} "
        f"{ev['nll']:>8.3f} {note:>14}",
        flush=True,
    )
    last_eval = done + 1
print("=" * 72)
print("  ppl 1024 = BPE random floor; the faster it drops, the better the sample efficiency.")