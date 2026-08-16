# =============================================================================
# BioNeural — QUALITY x SPEED SWEEP · paste into ONE Kaggle cell (GPU T4)
# =============================================================================
# For each batched window size, trains the real 8.2M model on REAL TinyStories
# text (falls back to synthetic if HF is unavailable) and reports throughput AND
# held-out quality together, so you can pick the window that is fastest WITHOUT
# losing ppl/acc. A legacy (window=0) row is included at reduced budget for scale.
# =============================================================================

import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

TRAIN_TOK = 2000  # tokens each config trains on
EVAL_TOK = 200  # held-out tokens evaluated (legacy eval is per-token, keep small)
LEGACY_TRAIN_TOK = 300  # legacy path is ~300x slower; small budget only
WINDOWS = [64, 128, 256]

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

# ---- 3. corpus + tokenizer (real text; synthetic fallback if HF blocked) ----
from bioneural.config import BioNeuralConfig
from bioneural.data.loader import load_dataset
from bioneural.io.tokenizer import build_tokenizer
from bioneural.runtime.organism import BioNeural

texts = load_dataset("tiny-stories", max_examples=200, seed=0)
train_texts = texts[:160]
val_texts = texts[160:]
tok = build_tokenizer(train_texts[:60], 1024)
cfg_vocab = tok.vocab_size if hasattr(tok, "vocab_size") else tok.vocab
flat_train = [x for t in train_texts for x in tok.encode(t)][:4000]
flat_val = [x for t in val_texts for x in tok.encode(t)][:EVAL_TOK]
print(f"    tokenizer vocab={cfg_vocab} train_tokens={len(flat_train)} val_tokens={len(flat_val)}")


def run(batch_window: int, train_n: int) -> dict:
    cfg = BioNeuralConfig()
    cfg.device = "cuda"
    cfg.vocab_size = cfg_vocab
    cfg.batch_window = batch_window
    org = BioNeural(cfg)
    seg = flat_train[:train_n]
    _sync()
    t0 = time.monotonic()
    org.train_sequence(seg)
    _sync()
    dt = time.monotonic() - t0
    tps = (train_n - 1) / dt
    ev = org.evaluate(flat_val)
    return {
        "window": batch_window,
        "trained": train_n,
        "tok_s": tps,
        "ppl": ev["ppl"],
        "acc": ev["acc"],
        "nll": ev["nll"],
    }


rows = []
# legacy reference (slow)
rows.append(run(0, LEGACY_TRAIN_TOK))
for w in WINDOWS:
    rows.append(run(w, TRAIN_TOK))

print()
print("=" * 66)
print(f"  {'window':>6} {'trained':>8} {'tok/s':>9} {'ppl':>8} {'top1':>6}")
print("=" * 66)
for r in rows:
    print(
        f"  {r['window']:>6} {r['trained']:>8} {r['tok_s']:>9.0f} "
        f"{r['ppl']:>8.2f} {r['acc']:>6.3f}"
    )
print("=" * 66)
batched = rows[1:]
best_ppl = min(r["ppl"] for r in batched)
pick = None
for r in batched:
    if r["ppl"] <= best_ppl * 1.05 and (pick is None or r["tok_s"] > pick["tok_s"]):
        pick = r
print(
    f"  suggested window: {pick['window']}  "
    f"(tok/s={pick['tok_s']:.0f}, ppl={pick['ppl']:.2f})"
)
print("  = largest window whose ppl stays within 5% of the best-ppl config.")