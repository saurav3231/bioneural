# =============================================================================
# BioNeural — QUALITY CHECK (windowed vs legacy, SAME token budget)
# =============================================================================
# ruff: noqa: E402  (imports must follow the clone/install steps in a Kaggle cell)
# The real question is whether the batched path learns as well per token as the
# legacy path. All configs train the SAME number of tokens on the same text and
# are evaluated identically; ppl ~ vocab (=1024) means NO learning yet, so if a
# row's ppl is far above the others that config's sample-efficiency regressed.
# Legacy is ~150x slower, so it sets the budget (bump TRAIN_TOK up to ~5000 if
# you want to see ppl drop below random, but legacy will take longer).
# =============================================================================

import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

TRAIN_TOK = 2400  # SAME budget for every config (legacy needs ~3.5 min at ~11 tok/s)
EVAL_TOK = 200  # held-out tokens evaluated (legacy eval is per-token; keep small)
WINDOWS = [0, 64, 128, 256]

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


def run(batch_window: int) -> dict:
    cfg = BioNeuralConfig()
    cfg.device = "cuda"
    cfg.vocab_size = cfg_vocab
    cfg.batch_window = batch_window
    org = BioNeural(cfg)
    seg = flat_train[:TRAIN_TOK]
    _sync()
    t0 = time.monotonic()
    org.train_sequence(seg)
    _sync()
    dt = time.monotonic() - t0
    tps = (TRAIN_TOK - 1) / dt
    ev = org.evaluate(flat_val)
    return {
        "window": batch_window,
        "tok_s": tps,
        "ppl": ev["ppl"],
        "acc": ev["acc"],
        "nll": ev["nll"],
    }


rows = [run(w) for w in WINDOWS]
legacy = rows[0]

print()
print("=" * 66)
print(f"  {'window':>6} {'tok/s':>9} {'ppl':>9} {'top1':>6} {'ppl/legacy':>10}")
print("=" * 66)
for r in rows:
    ratio = r["ppl"] / max(legacy["ppl"], 1e-9)
    print(
        f"  {r['window']:>6} {r['tok_s']:>9.0f} {r['ppl']:>9.2f} "
        f"{r['acc']:>6.3f} {ratio:>10.3f}"
    )
print("=" * 66)
print("  ppl/legacy = 1.0 means the batched path learns AS WELL as legacy per token.")
print("  >1.5 -> the packet approximation is hurting sample-efficiency; reduce the window.")
