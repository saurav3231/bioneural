# =============================================================================
# BioNeural — ctx ABLATION diagnostic (3-5 min) + peak-ppl report
# =============================================================================
# Trains at batch_window=384 for MINUTES minutes, then prints:
#   * the best ppl/top1 seen (the ~103 peak, not the overfit 15-min number)
#   * full ctx  : ppl / acc        (what the model actually reports)
#   * emb only  : ppl / acc        (embedding anchor alone — the bigram floor)
#   * cortex    : ppl / acc        (backbone/cortex alone)  [EMBSSM=False]
#   * ssm       : ppl / acc        (EmbSSM alone)           [EMBSSM=True]
# Decision rule:
#   emb ≈ full   -> cortex adds nothing; invest in embeddings/bigram
#   cortex > rnd -> strengthen cortex (task-aligned backbone)
#   ssm ≪ cortex -> EmbSSM's trained state carries the task signal
# =============================================================================
# ruff: noqa: E402

import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/saurav3231/bioneural.git"
REPO_DIR = "/kaggle/working/bioneural"

WINDOW = 384
TICKS = 1
MINUTES = 5        # 3 min hits the ~103 ppl peak; 5 for more stability
EVAL_TOK = 512
EVAL_EVERY = 250_000
EMBSSM = True     # train the EmbSSM readout path (continuous SSM over embeddings)
EMBSSM_HIDDEN = 0  # 0 = linear state; >0 = frozen random-feature (ELM) map before the SSM head
QSTATE = True    # swap the linear EmbSSM for the quantum-inspired complex QState channel (sign-fixed)
QSTATE_PAIRS = 16  # QState entanglement features (pairwise Re(h_i conj(h_j)) on the first N amplitudes)
QSTATE_LEARN_R = True  # train the QState qubit angles (θ, φ) so R adapts to the task (verified vs autograd)
QSTATE_SCALES = ()  # single QState decay; multi-scale would be (0.5, 0.9, 0.99)
QSTATE_TAPS = 3  # readout taps: concat [Re, Im] of the last K state rows (recent-token decode)
QSTATE_HIDDEN = 1024  # >0: frozen sparse random-feature (ELM) map on the QState features before the head
QSTATE_COMPILE = True  # torch.compile the QState hot kernels (CUDA only; eager fallback)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("==> GPU check:")
import torch

print(
    f"    torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)
torch.manual_seed(0)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if not os.path.isdir(REPO_DIR):
    print("==> Cloning repo")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print("==> Pulling latest")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
os.chdir(REPO_DIR)

print("==> Installing dependencies")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-build-isolation"], check=True
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", "psutil"], check=True)

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
print(f"    tokenizer vocab={cfg_vocab} train_tokens={n} val_tokens={len(flat_val)}")

cfg = BioNeuralConfig()
cfg.device = "cuda"
cfg.vocab_size = cfg_vocab
cfg.batch_window = WINDOW
cfg.spike_ticks = TICKS
cfg.profile = False
cfg.embssm_readout = EMBSSM
cfg.embssm_qstate = QSTATE
cfg.embssm_qpairs = QSTATE_PAIRS
cfg.embssm_qlearn = QSTATE_LEARN_R
cfg.embssm_qdecays = QSTATE_SCALES
cfg.embssm_qtaps = QSTATE_TAPS
cfg.embssm_qhidden = QSTATE_HIDDEN
cfg.embssm_qcompile = QSTATE_COMPILE
if QSTATE:
    cfg.embssm_qdim = cfg.cortex.readout_dim // 2
org = BioNeural(cfg)

budget = MINUTES * 60.0
t0 = time.monotonic()
i = 0
last_eval = 0
best = None
print()
while time.monotonic() - t0 < budget:
    seg = flat_train[i : i + 100_000]
    org.train_sequence(seg)
    i = (i + len(seg)) % max(n, 1)
    torch.cuda.empty_cache()
    if org.total_tokens - last_eval >= EVAL_EVERY:
        last_eval = org.total_tokens
        _sync()
        ev = org.evaluate_window(flat_val, window=WINDOW)
        if best is None or ev["ppl"] < best["ppl"]:
            best = dict(ppl=ev["ppl"], acc=ev["acc"], t=time.monotonic() - t0)
        print(
            f"    {time.monotonic() - t0:>5.0f}s {org.total_tokens:>9} tok "
            f"ppl {ev['ppl']:>7.2f} top1 {ev['acc']:.3f}  "
            f"emb {ev['ppl_emb']:>6.2f} ssm {ev['ppl_ssm']:>6.2f}",
            flush=True,
        )

_sync()
ev = org.evaluate_window(flat_val, window=WINDOW)
print()
print("=" * 72)
print("  ctx ablation (eval-only probes of the trained head):")
print(f"    full ctx : ppl {ev['ppl']:>8.2f}  top1 {ev['acc']:.4f}  nll {ev['nll']:.3f}")
print(f"    emb only : ppl {ev['ppl_emb']:>8.2f}  top1 {ev['acc_emb']:.4f}")
if EMBSSM:
    chan = "QState (complex unitary)" if QSTATE else "EmbSSM (linear)"
    print(f"    ssm      : ppl {ev['ppl_ssm']:>8.2f}  top1 {ev['acc_ssm']:.4f}   [{chan}, sign-fixed]")
    print("    baseline v1 full ~104 / emb ~104;  cortex dead ~1034")
else:
    print(f"    cortex   : ppl {ev['ppl_noemb']:>8.2f}  top1 {ev['acc_noemb']:.4f}")
    print("    BPE random floor ~= 1024  (cortex ~1024 = dead)")
if best is not None:
    print(f"    best seen: ppl {best['ppl']:.2f} top1 {best['acc']:.3f} @ {best['t']:.0f}s")
print("=" * 72)
