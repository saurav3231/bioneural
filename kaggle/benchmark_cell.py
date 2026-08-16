# =============================================================================
# BioNeural — KAGGLE BENCHMARK · PASTE THIS WHOLE FILE INTO ONE CELL
# =============================================================================
# 1) After you run `.\push.ps1 bioneural public`, set REPO_URL to your repo.
# 2) Paste this cell into a Kaggle notebook with GPU T4x2 accelerator.
# 3) Run. It clones, installs, trains both models for `MINUTES` and writes a
#    detailed report to /kaggle/working/results/<run>/ (report.md, report.json,
#    plots.png).
# =============================================================================

import glob
import json
import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/YOUR_USERNAME/bioneural.git"  # <-- SET THIS
REPO_DIR = "/kaggle/working/bioneural"
MINUTES = 15  # mid-time benchmark budget per model (matched wall-clock)
DATASET = "tiny-stories"  # tiny-stories | wikitext-2 | synthetic
MAX_EXAMPLES = 2000
RESULTS = "/kaggle/working/results"

print("==> GPU check:")
try:
    import torch

    print(
        f"    torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
except Exception as e:
    print(f"    torch import failed: {e}")

# ---- 1. clone ----
if not os.path.isdir(REPO_DIR):
    print(f"==> Cloning {REPO_URL}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print(f"==> Repo already present at {REPO_DIR}, pulling latest")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)

# ---- 2. install ----
print("==> Installing dependencies (this is the slow step, ~2-4 min)")
# torch on the T4 image already bundles Triton — install core deps + power metering only
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-build-isolation"], check=True
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", "psutil"], check=True)
try:
    import triton

    print(f"    triton={triton.__version__} (GPU kernels ENABLED)")
except Exception:
    print("    triton unavailable — falling back to torch reference path")

# ---- 3. run the benchmark ----
print(f"==> Running benchmark: {MINUTES} min/model, dataset={DATASET}, examples={MAX_EXAMPLES}")
cmd = [
    sys.executable,
    "-m",
    "scripts.bench",
    "--minutes",
    str(MINUTES),
    "--dataset",
    DATASET,
    "--max-examples",
    str(MAX_EXAMPLES),
    "--results",
    RESULTS,
]
t0 = time.time()
subprocess.run(cmd, check=True)
print(f"==> Benchmark finished in {(time.time() - t0) / 60:.1f} minutes")

# ---- 4. show the report ----
runs = sorted(glob.glob(os.path.join(RESULTS, "run_*")))
latest = runs[-1]
report = json.load(open(os.path.join(latest, "report.json")))
h = report["head2head"]
print("\n" + "=" * 64)
print("  HEAD-TO-HEAD  (matched params, matched wall-clock, matched data)")
print("=" * 64)
print(f"  Dataset:        {report['dataset']}   Device: {report['device']}")
print(
    f"  Params:         BioNeural={report['bioneural']['params']:,}  "
    f"Standard={report['standard']['params']:,}"
)
print(f"  Perplexity:     BioNeural={h['ppl_bio']:.3f}  Standard={h['ppl_std']:.3f}")
print(f"  Top-1 acc:      BioNeural={h['acc_bio']:.4f}  Standard={h['acc_std']:.4f}")
print(f"  Tokens/s:       BioNeural={h['tokps_bio']:.1f}  Standard={h['tokps_std']:.1f}")
print(f"  I/J (acc·ppl^-1):BioNeural={h['ij_bio']:.4f}  Standard={h['ij_std']:.4f}")
print("=" * 64)
b = report["bioneural"]
print(
    f"  BioNeural extras: active_cols/tick={b['col_stats']['active_cols_frac']:.3f}, "
    f"retention@32={b['memory']['scores'].get('32', b['memory']['scores'].get(32, 0)):.3f}, "
    f"idle_duty={b['liveness']['idle']['duty_cycle']:.5f}, "
    f"acts={b['liveness']['autonomy']['n_acts']}"
)
print(f"  Report saved to: {latest}")
print("  -> report.md  (human readable)")
print("  -> report.json (machine readable)")
print("  -> plots.png   (learning curves, retention, quality)")
print("=" * 64)
print(
    "If you find this useful, run a longer budget (60-120 min) and a second "
    "dataset for the full picture."
)
