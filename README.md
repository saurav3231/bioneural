# BioNeural

> A living, event-driven, quantized, backprop-free neural organism designed to run 24/7 on a single
> GPU. **Intelligence per joule, not intelligence per parameter.**

This repository is **v1** of the architecture concept: a real, runnable implementation of a
ternary-quantized, event-driven cortical model with local (backprop-free) learning, a five-tier
memory fabric, a global workspace, a drive engine, a neuromodulator bus, a time-clock bank, and a
sleep-consolidation daemon — plus an **ultra-detailed benchmark harness** that pits it against a
standard Transformer under identical conditions (matched parameters, matched wall-clock budget,
matched dataset) and reports every field: quality, speed, energy, memory, liveness, stability,
continuity.

> **Honest framing.** v1 is a *measurement instrument*, not a frontier LM. The whole point is the
> experiment the README of the concept promised: build it, benchmark it head-to-head against a
> standard model on the same conditions, and find out whether the "capacity without activity" trick
> holds any water. If it loses, we learn exactly where and why — every dimension is instrumented.

---

## 1. Quick start

> **Where do I run the heavy stuff? On Kaggle, not on a laptop.** The benchmark is built for the
> Kaggle T4 (GPU + Triton + pynvml power metering). Everything below marked `(optional)` runs on
> a CPU laptop but is slow there — use `--quick` for that.

```bash
# install once (any machine)
pip install -e ".[dev]"

# (optional) 2-minute offline self-test on a laptop
python scripts/smoke.py --quick

# (optional) tiny sanity benchmark on a laptop (~1-2 min)
python scripts/bench.py --quick

# full benchmark (run on Kaggle T4, NOT a laptop):
#   paste kaggle/benchmark_cell.py into a T4 notebook — see section 3.
```

## 2. One-click push to GitHub

`push.ps1` (Windows) / `push.sh` (Linux/macOS) create the repo on GitHub and push the entire
project, then print the URL. Requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be
installed and authenticated (`gh auth login`).

```powershell
# from repo root
.\push.ps1 bioneural public
# or just .\push.ps1   (defaults: repo name = folder name, private)
```

## 3. The benchmark (Kaggle, single cell)

Open `kaggle/benchmark_cell.py` — that exact code is one cell. It clones your repo, installs
deps, runs the full head-to-head benchmark on the T4 GPU, and writes a detailed report
(`results/<run>/report.md`, `report.json`, `*.png`).

After you push the repo, just paste the cell into a Kaggle notebook (GPU T4x2), change
`REPO_URL`, and run.

```python
# quick edit at top of the cell
REPO_URL = "https://github.com/<you>/bioneural.git"   # ← set this
MINUTES  = 15                                          # mid-time benchmark
DATASET  = "tiny-stories"
```

## 4. What v1 actually implements

| Concept | Status in v1 | Where |
|---|---|---|
| Ternary weights `{-1,0,+1}` + per-group INT8 scales + latent shadows | ✅ | `bioneural/quant/` |
| Triton ternary matmul + sparse event-gather kernels (GPU) with torch fallback (CPU) | ✅ | `bioneural/quant/kernels.py` |
| Quantized Event Units: INT16 membrane, adaptive thresholds, soft reset, eligibility traces | ✅ | `bioneural/cortex/qeu.py` |
| Cortical columns with local recurrence + k-WTA inhibition + column-batched execution | ✅ | `bioneural/cortex/column.py` |
| Event bus (AER tuples, GPU ring buffer = M0 sensory ring) | ✅ | `bioneural/cortex/event_bus.py` |
| Event-driven linear-attention backbone (long-range context) | ✅ | `bioneural/cortex/backbone.py` |
| Predictive coding (local targets, free surprise/NE signal) | ✅ | `bioneural/learning/predictive.py` |
| Three-factor Hebbian + eligibility traces, gated by neuromodulators | ✅ | `bioneural/learning/hebbian.py` |
| Readout heads via contrastive/forward-forward local rule (no backprop) | ✅ | `bioneural/learning/readout.py` |
| Homeostatic thresholds + synaptic scaling | ✅ | `bioneural/learning/homeostat.py` |
| SDC sparse-distributed codes | ✅ | `bioneural/memory/codes.py` |
| Memory Fabric: M0 ring, M1 working slots, M2a one-shot fast weights, M2b episodic log, M3 semantic graph, M4 procedural | ✅ | `bioneural/memory/` |
| Global Workspace loop (compete → broadcast → elaborate → decide → learn) | ✅ | `bioneural/workspace/` |
| Drive engine (curiosity/social/coherence/competence/energy → self-initiated acts) | ✅ | `bioneural/drives/` |
| Neuromodulator bus (DA/NE/ACh/5HT) | ✅ | `bioneural/neuromod/` |
| Oscillator clock bank + time cells + closed-form async decay | ✅ | `bioneural/time/` |
| Sleep consolidation daemon (replay, M3 distillation, downscaling) | ✅ | `bioneural/runtime/sleep.py` |
| Checkpointing = "the body" (continuity across restarts) | ✅ | `bioneural/runtime/checkpoint.py` |
| Tokenizer (BPE via HF `tokenizers`, char fallback) | ✅ | `bioneural/io/` |
| Spike code conversion (token → event bursts) | ✅ | `bioneural/io/spikes.py` |
| Dataset loader (TinyStories / WikiText-2 / offline synthetic fallback) | ✅ | `bioneural/data/loader.py` |
| Standard Transformer baseline, matched-params + matched-time | ✅ | `bioneural/eval/standard_model.py` |
| Ultra-detailed benchmark + metrics across 7 domains | ✅ | `bioneural/eval/` |

## 5. Benchmark domains measured

1. **Quality** — train/val loss, perplexity, top-1/top-5 accuracy, BLEU-2, distinct-2, samples.
2. **Speed / efficiency** — tokens/s (train & generate), events/s, % neurons active, FLOPs/token,
   GPU util, throughput per watt.
3. **Energy** — watts (pynvml), joules/token, **intelligence-per-joule** proxy for both models.
4. **Memory** — one-shot fact retention (tell once, recall after N steps), retrieval latency,
   engram count, hit-rate, temporal-query accuracy.
5. **Liveness / autonomy** — idle duty cycle, drive levels, self-initiated acts.
6. **Stability** — firing-rate drift, ternary flip rate, output-entropy over time.
7. **Continuity** — checkpoint round-trip integrity, offline-time perception.

Every run writes `report.json` (machine-readable) + `report.md` + plots to `results/<run>/`.

## 6. Project layout

```
bioneural/
  quant/        ternary quantization + Triton kernels
  cortex/       QEU neurons, columns, event bus, SSM backbone
  learning/     predictive coding, 3-factor Hebbian, readout, homeostasis
  memory/       SDC codes + M0–M4 tiers + MemoryFabric
  workspace/    global workspace loop
  drives/       homeostat
  neuromod/     DA/NE/ACh/5HT bus
  time/         clock bank + time cells
  runtime/      organism glue, sleep daemon, checkpoint
  io/           tokenizer, spike codes
  data/         dataset loader
  eval/         standard baseline + metrics + benchmark harness
scripts/        train / bench / smoke CLIs
kaggle/         the single-cell benchmark
tests/          pytest smoke tests
configs/        yaml configs
```

## 7. Contributing / status

Experimental research code, **v0.1-alpha**. See `#` risk register in the design docs. Open issues
freely; the benchmark harness is the oracle — any change to the substrate should show up in
`report.json`.

License: MIT.
