"""Metrics: quality, speed, energy, memory, liveness, stability, continuity.

Energy is measured via pynvml (real GPU watts) when available; otherwise a rough system-wide
estimate from psutil (labeled `estimated`). `intelligence_per_joule` is the whole point of the
project: it is a proxy for *intelligence per joule*, not intelligence per parameter.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Callable

import torch


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------
def softmax_cross_entropy(logits: torch.Tensor, target: int) -> float:
    mx = logits.max().item()
    lse = mx + math.log(torch.exp(logits - mx).sum().item())
    return float(lse - logits[target].item())


def perplexity_from_nll(nll: float) -> float:
    return math.exp(min(nll, 20.0))


def token_accuracy(correct: int, total: int) -> float:
    return correct / max(total, 1)


def bleu_2(reference: list[int], hypothesis: list[int]) -> float:
    """BLEU with up to 2-grams + brevity penalty. Handles OOV tokens fine."""
    if not hypothesis:
        return 0.0
    ref_counts = (
        Counter(zip(reference, reference[1:], strict=False)) if len(reference) > 1 else Counter()
    )
    hyp_counts = (
        Counter(zip(hypothesis, hypothesis[1:], strict=False)) if len(hypothesis) > 1 else Counter()
    )
    unigram_prec = sum(min(1, hypothesis.count(t)) for t in set(hypothesis)) / len(hypothesis)
    if ref_counts and hyp_counts:
        bigram_hits = sum(min(v, ref_counts.get(k, 0)) for k, v in hyp_counts.items())
        bigram_prec = bigram_hits / max(sum(hyp_counts.values()), 1)
    else:
        bigram_prec = 0.0
    bp = math.exp(min(0.0, 1.0 - len(reference) / len(hypothesis)))
    return bp * (unigram_prec * bigram_prec) ** 0.5


def distinct_2(ids: list[int]) -> float:
    if len(ids) < 2:
        return 0.0
    bigrams = list(zip(ids, ids[1:], strict=False))
    return len(set(bigrams)) / max(len(bigrams), 1)


# ---------------------------------------------------------------------------
# speed / energy
# ---------------------------------------------------------------------------
class EnergyMeter:
    """Samples GPU watts (pynvml) + GPU util + memory while a function runs."""

    def __init__(self):
        self._nvml = None
        self._handle = None
        self._sampler_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list[tuple[float, float, float, float]] = []  # (t, watts, util, memGB)
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _sample(self) -> None:
        if self._nvml is None:
            return
        try:
            w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024**3)
            self._samples.append((time.time(), float(w), float(u), float(m)))
        except Exception:
            pass

    def start(self) -> None:
        self._stop.clear()
        self._samples.clear()
        self._sample()
        if self._nvml is not None:
            self._sampler_thread = threading.Thread(target=self._loop, daemon=True)
            self._sampler_thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(0.1)

    def stop(self) -> dict:
        self._sample()
        self._stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=1.0)
        if not self._samples:
            return {"watts_avg": None, "gpu_util_avg": None, "mem_gb_max": None}
        watts = [s[1] for s in self._samples if s[1] is not None]
        util = [s[2] for s in self._samples if s[2] is not None]
        mem = [s[3] for s in self._samples if s[3] is not None]
        return {
            "watts_avg": (sum(watts) / len(watts)) if watts else None,
            "gpu_util_avg": (sum(util) / len(util)) if util else None,
            "mem_gb_max": max(mem) if mem else None,
            "n_samples": float(len(self._samples)),
        }

    @staticmethod
    def cpu_watts_estimate() -> float | None:
        try:
            import psutil

            return psutil.cpu_percent(interval=0.2) / 100.0 * 45.0
        except Exception:
            return None


def timed(
    fn: Callable, meter: EnergyMeter | None = None, budget_seconds: float | None = None
) -> dict:
    """Run `fn` (optionally with a wall-clock budget); return timing + energy metrics."""
    meter = meter or EnergyMeter()
    meter.start()
    t0 = time.monotonic()
    if budget_seconds is not None:
        budget_seconds = max(budget_seconds, 1.0)
        result = fn(budget_seconds)
    else:
        result = fn()
    dt = time.monotonic() - t0
    energy = meter.stop()
    return {
        "seconds": dt,
        "result": result,
        "energy": energy,
        "joules": (energy["watts_avg"] or 0.0) * dt,
    }


def intelligence_per_joule(score: float, joules: float) -> float:
    """A proxy for intelligence-per-joule. `score` = task metric (e.g. -ppl or acc)."""
    if joules <= 0:
        return float("nan")
    return score / joules
