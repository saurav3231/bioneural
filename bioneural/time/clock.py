"""Time perception: oscillator clock bank + time cells.

The master clock is an *input*, not a framework detail. A bank of oscillator neurons
(fixed frequencies, sine/cos pairs) is a positional encoding over *wall-clock life*. Phases are
computed in closed form from elapsed real time, so 4 idle hours cost one update, not 4 hours of
ticks. Binarized phases form the time-part of memory SDCs -> native temporal queries.
"""

from __future__ import annotations

import math
import time

import torch

from bioneural.config import TimeConfig
from bioneural.memory.codes import make_sdc


class ClockBank:
    def __init__(self, cfg: TimeConfig | None = None):
        self.cfg = cfg or TimeConfig()
        self.t0 = time.time()  # organism birth (t=0 for subjective time)
        # log-spaced frequencies from base_hz down to max_hz (one cycle / day)
        self.freqs = torch.logspace(
            math.log10(self.cfg.base_hz),
            math.log10(self.cfg.max_hz),
            self.cfg.n_freq,
        )

    def subjective_now(self) -> float:
        return time.time() - self.t0

    def phase_at(self, t: float) -> torch.Tensor:
        """(2*n_freq,) sine/cos phase vector at subjective time t."""
        theta = 2.0 * math.pi * self.freqs * t
        return torch.cat([torch.sin(theta), torch.cos(theta)])

    def time_code(self, t: float, active_frac: float = 0.3) -> torch.Tensor:
        """Binarized time code — SDC for temporal keys/queries."""
        return make_sdc(self.phase_at(t), active_frac=active_frac, ternary=True)

    def perceive_elapsed(self, start: float, end: float) -> str:
        dt = end - start
        if dt < 60:
            return f"{dt:.0f}s"
        if dt < 3600:
            return f"{dt / 60:.1f}m"
        if dt < 86400:
            return f"{dt / 3600:.1f}h"
        return f"{dt / 86400:.1f}d"

    def phase_similarity(self, a: float, b: float) -> float:
        va = self.phase_at(a)
        vb = self.phase_at(b)
        return float((va * vb).sum() / (va.norm() * vb.norm() + 1e-9))

    def is_circadian_night(self, t: float | None = None) -> bool:
        """Very cheap circadian heuristic: deep night when the slow oscillator is low."""
        t = self.subjective_now() if t is None else t
        day_osc = math.sin(2.0 * math.pi * t / 86400.0)
        return day_osc < -0.5
