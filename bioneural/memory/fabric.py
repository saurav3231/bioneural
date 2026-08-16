"""MemoryFabric: owns all tiers and drives memory -> behavior coupling.

Every workspace cycle the fabric injects three influence streams:

1. **Associative priming** — M2a pattern completion biases which columns win the next spotlight.
2. **Affective bias** — recalled neuromod snapshots nudge the live bus (memory shapes mood).
3. **Prospective memory** — engrams with a future time-code fire when their moment arrives.
"""

from __future__ import annotations

from pathlib import Path

import torch

from bioneural.config import MemoryConfig
from bioneural.cortex.event_bus import EventBus
from bioneural.memory.tiers import (
    EpisodicLog,
    FastWeights,
    ProceduralReflexes,
    SemanticGraph,
    SensoryRing,
    WorkingMemory,
)


class MemoryFabric:
    def __init__(
        self, cfg: MemoryConfig, dim: int, event_bus: EventBus, spill_dir: Path | None = None
    ):
        self.cfg = cfg
        self.dim = dim
        self.m0 = SensoryRing(event_bus)
        self.m1 = WorkingMemory(cfg.m1_slots, cfg.m1_decay, dim)
        self.m2a = FastWeights(cfg.m2a_capacity, dim, cfg.m2a_sim_threshold)
        self.m2b = EpisodicLog(cfg.m2b_max_engrams, dim, spill_dir)
        self.m3 = SemanticGraph(cfg.m3_max_concepts, dim)
        self.m4 = ProceduralReflexes()

    # ------------------------------------------------------------------
    def write_experience(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mod: dict[str, float] | None = None,
        meta: str = "",
        episodic: bool = True,
    ) -> None:
        """Write one experience across M1 (working), M2a (one-shot) and M2b (life log).

        `episodic=False` updates working/one-shot memory but skips the life-log (novelty gating).
        """
        self.m1.write(key, value, aCh=(mod or {}).get("ACh", 0.5))
        self.m2a.write(key, value, mod=(mod or {}).get("DA", 0.5))
        if episodic:
            self.m2b.add(key, mod, meta=meta)
        self.m1.tick()

    def recall(self, key: torch.Tensor) -> dict:
        """Return associative priming + affective bias + episodic hits for `key`."""
        primed = self.m2a.recall(key)  # content-addressable pattern completion
        episodes = self.m2b.query(key, self.cfg.m2b_k)
        affect = self._affective_bias(episodes)
        return {"primed": primed, "episodes": episodes, "affect": affect}

    def _affective_bias(self, episodes) -> dict[str, float]:
        if not episodes:
            return {"DA": 0.5, "NE": 0.5, "ACh": 0.5, "5HT": 0.5}
        agg = {}
        for k in ("DA", "NE", "ACh", "5HT"):
            vals = [e.mod_snapshot.get(k, 0.5) for e in episodes]
            agg[k] = sum(vals) / len(vals)
        return agg

    def prospective_due(self, now: float, window: float = 5.0) -> list:
        """Engrams carrying a future time-code that are due 'now'."""
        due = []
        for e in self.m2b.engrams:
            future = e.mod_snapshot.get("future_time")
            if future is not None and 0 <= future - now <= window:
                due.append(e)
        return due

    def remember_latest(self, k: int = 8) -> list:
        return self.m2b.latest(k)

    def stats(self) -> dict[str, float]:
        return {
            "m1_slots": float(len(self.m1)),
            "m2a_occupancy": self.m2a.occupancy(),
            "m2a_hits": float(self.m2a.hits),
            "m2a_writes": float(self.m2a.writes),
            "m2b_engrams": float(len(self.m2b)),
            "m2b_spilled": float(self.m2b.spilled),
            "m3_concepts": float(len(self.m3)),
        }
