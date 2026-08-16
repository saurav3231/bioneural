"""Drive engine (hypothalamus): homeostatic variables that make the organism act unprompted.

Drives live in [0,1]; action selection is drive-reduction: *reward = homeostasis*. Conversational
initiation is simply one of the organism's regulatory actions. Every self-initiated act is logged
with its cause so it can explain why it acted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from bioneural.config import DriveConfig


@dataclass
class DriveSignals:
    surprise: float = 0.5  # NE
    reward: float = 0.5  # DA
    novelty: float = 0.5  # ACh
    silence_since_user: float = 0.0
    contradiction_count: int = 0
    failure_signature: float = 0.0
    activity: float = 0.0  # sustained activity level (0..1)


class DriveEngine:
    def __init__(self, cfg: DriveConfig | None = None):
        self.cfg = cfg or DriveConfig()
        self.drives: dict[str, float] = dict(self.cfg.init)
        self.last_user_interaction = time.time()
        self.last_initiation = 0.0
        self.initiation_count = 0
        self.causes: list[dict] = []

    # ------------------------------------------------------------------
    def user_spoke(self) -> None:
        self.last_user_interaction = time.time()

    def update(self, s: DriveSignals) -> None:
        d = self.drives
        # curiosity: high in high-NE regions yet unexplored; falls as questions resolve
        d["curiosity"] = min(1.0, d["curiosity"] * 0.995 + 0.02 * max(0.0, s.surprise - 0.4))
        # social: rises with silence from the user, falls on interaction
        silence = time.time() - self.last_user_interaction
        if silence > 10:
            d["social"] = min(1.0, d["social"] + (silence / 3600.0) * 0.05)
        else:
            d["social"] = max(0.0, d["social"] - 0.2)
        # coherence: contradictions + self-prediction errors pressure sleep / clarification
        d["coherence"] = min(
            1.0, d["coherence"] * 0.99 + s.contradiction_count * 0.05 + 0.02 * s.surprise
        )
        # competence: repeated failures build pressure; success (DA) reduces it
        d["competence"] = min(
            1.0, d["competence"] * 0.995 + s.failure_signature * 0.05 - 0.1 * s.reward
        )
        # energy: sustained activity drains; rest restores
        d["energy"] = min(1.0, max(0.0, d["energy"] - 0.005 * s.activity + 0.001))

    # ------------------------------------------------------------------
    def wants_to_initiate(self) -> str | None:
        """Returns the drive name that crossed the initiate threshold, or None."""
        now = time.time()
        if now - self.last_initiation < 60:
            return None
        for name, level in sorted(self.drives.items(), key=lambda kv: -kv[1]):
            if level >= self.cfg.initiate_threshold:
                self.last_initiation = now
                self.initiation_count += 1
                return name
        return None

    def log_cause(self, drive: str) -> None:
        self.causes.append({"drive": drive, "time": time.time(), "levels": dict(self.drives)})

    def state(self) -> dict[str, float]:
        return dict(self.drives)
