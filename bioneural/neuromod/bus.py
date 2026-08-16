"""Neuromodulator bus: four global scalars gating plasticity, gain and thresholds.

DA (reward), NE (surprise), ACh (focus/novelty), 5HT (mood/conservatism). Single source of
truth for the learning gate and the workspace's "urgency" signal.
"""

from __future__ import annotations

from dataclasses import dataclass

MOD_KEYS = ("DA", "NE", "ACh", "5HT")


@dataclass
class ModState:
    DA: float = 0.5
    NE: float = 0.5
    ACh: float = 0.5
    SHT: float = 0.5  # 5HT (serotonin analog) — valid identifier, exported as "5HT"

    def as_dict(self) -> dict[str, float]:
        return {"DA": self.DA, "NE": self.NE, "ACh": self.ACh, "5HT": self.SHT}

    def tuple(self) -> tuple[float, float, float, float]:
        return (self.DA, self.NE, self.ACh, self.SHT)

    def __getitem__(self, key: str) -> float:
        return self.as_dict()[key]

    def __setitem__(self, key: str, value: float) -> None:
        if key == "5HT":
            self.SHT = value
        elif key in MOD_KEYS:
            setattr(self, key, value)


class NeuromodBus:
    """Broadcasts the modulator state; provides updates from physiological signals."""

    def __init__(self):
        self.mod = ModState()
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    def broadcast(self) -> dict[str, float]:
        return self.mod.as_dict()

    def gate(self, strength: float = 1.0) -> float:
        """Plasticity gate M(t) in [0,1] — see learning.hebbian.gate_plasticity."""
        m = self.mod.as_dict()
        return float(
            max(
                0.0,
                min(
                    1.0,
                    (0.4 * m["DA"] + 0.3 * m["NE"] + 0.3 * m["ACh"])
                    * (1.0 - 0.5 * (1.0 - m["5HT"])),
                ),
            )
            * strength
        )

    # ------------------------------------------------------------------
    def from_signals(
        self,
        surprise: float,
        reward: float | None = None,
        novelty: float | None = None,
        stability: float = 1.0,
    ) -> None:
        m = self.mod.as_dict()
        ne = float(min(1.0, max(0.0, surprise)))
        da = m["DA"] * 0.9 + (
            float(max(0.0, min(1.0, reward))) * 0.1 if reward is not None else 0.0
        )
        ach = m["ACh"] * 0.9 + (
            float(max(0.0, min(1.0, novelty))) * 0.1 if novelty is not None else 0.0
        )
        sht = m["5HT"] * 0.9 + float(max(0.0, min(1.0, 1.0 - stability))) * 0.1
        self.mod.DA, self.mod.NE, self.mod.ACh, self.mod.SHT = da, ne, ach, sht
        self.history.append(self.mod.as_dict())

    def push(self, key: str, value: float) -> None:
        self.mod[key] = float(max(0.0, min(1.0, value)))

    def reset(self) -> None:
        self.mod = ModState()
        self.history.clear()

    def recent_mean(self, key: str, window: int = 256) -> float:
        h = [m[key] for m in self.history[-window:]]
        return sum(h) / max(len(h), 1)
