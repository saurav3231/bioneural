"""Surprise (NE) tracking.

The NE neuromodulator is literally the network-wide mean prediction error. Big surprise
-> learning rates spike -> the model updates hardest exactly when reality violates its model.
Energy efficiency follows too: well-predicted input produces few error events.
"""

from __future__ import annotations


class SurpriseTracker:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.value = 0.0
        self.history: list[float] = []

    def update(self, surprise: float) -> float:
        self.value = (1 - self.alpha) * self.value + self.alpha * surprise
        self.history.append(surprise)
        return self.value

    def recent_mean(self, window: int = 256) -> float:
        h = self.history[-window:]
        return sum(h) / max(len(h), 1)

    def reset(self) -> None:
        self.value = 0.0
        self.history.clear()
