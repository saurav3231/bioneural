"""Address-Event-Representation (AER) event bus + M0 sensory ring.

Events are `(column_id, neuron_id, t, magnitude)` tuples — literally the protocol neuromorphic
hardware uses. The bus also backs the **M0 sensory ring**: raw recent events that enable
"wait, what did I just hear?" re-processing and STDP windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    column_id: int
    neuron_id: int
    t: float
    magnitude: float = 1.0


class EventBus:
    """AER event bus with a fixed-capacity GPU-friendly ring (M0)."""

    def __init__(self, ring_size: int = 4096, name: str = "global"):
        self.name = name
        self.ring_size = ring_size
        self._ring: deque[Event] = deque(maxlen=ring_size)
        self.tick: float = 0.0
        self.total_events: int = 0
        self.tick_events: int = 0

    # ------------------------------------------------------------------
    def emit(self, column_id: int, neuron_id: int, magnitude: float = 1.0) -> None:
        self._ring.append(Event(column_id, neuron_id, self.tick, magnitude))
        self.total_events += 1
        self.tick_events += 1

    def advance(self, dt: float = 1.0) -> None:
        self.tick += dt
        self.tick_events = 0

    # ------------------------------------------------------------------
    def recent(self, n: int | None = None) -> list[Event]:
        events = list(self._ring)
        if n is not None:
            events = events[-n:]
        return events

    def count_recent(self, t_window: float) -> int:
        cutoff = self.tick - t_window
        return sum(1 for e in self._ring if e.t >= cutoff)

    def events_last_tick(self) -> list[Event]:
        cutoff = self.tick - 1.0
        return [e for e in self._ring if e.t >= cutoff]

    def clear(self) -> None:
        self._ring.clear()

    def __len__(self) -> int:
        return len(self._ring)

    def stats(self) -> dict[str, float]:
        return {
            "total_events": float(self.total_events),
            "ring_occupancy": float(len(self._ring)) / self.ring_size,
            "tick": self.tick,
        }
