"""Memory tiers M0-M4.

M0  Sensory ring     — raw recent events, milliseconds-seconds
M1  Working memory   — active recirculation slots, seconds-minutes
M2a Fast weights     — one-shot Hebbian cache, the "today" memory
M2b Episodic log     — the "life" memory, product-quantized engrams
M3  Semantic graph   — consolidated concepts, months-lifetime
M4  Procedural       — named reflex circuits (locked column coalitions)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from bioneural.cortex.event_bus import EventBus
from bioneural.memory.codes import sdc_similarity


# ---------------------------------------------------------------------------
# M0
# ---------------------------------------------------------------------------
class SensoryRing:
    """M0: the raw event ring (thin wrapper over the AER event bus)."""

    def __init__(self, event_bus: EventBus):
        self.bus = event_bus

    def recent(self, n: int = 64) -> list:
        return self.bus.recent(n)

    def recall_window(self, t_window: float) -> list:
        return [e for e in self.bus.recent() if e.t >= self.bus.tick - t_window]


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------
@dataclass
class WorkingSlot:
    key: torch.Tensor  # SDC key
    payload: torch.Tensor  # INT8 payload (e.g. readout vector)
    age: float = 0.0
    aCh_protected: bool = False
    last_refresh: float = 0.0


class WorkingMemory:
    """M1: active-recirculation slots. Unrehearsed slots decay; ACh refresh-protects."""

    def __init__(self, n_slots: int = 32, decay: float = 0.9, dim: int = 256):
        self.n_slots = n_slots
        self.decay = decay
        self.dim = dim
        self.slots: list[WorkingSlot] = []

    def write(self, key: torch.Tensor, payload: torch.Tensor, aCh: float = 0.5, now: float | None = None) -> None:
        # refresh-protection from ACh
        protected = aCh > 0.7
        ts = now if now is not None else time.time()
        if len(self.slots) < self.n_slots:
            self.slots.append(WorkingSlot(key, payload, 0.0, protected, ts))
            return
        # evict oldest (FIFO; protection is only a soft hint, age-based tick still drops old slots)
        self.slots.pop(0)
        self.slots.append(WorkingSlot(key, payload, 0.0, protected, ts))

    def tick(self) -> None:
        for s in self.slots:
            s.age += 1
        # drop slots whose age exceeds the decay lifetime unless protected
        self.slots = [s for s in self.slots if s.aCh_protected or s.age < 40]

    def refresh(self, idx: int) -> None:
        if 0 <= idx < len(self.slots):
            self.slots[idx].age = 0.0

    def read(self, key: torch.Tensor, top: int = 3) -> list[WorkingSlot]:
        scored = sorted(self.slots, key=lambda s: sdc_similarity(key, s.key), reverse=True)
        return scored[:top]

    def __len__(self) -> int:
        return len(self.slots)


# ---------------------------------------------------------------------------
# M2a
# ---------------------------------------------------------------------------
class FastWeights:
    """M2a: one-shot modern-Hopfield / fast-weight associative memory.

    An experience is written by a single key-value store as it happens; recall is
    content-addressable pattern completion, O(active neurons). This is how a fact told
    mid-conversation is *known* three sentences later without any context window.

    Tensor ring buffer (int8 SDCs): FIFO semantics identical to the old list store, but a
    whole window writes in a few vectorized slice assignments instead of per-token appends.
    """

    def __init__(self, capacity: int = 256, dim: int = 256, threshold: float = 0.55):
        self.capacity = capacity
        self.dim = dim
        self.threshold = threshold
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.strength: torch.Tensor | None = None
        self._head = 0  # next write position
        self._len = 0  # number of stored rows
        self.hits = 0
        self.writes = 0

    def _ensure(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self.keys is None or self.keys.device != key.device:
            dev = key.device
            self.keys = torch.zeros(self.capacity, self.dim, dtype=torch.int8, device=dev)
            self.values = torch.zeros(self.capacity, self.dim, dtype=torch.int8, device=dev)
            self.strength = torch.zeros(self.capacity, dtype=torch.float32, device=dev)
            self._head = 0
            self._len = 0

    def write_batch(
        self, keys: torch.Tensor, values: torch.Tensor, mod: float = 1.0
    ) -> None:
        """Vectorized FIFO write of a `(W, dim)` window (ring buffer, O(1) eviction)."""
        n = keys.shape[0]
        if n <= 0:
            return
        self._ensure(keys, values)
        k = keys.to(torch.int8)
        v = values.to(torch.int8)
        head = self._head % self.capacity  # head can sit at capacity after an exact fill; wrap to 0
        if self._len < self.capacity:
            space = self.capacity - self._len
            first = min(n, space)
            self.keys[head : head + first] = k[:first]
            self.values[head : head + first] = v[:first]
            self.strength[head : head + first] = mod
            head += first
            self._len += first
            self.writes += first
            if first < n:
                rest = n - first
                self.keys[:rest] = k[first:]
                self.values[:rest] = v[first:]
                self.strength[:rest] = mod
                head = rest
                self.writes += rest
        else:
            room = self.capacity - head
            first = min(n, room)
            self.keys[head : head + first] = k[:first]
            self.values[head : head + first] = v[:first]
            self.strength[head : head + first] = mod
            self.writes += first
            if first < n:
                rest = n - first
                self.keys[:rest] = k[first:]
                self.values[:rest] = v[first:]
                self.strength[:rest] = mod
                head = rest
                self.writes += rest
            else:
                head = (head + first) % self.capacity
        self._head = head

    def write(self, key: torch.Tensor, value: torch.Tensor, mod: float = 1.0) -> None:
        self.write_batch(key.unsqueeze(0), value.unsqueeze(0), mod)

    def recall(self, key: torch.Tensor) -> torch.Tensor | None:
        if self.keys is None or self._len == 0:
            return None
        k = key.detach().to(self.keys.device).to(torch.int16)
        kb = self.keys.to(torch.int16)  # (capacity, dim)
        agree = (kb * k[None, :] > 0).sum(dim=1).float()  # zeros in unwritten rows -> 0 sim
        union = ((kb != 0) | (k[None, :] != 0)).sum(dim=1).float() + 1e-9
        sims = agree / union
        best, best_i = sims.max(dim=0)
        if float(best) >= self.threshold:
            self.hits += 1
            return self.values[best_i]
        return None

    def forget(self, factor: float = 0.9) -> None:
        # synaptic downscaling during sleep
        if self.strength is not None:
            self.strength = self.strength * factor

    def occupancy(self) -> float:
        return self._len / self.capacity


# ---------------------------------------------------------------------------
# M2b
# ---------------------------------------------------------------------------
@dataclass
class Engram:
    time: float
    key_bytes: bytes
    key: torch.Tensor
    mod_snapshot: dict = field(default_factory=dict)
    links: list = field(default_factory=list)
    payload_meta: str = ""


class EpisodicLog:
    """M2b: the "life" memory. Product-quantized (~bytes/engram) engram records."""

    def __init__(self, max_engrams: int = 4096, dim: int = 256, spill_dir: Path | None = None):
        self.max_engrams = max_engrams
        self.dim = dim
        self.spill_dir = spill_dir
        self.engrams: list[Engram] = []
        self.spilled = 0

    def add(
        self,
        key: torch.Tensor,
        mod_snapshot: dict | None = None,
        links: list | None = None,
        meta: str = "",
    ) -> None:
        blob = (key > 0).to(torch.uint8).cpu().numpy().tobytes()
        e = Engram(
            time=time.time(),
            key_bytes=blob,
            key=key.clone(),
            mod_snapshot=mod_snapshot or {},
            links=links or [],
            payload_meta=meta,
        )
        if len(self.engrams) >= self.max_engrams:
            if self.spill_dir is not None:
                self._spill(e)
                self.spilled += 1
            else:
                self.engrams.pop(0)
        else:
            self.engrams.append(e)

    def add_precloned(
        self,
        key: torch.Tensor,
        blob: bytes,
        mod_snapshot: dict | None = None,
        links: list | None = None,
        meta: str = "",
        now: float | None = None,
    ) -> None:
        """Same as `add` with the packed bytes pre-computed (batched fast path)."""
        e = Engram(
            time=now if now is not None else time.time(),
            key_bytes=blob,
            key=key,
            mod_snapshot=mod_snapshot or {},
            links=links or [],
            payload_meta=meta,
        )
        if len(self.engrams) >= self.max_engrams:
            if self.spill_dir is not None:
                self._spill(e)
                self.spilled += 1
            else:
                self.engrams.pop(0)
        else:
            self.engrams.append(e)

    def _spill(self, e: Engram) -> None:
        path = self.spill_dir / "episodic.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "time": e.time,
                        "key_hex": e.key_bytes.hex(),
                        "mod": e.mod_snapshot,
                        "links": e.links,
                        "meta": e.payload_meta,
                    }
                )
                + "\n"
            )

    def query(self, key: torch.Tensor, k: int = 8) -> list[Engram]:
        scored = sorted(self.engrams, key=lambda e: sdc_similarity(key, e.key), reverse=True)
        return scored[:k]

    def query_time(self, start: float, end: float) -> list[Engram]:
        return [e for e in self.engrams if start <= e.time <= end]

    def latest(self, k: int = 8) -> list[Engram]:
        return self.engrams[-k:]

    def __len__(self) -> int:
        return len(self.engrams)


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------
@dataclass
class Concept:
    proto: torch.Tensor
    count: int
    first_seen: float
    last_seen: float


class SemanticGraph:
    """M3: consolidated concept graph distilled from M2 during sleep."""

    def __init__(self, max_concepts: int = 1024, dim: int = 256, merge_sim: float = 0.7):
        self.max_concepts = max_concepts
        self.dim = dim
        self.merge_sim = merge_sim
        self.concepts: dict[str, Concept] = {}
        self.edges: dict[str, dict[str, float]] = {}  # concept -> {concept: weight}

    def add(self, key: torch.Tensor, meta: str = "") -> str:
        best_name, best_sim = self.find(key)
        if best_sim >= self.merge_sim:
            c = self.concepts[best_name]
            c.proto = (c.proto * c.count + key.float()) / (c.count + 1)
            c.count += 1
            c.last_seen = time.time()
            return best_name
        name = meta or f"c{len(self.concepts)}"
        if len(self.concepts) >= self.max_concepts:
            # evict least-used
            name0 = min(self.concepts, key=lambda k: self.concepts[k].count)
            del self.concepts[name0]
            self.edges.pop(name0, None)
        self.concepts[name] = Concept(key.clone().float(), 1, time.time(), time.time())
        return name

    def find(self, key: torch.Tensor) -> tuple[str, float]:
        best, best_sim = "", -1.0
        for name, c in self.concepts.items():
            sim = sdc_similarity(key, c.proto)
            if sim > best_sim:
                best, best_sim = name, sim
        return best, best_sim

    def link(self, a: str, b: str, weight: float = 1.0) -> None:
        self.edges.setdefault(a, {})[b] = self.edges.get(a, {}).get(b, 0.0) + weight

    def contradictions(self) -> list[tuple[str, str]]:
        """Find concept pairs that co-occur but have low prototype similarity — stress-test."""
        out = []
        names = list(self.concepts.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                w = self.edges.get(names[i], {}).get(names[j], 0.0)
                sim = sdc_similarity(self.concepts[names[i]].proto, self.concepts[names[j]].proto)
                if w > 3 and sim < 0.3:
                    out.append((names[i], names[j]))
        return out

    def __len__(self) -> int:
        return len(self.concepts)


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------
@dataclass
class Reflex:
    name: str
    signature: torch.Tensor  # SDC of the stimulus that triggers the reflex
    locked: bool = False
    use_count: int = 0


class ProceduralReflexes:
    """M4: named reflex circuits — practiced behavior migrates out of deliberation."""

    def __init__(self):
        self.reflexes: dict[str, Reflex] = {}

    def register(self, name: str, signature: torch.Tensor) -> None:
        self.reflexes[name] = Reflex(name, signature.clone())

    def lookup(self, key: torch.Tensor, threshold: float = 0.75) -> Reflex | None:
        best, best_sim = None, -1.0
        for r in self.reflexes.values():
            sim = sdc_similarity(key, r.signature)
            if sim > best_sim:
                best, best_sim = r, sim
        if best is not None and best_sim >= threshold:
            best.use_count += 1
            return best
        return None

    def stats(self) -> dict[str, float]:
        return {"n_reflexes": float(len(self.reflexes))}
