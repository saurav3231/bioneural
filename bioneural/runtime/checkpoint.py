"""Checkpointing — "the body".

Full state (weights + fast weights + neuron states + stores + drive state + clock) serializes to
a single archive so the organism survives session restarts with memory intact = continuity of
self. Elapsed offline time re-enters the clock bank on resume ("I was off for 5 hours").
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from bioneural.runtime.organism import BioNeural


def save_body(org: BioNeural, path: str | Path) -> Path:
    """Serialize the entire organism to `path` (a directory)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    aux = {
        "clock_t0": org.clock.t0,
        "bus": org.bus,
        "drives": org.drives,
        "surprise": org.surprise,
        "workspace_slots": org.workspace.slots,
        "fabric_m1": org.fabric.m1.slots,
        "fabric_m2a_keys": org.fabric.m2a.keys,
        "fabric_m2a_values": org.fabric.m2a.values,
        "fabric_m2a_strength": org.fabric.m2a.strength,
        "fabric_m2b_engrams": org.fabric.m2b.engrams,
        "fabric_m3_concepts": org.fabric.m3.concepts,
        "fabric_m3_edges": org.fabric.m3.edges,
        "total_tokens": org.total_tokens,
        "acc_ema": org._acc_ema,
        "state": org.state,
        "last_ctx": org._last_ctx,
        "prev_r": org._prev_r,
        "prev_sdc": org._prev_sdc,
        "pred_ready": org._pred_ready,
    }
    payload = {
        "state_dict": org.state_dict(),
        "aux": aux,
        "saved_at": time.time(),
    }
    torch.save(payload, path / "body.pt")
    (path / "meta.json").write_text(
        f'{{"tokens": {org.total_tokens}, "saved_at": {time.time():.2f}, "version": "0.1.0"}}',
        encoding="utf-8",
    )
    return path


def load_body(org: BioNeural, path: str | Path) -> BioNeural:
    """Restore a previously saved body into `org` (which must have the same config)."""
    path = Path(path)
    payload = torch.load(path / "body.pt", map_location=org.device, weights_only=False)
    org.load_state_dict(payload["state_dict"])
    aux = payload["aux"]
    org.clock.t0 = aux["clock_t0"]
    org.bus = aux["bus"]
    org.drives = aux["drives"]
    org.surprise = aux["surprise"]
    org.workspace.slots = aux["workspace_slots"]
    org.fabric.m1.slots = aux["fabric_m1"]
    org.fabric.m2a.keys = aux["fabric_m2a_keys"]
    org.fabric.m2a.values = aux["fabric_m2a_values"]
    org.fabric.m2a.strength = aux["fabric_m2a_strength"]
    org.fabric.m2b.engrams = aux["fabric_m2b_engrams"]
    org.fabric.m3.concepts = aux["fabric_m3_concepts"]
    org.fabric.m3.edges = aux["fabric_m3_edges"]
    org.total_tokens = aux["total_tokens"]
    org._acc_ema = aux["acc_ema"]
    org.state = aux["state"]
    org._last_ctx = aux.get("last_ctx")
    org._prev_r = aux.get("prev_r")
    org._prev_sdc = aux.get("prev_sdc")
    org._pred_ready = aux.get("pred_ready", False)
    return org
