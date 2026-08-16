"""Token -> spike burst conversion.

Each token becomes a short burst of events over `ticks` ticks (~5% column activation) via
temporal multiplexing: the embedding dimension is split across ticks, and the top-k magnitudes
within each tick-bin become the events. Familiar phrases -> little prediction error -> skimming
is nearly free; novel content is expensive. It reads the way people read.
"""

from __future__ import annotations

import torch


def spike_encode(
    emb: torch.Tensor,
    ticks: int,
    k_active: int,
) -> list[torch.Tensor]:
    """Convert a dense token embedding into a list of sparse per-tick event vectors."""
    d = emb.numel()
    device = emb.device
    idx = torch.arange(d, device=device)
    bins = idx % ticks
    vecs: list[torch.Tensor] = []
    for t in range(ticks):
        mask = bins == t
        sub_idx = idx[mask]
        sub_vals = emb[sub_idx]
        k = min(k_active, sub_idx.numel())
        if k == 0:
            vecs.append(torch.zeros(d, device=device))
            continue
        top = torch.topk(sub_vals, k).indices
        v = torch.zeros(d, device=device)
        v[sub_idx[top]] = sub_vals[top]
        vecs.append(v)
    return vecs


def spike_events(emb: torch.Tensor, ticks: int, k_active: int) -> list[tuple[int, float]]:
    """Return the raw `(input_neuron_id, magnitude)` events (AER form) for the burst."""
    events: list[tuple[int, float]] = []
    for _t, vec in enumerate(spike_encode(emb, ticks, k_active)):
        nz = vec.nonzero(as_tuple=False).flatten()
        for i in nz:
            events.append((int(i), float(vec[i].item())))
    return events


def spike_encode_batch(
    embs: torch.Tensor,
    ticks: int,
    k_active: int,
) -> list[torch.Tensor]:
    """Vectorized `spike_encode` for a `(W, d)` embedding batch -> `ticks` tensors of `(W, d)`.

    Same encoding rule as `spike_encode` (per-tick bins, top-k by magnitude per row) but the whole
    window of tokens is encoded in a handful of tensor ops so the GPU stays busy.
    """
    w, d = embs.shape
    device = embs.device
    idx = torch.arange(d, device=device)
    bins = idx % ticks
    bursts: list[torch.Tensor] = []
    for t in range(ticks):
        m = bins == t
        sub_idx = idx[m]
        sub_vals = embs[:, m]
        k = min(k_active, sub_idx.numel())
        if k == 0:
            bursts.append(torch.zeros(w, d, device=device))
            continue
        top = torch.topk(sub_vals, k, dim=1).indices  # (W, k) local indices within the bin
        v = torch.zeros(w, d, device=device)
        v.scatter_(1, sub_idx[top], sub_vals.gather(1, top))
        bursts.append(v)
    return bursts
