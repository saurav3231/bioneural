"""Sparse distributed codes (SDCs) — the single addressing scheme of the Memory Fabric.

An SDC is a ~2048-dim, ~5%-active binary vector produced by the cortex. SDCs are cheap to
compare (popcount of AND), robust, and naturally support partial/compositional matching.
"""

from __future__ import annotations

import numpy as np
import torch


def make_sdc(
    vec: torch.Tensor,
    active_frac: float = 0.05,
    k: int | None = None,
    ternary: bool = True,
) -> torch.Tensor:
    """Binarize a dense vector into a sparse distributed code (int8, ±1/0)."""
    vec = vec.detach().float().flatten()
    n = vec.numel()
    if k is None:
        k = max(1, int(round(n * active_frac)))
    k = min(k, n)
    topk = torch.topk(vec.abs(), k).indices
    code = torch.zeros(n, dtype=torch.int8, device=vec.device)
    code[topk] = 1
    if ternary:
        botk = torch.topk(-vec.abs(), k).indices
        code[botk] = -1
    return code


def sdc_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Similarity between two SDCs (popcount of AND over ternary-active elements)."""
    if a.shape != b.shape:
        raise ValueError(f"SDC shape mismatch {a.shape} vs {b.shape}")
    a = a.to(torch.int16)
    b = b.to(torch.int16)
    agree = (a * b > 0).sum().float().item()
    union = ((a != 0) | (b != 0)).sum().float().item() + 1e-9
    return float(agree / union)


def pack_bits(code: torch.Tensor) -> bytes:
    """Pack a binary (+1/0) SDC into bytes (for the ~48-64 B/engram episodic store)."""
    code = (code > 0).to(torch.uint8)
    return np.packbits(code.cpu().numpy()).tobytes()


def unpack_bits(blob: bytes, dim: int) -> torch.Tensor:
    return torch.from_numpy(np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:dim].copy())


def quantize_sdc(code: torch.Tensor) -> torch.Tensor:
    """Quantize an SDC to uint8 bits (for product-quantized store)."""
    return (code > 0).to(torch.uint8)
