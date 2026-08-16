"""Homeostatic plasticity: stability, sparsity, no runaway.

* **Adaptive thresholds** (in the QEU) enforce a target firing rate without global normalization.
* **Synaptic scaling** here down-scales bottom-up latents of over-active neurons (rate >> target),
  and up-scales under-active ones — a purely local, multiplicative fix that prevents collapse.
"""

from __future__ import annotations

import torch

from bioneural.cortex.column import ColumnLayer


def apply_synaptic_scaling(
    layer: ColumnLayer,
    target_rate: float,
    scale_floor: float = 0.5,
    scale_ceil: float = 1.5,
    max_change: float = 0.05,
) -> float:
    """Multiply each neuron's incoming bottom-up row by a homeostatic scale factor.

    Returns the mean applied scaling magnitude (diagnostic).
    """
    rate = layer.rate  # (C, K)
    target = target_rate
    factors = torch.clamp(
        torch.sqrt(target / (rate + 1e-6)),
        scale_floor,
        scale_ceil,
    )
    factors = 1.0 + (factors - 1.0).clamp(-max_change, max_change)
    # scale W_in rows per neuron: W_in is (C*K, input_dim); neuron (c, k) -> row c*K+k
    w = layer.W_in.latent
    scaled = w * factors.view(-1, 1)
    layer.W_in.latent = scaled
    layer.W_in._clamp_mask()
    layer.W_in.version += 1
    layer.W_in._cache = None
    return float(factors.mean().item())
