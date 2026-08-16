"""Three-factor Hebbian helpers.

    Δw_ij = η · M(t) · e_ij(t)     where  e_ij = decay(e_ij) + f(pre_j, post_i)

The eligibility trace encodes "these two neurons were co-involved recently" (STDP-flavored,
stored locally at INT8, decaying over seconds-to-minutes); M(t) is the broadcast neuromodulator
that says "and it mattered". This solves distal credit assignment *in time* the way brains do:
tag now, confirm later.
"""

from __future__ import annotations

import torch


def eligibility_coact(
    post_fire: torch.Tensor,
    pre_trace: torch.Tensor,
) -> torch.Tensor:
    """Outer product of post-firing (now) and pre-eligibility-trace (recent).

    Args:
        post_fire: (..., N_post) float firing this tick.
        pre_trace: (..., N_pre) float eligibility trace of presynaptic neurons.

    Returns:
        (..., N_post, N_pre) per-synapse eligibility (the e_ij term).
    """
    return torch.einsum("...i,...j->...ij", post_fire, pre_trace)


def gate_plasticity(mod: dict[str, float], strength: float = 1.0) -> float:
    """Combine the neuromodulator scalars into a single plasticity gate M(t) in [0, 1]."""
    da = mod.get("DA", 0.5)
    ne = mod.get("NE", 0.5)
    ach = mod.get("ACh", 0.5)
    sht = mod.get("5HT", 0.5)
    # surprise + reward + attention gate learning; 5HT is a brake (low 5HT = conservatism)
    gate = (0.4 * da + 0.3 * ne + 0.3 * ach) * (1.0 - 0.5 * (1.0 - sht))
    return float(max(0.0, min(1.0, gate))) * strength
