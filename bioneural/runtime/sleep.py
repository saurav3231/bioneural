"""Sleep — the consolidation daemon.

* **NREM-analog:** high-salience engrams replayed into the cortex at 10-20x speed; replayed
  coactivations -> eligibility traces -> slow ternary weights (M2->M4 transfer).
* **REM-analog:** free-running imagination (workspace self-loop, high temperature); generated
  rollouts are the *negative data* for the readout and the stress-tests for the semantic graph.
* **Synaptic downscaling:** global multiplicative decay of fast weights + prune pass
  (sleep-homeostasis hypothesis) -> M2a is fresh each "morning", VRAM reclaimed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from bioneural.runtime.organism import BioNeural


def sleep_cycle(org: BioNeural, replay_n: int = 64, temperature: float = 1.4) -> dict[str, float]:
    """Run one full sleep cycle (NREM + REM + downscaling). Returns diagnostics."""
    info: dict[str, float] = {}
    m2_before = len(org.fabric.m2b)

    # ---- NREM-analog: replay recent engrams into the cortex ----
    engrams = org.fabric.remember_latest(replay_n)
    gate = org.bus.gate()
    for e in engrams:
        key = e.key.to(org.device).float()
        org.columns.forward(key, gate)
        org.columns.learn_hebbian(gate)
        org.columns.learn_predictive(key, gate)
        org.surprise.update(0.01)
        # consolidate into the semantic graph (M2 -> M3) with links
        org.fabric.m3.add(key, meta=e.payload_meta)
    info["replayed"] = float(len(engrams))
    info["m3_concepts"] = float(len(org.fabric.m3))

    # ---- REM-analog: imagination as negative data + contradiction discovery ----
    n_neg = 0
    ctx = (
        org._last_ctx
        if org._last_ctx is not None
        else torch.zeros(org.c.readout_dim, device=org.device)
    )
    for _ in range(replay_n // 2):
        sample = org.readout.imagine(ctx, temperature)
        org.readout.negative_phase(ctx, sample, mod=gate)
        n_neg += 1
    contradictions = org.fabric.m3.contradictions()
    info["negatives_imagination"] = float(n_neg)
    info["contradictions_found"] = float(len(contradictions))
    org.drives.drives["coherence"] = max(
        0.0, org.drives.drives["coherence"] - 0.3 * len(contradictions)
    )
    org.drives.drives["curiosity"] = max(0.0, org.drives.drives["curiosity"] - 0.1)

    # ---- Synaptic downscaling + structural pruning ----
    org.fabric.m2a.forget(0.9)
    for mod in org.modules():
        if isinstance(mod, torch.nn.Module) and hasattr(mod, "apply_decay"):
            mod.apply_decay(0.999)
    pruned = org.prune_latents(threshold_frac=0.5)
    info["pruned_weights"] = float(pruned)
    info["engrams_before"] = float(m2_before)
    info["engrams_after"] = float(len(org.fabric.m2b))
    return info
