"""Global Workspace Theory implementation: a small, high-bandwidth arena that all columns
compete to broadcast into.

    COMPETE -> BROADCAST -> ELABORATE -> DECIDE -> LEARN

Thinking = internal event loops that never touch the output head. `imagine()` feeds the
workspace's own broadcast back as pseudo-input (the negative-phase data for forward-forward and
the planning rollouts for agency).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from bioneural.config import WorkspaceConfig

ACTIONS = ("speak", "tool", "query_memory", "note_to_self", "imagine", "rest")


class Workspace:
    def __init__(self, cfg: WorkspaceConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.slots: list[torch.Tensor] = []
        self.spotlight: list[int] = []
        self.thought_count = 0

    # ------------------------------------------------------------------
    def compete(
        self, candidates: list[torch.Tensor], salience: list[float] | None = None
    ) -> list[int]:
        """k-WTA: columns bid events; the highest-salience coalition wins the spotlight."""
        salience = salience or [1.0] * len(candidates)
        order = sorted(range(len(candidates)), key=lambda i: -salience[i])
        coalition = order[: self.cfg.wsa_k]
        self.spotlight = coalition
        return coalition

    def broadcast(self, coalition_codes: list[torch.Tensor]) -> torch.Tensor:
        """Broadcast the winning coalition to all columns + memory fabric."""
        if not coalition_codes:
            code = torch.zeros(self.dim, dtype=torch.int8)
        else:
            acc = torch.zeros(self.dim)
            for c in coalition_codes:
                acc += c.float()
            code = (acc > 0).to(torch.int8)
        # keep the top coalition slots in the workspace
        self.slots = coalition_codes[-self.cfg.n_slots :]
        self.thought_count += 1
        return code

    def elaborate(self, fabric, key: torch.Tensor) -> dict:
        """Memory returns associations; drives inject urgency; a THOUGHT forms."""
        return fabric.recall(key)

    def decide(self, drives: dict[str, float], mod: dict[str, float], novelty: float) -> str:
        """Action head. System-2 (deliberation) when surprise is high; reflex otherwise."""
        if mod["NE"] > 0.65 or novelty > 0.6:
            return "imagine"
        if drives["social"] > 0.8:
            return "speak"
        if drives["curiosity"] > 0.7:
            return "query_memory"
        if drives["energy"] < 0.2:
            return "rest"
        return "note_to_self"

    def imagine(
        self,
        generate: Callable[[torch.Tensor, float], torch.Tensor],
        ctx: torch.Tensor,
        temperature: float = 1.2,
    ) -> torch.Tensor:
        """Internal simulation: feed the workspace broadcast back as pseudo-input."""
        return generate(ctx, temperature)

    def stats(self) -> dict[str, float]:
        return {
            "thought_count": float(self.thought_count),
            "spotlight_size": float(len(self.spotlight)),
        }
