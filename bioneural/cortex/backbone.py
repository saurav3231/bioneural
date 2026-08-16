"""Event-driven recurrent state backbone (thalamo-cortical loops).

A Mamba/GLA-class linear-attention cell whose inputs and outputs are events. It gives the
cortex long-range language structure without a KV cache:

    h_t  = (1 - f_t) * h_{t-1} + r_t                 # linear recurrence, O(1) state
    f_t  = selective forget gate                      # learns "remember when surprised"
    o_t  = W_out @ h_t                                # event output to the workspace

The state `h_t` IS part of working memory (M1). All learning is local (predictive coding on the
next input + Hebbian forget adjustment). No backprop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioneural.config import CortexConfig, LearningConfig
from bioneural.quant.ternary import TernaryParam


class EventSSM(nn.Module):
    def __init__(self, dim_in: int, dim: int, cfg: CortexConfig, lcfg: LearningConfig):
        super().__init__()
        self.dim = dim
        self.lcfg = lcfg
        # W_in: (dim, dim_in), W_out: (dim_in, dim) readback, forget latent (dim,)
        self.W_in = TernaryParam((dim, dim_in))
        self.W_out = TernaryParam((dim_in, dim))
        self.forget = nn.Parameter(torch.full((dim,), -2.0))  # logit -> sigmoid ~0.12 baseline
        self.register_buffer("h", torch.zeros(dim))
        self.decay_history: list[float] = []
        self.pred_ctx: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def context(self) -> torch.Tensor:
        """Recurrent state projected back into readout space (feeds the readout head)."""
        return self.W_out.forward(self.h.detach())

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Update the recurrent state with one event-driven readout vector `r`."""
        f = torch.sigmoid(self.forget)
        r_proj = self.W_in.forward(r)
        self.h = (1 - f) * self.h + r_proj
        # predict the next readout from current state (for local predictive learning)
        self.pred_ctx = self.W_out.forward(self.h.detach())
        return self.h

    def learn(self, r_next: torch.Tensor, mod: float = 1.0) -> float:
        """Predictive-coding update on the readback projection + adaptive forget."""
        if self.pred_ctx is None:
            return 0.0
        err = r_next - self.pred_ctx
        # local gradient for W_out: dL/dW_out ~ outer(err, h); W_out is (dim_in, dim)
        grad = torch.einsum("i,j->ij", err, self.h.detach())
        self.W_out.update_latent(grad, lr=self.lcfg.lr_predict * mod)
        # adaptive forget: remember more when surprised (prediction error is high)
        surprise = float(err.abs().mean().item())
        self.forget.data = torch.clamp(
            self.forget.data + self.lcfg.lr_homeo * mod * (surprise - 0.3), -4.0, 1.0
        )
        self.decay_history.append(surprise)
        return surprise

    def reset(self) -> None:
        self.h.zero_()
        self.pred_ctx = None

    def stats(self) -> dict[str, float]:
        return {
            "mean_surprise": float(
                sum(self.decay_history[-256:]) / max(len(self.decay_history[-256:]), 1)
            ),
            "mean_forget": float(torch.sigmoid(self.forget).mean().item()),
        }
