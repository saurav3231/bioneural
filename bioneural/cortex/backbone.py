"""Event-driven recurrent state backbone (thalamo-cortical loops).

A Mamba/GLA-class linear-attention cell whose inputs are events. It gives the cortex long-range
language structure without a KV cache:

    h_t  = (1 - f_t) * h_{t-1} + r_t                 # linear recurrence, O(1) state
    f_t  = selective forget gate                      # learns "remember when surprised"
    o_t  = W_out @ h_t                                # event output to the workspace

The state `h_t` IS part of working memory (M1). W_in/W_out are CONTINUOUS (fp32) so the
recurrence can be trained by predictive coding against the NEXT TOKEN's embedding (passed in as
`target`), turning the state formation itself into a next-token predictor — the ternary cortex
cannot carry task signal, so the SSM readback is the supervised long-range channel. All learning
is local (predictive coding on the next target + Hebbian forget adjustment). No backprop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioneural.config import CortexConfig, LearningConfig


class EventSSM(nn.Module):
    def __init__(self, dim_in: int, dim: int, cfg: CortexConfig, lcfg: LearningConfig):
        super().__init__()
        self.dim = dim
        self.dim_in = dim_in
        self.lcfg = lcfg
        # continuous (fp32) state projections: h_t = a⊙h_{t-1} + W_in·r_t, o_t = W_out·h_t
        self.W_in = nn.Parameter(torch.randn(dim, dim_in) * 0.05)
        self.W_out = nn.Parameter(torch.randn(dim_in, dim) * 0.05)
        self.forget = nn.Parameter(torch.full((dim,), -2.0))  # logit -> sigmoid ~0.12 baseline
        self.register_buffer("h", torch.zeros(dim))
        self.decay_history: list[float] = []
        self.pred_ctx: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def context(self) -> torch.Tensor:
        """Recurrent state projected back into readout space (feeds the readout head)."""
        return self.W_out.detach() @ self.h.detach()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Update the recurrent state with one event-driven readout vector `r`."""
        f = torch.sigmoid(self.forget)
        r_proj = self.W_in @ r
        self.h = ((1 - f) * self.h + r_proj).detach()
        self.pred_ctx = self.W_out @ self.h.detach()
        return self.h

    def learn(self, r_next: torch.Tensor, mod: float = 1.0, target: torch.Tensor | None = None) -> float:
        """Predictive-coding update on the readback projection + adaptive forget.

        `target` (next token's embedding) overrides the sensory target `r_next` when the task
        wants the state to become a next-token predictor instead of a sensory auto-encoder.
        """
        if self.pred_ctx is None:
            return 0.0
        tgt = target if target is not None else r_next
        err = tgt - self.pred_ctx
        grad = torch.einsum("i,j->ij", err, self.h.detach())
        self.W_out.data.add_(self.lcfg.lr_backbone * mod * grad)
        surprise = float(err.abs().mean().item())
        self.forget.data = torch.clamp(
            self.forget.data + self.lcfg.lr_homeo * mod * (surprise - 0.3), -4.0, 1.0
        )
        self.decay_history.append(surprise)
        return surprise

    def window(
        self,
        r: torch.Tensor,
        learn: bool = True,
        mod: float = 1.0,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Batched window pass over the linear recurrence (many tokens per GPU op).

        h_t = a ⊙ h_{t-1} + W_in·r_t, a = 1 - sigmoid(forget), has the closed form
        h_t = a^t ⊙ h_prev + Σ_{i<=t} a^{t-i} ⊙ b_i (b_i = W_in·r_i), evaluated with a
        single triangular einsum over the whole window instead of a per-token loop.

        When `learn` and `target` is given (emb[y] for the window), W_out/W_in are trained by
        predictive coding to output the next token's embedding from the state — the recurrence
        becomes a genuinely task-trained sequence model.

        Returns `(h_window (W, dim), surprise (0-d or None))`. The recurrence state `h` carries
        the window's last state so consecutive windows stay connected.
        """
        w = r.shape[0]
        r_proj = self.W_in @ r  # (W, dim)
        a = (1.0 - torch.sigmoid(self.forget)).clamp(min=0.1)  # (dim,), >=0.1 keeps inverses finite
        # exact linear-recurrence scan, O(W·dim): h_t = a⊙h_{t-1} + r_proj_t
        # chunked (C<=32) so a^{-j} exponents stay finite; done in fp64 so the huge intermediates
        # don't leak relative error into the carry.
        rp64 = r_proj.double()
        a64 = a.double()
        h = torch.empty(w, self.dim, device=r.device)
        h_prev = self.h.detach().clone()
        carry = h_prev.double()
        C = 32
        inv_a = (1.0 / a64).reshape(1, -1)
        rel = torch.arange(C, device=r.device).double().reshape(-1, 1)
        a_pow_t = a64.reshape(1, -1).pow(rel)  # (C, dim) a^i
        inv_pow_t = inv_a.pow(rel)  # (C, dim) a^{-i}
        for s in range(0, w, C):
            e = min(s + C, w)
            n = e - s
            a_pow = a_pow_t[:n]
            scaled = rp64[s:e] * inv_pow_t[:n]
            res = a_pow * (carry.unsqueeze(0) + scaled.cumsum(0))
            h[s:e] = res.float()
            carry = (a64 * res[-1]).detach()

        surprise = None
        if learn:
            h_prev_cat = torch.cat([h_prev[None, :], h[:-1]], dim=0)  # (W, dim)
            ctx_prev = h_prev_cat @ self.W_out.t()  # (W, dim_in)
            err = (target if target is not None else r) - ctx_prev  # (W, dim_in)
            grad_out = (err.t() @ h_prev_cat) / w  # (dim_in, dim)
            self.W_out.data.add_(self.lcfg.lr_backbone * mod * grad_out)
            dh = err @ self.W_out  # (W, dim): how the state should move
            grad_in = (dh.t() @ r) / w  # (dim, dim_in)
            self.W_in.data.add_(self.lcfg.lr_backbone * mod * grad_in)
            surprise = err.abs().mean()
            s = float(surprise.item())
            self.forget.data = torch.clamp(
                self.forget.data + self.lcfg.lr_homeo * mod * (s - 0.3), -4.0, 1.0
            )
            self.decay_history.append(s)

        self.h = h[-1].detach()
        return h, surprise

    def context_batch(self, h: torch.Tensor) -> torch.Tensor:
        """Recurrent state window projected back into readout space."""
        return h.detach() @ self.W_out.t()

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
