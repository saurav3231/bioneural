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
        self.h = ((1 - f) * self.h + r_proj).detach()
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
        self.W_out.update_latent(grad, lr=self.lcfg.lr_predict * mod, count_flips=False)
        # adaptive forget: remember more when surprised (prediction error is high)
        surprise = float(err.abs().mean().item())
        self.forget.data = torch.clamp(
            self.forget.data + self.lcfg.lr_homeo * mod * (surprise - 0.3), -4.0, 1.0
        )
        self.decay_history.append(surprise)
        return surprise

    def window(
        self, r: torch.Tensor, learn: bool = True, mod: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Batched window pass over the linear recurrence (many tokens per GPU op).

        h_t = a ⊙ h_{t-1} + W_in.forward(r_t), a = 1 - sigmoid(forget), has the closed form
        h_t = a^t ⊙ h_prev + Σ_{i<=t} a^{t-i} ⊙ b_i (b_i = W_in.forward(r_i)), evaluated with a
        single triangular einsum over the whole window instead of a per-token loop.

        Returns `(h_window (W, dim), surprise (0-d or None))`. The recurrence state `h` carries
        the window's last state so consecutive windows stay connected.
        """
        w = r.shape[0]
        r_proj = self.W_in.forward(r)  # (W, dim)
        a = (1.0 - torch.sigmoid(self.forget)).clamp(min=0.1)  # (dim,), >=0.1 keeps inverses finite
        # exact linear-recurrence scan, O(W·dim) instead of the old (W,W,dim) triangular tensor:
        # h_t = a⊙h_{t-1} + r_proj_t  =>  h_{s+i} = a^i⊙h_{s-1} + a^i⊙Σ_{j<=i} a^{-j}⊙r_{s+j}
        # chunked (C<=32) so a^{-j} exponents stay finite; done in fp64 so the huge intermediates
        # (up to ~1e28) don't leak ~1e-7 relative error into the carry (fp32 test showed drift).
        rp64 = r_proj.double()
        a64 = a.double()
        h = torch.empty(w, self.dim, device=r.device)
        h_prev = self.h.detach().clone()
        carry = h_prev.double()
        C = 32  # a^{-C}·r must stay small enough that the O(1) carry isn't lost adding into the cumulative sum
        inv_a = (1.0 / a64).reshape(1, -1)
        for s in range(0, w, C):
            e = min(s + C, w)
            rel = torch.arange(e - s, device=r.device).double().reshape(-1, 1)
            a_pow = a64.reshape(1, -1).pow(rel)  # (C', dim) a^i
            scaled = rp64[s:e] * inv_a.pow(rel)  # r_{s+j} ⊙ a^{-j}
            res = a_pow * (carry.unsqueeze(0) + scaled.cumsum(0))  # (C', dim), fp64
            h[s:e] = res.float()
            # next chunk needs a ⊙ h_{s-1} (old closed form is h_t = a^t h_prev + Σ a^{t-i} r_i)
            carry = (a64 * res[-1]).detach()

        surprise = None
        if learn:
            h_prev_cat = torch.cat([h_prev[None, :], h[:-1]], dim=0)  # (W, dim)
            ctx_prev = self.W_out.forward(h_prev_cat)
            err = r - ctx_prev
            grad = torch.einsum("wd,wi->di", err, h_prev_cat).detach()  # (rd, dim)
            self.W_out.latent = (self.W_out.latent + grad * (self.lcfg.lr_predict * mod)).detach()
            self.W_out._clamp_mask()
            self.W_out.version += 1
            self.W_out._cache = None
            # local predictive update for the state projection W_in (also untrained before this):
            # to reduce the reconstruction error, h should move by dh = W_outᵀ·err, and the
            # readout input r_t caused that state, so dW_in ∝ (dh ⊗ r) (predictive coding, local).
            dh = err @ self.W_out.materialized().float()  # (W, dim)
            grad_in = (dh.t() @ r.detach().float()) / w  # (dim, rd)
            self.W_in.update_latent(grad_in, lr=self.lcfg.lr_predict * mod, count_flips=False)
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
        return self.W_out.forward(h.detach())

    def learn_topdown(
        self, d_ctx: torch.Tensor, h: torch.Tensor, r: torch.Tensor, mod: float = 1.0
    ) -> int:
        """Three-factor top-down (dopamine-style) error on the readback projection.

        The head's exact gradient w.r.t. ctx tells the backbone how its `W_out·h` term should
        move (−d_ctx). Applied to W_out directly and to W_in via the projected error
        (dh = W_outᵀ·d_ctx, dW_in ∝ dh ⊗ r) — the same local pattern as the self-predictive
        pass, but supervised by the task. No backprop.
        """
        w = h.shape[0]
        err_proj = -d_ctx.detach().float()  # (W, rd): desired Δ(W_out·h)
        grad_out = torch.einsum("wd,wi->di", err_proj, h.detach().float()) / w  # (rd, dim)
        grad_out = grad_out / (grad_out.norm() + 1e-8)  # bounded latent update
        self.W_out.latent = (
            self.W_out.latent + grad_out * (self.lcfg.lr_topdown * mod)
        ).detach()
        self.W_out._clamp_mask()
        self.W_out.version += 1
        self.W_out._cache = None
        dh = err_proj @ self.W_out.materialized().float()  # (W, dim)
        grad_in = (dh.t() @ r.detach().float()) / w  # (dim, rd)
        grad_in = grad_in / (grad_in.norm() + 1e-8)  # bounded latent update
        self.W_in.update_latent(grad_in, lr=self.lcfg.lr_topdown * mod, count_flips=False)
        return 0

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
