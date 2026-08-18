"""EmbSSM — the task-aligned predictive cortex: a linear-attention state over token embeddings.

The recurrence is fully continuous (fp32) so its *state formation* can be trained — the ternary
sensory columns and the one-step-local predictive backbone cannot carry task signal (measured:
cortex-only probes stayed ~475 ppl ≈ near-random). EmbSSM fixes that with exact analytic
gradients on the closed-form scan, no backprop-through-time and no autograd:

    h_t   = a ⊙ h_{t-1} + W_in · emb[x_t]      # O(1) state, closed-form scan
    ctx_t = W_out · h_t                         # trained so ctx_t ~ emb[y_{t+1}]

Loss:  L = (1/W) · Σ_t || ctx_t − emb[y_{t+1}] ||²

The state h is L2-normalized before the output projection, so the regression error stays O(1)
and the exact-gradient training is a bounded linear regression (no scale feedback loop: a
growing W_in would otherwise inflate ctx → err → gradients, compounding to NaN).

Exact gradients (closed form, one forward + one backward scan per window):
    dW_out = (1/W) · Σ_t err_t ⊗ h_t
    dW_in  = (1/W) · Σ_t b_t ⊗ emb[x_t],   b_t = Σ_{s≥t} a^{s−t}·(err_s @ W_out)

The embedding anchor emb[x] is excluded from this module's target dynamics: ctx_ssm predicts the
next token from history, the head combines it with the current-token anchor emb[x].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbSSM(nn.Module):
    def __init__(self, dim: int, lr: float = 0.1, decay: float = 0.9, chunk: int = 32):
        super().__init__()
        self.dim = dim
        self.lr = lr
        self.a = decay
        self.chunk = chunk
        self.wd = 1e-4
        self.W_in = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.W_out = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.register_buffer("h", torch.zeros(dim))
        C = chunk
        rel = torch.arange(C).float().reshape(-1, 1)
        self.register_buffer("_apow", (self.a ** rel).to(torch.float32))
        self.register_buffer("_invpow", ((1.0 / self.a) ** rel).to(torch.float32))

    # ------------------------------------------------------------------
    def _scan(self, r: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
        """Closed-form forward scan h_t = a·h_{t-1} + r_t, chunked for low launch count."""
        w = r.shape[0]
        h = torch.empty(w, self.dim, device=r.device)
        C = self.chunk
        for s in range(0, w, C):
            e = min(s + C, w)
            n = e - s
            scaled = r[s:e] * self._invpow[:n]  # r_{s+j} · a^{-j}
            res = self._apow[:n] * (carry.unsqueeze(0) + scaled.cumsum(0))
            h[s:e] = res
            carry = (self.a ** n) * res[-1]
        return h

    def _norm_state(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Single-token step (generation / legacy path)."""
        r = e @ self.W_in.t()
        self.h = (self.a * self.h + r).detach()
        return self._norm_state(self.h) @ self.W_out.t()

    def window(
        self,
        e: torch.Tensor,
        target: torch.Tensor,
        learn: bool = True,
        mod: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched window: forward scan + exact-gradient training against `target` (= emb[y]).

        Returns (ctx_ssm (W, dim), h (W, dim) — normalized state). On `learn=False` only the
        scan runs.
        """
        w = e.shape[0]
        r = e @ self.W_in.t()  # (W, dim)
        h_raw = self._scan(r, self.h)
        h = self._norm_state(h_raw)
        ctx = h @ self.W_out.t()  # (W, dim)
        if learn and target is not None:
            err = ctx - target  # (W, dim)
            dW_out = (err.t() @ h) / w  # (dim, dim)
            g = err @ self.W_out  # (W, dim)
            # backward scan: b_t = g_t + a·b_{t+1}, b_w = 0  ->  scan on the reversed stream
            zero = torch.zeros_like(self.h)
            b = self._scan(g.flip(0), zero).flip(0)
            dW_in = (b.t() @ e) / w  # (dim, dim)
            self.W_out.data.add_(self.lr * mod * (dW_out - self.wd * self.W_out.data))
            self.W_in.data.add_(self.lr * mod * (dW_in - self.wd * self.W_in.data))
        self.h = h_raw[-1].detach()
        return ctx, h

    def reset(self) -> None:
        self.h.zero_()
