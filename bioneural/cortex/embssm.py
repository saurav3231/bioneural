"""EmbSSM — task-aligned predictive state: a linear-attention over token embeddings.

The ternary sensory columns and one-step-local predictive backbone cannot carry task signal
(measured: cortex-only probes ~475 ppl ≈ random floor). EmbSSM replaces them with a continuous
fp32 linear recurrence over the input embeddings, trained END-TO-END against the head's
cross-entropy with exact closed-form gradients (one forward + one backward scan, chunked) — no
autograd, no backprop-through-time:

    h_t   = a ⊙ h_{t-1} + W_in · emb[x_t]        # O(1) linear state, closed-form scan
    logits_ssm = W_vocab · h_n                    # second head over the L2-normalized state

The bigram head (over emb[x]) is the proven ~104 floor and stays untouched; the SSM channel
only adds logits, and its head is trained on the exact COMBINED cross-entropy gradient, so an
uninformative state drives it to ~0 — the model can never be worse than the bigram, and when
the state carries higher-order structure it corrects the bigram's mistakes.

Gradients (exact, closed-form):
    dW_vocab = (1/W) · Σ_t d_ssm_t ⊗ h_n_t              (d_ssm = β·(p − onehot) on the combined)
    dW_in    = (1/W) · Σ_t b_t ⊗ emb[x_t],  b_t = Σ_{s≥t} a^{s−t}·d_ctx_s

where d_ctx = d_ssm @ W_vocab is the CE gradient routed back into the state. The state is
normalized before the head so the regression stays bounded (no scale feedback loop).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbSSM(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        lr: float = 0.1,
        head_lr: float = 0.05,
        decay: float = 0.9,
        chunk: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.lr = lr
        self.head_lr = head_lr
        self.a = decay
        self.chunk = chunk
        self.wd = 1e-4
        self.W_in = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.W_vocab = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)
        self.beta = nn.Parameter(torch.tensor(1.0))  # mixing: logits += β·logits_ssm
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

    def scan_window(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward scan over a window of input embeddings. Returns (h_n, h_raw) (W, dim) and
        advances the running carry `h`."""
        r = e @ self.W_in.t()  # (W, dim)
        h_raw = self._scan(r, self.h)
        self.h = h_raw[-1].detach()
        return self._norm_state(h_raw), h_raw

    def logits(self, h_n: torch.Tensor) -> torch.Tensor:
        """SSM channel logits: h_n @ W_vocab.t() -> (W, vocab)."""
        return h_n.to(torch.float16) @ self.W_vocab.to(torch.float16).t()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Single-token step (generation / legacy path). Returns the SSM logits (vocab,)."""
        r = e @ self.W_in.t()
        self.h = (self.a * self.h + r).detach()
        return self.W_vocab.to(torch.float16) @ self._norm_state(self.h).to(torch.float16)

    def train_head(self, d_logits: torch.Tensor, h_n: torch.Tensor, mod: float = 1.0) -> None:
        """Update W_vocab from the CE gradient on the SSM logits (d_logits = β·(p − onehot))."""
        w = h_n.shape[0]
        dW = (d_logits.float().t() @ h_n) / w
        self.W_vocab.data.add_(self.head_lr * mod * dW.to(self.W_vocab.dtype))

    def train_beta(self, grad_beta: torch.Tensor, mod: float = 1.0) -> None:
        """Update the mixing scalar from dL/dβ = Σ_v (p − onehot)_v · logits_ssm_v.
        Uses the (larger) state lr — the mixing decision should respond quickly."""
        self.beta.data.add_(self.lr * mod * grad_beta)
        self.beta.data.clamp_(0.0, 4.0)

    def apply_grad_ctx(self, d_ctx: torch.Tensor, e: torch.Tensor, mod: float = 1.0) -> None:
        """Train W_in by a supplied ctx-space loss gradient d_ctx (= CE gradient routed back
        into the state). Exact closed-form backward scan of the linear recurrence."""
        w = d_ctx.shape[0]
        zero = torch.zeros_like(self.h)
        b = self._scan(d_ctx.float().flip(0), zero).flip(0)  # b_t = Σ_{s≥t} a^{s−t}·d_ctx_s
        dW_in = (b.t() @ e.float()) / w  # (dim, dim)
        self.W_in.data.add_(self.lr * mod * (dW_in - self.wd * self.W_in.data))

    def reset(self) -> None:
        self.h.zero_()
