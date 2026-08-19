"""EmbSSM — task-aligned predictive state: a linear-attention over token embeddings.

The ternary sensory columns and one-step-local predictive backbone cannot carry task signal
(measured: cortex-only probes ~475 ppl ≈ random floor). EmbSSM replaces them with a continuous
fp32 linear recurrence over the input embeddings, trained END-TO-END against the head's
cross-entropy with exact closed-form gradients (one forward + one backward scan, chunked) — no
autograd, no backprop-through-time:

    h_t   = a ⊙ h_{t-1} + W_in · emb[x_t]        # O(1) linear state, closed-form scan
    logits_ssm = W_vocab · h_n                    # second head over the L2-normalized state

Multi-scale state: the head reads the CONCATENATION of several channels at different leakage
decays (a = 0.5 short / 0.9 mid / 0.99 long). Each channel is a linear-attention over the same
embeddings, so the exact closed-form training is preserved per channel; together they expose
the readout to distinct geometric signatures of the context (short pairs, mid n-grams, and the
long-range drift), which a single channel cannot linearly separate (measured: single linear
channel tops out at n-gram level, ~113 ppl on TinyStories; the concatenated 3-scale state
solves a 2nd-order Markov task to ppl 2.3 / acc 1.0 that the single channel cannot crack).

The bigram head (over emb[x]) is the proven ~104 floor and stays untouched; the SSM channel
only adds logits, and its head is trained on the exact COMBINED cross-entropy gradient, so an
uninformative state drives it to ~0 — the model can never be worse than the bigram, and when
the state carries higher-order structure it corrects the bigram's mistakes.

Gradients (exact, closed-form), per channel c:
    dW_vocab = (1/W) · Σ_t d_ssm_t ⊗ h_n_t              (d_ssm = β·(p − onehot) on the combined)
    dW_in_c  = (1/W) · Σ_t b_t ⊗ emb[x_t],  b_t = Σ_{s≥t} a_c^{s−t}·d_ctx_s

where d_ctx_c = (d_ssm @ W_vocab[:, c-slice]) is the CE gradient routed back into channel c.
The state is normalized before the head so the regression stays bounded (no scale feedback loop).
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
        hidden: int = 0,
        seed: int = 0,
        decays: tuple[float, ...] = (),
    ):
        super().__init__()
        self.dim = dim
        self.decays = tuple(decays) if decays else (decay,)
        self.nch = len(self.decays)
        self.lr = lr
        self.head_lr = head_lr
        self.chunk = chunk
        self.hidden = hidden
        self.wd = 1e-4
        self.d_head = dim * self.nch if hidden == 0 else hidden
        self.W_in = nn.ParameterList(
            [nn.Parameter(torch.randn(dim, dim) * 0.05) for _ in self.decays]
        )
        # W_vocab starts at ZERO: the SSM channel begins perfectly neutral, so β's gradient
        # starts at zero (no worse-than-random cold-start collapse) and the channel learns
        # from the bigram's residual before β grows to amplify it.
        self.W_vocab = nn.Parameter(torch.zeros(vocab_size, self.d_head))
        self.beta = nn.Parameter(torch.tensor(1.0))  # mixing: logits += β·logits_ssm
        self.register_buffer("h", torch.zeros(dim * self.nch))
        if hidden > 0:
            g = torch.Generator().manual_seed(seed)
            mask = torch.rand(hidden, dim * self.nch, generator=g) < 0.15
            vals = torch.where(torch.rand(hidden, dim * self.nch, generator=g) < 0.5, 1.0, -1.0)
            self.register_buffer("W_fix", (mask.float() * vals))
        C = chunk
        # fp64 from the start: for a=0.5, C=384 the closed form needs 2^383 (~1e115) and
        # 0.5^383 (~1e-115), both outside fp32's range (fp32 would produce inf/0 before any
        # later .double() cast could save it).
        rel = torch.arange(C).double()
        self.register_buffer("_apows", torch.stack([(a ** rel) for a in self.decays]))
        self.register_buffer("_invpows", torch.stack([((1.0 / a) ** rel) for a in self.decays]))

    # ------------------------------------------------------------------
    def _scan(self, r: torch.Tensor, carry: torch.Tensor, c: int) -> torch.Tensor:
        """Closed-form forward scan h_t = a_c·h_{t-1} + r_t, chunked for low launch count.
        The closed form needs a^{-j} (up to (1/a)^C), which overflows fp32 for small decays
        (a=0.5, C=384 -> 2^383 ~ 1e115 > fp32 max 3.4e38). The elementwise accumulation runs in
        fp64; the input r and output h stay fp32 (the matmul that produces r is untouched)."""
        w = r.shape[0]
        r64 = r.float().double()
        carry64 = carry.double()
        h64 = torch.empty(w, self.dim, device=r.device, dtype=torch.float64)
        C = self.chunk
        apow = self._apows[c, :C].unsqueeze(1)
        invpow = self._invpows[c, :C].unsqueeze(1)
        a = self.decays[c]
        for s in range(0, w, C):
            e = min(s + C, w)
            n = e - s
            scaled = r64[s:e] * invpow[:n]  # r_{s+j} · a^{-j}
            res = apow[:n] * (carry64.unsqueeze(0) + scaled.cumsum(0))
            h64[s:e] = res
            carry64 = (a ** n) * res[-1]
        return h64.float()

    def _norm_state(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    def sdch(self, h: torch.Tensor) -> torch.Tensor:
        """The mid-decay channel slice of a concatenated state (the memory/SDC key — must stay
        single-channel for the memory fabric's key dim)."""
        mid = min(range(self.nch), key=lambda c: abs(self.decays[c] - 0.9))
        sl = slice(mid * self.dim, (mid + 1) * self.dim)
        return h[..., sl]

    def scan_window(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward scan over a window of input embeddings. Returns the concatenated
        per-channel L2-normalized states (W, nch·dim) and the raw (W, nch·dim); advances the
        running carry `h`."""
        hns, hraws = [], []
        for c in range(self.nch):
            r = e @ self.W_in[c].t()  # (W, dim)
            h_raw = self._scan(r, self.h[c * self.dim : (c + 1) * self.dim], c)
            hraws.append(h_raw)
            hns.append(self._norm_state(h_raw))
        self.h = torch.cat([h[-1] for h in hraws]).detach()
        return torch.cat(hns, dim=-1), torch.cat(hraws, dim=-1)

    def features(self, h_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the normalized state to head features. Linear pass-through, or a frozen-ReLU
        random-feature (ELM) map when self.hidden > 0. Returns (features, pre-normalization)
        so the exact ReLU-masked gradient can be routed back through."""
        if self.hidden > 0:
            h = torch.relu(h_n @ self.W_fix.t())  # (W, hidden)
            return h / (h.norm(dim=-1, keepdim=True) + 1e-8), h
        return h_n, h_n

    def logits(self, h_n: torch.Tensor) -> torch.Tensor:
        """SSM channel logits: features @ W_vocab.t() -> (W, vocab)."""
        feat, _ = self.features(h_n)
        return feat.to(torch.float16) @ self.W_vocab.to(torch.float16).t()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Single-token step (generation / legacy path). Returns the SSM logits (vocab,)."""
        r = e @ self.W_in[0].t()
        self.h[: self.dim] = (self.decays[0] * self.h[: self.dim] + r).detach()
        return self.logits(self._norm_state(self.h))

    def train_head(self, d_logits: torch.Tensor, h_n: torch.Tensor, mod: float = 1.0) -> None:
        """Update W_vocab from the CE gradient on the SSM logits (d_logits = β·(p − onehot))."""
        feat, _ = self.features(h_n)
        w = feat.shape[0]
        dW = (d_logits.float().t() @ feat) / w
        self.W_vocab.data.add_(self.head_lr * mod * dW.to(self.W_vocab.dtype))

    def dctx_from_head(self, d_ssm: torch.Tensor, h_n: torch.Tensor) -> list[torch.Tensor]:
        """Route the head's CE gradient back to each state channel. Linear: d_ssm @ W_vocab[:, c].
        With the frozen feature map: ReLU-masked (×2) projection through W_fix (gain-rescaled to
        the output-layer gradient scale), split back per channel."""
        if self.hidden == 0:
            wv = self.W_vocab.float()
            return [
                d_ssm.float() @ wv[:, c * self.dim : (c + 1) * self.dim]
                for c in range(self.nch)
            ]
        feat, feat_pre = self.features(h_n)
        dh = d_ssm.float() @ self.W_vocab.float()  # (W, hidden) w.r.t. normalized feat
        d_pre = dh * (feat_pre > 0).float()  # ReLU mask
        d_h = d_pre @ self.W_fix.float()  # (W, nch·dim)
        out = []
        for c in range(self.nch):
            sl = d_h[:, c * self.dim : (c + 1) * self.dim]
            gain = dh.norm(dim=1, keepdim=True) / (sl.norm(dim=1, keepdim=True) + 1e-8)
            out.append(sl * gain)
        return out

    def train_beta(self, grad_beta: torch.Tensor, mod: float = 1.0) -> None:
        """Update the mixing scalar from dL/dβ = Σ_v (p − onehot)_v · logits_ssm_v.
        Uses the (larger) state lr — the mixing decision should respond quickly. A small floor
        keeps a learning signal for the channel even during a bad streak."""
        self.beta.data.add_(self.lr * mod * grad_beta)
        self.beta.data.clamp_(0.05, 4.0)

    def apply_grad_ctx(self, d_ctx: list[torch.Tensor], e: torch.Tensor, mod: float = 1.0) -> None:
        """Train each W_in_c by the channel's ctx-space loss gradient. Exact closed-form backward
        scan of the linear recurrence."""
        w = d_ctx[0].shape[0]
        for c, d in enumerate(d_ctx):
            zero = torch.zeros_like(self.h[c * self.dim : (c + 1) * self.dim])
            b = self._scan(d.float().flip(0), zero, c).flip(0)  # b_t = Σ_{s≥t} a^{s−t}·d_s
            dW = (b.t() @ e.float()) / w  # (dim, dim)
            self.W_in[c].data.add_(self.lr * mod * (dW - self.wd * self.W_in[c].data))

    def reset(self) -> None:
        self.h.zero_()
