"""Readout heads trained by a local contrastive / forward-forward rule (no backprop).

A linear head over the cortex's sparse distributed state. It learns in two phases:

* **Positive phase** (real outcome): push the correct token's prototype toward the current state.
* **Negative phase** (counterfactual, sampled or imagined): pull sampled wrong tokens away.

Updates are normalized per-class (McNaughton-style diminishing updates) for stability. This is
the closest cheap local approximation to a softmax head's gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioneural.config import LearningConfig


class ReadoutHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        vocab_size: int,
        lcfg: LearningConfig,
        seed: int = 0,
        tied_emb: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.vocab_size = vocab_size
        self.lcfg = lcfg
        g = torch.Generator().manual_seed(seed)
        self.tied = tied_emb is not None
        self.head_hidden = 0 if self.tied else (lcfg.head_hidden if lcfg else 0)
        if self.tied:
            object.__setattr__(self, "_emb_ref", tied_emb)
            self.register_buffer("logit_scale", torch.tensor(1.0))
        else:
            out_dim = self.head_hidden if self.head_hidden > 0 else dim_in
            self.register_buffer(
                "W", (torch.randn(vocab_size, out_dim, generator=g) * 0.02).to(torch.float16)
            )
            if self.head_hidden > 0:
                mask = torch.rand(self.head_hidden, dim_in, generator=g) < 0.15
                vals = torch.where(
                    torch.rand(self.head_hidden, dim_in, generator=g) < 0.5, 1.0, -1.0
                )
                self.register_buffer("W_fixed", (mask.float() * vals).to(torch.float16))
        self.register_buffer("count", torch.zeros(vocab_size))
        self._rng = g

    # ------------------------------------------------------------------
    def normalize(self, ctx: torch.Tensor) -> torch.Tensor:
        n = ctx.norm() + 1e-8
        return ctx / n

    def _features(self, ctx: torch.Tensor) -> torch.Tensor:
        """Map a context vector to the head's input features (frozen ReLU features when deep)."""
        if self.head_hidden > 0:
            h = torch.relu(ctx @ self.W_fixed.float().t())
            return h / (h.norm() + 1e-8)
        return ctx

    def _features_batch(self, ctx: torch.Tensor) -> torch.Tensor:
        if self.head_hidden > 0:
            h = torch.relu(ctx.float() @ self.W_fixed.float().t())  # (W, hidden)
            return h / (h.norm(dim=1, keepdim=True) + 1e-8)
        return ctx

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """Logits over vocab for a (normalized) context vector."""
        ctx = self.normalize(ctx)
        if self.tied:
            return self.logit_scale * (ctx.to(torch.float16) @ self._emb_ref.to(torch.float16).t())
        return self._features(ctx).to(torch.float16) @ self.W.T  # (vocab,)

    # ------------------------------------------------------------------
    def normalize_batch(self, ctx: torch.Tensor) -> torch.Tensor:
        return ctx / (ctx.norm(dim=1, keepdim=True) + 1e-8)

    def forward_batch(self, ctx: torch.Tensor) -> torch.Tensor:
        """Logits for a (W, dim) batch of contexts -> (W, vocab)."""
        ctx = self.normalize_batch(ctx)
        if self.tied:
            return self.logit_scale * (ctx.to(torch.float16) @ self._emb_ref.to(torch.float16).t())
        return self._features_batch(ctx).to(torch.float16) @ self.W.T

    def learn_batch(
        self,
        ctx: torch.Tensor,
        y_pos: list[int],
        n_neg: int = 4,
        mod: float = 1.0,
    ) -> list[list[int]]:
        """Online softmax-gradient update for the whole window (many tokens per GPU op).

        Per token: g = softmax(W·ctx) − onehot(y), then W -= lr·(g ⊗ ctx). This is the exact
        gradient of the head's cross-entropy, computed locally (ctx is a fixed input; no gradient
        flows back into the cortex). Positive and negative signal are balanced per token, so the
        head cannot collapse the way the old sampled-contrastive rule did when pulls outran pushes.
        """
        ctx = self.normalize_batch(ctx).to(torch.float16)
        ys = torch.tensor(y_pos, dtype=torch.long, device=ctx.device)
        if self.tied:
            # Tied embeddings: the shared embedding matrix is the output prototype store (GPT-2
            # style). The softmax-gradient update below is the OUTPUT role of emb[v]; the returned
            # d_ctx feeds the INPUT role (emb[x_t] -= lr*d_ctx) in the organism, so both directions
            # get exact supervised signal. Prototype rows are L2-normalized each window, so logit
            # sharpness must come from a learnable logit_scale (calibrated by the CE margin) instead
            # of growing prototype norms.
            emb32 = self._emb_ref.float()
            logits = self.logit_scale * (ctx.float() @ emb32.t())  # (W, vocab)
            p = torch.softmax(logits, dim=-1)
            onehot = torch.zeros_like(p)
            onehot.scatter_(1, ys[:, None], 1.0)
            lr = self.lcfg.lr_readout * mod / (1.0 + self.count[ys].sqrt())  # (W,)
            grad = (p - onehot) * lr[:, None]
            upd = grad.t() @ ctx.float()  # (vocab, dim)
            self._emb_ref.sub_(upd.to(self._emb_ref.dtype))
            margin = (logits * onehot).sum(-1) - (logits * p).sum(-1)
            self.logit_scale.add_(
                (0.1 * self.lcfg.lr_readout * mod) * margin.clamp(-2.0, 2.0).mean()
            )
            self.logit_scale.clamp_(0.1, 20.0)
            self.count[ys] += 1
            # input-role top-down in ctx space. Scaled by logit_scale so it co-evolves with
            # confidence exactly like the linear head's d_ctx grows with its prototype norm.
            d_ctx = self.logit_scale * (p - onehot) @ emb32  # (W, dim)
            d_ctx = d_ctx.clamp(-10.0, 10.0)
            return d_ctx
        W32 = self.W.float()
        if self.head_hidden > 0:
            h = self._features_batch(ctx.float())  # (W, hidden)
            logits = h @ W32.t()  # (W, vocab)
            p = torch.softmax(logits, dim=-1)
            onehot = torch.zeros_like(p)
            onehot.scatter_(1, ys[:, None], 1.0)
            lr = self.lcfg.lr_readout * mod / (1.0 + self.count[ys].sqrt())  # (W,)
            grad = (p - onehot) * lr[:, None]
            grad = grad.t() @ h  # (vocab, hidden)
            self.W -= grad.to(torch.float16)
            self.count[ys] += 1
            # exact d_ctx through the ReLU + the *frozen* projection (no trained W1 -> no
            # W1*W2 compounding, so the top-down signal stays in the linear head's envelope).
            dh = (p - onehot) @ W32  # (W, hidden)
            d_ctx = dh * (h > 0).float() @ self.W_fixed.float()  # (W, dim)
            # the fixed projection inflates gradient magnitude; rescale per-token back to the
            # output-layer gradient scale so the embedding top-down lr keeps its meaning.
            gain = dh.norm(dim=1, keepdim=True) / (d_ctx.norm(dim=1, keepdim=True) + 1e-8)
            d_ctx = d_ctx * gain
            d_ctx = d_ctx.clamp(-10.0, 10.0)
            return d_ctx
        logits = ctx.float() @ W32.t()  # (W, vocab)
        p = torch.softmax(logits, dim=-1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, ys[:, None], 1.0)
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[ys].sqrt())  # (W,)
        grad = (p - onehot) * lr[:, None]
        grad = grad.t() @ ctx.float()  # (vocab, dim)
        self.W -= grad.to(torch.float16)
        self.count[ys] += 1
        # top-down error in context space (reciprocal projection through the head's own weights).
        # This is the exact gradient of the head's loss w.r.t. the (normalized) context, which the
        # cortex/embedding can use as a supervised, dopamine-style neuromodulatory signal.
        d_ctx = (p - onehot) @ W32  # (W, dim)
        return d_ctx

    # ------------------------------------------------------------------
    def positive_phase(self, ctx: torch.Tensor, y_pos: int, mod: float = 1.0) -> None:
        protos = self._emb_ref if self.tied else self.W
        ctx = self._features(self.normalize(ctx)).to(torch.float16)
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[y_pos].sqrt().item())
        protos[y_pos] += lr * ctx
        protos[y_pos] /= protos[y_pos].norm() + 1e-8
        self.count[y_pos] += 1

    def negative_phase(self, ctx: torch.Tensor, y_neg: int, mod: float = 1.0) -> None:
        protos = self._emb_ref if self.tied else self.W
        ctx = self._features(self.normalize(ctx)).to(torch.float16)
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[y_neg].sqrt().item())
        protos[y_neg] -= lr * ctx
        protos[y_neg] /= protos[y_neg].norm() + 1e-8

    def learn(
        self,
        ctx: torch.Tensor,
        y_pos: int,
        n_neg: int = 4,
        mod: float = 1.0,
        negatives: list[int] | None = None,
    ) -> list[int]:
        """Contrastive local update. Returns the negative samples used."""
        self.positive_phase(ctx, y_pos, mod)
        if negatives is None:
            negatives = torch.randint(
                0, self.vocab_size - 1, (n_neg,), generator=self._rng
            ).tolist()
            negatives = [y if y < y_pos else y + 1 for y in negatives]
        for y_neg in negatives:
            self.negative_phase(ctx, y_neg, mod)
        return negatives

    # ------------------------------------------------------------------
    def imagine(self, ctx: torch.Tensor, temperature: float = 1.0) -> int:
        """Counterfactual 'imagination' during the negative phase / REM sleep."""
        logits = self.forward(ctx) / max(temperature, 0.05)
        probs = torch.softmax(logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    def stats(self) -> dict[str, float]:
        protos = self._emb_ref if self.tied else self.W
        return {
            "prototype_norm_mean": float(protos.norm(dim=1).mean().item()),
            "logit_scale": float(self.logit_scale.item()) if self.tied else 0.0,
            "rows_touched": float((self.count > 0).sum().item()) / self.vocab_size,
        }
