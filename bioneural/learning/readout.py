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
    def __init__(self, dim_in: int, vocab_size: int, lcfg: LearningConfig, seed: int = 0):
        super().__init__()
        self.dim_in = dim_in
        self.vocab_size = vocab_size
        self.lcfg = lcfg
        g = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "W", (torch.randn(vocab_size, dim_in, generator=g) * 0.02).to(torch.float16)
        )
        self.register_buffer("count", torch.zeros(vocab_size))
        self._rng = g

    # ------------------------------------------------------------------
    def normalize(self, ctx: torch.Tensor) -> torch.Tensor:
        n = ctx.norm() + 1e-8
        return ctx / n

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """Logits over vocab for a (normalized) context vector."""
        ctx = self.normalize(ctx)
        return ctx.to(torch.float16) @ self.W.T  # (vocab,)

    # ------------------------------------------------------------------
    def normalize_batch(self, ctx: torch.Tensor) -> torch.Tensor:
        return ctx / (ctx.norm(dim=1, keepdim=True) + 1e-8)

    def forward_batch(self, ctx: torch.Tensor) -> torch.Tensor:
        """Logits for a (W, dim) batch of contexts -> (W, vocab)."""
        ctx = self.normalize_batch(ctx)
        return ctx.to(torch.float16) @ self.W.T

    def learn_batch(
        self,
        ctx: torch.Tensor,
        y_pos: list[int],
        n_neg: int = 4,
        mod: float = 1.0,
    ) -> list[list[int]]:
        """Contrastive local update for a whole window (many tokens per GPU op).

        Returns the negative samples per token. Semantics match `learn` (positive push toward the
        observed token + negative pull away from sampled counterfactuals, normalized per row) but
        the whole window is applied with scatter-adds instead of a per-token Python loop.
        """
        ctx = self.normalize_batch(ctx).to(torch.float16)
        ys = torch.tensor(y_pos, dtype=torch.long, device=ctx.device)
        w = ctx.shape[0]

        # positive phase
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[ys].sqrt())  # (W,)
        ys_idx = ys[:, None].expand(w, self.dim_in)
        self.W.scatter_add_(0, ys_idx, (ctx.float() * lr[:, None]).to(torch.float16))
        self.W[ys] = (self.W[ys] / (self.W[ys].norm(dim=1, keepdim=True) + 1e-8)).to(
            torch.float16
        )
        self.count[ys] += 1

        # negative phase (counterfactual pull-away, sampled per token)
        negs = torch.randint(0, self.vocab_size - 1, (w, n_neg), device=ctx.device)
        negs = torch.where(negs >= ys[:, None], negs + 1, negs)  # avoid the positive class
        neg_ids = negs.reshape(-1)
        neg_ctx = ctx.repeat_interleave(n_neg, dim=0) * (self.lcfg.lr_readout * mod)
        self.W.scatter_add_(
            0, neg_ids[:, None].expand(-1, self.dim_in), (-neg_ctx).to(torch.float16)
        )
        uniq = neg_ids.unique()
        self.W[uniq] = (self.W[uniq] / (self.W[uniq].norm(dim=1, keepdim=True) + 1e-8)).to(
            torch.float16
        )
        return negs.tolist()

    # ------------------------------------------------------------------
    def positive_phase(self, ctx: torch.Tensor, y_pos: int, mod: float = 1.0) -> None:
        ctx = self.normalize(ctx).to(torch.float16)
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[y_pos].sqrt().item())
        self.W[y_pos] += lr * ctx
        self.W[y_pos] /= self.W[y_pos].norm() + 1e-8
        self.count[y_pos] += 1

    def negative_phase(self, ctx: torch.Tensor, y_neg: int, mod: float = 1.0) -> None:
        ctx = self.normalize(ctx).to(torch.float16)
        lr = self.lcfg.lr_readout * mod
        self.W[y_neg] -= lr * ctx
        self.W[y_neg] /= self.W[y_neg].norm() + 1e-8

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
        return {
            "prototype_norm_mean": float(self.W.norm(dim=1).mean().item()),
            "rows_touched": float((self.count > 0).sum().item()) / self.vocab_size,
        }
