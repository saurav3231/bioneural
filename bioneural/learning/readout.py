"""Readout heads trained by local (backprop-free) rules over the cortex state.

A two-layer perceptron head (ctx -> hidden -> vocab) with exact local gradients computed
manually at each layer. All weights are continuous (fp32/fp16) — unlike the ternary cortex,
continuous surfaces can take supervised error gradients without discrete-flip noise, so this
is where the task supervises the model. The hidden layer's gradient w.r.t. ctx (`d_ctx`) is
returned so the embedding can use it as a three-factor (dopamine-style) top-down signal.

Forward (batched):
    ctx = l2_normalize(ctx)
    a1  = tanh(W1 · ctx)
    logits = W2 · a1

Local gradients (exact for this MLP, no autograd graph):
    g2 = softmax(logits) - onehot(y);      W2 -= lr · g2 ⊗ a1
    g1 = (g2 · W2) ⊙ (1 - a1²);            W1 -= lr · g1 ⊗ ctx
    d_ctx = g1 · W1                        (three-factor signal for the embedding)
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
        hidden = lcfg.head_hidden
        g = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "W1", (torch.randn(hidden, dim_in, generator=g) * 0.02).to(torch.float16)
        )
        self.register_buffer(
            "W2", (torch.randn(vocab_size, hidden, generator=g) * 0.02).to(torch.float16)
        )
        self.register_buffer("count", torch.zeros(vocab_size))
        self._rng = g

    # ------------------------------------------------------------------
    def _forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ctx (W, dim) -> (a1 (W, hidden), logits (W, vocab))."""
        ctx = ctx / (ctx.norm(dim=-1, keepdim=True) + 1e-8)
        a1 = torch.tanh((ctx.to(torch.float16) @ self.W1.t()).float())  # (W, hidden)
        logits = a1.to(torch.float16) @ self.W2.t()  # (W, vocab)
        return a1, logits

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """Logits over vocab for a context vector."""
        _, logits = self._forward(ctx)
        return logits.squeeze(0)

    def forward_batch(self, ctx: torch.Tensor) -> torch.Tensor:
        """Logits for a (W, dim) batch of contexts -> (W, vocab)."""
        _, logits = self._forward(ctx)
        return logits

    # ------------------------------------------------------------------
    def learn_batch(
        self,
        ctx: torch.Tensor,
        y_pos: list[int],
        n_neg: int = 4,
        mod: float = 1.0,
    ) -> torch.Tensor:
        """Exact local gradients for the 2-layer head over a whole window.

        Returns `d_ctx = g1 · W1` — the gradient of the head's loss w.r.t. the (normalized)
        context, used by the embedding as a three-factor top-down signal.
        """
        a1, logits = self._forward(ctx)
        ys = torch.tensor(y_pos, dtype=torch.long, device=ctx.device)
        p = torch.softmax(logits.float(), dim=-1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, ys[:, None], 1.0)
        lr2 = self.lcfg.lr_readout * mod / (1.0 + self.count[ys].sqrt())  # (W,)
        lr1 = self.lcfg.lr_hidden * mod / (1.0 + self.count[ys].sqrt())
        g2 = p - onehot  # (W, vocab) unscaled error
        self.W2 -= ((g2 * lr2[:, None]).t() @ a1).to(torch.float16)  # (vocab, hidden)
        g1 = (g2 @ self.W2.float()) * (1.0 - a1**2)  # (W, hidden)
        ctxn = ctx / (ctx.norm(dim=-1, keepdim=True) + 1e-8)
        self.W1 -= ((g1 * lr1[:, None]).t() @ ctxn.float()).to(torch.float16)  # (hidden, dim)
        self.count[ys] += 1
        d_ctx = (g1 * lr1[:, None]) @ self.W1.float()  # (W, dim)
        return d_ctx

    # ------------------------------------------------------------------
    def positive_phase(self, ctx: torch.Tensor, y_pos: int, mod: float = 1.0) -> None:
        a1, _ = self._forward(ctx)
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[y_pos].sqrt().item())
        self.W2[y_pos] += (lr * a1).to(torch.float16)
        self.W2[y_pos] /= self.W2[y_pos].norm() + 1e-8
        self.count[y_pos] += 1

    def negative_phase(self, ctx: torch.Tensor, y_neg: int, mod: float = 1.0) -> None:
        a1, _ = self._forward(ctx)
        lr = self.lcfg.lr_readout * mod / (1.0 + self.count[y_neg].sqrt().item())
        self.W2[y_neg] -= (lr * a1).to(torch.float16)
        self.W2[y_neg] /= self.W2[y_neg].norm() + 1e-8

    def learn(
        self,
        ctx: torch.Tensor,
        y_pos: int,
        n_neg: int = 4,
        mod: float = 1.0,
        negatives: list[int] | None = None,
    ) -> list[int]:
        """Contrastive local update (legacy one-token path). Returns the negatives used."""
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
            "prototype_norm_mean": float(self.W2.norm(dim=1).mean().item()),
            "hidden_norm_mean": float(self.W1.norm(dim=1).mean().item()),
            "rows_touched": float((self.count > 0).sum().item()) / self.vocab_size,
        }
