"""The standard baseline: a compact GPT-style Transformer, trained with backprop (AdamW).

This is the honest point of comparison. Same dataset, same tokenizer, matched parameter count,
matched *wall-clock* budget. Anything BioNeural does, this must be measured against.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class StandardTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        max_len: int = 512,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.dim = dim
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([SelfAttnBlock(dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def _dev(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        h = self.emb(x) + self.pos(torch.arange(s, device=x.device))
        mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))  # (b, s, vocab)

    # ------------------------------------------------------------------
    def fit(
        self, tokens: list[int], seconds: float, lr: float = 3e-4, batch: int = 16, seq: int = 128
    ) -> dict:
        """Train for `seconds` wall-clock on a flat token stream. Returns steps taken."""
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        tokens_t = torch.tensor(tokens, dtype=torch.long, device=self._dev())
        n = tokens_t.numel()
        t0 = time.monotonic()
        steps = 0
        losses: list[float] = []
        while time.monotonic() - t0 < seconds:
            starts = torch.randint(
                0, max(n - seq, 1), (batch,), device=self._dev()
            )
            x = torch.stack([tokens_t[s : s + seq] for s in starts])
            y = torch.stack([tokens_t[s + 1 : s + seq + 1] for s in starts])
            logits = self.forward(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.head.out_features), y.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            steps += 1
        return {
            "steps": steps,
            "final_loss": float(losses[-1]) if losses else float("nan"),
            "loss_mean_last100": float(sum(losses[-100:]) / max(len(losses[-100:]), 1)),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, tokens: list[int], seq: int = 128, max_batches: int = 50) -> dict:
        self.eval()
        tokens_t = torch.tensor(tokens, dtype=torch.long, device=self._dev())
        n = tokens_t.numel()
        starts = torch.arange(0, max(n - seq, 1), seq)
        starts = starts[:max_batches]
        total_nll = 0.0
        correct = 0
        total = 0
        for s0 in starts:
            x = tokens_t[s0 : s0 + seq].unsqueeze(0)
            y = tokens_t[s0 + 1 : s0 + seq + 1].unsqueeze(0)
            logits = self.forward(x)[0]
            total_nll += float(
                nn.functional.cross_entropy(
                    logits.reshape(-1, self.head.out_features), y.reshape(-1), reduction="sum"
                )
            )
            correct += int((logits.argmax(-1) == y).sum().item())
            total += y.numel()
        nll = total_nll / max(total, 1)
        return {
            "nll": nll,
            "ppl": math.exp(min(nll, 20.0)),
            "acc": correct / max(total, 1),
            "n_tokens": total,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self, prompt_ids: list[int], n_tokens: int, temperature: float = 0.8, max_len: int = 512
    ) -> list[int]:
        self.eval()
        ids = list(prompt_ids)
        for _ in range(n_tokens):
            ctx = ids[-max_len:]
            x = torch.tensor([ctx], device=self._dev())
            logits = self.forward(x)[0, -1] / max(temperature, 0.05)
            tok = int(torch.multinomial(torch.softmax(logits, -1), 1).item())
            ids.append(tok)
        return ids[len(prompt_ids) :]

    # ------------------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
