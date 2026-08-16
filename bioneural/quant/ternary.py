"""Ternary weight parameters with latent shadow magnitudes.

Each weight keeps an fp16 *latent* value; the ternary value is its sign/deadzone quantization.
Small local updates accumulate in the latent and only rarely flip the ternary state (flip events
are cheap because they are rare). This is BitNet-style training made online and local.

Materialization is cached and invalidated by a version counter so the forward hot path is a plain
fast fp16 matmul.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioneural.config import QuantConfig
from bioneural.quant.kernels import materialize_ternary, ternary_matmul_triton


class TernaryParam(nn.Module):
    """A ternary weight matrix parameterized by latent shadow magnitudes."""

    def __init__(
        self, shape: tuple[int, int], config: QuantConfig | None = None, sparsity: float = 0.0
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.config = config or QuantConfig()
        self.register_buffer("latent", torch.randn(shape) * 0.05)
        if sparsity > 0:
            mask = torch.rand(shape) > sparsity
            self.latent *= mask
        self.register_buffer("version", torch.tensor(0, dtype=torch.long))
        self.register_buffer("flip_count", torch.tensor(0, dtype=torch.long))
        self._cache: torch.Tensor | None = None
        self._flip_bookkeeping = torch.tensor(0.0)

    # ------------------------------------------------------------------
    def _materialized(self) -> torch.Tensor:
        if self._cache is None:
            w_t, _ = materialize_ternary(
                self.latent,
                group_size=self.config.group_size,
                deadzone=self.config.deadzone,
                scale_mode=self.config.scale_mode,
            )
            self._cache = w_t.detach()
        return self._cache

    # ------------------------------------------------------------------
    def materialized(self) -> torch.Tensor:
        """The cached ternary form (fp16, group-scaled) of the latent shadows."""
        return self._materialized()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialized()
        if x.device != w.device:
            w = w.to(x.device)
        out = ternary_matmul_triton(x, w, self.config)
        if out is None:
            if x.dtype != torch.float16:
                x = x.to(torch.float16)
            out = x @ w.T
        return out.float()

    # ------------------------------------------------------------------
    def update_latent(self, grad: torch.Tensor, lr: float = 1.0, count_flips: bool = True) -> int:
        """Accumulate a local (backprop-free) gradient into the latent shadows.

        Returns the number of ternary flips this update caused (a stability diagnostic).
        `count_flips=False` skips the two full-matrix sign passes (hot path).
        """
        g = grad.to(self.latent.dtype) * lr
        flips = 0
        if count_flips:
            old = self._ternary_signs()
            self.latent = self.latent + g
            new = self._ternary_signs()
            flips = int((old != new).sum().item())
            self.flip_count += flips
        else:
            self.latent = self.latent + g
        self.version += 1
        self._cache = None
        return flips

    def _ternary_signs(self) -> torch.Tensor:
        w = self.latent
        thresh = self.config.deadzone * (w.abs().amax(dim=1, keepdim=True).clamp_min(1e-9))
        return torch.where(w.abs() > thresh, w.sign(), torch.zeros_like(w))

    # ------------------------------------------------------------------
    def apply_decay(self, factor: float) -> None:
        self.latent = self.latent * factor
        self.version += 1
        self._cache = None

    def stats(self) -> dict[str, float]:
        w = self._materialized()
        nz = (w != 0).float().mean().item()
        plus = (w > 0).float().mean().item()
        neg = (w < 0).float().mean().item()
        return {
            "density": nz,
            "frac_plus": plus,
            "frac_neg": neg,
            "flip_count": int(self.flip_count.item()),
        }

    def extra_repr(self) -> str:
        return f"shape={self.shape}, group_size={self.config.group_size}, deadzone={self.config.deadzone}"
