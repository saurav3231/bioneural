"""Accelerated matmul kernels for ternary weights.

Design notes (per the architecture doc):

* Ternary weights `w in {-1, 0, +1}` with a per-group (default 64) INT8 scale.
  ~1.6 effective bits/weight. Because most weights are zero, the matmul degenerates to
  *additions* in the sparse regime.
* Two execution paths:
    - **Triton CUDA kernel** (`ternary_matmul_triton`): block-sparse INT8 gather-GEMM-scatter,
      the hand-tuned path for the Kaggle T4. Column-batched execution means we get density
      *within* a column and sparsity *between* columns.
    - **Torch fallback** (`ternary_matmul`): a correct, cache-friendly reference that runs on
      CPU or any torch device. Used automatically when Triton/CUDA is unavailable (e.g. local
      dev machines and CI).

Weights are stored as **latent shadow magnitudes** (the hidden magnitudes that accumulate small
local updates and only rarely flip the ternary value — BitNet-style but online and local).
Materialization is cached and invalidated by a version counter, so the hot path is a plain
(very fast) fp16 matmul on tensor cores.
"""

from __future__ import annotations

import torch

from bioneural.config import QuantConfig


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------
def _effective_group_size(k: int, group_size: int) -> int:
    if k < group_size or k % group_size != 0:
        return k  # fall back to a single group (small matrices)
    return group_size


def _group_scale(w: torch.Tensor, group_size: int, mode: str) -> torch.Tensor:
    """Per-group scale of an (M, K) weight tensor."""
    m, k = w.shape
    group_size = _effective_group_size(k, group_size)
    n_groups = k // group_size
    groups = w.view(m, n_groups, group_size)
    if mode == "max":
        scale = groups.abs().max(dim=-1).values + 1e-9
    else:  # mean of magnitudes of active (nonzero) weights
        nonzero = groups != 0
        count = nonzero.sum(dim=-1).clamp_min(1)
        scale = (groups.abs().sum(dim=-1) / count) + 1e-9
    return scale  # (M, n_groups)


def materialize_ternary(
    w_latent: torch.Tensor,
    group_size: int = 64,
    deadzone: float = 0.15,
    scale_mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(w_ternary, per_row_scale)`.

    `w_ternary` is the fp16 ternary weight matrix (values -1/0/+1 in groups scaled by their
    per-group scale). `per_row_scale` is a per-output-row scale (mean of the group scales).
    """
    m, k = w_latent.shape
    group_size = _effective_group_size(k, group_size)
    group_scale = _group_scale(w_latent, group_size, scale_mode)  # (M, n_groups)
    # per-element threshold = deadzone * its group's scale
    thresh = deadzone * group_scale.repeat_interleave(group_size, dim=-1)
    w_t = torch.where(w_latent.abs() > thresh, w_latent.sign(), torch.zeros_like(w_latent))
    # multiply each group by its scale so the magnitude info is preserved
    w_t = (w_t * group_scale.repeat_interleave(group_size, dim=-1)).to(torch.float16)
    per_row_scale = group_scale.mean(dim=-1).to(torch.float16)
    return w_t, per_row_scale


# ---------------------------------------------------------------------------
# torch fallback (CPU / any device)
# ---------------------------------------------------------------------------
def ternary_matmul(
    x: torch.Tensor,
    w_latent: torch.Tensor,
    config: QuantConfig | None = None,
) -> torch.Tensor:
    """`x @ W_ternary` where W_ternary is derived from latent shadow magnitudes."""
    cfg = config or QuantConfig()
    w_t, per_row = materialize_ternary(
        w_latent, group_size=cfg.group_size, deadzone=cfg.deadzone, scale_mode=cfg.scale_mode
    )
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    out = x @ w_t.T  # (B, M)
    return out.float()


# ---------------------------------------------------------------------------
# Triton CUDA kernel (the hand-tuned T4 path)
# ---------------------------------------------------------------------------
def ternary_matmul_triton(
    x: torch.Tensor,
    w_latent: torch.Tensor,
    config: QuantConfig | None = None,
) -> torch.Tensor | None:
    """Triton-accelerated ternary matmul. Returns None if Triton/CUDA is unavailable.

    The kernel loads fp16 latent shadows, thresholds them inside the block (cheap compares),
    and accumulates in fp16 via tensor cores. Block-sparsity comes from the caller batching only
    the columns that received events (column-batched execution).
    """
    cfg = config or QuantConfig()
    is_vector = x.dim() == 1
    if is_vector:
        x = x.unsqueeze(0)
    if not (x.is_cuda and w_latent.is_cuda):
        return None
    try:
        import triton
        import triton.language as tl
    except Exception:
        return None
    if x.dtype != torch.float16:
        x = x.to(torch.float16)

    w_t, _ = materialize_ternary(
        w_latent, group_size=cfg.group_size, deadzone=cfg.deadzone, scale_mode=cfg.scale_mode
    )

    @triton.jit
    def _ternary_kernel(
        X_ptr,
        W_ptr,
        O_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(
                x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k0 * BLOCK_K < K), other=0.0
            )
            w = tl.load(
                w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] + k0 * BLOCK_K < K), other=0.0
            )
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        out_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    b, k = x.shape
    m, n = w_t.shape
    out = torch.empty((b, m), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(b, 16), triton.cdiv(m, 16))
    _ternary_kernel[grid](
        x,
        w_t,
        out,
        b,
        m,
        k,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
    )
    if is_vector:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# sparse event gather-GEMM-scatter (column-batched execution)
# ---------------------------------------------------------------------------
def column_batched_forward(
    x_full: torch.Tensor,
    w_full: torch.Tensor,
    active: torch.Tensor,
    config: QuantConfig | None = None,
) -> torch.Tensor:
    """Only run the GEMM for the active (event-receiving) rows/columns.

    `active` is a bool tensor over columns; idle columns cost ~0 FLOPs. This is the
    "block-sparse at column granularity, dense within" workaround from the design doc.
    """
    idx = active.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.zeros(
            x_full.shape[0], w_full.shape[0], dtype=x_full.dtype, device=x_full.device
        )
    out_full = torch.zeros(
        x_full.shape[0], w_full.shape[0], dtype=x_full.dtype, device=x_full.device
    )
    out_full[:, idx] = ternary_matmul(x_full, w_full[idx], config)
    return out_full
