"""QState — a quantum-inspired complex phase state for the predictive readout.

The head reads a COMPLEX state

    h_t = a · R · h_{t-1} + W_in · emb[x_t]

where R is a fixed block-diagonal complex unitary: dim/2 independent "qubits", each a 2x2
rotation with a phase (cosθ, sinθ·e^{iφ}). Because R is unitary and a < 1, the closed-form
scan powers a^j·R^j stay bounded (a=0.9, j=384 -> 0.9^384 ~ 4e-18, representable in fp32) —
no fp64 overflow, and the adjoint backward scan is well-conditioned. A real decaying sum's
gradient dies for the far past (a^j); the unitary phase ROTATES the past instead of shrinking
it, so long-range structure keeps a bounded learning signal (the measured bootstrap deadlock
of the linear SSM is structurally avoided).

Capacity: a real decaying sum cannot linearly represent combination/order structure — the
measured gap that kept the linear SSM at the bigram floor (~100 ppl vs the 8M transformer's
~31). QState exposes (i) the phase/interference information of the complex state (Re/Im of the
normalized amplitudes) and (ii) pairwise "entanglement" features Re(h_i · conj(h_j)) on the
first `pairs` amplitudes — a quadratic feature space that can linearly separate conjunctions a
linear map cannot.

Training is exact closed-form (one complex forward scan + one complex adjoint backward scan,
no autograd), matching the EmbSSM module interface so it can be swapped in as
`organism.embssm` when `cfg.embssm_qstate` is set. The mixed head is CE-trained on the combined
logits exactly like EmbSSM, so an uninformative state drives β -> 0 / W_vocab -> ~0 (pure
bigram) — never worse than the bigram.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QState(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        vocab_size: int,
        lr: float = 0.1,
        head_lr: float = 0.05,
        decay: float = 0.9,
        pairs: int = 16,
        seed: int = 0,
    ):
        super().__init__()
        assert dim % 2 == 0, "QState dim must be even (pairs of qubits)"
        self.dim = dim
        self.emb_dim = emb_dim
        self.a = decay
        self.lr = lr
        self.head_lr = head_lr
        self.wd = 1e-4
        self.pairs = min(pairs, dim)
        # feature layout: [Re(h_n); Im(h_n); |h_n|^2 (per-amplitude "measurement probabilities");
        # Re(h_i·conj(h_j)) for i,j < pairs]
        self.fdim = 3 * dim + self.pairs * self.pairs

        # fixed block-diagonal complex unitary R: dim/2 independent 2x2 qubit rotations
        g = torch.Generator().manual_seed(seed)
        self.nb = dim // 2
        theta = torch.rand(self.nb, generator=g) * torch.pi
        phi = torch.rand(self.nb, generator=g) * 2.0 * torch.pi
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.zeros(self.nb, 2, 2, dtype=torch.complex64)
        R[:, 0, 0] = c
        R[:, 0, 1] = s * torch.exp(1j * phi)
        R[:, 1, 0] = -s * torch.exp(-1j * phi)
        R[:, 1, 1] = c
        self.register_buffer("R", R)

        # Near-identity init: the state starts as a rotated/decayed sum of the raw embedding
        # prefix (already bigram-predictive), so W_vocab can learn something real before the
        # routed gradient refines W_in (same bootstrap fix as EmbSSM).
        self.W_in = nn.Parameter(torch.eye(dim, emb_dim) + torch.randn(dim, emb_dim) * 0.05)
        # W_vocab starts at ZERO (neutral channel -> β's gradient starts at zero, no cold-start
        # collapse); learns from the bigram's residual before β grows to amplify it.
        self.W_vocab = nn.Parameter(torch.zeros(vocab_size, self.fdim))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("h", torch.zeros(dim, dtype=torch.complex64))
        self._pcache: dict[tuple[int, torch.device], torch.Tensor] = {}

    # ------------------------------------------------------------------
    def _powers(self, n: int) -> torch.Tensor:
        """Block-diagonal powers P[j] = a^j · R^j for j = 0..n-1, as (n, nb, 2, 2)."""
        key = (n, self.R.device)
        if key in self._pcache:
            return self._pcache[key]
        R = self.R
        P = torch.zeros(n, self.nb, 2, 2, dtype=torch.complex64, device=self.R.device)
        acc = torch.eye(2, dtype=torch.complex64).repeat(self.nb, 1, 1).to(self.R.device)
        a = self.a
        for j in range(n):
            P[j] = (a**j) * acc
            acc = acc @ R
        self._pcache[key] = P
        return P

    def _powers_adj(self, n: int) -> torch.Tensor:
        """Adjoint powers P†[j] = a^j · (R†)^j for the backward scan."""
        key = (n, self.R.device, "adj")
        if key in self._pcache:
            return self._pcache[key]
        Radj = self.R.conj().transpose(-1, -2)
        P = torch.zeros(n, self.nb, 2, 2, dtype=torch.complex64, device=self.R.device)
        acc = torch.eye(2, dtype=torch.complex64).repeat(self.nb, 1, 1).to(self.R.device)
        a = self.a
        for j in range(n):
            P[j] = (a**j) * acc
            acc = acc @ Radj
        self._pcache[key] = P
        return P

    def _unit(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.abs() + 1e-8)

    # ------------------------------------------------------------------
    def scan_window(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form complex forward scan over a window of embeddings. Returns the
        L2-normalized complex state (W, dim), the raw state (W, dim); advances the carry."""
        w = e.shape[0]
        device = e.device
        r = (e.float() @ self.W_in.t()).to(torch.complex64).reshape(w, self.nb, 2)
        P = self._powers(w)
        carry = self.h.clone().reshape(self.nb, 2)
        h_raw = torch.zeros(w, self.dim, dtype=torch.complex64, device=device)
        for j in range(w):
            # carry contributes to h[j] via P[j]; inputs r[s] contribute to h[s+j] via P[j]
            cterm = torch.einsum("bij,bj->bi", P[j], carry).reshape(-1)
            h_raw[j] += cterm
            contrib = torch.einsum("bij,tbj->tbi", P[j], r[: w - j]).reshape(w - j, self.dim)
            h_raw[j:] += contrib
        self.h = h_raw[-1].detach()
        return self._unit(h_raw), h_raw

    def features(self, h_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the (complex) normalized state to real head features: Re/Im amplitudes,
        per-amplitude measurement probabilities |h|^2, and pairwise entanglement features
        Re(h_i·conj(h_j)). Returns (features, pre-activation) — here pass-through, so they
        coincide."""
        f = [h_n.real, h_n.imag, h_n.abs() ** 2]
        p = self.pairs
        if p > 0:
            sub = h_n[:, :p]
            f.append(torch.einsum("bi,bj->bij", sub, sub.conj()).real.reshape(h_n.shape[0], -1))
        return torch.cat(f, dim=-1).float(), None

    def logits(self, h_n: torch.Tensor) -> torch.Tensor:
        feat, _ = self.features(h_n)
        return feat.to(torch.float16) @ self.W_vocab.to(torch.float16).t()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Single-token step (generation / per-token path). Returns the SSM logits (vocab,)."""
        r = e.float() @ self.W_in.t()
        carry = self.h.clone().reshape(self.nb, 2)
        R = self.R
        nxt = (self.a * torch.einsum("bij,bj->bi", R, carry).reshape(-1) + r).to(torch.complex64)
        self.h = nxt.detach()
        hn = self._unit(nxt)
        return self.logits(hn.unsqueeze(0))[0]

    def sdch(self, h: torch.Tensor) -> torch.Tensor:
        """The memory/SDC key: concat(Re, Im) of the complex state -> 2·dim real values
        (matches the fabric's readout dim for dim = readout_dim / 2)."""
        return torch.cat([h.real, h.imag], dim=-1)

    # ------------------------------------------------------------------
    def train_head(self, d_logits: torch.Tensor, h_n: torch.Tensor, mod: float = 1.0) -> None:
        """Update W_vocab by gradient DESCENT on the CE (d_logits = β·(p − onehot)). The old
        add_-sign in the EmbSSM head was the root cause of the "dead channel" (ascent -> the
        head learned confidently-wrong -> β collapsed to the floor)."""
        feat, _ = self.features(h_n)
        w = feat.shape[0]
        dW = (d_logits.float().t() @ feat) / w
        self.W_vocab.data.sub_(self.head_lr * mod * dW.to(self.W_vocab.dtype))

    def dctx_from_head(self, d_ssm: torch.Tensor, h_n: torch.Tensor) -> list[torch.Tensor]:
        """Route the head's CE gradient back to the complex state (single channel -> one element).
        Covers the Re/Im features, the |h|^2 measurement features, and the pairwise
        entanglement features."""
        dF = d_ssm.float() @ self.W_vocab.float()  # (W, fdim)
        dim = self.dim
        p = self.pairs
        d_re = dF[:, :dim].clone()
        d_im = dF[:, dim : 2 * dim].clone()
        d_abs = dF[:, 2 * dim : 3 * dim]  # (W, dim)
        if p > 0:
            dp = dF[:, 3 * dim : 3 * dim + p * p].reshape(-1, p, p)
            sym = dp + dp.transpose(1, 2)  # (W, p, p)
            hn_p = h_n[:, :p]
            d_re[:, :p] += (sym * hn_p.real.unsqueeze(1)).sum(dim=2)
            d_im[:, :p] += (sym * hn_p.imag.unsqueeze(1)).sum(dim=2)
        d_hn = d_re + 2.0 * h_n.real * d_abs
        d_hn = (d_hn + 1j * (d_im + 2.0 * h_n.imag * d_abs)).to(torch.complex64)
        return [d_hn]

    def train_beta(self, grad_beta: torch.Tensor, mod: float = 1.0) -> None:
        self.beta.data.add_(self.lr * mod * grad_beta)
        self.beta.data.clamp_(0.05, 4.0)

    def apply_grad_ctx(self, d_ctx: list[torch.Tensor], e: torch.Tensor, mod: float = 1.0) -> None:
        """Train W_in by the exact complex adjoint backward scan of the recurrence."""
        d = d_ctx[0]
        w = d.shape[0]
        Padj = self._powers_adj(w)
        b = torch.zeros_like(d)
        dj = d.reshape(w, self.nb, 2)
        for j in range(w):
            contrib = torch.einsum("bij,tbj->tbi", Padj[j], dj[j:])
            b[: w - j] += contrib.reshape(w - j, self.dim)
        dW = (b.real.t() @ e.float()) / w  # W_in is real
        self.W_in.data.add_(self.lr * mod * (dW - self.wd * self.W_in.data))

    def reset(self) -> None:
        self.h.zero_()
