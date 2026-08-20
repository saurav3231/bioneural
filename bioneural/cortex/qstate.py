"""QState — a quantum-inspired complex phase state for the predictive readout.

The head reads a COMPLEX state

    h_t = a · R · h_{t-1} + W_in · emb[x_t]

where R is a block-diagonal complex unitary: dim/2 independent "qubits", each a 2x2
rotation with a phase (cosθ, sinθ·e^{iφ}). The angles are trainable (learn=True), so the
state can ADAPT its phase structure to the task via the exact local gradient
dL/dR = Σ_t a·b_t·h_{t-1}^H. Because R is unitary and a < 1, the closed-form
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
        learn: bool = False,
    ):
        super().__init__()
        assert dim % 2 == 0, "QState dim must be even (pairs of qubits)"
        self.dim = dim
        self.emb_dim = emb_dim
        self.a = decay
        self.lr = lr
        self.head_lr = head_lr
        self.wd = 1e-4
        self.learn = learn
        self.pairs = min(pairs, dim)
        # feature layout: [Re(h_n); Im(h_n); |h_n|^2 (per-amplitude "measurement probabilities");
        # Re(h_i·conj(h_j)) for i,j < pairs]
        self.fdim = 3 * dim + self.pairs * self.pairs

        # block-diagonal complex unitary R: dim/2 independent 2x2 qubit rotations. The angles
        # (θ, φ) are trainable when learn=True, so the state can ADAPT its phase structure to the
        # task (the gradient is exact and local: dL/dR = Σ_t a·b_t·h_{t-1}^H). Fixed otherwise.
        g = torch.Generator().manual_seed(seed)
        self.nb = dim // 2
        self.theta = nn.Parameter(torch.rand(self.nb, generator=g) * torch.pi)
        self.phi = nn.Parameter(torch.rand(self.nb, generator=g) * 2.0 * torch.pi)
        if not learn:
            self.theta.requires_grad_(False)
            self.phi.requires_grad_(False)
        self.register_buffer("R", self._build_R())
        self._pcache: dict[tuple[int, torch.device], torch.Tensor] = {}

        # Near-identity init: the state starts as a rotated/decayed sum of the raw embedding
        # prefix (already bigram-predictive), so W_vocab can learn something real before the
        # routed gradient refines W_in (same bootstrap fix as EmbSSM).
        self.W_in = nn.Parameter(torch.eye(dim, emb_dim) + torch.randn(dim, emb_dim) * 0.05)
        # W_vocab starts at ZERO (neutral channel -> β's gradient starts at zero, no cold-start
        # collapse); learns from the bigram's residual before β grows to amplify it.
        self.W_vocab = nn.Parameter(torch.zeros(vocab_size, self.fdim))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("h", torch.zeros(dim, dtype=torch.complex64))
        self.register_buffer("_hprev", torch.zeros(0, dim, dtype=torch.complex64))
        self.chunk = 32  # closed-form scan chunk: loop over 32 lags, batched across chunks

    def _build_R(self) -> torch.Tensor:
        """Block-diagonal R from the current angles: 2x2 blocks
        [[cosθ, sinθ·e^{iφ}], [-sinθ·e^{-iφ}, cosθ]] — unitary by construction."""
        nb = self.nb
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        ep = torch.exp(1j * self.phi)
        R = torch.zeros(nb, 2, 2, dtype=torch.complex64, device=self.theta.device)
        R[:, 0, 0] = c
        R[:, 0, 1] = s * ep
        R[:, 1, 0] = -s * torch.conj(ep)
        R[:, 1, 1] = c
        return R

    def _refresh_R(self) -> None:
        """Sync the R buffer with the trainable angles and invalidate the power cache."""
        with torch.no_grad():
            self.R.copy_(self._build_R())
        self._pcache.clear()

    # ------------------------------------------------------------------
    def _toeplitz(self, adj: bool = False, backward: bool = False) -> torch.Tensor:
        """Block-Toeplitz convolution matrix T (nb, 2C, 2C): the cross-lag convolution becomes
        one batched matmul T @ r̃ (r̃ interleaves the 2 qubit dims across the C positions:
        index = pos·2 + dim). Two orientations: forward (h[j] = Σ_{i≤j} P[j-i]·r[i],
        lower-triangular) and backward (G[s] = Σ_{t≥s} P[t-s]·r[t], upper-triangular — the
        adjoint scan's r-gradient). Fully vectorized via the closed-form block powers
        P[j] = a^j·[[cos(jθ), sin(jθ)e^{iφ}], [-sin(jθ)e^{-iφ}, cos(jθ)]]; T depends only on R,
        so it is cached and rebuilt only when the angles change."""
        key = ("T", self.R.device, adj, backward)
        if key in self._pcache:
            return self._pcache[key]
        C = self.chunk
        rows = torch.arange(C, device=self.R.device)[:, None]  # (C,1) block row
        cols = torch.arange(C, device=self.R.device)[None, :]  # (1,C) block col
        if backward:
            d = (cols - rows).clamp(min=0).float()  # (C,C), lag = t - s
            mask = (rows <= cols).float()  # upper-triangular support
        else:
            d = (rows - cols).clamp(min=0).float()  # (C,C), lag = j - i
            mask = (rows >= cols).float()  # lower-triangular support
        th = self.theta.detach()[:, None, None]  # (nb,1,1)
        dd = d[None]  # (1,C,C)
        c = torch.cos(dd * th) * mask  # (nb,C,C)
        s = torch.sin(dd * th) * mask
        ep = torch.exp(1j * self.phi.detach())[:, None, None]  # (nb,1,1)
        ad = (self.a**d[None]) * mask  # (1,C,C)
        T = torch.zeros(self.nb, C, C, 2, 2, dtype=torch.complex64, device=self.R.device)
        T[:, :, :, 0, 0] = c
        T[:, :, :, 0, 1] = (-s * ep) if adj else (s * ep)
        T[:, :, :, 1, 0] = (s * torch.conj(ep)) if adj else (-s * torch.conj(ep))
        T[:, :, :, 1, 1] = c
        T = T * ad[..., None, None]  # (nb,C,C,2,2)
        T = T.permute(0, 1, 3, 2, 4).reshape(self.nb, 2 * C, 2 * C)  # (nb, 2C, 2C)
        self._pcache[key] = T
        return T

    def _conv(self, r: torch.Tensor, adj: bool = False, backward: bool = False) -> torch.Tensor:
        """Batched Toeplitz convolution over chunks. r is (M, C, nb, 2); forward:
        Σ_{i≤j} P[j-i]·r[m, i]; backward: Σ_{t≥s} P[t-s]·r[m, t] (adjoint r-gradient)."""
        T = self._toeplitz(adj, backward)
        r̃ = r.permute(0, 2, 1, 3).reshape(r.shape[0], self.nb, 2 * self.chunk)  # (M, nb, 2C)
        out = torch.einsum("blk,mbk->mbl", T, r̃)  # (M, nb, 2C)
        return out.reshape(r.shape[0], self.nb, self.chunk, 2).permute(0, 2, 1, 3)

    def _powers(self, n: int) -> torch.Tensor:
        """Block-diagonal powers P[j] = a^j · R^j for j = 0..n-1, as (n, nb, 2, 2). Closed form:
        a 2x2 Givens-with-phase block satisfies R_b^j = [[cos(jθ), sin(jθ)e^{iφ}],
        [-sin(jθ)e^{-iφ}, cos(jθ)]], so the whole power table is one vectorized op (no sequential
        matmul) — keeps the learned-R rebuild cheap when the angles change."""
        key = (n, self.R.device)
        if key in self._pcache:
            return self._pcache[key]
        j = torch.arange(n, dtype=torch.float32, device=self.R.device)
        theta = self.theta.detach()
        phi = self.phi.detach()
        jth = j[None, :] * theta[:, None]  # (nb, n)
        cT = torch.cos(jth).t()  # (n, nb)
        sT = torch.sin(jth).t()
        ep = torch.exp(1j * phi)[None, :]  # (1, nb)
        aj = (self.a**j)[:, None]  # (n, 1)
        P = torch.zeros(n, self.nb, 2, 2, dtype=torch.complex64, device=self.R.device)
        P[:, :, 0, 0] = aj * cT
        P[:, :, 0, 1] = aj * sT * ep
        P[:, :, 1, 0] = -aj * sT * torch.conj(ep)
        P[:, :, 1, 1] = aj * cT
        self._pcache[key] = P
        return P

    def _powers_adj(self, n: int) -> torch.Tensor:
        """Adjoint powers P†[j] = a^j · (R†)^j for the backward scan. (R†)^j block is
        [[cos(jθ), -sin(jθ)e^{-iφ}], [sin(jθ)e^{iφ}, cos(jθ)]], one vectorized op."""
        key = (n, self.R.device, "adj")
        if key in self._pcache:
            return self._pcache[key]
        j = torch.arange(n, dtype=torch.float32, device=self.R.device)
        theta = self.theta.detach()
        phi = self.phi.detach()
        jth = j[None, :] * theta[:, None]  # (nb, n)
        cT = torch.cos(jth).t()  # (n, nb)
        sT = torch.sin(jth).t()
        ep = torch.exp(1j * phi)[None, :]  # (1, nb)
        aj = (self.a**j)[:, None]  # (n, 1)
        P = torch.zeros(n, self.nb, 2, 2, dtype=torch.complex64, device=self.R.device)
        P[:, :, 0, 0] = aj * cT
        P[:, :, 0, 1] = -aj * sT * ep
        P[:, :, 1, 0] = aj * sT * torch.conj(ep)
        P[:, :, 1, 1] = aj * cT
        self._pcache[key] = P
        return P

    def _unit(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.abs() + 1e-8)

    # ------------------------------------------------------------------
    def scan_window(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form complex forward scan over a window of embeddings, CHUNKED for speed.
        The window is split into M = w/C chunks; the internal cross-lag convolution runs as one
        batched loop over the C lags (vectorized across chunks), and only the M-chunk carry chain
        is sequential. Returns the L2-normalized complex state (W, dim), the raw state (W, dim);
        advances the carry."""
        w = e.shape[0]
        nb = self.nb
        C = self.chunk
        M = max(1, (w + C - 1) // C)
        if self.learn:
            self._refresh_R()
        r = (e.float() @ self.W_in.t()).to(torch.complex64).reshape(w, nb, 2)
        if M * C != w:
            r = torch.nn.functional.pad(r, (0, 0, 0, 0, 0, M * C - w))
        r = r.reshape(M, C, nb, 2)
        h = self._conv(r)  # one cached Toeplitz matmul instead of a 32-lag einsum loop
        P = self._powers(C + 1)  # (C+1, nb, 2, 2); carry uses [1:]
        carry = self.h.clone().reshape(nb, 2)
        carry0 = carry  # the state BEFORE this window = h_{-1} for t=0 (needed for the R grad)
        for m in range(M):
            h[m] += torch.einsum("cbij,bj->cbi", P[1:], carry)
            carry = h[m, C - 1]
        hf = h.reshape(M * C, nb, 2)[:w].reshape(w, self.dim)
        self.h = hf[-1].detach()
        self._hprev = torch.cat([carry0.reshape(1, self.dim), hf[:-1]], dim=0).detach()
        return self._unit(hf), hf

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
        if self.learn:
            self._refresh_R()
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
        """Train W_in by the exact complex adjoint backward scan of the recurrence (chunked).

        dL/dr[m, i] = Σ_{j≥i} P†[j−i]·dL/dh[m, j], with the cross-chunk carry gradient
        dL/dcarry_m = Σ_j P†[j+1]·dL/dh[m, j]. The internal adjoint is batched over chunks; the
        carry chain (M steps) contributes the tail terms P†[C−1−i]·dL/dcarry_{m+1}."""
        d = d_ctx[0]
        w = d.shape[0]
        nb = self.nb
        C = self.chunk
        M = max(1, (w + C - 1) // C)
        dj = d.reshape(w, nb, 2)
        if M * C != w:
            dj = torch.nn.functional.pad(dj, (0, 0, 0, 0, 0, M * C - w))
        dj = dj.reshape(M, C, nb, 2)
        G = self._conv(dj, adj=True, backward=True)  # batched adjoint r-gradient (P†, upper-tri)
        Padj = self._powers_adj(C + 1)  # carry terms use P†[1:], P†[C]
        c = torch.einsum("mcbj,cbij->mbi", dj, Padj[1:])  # (M, nb, 2) = dL/dcarry_m
        acc = torch.zeros(nb, 2, dtype=torch.complex64, device=d.device)
        for m in reversed(range(M)):
            tail = torch.einsum("cbij,bj->cbi", torch.flip(Padj[:C], dims=[0]), acc)
            G[m] += tail
            acc = c[m] + torch.einsum("bij,bj->bi", Padj[C], acc)
        Gf = G.reshape(M * C, nb, 2)[:w].reshape(w, self.dim)
        dW = (Gf.real.t() @ e.float()) / w  # W_in is real
        self.W_in.data.add_(self.lr * mod * (dW - self.wd * self.W_in.data))
        if self.learn:
            self._update_angles(Gf, w, mod)

    def _update_angles(self, Gf: torch.Tensor, w: int, mod: float) -> None:
        """Train the qubit angles by the exact local gradient dL/dR = Σ_t a·b_t·h_{t-1}^H
        (b_t = dL/dh_t = the adjoint state gradient) projected onto θ and φ. Gradient DESCENT
        (verified against autograd to ~1e-4; strictly better or equal to the mirrored W_in
        convention on the 2nd-order organism test)."""
        nb = self.nb
        G2 = Gf.reshape(w, nb, 2)
        hp = self._hprev.reshape(w, nb, 2)
        dR = (self.a / w) * torch.einsum("tbj,tbk->bjk", G2, hp.conj())  # ∂L/∂R, (nb,2,2)
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        ep = torch.exp(1j * self.phi)
        dR_dtheta = torch.zeros(nb, 2, 2, dtype=torch.complex64, device=self.theta.device)
        dR_dtheta[:, 0, 0] = -s
        dR_dtheta[:, 0, 1] = c * ep
        dR_dtheta[:, 1, 0] = -c * torch.conj(ep)
        dR_dtheta[:, 1, 1] = -s
        dR_dphi = torch.zeros(nb, 2, 2, dtype=torch.complex64, device=self.theta.device)
        dR_dphi[:, 0, 1] = 1j * s * ep
        dR_dphi[:, 1, 0] = 1j * s * torch.conj(ep)
        dtheta = torch.einsum("bij,bij->b", dR.conj(), dR_dtheta).real
        dphi = torch.einsum("bij,bij->b", dR.conj(), dR_dphi).real
        self.theta.data.sub_(self.lr * mod * (dtheta - self.wd * self.theta.data))
        self.phi.data.sub_(self.lr * mod * (dphi - self.wd * self.phi.data))

    def reset(self) -> None:
        self.h.zero_()
        self._hprev = torch.zeros(0, self.dim, dtype=torch.complex64, device=self.h.device)
