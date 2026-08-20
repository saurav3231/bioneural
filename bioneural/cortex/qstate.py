"""QState — a quantum-inspired complex phase state for the predictive readout.

The head reads a COMPLEX state per channel c

    h_t^c = a_c · R_c · h_{t-1}^c + W_in^c · emb[x_t]

where R_c is a block-diagonal complex unitary: dim/2 independent "qubits", each a 2x2
rotation with a phase (cosθ, sinθ·e^{iφ}). The angles are trainable (learn=True), so the
state can ADAPT its phase structure to the task via the exact local gradient
dL/dR = Σ_t a·b_t·h_{t-1}^H. Several channels at different leakage decays (short/mid/long)
can be stacked (decays=() -> the single `decay`), each with its own R and W_in; the head
reads the CONCATENATION, exposing multiple time scales to a linear map.

Because R is unitary and a < 1, the closed-form scan powers a^j·R^j stay bounded
(a=0.9, j=384 -> 0.9^384 ~ 4e-18, representable in fp32) — no fp64 overflow, and the
adjoint backward scan is well-conditioned. A real decaying sum's gradient dies for the far
past (a^j); the unitary phase ROTATES the past instead of shrinking it, so long-range
structure keeps a bounded learning signal.

Capacity: a real decaying sum cannot linearly represent combination/order structure. QState
exposes (i) the phase/interference information of the complex state (Re/Im of the normalized
amplitudes) and (ii) pairwise "entanglement" features Re(h_i · conj(h_j)) on the first
`pairs` amplitudes — a quadratic feature space that can linearly separate conjunctions a
linear map cannot.

Training is exact closed-form (complex forward scan + complex adjoint backward scan, no
autograd), matching the EmbSSM module interface so it can be swapped in as
`organism.embssm` when `cfg.embssm_qstate` is set. The mixed head is CE-trained on the
combined logits exactly like EmbSSM, so an uninformative state drives β -> 0 / W_vocab ->
~0 (pure bigram) — never worse than the bigram.

The cross-lag convolutions are block-Toeplitz matrix products (cached, rebuilt only when the
learned angles change), so each scan is one batched matmul + an M-step carry chain instead of
a C-lag einsum loop.
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
        decays: tuple[float, ...] = (),
        slim: bool = False,
    ):
        super().__init__()
        assert dim % 2 == 0, "QState dim must be even (pairs of qubits)"
        self.dim = dim  # per-channel complex dim
        self.emb_dim = emb_dim
        self.decays = tuple(decays) if decays else (decay,)
        self.nch = len(self.decays)
        self.a = self.decays[0]  # legacy single-decay accessor
        self.lr = lr
        self.head_lr = head_lr
        self.wd = 1e-4
        self.learn = learn
        self.pairs = min(pairs, dim)
        self.slim = slim
        # Feature layout. The MID-decay channel (closest to 0.9, the SDC/SDC-key carrier) keeps
        # the full feature set [Re(h); Im(h); |h|^2; pairwise Re(h_i·conj(h_j)) i,j<pairs].
        # When slim=True the OTHER channels contribute only [Re(h); Im(h)] (2·dim) so stacking
        # timescales does not triple the head: fdim = (nch-1)·2·dim + (3·dim + pairs²) instead of
        # nch·(3·dim + pairs²). Redundant near-identity channels otherwise dilute W_vocab and the
        # full mix collapses back onto the embedding anchor (measured: fdim 1920 -> full 104 ppl
        # vs 85 single-channel).
        self.mid = min(range(self.nch), key=lambda c: abs(self.decays[c] - 0.9))
        self.fdim_c = 3 * dim + self.pairs * self.pairs  # full feature block (mid channel)
        if slim:
            self.fdim = (self.nch - 1) * 2 * dim + self.fdim_c
        else:
            self.fdim = self.nch * self.fdim_c

        # block-diagonal complex unitary R per channel: dim/2 independent 2x2 qubit rotations.
        # The angles (θ, φ) are trainable when learn=True, so R adapts to the task.
        g = torch.Generator().manual_seed(seed)
        self.nb = dim // 2
        self.theta = nn.Parameter(torch.rand(self.nch, self.nb, generator=g) * torch.pi)
        self.phi = nn.Parameter(torch.rand(self.nch, self.nb, generator=g) * 2.0 * torch.pi)
        if not learn:
            self.theta.requires_grad_(False)
            self.phi.requires_grad_(False)
        self.register_buffer("R", self._build_R())
        self._pcache: dict[tuple, torch.Tensor] = {}

        # Near-identity init: each state starts as a rotated/decayed sum of the raw embedding
        # prefix (already bigram-predictive), so W_vocab can learn something real before the
        # routed gradient refines W_in (same bootstrap fix as EmbSSM).
        eye = torch.eye(dim, emb_dim)
        self.W_in = nn.Parameter(
            eye.unsqueeze(0).expand(self.nch, dim, emb_dim).clone()
            + torch.randn(self.nch, dim, emb_dim) * 0.05
        )
        # W_vocab starts at ZERO (neutral channel -> β's gradient starts at zero, no cold-start
        # collapse); learns from the bigram's residual before β grows to amplify it.
        self.W_vocab = nn.Parameter(torch.zeros(vocab_size, self.fdim))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("h", torch.zeros(self.nch, dim, dtype=torch.complex64))
        self.register_buffer("_hprev", torch.zeros(self.nch, 0, dim, dtype=torch.complex64))
        self.chunk = 32  # closed-form scan chunk

    def _build_R(self) -> torch.Tensor:
        """Block-diagonal R (nch, nb, 2, 2): [[cosθ, sinθ·e^{iφ}], [-sinθ·e^{-iφ}, cosθ]]."""
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        ep = torch.exp(1j * self.phi)
        R = torch.zeros(self.nch, self.nb, 2, 2, dtype=torch.complex64, device=self.theta.device)
        R[:, :, 0, 0] = c
        R[:, :, 0, 1] = s * ep
        R[:, :, 1, 0] = -s * torch.conj(ep)
        R[:, :, 1, 1] = c
        return R

    def _refresh_R(self) -> None:
        """Sync the R buffer with the trainable angles and invalidate the power cache."""
        with torch.no_grad():
            self.R.copy_(self._build_R())
        self._pcache.clear()

    # ------------------------------------------------------------------
    def _toeplitz(self, adj: bool = False, backward: bool = False) -> torch.Tensor:
        """Block-Toeplitz convolution matrices T (nch, nb, 2C, 2C): the cross-lag convolution
        becomes one batched matmul T @ r̃ (r̃ interleaves the 2 qubit dims across the C
        positions: index = pos·2 + dim). Two orientations: forward (h[j] = Σ_{i≤j} P[j-i]·r[i],
        lower-triangular) and backward (G[s] = Σ_{t≥s} P[t-s]·r[t], upper-triangular — the
        adjoint scan's r-gradient). Fully vectorized via the closed-form block powers
        P[j] = a^j·[[cos(jθ), sin(jθ)e^{iφ}], [-sin(jθ)e^{-iφ}, cos(jθ)]]; T depends only on the
        angles, so it is cached and rebuilt only when those change."""
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
        th = self.theta.detach()[:, :, None, None]  # (nch, nb, 1, 1)
        dd = d[None, None]  # (1,1,C,C)
        c = torch.cos(dd * th) * mask  # (nch, nb, C, C)
        s = torch.sin(dd * th) * mask
        ep = torch.exp(1j * self.phi.detach())[:, :, None, None]  # (nch, nb, 1, 1)
        a = torch.tensor(self.decays, device=self.R.device)[:, None, None, None]  # (nch,1,1,1)
        ad = (a**d[None, None]) * mask  # (nch,1,C,C)
        T = torch.zeros(self.nch, self.nb, C, C, 2, 2, dtype=torch.complex64, device=self.R.device)
        T[:, :, :, :, 0, 0] = c
        T[:, :, :, :, 0, 1] = (-s * ep) if adj else (s * ep)
        T[:, :, :, :, 1, 0] = (s * torch.conj(ep)) if adj else (-s * torch.conj(ep))
        T[:, :, :, :, 1, 1] = c
        T = T * ad[..., None, None]  # (nch, nb, C, C, 2, 2)
        T = T.permute(0, 1, 2, 4, 3, 5).reshape(self.nch, self.nb, 2 * C, 2 * C)
        self._pcache[key] = T
        return T

    def _conv(self, r: torch.Tensor, adj: bool = False, backward: bool = False) -> torch.Tensor:
        """Batched Toeplitz convolution over chunks and channels. r is (M, C, nch, nb, 2);
        forward: Σ_{i≤j} P[j-i]·r[m, i]; backward: Σ_{t≥s} P[t-s]·r[m, t] (adjoint r-gradient)."""
        T = self._toeplitz(adj, backward)  # (nch, nb, 2C, 2C)
        r̃ = r.permute(0, 2, 3, 1, 4).reshape(r.shape[0], self.nch, self.nb, 2 * self.chunk)
        out = torch.einsum("cblk,mcbk->mcbl", T, r̃)  # (M, nch, nb, 2C)
        return out.reshape(r.shape[0], self.nch, self.nb, self.chunk, 2).permute(0, 3, 1, 2, 4)

    def _powers(self, n: int) -> torch.Tensor:
        """Block-diagonal powers P[j] = a^j · R^j for j = 0..n-1, as (nch, n, nb, 2, 2).
        Closed form: R_b^j = [[cos(jθ), sin(jθ)e^{iφ}], [-sin(jθ)e^{-iφ}, cos(jθ)]]."""
        key = (n, self.R.device)
        if key in self._pcache:
            return self._pcache[key]
        j = torch.arange(n, dtype=torch.float32, device=self.R.device)
        theta = self.theta.detach()
        phi = self.phi.detach()
        jth = j[None, None, :] * theta[:, :, None]  # (nch, nb, n)
        cT = torch.cos(jth).transpose(2, 1)  # (nch, n, nb)
        sT = torch.sin(jth).transpose(2, 1)
        ep = torch.exp(1j * phi)[:, None, :]  # (nch, 1, nb)
        a = torch.tensor(self.decays, device=self.R.device)[:, None]
        aj = (a**j[None, :])[:, :, None]  # (nch, n, 1)
        P = torch.zeros(self.nch, n, self.nb, 2, 2, dtype=torch.complex64, device=self.R.device)
        P[:, :, :, 0, 0] = aj * cT
        P[:, :, :, 0, 1] = aj * sT * ep
        P[:, :, :, 1, 0] = -aj * sT * torch.conj(ep)
        P[:, :, :, 1, 1] = aj * cT
        self._pcache[key] = P
        return P

    def _powers_adj(self, n: int) -> torch.Tensor:
        """Adjoint powers P†[j] = a^j · (R†)^j; block [[cos(jθ), -sin(jθ)e^{iφ}],
        [sin(jθ)e^{-iφ}, cos(jθ)]]."""
        key = (n, self.R.device, "adj")
        if key in self._pcache:
            return self._pcache[key]
        j = torch.arange(n, dtype=torch.float32, device=self.R.device)
        theta = self.theta.detach()
        phi = self.phi.detach()
        jth = j[None, None, :] * theta[:, :, None]  # (nch, nb, n)
        cT = torch.cos(jth).transpose(2, 1)  # (nch, n, nb)
        sT = torch.sin(jth).transpose(2, 1)
        ep = torch.exp(1j * phi)[:, None, :]  # (nch, 1, nb)
        a = torch.tensor(self.decays, device=self.R.device)[:, None]
        aj = (a**j[None, :])[:, :, None]  # (nch, n, 1)
        P = torch.zeros(self.nch, n, self.nb, 2, 2, dtype=torch.complex64, device=self.R.device)
        P[:, :, :, 0, 0] = aj * cT
        P[:, :, :, 0, 1] = -aj * sT * ep
        P[:, :, :, 1, 0] = aj * sT * torch.conj(ep)
        P[:, :, :, 1, 1] = aj * cT
        self._pcache[key] = P
        return P

    def _unit(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.abs() + 1e-8)

    # ------------------------------------------------------------------
    def scan_window(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form complex forward scan over a window of embeddings, CHUNKED. Returns the
        L2-normalized complex state (W, nch·dim), the raw state (W, nch·dim); advances the
        per-channel carries."""
        w = e.shape[0]
        nb = self.nb
        C = self.chunk
        M = max(1, (w + C - 1) // C)
        if self.learn:
            self._refresh_R()
        r = torch.einsum("wd,cdk->wck", e.float(), self.W_in.transpose(1, 2)).to(
        torch.complex64
    )  # (w, nch, dim)
        r = r.reshape(w, self.nch, nb, 2)
        if M * C != w:
            r = torch.nn.functional.pad(r, (0, 0, 0, 0, 0, 0, M * C - w, 0))
        r = r.reshape(M, C, self.nch, nb, 2)
        h = self._conv(r)  # internal convolution, (M, C, nch, nb, 2)
        P = self._powers(C + 1)  # (nch, C+1, nb, 2, 2); carry uses [1:]
        carry = self.h.reshape(self.nch, nb, 2).clone()  # (nch, nb, 2)
        carry0 = carry
        for m in range(M):
            h[m] += torch.einsum("cjbik,cbk->cjbi", P[:, 1:], carry).permute(1, 0, 2, 3)
            carry = h[m, C - 1]
        hf = h.reshape(M * C, self.nch, nb, 2)[:w].reshape(w, self.nch * self.dim)
        self.h = hf[-1].reshape(self.nch, self.dim).detach()
        self._hprev = torch.cat(
            [carry0.reshape(1, self.nch, self.dim), hf[:-1].reshape(w - 1, self.nch, self.dim)],
            dim=0,
        ).detach()
        return self._unit(hf), hf

    def features(self, h_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the (complex) normalized state to real head features. The mid-decay channel
        contributes [Re, Im, |h|^2, pairwise]; when slim=True the other channels contribute only
        [Re, Im] (their time-scale phase info) so the head stays small."""
        w = h_n.shape[0]
        hw = h_n.reshape(w, self.nch, self.dim)
        base = torch.cat([hw.real, hw.imag, hw.abs() ** 2], dim=-1)  # (w, nch, 3·dim)
        p = self.pairs
        if p > 0:
            sub = hw[:, :, :p]
            f_pairs = torch.einsum("wci,wcj->wcij", sub, sub.conj()).real.reshape(
                w, self.nch, -1
            )
            full = torch.cat([base, f_pairs], dim=-1)  # (w, nch, fdim_c)
        else:
            full = base
        if not self.slim:
            return full.reshape(w, self.nch * self.fdim_c), None
        slim_part = torch.cat([hw.real, hw.imag], dim=-1)  # (w, nch, 2·dim)
        out = torch.cat(
            [slim_part[:, c] if c != self.mid else full[:, c] for c in range(self.nch)],
            dim=-1,
        )
        return out, None

    def logits(self, h_n: torch.Tensor) -> torch.Tensor:
        feat, _ = self.features(h_n)
        return feat.to(torch.float16) @ self.W_vocab.to(torch.float16).t()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Single-token step (generation / per-token path). Returns the SSM logits (vocab,)."""
        if self.learn:
            self._refresh_R()
        r = (e.float() @ self.W_in.transpose(1, 2)).to(torch.complex64)  # (nch, dim)
        R = self.R
        carry = self.h.reshape(self.nch, self.nb, 2).clone()
        a = torch.tensor(self.decays, device=e.device)[:, None]
        nxt = a * torch.einsum("cbij,cbj->cbi", R, carry).reshape(self.nch, self.dim) + r
        self.h = nxt.detach()
        hn = self._unit(nxt.unsqueeze(0)).reshape(1, self.nch * self.dim)
        return self.logits(hn)[0]

    def sdch(self, h: torch.Tensor) -> torch.Tensor:
        """The memory/SDC key: concat(Re, Im) of the MID-decay channel's complex state ->
        2·dim real values (matches the fabric's readout dim for dim = readout_dim / 2).
        Accepts the windowed state (W, nch·dim), a per-token state (nch, dim), or the raw
        carry buffer (nch, nb, 2)."""
        if h.ndim == 3 and h.shape[1] == self.nch and h.shape[2] == self.dim:
            hw = h
        elif h.ndim == 3:
            hw = h.reshape(1, self.nch, self.dim)
        elif h.ndim == 2 and h.shape == (self.nch, self.dim):
            hw = h.unsqueeze(0)
        else:
            hw = h.reshape(-1, self.nch, self.dim)
        mid = min(range(self.nch), key=lambda c: abs(self.decays[c] - 0.9))
        hc = hw[:, mid]
        return torch.cat([hc.real, hc.imag], dim=-1)

    # ------------------------------------------------------------------
    def train_head(self, d_logits: torch.Tensor, h_n: torch.Tensor, mod: float = 1.0) -> None:
        """Update W_vocab by gradient DESCENT on the CE (d_logits = β·(p − onehot))."""
        feat, _ = self.features(h_n)
        w = feat.shape[0]
        dW = (d_logits.float().t() @ feat) / w
        self.W_vocab.data.sub_(self.head_lr * mod * dW.to(self.W_vocab.dtype))

    def dctx_from_head(self, d_ssm: torch.Tensor, h_n: torch.Tensor) -> list[torch.Tensor]:
        """Route the head's CE gradient back to each channel's complex state. The mid channel
        gets the full [Re, Im, |h|^2, pairs] routing; slim extra channels only [Re, Im]."""
        w = h_n.shape[0]
        dim = self.dim
        p = self.pairs
        dF = d_ssm.float() @ self.W_vocab.float()  # (w, fdim)
        out = []
        offset = 0
        for c in range(self.nch):
            bsize = self.fdim_c if (c == self.mid or not self.slim) else 2 * dim
            dFc = dF[:, offset : offset + bsize]
            offset += bsize
            hn_c = h_n.reshape(w, self.nch, dim)[:, c]
            if c == self.mid or not self.slim:
                d_re = dFc[:, :dim].clone()
                d_im = dFc[:, dim : 2 * dim].clone()
                d_abs = dFc[:, 2 * dim : 3 * dim]
                if p > 0:
                    dp = dFc[:, 3 * dim : 3 * dim + p * p].reshape(-1, p, p)
                    sym = dp + dp.transpose(1, 2)  # (W, p, p)
                    hn_p = hn_c[:, :p]
                    d_re[:, :p] += (sym * hn_p.real.unsqueeze(1)).sum(dim=2)
                    d_im[:, :p] += (sym * hn_p.imag.unsqueeze(1)).sum(dim=2)
                d_hn = d_re + 2.0 * hn_c.real * d_abs
                d_hn = d_hn + 1j * (d_im + 2.0 * hn_c.imag * d_abs)
            else:
                d_hn = dFc[:, :dim] + 1j * dFc[:, dim : 2 * dim]
            out.append(d_hn.to(torch.complex64))
        return out

    def train_beta(self, grad_beta: torch.Tensor, mod: float = 1.0) -> None:
        self.beta.data.add_(self.lr * mod * grad_beta)
        self.beta.data.clamp_(0.05, 4.0)

    def apply_grad_ctx(self, d_ctx: list[torch.Tensor], e: torch.Tensor, mod: float = 1.0) -> None:
        """Train each W_in^c by the exact complex adjoint backward scan of the recurrence
        (chunked, batched over channels)."""
        w = d_ctx[0].shape[0]
        nb = self.nb
        C = self.chunk
        M = max(1, (w + C - 1) // C)
        d = torch.stack(d_ctx, dim=1)  # (w, nch, dim)
        dj = d.reshape(w, self.nch, nb, 2)
        if M * C != w:
            dj = torch.nn.functional.pad(dj, (0, 0, 0, 0, 0, 0, M * C - w, 0))
        dj = dj.reshape(M, C, self.nch, nb, 2)
        G = self._conv(dj, adj=True, backward=True)  # internal adjoint r-gradient
        Padj = self._powers_adj(C + 1)
        c_term = torch.einsum("mjcbi,cjbki->mcbk", dj, Padj[:, 1:])  # (M, nch, nb, 2)
        acc = torch.zeros(self.nch, nb, 2, dtype=torch.complex64, device=d.device)
        for m in reversed(range(M)):
            tail = torch.einsum("cjbik,cbk->cjbi", torch.flip(Padj[:, :C], dims=[1]), acc)
            G[m] += tail.permute(1, 0, 2, 3)
            acc = c_term[m] + torch.einsum(
                "cjbik,cbk->cjbi", Padj[:, C : C + 1], acc
            ).squeeze(1)
        Gf = G.reshape(M * C, self.nch, nb, 2)[:w]  # (w, nch, nb, 2)
        dW = (
            torch.einsum("wcd,we->cde", Gf.reshape(w, self.nch, self.dim).real, e.float()) / w
        )
        self.W_in.data.add_(self.lr * mod * (dW - self.wd * self.W_in.data))
        if self.learn:
            self._update_angles(Gf, w, mod)

    def _update_angles(self, Gf: torch.Tensor, w: int, mod: float) -> None:
        """Train the qubit angles by the exact local gradient dL/dR = Σ_t a·b_t·h_{t-1}^H
        (b_t = dL/dh_t = the adjoint state gradient) projected onto θ and φ, per channel.
        Gradient DESCENT (verified against autograd to ~1e-4)."""
        hp = self._hprev.reshape(w, self.nch, self.nb, 2)
        a = torch.tensor(self.decays, device=Gf.device)[:, None, None, None]
        dR = (a / w) * torch.einsum("wcbj,wcbk->cbjk", Gf, hp.conj())  # (nch, nb, 2, 2)
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        ep = torch.exp(1j * self.phi)
        dR_dtheta = torch.zeros(self.nch, self.nb, 2, 2, dtype=torch.complex64, device=Gf.device)
        dR_dtheta[:, :, 0, 0] = -s
        dR_dtheta[:, :, 0, 1] = c * ep
        dR_dtheta[:, :, 1, 0] = -c * torch.conj(ep)
        dR_dtheta[:, :, 1, 1] = -s
        dR_dphi = torch.zeros(self.nch, self.nb, 2, 2, dtype=torch.complex64, device=Gf.device)
        dR_dphi[:, :, 0, 1] = 1j * s * ep
        dR_dphi[:, :, 1, 0] = 1j * s * torch.conj(ep)
        dtheta = torch.einsum("cbij,cbij->cb", dR.conj(), dR_dtheta).real
        dphi = torch.einsum("cbij,cbij->cb", dR.conj(), dR_dphi).real
        self.theta.data.sub_(self.lr * mod * (dtheta - self.wd * self.theta.data))
        self.phi.data.sub_(self.lr * mod * (dphi - self.wd * self.phi.data))

    def reset(self) -> None:
        self.h.zero_()
        self._hprev = torch.zeros(
            self.nch, 0, self.dim, dtype=torch.complex64, device=self.h.device
        )
