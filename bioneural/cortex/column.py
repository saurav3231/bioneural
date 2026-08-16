"""Cortical columns: small recurrent QEU networks + local inhibitory pool (k-WTA).

Execution model (per the design doc):
* **Within-column:** dense-ish local recurrence — small matrices, stays in SRAM/L2.
* **Between-columns:** sparse long-range event routing via the event bus (AER).
* **Block-sparse at column granularity:** each tick only the columns that received events are
  gathered, run, and scattered. Idle columns cost ~0 FLOPs. `active%` is instrumented and
  reported by the benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bioneural.config import CortexConfig, LearningConfig
from bioneural.cortex.qeu import advance_qeu_tensors
from bioneural.quant.ternary import TernaryParam


def _build_sparse_conn(shape: tuple[int, int], sparsity: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) > sparsity


class ColumnLayer(nn.Module):
    """All cortical columns as one block-sparse layer."""

    def __init__(
        self,
        num_columns: int,
        neurons_per_column: int,
        input_dim: int,
        readout_dim: int,
        cfg: CortexConfig,
        lcfg: LearningConfig,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.lcfg = lcfg
        self.c = num_columns
        self.k = neurons_per_column
        self.input_dim = input_dim
        self.readout_dim = readout_dim

        self.W_in = TernaryParam((num_columns * neurons_per_column, input_dim), sparsity=0.85)
        self.W_rec = TernaryParam(
            (num_columns * neurons_per_column, neurons_per_column), sparsity=0.90
        )
        self.W_pred = TernaryParam((num_columns * neurons_per_column, input_dim), sparsity=0.85)
        self.out_basis = TernaryParam(
            (num_columns * neurons_per_column, readout_dim), sparsity=0.90
        )

        # per-column input connectivity: column c connects to input dim j if any of its K rows does
        conn = _build_sparse_conn((num_columns, input_dim), 0.70, seed)
        self.register_buffer("in_conn", conn)
        self._conn_version = int(self.W_in.version.item()) - 1
        self._refresh_conn()

        # fixed-point state
        self.register_buffer("v", torch.zeros((num_columns, neurons_per_column), dtype=torch.int16))
        self.register_buffer(
            "theta",
            torch.full((num_columns, neurons_per_column), 1000, dtype=torch.int16),
        )
        self.register_buffer(
            "trace", torch.zeros((num_columns, neurons_per_column), dtype=torch.int8)
        )
        self.register_buffer("rate", torch.zeros((num_columns, neurons_per_column)))
        self.register_buffer("prev_fire", torch.zeros((num_columns, neurons_per_column)))
        self.register_buffer("last_pred", torch.zeros((num_columns, input_dim)))
        self.register_buffer("last_fire", torch.zeros((num_columns, neurons_per_column)))

        # instrumentation
        self.ticks_run = 0
        self.total_active_cols = 0
        self.total_fires = 0
        self._arange_k = torch.arange(neurons_per_column)

    # ------------------------------------------------------------------
    def _refresh_conn(self) -> None:
        if int(self.W_in.version.item()) == self._conn_version:
            return
        latent = self.W_in.latent
        nz = (latent.view(self.c, self.k, self.input_dim) != 0).any(dim=1)
        self.in_conn = nz
        self._conn_version = int(self.W_in.version.item())

    def _rows_for(self, col_idx: torch.Tensor) -> torch.Tensor:
        base = col_idx * self.k
        return base[:, None] + self._arange_k

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mod: float = 1.0) -> dict[str, torch.Tensor]:
        """Advance the layer for one tick of sparse input events.

        Args:
            x: (input_dim,) float event vector (mostly zeros).
            mod: neuromodulator scalar broadcast this tick.

        Returns a dict with `readout` ((readout_dim,) contribution), `fire`, `potential`,
        and instrumentation.
        """
        self._refresh_conn()
        x = x.detach().float()
        active_inputs = x != 0
        col_active = (self.in_conn & active_inputs).any(dim=1)
        idx = col_active.nonzero(as_tuple=False).flatten()
        n_act = int(idx.numel())
        self.ticks_run += 1
        self.total_active_cols += n_act

        readout = torch.zeros(self.readout_dim, device=self.v.device)

        if n_act == 0:
            return {"readout": readout, "fire": torch.zeros(0), "n_active": 0, "idx": idx}

        rows = self._rows_for(idx)  # (n_act, K)

        # ---- sparse bottom-up: gather only active input dims ----
        adims = active_inputs.nonzero(as_tuple=False).flatten()
        win = self.W_in.materialized()[rows][:, :, adims]  # (n_act, K, n_a)
        xa = x[adims]
        contrib_bu = (win * xa[None, None, :]).sum(-1)

        # ---- local recurrence (dense within column) ----
        prev_fire = self.prev_fire[idx]
        wrec = self.W_rec.materialized()[rows]  # (n_act, K, K)
        contrib_rec = (wrec * prev_fire[:, None, :]).sum(-1)

        contrib = contrib_bu + contrib_rec

        # ---- advance QEU fixed-point state on the active subset ----
        v_new, theta_new, trace_new, _rd, fire, potential = advance_qeu_tensors(
            self.v[idx].to(torch.int32),
            self.theta[idx].to(torch.int32),
            self.trace[idx].to(torch.int32),
            contrib,
            leak_bits=self.cfg.leak_bits,
            theta_delta_plus=self.cfg.theta_delta_plus,
            theta_delta_minus=self.cfg.theta_delta_minus,
            wta_k=self.cfg.wta_k,
        )
        self.v[idx] = v_new
        self.theta[idx] = theta_new
        self.trace[idx] = trace_new
        self.prev_fire[idx] = fire.float()
        self.total_fires += int(fire.sum().item())

        # ---- predictive coding: predict the NEXT tick input from this state ----
        fire_vec = fire.float()
        wpred = self.W_pred.materialized()[rows][:, :, adims]
        pred_here = (wpred * fire_vec[:, :, None]).sum(1)  # (n_act, n_a)
        self.last_pred[idx[:, None], adims[None, :]] = pred_here
        self.last_fire[idx] = fire_vec

        # ---- readout contribution to the global broadcast ----
        ob = self.out_basis.materialized()[rows]  # (n_act, K, readout_dim)
        contrib_read = (ob * fire_vec[:, :, None]).sum(1).sum(0)  # (readout_dim,)
        readout = contrib_read

        return {
            "readout": readout,
            "fire": fire,
            "potential": potential,
            "n_active": n_act,
            "idx": idx,
        }

    # ------------------------------------------------------------------
    def learn_predictive(self, x_next: torch.Tensor, mod: float = 1.0) -> float:
        """Predictive-coding update: error vs the prediction made last tick (local target).

        Returns the mean absolute prediction error (the column's NE contribution).
        """
        x_next = x_next.detach().float()
        active = self.last_fire.any(dim=1)
        idx = active.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0.0
        rows = self._rows_for(idx)
        err = x_next - self.last_pred[idx]
        grad = torch.einsum("ak,ai->aki", self.last_fire[idx], err)  # (n_act, K, input_dim)
        lr = self.lcfg.lr_predict * mod * self.lcfg.mod_gate_strength
        grad_full = torch.zeros_like(self.W_pred.latent)
        grad_full[rows] = grad
        self.W_pred.update_latent(grad_full, lr=lr)
        return float(err.abs().sum().item())

    # ------------------------------------------------------------------
    def learn_hebbian(self, mod: float = 1.0) -> int:
        """Three-factor Hebbian on the within-column recurrent weights.

        Δw_ij = η · M(t) · fire_i(t) · trace_j(t-1)  (post x eligibility trace of pre).
        """
        active = self.trace.any(dim=1)
        idx = active.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0
        rows = self._rows_for(idx)
        post = self.prev_fire[idx]
        pre_trace = self.trace[idx].float() / 127.0
        grad = torch.einsum("ak,aj->akj", post, pre_trace)
        lr = self.lcfg.lr_hebb * mod * self.lcfg.mod_gate_strength
        grad_full = torch.zeros_like(self.W_rec.latent)
        grad_full[rows] = grad
        flips = self.W_rec.update_latent(grad_full, lr=lr)
        return flips

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, float]:
        return {
            "active_cols_frac": (self.total_active_cols / max(self.ticks_run * self.c, 1)),
            "fires_per_tick": self.total_fires / max(self.ticks_run, 1),
            "firing_rate_mean": float(self.rate.mean().item()),
            "w_in_density": self.W_in.stats()["density"],
            "w_rec_density": self.W_rec.stats()["density"],
        }

    def reset(self) -> None:
        self.v.zero_()
        self.theta.fill_(1000)
        self.trace.zero_()
        self.rate.zero_()
        self.prev_fire.zero_()
        self.last_pred.zero_()
        self.last_fire.zero_()
        self.ticks_run = 0
        self.total_active_cols = 0
        self.total_fires = 0
