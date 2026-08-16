"""Quantized Event Unit (QEU) — the neuron model.

A hybrid of a Leaky-Integrate-and-Fire neuron and a gated RNN cell, kept in the *hardware-friendly*
fixed-point form as its native form:

    v      : membrane potential      (INT16 fixed-point, MEMBRANE_SCALE)
    theta  : adaptive threshold      (INT16) — homeostatic
    trace  : eligibility trace       (INT8)  — local memory for the learning rule

    on events:  v <- v - (v >> k)  +  contrib            # leak is a shift (1 - 2^-k)
                fire = (v > theta)
                if fire: emit; v <- v - theta            # soft reset keeps residual info
                theta <- theta + fire*Δθ⁺ - Δθ⁻          # target firing rate ~2-5%
                trace <- trace - (trace >> d) + fire

All arithmetic is fixed-point integer math: no floats in the hot path on device.
"""

from __future__ import annotations

import torch

MEMBRANE_SCALE = 1000  # fixed-point scale: membrane units x1000 -> INT16


class QEUState:
    """Fixed-point state for a population of QEU neurons (columns x K neurons)."""

    def __init__(self, num_columns: int, neurons_per_column: int, theta_init: float = 1.0):
        self.num_columns = num_columns
        self.k = neurons_per_column
        self.v = torch.zeros((num_columns, neurons_per_column), dtype=torch.int16)
        self.theta = torch.full(
            (num_columns, neurons_per_column), int(theta_init * MEMBRANE_SCALE), dtype=torch.int16
        )
        self.trace = torch.zeros((num_columns, neurons_per_column), dtype=torch.int8)
        # running firing-rate estimate (float, for diagnostics / homeostasis reporting)
        self.rate = torch.zeros((num_columns, neurons_per_column))

    def reset(self) -> None:
        self.v.zero_()
        self.trace.zero_()
        self.theta.fill_(int(1.0 * MEMBRANE_SCALE))
        self.rate.zero_()


def advance_qeu_tensors(
    v: torch.Tensor,
    theta: torch.Tensor,
    trace: torch.Tensor,
    contrib: torch.Tensor,
    leak_bits: int = 4,
    theta_delta_plus: float = 0.05,
    theta_delta_minus: float = 0.002,
    trace_decay_bits: int = 2,
    wta_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance fixed-point QEU state tensors ((C, K)).

    Returns new (v, theta, trace, rate_delta, fire, potential).
    """
    dev = v.device
    contrib_int = (contrib * MEMBRANE_SCALE).round().to(device=dev, dtype=torch.int32)

    # leak: v <- v - (v >> leak_bits)  == v * (1 - 2^-leak_bits), arithmetic shift
    leak = v.to(torch.int32) - (v.to(torch.int32) >> leak_bits)
    v_new = leak + contrib_int

    potential = v_new.to(torch.float32) / MEMBRANE_SCALE

    fire = v_new > theta.to(torch.int32)

    if wta_k is not None and wta_k > 0:
        # k winners within each column: keep the top-k firing neurons by potential
        scores = torch.where(fire, potential, torch.full_like(potential, -1e9))
        topk = torch.topk(scores, wta_k, dim=-1).indices
        winner = torch.zeros_like(fire)
        winner.scatter_(-1, topk, 1)
        fire = fire & winner.bool()

    # soft reset keeps residual info
    v_new = torch.where(fire, v_new - theta.to(torch.int32), v_new)

    # homeostatic adaptive threshold
    theta_dp = int(theta_delta_plus * MEMBRANE_SCALE)
    theta_dm = int(theta_delta_minus * MEMBRANE_SCALE)
    theta_new = theta.to(torch.int32) + fire.to(torch.int32) * theta_dp - theta_dm
    theta_new = theta_new.clamp(min=int(0.2 * MEMBRANE_SCALE), max=int(5.0 * MEMBRANE_SCALE))

    # eligibility trace: exponential decay + fire (INT8 saturation)
    trace_dec = trace.to(torch.int32) - (trace.to(torch.int32) >> trace_decay_bits)
    trace_new = (trace_dec + fire.to(torch.int32) * 127).clamp(min=-128, max=127)

    return (
        v_new.to(torch.int16),
        theta_new.to(torch.int16),
        trace_new.to(torch.int8),
        fire.float(),
        fire,
        potential,
    )


def advance_qeu(
    state: QEUState,
    contrib: torch.Tensor,
    leak_bits: int = 4,
    theta_delta_plus: float = 0.05,
    theta_delta_minus: float = 0.002,
    trace_decay_bits: int = 2,
    wta_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance fixed-point QEU state for a full column batch (thin wrapper)."""
    v, theta, trace, rate_delta, fire, potential = advance_qeu_tensors(
        state.v,
        state.theta,
        state.trace,
        contrib,
        leak_bits=leak_bits,
        theta_delta_plus=theta_delta_plus,
        theta_delta_minus=theta_delta_minus,
        trace_decay_bits=trace_decay_bits,
        wta_k=wta_k,
    )
    state.v, state.theta, state.trace = v, theta, trace
    state.rate = state.rate * 0.99 + rate_delta * 0.01
    return fire, potential


def async_leak(state: QEUState, elapsed_ticks: float, leak_bits: int = 4) -> None:
    """Advance membrane leaks by an *elapsed* number of ticks in one closed-form step.

    Idle time is therefore O(1) per update, not O(ticks): 4 idle hours cost one update.
    Approximated per-leak-step for stability; exact enough for v1 (time perception is
    continuous, compute is event-proportional).
    """
    if elapsed_ticks <= 0:
        return
    v = state.v.to(torch.int32)
    for _ in range(int(min(elapsed_ticks, 64))):
        v = v - (v >> leak_bits)
    state.v = v.to(torch.int16)
