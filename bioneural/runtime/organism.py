"""BioNeural — the organism. Glue between cortex, learning, memory, workspace, drives,
neuromodulators, time, and sleep.

Lifecycle:
    process_token  : one token through cortex + workspace + memory (the "present").
    step           : one training step (predict next token, learn locally, update modulators).
    generate       : deliberative / reflex text generation (no learning).
    act_autonomously : drive-initiated behavior (self-initiated acts, logged with causes).
    idle_update    : async closed-form decay + drive drift while nobody talks to it.
    sleep_cycle    : consolidation daemon (replay, REM, downscaling).
    save_body / load_body : continuity of self across restarts.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from bioneural.config import BioNeuralConfig
from bioneural.cortex.backbone import EventSSM
from bioneural.cortex.column import ColumnLayer
from bioneural.cortex.event_bus import EventBus
from bioneural.cortex.qeu import async_leak
from bioneural.drives.homeostat import DriveEngine, DriveSignals
from bioneural.io.spikes import spike_encode
from bioneural.learning.homeostat import apply_synaptic_scaling
from bioneural.learning.predictive import SurpriseTracker
from bioneural.learning.readout import ReadoutHead
from bioneural.memory.codes import make_sdc, sdc_similarity
from bioneural.memory.fabric import MemoryFabric
from bioneural.neuromod.bus import NeuromodBus
from bioneural.time.clock import ClockBank
from bioneural.workspace.workspace import Workspace


class BioNeural(nn.Module):
    def __init__(self, cfg: BioNeuralConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.resolve_device())
        self.c = cfg.cortex
        self.lc = cfg.learning
        self.state: str = "awake"

        self.emb = nn.Embedding(cfg.vocab_size, cfg.token_dim)
        self.columns = ColumnLayer(
            self.c.num_columns,
            self.c.neurons_per_column,
            self.c.input_dim,
            self.c.readout_dim,
            self.c,
            self.lc,
            seed=cfg.seed,
        )
        self.backbone = EventSSM(self.c.readout_dim, self.c.backbone_dim, self.c, self.lc)
        self.readout = ReadoutHead(self.c.readout_dim, cfg.vocab_size, self.lc, seed=cfg.seed)

        self.bus = NeuromodBus()
        self.clock = ClockBank(cfg.time)
        self.event_bus = EventBus(cfg.memory.m0_ring_size)
        self.fabric = MemoryFabric(cfg.memory, self.c.readout_dim, self.event_bus)
        self.workspace = Workspace(cfg.workspace, self.c.readout_dim)
        self.drives = DriveEngine(cfg.drives)
        self.surprise = SurpriseTracker()

        # running state
        self._prev_r: torch.Tensor | None = None
        self._prev_sdc: torch.Tensor | None = None
        self._pred_ready = False
        self._last_ctx: torch.Tensor | None = None
        self._acc_ema = 0.0
        self._steps_since_homeo = 0
        self.total_tokens = 0
        self.total_correct = 0
        self.total_acts = 0
        self.last_act_time = 0.0
        self.init_state = time.time()

        self.to(self.device)

    # ==================================================================
    # core: one token through the substrate
    # ==================================================================
    def process_token(self, tok_id: int, learn: bool = True) -> dict:
        emb = self.emb.weight[tok_id]
        vecs = spike_encode(emb, self.cfg.spike_ticks, self.cfg.k_active_per_tick)
        gate = self.bus.gate()
        readout_acc = torch.zeros(self.c.readout_dim, device=self.device)

        for tv in vecs:
            if learn and self._pred_ready:
                ne_col = self.columns.learn_predictive(tv, gate)
                self.surprise.update(ne_col)
            out = self.columns.forward(tv, gate)
            readout_acc = readout_acc + out["readout"]
            if out["n_active"] > 0:
                fired = torch.nonzero(out["fire"], as_tuple=False)
                if fired.numel() > 0:
                    cids = out["idx"][fired[:, 0]]
                    for cid, ni in torch.stack([cids, fired[:, 1]], dim=1).tolist():
                        self.event_bus.emit(int(cid), int(ni), magnitude=1.0)
            self.event_bus.advance(1.0)
            self._pred_ready = True

        r = readout_acc
        # long-range backbone (predictive local learning, one-step lag)
        if learn and self._prev_r is not None:
            self.surprise.update(self.backbone.learn(r, gate))
        self.backbone.forward(r)

        # context for the readout head: local readout + recurrent projection
        ctx = r + 0.5 * self.backbone.context()
        logits = self.readout.forward(ctx)

        # workspace + memory
        sdc = make_sdc(r, active_frac=0.05, ternary=True)
        novelty = (1.0 - sdc_similarity(sdc, self._prev_sdc)) if self._prev_sdc is not None else 0.5
        self.workspace.compete([sdc], [novelty])
        self.workspace.broadcast([sdc])
        self.workspace.elaborate(self.fabric, sdc)
        if learn and self._prev_sdc is not None:
            self.fabric.write_experience(
                self._prev_sdc,
                sdc,
                self.bus.broadcast(),
                meta="tok",
                episodic=novelty >= 0.3,
            )

        self._prev_r = r.detach()
        self._prev_sdc = sdc.detach()
        self._last_ctx = ctx
        self.total_tokens += 1
        return {"ctx": ctx, "logits": logits, "sdc": sdc, "readout": r, "novelty": novelty}

    # ==================================================================
    # training
    # ==================================================================
    def step(self, tok_id: int, next_tok_id: int) -> dict:
        info = self.process_token(tok_id, learn=True)
        ctx, logits = info["ctx"], info["logits"]
        pred = int(logits.argmax().item())
        correct = pred == next_tok_id
        gate = self.bus.gate()
        self.readout.learn(ctx, int(next_tok_id), mod=gate)

        ne = float(min(1.0, self.surprise.value))
        reward = 1.0 if correct else 0.0
        novelty = info["novelty"]
        stability = max(0.0, 1.0 - ne)
        self.bus.from_signals(ne, reward, novelty, stability)

        self._acc_ema = 0.98 * self._acc_ema + 0.02 * float(correct)
        failure = 1.0 - self._acc_ema
        self.drives.update(
            DriveSignals(
                surprise=ne,
                reward=reward,
                novelty=novelty,
                failure_signature=failure,
                activity=1.0,
            )
        )

        self.total_correct += int(correct)
        self._steps_since_homeo += 1
        if self._steps_since_homeo >= 50:
            apply_synaptic_scaling(self.columns, self.c.target_rate)
            self._steps_since_homeo = 0

        return {"correct": correct, "pred": pred, "ne": ne, "reward": reward, "gate": gate}

    def train_sequence(self, token_ids: list[int]) -> dict:
        """Train on a sequence of tokens; returns aggregate metrics over the sequence."""
        correct = 0
        n = 0
        total_ne = 0.0
        for i in range(len(token_ids) - 1):
            out = self.step(token_ids[i], token_ids[i + 1])
            correct += int(out["correct"])
            total_ne += out["ne"]
            n += 1
        return {
            "acc": correct / max(n, 1),
            "n": n,
            "ne_mean": total_ne / max(n, 1),
        }

    # ==================================================================
    # inference / generation
    # ==================================================================
    def evaluate(self, token_ids: list[int]) -> dict:
        """No-learning evaluation: top-1 acc + per-token NLL (-> perplexity)."""
        correct = 0
        nll = 0.0
        n = 0
        for i in range(len(token_ids) - 1):
            info = self.process_token(token_ids[i], learn=False)
            logits = info["logits"]
            t = token_ids[i + 1]
            correct += int(logits.argmax().item() == t)
            mx = logits.max().item()
            lse = mx + math.log(torch.exp(logits - mx).sum().item())
            nll += lse - logits[t].item()
            n += 1
        nll_mean = nll / max(n, 1)
        return {
            "acc": correct / max(n, 1),
            "nll": nll_mean,
            "ppl": math.exp(min(nll_mean, 20.0)),
            "n_tokens": n,
        }

    def generate(self, prompt_ids: list[int], n_tokens: int, temperature: float = 0.8) -> list[int]:
        for p in prompt_ids:
            self.process_token(p, learn=False)
        out: list[int] = []
        for _ in range(n_tokens):
            ctx = self._last_ctx
            logits = self.readout.forward(ctx) / max(temperature, 0.05)
            probs = torch.softmax(logits, dim=0)
            tok = int(torch.multinomial(probs, 1).item())
            out.append(tok)
            self.process_token(tok, learn=False)
        return out

    # ==================================================================
    # autonomy & liveness
    # ==================================================================
    def act_autonomously(self, generate_fn=None) -> dict | None:
        """Drive-initiated behavior. Returns the act record (with its cause) or None."""
        drive = self.drives.wants_to_initiate()
        if drive is None:
            return None
        if time.time() - self.last_act_time < 10:
            return None
        self.last_act_time = time.time()
        self.drives.log_cause(drive)
        # a self-initiated "message": imagine a continuation from current context
        thought = None
        if self._last_ctx is not None and generate_fn is not None:
            thought = generate_fn(self._last_ctx, 1.3)
        meta = f"autonomous:{drive}"
        if self._prev_sdc is not None:
            self.fabric.write_experience(
                self._prev_sdc, self._prev_sdc, self.bus.broadcast(), meta=meta
            )
        self.total_acts += 1
        return {"drive": drive, "time": time.time(), "thought": thought, "cause": meta}

    def idle_update(self, dt_seconds: float) -> None:
        """Async closed-form decay: 4 idle hours cost one update, not 4 hours of ticks."""
        elapsed_ticks = dt_seconds  # 1 tick ~ 1 s in v1
        async_leak(self.columns, elapsed_ticks, self.c.leak_bits)
        # energy restores during idle
        self.drives.drives["energy"] = min(1.0, self.drives.drives["energy"] + 0.001 * dt_seconds)
        self.drives.update(
            DriveSignals(surprise=self.surprise.value, reward=0.5, novelty=0.1, activity=0.0)
        )
        # occasionally let the drive engine initiate
        if self.drives.wants_to_initiate() is not None:
            self.act_autonomously()

    # ==================================================================
    # structural plasticity
    # ==================================================================
    def prune_latents(self, threshold_frac: float = 0.5) -> int:
        """Structural plasticity: free capacity from synapses whose latent magnitude decays
        below the ternary deadzone (they are already zero-valued on the wire). Returns the
        number of weights zeroed."""
        from bioneural.quant.kernels import _effective_group_size, _group_scale

        pruned = 0
        for mod in self.modules():
            latent = getattr(mod, "latent", None)
            if latent is None or not hasattr(mod, "version"):
                continue
            m, k = latent.shape
            gs = _effective_group_size(k, mod.config.group_size)
            scale = _group_scale(latent, mod.config.group_size, mod.config.scale_mode)
            thresh = mod.config.deadzone * scale.repeat_interleave(gs, dim=-1) * threshold_frac
            mask = latent.abs() < thresh
            if mask.any().item():
                mod.latent = torch.where(mask, torch.zeros_like(latent), latent)
                mod.version += 1
                mod._cache = None
                pruned += int(mask.sum().item())
        return pruned

    def sleep_cycle(self, replay_n: int = 64, temperature: float = 1.4) -> dict[str, float]:
        from bioneural.runtime.sleep import sleep_cycle as _sleep

        self.state = "sleep"
        info = _sleep(self, replay_n, temperature)
        self.state = "awake"
        return info

    # ==================================================================
    # serialization / stats
    # ==================================================================
    def save_body(self, path: str) -> None:
        from bioneural.runtime.checkpoint import save_body as _save

        _save(self, path)

    def load_body(self, path: str) -> BioNeural:
        from bioneural.runtime.checkpoint import load_body as _load

        return _load(self, path)

    def n_params(self) -> int:
        total = 0
        for mod in self.modules():
            for b in mod._buffers.values():
                if b is not None and b.dtype.is_floating_point:
                    total += b.numel()
            for p in mod.parameters():
                total += p.numel()
        return total

    def stats(self) -> dict[str, float]:
        s: dict[str, float] = {}
        s.update(
            {
                "total_tokens": float(self.total_tokens),
                "token_acc": self.total_correct / max(self.total_tokens, 1),
                "acc_ema": self._acc_ema,
                "autonomous_acts": float(self.total_acts),
                "surprise": self.surprise.value,
                "n_params": float(self.n_params()),
            }
        )
        s.update({f"col_{k}": v for k, v in self.columns.stats().items()})
        s.update({f"ssm_{k}": v for k, v in self.backbone.stats().items()})
        s.update({f"mem_{k}": v for k, v in self.fabric.stats().items()})
        s.update({f"drive_{k}": v for k, v in self.drives.state().items()})
        s.update({f"mod_{k}": v for k, v in self.bus.broadcast().items()})
        s.update(self.workspace.stats())
        return s
