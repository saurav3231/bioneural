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
from bioneural.io.spikes import spike_encode, spike_encode_batch
from bioneural.learning.homeostat import apply_synaptic_scaling
from bioneural.learning.predictive import SurpriseTracker
from bioneural.learning.readout import ReadoutHead
from bioneural.memory.codes import make_sdc, make_sdc_batch, sdc_similarity, sdc_similarity_batch
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
        self.columns.profile = cfg.profile
        self.backbone = EventSSM(self.c.readout_dim, self.c.backbone_dim, self.c, self.lc)
        self.readout = ReadoutHead(
            self.c.readout_dim,
            cfg.vocab_size,
            self.lc,
            seed=cfg.seed,
            tied_emb=self.emb.weight if self.lc.tied_embeddings else None,
        )
        # task-aligned context projector: continuous fp32 map from the recurrent state h into
        # readout space, trained by regression to the next token's embedding. The ternary W_out·h
        # channel is self-predictive only; P gives the long-range state a supervised path into
        # the readout. Zero-init -> no initial effect on ctx.
        self.ctx_proj = nn.Parameter(torch.zeros(self.c.readout_dim, self.c.backbone_dim))

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
        self._sim_time = 0.0  # organism-relative clock (1 tick ~ 1 s), drives cooldowns/ripples
        self.init_state = time.time()
        self._prof: dict[str, float] = {}  # per-phase ms accumulator (cfg.profile)
        self._prof_n = 0

        self.to(self.device)

    # ==================================================================
    # core: one token through the substrate
    # ==================================================================
    @torch.inference_mode()
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
        h = self.backbone.forward(r)

        # context for the readout head: local readout + recurrent projection + task-aligned P·h
        h_p = h / (h.norm() + 1e-8)
        ctx_hebb = (
            r + self.cfg.ctx_embed_weight * emb
        )
        ctx = (
            r
            + 0.5 * self.backbone.context()
            + self.cfg.ctx_embed_weight * emb
            + self.cfg.ctx_proj_weight * (self.ctx_proj @ h_p)
        )
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
        self._sim_time += 1.0
        return {"ctx": ctx, "ctx_hebb": ctx_hebb, "logits": logits, "sdc": sdc, "readout": r, "novelty": novelty}

    # ==================================================================
    # training
    # ==================================================================
    @torch.inference_mode()
    def step(self, tok_id: int, next_tok_id: int) -> dict:
        info = self.process_token(tok_id, learn=True)
        ctx, logits = info["ctx"], info["logits"]
        pred = int(logits.argmax().item())
        correct = pred == next_tok_id
        gate = self.bus.gate()
        self.readout.learn(ctx, int(next_tok_id), mod=gate)
        delta = info["ctx_hebb"].detach().float() - self.emb.weight[next_tok_id]
        self.emb.weight[next_tok_id] = self.emb.weight[next_tok_id] + delta * (0.04 * gate)
        self.emb.weight[next_tok_id] /= self.emb.weight[next_tok_id].norm() + 1e-8

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

    @torch.inference_mode()
    def train_sequence(self, token_ids: list[int]) -> dict:
        """Train on a sequence of tokens; returns aggregate metrics over the sequence."""
        if self.cfg.batch_window >= 2:
            info = self._train_sequence_windowed(token_ids)
            self._sim_time += max(len(token_ids) - 1, 0)
            if self.cfg.profile and self._prof_n >= 5:
                tot = sum(self._prof.values())
                pct = {k: f"{100.0 * v / max(tot, 1e-9):.0f}%" for k, v in self._prof.items()}
                print(
                    f"  PROF window_ms={tot / self._prof_n:.2f}  "
                    + "  ".join(f"{k}={pct[k]}" for k in sorted(pct)),
                    flush=True,
                )
                if self.columns._pcol_n >= 5:
                    ct = sum(self.columns._pcol.values())
                    cp = {k: f"{100.0 * v / max(ct, 1e-9):.0f}%" for k, v in self.columns._pcol.items()}
                    print(
                        f"  COL  ms/tick={ct / self.columns._pcol_n:.2f}  "
                        + "  ".join(f"{k}={cp[k]}" for k in sorted(cp)),
                        flush=True,
                    )
                    self.columns._pcol.clear()
                    self.columns._pcol_n = 0
                self._prof.clear()
                self._prof_n = 0
            return info
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

    @torch.inference_mode()
    def _train_sequence_windowed(self, token_ids: list[int]) -> dict:
        w = self.cfg.batch_window
        correct = 0
        n = 0
        total_ne = 0.0
        for i in range(0, len(token_ids) - 1, w):
            seg = token_ids[i : i + w + 1]
            if len(seg) < 2:
                break
            out = self.train_window(seg, window=w)
            correct += int(round(out["acc"] * out["n"]))
            total_ne += out["ne_mean"] * out["n"]
            n += out["n"]
        return {
            "acc": correct / max(n, 1),
            "n": n,
            "ne_mean": total_ne / max(n, 1),
        }

    @torch.inference_mode()
    def train_window(self, token_ids: list[int], window: int = 64) -> dict:
        """Batched training: `window` tokens flow through the neural path per GPU op.

        Packet approximation on the columns (each tick's contributions summed per column before
        the QEU threshold step) + a closed-form backbone recurrence + scatter-add readout learning
        + a batched memory flush. This replaces ~120 per-token torch ops with a handful per
        window, which is what the T4's launch-latency floor demands.
        """
        w = min(window, len(token_ids) - 1)
        if w < 1:
            return {"acc": 0.0, "n": 0, "ne_mean": 0.0}
        xs = token_ids[:w]
        ys = token_ids[1 : w + 1]
        xids = torch.tensor(xs, dtype=torch.long, device=self.device)
        embs = self.emb.weight[xids]  # (W, D)
        bursts = spike_encode_batch(embs, self.cfg.spike_ticks, self.cfg.k_active_per_tick)
        gate = self.bus.gate()

        readout = torch.zeros(w, self.c.readout_dim, device=self.device)
        pred_errs: list[torch.Tensor] = []
        prof = self.cfg.profile and torch.cuda.is_available()
        if prof:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record()
        for tv in bursts:
            out = self.columns.forward_batch(tv, gate, learn=True)
            readout += out["readout"]
            if out["pred_err"] is not None:
                pred_errs.append(out["pred_err"])
            self.event_bus.advance(1.0)
        if prof:
            ev[1].record()

        if pred_errs:
            ne_col = float(torch.stack(pred_errs).mean().item())  # one sync per window
            self.surprise.update(ne_col)

        # backbone: closed-form linear recurrence over the whole window, trained by predictive
        # coding AGAINST the next token's embedding (task-aligned state formation).
        r = readout
        ys_t = torch.tensor(ys, dtype=torch.long, device=self.device)
        h, bsurp = self.backbone.window(
            r, learn=True, mod=gate, target=self.emb.weight[ys_t]
        )
        if bsurp is not None:
            self.surprise.update(float(bsurp.item()))
        if prof:
            ev[2].record()
        # task-aligned context projector P: maps the (L2-normalized) recurrent state h_p into
        # readout space. P is a plain fp32 matrix (no forced row-norm) trained by regression to
        # predict the NEXT token's embedding, so ph converges toward emb[y] (~unit norm, bounded).
        h_p = h / (h.norm(dim=1, keepdim=True) + 1e-8)
        ph = self.cfg.ctx_proj_weight * (h_p @ self.ctx_proj.t())
        # context for the head: processed (cortex+SSM) + direct sensory (current token embedding)
        # + the task-aligned P·h next-token prediction.
        ctx = (
            r
            + 0.5 * self.backbone.context_batch(h)
            + self.cfg.ctx_embed_weight * self.emb.weight[xids]
            + ph
        )

        # readout head: batched forward + contrastive learn
        logits = self.readout.forward_batch(ctx).float()
        preds = logits.argmax(dim=-1).tolist()  # one sync per window
        correct = sum(1 for p, y in zip(preds, ys, strict=False) if p == y)
        self.readout.learn_batch(ctx, ys, mod=gate)
        if prof:
            ev[3].record()

        # top-down task error is applied ONLY to continuous weight surfaces (embeddings, readout
        # head, and the continuous SSM) — ternary-quantized cortex columns update by discrete
        # flips, so continuous error gradients become noise there and collapse learning. The
        # columns learn self-predictively; the embeddings+head+SSM carry the task.

        # embedding learning: the token that was PREDICTED moves toward the state that
        # predicted it (emb[x_{t+1}] += lr·(ctx_t − emb[x_{t+1}])). Local Hebbian rule that
        # makes input and output embeddings converge to the same "token -> internal state"
        # map (like tied embeddings), so the cortex gets structured, learnable input features.
        if not self.lc.tied_embeddings:
            # when tied, the head's output-role softmax gradient (which touches every prototype row
            # with exact CE signal) subsumes this local Hebbian pull toward the predicting ctx.
            # The target EXCLUDES the backbone: the SSM now outputs ~emb[y], so including it
            # would drag emb[y] toward itself (0.5·emb[y] term) and decay the embedding map.
            ctx_hebb = (
                r + self.cfg.ctx_embed_weight * self.emb.weight[xids]
            )
            delta = ctx_hebb.detach().float() - self.emb.weight[ys_t]
            self.emb.weight.index_add_(
                0, ys_t, (delta * (0.04 * gate)).to(self.emb.weight.dtype)
            )
        # top-down (dopamine-style) supervised error through the head's reciprocal weights:
        # emb[x_t] -= lr·d_ctx. The gradient is computed on the emb-only context (r + emb[x])
        # so neither the SSM's ~emb[y] output nor P·h can contaminate the embedding map.
        d_ctx = self.readout.grad_ctx(r + self.cfg.ctx_embed_weight * self.emb.weight[xids], ys)
        td = (-self.lc.lr_emb_top * gate * d_ctx).to(self.emb.weight.dtype)
        self.emb.weight.index_add_(0, xids, td)
        touched = torch.unique(torch.cat([ys_t, xids]))
        self.emb.weight[touched] = (
            self.emb.weight[touched] / (self.emb.weight[touched].norm(dim=1, keepdim=True) + 1e-8)
        ).to(self.emb.weight.dtype)

        # task-aligned context projector update: P is a next-token-embedding predictor, so its
        # target is emb[y_{t+1}] (the unit-norm token embedding) — dL/dP ∝ (ph − emb[y]) ⊗ h_p.
        # Regression is stable (no dependence on the head's normalization) and self-bounding.
        # Gradients are MEAN-normalized over the window (/w) so P can't blow up into ctx.
        errP = ph.float() - self.emb.weight[ys_t].float()  # (W, rd)
        gP = (self.cfg.ctx_proj_weight * errP).t() @ h_p.float() / w  # (rd, dim)
        self.ctx_proj.sub_((self.lc.lr_ctx_proj * gate * gP).to(self.ctx_proj.dtype))

        # sparse codes + novelty (batched)
        sdcs = make_sdc_batch(r, active_frac=0.05, ternary=True)
        if w > 1:
            sims = sdc_similarity_batch(sdcs[:-1], sdcs[1:])
            novelties = [0.5] + [1.0 - s for s in sims]
        else:
            novelties = [0.5]

        # workspace + memory (per-window, batched flush)
        self.workspace.compete([sdcs[-1]], [novelties[-1]])
        if self._prev_sdc is not None and w > 1:
            keys = torch.cat([self._prev_sdc.unsqueeze(0), sdcs[:-1]], dim=0)
        else:
            keys = sdcs
        self.fabric.write_experience_batch(keys, sdcs, novelties, self.bus.broadcast())
        if prof:
            ev[4].record()

        # neuromodulators / drives / stats
        ne = float(min(1.0, self.surprise.value))
        novelty_mean = sum(novelties) / max(w, 1)
        reward = correct / max(w, 1)
        self.bus.from_signals(ne, reward, novelty_mean, stability=max(0.0, 1.0 - ne))
        self._acc_ema = 0.98 * self._acc_ema + 0.02 * reward
        self.drives.update(
            DriveSignals(
                surprise=ne,
                reward=reward,
                novelty=novelty_mean,
                failure_signature=1.0 - self._acc_ema,
                activity=1.0,
            )
        )

        self._prev_r = r[-1].detach()
        self._prev_sdc = sdcs[-1].detach()
        self._last_ctx = ctx[-1]
        self.total_tokens += w
        self.total_correct += correct

        self._steps_since_homeo += w
        if self._steps_since_homeo >= 50:
            apply_synaptic_scaling(self.columns, self.c.target_rate)
            self._steps_since_homeo = 0

        if prof:
            ev[5].record()
            torch.cuda.synchronize()
            names = ["columns", "backbone", "head+embed", "sdc/mem", "rest"]
            for i, nm in enumerate(names):
                self._prof[nm] = self._prof.get(nm, 0.0) + ev[i].elapsed_time(ev[i + 1])
            self._prof_n += 1

        return {"acc": correct / max(w, 1), "n": w, "ne_mean": ne}

    # ==================================================================
    # inference / generation
    # ==================================================================
    @torch.inference_mode()
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

    @torch.inference_mode()
    def evaluate_window(self, token_ids: list[int], window: int = 0) -> dict:
        """No-learning evaluation using the batched windowed path (fast on GPU).

        Runs the same packet forward the model is trained with, so eval and train share
        dynamics; used instead of the per-token `evaluate` when `cfg.batch_window >= 2`.
        """
        w = max(2, min(window or self.cfg.batch_window, len(token_ids) - 1))
        if w < 2:
            return {"acc": 0.0, "nll": float("inf"), "ppl": float("inf"), "n_tokens": 0}
        xs = token_ids[:-1]
        ys = token_ids[1:]
        correct = 0
        nll = 0.0
        correct_emb = 0
        nll_emb = 0.0
        correct_noemb = 0
        nll_noemb = 0.0
        n = 0
        gate = self.bus.gate()
        for i in range(0, len(xs), w):
            seg = xs[i : i + w]
            if len(seg) < 1:
                break
            xids = torch.tensor(seg, dtype=torch.long, device=self.device)
            embs = self.emb.weight[xids]
            bursts = spike_encode_batch(embs, self.cfg.spike_ticks, self.cfg.k_active_per_tick)
            readout = torch.zeros(len(seg), self.c.readout_dim, device=self.device)
            for tv in bursts:
                out = self.columns.forward_batch(tv, gate, learn=False)
                readout += out["readout"]
                self.event_bus.advance(1.0)
            h, _ = self.backbone.window(readout, learn=False, mod=gate)
            backbone_ctx = self.backbone.context_batch(h)
            h_p = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            ph = self.cfg.ctx_proj_weight * (h_p @ self.ctx_proj.t())
            ctx = (
                readout
                + 0.5 * backbone_ctx
                + self.cfg.ctx_embed_weight * embs
                + ph
            )
            logits = self.readout.forward_batch(ctx).float()
            # ctx ablation (diagnostic): ppl under the embedding anchor alone vs the
            # cortex/backbone alone. If ppl_emb ~= ppl, the cortex adds nothing for the head.
            logits_emb = self.readout.forward_batch(self.cfg.ctx_embed_weight * embs).float()
            logits_noemb = self.readout.forward_batch(
                readout + 0.5 * backbone_ctx + ph
            ).float()
            lsm = torch.log_softmax(logits, dim=-1)
            lsm_emb = torch.log_softmax(logits_emb, dim=-1)
            lsm_noemb = torch.log_softmax(logits_noemb, dim=-1)
            y = torch.tensor(ys[i : i + w], dtype=torch.long, device=self.device)[: len(seg)]
            lsm = lsm[: len(y)]
            lsm_emb = lsm_emb[: len(y)]
            lsm_noemb = lsm_noemb[: len(y)]
            nll += float(-lsm.gather(1, y[:, None]).sum().item())
            nll_emb += float(-lsm_emb.gather(1, y[:, None]).sum().item())
            nll_noemb += float(-lsm_noemb.gather(1, y[:, None]).sum().item())
            correct += int((logits[: len(y)].argmax(dim=-1) == y).sum().item())
            correct_emb += int((logits_emb[: len(y)].argmax(dim=-1) == y).sum().item())
            correct_noemb += int((logits_noemb[: len(y)].argmax(dim=-1) == y).sum().item())
            n += len(y)
        nll_mean = nll / max(n, 1)
        nll_emb_mean = nll_emb / max(n, 1)
        nll_noemb_mean = nll_noemb / max(n, 1)
        return {
            "acc": correct / max(n, 1),
            "nll": nll_mean,
            "ppl": math.exp(min(nll_mean, 20.0)),
            "ppl_emb": math.exp(min(nll_emb_mean, 20.0)),
            "ppl_noemb": math.exp(min(nll_noemb_mean, 20.0)),
            "acc_emb": correct_emb / max(n, 1),
            "acc_noemb": correct_noemb / max(n, 1),
            "n_tokens": n,
        }

    @torch.inference_mode()
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
        if self._sim_time - self.last_act_time < 10:
            return None
        self.last_act_time = self._sim_time
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
        self._sim_time += dt_seconds
        async_leak(self.columns, elapsed_ticks, self.c.leak_bits)
        # energy restores during idle
        self.drives.drives["energy"] = min(1.0, self.drives.drives["energy"] + 0.001 * dt_seconds)
        self.drives.update(
            DriveSignals(surprise=self.surprise.value, reward=0.5, novelty=0.1, activity=0.0)
        )
        # hippocampal replay (sharp-wave ripples): spontaneous reactivation of recent engrams
        # during rest, ~1 ripple per 2 simulated seconds — the organism is alive when idle.
        gate = self.bus.gate()
        n_ripples = int(dt_seconds // 2)
        engrams = self.fabric.remember_latest(max(n_ripples, 1))
        for e in engrams:
            key = e.key.to(self.device).float()
            self.columns.forward(key, gate)  # reactivation (no learning: cheap, memory-only)
            self.event_bus.advance(1.0)
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
