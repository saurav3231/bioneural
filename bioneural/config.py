from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QuantConfig:
    group_size: int = 64
    deadzone: float = 0.15
    scale_mode: str = "mean"  # "mean" | "max"
    accum_dtype: str = "int32"
    use_kernel: bool = True


@dataclass
class CortexConfig:
    num_columns: int = 64
    neurons_per_column: int = 128
    input_dim: int = 256
    readout_dim: int = 256
    leak_bits: int = 4  # leak = 1 - 2^-k
    theta_delta_plus: float = 0.05
    theta_delta_minus: float = 0.002
    target_rate: float = 0.03  # ~3% neurons active per tick
    wta_k: int = 8  # k winners within column
    backbone_dim: int = 128
    backbone_layers: int = 1


@dataclass
class LearningConfig:
    lr_homeo: float = 0.01
    lr_predict: float = 0.05
    lr_backbone: float = 0.1  # continuous SSM state projections (predictive coding vs next-token emb)
    lr_hebb: float = 0.02
    lr_readout: float = 0.1
    lr_emb_top: float = 0.1  # top-down (dopamine-style) error lr applied to input embeddings
    lr_topdown: float = 0.05  # top-down error lr applied to the cortex readout projections
    lr_hidden: float = 0.05  # hidden-layer lr of the (2-layer) readout head
    head_hidden: int = 0  # >0: frozen random-feature (ELM) head width; 0 = plain linear head
    tied_embeddings: bool = False  # tied head: unstable when ctx contains emb[x] (self-referential)
    lr_ctx_proj: float = 0.05  # task-aligned context projector P (continuous fp32) lr: P -= lr·d_ctx⊗h_p
    trace_decay: float = 0.95
    mod_gate_strength: float = 1.0
    shadow_update: float = 0.05


@dataclass
class MemoryConfig:
    m0_ring_size: int = 4096
    m1_slots: int = 32
    m1_decay: float = 0.9
    m2a_capacity: int = 256  # one-shot fast-weight entries (the "today" memory)
    m2a_sim_threshold: float = 0.55
    m2b_max_engrams: int = 4096  # hot in-RAM; > this spills to disk
    m2b_k: int = 8  # retrieval neighbours
    m3_max_concepts: int = 1024


@dataclass
class WorkspaceConfig:
    n_slots: int = 16
    wsa_k: int = 4  # coalition size


@dataclass
class TimeConfig:
    n_freq: int = 12  # oscillator pairs
    base_hz: float = 1.0
    max_hz: float = 1.0 / 86400.0  # 1 per day


@dataclass
class DriveConfig:
    init: dict = field(
        default_factory=lambda: {
            "curiosity": 0.1,
            "social": 0.2,
            "coherence": 0.1,
            "competence": 0.1,
            "energy": 0.8,
        }
    )
    silence_decay: float = 0.98
    initiate_threshold: float = 0.75


@dataclass
class EvalConfig:
    seed: int = 0
    train_budget_minutes: float = 15.0
    dataset: str = "tiny-stories"
    max_examples: int = 2000
    max_val_examples: int = 200
    gen_length: int = 64
    eval_every_steps: int = 50
    retention_checkpoints: list = field(default_factory=lambda: [16, 32, 64])
    idle_seconds: float = 10.0


@dataclass
class BioNeuralConfig:
    vocab_size: int = 1024
    token_dim: int = 256
    spike_ticks: int = 1  # token -> burst over N ticks (1 = fastest; 2 = richer temporal dynamics)
    k_active_per_tick: int = 8
    batch_size: int = 8
    log_every: int = 20
    seed: int = 0
    device: str = "auto"
    batch_window: int = 0  # >1 enables the batched training path (many tokens per GPU op)
    ctx_embed_weight: float = 1.0  # direct sensory (bottom-up) strength in the readout context
    ctx_proj_weight: float = 1.0  # task-aligned next-token-embedding predictor (P·h_p) in the readout context
    embssm_readout: bool = False  # predictive path via trained linear-attention over embeddings (EmbSSM)
    embssm_decay: float = 0.9  # EmbSSM state leakage a (h_t = a·h_{t-1} + W_in·emb[x_t])
    profile: bool = False  # per-phase wall-clock breakdown in the windowed training path

    quant: QuantConfig = field(default_factory=QuantConfig)
    cortex: CortexConfig = field(default_factory=CortexConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    drives: DriveConfig = field(default_factory=DriveConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BioNeuralConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BioNeuralConfig":
        merged = asdict(cls())

        def _recurse(schema: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
            for key, value in schema.items():
                if isinstance(value, dict) and isinstance(user.get(key), dict):
                    schema[key] = _recurse(dict(value), user[key])
                elif key in user:
                    schema[key] = user[key]
            return schema

        merged = _recurse(merged, data)

        def _build(dc_cls: Any, raw: dict[str, Any]) -> Any:
            kwargs = {}
            for f in fields(dc_cls):
                if f.name not in raw:
                    kwargs[f.name] = getattr(dc_cls(), f.name)
                    continue
                val = raw[f.name]
                if is_dataclass(getattr(dc_cls(), f.name)) and isinstance(val, dict):
                    kwargs[f.name] = _build(type(getattr(dc_cls(), f.name)), val)
                elif isinstance(val, dict):
                    kwargs[f.name] = val
                else:
                    kwargs[f.name] = val
            return dc_cls(**kwargs)

        return _build(cls, merged)

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self))

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
