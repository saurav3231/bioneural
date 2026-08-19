import tempfile

from bioneural.config import BioNeuralConfig
from bioneural.data.loader import _synthetic_corpus
from bioneural.eval.standard_model import StandardTransformer
from bioneural.io.tokenizer import CharTokenizer
from bioneural.runtime.organism import BioNeural


def _tiny_cfg() -> BioNeuralConfig:
    cfg = BioNeuralConfig(vocab_size=32, token_dim=64, spike_ticks=2, k_active_per_tick=4)
    cfg.cortex.num_columns = 16
    cfg.cortex.neurons_per_column = 32
    cfg.cortex.input_dim = 64
    cfg.cortex.readout_dim = 64
    cfg.cortex.backbone_dim = 32
    cfg.memory.m2a_capacity = 32
    cfg.memory.m2b_max_engrams = 128
    return cfg


def _flat(cfg: BioNeuralConfig) -> list[int]:
    tok = CharTokenizer()
    cfg.vocab_size = tok.vocab_size
    return [x for t in _synthetic_corpus(20) for x in tok.encode(t)]


def test_organism_train_and_eval():
    cfg = _tiny_cfg()
    tokens = _flat(cfg)
    org = BioNeural(cfg)
    org.train_sequence(tokens[:96])
    ev = org.evaluate(tokens[:96])
    assert 0.0 <= ev["acc"] <= 1.0
    assert ev["ppl"] > 1.0


def test_ternary_materialization():
    import torch

    from bioneural.quant.kernels import materialize_ternary

    latent = torch.randn(64, 128) * 0.5
    wt, scale = materialize_ternary(latent, group_size=64, deadzone=0.15)
    assert wt.shape == latent.shape
    assert (wt[wt != 0].abs() > 0).all()


def test_one_shot_fast_weights():
    cfg = _tiny_cfg()
    tokens = _flat(cfg)
    org = BioNeural(cfg)
    org.process_token(tokens[0], learn=False)
    k = org._prev_sdc
    org.fabric.m2a.write(k, k, mod=1.0)
    assert org.fabric.m2a.recall(k) is not None


def test_checkpoint_roundtrip():
    cfg = _tiny_cfg()
    tokens = _flat(cfg)
    org = BioNeural(cfg)
    org.train_sequence(tokens[:64])
    with tempfile.TemporaryDirectory() as td:
        org.save_body(td)
        org2 = BioNeural(cfg)
        org2.load_body(td)
        assert org2.total_tokens == org.total_tokens


def test_standard_model_trains():
    cfg = _tiny_cfg()
    tokens = _flat(cfg)
    std = StandardTransformer(vocab_size=cfg.vocab_size, dim=32, n_layer=1, n_head=2, max_len=64)
    std.fit(tokens[:200], seconds=0.5, batch=4, seq=32)
    ev = std.evaluate(tokens[:200], seq=32, max_batches=2)
    assert ev["ppl"] > 1.0


def test_config_yaml_roundtrip():
    import yaml

    cfg = BioNeuralConfig()
    data = yaml.safe_load(cfg.to_yaml())
    cfg2 = BioNeuralConfig.from_dict(data)
    assert cfg2.cortex.num_columns == cfg.cortex.num_columns
    assert cfg2.eval.dataset == cfg.eval.dataset


def test_embssm_windowed_path():
    cfg = _tiny_cfg()
    cfg.batch_window = 16
    cfg.embssm_readout = True
    tokens = _flat(cfg)
    org = BioNeural(cfg)
    org.train_sequence(tokens[:96])
    tr = org.train_sequence(tokens[:96])
    assert 0.0 <= tr["acc"] <= 1.0
    assert tr["n"] == 95
    ev = org.evaluate_window(tokens[:96])
    assert 0.0 <= ev["acc"] <= 1.0
    assert ev["ppl"] > 1.0
    assert "ppl_ssm" in ev and "ppl_emb" in ev
    gen = org.generate([tokens[0]] * 4, n_tokens=8)
    assert len(gen) == 8


def _second_order_tokens(n: int, seed: int = 0) -> list[int]:
    import random

    rng = random.Random(seed)
    ids = [rng.randrange(4), rng.randrange(4)]
    for _ in range(n):
        ids.append((2 * ids[-1] + ids[-2]) % 4)
    return ids


def test_qstate_second_order_organism():
    """QState must crack a 2nd-order Markov task inside the organism — the test the linear SSM
    failed (β collapsed to the floor, acc stuck at random). The rule is unpredictable from the
    bigram alone, so only a stateful higher-order channel can lift acc off the 0.25 floor."""
    cfg = _tiny_cfg()
    cfg.vocab_size = 4
    cfg.batch_window = 16
    cfg.embssm_readout = True
    cfg.embssm_qstate = True
    cfg.embssm_qdim = cfg.cortex.readout_dim // 2
    tokens = _second_order_tokens(2048)
    org = BioNeural(cfg)
    org.train_sequence(tokens[:1536])
    ev = org.evaluate_window(tokens[1536:])
    assert ev["ppl"] > 1.0
    assert org.embssm.beta.detach().item() > 0.1, f"beta collapsed: {org.embssm.beta.item()}"
    assert ev["acc"] > 0.4, f"qstate did not learn 2nd-order structure: acc={ev['acc']}"
