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


def test_qstate_learned_R_gradient():
    """The QState qubit-angle gradient must match autograd on the equivalent step-by-step
    recurrence (scaled by 1/w, the module's W_in-consistent normalization)."""
    import torch

    from bioneural.cortex.qstate import QState

    torch.manual_seed(0)
    dim, emb, nb, w = 16, 8, 8, 96
    q = QState(dim=dim, emb_dim=emb, vocab_size=8, lr=0.1, head_lr=0.05, decay=0.9, pairs=8, seed=0, learn=True)
    a = q.a
    e = torch.randn(w, emb)

    theta = q.theta.detach().clone().requires_grad_(True)
    phi = q.phi.detach().clone().requires_grad_(True)

    def build_R(th, ph):
        c = torch.cos(th)
        s = torch.sin(th)
        ep = torch.exp(1j * ph)
        R = torch.zeros(nb, 2, 2, dtype=torch.complex64)
        R[:, 0, 0] = c
        R[:, 0, 1] = s * ep
        R[:, 1, 0] = -s * torch.conj(ep)
        R[:, 1, 1] = c
        return R

    r = (e.float() @ q.W_in.detach()[0].t()).to(torch.complex64).reshape(w, nb, 2)
    R = build_R(theta, phi)
    carry = torch.zeros(nb, 2, dtype=torch.complex64)
    h_list = []
    for t in range(w):
        carry = a * torch.einsum("bij,bj->bi", R, carry) + r[t]
        h_list.append(carry)
    h_ag = torch.stack(h_list)
    target = torch.randn(w, nb, 2, dtype=torch.complex64)
    L = (torch.conj(target) * h_ag).real.sum()
    L.backward()
    gth_ref, gph_ref = theta.grad.clone(), phi.grad.clone()

    theta0 = q.theta.data.clone()
    phi0 = q.phi.data.clone()
    q.scan_window(e)
    q.apply_grad_ctx([target.reshape(w, dim)], e, mod=1.0)
    dth = (theta0 - q.theta.data) / q.lr + q.wd * theta0  # sub_ (descent) -> (old-new)/lr = +grad
    dph = (phi0 - q.phi.data) / q.lr + q.wd * phi0
    assert (dth * w - gth_ref).abs().max().item() < 1e-3
    assert (dph * w - gph_ref).abs().max().item() < 1e-3


def test_qstate_learned_R_organism():
    """With learn=True the qubit angles must actually move and the organism must still learn
    the 2nd-order rule."""
    cfg = _tiny_cfg()
    cfg.vocab_size = 4
    cfg.batch_window = 16
    cfg.embssm_readout = True
    cfg.embssm_qstate = True
    cfg.embssm_qdim = cfg.cortex.readout_dim // 2
    cfg.embssm_qlearn = True
    tokens = _second_order_tokens(2048)
    org = BioNeural(cfg)
    th0 = org.embssm.theta.detach().clone()
    ph0 = org.embssm.phi.detach().clone()
    org.train_sequence(tokens[:1536])
    ev = org.evaluate_window(tokens[1536:])
    moved = (org.embssm.theta.detach() - th0).abs().max().item() > 1e-6 or (
        org.embssm.phi.detach() - ph0
    ).abs().max().item() > 1e-6
    assert moved, "learned R angles did not move"
    assert ev["acc"] > 0.4, f"learned R broke 2nd-order learning: acc={ev['acc']}"


def test_qstate_multiscale_gradient():
    """Multi-scale QState (nch=3 channels) must match a step-by-step brute-force recurrence on
    both the forward scan and the qubit-angle gradient, per channel."""
    import torch

    from bioneural.cortex.qstate import QState

    torch.manual_seed(0)
    dim, emb, nb, w = 16, 8, 8, 96
    decays = (0.5, 0.9, 0.99)
    q = QState(
        dim=dim,
        emb_dim=emb,
        vocab_size=8,
        lr=0.1,
        head_lr=0.05,
        decay=0.9,
        pairs=8,
        seed=0,
        learn=True,
        decays=decays,
    )
    nch = len(decays)
    e = torch.randn(w, emb)

    def build_R(th, ph):
        c = torch.cos(th)
        s = torch.sin(th)
        ep = torch.exp(1j * ph)
        R = torch.zeros(nb, 2, 2, dtype=torch.complex64)
        R[:, 0, 0] = c
        R[:, 0, 1] = s * ep
        R[:, 1, 0] = -s * torch.conj(ep)
        R[:, 1, 1] = c
        return R

    theta = q.theta.detach().clone().requires_grad_(True)
    phi = q.phi.detach().clone().requires_grad_(True)

    refs = []
    h_full = []
    for c in range(nch):
        R = build_R(theta[c], phi[c])
        r = (e.float() @ q.W_in.detach()[c].t()).to(torch.complex64).reshape(w, nb, 2)
        carry = torch.zeros(nb, 2, dtype=torch.complex64)
        h_list = []
        for t in range(w):
            carry = decays[c] * torch.einsum("bij,bj->bi", R, carry) + r[t]
            h_list.append(carry)
        h_ag = torch.stack(h_list)
        h_full.append(h_ag.reshape(w, dim))
        target = torch.randn(w, nb, 2, dtype=torch.complex64)
        L = (torch.conj(target) * h_ag).real.sum()
        L.backward(retain_graph=True)
        refs.append((theta.grad[c].clone(), phi.grad[c].clone(), target.reshape(w, dim)))

    hf = q.scan_window(e)[1].reshape(w, nch, dim)
    for c in range(nch):
        err = (hf[:, c] - h_full[c]).abs().max().item()
        assert err < 1e-3, f"multiscale channel {c} forward err {err}"

    theta0 = q.theta.data.clone()
    phi0 = q.phi.data.clone()
    q.apply_grad_ctx([t[2] for t in refs], e, mod=1.0)
    dth = (theta0 - q.theta.data) / q.lr + q.wd * theta0
    dph = (phi0 - q.phi.data) / q.lr + q.wd * phi0
    for c in range(nch):
        gth_ref, gph_ref, _ = refs[c]
        assert (dth[c] * w - gth_ref).abs().max().item() < 1e-2, f"multiscale theta grad ch {c}"
        assert (dph[c] * w - gph_ref).abs().max().item() < 1e-2, f"multiscale phi grad ch {c}"
