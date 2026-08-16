"""Quick offline self-test: imports, tiny synthetic training, memory, checkpoint round-trip.

Runs entirely on CPU with no internet. `--quick` runs a miniature version for CI.
"""

from __future__ import annotations

import argparse
import tempfile

from bioneural.config import BioNeuralConfig


def smoke(quick: bool = False) -> None:
    cfg = BioNeuralConfig(
        vocab_size=32,
        token_dim=64,
        spike_ticks=2,
        k_active_per_tick=4,
        cortex=BioNeuralConfig().cortex,
    )
    cfg.cortex.num_columns = 16
    cfg.cortex.neurons_per_column = 32
    cfg.cortex.input_dim = 64
    cfg.cortex.readout_dim = 64
    cfg.cortex.backbone_dim = 32
    cfg.memory.m2a_capacity = 32
    cfg.memory.m2b_max_engrams = 128
    cfg.memory.m3_max_concepts = 64
    cfg.eval.gen_length = 16

    from bioneural.data.loader import _synthetic_corpus
    from bioneural.io.tokenizer import CharTokenizer
    from bioneural.runtime.organism import BioNeural

    tok = CharTokenizer()
    cfg.vocab_size = tok.vocab_size
    texts = _synthetic_corpus(40, seed=0)
    train = [tok.encode(t) for t in texts]
    flat = [x for s in train for x in s]

    org = BioNeural(cfg)
    print(f"[smoke] created BioNeural, params={org.n_params():,} device={org.device}")
    assert org.device.type in ("cpu", "cuda")

    steps = 3 if quick else 12
    for _ in range(steps):
        seg = flat[:48]
        org.train_sequence(seg)
    print(f"[smoke] trained {steps} seqs, acc={org.total_correct / max(org.total_tokens, 1):.3f}")

    # one-shot retention
    org.fabric.m2a.write(org._prev_sdc, org._prev_sdc, mod=1.0)
    rec = org.fabric.m2a.recall(org._prev_sdc)
    assert rec is not None, "M2a recall failed"

    # generation
    gen = org.generate(flat[:8], 8)
    assert len(gen) == 8

    # sleep
    si = org.sleep_cycle(replay_n=4)
    print(f"[smoke] sleep: {si}")

    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as td:
        org.save_body(td)
        org2 = BioNeural(cfg)
        org2.load_body(td)
        assert org2.total_tokens == org.total_tokens, "checkpoint token mismatch"
    print("[smoke] checkpoint round-trip OK")

    # standard model sanity
    from bioneural.eval.standard_model import StandardTransformer

    std = StandardTransformer(vocab_size=cfg.vocab_size, dim=32, n_layer=1, n_head=2, max_len=64)
    std.fit(flat, seconds=1.0, batch=4, seq=32)
    ev = std.evaluate(flat[:128], seq=32, max_batches=2)
    print(f"[smoke] standard model: nll={ev['nll']:.3f} params={std.n_params()}")
    print("[smoke] ALL OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="minimal CI run")
    args = p.parse_args()
    smoke(quick=args.quick)


if __name__ == "__main__":
    main()
