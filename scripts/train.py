"""Train a BioNeural organism from a config file or defaults."""

from __future__ import annotations

import argparse

from bioneural.config import BioNeuralConfig
from bioneural.data.loader import load_dataset
from bioneural.io.tokenizer import build_tokenizer
from bioneural.runtime.organism import BioNeural


def main() -> None:
    p = argparse.ArgumentParser(description="Train a BioNeural organism (local, backprop-free).")
    p.add_argument("--config", default="configs/bioneural.yaml", help="YAML config path")
    p.add_argument("--dataset", default=None, help="tiny-stories | wikitext-2 | synthetic")
    p.add_argument("--steps", type=int, default=500, help="training sequences")
    p.add_argument("--checkpoint", default="checkpoints/bioneural", help="body output dir")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = BioNeuralConfig.from_yaml(args.config)
    if args.dataset:
        cfg.eval.dataset = args.dataset
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"[bioneural] loading dataset '{cfg.eval.dataset}' ...")
    texts = load_dataset(cfg.eval.dataset, cfg.eval.max_examples, seed=cfg.seed)
    tokenizer = build_tokenizer(texts[:200], cfg.vocab_size)
    cfg.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.vocab
    train = [tokenizer.encode(t) for t in texts[: -cfg.eval.max_val_examples]]
    flat = [t for seq in train for t in seq]

    org = BioNeural(cfg)
    print(f"[bioneural] params={org.n_params():,} device={org.device}")
    n = len(flat) - 64
    i = 0
    for step in range(args.steps):
        seg = flat[i : i + 64]
        r = org.train_sequence(seg)
        i = (i + 64) % max(n, 1)
        if step % cfg.log_every == 0:
            print(
                f"[step {step}] acc={r['acc']:.3f} NE={r['ne_mean']:.3f} "
                f"tokens={org.total_tokens} drives={ {k: round(v, 2) for k, v in org.drives.state().items()} }"
            )
    ev = org.evaluate([t for seq in train[-cfg.eval.max_val_examples :] for t in seq])
    print(f"[eval] acc={ev['acc']:.4f} ppl={ev['ppl']:.3f}")
    org.save_body(args.checkpoint)
    print(f"[bioneural] body saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
