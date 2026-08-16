"""Run the full head-to-head benchmark and write the report to results/<run>/."""

from __future__ import annotations

import argparse
import json

from bioneural.config import BioNeuralConfig
from bioneural.eval.benchmark import run_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description="BioNeural vs standard Transformer benchmark.")
    p.add_argument("--config", default="configs/bioneural.yaml")
    p.add_argument("--minutes", type=float, default=None, help="train budget in minutes")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--results", default="results")
    p.add_argument(
        "--quick",
        action="store_true",
        help="tiny config + synthetic data + ~1 min budget (for slow machines / sanity)",
    )
    args = p.parse_args()

    if args.quick:
        cfg = BioNeuralConfig(
            vocab_size=64,
            token_dim=64,
            spike_ticks=2,
            k_active_per_tick=4,
            batch_size=4,
        )
        cfg.cortex.num_columns = 16
        cfg.cortex.neurons_per_column = 32
        cfg.cortex.input_dim = 64
        cfg.cortex.readout_dim = 64
        cfg.cortex.backbone_dim = 32
        cfg.memory.m2a_capacity = 32
        cfg.memory.m2b_max_engrams = 256
        cfg.eval.train_budget_minutes = args.minutes if args.minutes else 0.5
        cfg.eval.dataset = args.dataset or "synthetic"
        cfg.eval.max_examples = args.max_examples or 300
        cfg.eval.gen_length = 16
    else:
        cfg = BioNeuralConfig.from_yaml(args.config)
        if args.minutes is not None:
            cfg.eval.train_budget_minutes = args.minutes
        if args.dataset is not None:
            cfg.eval.dataset = args.dataset
        if args.max_examples is not None:
            cfg.eval.max_examples = args.max_examples

    report = run_benchmark(cfg, results_dir=args.results)
    h = report["head2head"]
    print("\n" + "=" * 60)
    print(f"  RUN: {report['run_dir']}")
    print(f"  ppl    BioNeural={h['ppl_bio']:.3f}  Standard={h['ppl_std']:.3f}")
    print(f"  acc    BioNeural={h['acc_bio']:.4f}  Standard={h['acc_std']:.4f}")
    print(f"  tok/s  BioNeural={h['tokps_bio']:.1f}  Standard={h['tokps_std']:.1f}")
    print(f"  I/J    BioNeural={h['ij_bio']:.4f}  Standard={h['ij_std']:.4f}")
    print("=" * 60)
    print(json.dumps(h, indent=2))


if __name__ == "__main__":
    main()
