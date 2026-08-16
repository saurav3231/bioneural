"""Evaluation: the oracle. Every metric below is measured identically for BioNeural and the
standard model on the same conditions (matched params, matched wall-clock, matched data)."""

from bioneural.eval.benchmark import run_benchmark
from bioneural.eval.metrics import (
    EnergyMeter,
    bleu_2,
    distinct_2,
    intelligence_per_joule,
    perplexity_from_nll,
    softmax_cross_entropy,
    token_accuracy,
)
from bioneural.eval.standard_model import StandardTransformer

__all__ = [
    "bleu_2",
    "distinct_2",
    "EnergyMeter",
    "token_accuracy",
    "perplexity_from_nll",
    "softmax_cross_entropy",
    "intelligence_per_joule",
    "StandardTransformer",
    "run_benchmark",
]
