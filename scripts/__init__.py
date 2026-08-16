"""CLI entry points."""

from bioneural.eval.benchmark import run_benchmark
from bioneural.eval.standard_model import StandardTransformer

__all__ = ["run_benchmark", "StandardTransformer"]
