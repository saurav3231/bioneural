"""BioNeural quantization package: ternary weights and accelerated matmul kernels."""

from bioneural.quant.kernels import materialize_ternary, ternary_matmul, ternary_matmul_triton
from bioneural.quant.ternary import TernaryParam

__all__ = ["TernaryParam", "materialize_ternary", "ternary_matmul", "ternary_matmul_triton"]
