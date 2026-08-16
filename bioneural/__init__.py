"""BioNeural — a living, event-driven, quantized, backprop-free neural organism."""

__version__ = "0.1.0"

from bioneural.config import BioNeuralConfig, CortexConfig, QuantConfig
from bioneural.runtime.organism import BioNeural

__all__ = [
    "__version__",
    "BioNeuralConfig",
    "QuantConfig",
    "CortexConfig",
    "BioNeural",
]
