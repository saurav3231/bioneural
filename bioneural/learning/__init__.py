"""Learning without backpropagation: local plasticity rules + readout heads."""

from bioneural.learning.hebbian import eligibility_coact
from bioneural.learning.homeostat import apply_synaptic_scaling
from bioneural.learning.predictive import SurpriseTracker
from bioneural.learning.readout import ReadoutHead

__all__ = ["SurpriseTracker", "eligibility_coact", "apply_synaptic_scaling", "ReadoutHead"]
