"""
src/neurons_model/perturbations/synaptic_gain.py
Perturbation for changing synaptic gain on one pathway block.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import Perturbation


@dataclass(frozen=True, slots=True)
class SynapticGain(Perturbation):
    source_population: str
    target_population: str
    multiplier: float = 1.0

    def apply_to_network(self, net) -> None:
        src_idx = net.population_indices(self.source_population)
        tgt_idx = net.population_indices(self.target_population)
        net.weights[np.ix_(src_idx, tgt_idx)] *= self.multiplier
