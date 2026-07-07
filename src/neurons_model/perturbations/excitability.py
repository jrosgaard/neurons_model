"""
src/neurons_model/perturbations/excitability.py
Perturbation for adjusting intrinsic excitability of one population.

"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Perturbation


@dataclass(frozen=True, slots=True)
class ExcitabilityShift(Perturbation):
    target_population: str
    delta_mv: float = 2.0

    def apply_to_network(self, net) -> None:
        """
        Lower the spike threshold by ``delta_mv`` for the targeted population.

        Positive values therefore make the neurons more excitable.
        """
        indices = net.population_indices(self.target_population)
        net.v_threshold_mv[indices] -= self.delta_mv
