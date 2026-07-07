"""
src/neurons_model/perturbations/input_drive.py
Perturbation for changing external input drive to a target population.
"""


from __future__ import annotations

from dataclasses import dataclass

from .base import Perturbation


@dataclass(frozen=True, slots=True)
class InputDrive(Perturbation):
    target_population: str
    delta: float = 0.0

    def apply_to_network(self, net) -> None:
        """
        Increase the configured external drive on the targeted population.

        For pulse- and ramp-like inputs this shifts ``amplitude``. For poisson
        inputs it shifts ``rate_hz``.
        """
        ext = net.config.external_input
        ext.target_population = net.resolve_population_target_name(self.target_population)

        if ext.mode == "poisson":
            ext.rate_hz = max(0.0, ext.rate_hz + self.delta)
        else:
            ext.amplitude += self.delta
