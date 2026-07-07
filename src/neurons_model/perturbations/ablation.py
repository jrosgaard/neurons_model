"""
src/neurons_model/perturbations/ablation.py
Perturbations that silence selected neurons in the assembled network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import Perturbation


AblationMethod = Literal["zero_ablation",
                         "functional_ablation",
                         "synaptic_ablation",
                         ]


@dataclass(frozen=True, slots=True)
class Ablation(Perturbation):
    target_population: str | None = None
    fraction: float | None = None
    cell_list: list[int] | None = None
    method: AblationMethod = "zero_ablation"
    random_seed: int = 34

    def apply_to_network(self, net) -> None:
        """Apply the ablation directly to the runtime network."""
        if self.method != "zero_ablation":
            raise ValueError(f"Unsupported ablation method: {self.method}")

        if self.target_population is not None and self.fraction is not None:
            net.ablate_neurons_by_fraction(
                self.target_population,
                self.fraction,
                self.random_seed,
            )
            return

        if self.cell_list is not None:
            net.ablate_neurons_by_cell_list(self.cell_list)
            return

        raise ValueError(
            "Invalid ablation configuration: must specify either "
            "target_population and fraction, or cell_list."
        )


def neuron_fraction_ablation(target_population: str, fraction: float = 0.1, 
                             method: AblationMethod = "zero_ablation", seed: int = 34,
                             ) -> Ablation:
    """Create an ablation perturbation for a fraction of a population."""
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0, 1].")

    return Ablation(target_population=target_population, fraction=fraction,
                    method=method, random_seed=seed,)


def neuron_list_ablation(cell_list: list[int], 
                         method: AblationMethod = "zero_ablation", seed: int = 34,
                         ) -> Ablation:
    """Create an ablation perturbation for an explicit list of neurons."""
    if len(cell_list) == 0:
        raise ValueError("cell_list must not be empty.")

    return Ablation(cell_list=sorted(set(cell_list)),
                    method=method, random_seed=seed,)


def functional_ablation(*args, **kwargs):
    raise NotImplementedError(
        "functional_ablation is not implemented yet. "
        "Define neuron selection criteria first."
    )
