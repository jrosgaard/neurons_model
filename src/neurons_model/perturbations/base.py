"""
src/neurons_model/perturbations/base.py
Base perturbation interfaces and config-to-runtime conversion helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .ablation import Ablation
from .excitability import ExcitabilityShift
from .hebbian import HebbianPlasticity
from .input_drive import InputDrive
from .synaptic_gain import SynapticGain

if TYPE_CHECKING:
    from ..simulation.config import PerturbationConfig
    from ..simulation.network import Network


class Perturbation(ABC):
    """Abstract contract for perturbations that mutate a runtime network."""

    @abstractmethod
    def apply_to_network(self, net: Network) -> None:
        """Apply this perturbation in place to the assembled runtime network."""


def perturbation_from_config(spec: PerturbationConfig) -> Perturbation:
    """
    Build a runtime perturbation object from a validated config entry.

    The network config stores perturbations in a compact schema. At runtime we
    expand those into explicit mutators so integration stays localized here.
    """
    if spec.kind == "ablation":
        from .ablation import Ablation

        return Ablation(target_population=spec.target, fraction=spec.value)

    if spec.kind == "excitability_shift":
        from .excitability import ExcitabilityShift

        return ExcitabilityShift(target_population=spec.target, delta_mv=spec.value)
    
    if spec.kind == "hebbian":
        from .hebbian import HebbianPlasticity

        return HebbianPlasticity(learning_rate=spec.value)

    if spec.kind == "input_drive":
        from .input_drive import InputDrive

        return InputDrive(target_population=spec.target, delta=spec.value)

    if spec.kind == "synaptic_gain":
        from .synaptic_gain import SynapticGain

        source, target = [part.strip() for part in spec.target.split("->", 1)]
        return SynapticGain(source_population=source,target_population=target,multiplier=spec.value)

    raise ValueError(f"Unsupported perturbation kind: {spec.kind}")
