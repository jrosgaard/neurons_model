"""
src/neurons_model/readouts/base.py
Base readout contract.

TODO: Keep only to formalize readouts behind a shared interface.
The current simulation path returns raw outputs directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Readout(ABC):
    """Readout API for reducing simulation outputs."""

    @abstractmethod
    def compute(self, simulation_result: dict[str, Any]) -> dict[str, Any]:
        """Compute readout values from simulation output."""
