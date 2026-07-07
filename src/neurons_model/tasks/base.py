"""Base task contract for benchmark routines.

TODO: Keep only if task evaluation becomes a first-class pipeline feature.
Current simulations do not dispatch through task objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """Abstract task API."""

    @abstractmethod
    def run(self, simulation_result: dict[str, Any]) -> dict[str, Any]:
        """Consume simulation output and return task-specific metrics."""


class DiscriminationTask(Task):
    """Evaluate simple condition discrimination performance."""

    def run(self, simulation_result: dict[str, Any]) -> dict[str, Any]:
        cell_types = float(simulation_result.get("cell_types", 0))
        return {"discrimination_index": cell_types / (cell_types + 1.0)}
    

class RelayTask(Task):
    """Evaluate information relay quality."""

    def run(self, simulation_result: dict[str, Any]) -> dict[str, Any]:
        steps = float(simulation_result.get("steps", 0))
        return {"relay_score": steps / (steps + 1.0)}
