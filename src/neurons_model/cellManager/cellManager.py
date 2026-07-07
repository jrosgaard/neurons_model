"""
src/neurons_model/cellManager/cellManager.py

Registry of custom cell types (CellTypeSpec) that can be injected into a
NetworkConfig before building a Network. This complements the built-in presets
by allowing additional cell types (e.g. glial populations) to be registered
from JSON or YAML files at runtime.
"""

from __future__ import annotations

import dataclasses

from ..cell_types import CellTypeSpec
from ..simulation.config import PopulationConfig


class CellManager:
    """
    Registry of custom cell type specs.

    Cell types are keyed by their name and stored as CellTypeSpec objects,
    which can be converted to PopulationConfig for injection into a NetworkConfig.
    """

    def __init__(self) -> None:
        self._registry: dict[str, CellTypeSpec] = {}

    def register(self, spec: CellTypeSpec) -> None:
        """Add or overwrite a cell type spec in the registry."""
        self._registry[spec.name] = spec

    def get(self, name: str) -> CellTypeSpec:
        """Return the spec for *name*, raising KeyError if not registered."""
        return self._registry[name]

    def list_available(self) -> list[str]:
        """Return the names of all registered cell types."""
        return list(self._registry.keys())
    
    def cell_data(self, name: str) -> dict:
        """
        Return a serialisable dict for the registered spec *name*.

        The dict shape mirrors a single entry in a YAML populations list and
        can be passed directly to ``CellImport.save_to_cellClasses()`` or
        round-tripped back through ``CellTypeSpec.from_mapping()``.
        """
        return dataclasses.asdict(self._registry[name])

    def to_population_config(self, name: str) -> PopulationConfig:
        """
        Convert the registered spec for *name* to a PopulationConfig that can
        be appended to a NetworkConfig's populations list.

        Example::

            cfg.populations.append(cell_manager.to_population_config("astro"))
        """
        return PopulationConfig.from_cell_type_spec(self._registry[name])
