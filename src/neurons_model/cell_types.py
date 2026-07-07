"""
src/neurons_model/cell_types.py
Definitions of cell populations used by the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_POPULATION_PARAMETER_ALIASES = {
    "baseline_input_au": "baseline_input",
    "noise_std_au": "noise_std",
    "perturb_std_au": "noise_std",
}

_POPULATION_PARAMETER_FIELDS = {
    "v_rest_mv",
    "v_reset_mv",
    "v_threshold_mv",
    "tau_m_ms",
    "tau_refractory_ms",
    "adaptation_strength",
    "baseline_input",
    "noise_std",
}


@dataclass(slots=True)
class CellTypeSpec:
    """
    Declarative spec for one cell type in the network.

    The common fields identify and size the population. Type-specific numeric
    knobs live in ``parameters`` and structured biology/extension metadata lives
    in ``metadata``. This lets extension cell types, such as GBMsim glioma
    cells, share one core shape without forcing every field into the neuron
    simulator immediately.
    """
    name: str
    kind: str
    count_n: int
    cell_class: str = "neuron"
    dynamics: str = "lif"
    is_spiking: bool = True
    parameters: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Backward-compatible alias for older code that used ``count``."""
        return self.count_n

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CellTypeSpec":
        """Build a CellTypeSpec from a preset/extension mapping."""
        name = str(data["name"])
        kind = str(data["kind"])
        count_n = int(data.get("count_n", data.get("count", 0)))
        if count_n <= 0:
            raise ValueError(f"Cell type '{name}' must have a positive count_n.")

        parameters: dict[str, float] = {}
        metadata: dict[str, Any] = {}
        reserved = {
            "name",
            "kind",
            "count",
            "count_n",
            "class",
            "cell_class",
            "dynamics",
            "is_spiking",
            "parameters",
            "metadata",
            "membrane",
        }

        for key, value in dict(data.get("parameters") or {}).items():
            _add_numeric_parameter(parameters, key, value)

        membrane = data.get("membrane")
        if isinstance(membrane, Mapping):
            metadata["membrane"] = dict(membrane)
            for key, value in membrane.items():
                if key == "enabled":
                    continue
                _add_numeric_parameter(parameters, key, value)

        for key, value in data.items():
            if key in reserved:
                continue
            if _add_numeric_parameter(parameters, key, value):
                continue
            metadata[key] = value

        metadata.update(dict(data.get("metadata") or {}))

        return cls(
            name=name,
            kind=kind,
            count_n=count_n,
            cell_class=str(data.get("cell_class", data.get("class", "neuron"))),
            dynamics=str(data.get("dynamics", "lif")),
            is_spiking=bool(data.get("is_spiking", True)),
            parameters=parameters,
            metadata=metadata,
        )

    def to_population_config_dict(self) -> dict[str, Any]:
        """
        Convert the spec to the dictionary shape accepted by PopulationConfig.

        Only simulator-known membrane fields are promoted. Extension-specific
        numeric parameters remain attached under ``parameters``.
        """
        population_data: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "count_n": self.count_n,
            "cell_class": self.cell_class,
            "dynamics": self.dynamics,
            "is_spiking": self.is_spiking,
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }
        for source_key, value in self.parameters.items():
            target_key = _POPULATION_PARAMETER_ALIASES.get(source_key, source_key)
            if target_key in _POPULATION_PARAMETER_FIELDS:
                population_data[target_key] = value

        return population_data


def load_cell_type_specs(path: str | Path, section: str | None = None) -> list[CellTypeSpec]:
    """
    Load cell type specs from a YAML file.

    If ``section`` is omitted, the first list-valued key that looks like a cell
    collection is used, e.g. ``glioma_cells`` or ``glial_populations``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, Mapping):
        raise ValueError(f"Cell type file {path} must define a mapping at the top level.")

    if section is None:
        section = _infer_cell_type_section(raw)

    entries = raw.get(section)
    if not isinstance(entries, list):
        raise ValueError(f"Cell type section '{section}' in {path} must be a list.")

    return [CellTypeSpec.from_mapping(entry) for entry in entries]


def _infer_cell_type_section(raw: Mapping[str, Any]) -> str:
    for key, value in raw.items():
        if isinstance(value, list) and ("cell" in key or "population" in key or key == "neurons"):
            return key
    raise ValueError("Could not infer a cell type section.")


def _add_numeric_parameter(parameters: dict[str, float], key: str, value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    parameters[key] = float(value)
    return True
