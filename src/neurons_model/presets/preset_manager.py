"""
src/neurons_model/presets/preset_manager.py
Manager for loading and handling presets of neuron models and simulations.
"""


from __future__ import annotations

import os
from pathlib import Path
import yaml
from typing import Dict, Any
import numpy as np
import pandas as pd

from neurons_model.cellManager.cellManager import CellManager
from neurons_model.cellManager.cellImport import CellImport

from neurons_model.simulation.config import NetworkConfig, PathwayConfig


_PRESET_DIR = Path(__file__).resolve().parent

# Curated presets that are listed by default in the CLI and documentation. 
# The keys are the preset names and the values are the corresponding YAML file names in the preset directory.
_PRESETS = {
    "healthy_quiet": "healthy_quiet.yaml",
    "healthy_active_1": "healthy_active_1.yaml",
    "healthy_transition_1": "healthy_transition_1.yaml",
    "healthy_unstable_1": "healthy_unstable_1.yaml",
}


def available_preset_paths(include_unlisted: bool = False) -> dict[str, Path]:
    """Return visible preset files keyed by preset stem.

    When ``include_unlisted`` is true, include every YAML preset in the preset
    directory in addition to the curated visible set.
    """
    presets = {name: _PRESET_DIR / filename 
               for name, filename in _PRESETS.items()}

    if include_unlisted:
        for path in sorted(_PRESET_DIR.glob("*.yaml")):
            presets.setdefault(path.stem, path)

    return presets


def _read_preset_yaml(path: Path) -> dict[str, Any]:
    """Read a preset YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Preset file {path} must define a mapping at the top level.")

    return raw


def _validate_preset(raw: dict[str, Any], path: Path) -> NetworkConfig:
    """Validate a preset mapping as a NetworkConfig."""
    try:
        return NetworkConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid preset at {path}: {exc}") from exc


def resolve_preset_path(preset_name: str, allow_unlisted: bool | None = None) -> Path:
    """Resolve a preset name to a file path."""
    if allow_unlisted is None:
        allow_unlisted = False
    
    presets = available_preset_paths(include_unlisted=allow_unlisted)
    normalized_name = Path(preset_name).stem

    if normalized_name in presets:
        return presets[normalized_name]

    if not allow_unlisted:
        raise ValueError(f"Preset '{preset_name}' not found. "
                         f"Available presets: {list(presets.keys())}")

    candidate_inputs = [Path(preset_name).expanduser()]
    if candidate_inputs[0].suffix == "":
        candidate_inputs.append(candidate_inputs[0].with_suffix(".yaml"))

    candidates: list[Path] = []
    for candidate in candidate_inputs:
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append((_PRESET_DIR / candidate).resolve())
            candidates.append(candidate.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise ValueError(
        f"Unlisted preset '{preset_name}' not found. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def load_named_preset(preset_name: str, unlisted_preset: bool | None = None,
                      ) -> NetworkConfig:
    """Load and validate a preset by name or, optionally, custom path."""
    if unlisted_preset is None:
        unlisted_preset = False
    preset_path = resolve_preset_path(preset_name, allow_unlisted=unlisted_preset)
    raw = _read_preset_yaml(preset_path)
    return _validate_preset(raw, preset_path)


def select_preset(preset_name: str, unlisted_preset: bool | None = None,
                  ) -> dict[str, Any]:
    """
    Select a preset by name and return the validated config as a dictionary.
    """
    if unlisted_preset is None:
        unlisted_preset = False
    
    cfg = load_named_preset(preset_name, unlisted_preset=unlisted_preset)
    return cfg.model_dump(mode="python")


def list_presets() -> dict[str, str]:
    """
    List available presets and their descriptions.
    """
    descriptions: dict[str, str] = {}
    preset_paths = {name: _PRESET_DIR / filename 
            for name, filename in _PRESETS.items()}
    
    for preset_name, preset_path in preset_paths.items():
        raw = _read_preset_yaml(preset_path)
        descriptions[preset_name] = raw.get("description",
                                            f"No description for '{preset_name}'.",
                                            )
    return descriptions


def describe_preset(preset_name: str) -> tuple[str, dict[str, float]]:
    """
    Return a preset description and a compact mapping of pathway weights.
    """
    preset_path = resolve_preset_path(preset_name)
    raw = _read_preset_yaml(preset_path)

    text_description = raw.get("description", 
                               "No description for this preset.",
                               )

    pathways = raw.get("pathways", [])
    pathway_weights: dict[str, float] = {}
    
    if isinstance(pathways, list):
        for pathway in pathways:
            if not isinstance(pathway, dict):
                continue
            source = pathway.get("source")
            target = pathway.get("target")
            weight = pathway.get("weight")
            if source is None or target is None or weight is None:
                continue
            try:
                pathway_weights[f"{source}->{target}"] = float(weight)
            except (TypeError, ValueError):
                continue

    return text_description, pathway_weights


# ----------------------------------------------------------------------------------------------------------
# The following functions are for the "expanded" preset, which is a work in progress and not listed by default.

# The "expanded" preset is a work in progress that includes more cell types and pathways than the curated presets.
# It is not listed by default but can be accessed by name or path.

_EXPANDED_PRESET_DIR = _PRESET_DIR / "expanded_preset"

_PRESET_EXPANDED = {
    "expanded_parameters": "expanded_params.yaml",
    "neurons": "neurons.yaml",
    "glia": "glia.yaml",
    "neuron_synapses": "neuron_synapses.yaml",
    "glial_modulation": "glial_modulation.yaml",
}

def load_expanded_preset() -> NetworkConfig:
    """
    Assemble the full expanded preset as a validated NetworkConfig.

    Reads simulation parameters from expanded_params.yaml, cell type specs
    from neurons.yaml and glia.yaml, and synaptic pathways from
    neuron_synapses.yaml and glial_modulation.yaml.

    Pathway entries that are missing required PathwayConfig fields (e.g. glial
    modulation entries that lack tau_rise_ms / reversal_mv) are skipped with a
    warning until PathwayConfig gains a metadata field for extension data.
    """
    # --- Parameters (simulation, task, external_input, readouts, name) ---
    raw_params = load_expanded_preset_parameters()

    # Keep only keys NetworkConfig recognises; drop include, fields, schema_version etc.
    _KNOWN_KEYS = {
        "name", "description", "populations", "pathways",
        "simulation", "external_input", "task", "perturbations", "readouts",
    }
    config_dict: dict[str, Any] = {k: v for k, v in raw_params.items() if k in _KNOWN_KEYS}

    # Restrict readouts to currently supported kinds so validation does not fail
    # on extended readout types (connectivity, synchrony, cytotoxicity, etc.)
    _SUPPORTED_READOUT_KINDS = {"spikes", "rates", "lfp_proxy", "metrics", "spectra"}
    raw_readouts = config_dict.get("readouts", {})
    if isinstance(raw_readouts, dict):
        enabled = raw_readouts.get("enabled") or []
        config_dict["readouts"] = {
            "enabled": [k for k in enabled if k in _SUPPORTED_READOUT_KINDS]
        }

    # --- Populations from CellManager ---
    cell_manager = load_expanded_preset_cells()
    config_dict["populations"] = [
        cell_manager.to_population_config(name).model_dump(mode="python")
        for name in cell_manager.list_available()
    ]

    # --- Pathways: skip entries missing required PathwayConfig fields ---
    config_dict["pathways"] = []
    for pathway_data in load_expanded_preset_behavior():
        try:
            PathwayConfig.model_validate(pathway_data)
            config_dict["pathways"].append(pathway_data)
        except Exception as exc:
            src = pathway_data.get("source", "?")
            tgt = pathway_data.get("target", "?")
            print(f"Warning: skipping pathway {src}->{tgt} (missing required fields): {exc}")

    return NetworkConfig.model_validate(config_dict)


def load_expanded_preset_parameters() -> dict[str, Any]:
    """
    Load the simulation/task/readout parameters from the expanded preset.
    Returns the raw parameter dict (not a full NetworkConfig, as this file
    does not contain populations or pathways).
    """
    preset_path = _EXPANDED_PRESET_DIR / _PRESET_EXPANDED["expanded_parameters"]
    return _read_preset_yaml(preset_path)


def load_expanded_preset_cells() -> CellManager:
    """
    Load neuron and glial cell type specs from the expanded preset and
    return a populated CellManager registry.

    The returned CellManager can be used to inject populations into an
    existing NetworkConfig via ``cell_manager.to_population_config(name)``.
    """
    manager = CellManager()
    importer = CellImport(manager)
    for key in ("neurons", "glia"):
        preset_path = _EXPANDED_PRESET_DIR / _PRESET_EXPANDED[key]
        importer.import_from_yaml(preset_path)
    return manager


def load_expanded_preset_behavior() -> list[dict[str, Any]]:
    """
    Load synaptic and glial modulation pathways from the expanded preset.
    Returns a combined list of raw pathway dicts from both neuron_synapses.yaml
    and glial_modulation.yaml.

    Note: glial pathways carry extra fields (e.g. ``dopa_modulation``,
    ``cytotoxicity``) that are not yet part of PathwayConfig; they are
    returned as raw dicts so callers can handle them appropriately.
    """
    pathways: list[dict[str, Any]] = []
    for key in ("neuron_synapses", "glial_modulation"):
        preset_path = _EXPANDED_PRESET_DIR / _PRESET_EXPANDED[key]
        raw = _read_preset_yaml(preset_path)
        entries = raw.get("pathways", [])
        if isinstance(entries, list):
            pathways.extend(entries)
    return pathways
