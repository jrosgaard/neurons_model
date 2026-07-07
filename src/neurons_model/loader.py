"""
src/neurons_model/loader.py

This module provides functionality to load preset configurations from YAML files.
"""

from __future__ import annotations

from pathlib import Path
import yaml
import json
from typing import Any

from .simulation.config import NetworkConfig
from .simulation.network import Network
from .simulation.scaling import scale_network_config
from .presets.preset_manager import load_named_preset


DEFAULT_SIM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "examples" / "example_sim1.json"


def load_preset(path: str | Path) -> NetworkConfig:
    """Load a YAML preset file and return a validated NetworkConfig."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return NetworkConfig.model_validate(data)


def load_sim_config(path: str | Path | None = None) -> tuple[Network, dict[str, Any]]:
    """
    Load a JSON simulation configuration.

    Returns a built Network plus keyword arguments that can be passed directly
    to ``run_simulation``.
    """
    path = DEFAULT_SIM_CONFIG_PATH if path is None else Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Simulation config {path} must define a JSON object.")

    preset_name = data.get("preset_name", data.get("preset"))
    if not isinstance(preset_name, str) or not preset_name:
        raise ValueError("Simulation config must include 'preset_name' or 'preset'.")

    cfg = _load_config_preset(preset_name, base_dir=path.parent)

    scale_factor = float(data.get("scale_factor", 1.0))
    scaling_strategy = str(data.get("scaling_strategy", "preserve_in_degree"))
    cfg = scale_network_config(
        cfg,
        scale_factor=scale_factor,
        strategy=scaling_strategy,
    )

    if "dt_ms" in data:
        cfg.simulation.dt_ms = float(data["dt_ms"])
    if "duration_ms" in data:
        cfg.simulation.duration_ms = float(data["duration_ms"])
    if "seed" in data:
        cfg.simulation.seed = int(data["seed"])

    net = Network.from_config(cfg)

    sim_kwargs = {
        "integrator": str(data.get("integrator", "rk4")),
        "rhythm": data.get("rhythm", None),
        "full_result": bool(data.get("full_result", True)),
    }

    return net, sim_kwargs


def _load_config_preset(preset_name: str, *, base_dir: Path) -> NetworkConfig:
    """Load a preset named in a simulation config."""
    candidate = Path(preset_name).expanduser()

    if candidate.is_absolute() and candidate.exists():
        return load_preset(candidate)

    relative_candidate = (base_dir / candidate).resolve()
    if relative_candidate.exists():
        return load_preset(relative_candidate)

    return load_named_preset(preset_name, unlisted_preset=True)
