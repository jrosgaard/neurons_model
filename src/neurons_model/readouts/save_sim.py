"""
src/neurons_model/readouts/save_sim.py
Helpers for saving simulation configs and metadata.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import numpy as np


DEFAULT_SIM_CONFIG_PATH = Path(__file__).resolve().parents[3] / "examples" / "example_sim1.json"
DEFAULT_SAVE_SIM_PATH = Path(__file__).resolve().parents[1] / "output_folder"


def save_sim_result(sim_result: Any,
            *,
            config: Any,
            save_path: Path | None = None,
            other_save_path: Path | None = None,
            config_path: Path | None = None,
            integrator: str = "rk4",
            rhythm: str = "beta",
            scale_factor: float = 1.0,
            scaling_strategy: str = "preserve_in_degree",
            save_preset: bool = False,
            ) -> dict[str, Path]:
    """Save one simulation result, metadata, and optionally the preset config."""

    if sim_result is None:
        raise ValueError("sim_result must not be None.")

    data: dict[str, Any] = {}
    path = DEFAULT_SIM_CONFIG_PATH if config_path is None else Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Simulation config {path} must define a JSON object.")
        data = loaded

    save_dir = _resolve_save_path(
        save_path=save_path,
        other_save_path=other_save_path,
        config_data=data,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(data.get("save_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_name = str(data.get("save_name") or "simulation_result")

    saved_paths: dict[str, Path] = {}
    if save_preset:
        preset_json = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
        preset_name = str(data.get("preset_name") or getattr(config, "name", "preset"))
        preset_path = save_dir / f"{preset_name}_preset_{timestamp}.json"
        with preset_path.open("w", encoding="utf-8") as f:
            json.dump(preset_json, f, indent=2)
        saved_paths["preset"] = preset_path
        print(f"Saved preset to {preset_path}")

    metadata = {
        "preset_name": str(data.get("preset_name") or getattr(config, "name", "")),
        "save_timestamp": timestamp,
        "integrator": integrator,
        "rhythm": rhythm,
        "dt_ms": config.simulation.dt_ms,
        "duration_ms": config.simulation.duration_ms,
        "scale_factor": scale_factor,
        "scaling_strategy": scaling_strategy,
        "config_path": str(path) if path.exists() else None,
    }

    metadata_path = save_dir / f"{save_name}_metadata_{timestamp}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    saved_paths["metadata"] = metadata_path

    output_data = asdict(sim_result) if is_dataclass(sim_result) else sim_result
    if hasattr(output_data, "__dict__") and not isinstance(output_data, dict):
        output_data = vars(output_data)
    if not isinstance(output_data, dict):
        raise ValueError("sim_result must be a dataclass instance or a dictionary.")

    output_npz, output_json = _split_output_data(output_data)

    output_path = save_dir / f"{save_name}_output_{timestamp}.npz"
    np.savez_compressed(output_path, **output_npz)
    print(f"Saved numeric output data to {output_path}")
    saved_paths["output"] = output_path

    if output_json:
        json_path = save_dir / f"{save_name}_output_{timestamp}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2)
        print(f"Saved structured output data to {json_path}")
        saved_paths["output_json"] = json_path

    return saved_paths


def _resolve_save_path(
    *,
    save_path: Path | None,
    other_save_path: Path | None,
    config_data: dict[str, Any],
) -> Path:
    if other_save_path is not None:
        return Path(other_save_path)
    if save_path is not None:
        return Path(save_path)
    if config_data.get("save_path"):
        return Path(str(config_data["save_path"]))
    return DEFAULT_SAVE_SIM_PATH


def _split_output_data(output_data: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Split simulation outputs into NPZ-safe arrays and JSON-safe structures."""
    npz_data: dict[str, np.ndarray] = {}
    json_data: dict[str, Any] = {}

    for key, value in output_data.items():
        if value is None:
            json_data[key] = None
        elif isinstance(value, np.ndarray):
            npz_data[key] = value
        elif isinstance(value, dict):
            _collect_dict_output(key, value, npz_data, json_data)
        else:
            array = _try_numeric_array(value)
            if array is None:
                json_data[key] = _to_jsonable(value)
            else:
                npz_data[key] = array

    if not npz_data:
        raise ValueError("sim_result did not contain any numeric array data to save.")

    return npz_data, json_data


def _collect_dict_output(
    prefix: str,
    value: dict[Any, Any],
    npz_data: dict[str, np.ndarray],
    json_data: dict[str, Any],
) -> None:
    json_dict: dict[str, Any] = {}

    for subkey, subvalue in value.items():
        output_key = f"{prefix}__{subkey}"
        array = _try_numeric_array(subvalue)
        if array is None:
            json_dict[str(subkey)] = _to_jsonable(subvalue)
        else:
            npz_data[output_key] = array

    if json_dict:
        json_data[prefix] = json_dict


def _try_numeric_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except ValueError:
        return None

    if array.dtype.kind in {"b", "i", "u", "f", "c"}:
        return array
    return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
