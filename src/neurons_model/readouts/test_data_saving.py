"""
src/neurons_model/readouts/test_data_saving.py
Helpers for saving sweep test data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import json
import numpy as np


def save_test_data(
    *,
    summaries: list[dict[str, Any]],
    config: Any,
    save_dir: Path,
    preset_path: Path | str,
    sweep_weights: dict[str, tuple[float, ...]] | tuple[float, ...] | None = None,
    integrator: str = "rk4",
    rhythm: str = "beta",
    scale_factor: float = 0.20,
    scaling_strategy: str = "preserve_in_degree",
    timestamp: str | None = None,
) -> dict[str, Path]:
    """
    Save sweep config, metadata, and summary arrays using the same schema as sweep_test_0002.py.
    """

    if not summaries:
        raise ValueError("summaries must contain at least one entry.")

    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    config_json = config.model_dump(mode="json") if hasattr(config, "model_dump") else config

    config_path = save_dir / f"sweep_config_{timestamp}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)
    print(f"Saved config to {config_path}")

    metadata = {
        "preset_path": str(preset_path),
        "save_timestamp": timestamp,
        "n_cases": len(summaries),
        "sweep_weights": ({pathway: list(weights) for pathway, weights in sweep_weights.items()}
                          if isinstance(sweep_weights, dict)
                          else list(sweep_weights) if sweep_weights is not None else None),
        "integrator": integrator,
        "rhythm": rhythm,
        "dt_ms": config.simulation.dt_ms,
        "duration_ms": config.simulation.duration_ms,
        "scale_factor": scale_factor,
        "scaling_strategy": scaling_strategy,
    }

    metadata_path = save_dir / f"sweep_metadata_{timestamp}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    required_output_keys: list[tuple[str, type]] = [
        ("name", str),
        ("pathway", str),
        ("base_weight", float),
        ("swept_weight", float),
        ("n_steps", int),
        ("n_neurons", int),
        ("total_spikes", int),
        ("field_proxy_max", float),
        ("functional_val", float),
        ("mean_membrane_potential_mv", float),
        ("min_membrane_potential_mv", float),
        ("max_membrane_potential_mv", float),
        ("Ex_mean_firing_rate_hz", float),
        ("In_fast_mean_firing_rate_hz", float),
        ("In_adapt_mean_firing_rate_hz", float),
        ("v_finite", bool),
        ("field_finite", bool),
    ]

    optional_output_keys: list[tuple[str, type]] = [
        ("multiplier", float),
        ]

    missing_keys = [key for key, _ in required_output_keys if key not in summaries[0]]
    if missing_keys:
        raise KeyError(f"summaries are missing required keys: {missing_keys}")

    output_keys = required_output_keys + [
        (key, dtype) for key, dtype in optional_output_keys if key in summaries[0]
    ]

    output_data = {
        key: np.array([summary[key] for summary in summaries], dtype=dtype)
        for key, dtype in output_keys
    }

    # Save output data as .npz file
    output_path = save_dir / f"sweep_output_data_{timestamp}.npz"
    np.savez(output_path, **output_data)
    print(f"Saved output data to {output_path}")

    return {"config": config_path,
            "metadata": metadata_path, 
            "output": output_path,
    }
