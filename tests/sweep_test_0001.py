"""
src/neurons_model/tests/sweep_test_0001.py

This is a smaller smoke sweep.
"""


from __future__ import annotations


from copy import deepcopy
from pathlib import Path

import numpy as np
import json
from datetime import datetime


from neurons_model.functionals import functional_J
from neurons_model.loader import load_preset
from neurons_model.readouts.sim_metrics import compute_metrics
from neurons_model.simulation.network import Network
from neurons_model.simulation.scaling import scale_network_config
from neurons_model.simulation.simulation import run_simulation

SAVE_DIR = Path(__file__).resolve().parent / "sweep_results"
SAVE_DIR.mkdir(exist_ok=True)
SAVE = True


PRESET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "neurons_model"
    / "presets"
    / "healthy_quiet.yaml"
)

SWEEP_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5)


def pathway_label(pathway) -> str:
    return f"{pathway.source}->{pathway.target}"


def build_base_config(
    *,
    dt_ms: float = 0.10,
    duration_ms: float = 20.0,
    scale_factor: float = 0.20,
    strategy: str = "preserve_in_degree",
):
    cfg = load_preset(PRESET_PATH)
    cfg = scale_network_config(cfg, scale_factor=scale_factor, strategy=strategy)
    cfg.simulation.dt_ms = dt_ms
    cfg.simulation.duration_ms = duration_ms
    return cfg


def build_sweep_config(pathway_index: int, multiplier: float):
    cfg = build_base_config()
    base_weight = cfg.pathways[pathway_index].weight
    cfg.pathways[pathway_index].weight = base_weight * multiplier
    return cfg, base_weight


def run_case(pathway_index: int, multiplier: float) -> dict[str, float | int | str]:
    cfg, base_weight = build_sweep_config(pathway_index, multiplier)
    pathway = cfg.pathways[pathway_index]
    case_name = f"{pathway_label(pathway)}_x{multiplier:.2f}"

    n_steps = int(round(cfg.simulation.duration_ms / cfg.simulation.dt_ms))
    assert n_steps < 500, f"{case_name}: expected fewer than 500 timesteps, got {n_steps}"

    net = Network.from_config(cfg)
    result = run_simulation(
        net,
        integrator="rk4",
        rhythm="beta",
        full_result=False,
    )

    metrics = compute_metrics(result, save=False)
    functional_val = functional_J(metrics, weight_matrix=net.weights)

    assert result.t_ms.shape == (n_steps,), f"{case_name}: unexpected time vector shape"
    assert result.v_trace.shape == (net.v_mv.shape[0], n_steps), (
        f"{case_name}: unexpected voltage trace shape"
    )
    assert result.field_proxy.shape == (n_steps,), (
        f"{case_name}: unexpected field proxy shape"
    )

    assert np.all(np.isfinite(result.v_trace)), f"{case_name}: non-finite voltages"
    assert np.all(np.isfinite(result.field_proxy)), f"{case_name}: non-finite field proxy"
    assert metrics["v_finite"], f"{case_name}: metrics flagged non-finite voltages"
    assert metrics["field_finite"], f"{case_name}: metrics flagged non-finite field proxy"
    assert np.isfinite(functional_val), f"{case_name}: non-finite functional_J score"

    total_spikes = int(sum(arr.sum() for arr in result.spike_counts.values()))

    return {
        "name": case_name,
        "pathway": pathway_label(pathway),
        "multiplier": float(multiplier),
        "base_weight": float(base_weight),
        "swept_weight": float(pathway.weight),
        "n_steps": n_steps,
        "n_neurons": int(net.v_mv.shape[0]),
        "total_spikes": total_spikes,
        "field_proxy_max": float(np.max(result.field_proxy)),
        "functional_val": float(functional_val),
        "mean_membrane_potential_mv": float(metrics["mean_membrane_potential_mv"]),
        "min_membrane_potential_mv": float(metrics["min_membrane_potential_mv"]),
        "max_membrane_potential_mv": float(metrics["max_membrane_potential_mv"]),
        "Ex_mean_firing_rate_hz": float(metrics["Ex_mean_firing_rate_hz"]),
        "In_fast_mean_firing_rate_hz": float(metrics["In_fast_mean_firing_rate_hz"]),
        "In_adapt_mean_firing_rate_hz": float(metrics["In_adapt_mean_firing_rate_hz"]),
        "v_finite": bool(metrics["v_finite"]),
        "field_finite": bool(metrics["field_finite"]),
    }


def main() -> None:
    base_cfg = build_base_config()
    summaries: list[dict[str, float | int | str]] = []

    for pathway_index, pathway in enumerate(base_cfg.pathways):
        baseline_pathway = deepcopy(pathway)
        print(
            f"\nSweeping {pathway_label(baseline_pathway)} "
            f"around baseline weight {baseline_pathway.weight:.6f}"
        )
        for multiplier in SWEEP_MULTIPLIERS:
            summary = run_case(pathway_index, multiplier)
            summaries.append(summary)
            print(
                f"  {summary['name']}: "
                f"weight = {summary['swept_weight']:.6f}, "
                f"spikes = {summary['total_spikes']}, "
                f"field_max = {summary['field_proxy_max']:.4f}, "
                f"functional_val = {summary['functional_val']:.4f}"
            )

    print(f"\nCompleted {len(summaries)} sweep cases.")

    if SAVE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        config = base_cfg.model_dump(mode="json")
        
        save_path_config = SAVE_DIR / f"sweep_config_{timestamp}.json"
        with open(save_path_config, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Saved config to {save_path_config}")


        metadata = {
            "preset_path": str(PRESET_PATH),
            "save_timestamp": timestamp,
            "n_cases": len(summaries),
            "sweep_multipliers": list(SWEEP_MULTIPLIERS),
            "integrator": "rk4",
            "rhythm": "beta",
            "dt_ms": base_cfg.simulation.dt_ms,
            "duration_ms": base_cfg.simulation.duration_ms,
            "scale_factor": 0.20,
            "scaling_strategy": "preserve_in_degree",
        }

        save_path_metadata = SAVE_DIR / f"sweep_metadata_{timestamp}.json"
        with open(save_path_metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {save_path_metadata}")


        output_data = {
            "name": np.array([summary["name"] for summary in summaries], dtype=str),
            "pathway": np.array([summary["pathway"] for summary in summaries], dtype=str),
            "multiplier": np.array([summary["multiplier"] for summary in summaries], dtype=float),
            "base_weight": np.array([summary["base_weight"] for summary in summaries], dtype=float),
            "swept_weight": np.array([summary["swept_weight"] for summary in summaries], dtype=float),
            "n_steps": np.array([summary["n_steps"] for summary in summaries], dtype=int),
            "n_neurons": np.array([summary["n_neurons"] for summary in summaries], dtype=int),
            "total_spikes": np.array([summary["total_spikes"] for summary in summaries], dtype=int),
            "field_proxy_max": np.array([summary["field_proxy_max"] for summary in summaries], dtype=float),
            "functional_val": np.array([summary["functional_val"] for summary in summaries], dtype=float),
            "mean_membrane_potential_mv": np.array(
                [summary["mean_membrane_potential_mv"] for summary in summaries], dtype=float
            ),
            "min_membrane_potential_mv": np.array(
                [summary["min_membrane_potential_mv"] for summary in summaries], dtype=float
            ),
            "max_membrane_potential_mv": np.array(
                [summary["max_membrane_potential_mv"] for summary in summaries], dtype=float
            ),
            "Ex_mean_firing_rate_hz": np.array(
                [summary["Ex_mean_firing_rate_hz"] for summary in summaries], dtype=float
            ),
            "In_fast_mean_firing_rate_hz": np.array(
                [summary["In_fast_mean_firing_rate_hz"] for summary in summaries], dtype=float
            ),
            "In_adapt_mean_firing_rate_hz": np.array(
                [summary["In_adapt_mean_firing_rate_hz"] for summary in summaries], dtype=float
            ),
            "v_finite": np.array([summary["v_finite"] for summary in summaries], dtype=bool),
            "field_finite": np.array([summary["field_finite"] for summary in summaries], dtype=bool),
        }

        save_path_output = SAVE_DIR / f"sweep_output_data_{timestamp}.npz"
        np.savez(save_path_output, **output_data)

        print(f"Saved output data to {save_path_output}")


if __name__ == "__main__":
    main()
