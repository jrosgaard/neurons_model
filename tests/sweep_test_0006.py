"""
src/neurons_model/tests/sweep_test_0006.py

This is a sweep.
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
from neurons_model.readouts.test_data_saving import save_test_data
from neurons_model.simulation.network import Network
from neurons_model.simulation.scaling import scale_network_config
from neurons_model.simulation.simulation import run_simulation

# -----------------------------------------------------------------------------
test_number = "0006"

SAVE_DIR = Path(__file__).resolve().parent / "sweep_results"
SAVE_DIR.mkdir(exist_ok=True)
SAVE = True


PRESET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "neurons_model"
    / "presets"
    / "healthy_active_1.yaml"
)

SWEEP_PATHWAYS = ("Ex->In_fast",
                  "Ex->In_adapt", 
                  )

# Ex->Ex weight, default is 0.014796
Ex_to_Ex_weights = (0.014796,)


# Sweep Ex->In weights, this includes both Ex->In_fast and Ex->In_adapt
# Default for both is 0.008
Ex_to_In_weights = np.linspace(0.000, 0.020, num=50)



SWEEP_WEIGHTS = {
    "Ex->Ex": Ex_to_Ex_weights,
    "Ex->In_fast": Ex_to_In_weights,
    "Ex->In_adapt": Ex_to_In_weights,}

# -----------------------------------------------------------------------------




def pathway_label(pathway) -> str:
    return f"{pathway.source}->{pathway.target}"


def pathway_index_from_label(cfg, pathway_name: str) -> int:
    for index, pathway in enumerate(cfg.pathways):
        if pathway_label(pathway) == pathway_name:
            return index
    raise ValueError(f"Pathway {pathway_name!r} not found in config.")


def build_base_config(
    *,
    dt_ms: float = 0.10,
    duration_ms: float = 500.0,
    scale_factor: float = 0.20,
    strategy: str = "preserve_in_degree",
):
    cfg = load_preset(PRESET_PATH)
    cfg = scale_network_config(cfg, scale_factor=scale_factor, strategy=strategy)
    cfg.simulation.dt_ms = dt_ms
    cfg.simulation.duration_ms = duration_ms
    return cfg


def build_sweep_config(pathway_name: str, swept_weight: float):
    cfg = build_base_config()
    pathway_index = pathway_index_from_label(cfg, pathway_name)
    base_weight = cfg.pathways[pathway_index].weight
    cfg.pathways[pathway_index].weight = swept_weight
    return cfg, pathway_index, base_weight


def run_case(pathway_name: str, swept_weight: float) -> dict[str, float | int | str]:
    cfg, pathway_index, base_weight = build_sweep_config(pathway_name, swept_weight)
    pathway = cfg.pathways[pathway_index]
    case_name = f"{pathway_label(pathway)}_w{swept_weight:.3f}"

    n_steps = int(round(cfg.simulation.duration_ms / cfg.simulation.dt_ms))
    assert n_steps < 10000, f"{case_name}: expected fewer than 10000 timesteps, got {n_steps}"

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
        "base_weight": float(base_weight),
        "swept_weight": float(swept_weight),
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

    for pathway_name in SWEEP_PATHWAYS:
        pathway_index = pathway_index_from_label(base_cfg, pathway_name)
        baseline_pathway = deepcopy(base_cfg.pathways[pathway_index])
        print(
            f"\nSweeping {pathway_label(baseline_pathway)} "
            f"around baseline weight {baseline_pathway.weight:.6f}"
        )
        for swept_weight in SWEEP_WEIGHTS[pathway_name]:
            summary = run_case(pathway_name, swept_weight)
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

        save_directory = SAVE_DIR / f"sweep_test_{test_number}"
        save_directory.mkdir(exist_ok=True)

        save_test_data(
            summaries=summaries,
            config=base_cfg,
            save_dir=save_directory,
            preset_path=PRESET_PATH,
            sweep_weights=SWEEP_WEIGHTS,
            integrator="rk4",
            rhythm="beta",
            scale_factor=0.20,
            scaling_strategy="preserve_in_degree",
            timestamp=timestamp,
        )


if __name__ == "__main__":
    main()
