"""
src/neurons_model/simulation/sweep_function.py
Helpers for running synaptic weight sweeps and saving summary results.
"""


from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from neurons_model.functionals import functional_J
from neurons_model.loader import load_preset
from neurons_model.readouts.sim_metrics import compute_metrics
from neurons_model.readouts.test_data_saving import save_test_data
from neurons_model.simulation.network import Network
from neurons_model.simulation.scaling import scale_network_config
from neurons_model.simulation.simulation import run_simulation


# -----------------------------------------------------------------------------
# Default paths and parameters for sweeps. These can be overridden by individual sweep tests.

DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[3] / "tests" / "sweep_results"
DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PRESET_PATH = (
    Path(__file__).resolve().parents[1]
    / "presets"
    / "healthy_active_1.yaml"
)

# -----------------------------------------------------------------------------

def pathway_label(pathway) -> str:
    return f"{pathway.source}->{pathway.target}"


def pathway_index_from_label(cfg, pathway_name: str) -> int:
    for index, pathway in enumerate(cfg.pathways):
        if pathway_label(pathway) == pathway_name:
            return index
    raise ValueError(f"Pathway {pathway_name!r} not found in config.")


def build_base_config(
    preset_path: Path | str = DEFAULT_PRESET_PATH,
    *,
    dt_ms: float = 0.10,
    duration_ms: float = 500.0,
    scale_factor: float = 0.20,
    strategy: str = "preserve_in_degree",
):
    cfg = load_preset(preset_path)
    cfg = scale_network_config(cfg, scale_factor=scale_factor, strategy=strategy)
    cfg.simulation.dt_ms = dt_ms
    cfg.simulation.duration_ms = duration_ms
    return cfg


def build_sweep_config(
    pathway_name: str,
    swept_weight: float,
    *,
    preset_path: Path | str = DEFAULT_PRESET_PATH,
    dt_ms: float = 0.10,
    duration_ms: float = 500.0,
    scale_factor: float = 0.20,
    strategy: str = "preserve_in_degree",
):
    cfg = build_base_config(
        preset_path=preset_path,
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        scale_factor=scale_factor,
        strategy=strategy,
    )
    pathway_index = pathway_index_from_label(cfg, pathway_name)
    base_weight = cfg.pathways[pathway_index].weight
    cfg.pathways[pathway_index].weight = swept_weight
    return cfg, pathway_index, base_weight


def run_case(
    pathway_name: str,
    swept_weight: float,
    *,
    preset_path: Path | str = DEFAULT_PRESET_PATH,
    dt_ms: float = 0.10,
    duration_ms: float = 500.0,
    scale_factor: float = 0.20,
    scaling_strategy: str = "preserve_in_degree",
    integrator: str = "rk4",
    rhythm: str = "beta",
    full_result: bool = False,
) -> dict[str, float | int | str | bool]:
    cfg, pathway_index, base_weight = build_sweep_config(
        pathway_name,
        swept_weight,
        preset_path=preset_path,
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        scale_factor=scale_factor,
        strategy=scaling_strategy,
    )
    pathway = cfg.pathways[pathway_index]
    case_name = f"{pathway_label(pathway)}_w{swept_weight:.3f}"

    n_steps = int(round(cfg.simulation.duration_ms / cfg.simulation.dt_ms))
    assert n_steps < 10000, f"{case_name}: expected fewer than 10000 timesteps, got {n_steps}"

    net = Network.from_config(cfg)
    result = run_simulation(
        net,
        integrator=integrator,
        rhythm=rhythm,
        full_result=full_result,
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


def run_sweep(
    sweep_pathways: Iterable[str],
    sweep_weights: dict[str, Iterable[float]],
    *,
    preset_path: Path | str = DEFAULT_PRESET_PATH,
    save_dir: Path | str = DEFAULT_SAVE_DIR,
    test_number: str = "sweep",
    save: bool = True,
    dt_ms: float = 0.10,
    duration_ms: float = 500.0,
    scale_factor: float = 0.20,
    scaling_strategy: str = "preserve_in_degree",
    integrator: str = "rk4",
    rhythm: str = "beta",
    full_result: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a synaptic weight sweep and optionally save the results.

    Returns a dictionary containing the sweep summaries, the base config, and any
    saved artifact paths.
    """
    sweep_pathways = tuple(sweep_pathways)
    if not sweep_pathways:
        raise ValueError("sweep_pathways must contain at least one pathway.")

    missing_weights = [pathway for pathway in sweep_pathways if pathway not in sweep_weights]
    if missing_weights:
        raise KeyError(f"Missing sweep weights for pathways: {missing_weights}")

    base_cfg = build_base_config(
        preset_path=preset_path,
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        scale_factor=scale_factor,
        strategy=scaling_strategy,
    )
    summaries: list[dict[str, float | int | str | bool]] = []

    for pathway_name in sweep_pathways:
        pathway_index = pathway_index_from_label(base_cfg, pathway_name)
        baseline_pathway = deepcopy(base_cfg.pathways[pathway_index])
        weights = tuple(sweep_weights[pathway_name])
        if not weights:
            raise ValueError(f"sweep_weights[{pathway_name!r}] must contain at least one weight.")

        if verbose:
            print(
                f"\nSweeping {pathway_label(baseline_pathway)} "
                f"around baseline weight {baseline_pathway.weight:.6f}"
            )

        for swept_weight in weights:
            summary = run_case(
                pathway_name,
                float(swept_weight),
                preset_path=preset_path,
                dt_ms=dt_ms,
                duration_ms=duration_ms,
                scale_factor=scale_factor,
                scaling_strategy=scaling_strategy,
                integrator=integrator,
                rhythm=rhythm,
                full_result=full_result,
            )
            summaries.append(summary)

            if verbose:
                print(
                    f"  {summary['name']}: "
                    f"weight = {summary['swept_weight']:.6f}, "
                    f"spikes = {summary['total_spikes']}, "
                    f"field_max = {summary['field_proxy_max']:.4f}, "
                    f"functional_val = {summary['functional_val']:.4f}"
                )

    if verbose:
        print(f"\nCompleted {len(summaries)} sweep cases.")

    saved_paths: dict[str, Path] | None = None
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(save_dir) / f"sweep_test_{test_number}"
        target_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = save_test_data(
            summaries=summaries,
            config=base_cfg,
            save_dir=target_dir,
            preset_path=preset_path,
            sweep_weights={pathway: tuple(sweep_weights[pathway]) for pathway in sweep_pathways},
            integrator=integrator,
            rhythm=rhythm,
            scale_factor=scale_factor,
            scaling_strategy=scaling_strategy,
            timestamp=timestamp,
        )

    return {
        "summaries": summaries,
        "config": base_cfg,
        "saved_paths": saved_paths,
    }
