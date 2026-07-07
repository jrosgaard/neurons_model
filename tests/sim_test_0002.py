"""
src/neurons_model/tests/sim_test_0002.py
Test the simulation pipeline with various configurations and perturbations.
This test runs multiple simulations with different parameters.
"""


from __future__ import annotations


from pathlib import Path

import numpy as np


from neurons_model.functionals import functional_J
from neurons_model.loader import load_preset
from neurons_model.readouts.sim_metrics import compute_metrics
from neurons_model.simulation.config import PerturbationConfig
from neurons_model.simulation.network import Network
from neurons_model.simulation.scaling import scale_network_config
from neurons_model.simulation.simulation import run_simulation


PRESET_PATH = (Path(__file__).resolve().parents[1]
               / "src" / "neurons_model"
               / "presets"
               / "healthy_quiet.yaml")


def build_case_config(*, dt_ms: float, duration_ms: float,
                      amplitude_delta: float = 0.0, scale_factor: float = 0.2,
                      strategy: str = "preserve_in_degree",
                      perturbations: list[PerturbationConfig] | None = None,
                      ):
       
       cfg = load_preset(PRESET_PATH)
       cfg = scale_network_config(cfg, scale_factor=scale_factor, strategy=strategy)
       
       cfg.simulation.dt_ms = dt_ms
       cfg.simulation.duration_ms = duration_ms
       cfg.external_input.amplitude += amplitude_delta
       cfg.perturbations = perturbations or []
       
       return cfg


def run_case(
    name: str,
    *,
    integrator: str,
    rhythm: str,
    dt_ms: float,
    duration_ms: float,
    amplitude_delta: float = 0.0,
    scale_factor: float = 0.2,
    strategy: str = "preserve_in_degree",
    full_result: bool = False,
    perturbations: list[PerturbationConfig] | None = None,
) -> dict[str, float | int | str]:
    
    cfg = build_case_config(
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        amplitude_delta=amplitude_delta,
        scale_factor=scale_factor,
        strategy=strategy,
        perturbations=perturbations,
    )

    n_steps = int(round(cfg.simulation.duration_ms / cfg.simulation.dt_ms))
    assert n_steps < 500, f"{name}: expected fewer than 500 timesteps, got {n_steps}"

    net = Network.from_config(cfg)
    result = run_simulation(
        net,
        integrator=integrator,
        rhythm=rhythm,
        full_result=full_result,
    )

    metrics = compute_metrics(result, save=False)
    functional_val = functional_J(metrics, weight_matrix=net.weights)

    assert result.t_ms.shape == (n_steps,), f"{name}: unexpected time vector shape"
    assert result.v_trace.shape == (net.v_mv.shape[0], n_steps), (
        f"{name}: unexpected voltage trace shape"
    )

    assert result.field_proxy.shape == (n_steps,), (
        f"{name}: unexpected field proxy shape"
    )

    assert np.all(np.isfinite(result.v_trace)), f"{name}: non-finite voltages"
    assert np.all(np.isfinite(result.field_proxy)), f"{name}: non-finite field proxy"
    assert metrics["v_finite"], f"{name}: metrics flagged non-finite voltages"
    assert metrics["field_finite"], f"{name}: metrics flagged non-finite field proxy"
    assert np.isfinite(functional_val), f"{name}: non-finite functional_J score"

    total_spikes = int(sum(arr.sum() for arr in result.spike_counts.values()))

    if full_result:
        assert result.I_syn_trace is not None, f"{name}: missing I_syn_trace"
        assert result.I_ext_trace is not None, f"{name}: missing I_ext_trace"
        assert result.w_trace is not None, f"{name}: missing w_trace"


    return {
        "name": name,
        "integrator": integrator,
        "rhythm": rhythm,
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
    cases = [
        {
            "name": "baseline_small_rk4",
            "integrator": "rk4",
            "rhythm": "beta",
            "dt_ms": 0.10,
            "duration_ms": 20.0,
            "amplitude_delta": 0.0,
            "scale_factor": 0.20,
            "full_result": True,
        },
        {
            "name": "gamma_stronger_drive",
            "integrator": "exp_euler",
            "rhythm": "gamma",
            "dt_ms": 0.10,
            "duration_ms": 20.0,
            "amplitude_delta": 0.25,
            "scale_factor": 0.20,
            "full_result": True,
        },
        {
            "name": "finer_dt_scaled_drive",
            "integrator": "rk4",
            "rhythm": "alpha",
            "dt_ms": 0.05,
            "duration_ms": 20.0,
            "amplitude_delta": -0.10,
            "scale_factor": 0.15,
            "strategy": "preserve_total_drive",
            "full_result": True,
        },
        {
            "name": "perturbation_smoke",
            "integrator": "rk4",
            "rhythm": "theta",
            "dt_ms": 0.10,
            "duration_ms": 15.0,
            "scale_factor": 0.20,
            "full_result": True,
            "perturbations": [
                PerturbationConfig(
                    kind="excitability_shift",
                    target="excitatory",
                    value=1.5,
                ),
                PerturbationConfig(
                    kind="synaptic_gain",
                    target="excitatory->inhibitory_fast",
                    value=0.9,
                ),
                PerturbationConfig(
                    kind="input_drive",
                    target="excitatory",
                    value=0.1,
                ),
            ],
        },
    ]

    summaries = [run_case(**case) for case in cases]

    print("\nSimulation sweep summary")
    for summary in summaries:
        print(
            f"{summary['name']}: "
            f"integrator = {summary['integrator']}, "
            f"rhythm = {summary['rhythm']}, "
            f"steps = {summary['n_steps']}, "
            f"neurons = {summary['n_neurons']}, "
            f"spikes = {summary['total_spikes']}, "
            f"field_max = {summary['field_proxy_max']:.4f}, "
            f"\nfunctional_val = {summary['functional_val']:.4f}")


if __name__ == "__main__":
    main()
