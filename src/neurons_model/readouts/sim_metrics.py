"""
src/neurons_model/sim_metrics.py
Defines functions to compute summary metrics from simulation results.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
import json

from ..simulation.simulation import SimulationResult


def compute_metrics(sim_result: SimulationResult, save: bool = False) -> dict:
    """
    Compute summary metrics from a SimulationResult object.

    Returns
    -------
    dict
        Dictionary of scalar metrics and per-population firing rates.
    """

    metrics: dict[str, float | bool] = {}

    # --------------------------------------------------------------------------------------------
    # Basic sanity checks
    metrics["v_finite"] = bool(np.all(np.isfinite(sim_result.v_trace)))
    metrics["field_finite"] = bool(np.all(np.isfinite(sim_result.field_proxy)))

    # --------------------------------------------------------------------------------------------
    # Time duration
    if sim_result.t_ms.size == 0:
        raise ValueError("sim_result.t_ms is empty.")

    total_time_sec = sim_result.t_ms[-1] / 1000.0
    if total_time_sec <= 0:
        raise ValueError("Simulation duration must be positive.")

    # --------------------------------------------------------------------------------------------
    # Voltage summary
    if sim_result.v_trace.size > 0:
        metrics["mean_membrane_potential_mv"] = float(np.mean(sim_result.v_trace))
        metrics["min_membrane_potential_mv"] = float(np.min(sim_result.v_trace))
        metrics["max_membrane_potential_mv"] = float(np.max(sim_result.v_trace))
    else:
        metrics["mean_membrane_potential_mv"] = np.nan
        metrics["min_membrane_potential_mv"] = np.nan
        metrics["max_membrane_potential_mv"] = np.nan

    # --------------------------------------------------------------------------------------------
    # Field proxy summary
    if sim_result.field_proxy.size > 0:
        metrics["field_proxy_mean"] = float(np.mean(sim_result.field_proxy))
        metrics["field_proxy_std"] = float(np.std(sim_result.field_proxy))
        metrics["field_proxy_max"] = float(np.max(sim_result.field_proxy))
    else:
        metrics["field_proxy_mean"] = np.nan
        metrics["field_proxy_std"] = np.nan
        metrics["field_proxy_max"] = np.nan

    # --------------------------------------------------------------------------------------------
    # Spike metrics
    total_spikes_all = 0

    for pop_name, spike_arr in sim_result.spike_counts.items():
        # spike_arr expected shape: (n_neurons_in_population, n_steps)
        if spike_arr.ndim != 2:
            raise ValueError(
                f"spike_counts['{pop_name}'] must be 2D, got shape {spike_arr.shape}"
            )

        n_neurons = spike_arr.shape[0]
        total_spikes = int(np.sum(spike_arr))
        total_spikes_all += total_spikes

        mean_firing_rate_hz = total_spikes / (n_neurons * total_time_sec)
        metrics[f"{pop_name}_spike_count"] = total_spikes
        metrics[f"{pop_name}_mean_firing_rate_hz"] = float(mean_firing_rate_hz)

    # --------------------------------------------------------------------------------------------
        # Simple population synchrony proxy:
        # average pairwise correlation across spike trains
        if n_neurons > 1:
            spike_trains = spike_arr.astype(float)

            # Remove zero-variance neurons to avoid NaNs in corrcoef
            stds = np.std(spike_trains, axis=1)
            valid = stds > 0

            if np.sum(valid) > 1:
                corr = np.corrcoef(spike_trains[valid])
                upper = np.triu_indices_from(corr, k=1)
                metrics[f"{pop_name}_synchrony_index"] = float(np.nanmean(corr[upper]))
            else:
                metrics[f"{pop_name}_synchrony_index"] = np.nan
        else:
            metrics[f"{pop_name}_synchrony_index"] = np.nan

    # --------------------------------------------------------------------------------------------
    # Population count
    metrics["total_spike_count"] = total_spikes_all
    metrics["n_populations"] = len(sim_result.spike_counts)

    # --------------------------------------------------------------------------------------------
    # Population count entropy
    pop_count_ts = np.sum(
        np.vstack(list(sim_result.spike_counts.values())),
        axis=0)

    values, counts = np.unique(pop_count_ts, return_counts=True)
    probs = counts / counts.sum()

    metrics["population_count_entropy_bits"] = float(stats.entropy(probs, base=2))

# --------------------------------------------------------------------------------------------
    # Population shannon entropy
    all_spikes_flat = np.concatenate(
        [spike_arr.flatten() for spike_arr in sim_result.spike_counts.values()])
    
    if all_spikes_flat.size > 0:
        p_spike = np.mean(all_spikes_flat)
        probs = np.array([1.0 - p_spike, p_spike])

        # avoid log(0) issues
        probs = probs[probs > 0]
        metrics["overall_shannon_entropy"] = float(stats.entropy(probs, base=2))

    else:
        metrics["overall_shannon_entropy"] = np.nan

    # --------------------------------------------------------------------------------------------
    # Save as json if requested
    if save:
        with open("sim_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


def firing_rate_hz(spike_count: int, duration_ms: float) -> float:
    """
    Compute firing rate in Hz given spike count and duration in ms.
    """
    if duration_ms <= 0:
        return 0.0
    return 1000.0 * spike_count / duration_ms


def spike_count(spike_times: list[float]) -> int:
    """
    Compute spike count given a list of spike times.
    """
    return len(spike_times)

