"""
src/neurons_model/readouts/plot_metrics.py
Defines functions to plot summary metrics from simulation results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurons_model.simulation.simulation import SimulationResult


def plot_spike_raster(sim_result: SimulationResult, num_neurons: int = 5) -> None:
    """
    Plot a spike raster from a SimulationResult object.
    This function will plot the membrane potential, spike counts, synaptic current, and external current for a subset of neurons.
    Inputs:
    - sim_result: SimulationResult object containing the simulation data.
    - num_neurons: Number of neurons to plot from each population (default: 5).

    Raises:
    - ValueError: If the SimulationResult object does not contain the necessary data for plotting.

    Outputs:
    - A figure with 4 subplots showing 
        membrane potential, 
        spike counts, 
        synaptic current, and 
        external current for the specified neurons.
    """
    if sim_result.spike_counts is None:
        raise ValueError("SimulationResult must contain spike_counts to plot raster.")
    
    if sim_result.t_ms is None:
        raise ValueError("SimulationResult must contain t_ms to plot raster.")
    
    if sim_result.v_trace is None:
        raise ValueError("SimulationResult must contain v_trace to plot raster.")
    
    if sim_result.I_syn_trace is None:
        raise ValueError("SimulationResult must contain I_syn_trace to plot raster.")
    
    if sim_result.I_ext_trace is None:
        raise ValueError("SimulationResult must contain I_ext_trace to plot raster.")
    

    plt.figure(figsize=(14, 16))

    plt.subplot(4, 1, 1)
    plt.title("Membrane Potential Trace")
    for pop_name, spike_arr in sim_result.spike_counts.items():
        for i in range(num_neurons):
            plt.plot(sim_result.t_ms, sim_result.v_trace[i, :], alpha=0.7, label=f"{pop_name} Neuron {i}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.title("Spike counts")
    for pop_name, spike_arr in sim_result.spike_counts.items():
        for i in range(num_neurons):
            plt.plot(sim_result.t_ms, spike_arr[i, :], label=f"{pop_name} Neuron {i}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Count")
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.title("Synaptic Current Trace")
    for pop_name, spike_arr in sim_result.spike_counts.items():
        for i in range(num_neurons):
            plt.plot(sim_result.t_ms, sim_result.I_syn_trace[i, :], alpha=0.7, label=f"{pop_name} Neuron {i}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Synaptic Current (pA)")
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.title("External Current Trace")
    for pop_name, spike_arr in sim_result.spike_counts.items():
        for i in range(num_neurons):
            plt.plot(sim_result.t_ms, sim_result.I_ext_trace[i, :], alpha=0.7, label=f"{pop_name} Neuron {i}")
    plt.xlabel("Time (ms)")
    plt.ylabel("External Current (pA)")
    plt.grid()

    plt.tight_layout()
    plt.show()