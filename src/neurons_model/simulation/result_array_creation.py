"""
src/neurons_model/simulation/result_array_creation.py
Functions for creating result arrays for storing simulation outputs, e.g. voltages and spikes.

TODO: The simulator currently allocates these arrays inline. Keep this only if
we want to refactor allocation into a shared helper again.
"""

from __future__ import annotations

import numpy as np


def create_result_arrays(net, n_steps, dt, n_neurons, full_result: bool = False):
    
    # Recording arrays
    t_ms         = np.arange(n_steps) * dt
    v_trace      = np.zeros((n_neurons, n_steps))
    spike_rec    = np.zeros((n_neurons, n_steps), dtype=bool)
    field_proxy    = np.zeros(n_steps)

    I_syn_trace = None
    I_ext_trace = None
    I_adapt_current_trace = None
    w_trace = None

    # Optional traces for synaptic current, external current, adaptation current, and adaptation variable.
    if full_result:
        I_syn_trace  = np.zeros((n_neurons, n_steps))
        I_ext_trace  = np.zeros((n_neurons, n_steps))
        I_adapt_current_trace = np.zeros((n_neurons, n_steps))
        w_trace      = np.zeros((n_neurons, n_steps))

    return t_ms, v_trace, spike_rec, field_proxy, \
           I_syn_trace, I_ext_trace, I_adapt_current_trace, w_trace
