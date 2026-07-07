"""
src/neurons_model/scaling.py
Helpers for scaling network configurations across model sizes.


Conceptual categories
---------------------


Intensive-like properties

    Quantities that usually remain unchanged when network size changes,
    because they are properties of individual neurons or synapses:
    - intrinsic membrane parameters (v_rest_mv, v_threshold_mv, v_reset_mv)
    - membrane and refractory time constants
    - adaptation parameters
    - synaptic kinetics (tau_rise_ms, tau_decay_ms)
    - reversal potentials
    - delays
    - relative population composition

    
Extensive-like properties

    Quantities that naturally grow with network size unless normalized:
    - population counts
    - total spike count
    - summed field / LFP-like proxy
    - total recurrent synaptic drive
    - absolute number of stimulated neurons

    
Strategy-dependent properties

    Quantities whose scaling depends on what one wants to preserve:
    - connection probabilities
    - synaptic weights
    - external input target_fraction

"""


from __future__ import annotations


from copy import deepcopy
import numpy as np


from .config import NetworkConfig


def scale_network_config(cfg: NetworkConfig, scale_factor: float, 
                         strategy: str = "preserve_in_degree",) -> NetworkConfig:
    """
    Scale a validated NetworkConfig according to a chosen invariance strategy.

    This function preserves intensive neuron/synapse parameters
    (intrinsic voltages, time constants, kinetics, delays) while scaling
    network-size-dependent properties according to the selected strategy.

    Strategy-dependent parameters such as connection probability,
    synaptic weight, and stimulus coverage may be adjusted differently
    depending on what is being preserved.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")

    new_cfg = deepcopy(cfg)

    # --------------------------------------------------------------------------------------------
    # Scale population sizes
    for pop in new_cfg.populations:
        pop.count_n = max(1, int(round(pop.count_n * scale_factor)))

    # --------------------------------------------------------------------------------------------
    # Scale pathway parameters
    if strategy == "preserve_in_degree":
        for pathway in new_cfg.pathways:
            pathway.probability = max(
                0.0,
                min(1.0, pathway.probability / scale_factor),
            )

    elif strategy == "preserve_total_drive":
        for pathway in new_cfg.pathways:
            pathway.weight = pathway.weight / scale_factor

    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}. "
                         f"Choose 'preserve_in_degree' or 'preserve_total_drive'.")


    return new_cfg