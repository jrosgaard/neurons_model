"""
src/neurons_model/simulation/ions.py
Ion channel dynamics and related functions.

TODO: This is experimental scaffolding and is not connected to the current simulation path. 
Revisit if detailed conductance/channel modeling returns.
"""

from __future__ import annotations

import numpy as np


# Define ion channel properties and dynamics here for use in the simulation.
# These values have NOT been biologically validated yet.
ionChannels = {
    "Na": {"conductance": 120.0, "reversal_potential": 50.0, "extracellular_concentration": 145.0, "intracellular_concentration": 15.0},
    "K": {"conductance": 36.0, "reversal_potential": -77.0, "extracellular_concentration": 5.0, "intracellular_concentration": 150.0},
    "Ca": {"conductance": 0.1, "reversal_potential": 120.0, "extracellular_concentration": 1.8, "intracellular_concentration": 0.0001},
    "Cl": {"conductance": 0.1, "reversal_potential": -65.0, "extracellular_concentration": 110.0, "intracellular_concentration": 10.0},
    "Leak": {"conductance": 0.3, "reversal_potential": -54.387},
}


class IonChannel:
    """Represents an ion channel with specific properties."""
    
    def __init__(self, name: str, conductance: float, reversal_potential: float):
        self.name = name
        self.conductance = conductance
        self.reversal_potential = reversal_potential
    
    def compute_current(self, voltage: float) -> float:
        """Compute the current through the channel based on the voltage."""
        return self.conductance * (voltage - self.reversal_potential)
    
    def __repr__(self) -> str:
        return f"IonChannel(name={self.name}, conductance={self.conductance}, reversal_potential={self.reversal_potential})"


