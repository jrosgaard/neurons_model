"""
src/neurons_model/__init__.py
neurons_model core package.
"""

__version__ = "0.1.0"

__all__ = [
    # Config primitives
    "PopulationKind",
    "PerturbationKind",
    "TaskKind",
    "ReadoutKind",
    "PopulationConfig",
    "PathwayConfig",
    "SimulationConfig",
    "InputConfig",
    "TaskConfig",
    "PerturbationConfig",
    "ReadoutConfig",
    "NetworkConfig",
    # Cell types
    "CellTypeSpec",
    "load_cell_type_specs",
    # Network
    "Network",
    # Simulation
    "run_simulation",
    "SimulationResult",
    # Registry
    "register",
    "get",
    "list_registered",
    # Loader
    "load_preset",
]

from .simulation.config import (
    PopulationKind,
    PerturbationKind,
    TaskKind,
    ReadoutKind,
    PopulationConfig,
    PathwayConfig,
    SimulationConfig,
    InputConfig,
    TaskConfig,
    PerturbationConfig,
    ReadoutConfig,
    NetworkConfig,
)

from .cell_types import CellTypeSpec, load_cell_type_specs
from .loader import load_preset

from .registry import register, get, list_registered
from .presets.preset_manager import select_preset, list_presets, load_named_preset, resolve_preset_path

from neurons_model.simulation.config import NetworkConfig
from .simulation.network import Network
from neurons_model.simulation.scaling import scale_network_config

from .simulation.simulation import run_simulation, SimulationResult

from neurons_model.readouts.sim_metrics import compute_metrics
from neurons_model.readouts.plot_metrics import plot_spike_raster
from neurons_model.readouts.pop_plot import plot_weight_heatmap, plot_connectivity_heatmap, plot_population_weight_summary
from neurons_model.readouts.sweep_plot import sweep_plot_general, sweep_plot_norm_summary
from neurons_model.readouts.calvo_analysis import calvo_pca, calvo_pca_plot, pca_analysis
from neurons_model.functionals import functional_J
