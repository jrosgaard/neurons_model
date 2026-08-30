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

from .simulation.network import Network
from .simulation.scaling import scale_network_config

from .simulation.simulation import run_simulation, SimulationResult

from .readouts.sim_metrics import compute_metrics
from .functionals import functional_J
