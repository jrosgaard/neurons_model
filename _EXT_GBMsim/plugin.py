"""
_EXT_GBMsim.plugin.py
Plugin registration for the GBMsim extension package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --- Extension metadata ---
_EXTENSION_NAME = "GBMsim"
_EXTENSION_VERSION = "0.1.0"
_EXTENSION_DESCRIPTION = "Simulation utilities for a Kirschner-Panetta-style glioblastoma growth model."
_EXTENSION_AUTHOR = "Johan Rosgaard"


def register(registry):
    """Register GBMsim with the neurons_model extension registry."""
    return registry.register_extension(
        package=__package__ or "_EXT_GBMsim",
        name=_EXTENSION_NAME,
        version=_EXTENSION_VERSION,
        description=_EXTENSION_DESCRIPTION,
        author=_EXTENSION_AUTHOR,
        functions=_EXTENSION_FUNCTIONS,
        variables=_EXTENSION_VARIABLES,
    )


def _main(*args: Any, **kwargs: Any):
    from .main import main

    return main(*args, **kwargs)


def _config_model(*args: Any, **kwargs: Any):
    from .model_setup import config_model

    return config_model(*args, **kwargs)


def _run_tumor_simulation(*args: Any, **kwargs: Any):
    from .model_setup import run_tumor_simulation

    return run_tumor_simulation(*args, **kwargs)


def _save_results_to_csv(*args: Any, **kwargs: Any):
    from .model_setup import save_results_to_csv

    return save_results_to_csv(*args, **kwargs)


def _load_glioma_cell_types(*args: Any, **kwargs: Any):
    from neurons_model.cell_types import load_cell_type_specs

    path = Path(__file__).resolve().with_name("glioma.yaml")
    return load_cell_type_specs(path, section="glioma_cells")


_EXTENSION_FUNCTIONS = {
    "main": _main,
    "config_model": _config_model,
    "run_tumor_simulation": _run_tumor_simulation,
    "save_results_to_csv": _save_results_to_csv,
    "load_glioma_cell_types": _load_glioma_cell_types,
}

_EXTENSION_VARIABLES = {
    "package": __package__ or "_EXT_GBMsim",
    "cell_type_sections": {"glioma": "glioma_cells"},
}
