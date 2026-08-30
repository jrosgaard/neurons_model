"""
_EXT_GBMsim.__init__.py
GBMsim extension package for tumor growth simulations.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["config_model",
           "main",
           "run_tumor_simulation",
           "save_results_to_csv",
           ]


def __getattr__(name: str):
    if name == "main":
        from .main import main

        return main
    if name in {"config_model", "run_tumor_simulation", "save_results_to_csv"}:
        from .model_setup import config_model, run_tumor_simulation, save_results_to_csv

        exports = {
            "config_model": config_model,
            "run_tumor_simulation": run_tumor_simulation,
            "save_results_to_csv": save_results_to_csv,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
