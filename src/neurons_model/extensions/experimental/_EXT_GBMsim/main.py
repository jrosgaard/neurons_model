"""
_EXT_GBMsim.main.py
Entry points for running GBMsim simulations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_setup import run_tumor_simulation

def main(output_path: str | Path | None = None, **simulation_kwargs: Any) -> dict[str, Any]:
    """Run a GBM simulation and optionally persist the results to CSV."""
    return run_tumor_simulation(output_path=output_path, **simulation_kwargs)

__all__ = ["main"]
