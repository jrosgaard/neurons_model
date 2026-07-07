"""
neurons_model/__main__.py
CLI entry point: python -m neurons_model [sim_config_path]
"""

from __future__ import annotations

import argparse

from .loader import DEFAULT_SIM_CONFIG_PATH, load_sim_config
from .simulation.simulation import run_simulation


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m neurons_model",
        description="Run a neurons_model simulation from a JSON simulation config.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_SIM_CONFIG_PATH),
        help=f"Path to a simulation JSON config. Default: {DEFAULT_SIM_CONFIG_PATH}",
    )
    return parser


def main(argv: list[str] | None = None):
    """Run the configured simulation and return the SimulationResult."""
    args = _build_parser().parse_args(argv)
    net, sim_kwargs = load_sim_config(args.config)
    result = run_simulation(net, **sim_kwargs)


    print(f"Preset: {net.config.name}")
    print(f"Neurons: {net.v_mv.size}")
    print(f"dt_ms: {net.config.simulation.dt_ms}")
    print(f"duration_ms: {net.config.simulation.duration_ms}")
    print(f"Integrator: {sim_kwargs['integrator']}")
    print(f"Rhythm: {sim_kwargs['rhythm']}")

    return result


if __name__ == "__main__":
    main()
