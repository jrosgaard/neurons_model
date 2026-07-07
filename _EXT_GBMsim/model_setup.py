"""
_EXT_GBMsim.model_setup.py
Configuration and execution helpers for GBMsim tumor simulations.
"""

from __future__ import annotations

import csv
from numbers import Real
from pathlib import Path
import numpy as np

from .integrator import KPIntegrator
from .tumor_model import nondim

# Default initial conditions for dimensional variables
_default_E0 = 10
_default_T0 = 5
_default_IL0 = 10


def _validate_optional_int(name: str, value: int | None, *, minimum: int = 1) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or None")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_optional_real(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number or None")


def run_tumor_simulation(*,num_steps: int | None = None, total_time: float | None = None,
                         c: float | None = None, 
                         s_1: float | None = None, s_2: float | None = None,
                         output_path: str | Path | None = None,
                         ) -> dict[str, np.ndarray | float]:
    """
    Run the Kirschner-Panetta tumor simulation and return the full time series.
    """
    dimensional_params, model_run_params = config_model(num_steps = num_steps, total_time = total_time,
                                                        c = c, s_1 = s_1, s_2 = s_2,)

    (c_dim, p_1_dim, g_1_dim, mu_2_dim, g_2_dim, b_dim, r_2_dim, alpha_dim,
     mu_3_dim, p_2_dim, g_3_dim, s_1_dim, s_2_dim,) = dimensional_params
    
    num_steps, total_time, t_s, t, tau = model_run_params

    x = np.zeros(num_steps, dtype=float)
    y = np.zeros(num_steps, dtype=float)
    z = np.zeros(num_steps, dtype=float)
    s_1_array = np.full(num_steps, s_1_dim, dtype=float)
    s_2_array = np.full(num_steps, s_2_dim, dtype=float)

    E = np.zeros(num_steps, dtype=float)
    T = np.zeros(num_steps, dtype=float)
    IL = np.zeros(num_steps, dtype=float)
    IL_input = np.zeros(num_steps, dtype=float)

    E[0], T[0], IL[0] = _default_E0, _default_T0, _default_IL0

    (c_nd, p_1_nd, g_1_nd, mu_2_nd, g_2_nd, b_nd, r_2_nd, 
     alpha_nd, mu_3_nd, p_2_nd, g_3_nd, s_1_nd, s_2_nd,) = nondim(
         E[0], T[0], IL[0], t_s, c_dim, p_1_dim, g_1_dim, mu_2_dim, g_2_dim, b_dim, r_2_dim, alpha_dim,
         mu_3_dim, p_2_dim, g_3_dim, s_1_dim, s_2_dim,)

    x[0] = 1.0
    y[0] = 1.0
    z[0] = 1.0

    # Initialize the integrator
    integrator = KPIntegrator()

    for step in range(1, num_steps):
        params = [c_nd, mu_2_nd, p_1_nd, g_1_nd, s_1_nd, r_2_nd, b_nd, alpha_nd,
                  g_2_nd, p_2_nd, g_3_nd, mu_3_nd, s_2_nd,
                  ]

        x[step], y[step], z[step] = integrator.integrate(state=[x[step - 1], y[step - 1], z[step - 1]],
                                                         params=params,t_span=(float(tau[step - 1]), float(tau[step])),
                                                         )

        E[step] = x[step] * E[0]
        T[step] = y[step] * T[0]
        IL[step] = z[step] * IL[0]
        IL_input[step] = s_2_array[step]

    results: dict[str, np.ndarray | float] = {"t": t, "tau": tau,
                                              "x": x, "y": y, "z": z,
                                              "E": E, "T": T, "IL": IL,
                                              "s_1": s_1_array, "s_2": s_2_array,
                                              "IL_input": IL_input,
                                              "t_s": float(t_s),
                                              "total_time": float(total_time),
                                              }

    if output_path is not None:
        save_results_to_csv(output_path, results)
    return results


def config_model(num_steps: int | None = None, total_time: float | None = None,
                 c: float | None = None, s_1: float | None = None, s_2: float | None = None,
                 ) -> list[list]:
    """
    Build the dimensional parameter set and time grids for a GBMsim run.
    """
    _validate_optional_int("num_steps", num_steps, minimum=2)
    _validate_optional_real("total_time", total_time)
    _validate_optional_real("c", c)
    _validate_optional_real("s_1", s_1)
    _validate_optional_real("s_2", s_2)

    model_dim_params = _configure_dimensional_params(c=c, s_1=s_1, s_2=s_2)
    num_steps_out, total_time_out, t_s, t, tau = _configure_model_run(
        num_steps=num_steps,
        total_time=total_time,
        time_scale=model_dim_params[6],
    )

    model_run_params = [num_steps_out, total_time_out, t_s, t, tau]
    return [model_dim_params, model_run_params]


def save_results_to_csv(filename: str | Path, results: dict[str, np.ndarray | float]) -> None:
    """
    Save simulation results to CSV.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "tau", "x", "y", "z", "E", "T", "IL", "s_1", "s_2", "IL_input", "t_s"])

        t = results["t"]
        tau = results["tau"]
        x = results["x"]
        y = results["y"]
        z = results["z"]
        E = results["E"]
        T = results["T"]
        IL = results["IL"]
        s_1_array = results["s_1"]
        s_2_array = results["s_2"]
        IL_input = results["IL_input"]
        t_s = float(results["t_s"])

        for step in range(len(t)):
            writer.writerow(
                [t[step], tau[step], x[step], y[step], z[step],
                 E[step], T[step], IL[step],
                 s_1_array[step], s_2_array[step], IL_input[step], t_s,])


def _configure_dimensional_params(c: float | None = None,
                                  s_1: float | None = None,
                                  s_2: float | None = None,
                                  ) -> list[float]:
    """
    Dimensional parameters for the Kirschner-Panetta model.
    """
    c_value = 0.0297 if c is None else float(c)
    s_1_value = 0.0 if s_1 is None else float(s_1)
    s_2_value = 0.0 if s_2 is None else float(s_2)

    p_1 = 0.1245
    g_1 = 2.0e7
    mu_2 = 0.03
    g_2 = 1.0e5
    r_2 = 0.18
    b = 1.0e-9
    alpha = 1.0
    mu_3 = 10.0
    p_2 = 5.0
    g_3 = 1.0e3

    return [c_value, p_1, g_1, mu_2, g_2, b, r_2, alpha, mu_3, p_2, g_3, s_1_value, s_2_value]


def _configure_model_run(num_steps: int | None = None, total_time: float | None = None, 
                         *, time_scale: float = 0.18,
                         ) -> list[int | float | np.ndarray]:
    """
    Time grids and run parameters for a GBMsim simulation.
    """
    steps = 2000 if num_steps is None else num_steps
    duration = 4000.0 if total_time is None else float(total_time)

    t = np.linspace(0.0, duration, steps)
    tau = t * time_scale

    return [steps, duration, time_scale, t, tau]


_SAVE_RESULTS_TO_CSV = save_results_to_csv
_CONFIG_MODEL__DIMENSIONAL_PARAMS = _configure_dimensional_params
_CONFIG_MODEL_RUN = _configure_model_run

__all__ = [
    "config_model",
    "run_tumor_simulation",
    "save_results_to_csv",
]
