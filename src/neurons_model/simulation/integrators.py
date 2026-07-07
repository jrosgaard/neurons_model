"""
src/neurons_model/integrators.py
Numerical integration methods for neuron voltage dynamics.
Each function returns the updated state array.
Choose:
Euler,
exponential euler,
runge-kutta 4,
runge-kutta 45 (SciPy's adaptive RK45),
leapfrog 
"""


from __future__ import annotations

import scipy.integrate
import math
import numpy as np


def euler(v, v_rest, tau_m, I_total_fn, dt, t=0.0, **kwargs) -> np.ndarray:
    """
    Forward Euler integrator.
    Simple and fast, but can be unstable for stiff equations or large dt.
    Not recommended for this network, but included for comparison.
    """

    I_total = I_total_fn(t, v)

    dv = (v_rest - v) / tau_m + I_total

    return v + dv * dt


def exp_euler(v, v_rest, tau_m, I_total, dt, **kwargs) -> np.ndarray:
    """
    Exponential Euler integrator.
    Solves the linear leak term exactly using the known Jacobian (-1/tau_m),
    then applies remaining input as a forcing term.
    Exact for pure LIF; best default choice for this network.

    For dv/dt = (v_rest - v)/tau_m + I:
        v_new = v_rest + (v - v_rest)*exp(-dt/tau_m) + I*tau_m*(1 - exp(-dt/tau_m))
    """

    decay = np.exp(-dt / tau_m)

    return v_rest + (v - v_rest) * decay + I_total * tau_m * (1.0 - decay)


def rk4(v, v_rest, tau_m, I_total_fn, dt, t=0.0, **kwargs) -> np.ndarray:
    """
    Classic 4th-order Runge-Kutta integrator.
    Most accurate fixed-step method; evaluates the derivative four times per step.
    I_total_fn is callable (t, v) -> total current, because RK4 needs intermediate
    evaluations at fractional timesteps.
    """
    def dv(t_, v_):
        return (v_rest - v_) / tau_m + I_total_fn(t_, v_)

    # Compute the four RK4 slopes
    k1 = dv(t,            v)
    k2 = dv(t + dt / 2,   v + dt / 2 * k1)
    k3 = dv(t + dt / 2,   v + dt / 2 * k2)
    k4 = dv(t + dt,       v + dt     * k3)

    return v + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk45(v, v_rest, tau_m, I_total_fn, dt, t=0.0, **kwargs) -> np.ndarray:
    """
    Adaptive Runge-Kutta-Fehlberg integrator.
    Automatically adjusts timestep to maintain error tolerance, which can be
    more efficient for stiff or highly nonlinear dynamics. Uses RK4 and RK5
    estimates to compute error and adjust dt.

    Note: This is a more complex integrator that may require additional state
    (e.g. previous error estimates) to be carried between steps for optimal performance.
    For simplicity, this implementation does not carry state and may not achieve
    full adaptive benefits without modification.
    """
    # For simplicity, we'll just call scipy's implementation here, which handles the adaptive logic internally.
    # In a real implementation, you might want to implement the RK45 logic yourself to have more control over state and performance.
    def dv(t_, v_):
        v_scalar = v_[0]
        return [(v_rest - v_scalar) / tau_m + I_total_fn(t_, v_scalar)]
    
    # Use scipy's solve_ivp with method='RK45' for adaptive integration
    sol = scipy.integrate.solve_ivp(dv, [t, t + dt], [v], method='RK45', rtol=1e-6)

    if not sol.success:
        raise RuntimeError(f"rk45 integration failed: {sol.message}")
    
    return sol.y[0, -1]  # Return the final value at t + dt


def leapfrog_step(v, v_half, v_rest, tau_m, I_total, dt, first_step: bool = False, **kwargs):
    """
    Störmer-Verlet (leapfrog) integrator.
    Symplectic — conserves phase-space volume, making it better than Euler
    for long runs with oscillatory dynamics.

    Requires v_half (voltage at previous half-step) to be carried between
    timesteps. On the first step, bootstraps v_half using Euler.

    Returns (v_new, v_half_new).
    """
    def dv(v_):
        return (v_rest - v_) / tau_m + I_total

    if first_step or v_half is None:
        # Bootstrap: estimate half-step with Euler
        v_half = v + 0.5 * dt * dv(v)

    # Full leapfrog update
    v_new = v_half + 0.5 * dt * dv(v_half)
    v_half_new = v_half + dt * dv(v_new)

    return v_new, v_half_new


# Integrators registry for simulation.py
INTEGRATORS = {"euler": euler,
               "exp_euler": exp_euler,
               "rk4":       rk4,
               "rk45":      rk45,
               "leapfrog":  leapfrog_step,
               }