"""
_EXT_GBMsim.integrator.py
Numerical integration helper using SciPy solve_ivp RK45.
"""

from __future__ import annotations

from scipy.integrate import solve_ivp

from .tumor_model import kp_coupled


class KPIntegrator:
    """Integrate the coupled Kirschner-Panetta model over one time window."""

    def integrate(self, state: list[float], params: list[float], t_span: tuple[float, float],
                  ) -> list[float]:
        solution = solve_ivp(fun=kp_coupled, t_span=t_span, y0=state, method="RK45", args=tuple(params),
                             rtol=1e-7, atol=1e-9, max_step=0.1,)

        if not solution.success:
            raise RuntimeError(f"GBMsim integration failed: {solution.message}")

        x_out, y_out, z_out = (float(value) for value in solution.y[:, -1])

        return [x_out, y_out, z_out]


kp_integrate = KPIntegrator

__all__ = ["KPIntegrator", "kp_integrate"]
