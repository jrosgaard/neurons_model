"""
src/neurons_model/functionals.py
Score functions for summarizing network behavior and model regimes.

Functions:
- functional_J
- connectivity_spectral_proxy
- activity_score
- health_score
- shannon_entropy
"""

from __future__ import annotations

import numpy as np


def functional_J(metrics: dict, weight_matrix: np.ndarray | None = None) -> float:
    """
    Combined score for tuning and comparing regimes.
    """
    score = 0.0
    score += health_score(metrics)
    score += activity_score(metrics)

    if weight_matrix is not None:
        score += connectivity_spectral_proxy(weight_matrix)

    return float(score)

def connectivity_spectral_proxy(weight_matrix: np.ndarray) -> float:
    """
    Heuristic proxy based on the spectral radius of the weight matrix.

    Notes
    -----
    This is not a true dynamical stability measure for the full spiking system.
    It is only a rough structural proxy for recurrent gain.
    """
    eigenvalues = np.linalg.eigvals(weight_matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    return 1.0 / (1.0 + spectral_radius)


def activity_score(metrics: dict) -> float:
    """
    Reward moderate excitatory activity and penalize silence or overactivity.
    """
    score = 0.0

    ex_rate = metrics.get("Ex_mean_firing_rate_hz", np.nan)
    entropy = metrics.get("population_count_entropy_bits", np.nan)

    if np.isfinite(ex_rate):
        score -= max(0.0, 1.0 - ex_rate) * 2.0
        score -= max(0.0, ex_rate - 20.0) * 2.0

    if np.isfinite(entropy):
        score += entropy

    return score


def health_score(metrics: dict) -> float:
    """
    Reward voltage plausibility, moderate firing, and low synchrony.
    """
    if not metrics.get("v_finite", False) or not metrics.get("field_finite", False):
        return -1e9

    score = 0.0

    mean_v = metrics.get("mean_membrane_potential_mv", np.nan)
    min_v = metrics.get("min_membrane_potential_mv", np.nan)
    ex_rate = metrics.get("Ex_mean_firing_rate_hz", np.nan)
    ifast_rate = metrics.get("In_fast_mean_firing_rate_hz", np.nan)
    iadapt_rate = metrics.get("In_adapt_mean_firing_rate_hz", np.nan)
    ex_sync = metrics.get("Ex_synchrony_index", np.nan)

    if np.isfinite(mean_v):
        score -= abs(mean_v + 70.0)

    if np.isfinite(min_v):
        score -= max(0.0, -120.0 - min_v) * 5.0

    if np.isfinite(ex_rate):
        score -= max(0.0, ex_rate - 20.0) * 2.0
        score -= max(0.0, 1.0 - ex_rate) * 2.0

    if np.isfinite(ifast_rate):
        score -= max(0.0, ifast_rate - 30.0)

    if np.isfinite(iadapt_rate):
        score -= max(0.0, iadapt_rate - 30.0)

    if np.isfinite(ex_sync):
        score -= ex_sync * 10.0

    return score


def shannon_entropy(p):
    """Compute the Shannon entropy of a probability distribution p."""
    p = p[p > 0]
    shannon_entropy = -np.sum(p * np.log2(p))
    return shannon_entropy
