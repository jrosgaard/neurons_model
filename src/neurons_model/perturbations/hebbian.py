"""
src/neurons_model/perturbations/hebbian.py
Hebbian-style plasticity perturbation and per-timestep update rules.

During network assembly, HebbianPlasticity.apply_to_network() registers the
rule on net.plasticity_rules so the simulation loop can call
rule.update(net.weights, pre_spikes, post_spikes) each timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import Perturbation


@dataclass(frozen=True, slots=True)
class HebbianPlasticity(Perturbation):
    """
    Registers a Hebbian plasticity rule on the network.

    During network assembly (apply_to_network) this perturbation appends
    itself to net.plasticity_rules so the simulation loop can call
    rule.update(net.weights, pre_spikes, post_spikes) each timestep.
    """

    learning_rate: float
    rule: Literal["hebbian", "anti_hebbian", "hebbian_with_decay"] = "hebbian"
    max_weight: float | None = None
    decay_rate: float = 0.0
    min_weight: float = 0.0

    def apply_to_network(self, net) -> None:
        """Register this plasticity rule on the network for per-step application."""
        if not hasattr(net, "plasticity_rules"):
            net.plasticity_rules = []
        net.plasticity_rules.append(self)

    def update(
        self,
        w: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Apply one plasticity step and return the updated weight matrix.

        Parameters
        ----------
        w : np.ndarray
            Synaptic weight matrix, convention w[i, j] = pre i -> post j.
        pre_spikes : np.ndarray
            Boolean/0-1 array of presynaptic spikes at the previous step.
        post_spikes : np.ndarray
            Boolean/0-1 array of postsynaptic spikes at the current step.

        Returns
        -------
        np.ndarray
            Updated weight matrix (same shape as w).
        """
        if self.rule == "hebbian":
            return hebbian_learning(
                w, pre_spikes, post_spikes,
                self.learning_rate, self.max_weight,
            )
        if self.rule == "anti_hebbian":
            return anti_hebbian_learning(
                w, pre_spikes, post_spikes,
                self.learning_rate, self.min_weight,
            )
        if self.rule == "hebbian_with_decay":
            return hebbian_with_decay(
                w, pre_spikes, post_spikes,
                self.learning_rate, self.decay_rate, self.max_weight,
            )
        raise ValueError(f"Unsupported Hebbian plasticity rule: {self.rule}")


def hebbian_learning(
    w: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    learning_rate: float,
    max_weight: float | None = None,
) -> np.ndarray:
    """
    Apply a simple coincidence-based Hebbian update.

    Parameters
    ----------
    w : np.ndarray
        Synaptic weight matrix with convention w[i, j] = pre i -> post j.
    pre_spikes : np.ndarray
        Boolean or 0/1 array for presynaptic spikes at previous step.
    post_spikes : np.ndarray
        Boolean or 0/1 array for postsynaptic spikes at current step.
    learning_rate : float
        Learning rate for weight updates.
    max_weight : float | None
        Optional upper bound on weights.

    Returns
    -------
    np.ndarray
        Updated weight matrix.
    """
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative.")

    pre_spikes = pre_spikes.astype(float)
    post_spikes = post_spikes.astype(float)

    delta_w = learning_rate * np.outer(pre_spikes, post_spikes)
    w_new = w + delta_w

    # Keep conductance-like weights nonnegative
    w_new = np.clip(w_new, 0.0, None)

    if max_weight is not None:
        w_new = np.minimum(w_new, max_weight)

    return w_new


def anti_hebbian_learning(
          w: np.ndarray,
          pre_spikes: np.ndarray,
          post_spikes: np.ndarray,
          learning_rate: float,
          min_weight: float = 0.0,
          ) -> np.ndarray:
    """
    Weaken synapses with coincident pre/post activity.
    Assumes w[i, j] = pre i -> post j.
    """
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative.")

    pre_spikes = pre_spikes.astype(float)
    post_spikes = post_spikes.astype(float)

    delta_w = learning_rate * np.outer(pre_spikes, post_spikes)
    w_new = w - delta_w
    w_new = np.maximum(w_new, min_weight)

    return w_new


def hebbian_with_decay(
    w: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    learning_rate: float,
    decay_rate: float = 0.0,
    max_weight: float | None = None,
) -> np.ndarray:
    """
    Simple Hebbian potentiation with optional global weight decay.

    This is a more useful first combined plasticity rule than averaging
    Hebbian and anti-Hebbian updates.
    """
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative.")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative.")

    pre_spikes = pre_spikes.astype(float)
    post_spikes = post_spikes.astype(float)

    delta_w = learning_rate * np.outer(pre_spikes, post_spikes)
    w_new = w + delta_w

    if decay_rate > 0:
        w_new *= (1.0 - decay_rate)

    w_new = np.clip(w_new, 0.0, None)

    if max_weight is not None:
        w_new = np.minimum(w_new, max_weight)

    return w_new


def hebbian_tracker(
    w: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    learning_rate: float,
    max_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hebbian update with tracking of actual weight changes.

    Returns a tuple of (updated_weights, delta_weights) where delta_weights
    reflects the true change after clipping, not the raw outer product.
    """
    w_new = hebbian_learning(w, pre_spikes, post_spikes, learning_rate, max_weight)
    return w_new, w_new - w