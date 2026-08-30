"""
neurons_model/network.py
Network assembly data structures.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np
from dataclasses import dataclass, field

from .config import NetworkConfig, PopulationConfig
from ..perturbations.base import perturbation_from_config

# Mapping of receptor names to integer codes for efficient processing.
RECEPTOR_TO_CODE = {"ampa": 1,
                    "nmda": 2,
                    "gaba_a": 3,
                    "gaba_b": 4,
                    }

@dataclass
class PopulationSlice:
    """Index range and metadata for one population within the flat neuron arrays."""
    name: str
    kind: str
    start: int
    end: int
    config: PopulationConfig

    @property
    def count(self) -> int:
        return self.end - self.start
    
    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.end)
    

@dataclass
class Network:
    """Runtime network state assembled from a NetworkConfig.
    All neuron state lives in flat arrays of length n_neurons.
    Populations are tracked as named slices into those arrays."""
    config: NetworkConfig
    populations: list[PopulationSlice]

    # Neuron state arrays
    v_mv: np.ndarray
    refractory: np.ndarray
    v_rest_mv: np.ndarray
    v_reset_mv: np.ndarray
    v_threshold_mv: np.ndarray
    tau_m_ms: np.ndarray
    tau_refractory_ms: np.ndarray
    adaptation_strength: np.ndarray
    baseline_input: np.ndarray
    noise_std: np.ndarray
    ablated_mask: np.ndarray

    # Connectivity matrices
    weights: np.ndarray
    delays_ms: np.ndarray
    reversal_mv: np.ndarray
    tau_rise_ms: np.ndarray
    tau_decay_ms: np.ndarray

    # Optional integer code for receptor type (e.g. AMPA, GABA) for each synapse
    receptor_code: np.ndarray | None

    is_spiking: bool = True
    is_modulatory: bool = False

    # Index to look up population slices by name
    pop_index: dict[str, PopulationSlice] = field(default_factory=dict)

    # Plasticity rules registered via HebbianPlasticity.apply_to_network()
    plasticity_rules: list = field(default_factory=list)

    @classmethod
    def from_config(cls, config: NetworkConfig, rng: np.random.Generator | None = None, weight_loop_threshold: int = 10_000) -> "Network":
        """Build a Network from a validated NetworkConfig."""
        config = deepcopy(config)
        if rng is None:
            rng = np.random.default_rng(config.simulation.seed)

        # Assign each population slice of the neuron array
        populations = []
        cursor = 0
        for pop_cfg in config.populations:
            start = cursor
            stop = cursor + pop_cfg.count_n
            populations.append(PopulationSlice(
                name=pop_cfg.name,
                kind=pop_cfg.kind,
                start=start,
                end=stop,
                config=pop_cfg,
            ))
            
            cursor = stop

        n_cells = cursor
        pop_index = {p.name: p for p in populations}

        # Initialise neuron state
        v_mv = np.full(n_cells, -65.0)
        v_rest_mv = np.empty(n_cells)
        v_reset_mv = np.empty(n_cells)
        v_threshold_mv = np.empty(n_cells)
        tau_m_ms = np.empty(n_cells)
        tau_refractory_ms = np.empty(n_cells)
        adaptation_strength = np.empty(n_cells)
        baseline_input = np.empty(n_cells)
        noise_std = np.empty(n_cells)
        for pop in populations:
            sl = slice(pop.start, pop.end)
            v_mv[sl] = pop.config.v_rest_mv
            v_rest_mv[sl] = pop.config.v_rest_mv
            v_reset_mv[sl] = pop.config.v_reset_mv
            v_threshold_mv[sl] = pop.config.v_threshold_mv
            tau_m_ms[sl] = pop.config.tau_m_ms
            tau_refractory_ms[sl] = pop.config.tau_refractory_ms
            adaptation_strength[sl] = pop.config.adaptation_strength
            baseline_input[sl] = pop.config.baseline_input
            noise_std[sl] = pop.config.noise_std

        refractory = np.zeros(n_cells)
        ablated_mask = np.zeros(n_cells, dtype=bool)

        # Build weight and delay matrices
        weights = np.zeros((n_cells, n_cells))
        delays_ms = np.zeros((n_cells, n_cells))
        reversal_mv = np.zeros((n_cells, n_cells))
        tau_rise_ms = np.zeros((n_cells, n_cells))
        tau_decay_ms = np.zeros((n_cells, n_cells))
        receptor_code = np.zeros((n_cells, n_cells), dtype=np.int8)

        for pathway in config.pathways:
            src = pop_index[pathway.source]
            tgt = pop_index[pathway.target]

            if src.count * tgt.count < weight_loop_threshold:
                # Looping is faster for small populations
                for i in src.indices:
                    for j in tgt.indices:
                        if src.name == tgt.name and i == j:
                            # Avoid self-connections for now
                            continue
                        if rng.random() < pathway.probability:
                            weights[i, j] = pathway.weight
                            delays_ms[i, j] = pathway.delay_ms
                            reversal_mv[i, j] = pathway.reversal_mv
                            tau_rise_ms[i, j] = pathway.tau_rise_ms
                            tau_decay_ms[i, j] = pathway.tau_decay_ms
                            receptor_code[i, j] = RECEPTOR_TO_CODE[pathway.receptor]
            else:
                # Vectorized for large populations
                mask = rng.random((src.count, tgt.count)) < pathway.probability
                if src.name == tgt.name:
                    np.fill_diagonal(mask, False)
                
                weights[np.ix_(src.indices, tgt.indices)] = np.where(mask, pathway.weight, 0.0)
                delays_ms[np.ix_(src.indices, tgt.indices)] = np.where(mask, pathway.delay_ms, 0.0)
                reversal_mv[np.ix_(src.indices, tgt.indices)] = np.where(mask, pathway.reversal_mv, 0.0)
                tau_rise_ms[np.ix_(src.indices, tgt.indices)] = np.where(mask, pathway.tau_rise_ms, 0.0)
                tau_decay_ms[np.ix_(src.indices, tgt.indices)] = np.where(mask, pathway.tau_decay_ms, 0.0)
                receptor_code[np.ix_(src.indices, tgt.indices)] = np.where(mask, RECEPTOR_TO_CODE[pathway.receptor], 0)

        net = cls(
            config=config,
            populations=populations,
            v_mv=v_mv,
            refractory=refractory,
            v_rest_mv=v_rest_mv,
            v_reset_mv=v_reset_mv,
            v_threshold_mv=v_threshold_mv,
            tau_m_ms=tau_m_ms,
            tau_refractory_ms=tau_refractory_ms,
            adaptation_strength=adaptation_strength,
            baseline_input=baseline_input,
            noise_std=noise_std,
            ablated_mask=ablated_mask,
            weights=weights,
            delays_ms=delays_ms,
            reversal_mv=reversal_mv,
            tau_rise_ms=tau_rise_ms,
            tau_decay_ms=tau_decay_ms,
            receptor_code=receptor_code,
            pop_index=pop_index,
        )

        for spec in net.config.perturbations:
            perturbation_from_config(spec).apply_to_network(net)

        return net

    def population_indices(self, population_name: str) -> np.ndarray:
        """
        Return neuron indices for a population selector.

        The selector may be either an explicit population name such as
        ``pyramidal`` or a population kind such as ``excitatory``.
        """
        if population_name in self.pop_index:
            return self.pop_index[population_name].indices

        matching = [pop.indices for pop in self.populations if pop.kind == population_name]
        if matching:
            return np.concatenate(matching)

        raise KeyError(population_name)

    def resolve_population_target_name(self, selector: str) -> str:
        """
        Resolve a selector to one explicit population name.

        This is used for APIs that currently accept only one target population,
        such as ``external_input.target_population``.
        """
        if selector in self.pop_index:
            return selector

        matches = [pop.name for pop in self.populations if pop.kind == selector]
        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                f"Population selector '{selector}' matches multiple populations: {matches}"
            )

        raise KeyError(selector)

    def ablate_neurons_by_fraction(
        self,
        target_population: str,
        fraction: float,
        seed: int,
    ) -> None:
        """Ablate a fraction of neurons from the chosen population."""
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("fraction must be in [0, 1].")

        indices = self.population_indices(target_population)
        if fraction == 0.0 or indices.size == 0:
            return

        n_ablate = int(round(indices.size * fraction))
        n_ablate = min(indices.size, n_ablate)
        if n_ablate == 0:
            return

        rng = np.random.default_rng(seed)
        selected = rng.choice(indices, size=n_ablate, replace=False)
        self.ablate_neurons_by_cell_list(selected.tolist())

    def ablate_neurons_by_cell_list(self, cell_list: list[int]) -> None:
        """Ablate the provided global neuron indices in place."""
        if len(cell_list) == 0:
            return

        indices = np.array(sorted(set(cell_list)), dtype=int)
        n_neurons = self.v_mv.shape[0]
        if np.any(indices < 0) or np.any(indices >= n_neurons):
            raise IndexError("cell_list contains neuron indices outside the network.")

        self.weights[indices, :] = 0.0
        self.weights[:, indices] = 0.0
        self.delays_ms[indices, :] = 0.0
        self.delays_ms[:, indices] = 0.0
        self.reversal_mv[indices, :] = 0.0
        self.reversal_mv[:, indices] = 0.0
        self.tau_rise_ms[indices, :] = 0.0
        self.tau_rise_ms[:, indices] = 0.0
        self.tau_decay_ms[indices, :] = 0.0
        self.tau_decay_ms[:, indices] = 0.0
        self.receptor_code[indices, :] = 0
        self.receptor_code[:, indices] = 0

        self.v_mv[indices] = self.v_rest_mv[indices]
        self.refractory[indices] = 0.0
        self.v_threshold_mv[indices] = np.inf
        self.baseline_input[indices] = 0.0
        self.noise_std[indices] = 0.0
        self.adaptation_strength[indices] = 0.0
        self.ablated_mask[indices] = True
