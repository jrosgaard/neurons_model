"""
neurons_model/config.py
Configuration primitives for microcircuit simulations.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass, field



# Population kind defines the functional role of a population. Existing presets
# use values such as "excitatory", "inhibitory_fast", "inhibitory_adapting",
# and "modulatory", but extensions may introduce values such as "glioma".
PopulationKind = str

# Perturbation kind defines the type of manipulation applied to a population or pathway
PerturbationKind = Literal[
    "ablation",
    "excitability_shift",
    "hebbian",
    "synaptic_gain",
    "input_drive",
    ]

# Task kind defines the type of computational task the network is performing
TaskKind = Literal[
    "relay",
    "binary_discrimination",
    ]

# Readout kind defines the types of outputs we want to record from the simulation
ReadoutKind = Literal[
    "spikes",
    "rates",
    "lfp_proxy",
    "metrics",
    "spectra",
    ]

# Cell_class defines the biological class of a cell type
CellClass = Literal[
    "neuron", 
    "interneuron", 
    "glia", 
    "other"
    ]

# State defines the functional state of a cell type
state = Literal[
    "dormant", 
    "active", 
    "lesioned", 
    "homeostatic", 
    "stem-like", 
    "mature",
    "ramified",
    "amoeboid",
    ]

# Dynamics defines the mathematical model used to simulate the population's activity
dynamics = Literal[
    "lif", 
    "izhikevich", 
    "hh", 
    "FHN", 
    "other", 
    "astrocyte_modulatory",
    "myelination_modulatory",
    "immune_modulatory",
    ]

# PopulationConfig defines the properties of a neural population in the network
class PopulationConfig(BaseModel):
    name: str = Field(..., description="Unique population name, e.g. E, I_fast, I_adapt")
    kind: PopulationKind
    count_n: int = Field(..., gt=0)
    cell_class: CellClass = "neuron"
    state: state = "dormant"
    dynamics: dynamics = "lif"
    is_spiking: bool = True

    # Simple point-neuron parameters
    v_rest_mv: float = -65.0
    v_reset_mv: float = -65.0
    v_threshold_mv: float = -50.0
    tau_m_ms: float = Field(20.0, gt=0)
    tau_refractory_ms: float = Field(2.0, ge=0)

    # Generic knobs for later model variants
    adaptation_strength: float = Field(0.0, ge=0)
    baseline_input: float = 0.0
    noise_std: float = Field(0.0, ge=0)
    parameters: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_cell_type_spec(cls, spec: Any) -> "PopulationConfig":
        """Create a validated population from a CellTypeSpec-like object."""
        return cls.model_validate(spec.to_population_config_dict())


# ReceptorKind defines the types of synaptic receptors mediating pathways
ReceptorKind = Literal["ampa", "nmda", "gaba_a", "gaba_b"]

# PathwayConfig defines the properties of synaptic connections between populations
class PathwayConfig(BaseModel):
    source: str
    target: str
    receptor: ReceptorKind
    probability: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., description="Signed synaptic weight")
    delay_ms: float = Field(1.0, ge=0.0)
    tau_rise_ms: float = Field(..., gt=0.0)
    tau_decay_ms: float = Field(..., gt=0.0)
    reversal_mv: float
    metadata: dict[str, Any] = Field(default_factory=dict)


    @model_validator(mode="after")
    def validate_kinetics(self) -> "PathwayConfig":
        if self.tau_decay_ms <= self.tau_rise_ms:
            raise ValueError("tau_decay_ms must be greater than tau_rise_ms.")
        return self

# SimulationConfig defines the parameters for running the simulation
class SimulationConfig(BaseModel):
    dt_ms: float = Field(0.1, gt=0)
    duration_ms: float = Field(500.0, gt=0)
    seed: int = 1

# InputConfig defines the properties of external inputs to the network
class InputConfig(BaseModel):
    mode: Literal["none", "poisson", "pulse", "pattern"] = "poisson"
    target_population: Optional[str] = None

    # Poisson-like input
    rate_hz: float = Field(5.0, ge=0.0)

    # Pulse input
    start_ms: float = Field(100.0, ge=0.0)
    stop_ms: float = Field(200.0, ge=0.0)
    amplitude: float = 1.0

# TaskConfig defines the computational task the network is performing, which can guide input patterns and readouts
class TaskConfig(BaseModel):
    kind: TaskKind = "relay"
    input_population: Optional[str] = None
    output_population: Optional[str] = None

    # For binary discrimination later
    pattern_a_strength: float = 1.0
    pattern_b_strength: float = 1.5

# PerturbationConfig defines manipulations to apply to populations or pathways, which can be used to simulate lesions, pharmacological effects, or other interventions
class PerturbationConfig(BaseModel):
    kind: PerturbationKind
    target: str = Field(..., description="Population name or pathway like E->I_fast")
    value: float = Field(..., description="Meaning depends on perturbation type")

    @model_validator(mode="after")
    def validate_value(self) -> "PerturbationConfig":
        if self.kind == "ablation" and not (0.0 <= self.value <= 1.0):
            raise ValueError("For ablation, value must be a fraction in [0, 1].")
        return self

# ReadoutConfig defines which types of outputs to record from the simulation, which can be used for analysis and comparison to empirical data
class ReadoutConfig(BaseModel):
    enabled: list[ReadoutKind] = Field(
        default_factory=lambda: ["spikes", "rates", "lfp_proxy", "metrics"]
        )

# NetworkConfig is the top-level configuration object that encapsulates all aspects of the network architecture, simulation parameters, inputs, tasks, perturbations, and readouts. It also includes validation to ensure internal consistency of references.
class NetworkConfig(BaseModel):
    name: str
    description: str = ""

    populations: list[PopulationConfig]
    pathways: list[PathwayConfig]

    # These defaults only work because all fields in SimulationConfig 
    # and InputConfig have their own defaults. If required fields
    # are added to either class, an explicit default must be provided here instead.
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    external_input: InputConfig = Field(default_factory=InputConfig)

    task: TaskConfig = Field(default_factory=TaskConfig)
    perturbations: list[PerturbationConfig] = Field(default_factory=list)
    readouts: ReadoutConfig = Field(default_factory=ReadoutConfig)

    @model_validator(mode="after")
    def validate_references(self) -> "NetworkConfig":
        population_names = {p.name for p in self.populations}
        population_kinds = {p.kind for p in self.populations}

        if len(population_names) != len(self.populations):
            raise ValueError("Population names must be unique.")

        total_neurons = sum(p.count_n for p in self.populations)
        if total_neurons <= 0:
            raise ValueError("Total neuron count must be positive.")

        for pathway in self.pathways:
            if pathway.source not in population_names:
                raise ValueError(f"Unknown pathway source: {pathway.source}")
            
            if pathway.target not in population_names:
                raise ValueError(f"Unknown pathway target: {pathway.target}")

        if (self.external_input.target_population is not None
            and self.external_input.target_population not in population_names):
            raise ValueError(f"Unknown external_input.target_population: "
                             f"{self.external_input.target_population}")

        if self.task.input_population is not None and self.task.input_population not in population_names:
            raise ValueError(f"Unknown task.input_population: {self.task.input_population}")

        if self.task.output_population is not None and self.task.output_population not in population_names:
            raise ValueError(f"Unknown task.output_population: {self.task.output_population}")

        for perturbation in self.perturbations:
            if perturbation.kind in {"ablation", "excitability_shift", "input_drive"}:
                if (
                    perturbation.target not in population_names
                    and perturbation.target not in population_kinds
                ):
                    raise ValueError(
                        f"Unknown perturbation target population or kind: {perturbation.target}"
                    )

            if perturbation.kind == "synaptic_gain":
                if "->" not in perturbation.target:
                    raise ValueError("Synaptic gain perturbation target must look like 'E->I_fast'.")
                
                src, dst = [s.strip() for s in perturbation.target.split("->", 1)]

                if src not in population_names or dst not in population_names:
                    raise ValueError(f"Unknown synaptic target pathway: {perturbation.target}")

        return self
