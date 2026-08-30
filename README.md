# neurons_model

# Welcome to the neuronSandBox package (neurons_model).

NeuronSandbox is a biological neural network-inspired modeling and simulation sandbox for in silico experimentation with small spiking-network behaviors and perturbations. Try out generating readouts from the presets and see what interesting network behavior you can create.

This package is maintained by Johan Rosgaard and was designed to be a base package for biologically inspired neuron network modeling. AI/LLM tools were used for creating and correcting sections of the codebase.

Cells in this package were based on excitatory Pyramidal cells and inhibitory interneurons found in the cerebral cortex. While based on behavior of true biological cells, this is not a perfect cortical model; still, many interesting network behaviors and results can be achieved. 

## Quick start

1. Create and activate a virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `PYTHONPATH=src python3 -m neurons_model`
4. Run tests:
   - `PYTHONPATH=src pytest -q`

## Project layout

- `src/neurons_model`: application package
- `tests`: unit tests
