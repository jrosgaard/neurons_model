# neurons_model

# Welcome to the neuronSandBox package (neuronSB).
This package is maintained by Johan Rosgaard and was designed to be a base package for biologically inspired neuron network modeling. AI/LLM tools were used for correcting and generating various sections of the codebase.

Cells in this package were based on excitatory Pyramidal cells and inhibitory interneurons found in the cerebral cortex. While based on behavoir of true biological cells, this is far from a perfect cortical model, but many interesting network behaviors and results can still be achieved. 

## Quick start

1. Create and activate a virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `PYTHONPATH=src python -m neurons_model`
4. Run tests:
   - `PYTHONPATH=src pytest -q`

## Project layout

- `src/neurons_model`: application package
- `tests`: unit tests
