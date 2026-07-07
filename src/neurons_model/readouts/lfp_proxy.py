"""
src/neurons_model/readouts/lfp_proxy.py
LFP proxy calculations.

TODO: The simulator already returns a field proxy directly. Revisit whether
this file should become the canonical LFP reduction helper or be removed.
"""


def lfp_proxy(currents: list[float]) -> float:
    if not currents:
        return 0.0
    return sum(currents) / len(currents)
