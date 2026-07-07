"""
src/neurons_model/readouts/spectra.py
Lightweight spectral summary helpers.

TODO: Placeholder helper for future spectral analysis. Not currently called by
the active simulation/readout flow.
"""

from __future__ import annotations


def power(signal: list[float]) -> float:
    if not signal:
        return 0.0
    return sum(v * v for v in signal) / len(signal)
