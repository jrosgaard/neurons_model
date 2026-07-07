"""
src/neurons_model/waveform.py
Generates EEG-band sinusoidal waveforms for use as neural input signals.

Based on:
Nayak CS, Anilkumar AC. Normal EEG Waveforms. 
[Updated 2025 Aug 3]. In: StatPearls [Internet]. 
Treasure Island (FL): StatPearls Publishing; 2026 Jan-. 
Available from: https://www.ncbi.nlm.nih.gov/books/NBK539805/
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class BandSpec:
    low_hz:        float
    high_hz:       float
    default_amp: float   # default amplitude

BAND_SPECS = {
    "infraslow": BandSpec(0.01, 0.1, 0.3),
    "delta":     BandSpec(0.5, 4.0, 0.7),
    "theta":     BandSpec(4.0, 8.0, 0.8),
    "alpha":     BandSpec(8.0, 13.0, 1.0),
    "sigma":     BandSpec(12.0, 16.0, 0.7),
    "beta":      BandSpec(13.0, 30.0, 0.6),
    "gamma":     BandSpec(30.0, 80.0, 0.4),
}

# Maximum amplitude of 1 across all bands for normalization purposes
_max_amp = max(spec.default_amp for spec in BAND_SPECS.values()) 

# waveform generator function
def generate_waveform(duration_ms: float, dt_ms: float, rhythm: str, 
                      amplitude: float | None = None, freq_hz: float | None = None,
                      phase_jitter: float = 0.1, amplitude_std_fraction: float | None = None, 
                      seed: int   = 42
                      ) -> np.ndarray:
    """
    Generate a sinusoidal waveform for a given EEG rhythm band.

    Parameters
    ----------
    duration_ms            : total duration in milliseconds
    dt_ms                  : timestep in ms — must match SimulationConfig.dt_ms
    rhythm                 : one of the keys in BAND_SPECS
    amplitude              : optional dimensionless peak amplitude override 
                            (if None, uses default_amp from BAND_SPECS normalized by _max_amp)
    freq_hz                : frequency in Hz, if None uses the midpoint of the rhythm band
    phase_jitter           : std of per-cycle Gaussian phase perturbation (radians)
    amplitude_std_fraction : std of Gaussian noise added each timestep, relative to amplitude
    seed                   : random seed

    Returns
    -------
    dimensionless and normalized signal : np.ndarray of shape (n_steps,)
    """
    
    if rhythm not in BAND_SPECS:
        raise ValueError(f"Unknown rhythm '{rhythm}'. Choose from: {list(BAND_SPECS)}")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if phase_jitter < 0:
        raise ValueError("phase_jitter must be >= 0")
    if amplitude_std_fraction is not None and amplitude_std_fraction < 0:
        raise ValueError("amplitude_std_fraction must be >= 0")

    # Pull frequency and amplitude for the specified rhythm
    spec = BAND_SPECS[rhythm]
    rng = np.random.default_rng(seed)

    # Set frequency to midpoint of band if not specified, and validate it's positive
    freq_hz = freq_hz if freq_hz is not None else 0.5 * (spec.low_hz + spec.high_hz)
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")

    # Determine amplitude with noise
    base_amp = amplitude if amplitude is not None else spec.default_amp / _max_amp
    amp_std = amplitude_std_fraction * base_amp if amplitude_std_fraction is not None else 0.1 * base_amp

    # Generate time vector
    t_ms  = np.arange(0, duration_ms, dt_ms)
    t_s   = t_ms / 1000.0
    n     = len(t_ms)

    # Per-cycle phase jitter
    period_steps = max(1, int(round(1.0 / (freq_hz * dt_ms / 1000.0))))
    phase = np.zeros(n)
    cumulative_jitter = 0.0
    for i in range(n):
        if i % period_steps == 0:
            cumulative_jitter += rng.normal(0.0, phase_jitter)
        phase[i] = 2.0 * np.pi * freq_hz * t_s[i] + cumulative_jitter

    # Generate the sinusoidal signal and add amplitude noise
    amp = max(0.0, rng.normal(loc=base_amp, scale=amp_std))
    signal = amp * np.sin(phase)

    return signal