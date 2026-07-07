"""
src/neurons_model/simulation/drive.py
Functions for generating input drive signals to the network, e.g. sinusoidal waveforms.
"""

from __future__ import annotations

import numpy as np


def ext_drive(net, ext, tgt_pop, drive, I_ext, step, t_now, n_steps):
        """
        Apply external drive to target population if specified.

        Inputs:
        - ext.target_population: name of target population to apply drive to, or None for no drive
        - ext.mode: "waveform", "pulse", "pulse_waveform", "ramp", or "ramp_waveform"
        - ext.amplitude: amplitude of drive for pulse and ramp modes
        - drive: array of drive values for current step, used for waveform modes
        - I_ext: array of external currents to be updated with drive
        - step: current simulation step index
        - t_now: current simulation time in ms
        - n_steps: total number of simulation steps (for validating drive length)
        """
        

        if len(drive) != n_steps:
            raise ValueError(f"Drive length {len(drive)} does not match n_steps {n_steps}")
            
        if ext.mode == "waveform":
            if ext.start_ms <= t_now <= ext.stop_ms:
                I_ext[tgt_pop.start:tgt_pop.end] += drive[step]
                return I_ext

        elif ext.mode == "pulse":
            if ext.start_ms <= t_now <= ext.stop_ms:
                I_ext[tgt_pop.start:tgt_pop.end] += ext.amplitude
                return I_ext

        elif ext.mode == "pulse_waveform":
            if ext.start_ms <= t_now <= ext.stop_ms:
                I_ext[tgt_pop.start:tgt_pop.end] += ext.amplitude + drive[step]
                return I_ext

        elif ext.mode == "ramp":
            if ext.start_ms <= t_now <= ext.stop_ms:
                ramp_factor = (t_now - ext.start_ms) / (ext.stop_ms - ext.start_ms)
                I_ext[tgt_pop.start:tgt_pop.end] += ramp_factor * ext.amplitude
                return I_ext

        elif ext.mode == "ramp_waveform":
            if ext.start_ms <= t_now <= ext.stop_ms:
                ramp_factor = (t_now - ext.start_ms) / (ext.stop_ms - ext.start_ms)
                I_ext[tgt_pop.start:tgt_pop.end] += ramp_factor * ext.amplitude + drive[step]
                return I_ext
                
        else:
            raise ValueError(f"Unknown external input mode: {ext.mode}")
            
