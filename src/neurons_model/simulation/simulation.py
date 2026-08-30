"""
src/neurons_model/simulation/simulation.py
Time-loop simulation engine for the microcircuit sandbox.
Supports AdEx (excitatory), LIF+adaptation (inhibitory_adapting),
and simple LIF (inhibitory_fast) neuron models.
Synaptic transmission is conductance-based with double-exponential kinetics.
Delays are rounded to the nearest timestep.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import time

from .network import Network
from .integrators import INTEGRATORS
from ..waveform import generate_waveform
from .drive import ext_drive


# Simulation result container
@dataclass
class SimulationResult:
    """
    All recorded outputs from one simulation run.
    """
    t_ms: np.ndarray                            # (n_steps,) time
    spike_times: dict[str, list[list[float]]]   # population name -> spike times (ms)
    spike_counts: dict[str, np.ndarray]         # population name -> (n_neurons, n_steps) 
    v_trace: np.ndarray                         # (n_neurons, n_steps) voltage traces
    field_proxy: np.ndarray                     # (n_steps,) summed abs(synaptic current)
    I_syn_trace: np.ndarray | None              # (n_neurons, n_steps) synaptic current trace (optional)
    I_ext_trace: np.ndarray | None              # (n_neurons, n_steps) external current trace (optional)
    I_adapt_current_trace: np.ndarray | None    # (n_neurons, n_steps) adaptation current trace (optional)
    w_trace: np.ndarray | None                  # (n_neurons, n_steps) adaptation variable trace (optional)


# AdEx constants
ADEX_DELTA_T = 2.0      # sharpness of spike initiation (mV)
ADEX_V_PEAK  = 20.0     # spike peak voltage for recording (mV)
ADEX_A       = 4.0      # subthreshold adaptation coupling (nS)
ADEX_B       = 80.5     # spike-triggered adaptation increment (pA)
ADEX_TAU_W   = 144.0    # adaptation time constant (ms)


# Main simulation function
def run_simulation(net: Network, 
                   integrator: str = 'rk4', 
                   rhythm: str | None = None, 
                   full_result: bool = True,
                   diagnostic_mode: bool = False) -> SimulationResult:
    """
    Run the simulation defined by net.config and return all readouts.
    Default integrator is RK4, but can choose "exp_euler" or "leapfrog".
    """
    # Load the integrator function from the INTEGRATORS dictionary
    if integrator not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{integrator}'. Choose from: {list(INTEGRATORS)}")
    integrate = INTEGRATORS[integrator] 

    # Unpack simulation parameters
    cfg = net.config
    sim = cfg.simulation
    dt = sim.dt_ms

    # Diagnostic printout of simulation parameters
    if diagnostic_mode:
        print(f"Running simulation with integrator='{integrator}'\n"
              f"rhythm='{rhythm}'\n"
              f"dt={dt} ms\n"
              f"duration={sim.duration_ms} ms\n")

    # Precompute simulation constants
    n_steps = int(np.round(sim.duration_ms / dt))
    n_neurons = net.v_mv.shape[0]

    # Working copies of state (don't mutate net in-place)
    v = net.v_mv.copy()
    ref = net.refractory.copy()

    # Per-neuron model parameters (flat arrays)
    v_rest = net.v_rest_mv
    v_thresh = net.v_threshold_mv
    v_reset = net.v_reset_mv
    tau_m = net.tau_m_ms
    tau_ref = net.tau_refractory_ms
    adapt_str = net.adaptation_strength
    baseline = net.baseline_input
    noise_std = net.noise_std

    # Neuron kind masks
    kind_arr = np.array(
        [kind
         for pop in net.populations
         for kind in [pop.kind] * pop.count
         ])
    
    # For now, excitatory populations use AdEx, inhibitory_adapting uses LIF
    # with adaptation, and other known population kinds use simple LIF dynamics.
    is_adex = kind_arr == "excitatory"
    is_adapt = kind_arr == "inhibitory_adapting"

    # Adaptation variable (used by AdEx and LIF+adapt)
    w = np.zeros(n_neurons)   # pA

    # Synaptic conductance arrays
    # One conductance trace per pathway, stored as (n_neurons,) target arrays
    # We track rise and decay components separately for double-exponential
    n_pathways = len(cfg.pathways)
    g_rise = np.zeros((n_pathways, n_neurons))   # fast rise component
    g_decay= np.zeros((n_pathways, n_neurons))   # slow decay component

    # Precompute pathway kinetic factors and reversal potentials
    pathway_reversal = np.array([p.reversal_mv for p in cfg.pathways])
    tau_rise_arr     = np.array([p.tau_rise_ms for p in cfg.pathways])
    tau_decay_arr    = np.array([p.tau_decay_ms for p in cfg.pathways])
    decay_rise  = np.exp(-dt / tau_rise_arr)    # (n_pathways,)
    decay_decay = np.exp(-dt / tau_decay_arr)

    # Precompute delay in timesteps (rounded)
    delay_steps = np.round(net.delays_ms / dt).astype(int)
    max_delay   = int(delay_steps.max()) + 1
    spike_buffer= np.zeros((max_delay, n_neurons), dtype=bool)  # ring buffer

    # Recording arrays
    t_ms         = np.arange(n_steps) * dt
    v_trace      = np.zeros((n_neurons, n_steps))
    spike_rec    = np.zeros((n_neurons, n_steps), dtype=bool)
    field_proxy    = np.zeros(n_steps)

    I_syn_trace = None
    I_ext_trace = None
    I_adapt_current_trace = None
    w_trace = None

    # Optional traces for synaptic current, external current, adaptation current, and adaptation variable.
    if full_result:
        I_syn_trace  = np.zeros((n_neurons, n_steps))
        I_ext_trace  = np.zeros((n_neurons, n_steps))
        I_adapt_current_trace = np.zeros((n_neurons, n_steps))
        w_trace      = np.zeros((n_neurons, n_steps))
 
    # Random number generator for noise and waveform
    rng = np.random.default_rng(sim.seed + 1)

    # Default rhythm is beta which is the most frequently observed pattern in normal adults and children.
    if rhythm is None:
        rhythm = "beta"

    # Custom rhythm parameters are not yet supported, but the framework is in place for future extension.
    if rhythm == "custom":
        print("Custom rhythm parameters not supported yet.")

    # Drive parameters for external input waveform.
    drive_amplitude = None
    drive_freq_hz = None
    drive_phase_jitter = 0.1
    drive_amplitude_std_fraction = None

    # Generate external drive waveform for the simulation duration
    drive = generate_waveform(
        cfg.simulation.duration_ms,
        dt,
        rhythm=rhythm,
        amplitude=drive_amplitude,
        freq_hz=drive_freq_hz,
        phase_jitter=drive_phase_jitter,
        amplitude_std_fraction=drive_amplitude_std_fraction,
        seed=sim.seed,
    )

    # Estimate oscillation frequency from local maxima rather than every sample
    # above a threshold, which overcounts and inflates the reported Hz.
    peak_idx = np.where((drive[1:-1] > drive[:-2]) & (drive[1:-1] >= drive[2:]))[0] + 1
    if len(peak_idx) > 1:
        diffs = np.diff(peak_idx) * dt / 1000.0
        est_freq = 1.0 / np.mean(diffs) if np.all(diffs > 0) else np.nan
    else:
        est_freq = np.nan

    if diagnostic_mode:
        print("Drive parameters: \n")
        print(f"Rhythm: {rhythm}")
        print(f"Amplitude (dimensionless, normalized): {drive.max():.3f}")
        print(f"Frequency estimate (Hz): {est_freq:.2f}" if np.isfinite(est_freq) else "Frequency estimate (Hz): unavailable")
        print(f"Phase jitter (radians): {drive_phase_jitter}")
        print(
            "Amplitude noise std fraction: "
            f"{0.1 if drive_amplitude_std_fraction is None else drive_amplitude_std_fraction}"
        )
        print("\n****** Starting simulation... ******\n")

    # --------------------------------------------------------------------------------------------
    # Main time loop

    # Initialize half-step voltage for leapfrog integrator if selected
    if integrator == "leapfrog":
        v_half = None  # leapfrog needs to track half-step voltage

    # Precompute normalisation factors for double-exponential peak = 1
    t_peak_arr = (
        (tau_decay_arr * tau_rise_arr) / 
        (tau_decay_arr - tau_rise_arr)
        ) * np.log(tau_decay_arr / tau_rise_arr)
    
    # Normalization factors for double-exponential conductance to have a peak of 1
    norm_factors = np.exp(-t_peak_arr / tau_decay_arr) - np.exp(-t_peak_arr / tau_rise_arr)

    # Check for non-positive normalization factors which would indicate invalid tau values.
    if norm_factors.min() <= 0:
        raise ValueError("Invalid tau_rise and tau_decay values leading to non-positive normalization factors.")

    # Set up a timer to measure the total simulation runtime.
    runtimer = [0.0, 0.0]
    runtimer[0] = time.time()

    # -----------------------------------------------------------------------------

    # Main simulation loop over time steps
    for step in range(n_steps):
        buf_idx = step % max_delay

        # Compute synaptic input for each pathway
        I_syn = np.zeros(n_neurons)

        # Loop over each pathway to compute synaptic currents based on pre-synaptic spikes and conductance dynamics.
        for k, pathway in enumerate(cfg.pathways):
            src = net.pop_index[pathway.source]
            tgt = net.pop_index[pathway.target]

            # Which source neurons spiked and arrive at this target now
            # Use per-connection delay from delay_steps matrix
            # (simplified: use pathway mean delay for the ring buffer index)
            pathway_delay = int(round(pathway.delay_ms / dt))
            buf_read = (step - pathway_delay) % max_delay
            pre_spikes = spike_buffer[buf_read, src.start:src.end]  # (src.count,)

            if pre_spikes.any():
                # Increment conductance rise component for target neurons
                # Weight matrix selects which targets each source connects to
                w_sub = net.weights[src.start:src.end, tgt.start:tgt.end]  # (src, tgt)
                spike_input = pre_spikes.astype(float) @ w_sub             # (tgt.count,)
                g_rise[k, tgt.start:tgt.end]  += spike_input / norm_factors[k]
                g_decay[k, tgt.start:tgt.end] += spike_input / norm_factors[k]

            # Conductance = decay - rise (double exponential), (n_neurons,)
            g = g_decay[k] - g_rise[k]

            # Synaptic current conductance-based
            I_syn += g * (pathway_reversal[k] - v)

        # Noise + baseline
        I_ext = baseline + noise_std * rng.standard_normal(n_neurons)
        
        # External pulse input
        t_now = step * dt
        ext = cfg.external_input

        # Add external drive to target population if specified
        if ext.target_population is not None:
            tgt_pop = net.pop_index[ext.target_population]

            # Apply external drive to target population if specified, modifying I_ext
            ext_drive(net, ext, tgt_pop, drive, I_ext, step, t_now, n_steps)

        else:
            # If no external drive, just use baseline + noise as I_ext
            pass

        # Voltage update — Euler step
        in_refractory = ref > 0

        # Leak current
        I_leak = (v_rest - v) / tau_m

        # AdEx exponential term (excitatory only)
        exp_term = np.clip((v - v_thresh) / ADEX_DELTA_T, -20.0, 20.0)
        I_exp = np.where(is_adex, (ADEX_DELTA_T / tau_m) * np.exp(exp_term), 0.0)

        # Adaptation current
        I_adapt_current = w / tau_m

        # --------------------------------------------------------------------------------------------
        # Integration code

        # Current total
        I_total = I_leak + I_exp + I_syn - I_adapt_current + I_ext
        I_other = I_syn - I_adapt_current + I_ext

        # I_total_fn is a function of (t, v) that returns the total current.
        # Used for euler, exp_euler, and rk4.
        def I_total_fn(t_, v_):
            exp_term = np.clip((v_ - v_thresh) / ADEX_DELTA_T, -20.0, 20.0)
            i_exp = np.where(is_adex, (ADEX_DELTA_T / tau_m) * np.exp(exp_term), 0.0,)
            return i_exp + I_other

        # Route and run chosen integration method.
        if integrator == "euler":
            v_non_ref = integrate(v, v_rest, tau_m, I_total_fn, dt)

        elif integrator == "exp_euler":
            v_non_ref = integrate(v, v_rest, tau_m, I_other, dt) 

        elif integrator == "rk4":
            v_non_ref = integrate(v, v_rest, tau_m, I_total_fn, dt)
        
        elif integrator == "rk45":
            raise NotImplementedError("RK45 integrator not implemented yet")

        elif integrator == "leapfrog":
            v_non_ref, v_half = integrate(v, v_half, v_rest, tau_m, I_total, dt,
                                          first_step=(step == 0))
            
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # Update voltage with refractory reset
        v_new = np.where(in_refractory, v_reset, v_non_ref)

        # --------------------------------------------------------------------------------------------
        # Spike detection and state updates

        # Spike detection
        spiked = (~in_refractory) & (v_new >= v_thresh)

        # Check for voltage spike, setting to v_reset or v_new depending on whether it spiked.
        v_new = np.where(spiked, v_reset, v_new)

        # Finite voltage check
        if not np.all(np.isfinite(v_new)):
            raise FloatingPointError(f"Non-finite voltage at step {step}, t={t_ms[step]} ms")
        
        # -----------------------------------------------------------------------------
        # Per-step Hebbian weight updates (static perturbations are applied once in Network.from_config())

        if net.plasticity_rules:
            prev_buf = (step - 1) % max_delay
            pre_spikes = spike_buffer[prev_buf]   # (n_neurons,) spikes from previous step
            for rule in net.plasticity_rules:
                net.weights = rule.update(net.weights, pre_spikes, spiked)

        # -----------------------------------------------------------------------------
        # Reset, update adaptation, decay conductances, and record state

        # Reset spiking neurons
        ref   = np.where(spiked, tau_ref, np.maximum(0.0, ref - dt))

        # Update adaptation variable w
        dw_adex  = dt * (ADEX_A * (v - v_rest) - w) / ADEX_TAU_W
        dw_adapt = dt * (adapt_str * (v - v_rest) - w) / ADEX_TAU_W
        w += np.where(is_adex,  dw_adex,
             np.where(is_adapt, dw_adapt, 0.0))
        
        # spike-triggered increment
        w  = np.where(spiked, w + ADEX_B, w)

        # Decay conductances
        g_rise  *= decay_rise[:, np.newaxis]
        g_decay *= decay_decay[:, np.newaxis]

        # Write spikes into ring buffer for future steps
        spike_buffer[buf_idx] = False          # clear slot
        spike_buffer[buf_idx] = spiked         # current spikes

        # Record current state
        v[:] = v_new
        v_trace[:, step]   = v
        spike_rec[:, step] = spiked
        field_proxy[step]    = np.abs(I_syn).sum()

        # Optional full traces for synaptic current, external current, adaptation current, and adaptation variable.
        if full_result:
            I_syn_trace[:, step] = I_syn
            I_ext_trace[:, step] = I_ext
            I_adapt_current_trace[:, step] = I_adapt_current
            w_trace[:, step] = w

    # --------------------------------------------------------------------------------------------
    # Package results

    # End the timer and compute total runtime
    runtimer[1] = time.time()

    # Process spike times and counts for each population
    spike_times  = {}
    spike_counts = {}

    # Loop over populations to extract spike times and counts from the spike_rec array
    for pop in net.populations:
        sl = slice(pop.start, pop.end)
        counts = spike_rec[sl, :]
        spike_counts[pop.name] = counts
        times = []

        for n_idx in range(pop.count):
            times.append(t_ms[counts[n_idx]].tolist())

        spike_times[pop.name] = times

    if diagnostic_mode:
        print("Total sim. run time: ", runtimer[1] - runtimer[0], "seconds")
        print("\n****** Simulation completed. ******\n")

    # Return the simulation results as a SimulationResult dataclass instance.
    return SimulationResult(t_ms=t_ms,
                            spike_times=spike_times,
                            spike_counts=spike_counts,
                            v_trace=v_trace,
                            I_syn_trace=I_syn_trace,
                            I_ext_trace=I_ext_trace,
                            I_adapt_current_trace=I_adapt_current_trace,
                            w_trace=w_trace,
                            field_proxy=field_proxy,)
