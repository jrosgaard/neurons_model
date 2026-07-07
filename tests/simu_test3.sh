python - <<'EOF'
import numpy as np
from neurons_model.loader import load_preset
from neurons_model.network import Network
from neurons_model.simulation import run_simulation

v_rest   = -65.0
v_thresh = -50.0
tau_m    = 20.0
baseline = 0.6
ext_amp  = 0.30
ADEX_DELTA_T = 2.0
ADEX_A   = 4.0
ADEX_TAU_W = 144.0
dt = 0.1

v = -65.0
w = 0.0
print("step |    v     | I_leak  | I_exp   | I_adapt | I_ext  |   dv")
for step in range(100):
    I_leak  = (v_rest - v) / tau_m
    I_exp   = (ADEX_DELTA_T / tau_m) * np.exp((v - v_thresh) / ADEX_DELTA_T)
    I_adapt = w / tau_m
    I_ext   = baseline + ext_amp
    dv      = dt * (I_leak + I_exp - I_adapt + I_ext)
    if step % 10 == 0:
        print(f"  {step:2d} | {v:7.3f} | {I_leak:.4f} | {I_exp:.4f} | {I_adapt:.4f} | {I_ext:.4f} | {dv:.4f}")
    v += dv
    if v >= v_thresh:
        print(f"  SPIKE at step {step}, v={v:.3f}")
        break
else:
    print(f"\nNo spike. Final v={v:.3f}")

cfg = load_preset("src/neurons_model/presets/healthy.yaml")
net = Network.from_config(cfg)

# Monkey-patch to record refractory state
import neurons_model.simulation as sim_module
original_run = sim_module.run_simulation

result = run_simulation(net, integrator="exp_euler")

dt = cfg.simulation.dt_ms
pulse_start = int(cfg.external_input.start_ms / dt)
pulse_stop  = int(cfg.external_input.stop_ms  / dt)

v0 = result.v_trace[0, :]
print(f"Neuron 0 max voltage during pulse: {v0[pulse_start:pulse_stop].max():.3f}")
print(f"Neuron 0 voltage at step {pulse_start+325}: {v0[pulse_start+325]:.3f}")
print(f"Expected spike at ~step {pulse_start+325} ({(pulse_start+325)*dt:.1f}ms)")

# Check if any neuron reaches threshold
print(f"\nMax voltage across ALL neurons during pulse: {result.v_trace[:, pulse_start:pulse_stop].max():.3f}")
print(f"v_thresh = -50.0")

# Check if waveform is subtracting during pulse
from neurons_model.waveform import generate_waveform
drive = generate_waveform(cfg.simulation.duration_ms, dt, rhythm="beta",
                          amplitude=None, seed=cfg.simulation.seed)
print(f"\nDrive during pulse: min={drive[pulse_start:pulse_stop].min():.4f}  max={drive[pulse_start:pulse_stop].max():.4f}")
print(f"Note: negative drive values subtract from I_ext")
print(f"Worst case total: {0.6 + 0.3 + drive[pulse_start:pulse_stop].min():.4f} mV/ms")
EOF