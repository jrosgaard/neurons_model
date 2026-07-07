python - <<'EOF'
from neurons_model.loader import load_preset
from neurons_model.network import Network
import numpy as np

cfg = load_preset("src/neurons_model/presets/healthy.yaml")
net = Network.from_config(cfg)

print("=== Voltage thresholds ===")
for pop in net.populations:
    print(f"  {pop.name}: v_rest={pop.config.v_rest_mv}, v_thresh={pop.config.v_threshold_mv}, baseline={pop.config.baseline_input}")

print("\n=== Weight matrix ===")
print(f"  Min weight: {net.weights.min():.4f}")
print(f"  Max weight: {net.weights.max():.4f}")
print(f"  Non-zero: {(net.weights != 0).sum()}")

print("\n=== Pathway summary ===")
for p in cfg.pathways:
    print(f"  {p.source}->{p.target}: weight={p.weight}, reversal={p.reversal_mv}, tau_rise={p.tau_rise_ms}, tau_decay={p.tau_decay_ms}")

print("\n=== External input ===")
print(f"  mode={cfg.external_input.mode}")
print(f"  target={cfg.external_input.target_population}")
print(f"  amplitude={cfg.external_input.amplitude}")
print(f"  start={cfg.external_input.start_ms}, stop={cfg.external_input.stop_ms}")

print("\n=== Baseline input sanity ===")
for pop in net.populations:
    gap = pop.config.v_threshold_mv - pop.config.v_rest_mv
    print(f"  {pop.name}: threshold gap = {gap:.1f} mV, baseline_input = {pop.config.baseline_input}")
EOF