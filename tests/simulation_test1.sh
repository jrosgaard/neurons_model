python - <<'EOF'
from neurons_model.loader import load_preset
from neurons_model.network import Network
from neurons_model.simulation import run_simulation

cfg = load_preset("src/neurons_model/presets/healthy.yaml")
net = Network.from_config(cfg)
result = run_simulation(net)

total_spikes = sum(
    result.spike_counts[p].sum() for p in result.spike_counts
)
print(f"Simulation duration: {cfg.simulation.duration_ms} ms")
print(f"Total spikes: {total_spikes}")
for pop_name, counts in result.spike_counts.items():
    rate = counts.sum() / (counts.shape[0] * cfg.simulation.duration_ms / 1000)
    print(f"  {pop_name}: mean firing rate {rate:.1f} Hz")
print(f"Field proxy range: {result.field_proxy.min():.3f} – {result.field_proxy.max():.3f}")
EOF