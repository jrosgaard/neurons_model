python - <<'EOF'
from neurons_model.loader import load_preset
from neurons_model.network import Network

cfg = load_preset("src/neurons_model/presets/healthy.yaml")
net = Network.from_config(cfg)

print(f"Total neurons: {sum(p.count for p in net.populations)}")
for p in net.populations:
    print(f"  {p.name} ({p.kind}): neurons {p.start}–{p.end}")
print(f"Weight matrix shape: {net.weights.shape}")
print(f"Non-zero weights: {(net.weights != 0).sum()}")
EOF