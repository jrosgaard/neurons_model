python - <<'EOF'
from neurons_model.waveform import generate_waveform
import numpy as np

sig = generate_waveform(8000.0, 0.1, rhythm="infraslow", amplitude=None, seed=7)
print(f"infraslow amp=None: min={sig.min():.4f}  max={sig.max():.4f}")
print(f"Expected: ~±0.2 (3.0/15.0)")

spec_amp = 3.0 / 15.0
print(f"Expected peak: {spec_amp:.4f}")
EOF