from __future__ import annotations

import numpy as np

from neurons_model import (
    InputConfig,
    Network,
    compute_metrics,
    load_named_preset,
    run_simulation,
    scale_network_config,
)


def test_curated_preset_loads_and_builds_network() -> None:
    cfg = load_named_preset("healthy_quiet")
    cfg = scale_network_config(cfg, scale_factor=0.2)

    net = Network.from_config(cfg)

    assert net.v_mv.shape == (20,)
    assert net.weights.shape == (20, 20)
    assert set(net.pop_index) == {"Ex", "In_fast", "In_adapt"}
    assert np.all(np.isfinite(net.v_mv))


def test_network_assembly_is_seeded_and_deterministic() -> None:
    cfg = load_named_preset("healthy_quiet")
    cfg = scale_network_config(cfg, scale_factor=0.2)

    net_a = Network.from_config(cfg)
    net_b = Network.from_config(cfg)

    np.testing.assert_array_equal(net_a.weights, net_b.weights)
    np.testing.assert_array_equal(net_a.receptor_code, net_b.receptor_code)


def test_simulation_smoke_produces_finite_outputs(capsys) -> None:
    cfg = load_named_preset("healthy_quiet")
    cfg = scale_network_config(cfg, scale_factor=0.2)
    cfg.simulation.duration_ms = 20.0
    cfg.simulation.dt_ms = 0.1

    net = Network.from_config(cfg)
    result = run_simulation(net, integrator="rk4", rhythm="beta", full_result=False)
    capsys.readouterr()

    assert result.t_ms.shape == (200,)
    assert result.v_trace.shape == (20, 200)
    assert result.field_proxy.shape == (200,)
    assert result.I_syn_trace is None
    assert np.all(np.isfinite(result.v_trace))
    assert np.all(np.isfinite(result.field_proxy))

    metrics = compute_metrics(result)
    assert metrics["v_finite"]
    assert metrics["field_finite"]
    assert np.isfinite(metrics["Ex_mean_firing_rate_hz"])


def test_input_config_accepts_runtime_drive_modes() -> None:
    for mode in ("waveform", "pulse_waveform", "ramp", "ramp_waveform"):
        cfg = InputConfig(mode=mode, target_population="Ex")
        assert cfg.mode == mode
