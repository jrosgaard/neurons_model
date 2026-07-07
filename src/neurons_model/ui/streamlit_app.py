"""
src/neurons_model/ui/streamlit_app.py
Streamlit UI for interactive simulation inspection.

To run locally, bash:
conda activate neuronSB1
streamlit run /Users/johanrosgaard/Documents/BNQ/neurons_model/src/neurons_model/ui/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from neurons_model.loader import load_preset
from neurons_model.presets.preset_manager import available_preset_paths, list_presets
from neurons_model.readouts.sim_metrics import compute_metrics
from neurons_model.simulation.network import Network
from neurons_model.simulation.scaling import scale_network_config
from neurons_model.simulation.simulation import SimulationResult, run_simulation


_INTEGRATOR_OPTIONS = {
    "Euler": "euler",
    "Exponential Euler": "exp_euler",
    "Runge-Kutta 4": "rk4",
    "Leapfrog": "leapfrog",
}
_RHYTHM_OPTIONS = (
    "infraslow",
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta",
    "gamma",
)
_PANEL_SPECS = {
    "Membrane potential": {
        "field": "v_trace",
        "ylabel": "Membrane potential (mV)",
        "title": "Membrane Potential Trace",
    },
    "Spike counts": {
        "field": "spike_counts",
        "ylabel": "Spike count",
        "title": "Spike Counts",
    },
    "Synaptic current": {
        "field": "I_syn_trace",
        "ylabel": "Synaptic current (pA)",
        "title": "Synaptic Current Trace",
    },
    "External current": {
        "field": "I_ext_trace",
        "ylabel": "External current (pA)",
        "title": "External Current Trace",
    },
}


def main() -> None:
    """Run the Streamlit app."""
    startup()


def _initialize_session_state() -> None:
    """Create persistent keys used across Streamlit reruns."""
    st.session_state.setdefault("sim_result", None)
    st.session_state.setdefault("sim_metrics", None)
    st.session_state.setdefault("last_run_summary", None)


def _format_preset_label(preset_name: str) -> str:
    """Convert a preset stem into a human-friendly label."""
    return preset_name.replace("_", " ").title()


def _preset_options() -> dict[str, Path]:
    """Return the available preset paths keyed by UI label."""
    return {
        _format_preset_label(name): path
        for name, path in available_preset_paths().items()
    }


def _validate_spike_raster_inputs(sim_result: SimulationResult) -> None:
    """Validate that the simulation result contains the traces needed for plotting."""
    if sim_result.spike_counts is None:
        raise ValueError("SimulationResult must contain spike_counts to plot raster.")

    if sim_result.t_ms is None:
        raise ValueError("SimulationResult must contain t_ms to plot raster.")

    if sim_result.v_trace is None:
        raise ValueError("SimulationResult must contain v_trace to plot raster.")

    if sim_result.I_syn_trace is None:
        raise ValueError(
            "SimulationResult must contain I_syn_trace to plot raster. "
            "Run the simulation with `full_result=True`."
        )

    if sim_result.I_ext_trace is None:
        raise ValueError(
            "SimulationResult must contain I_ext_trace to plot raster. "
            "Run the simulation with `full_result=True`."
        )


def _population_offsets(sim_result: SimulationResult) -> dict[str, tuple[int, int]]:
    """Map each population name to its slice in the flattened neuron trace arrays."""
    offsets: dict[str, tuple[int, int]] = {}
    start = 0

    for pop_name, spike_arr in sim_result.spike_counts.items():
        stop = start + spike_arr.shape[0]
        offsets[pop_name] = (start, stop)
        start = stop

    return offsets


def _build_summary_table(
    sim_result: SimulationResult,
    selected_populations: list[str],
    num_neurons: int,
    time_mask,
) -> pd.DataFrame:
    """Build a compact summary table for the selected traces."""
    offsets = _population_offsets(sim_result)
    rows: list[dict[str, float | int | str]] = []

    for pop_name in selected_populations:
        pop_spikes = sim_result.spike_counts[pop_name]
        start, _ = offsets[pop_name]
        n_to_plot = min(num_neurons, pop_spikes.shape[0])

        for local_idx in range(n_to_plot):
            global_idx = start + local_idx
            rows.append(
                {
                    "population": pop_name,
                    "neuron": local_idx,
                    "spike_count": int(pop_spikes[local_idx, time_mask].sum()),
                    "mean_v_mv": float(sim_result.v_trace[global_idx, time_mask].mean()),
                    "mean_I_syn": float(sim_result.I_syn_trace[global_idx, time_mask].mean()),
                    "mean_I_ext": float(sim_result.I_ext_trace[global_idx, time_mask].mean()),
                }
            )

    return pd.DataFrame(rows)


def _run_configured_simulation(
    *,
    preset_path: Path,
    integrator: str,
    rhythm: str,
    duration_ms: float,
    dt_ms: float,
    scale_factor: float,
    scaling_strategy: str = "preserve_in_degree",
) -> tuple[SimulationResult, dict[str, float | bool], dict[str, float | str | int]]:
    """Build the configured network, run the simulation, and summarize the output."""
    cfg = load_preset(preset_path)
    cfg.simulation.duration_ms = duration_ms
    cfg.simulation.dt_ms = dt_ms

    scaled_cfg = scale_network_config(
        cfg,
        scale_factor=scale_factor,
        strategy=scaling_strategy,
    )
    net = Network.from_config(scaled_cfg)
    sim_result = run_simulation(
        net,
        integrator=integrator,
        rhythm=rhythm,
        full_result=True,
    )
    metrics = compute_metrics(sim_result, save=False)
    summary = {
        "preset": scaled_cfg.name,
        "integrator": integrator,
        "rhythm": rhythm,
        "duration_ms": float(scaled_cfg.simulation.duration_ms),
        "dt_ms": float(scaled_cfg.simulation.dt_ms),
        "n_neurons": int(net.v_mv.shape[0]),
        "total_spikes": int(sum(arr.sum() for arr in sim_result.spike_counts.values())),
        "field_proxy_max": float(sim_result.field_proxy.max()) if sim_result.field_proxy.size else 0.0,
    }
    return sim_result, metrics, summary


def st_plot1(sim_result: SimulationResult) -> None:
    """Render an interactive Streamlit version of ``plot_spike_raster``."""
    _validate_spike_raster_inputs(sim_result)

    population_names = list(sim_result.spike_counts)
    if not population_names:
        st.warning("No populations are available in the provided simulation result.")
        return

    max_neurons = max(spike_arr.shape[0] for spike_arr in sim_result.spike_counts.values())
    default_neurons = min(5, max_neurons)
    panel_names = list(_PANEL_SPECS)
    t_min = float(sim_result.t_ms[0])
    t_max = float(sim_result.t_ms[-1])
    dt = float(sim_result.t_ms[1] - sim_result.t_ms[0]) if sim_result.t_ms.size > 1 else 1.0

    st.write(
        "Interactive spike-raster view. Choose populations, zoom the time window, "
        "and decide which trace panels to show."
    )

    with st.container(border=True):
        left, middle, right = st.columns([1.4, 1.0, 1.2])
        selected_populations = left.multiselect(
            "Populations",
            population_names,
            default=population_names,
        )
        num_neurons = middle.slider(
            "Neurons per population",
            min_value=1,
            max_value=max_neurons,
            value=default_neurons,
        )
        selected_panels = right.multiselect(
            "Panels",
            panel_names,
            default=panel_names,
        )

        time_window = st.slider(
            "Time window (ms)",
            min_value=t_min,
            max_value=t_max,
            value=(t_min, t_max),
            step=dt,
        )
        show_legend = st.toggle(
            "Show legend",
            value=len(selected_populations) * num_neurons <= 8 if selected_populations else False,
        )

    if not selected_populations:
        st.info("Select at least one population to draw the traces.")
        return

    if not selected_panels:
        st.info("Select at least one panel to render the plot.")
        return

    time_mask = (sim_result.t_ms >= time_window[0]) & (sim_result.t_ms <= time_window[1])
    if not time_mask.any():
        st.warning("The selected time window contains no samples.")
        return

    offsets = _population_offsets(sim_result)
    fig, axes = plt.subplots(
        len(selected_panels),
        1,
        figsize=(14, 3.5 * len(selected_panels)),
        sharex=True,
    )

    if len(selected_panels) == 1:
        axes = [axes]

    for axis, panel_name in zip(axes, selected_panels):
        spec = _PANEL_SPECS[panel_name]
        axis.set_title(spec["title"])

        for pop_name in selected_populations:
            pop_spikes = sim_result.spike_counts[pop_name]
            start, _ = offsets[pop_name]
            n_to_plot = min(num_neurons, pop_spikes.shape[0])

            for local_idx in range(n_to_plot):
                global_idx = start + local_idx
                label = f"{pop_name} neuron {local_idx}"

                if spec["field"] == "spike_counts":
                    axis.step(
                        sim_result.t_ms[time_mask],
                        pop_spikes[local_idx, time_mask],
                        where="post",
                        alpha=0.8,
                        label=label,
                    )
                else:
                    trace = getattr(sim_result, spec["field"])
                    axis.plot(
                        sim_result.t_ms[time_mask],
                        trace[global_idx, time_mask],
                        alpha=0.8,
                        label=label,
                    )

        axis.set_ylabel(spec["ylabel"])
        axis.grid(True, alpha=0.3)

        if show_legend:
            axis.legend(loc="upper right", fontsize="small", ncols=2)

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()

    plot_tab, table_tab = st.tabs(["Plot", "Summary"])
    plot_tab.pyplot(fig, clear_figure=True, use_container_width=True)

    summary_df = _build_summary_table(
        sim_result,
        selected_populations=selected_populations,
        num_neurons=num_neurons,
        time_mask=time_mask,
    )
    table_tab.dataframe(summary_df, use_container_width=True, height=320)

    plt.close(fig)


def st_run_simulation(
    *,
    preset_path: Path | str,
    integrator: str,
    rhythm: str,
    duration_ms: float,
    dt_ms: float,
    scale_factor: float = 1.0,
    scaling_strategy: str = "preserve_in_degree",
) -> SimulationResult:
    """Run a configured simulation inside Streamlit and persist the result."""
    with st.spinner("Running simulation..."):
        sim_result, metrics, summary = _run_configured_simulation(
            preset_path=Path(preset_path),
            integrator=integrator,
            rhythm=rhythm,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            scale_factor=scale_factor,
            scaling_strategy=scaling_strategy,
        )

    st.session_state["sim_result"] = sim_result
    st.session_state["sim_metrics"] = metrics
    st.session_state["last_run_summary"] = summary
    st.success("Simulation completed.")
    return sim_result


def startup() -> None:
    """Initialize the Streamlit app and expose simulation controls."""
    st.set_page_config(page_title="Neuron Sandbox", layout="wide")
    _initialize_session_state()
    preset_options = _preset_options()
    preset_descriptions = list_presets()

    st.title("Neuron Sandbox")
    st.write(
        "Configure a preset, run a simulation, and inspect the result below. "
        "The last successful run stays available while you adjust the plot controls."
    )

    with st.container(border=True):
        preset_label = st.selectbox(
            "Preset",
            list(preset_options),
            index=0,
        )
        selected_preset_name = preset_options[preset_label].stem
        preset_description = preset_descriptions.get(selected_preset_name)
        if preset_description:
            st.caption(preset_description)
        integrator_label = st.selectbox(
            "Integrator",
            list(_INTEGRATOR_OPTIONS),
            index=2,
        )
        rhythm = st.selectbox(
            "Rhythm",
            list(_RHYTHM_OPTIONS),
            index=5,
        )
        duration_ms = st.number_input(
            "Simulation duration (ms)",
            min_value=10.0,
            max_value=10000.0,
            value=400.0,
            step=10.0,
        )
        dt_ms = st.number_input(
            "Simulation timestep (ms)",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f",
        )
        scale_factor = st.slider(
            "Scaling factor",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        run_clicked = st.button("Start simulation", type="primary")

    if run_clicked:
        try:
            st_run_simulation(
                preset_path=preset_options[preset_label],
                integrator=_INTEGRATOR_OPTIONS[integrator_label],
                rhythm=rhythm,
                duration_ms=float(duration_ms),
                dt_ms=float(dt_ms),
                scale_factor=float(scale_factor),
            )
        except Exception as exc:
            st.session_state["sim_result"] = None
            st.session_state["sim_metrics"] = None
            st.session_state["last_run_summary"] = None
            st.exception(exc)

    summary = st.session_state.get("last_run_summary")
    metrics = st.session_state.get("sim_metrics")
    sim_result = st.session_state.get("sim_result")

    if summary is None or metrics is None or sim_result is None:
        st.info("Run a simulation to inspect traces and summary metrics.")
        return

    top_1, top_2, top_3, top_4 = st.columns(4)
    top_1.metric("Preset", str(summary["preset"]))
    top_2.metric("Neurons", int(summary["n_neurons"]))
    top_3.metric("Total spikes", int(summary["total_spikes"]))
    top_4.metric("Field proxy max", f"{float(summary['field_proxy_max']):.3f}")

    detail_1, detail_2, detail_3 = st.columns(3)
    detail_1.metric("Integrator", str(summary["integrator"]))
    detail_2.metric("Rhythm", str(summary["rhythm"]))
    detail_3.metric("Mean membrane V", f"{float(metrics['mean_membrane_potential_mv']):.3f} mV")

    st_plot1(sim_result)


if __name__ == "__main__":
    main()
