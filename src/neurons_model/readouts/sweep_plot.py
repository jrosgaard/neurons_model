"""
src/neurons_model/readouts/sweep_plot.py
Defines functions to plot results from parameter sweeps.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(x):
    """
    Normalize an array to the range [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)


def _prepare_sweep_df(sweepdf: pd.DataFrame) -> pd.DataFrame:
    """
    Return a validated, sorted copy of the sweep DataFrame.
    """
    required_columns = {
        "swept_weight",
        "functional_val",
        "mean_membrane_potential_mv",
        "min_membrane_potential_mv",
        "max_membrane_potential_mv",
    }
    missing = sorted(required_columns - set(sweepdf.columns))
    if missing:
        raise KeyError(f"sweepdf is missing required columns: {missing}")

    prepared = sweepdf.copy()
    if "pathway" not in prepared.columns:
        prepared["pathway"] = "sweep"

    return prepared.sort_values(["pathway", "swept_weight"]).reset_index(drop=True)


def _plot_grouped_lines(ax, sweepdf: pd.DataFrame, y_col: str, *, ylabel: str, marker: str = "o") -> bool:
    """
    Plot one line per pathway for a single metric.
    """
    if y_col not in sweepdf.columns:
        return False

    plotted = False
    for pathway, group in sweepdf.groupby("pathway", sort=False):
        ax.plot(
            group["swept_weight"],
            group[y_col],
            marker=marker,
            label=str(pathway),
        )
        plotted = True

    ax.set_xlabel("Swept weight")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return plotted


def sweep_plot_general(sweepdf: pd.DataFrame) -> None:
    """
    Plot summary metrics from a sweep DataFrame.
    This function plots key metrics against the swept weight, grouped by pathway
    when multiple sweep series are present.
    """
    sweepdf = _prepare_sweep_df(sweepdf)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_activity, ax_voltage, ax_spikes, ax_scatter = axes.flat

    activity_col = (
        "Ex_mean_firing_rate_hz"
        if "Ex_mean_firing_rate_hz" in sweepdf.columns
        else next((col for col in sweepdf.columns if col.endswith("_mean_firing_rate_hz")), None)
    )

    if activity_col is not None:
        line_handles = []
        line_labels = []
        for pathway, group in sweepdf.groupby("pathway", sort=False):
            (line,) = ax_activity.plot(
                group["swept_weight"],
                group[activity_col],
                marker="o",
                label=f"{pathway} rate",
            )
            line_handles.append(line)
            line_labels.append(f"{pathway} rate")

        ax_activity.set_ylabel(activity_col.replace("_", " "))
        ax_activity.set_xlabel("Swept weight")
        ax_activity.grid(True)

        ax_activity_2 = ax_activity.twinx()
        for pathway, group in sweepdf.groupby("pathway", sort=False):
            (line,) = ax_activity_2.plot(
                group["swept_weight"],
                group["functional_val"],
                marker="s",
                linestyle="--",
                label=f"{pathway} functional",
                
            )
            line_handles.append(line)
            line_labels.append(f"{pathway} functional")
        ax_activity_2.set_ylabel("Functional value")
        ax_activity.legend(line_handles, line_labels, loc="best", fontsize="small")
    else:
        _plot_grouped_lines(
            ax_activity,
            sweepdf,
            "functional_val",
            ylabel="Functional value",
        )

    ax_activity.set_title("Sweep: activity vs functional score")

    for pathway, group in sweepdf.groupby("pathway", sort=False):
        ax_voltage.plot(group["swept_weight"], group["mean_membrane_potential_mv"], marker="o", label=f"{pathway} mean V")
        ax_voltage.plot(group["swept_weight"], group["min_membrane_potential_mv"], marker="s", linestyle="--", label=f"{pathway} min V")
        ax_voltage.plot(group["swept_weight"], group["max_membrane_potential_mv"], marker="^", linestyle=":", label=f"{pathway} max V")
    ax_voltage.axhline(-120, linestyle="--", alpha=0.7, color="k", label="Min-V warning")
    ax_voltage.set_xlabel("Swept weight")
    ax_voltage.set_ylabel("Membrane potential (mV)")
    ax_voltage.set_title("Sweep: voltage regime")
    ax_voltage.grid(True)
    ax_voltage.legend(loc="best", fontsize="small")

    spikes_plotted = _plot_grouped_lines(
        ax_spikes,
        sweepdf,
        "total_spikes",
        ylabel="Total spikes",
        marker="o",
    )
    if "field_proxy_max" in sweepdf.columns:
        ax_spikes_2 = ax_spikes.twinx()
        field_handles = []
        field_labels = []
        for pathway, group in sweepdf.groupby("pathway", sort=False):
            (line,) = ax_spikes_2.plot(
                group["swept_weight"],
                group["field_proxy_max"],
                marker="s",
                linestyle="--",
                label=f"{pathway} field",
            )
            field_handles.append(line)
            field_labels.append(f"{pathway} field")
        ax_spikes_2.set_ylabel("Field proxy max")
        if spikes_plotted:
            handles, labels = ax_spikes.get_legend_handles_labels()
            ax_spikes.legend(handles + field_handles, labels + field_labels, loc="best", fontsize="small")
        else:
            ax_spikes.legend(field_handles, field_labels, loc="best", fontsize="small")
    elif spikes_plotted:
        ax_spikes.legend(loc="best", fontsize="small")
    ax_spikes.set_title("Sweep: spiking and field recruitment")

    if activity_col is None:
        x_col = "functional_val"
        x_label = "Functional value"
    else:
        x_col = activity_col
        x_label = activity_col.replace("_", " ")

    for pathway, group in sweepdf.groupby("pathway", sort=False):
        sc = ax_scatter.scatter(
            group[x_col],
            group["min_membrane_potential_mv"],
            c=group["swept_weight"],
            cmap="viridis",
            label=str(pathway),
        )
    ax_scatter.set_xlabel(x_label)
    ax_scatter.set_ylabel("Min membrane potential (mV)")
    ax_scatter.set_title("Sweep: activity vs voltage cost")
    ax_scatter.grid(True)
    ax_scatter.legend(loc="best", fontsize="small")
    fig.colorbar(sc, ax=ax_scatter, label="Swept weight")

    fig.tight_layout()
    plt.show()


def sweep_plot_norm_summary(sweepdf: pd.DataFrame) -> None:
    """
    Plot normalized summary metrics from a sweep DataFrame.
    This function normalizes each metric within each pathway sweep and plots the
    normalized curves against the swept weight.
    """
    sweepdf = _prepare_sweep_df(sweepdf)
    sweepdf = sweepdf.copy()

    activity_col = (
        "Ex_mean_firing_rate_hz"
        if "Ex_mean_firing_rate_hz" in sweepdf.columns
        else next((col for col in sweepdf.columns if col.endswith("_mean_firing_rate_hz")), None)
    )

    norm_frames = []
    for _, group in sweepdf.groupby("pathway", sort=False):
        group = group.copy()
        if activity_col is not None:
            group["activity_norm"] = normalize(group[activity_col])
        group["field_norm"] = normalize(group["field_proxy_max"]) if "field_proxy_max" in group.columns else np.nan
        group["minV_penalty_norm"] = normalize(-group["min_membrane_potential_mv"])
        group["functional_norm"] = normalize(group["functional_val"])
        norm_frames.append(group)

    sweepdf = pd.concat(norm_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    metric_specs = []
    if activity_col is not None:
        metric_specs.append(("activity_norm", f"{activity_col.replace('_', ' ')} (norm)", "o"))
    if "field_proxy_max" in sweepdf.columns:
        metric_specs.append(("field_norm", "Field max (norm)", "s"))
    metric_specs.extend([
        ("minV_penalty_norm", "Voltage penalty (norm)", "^"),
        ("functional_norm", "Functional (norm)", "d"),
    ])

    for metric_col, metric_label, marker in metric_specs:
        for pathway, group in sweepdf.groupby("pathway", sort=False):
            ax.plot(
                group["swept_weight"],
                group[metric_col],
                marker=marker,
                label=f"{pathway} {metric_label}",
            )

    ax.set_xlabel("Swept weight")
    ax.set_ylabel("Normalized value")
    ax.set_title("Sweep: normalized regime summary")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)

    fig.tight_layout()
    plt.show()
