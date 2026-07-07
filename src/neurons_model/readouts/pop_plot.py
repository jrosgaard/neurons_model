"""
src/neurons_model/readouts/pop_plot.py
Functions for plotting population-level readouts from simulations.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_weight_heatmap(net):
    """
    Plot a heatmap of the network's weight matrix.
    """
    weights = net.weights

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(weights, aspect="auto")
    fig.colorbar(im, ax=ax, label="Synaptic weight")

    ax.set_title("Network weight matrix")
    ax.set_xlabel("Post-synaptic neuron index")
    ax.set_ylabel("Pre-synaptic neuron index")

    tick_positions = []
    tick_labels = []

    for pop in net.populations:

        ax.axhline(pop.start - 0.5, linestyle="--", alpha=0.5)
        ax.axvline(pop.start - 0.5, linestyle="--", alpha=0.5)

        center = 0.5 * (pop.start + pop.end - 1)
        tick_positions.append(center)
        tick_labels.append(pop.name)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.show()


def plot_connectivity_heatmap(net):
    """
    Plot a heatmap of the network's connectivity matrix.
    """
    connectivity = (net.weights != 0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(connectivity, aspect="auto")
    fig.colorbar(im, ax=ax, label="Connection")

    ax.set_title("Network connectivity matrix")
    ax.set_xlabel("Post-synaptic neuron index")
    ax.set_ylabel("Pre-synaptic neuron index")

    tick_positions = []
    tick_labels = []

    for pop in net.populations:

        ax.axhline(pop.start - 0.5, linestyle="--", alpha=0.5)
        ax.axvline(pop.start - 0.5, linestyle="--", alpha=0.5)

        center = 0.5 * (pop.start + pop.end - 1)
        tick_positions.append(center)
        tick_labels.append(pop.name)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.show()


def plot_population_weight_summary(net):
    """
    Plot a summary of mean nonzero weights between populations.
    """
    n_pop = len(net.populations)
    summary = np.zeros((n_pop, n_pop))

    for i, src in enumerate(net.populations):
        for j, tgt in enumerate(net.populations):
            block = net.weights[np.ix_(src.indices, tgt.indices)]
            nonzero = block[block != 0]
            summary[i, j] = nonzero.mean() if nonzero.size > 0 else 0.0

    labels = [p.name for p in net.populations]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(summary)
    fig.colorbar(im, ax=ax, label="Mean nonzero weight")

    ax.set_xticks(range(n_pop))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n_pop))
    ax.set_yticklabels(labels)

    ax.set_title("Population-to-population mean weight")
    ax.set_xlabel("Target population")
    ax.set_ylabel("Source population")

    plt.tight_layout()
    plt.show()

