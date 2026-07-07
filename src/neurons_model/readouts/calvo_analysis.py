"""
src/neurons_model/readouts/calvo_analysis.py
Analysis similar to paper below:

Robust Scaling in Human Brain Dynamics Despite Correlated Inputs and Limited Sampling Distortions,
Rubén Calvo, Carles Martorell, Adrián Roig, and Miguel A. Muñoz
DOI: https://doi-org.proxy.lib.ohio-state.edu/10.1103/36v9-wtm8

https://link.aps.org/doi/10.1103/36v9-wtm8

"""


from __future__ import annotations


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calvo_pca(data: pd.DataFrame,n_components: int = 10, all_eigen: bool = False
              ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform covariance-based PCA on a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Rows are observations, columns are variables.
    n_components : int, default=10
        Number of principal components to retain.

    Returns
    -------
    projected_df : pd.DataFrame
        Data projected onto the top principal components.
    eigenvalues : np.ndarray
        Top n_components eigenvalues, sorted descending.
    eigenvectors : np.ndarray
        Eigenvectors corresponding to the top n_components eigenvalues.
        Shape: (n_features, n_components)
    explained_variance_ratio : np.ndarray
        Fraction of total variance explained by the top n_components.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("data must not be empty")

    if data.isna().any().any():
        raise ValueError("data contains NaN values")

    n_features = data.shape[1]
    if not (1 <= n_components <= n_features):
        raise ValueError(f"n_components must be between 1 and {n_features}")

    data_centered = data - data.mean(axis=0)
    covariance_matrix = np.cov(data_centered, rowvar=False)

    eigenvalues_all, eigenvectors_all = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = eigenvalues_all[sorted_indices]
    eigenvectors_all = eigenvectors_all[:, sorted_indices]

    total_variance = np.sum(eigenvalues_all)
    explained_variance_ratio_all = (eigenvalues_all / total_variance 
                                    if total_variance > 0 else 
                                    np.zeros_like(eigenvalues_all))

    eigenvalues = eigenvalues_all[:n_components]
    eigenvectors = eigenvectors_all[:, :n_components]
    explained_variance_ratio = explained_variance_ratio_all[:n_components]

    projected_data = data_centered.to_numpy() @ eigenvectors

    projected_df = pd.DataFrame(projected_data, index=data.index, 
                                columns=[f"PC{i + 1}" for i in range(n_components)],)

    if all_eigen:
        return projected_df, eigenvalues_all, eigenvectors_all, explained_variance_ratio_all
    else:
        return projected_df, eigenvalues, eigenvectors, explained_variance_ratio


def calvo_pca_plot(sim_result, n_components: int = 10, z_score_norm: bool = True,
                   explained_variance_ratio_length: int = 10, 
                   title: str = "", show: bool = True,
                   ) -> tuple[plt.Figure | None, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot PCA of per-neuron voltage traces.

    Parameters
    ----------
    sim_result : SimulationResult
        Simulation result object containing v_trace and t_ms.
    n_components : int, default=10
        Number of principal components to retain. Must be at least 2.
    z_score_norm : bool, default=True
        Whether to z-score normalize neuron voltage traces before PCA.
    explained_variance_ratio_length : int, default=10
        Number of explained variance components to plot.
    title : str, default=""
        Figure-level title prefix.
    show : bool, default=True
        Whether to call plt.show() before returning.

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        The created figure, or None if show is False.
    projected_df : pd.DataFrame
        PCA projections with time column added.
    eigenvalues : np.ndarray
        Top n_components eigenvalues.
    eigenvectors : np.ndarray
        Top n_components eigenvectors.
    explained_variance_ratio : np.ndarray
        Explained variance ratio for the returned components.
    """
    if sim_result is None:
        raise ValueError("sim_result must not be None")

    if not hasattr(sim_result, "v_trace") or not hasattr(sim_result, "t_ms"):
        raise TypeError("sim_result must have v_trace and t_ms attributes")

    v_df = pd.DataFrame(sim_result.v_trace.T)
    v_df.columns = [f"v_neuron_{i}" for i in range(v_df.shape[1])]

    if v_df.empty:
        raise ValueError("Voltage trace DataFrame must not be empty")

    if z_score_norm:
        v_df_proc = (v_df - v_df.mean()) / v_df.std(ddof=0)
        v_df_proc = v_df_proc.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    else:
        v_df_proc = v_df.copy()

    n_features = v_df_proc.shape[1]
    if n_components < 2 or n_components > n_features:
        raise ValueError(f"n_components must be between 2 and {n_features}")

    projected_df, eigenvalues, eigenvectors, explained_variance_ratio = calvo_pca(
        v_df_proc, n_components=n_components,)

    projected_df["t_ms"] = sim_result.t_ms

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        ax0 = axes[0]
        ax0.plot(projected_df["PC1"], projected_df["PC2"], alpha=0.7, c="black")

        sc = ax0.scatter(projected_df["PC1"], projected_df["PC2"], 
                        c=projected_df["t_ms"], cmap="viridis", s=10,)

        for idx in range(0, len(projected_df), max(1, len(projected_df) // 20)):
            row = projected_df.iloc[idx]
            ax0.annotate(f"{row['t_ms']:.1f}", (row["PC1"], row["PC2"]), fontsize=6)

        ax0.scatter(projected_df["PC1"].iloc[0], projected_df["PC2"].iloc[0],
                    marker="o", s=50, label="start",)
        ax0.scatter(projected_df["PC1"].iloc[-1], projected_df["PC2"].iloc[-1], 
                    marker="x", s=50, label="end",)

        fig.colorbar(sc, ax=ax0, label="Time (ms)")
        ax0.set_xlabel("PC1")
        ax0.set_ylabel("PC2")
        ax0.set_title("PCA of Per-Neuron Voltage Traces")
        ax0.legend()
        ax0.grid(True)

        ax1 = axes[1]
        n_plot = min(explained_variance_ratio_length, len(explained_variance_ratio))
        ax1.plot(range(1, n_plot + 1), explained_variance_ratio[:n_plot], marker="o",)

        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Explained variance ratio")
        ax1.set_title("PCA explained variance")
        ax1.grid(True)

        fig.suptitle(f"{title}: z-scored PCA" if z_score_norm else f"{title}: PCA", fontsize=16)
        fig.tight_layout()
        plt.show()
    
    else:
        fig = None

    return fig, projected_df, eigenvalues, eigenvectors, explained_variance_ratio


def pca_analysis(sim_result, n_components: int = 10, z_score_norm: bool = True,) -> dict[str, object]:
    """
    Compute PCA summary measures for per-neuron voltage traces.
    """
    fig, projected_df, eigenvalues, eigenvectors, explained_variance_ratio = calvo_pca_plot(
        sim_result, n_components=n_components, z_score_norm=z_score_norm, show=False,)

    coords = projected_df[["PC1", "PC2"]].to_numpy()

    early = projected_df[projected_df["t_ms"] < projected_df["t_ms"].max() * 0.25]
    late = projected_df[projected_df["t_ms"] > projected_df["t_ms"].max() * 0.75]

    # Centroid shift between early and late trajectory segments
    centroid_shift = np.linalg.norm(
        early[["PC1", "PC2"]].mean().to_numpy()
        - late[["PC1", "PC2"]].mean().to_numpy()
    )

    # Participation ratio as a dimensionality measure
    participation_ratio = (eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2)

    # Trajectory geometry in PC1-PC2 space
    step_vectors = np.diff(coords, axis=0)
    step_lengths = np.linalg.norm(step_vectors, axis=1)
    path_length = np.sum(step_lengths)

    # Distance from each point to the initial point in PC1-PC2 space
    path_deviation = np.linalg.norm(coords - coords[0], axis=1)

    net_displacement = np.linalg.norm(coords[-1] - coords[0])
    directionality_ratio = net_displacement / path_length if path_length > 0 else 0.0

    # Dispersion / thickness proxies
    centroid = coords.mean(axis=0)
    radial_distances = np.linalg.norm(coords - centroid, axis=1)
    mean_radial_distance = np.mean(radial_distances)
    std_radial_distance = np.std(radial_distances)

    # Variance summaries
    pc1_variance_ratio = explained_variance_ratio[0]
    pc2_variance_ratio = explained_variance_ratio[1] if len(explained_variance_ratio) > 1 else np.nan
    pc1_pc2_cumulative_variance = explained_variance_ratio[:2].sum()

    # Loadings
    pc1_loadings = eigenvectors[:, 0]
    pc2_loadings = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.full(eigenvectors.shape[0], np.nan)

    PCA_char = {
        "fig": fig,
        "summary": {
            "centroid_shift": centroid_shift,
            "participation_ratio": participation_ratio,
            "path_length": path_length,
            "directionality_ratio": directionality_ratio,
            "net_displacement": net_displacement,
            "mean_radial_distance": mean_radial_distance,
            "std_radial_distance": std_radial_distance,
            "pc1_variance_ratio": pc1_variance_ratio,
            "pc2_variance_ratio": pc2_variance_ratio,
            "pc1_pc2_cumulative_variance": pc1_pc2_cumulative_variance,
        },
        "series": {
            "path_deviation": path_deviation,
            "projected_df": projected_df,
        },
        "loadings": {
            "pc1_loadings": pc1_loadings,
            "pc2_loadings": pc2_loadings,
        },
        "raw": {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "explained_variance_ratio": explained_variance_ratio,
        },
    }

    return PCA_char