import numpy as np
import pandas as pd


def calculate_grid(density, legacy=False):
    """
    Calculates a grid mapping from node indices to their coordinates.

    Parameters
    ----------
    density : ndarray
        The density array whose length equals the total number of nodes.
    legacy : bool, optional
        Whether to use a legacy mode for grid calculation. Default is False.

    Returns
    -------
    DataFrame
        A DataFrame containing node indices and their respective coordinates in the grid.

    Notes
    -----
    This function is used to calculate the (x, y) coordinates for a given density map.
    If `legacy` is False, the node indexing is transposed to match expected grid layouts.
    """
    n_nodes = density.shape[-1]
    axis_len = int(np.sqrt(n_nodes))
    x, y = np.meshgrid(range(1, axis_len + 1), range(1, axis_len + 1))

    if legacy:
        nodes = np.arange(1, n_nodes + 1)
    else:
        nodes = np.arange(1, n_nodes + 1).reshape((axis_len, axis_len)).T.ravel()

    source = {
        "x": x.ravel(),
        "y": y.ravel(),
        "nodes": nodes,
    }

    source_df = pd.DataFrame(source)
    source_df = source_df.sort_values(by=["nodes"])
    source_df["density"] = density

    return source_df


def get_density_matrix(responsibilities: np.ndarray):
    assert len(responsibilities.shape) == 2
    return responsibilities.sum(axis=0)


def density_to_table(density, node_threshold=0.0, output_csv_file=""):
    source = calculate_grid(density)
    source["filtered_density"] = filter_by_threshold(density, density, node_threshold)
    if output_csv_file:
        source.to_csv(output_csv_file, index=False)
    return source


def filter_by_threshold(modified_density, reference_density, node_threshold):
    """
    Apply a threshold to a density matrix based on a reference density to filter nodes.

    Parameters
    ----------
    modified_density : ndarray
        The density values to be filtered.
    reference_density : ndarray
        The reference density values used for applying the threshold.
    node_threshold : float
        The density threshold below which nodes will be set to NaN in the modified density.

    Returns
    -------
    ndarray
        The modified density array with nodes set to NaN where the reference density is below the threshold.

    Notes
    -----
    This function is used to selectively remove or ignore nodes in density calculations
    that do not meet a specified threshold based on a reference set.
    """
    new_density = modified_density.copy()
    new_density[reference_density < node_threshold] = np.nan
    return new_density
