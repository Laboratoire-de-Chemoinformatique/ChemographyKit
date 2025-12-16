from typing import List, Union

import numpy as np
import pandas as pd

from chemographykit.utils.density import calculate_grid, filter_by_threshold, get_density_matrix


def norm_reg_density(reg_density: np.ndarray, density: np.ndarray):
    """
    Normalize the regression density by the overall density.

    Parameters
    ----------
    reg_density : ndarray
        The regression density matrix for each node.
    density : ndarray
        The overall density matrix for each node.

    Returns
    -------
    ndarray
        The normalized regression density matrix.

    Notes
    -----
    Performs safe division to prevent division by zero. This function is used to ensure
    that regression values are properly scaled by the underlying density, facilitating
    correct interpretation and further analysis.
    """
    reg_density = np.divide(
        reg_density,
        density,
        out=np.zeros_like(reg_density),
        where=density != 0,
    )
    return reg_density


def get_reg_density_matrix(
    responsibilities: np.ndarray, reg_values: Union[List[float], np.ndarray]
):
    """
    Compute the regression density matrix from responsibilities and regression values.

    Parameters
    ----------
    responsibilities : ndarray
        The responsibility matrix where rows correspond to samples and columns to nodes.
    reg_values : Union[List[NUMERIC], np.ndarray]
        Regression values as a list, an array of numeric values.

    Returns
    -------
    Tuple[ndarray, ndarray]
        A tuple containing the overall density matrix and the regression density matrix.

    Notes
    -----
    This function calculates the regression impact of each node based on the provided responsibilities and regression values.
    """
    assert len(responsibilities.shape) == 2

    readed_reg_vals = reg_values

    if isinstance(readed_reg_vals, list):
        readed_reg_vals = np.array(readed_reg_vals)

    density = get_density_matrix(responsibilities)
    reg_density = responsibilities.T @ readed_reg_vals

    reg_density = norm_reg_density(reg_density, density)

    return density, reg_density


def reg_density_to_table(
    density: np.ndarray,
    regression_density: np.ndarray,
    node_threshold: float = 0.0,
    output_csv_file: str = "",
):
    """
    Convert regression density data to a tabular format and optionally save to a CSV file.

    Parameters
    ----------
    density : ndarray
        Overall density at each node.
    regression_density : ndarray
        Regression-specific density at each node.
    node_threshold : float, optional
        Minimum density threshold for including a node in the output. Defaults to 0.0.
    output_csv_file : str, optional
        Path to save the output CSV file. If not specified, no file is saved.

    Returns
    -------
    DataFrame
        A DataFrame containing the grid mapping and regression densities.

    Notes
    -----
    This function is used to visualize or further analyze regression densities in a structured format.
    """
    source = calculate_grid(density)

    source["filtered_reg_density"] = filter_by_threshold(
        regression_density, density, node_threshold
    )

    if output_csv_file:
        source.to_csv(output_csv_file, index=False)
    return source
