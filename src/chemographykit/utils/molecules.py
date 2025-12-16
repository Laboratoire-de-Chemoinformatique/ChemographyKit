import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_latent_coords(
    responsibilities, legacy=False, correction=False, return_node=False
):
    """
    Calculate molecular coordinates based on responsibility matrices.

    Parameters
    ----------
    responsibilities : ndarray
        A 2D array of responsibilities, where each row corresponds to a molecule and each column to a node.
    legacy : bool, optional
        Whether to use a legacy coordinate system. Default is False.
    correction : bool, optional
        Apply a correction to coordinates for visualization tools like Altair. Default is False.
    return_node : bool, optional
        Whether to return the primary (most responsible) node for each molecule. Default is False.
    Returns
    -------
    DataFrame
        A DataFrame with columns:
        - 'x' and 'y': The calculated coordinates for each molecule.
        - 'node_index' (optional): The index of the most responsible node.

    Notes
    -----
    This function calculates the weighted average position of each molecule based on its node responsibilities.
    - If `correction` is True, it adjusts coordinates to align visually on grid-based plots.
    - If `return_node` is True, the function also includes the most responsible node per molecule.

    """
    assert len(responsibilities.shape) == 2

    n_nodes = responsibilities.shape[-1]
    axis_len = int(np.sqrt(n_nodes))

    if legacy:
        x, y = np.meshgrid(range(1, axis_len + 1), range(1, axis_len + 1))
    else:
        y, x = np.meshgrid(range(1, axis_len + 1), range(1, axis_len + 1))

    x_coord = (x.ravel() * responsibilities).sum(-1)
    y_coord = (y.ravel() * responsibilities).sum(-1)

    if (
        correction
    ):  # correcting coordinates for proper visualisation for altair and others
        x_coord += 0.5
        y_coord += 0.5

    if return_node:
        node_indices = (
            responsibilities.argmax(axis=-1) + 1
        )  # Get the index of max responsibility
    else:
        node_indices = None

    source = pd.DataFrame({"x": x_coord, "y": y_coord})

    if return_node:
        source["node_index"] = node_indices  # Include the most responsible node

    return source
