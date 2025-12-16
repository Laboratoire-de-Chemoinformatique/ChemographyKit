from typing import List, Union

import numpy as np
import pandas as pd

from chemographykit.utils.density import calculate_grid, filter_by_threshold


def class_prob_from_density(class_density, classes_counts=None):
    """
    Calculate the probability of each class from the class density matrix.

    Parameters
    ----------
    class_density : ndarray
        A 2D array where each column represents the density for a class at each node.
    classes_counts : ndarray, optional
        An array representing the counts of each class, used for normalization.

    Returns
    -------
    ndarray
        A 2D array of class probabilities at each node.

    Notes
    -----
    Normalizes class density by the total density and optionally by the minimum class count
    to handle class imbalance.
    """
    if classes_counts is not None:
        norm_coeff = classes_counts.min() / classes_counts
        class_density *= norm_coeff[np.newaxis, :]

    density = class_density.sum(-1)[:, np.newaxis]

    # safe division to bypass division by zero
    class_prob = np.divide(
        class_density, density, out=np.zeros_like(class_density), where=density != 0
    )

    return class_prob


def get_class_inds(class_labels, classes):
    """
    Convert class labels to indices based on a list of class names.

    Parameters
    ----------
    class_labels : Union[List[int], np.ndarray]
        Class labels as a list, an array of integers.
    classes : List[str]
        A list of class names corresponding to the classes in the dataset.

    Returns
    -------
    np.ndarray
        Array of indices corresponding to class labels.

    Notes
    -----
    If `class_labels` is a string, it is assumed to be a path to a file from which labels are read.
    """
    class_inds = [classes.index(cl) for cl in class_labels]
    return np.array(class_inds)


def get_class_density_matrix(
    responsibilities,
    class_labels: Union[str, List[int], np.ndarray],
    class_name: List[str] = ["1", "2"],
    normalize: bool = False,
):
    """
    Compute the class density matrix from responsibility data.

    Parameters
    ----------
    responsibilities : ndarray
        A matrix of responsibilities, where rows correspond to samples and columns to nodes.
    class_labels : Union[str, List[int], np.ndarray]
        Class labels for each sample as a list, array, or path to a label file.
    classes : List[str], optional
        A list of unique class identifiers. Defaults to ["1", "2"].
    normalize : bool, optional
        Whether to normalize class densities. Defaults to False. This will normalize both the class density and class probability matrices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the overall density matrix, the class density matrix, and the class probability matrix.

    Notes
    -----
    This function calculates how much each class contributes to each node in terms of responsibility.
    Optionally normalizes these contributions to probabilities.
    """
    n_lines_rsvm, num_nodes = responsibilities.shape

    class_count = np.zeros((len(class_name),))
    density = np.zeros((num_nodes,))
    class_density = np.zeros(
        (
            num_nodes,
            len(class_name),
        )
    )

    if not isinstance(class_labels, np.ndarray):
        class_inds = get_class_inds(class_labels, class_name)
    else:
        class_inds = class_labels

    for class_label, line_id in zip(class_inds, range(n_lines_rsvm)):
        density += responsibilities[line_id]
        class_density[:, class_label] += responsibilities[line_id]
        class_count[class_label] += 1

    if normalize:
        class_prob = class_prob_from_density(class_density, class_count)
    else:
        class_prob = class_prob_from_density(class_density)
    return density, class_density, class_prob


def class_density_to_table(
    density,
    class_density,
    class_prob,
    node_threshold=0.0,
    output_csv_file="",
    class_name=["1", "2"],
    normalized=False,
):
    """
    Convert class density data to a tabular format and optionally save to a CSV file.

    Parameters
    ----------
    density : ndarray
        Overall density at each node.
    class_density : ndarray
        Class-specific density at each node.
    class_prob : ndarray
        Class probabilities at each node.
    node_threshold : float, optional
        Minimum density threshold for including a node in the output. Defaults to 0.0.
    output_csv_file : str, optional
        Path to save the output CSV file. If not specified, no file is saved.
    class_name: List[str], optional
        A list of unique class identifiers. Defaults to ["1", "2"].
    normalized: Boolean
        Was the get_class_density_matrix function called using normalize=True to construct the densities and prob ndarrays.

    Returns
    -------
    DataFrame
        A DataFrame containing the grid mapping and class densities and probabilities.

    Notes
    -----
    This function is used to visualize or further analyze class densities in a structured format.
    """

    if normalized:
        norm_text = "_norm"
    else:
        norm_text = ""

    source = calculate_grid(density)
    source[str(class_name[0]) + norm_text + "_prob"] = filter_by_threshold(
        class_prob[:, 0], density, node_threshold
    )
    source[str(class_name[1]) + norm_text + "_prob"] = filter_by_threshold(
        class_prob[:, 1], density, node_threshold
    )
    source[str(class_name[0]) + norm_text + "_density"] = class_density[:, 0]
    source[str(class_name[1]) + norm_text + "_density"] = class_density[:, 1]

    if output_csv_file:
        source.to_csv(output_csv_file, index=False)
    return source
