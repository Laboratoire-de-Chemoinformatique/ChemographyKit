from typing import List, Tuple, Union
from sklearn.neighbors import NearestNeighbors

import numpy as np


def resp_to_pattern(
    responsibilities: np.ndarray, n_bins: int = 10, threshold: float = 0.01
) -> np.ndarray:
    """
    Convert a vector of responsibility values into a Responsibility Pattern fingerprint.

    Parameters:
        responsibilities (np.ndarray): Array of responsibility values (typically in [0, 1]).
        n_bins (int): Number of bins to map into (values 1..n_bins). Default is 10.
        threshold (float): Values < threshold are set to 0. Default is 0.01.

    Returns:
        np.ndarray: Integer fingerprint vector in [0, n_bins], where values below `threshold`
                    are 0 and others are binned using floor(n_bins*x + 0.9) for stability
                    around bin edges (matches the original behavior when n_bins=10).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    resp = np.asarray(responsibilities, dtype=float)

    # Robust binning equivalent to the original (avoids float issues at exact edges)
    rp = np.floor(n_bins * resp + 0.9).astype(int)

    # Apply threshold and keep within bounds just in case
    rp[resp < threshold] = 0
    np.clip(rp, 0, n_bins, out=rp)

    return rp


def get_fingerprint_counts(fingerprint_array: np.ndarray) -> dict[Tuple[int, ...], int]:
    """
    Compute counts of unique fingerprints from an array of fingerprints.

    Parameters:
        fingerprint_array (np.ndarray): Array of fingerprints (each row is a pattern).

    Returns:
        dict: A dictionary where keys are tuples representing unique patterns,
              and values are the counts of occurrences.
    """
    unique_patterns, counts = np.unique(fingerprint_array, axis=0, return_counts=True)
    return {tuple(pattern): count for pattern, count in zip(unique_patterns, counts)}


def compute_rp_coverage(
    ref_lib: np.ndarray, test_lib: np.ndarray, use_weight: bool = True
) -> float:
    """
    Compute coverage or weighted coverage, starting directly from two NumPy arrays of
    responsibilities.

    If use_weight=True, Weighted coverage is:
        sum_{patterns in both} ref_count(pattern) / sum_{all patterns in ref} ref_count(pattern)

    If use_weight=False, Unweighted coverage is:
        (# of patterns in both ref and test) / (# of patterns in ref).

    Parameters
    ----------
    ref_lib : np.ndarray
        Shape (N_ref, D). Each row => responsibilities for one compound in reference set.
    test_lib : np.ndarray
        Shape (N_test, D). Each row => responsibilities for one compound in test set.
    use_weight : bool
        If True => weighted coverage, else unweighted coverage.

    Returns
    -------
    float
        Coverage or weighted coverage in [0,1].
    """
    # 1) Compute dictionaries of pattern -> occurrence_count for ref and test
    counts_ref = get_fingerprint_counts(ref_lib)
    counts_test = get_fingerprint_counts(test_lib)

    # 2) Use dictionary-intersection logic
    if use_weight:
        #
        # Weighted coverage:
        # sum_{p in both} ref_count(p) / sum_{p in ref} ref_count(p)
        #
        total_ref_count = sum(counts_ref.values())  # total # of comps in ref
        if total_ref_count == 0:
            return 0.0
        # Intersection of patterns
        common_patterns = counts_ref.keys() & counts_test.keys()
        # Sum reference counts for these patterns
        coverage_sum = sum(counts_ref[p] for p in common_patterns)
        coverage_value = coverage_sum / total_ref_count

    else:
        #
        # Unweighted coverage:
        # (# of patterns in both) / (# of patterns in ref)
        #
        num_ref_patterns = len(counts_ref)
        if num_ref_patterns == 0:
            return 0.0
        common_patterns = counts_ref.keys() & counts_test.keys()
        coverage_value = len(common_patterns) / num_ref_patterns

    return coverage_value

def calculate_nn_preservation(
    X_high_dim: np.ndarray,
    X_low_dim: np.ndarray,
    k_neighbors: Union[int, List[int]],
    high_dim_indexes: np.ndarray = None,
    high_dim_metric: str = 'euclidean'
) -> Union[float, List[float]]:
    """
    Calculate the nearest neighbor preservation scores for different k values.

    Args:
        X_high_dim (np.ndarray): High-dimensional data of shape (n_samples, n_features_high).
        X_low_dim (np.ndarray): Low-dimensional data of shape (n_samples, n_features_low).
        k_neighbors (int or List[int]): Single k value or list of k values.
        high_dim_indexes (np.ndarray, optional): Precomputed high-dimensional neighbor indices.
        high_dim_metric (str, optional): Metric to use in low-dimensional space

    Returns:
        float or List[float]: Preservation score(s) as a percentage.
    """
    # Ensure k_neighbors is a list
    if isinstance(k_neighbors, int):
        k_list = [k_neighbors]
        single_k = True
    else:
        k_list = k_neighbors
        single_k = False

    nn_preservation_scores = []

    # Precompute high-dimensional nearest neighbors if not provided
    if high_dim_indexes is None:
        max_k = max(k_list)
        nbrs_high = NearestNeighbors(n_neighbors=max_k + 1, metric=high_dim_metric).fit(X_high_dim)
        _, indices_high = nbrs_high.kneighbors(X_high_dim)
        indices_high = indices_high[:, 1:]  # Exclude self
    else:
        indices_high = high_dim_indexes
        max_k = indices_high.shape[1]

    # Precompute nearest neighbors in low-dimensional space
    nbrs_low = NearestNeighbors(n_neighbors=max_k + 1).fit(X_low_dim)
    _, indices_low = nbrs_low.kneighbors(X_low_dim)
    indices_low = indices_low[:, 1:]  # Exclude self

    for k in k_list:
        indices_high_k = indices_high[:, :k]  # shape (n_samples, k)
        indices_low_k = indices_low[:, :k]    # shape (n_samples, k)

        # Vectorized computation
        combined_indices = np.concatenate((indices_high_k, indices_low_k), axis=1)
        sorted_indices = np.sort(combined_indices, axis=1)
        diffs = np.diff(sorted_indices, axis=1)
        overlaps_per_sample = np.sum(diffs == 0, axis=1)
        overlap_counts = overlaps_per_sample / k
        avg_preservation = np.mean(overlap_counts) * 100
        nn_preservation_scores.append(avg_preservation)

    if single_k:
        return nn_preservation_scores[0]
    else:
        return nn_preservation_scores
    
def shannon_entropy(responsibilities: np.ndarray) -> float:
    """
    Compute the Shannon entropy (in percent) of a GTM landscape.
    """
    cumR = responsibilities.sum(axis=0)
    p = cumR / cumR.sum()
    nonzero = p > 0
    H = -np.sum(p[nonzero] * np.log(p[nonzero]))
    K = responsibilities.shape[1]
    E = (H / np.log(K)) * 100
    return E