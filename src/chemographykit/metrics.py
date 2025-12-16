from typing import List, Tuple, Union

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
