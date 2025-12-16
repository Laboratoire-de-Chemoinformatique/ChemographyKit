"""
Tests for metrics functionality in ChemographyKit.
Tests responsibility pattern fingerprints and coverage calculations.
"""

import numpy as np
import pytest

from chemographykit.metrics import compute_rp_coverage, get_fingerprint_counts, resp_to_pattern


class TestRespToPattern:
    """Test responsibility pattern fingerprint generation."""

    def test_resp_to_pattern_basic(self):
        """Test basic responsibility to pattern conversion."""
        responsibilities = np.array([0.0, 0.5, 1.0])

        result = resp_to_pattern(responsibilities, n_bins=10, threshold=0.01)

        # Expected: floor(10 * [0.0, 0.5, 1.0] + 0.9) = [0, 5, 10]
        # But values below threshold (0.01) become 0
        expected = np.array([0, 5, 10])
        np.testing.assert_array_equal(result, expected)

    def test_resp_to_pattern_threshold(self):
        """Test threshold filtering in pattern conversion."""
        responsibilities = np.array([0.005, 0.02, 0.1, 0.5])

        result = resp_to_pattern(responsibilities, n_bins=10, threshold=0.01)

        # 0.005 < 0.01 should become 0
        # Others: floor(10 * [0.02, 0.1, 0.5] + 0.9) = [1, 1, 5]
        expected = np.array([0, 1, 1, 5])
        np.testing.assert_array_equal(result, expected)

    def test_resp_to_pattern_different_bins(self):
        """Test pattern conversion with different number of bins."""
        responsibilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        result_5 = resp_to_pattern(responsibilities, n_bins=5, threshold=0.0)
        result_10 = resp_to_pattern(responsibilities, n_bins=10, threshold=0.0)

        # With 5 bins: floor(5 * values + 0.9)
        expected_5 = np.array([0, 2, 3, 4, 5])
        np.testing.assert_array_equal(result_5, expected_5)

        # With 10 bins: floor(10 * values + 0.9)
        expected_10 = np.array([0, 3, 5, 8, 10])
        np.testing.assert_array_equal(result_10, expected_10)

    def test_resp_to_pattern_edge_values(self):
        """Test pattern conversion with edge values."""
        responsibilities = np.array([0.0, 0.1, 0.9, 1.0])

        result = resp_to_pattern(responsibilities, n_bins=10, threshold=0.01)

        # floor(10 * [0.0, 0.1, 0.9, 1.0] + 0.9) = [0, 1, 9, 10]
        expected = np.array([0, 1, 9, 10])
        np.testing.assert_array_equal(result, expected)

    def test_resp_to_pattern_invalid_bins(self):
        """Test error handling for invalid number of bins."""
        responsibilities = np.array([0.5])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            resp_to_pattern(responsibilities, n_bins=0)

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            resp_to_pattern(responsibilities, n_bins=-1)

    def test_resp_to_pattern_bounds_clipping(self):
        """Test that values are clipped to valid bounds."""
        # Test with values that might exceed bounds due to floating point precision
        responsibilities = np.array([0.0, 1.0000001])  # Slightly over 1.0

        result = resp_to_pattern(responsibilities, n_bins=10, threshold=0.0)

        # Should be clipped to valid range [0, n_bins]
        assert np.all(result >= 0)
        assert np.all(result <= 10)

    def test_resp_to_pattern_array_types(self):
        """Test pattern conversion with different array types."""
        # List input
        resp_list = [0.0, 0.5, 1.0]
        result_list = resp_to_pattern(resp_list, n_bins=10)

        # Numpy array input
        resp_array = np.array([0.0, 0.5, 1.0])
        result_array = resp_to_pattern(resp_array, n_bins=10)

        # Should give same results
        np.testing.assert_array_equal(result_list, result_array)

        # Result should always be numpy array
        assert isinstance(result_list, np.ndarray)
        assert isinstance(result_array, np.ndarray)


class TestGetFingerprintCounts:
    """Test fingerprint counting functionality."""

    def test_get_fingerprint_counts_basic(self):
        """Test basic fingerprint counting."""
        fingerprints = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],  # Duplicate
                [4, 5, 6],
                [1, 2, 3],  # Another duplicate
            ]
        )

        result = get_fingerprint_counts(fingerprints)

        expected = {
            (1, 2, 3): 3,
            (4, 5, 6): 1,
        }
        assert result == expected

    def test_get_fingerprint_counts_unique(self):
        """Test fingerprint counting with all unique patterns."""
        fingerprints = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        result = get_fingerprint_counts(fingerprints)

        expected = {
            (1, 0, 0): 1,
            (0, 1, 0): 1,
            (0, 0, 1): 1,
        }
        assert result == expected

    def test_get_fingerprint_counts_single_pattern(self):
        """Test fingerprint counting with single repeated pattern."""
        fingerprints = np.array(
            [
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5],
            ]
        )

        result = get_fingerprint_counts(fingerprints)

        expected = {(5, 5, 5): 3}
        assert result == expected

    def test_get_fingerprint_counts_empty(self):
        """Test fingerprint counting with empty array."""
        fingerprints = np.array([]).reshape(0, 3)  # Empty array with 3 columns

        result = get_fingerprint_counts(fingerprints)

        assert result == {}

    def test_get_fingerprint_counts_single_row(self):
        """Test fingerprint counting with single fingerprint."""
        fingerprints = np.array([[1, 2, 3, 4]])

        result = get_fingerprint_counts(fingerprints)

        expected = {(1, 2, 3, 4): 1}
        assert result == expected


class TestComputeRpCoverage:
    """Test responsibility pattern coverage calculation."""

    def test_compute_rp_coverage_weighted_basic(self):
        """Test basic weighted coverage calculation."""
        # Reference library has patterns with counts
        ref_lib = np.array(
            [
                [1, 2, 3],  # Pattern A
                [1, 2, 3],  # Pattern A (appears twice)
                [4, 5, 6],  # Pattern B
                [7, 8, 9],  # Pattern C
            ]
        )

        # Test library has some overlapping patterns
        test_lib = np.array(
            [
                [1, 2, 3],  # Pattern A (in ref)
                [4, 5, 6],  # Pattern B (in ref)
                [0, 0, 0],  # Pattern D (not in ref)
            ]
        )

        coverage = compute_rp_coverage(ref_lib, test_lib, use_weight=True)

        # Weighted coverage: (count_A + count_B) / total_ref_count = (2 + 1) / 4 = 0.75
        expected = 0.75
        assert abs(coverage - expected) < 1e-10

    def test_compute_rp_coverage_unweighted_basic(self):
        """Test basic unweighted coverage calculation."""
        ref_lib = np.array(
            [
                [1, 2, 3],  # Pattern A
                [1, 2, 3],  # Pattern A (duplicate)
                [4, 5, 6],  # Pattern B
                [7, 8, 9],  # Pattern C
            ]
        )

        test_lib = np.array(
            [
                [1, 2, 3],  # Pattern A (in ref)
                [4, 5, 6],  # Pattern B (in ref)
                [0, 0, 0],  # Pattern D (not in ref)
            ]
        )

        coverage = compute_rp_coverage(ref_lib, test_lib, use_weight=False)

        # Unweighted coverage: num_common_patterns / num_ref_patterns = 2 / 3 ≈ 0.667
        expected = 2.0 / 3.0
        assert abs(coverage - expected) < 1e-10

    def test_compute_rp_coverage_no_overlap(self):
        """Test coverage with no overlapping patterns."""
        ref_lib = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

        test_lib = np.array(
            [
                [7, 8, 9],
                [0, 0, 0],
            ]
        )

        coverage_weighted = compute_rp_coverage(ref_lib, test_lib, use_weight=True)
        coverage_unweighted = compute_rp_coverage(ref_lib, test_lib, use_weight=False)

        assert coverage_weighted == 0.0
        assert coverage_unweighted == 0.0

    def test_compute_rp_coverage_complete_overlap(self):
        """Test coverage with complete overlap."""
        ref_lib = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],  # Duplicate
            ]
        )

        test_lib = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],  # Extra occurrences don't matter for coverage
                [4, 5, 6],
            ]
        )

        coverage_weighted = compute_rp_coverage(ref_lib, test_lib, use_weight=True)
        coverage_unweighted = compute_rp_coverage(ref_lib, test_lib, use_weight=False)

        assert coverage_weighted == 1.0
        assert coverage_unweighted == 1.0

    def test_compute_rp_coverage_empty_ref(self):
        """Test coverage with empty reference library."""
        ref_lib = np.array([]).reshape(0, 3)
        test_lib = np.array([[1, 2, 3]])

        coverage_weighted = compute_rp_coverage(ref_lib, test_lib, use_weight=True)
        coverage_unweighted = compute_rp_coverage(ref_lib, test_lib, use_weight=False)

        assert coverage_weighted == 0.0
        assert coverage_unweighted == 0.0

    def test_compute_rp_coverage_empty_test(self):
        """Test coverage with empty test library."""
        ref_lib = np.array([[1, 2, 3], [4, 5, 6]])
        test_lib = np.array([]).reshape(0, 3)

        coverage_weighted = compute_rp_coverage(ref_lib, test_lib, use_weight=True)
        coverage_unweighted = compute_rp_coverage(ref_lib, test_lib, use_weight=False)

        assert coverage_weighted == 0.0
        assert coverage_unweighted == 0.0

    def test_compute_rp_coverage_identical_libraries(self):
        """Test coverage with identical libraries."""
        lib = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],  # Duplicate
            ]
        )

        coverage_weighted = compute_rp_coverage(lib, lib, use_weight=True)
        coverage_unweighted = compute_rp_coverage(lib, lib, use_weight=False)

        assert coverage_weighted == 1.0
        assert coverage_unweighted == 1.0


class TestMetricsIntegration:
    """Integration tests combining multiple metrics functions."""

    def test_full_pipeline(self):
        """Test complete pipeline from responsibilities to coverage."""
        # Create synthetic responsibility data
        np.random.seed(42)
        ref_responsibilities = np.random.rand(20, 5)  # 20 samples, 5 nodes
        test_responsibilities = np.random.rand(15, 5)  # 15 samples, 5 nodes

        # Convert to patterns
        ref_patterns = np.array(
            [
                resp_to_pattern(resp, n_bins=10, threshold=0.05)
                for resp in ref_responsibilities
            ]
        )
        test_patterns = np.array(
            [
                resp_to_pattern(resp, n_bins=10, threshold=0.05)
                for resp in test_responsibilities
            ]
        )

        # Calculate coverage
        coverage_weighted = compute_rp_coverage(
            ref_patterns, test_patterns, use_weight=True
        )
        coverage_unweighted = compute_rp_coverage(
            ref_patterns, test_patterns, use_weight=False
        )

        # Basic sanity checks
        assert 0.0 <= coverage_weighted <= 1.0
        assert 0.0 <= coverage_unweighted <= 1.0
        assert isinstance(coverage_weighted, float)
        assert isinstance(coverage_unweighted, float)

    def test_pattern_consistency(self):
        """Test that pattern generation is consistent."""
        responsibilities = np.array([0.1, 0.3, 0.6])

        # Generate patterns multiple times with same parameters
        pattern1 = resp_to_pattern(responsibilities, n_bins=10, threshold=0.05)
        pattern2 = resp_to_pattern(responsibilities, n_bins=10, threshold=0.05)

        # Should be identical
        np.testing.assert_array_equal(pattern1, pattern2)

    def test_coverage_symmetry_properties(self):
        """Test mathematical properties of coverage calculation."""
        # Create test data
        ref_lib = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],  # Duplicate
            ]
        )

        # Coverage of ref with itself should be 1.0
        self_coverage_w = compute_rp_coverage(ref_lib, ref_lib, use_weight=True)
        self_coverage_u = compute_rp_coverage(ref_lib, ref_lib, use_weight=False)

        assert abs(self_coverage_w - 1.0) < 1e-10
        assert abs(self_coverage_u - 1.0) < 1e-10

        # Coverage should be between 0 and 1
        test_lib = np.array([[1, 2, 3], [7, 8, 9]])
        coverage_w = compute_rp_coverage(ref_lib, test_lib, use_weight=True)
        coverage_u = compute_rp_coverage(ref_lib, test_lib, use_weight=False)

        assert 0.0 <= coverage_w <= 1.0
        assert 0.0 <= coverage_u <= 1.0

    def test_different_thresholds_effect(self):
        """Test how different thresholds affect pattern generation and coverage."""
        responsibilities = np.array(
            [
                [0.01, 0.02, 0.97],  # Clear pattern
                [0.005, 0.015, 0.98],  # Similar but with smaller values
            ]
        )

        # High threshold - small values become 0
        patterns_high_thresh = np.array(
            [
                resp_to_pattern(resp, n_bins=10, threshold=0.015)
                for resp in responsibilities
            ]
        )

        # Low threshold - more values preserved
        patterns_low_thresh = np.array(
            [
                resp_to_pattern(resp, n_bins=10, threshold=0.001)
                for resp in responsibilities
            ]
        )

        # Patterns should be different due to threshold effect
        assert not np.array_equal(patterns_high_thresh, patterns_low_thresh)

        # Calculate self-coverage (should be 1.0 for both)
        coverage_high = compute_rp_coverage(patterns_high_thresh, patterns_high_thresh)
        coverage_low = compute_rp_coverage(patterns_low_thresh, patterns_low_thresh)

        assert abs(coverage_high - 1.0) < 1e-10
        assert abs(coverage_low - 1.0) < 1e-10
