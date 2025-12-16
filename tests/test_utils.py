"""
Tests for utility functions in ChemographyKit.
Covers density, classification, regression, and molecule utilities.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from chemographykit.utils.classification import (
    class_density_to_table,
    class_prob_from_density,
    get_class_density_matrix,
    get_class_inds,
)
from chemographykit.utils.density import (
    calculate_grid,
    density_to_table,
    filter_by_threshold,
    get_density_matrix,
)
from chemographykit.utils.molecules import calculate_latent_coords
from chemographykit.utils.regression import (
    get_reg_density_matrix,
    norm_reg_density,
    reg_density_to_table,
)


class TestDensityUtils:
    """Test density calculation utilities."""

    def test_calculate_grid_basic(self):
        """Test basic grid calculation."""
        density = np.array([0.1, 0.2, 0.3, 0.4])  # 2x2 grid
        result = calculate_grid(density)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert set(result.columns) == {"x", "y", "nodes", "density"}
        assert result["density"].tolist() == [0.1, 0.2, 0.3, 0.4]

        # Check grid coordinates
        assert set(result["x"]) == {1, 2}
        assert set(result["y"]) == {1, 2}

    def test_calculate_grid_9_nodes(self):
        """Test grid calculation with 9 nodes (3x3)."""
        density = np.random.rand(9)
        result = calculate_grid(density)

        assert len(result) == 9
        assert set(result["x"]) == {1, 2, 3}
        assert set(result["y"]) == {1, 2, 3}
        assert len(result["nodes"].unique()) == 9

    def test_calculate_grid_legacy_mode(self):
        """Test grid calculation with legacy mode."""
        density = np.array([0.1, 0.2, 0.3, 0.4])
        result_normal = calculate_grid(density, legacy=False)
        result_legacy = calculate_grid(density, legacy=True)

        # Both should have same shape but potentially different node ordering
        assert len(result_normal) == len(result_legacy)
        assert set(result_normal.columns) == set(result_legacy.columns)

    def test_get_density_matrix(self):
        """Test density matrix calculation from responsibilities."""
        responsibilities = np.array(
            [
                [0.8, 0.1, 0.1],  # Sample 1
                [0.2, 0.7, 0.1],  # Sample 2
                [0.1, 0.1, 0.8],  # Sample 3
            ]
        )

        density = get_density_matrix(responsibilities)

        expected = np.array([1.1, 0.9, 1.0])  # Sum along axis 0
        np.testing.assert_array_almost_equal(density, expected)

    def test_filter_by_threshold(self):
        """Test threshold filtering."""
        modified = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.array([0.5, 1.5, 0.8, 2.0])
        threshold = 1.0

        result = filter_by_threshold(modified, reference, threshold)

        expected = np.array([np.nan, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_density_to_table(self):
        """Test conversion of density to table format."""
        density = np.array([0.5, 1.2, 0.3, 2.1])  # 2x2 grid

        result = density_to_table(density, node_threshold=1.0)

        assert isinstance(result, pd.DataFrame)
        assert "filtered_density" in result.columns

        # Check that values below threshold are NaN
        filtered_vals = result["filtered_density"].values
        assert np.isnan(filtered_vals[0])  # 0.5 < 1.0
        assert filtered_vals[1] == 1.2
        assert np.isnan(filtered_vals[2])  # 0.3 < 1.0
        assert filtered_vals[3] == 2.1


class TestClassificationUtils:
    """Test classification utility functions."""

    def test_class_prob_from_density_basic(self):
        """Test basic class probability calculation."""
        class_density = np.array(
            [
                [0.8, 0.2],  # Node 1: 80% class 0, 20% class 1
                [0.3, 0.7],  # Node 2: 30% class 0, 70% class 1
                [0.5, 0.5],  # Node 3: 50% class 0, 50% class 1
            ]
        )

        result = class_prob_from_density(class_density)

        expected = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.5, 0.5],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_class_prob_with_counts(self):
        """Test class probability calculation with class counts."""
        class_density = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )
        class_counts = np.array([100, 50])  # Imbalanced classes

        result = class_prob_from_density(class_density, class_counts)

        # Should apply normalization based on class imbalance
        assert result.shape == class_density.shape
        # Each row should still sum to approximately 1
        row_sums = result.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_class_prob_zero_density(self):
        """Test class probability with zero density nodes."""
        class_density = np.array(
            [
                [0.0, 0.0],  # Zero density node
                [0.3, 0.7],
            ]
        )

        result = class_prob_from_density(class_density)

        # First row should be [0, 0] due to zero division handling
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [0.3, 0.7])

    def test_get_class_inds(self):
        """Test conversion of class labels to indices."""
        class_labels = ["A", "B", "A", "C", "B"]
        classes = ["A", "B", "C"]

        result = get_class_inds(class_labels, classes)

        expected = np.array([0, 1, 0, 2, 1])
        np.testing.assert_array_equal(result, expected)

    def test_get_class_density_matrix(self):
        """Test class density matrix calculation."""
        responsibilities = np.array(
            [
                [0.8, 0.1, 0.1],  # Sample 1
                [0.2, 0.7, 0.1],  # Sample 2
                [0.1, 0.2, 0.7],  # Sample 3
            ]
        )
        class_labels = np.array([0, 1, 0])  # First and third samples are class 0

        density, class_density, class_prob = get_class_density_matrix(
            responsibilities, class_labels, class_name=["0", "1"]
        )

        # Check shapes
        assert density.shape == (3,)  # 3 nodes
        assert class_density.shape == (3, 2)  # 3 nodes, 2 classes
        assert class_prob.shape == (3, 2)

        # Check density calculation
        expected_density = np.array([1.1, 1.0, 0.9])  # Sum of all responsibilities
        np.testing.assert_array_almost_equal(density, expected_density)

        # Check class density (samples 0 and 2 are class 0, sample 1 is class 1)
        expected_class_0 = np.array([0.9, 0.3, 0.8])  # Sum of class 0 samples
        expected_class_1 = np.array([0.2, 0.7, 0.1])  # Sum of class 1 samples
        np.testing.assert_array_almost_equal(class_density[:, 0], expected_class_0)
        np.testing.assert_array_almost_equal(class_density[:, 1], expected_class_1)


class TestRegressionUtils:
    """Test regression utility functions."""

    def test_norm_reg_density(self):
        """Test regression density normalization."""
        reg_density = np.array([1.0, 2.0, 3.0])
        density = np.array([2.0, 4.0, 6.0])

        result = norm_reg_density(reg_density, density)

        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_norm_reg_density_zero_division(self):
        """Test regression density normalization with zero density."""
        reg_density = np.array([1.0, 2.0])
        density = np.array([2.0, 0.0])  # Zero density

        result = norm_reg_density(reg_density, density)

        expected = np.array([0.5, 0.0])  # Safe division should give 0
        np.testing.assert_array_equal(result, expected)

    def test_get_reg_density_matrix(self):
        """Test regression density matrix calculation."""
        responsibilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.3, 0.3, 0.4],
            ]
        )
        reg_values = np.array([1.0, 2.0, 3.0])

        density, reg_density = get_reg_density_matrix(responsibilities, reg_values)

        # Check shapes
        assert density.shape == (3,)
        assert reg_density.shape == (3,)

        # Manual calculation check
        expected_density = responsibilities.sum(axis=0)
        np.testing.assert_array_almost_equal(density, expected_density)

        # Regression density should be normalized
        raw_reg_density = responsibilities.T @ reg_values
        expected_reg_density = raw_reg_density / expected_density
        # Handle division by zero
        expected_reg_density = np.where(expected_density != 0, expected_reg_density, 0)
        np.testing.assert_array_almost_equal(reg_density, expected_reg_density)

    def test_get_reg_density_matrix_list_input(self):
        """Test regression density matrix with list input."""
        responsibilities = np.array([[0.8, 0.2], [0.3, 0.7]])
        reg_values = [1.0, 2.0]  # List instead of array

        density, reg_density = get_reg_density_matrix(responsibilities, reg_values)

        assert density.shape == (2,)
        assert reg_density.shape == (2,)
        assert isinstance(reg_density, np.ndarray)


class TestMoleculeUtils:
    """Test molecule coordinate calculation utilities."""

    def test_calculate_latent_coords_basic(self):
        """Test basic molecule coordinate calculation."""
        responsibilities = np.array(
            [
                [0.8, 0.1, 0.05, 0.05],  # Molecule 1: mostly at node 1
                [0.1, 0.8, 0.05, 0.05],  # Molecule 2: mostly at node 2
                [0.25, 0.25, 0.25, 0.25],  # Molecule 3: evenly distributed
            ]
        )

        result = calculate_latent_coords(responsibilities)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 molecules
        assert set(result.columns) == {"x", "y"}

        # Check that coordinates are reasonable
        assert result["x"].min() >= 1.0
        assert result["x"].max() <= 2.0  # 2x2 grid
        assert result["y"].min() >= 1.0
        assert result["y"].max() <= 2.0

    def test_calculate_latent_coords_with_correction(self):
        """Test molecule coordinates with correction for visualization."""
        responsibilities = np.array([[1.0, 0.0, 0.0, 0.0]])

        result_no_correction = calculate_latent_coords(
            responsibilities, correction=False
        )
        result_with_correction = calculate_latent_coords(
            responsibilities, correction=True
        )

        # With correction, coordinates should be shifted by 0.5
        assert (
            result_with_correction["x"].iloc[0]
            == result_no_correction["x"].iloc[0] + 0.5
        )
        assert (
            result_with_correction["y"].iloc[0]
            == result_no_correction["y"].iloc[0] + 0.5
        )

    def test_calculate_latent_coords_return_node(self):
        """Test molecule coordinates with node index return."""
        responsibilities = np.array(
            [
                [0.9, 0.1, 0.0, 0.0],  # Clearly at node 1 (index 0)
                [0.0, 0.0, 0.0, 1.0],  # Clearly at node 4 (index 3)
            ]
        )

        result = calculate_latent_coords(responsibilities, return_node=True)

        assert "node_index" in result.columns
        assert result["node_index"].iloc[0] == 1  # Node indices are 1-based
        assert result["node_index"].iloc[1] == 4

    def test_calculate_latent_coords_legacy_mode(self):
        """Test molecule coordinates with legacy coordinate system."""
        responsibilities = np.array([[1.0, 0.0, 0.0, 0.0]])

        result_normal = calculate_latent_coords(responsibilities, legacy=False)
        result_legacy = calculate_latent_coords(responsibilities, legacy=True)

        # Both should have same structure but potentially different coordinates
        assert set(result_normal.columns) == set(result_legacy.columns)
        assert len(result_normal) == len(result_legacy)

    def test_calculate_latent_coords_9_nodes(self):
        """Test molecule coordinates with 3x3 grid."""
        responsibilities = np.random.rand(5, 9)  # 5 molecules, 9 nodes
        responsibilities = responsibilities / responsibilities.sum(
            axis=1, keepdims=True
        )

        result = calculate_latent_coords(responsibilities)

        assert len(result) == 5
        # Coordinates should be in range [1, 3] for 3x3 grid
        assert result["x"].min() >= 1.0
        assert result["x"].max() <= 3.0
        assert result["y"].min() >= 1.0
        assert result["y"].max() <= 3.0


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_classification_pipeline(self):
        """Test complete classification pipeline."""
        # Create synthetic data
        responsibilities = np.random.rand(10, 4)  # 10 samples, 4 nodes
        responsibilities = responsibilities / responsibilities.sum(
            axis=1, keepdims=True
        )
        class_labels = np.random.choice([0, 1], size=10)

        # Run full pipeline
        density, class_density, class_prob = get_class_density_matrix(
            responsibilities, class_labels, class_name=["A", "B"]
        )

        table = class_density_to_table(
            density,
            class_density,
            class_prob,
            node_threshold=0.1,
            class_name=["A", "B"],
        )

        # Check final table structure
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 4  # 4 nodes
        expected_cols = {
            "x",
            "y",
            "nodes",
            "density",
            "A_prob",
            "B_prob",
            "A_density",
            "B_density",
        }
        assert set(table.columns) == expected_cols

    def test_regression_pipeline(self):
        """Test complete regression pipeline."""
        # Create synthetic data
        responsibilities = np.random.rand(15, 9)  # 15 samples, 9 nodes
        responsibilities = responsibilities / responsibilities.sum(
            axis=1, keepdims=True
        )
        reg_values = np.random.randn(15)  # Regression targets

        # Run pipeline
        density, reg_density = get_reg_density_matrix(responsibilities, reg_values)
        table = reg_density_to_table(density, reg_density, node_threshold=0.1)

        # Check results
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 9  # 9 nodes
        expected_cols = {"x", "y", "nodes", "density", "filtered_reg_density"}
        assert set(table.columns) == expected_cols

    def test_coordinate_consistency(self):
        """Test that coordinate systems are consistent across utilities."""
        density = np.ones(4)  # 2x2 grid
        responsibilities = np.array(
            [[1.0, 0.0, 0.0, 0.0]]
        )  # Single molecule at first node

        # Get grid from density
        grid_table = calculate_grid(density)

        # Get coordinates from molecule calculation
        mol_coords = calculate_latent_coords(responsibilities)

        # First node should have same coordinates
        first_node_grid = grid_table.iloc[0]
        mol_coord = mol_coords.iloc[0]

        # Should be at same position (allowing for floating point precision)
        assert abs(first_node_grid["x"] - mol_coord["x"]) < 1e-10
        assert abs(first_node_grid["y"] - mol_coord["y"]) < 1e-10
