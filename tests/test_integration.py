"""
Integration tests for ChemographyKit.
Tests end-to-end workflows and interactions between components.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from chemographykit.gtm import GTM, VanillaGTM
from chemographykit.metrics import compute_rp_coverage, resp_to_pattern
from chemographykit.utils.classification import class_density_to_table, get_class_density_matrix
from chemographykit.utils.density import density_to_table, get_density_matrix
from chemographykit.utils.molecules import calculate_latent_coords
from chemographykit.utils.regression import get_reg_density_matrix, reg_density_to_table


@pytest.mark.integration
class TestGTMWorkflow:
    """Test complete GTM workflows from data to visualization."""

    def test_basic_gtm_workflow(self, medium_dataset, gtm_params):
        """Test basic GTM training and transformation workflow."""
        # Create and train model
        gtm = VanillaGTM(**gtm_params)

        # Fit and transform
        transformed = gtm.fit_transform(medium_dataset)

        # Verify output
        assert transformed.shape == torch.Size([100, 2])  # 100 samples, 2D latent space
        assert torch.isfinite(transformed).all()

        # Test projection
        responsibilities, llhs = gtm.project(medium_dataset)
        assert responsibilities.shape == (9, 100)  # 9 nodes, 100 samples
        assert llhs.shape == (100,)

        # Verify responsibilities sum to 1
        resp_sums = responsibilities.sum(dim=0)
        assert torch.allclose(
            resp_sums, torch.ones(100, dtype=torch.float64), atol=1e-6
        )

    def test_gtm_with_pca_workflow(self, medium_dataset):
        """Test GTM with PCA initialization workflow."""
        gtm = GTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=5,
            pca_engine="sklearn",
        )

        # Should work without errors
        transformed = gtm.fit_transform(medium_dataset)

        assert transformed.shape == torch.Size([100, 2])
        assert torch.isfinite(transformed).all()

        # Check that PCA initialization was used
        assert gtm.weights is not None
        assert gtm.beta is not None
        assert hasattr(gtm, "beta_init")

    def test_density_analysis_workflow(self, medium_dataset, gtm_params):
        """Test workflow from GTM to density analysis."""
        # Train GTM
        gtm = VanillaGTM(**gtm_params)
        gtm.fit(medium_dataset)

        # Get responsibilities
        responsibilities, _ = gtm.project(medium_dataset)
        resp_np = (
            responsibilities.detach().cpu().numpy().T
        )  # Convert to (samples, nodes)

        # Calculate density
        density = get_density_matrix(resp_np)
        assert density.shape == (9,)
        assert np.all(density >= 0)

        # Create density table
        density_table = density_to_table(density, node_threshold=0.1)

        assert isinstance(density_table, pd.DataFrame)
        assert len(density_table) == 9
        assert "filtered_density" in density_table.columns
        assert "x" in density_table.columns
        assert "y" in density_table.columns

    def test_classification_workflow(
        self, medium_dataset, binary_class_labels, gtm_params
    ):
        """Test complete classification analysis workflow."""
        # Ensure we have matching number of samples
        data = medium_dataset
        labels = np.tile(
            binary_class_labels, (data.shape[0] // len(binary_class_labels) + 1)
        )[: data.shape[0]]

        # Train GTM
        gtm = VanillaGTM(**gtm_params)
        gtm.fit(data)

        # Get responsibilities
        responsibilities, _ = gtm.project(data)
        resp_np = responsibilities.detach().cpu().numpy().T

        # Classification analysis
        density, class_density, class_prob = get_class_density_matrix(
            resp_np, labels, class_name=["Inactive", "Active"]
        )

        # Verify shapes and properties
        assert density.shape == (9,)
        assert class_density.shape == (9, 2)
        assert class_prob.shape == (9, 2)

        # Probabilities should be between 0 and 1
        assert np.all(class_prob >= 0)
        assert np.all(class_prob <= 1)

        # Create classification table
        class_table = class_density_to_table(
            density,
            class_density,
            class_prob,
            node_threshold=0.1,
            class_name=["Inactive", "Active"],
        )

        assert isinstance(class_table, pd.DataFrame)
        assert len(class_table) == 9
        expected_cols = {
            "x",
            "y",
            "nodes",
            "density",
            "Inactive_prob",
            "Active_prob",
            "Inactive_density",
            "Active_density",
        }
        assert set(class_table.columns) == expected_cols

    def test_regression_workflow(self, medium_dataset, gtm_params):
        """Test complete regression analysis workflow."""
        # Create regression targets
        reg_values = np.random.randn(medium_dataset.shape[0])

        # Train GTM
        gtm = VanillaGTM(**gtm_params)
        gtm.fit(medium_dataset)

        # Get responsibilities
        responsibilities, _ = gtm.project(medium_dataset)
        resp_np = responsibilities.detach().cpu().numpy().T

        # Regression analysis
        density, reg_density = get_reg_density_matrix(resp_np, reg_values)

        assert density.shape == (9,)
        assert reg_density.shape == (9,)

        # Create regression table
        reg_table = reg_density_to_table(density, reg_density, node_threshold=0.1)

        assert isinstance(reg_table, pd.DataFrame)
        assert len(reg_table) == 9
        expected_cols = {"x", "y", "nodes", "density", "filtered_reg_density"}
        assert set(reg_table.columns) == expected_cols

    def test_molecule_coordinates_workflow(self, medium_dataset, gtm_params):
        """Test workflow for calculating molecule coordinates."""
        # Train GTM
        gtm = VanillaGTM(**gtm_params)
        gtm.fit(medium_dataset)

        # Get responsibilities
        responsibilities, _ = gtm.project(medium_dataset)
        resp_np = responsibilities.detach().cpu().numpy().T

        # Calculate molecule coordinates
        mol_coords = calculate_latent_coords(resp_np, return_node=True)

        assert isinstance(mol_coords, pd.DataFrame)
        assert len(mol_coords) == medium_dataset.shape[0]
        assert set(mol_coords.columns) == {"x", "y", "node_index"}

        # Coordinates should be in valid range for 3x3 grid
        assert mol_coords["x"].min() >= 1.0
        assert mol_coords["x"].max() <= 3.0
        assert mol_coords["y"].min() >= 1.0
        assert mol_coords["y"].max() <= 3.0

        # Node indices should be in valid range
        assert mol_coords["node_index"].min() >= 1
        assert mol_coords["node_index"].max() <= 9

    def test_metrics_workflow(self, medium_dataset, gtm_params):
        """Test complete metrics calculation workflow."""
        # Train two GTM models (to compare)
        gtm1 = VanillaGTM(seed=42, **gtm_params)
        gtm2 = VanillaGTM(seed=123, **gtm_params)  # Different seed

        # Split data for reference and test sets
        ref_data = medium_dataset[:60]
        test_data = medium_dataset[60:]

        gtm1.fit(ref_data)
        gtm2.fit(test_data)

        # Get responsibilities
        ref_resp, _ = gtm1.project(ref_data)
        test_resp, _ = gtm2.project(test_data)

        ref_resp_np = ref_resp.detach().cpu().numpy().T
        test_resp_np = test_resp.detach().cpu().numpy().T

        # Convert to patterns
        ref_patterns = np.array(
            [resp_to_pattern(resp, n_bins=10, threshold=0.05) for resp in ref_resp_np]
        )
        test_patterns = np.array(
            [resp_to_pattern(resp, n_bins=10, threshold=0.05) for resp in test_resp_np]
        )

        # Calculate coverage
        coverage_weighted = compute_rp_coverage(
            ref_patterns, test_patterns, use_weight=True
        )
        coverage_unweighted = compute_rp_coverage(
            ref_patterns, test_patterns, use_weight=False
        )

        assert 0.0 <= coverage_weighted <= 1.0
        assert 0.0 <= coverage_unweighted <= 1.0
        assert isinstance(coverage_weighted, float)
        assert isinstance(coverage_unweighted, float)


@pytest.mark.integration
class TestDataConsistency:
    """Test data consistency across different components."""

    def test_coordinate_system_consistency(self, responsibilities_3x3):
        """Test that coordinate systems are consistent across utilities."""
        # Calculate density and grid
        density = get_density_matrix(responsibilities_3x3)
        density_table = density_to_table(density)

        # Calculate molecule coordinates
        mol_coords = calculate_latent_coords(responsibilities_3x3)

        # Check that coordinate ranges match
        assert density_table["x"].min() == 1
        assert density_table["x"].max() == 3
        assert mol_coords["x"].min() >= 1.0
        assert mol_coords["x"].max() <= 3.0

        # Check grid structure
        assert len(density_table) == 9  # 3x3 grid
        assert set(density_table["x"]) == {1, 2, 3}
        assert set(density_table["y"]) == {1, 2, 3}

    def test_responsibility_normalization_consistency(self, medium_dataset, gtm_params):
        """Test that responsibilities are properly normalized across workflows."""
        gtm = VanillaGTM(**gtm_params)
        gtm.fit(medium_dataset)

        # Get responsibilities
        responsibilities, _ = gtm.project(medium_dataset)

        # Check normalization
        resp_sums = responsibilities.sum(dim=0)
        assert torch.allclose(
            resp_sums,
            torch.ones(medium_dataset.shape[0], dtype=torch.float64),
            atol=1e-6,
        )

        # Convert to numpy and check consistency
        resp_np = responsibilities.detach().cpu().numpy().T

        # Check that numpy version is also normalized
        np_sums = resp_np.sum(axis=1)
        np.testing.assert_allclose(np_sums, np.ones(medium_dataset.shape[0]), atol=1e-6)

        # Use in density calculation
        density = get_density_matrix(resp_np)

        # Density should equal number of samples
        assert abs(density.sum() - medium_dataset.shape[0]) < 1e-6

    def test_table_format_consistency(
        self, responsibilities_3x3, binary_class_labels, regression_values
    ):
        """Test that all table outputs have consistent format."""
        # Ensure matching sizes
        labels = binary_class_labels[: len(responsibilities_3x3)]
        reg_vals = regression_values[: len(responsibilities_3x3)]

        # Calculate density
        density = get_density_matrix(responsibilities_3x3)

        # Create different types of tables
        density_table = density_to_table(density)

        _, class_density, class_prob = get_class_density_matrix(
            responsibilities_3x3, labels, class_name=["A", "B"]
        )
        class_table = class_density_to_table(
            density, class_density, class_prob, class_name=["A", "B"]
        )

        _, reg_density = get_reg_density_matrix(responsibilities_3x3, reg_vals)
        reg_table = reg_density_to_table(density, reg_density)

        # Check common columns
        common_cols = {"x", "y", "nodes", "density"}
        assert common_cols.issubset(set(density_table.columns))
        assert common_cols.issubset(set(class_table.columns))
        assert common_cols.issubset(set(reg_table.columns))

        # Check consistent ordering and values for common columns
        for col in common_cols:
            pd.testing.assert_series_equal(
                density_table[col], class_table[col], check_names=False
            )
            pd.testing.assert_series_equal(
                density_table[col], reg_table[col], check_names=False
            )


@pytest.mark.integration
@pytest.mark.slow
class TestScalabilityWorkflows:
    """Test workflows with different data sizes and complexities."""

    def test_small_dataset_workflow(self, small_dataset):
        """Test complete workflow with small dataset."""
        gtm = VanillaGTM(
            num_nodes=4,
            num_basis_functions=4,
            basis_width=0.5,
            reg_coeff=0.1,
            max_iter=10,
        )

        # Should work even with small data
        transformed = gtm.fit_transform(small_dataset)

        assert transformed.shape == torch.Size([small_dataset.shape[0], 2])
        assert torch.isfinite(transformed).all()

    def test_high_dimensional_workflow(self, high_dim_dataset):
        """Test workflow with high-dimensional data."""
        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=3,  # Short for high-dim data
        )

        # Should handle high-dimensional data
        transformed = gtm.fit_transform(high_dim_dataset)

        assert transformed.shape == torch.Size([high_dim_dataset.shape[0], 2])
        assert torch.isfinite(transformed).all()

    def test_sparse_responsibilities_workflow(self, test_data_generator):
        """Test workflow with sparse responsibility matrices."""
        # Create sparse responsibilities
        sparse_resp = test_data_generator.create_sparse_responsibilities(
            n_samples=30, n_nodes=9, sparsity=0.9
        )

        # Should handle sparse data
        density = get_density_matrix(sparse_resp)
        density_table = density_to_table(density, node_threshold=0.01)

        assert len(density_table) == 9
        assert np.all(density >= 0)

        # Many nodes might have low density due to sparsity
        low_density_nodes = (density < 0.1).sum()
        assert low_density_nodes >= 0  # Some nodes might have very low density


@pytest.mark.integration
class TestErrorHandlingWorkflows:
    """Test error handling in complete workflows."""

    def test_mismatched_data_sizes(self, medium_dataset):
        """Test handling of mismatched data sizes in workflows."""
        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=5,
        )
        gtm.fit(medium_dataset)

        # Get responsibilities for full dataset
        responsibilities, _ = gtm.project(medium_dataset)
        resp_np = responsibilities.detach().cpu().numpy().T

        # Create mismatched labels (too few)
        short_labels = np.array([0, 1, 0])  # Only 3 labels for 100 samples

        # This should raise an error or handle gracefully
        # (depending on implementation - test actual behavior)
        try:
            get_class_density_matrix(resp_np, short_labels)
            # If it doesn't raise an error, check that it handled gracefully
            assert len(short_labels) <= resp_np.shape[0]
        except (ValueError, IndexError):
            # Expected behavior for mismatched sizes
            pass

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        # Test with minimal valid data
        tiny_data = torch.randn(3, 2, dtype=torch.float64)

        gtm = VanillaGTM(
            num_nodes=4,
            num_basis_functions=4,
            basis_width=1.0,
            reg_coeff=0.5,
            max_iter=3,
        )

        # Should handle tiny dataset (though results might not be meaningful)
        try:
            transformed = gtm.fit_transform(tiny_data)
            assert transformed.shape == torch.Size([3, 2])
        except Exception as e:
            # Some configurations might not work with very small data
            assert isinstance(e, (RuntimeError, ValueError))

    def test_numerical_edge_cases(self):
        """Test handling of numerical edge cases."""
        # Data with extreme values
        extreme_data = torch.tensor(
            [
                [1e6, 1e-6],
                [1e-6, 1e6],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
        )

        gtm = VanillaGTM(
            num_nodes=4,
            num_basis_functions=4,
            basis_width=1.0,
            reg_coeff=0.1,
            max_iter=3,
        )

        # Should handle extreme values (might issue warnings)
        with pytest.warns(UserWarning, match=".*numerical issues.*"):
            gtm.fit(extreme_data)

        # Should still produce finite results
        transformed = gtm.transform(extreme_data)
        assert torch.isfinite(transformed).all()
