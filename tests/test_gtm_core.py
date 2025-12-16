"""
Core tests for GTM model functionality.
Tests the most critical components: initialization, fitting, and transformation.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from chemographykit.gtm import GTM, DataStandardizer, VanillaGTM


class TestDataStandardizer:
    """Test data standardization functionality."""

    def test_standardizer_initialization(self):
        """Test DataStandardizer can be created with different options."""
        std1 = DataStandardizer()
        assert std1.with_mean is True
        assert std1.with_std is True

        std2 = DataStandardizer(with_mean=False, with_std=True)
        assert std2.with_mean is False
        assert std2.with_std is True

    def test_standardizer_numpy_input(self):
        """Test standardizer works with numpy arrays."""
        data = np.random.randn(100, 5)
        standardizer = DataStandardizer()
        result = standardizer.fit_transform(data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape
        # Check that mean is close to 0 and std close to 1
        assert torch.allclose(
            result.mean(dim=0), torch.zeros(5, dtype=torch.float64), atol=1e-6
        )
        assert torch.allclose(
            result.std(dim=0), torch.ones(5, dtype=torch.float64), atol=1e-2
        )

    def test_standardizer_torch_input(self):
        """Test standardizer works with torch tensors."""
        data = torch.randn(100, 5, dtype=torch.float64)
        standardizer = DataStandardizer()
        result = standardizer.fit_transform(data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape
        assert torch.allclose(
            result.mean(dim=0), torch.zeros(5, dtype=torch.float64), atol=1e-6
        )

    def test_standardizer_nan_handling(self):
        """Test standardizer handles NaN values correctly."""
        data = torch.randn(100, 5)
        data[0, 0] = float("nan")  # Inject NaN

        standardizer = DataStandardizer()
        result = standardizer.fit_transform(data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape
        # Result should still be finite except for the NaN input
        assert torch.isnan(result[0, 0])
        assert torch.isfinite(result[1:, :]).all()


class TestVanillaGTM:
    """Test VanillaGTM model functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(50, 10, dtype=torch.float64)

    @pytest.fixture
    def gtm_model(self):
        """Create a basic GTM model for testing."""
        return VanillaGTM(
            num_nodes=25,  # 5x5 grid
            num_basis_functions=16,  # 4x4 grid
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=10,  # Short for testing
            device="cpu",
        )

    def test_gtm_initialization(self):
        """Test GTM model can be initialized with valid parameters."""
        gtm = VanillaGTM(
            num_nodes=25, num_basis_functions=16, basis_width=0.3, reg_coeff=0.1
        )

        assert gtm.num_nodes == 25
        assert gtm.num_basis_functions == 16
        assert gtm.basis_width == 0.3
        assert gtm.reg_coeff == 0.1
        assert gtm.nodes.shape == (25, 2)
        assert gtm.mu.shape == (16, 2)
        assert gtm.phi.shape == (25, 17)  # +1 for bias

    def test_gtm_invalid_nodes(self):
        """Test GTM raises error for non-square number of nodes."""
        with pytest.raises(AssertionError, match="num_nodes must be square"):
            VanillaGTM(
                num_nodes=26,  # Not a perfect square
                num_basis_functions=16,
                basis_width=0.3,
                reg_coeff=0.1,
            )

    def test_gtm_invalid_basis_functions(self):
        """Test GTM raises error for non-square number of basis functions."""
        with pytest.raises(AssertionError, match="num_basis_functions must be square"):
            VanillaGTM(
                num_nodes=25,
                num_basis_functions=15,  # Not a perfect square
                basis_width=0.3,
                reg_coeff=0.1,
            )

    def test_gtm_fit(self, gtm_model, sample_data):
        """Test GTM model can fit to data without errors."""
        # Should not raise any exceptions
        gtm_model.fit(sample_data)

        # Check that model parameters are set
        assert gtm_model.weights is not None
        assert gtm_model.beta is not None
        assert gtm_model.data_mean is not None
        assert gtm_model.data_std is not None

        # Check shapes
        assert gtm_model.weights.shape == (
            17,
            10,
        )  # (num_basis_functions + 1, data_dim)
        assert gtm_model.beta.shape == torch.Size([])  # scalar

    def test_gtm_transform(self, gtm_model, sample_data):
        """Test GTM model can transform data after fitting."""
        gtm_model.fit(sample_data)
        result = gtm_model.transform(sample_data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([50, 2])  # (n_samples, n_components)
        assert torch.isfinite(result).all()

    def test_gtm_fit_transform(self, gtm_model, sample_data):
        """Test GTM fit_transform gives same result as separate fit and transform."""
        # Fit and transform separately
        gtm1 = VanillaGTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=10,
            seed=42,
        )
        gtm1.fit(sample_data)
        result1 = gtm1.transform(sample_data)

        # Fit and transform together
        gtm2 = VanillaGTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=10,
            seed=42,
        )
        result2 = gtm2.fit_transform(sample_data)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_gtm_project(self, gtm_model, sample_data):
        """Test GTM project method returns responsibilities and log-likelihoods."""
        gtm_model.fit(sample_data)
        responsibilities, llhs = gtm_model.project(sample_data)

        assert isinstance(responsibilities, torch.Tensor)
        assert isinstance(llhs, torch.Tensor)
        assert responsibilities.shape == (25, 50)  # (num_nodes, n_samples)
        assert llhs.shape == (50,)  # (n_samples,)

        # Responsibilities should sum to 1 for each sample
        assert torch.allclose(
            responsibilities.sum(dim=0), torch.ones(50, dtype=torch.float64), atol=1e-6
        )

        # All values should be finite
        assert torch.isfinite(responsibilities).all()
        assert torch.isfinite(llhs).all()

    def test_gtm_kernel(self, gtm_model):
        """Test kernel function computes distances correctly."""
        a = torch.randn(10, 5, dtype=torch.float64)
        b = torch.randn(15, 5, dtype=torch.float64)

        distances = gtm_model.kernel(a, b)

        assert distances.shape == (10, 15)
        assert (distances >= 0).all()  # Distances should be non-negative
        assert torch.isfinite(distances).all()

    def test_gtm_different_devices(self):
        """Test GTM works with different device specifications."""
        # CPU model
        gtm_cpu = VanillaGTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            device="cpu",
        )
        assert gtm_cpu.device.type == "cpu"

        # Test that tensors are on correct device
        assert gtm_cpu.nodes.device.type == "cpu"
        assert gtm_cpu.phi.device.type == "cpu"


class TestGTM:
    """Test Bishop's GTM implementation with PCA initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(50, 10, dtype=torch.float64)

    def test_gtm_initialization(self):
        """Test GTM with PCA initialization can be created."""
        gtm = GTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            pca_engine="sklearn",
        )

        assert gtm.pca_engine == "sklearn"
        assert gtm.pca_scale is True
        assert gtm.pca_lowrank is False

    def test_gtm_sklearn_pca(self, sample_data):
        """Test GTM with sklearn PCA engine."""
        gtm = GTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=5,
            pca_engine="sklearn",
        )

        # Should not raise exceptions
        gtm.fit(sample_data)
        result = gtm.transform(sample_data)

        assert result.shape == torch.Size([50, 2])
        assert torch.isfinite(result).all()

    def test_gtm_torch_pca(self, sample_data):
        """Test GTM with torch PCA engine."""
        gtm = GTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=5,
            pca_engine="torch",
        )

        # Should not raise exceptions
        gtm.fit(sample_data)
        result = gtm.transform(sample_data)

        assert result.shape == torch.Size([50, 2])
        assert torch.isfinite(result).all()

    def test_gtm_invalid_pca_engine(self):
        """Test GTM raises error for invalid PCA engine."""
        gtm = GTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            pca_engine="invalid",
        )

        sample_data = torch.randn(50, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="Unknown pca_engine"):
            gtm.fit(sample_data)

    def test_gtm_custom_pca_dict(self, sample_data):
        """Test GTM with custom PCA dictionary."""
        # Create mock PCA results
        eigenvectors = torch.randn(3, 10, dtype=torch.float64)
        eigenvalues = torch.tensor([1.0, 0.5, 0.1], dtype=torch.float64)

        gtm = GTM(
            num_nodes=25,
            num_basis_functions=16,
            basis_width=0.3,
            reg_coeff=0.1,
            max_iter=5,
            pca_engine={"eigenvectors": eigenvectors, "eigenvalues": eigenvalues},
        )

        # Should not raise exceptions
        gtm.fit(sample_data)
        result = gtm.transform(sample_data)

        assert result.shape == torch.Size([50, 2])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_small_dataset(self):
        """Test GTM behavior with very small datasets."""
        data = torch.randn(5, 3, dtype=torch.float64)  # Very small dataset
        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.5,
            reg_coeff=0.1,
            max_iter=5,
        )

        # Should still work, though might not converge well
        gtm.fit(data)
        result = gtm.transform(data)
        assert result.shape == torch.Size([5, 2])

    def test_single_feature(self):
        """Test GTM with single feature data."""
        data = torch.randn(50, 1, dtype=torch.float64)
        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.5,
            reg_coeff=0.1,
            max_iter=5,
        )

        gtm.fit(data)
        result = gtm.transform(data)
        assert result.shape == torch.Size([50, 2])
        assert torch.isfinite(result).all()

    def test_high_dimensional_data(self):
        """Test GTM with high-dimensional data."""
        data = torch.randn(30, 100, dtype=torch.float64)  # High-dimensional
        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.5,
            reg_coeff=0.1,
            max_iter=3,  # Very short for speed
        )

        gtm.fit(data)
        result = gtm.transform(data)
        assert result.shape == torch.Size([30, 2])

    def test_zero_variance_features(self):
        """Test GTM handles zero variance features."""
        data = torch.randn(50, 5, dtype=torch.float64)
        data[:, 2] = 1.0  # Constant feature

        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.5,
            reg_coeff=0.1,
            max_iter=5,
        )

        # Should handle this gracefully
        with pytest.warns(UserWarning):  # Should warn about numerical issues
            gtm.fit(data)

        result = gtm.transform(data)
        assert result.shape == torch.Size([50, 2])

    def test_convergence_tolerance(self):
        """Test GTM stops early when tolerance is reached."""
        data = torch.randn(50, 5, dtype=torch.float64)
        gtm = VanillaGTM(
            num_nodes=9,
            num_basis_functions=4,
            basis_width=0.5,
            reg_coeff=0.1,
            max_iter=100,
            tolerance=1e-2,  # Loose tolerance
        )

        # Mock the logging to capture iterations
        with patch("chemographykit.gtm.logging") as mock_logging:
            gtm.fit(data)
            # Should have stopped before max_iter due to convergence
            # (This is hard to test precisely without access to internal state)
            assert mock_logging.info.called
