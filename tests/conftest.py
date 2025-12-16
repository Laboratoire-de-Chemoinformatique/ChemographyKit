"""
Pytest configuration and shared fixtures for ChemographyKit tests.
"""

import warnings

import numpy as np
import pytest
import torch


def pytest_configure(config):
    """Configure pytest settings."""
    # Suppress specific warnings that are expected during testing
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*numerical issues.*"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*very large values.*"
    )


@pytest.fixture(scope="session")
def torch_seed():
    """Set torch random seed for reproducible tests."""
    torch.manual_seed(42)
    return 42


@pytest.fixture(scope="session")
def numpy_seed():
    """Set numpy random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_dataset(torch_seed):
    """Create a small dataset for testing."""
    return torch.randn(20, 5, dtype=torch.float64)


@pytest.fixture
def medium_dataset(torch_seed):
    """Create a medium-sized dataset for testing."""
    return torch.randn(100, 10, dtype=torch.float64)


@pytest.fixture
def high_dim_dataset(torch_seed):
    """Create a high-dimensional dataset for testing."""
    return torch.randn(50, 50, dtype=torch.float64)


@pytest.fixture
def responsibilities_2x2(numpy_seed):
    """Create responsibility matrix for 2x2 grid."""
    resp = np.random.rand(10, 4)
    # Normalize so each row sums to 1
    return resp / resp.sum(axis=1, keepdims=True)


@pytest.fixture
def responsibilities_3x3(numpy_seed):
    """Create responsibility matrix for 3x3 grid."""
    resp = np.random.rand(15, 9)
    # Normalize so each row sums to 1
    return resp / resp.sum(axis=1, keepdims=True)


@pytest.fixture
def binary_class_labels(numpy_seed):
    """Create binary class labels."""
    return np.random.choice([0, 1], size=20)


@pytest.fixture
def regression_values(numpy_seed):
    """Create regression target values."""
    return np.random.randn(20)


@pytest.fixture
def gtm_params():
    """Standard GTM parameters for testing."""
    return {
        "num_nodes": 9,  # 3x3 grid
        "num_basis_functions": 4,  # 2x2 grid
        "basis_width": 0.3,
        "reg_coeff": 0.1,
        "max_iter": 5,  # Short for testing
        "tolerance": 1e-3,
        "device": "cpu",
    }


class TestDataGenerator:
    """Helper class for generating test data with specific properties."""

    @staticmethod
    def create_clustered_data(n_samples=100, n_features=10, n_clusters=3, seed=42):
        """Create clustered data for testing GTM performance."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create cluster centers
        centers = torch.randn(n_clusters, n_features, dtype=torch.float64) * 3

        # Assign samples to clusters
        cluster_assignments = torch.randint(0, n_clusters, (n_samples,))

        # Generate data around centers
        data = torch.zeros(n_samples, n_features, dtype=torch.float64)
        for i in range(n_samples):
            cluster = cluster_assignments[i]
            data[i] = (
                centers[cluster] + torch.randn(n_features, dtype=torch.float64) * 0.5
            )

        return data, cluster_assignments

    @staticmethod
    def create_linear_data(n_samples=100, n_features=10, noise_level=0.1, seed=42):
        """Create data with linear structure for testing PCA initialization."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create linear subspace
        W = torch.randn(n_features, 2, dtype=torch.float64)
        z = torch.randn(n_samples, 2, dtype=torch.float64)

        # Generate data in linear subspace plus noise
        data = (
            z @ W.T
            + torch.randn(n_samples, n_features, dtype=torch.float64) * noise_level
        )

        return data

    @staticmethod
    def create_sparse_responsibilities(n_samples=50, n_nodes=9, sparsity=0.8, seed=42):
        """Create sparse responsibility matrix (many zeros)."""
        np.random.seed(seed)

        resp = np.random.rand(n_samples, n_nodes)

        # Make sparse by setting random entries to zero
        mask = np.random.rand(n_samples, n_nodes) < sparsity
        resp[mask] = 0

        # Ensure each row has at least one non-zero entry
        for i in range(n_samples):
            if resp[i].sum() == 0:
                resp[i, np.random.randint(n_nodes)] = 1.0

        # Normalize
        resp = resp / resp.sum(axis=1, keepdims=True)

        return resp


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator


# Performance testing utilities
@pytest.fixture
def benchmark_timer():
    """Simple timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time

        @property
        def elapsed(self):
            if self.end_time and self.start_time:
                return self.end_time - self.start_time
            return None

    return Timer


# Utility functions for test assertions
def assert_valid_responsibilities(responsibilities, tolerance=1e-6):
    """Assert that responsibility matrix is valid."""
    assert isinstance(responsibilities, (np.ndarray, torch.Tensor))

    if isinstance(responsibilities, torch.Tensor):
        responsibilities = responsibilities.detach().cpu().numpy()

    # Check shape
    assert responsibilities.ndim == 2

    # Check that all values are non-negative
    assert np.all(responsibilities >= 0), "Responsibilities must be non-negative"

    # Check that rows sum to approximately 1
    row_sums = responsibilities.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=tolerance
    ), f"Row sums should be 1, got {row_sums}"

    # Check for NaN or infinite values
    assert np.all(
        np.isfinite(responsibilities)
    ), "Responsibilities contain NaN or infinite values"


def assert_valid_coordinates(coordinates, min_val=1.0, max_val=None):
    """Assert that coordinate matrix is valid."""
    assert isinstance(coordinates, (np.ndarray, torch.Tensor, dict))

    if isinstance(coordinates, dict):
        # DataFrame-like structure
        x_vals = coordinates["x"] if "x" in coordinates else coordinates.get("x", [])
        y_vals = coordinates["y"] if "y" in coordinates else coordinates.get("y", [])
    else:
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()

        if coordinates.shape[0] == 2:  # (2, n_samples) format
            x_vals, y_vals = coordinates[0], coordinates[1]
        else:  # (n_samples, 2) format
            x_vals, y_vals = coordinates[:, 0], coordinates[:, 1]

    # Check bounds
    assert np.all(x_vals >= min_val), f"X coordinates should be >= {min_val}"
    assert np.all(y_vals >= min_val), f"Y coordinates should be >= {min_val}"

    if max_val is not None:
        assert np.all(x_vals <= max_val), f"X coordinates should be <= {max_val}"
        assert np.all(y_vals <= max_val), f"Y coordinates should be <= {max_val}"

    # Check for finite values
    assert np.all(np.isfinite(x_vals)), "X coordinates contain NaN or infinite values"
    assert np.all(np.isfinite(y_vals)), "Y coordinates contain NaN or infinite values"


# Make utility functions available to tests
@pytest.fixture
def assert_valid_responsibilities_fixture():
    """Provide responsibility validation function."""
    return assert_valid_responsibilities


@pytest.fixture
def assert_valid_coordinates_fixture():
    """Provide coordinate validation function."""
    return assert_valid_coordinates
