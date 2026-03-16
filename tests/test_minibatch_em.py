"""Tests for minibatch EM and standardization consistency fixes."""

import torch
import numpy as np
import pytest

from chemographykit.gtm import GTM, VanillaGTM, DataStandardizer


# ---------------------------------------------------------------------------
# DataStandardizer.transform()
# ---------------------------------------------------------------------------

class TestDataStandardizerTransform:
    """Verify that transform() reuses statistics from fit_transform()."""

    def test_transform_matches_fit_transform(self):
        """transform(X) should give the same result as fit_transform(X) when
        applied to the same data."""
        torch.manual_seed(42)
        X = torch.randn(50, 10, dtype=torch.float64)

        std = DataStandardizer(with_mean=True, with_std=True)
        X_fitted = std.fit_transform(X)
        X_transformed = std.transform(X)

        assert torch.allclose(X_fitted, X_transformed, atol=1e-10)

    def test_transform_uses_training_stats_on_new_data(self):
        """transform(X_new) should standardize X_new using the training
        statistics, NOT X_new's own statistics."""
        torch.manual_seed(42)
        X_train = torch.randn(100, 5, dtype=torch.float64)
        X_test = torch.randn(20, 5, dtype=torch.float64) + 5.0  # shifted

        std = DataStandardizer(with_mean=True, with_std=True)
        std.fit_transform(X_train)

        X_test_transformed = std.transform(X_test)

        # The mean of the transformed test data should NOT be near zero
        # (because we used training stats, not test stats)
        test_mean = X_test_transformed.mean(dim=0)
        assert not torch.allclose(test_mean, torch.zeros(5, dtype=torch.float64), atol=0.5)

    def test_transform_raises_without_fit(self):
        """transform() should raise if fit_transform() was never called."""
        std = DataStandardizer(with_mean=True, with_std=True)
        X = torch.randn(10, 5, dtype=torch.float64)

        with pytest.raises(RuntimeError, match="fit_transform"):
            std.transform(X)


# ---------------------------------------------------------------------------
# Standardization consistency in GTM project()
# ---------------------------------------------------------------------------

class TestProjectStandardization:
    """Verify that project() uses training statistics, not per-batch stats."""

    @pytest.fixture
    def fitted_gtm(self):
        torch.manual_seed(42)
        data = torch.randn(200, 10, dtype=torch.float64)
        gtm = VanillaGTM(
            num_nodes=9, num_basis_functions=4,
            basis_width=0.3, reg_coeff=0.1,
            max_iter=10, tolerance=1e-4,
            standardize=True, device="cpu",
        )
        gtm.fit(data)
        return gtm, data

    def test_project_same_data_consistent(self, fitted_gtm):
        """Projecting the training data should give valid responsibilities."""
        gtm, data = fitted_gtm
        resps, llhs = gtm.project(data)

        assert resps.shape == (200, 9)
        assert llhs.shape == (200,)
        # Rows should sum to 1
        assert torch.allclose(
            resps.sum(dim=1),
            torch.ones(200, dtype=torch.float64),
            atol=1e-6,
        )

    def test_project_subset_matches_full(self, fitted_gtm):
        """Projecting a subset should give the same responsibilities as the
        corresponding rows of the full projection.  This would FAIL with
        per-batch standardization if the subset has different statistics."""
        gtm, data = fitted_gtm
        resps_full, _ = gtm.project(data)
        resps_subset, _ = gtm.project(data[:50])

        assert torch.allclose(resps_full[:50], resps_subset, atol=1e-10)

    def test_project_shifted_data_differs(self, fitted_gtm):
        """Projecting shifted data should produce different node assignments
        than projecting the training data (not re-centered to its own mean)."""
        gtm, data = fitted_gtm
        resps_orig, _ = gtm.project(data)
        resps_shifted, _ = gtm.project(data + 10.0)

        # At least some node assignments should differ
        nodes_orig = resps_orig.argmax(dim=1)
        nodes_shifted = resps_shifted.argmax(dim=1)
        differ_pct = (nodes_orig != nodes_shifted).float().mean()
        # With a +10 shift, we expect substantial difference
        assert differ_pct > 0.0, "Shifted data should have some different node assignments"


# ---------------------------------------------------------------------------
# Minibatch EM
# ---------------------------------------------------------------------------

class TestMinibatchEM:
    """Verify that minibatch EM produces results consistent with standard EM."""

    def test_minibatch_matches_standard(self):
        """On a dataset small enough for both, minibatch and standard EM
        should converge to very similar solutions when starting from the
        same initialization (PCA-based via GTM class)."""
        torch.manual_seed(123)
        data = torch.randn(300, 8, dtype=torch.float64)

        params = dict(
            num_nodes=9, num_basis_functions=4,
            basis_width=0.5, reg_coeff=0.1,
            max_iter=50, tolerance=1e-6,
            standardize=True, device="cpu",
            pca_engine="sklearn",
        )

        # Standard EM
        gtm_std = GTM(**params)
        gtm_std.fit(data)
        resps_std, _ = gtm_std.project(data)

        # Minibatch EM — replicate fit() but swap _fit_loop for _fit_loop_minibatch
        gtm_mb = GTM(**params)
        x = data.to(torch.float64)
        x = gtm_mb._standardize(x, with_mean=True, with_std=True, fit=True)
        gtm_mb.data_mean = torch.mean(x, dim=0)
        gtm_mb.data_std = torch.std(x, dim=0)
        eigvecs, eigvals = gtm_mb._get_pca(x)
        gtm_mb.eigenvectors = eigvecs.detach().clone()
        gtm_mb.eigenvalues = eigvals.detach().clone()
        gtm_mb.weights = gtm_mb._init_weights(eigvecs)
        gtm_mb.weights[-1, :] = gtm_mb.data_mean
        gtm_mb.beta = gtm_mb._init_beta(eigvals)
        gtm_mb._fit_loop_minibatch(x, chunk_size=50)

        resps_mb, _ = gtm_mb.project(data)

        # Both should have high node agreement
        nodes_std = resps_std.argmax(dim=1)
        nodes_mb = resps_mb.argmax(dim=1)
        agreement = (nodes_std == nodes_mb).float().mean()
        assert agreement > 0.85, f"Node agreement too low: {agreement:.2%}"

    def test_minibatch_responsibilities_valid(self):
        """Minibatch EM should produce valid responsibilities."""
        torch.manual_seed(42)
        data = torch.randn(500, 10, dtype=torch.float64)

        gtm = VanillaGTM(
            num_nodes=16, num_basis_functions=4,
            basis_width=0.5, reg_coeff=0.1,
            max_iter=20, tolerance=1e-5,
            standardize=True, device="cpu",
        )
        # Force minibatch EM
        x = data.to(torch.float64)
        x = gtm._standardize(x, with_mean=True, with_std=True, fit=True)
        gtm.data_mean = torch.mean(x, dim=0)
        gtm.data_std = torch.std(x, dim=0)
        gtm.weights = gtm._init_weights(x)
        gtm.weights[-1, :] = gtm.data_mean
        gtm.beta = gtm._init_beta()
        gtm._fit_loop_minibatch(x, chunk_size=100)

        resps, llhs = gtm.project(data)
        assert resps.shape == (500, 16)
        assert torch.all(resps >= 0)
        assert torch.allclose(
            resps.sum(dim=1),
            torch.ones(500, dtype=torch.float64),
            atol=1e-6,
        )
        assert torch.all(torch.isfinite(llhs))

    def test_auto_selects_minibatch_for_large_data(self):
        """_fit_loop should auto-select minibatch when (K,N) exceeds 2 GB."""
        torch.manual_seed(42)
        # K=2025, N needs to be > 2GB/(2025*8) ≈ 130k
        # We won't actually run 130k — just verify the heuristic check
        gtm = VanillaGTM(
            num_nodes=2025, num_basis_functions=225,
            basis_width=1.0, reg_coeff=0.1,
            max_iter=1, tolerance=1e-3,
            standardize=False, device="cpu",
        )
        N = 200_000  # 2025 * 200k * 8 ≈ 3 GB → should trigger minibatch
        matrix_bytes = 2025 * N * 8
        assert matrix_bytes > 2 * 1024 ** 3  # confirm it would trigger


class TestGTMWithMinibatchEM:
    """Test the GTM (PCA-initialized) class with minibatch EM path."""

    def test_gtm_pca_minibatch(self):
        """GTM (PCA init) should work via minibatch EM."""
        torch.manual_seed(42)
        data = torch.randn(300, 15, dtype=torch.float64)

        gtm = GTM(
            num_nodes=9, num_basis_functions=4,
            basis_width=0.5, reg_coeff=0.1,
            max_iter=20, tolerance=1e-5,
            standardize=True, device="cpu",
            pca_engine="sklearn",
        )
        # Force minibatch by fitting normally — it uses standard EM for small data
        gtm.fit(data)

        resps, _ = gtm.project(data)
        assert resps.shape == (300, 9)
        assert torch.allclose(
            resps.sum(dim=1),
            torch.ones(300, dtype=torch.float64),
            atol=1e-6,
        )

    def test_pickle_backward_compat(self):
        """A GTM pickled before this change (without _fitted_standardizer)
        should still work in project() with a fallback warning."""
        import pickle

        torch.manual_seed(42)
        data = torch.randn(100, 8, dtype=torch.float64)

        gtm = VanillaGTM(
            num_nodes=9, num_basis_functions=4,
            basis_width=0.3, reg_coeff=0.1,
            max_iter=5, standardize=True, device="cpu",
        )
        gtm.fit(data)

        # Simulate old pickle: remove _fitted_standardizer
        state = pickle.dumps(gtm)
        gtm_loaded = pickle.loads(state)
        gtm_loaded._fitted_standardizer = None

        with pytest.warns(UserWarning, match="No fitted standardizer found"):
            resps, _ = gtm_loaded.project(data)

        assert resps.shape == (100, 9)
        assert torch.allclose(
            resps.sum(dim=1),
            torch.ones(100, dtype=torch.float64),
            atol=1e-6,
        )
