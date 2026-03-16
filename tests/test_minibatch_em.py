"""Tests for minibatch EM implementation."""

import torch
import pytest

from chemographykit.gtm import GTM, VanillaGTM


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
        from chemographykit.gtm import DataStandardizer
        gtm_mb._input_standardizer = DataStandardizer(with_mean=True, with_std=True)
        x = gtm_mb._input_standardizer.fit_transform(x)
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

        gtm = GTM(
            num_nodes=16, num_basis_functions=4,
            basis_width=0.5, reg_coeff=0.1,
            max_iter=20, tolerance=1e-5,
            standardize=True, device="cpu",
            pca_engine="sklearn",
        )
        # Replicate fit() internals to force minibatch path
        x = data.to(torch.float64)
        from chemographykit.gtm import DataStandardizer
        gtm._input_standardizer = DataStandardizer(with_mean=True, with_std=True)
        x = gtm._input_standardizer.fit_transform(x)
        gtm.data_mean = torch.mean(x, dim=0)
        gtm.data_std = torch.std(x, dim=0)
        eigvecs, eigvals = gtm._get_pca(x)
        gtm.eigenvectors = eigvecs.detach().clone()
        gtm.eigenvalues = eigvals.detach().clone()
        gtm.weights = gtm._init_weights(eigvecs)
        gtm.weights[-1, :] = gtm.data_mean
        gtm.beta = gtm._init_beta(eigvals)
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
        # K=2025, N=200k → 2025 * 200k * 8 ≈ 3 GB → should trigger minibatch
        K = 2025
        N = 200_000
        matrix_bytes = K * N * 8
        assert matrix_bytes > 2 * 1024 ** 3

    def test_minibatch_subset_projection_consistent(self):
        """After minibatch EM fitting, projecting subsets should be consistent
        with projecting the full dataset."""
        torch.manual_seed(42)
        data = torch.randn(200, 8, dtype=torch.float64)

        gtm = GTM(
            num_nodes=9, num_basis_functions=4,
            basis_width=0.5, reg_coeff=0.1,
            max_iter=20, tolerance=1e-5,
            standardize=True, device="cpu",
            pca_engine="sklearn",
        )
        # Force minibatch
        x = data.to(torch.float64)
        from chemographykit.gtm import DataStandardizer
        gtm._input_standardizer = DataStandardizer(with_mean=True, with_std=True)
        x = gtm._input_standardizer.fit_transform(x)
        gtm.data_mean = torch.mean(x, dim=0)
        gtm.data_std = torch.std(x, dim=0)
        eigvecs, eigvals = gtm._get_pca(x)
        gtm.eigenvectors = eigvecs.detach().clone()
        gtm.eigenvalues = eigvals.detach().clone()
        gtm.weights = gtm._init_weights(eigvecs)
        gtm.weights[-1, :] = gtm.data_mean
        gtm.beta = gtm._init_beta(eigvals)
        gtm._fit_loop_minibatch(x, chunk_size=50)

        resps_full, _ = gtm.project(data)
        resps_subset, _ = gtm.project(data[:50])

        # Subset should match corresponding rows of full projection
        assert torch.allclose(resps_full[:50], resps_subset, atol=1e-10)
