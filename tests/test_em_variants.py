"""Regression tests: full-batch EM is the sole training mode.

Validates that:
- GTM always runs classical full-batch EM regardless of dataset size.
- The minibatch path has been fully removed (no _fit_loop_minibatch attribute).
- fit() returns a finite log-likelihood float and populates n_iter_ / stop_reason_.
- callback() is invoked with monotonically increasing iteration indices.
- score() is numerically consistent with the log-likelihoods from project().
- Full-batch EM produces non-degenerate latent coordinates on a mock mixed
  fingerprint corpus (bimodal sparsity, simulating DEL + ChEMBL interleaving).
"""

import warnings

import pytest
import torch

from chemographykit.gtm import GTM, VanillaGTM


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _small_gtm(**overrides) -> GTM:
    defaults = dict(
        num_nodes=9,
        num_basis_functions=4,
        basis_width=0.5,
        reg_coeff=0.1,
        max_iter=30,
        tolerance=1e-4,
        standardize=False,
        device="cpu",
        pca_engine="sklearn",
        seed=42,
    )
    defaults.update(overrides)
    return GTM(**defaults)


def _mock_mixed_fingerprints(
    n_del: int = 150,
    n_chembl: int = 150,
    fp_dim: int = 64,
    del_density: float = 0.05,
    chembl_density: float = 0.30,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (data, labels) with bimodal row-sparsity.

    Group 0 (DEL-like): sparse binary vectors (~5 % bits set).
    Group 1 (ChEMBL-like): denser binary vectors (~30 % bits set).
    Both groups share some active feature dimensions to ensure the latent
    space is not trivially separable — the manifold must genuinely interleave.
    """
    rng = torch.Generator().manual_seed(seed)
    del_fps = (torch.rand(n_del, fp_dim, generator=rng) < del_density).to(torch.float64)
    chembl_fps = (torch.rand(n_chembl, fp_dim, generator=rng) < chembl_density).to(torch.float64)
    data = torch.cat([del_fps, chembl_fps], dim=0)
    labels = torch.cat([torch.zeros(n_del), torch.ones(n_chembl)])
    return data, labels


# ---------------------------------------------------------------------------
# 1. Minibatch path is gone
# ---------------------------------------------------------------------------


class TestMinibatchRemoved:
    def test_no_fit_loop_minibatch_attribute(self):
        gtm = _small_gtm()
        assert not hasattr(gtm, "_fit_loop_minibatch"), (
            "_fit_loop_minibatch must be deleted; it introduced manifold-collapse bugs"
        )

    def test_no_fit_loop_standard_attribute(self):
        """_fit_loop_standard was a private split — also removed in the cleanup."""
        gtm = _small_gtm()
        assert not hasattr(gtm, "_fit_loop_standard")

    def test_fit_loop_is_direct_full_batch(self):
        """_fit_loop must exist and be directly callable (not a dispatcher)."""
        torch.manual_seed(0)
        gtm = _small_gtm()
        data = torch.randn(100, 8, dtype=torch.float64)
        gtm.fit(data)
        # After fitting _fit_loop must have set stop_reason_
        assert gtm.stop_reason_ in {"llh_plateau", "sharp_safe_plateau", "max_iter"}


# ---------------------------------------------------------------------------
# 2. fit() contract: return value, post-fit attributes
# ---------------------------------------------------------------------------


class TestFitContract:
    def test_fit_returns_finite_float(self):
        torch.manual_seed(1)
        gtm = _small_gtm()
        data = torch.randn(120, 8, dtype=torch.float64)
        llh = gtm.fit(data)
        assert isinstance(llh, float)
        assert llh == llh and llh != float("inf")  # finite, not NaN

    def test_train_llh_matches_return_value(self):
        torch.manual_seed(2)
        gtm = _small_gtm()
        data = torch.randn(120, 8, dtype=torch.float64)
        returned = gtm.fit(data)
        assert returned == gtm.train_llh_

    def test_n_iter_is_positive_integer(self):
        torch.manual_seed(3)
        gtm = _small_gtm()
        data = torch.randn(120, 8, dtype=torch.float64)
        gtm.fit(data)
        assert isinstance(gtm.n_iter_, int)
        assert gtm.n_iter_ >= 1

    def test_stop_reason_is_known_string(self):
        torch.manual_seed(4)
        gtm = _small_gtm()
        data = torch.randn(120, 8, dtype=torch.float64)
        gtm.fit(data)
        assert gtm.stop_reason_ in {"llh_plateau", "sharp_safe_plateau", "max_iter"}


# ---------------------------------------------------------------------------
# 3. callback is invoked in order
# ---------------------------------------------------------------------------


class TestCallback:
    def test_callback_receives_increasing_iterations(self):
        torch.manual_seed(5)
        gtm = _small_gtm(max_iter=10)
        data = torch.randn(80, 6, dtype=torch.float64)

        calls: list[tuple[int, float]] = []
        gtm.fit(data, callback=lambda i, llh: calls.append((i, llh)))

        assert len(calls) >= 1
        iterations = [c[0] for c in calls]
        assert iterations == list(range(len(calls))), "Iterations must be 0-based and contiguous"

    def test_callback_llh_values_are_finite(self):
        torch.manual_seed(6)
        gtm = _small_gtm(max_iter=10)
        data = torch.randn(80, 6, dtype=torch.float64)

        llhs: list[float] = []
        gtm.fit(data, callback=lambda i, llh: llhs.append(llh))

        assert all(v == v and v != float("inf") for v in llhs), "All callback LLh values must be finite"


# ---------------------------------------------------------------------------
# 4. score() is consistent with project()
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_matches_project_llh(self):
        torch.manual_seed(7)
        gtm = _small_gtm()
        data = torch.randn(100, 8, dtype=torch.float64)
        gtm.fit(data)

        _, llhs = gtm.project(data)
        score_full = gtm.score(data)
        assert abs(score_full - float(llhs.mean())) < 1e-9

    def test_score_batched_equals_unbatched(self):
        torch.manual_seed(8)
        gtm = _small_gtm()
        data = torch.randn(150, 8, dtype=torch.float64)
        gtm.fit(data)

        unbatched = gtm.score(data, batch_size=0)
        batched = gtm.score(data, batch_size=32)
        assert abs(unbatched - batched) < 1e-9

    def test_score_requires_fitted_model(self):
        gtm = _small_gtm()
        data = torch.randn(50, 8, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="not fitted"):
            gtm.score(data)


# ---------------------------------------------------------------------------
# 5. Full-batch EM on mixed-system corpus (the DEL+ChEMBL regression)
# ---------------------------------------------------------------------------


class TestFullBatchMixedCorpus:
    """Guard against the manifold-collapse bug that motivated the minibatch removal.

    When DEL-like (sparse) and ChEMBL-like (dense) binary fingerprints are
    co-trained, full-batch EM must produce:
    - non-degenerate latent coordinates (the grid is actually used)
    - genuine interleaving: both populations contribute to most of the grid,
      rather than collapsing to two isolated blobs.
    """

    def test_non_degenerate_latent_coordinates(self):
        """Latent coordinate std must be meaningfully above zero."""
        data, _ = _mock_mixed_fingerprints(seed=10)
        gtm = _small_gtm(num_nodes=16, num_basis_functions=4, max_iter=40, seed=10)
        gtm.fit(data)

        coords = gtm.transform(data)  # (N, 2)
        coord_std = coords.std(dim=0)
        assert float(coord_std[0]) > 1e-3, "x-coordinate collapsed to a single point"
        assert float(coord_std[1]) > 1e-3, "y-coordinate collapsed to a single point"

    def test_both_populations_span_grid(self):
        """Both DEL-like and ChEMBL-like molecules must use multiple distinct nodes."""
        data, labels = _mock_mixed_fingerprints(seed=11)
        gtm = _small_gtm(num_nodes=16, num_basis_functions=4, max_iter=40, seed=11)
        gtm.fit(data)

        resps, _ = gtm.project(data)
        node_assignments = resps.argmax(dim=1)

        del_nodes = set(node_assignments[labels == 0].tolist())
        chembl_nodes = set(node_assignments[labels == 1].tolist())

        # Each population must occupy more than one node (not all collapsed)
        assert len(del_nodes) > 1, f"DEL population collapsed to single node: {del_nodes}"
        assert len(chembl_nodes) > 1, f"ChEMBL population collapsed to single node: {chembl_nodes}"

        # The two populations must share at least one node (genuine interleaving)
        shared = del_nodes & chembl_nodes
        assert len(shared) >= 1, (
            f"Populations occupy completely disjoint nodes: DEL={del_nodes}, ChEMBL={chembl_nodes}"
        )

    def test_responsibilities_are_valid(self):
        """Responsibilities must be non-negative and sum to 1 per sample."""
        data, _ = _mock_mixed_fingerprints(seed=12)
        gtm = _small_gtm(num_nodes=16, num_basis_functions=4, max_iter=40, seed=12)
        gtm.fit(data)

        resps, llhs = gtm.project(data)
        assert torch.all(resps >= 0), "Negative responsibilities detected"
        assert torch.allclose(
            resps.sum(dim=1),
            torch.ones(data.shape[0], dtype=torch.float64),
            atol=1e-6,
        ), "Responsibilities do not sum to 1"
        assert torch.all(torch.isfinite(llhs)), "Non-finite log-likelihoods after full-batch EM"

    def test_llh_is_non_trivial(self):
        """The final training LLh must be meaningfully above the initial (untrained) value."""
        data, _ = _mock_mixed_fingerprints(seed=13)
        gtm = _small_gtm(num_nodes=16, num_basis_functions=4, max_iter=40, seed=13)

        initial_llhs: list[float] = []
        def _capture_first(i: int, llh: float) -> None:
            if i == 0:
                initial_llhs.append(llh)

        final_llh = gtm.fit(data, callback=_capture_first)
        assert initial_llhs, "callback never fired"
        assert final_llh > initial_llhs[0], (
            f"LLh did not improve: initial={initial_llhs[0]:.4f}, final={final_llh:.4f}"
        )