from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]
LatentSignatureFn = Callable[[torch.Tensor], torch.Tensor]


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
    counts_ref = get_fingerprint_counts(ref_lib)
    counts_test = get_fingerprint_counts(test_lib)

    if use_weight:
        total_ref_count = sum(counts_ref.values())
        if total_ref_count == 0:
            return 0.0
        common_patterns = counts_ref.keys() & counts_test.keys()
        coverage_sum = sum(counts_ref[p] for p in common_patterns)
        coverage_value = coverage_sum / total_ref_count
    else:
        num_ref_patterns = len(counts_ref)
        if num_ref_patterns == 0:
            return 0.0
        common_patterns = counts_ref.keys() & counts_test.keys()
        coverage_value = len(common_patterns) / num_ref_patterns

    return coverage_value


@dataclass(frozen=True)
class SharpSafeStoppingConfig:
    """Validated configuration for staged, topology-aware EM stopping."""

    tolerance: float
    min_iter: int = 0
    convergence_patience: int = 1
    scale_tolerance: bool = False
    tolerance_reference_samples: int = 5_000
    topology_guard: bool = False
    topology_check_interval: int = 10
    topology_check_samples: int = 256
    topology_k_neighbors: int = 10
    topology_stability_threshold: float = 0.98

    def __post_init__(self) -> None:
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be >= 0")
        if self.min_iter < 0:
            raise ValueError("min_iter must be >= 0")
        if self.convergence_patience < 1:
            raise ValueError("convergence_patience must be >= 1")
        if self.tolerance_reference_samples < 1:
            raise ValueError("tolerance_reference_samples must be >= 1")
        if self.topology_check_interval < 1:
            raise ValueError("topology_check_interval must be >= 1")
        if self.topology_check_samples < 2:
            raise ValueError("topology_check_samples must be >= 2")
        if self.topology_k_neighbors < 1:
            raise ValueError("topology_k_neighbors must be >= 1")
        if not 0.0 <= self.topology_stability_threshold <= 1.0:
            raise ValueError("topology_stability_threshold must lie in [0, 1]")


@dataclass
class SharpSafeStoppingState:
    """Mutable convergence state owned by the stopping engine."""

    effective_tolerance: float
    anchor_data: Optional[torch.Tensor]
    previous_signature: Optional[torch.Tensor] = None
    converged_checks: int = 0
    last_topology_stability: Optional[float] = None
    last_topology_ok: bool = True


class SharpSafeConvergenceEngine:
    """Reusable staged convergence policy for EM-style manifold learners.

    The learner supplies only:
    - the current log-likelihood delta
    - a callback that maps a fixed anchor subset to a latent kNN signature

    This keeps the stopping rule independent from any specific GTM state layout.
    """

    def __init__(
        self,
        config: SharpSafeStoppingConfig,
        *,
        data: torch.Tensor,
        seed: int,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.state = SharpSafeStoppingState(
            effective_tolerance=self._effective_llh_tolerance(int(data.shape[0])),
            anchor_data=self._topology_anchor_subset(data),
            last_topology_ok=not config.topology_guard,
        )

    def _effective_llh_tolerance(self, n_samples: int) -> float:
        tol = float(self.config.tolerance)
        if not self.config.scale_tolerance:
            return tol
        scale = max(1.0, float(n_samples) / float(self.config.tolerance_reference_samples))
        return tol / float(np.sqrt(scale))

    def _topology_anchor_subset(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.config.topology_guard or data.shape[0] < 3:
            return None
        sample_count = min(int(data.shape[0]), self.config.topology_check_samples)
        if sample_count < 3:
            return None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            anchor_idx = torch.randperm(data.shape[0], device="cpu")[:sample_count]
        return data[anchor_idx.to(data.device)]

    @staticmethod
    def latent_neighbourhood_signature(
        coords: torch.Tensor,
        *,
        k_neighbors: int,
    ) -> torch.Tensor:
        if coords.ndim != 2:
            raise ValueError(f"coords must be 2D; got {tuple(coords.shape)}")
        dist = torch.cdist(
            coords.contiguous(),
            coords.contiguous(),
            compute_mode="use_mm_for_euclid_dist",
        )
        dist.fill_diagonal_(float("inf"))
        k = min(int(k_neighbors), max(1, coords.shape[0] - 1))
        return torch.topk(dist, k=k, largest=False).indices

    @staticmethod
    def topology_stability_score(
        current_signature: torch.Tensor,
        previous_signature: Optional[torch.Tensor],
    ) -> float:
        if previous_signature is None:
            return 0.0
        current_sorted = torch.sort(current_signature, dim=1).values
        previous_sorted = torch.sort(previous_signature, dim=1).values
        overlap = (
            (current_sorted.unsqueeze(-1) == previous_sorted.unsqueeze(-2))
            .any(dim=-1)
            .sum(dim=-1)
            .to(torch.float64)
        )
        return float((overlap / current_signature.shape[1]).mean())

    def check(
        self,
        *,
        iteration: int,
        llh_diff: Union[torch.Tensor, float],
        signature_fn: LatentSignatureFn,
    ) -> tuple[bool, Dict[str, float | bool]]:
        """Return whether training should stop at the current iteration."""
        info: Dict[str, float | bool] = {
            "llh_tolerance": float(self.state.effective_tolerance),
        }

        llh_diff_value = float(llh_diff)
        llh_plateau = llh_diff_value <= self.state.effective_tolerance
        info["llh_plateau"] = llh_plateau

        if iteration + 1 < self.config.min_iter:
            self.state.converged_checks = 0
            return False, info

        if not self.config.topology_guard:
            if llh_plateau:
                self.state.converged_checks += 1
            else:
                self.state.converged_checks = 0
            info["topology_ok"] = True
            info["converged_checks"] = self.state.converged_checks
            should_stop = self.state.converged_checks >= self.config.convergence_patience
            return should_stop, info

        check_due = (
            ((iteration + 1) % self.config.topology_check_interval == 0)
            or (self.state.anchor_data is not None and iteration + 1 == self.config.min_iter)
        )

        if check_due and self.state.anchor_data is not None:
            current_signature = signature_fn(self.state.anchor_data)
            stability = self.topology_stability_score(
                current_signature,
                self.state.previous_signature,
            )
            self.state.previous_signature = current_signature.detach()
            self.state.last_topology_stability = stability
            self.state.last_topology_ok = (
                stability >= self.config.topology_stability_threshold
            )

        info["topology_stability"] = (
            float(self.state.last_topology_stability)
            if self.state.last_topology_stability is not None
            else float("nan")
        )
        info["topology_ok"] = bool(self.state.last_topology_ok)

        if check_due:
            if llh_plateau and self.state.last_topology_ok:
                self.state.converged_checks += 1
            else:
                self.state.converged_checks = 0
        info["converged_checks"] = self.state.converged_checks

        should_stop = (
            check_due and self.state.converged_checks >= self.config.convergence_patience
        )
        return should_stop, info


@dataclass
class GeometryPrimitives:
    """Low-level latent geometry primitives with a clear tensor contract."""

    g11: torch.Tensor
    g12: torch.Tensor
    g22: torch.Tensor
    jacobian: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        n_points = int(self.g11.shape[0])
        if self.g11.ndim != 1 or self.g12.ndim != 1 or self.g22.ndim != 1:
            raise ValueError("g11, g12, and g22 must be 1D tensors")
        if self.g12.shape[0] != n_points or self.g22.shape[0] != n_points:
            raise ValueError("g11, g12, and g22 must have the same length")
        if self.jacobian is not None and (
            self.jacobian.ndim != 3
            or self.jacobian.shape[0] != n_points
            or self.jacobian.shape[2] != 2
        ):
            raise ValueError(
                "jacobian must have shape (n_points, n_features, 2) when provided"
            )

    def metric_tensor(self) -> torch.Tensor:
        metric = torch.empty((self.g11.shape[0], 2, 2), dtype=self.g11.dtype, device=self.g11.device)
        metric[:, 0, 0] = self.g11
        metric[:, 0, 1] = self.g12
        metric[:, 1, 0] = self.g12
        metric[:, 1, 1] = self.g22
        return metric

    def magnification(self) -> torch.Tensor:
        det_metric = torch.clamp(self.g11 * self.g22 - self.g12.square(), min=0.0)
        return torch.sqrt(det_metric)

    def condition_number(self) -> torch.Tensor:
        trace = self.g11 + self.g22
        spectral_gap = torch.sqrt(torch.clamp((self.g11 - self.g22).square() + 4.0 * self.g12.square(), min=0.0))
        eig_max = 0.5 * (trace + spectral_gap)
        eig_min = torch.clamp(0.5 * (trace - spectral_gap), min=1e-12)
        return eig_max / eig_min

    def as_dict(self) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {
            "g11": self.g11,
            "g12": self.g12,
            "g22": self.g22,
        }
        if self.jacobian is not None:
            result["jacobian"] = self.jacobian
        return result


@dataclass
class GeometryDiagnostics:
    """High-level GTM geometry outputs used by HPO and diagnostics."""

    magnification: torch.Tensor
    condition_number: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "magnification": self.magnification,
            "condition_number": self.condition_number,
        }


def resolve_latent_points(
    points: Optional[TensorLike],
    *,
    default_points: torch.Tensor,
    device: Union[str, torch.device],
    n_components: int,
) -> torch.Tensor:
    """Return latent coordinates as a validated float64 tensor."""
    if points is None:
        points_t = default_points
    elif isinstance(points, torch.Tensor):
        points_t = points
    else:
        points_t = torch.as_tensor(points, dtype=torch.float64)

    points_t = points_t.to(device=device, dtype=torch.float64)
    if points_t.ndim != 2 or points_t.shape[1] != n_components:
        raise ValueError(
            f"points must have shape (n_points, {n_components}); got {tuple(points_t.shape)}"
        )
    return points_t.contiguous()


def geometry_columns_chunk(
    points_chunk: torch.Tensor,
    *,
    mu: torch.Tensor,
    rbf_weights: torch.Tensor,
    inv_sigma2: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the two Jacobian columns for a latent-point chunk.

    The hot path is two GEMMs:
    ``gx = dphi_dx @ W`` and ``gy = dphi_dy @ W``.
    """
    mu = mu.contiguous()
    rbf_weights = rbf_weights.contiguous()

    mu_x = mu[:, 0].unsqueeze(0)
    mu_y = mu[:, 1].unsqueeze(0)

    dx = points_chunk[:, 0:1] - mu_x
    dy = points_chunk[:, 1:2] - mu_y

    phi = dx.square()
    phi.addcmul_(dy, dy, value=1.0)
    phi.mul_(-0.5 * inv_sigma2).exp_()

    dx.mul_(-inv_sigma2).mul_(phi)
    dy.mul_(-inv_sigma2).mul_(phi)

    gx = dx @ rbf_weights
    gy = dy @ rbf_weights
    return gx, gy


def geometry_primitives(
    *,
    points: Optional[TensorLike],
    default_points: torch.Tensor,
    mu: torch.Tensor,
    rbf_weights: torch.Tensor,
    inv_sigma2: float,
    device: Union[str, torch.device],
    n_components: int = 2,
    chunk_size: Optional[int] = None,
    include_jacobian: bool = False,
) -> GeometryPrimitives:
    """Return GTM latent geometry primitives with bounded peak memory."""
    points_t = resolve_latent_points(
        points,
        default_points=default_points,
        device=device,
        n_components=n_components,
    )
    n_points = points_t.shape[0]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n_points:
        chunk_size = n_points

    g11_parts: list[torch.Tensor] = []
    g12_parts: list[torch.Tensor] = []
    g22_parts: list[torch.Tensor] = []
    jacobian_parts: Optional[list[torch.Tensor]] = [] if include_jacobian else None

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        gx, gy = geometry_columns_chunk(
            points_t[start:stop],
            mu=mu,
            rbf_weights=rbf_weights,
            inv_sigma2=inv_sigma2,
        )
        g11_parts.append(torch.sum(gx * gx, dim=1))
        g12_parts.append(torch.sum(gx * gy, dim=1))
        g22_parts.append(torch.sum(gy * gy, dim=1))
        if jacobian_parts is not None:
            jacobian_parts.append(torch.stack((gx, gy), dim=-1))

    return GeometryPrimitives(
        g11=torch.cat(g11_parts, dim=0),
        g12=torch.cat(g12_parts, dim=0),
        g22=torch.cat(g22_parts, dim=0),
        jacobian=torch.cat(jacobian_parts, dim=0) if jacobian_parts is not None else None,
    )


def geometry_diagnostics_from_primitives(
    primitives: GeometryPrimitives,
) -> GeometryDiagnostics:
    """Return magnification and anisotropy from a single primitive bundle."""
    return GeometryDiagnostics(
        magnification=primitives.magnification(),
        condition_number=primitives.condition_number(),
    )
