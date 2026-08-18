from __future__ import annotations

"""Region-owned latent surface support: a queryable, position-based robust
local surface estimator (moving-least-squares-style) over Gaussian centers.

Gaussian centers are treated as noisy observations of an underlying surface,
never as surface vertices or a connectivity graph to be triangulated. This
module never reads Gaussian covariance and never mutates the positions it is
built from -- every query result is computed fresh and returned, nothing is
written back.

Public surface: :func:`build_latent_surface_support` builds a
:class:`LatentSurfaceSupport` from one region's owned centers;
``LatentSurfaceSupport.query``/``query_batch`` project arbitrary 3D positions
onto the locally supported surface and report a local tangent frame,
confidence, and an explicit ``supported`` flag. A query with no nearby
support is never silently accepted -- ``supported=False`` and the caller must
treat the projection as invalid (fail closed), not usable geometry.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# Fixed conventions (never tuned toward a favorable replay outcome):
# - k: local neighborhood size for the weighted PCA fit. Matches the k=8
#   used throughout Worklog 82/92/93's own local-neighborhood diagnostics,
#   widened slightly (16) because this module must ALSO serve out-of-sample
#   query positions (curve steps), not just the support points themselves,
#   so a modestly larger neighborhood improves fit stability.
DEFAULT_K = 16
# - bandwidth_scale: Gaussian kernel bandwidth as a multiple of the local
#   median support-point spacing.
DEFAULT_BANDWIDTH_SCALE = 1.5
# - support_radius_scale: a query farther than this multiple of local
#   spacing from its nearest support point is unsupported by construction
#   (never extrapolated).
DEFAULT_SUPPORT_RADIUS_SCALE = 3.0
# - min_effective_support: minimum Gaussian-kernel-weighted effective
#   neighbor count (sum of weights normalized by the max possible weight)
#   required to trust the local fit.
DEFAULT_MIN_EFFECTIVE_SUPPORT = 3.0
# - min_planarity: minimum ratio of the 2nd to 3rd eigenvalue of the local
#   weighted scatter matrix required to accept a normal direction (guards
#   against fitting a plane to a near-isotropic/degenerate local cloud).
DEFAULT_MIN_PLANARITY = 1.5
MLS_ITERATIONS = 2


@dataclass(frozen=True)
class LatentSurfaceQueryBatch:
    """Batched query result. All tensors share leading dimension M."""

    positions: Any  # (M, 3) MLS-projected positions (only meaningful where supported)
    normals: Any  # (M, 3)
    tangent_u: Any  # (M, 3)
    tangent_v: Any  # (M, 3)
    confidence: Any  # (M,) effective-neighbor-count based confidence, unitless
    supported: Any  # (M,) bool


@dataclass(frozen=True)
class LatentSurfaceSupport:
    """Read-only region-owned latent surface estimator.

    ``support_points`` are the raw Gaussian centers this estimator was built
    from -- never mutated, never covariance-derived. Every other field is a
    fixed convention (see module docstring), not swept per replay.
    """

    support_points: Any  # (N, 3)
    median_spacing: float
    k: int
    bandwidth: float
    support_radius: float
    min_effective_support: float
    min_planarity: float

    def query(self, position: Any) -> LatentSurfaceQueryBatch:
        torch = require_torch()
        return self.query_batch(torch.as_tensor(position, dtype=self.support_points.dtype,
                                                  device=self.support_points.device).reshape(1, 3))

    def query_batch(self, positions: Any) -> LatentSurfaceQueryBatch:
        """Project a batch of arbitrary 3D positions onto the local surface.

        Each query independently: (1) finds its k nearest support points,
        (2) fits a Gaussian-kernel-weighted local plane (weighted PCA), (3)
        projects the query onto that plane, (4) re-centers the neighbor
        search at the projected position and refits -- ``MLS_ITERATIONS``
        passes total, the minimal iterative moving-least-squares scheme.
        Unsupported queries (too far from any support point, too few
        effective neighbors, or a degenerate/non-planar local scatter) are
        flagged, never silently projected onto an invented plane.
        """

        torch = require_torch()
        positions = torch.as_tensor(positions, dtype=self.support_points.dtype, device=self.support_points.device)
        current = positions.clone()
        count = current.shape[0]
        k = min(self.k, self.support_points.shape[0])

        normals = torch.zeros((count, 3), dtype=current.dtype, device=current.device)
        tangent_u = torch.zeros((count, 3), dtype=current.dtype, device=current.device)
        tangent_v = torch.zeros((count, 3), dtype=current.dtype, device=current.device)
        confidence = torch.zeros(count, dtype=current.dtype, device=current.device)
        supported = torch.zeros(count, dtype=torch.bool, device=current.device)

        for _iteration in range(MLS_ITERATIONS):
            distance = torch.cdist(current, self.support_points)
            nearest_distance, nearest_index = torch.topk(distance, k, dim=1, largest=False)
            neighbor_points = self.support_points[nearest_index]  # (M, k, 3)

            nearest_support_distance = nearest_distance[:, 0]
            weights = torch.exp(-0.5 * (nearest_distance / max(self.bandwidth, _EPS)).square())
            effective_count = weights.sum(dim=1) / weights.max(dim=1).values.clamp_min(_EPS)

            weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(_EPS)
            mean = (neighbor_points * weights.unsqueeze(-1)).sum(dim=1) / weight_sum
            centered = neighbor_points - mean.unsqueeze(1)
            weighted_centered = centered * weights.unsqueeze(-1).sqrt()
            # (M, 3, 3) weighted scatter matrix per query.
            scatter = torch.einsum("mki,mkj->mij", weighted_centered, weighted_centered)
            eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
            # Ascending eigenvalues; normal = smallest-variance eigenvector.
            normal = eigenvectors[:, :, 0]
            tu = eigenvectors[:, :, 2]
            tv = eigenvectors[:, :, 1]

            offset = ((current - mean) * normal).sum(dim=1, keepdim=True)
            current = current - offset * normal

            lambda3, lambda2, _lambda1 = eigenvalues.unbind(-1)  # ascending: normal, mid, major
            planarity = lambda2 / lambda3.clamp_min(_EPS)

            normals, tangent_u, tangent_v, confidence = normal, tu, tv, effective_count
            supported = (
                (nearest_support_distance <= self.support_radius)
                & (effective_count >= self.min_effective_support)
                & (planarity >= self.min_planarity)
            )

        return LatentSurfaceQueryBatch(current, normals, tangent_u, tangent_v, confidence, supported)


def _median_pairwise_nn(points: Any) -> float:
    torch = require_torch()
    count = points.shape[0]
    if count < 2:
        return 1e-6
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    value = float(distance.min(dim=1).values.median().item())
    return value if value > 0 else 1e-6


def build_latent_surface_support(
    points: Any,
    *,
    k: int = DEFAULT_K,
    bandwidth_scale: float = DEFAULT_BANDWIDTH_SCALE,
    support_radius_scale: float = DEFAULT_SUPPORT_RADIUS_SCALE,
    min_effective_support: float = DEFAULT_MIN_EFFECTIVE_SUPPORT,
    min_planarity: float = DEFAULT_MIN_PLANARITY,
) -> LatentSurfaceSupport:
    """Build a latent surface estimator from one region's owned Gaussian
    centers. ``points`` is never modified or stored mutably elsewhere --
    the returned object holds its own reference for query-time neighbor
    search only."""

    torch = require_torch()
    points = torch.as_tensor(points)
    spacing = _median_pairwise_nn(points)
    return LatentSurfaceSupport(
        support_points=points,
        median_spacing=spacing,
        k=k,
        bandwidth=bandwidth_scale * spacing,
        support_radius=support_radius_scale * spacing,
        min_effective_support=min_effective_support,
        min_planarity=min_planarity,
    )
