from __future__ import annotations

"""Deterministic review metrics for observed-boundary source fidelity."""
from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class BoundarySourceFidelity:
    boundary_sample_count: int
    source_point_count: int
    local_spacing_median: float
    boundary_to_source_minimum: float
    boundary_to_source_median: float
    boundary_to_source_mean: float
    boundary_to_source_maximum: float
    normalized_median_distance: float
    normalized_maximum_distance: float

    def payload(self) -> dict[str, float | int]:
        return {
            "boundary_sample_count": self.boundary_sample_count,
            "source_point_count": self.source_point_count,
            "local_spacing_median": self.local_spacing_median,
            "boundary_to_source_minimum": self.boundary_to_source_minimum,
            "boundary_to_source_median": self.boundary_to_source_median,
            "boundary_to_source_mean": self.boundary_to_source_mean,
            "boundary_to_source_maximum": self.boundary_to_source_maximum,
            "normalized_median_distance": self.normalized_median_distance,
            "normalized_maximum_distance": self.normalized_maximum_distance,
        }


def measure_observed_boundary_source_fidelity(boundary_curve: Any, component_points: Any) -> BoundarySourceFidelity:
    """Measure an observed boundary against its own raw support points.

    This is a diagnostic/review metric.  It does not assert that a sampled
    point cloud lies exactly on a continuous boundary and therefore never
    changes construction eligibility by itself.
    """
    torch = require_torch()
    points = torch.as_tensor(component_points)
    boundary = torch.as_tensor(boundary_curve.world, dtype=points.dtype, device=points.device)
    if boundary.ndim != 2 or boundary.shape[1] != 3:
        raise ValueError("boundary_curve.world must have shape (N, 3).")
    if points.ndim != 2 or points.shape[1] != 3 or int(points.shape[0]) < 2:
        raise ValueError("component_points must have shape (N>=2, 3).")
    if int(boundary.shape[0]) < 1:
        raise ValueError("boundary_curve must contain at least one sample.")
    if int(boundary.shape[0]) > 1 and bool(torch.allclose(boundary[0], boundary[-1])):
        boundary = boundary[:-1]
    if int(boundary.shape[0]) < 1:
        raise ValueError("boundary_curve has no unique sample.")
    boundary_distances = torch.cdist(boundary, points).min(dim=1).values
    source_distances = torch.cdist(points, points)
    source_distances.fill_diagonal_(float("inf"))
    spacing = source_distances.min(dim=1).values.median()
    scale = max(float(spacing), 1e-12)
    return BoundarySourceFidelity(
        boundary_sample_count=int(boundary.shape[0]),
        source_point_count=int(points.shape[0]),
        local_spacing_median=float(spacing),
        boundary_to_source_minimum=float(boundary_distances.min()),
        boundary_to_source_median=float(boundary_distances.median()),
        boundary_to_source_mean=float(boundary_distances.mean()),
        boundary_to_source_maximum=float(boundary_distances.max()),
        normalized_median_distance=float(boundary_distances.median()) / scale,
        normalized_maximum_distance=float(boundary_distances.max()) / scale,
    )