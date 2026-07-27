from __future__ import annotations

"""Deterministic geometry checks for isolated Boundary-first surfaces."""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_boundary_constrained_surface import (
    BoundaryConstrainedSurfaceResult,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class BoundaryFirstSurfaceQuality:
    """Measured constraints, not an eligibility or dispatcher decision."""

    boundary_sample_max_error: float
    boundary_sample_rms_error: float
    seam_max_error: float
    minimum_jacobian_norm: float
    pole_excluded_minimum_jacobian_norm: float | None
    finite: bool
    patch_count: int

    def payload(self) -> dict[str, Any]:
        return {
            "boundary_sample_max_error": self.boundary_sample_max_error,
            "boundary_sample_rms_error": self.boundary_sample_rms_error,
            "seam_max_error": self.seam_max_error,
            "minimum_jacobian_norm": self.minimum_jacobian_norm,
            "pole_excluded_minimum_jacobian_norm": self.pole_excluded_minimum_jacobian_norm,
            "finite": self.finite,
            "patch_count": self.patch_count,
            "measurement": "isolated_boundary_first_geometry",
        }


def measure_boundary_first_surface_quality(
    result: BoundaryConstrainedSurfaceResult,
) -> BoundaryFirstSurfaceQuality:
    """Measure exact sampled-boundary/seam preservation and non-degeneracy.

    The result is intentionally diagnostic only: a numeric observation must not
    silently route a chart through the legacy dispatcher or production path.
    """

    torch = require_torch()
    surfaces = tuple(result.surfaces)
    if not surfaces:
        raise ValueError("Boundary-first quality requires at least one surface.")
    # Payloads deliberately stay compact and do not carry tensors.  Read the
    # boundary values from the constructed degree-one control grids instead.
    errors: list[Any] = []
    jacobians: list[Any] = []
    seams: list[Any] = []
    non_pole_jacobians: list[Any] = []
    pole_aware = bool(getattr(result, "provenance", {}).get("pole_singularity"))
    for index, surface in enumerate(surfaces):
        dtype, device = surface.control_grid.dtype, surface.control_grid.device
        uv = torch.tensor(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=dtype, device=device)
        evaluated, derivative_u, derivative_v = surface.evaluate_with_derivatives(uv)
        expected = torch.stack((
            surface.control_grid[0, 0], surface.control_grid[-1, 0],
            surface.control_grid[0, -1], surface.control_grid[-1, -1],
        ))
        errors.append(torch.linalg.vector_norm(evaluated - expected, dim=1))
        patch_jacobians = torch.linalg.vector_norm(torch.cross(derivative_u, derivative_v, dim=1), dim=1)
        jacobians.append(patch_jacobians)
        if pole_aware:
            controls = surface.control_grid
            if bool(torch.allclose(controls[:, 0], controls[0, 0].expand_as(controls[:, 0]))):
                non_pole_jacobians.append(patch_jacobians[[2, 3]])
            elif bool(torch.allclose(controls[0, :], controls[0, 0].expand_as(controls[0, :]))):
                non_pole_jacobians.append(patch_jacobians[[1, 3]])
            else:
                non_pole_jacobians.append(patch_jacobians)
        else:
            non_pole_jacobians.append(patch_jacobians)
        if len(surfaces) > 1:
            next_surface = surfaces[(index + 1) % len(surfaces)]
            seams.append(torch.linalg.vector_norm(surface.control_grid[-1] - next_surface.control_grid[0], dim=1))
    all_errors = torch.cat(errors)
    all_jacobians = torch.cat(jacobians)
    all_seams = torch.cat(seams) if seams else torch.zeros((1,), dtype=all_errors.dtype, device=all_errors.device)
    all_non_pole_jacobians = torch.cat(non_pole_jacobians)
    return BoundaryFirstSurfaceQuality(
        boundary_sample_max_error=float(all_errors.max().detach().cpu()),
        boundary_sample_rms_error=float(torch.sqrt(torch.mean(all_errors.square())).detach().cpu()),
        seam_max_error=float(all_seams.max().detach().cpu()),
        minimum_jacobian_norm=float(all_jacobians.min().detach().cpu()),
        pole_excluded_minimum_jacobian_norm=float(all_non_pole_jacobians.min().detach().cpu()) if pole_aware else None,
        finite=bool(torch.isfinite(all_errors).all() and torch.isfinite(all_jacobians).all() and torch.isfinite(all_seams).all()),
        patch_count=len(surfaces),
    )
