from __future__ import annotations

"""Worklog 98 -- fit a NURBS surface from one coherent synchronized-frame
curve-lattice component (path C), reusing Worklog 97's external-UV fitter
unchanged. No PCA anywhere in this path."""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_lattice import CurveLatticeUV
from osn_gs.surface.torch_curve_network_native_fit import DEGREE_U, DEGREE_V, RESOLUTION_U, RESOLUTION_V, ResidualStats
from osn_gs.surface.torch_nurbs import TorchNURBSSurface, fit_torch_visible_surface_from_uv
from osn_gs.utils.torch_ops import require_torch


def _residual_stats(surface: TorchNURBSSurface, points: Any, uv: Any) -> ResidualStats:
    torch = require_torch()
    if points is None or int(points.shape[0]) == 0:
        return ResidualStats(None, None, None, None, 0)
    predicted = surface.evaluate(uv)
    error = (predicted - points).norm(dim=1)
    values = error.detach().cpu()
    if values.numel() == 0:
        return ResidualStats(None, None, None, None, 0)
    return ResidualStats(
        float(values.mean().item()), float(values.median().item()),
        float(torch.quantile(values, 0.95).item()), float(values.max().item()), int(values.numel()),
    )


@dataclass(frozen=True)
class CurveLatticeNativeFitResult:
    valid_lattice: bool
    invalid_reason: str | None
    surface: TorchNURBSSurface | None
    lattice: CurveLatticeUV
    overall_residual: ResidualStats
    u_curve_residual: ResidualStats
    v_curve_residual: ResidualStats


def fit_curve_lattice_native(lattice: CurveLatticeUV) -> CurveLatticeNativeFitResult:
    """Fit path C: the synchronized-field lattice's own (u, v), never PCA,
    via the same :func:`~osn_gs.surface.torch_nurbs.fit_torch_visible_surface_from_uv`
    Worklog 97 already added -- reused unchanged, at the same fixed
    6x6/degree-2 capacity."""

    empty = ResidualStats(None, None, None, None, 0)
    if not lattice.valid:
        return CurveLatticeNativeFitResult(False, lattice.invalid_reason, None, lattice, empty, empty, empty)

    surface = fit_torch_visible_surface_from_uv(
        lattice.points, lattice.uv, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
        degree_u=DEGREE_U, degree_v=DEGREE_V,
    )
    overall = _residual_stats(surface, lattice.points, lattice.uv)

    torch = require_torch()
    if lattice.u_curves:
        u_points = torch.cat([curve.points for curve in lattice.u_curves], dim=0)
        u_uv = torch.cat([curve.uv for curve in lattice.u_curves], dim=0)
        u_residual = _residual_stats(surface, u_points, u_uv)
    else:
        u_residual = empty
    if lattice.v_curves:
        v_points = torch.cat([curve.points for curve in lattice.v_curves], dim=0)
        v_uv = torch.cat([curve.uv for curve in lattice.v_curves], dim=0)
        v_residual = _residual_stats(surface, v_points, v_uv)
    else:
        v_residual = empty

    return CurveLatticeNativeFitResult(True, None, surface, lattice, overall, u_residual, v_residual)
