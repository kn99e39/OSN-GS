from __future__ import annotations

"""Phase 3 Trimmed Component Correctness Baseline.

Implements ``OSN_GS_Final_Boundary_First_NURBS_Direction.md`` §Phase 3: fits
ONE existing-fitter NURBS chart per Phase 1 ``SurfaceComponent``, using
Phase 2's own shared UV frame (so the trim mask and the fitted chart agree on
what "u, v" means) and Phase 2's refined support mask as the trim.

This is explicitly a *correctness baseline*, not the final architecture:

- the control grid spans the component's whole rectangular UV domain and MAY
  cross the hole (§3.3) -- topology is carried entirely by the trim mask, not
  by control-grid structure;
- the existing production fitter (IDW seed -> regularized LSQ -> foot-point
  correction, ``osn_gs/surface/torch_nurbs.py``) is reused UNMODIFIED (§3.1);
  this module only wires Phase 1/2 outputs into it and adds fit-quality
  diagnostics (§3.4 "Jacobian degeneracy", "control-grid collapse") that the
  existing fitter does not report on its own.

Foot-point correction can move a point's fitted UV away from the frame's
initial (pre-fit) mapping that Phase 2's mask was built from; ``mask_hit_rate``
below is an explicit, exported diagnostic of how much drift actually
happened, rather than an unstated assumption that the mask still matches.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_nurbs import (
    NURBSFitDiagnostics,
    TorchNURBSSurface,
    fit_torch_visible_surface_lsq,
)
from osn_gs.surface.torch_surface_components import SurfaceComponent
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TrimmedComponentFitResult:
    component_id: int
    surface: TorchNURBSSurface  # control_grid spans the whole UV domain; uv_support_mask carries the trim
    uv: Any  # (N, 2) final foot-point UV of the component's raw Gaussians
    diagnostics: NURBSFitDiagnostics
    fit_metrics: dict[str, Any]  # point-to-surface residual, Jacobian, control-grid collapse, mask consistency


def _jacobian_metrics(surface: TorchNURBSSurface, resolution: int = 24) -> dict[str, float]:
    """Surface Jacobian sanity over a dense UV grid (§3.4 "Jacobian degeneracy")."""

    torch = require_torch()
    t = torch.linspace(0.0, 1.0, resolution, device=surface.control_grid.device)
    u, v = torch.meshgrid(t, t, indexing="ij")
    _, du, dv = surface.evaluate_with_derivatives(torch.stack((u.flatten(), v.flatten()), dim=1))
    jacobian = torch.cross(du, dv, dim=1).norm(dim=1)
    median = jacobian.median().clamp_min(1e-12)
    return {
        "jacobian_median": float(median.cpu()),
        "jacobian_min": float(jacobian.min().cpu()),
        "degenerate_fraction": float((jacobian <= median * 1e-3).float().mean().cpu()),
    }


def _control_grid_metrics(control_grid: Any) -> dict[str, float]:
    """Control-grid collapse diagnostics (§3.4)."""

    torch = require_torch()
    flat = control_grid.reshape(-1, 3)
    extent = float((flat.amax(dim=0) - flat.amin(dim=0)).norm())
    edges = []
    if control_grid.shape[0] > 1:
        edges.append((control_grid[1:] - control_grid[:-1]).norm(dim=-1).reshape(-1))
    if control_grid.shape[1] > 1:
        edges.append((control_grid[:, 1:] - control_grid[:, :-1]).norm(dim=-1).reshape(-1))
    edge = torch.cat(edges) if edges else torch.zeros(1, device=control_grid.device)
    return {
        "extent": extent,
        "edge_median": float(edge.median().cpu()),
        "edge_min": float(edge.min().cpu()),
        "collapsed": bool(edge.numel() and float(edge.min().cpu()) <= 1e-8),
    }


def _mask_hit_rate(uv: Any, mask: Any) -> float:
    """Fraction of ``uv`` (final, post-correction) that lands inside ``mask``.

    Exposes any drift between Phase 2's mask (built from the frame's initial,
    pre-fit UV mapping) and where the fitted+corrected surface actually put
    each point, instead of silently assuming they still agree.
    """

    torch = require_torch()
    res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
    cell_u = torch.clamp((uv[:, 0] * res_u).long(), 0, res_u - 1)
    cell_v = torch.clamp((uv[:, 1] * res_v).long(), 0, res_v - 1)
    return float(mask[cell_u, cell_v].float().mean())


def fit_trimmed_component(
    component: SurfaceComponent,
    points: Any,
    frame: Any,  # UVFrame, from Phase 2's extract_component_boundary(...).frame
    refined_mask: Any,  # (R, R) bool, from Phase 2's extract_component_boundary(...).refined_mask
    resolution_u: int = 12,
    resolution_v: int = 12,
    degree_u: int = 2,
    degree_v: int = 2,
    smoothness_lambda: float = 1e-4,
    tikhonov_lambda: float = 1e-4,
    correction_rounds: int = 2,
    projection_iterations: int = 4,
    chunk_size: int = 4096,
) -> TrimmedComponentFitResult:
    """Fit one trimmed NURBS chart for a Phase 1 component, using Phase 2's frame/mask."""

    torch = require_torch()
    component_points = points[component.gaussian_indices]
    # Same frame Phase 2 used to build refined_mask, so the trim mask and the
    # chart's initial UV parameterization agree on what "u, v" means.
    initial_uv = frame.apply(component_points, clamp=True)

    surface, uv, diagnostics = fit_torch_visible_surface_lsq(
        component_points,
        resolution_u=resolution_u,
        resolution_v=resolution_v,
        degree_u=degree_u,
        degree_v=degree_v,
        smoothness_lambda=smoothness_lambda,
        tikhonov_lambda=tikhonov_lambda,
        correction_rounds=correction_rounds,
        projection_iterations=projection_iterations,
        chunk_size=chunk_size,
        initial_uv=initial_uv,
        collect_diagnostics=True,
    )
    surface.uv_support_mask = refined_mask

    residual = (surface.evaluate(uv).detach() - component_points).norm(dim=1)
    fit_metrics = {
        "point_to_surface_rms": float(residual.square().mean().sqrt().cpu()),
        "point_to_surface_max": float(residual.max().cpu()),
        "mask_hit_rate": _mask_hit_rate(uv.detach(), refined_mask),
        **{f"control_grid_{k}": v for k, v in _control_grid_metrics(surface.control_grid.detach()).items()},
        **_jacobian_metrics(surface),
    }

    return TrimmedComponentFitResult(
        component_id=component.component_id,
        surface=surface,
        uv=uv,
        diagnostics=diagnostics,
        fit_metrics=fit_metrics,
    )
