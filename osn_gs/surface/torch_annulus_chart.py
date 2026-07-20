from __future__ import annotations

"""Phase 4 §4.3/4.4 Annulus O-grid chart generator.

For an ``annulus``-classified component (Phase 2's outer loop + exactly one
hole loop), builds N radial "pie slice" quadrilateral charts instead of one
rectangle + trim mask (Phase 3). Each slice's own UV domain is, by
construction, the wedge between two angles and between the inner/outer
boundary -- so unlike Phase 3, there is nothing to trim: the chart shape
itself follows the true boundary. This directly targets Phase 3's measured
weakness (a single rectangular chart trimmed to a circular/annular boundary
loses ~0.2 of the true hole to the mismatch between the domain shapes).

Each slice reuses the EXISTING production fitter unmodified (IDW seed +
regularized LSQ + foot-point correction, same call Phase 3 makes) with a
polar-local initial UV mapping (tangential = angle fraction within the
slice, radial = radius fraction between the slice's own inner/outer bound)
instead of Phase 3's flat linear frame.

Continuity is C0-by-construction only where two slices are seeded from the
*exact same* angle/radius bounds; the plan (§4.5) explicitly scopes this as
"start with C0, tighten to G1/C1 later", so this module *measures and
reports* the actual world-space seam gap between adjacent independently-fit
patches (§4.4 "seam metric") rather than forcing exact continuity through
shared-control-point machinery.

Seam placement (§4.4 "deterministic seam placement, low-curvature/low-
confidence region 우선"): the synthetic benchmark scenes this module is
validated against are rotationally close to uniform, so there is no
distinguished low-curvature location to prefer -- the seam is placed at a
fixed angle=0 (documented simplification; curvature/confidence-driven seam
placement is deferred, not implemented).
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
class AnnulusChartSlice:
    slice_index: int
    angle_range: tuple[float, float]
    inner_radius: float
    outer_radius: float
    gaussian_indices: Any  # (N,) long tensor into the component's own points
    surface: TorchNURBSSurface
    uv: Any  # (N, 2) final foot-point UV of this slice's own Gaussians, same order as gaussian_indices
    diagnostics: NURBSFitDiagnostics
    fit_metrics: dict[str, Any]


@dataclass
class SeamDiagnostic:
    slice_a: int
    slice_b: int
    mean_gap: float
    max_gap: float
    sample_count: int


@dataclass
class AnnulusChartResult:
    component_id: int
    origin_world: Any  # (3,)
    origin_uv: Any  # (2,)
    segments: int
    slices: list[AnnulusChartSlice]
    seams: list[SeamDiagnostic]
    topology_checks: dict[str, Any]


def _jacobian_min(surface: TorchNURBSSurface, resolution: int = 12) -> float:
    torch = require_torch()
    t = torch.linspace(0.0, 1.0, resolution, device=surface.control_grid.device)
    u, v = torch.meshgrid(t, t, indexing="ij")
    _, du, dv = surface.evaluate_with_derivatives(torch.stack((u.flatten(), v.flatten()), dim=1))
    return float(torch.cross(du, dv, dim=1).norm(dim=1).min().cpu())



def build_annulus_chart(
    component: SurfaceComponent,
    points: Any,
    boundary_frame: Any,  # UVFrame, from Phase 2's extract_component_boundary(...).frame
    refined_mask: Any,  # (R, R) bool, from Phase 2's extract_component_boundary(...).refined_mask
    hole_boundary_world_points: list[list[float]],  # from boundary_result.hole_loops[0]
    segments: int = 8,
    slice_resolution_u: int = 8,
    slice_resolution_v: int = 4,
    degree_u: int = 2,
    degree_v: int = 1,
    angular_overlap_fraction: float = 0.0,
    seam_sample_count: int = 9,
) -> AnnulusChartResult:
    """Build an O-grid of ``segments`` radial NURBS charts for one annulus component.

    ``angular_overlap_fraction`` defaults to 0 (no overlap between adjacent
    slices' point-selection windows). This was NOT the original design: an
    overlap window was meant to give each patch shared context near its
    boundary for continuity, but ``local_s`` is clamped to ``[0, 1]``, so any
    positive overlap piles the overlap-window points up AT the s=0/s=1 edge
    -- swept 0/0.08/0.15/0.25/0.35 (2026-07-20): overlap monotonically WORSENS
    both fit RMS (0.00028 -> 0.01572) and seam gap (0.005 -> 0.156) on
    planar_hole. The Coons-style shared boundary radius (computed once per
    slice-boundary angle below, §4.3) turned out to be what actually matters
    for continuity; overlap was actively counterproductive and is disabled.
    """

    torch = require_torch()
    if segments < 3:
        raise ValueError("An O-grid annulus chart needs at least 3 angular segments.")

    component_points = points[component.gaussian_indices]
    # Hole center estimate: centroid of the hole loop's own boundary cells,
    # in world space, then re-expressed in the shared component UV frame so
    # every angle/radius computation below stays in one consistent frame.
    hole_boundary_world = torch.tensor(
        hole_boundary_world_points, dtype=component_points.dtype, device=component_points.device
    )
    origin_world = hole_boundary_world.mean(dim=0)
    origin_uv = boundary_frame.apply(origin_world.unsqueeze(0), clamp=False)[0]

    own_uv = boundary_frame.apply(component_points, clamp=False)
    relative = own_uv - origin_uv
    point_angle = torch.atan2(relative[:, 1], relative[:, 0])
    point_radius = relative.norm(dim=1)

    # Same polar decomposition of the REFINED MASK's own True cells, used to
    # seed each slice's inner/outer radius bound from the actual density-
    # refined support (not from the raw, possibly noisier, point cloud).
    resolution = int(refined_mask.shape[0])
    centers = (torch.arange(resolution, dtype=own_uv.dtype, device=own_uv.device) + 0.5) / resolution
    grid_u, grid_v = torch.meshgrid(centers, centers, indexing="ij")
    cell_uv = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)
    cell_supported = refined_mask.reshape(-1)
    cell_relative = cell_uv - origin_uv
    cell_angle = torch.atan2(cell_relative[:, 1], cell_relative[:, 0])
    cell_radius = cell_relative.norm(dim=1)

    two_pi = 2.0 * torch.pi
    point_angle_wrapped = torch.remainder(point_angle, two_pi)
    cell_angle_wrapped = torch.remainder(cell_angle, two_pi)
    angle_step = two_pi / segments
    overlap = angular_overlap_fraction * angle_step

    # Coons-style shared boundary values (§4.3 "Seed: Coons patch 또는
    # transfinite interpolation"): compute inner/outer radius ONCE per slice
    # BOUNDARY angle (not per slice interior), in a small window straddling
    # that angle. Slice k then uses (inner_boundary[k], inner_boundary[k+1])
    # as its s=0/s=1 radius bounds -- by construction the SAME value slice
    # k+1 uses at its own s=0 edge, so the two independently-fit patches'
    # domains agree exactly at the shared boundary instead of each guessing
    # its own radius range from only its own interior cells (which measurably
    # disagreed: pre-fix seam gaps were ~0.03-0.08 on a flat unit-scale plane
    # purely from this radius-bound mismatch, not from the LSQ fit itself).
    boundary_angles = torch.remainder(torch.arange(segments, dtype=own_uv.dtype, device=own_uv.device) * angle_step, two_pi)
    boundary_window = 0.5 * angle_step * max(angular_overlap_fraction * 4.0, 0.25)
    inner_boundary = torch.empty((segments,), dtype=own_uv.dtype, device=own_uv.device)
    outer_boundary = torch.empty((segments,), dtype=own_uv.dtype, device=own_uv.device)
    for k in range(segments):
        theta = float(boundary_angles[k])
        delta = torch.remainder(cell_angle_wrapped - theta + torch.pi, two_pi) - torch.pi
        near = (delta.abs() <= boundary_window) & cell_supported
        if bool(near.any()):
            inner_boundary[k] = cell_radius[near].min()
            outer_boundary[k] = cell_radius[near].max()
        else:
            delta_p = torch.remainder(point_angle_wrapped - theta + torch.pi, two_pi) - torch.pi
            near_p = delta_p.abs() <= boundary_window
            inner_boundary[k] = point_radius[near_p].min() if bool(near_p.any()) else 0.0
            outer_boundary[k] = point_radius[near_p].max() if bool(near_p.any()) else 1.0

    slices: list[AnnulusChartSlice] = []
    for k in range(segments):
        theta_lo = k * angle_step
        theta_hi = (k + 1) * angle_step
        inner_lo, inner_hi = float(inner_boundary[k]), float(inner_boundary[(k + 1) % segments])
        outer_lo, outer_hi = float(outer_boundary[k]), float(outer_boundary[(k + 1) % segments])
        inner_radius = min(inner_lo, inner_hi)  # reported/diagnostic summary only
        outer_radius = max(outer_lo, outer_hi)
        radius_span = max(outer_radius - inner_radius, 1e-6)

        # angular_overlap_fraction is 0 by default (see the function
        # docstring); kept as a parameter in case a future scene needs it,
        # but each slice's own point-selection window is exactly its angular
        # range unless explicitly widened.
        selected = (point_angle_wrapped >= theta_lo - overlap) & (point_angle_wrapped < theta_hi + overlap)
        indices = torch.nonzero(selected, as_tuple=False).reshape(-1)
        if int(indices.numel()) < 4:
            # Too few points to fit a wedge (only possible on pathologically
            # sparse/fine slicing): widen once more to the whole annulus
            # ring at this radius band rather than silently degrading.
            selected = (point_radius >= inner_radius - 0.1 * radius_span) & (
                point_radius <= outer_radius + 0.1 * radius_span
            )
            indices = torch.nonzero(selected, as_tuple=False).reshape(-1)

        slice_points = component_points[indices]
        slice_angle = point_angle_wrapped[indices]
        slice_radius = point_radius[indices]
        # Bilinear (Coons) local_s/local_t: the radius bounds themselves vary
        # linearly across the slice's angular extent, tying s=0/s=1 exactly
        # to the shared boundary radii computed above.
        local_s = torch.clamp((slice_angle - theta_lo) / angle_step, 0.0, 1.0)
        radius_lo_at_s = inner_lo + local_s * (inner_hi - inner_lo)
        radius_hi_at_s = outer_lo + local_s * (outer_hi - outer_lo)
        local_t = torch.clamp(
            (slice_radius - radius_lo_at_s) / (radius_hi_at_s - radius_lo_at_s).clamp_min(1e-6), 0.0, 1.0
        )
        initial_uv = torch.stack([local_s, local_t], dim=1)

        surface, uv, diagnostics = fit_torch_visible_surface_lsq(
            slice_points,
            resolution_u=slice_resolution_u,
            resolution_v=slice_resolution_v,
            degree_u=degree_u,
            degree_v=degree_v,
            initial_uv=initial_uv,
            collect_diagnostics=True,
        )
        # NOT hard-enforced as literal shared control points. An earlier
        # version of this function overwrote control_grid[:, 0]/[:, -1] (and
        # the u=0/u=1 radial edges) with identical values on both sides of a
        # seam, which does give exact (~1e-7) C0 continuity -- but measured
        # WORSE on both required accuracy metrics than the free LSQ fit below,
        # regardless of whether the imposed boundary curve was a 2-point
        # chord (planar_hole: chamfer 0.0058 -> 0.0061, false-fill 0.180 ->
        # 0.200) or sampled from the actual Phase 2 loop points at every
        # slice angle (chamfer -> 0.0095, false-fill -> 0.311 -- WORSE again,
        # because Phase 2's loop points are raster-cell centers with their
        # own staircase quantization noise, and forcing the fit through that
        # noisy curve added error the smooth chord did not). Since neither
        # hard-constraint variant beat "fit freely, then measure the gap",
        # and the plan (§4.5) explicitly scopes v1 as "measure, don't force,
        # C0" with continuity as later refinement, this stays a FREE fit; the
        # Coons-seeded ``initial_uv`` above is the only continuity mechanism.
        # ``boundary_anchor_max_error`` is kept as a diagnostic (how far the
        # free fit's own boundary drifted from the Coons chord seed) without
        # being enforced.
        edge_angles = torch.linspace(theta_lo, theta_hi, slice_resolution_u, dtype=own_uv.dtype, device=own_uv.device)
        edge_fraction = torch.linspace(0.0, 1.0, slice_resolution_u, dtype=own_uv.dtype, device=own_uv.device)
        inner_radius_edge = inner_lo + edge_fraction * (inner_hi - inner_lo)
        outer_radius_edge = outer_lo + edge_fraction * (outer_hi - outer_lo)
        direction = torch.stack((torch.cos(edge_angles), torch.sin(edge_angles)), dim=1)
        inner_edge = boundary_frame.to_world(origin_uv.unsqueeze(0) + direction * inner_radius_edge[:, None])
        outer_edge = boundary_frame.to_world(origin_uv.unsqueeze(0) + direction * outer_radius_edge[:, None])
        residual = (surface.evaluate(uv).detach() - slice_points).norm(dim=1)
        boundary_uv = torch.cat((
            torch.stack((torch.linspace(0.0, 1.0, slice_resolution_u, device=own_uv.device), torch.zeros(slice_resolution_u, device=own_uv.device)), dim=1),
            torch.stack((torch.linspace(0.0, 1.0, slice_resolution_u, device=own_uv.device), torch.ones(slice_resolution_u, device=own_uv.device)), dim=1),
        ), dim=0)
        boundary_error = (surface.evaluate(boundary_uv).detach() - torch.cat((inner_edge, outer_edge), dim=0)).norm(dim=1)
        fit_metrics = {
            "point_to_surface_rms": float(residual.square().mean().sqrt().cpu()),
            "point_count": int(slice_points.shape[0]),
            "jacobian_min": _jacobian_min(surface),
            "boundary_anchor_max_error": float(boundary_error.max().cpu()),
        }

        slices.append(
            AnnulusChartSlice(
                slice_index=k,
                angle_range=(theta_lo, theta_hi),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                gaussian_indices=component.gaussian_indices[indices],
                surface=surface,
                uv=uv.detach(),
                diagnostics=diagnostics,
                fit_metrics=fit_metrics,
            )
        )

    seams = _measure_seams(slices, seam_sample_count)
    topology_checks = {
        "uv_overlap": False,  # true by construction: angle ranges partition [0, 2pi) exactly
        "jacobian_fold_count": sum(1 for s in slices if s.fit_metrics["jacobian_min"] <= 0.0),
        "min_slice_point_count": min(s.fit_metrics["point_count"] for s in slices),
        "boundary_anchor_max_error": max(s.fit_metrics["boundary_anchor_max_error"] for s in slices),
        # NOT hard-enforced (see the free-fit vs. hard-constraint comparison
        # in the per-slice loop above); C0 is measured via `seams`, not forced.
        "shared_boundary_constraint": False,
    }

    return AnnulusChartResult(
        component_id=component.component_id,
        origin_world=origin_world,
        origin_uv=origin_uv,
        segments=segments,
        slices=slices,
        seams=seams,
        topology_checks=topology_checks,
    )


def _measure_seams(slices: list[AnnulusChartSlice], sample_count: int) -> list[SeamDiagnostic]:
    """World-space gap between adjacent slices' shared boundary (§4.4 seam metric).

    Slice ``k``'s local_s=1 edge and slice ``k+1``'s local_s=0 edge are
    nominally the same physical curve (both parameterized over the same
    inner->outer radius range at the shared angle); sampled independently
    from each patch's own fit, so the reported gap directly measures the
    continuity this v1 implementation does NOT enforce by construction.
    """

    torch = require_torch()
    t = torch.linspace(0.0, 1.0, sample_count, device=slices[0].surface.control_grid.device)
    seams = []
    n = len(slices)
    for k in range(n):
        a, b = slices[k], slices[(k + 1) % n]
        edge_a = torch.stack([torch.ones_like(t), t], dim=1)  # a's local_s=1 edge
        edge_b = torch.stack([torch.zeros_like(t), t], dim=1)  # b's local_s=0 edge
        points_a = a.surface.evaluate(edge_a).detach()
        points_b = b.surface.evaluate(edge_b).detach()
        gap = (points_a - points_b).norm(dim=1)
        seams.append(
            SeamDiagnostic(
                slice_a=a.slice_index,
                slice_b=b.slice_index,
                mean_gap=float(gap.mean().cpu()),
                max_gap=float(gap.max().cpu()),
                sample_count=sample_count,
            )
        )
    return seams


def annulus_iso_line_payload(
    result: AnnulusChartResult,
    interior_lines: int = 3,
    samples_per_line: int = 17,
) -> dict[str, Any]:
    """Sample NURBS O-grid iso-lines for export and audit.

    ``u`` is tangential within a wedge and ``v`` is radial. The ``v=0`` and
    ``v=1`` lines trace the inner and outer chart boundaries; the ``u=0`` and
    ``u=1`` lines are the radial chart connectors. These polylines are
    evaluated from the exported NURBS geometry, not drawn as a viewer overlay.
    """
    torch = require_torch()
    if interior_lines < 0:
        raise ValueError("interior_lines must be non-negative.")
    if samples_per_line < 2:
        raise ValueError("samples_per_line must be at least 2.")

    device = result.slices[0].surface.control_grid.device
    dtype = result.slices[0].surface.control_grid.dtype
    sample = torch.linspace(0.0, 1.0, samples_per_line, device=device, dtype=dtype)
    interior = torch.linspace(0.0, 1.0, interior_lines + 2, device=device, dtype=dtype)[1:-1]
    values = torch.cat((torch.zeros(1, device=device, dtype=dtype), interior, torch.ones(1, device=device, dtype=dtype)))
    slices: list[dict[str, Any]] = []
    for sl in result.slices:
        u_lines, v_lines = [], []
        for u in values:
            uv = torch.stack((torch.full_like(sample, u), sample), dim=1)
            u_lines.append({"u": float(u.cpu()), "points": sl.surface.evaluate(uv).detach().cpu().tolist()})
        for v in values:
            uv = torch.stack((sample, torch.full_like(sample, v)), dim=1)
            v_lines.append({"v": float(v.cpu()), "points": sl.surface.evaluate(uv).detach().cpu().tolist()})
        slices.append({"slice_index": sl.slice_index, "u_lines": u_lines, "v_lines": v_lines})
    return {
        "coordinate_semantics": {
            "u": "periodic tangential coordinate within the O-grid wedge",
            "v": "radial coordinate from the inner boundary (0) to the outer boundary (1)",
        },
        "samples_per_line": samples_per_line,
        "interior_lines_per_family": interior_lines,
        "slices": slices,
    }


def annulus_chart_payload(result: AnnulusChartResult) -> dict[str, Any]:
    """JSON-serializable provenance of an O-grid annulus chart."""

    return {
        "component_id": result.component_id,
        "origin_world": result.origin_world.detach().cpu().tolist(),
        "origin_uv": result.origin_uv.detach().cpu().tolist(),
        "segments": result.segments,
        "topology_checks": result.topology_checks,
        "iso_lines": annulus_iso_line_payload(result),
        "seams": [
            {
                "slice_a": s.slice_a, "slice_b": s.slice_b,
                "mean_gap": s.mean_gap, "max_gap": s.max_gap, "sample_count": s.sample_count,
            }
            for s in result.seams
        ],
        "slices": [
            {
                "slice_index": sl.slice_index,
                "angle_range": list(sl.angle_range),
                "inner_radius": sl.inner_radius,
                "outer_radius": sl.outer_radius,
                "control_grid_shape": [int(x) for x in sl.surface.control_grid.shape],
                "control_grid": sl.surface.control_grid.detach().cpu().tolist(),
                "weights": sl.surface.weights.detach().cpu().tolist(),
                "degree_u": int(sl.surface.degree_u),
                "degree_v": int(sl.surface.degree_v),
                "fit_metrics": sl.fit_metrics,
            }
            for sl in result.slices
        ],
    }
