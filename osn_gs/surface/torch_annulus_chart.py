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

**Coordinate convention (load-bearing for every diagnostic below, verified
by ``tests/test_annulus_chart.py``'s orientation-invariant tests, not just
asserted here):** ``u`` (``local_s``) is tangential/angular, ``v``
(``local_t``) is radial, for EVERY slice. Two consequences follow directly
from the construction below and are relied on by the seam metrics:

1. ``local_s`` increases with global angle for every slice (``local_s =
   (angle - theta_lo) / angle_step``), so ``Su = dS/d(local_s)`` points in
   the SAME physical rotational direction on both sides of every seam. No
   sign correction is needed when comparing slice A's ``Su`` at
   ``local_s=1`` against slice B's ``Su`` at ``local_s=0``.
2. ``local_t`` increases from the inner boundary (0) to the outer boundary
   (1) for every slice identically, so ``Sv`` is likewise directly
   comparable with no sign correction.

This is specific to this O-grid's own construction (every slice shares one
global angle/radius orientation) -- it is not a general NURBS-continuity
assumption and would not hold for arbitrarily-oriented independent charts.
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

_EPS = 1e-8


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
    """Continuity between adjacent slices A (``local_s=1`` edge) and B (``local_s=0`` edge).

    Two independent metric families, kept separate because they diagnose
    different failure modes (§ plan review, 2026-07-20):

    - Along-seam continuity: does the shared curve itself agree between the
      two independently-fit patches (position gap, ``Sv`` tangent angle --
      both slices parameterize the seam curve by ``v``/radius identically).
    - Across-seam continuity: does the surface behave consistently crossing
      the seam (``Su`` cross-boundary derivative angle, normal angle,
      derivative magnitude ratio).
    """

    slice_a: int
    slice_b: int
    sample_count: int
    mean_gap: float
    max_gap: float
    seam_tangent_angle_deg_mean: float
    seam_tangent_angle_deg_max: float
    seam_cross_derivative_angle_deg_mean: float
    seam_cross_derivative_angle_deg_max: float
    seam_normal_angle_deg_mean: float
    seam_normal_angle_deg_max: float
    seam_derivative_ratio_mean: float
    seam_derivative_ratio_max: float


@dataclass
class AnnulusChartResult:
    component_id: int
    origin_world: Any  # (3,)
    origin_uv: Any  # (2,)
    segments: int
    slices: list[AnnulusChartSlice]
    seams: list[SeamDiagnostic]
    topology_checks: dict[str, Any]
    chart_quality: dict[str, Any]


def _jacobian_diagnostics(
    surface: TorchNURBSSurface,
    resolution: int = 12,
    eps: float = _EPS,
    characteristic_length: float = 1.0,
    collect_samples: bool = False,
) -> dict[str, Any]:
    """Per-sample singular values of ``J = [Su Sv] in R^{3x2}``, plus orientation.

    ``sigma_min``/``sigma_max`` are the true parameterization singular
    values (via the closed-form eigenvalues of the 2x2 ``J^T J``), NOT
    ``||Su x Sv||`` (the local area scale, which equals ``sigma_min *
    sigma_max`` exactly for a 3x2 matrix -- kept separately as
    ``min_area_jacobian``). ``sigma_min -> 0`` is local collapse/compression;
    a large ``sigma_max/sigma_min`` ratio is anisotropic distortion; neither
    is visible from the area product alone (e.g. a very long, very thin
    patch can have a "healthy" mid-sized area with a terrible condition
    number).

    ``characteristic_length`` (Phase 4 hardening Step 4-A, per plan review
    2026-07-21 point 3.2): ``min_jacobian_singular_value`` alone is in
    absolute physical units, so it isn't comparable across components/scenes
    of different scale. ``min_jacobian_singular_value_normalized = sigma_min
    / (characteristic_length + eps)`` is reported alongside it (not instead
    of it). Defaults to ``1.0`` (i.e. normalized == absolute) so this
    function stays usable standalone/in unit tests without a caller having
    to supply a real scale; ``build_annulus_chart`` passes each component's
    own median radial width.

    Orientation reference: computed PER SLICE from that slice's OWN normal
    field (median-sign self-consistency, seeded from the slice's own
    central sample and majority-aligned), not one fixed global vector for
    the whole component -- a genuinely curved annulus legitimately rotates
    its true normal around the ring, so a single global reference would
    misfire. This catches orientation reversal / fold-over WITHIN one
    patch; reversal BETWEEN adjacent patches is a distinct condition,
    already covered by ``SeamDiagnostic.seam_normal_angle_deg_*`` AND (Step
    4-A addition) the component-level holonomy check in
    ``_orientation_holonomy`` -- verified NOT to hide real flips: the
    ``orientation_dot`` test below uses ``normal_unit`` (the RAW, never
    sign-corrected per-sample normal), never the sign-aligned ``aligned``
    array, which exists only to compute ``reference``'s mean direction.

    ``collect_samples`` (Step 4-A): when true, also returns the full
    per-sample ``(u, v, sigma_min, condition, orientation_dot, norm_su,
    norm_sv)`` grid under ``"samples"`` (default false -- no effect on any
    existing call site's output shape/performance) so a spatial heatmap can
    be exported/compared across seed-change candidates instead of only
    aggregate counts.
    """

    torch = require_torch()
    t = torch.linspace(0.0, 1.0, resolution, device=surface.control_grid.device)
    u, v = torch.meshgrid(t, t, indexing="ij")
    uv = torch.stack((u.reshape(-1), v.reshape(-1)), dim=1)
    _, du, dv = surface.evaluate_with_derivatives(uv)
    du, dv = du.detach(), dv.detach()

    a = (du * du).sum(dim=1)
    d = (dv * dv).sum(dim=1)
    b = (du * dv).sum(dim=1)
    trace = a + d
    disc = (trace.square() - 4.0 * (a * d - b * b)).clamp_min(0.0).sqrt()
    sigma_max = ((trace + disc).clamp_min(0.0) * 0.5).sqrt()
    sigma_min = ((trace - disc).clamp_min(0.0) * 0.5).sqrt()

    normal = torch.cross(du, dv, dim=1)
    area = normal.norm(dim=1)
    normal_unit = normal / area.clamp_min(eps)[:, None]
    seed = normal_unit[normal_unit.shape[0] // 2]
    aligned = torch.where((normal_unit @ seed < 0.0)[:, None], -normal_unit, normal_unit)
    reference = aligned.mean(dim=0)
    reference = reference / reference.norm().clamp_min(eps)
    orientation_dot = normal_unit @ reference

    condition = sigma_max / sigma_min.clamp_min(eps)
    sigma_min_normalized = sigma_min / (characteristic_length + eps)

    result = {
        "min_area_jacobian": float(area.min().cpu()),
        "min_jacobian_singular_value": float(sigma_min.min().cpu()),
        "min_jacobian_singular_value_normalized": float(sigma_min_normalized.min().cpu()),
        "characteristic_length": float(characteristic_length),
        "jacobian_condition_mean": float(condition.mean().cpu()),
        "jacobian_condition_p95": float(condition.quantile(0.95).cpu()),
        "max_jacobian_condition": float(condition.max().cpu()),
        "orientation_flip_count": int((orientation_dot < 0.0).sum()),
        "near_degenerate_count": int((sigma_min < eps).sum()),
        "sample_count": int(area.shape[0]),
        "reference_normal": reference.detach().cpu().tolist(),
    }
    if collect_samples:
        result["samples"] = {
            "u": u.reshape(-1).detach().cpu().tolist(),
            "v": v.reshape(-1).detach().cpu().tolist(),
            "sigma_min": sigma_min.detach().cpu().tolist(),
            "condition": condition.detach().cpu().tolist(),
            "orientation_dot": orientation_dot.detach().cpu().tolist(),
            "norm_su": du.norm(dim=1).detach().cpu().tolist(),
            "norm_sv": dv.norm(dim=1).detach().cpu().tolist(),
        }
    return result


def _orientation_holonomy(slices: list["AnnulusChartSlice"]) -> dict[str, Any]:
    """Component-level check (Phase 4 hardening Step 4-A, plan review point
    3.1/C): do the per-slice orientation references stay mutually consistent
    all the way around the closed ring?

    Each slice's ``reference_normal`` (``_jacobian_diagnostics``) is seeded
    and sign-aligned independently from that slice's OWN samples -- there is
    no guarantee two non-adjacent slices, or the ring as a whole, agree on
    which way is "up". A per-seam pairwise normal-angle number alone
    (``SeamDiagnostic.seam_normal_angle_deg_*``, always unsigned in
    ``[0, 180]``) cannot by itself reveal a genuine GLOBAL (topological)
    inconsistency, since it never sees more than two neighbors at once.

    Implementation: the sign of ``dot(reference_k, reference_{k+1 mod n})``
    for every adjacent pair around the ring (including the closing
    n-1 -> 0 pair). ``holonomy_consistent`` is the PRODUCT of these n signs
    being positive -- the standard parity invariant for a closed cyclic
    sequence of sign labels (a cyclic sequence of two states always has an
    EVEN number of state changes, a basic combinatorial fact, unless the
    underlying field is genuinely non-orientable at some point).

    **Known, deliberate limitation, not a false sense of coverage:** for
    this specific per-slice reference (effectively a single discrete +/-
    direction per slice, not a continuously rotating field), any isolated
    "flipped" slice necessarily produces exactly TWO local sign
    disagreements (entering and leaving it) -- an EVEN count, so a lone
    flipped slice is mathematically guaranteed to read as
    ``holonomy_consistent=True`` here (it is a real local anomaly, already
    caught by ``near_degenerate_count``/``orientation_flip_count`` on that
    slice and by the adjacent seam's own ``seam_normal_angle_deg``, just not
    by this check). This function exists as a general-purpose safety net
    for a genuinely non-orientable construction bug (e.g. an even/odd
    seam-count defect producing a real net twist around the ring), not as
    an additional detector for the single/paired-flip failure mode already
    covered elsewhere.
    """

    torch = require_torch()
    n = len(slices)
    refs = [torch.tensor(s.fit_metrics["reference_normal"]) for s in slices]
    pairwise_signs = [1 if float(refs[k] @ refs[(k + 1) % n]) >= 0.0 else -1 for k in range(n)]
    total_sign = 1
    for sign in pairwise_signs:
        total_sign *= sign
    return {
        "holonomy_consistent": bool(total_sign > 0),
        "holonomy_local_disagreement_count": sum(1 for sign in pairwise_signs if sign < 0),
    }


def _parameter_quality(surface: TorchNURBSSurface, resolution: int = 12, line_samples: int = 9) -> dict[str, Any]:
    """Chart parameterization quality: iso-line spacing uniformity, directional
    stretch, anisotropy, and orthogonality -- NOT just visual iso-line spacing.

    ``cv_u``/``cv_v`` (coefficient of variation of consecutive world-space
    point spacing along constant-v/constant-u lines) are reported as raw
    diagnostics only; a polar O-grid has an EXPECTED radial contraction
    near the inner (small-circumference) edge that raw CV cannot by itself
    distinguish from genuine crowding/collapse -- a detrended version is a
    candidate refinement, deferred until Step 3's real multi-scene numbers
    show it is actually needed.
    """

    torch = require_torch()
    device = surface.control_grid.device
    t_lines = torch.linspace(0.0, 1.0, resolution, device=device)
    t_samples = torch.linspace(0.0, 1.0, line_samples, device=device)

    def _cv(points: Any) -> float:
        seg = (points[1:] - points[:-1]).norm(dim=1)
        mean = seg.mean().clamp_min(_EPS)
        return float((seg.std(unbiased=False) / mean).cpu())

    cv_v_per_u_line = []
    for uc in t_lines:
        uv = torch.stack((torch.full_like(t_samples, float(uc)), t_samples), dim=1)
        cv_v_per_u_line.append(_cv(surface.evaluate(uv).detach()))
    cv_u_per_v_line = []
    for vc in t_lines:
        uv = torch.stack((t_samples, torch.full_like(t_samples, float(vc))), dim=1)
        cv_u_per_v_line.append(_cv(surface.evaluate(uv).detach()))

    grid_u, grid_v = torch.meshgrid(t_lines, t_lines, indexing="ij")
    uv = torch.stack((grid_u.reshape(-1), grid_v.reshape(-1)), dim=1)
    _, du, dv = surface.evaluate_with_derivatives(uv)
    du, dv = du.detach(), dv.detach()
    norm_u, norm_v = du.norm(dim=1), dv.norm(dim=1)
    anisotropy = torch.minimum(norm_u, norm_v) / torch.maximum(norm_u, norm_v).clamp_min(_EPS)
    orthogonality = (du * dv).sum(dim=1).abs() / (norm_u * norm_v).clamp_min(_EPS)

    return {
        "cv_v_along_u_line_mean": float(sum(cv_v_per_u_line) / len(cv_v_per_u_line)),
        "cv_u_along_v_line_mean": float(sum(cv_u_per_v_line) / len(cv_u_per_v_line)),
        "stretch_u_mean": float(norm_u.mean().cpu()),
        "stretch_u_min": float(norm_u.min().cpu()),
        "stretch_u_max": float(norm_u.max().cpu()),
        "stretch_v_mean": float(norm_v.mean().cpu()),
        "stretch_v_min": float(norm_v.min().cpu()),
        "stretch_v_max": float(norm_v.max().cpu()),
        "anisotropy_mean": float(anisotropy.mean().cpu()),
        "anisotropy_min": float(anisotropy.min().cpu()),
        "orthogonality_mean": float(orthogonality.mean().cpu()),
        "orthogonality_max": float(orthogonality.max().cpu()),
    }


def _boundary_conformance(chart_edge_world: Any, reference_world: Any, coverage_tolerance: float) -> dict[str, Any] | None:
    """Symmetric distance between a chart edge and Phase 2's OBSERVED-SUPPORT
    boundary for the corresponding loop (not ground truth -- Phase 2's own
    contour is itself density-threshold + marching-squares estimated).

    One-directional nearest-point distance alone can look perfect even if
    the chart edge has collapsed onto a small sub-arc of the true boundary
    (every chart sample near SOME reference point), so both directions are
    reported plus a coverage ratio that would catch exactly that failure.
    """

    torch = require_torch()
    if reference_world is None or int(reference_world.shape[0]) == 0 or int(chart_edge_world.shape[0]) == 0:
        return None
    d = torch.cdist(chart_edge_world, reference_world)
    edge_to_ref = d.min(dim=1).values
    ref_to_edge = d.min(dim=0).values
    coverage = float((ref_to_edge <= coverage_tolerance).float().mean().cpu())
    return {
        "edge_to_reference_mean": float(edge_to_ref.mean().cpu()),
        "edge_to_reference_max": float(edge_to_ref.max().cpu()),
        "reference_to_edge_mean": float(ref_to_edge.mean().cpu()),
        "reference_to_edge_max": float(ref_to_edge.max().cpu()),
        "symmetric_chamfer": float(0.5 * (edge_to_ref.mean() + ref_to_edge.mean()).cpu()),
        "hausdorff": float(torch.maximum(edge_to_ref.max(), ref_to_edge.max()).cpu()),
        "boundary_coverage_ratio": coverage,
        "coverage_tolerance": coverage_tolerance,
        "reference_point_count": int(reference_world.shape[0]),
    }


def _outer_radius_weighted_boundary_angles(
    cell_angle_wrapped: Any, cell_radius: Any, cell_supported: Any,
    point_angle_wrapped: Any, point_radius: Any,
    segments: int, bins: int = 72,
) -> Any:
    """Phase 4 hardening Step 4 (arc-length reparameterization, low-risk seed
    change per ``OSN_GS_Phase4_Hardening_Plan.md``): place the ``segments``
    O-grid seam angles so each wedge spans roughly EQUAL ARC LENGTH along the
    OUTER boundary, instead of equal ANGLE from the hole's centroid.

    Motivation (Step 3 baseline finding): equal-angle segments produced 4x
    the orientation-flip count and 3x the Jacobian condition number on
    ``planar_hole_offcenter`` versus the centered case -- an off-center (or
    elliptical) hole makes equal-angle wedges wildly unequal in physical
    size, with the worst wedge's inner corner becoming the near-degenerate
    corner documented in the Step 1 root-cause finding. Equalizing arc
    length directly targets this without touching the free-LSQ-fit
    mechanism at all (still a seed-only change, per the plan's discipline).

    Builds a coarse (``bins``-bucket) histogram of the OUTER radius as a
    function of angle from the same refined-mask cell data the uniform-angle
    path already uses (falls back to the raw point cloud for any bucket with
    no supported cells), then treats ``ds/dtheta ~= r_outer(theta)`` to get a
    cumulative arc-length function of angle, and inverts it at ``segments``
    equally-spaced arc-length fractions.
    """

    torch = require_torch()
    two_pi = 2.0 * torch.pi
    dtype, device = cell_angle_wrapped.dtype, cell_angle_wrapped.device
    edges = torch.linspace(0.0, two_pi, bins + 1, dtype=dtype, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    outer_per_bin = torch.full((bins,), float("nan"), dtype=dtype, device=device)
    bin_width = two_pi / bins
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        in_bin = (cell_angle_wrapped >= lo) & (cell_angle_wrapped < hi) & cell_supported
        if bool(in_bin.any()):
            outer_per_bin[i] = cell_radius[in_bin].max()
        else:
            in_bin_p = (point_angle_wrapped >= lo) & (point_angle_wrapped < hi)
            if bool(in_bin_p.any()):
                outer_per_bin[i] = point_radius[in_bin_p].max()
    # Fill any still-empty bins (sparse data) from the nearest filled
    # neighbor -- a plain scan is fine at this small bin count.
    values = outer_per_bin.tolist()
    if all(v != v for v in values):  # every bin empty: nothing to place from
        return torch.remainder(torch.arange(segments, dtype=dtype, device=device) * (two_pi / segments), two_pi)
    n = len(values)
    for i in range(n):
        if values[i] != values[i]:
            for d in range(1, n):
                lo_idx, hi_idx = (i - d) % n, (i + d) % n
                if values[lo_idx] == values[lo_idx]:
                    values[i] = values[lo_idx]
                    break
                if values[hi_idx] == values[hi_idx]:
                    values[i] = values[hi_idx]
                    break
    outer_per_bin = torch.tensor(values, dtype=dtype, device=device)

    ds = outer_per_bin * bin_width
    cumulative = torch.cumsum(ds, dim=0)
    cumulative = cumulative / cumulative[-1]
    targets = torch.arange(segments, dtype=dtype, device=device) / segments
    # searchsorted needs a leading zero so fraction 0 maps to bin 0's start.
    cumulative_padded = torch.cat((torch.zeros(1, dtype=dtype, device=device), cumulative))
    idx = torch.searchsorted(cumulative_padded, targets, right=False).clamp(1, bins) - 1
    return centers[idx]


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
    outer_boundary_world_points: list[list[float]] | None = None,
    boundary_conformance_tolerance: float = 0.05,
    segment_placement: str = "uniform_angle",
    collect_diagnostic_samples: bool = False,
    seam_phase_offset: float = 0.0,
    hermite_boundary_seed: bool = False,
) -> AnnulusChartResult:
    """Build an O-grid of ``segments`` radial NURBS charts for one annulus component.

    ``outer_boundary_world_points`` (from ``boundary_result.outer_loops[0]``,
    optional): if provided, enables the outer-edge Phase-2 boundary
    conformance check alongside the inner (hole) one, which is always
    available via ``hole_boundary_world_points``. ``boundary_conformance_
    tolerance`` is a provisional diagnostic default (domain-scale-relative,
    not yet derived from a multi-scene distribution -- see Step 3 of the
    Phase 4 hardening plan) and is reported alongside the coverage ratio it
    produces so it is never opaque in the output.

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

    ``segment_placement`` (Phase 4 hardening Step 4, opt-in, default
    ``"uniform_angle"`` byte-identical to the original Phase 4 behavior):
    ``"outer_radius_weighted_segment_placement"`` RELOCATES the ``segments``
    seam angles (i.e. changes the chart decomposition itself -- which
    Gaussians each wedge owns) so consecutive seams span equal arc length
    along the outer boundary, instead of equal angle from the hole's
    centroid. This is NOT a boundary-curve reparameterization (that would
    change sample correspondence along a FIXED curve without moving the
    seams themselves) -- it changes where the seams are, a bigger-regression
    -risk change than that name would suggest, which is why it is named for
    what it actually does. A/B tested against the Step 3 baseline scenes and
    REJECTED as the default (see ``OSN_GS_Phase4_Hardening_Plan.md`` Step 4
    candidate 1) -- kept only as a documented, tested ablation tool via
    ``--bf-annulus-segment-placement``.

    ``seam_phase_offset`` (Phase 4 hardening Step 4-B, only meaningful for
    ``segment_placement="uniform_angle"``, default ``0.0`` byte-identical to
    before): rotates all ``segments`` seam angles by this constant, keeping
    every wedge's angular WIDTH and the topology/point-assignment mechanism
    otherwise identical -- unlike ``outer_radius_weighted_segment_placement``,
    this cannot change wedge count or create unequal widths, so it carries
    much less regression risk while still letting the narrow off-center
    inner corner NOT necessarily land exactly on a seam.

    ``hermite_boundary_seed`` (Phase 4 hardening Step 4-C, default
    ``False`` byte-identical to before): replaces the bilinear Coons seed's
    linear radius interpolation with a cubic Hermite blend using a SHARED
    (central-difference) boundary slope, so adjacent slices' seeds start
    from matching d(radius)/d(local_s) at the shared boundary, not just a
    matching radius value. Explicitly scoped to seam CONTINUITY only --
    it does not address the inner-corner Jacobian collapse mechanism
    (``Su -> 0``), which a smoother seed does not change.
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
    if segment_placement == "outer_radius_weighted_segment_placement":
        boundary_angles = _outer_radius_weighted_boundary_angles(
            cell_angle_wrapped, cell_radius, cell_supported, point_angle_wrapped, point_radius, segments
        )
    elif segment_placement == "uniform_angle":
        boundary_angles = torch.remainder(
            torch.arange(segments, dtype=own_uv.dtype, device=own_uv.device) * angle_step + seam_phase_offset, two_pi
        )
    else:
        raise ValueError(f"Unknown segment_placement: {segment_placement!r}")
    # Local angular spacing AT each boundary index (average of the segment
    # widths on either side) -- reduces to exactly ``angle_step`` for
    # ``uniform_angle`` (byte-identical to the pre-Step-4 window), and
    # adapts per-boundary for ``outer_radius_weighted_segment_placement``'s non-uniform spacing.
    spacing_next = torch.remainder(boundary_angles.roll(-1) - boundary_angles, two_pi)
    spacing_next = torch.where(spacing_next == 0.0, torch.full_like(spacing_next, two_pi), spacing_next)
    spacing_prev = spacing_next.roll(1)
    local_spacing = 0.5 * (spacing_prev + spacing_next)
    inner_boundary = torch.empty((segments,), dtype=own_uv.dtype, device=own_uv.device)
    outer_boundary = torch.empty((segments,), dtype=own_uv.dtype, device=own_uv.device)
    for k in range(segments):
        theta = float(boundary_angles[k])
        boundary_window = 0.5 * float(local_spacing[k]) * max(angular_overlap_fraction * 4.0, 0.25)
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

    # Step 4-A: component-level characteristic length for scale-normalized
    # Jacobian singular values (median radial width across all boundary
    # samples) -- computed once, shared by every slice's diagnostics call.
    characteristic_length = float((outer_boundary - inner_boundary).median())

    # Step 4-C (Hermite/derivative-aware Coons seed, opt-in via
    # ``hermite_boundary_seed``): a SHARED d(radius)/d(theta) slope at each
    # boundary index, via central difference -- shared because it depends
    # only on ``inner_boundary``/``outer_boundary``/``boundary_angles``,
    # never on which slice is asking, so slice k's local_s=1 edge and slice
    # k+1's local_s=0 edge always reference the exact same slope value.
    # Denominator uses the SAME ``local_spacing`` already computed above
    # (half of prev+next boundary spacing), so this reduces to the standard
    # central-difference formula for uniform_angle spacing.
    inner_slope = (inner_boundary.roll(-1) - inner_boundary.roll(1)) / (2.0 * local_spacing)
    outer_slope = (outer_boundary.roll(-1) - outer_boundary.roll(1)) / (2.0 * local_spacing)

    slices: list[AnnulusChartSlice] = []
    inner_edges_world: list[Any] = []
    outer_edges_world: list[Any] = []
    for k in range(segments):
        theta_lo = float(boundary_angles[k])
        theta_hi = float(boundary_angles[(k + 1) % segments])
        if theta_hi <= theta_lo:
            theta_hi += two_pi  # wrap-around segment (last -> first boundary angle)
        slice_width = theta_hi - theta_lo
        inner_lo, inner_hi = float(inner_boundary[k]), float(inner_boundary[(k + 1) % segments])
        outer_lo, outer_hi = float(outer_boundary[k]), float(outer_boundary[(k + 1) % segments])
        inner_radius = min(inner_lo, inner_hi)  # reported/diagnostic summary only
        outer_radius = max(outer_lo, outer_hi)
        radius_span = max(outer_radius - inner_radius, 1e-6)

        # Shift point angles into [theta_lo, theta_lo + two_pi) before
        # comparing against theta_hi -- theta_hi can exceed two_pi for the
        # wrap-around segment (or, for non-uniform ``outer_radius_weighted_segment_placement``
        # placement, any segment could in principle straddle the 0/two_pi
        # seam depending on where the first boundary angle falls). For
        # ``uniform_angle`` this reduces to the original unshifted
        # comparison exactly (verified byte-identical numbers).
        shifted_point_angle = theta_lo + torch.remainder(point_angle_wrapped - theta_lo, two_pi)
        overlap_k = angular_overlap_fraction * slice_width
        # angular_overlap_fraction is 0 by default (see the function
        # docstring); kept as a parameter in case a future scene needs it,
        # but each slice's own point-selection window is exactly its angular
        # range unless explicitly widened.
        selected = (shifted_point_angle >= theta_lo - overlap_k) & (shifted_point_angle < theta_hi + overlap_k)
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
        slice_angle = shifted_point_angle[indices]
        slice_radius = point_radius[indices]
        # Bilinear (Coons) local_s/local_t: the radius bounds themselves vary
        # linearly across the slice's angular extent, tying s=0/s=1 exactly
        # to the shared boundary radii computed above.
        local_s = torch.clamp((slice_angle - theta_lo) / slice_width, 0.0, 1.0)
        if hermite_boundary_seed:
            # Step 4-C: cubic Hermite blend using the SHARED boundary slopes
            # (inner_slope/outer_slope, computed once above from central
            # differences, identical for both slices meeting at a boundary)
            # instead of pure linear interpolation -- matches d(radius)/
            # d(local_s) at both edges, not just the radius value itself.
            # Reduces to the exact linear formula when both slopes are
            # equal (e.g. a perfectly circular boundary), since h10+h11
            # then contribute a term proportional to (t^3-t^2)+(t^3-2t^2+t)
            # ... this is NOT claimed to fix inner-corner collapse (that
            # failure is ``Su`` magnitude going to zero, untouched by a
            # smoother seed) -- only measured against seam CONTINUITY.
            t = local_s
            h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
            h10 = t**3 - 2.0 * t**2 + t
            h01 = -2.0 * t**3 + 3.0 * t**2
            h11 = t**3 - t**2
            m_inner_lo = float(inner_slope[k]) * slice_width
            m_inner_hi = float(inner_slope[(k + 1) % segments]) * slice_width
            m_outer_lo = float(outer_slope[k]) * slice_width
            m_outer_hi = float(outer_slope[(k + 1) % segments]) * slice_width
            radius_lo_at_s = h00 * inner_lo + h10 * m_inner_lo + h01 * inner_hi + h11 * m_inner_hi
            radius_hi_at_s = h00 * outer_lo + h10 * m_outer_lo + h01 * outer_hi + h11 * m_outer_hi
        else:
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
        # ``seed_boundary_anchor_error`` is kept as a diagnostic (how far the
        # free fit's own boundary drifted from the Coons chord seed) without
        # being enforced.
        edge_angles = torch.linspace(theta_lo, theta_hi, slice_resolution_u, dtype=own_uv.dtype, device=own_uv.device)
        edge_fraction = torch.linspace(0.0, 1.0, slice_resolution_u, dtype=own_uv.dtype, device=own_uv.device)
        inner_radius_edge = inner_lo + edge_fraction * (inner_hi - inner_lo)
        outer_radius_edge = outer_lo + edge_fraction * (outer_hi - outer_lo)
        direction = torch.stack((torch.cos(edge_angles), torch.sin(edge_angles)), dim=1)
        inner_edge = boundary_frame.to_world(origin_uv.unsqueeze(0) + direction * inner_radius_edge[:, None])
        outer_edge = boundary_frame.to_world(origin_uv.unsqueeze(0) + direction * outer_radius_edge[:, None])
        inner_edges_world.append(inner_edge.detach())
        outer_edges_world.append(outer_edge.detach())
        residual = (surface.evaluate(uv).detach() - slice_points).norm(dim=1)
        boundary_uv = torch.cat((
            torch.stack((torch.linspace(0.0, 1.0, slice_resolution_u, device=own_uv.device), torch.zeros(slice_resolution_u, device=own_uv.device)), dim=1),
            torch.stack((torch.linspace(0.0, 1.0, slice_resolution_u, device=own_uv.device), torch.ones(slice_resolution_u, device=own_uv.device)), dim=1),
        ), dim=0)
        boundary_error = (surface.evaluate(boundary_uv).detach() - torch.cat((inner_edge, outer_edge), dim=0)).norm(dim=1)

        jacobian_diag = _jacobian_diagnostics(
            surface, characteristic_length=characteristic_length, collect_samples=collect_diagnostic_samples
        )
        parameter_quality = _parameter_quality(surface)
        fit_metrics = {
            "point_to_surface_rms": float(residual.square().mean().sqrt().cpu()),
            "point_count": int(slice_points.shape[0]),
            "seed_boundary_anchor_error": float(boundary_error.max().cpu()),
            **jacobian_diag,
            "parameter_quality": parameter_quality,
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
        "near_degenerate_slice_count": sum(1 for s in slices if s.fit_metrics["near_degenerate_count"] > 0),
        "min_jacobian_singular_value": min(s.fit_metrics["min_jacobian_singular_value"] for s in slices),
        "max_jacobian_condition": max(s.fit_metrics["max_jacobian_condition"] for s in slices),
        "total_orientation_flip_samples": sum(s.fit_metrics["orientation_flip_count"] for s in slices),
        "min_slice_point_count": min(s.fit_metrics["point_count"] for s in slices),
        "seed_boundary_anchor_max_error": max(s.fit_metrics["seed_boundary_anchor_error"] for s in slices),
        # NOT hard-enforced (see the free-fit vs. hard-constraint comparison
        # in the per-slice loop above); C0 is measured via `seams`, not forced.
        "shared_boundary_constraint": False,
    }

    inner_reference = hole_boundary_world if hole_boundary_world.numel() else None
    outer_reference = (
        torch.tensor(outer_boundary_world_points, dtype=component_points.dtype, device=component_points.device)
        if outer_boundary_world_points
        else None
    )
    chart_quality = {
        "jacobian": {
            "min_area_jacobian": min(s.fit_metrics["min_area_jacobian"] for s in slices),
            "min_jacobian_singular_value": topology_checks["min_jacobian_singular_value"],
            "min_jacobian_singular_value_normalized": min(s.fit_metrics["min_jacobian_singular_value_normalized"] for s in slices),
            "characteristic_length": characteristic_length,
            # Component-level rollup of already-per-slice-aggregated
            # mean/p95/max (slice -> sample rollup happens inside
            # ``_jacobian_diagnostics``); this is a coarser second-level
            # rollup, not a re-aggregation over raw per-sample values.
            "jacobian_condition_mean_of_slice_means": sum(s.fit_metrics["jacobian_condition_mean"] for s in slices) / len(slices),
            "jacobian_condition_max_of_slice_p95": max(s.fit_metrics["jacobian_condition_p95"] for s in slices),
            "max_jacobian_condition": topology_checks["max_jacobian_condition"],
            "total_orientation_flip_samples": topology_checks["total_orientation_flip_samples"],
            "total_near_degenerate_samples": sum(s.fit_metrics["near_degenerate_count"] for s in slices),
            **_orientation_holonomy(slices),
        },
        "seams": {
            "position_gap_mean": sum(s.mean_gap for s in seams) / len(seams) if seams else 0.0,
            "position_gap_max": max((s.max_gap for s in seams), default=0.0),
            "tangent_angle_deg_mean": sum(s.seam_tangent_angle_deg_mean for s in seams) / len(seams) if seams else 0.0,
            "tangent_angle_deg_max": max((s.seam_tangent_angle_deg_max for s in seams), default=0.0),
            "cross_derivative_angle_deg_mean": sum(s.seam_cross_derivative_angle_deg_mean for s in seams) / len(seams) if seams else 0.0,
            "cross_derivative_angle_deg_max": max((s.seam_cross_derivative_angle_deg_max for s in seams), default=0.0),
            "normal_angle_deg_mean": sum(s.seam_normal_angle_deg_mean for s in seams) / len(seams) if seams else 0.0,
            "normal_angle_deg_max": max((s.seam_normal_angle_deg_max for s in seams), default=0.0),
            "derivative_ratio_mean": sum(s.seam_derivative_ratio_mean for s in seams) / len(seams) if seams else 1.0,
            "derivative_ratio_max": max((s.seam_derivative_ratio_max for s in seams), default=1.0),
        },
        "parameter_quality": {
            "cv_v_along_u_line_mean": sum(s.fit_metrics["parameter_quality"]["cv_v_along_u_line_mean"] for s in slices) / len(slices),
            "cv_u_along_v_line_mean": sum(s.fit_metrics["parameter_quality"]["cv_u_along_v_line_mean"] for s in slices) / len(slices),
            "anisotropy_mean": sum(s.fit_metrics["parameter_quality"]["anisotropy_mean"] for s in slices) / len(slices),
            "anisotropy_min": min(s.fit_metrics["parameter_quality"]["anisotropy_min"] for s in slices),
            "orthogonality_mean": sum(s.fit_metrics["parameter_quality"]["orthogonality_mean"] for s in slices) / len(slices),
            "orthogonality_max": max(s.fit_metrics["parameter_quality"]["orthogonality_max"] for s in slices),
        },
        "phase2_boundary_conformance": {
            "inner": _boundary_conformance(torch.cat(inner_edges_world, dim=0), inner_reference, boundary_conformance_tolerance),
            "outer": _boundary_conformance(torch.cat(outer_edges_world, dim=0), outer_reference, boundary_conformance_tolerance),
        },
    }

    return AnnulusChartResult(
        component_id=component.component_id,
        origin_world=origin_world,
        origin_uv=origin_uv,
        segments=segments,
        slices=slices,
        seams=seams,
        topology_checks=topology_checks,
        chart_quality=chart_quality,
    )


def _measure_seams(slices: list[AnnulusChartSlice], sample_count: int) -> list[SeamDiagnostic]:
    """Along-seam and across-seam continuity between adjacent slices (§4.4 seam metric).

    Slice ``k``'s ``local_s=1`` edge and slice ``k+1``'s ``local_s=0`` edge
    are nominally the same physical curve; sampled independently from each
    patch's own fit, so every reported angle/gap directly measures
    continuity this v1 implementation does NOT enforce by construction
    (see the module docstring for why no orientation sign correction is
    needed for either ``Su`` or ``Sv`` in this specific O-grid).
    """

    torch = require_torch()
    t = torch.linspace(0.0, 1.0, sample_count, device=slices[0].surface.control_grid.device)
    seams = []
    n = len(slices)

    def _angle_deg(x: Any, y: Any, eps: float = _EPS) -> Any:
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(eps)
        yn = y / y.norm(dim=1, keepdim=True).clamp_min(eps)
        cos = (xn * yn).sum(dim=1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cos))

    for k in range(n):
        a, b = slices[k], slices[(k + 1) % n]
        edge_a = torch.stack([torch.ones_like(t), t], dim=1)  # a's local_s=1 edge
        edge_b = torch.stack([torch.zeros_like(t), t], dim=1)  # b's local_s=0 edge
        points_a, su_a, sv_a = a.surface.evaluate_with_derivatives(edge_a)
        points_b, su_b, sv_b = b.surface.evaluate_with_derivatives(edge_b)
        points_a, points_b = points_a.detach(), points_b.detach()
        su_a, sv_a, su_b, sv_b = su_a.detach(), sv_a.detach(), su_b.detach(), sv_b.detach()

        gap = (points_a - points_b).norm(dim=1)
        # Along-seam: Sv is the seam curve's own tangent on both sides.
        tangent_angle = _angle_deg(sv_a, sv_b)
        # Across-seam: Su is the cross-boundary derivative on both sides.
        cross_angle = _angle_deg(su_a, su_b)
        normal_a = torch.cross(su_a, sv_a, dim=1)
        normal_b = torch.cross(su_b, sv_b, dim=1)
        normal_angle = _angle_deg(normal_a, normal_b)
        mag_a, mag_b = su_a.norm(dim=1), su_b.norm(dim=1)
        ratio = mag_a / mag_b.clamp_min(_EPS)
        ratio = torch.maximum(ratio, 1.0 / ratio.clamp_min(_EPS))

        seams.append(
            SeamDiagnostic(
                slice_a=a.slice_index,
                slice_b=b.slice_index,
                sample_count=sample_count,
                mean_gap=float(gap.mean().cpu()),
                max_gap=float(gap.max().cpu()),
                seam_tangent_angle_deg_mean=float(tangent_angle.mean().cpu()),
                seam_tangent_angle_deg_max=float(tangent_angle.max().cpu()),
                seam_cross_derivative_angle_deg_mean=float(cross_angle.mean().cpu()),
                seam_cross_derivative_angle_deg_max=float(cross_angle.max().cpu()),
                seam_normal_angle_deg_mean=float(normal_angle.mean().cpu()),
                seam_normal_angle_deg_max=float(normal_angle.max().cpu()),
                seam_derivative_ratio_mean=float(ratio.mean().cpu()),
                seam_derivative_ratio_max=float(ratio.max().cpu()),
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
        "chart_quality": result.chart_quality,
        "iso_lines": annulus_iso_line_payload(result),
        "seams": [
            {
                "slice_a": s.slice_a, "slice_b": s.slice_b, "sample_count": s.sample_count,
                "mean_gap": s.mean_gap, "max_gap": s.max_gap,
                "seam_tangent_angle_deg_mean": s.seam_tangent_angle_deg_mean,
                "seam_tangent_angle_deg_max": s.seam_tangent_angle_deg_max,
                "seam_cross_derivative_angle_deg_mean": s.seam_cross_derivative_angle_deg_mean,
                "seam_cross_derivative_angle_deg_max": s.seam_cross_derivative_angle_deg_max,
                "seam_normal_angle_deg_mean": s.seam_normal_angle_deg_mean,
                "seam_normal_angle_deg_max": s.seam_normal_angle_deg_max,
                "seam_derivative_ratio_mean": s.seam_derivative_ratio_mean,
                "seam_derivative_ratio_max": s.seam_derivative_ratio_max,
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
