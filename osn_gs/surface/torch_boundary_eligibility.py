from __future__ import annotations

"""Boundary-leaf support ELIGIBILITY classification (ACTIVE_OBSERVED /
UNCERTAIN / INACTIVE / COMPLEX), promoted to production per
``docs/worklogs/45-48``.

Phase 1's ``plane_aabb_intersection_polygon`` union (Stage 1-C) uses each
leaf's FULL plane-AABB polygon regardless of actual point occupancy, which
systematically over-estimates support on BOUNDARY leaves (root-caused in
worklog 45). A convex-hull clip of boundary-leaf polygons fixes most of that
(worklog 46), but a plain hull clip alone under-covers leaves whose nearby
Gaussians are genuinely sparse -- which is not necessarily evidence of a
false boundary; it may just be thin observed data. OSN-GS already
distinguishes certain/observed from uncertain/inferred evidence elsewhere
([[project_osn_gs_direction]]), so this module extends that distinction to
per-leaf boundary support: each boundary leaf is classified into one of four
tiers from LOCAL, PRE-EXISTING Phase 1 signals only (no GT, no scene-specific
tuning), and only ``ACTIVE_OBSERVED`` leaves get the convex-hull clip;
``UNCERTAIN``/``COMPLEX`` leaves keep their original (unclipped) polygon;
``INACTIVE`` leaves are dropped entirely.

This is a STATIC, deterministic, per-leaf classifier -- explicitly NOT
hysteresis in the state-machine sense (no memory of a leaf's previous
classification across iterations); it is a two-threshold TERNARY
CLASSIFICATION BAND on one static pass.

Every vote is recorded individually (``plane_residual_vote`` /
``normal_consistency_vote`` / ``neighbor_continuity_vote`` +
``final_class`` + ``class_transition_reason``) rather than folded into a
hidden weighted score, so any classification is auditable after the fact.

Validated end-to-end (worklog 48, not just at the raster-mask level): fed
into the REAL ``build_annulus_chart`` fit and scored with production
``ground_truth_metrics``/``patch_union_metrics`` on all 4 annulus scenes.
``active_only`` alone regresses chamfer_rms (coverage loss costs more than
the false-fill gain helps) -- not used standalone. The
``active_plus_uncertain_plus_complex`` view (this module's
``build_eligibility_filtered_coarse_mask``) is what ``extract_component_
boundary`` now uses as its coarse-support mask: chamfer_rms at parity with
the old plain-union mask or better, false_fill cut 54-82% on 3/4 scenes.
COMPLEX leaves are included (not just ``active_plus_uncertain``) so a
Phase-1 ``STATE_COMPLEX`` leaf's contribution is unchanged from the
pre-eligibility behavior -- worklog 48's 4 validation scenes had zero
COMPLEX leaves, so this choice reproduces the exact validated numbers on
those scenes while not silently dropping COMPLEX-leaf support on scenes
that do have them (untested territory, so left at its old behavior rather
than guessed).
"""

import math
from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_boundary_refinement import sample_nn_spacings
from osn_gs.surface.torch_surface_components import SurfaceComponent
from osn_gs.surface.torch_voxel_hierarchy import (
    FACE_INTERIOR,
    STATE_ACTIVE,
    STATE_COMPLEX,
    STATE_EMPTY,
    STATE_INACTIVE,
    TorchVoxelGaussianHierarchy,
    compute_leaf_face_adjacency,
    plane_aabb_intersection_polygon,
    rasterize_convex_polygon_uv,
)
from osn_gs.utils.torch_ops import require_torch

ACTIVE_OBSERVED = "ACTIVE_OBSERVED"
UNCERTAIN = "UNCERTAIN"
INACTIVE = "INACTIVE"
COMPLEX = "COMPLEX"

DEFAULT_ELIGIBILITY_THRESHOLDS = {
    "spacing_ratio_low": 0.35,   # <= this -> spacing-based "active" candidate
    "spacing_ratio_high": 0.70,  # >= this -> spacing-based "inactive" candidate
    "plane_residual_normalized_high": 0.15,
    "normal_consistency_angle_high_deg": 25.0,
    "neighbor_phase1_active_ratio_low": 0.34,
}


@dataclass
class LeafBoundaryProvenance:
    """Which KIND of boundary this leaf's face contacts actually reflect --
    conflated in the raw ``is_boundary_leaf`` flag, which is why EVERY leaf
    on a flat (z=0) synthetic scene reads as "boundary" (every leaf touches
    the z-axis ROOT AABB face, unrelated to real x/y support edges).

    ``is_hole_boundary_leaf`` is deliberately NOT included: whether a leaf
    borders a hole vs. the outer edge is a Phase-2 (density-threshold +
    loop-labeling) concept, not decidable from Phase-1 leaf adjacency alone.
    """

    is_root_boundary_leaf: bool          # touches the analysis-domain AABB itself (no neighbor at all)
    is_inactive_neighbor_leaf: bool      # touches a REAL neighbor leaf classified inactive/empty
    is_cross_component_boundary_leaf: bool  # touches an active/complex leaf belonging to a DIFFERENT component


@dataclass
class LeafEligibilityResult:
    leaf_id: str
    spacing_ratio: float
    rho_u: float
    rho_v: float
    plane_residual_world: float
    plane_residual_normalized: float
    normal_consistency: float  # mean of |dot| (sign-ambiguity-safe), NaN if no interior neighbors
    normal_neighbor_count: int
    neighbor_phase1_active_ratio: float
    primary_spacing_class: str          # "active_candidate" | "uncertain_candidate" | "inactive_candidate"
    plane_residual_vote: str            # "good" | "bad"
    normal_consistency_vote: str        # "good" | "bad" | "neutral" (neutral = no neighbor data, not counted)
    neighbor_continuity_vote: str       # "good" | "bad"
    final_class: str
    class_transition_reason: str
    provenance: LeafBoundaryProvenance


def _convex_hull_2d(points: Any) -> Any:
    """Andrew's monotone chain, CCW-ordered. Torch-only, no new dependency."""

    torch = require_torch()
    pts = sorted(set(tuple(p) for p in points.tolist()))
    if len(pts) < 3:
        return torch.tensor(pts, dtype=points.dtype)

    def cross(o: tuple, a: tuple, b: tuple) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return torch.tensor(hull, dtype=points.dtype)


def _sutherland_hodgman_clip(subject: Any, clip_polygon: Any) -> Any:
    """Intersection of two convex polygons (standard Sutherland-Hodgman).
    ``clip_polygon`` is re-wound CCW internally for a consistent half-plane
    test; ``subject``'s own winding does not matter."""

    torch = require_torch()
    clip_list = clip_polygon.tolist()
    area2 = sum(
        clip_list[i][0] * clip_list[(i + 1) % len(clip_list)][1] - clip_list[(i + 1) % len(clip_list)][0] * clip_list[i][1]
        for i in range(len(clip_list))
    )
    if area2 < 0:
        clip_list = clip_list[::-1]

    def inside(p: tuple, a: tuple, b: tuple) -> bool:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def intersect(p1: tuple, p2: tuple, a: tuple, b: tuple) -> tuple:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = subject.tolist()
    for i in range(len(clip_list)):
        a, b = clip_list[i], clip_list[(i + 1) % len(clip_list)]
        if not output:
            break
        input_list, output = output, []
        for j in range(len(input_list)):
            cur, prev = input_list[j], input_list[j - 1]
            cur_in, prev_in = inside(cur, a, b), inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(intersect(prev, cur, a, b))
    return torch.tensor(output, dtype=subject.dtype) if output else torch.empty((0, 2), dtype=subject.dtype)


def _polygon_area_2d(polygon: Any) -> float:
    torch = require_torch()
    if int(polygon.shape[0]) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    x2, y2 = torch.roll(x, -1), torch.roll(y, -1)
    return float(0.5 * (x * y2 - x2 * y).sum().abs())


def _axis_nn_spacing(uv: Any) -> tuple[float, float]:
    """Per-axis (u, v) component of each point's own 2D nearest-neighbor
    displacement, median over points -- avoids collapsing an elongated
    leaf's anisotropic sampling into one scalar spacing number."""

    torch = require_torch()
    n = int(uv.shape[0])
    if n < 2:
        return float("inf"), float("inf")
    d = torch.cdist(uv, uv)
    d.fill_diagonal_(float("inf"))
    nearest = uv[d.argmin(dim=1)]
    diffs = (uv - nearest).abs()
    return float(diffs[:, 0].median()), float(diffs[:, 1].median())


def compute_leaf_boundary_provenance(
    leaf_id: str, adjacency: dict[str, dict[str, Any]], component_member_ids: set[str],
) -> LeafBoundaryProvenance:
    contacts = adjacency.get(leaf_id, {}).get("contacts", [])
    is_root = any(c["neighbor_id"] is None for c in contacts)
    is_inactive_neighbor = any(
        c["neighbor_id"] is not None and c.get("neighbor_state") in (STATE_INACTIVE, STATE_EMPTY) for c in contacts
    )
    is_cross_component = any(
        c["neighbor_id"] is not None and c["neighbor_id"] not in component_member_ids
        and c.get("neighbor_state") in (STATE_ACTIVE, STATE_COMPLEX)
        for c in contacts
    )
    return LeafBoundaryProvenance(is_root, is_inactive_neighbor, is_cross_component)


def compute_leaf_eligibility(
    leaf: Any,
    points: Any,
    frame: Any,
    polygon_uv: Any,
    adjacency: dict[str, dict[str, Any]],
    node_by_id: dict[str, Any],
    component_member_ids: set[str],
    thresholds: dict[str, float] = DEFAULT_ELIGIBILITY_THRESHOLDS,
) -> LeafEligibilityResult:
    torch = require_torch()
    provenance = compute_leaf_boundary_provenance(leaf.leaf_id if hasattr(leaf, "leaf_id") else leaf.node_id, adjacency, component_member_ids)

    if leaf.state == STATE_COMPLEX:
        return LeafEligibilityResult(
            leaf_id=leaf.node_id, spacing_ratio=float("nan"), rho_u=float("nan"), rho_v=float("nan"),
            plane_residual_world=float("nan"), plane_residual_normalized=float("nan"),
            normal_consistency=float("nan"), normal_neighbor_count=0, neighbor_phase1_active_ratio=float("nan"),
            primary_spacing_class="n/a", plane_residual_vote="n/a", normal_consistency_vote="n/a",
            neighbor_continuity_vote="n/a", final_class=COMPLEX, class_transition_reason="phase1_complex_state",
            provenance=provenance,
        )

    member_points_world = points[leaf.gaussian_indices] if leaf.gaussian_indices is not None else points[:0]
    member_uv = frame.apply(member_points_world, clamp=False) if int(member_points_world.shape[0]) else torch.empty((0, 2))

    lo_u, hi_u = float(polygon_uv[:, 0].min()), float(polygon_uv[:, 0].max())
    lo_v, hi_v = float(polygon_uv[:, 1].min()), float(polygon_uv[:, 1].max())
    L_u, L_v = max(hi_u - lo_u, 1e-9), max(hi_v - lo_v, 1e-9)
    cell_scale = math.sqrt(L_u * L_v)

    d_nn_u, d_nn_v = _axis_nn_spacing(member_uv)
    rho_u, rho_v = d_nn_u / L_u, d_nn_v / L_v
    spacing = float(sample_nn_spacings(member_uv).median()) if int(member_uv.shape[0]) >= 2 else float("inf")
    spacing_ratio = spacing / cell_scale

    if leaf.plane is not None and int(member_points_world.shape[0]):
        residuals = (member_points_world - leaf.plane.centroid) @ leaf.plane.normal
        plane_residual_world = float(residuals.square().mean().sqrt())
    else:
        plane_residual_world = float("inf")
    plane_residual_normalized = plane_residual_world / cell_scale

    contacts = adjacency.get(leaf.node_id, {}).get("contacts", [])
    # Real spatial contacts only -- excludes root-AABB-boundary contacts
    # (neighbor_id=None), which on a flat z=0 scene are a domain-box
    # artifact (every leaf touches both z faces), not spatial information;
    # including them in the denominator would dilute this ratio for every
    # leaf regardless of real x/y connectivity.
    real_contacts = [c for c in contacts if c["neighbor_id"] is not None]
    interior_contacts = [c for c in real_contacts if c["classification"] == FACE_INTERIOR]
    # neighbor_phase1_active_ratio uses ONLY the pre-existing Phase 1 leaf
    # STATE (active/inactive/complex/empty) of face-contact neighbors -- NOT
    # this eligibility classifier's own output, to avoid a circular
    # definition (a leaf's class depending on neighbors' not-yet-computed class).
    phase1_active_contacts = [c for c in interior_contacts if c.get("neighbor_state") == STATE_ACTIVE]
    neighbor_phase1_active_ratio = len(phase1_active_contacts) / len(real_contacts) if real_contacts else 0.0

    dots = []
    for c in interior_contacts:
        neighbor = node_by_id.get(c["neighbor_id"])
        if neighbor is not None and neighbor.plane is not None and leaf.plane is not None:
            dots.append(abs(float((leaf.plane.normal @ neighbor.plane.normal).clamp(-1.0, 1.0))))
    normal_neighbor_count = len(dots)
    normal_consistency = sum(dots) / len(dots) if dots else float("nan")

    # --- Votes (explicit trace, not a weighted-sum score -- avoids hidden
    # magic weights; each vote is independently readable). ---
    if spacing_ratio <= thresholds["spacing_ratio_low"]:
        primary_spacing_class = "active_candidate"
    elif spacing_ratio >= thresholds["spacing_ratio_high"]:
        primary_spacing_class = "inactive_candidate"
    else:
        primary_spacing_class = "uncertain_candidate"

    plane_residual_vote = "bad" if plane_residual_normalized > thresholds["plane_residual_normalized_high"] else "good"
    if normal_neighbor_count == 0:
        normal_consistency_vote = "neutral"
    else:
        angle_deg = math.degrees(math.acos(min(1.0, max(-1.0, normal_consistency))))
        normal_consistency_vote = "bad" if angle_deg > thresholds["normal_consistency_angle_high_deg"] else "good"
    neighbor_continuity_vote = "bad" if neighbor_phase1_active_ratio < thresholds["neighbor_phase1_active_ratio_low"] else "good"

    bad_votes = sum(1 for v in (plane_residual_vote, normal_consistency_vote, neighbor_continuity_vote) if v == "bad")

    # --- Decision table (explicit, conservative toward UNCERTAIN -- INACTIVE
    # must require sparse spacing AND corroborating secondary evidence
    # together, never spacing alone, so genuine thin structure isn't
    # discarded on one signal). ---
    if primary_spacing_class == "active_candidate":
        if bad_votes == 0:
            final_class, reason = ACTIVE_OBSERVED, "dense_and_consistent"
        else:
            bad_names = [n for n, v in (("plane_residual", plane_residual_vote), ("normal_consistency", normal_consistency_vote), ("neighbor_continuity", neighbor_continuity_vote)) if v == "bad"]
            final_class, reason = UNCERTAIN, f"downgraded_from_active_by_{'_'.join(bad_names)}"
    elif primary_spacing_class == "inactive_candidate":
        if bad_votes >= 2:
            final_class, reason = INACTIVE, "sparse_and_multiple_conflicting_signals"
        else:
            final_class, reason = UNCERTAIN, "sparse_but_insufficient_corroborating_evidence_for_inactive"
    else:
        final_class, reason = UNCERTAIN, "ambiguous_spacing_signal"

    return LeafEligibilityResult(
        leaf_id=leaf.node_id, spacing_ratio=spacing_ratio, rho_u=rho_u, rho_v=rho_v,
        plane_residual_world=plane_residual_world, plane_residual_normalized=plane_residual_normalized,
        normal_consistency=normal_consistency, normal_neighbor_count=normal_neighbor_count,
        neighbor_phase1_active_ratio=neighbor_phase1_active_ratio,
        primary_spacing_class=primary_spacing_class, plane_residual_vote=plane_residual_vote,
        normal_consistency_vote=normal_consistency_vote, neighbor_continuity_vote=neighbor_continuity_vote,
        final_class=final_class, class_transition_reason=reason, provenance=provenance,
    )


def build_eligibility_filtered_coarse_mask(
    component: SurfaceComponent,
    hierarchy: TorchVoxelGaussianHierarchy,
    points: Any,
    frame: Any,
    resolution: int,
    min_hull_points: int = 4,
    thresholds: dict[str, float] = DEFAULT_ELIGIBILITY_THRESHOLDS,
) -> Any:
    """Replaces the plain per-leaf polygon union (Stage 1-C, unfiltered)
    with the ``active_plus_uncertain_plus_complex`` eligibility view
    (worklog 48): interior leaves and COMPLEX-state boundary leaves keep
    their original plane-AABB polygon unchanged; ``ACTIVE_OBSERVED``
    boundary leaves get the convex-hull clip of their own member Gaussians;
    ``UNCERTAIN`` boundary leaves keep their original polygon (still
    included -- only ``INACTIVE`` leaves are dropped, and dropping requires
    sparse spacing AND >= 2 corroborating bad signals, so this is a
    conservative filter, not an aggressive one).
    """

    torch = require_torch()
    node_by_id = {node.node_id: node for node in hierarchy.nodes}
    boundary_ids = set(component.boundary_leaf_ids)
    member_ids = set(component.member_leaf_ids)
    adjacency = compute_leaf_face_adjacency(hierarchy, degenerate_axis_tolerant=True)

    mask = torch.zeros((resolution, resolution), dtype=torch.bool)
    for leaf_id in component.member_leaf_ids:
        leaf = node_by_id[leaf_id]
        if leaf.plane is None:
            continue
        polygon_world = plane_aabb_intersection_polygon(leaf.plane.centroid, leaf.plane.normal, leaf.aabb_min, leaf.aabb_max)
        if int(polygon_world.shape[0]) < 3:
            continue
        polygon_uv = frame.apply(polygon_world, clamp=False)
        is_boundary = leaf_id in boundary_ids
        point_count = int(leaf.gaussian_indices.shape[0]) if leaf.gaussian_indices is not None else 0

        if not is_boundary:
            mask = mask | rasterize_convex_polygon_uv(polygon_uv, resolution)
            continue

        eligibility = compute_leaf_eligibility(leaf, points, frame, polygon_uv, adjacency, node_by_id, member_ids, thresholds)
        if eligibility.final_class == INACTIVE:
            continue  # dropped entirely -- the only class that contributes nothing
        if eligibility.final_class == ACTIVE_OBSERVED and point_count >= min_hull_points:
            member_uv = frame.apply(points[leaf.gaussian_indices], clamp=False)
            hull_uv = _convex_hull_2d(member_uv)
            clipped_uv = polygon_uv
            if int(hull_uv.shape[0]) >= 3:
                candidate = _sutherland_hodgman_clip(polygon_uv, hull_uv)
                if int(candidate.shape[0]) >= 3:
                    clipped_uv = candidate
            mask = mask | rasterize_convex_polygon_uv(clipped_uv, resolution)
        else:
            # UNCERTAIN or COMPLEX (or ACTIVE_OBSERVED with too few points
            # to hull-clip reliably): keep the original, unclipped polygon.
            mask = mask | rasterize_convex_polygon_uv(polygon_uv, resolution)
    return mask
