from __future__ import annotations

"""Worklog 71: region-owned full-evidence boundary topology reconstruction.

worklog 70's per-edge densification of the 3-4-point REPRESENTATIVE boundary
is retired as canonical (its diagnostic results stand, but
`interior_outside_boundary` stayed >10% in every patch that even
materialized a simple loop -- the original edge/wedge topology itself
cannot trace real evidence shape no matter how densely each wedge is
filled).

This module does not densify an ASSUMED 3-4-edge topology (worklog 61's
parametric chart boundary is not read anywhere in this module). Instead it
recovers ordered boundary TOPOLOGY directly from the region's own typed
half-edge evidence, and only THEN densifies each recovered edge with
region-owned full-cloud evidence:

  1. Seed nodes come from the EXISTING, unmodified typed candidate
     extractors: `torch_boundary_support_termination.
     extract_support_termination_candidates` (physical termination /
     reliability frontier / sampling gap) and `torch_world_space_boundary_
     halfedges.extract_world_space_boundary_halfedge_candidates` (crease /
     parallel-sheet conflict / ambiguous continuation -- computed by
     production but never merged into `VisibleSurfaceConstructionResult.
     boundary_halfedge_candidates`, a dead-code gap found this round and
     worked around here additively rather than by touching production).
  2. Seed-to-seed ADJACENCY reuses the EXISTING, unmodified production
     `torch_ordered_world_boundary_graph.build_boundary_compatibility` /
     `recover_ordered_boundary_components` -- the same tangent/normal-
     alignment + same-reason + representative-density-scale compatibility
     rule production already uses, at representative density where it was
     designed and tested. (An earlier revision of this module instead built
     a full evidence-density graph directly and connected every within-
     radius pair; measured on the real dataset this collapsed almost every
     region into one giant ~50-260-node branching blob, because dense
     evidence near ANY single seed candidate is mutually close to itself far
     more than it bridges to a neighbouring seed's evidence -- reusing the
     production seed-level graph, which was built and tuned for exactly this
     representative-density regime, avoids that failure mode entirely.)
  3. Only for a recovered `ordered_closed_loop` / `ordered_open_chain` seed
     component: EACH EDGE is densified with region-owned full evidence
     (`torch_region_owned_full_evidence.collect_region_owned_evidence`-style
     gate, worklog 67, unchanged upstream) using the same per-edge world-3D
     ownership + local-evidence-scale-binned insertion worklog 70 built
     (`torch_region_owned_boundary_materialization.materialize_dense_
     boundary`) -- reimplemented here (not imported) because worklog 70's
     version couples densification to `validate_simple_closed_loop`, which
     conflates 3D nonplanarity with self-intersection failure; this round
     needs those two kept separate (see `evaluate_closed_loop_geometry`
     below), and worklog 70's module/tests stay untouched as a record of
     that earlier round.

Every new densified vertex inherits its edge's ORIGINAL boundary_reason
(never invented, never mixed across edges of different reasons -- the
seed-level compatibility rule already requires exact reason match to accept
an edge in the first place). Branch/ambiguous seed junctions are typed
fail-closed before any evidence is even attached. Multiple independent
closed loops recovered within one region are returned separately -- never
merged, no outer/inner role inferred.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_boundary_self_intersection import (
    NONPLANAR_AMBIGUOUS,
    PlanarityReport,
    _project_to_local_plane,
    _segments_intersect,
    compute_planarity,
)
from osn_gs.surface.torch_ordered_world_boundary_graph import (
    OrderedBoundaryComponent,
    build_boundary_compatibility,
    recover_ordered_boundary_components,
)
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

STATE_CLOSED_LOOP_RECOVERED = "boundary_topology_closed_loop_recovered"
STATE_SELF_INTERSECTING = "boundary_topology_self_intersecting"
STATE_BRANCH_DETECTED = "boundary_topology_branch_detected"
STATE_AMBIGUOUS_JUNCTION = "boundary_topology_ambiguous_junction"
STATE_OPEN_FRAGMENT = "boundary_topology_open_fragment"
STATE_INSUFFICIENT_EVIDENCE = "boundary_topology_insufficient_evidence"


@dataclass(frozen=True)
class LoopGeometryReport:
    planarity: PlanarityReport | None
    crossing_check: str  # "checked" | "not_checked_nonplanar" | "not_checked_too_few_points"
    proper_crossing_count: int
    endpoint_touch_count: int
    collinear_overlap_count: int


@dataclass(frozen=True)
class DensifiedBoundarySegment:
    node_a: Any
    node_b: Any
    boundary_reason: str
    is_extension: bool


@dataclass(frozen=True)
class DensifiedBoundary:
    ordered_ids: tuple[Any, ...]
    ordered_positions: Any  # (K, 3)
    segments: tuple[DensifiedBoundarySegment, ...]
    extension_count: int
    seed_vertex_count: int


@dataclass(frozen=True)
class RegionBoundaryTopologyResult:
    region_id: int
    seed_component: OrderedBoundaryComponent | None
    densified: DensifiedBoundary | None
    geometry: LoopGeometryReport | None
    status: str
    reasons: tuple[str, ...]


def evaluate_closed_loop_geometry(ordered_positions: Sequence[tuple[float, float, float]]) -> LoopGeometryReport:
    """Planarity (`compute_planarity`, unmodified) is always reported. The
    2D-projected proper-crossing check is only PERFORMED when the loop is
    not `NONPLANAR_AMBIGUOUS` -- nonplanarity alone is disclosed, never
    treated as a crossing failure by itself (unlike `torch_boundary_self_
    intersection.validate_simple_closed_loop`, which conflates the two;
    this function deliberately does not reuse that conflated contract).
    Real 3D surface self-intersection is out of scope here, as everywhere
    else in this pipeline."""

    n = len(ordered_positions)
    if n < 3:
        return LoopGeometryReport(None, "not_checked_too_few_points", 0, 0, 0)
    planarity = compute_planarity(ordered_positions)
    if planarity.planarity_class == NONPLANAR_AMBIGUOUS:
        return LoopGeometryReport(planarity, "not_checked_nonplanar", 0, 0, 0)

    planar = _project_to_local_plane(ordered_positions)
    proper = touches = collinear = 0
    for i in range(n):
        a1, a2 = planar[i], planar[(i + 1) % n]
        for j in range(i + 1, n):
            shared = {i, (i + 1) % n} & {j, (j + 1) % n}
            if shared:
                continue
            b1, b2 = planar[j], planar[(j + 1) % n]
            kind, _ = _segments_intersect(a1, a2, b1, b2)
            if kind == "proper":
                proper += 1
            elif kind == "endpoint_touch":
                touches += 1
            elif kind == "collinear_overlap":
                collinear += 1
    return LoopGeometryReport(planarity, "checked", proper, touches, collinear)


def _point_segment_distance_3d(points: Any, a: Any, b: Any) -> Any:
    ab = b - a
    ab_len2 = (ab * ab).sum().clamp_min(_EPS)
    t = ((points - a) @ ab) / ab_len2
    t = t.clamp(0.0, 1.0)
    projection = a[None, :] + t[:, None] * ab[None, :]
    return (points - projection).norm(dim=-1)


def densify_ordered_boundary_with_evidence(
    ordered_ids: Sequence[Any],
    ordered_positions: Any,
    boundary_reason_by_id: dict[Any, str],
    *,
    closed: bool,
    evidence_ids: Sequence[Any],
    evidence_positions: Any,
    local_evidence_scale: float,
) -> DensifiedBoundary:
    """Per-EDGE (never a global hull) world-3D densification of a seed-level
    ordered boundary (from `recover_ordered_boundary_components`, closed or
    open) with region-owned full evidence -- same ownership/binning
    algorithm as worklog 70's `materialize_dense_boundary`, reimplemented
    here to stay decoupled from that function's internal (conflated)
    `validate_simple_closed_loop` call; the caller validates the result with
    `evaluate_closed_loop_geometry` instead."""

    torch = require_torch()
    n = int(ordered_positions.shape[0])
    edge_count = n if closed else max(0, n - 1)
    m = int(evidence_positions.shape[0]) if hasattr(evidence_positions, "shape") else len(evidence_positions)

    if edge_count == 0 or m == 0:
        segments = tuple(
            DensifiedBoundarySegment(ordered_ids[i], ordered_ids[(i + 1) % n], boundary_reason_by_id[ordered_ids[i]], False)
            for i in range(edge_count)
        )
        return DensifiedBoundary(tuple(ordered_ids), ordered_positions, segments, 0, n)

    scale = max(float(local_evidence_scale), _EPS)
    centroid = ordered_positions.mean(dim=0)

    edge_distances = torch.stack([
        _point_segment_distance_3d(evidence_positions, ordered_positions[i], ordered_positions[(i + 1) % n])
        for i in range(edge_count)
    ], dim=1)
    nearest_edge = edge_distances.argmin(dim=1)

    vertex_dist_from_centroid = (ordered_positions - centroid[None, :]).norm(dim=-1)
    evidence_dist_from_centroid = (evidence_positions - centroid[None, :]).norm(dim=-1)

    new_ids: list[Any] = []
    new_positions: list[Any] = []
    new_segments: list[DensifiedBoundarySegment] = []
    extension_count = 0
    for i in range(n):
        new_ids.append(ordered_ids[i])
        new_positions.append(ordered_positions[i])
        if not closed and i == n - 1:
            break  # open chain: last vertex has no outgoing edge
        a_pos, b_pos = ordered_positions[i], ordered_positions[(i + 1) % n]
        reason = boundary_reason_by_id[ordered_ids[i]]
        endpoint_max = max(float(vertex_dist_from_centroid[i]), float(vertex_dist_from_centroid[(i + 1) % n]))
        owned_mask = nearest_edge == i
        owned_indices = torch.nonzero(owned_mask, as_tuple=False).reshape(-1)

        selected_local: list[int] = []
        if int(owned_indices.numel()) > 0:
            owned_dist = evidence_dist_from_centroid[owned_indices]
            qualifies = owned_dist > (endpoint_max + scale)
            selected_local = owned_indices[qualifies].tolist()

        if selected_local:
            tangent = b_pos - a_pos
            tangent_len = float(tangent.norm().clamp_min(_EPS))
            bin_count = max(1, int(math.ceil(tangent_len / scale)))
            projections = [
                float(((evidence_positions[idx] - a_pos) @ tangent) / (tangent_len * tangent_len))
                for idx in selected_local
            ]
            best_per_bin: dict[int, tuple[float, int]] = {}
            for local_index, proj in zip(selected_local, projections):
                bin_index = min(bin_count - 1, max(0, int(proj * bin_count)))
                dist = float(evidence_dist_from_centroid[local_index])
                current = best_per_bin.get(bin_index)
                if current is None or dist > current[0]:
                    best_per_bin[bin_index] = (dist, local_index)
            ordered_local = [best_per_bin[b][1] for b in sorted(best_per_bin)]

            prev_id = ordered_ids[i]
            for local_index in ordered_local:
                ext_id = evidence_ids[local_index]
                new_ids.append(ext_id)
                new_positions.append(evidence_positions[local_index])
                new_segments.append(DensifiedBoundarySegment(prev_id, ext_id, reason, True))
                prev_id = ext_id
            new_segments.append(DensifiedBoundarySegment(prev_id, ordered_ids[(i + 1) % n], reason, True))
            extension_count += len(ordered_local)
        else:
            new_segments.append(DensifiedBoundarySegment(ordered_ids[i], ordered_ids[(i + 1) % n], reason, False))

    new_positions_tensor = torch.stack(new_positions, dim=0)
    return DensifiedBoundary(tuple(new_ids), new_positions_tensor, tuple(new_segments), extension_count, n)


def _walk_simple_path(start: str, adjacency: dict[str, set[str]], *, close_at: str | None) -> tuple[str, ...]:
    """Deterministic single walk from ``start`` following, at each step, the
    lexicographically-first neighbour other than the one just visited.
    ``recover_ordered_boundary_components`` (reused above for STATE
    classification only) returns its ``ordered_source_ids`` sorted by
    gaussian id for determinism, NOT as a geometric walk -- confirmed by
    direct construction: a 4-node square whose ids sort out of cyclic order
    comes back with a non-adjacent id sequence. This function recovers the
    REAL adjacency order from the same accepted compatibility edges, for a
    component already known (via that classification) to have degree <= 2
    everywhere with either zero (open chain) or two (closed loop) endpoints."""

    order = [start]
    prev: str | None = None
    current = start
    while True:
        candidates = sorted(adjacency[current] - ({prev} if prev is not None else set()))
        if not candidates:
            break
        next_node = candidates[0]
        if close_at is not None and next_node == close_at:
            break
        order.append(next_node)
        prev, current = current, next_node
    return tuple(order)


def reconstruct_region_boundary_topology(
    region_id: int,
    seed_candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    evidence_ids: Sequence[Any],
    evidence_positions: Any,
    local_evidence_scale: float,
) -> tuple[RegionBoundaryTopologyResult, ...]:
    """Top-level orchestrator for one region.

    ``seed_candidates`` must already be filtered/available for the whole
    scene (region filtering happens inside, matching `recover_ordered_
    boundary_components`'s own contract). ``evidence_ids``/``evidence_
    positions`` are this region's OWN full-cloud evidence only (worklog 67's
    gate, unchanged, applied upstream by the caller) -- never merged across
    regions. ``local_evidence_scale`` is a single already-established scale
    for this region (worklog 32's per-representative `mean_spacing`,
    aggregated by the caller); no new scale is invented here."""

    region_seed = tuple(c for c in seed_candidates if c.source_region_id == region_id)
    if not region_seed:
        return (RegionBoundaryTopologyResult(region_id, None, None, None, STATE_INSUFFICIENT_EVIDENCE, ("no_typed_boundary_evidence_for_region",)),)

    compatibility = build_boundary_compatibility(region_seed)
    seed_components = recover_ordered_boundary_components(region_seed, compatibility)
    candidate_by_id = {c.half_edge_id: c for c in region_seed}

    # `recover_ordered_boundary_components` returns `ordered_source_ids`
    # sorted by gaussian id for DETERMINISM, not as a geometric walk
    # (confirmed by direct construction -- see `_walk_simple_path`'s
    # docstring) -- rebuild real adjacency from the same accepted edges and
    # walk it ourselves for closed_loop/open_chain components, using the
    # production classification only for STATE.
    adjacency: dict[str, set[str]] = {c.half_edge_id: set() for c in region_seed}
    for edge in compatibility:
        if edge.decision == "accepted":
            adjacency[edge.source_half_edge_id].add(edge.target_half_edge_id)
            adjacency[edge.target_half_edge_id].add(edge.source_half_edge_id)

    results = []
    for component in seed_components:
        if component.ordering_state == "isolated_boundary_candidate":
            results.append(RegionBoundaryTopologyResult(region_id, component, None, None, STATE_INSUFFICIENT_EVIDENCE, ("isolated_seed_candidate_no_chain",)))
            continue
        if component.ordering_state == "branching_boundary_graph":
            results.append(RegionBoundaryTopologyResult(region_id, component, None, None, STATE_BRANCH_DETECTED, (f"branch_node_count={len(component.branch_node_ids)}",)))
            continue
        if component.ordering_state == "ambiguous_ordering":
            results.append(RegionBoundaryTopologyResult(region_id, component, None, None, STATE_AMBIGUOUS_JUNCTION, ("degree_pattern_not_simple_path_or_cycle",)))
            continue

        # ordered_closed_loop / ordered_open_chain: densify with THIS
        # region's own owned evidence only, in TRUE adjacency-walk order.
        closed = component.closed
        members = set(component.ordered_half_edge_ids)
        start = min(members)
        if closed:
            seed_ids = _walk_simple_path(start, adjacency, close_at=start)
        else:
            endpoint = next(m for m in sorted(members) if len(adjacency[m] & members) == 1)
            seed_ids = _walk_simple_path(endpoint, adjacency, close_at=None)
        seed_positions = require_torch().stack([
            require_torch().as_tensor(candidate_by_id[hid].world_position, dtype=evidence_positions.dtype, device=evidence_positions.device)
            for hid in seed_ids
        ], dim=0)
        boundary_reason_by_id = {hid: candidate_by_id[hid].boundary_reason for hid in seed_ids}

        if int(evidence_positions.shape[0]) == 0:
            results.append(RegionBoundaryTopologyResult(region_id, component, None, None, STATE_INSUFFICIENT_EVIDENCE, ("no_region_owned_full_evidence",)))
            continue

        densified = densify_ordered_boundary_with_evidence(
            seed_ids, seed_positions, boundary_reason_by_id, closed=closed,
            evidence_ids=evidence_ids, evidence_positions=evidence_positions,
            local_evidence_scale=local_evidence_scale,
        )

        if not closed:
            results.append(RegionBoundaryTopologyResult(region_id, component, densified, None, STATE_OPEN_FRAGMENT, ("no_evidence_backed_closure",)))
            continue

        geometry = evaluate_closed_loop_geometry([tuple(float(v) for v in row) for row in densified.ordered_positions.detach().cpu().tolist()])
        if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
            results.append(RegionBoundaryTopologyResult(
                region_id, component, densified, geometry, STATE_SELF_INTERSECTING,
                (f"proper_crossing_count={geometry.proper_crossing_count}",),
            ))
        else:
            results.append(RegionBoundaryTopologyResult(region_id, component, densified, geometry, STATE_CLOSED_LOOP_RECOVERED, ()))
    return tuple(results)
