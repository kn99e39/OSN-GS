from __future__ import annotations

"""Region-local parametric chart boundary construction (worklog 61).

A visible NURBS chart's parametric boundary does not have to coincide with a
physical surface termination. This module builds a chart boundary purely
from a region's ALREADY-ACCEPTED topology (``SurfaceRegionCandidate.
internal_accepted_edge_ids``, computed by the existing manifold-affinity/
region-formation pipeline, unmodified) and the already-computed typed
boundary evidence (``WorldSpaceBoundaryHalfEdgeCandidate.boundary_reason``,
also unmodified) -- entirely independent of, and additive to, the
`eligible_closed_boundary` physical-termination path
(``torch_visible_boundary_region_status.py``, untouched by this module).

Boundary trace method (deliberately NOT a convex hull / bounding box / PCA
rectangle): starting from the region's own accepted-edge adjacency graph,
projected into a single canonical 2D tangent-plane frame
(``torch_canonical_region_tangent_frame.py``, unmodified), the OUTER FACE of
this planar graph is traced by a leftmost-turn walk -- the same
turn-maximizing technique ``torch_patch_boundary._trace_oriented_mask_loops``
already uses for a regular grid graph, generalized here to an arbitrary
graph via real-valued edge angles. This follows the graph's own (possibly
concave) shape exactly; it is never widened to a hull.

Every resulting boundary EDGE is classified into exactly one segment kind:

- ``physical_termination``: an endpoint carries an existing
  ``observed_support_termination`` candidate.
- ``crease``: an endpoint carries an existing ``crease_discontinuity``
  candidate.
- ``observation_frontier``: an endpoint carries any other typed boundary
  evidence (``reliability_frontier``/``unresolved_sampling_gap``/
  ``parallel_sheet_conflict``/``ambiguous_continuation``).
- ``partition_seam``: neither endpoint carries typed boundary evidence --
  this edge exists ONLY because the region's own accepted topology ends
  there, not because anything was physically observed to terminate. Never
  relabeled as ``physical_termination``.

Only a single validated simple closed, non-branching, non-self-intersecting
loop is ever eligible (``eligible_parametric_chart_boundary``). Every other
outcome is a stable typed rejection -- no forced closure, no gap
interpolation, no shape-specific dispatch.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_canonical_region_tangent_frame import CanonicalRegionTangentFrame
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationResult
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate

SCHEMA_VERSION = "region_parametric_chart_boundary_worklog61_v1"

STATUS_ELIGIBLE_PARAMETRIC_CHART = "eligible_parametric_chart_boundary"
STATUS_INSUFFICIENT_TOPOLOGY = "parametric_chart_insufficient_topology"
STATUS_TOPOLOGY_OPEN_OR_BRANCHING = "parametric_chart_topology_open_or_branching"
STATUS_SELF_INTERSECTING = "parametric_chart_self_intersection_failed"
STATUS_NO_TANGENT_FRAME = "parametric_chart_no_canonical_tangent_frame"
PARAMETRIC_CHART_STATES = {
    STATUS_ELIGIBLE_PARAMETRIC_CHART,
    STATUS_INSUFFICIENT_TOPOLOGY,
    STATUS_TOPOLOGY_OPEN_OR_BRANCHING,
    STATUS_SELF_INTERSECTING,
    STATUS_NO_TANGENT_FRAME,
}

SEGMENT_PHYSICAL_TERMINATION = "physical_termination"
SEGMENT_CREASE = "crease"
SEGMENT_OBSERVATION_FRONTIER = "observation_frontier"
SEGMENT_PARTITION_SEAM = "partition_seam"
SEGMENT_KINDS = {
    SEGMENT_PHYSICAL_TERMINATION, SEGMENT_CREASE, SEGMENT_OBSERVATION_FRONTIER, SEGMENT_PARTITION_SEAM,
}

_REASON_TO_SEGMENT_KIND = {
    "observed_support_termination": SEGMENT_PHYSICAL_TERMINATION,
    "crease_discontinuity": SEGMENT_CREASE,
    "reliability_frontier": SEGMENT_OBSERVATION_FRONTIER,
    "unresolved_sampling_gap": SEGMENT_OBSERVATION_FRONTIER,
    "parallel_sheet_conflict": SEGMENT_OBSERVATION_FRONTIER,
    "ambiguous_continuation": SEGMENT_OBSERVATION_FRONTIER,
}
# Precedence when an endpoint's evidence maps to more than one candidate kind
# (two different candidates at the same node): a genuine physical
# termination always wins disclosure over a softer frontier/crease read, and
# crease wins over a bare frontier -- never the reverse.
_SEGMENT_KIND_PRIORITY = (SEGMENT_PHYSICAL_TERMINATION, SEGMENT_CREASE, SEGMENT_OBSERVATION_FRONTIER)


@dataclass(frozen=True)
class ParametricChartBoundarySegment:
    node_a: Any
    node_b: Any
    segment_kind: str

    def __post_init__(self) -> None:
        if self.segment_kind not in SEGMENT_KINDS:
            raise ValueError(f"Unknown parametric chart segment kind: {self.segment_kind!r}")

    def payload(self) -> dict[str, Any]:
        return {"node_a": self.node_a, "node_b": self.node_b, "segment_kind": self.segment_kind}


@dataclass(frozen=True)
class RegionParametricChartBoundary:
    region_id: int
    status: str
    reason: str
    ordered_node_ids: tuple[Any, ...]
    segments: tuple[ParametricChartBoundarySegment, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in PARAMETRIC_CHART_STATES:
            raise ValueError(f"Unknown parametric chart status: {self.status!r}")

    def segment_kind_counts(self) -> dict[str, int]:
        counts = {kind: 0 for kind in SEGMENT_KINDS}
        for segment in self.segments:
            counts[segment.segment_kind] += 1
        return counts

    def payload(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "status": self.status,
            "reason": self.reason,
            "ordered_node_ids": list(self.ordered_node_ids),
            "segments": [item.payload() for item in self.segments],
            "segment_kind_counts": self.segment_kind_counts(),
            "schema_version": self.schema_version,
        }


def _edge_turn(incoming: tuple[float, float], outgoing: tuple[float, float]) -> float:
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    return math.atan2(cross, dot)


def _trace_leftmost_turn_boundary(
    uv: dict[Any, tuple[float, float]], adjacency: dict[Any, set[Any]],
) -> tuple[Any, ...] | None:
    """Trace the outer face of a planar straight-line graph via a
    leftmost-turn walk, generalizing
    ``torch_patch_boundary._trace_oriented_mask_loops`` from a regular grid
    to an arbitrary graph. Returns the ordered, unique node-id sequence
    (no closing duplicate), or ``None`` if the walk cannot close into a
    simple non-branching cycle within a bounded number of steps.
    """

    if len(uv) < 3:
        return None
    start = min(uv, key=lambda node: (uv[node][0], uv[node][1]))
    cur = start
    incoming = (0.0, 1.0)  # pretend we arrived moving "up" into the leftmost point
    prev = None
    order: list[Any] = [start]
    visited_directed_edges: set[tuple[Any, Any]] = set()
    budget = 2 * len(uv) + 2
    for _ in range(budget):
        candidates = [node for node in adjacency.get(cur, ()) if node != prev]
        if not candidates:
            candidates = [node for node in adjacency.get(cur, ()) if node == prev]
        if not candidates:
            return None  # dead end / pendant node -- not a closed boundary
        ranked = []
        for candidate in candidates:
            direction = (uv[candidate][0] - uv[cur][0], uv[candidate][1] - uv[cur][1])
            length = math.hypot(*direction)
            if length <= 1e-12:
                continue
            direction = (direction[0] / length, direction[1] / length)
            ranked.append((-_edge_turn(incoming, direction), candidate, direction))
        if not ranked:
            return None
        _, next_node, next_direction = min(ranked)
        directed_edge = (cur, next_node)
        if directed_edge in visited_directed_edges:
            return None  # revisiting a directed edge before closing -> branch/degenerate
        visited_directed_edges.add(directed_edge)
        if next_node == start:
            return tuple(order)
        if next_node in order:
            return None  # revisits a non-start vertex -> not a simple cycle
        order.append(next_node)
        prev, cur, incoming = cur, next_node, next_direction
    return None


def _segment_kind_for_endpoints(
    node_a: Any, node_b: Any, reason_by_node: dict[Any, tuple[str, ...]],
) -> str:
    found: set[str] = set()
    for node in (node_a, node_b):
        for reason in reason_by_node.get(node, ()):
            kind = _REASON_TO_SEGMENT_KIND.get(reason)
            if kind is not None:
                found.add(kind)
    for kind in _SEGMENT_KIND_PRIORITY:
        if kind in found:
            return kind
    return SEGMENT_PARTITION_SEAM


def construct_region_parametric_chart_boundaries(
    positions: Any,
    ids: Sequence[Any],
    regions: RegionFormationResult,
    canonical_frames: Sequence[CanonicalRegionTangentFrame | None],
    halfedge_candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
) -> tuple[RegionParametricChartBoundary, ...]:
    """One ``RegionParametricChartBoundary`` per region -- every region is
    accounted for, never silently skipped."""

    index_by_id = {item: i for i, item in enumerate(ids)}
    reason_by_node: dict[Any, tuple[str, ...]] = {}
    for candidate in halfedge_candidates:
        reason_by_node.setdefault(candidate.source_gaussian_id, ())
        reason_by_node[candidate.source_gaussian_id] = reason_by_node[candidate.source_gaussian_id] + (
            candidate.boundary_reason,
        )

    frame_by_region: dict[int, CanonicalRegionTangentFrame] = {}
    for frame in canonical_frames:
        if frame is not None and frame.region_id not in frame_by_region:
            frame_by_region[frame.region_id] = frame

    results: list[RegionParametricChartBoundary] = []
    for region in regions.regions:
        frame = frame_by_region.get(region.region_id)
        if frame is None or frame.gaussian_id not in index_by_id:
            results.append(RegionParametricChartBoundary(
                region.region_id, STATUS_NO_TANGENT_FRAME, "no_canonical_tangent_frame_for_region", (), (),
            ))
            continue

        origin = positions[index_by_id[frame.gaussian_id]]
        axis_u, axis_v = frame.tangent_axis_0, frame.tangent_axis_1
        member_set = set(region.member_ids)
        uv: dict[Any, tuple[float, float]] = {}
        for member in region.member_ids:
            member_index = index_by_id.get(member)
            if member_index is None:
                continue
            offset = positions[member_index] - origin
            uv[member] = (float(offset @ axis_u), float(offset @ axis_v))

        adjacency: dict[Any, set[Any]] = {node: set() for node in uv}
        for a, b in region.internal_accepted_edge_ids:
            if a in uv and b in uv:
                adjacency[a].add(b)
                adjacency[b].add(a)

        connected_nodes = {node for node in uv if adjacency.get(node)}
        if len(connected_nodes) < 3:
            results.append(RegionParametricChartBoundary(
                region.region_id, STATUS_INSUFFICIENT_TOPOLOGY,
                "fewer_than_3_accepted_edge_connected_members", (), (),
            ))
            continue

        # Restrict to the connected component containing the extremal
        # (leftmost, tie-broken lowest) node -- deterministic, no arbitrary
        # component choice.
        seed = min(connected_nodes, key=lambda node: (uv[node][0], uv[node][1]))
        stack, component = [seed], {seed}
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in component and neighbor in connected_nodes:
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) < 3:
            results.append(RegionParametricChartBoundary(
                region.region_id, STATUS_INSUFFICIENT_TOPOLOGY,
                "fewer_than_3_members_in_largest_accepted_component", (), (),
            ))
            continue

        component_uv = {node: uv[node] for node in component}
        component_adjacency = {node: {n for n in adjacency[node] if n in component} for node in component}
        ordered = _trace_leftmost_turn_boundary(component_uv, component_adjacency)
        if ordered is None:
            results.append(RegionParametricChartBoundary(
                region.region_id, STATUS_TOPOLOGY_OPEN_OR_BRANCHING,
                "leftmost_turn_walk_did_not_close_into_a_simple_cycle", (), (),
            ))
            continue

        world_points = [
            tuple(float(v) for v in positions[index_by_id[node]].detach().cpu().tolist()) for node in ordered
        ]
        report = validate_simple_closed_loop(world_points)
        if not report.is_simple_polygon:
            results.append(RegionParametricChartBoundary(
                region.region_id, STATUS_SELF_INTERSECTING,
                "closed_loop_failed_self_intersection_check", ordered, (),
            ))
            continue

        segments = tuple(
            ParametricChartBoundarySegment(
                ordered[i], ordered[(i + 1) % len(ordered)],
                _segment_kind_for_endpoints(ordered[i], ordered[(i + 1) % len(ordered)], reason_by_node),
            )
            for i in range(len(ordered))
        )
        results.append(RegionParametricChartBoundary(
            region.region_id, STATUS_ELIGIBLE_PARAMETRIC_CHART,
            "validated_topology_supported_chart_boundary", ordered, segments,
        ))

    return tuple(results)
