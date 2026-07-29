from __future__ import annotations

"""Review-only ordering of covariance-guided world-space boundary evidence.

No raster/PCA contour, loop materialization, builder adapter, or NURBS fitting
is performed here.  Components stay open/branching/ambiguous when the local
half-edge evidence cannot prove a deterministic closure.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


@dataclass(frozen=True)
class BoundaryCompatibilityEdge:
    source_half_edge_id: str
    target_half_edge_id: str
    normalized_distance: float
    tangent_alignment: float
    normal_alignment: float
    confidence: float
    decision: str
    reason: str


@dataclass(frozen=True)
class OrderedBoundaryComponent:
    component_id: str
    region_id: int
    ordered_half_edge_ids: tuple[str, ...]
    ordered_source_ids: tuple[Any, ...]
    ordering_state: str
    closed: bool
    branch_node_ids: tuple[Any, ...]
    boundary_reason_distribution: dict[str, int]
    confidence: float
    role_candidate: str
    coverage_semantics: str
    full_surface_coverage_claimed: bool
    unresolved_reasons: tuple[str, ...]


def build_boundary_compatibility(candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate]) -> tuple[BoundaryCompatibilityEdge, ...]:
    """Conservative local compatibility; no nearest-neighbor forced joins."""
    output = []
    for i, source in enumerate(candidates):
        for target in candidates[i + 1:]:
            if source.source_region_id != target.source_region_id:
                continue
            delta = sum((a - b) ** 2 for a, b in zip(source.world_position, target.world_position)) ** 0.5
            tangent = abs(sum(a * b for a, b in zip(source.boundary_tangent_direction if hasattr(source, 'boundary_tangent_direction') else source.boundary_direction, target.boundary_tangent_direction if hasattr(target, 'boundary_tangent_direction') else target.boundary_direction)))
            normal = abs(sum(a * b for a, b in zip(source.local_normal, target.local_normal)))
            accepted = delta <= 0.15 and tangent >= 0.5 and normal >= 0.8 and source.boundary_reason == target.boundary_reason
            output.append(BoundaryCompatibilityEdge(source.half_edge_id, target.half_edge_id, delta, tangent, normal, (tangent + normal) / 2.0, "accepted" if accepted else "review", "local_frame_and_reason_compatible" if accepted else "insufficient_local_chainability"))
    return tuple(output)


def recover_ordered_boundary_components(candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate], compatibility: Sequence[BoundaryCompatibilityEdge]) -> tuple[OrderedBoundaryComponent, ...]:
    by_id = {candidate.half_edge_id: candidate for candidate in candidates}
    adjacency = {key: set() for key in by_id}
    for edge in compatibility:
        if edge.decision == "accepted":
            adjacency[edge.source_half_edge_id].add(edge.target_half_edge_id)
            adjacency[edge.target_half_edge_id].add(edge.source_half_edge_id)
    output = []
    seen = set()
    for start in sorted(by_id):
        if start in seen:
            continue
        stack, members = [start], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current); members.append(current); stack.extend(sorted(adjacency[current] - seen))
        degrees = {member: len(adjacency[member]) for member in members}
        branches = tuple(by_id[item].source_gaussian_id for item, degree in degrees.items() if degree > 2)
        endpoints = [item for item, degree in degrees.items() if degree == 1]
        if len(members) == 1:
            state, closed = "isolated_boundary_candidate", False
        elif branches:
            state, closed = "branching_boundary_graph", False
        elif len(endpoints) == 0 and all(degree == 2 for degree in degrees.values()):
            state, closed = "ordered_closed_loop", True
        elif len(endpoints) == 2 and all(degree <= 2 for degree in degrees.values()):
            state, closed = "ordered_open_chain", False
        else:
            state, closed = "ambiguous_ordering", False
        ordered = tuple(sorted(members, key=lambda item: (by_id[item].source_gaussian_id, item)))
        reasons = {}
        for item in ordered:
            reasons[by_id[item].boundary_reason] = reasons.get(by_id[item].boundary_reason, 0) + 1
        role = "outer_boundary_candidate" if closed and reasons.get("observed_support_termination", 0) else ("crease_boundary_candidate" if reasons.get("crease_discontinuity", 0) else "open_boundary_candidate" if state == "ordered_open_chain" else "unresolved_boundary_role")
        output.append(OrderedBoundaryComponent(f"region:{by_id[start].source_region_id}:component:{min(ordered)}", by_id[start].source_region_id, ordered, tuple(by_id[item].source_gaussian_id for item in ordered), state, closed, branches, reasons, sum(by_id[item].confidence for item in ordered) / len(ordered), role, "reliable_core_only", False, () if state.startswith("ordered") else (state,)))
    return tuple(output)
