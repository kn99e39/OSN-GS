from __future__ import annotations

"""Read-only adversarial diagnostics for consensus-aware surface regions.

This module deliberately does not construct an ordered boundary, a half-edge
graph, or a NURBS chart.  It makes the evidence a future boundary-graph
experiment may consume explicit and keeps Worklog 116's region candidates
review-only.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_manifold_affinity import (
    RELATION_CREASE,
    RELATION_PARALLEL_SEPARATE,
    RELATION_SAME_SURFACE,
    ManifoldAffinityGraph,
)
from osn_gs.surface.torch_gaussian_structural_reliability import INTRINSIC_REJECTED
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    CONTRADICTION_STABLE,
    MEMBER_AMBIGUOUS_UNASSIGNED,
    MEMBER_CONFLICT_BOUNDARY,
    MEMBER_REJECTED,
    REGION_SMALL_REVIEW,
    REGION_STABLE,
    RegionFormationResult,
    SurfaceRegionCandidate,
)

READINESS_READY = "ready_for_boundary_graph_experiment"
READINESS_REVIEW = "review_before_boundary_graph"
READINESS_INSUFFICIENT = "insufficient_region_consistency"
READINESS_REJECTED = "rejected_for_boundary_graph"


@dataclass(frozen=True)
class BoundaryInputReadiness:
    region_id: int
    stable_core_present: bool
    membership_ambiguity_ratio: float
    internal_contradiction_state: str
    conflict_edge_count: int
    rejected_neighbor_adjacency_count: int
    graph_connected: bool
    possible_open_boundary_evidence: bool
    possible_crease_boundary_evidence: bool
    possible_parallel_sheet_conflict: bool
    ordering_readiness: str
    boundary_extraction_readiness: str
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "coverage_semantics": "reliable_core_only",
            "full_surface_coverage_claimed": False,
            "stable_core_present": self.stable_core_present,
            "membership_ambiguity_ratio": self.membership_ambiguity_ratio,
            "internal_contradiction_state": self.internal_contradiction_state,
            "conflict_edge_count": self.conflict_edge_count,
            "rejected_neighbor_adjacency_count": self.rejected_neighbor_adjacency_count,
            "graph_connected": self.graph_connected,
            "possible_open_boundary_evidence": self.possible_open_boundary_evidence,
            "possible_crease_boundary_evidence": self.possible_crease_boundary_evidence,
            "possible_parallel_sheet_conflict": self.possible_parallel_sheet_conflict,
            "ordering_readiness": self.ordering_readiness,
            "boundary_extraction_readiness": self.boundary_extraction_readiness,
            "reasons": list(self.reasons),
        }


def _connected(nodes: set[int], same: list[set[int]]) -> bool:
    if not nodes:
        return False
    seen = {next(iter(nodes))}
    todo = list(seen)
    while todo:
        current = todo.pop()
        for neighbor in same[current]:
            if neighbor in nodes and neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return seen == nodes


def diagnose_boundary_input_readiness(
    result: RegionFormationResult,
    graph: ManifoldAffinityGraph,
    *,
    ids: Sequence[Any] | None = None,
) -> tuple[BoundaryInputReadiness, ...]:
    """Produce review-only readiness evidence without materializing a boundary.

    Stable IDs are accepted only to map candidate payload IDs back to graph
    indices.  No geometric membership is changed by this diagnostic.
    """
    count = len(result.node_region_id)
    ids = tuple(range(count)) if ids is None else tuple(ids)
    if len(ids) != count:
        raise ValueError("ids must match the region-formation node count")
    index_by_id = {value: index for index, value in enumerate(ids)}
    same = graph.same_surface_neighbors(count)
    crease = [set() for _ in range(count)]
    parallel = [set() for _ in range(count)]
    rejected_neighbors = [set() for _ in range(count)]
    for edge in graph.edges:
        if edge.manifold_relation == RELATION_CREASE:
            crease[edge.source].add(edge.target); crease[edge.target].add(edge.source)
        elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
            parallel[edge.source].add(edge.target); parallel[edge.target].add(edge.source)
        if edge.manifold_relation != RELATION_SAME_SURFACE and edge.endpoint_status.endswith("unreliable"):
            rejected_neighbors[edge.source].add(edge.target); rejected_neighbors[edge.target].add(edge.source)

    reports: list[BoundaryInputReadiness] = []
    for region in result.regions:
        nodes = {index_by_id[value] for value in region.member_ids}
        state = region.tangent_frame_consistency_stats.get("internal_state", "unknown")
        ambiguity = sum(
            result.node_membership_state[n] in (MEMBER_AMBIGUOUS_UNASSIGNED, MEMBER_CONFLICT_BOUNDARY)
            for n in nodes
        ) / max(len(nodes), 1)
        conflict = len(region.boundary_conflict_edge_ids)
        rejected = sum(len(rejected_neighbors[n]) for n in nodes)
        has_crease = any(crease[n] for n in nodes)
        has_parallel = any(parallel[n] for n in nodes)
        connected = _connected(nodes, same)
        reasons: list[str] = []
        if region.region_state == REGION_SMALL_REVIEW:
            readiness = READINESS_REVIEW; reasons.append("small_review_region")
        elif not region.core_member_ids or not connected or state != CONTRADICTION_STABLE:
            readiness = READINESS_INSUFFICIENT; reasons.append("core_or_connectivity_or_internal_consistency_missing")
        elif ambiguity > 0.0 or conflict > 0 or has_parallel:
            readiness = READINESS_REVIEW; reasons.append("membership_or_conflict_evidence_requires_review")
        elif region.region_state == REGION_STABLE:
            readiness = READINESS_READY; reasons.append("stable_connected_core_without_internal_conflict")
        else:
            readiness = READINESS_REVIEW; reasons.append("core_region_still_requires_boundary_review")
        reports.append(BoundaryInputReadiness(
            region_id=region.region_id,
            stable_core_present=bool(region.core_member_ids),
            membership_ambiguity_ratio=ambiguity,
            internal_contradiction_state=state,
            conflict_edge_count=conflict,
            rejected_neighbor_adjacency_count=rejected,
            graph_connected=connected,
            possible_open_boundary_evidence=not connected or ambiguity > 0.0,
            possible_crease_boundary_evidence=has_crease,
            possible_parallel_sheet_conflict=has_parallel,
            ordering_readiness=readiness,
            boundary_extraction_readiness=readiness,
            reasons=tuple(reasons),
        ))
    return tuple(reports)
