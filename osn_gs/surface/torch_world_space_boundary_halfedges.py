from __future__ import annotations

"""Experimental, unordered world-space boundary half-edge candidates.

Consumes accepted local region topology only.  It intentionally does not
materialize a boundary chain, loop, chart, or NURBS surface.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_manifold_affinity import RELATION_CREASE, RELATION_PARALLEL_SEPARATE, ManifoldAffinityGraph
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationResult


@dataclass(frozen=True)
class WorldSpaceBoundaryHalfEdgeCandidate:
    half_edge_id: str
    source_region_id: int
    source_gaussian_id: Any
    adjacent_gaussian_id: Any | None
    world_position: tuple[float, float, float]
    local_normal: tuple[float, float, float]
    local_tangent_direction: tuple[float, float, float]
    boundary_direction: tuple[float, float, float]
    boundary_reason: str
    source_pair_ids: tuple[Any, Any] | None
    confidence: float
    ordering_state: str
    review_reasons: tuple[str, ...]


def extract_world_space_boundary_halfedge_candidates(positions: Any, normals: Any, region_result: RegionFormationResult, graph: ManifoldAffinityGraph, *, ids: Sequence[Any] | None = None) -> tuple[WorldSpaceBoundaryHalfEdgeCandidate, ...]:
    count = len(region_result.node_region_id)
    ids = tuple(range(count)) if ids is None else tuple(ids)
    accepted = {tuple(sorted(edge)) for region in region_result.regions for edge in region.internal_accepted_edge_ids}
    output = []
    for edge in graph.edges:
        pair = tuple(sorted((edge.source_id, edge.target_id)))
        source_region = region_result.node_region_id[edge.source]
        if source_region < 0 or pair in accepted:
            continue
        if edge.manifold_relation == RELATION_CREASE:
            reason = "crease_discontinuity"
        elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
            reason = "parallel_sheet_conflict"
        elif region_result.node_membership_state[edge.target] == "rejected_structural_node":
            reason = "rejected_neighbor_adjacency"
        else:
            reason = "ambiguous_continuation"
        point = positions[edge.source]
        normal = normals[edge.source]
        delta = positions[edge.target] - point
        tangent = delta - normal * (delta * normal).sum()
        tangent = tangent / tangent.norm().clamp_min(1e-12)
        output.append(WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"region:{source_region}:gaussian:{edge.source_id}:adjacent:{edge.target_id}:{reason}",
            source_region_id=source_region, source_gaussian_id=edge.source_id, adjacent_gaussian_id=edge.target_id,
            world_position=tuple(float(x) for x in point), local_normal=tuple(float(x) for x in normal),
            local_tangent_direction=tuple(float(x) for x in tangent), boundary_direction=tuple(float(x) for x in tangent),
            boundary_reason=reason, source_pair_ids=pair, confidence=0.5,
            ordering_state="locally_chainable" if reason == "crease_discontinuity" else "ambiguous_ordering",
            review_reasons=("experimental_halfedge_candidate_not_ordered_boundary",),
        ))
    return tuple(sorted(output, key=lambda item: item.half_edge_id))
