from __future__ import annotations

"""Directed mutual-successor ordering for local boundary evidence."""

from dataclasses import dataclass
from typing import Sequence

from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


@dataclass(frozen=True)
class DirectedBoundarySuccessor:
    source_half_edge_id: str
    target_half_edge_id: str
    forward_distance: float
    lateral_residual: float
    normalized_distance: float
    tangent_alignment: float
    normal_alignment: float
    outward_alignment: float
    score: float
    decision: str


def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _sub(a, b): return tuple(x-y for x,y in zip(a,b))
def _norm(a): return max(sum(x*x for x in a) ** .5, 1e-12)


def recover_directed_boundary_components(candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate], accepted_topology: Sequence[tuple[object, object]] = ()) -> tuple[tuple[DirectedBoundarySuccessor, ...], tuple[OrderedBoundaryComponent, ...]]:
    """Select one mutual forward successor, never all compatible pairs."""
    candidates = tuple(item for item in candidates if item.boundary_reason == "observed_support_termination")
    by_id = {item.half_edge_id: item for item in candidates}
    nearest = []
    for source in candidates:
        distances = [_norm(_sub(source.world_position, target.world_position)) for target in candidates if target.half_edge_id != source.half_edge_id and target.source_region_id == source.source_region_id]
        if distances: nearest.append(min(distances))
    local_spacing = sorted(nearest)[len(nearest) // 2] if nearest else 1.0
    max_distance, max_lateral = local_spacing * 1.6, local_spacing * 0.9
    accepted_pairs = {frozenset(pair) for pair in accepted_topology}
    best = {}
    diagnostics = []
    for source in candidates:
        options = []
        for target in candidates:
            if target.half_edge_id == source.half_edge_id or target.source_region_id != source.source_region_id or frozenset((source.source_gaussian_id, target.source_gaussian_id)) not in accepted_pairs:
                continue
            delta = _sub(target.world_position, source.world_position)
            distance = _norm(delta); tangent = source.boundary_direction
            forward = _dot(delta, tangent)
            if forward <= 1e-8 or distance > max_distance:
                continue
            lateral = (max(distance * distance - forward * forward, 0.0)) ** .5
            tan_align = _dot(source.boundary_direction, target.boundary_direction)
            normal_align = abs(_dot(source.local_normal, target.local_normal))
            # outward is normal x directed tangent; sign-consistent tangent makes this meaningful.
            outward_align = normal_align * max(tan_align, 0.0)
            if lateral > max_lateral or tan_align < -.15 or normal_align < .45:
                continue
            score = forward / distance + tan_align + normal_align + outward_align - lateral / max_lateral
            options.append((score, str(target.source_gaussian_id), target))
        if not options:
            continue
        options.sort(key=lambda item: (-item[0], item[1], item[2].half_edge_id))
        if len(options) > 1 and abs(options[0][0] - options[1][0]) < 1e-6:
            continue
        score, _, target = options[0]
        delta = _sub(target.world_position, source.world_position); distance = _norm(delta); forward = _dot(delta, source.boundary_direction)
        diagnostics.append(DirectedBoundarySuccessor(source.half_edge_id, target.half_edge_id, forward, (max(distance*distance-forward*forward,0))**.5, distance/max_distance, _dot(source.boundary_direction,target.boundary_direction), abs(_dot(source.local_normal,target.local_normal)), abs(_dot(source.local_normal,target.local_normal))*max(_dot(source.boundary_direction,target.boundary_direction),0), score, "best_forward"))
        best[source.half_edge_id] = target.half_edge_id
    # Score backward predecessor independently of the forward choice.
    predecessor = {}
    for target in candidates:
        options = []
        for source in candidates:
            if source.half_edge_id == target.half_edge_id or source.source_region_id != target.source_region_id or frozenset((source.source_gaussian_id, target.source_gaussian_id)) not in accepted_pairs:
                continue
            delta = _sub(source.world_position, target.world_position)
            distance = _norm(delta); backward = -_dot(delta, target.boundary_direction)
            if backward <= 1e-8 or distance > max_distance:
                continue
            lateral = (max(distance * distance - backward * backward, 0.0)) ** .5
            tangent = _dot(source.boundary_direction, target.boundary_direction)
            normal = abs(_dot(source.local_normal, target.local_normal))
            if lateral > max_lateral or tangent < -.15 or normal < .45:
                continue
            score = backward / distance + tangent + normal + normal * max(tangent, 0.0) - lateral / max_lateral
            options.append((score, str(source.source_gaussian_id), source))
        if options:
            options.sort(key=lambda item: (-item[0], item[1], item[2].half_edge_id))
            if len(options) == 1 or abs(options[0][0] - options[1][0]) >= 1e-6:
                predecessor[target.half_edge_id] = options[0][2].half_edge_id
    mutual = tuple(item for item in diagnostics if predecessor.get(item.target_half_edge_id) == item.source_half_edge_id)
    selected = list(mutual)
    used_out = {edge.source_half_edge_id for edge in selected}
    used_in = {edge.target_half_edge_id for edge in selected}
    for edge in sorted(diagnostics, key=lambda item: (-item.score, item.source_half_edge_id, item.target_half_edge_id)):
        if edge.source_half_edge_id in used_out or edge.target_half_edge_id in used_in:
            continue
        selected.append(edge); used_out.add(edge.source_half_edge_id); used_in.add(edge.target_half_edge_id)
    mutual = tuple(selected)
    adjacency = {item.half_edge_id: [] for item in candidates}
    for edge in mutual: adjacency[edge.source_half_edge_id].append(edge.target_half_edge_id)
    output=[]; seen=set()
    for start in sorted(adjacency):
        if start in seen: continue
        chain=[]; current=start
        while current not in chain and current in adjacency and len(adjacency[current]) == 1:
            chain.append(current); seen.add(current); current=adjacency[current][0]
        closed = current == start and len(chain) >= 3
        if not chain: continue
        source = by_id[start]; state = "ordered_closed_loop" if closed else "ambiguous_ordering"
        role = "outer_boundary_candidate" if closed else "unresolved_boundary_role"
        output.append(OrderedBoundaryComponent(f"region:{source.source_region_id}:directed:{min(chain)}",source.source_region_id,tuple(chain),tuple(by_id[x].source_gaussian_id for x in chain),state,closed,(),{"observed_support_termination":len(chain)},.7,role,"reliable_core_only",False,() if closed else ("mutual_successor_cycle_not_proven",)))
    return mutual, tuple(output)
