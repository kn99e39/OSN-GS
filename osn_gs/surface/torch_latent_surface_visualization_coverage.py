from __future__ import annotations

"""Worklog 104 -- visualization-completeness for a Worklog 103 latent
support unit.

Worklog 103 attempted exactly ONE visualization-only NURBS per latent
support unit; 37 of 86 units failed the fixed resolution ladder outright,
making those units' latent-projected geometry invisible in the
``ALL_LATENT_SURFACES`` NURBS view even though the underlying supported
samples were correctly exported as raw point data. This module changes
the VISUALIZATION UNIT, never the latent surface itself: a unit that
cannot be represented by one visualization NURBS is deterministically
subdivided, using only the unit's own continuously-supported graph
connectivity (never convex hull / bounding box / PCA rectangle / arbitrary
Euclidean bridging), into smaller connected pieces, each retried
recursively. A piece that still cannot be represented after subdivision
down to a minimum floor is reported as an explicit
``UNREPRESENTED_LATENT_FRAGMENT`` with its exact source node IDs and
failure reason -- never silently dropped.

Subdivision never joins disconnected components, never invents geometry
outside the unit's own supported samples, and never uses production
chart/identifiability/safety criteria to decide where to cut.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_coverage_audit import LatentSupportUnit
from osn_gs.surface.torch_latent_surface_visualization_nurbs import (
    VisualizationNurbsResult,
    fit_visualization_nurbs,
)
from osn_gs.utils.torch_ops import require_torch

# Fixed, not tuned from visual appearance: a fragment smaller than this can
# never usefully subdivide further (subdividing a 2-node piece cannot
# produce two non-trivial connected pieces), so recursion stops here and
# the fragment is reported as unrepresented rather than looping forever.
MIN_SUBDIVIDABLE_SIZE = 3
# Defensive recursion cap -- purely to guarantee termination on pathological
# topology, never reached in practice for the unit sizes this module sees.
MAX_SUBDIVISION_DEPTH = 8


@dataclass(frozen=True)
class MaterializedFragment:
    unit_id: int
    fragment_id: int
    node_indices: tuple[int, ...]  # GLOBAL indices into the region's raw evidence tensor
    result: VisualizationNurbsResult


@dataclass(frozen=True)
class UnrepresentedFragment:
    unit_id: int
    node_indices: tuple[int, ...]  # GLOBAL indices into the region's raw evidence tensor
    reason: str


def _bfs_distances(adjacency: dict[int, list[int]], source: int) -> dict[int, int]:
    from collections import deque

    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        current = frontier.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
    return distances


def _adjacency_from_edges(node_count: int, edges: tuple[tuple[int, int], ...]) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(node_count)}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _connected_components(node_count: int, edges: tuple[tuple[int, int], ...]) -> list[list[int]]:
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for node in range(node_count):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def _split_by_graph_distance(local_count: int, edges: tuple[tuple[int, int], ...]) -> list[list[int]] | None:
    """Deterministic connectivity-only bisection: farthest-pair BFS anchors,
    assign every node to its nearer anchor (ties -> the first anchor), then
    re-run connected-components on each half's OWN induced edges so a half
    that itself splits into multiple disconnected pieces is reported as
    such rather than forced together. Returns ``None`` if the split cannot
    separate the unit at all (e.g. every node is equidistant, or the graph
    itself is already a single unsplittable piece)."""

    adjacency = _adjacency_from_edges(local_count, edges)
    anchor_a = 0  # deterministic: lowest local index
    distances_from_a = _bfs_distances(adjacency, anchor_a)
    if not distances_from_a:
        return None
    anchor_b = max(distances_from_a, key=lambda node: (distances_from_a[node], -node))
    if anchor_b == anchor_a:
        return None
    distances_from_b = _bfs_distances(adjacency, anchor_b)

    side_a: list[int] = []
    side_b: list[int] = []
    for node in range(local_count):
        distance_a = distances_from_a.get(node)
        distance_b = distances_from_b.get(node)
        if distance_a is None and distance_b is None:
            side_a.append(node)  # unreachable from either anchor -- keep deterministic, goes to side_a
        elif distance_b is None or (distance_a is not None and distance_a <= distance_b):
            side_a.append(node)
        else:
            side_b.append(node)

    if not side_a or not side_b:
        return None

    def _induced(side: list[int]) -> tuple[tuple[int, int], ...]:
        side_set = set(side)
        return tuple((a, b) for a, b in edges if a in side_set and b in side_set)

    pieces: list[list[int]] = []
    for side in (side_a, side_b):
        local_to_new = {old: new for new, old in enumerate(side)}
        induced_edges = tuple((local_to_new[a], local_to_new[b]) for a, b in _induced(side))
        for component in _connected_components(len(side), induced_edges):
            pieces.append([side[i] for i in component])
    if len(pieces) < 2:
        return None
    return pieces


def _sub_unit(
    unit_id: int, parent_node_indices: tuple[int, ...], parent_raw: Any, parent_latent: Any,
    parent_normals: Any, parent_edges: tuple[tuple[int, int], ...], local_indices: list[int],
) -> tuple[tuple[int, ...], Any, Any, Any, tuple[tuple[int, int], ...]]:
    torch = require_torch()
    selector = torch.tensor(local_indices, dtype=torch.long, device=parent_latent.device)
    local_to_new = {old: new for new, old in enumerate(local_indices)}
    node_set = set(local_indices)
    remapped_edges = tuple(
        (local_to_new[a], local_to_new[b]) for a, b in parent_edges if a in node_set and b in node_set
    )
    global_node_indices = tuple(parent_node_indices[i] for i in local_indices)
    return (
        global_node_indices, parent_raw[selector], parent_latent[selector], parent_normals[selector],
        remapped_edges,
    )


def materialize_unit_with_subdivision(
    unit: LatentSupportUnit,
) -> tuple[list[MaterializedFragment], list[UnrepresentedFragment]]:
    """Attempt one visualization NURBS for the whole unit; on failure,
    deterministically subdivide by graph connectivity and recurse. Every
    global node index in ``unit.node_indices`` ends up in EXACTLY one
    returned fragment (materialized or unrepresented) -- never both, never
    neither."""

    materialized: list[MaterializedFragment] = []
    unrepresented: list[UnrepresentedFragment] = []
    fragment_counter = [0]

    def _recurse(
        node_indices: tuple[int, ...], raw_positions: Any, latent_positions: Any, normals: Any,
        edges: tuple[tuple[int, int], ...], depth: int,
    ) -> None:
        # Never attempt a fit across genuinely disconnected geometry, even
        # if the fitter would numerically "succeed" anyway -- a NURBS
        # spanning two disconnected pieces would misrepresent connectivity
        # that was never actually supported. Split into real connected
        # components FIRST, before ever trying a whole-piece fit.
        components = _connected_components(len(node_indices), edges)
        if len(components) > 1:
            for local_piece in components:
                piece_node_indices, piece_raw, piece_latent, piece_normals, piece_edges = _sub_unit(
                    unit.unit_id, node_indices, raw_positions, latent_positions, normals, edges, local_piece,
                )
                _recurse(piece_node_indices, piece_raw, piece_latent, piece_normals, piece_edges, depth + 1)
            return

        result = fit_visualization_nurbs(unit.unit_id, latent_positions)
        if result.materialized:
            materialized.append(MaterializedFragment(unit.unit_id, fragment_counter[0], node_indices, result))
            fragment_counter[0] += 1
            return

        if len(node_indices) < MIN_SUBDIVIDABLE_SIZE or depth >= MAX_SUBDIVISION_DEPTH:
            unrepresented.append(UnrepresentedFragment(
                unit.unit_id, node_indices,
                f"materialization_failed_at_minimum_fragment_size:{result.invalid_reason}",
            ))
            return

        split = _split_by_graph_distance(len(node_indices), edges)
        if split is None:
            unrepresented.append(UnrepresentedFragment(
                unit.unit_id, node_indices, f"connectivity_not_further_subdividable:{result.invalid_reason}",
            ))
            return

        for local_piece in split:
            piece_node_indices, piece_raw, piece_latent, piece_normals, piece_edges = _sub_unit(
                unit.unit_id, node_indices, raw_positions, latent_positions, normals, edges, local_piece,
            )
            _recurse(piece_node_indices, piece_raw, piece_latent, piece_normals, piece_edges, depth + 1)

    _recurse(unit.node_indices, unit.raw_positions, unit.latent_positions, unit.normals, unit.edges, 0)
    return materialized, unrepresented
