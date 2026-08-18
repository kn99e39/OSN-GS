from __future__ import annotations

"""Worklog 98 -- globally synchronized in-plane tangent frame over a
region's latent surface support.

Worklog 97 showed that 91.7% of Worklog 96's own coherent curve blocks
fail intrinsic UV construction because adjacent transversal curves are
seeded with INDEPENDENTLY chosen directions (one per seed sample) that can
disagree or outright oppose each other. This module replaces that
per-sample choice with one coherent field: a single in-plane basis
``(e_u, e_v)`` synchronized by parallel transport across a support
adjacency graph, so every point in a coherent component shares one
consistent parametric orientation before any curve is traced.

The latent-surface estimator's own local eigenvectors are used ONLY to
recover a tangent PLANE (the normal); they are never treated as an
authoritative in-plane direction (Worklog 98 directive section 1). This
module builds its own oriented in-plane basis from scratch at the anchor
and propagates it outward.

No raw Gaussian-center manifold topology is built or required. The support
adjacency here is a lightweight kNN graph over already latent-surface-
supported points, used ONLY to synchronize the frame -- every edge is
independently validated by
:func:`~osn_gs.surface.torch_latent_surface_curve_tracer.sample_segment_continuous_support`
before it may carry a transported frame.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_tracer import (
    propagate_tangent_onto_plane,
    sample_segment_continuous_support,
)
from osn_gs.surface.torch_latent_surface_support import DEFAULT_K, LatentSurfaceSupport
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# Fixed, not tuned from any real replay outcome: reuses the same kNN width
# already established for latent-surface queries (Worklog 95's own
# DEFAULT_K) as the field-synchronization adjacency width.
FIELD_NEIGHBOR_COUNT = DEFAULT_K
# A coherent component candidate needs at least this many successfully
# framed nodes -- matches Worklog 96's own MIN_FAMILY_CURVE_COUNT-scale
# minimum-evidence floor, not a new swept parameter.
MIN_COMPONENT_SIZE = 8
# Same principled floor as Worklog 97's own direction-consistency check:
# a transported frame that disagrees by more than 90 degrees (cosine <= 0)
# with the frame already assigned at the target node is a genuine
# orientation reversal, not noise.
_CONSISTENT_COSINE_FLOOR = 0.0


@dataclass(frozen=True)
class FieldSingularity:
    node_index: int
    reason: str  # "degenerate_transport" | "unsupported_frame_propagation"


@dataclass(frozen=True)
class HolonomyEdge:
    node_a: int
    node_b: int
    consistent: bool
    angular_disagreement_degrees: float | None


@dataclass(frozen=True)
class TangentFrameFieldComponent:
    node_indices: tuple[int, ...]  # indices into the original points tensor
    positions: Any  # (M, 3)
    normals: Any  # (M, 3)
    e_u: Any  # (M, 3)
    e_v: Any  # (M, 3)
    u: Any  # (M,) tree-derived arc-length potential
    v: Any  # (M,)
    tree_edges: tuple[tuple[int, int], ...]  # indices are LOCAL (0..M-1)
    holonomy_edges: tuple[HolonomyEdge, ...]  # non-tree edges tested for consistency (LOCAL indices)
    singularities: tuple[FieldSingularity, ...]
    coherent: bool
    incoherence_reason: str | None
    anchor_seed_type: str | None


@dataclass(frozen=True)
class TangentFrameFieldResult:
    components: tuple[TangentFrameFieldComponent, ...]
    total_candidate_edges: int
    unsupported_edge_count: int


def _knn_edges(points: Any, k: int) -> Any:
    torch = require_torch()
    count = points.shape[0]
    k = min(k, count - 1)
    if k <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=points.device)
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    _, indices = torch.topk(distance, k, dim=1, largest=False)
    rows = torch.arange(count, device=points.device).unsqueeze(1).expand(-1, k)
    edges = torch.stack([rows.reshape(-1), indices.reshape(-1)], dim=1)
    # Symmetrize and dedupe (a<b canonical form).
    a = torch.minimum(edges[:, 0], edges[:, 1])
    b = torch.maximum(edges[:, 0], edges[:, 1])
    unique_edges = torch.unique(torch.stack([a, b], dim=1), dim=0)
    return unique_edges


def _canonical_in_plane_basis(normal: Any) -> Any:
    """Deterministic Gram-Schmidt in-plane axis, independent of any
    Gaussian's own covariance/PCA sign -- used only as the gauge choice at
    an anchor with no boundary tangent to inherit."""

    torch = require_torch()
    reference = torch.tensor([1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device)
    if float(torch.abs((normal * reference).sum())) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype, device=normal.device)
    projected = reference - (reference * normal).sum() * normal
    return projected / projected.norm().clamp_min(_EPS)


def _union_find_components(count: int, edges: Any) -> list[list[int]]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for a, b in edges.tolist():
        union(a, b)

    groups: dict[int, list[int]] = {}
    for node in range(count):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def _build_component_frame(
    points: Any, normals: Any, node_indices: list[int], local_edges: list[tuple[int, int]],
    root_local: int, anchor_hint_direction: Any | None,
) -> tuple[Any, Any, Any, Any, tuple[tuple[int, int], ...], tuple[FieldSingularity, ...], set[int]]:
    """Geometric-shortest-path spanning-tree frame propagation + arc-length
    potential integration over one topological component. Dijkstra by
    actual 3D edge length (not raw BFS hop count) keeps each node's
    transport path as geometrically direct as possible, which is what
    keeps the tree-integrated ``(u, v)`` potential close to a true
    path-independent arc-length coordinate despite the connection's
    curvature-induced holonomy being only checked, not zero. Returns
    (e_u, e_v, u, v, tree_edges, singularities, framed_local_indices)."""

    torch = require_torch()
    count = len(node_indices)
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(count)}
    for a, b in local_edges:
        edge_length = float((points[a] - points[b]).norm().item())
        adjacency[a].append((b, edge_length))
        adjacency[b].append((a, edge_length))

    e_u = torch.zeros((count, 3), dtype=points.dtype, device=points.device)
    e_v = torch.zeros((count, 3), dtype=points.dtype, device=points.device)
    u = torch.zeros(count, dtype=points.dtype, device=points.device)
    v = torch.zeros(count, dtype=points.dtype, device=points.device)
    framed = {root_local}
    tree_edges: list[tuple[int, int]] = []
    singularities: list[FieldSingularity] = []

    root_normal = normals[root_local]
    root_e_u = (
        anchor_hint_direction if anchor_hint_direction is not None else _canonical_in_plane_basis(root_normal)
    )
    projected_root = root_e_u - (root_e_u * root_normal).sum() * root_normal
    root_norm = projected_root.norm()
    if float(root_norm.item()) < _EPS:
        root_e_u = _canonical_in_plane_basis(root_normal)
    else:
        root_e_u = projected_root / root_norm
    e_u[root_local] = root_e_u
    e_v[root_local] = torch.linalg.cross(root_normal, root_e_u)

    import heapq

    frontier: list[tuple[float, int, int]] = [(0.0, root_local, root_local)]
    visited_or_failed: set[int] = set()
    while frontier:
        distance, current, parent = heapq.heappop(frontier)
        if current in visited_or_failed:
            continue
        if current != root_local:
            propagated = propagate_tangent_onto_plane(e_u[parent], normals[current])
            if propagated is None:
                singularities.append(FieldSingularity(current, "degenerate_transport"))
                visited_or_failed.add(current)
                continue
            e_u[current] = propagated
            e_v[current] = torch.linalg.cross(normals[current], propagated)
            delta = points[current] - points[parent]
            u[current] = u[parent] + (delta * e_u[parent]).sum()
            v[current] = v[parent] + (delta * e_v[parent]).sum()
            framed.add(current)
            tree_edges.append((parent, current))
        visited_or_failed.add(current)
        for neighbor, edge_length in adjacency[current]:
            if neighbor not in visited_or_failed:
                heapq.heappush(frontier, (distance + edge_length, neighbor, current))

    return e_u, e_v, u, v, tuple(tree_edges), tuple(singularities), framed


def _check_holonomy(
    points: Any, e_u: Any, e_v: Any, normals: Any, non_tree_edges: list[tuple[int, int]], framed: set[int],
) -> tuple[HolonomyEdge, ...]:
    torch = require_torch()
    results: list[HolonomyEdge] = []
    for a, b in non_tree_edges:
        if a not in framed or b not in framed:
            continue
        transported = propagate_tangent_onto_plane(e_u[a], normals[b])
        if transported is None:
            results.append(HolonomyEdge(a, b, False, None))
            continue
        cosine = float((transported * e_u[b]).sum().clamp(-1.0, 1.0).item())
        consistent = cosine > _CONSISTENT_COSINE_FLOOR
        angle_degrees = float(torch.rad2deg(torch.arccos(torch.tensor(min(1.0, max(-1.0, cosine))))).item())
        results.append(HolonomyEdge(a, b, consistent, angle_degrees))
    return tuple(results)


def build_tangent_frame_field(
    points: Any,
    support: LatentSurfaceSupport,
    *,
    k: int = FIELD_NEIGHBOR_COUNT,
    min_component_size: int = MIN_COMPONENT_SIZE,
    anchor_position: Any | None = None,
    anchor_hint_direction: Any | None = None,
    anchor_seed_type: str | None = None,
) -> TangentFrameFieldResult:
    """Build a globally synchronized tangent frame over ``points`` (all
    assumed drawn from ``support``'s own evidence or otherwise verified
    supported by the caller). ``anchor_position``/``anchor_hint_direction``
    optionally anchor one component's gauge to an observed boundary/feature
    tangent (Worklog 98 section 3) -- always optional, never required.
    """

    torch = require_torch()
    count = int(points.shape[0])
    query = support.query_batch(points)
    normals = query.normals

    candidate_edges = _knn_edges(points, k)
    supported_mask = []
    for a, b in candidate_edges.tolist():
        _seg_points, fully_supported = sample_segment_continuous_support(support, points[a], points[b])
        supported_mask.append(fully_supported)
    supported_mask_tensor = torch.tensor(supported_mask, dtype=torch.bool)
    supported_edges = candidate_edges[supported_mask_tensor] if candidate_edges.shape[0] else candidate_edges
    unsupported_count = int((~supported_mask_tensor).sum().item()) if supported_mask else 0

    topo_components = _union_find_components(count, supported_edges)

    anchor_local_by_component: dict[int, int] = {}
    if anchor_position is not None:
        distances_to_anchor = torch.cdist(points, anchor_position.reshape(1, 3)).reshape(-1)
        nearest_global = int(distances_to_anchor.argmin().item())
        for component_index, node_list in enumerate(topo_components):
            if nearest_global in node_list:
                anchor_local_by_component[component_index] = node_list.index(nearest_global)

    components: list[TangentFrameFieldComponent] = []
    edge_set_by_component: dict[int, list[tuple[int, int]]] = {}
    node_position_by_component: dict[int, dict[int, int]] = {}
    for component_index, node_list in enumerate(topo_components):
        node_position_by_component[component_index] = {node: local for local, node in enumerate(node_list)}
    for a, b in supported_edges.tolist():
        for component_index, mapping in node_position_by_component.items():
            if a in mapping:
                edge_set_by_component.setdefault(component_index, []).append((mapping[a], mapping[b]))
                break

    for component_index, node_list in enumerate(topo_components):
        if len(node_list) < min_component_size:
            continue
        local_edges = edge_set_by_component.get(component_index, [])
        component_points = points[node_list]
        component_normals = normals[node_list]

        if component_index in anchor_local_by_component:
            root_local = anchor_local_by_component[component_index]
            root_normal = component_normals[root_local]
            hint = None
            if anchor_hint_direction is not None:
                candidate = anchor_hint_direction - (anchor_hint_direction * root_normal).sum() * root_normal
                if float(candidate.norm().item()) > _EPS:
                    hint = candidate / candidate.norm()
        else:
            centroid = component_points.mean(dim=0)
            root_local = int(torch.cdist(component_points, centroid.reshape(1, 3)).reshape(-1).argmin().item())
            hint = None

        e_u, e_v, u, v, tree_edges, singularities, framed = _build_component_frame(
            component_points, component_normals, node_list, local_edges, root_local, hint,
        )
        tree_edge_set = {tuple(sorted(edge)) for edge in tree_edges}
        non_tree_edges = [edge for edge in local_edges if tuple(sorted(edge)) not in tree_edge_set]
        holonomy_edges = _check_holonomy(component_points, e_u, e_v, component_normals, non_tree_edges, framed)

        framed_indices = sorted(framed)
        if len(framed_indices) < min_component_size:
            continue

        selector = torch.tensor(framed_indices, dtype=torch.long)
        local_to_new = {old: new for new, old in enumerate(framed_indices)}
        remapped_tree_edges = tuple((local_to_new[a], local_to_new[b]) for a, b in tree_edges)
        remapped_holonomy = tuple(
            HolonomyEdge(local_to_new[edge.node_a], local_to_new[edge.node_b], edge.consistent, edge.angular_disagreement_degrees)
            for edge in holonomy_edges if edge.node_a in local_to_new and edge.node_b in local_to_new
        )
        inconsistent_count = sum(1 for edge in remapped_holonomy if not edge.consistent)
        coherent = inconsistent_count == 0
        reason = None if coherent else "holonomy_inconsistency"

        components.append(TangentFrameFieldComponent(
            node_indices=tuple(node_list[i] for i in framed_indices),
            positions=component_points[selector], normals=component_normals[selector],
            e_u=e_u[selector], e_v=e_v[selector], u=u[selector], v=v[selector],
            tree_edges=remapped_tree_edges, holonomy_edges=remapped_holonomy,
            singularities=singularities, coherent=coherent, incoherence_reason=reason,
            anchor_seed_type=anchor_seed_type if component_index in anchor_local_by_component else None,
        ))

    return TangentFrameFieldResult(
        components=tuple(components), total_candidate_edges=int(candidate_edges.shape[0]),
        unsupported_edge_count=unsupported_count,
    )
