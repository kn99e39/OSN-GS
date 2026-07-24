from __future__ import annotations

"""Diagnostics-only scale-aware spatial graph over atomic surface cells.

The diagnostics-only Stage 2 recorded in
``docs/worklogs/62_proxy_decomposition_stage2_candidate_graph.md`` replaces face
contact as the *candidate generation* mechanism, not as a production component
rule. This module therefore knows nothing about scene names,
ground-truth topology, component counts, merge scores, or admissibility.

Candidate generation uses expanded adaptive-leaf AABBs with a deterministic
sweep-and-prune broad phase followed by the exact symmetric condition::

    aabb_distance(a, b) <= radius_factor * max(cell_scale(a), cell_scale(b))

Face/edge/corner contact is computed only after an edge has been accepted and
is stored as diagnostic provenance.  It never changes candidate membership.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from osn_gs.surface.torch_voxel_hierarchy import (
    STATE_ACTIVE,
    STATE_COMPLEX,
    TorchVoxelGaussianHierarchy,
)
from osn_gs.utils.torch_ops import require_torch


CANDIDATE_SOURCE = "scale_aware_expanded_aabb_sweep"


@dataclass(frozen=True)
class SurfaceCandidateEdge:
    cell_a: str
    cell_b: str
    aabb_distance: float
    centroid_distance: float
    support_gap: float
    scale_a: float
    scale_b: float
    radius_limit: float
    scale_normalized_aabb_distance: float
    scale_normalized_gap: float
    contact_relation: str
    candidate_source: str = CANDIDATE_SOURCE

    @property
    def pair(self) -> tuple[str, str]:
        return (self.cell_a, self.cell_b)

    def payload(self, diagnostic_tags: Iterable[str] = ()) -> dict[str, Any]:
        return {
            "pair_id": f"{self.cell_a}|{self.cell_b}",
            "cell_a": self.cell_a,
            "cell_b": self.cell_b,
            "aabb_distance": self.aabb_distance,
            "centroid_distance": self.centroid_distance,
            "support_gap": self.support_gap,
            "scale_a": self.scale_a,
            "scale_b": self.scale_b,
            "radius_limit": self.radius_limit,
            "scale_normalized_aabb_distance": self.scale_normalized_aabb_distance,
            "scale_normalized_gap": self.scale_normalized_gap,
            "contact_relation": self.contact_relation,
            "candidate_source": self.candidate_source,
            "diagnostic_tags": sorted(set(diagnostic_tags)),
        }


@dataclass
class SurfaceCandidateGraph:
    node_ids: list[str]
    edges: list[SurfaceCandidateEdge]
    config: dict[str, Any]

    def edge_pairs(self) -> set[tuple[str, str]]:
        return {edge.pair for edge in self.edges}

    def degree_by_node(self) -> dict[str, int]:
        degree = {node_id: 0 for node_id in self.node_ids}
        for edge in self.edges:
            degree[edge.cell_a] += 1
            degree[edge.cell_b] += 1
        return degree


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _percentile_nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(math.ceil(float(quantile) * (len(ordered) - 1)))
    return ordered[index]


def _aabb_distance(aabb_min_a: Any, aabb_max_a: Any, aabb_min_b: Any, aabb_max_b: Any) -> float:
    torch = require_torch()
    gap = torch.maximum(aabb_min_a - aabb_max_b, aabb_min_b - aabb_max_a).clamp_min(0)
    return float(torch.linalg.norm(gap))


def _point_support_gap(node_a: Any, node_b: Any, points: Any) -> float:
    """Return exact closest raw-point distance for accepted-edge diagnostics."""

    torch = require_torch()
    indices_a = node_a.gaussian_indices
    indices_b = node_b.gaussian_indices
    if indices_a is None or indices_b is None:
        return math.inf
    if int(indices_a.numel()) == 0 or int(indices_b.numel()) == 0:
        return math.inf
    return float(torch.cdist(points[indices_a], points[indices_b]).min())


def classify_aabb_contact(
    aabb_min_a: Any,
    aabb_max_a: Any,
    aabb_min_b: Any,
    aabb_max_b: Any,
    tolerance: float,
) -> str:
    """Classify AABB relation for diagnostics only.

    An axis on which both cells are degenerate is treated as overlapping, so
    an x-face contact on a perfectly flat z=0 surface remains ``face`` rather
    than being mislabeled ``edge`` solely because both z extents are zero.
    """

    touching_axes = 0
    for axis in range(3):
        lo = max(float(aabb_min_a[axis]), float(aabb_min_b[axis]))
        hi = min(float(aabb_max_a[axis]), float(aabb_max_b[axis]))
        overlap = hi - lo
        extent_a = float(aabb_max_a[axis] - aabb_min_a[axis])
        extent_b = float(aabb_max_b[axis] - aabb_min_b[axis])
        if overlap < -float(tolerance):
            return "disjoint"
        both_degenerate = extent_a <= float(tolerance) and extent_b <= float(tolerance)
        if not both_degenerate and overlap <= float(tolerance):
            touching_axes += 1
    if touching_axes == 0:
        return "overlap"
    if touching_axes == 1:
        return "face"
    if touching_axes == 2:
        return "edge"
    return "corner"


def _node_centroid(node: Any, points: Any) -> Any:
    if node.gaussian_indices is not None and int(node.gaussian_indices.numel()) > 0:
        return points[node.gaussian_indices].mean(dim=0)
    if node.plane is not None:
        return node.plane.centroid
    return (node.aabb_min + node.aabb_max) * 0.5


def build_surface_cell_candidate_graph(
    hierarchy: TorchVoxelGaussianHierarchy,
    points: Any,
    radius_factor: float = 0.25,
    max_neighbors: int = 0,
    fit_complex_leaves: bool = True,
) -> SurfaceCandidateGraph:
    """Build a deterministic spatial candidate graph over mergeable leaves.

    ``max_neighbors=0`` means unlimited and is the diagnostics default so
    recall is measured before any degree cap.  A positive cap is applied by a
    deterministic shortest-normalized-distance greedy pass and is intended
    only for explicit scalability sweeps.
    """

    torch = require_torch()
    radius_factor = float(radius_factor)
    max_neighbors = int(max_neighbors)
    if radius_factor < 0.0 or not math.isfinite(radius_factor):
        raise ValueError("radius_factor must be finite and non-negative")
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be >= 0")
    if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")

    mergeable_states = {STATE_ACTIVE} | ({STATE_COMPLEX} if fit_complex_leaves else set())
    leaves = sorted(
        (leaf for leaf in hierarchy.leaves() if leaf.state in mergeable_states),
        key=lambda leaf: leaf.node_id,
    )
    node_ids = [leaf.node_id for leaf in leaves]
    if not leaves:
        return SurfaceCandidateGraph(
            node_ids=[],
            edges=[],
            config={
                "radius_factor": radius_factor,
                "max_neighbors": max_neighbors,
                "fit_complex_leaves": bool(fit_complex_leaves),
                "candidate_source": CANDIDATE_SOURCE,
            },
        )

    root = hierarchy.nodes[0]
    root_extent = float((root.aabb_max - root.aabb_min).abs().max())
    tolerance = max(root_extent, 1e-6) * 1e-8
    entries = []
    for leaf in leaves:
        extent = leaf.aabb_max - leaf.aabb_min
        scale = max(float(torch.linalg.norm(extent)), tolerance)
        radius = radius_factor * scale
        entries.append(
            {
                "leaf": leaf,
                "scale": scale,
                "radius": radius,
                "expanded_min": leaf.aabb_min - radius,
                "expanded_max": leaf.aabb_max + radius,
                "centroid": _node_centroid(leaf, points),
            }
        )

    # Deterministic sweep-and-prune broad phase.  Contact relation is not read
    # here; only expanded interval overlap controls which exact distance checks
    # are necessary.
    entries.sort(key=lambda entry: (float(entry["expanded_min"][0]), entry["leaf"].node_id))
    active: list[dict[str, Any]] = []
    candidate_edges: list[SurfaceCandidateEdge] = []
    seen: set[tuple[str, str]] = set()
    for current in entries:
        current_min_x = float(current["expanded_min"][0])
        active = [
            other
            for other in active
            if float(other["expanded_max"][0]) >= current_min_x - tolerance
        ]
        for other in active:
            if any(
                float(other["expanded_max"][axis]) < float(current["expanded_min"][axis]) - tolerance
                or float(current["expanded_max"][axis]) < float(other["expanded_min"][axis]) - tolerance
                for axis in (1, 2)
            ):
                continue
            leaf_a, leaf_b = other["leaf"], current["leaf"]
            pair = _canonical_pair(leaf_a.node_id, leaf_b.node_id)
            if pair in seen:
                continue
            aabb_distance = _aabb_distance(
                leaf_a.aabb_min, leaf_a.aabb_max, leaf_b.aabb_min, leaf_b.aabb_max
            )
            scale_a, scale_b = float(other["scale"]), float(current["scale"])
            pair_scale = max(scale_a, scale_b, tolerance)
            radius_limit = radius_factor * pair_scale
            if aabb_distance > radius_limit + tolerance:
                continue
            seen.add(pair)
            centroid_distance = float(torch.linalg.norm(other["centroid"] - current["centroid"]))
            support_gap = _point_support_gap(leaf_a, leaf_b, points)
            relation = classify_aabb_contact(
                leaf_a.aabb_min,
                leaf_a.aabb_max,
                leaf_b.aabb_min,
                leaf_b.aabb_max,
                tolerance,
            )
            values = {
                leaf_a.node_id: (scale_a, other),
                leaf_b.node_id: (scale_b, current),
            }
            ordered_scale_a = values[pair[0]][0]
            ordered_scale_b = values[pair[1]][0]
            candidate_edges.append(
                SurfaceCandidateEdge(
                    cell_a=pair[0],
                    cell_b=pair[1],
                    aabb_distance=aabb_distance,
                    centroid_distance=centroid_distance,
                    support_gap=support_gap,
                    scale_a=ordered_scale_a,
                    scale_b=ordered_scale_b,
                    radius_limit=radius_limit,
                    scale_normalized_aabb_distance=aabb_distance / pair_scale,
                    scale_normalized_gap=support_gap / pair_scale,
                    contact_relation=relation,
                )
            )
        active.append(current)

    if max_neighbors > 0:
        degree = {node_id: 0 for node_id in node_ids}
        accepted: list[SurfaceCandidateEdge] = []
        for edge in sorted(
            candidate_edges,
            key=lambda item: (
                item.scale_normalized_aabb_distance,
                item.centroid_distance / max(item.scale_a, item.scale_b, tolerance),
                item.cell_a,
                item.cell_b,
            ),
        ):
            if degree[edge.cell_a] >= max_neighbors or degree[edge.cell_b] >= max_neighbors:
                continue
            degree[edge.cell_a] += 1
            degree[edge.cell_b] += 1
            accepted.append(edge)
        candidate_edges = accepted

    candidate_edges.sort(key=lambda edge: (edge.cell_a, edge.cell_b))
    return SurfaceCandidateGraph(
        node_ids=node_ids,
        edges=candidate_edges,
        config={
            "radius_factor": radius_factor,
            "max_neighbors": max_neighbors,
            "fit_complex_leaves": bool(fit_complex_leaves),
            "candidate_source": CANDIDATE_SOURCE,
            "broad_phase": "expanded_aabb_sweep_and_prune",
            "exact_distance": "aabb_euclidean",
        },
    )


def candidate_graph_payload(
    graph: SurfaceCandidateGraph,
    reference_pairs: Mapping[str, Iterable[tuple[str, str]]] | None = None,
    diagnostic_tags: Mapping[tuple[str, str], Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Serialize graph plus optional evaluation-only recall/tag diagnostics."""

    edge_pairs = graph.edge_pairs()
    degrees = graph.degree_by_node()
    degree_values = list(degrees.values())
    degree_histogram: dict[str, int] = {}
    for degree in degree_values:
        key = str(degree)
        degree_histogram[key] = degree_histogram.get(key, 0) + 1
    relation_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for edge in graph.edges:
        relation_counts[edge.contact_relation] = relation_counts.get(edge.contact_relation, 0) + 1
        source_counts[edge.candidate_source] = source_counts.get(edge.candidate_source, 0) + 1

    recalls: dict[str, Any] = {}
    for name, pairs in sorted((reference_pairs or {}).items()):
        canonical = sorted({_canonical_pair(a, b) for a, b in pairs if a != b})
        found = [pair for pair in canonical if pair in edge_pairs]
        recalls[name] = {
            "reference_count": len(canonical),
            "found_count": len(found),
            "recall": len(found) / len(canonical) if canonical else 1.0,
            "missing_pairs": [list(pair) for pair in canonical if pair not in edge_pairs],
        }

    node_count = len(graph.node_ids)
    possible_edges = node_count * (node_count - 1) // 2
    tags = diagnostic_tags or {}
    tag_counts: dict[str, int] = {}
    for edge in graph.edges:
        for tag in sorted(set(tags.get(edge.pair, ()))):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "config": dict(graph.config),
        "node_count": node_count,
        "edge_count": len(graph.edges),
        "possible_edge_count": possible_edges,
        "candidate_fraction_of_complete_graph": (
            len(graph.edges) / possible_edges if possible_edges else 0.0
        ),
        "degree": {
            "by_node": degrees,
            "histogram": degree_histogram,
            "min": min(degree_values, default=0),
            "median": _percentile_nearest_rank(degree_values, 0.5),
            "p95": _percentile_nearest_rank(degree_values, 0.95),
            "max": max(degree_values, default=0),
            "mean": sum(degree_values) / len(degree_values) if degree_values else 0.0,
            "isolated_node_count": sum(1 for value in degree_values if value == 0),
        },
        "contact_relation_counts": dict(sorted(relation_counts.items())),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "diagnostic_tag_counts": dict(sorted(tag_counts.items())),
        "reference_recall": recalls,
        "edges": [
            edge.payload(tags.get(edge.pair, ()))
            for edge in graph.edges
        ],
    }


__all__ = [
    "CANDIDATE_SOURCE",
    "SurfaceCandidateEdge",
    "SurfaceCandidateGraph",
    "build_surface_cell_candidate_graph",
    "candidate_graph_payload",
    "classify_aabb_contact",
]
