from __future__ import annotations

"""Worklog 82: evidence-scale surface-consistency chart-unit decomposition.

Worklog 80 separated topology (sparse accepted cycle) from geometry (dense
boundary-support), but kept an implicit assumption that ONE region owns
exactly ONE chart. Worklog 81 showed the resulting single-chart fits fail
not because of parameterization choice but because the region's OWN evidence
is not locally flat (local normal disagreement 16-37%, thickness ratio
17-55%) -- i.e. some regions may genuinely contain more than one coherent
surface sheet, or evidence that resolves into no clean sheet at all.

This module resolves that inside a region, at EVIDENCE scale, using only
already-computed local structural evidence -- never spatial proximity alone:

  * Candidate adjacency is a bounded-degree kNN graph (never a raw radius
    graph or dense clique -- degree is capped exactly as
    `torch_gaussian_manifold_affinity.py`'s `max_candidate_count_per_node`
    already does at representative scale).
  * A kNN candidate edge is only ACCEPTED as `same_surface` when normal
    alignment is high AND mutual tangent residual is low -- the same
    same-surface criterion `torch_gaussian_manifold_affinity._classify_relation`
    already uses, evaluated here directly on per-Gaussian covariance frames
    (`extract_covariance_frame`, unmodified) at evidence density instead of
    at representative density.
  * A candidate edge whose two endpoints fall on OPPOSITE SIDES of an
    already-typed crease/frontier arc (worklog 80's per-arc `segment_kind`,
    via nearest-arc assignment) is VETOED regardless of how well its
    normal/residual line up -- typed provenance is a legitimate separator,
    reused, not reinvented from desired output geometry.
  * Chart-unit components are the connected components of the ACCEPTED
    (same_surface) edges only. Crease/ambiguous/rejected edges never merge
    components -- exactly `diagnose_same_surface_regions`'s own convention,
    reused here at evidence scale.
  * A point with zero accepted same_surface edges is UNRESOLVED, never
    silently assigned to a nearby component.
  * A component whose own accepted-edge subgraph contains an internal normal
    sign disagreement inconsistent with a single continuous sheet (worklog
    81's own measured signal) is flagged `non_manifold_suspected` rather than
    materialized as if it were clean -- fail-closed, not force-accepted.

This module does not decide chart BOUNDARIES (worklog 80 does that, given a
component's own evidence) and does not parameterize or fit (worklog
61/68/81's existing paths do that, unmodified). It only decides how many
chart-unit components a region's owned evidence actually supports.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

RELATION_SAME_SURFACE = "same_surface"
RELATION_CREASE_VETOED = "crease_vetoed"
RELATION_AMBIGUOUS = "ambiguous"

# Same thresholds `torch_gaussian_manifold_affinity.ManifoldAffinityConfig`
# already uses for its same_surface decision (worklog 111/113/114) -- reused,
# not retuned, so this evidence-scale decision is not a fresh threshold sweep.
DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT = 0.85
DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL = 0.35
DEFAULT_CANDIDATE_NEIGHBOR_COUNT = 8
DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE = 12


@dataclass(frozen=True)
class DenseConsistencyEdge:
    a: int
    b: int
    relation: str  # RELATION_SAME_SURFACE | RELATION_CREASE_VETOED | RELATION_AMBIGUOUS
    normal_alignment: float
    mutual_tangent_residual: float


@dataclass(frozen=True)
class DenseSurfaceConsistencyComponent:
    member_indices: tuple[int, ...]
    non_manifold_suspected: bool
    internal_normal_disagreement_fraction: float


@dataclass(frozen=True)
class DenseSurfaceConsistencyResult:
    region_id: int
    point_count: int
    edges: tuple[DenseConsistencyEdge, ...]
    components: tuple[DenseSurfaceConsistencyComponent, ...]
    unresolved_indices: tuple[int, ...]
    crease_vetoed_edge_count: int


def _residual_scale(positions: Any, k: int) -> Any:
    """Per-point local spacing (median kNN distance), the same role
    `residual_scale`/`candidate_scale` play in the representative-graph
    version -- normalizes tangent residual so the same_surface criterion is
    scale-free across sparse and dense regions of the same region."""
    torch = require_torch()
    n = int(positions.shape[0])
    neighbors = min(k, max(1, n - 1))
    d = torch.cdist(positions, positions)
    d.fill_diagonal_(float("inf"))
    nearest = d.topk(neighbors, dim=1, largest=False).values
    return nearest.median(dim=1).values.clamp_min(_EPS)


def _nearest_arc_side(
    positions: Any, arc_starts: Any, arc_ends: Any, arc_kinds: Sequence[str],
) -> list[str]:
    """Assign each point the `segment_kind` of its nearest worklog-80 chart
    arc, purely so two points on opposite sides of an already-typed
    crease/frontier arc can be told apart -- this NEVER creates a boundary,
    it only reads one that already exists."""
    torch = require_torch()
    n_arcs = int(arc_starts.shape[0])
    if n_arcs == 0:
        return ["" for _ in range(int(positions.shape[0]))]
    ab = arc_ends - arc_starts
    ab_len2 = (ab * ab).sum(dim=1).clamp_min(_EPS)
    best_kind = []
    for i in range(int(positions.shape[0])):
        p = positions[i]
        t = (((p[None, :] - arc_starts) * ab).sum(dim=1) / ab_len2).clamp(0.0, 1.0)
        projection = arc_starts + t[:, None] * ab
        dist = (p[None, :] - projection).norm(dim=1)
        nearest = int(dist.argmin())
        best_kind.append(arc_kinds[nearest])
    return best_kind


def build_dense_surface_consistency_components(
    region_id: int,
    positions: Any,
    *,
    covariance: Any,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
    candidate_neighbor_count: int = DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
    max_candidate_count_per_node: int = DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
    same_surface_min_normal_alignment: float = DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    same_surface_max_mutual_residual: float = DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
) -> DenseSurfaceConsistencyResult:
    """Decompose one region's owned evidence into surface-consistency
    components. ``covariance`` is the (N, 3, 3) per-point covariance already
    used elsewhere (``covariance_from_scale_rotation``, unmodified). Typed
    crease-arc veto is applied only when ``arc_starts``/``arc_ends``/``arc_kinds``
    (worklog 80's own per-arc chart segments, in the SAME frame as
    ``positions``) are supplied; omitting them runs normal/residual-only
    classification (still never spatial proximity alone)."""

    torch = require_torch()
    n = int(positions.shape[0])
    if n == 0:
        return DenseSurfaceConsistencyResult(region_id, 0, (), (), (), 0)

    frame = extract_covariance_frame(covariance)
    normals = frame.normal_candidate
    scale = _residual_scale(positions, candidate_neighbor_count)

    arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        arc_side = _nearest_arc_side(positions, arc_starts, arc_ends, arc_kinds)

    k = min(candidate_neighbor_count, max(1, n - 1))
    d = torch.cdist(positions, positions)
    d.fill_diagonal_(float("inf"))
    knn_indices = d.topk(k, dim=1, largest=False).indices
    knn_set = [set(row.tolist()) for row in knn_indices]

    seen: set[tuple[int, int]] = set()
    pending: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in knn_indices[i].tolist():
            key = (min(i, j), max(i, j))
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            pending.append((key[0], key[1], float(d[key[0], key[1]])))
    # Deterministic-cap ordering (worklog 114 convention, reused): sort by
    # distance before applying the per-node degree cap so the accepted graph
    # never depends on input array order.
    pending.sort(key=lambda row: (row[2], row[0], row[1]))

    per_node_count = [0] * n
    edges: list[DenseConsistencyEdge] = []
    adjacency_same_surface: list[set[int]] = [set() for _ in range(n)]
    crease_vetoed = 0

    for a, b, _dist in pending:
        if per_node_count[a] >= max_candidate_count_per_node or per_node_count[b] >= max_candidate_count_per_node:
            continue

        # Typed-provenance veto: opposite sides of an already-typed
        # crease/frontier arc can never become same_surface, no matter how
        # well normal/residual line up -- reused separator, not invented.
        if arc_side is not None and arc_side[a] and arc_side[b] and arc_side[a] != arc_side[b]:
            crease_vetoed += 1
            edges.append(DenseConsistencyEdge(a, b, RELATION_CREASE_VETOED, 0.0, 0.0))
            per_node_count[a] += 1
            per_node_count[b] += 1
            continue

        displacement = positions[b] - positions[a]
        normal_a, normal_b = normals[a], normals[b]
        alignment = float((normal_a * normal_b).sum().abs())

        residual_a = float((displacement * normal_a).sum().abs() / scale[a])
        residual_b = float((-displacement * normal_b).sum().abs() / scale[b])
        mutual_residual = max(residual_a, residual_b)

        if alignment >= same_surface_min_normal_alignment and mutual_residual <= same_surface_max_mutual_residual:
            edges.append(DenseConsistencyEdge(a, b, RELATION_SAME_SURFACE, alignment, mutual_residual))
            adjacency_same_surface[a].add(b)
            adjacency_same_surface[b].add(a)
        else:
            edges.append(DenseConsistencyEdge(a, b, RELATION_AMBIGUOUS, alignment, mutual_residual))
        per_node_count[a] += 1
        per_node_count[b] += 1

    # Connected components of same_surface edges ONLY (crease/ambiguous never merge).
    component_id = [-1] * n
    components: list[DenseSurfaceConsistencyComponent] = []
    for start in range(n):
        if component_id[start] != -1 or not adjacency_same_surface[start]:
            continue
        current = len(components)
        frontier = [start]
        component_id[start] = current
        member: list[int] = []
        while frontier:
            node = frontier.pop()
            member.append(node)
            for neighbor in adjacency_same_surface[node]:
                if component_id[neighbor] == -1:
                    component_id[neighbor] = current
                    frontier.append(neighbor)

        # Internal non-manifold check: within a claimed single sheet, a
        # member's normal should agree (unsigned) with the component's own
        # dominant normal direction. Large internal disagreement means the
        # accepted-edge chain folded back through near-orthogonal patches
        # (a bow-tie/self-crossing sheet), which same_surface's LOCAL
        # pairwise test alone cannot see -- disclosed, not silently accepted.
        member_normals = normals[torch.tensor(member, dtype=torch.long, device=positions.device)]
        reference = member_normals[0]
        aligned = member_normals.clone()
        flip = (aligned @ reference) < 0.0
        aligned[flip] = -aligned[flip]
        mean_normal = torch.nn.functional.normalize(aligned.mean(dim=0), dim=0)
        dots = (member_normals @ mean_normal).abs()
        disagreement_fraction = float((dots < 0.5).float().mean())
        non_manifold_suspected = disagreement_fraction > 0.15

        components.append(
            DenseSurfaceConsistencyComponent(tuple(member), non_manifold_suspected, disagreement_fraction)
        )

    unresolved = tuple(i for i in range(n) if component_id[i] == -1)
    return DenseSurfaceConsistencyResult(
        region_id, n, tuple(edges), tuple(components), unresolved, crease_vetoed,
    )
