from __future__ import annotations

"""Worklog 101 -- intrinsic-integrability-driven local chart atlas over a
Worklog 98 synchronized tangent frame field component.

Worklog 100 found that forcing ONE global (u, v) chart over an entire
coherent component is the wrong scale for a majority of components: global
differential integration (candidate B) only modestly improves domain
validity over tree integration, and a fixed local-injectivity refinement
(candidate C) rescues ZERO additional components -- every remaining fold is
a genuine, spatially local orientation reversal, not tree-path noise. This
module stops forcing one chart per component and instead decomposes each
component into a deterministic ATLAS of overlapping, individually valid
local charts.

A chart is a subset of the component's OWN continuously-supported source
graph (never a Euclidean bounding box, PCA rectangle, convex hull, or
physical-boundary closure). Chart construction is entirely upstream of any
NURBS fit: fit error, held-out error, extrapolative/unsafe classification
never influence chart creation, growth, or termination (verified by AST in
tests).

Construction, per coherent component:

1. Deterministic anchor (centroid-nearest node, same convention the field
   itself uses for un-anchored components).
2. BFS hop-count RINGS over the component's own supported-edge adjacency
   (:func:`~osn_gs.surface.torch_parametric_domain_validity._source_graph_adjacency`)
   from that anchor -- graph-topological rings, not a tuned Euclidean/PCA
   radius.
3. Grow the candidate chart ring-by-ring: at each ring, integrate Worklog
   100's global differential UV (:func:`~osn_gs.surface.torch_global_differential_uv_integration.integrate_global_differential_uv`)
   over ALL supported edges internal to the candidate node set (never a
   spanning tree), validate with the corrected Worklog 100 validator
   (:func:`~osn_gs.surface.torch_parametric_domain_validity.assess_parametric_domain_validity`).
   Keep growing while valid; stop at the first ring that makes the domain
   invalid, keeping the last valid ring as the maximal accepted chart.
4. Repeat from a new anchor -- the uncovered node with maximum BFS graph
   distance from the current coverage -- until every chartable node is
   covered by at least one valid chart, or the remaining connected
   uncovered support is too small to grow even a minimal chart.

Charts may overlap: growth is never restricted to exclude already-covered
nodes, so a chart's own maximal ring naturally extends into territory a
previous chart already covers when the graph supports it. That overlap is
for geometric reconciliation only -- it is never physical evidence of two
surfaces. Edges connecting two different charts (or a charted node to an
uncovered one) are tagged ``partition_seam`` -- a parametric label only,
never physical-boundary/crease/observation-frontier semantics.

Each chart inherits the parent component's synchronized ``e_u``/``e_v``
(and thus tangent plane) UNCHANGED at every node -- no new tangent field is
estimated per chart. Only the chart's own intrinsic ``(u, v)`` gauge (via
its own global differential integration) is chart-local.
"""

from dataclasses import dataclass, replace
from typing import Any

from osn_gs.surface.torch_global_differential_uv_integration import (
    GlobalIntegrationResult,
    integrate_global_differential_uv,
)
from osn_gs.surface.torch_latent_surface_edge_differential import build_edge_differentials
from osn_gs.surface.torch_latent_surface_tangent_frame_field import TangentFrameFieldComponent
from osn_gs.surface.torch_parametric_domain_validity import (
    ParametricDomainValidityReport,
    _source_graph_adjacency,
    assess_parametric_domain_validity,
)
from osn_gs.utils.torch_ops import require_torch

# Fixed, structural: matches Worklog 98's own MIN_COMPONENT_SIZE (never
# tuned from fit/held-out outcome) -- the minimum node count worth trying
# to grow a chart from at all.
MIN_CHART_SEED_SIZE = 3
# A remaining uncovered connected subgraph smaller than this is reported as
# unchartable rather than grown into a chart -- same floor Worklog 98 uses
# for a component to be considered at all.
MIN_REMAINING_COVERAGE_SIZE = 3


@dataclass(frozen=True)
class Chart:
    chart_id: int
    anchor_node_index: int  # local index into the PARENT component
    node_indices: tuple[int, ...]  # local indices into the PARENT component, sorted
    component: TangentFrameFieldComponent  # restricted view, own (u, v) from candidate B
    integration: GlobalIntegrationResult
    domain_report: ParametricDomainValidityReport
    ring_reached: int


@dataclass(frozen=True)
class AtlasResult:
    charts: tuple[Chart, ...]
    covered_node_indices: frozenset[int]
    multiply_covered_node_indices: frozenset[int]
    uncovered_node_indices: frozenset[int]
    unchartable_seed_node_indices: frozenset[int]  # single/tiny seeds that failed to grow even a minimal valid chart
    seam_edges: tuple[tuple[int, int], ...]  # (a, b) pairs spanning two different charts, or chart<->uncovered


def _hop_distances(adjacency: dict[int, list[int]], source_nodes: set[int]) -> dict[int, int]:
    """Unweighted BFS hop distance from ANY node in ``source_nodes`` (multi-
    source) over ``adjacency`` -- pure graph topology, never a Euclidean or
    PCA metric."""

    from collections import deque

    distances: dict[int, int] = {node: 0 for node in source_nodes}
    frontier = deque(source_nodes)
    while frontier:
        current = frontier.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
    return distances


def _induced_edges(component: TangentFrameFieldComponent, node_set: set[int]) -> tuple[tuple[int, ...], tuple]:
    """Split the parent component's own supported edges (tree_edges union
    holonomy_edges -- exactly the same edge set the validator/field already
    verified) into those fully internal to ``node_set``, preserving the
    tree/holonomy split so downstream diagnostics (cycle-edge residual)
    still work on the restriction."""

    tree_edges = tuple((a, b) for a, b in component.tree_edges if a in node_set and b in node_set)
    holonomy_edges = tuple(
        edge for edge in component.holonomy_edges if edge.node_a in node_set and edge.node_b in node_set
    )
    return tree_edges, holonomy_edges


def _restrict_component(
    component: TangentFrameFieldComponent, node_indices: list[int],
) -> TangentFrameFieldComponent:
    """Build a chart-scoped view: SAME synchronized e_u/e_v/normals at every
    retained node (no new tangent field estimated), edges restricted to
    those fully internal to the subset. ``u``/``v`` are copied through as a
    placeholder (the caller overwrites them with the chart's own candidate-B
    integration immediately after)."""

    torch = require_torch()
    node_set = set(node_indices)
    selector = torch.tensor(sorted(node_indices), dtype=torch.long, device=component.positions.device)
    local_to_new = {old: new for new, old in enumerate(sorted(node_indices))}

    tree_edges, holonomy_edges = _induced_edges(component, node_set)
    remapped_tree = tuple((local_to_new[a], local_to_new[b]) for a, b in tree_edges)
    remapped_holonomy = tuple(
        replace(edge, node_a=local_to_new[edge.node_a], node_b=local_to_new[edge.node_b])
        for edge in holonomy_edges
    )

    # Geometry provenance (Worklog 103): a chart is a restriction, never a
    # re-derivation -- raw_positions/projection_displacement/latent_supported
    # must survive the restriction unchanged, indexed the same way
    # positions/normals/e_u/e_v already are. Older parent components built
    # before this contract (raw_positions=None) leave the restriction's
    # provenance fields None too, rather than fabricating them.
    raw_positions = component.raw_positions[selector] if component.raw_positions is not None else None
    projection_displacement = (
        component.projection_displacement[selector] if component.projection_displacement is not None else None
    )
    latent_supported = component.latent_supported[selector] if component.latent_supported is not None else None

    return TangentFrameFieldComponent(
        node_indices=tuple(component.node_indices[i] for i in sorted(node_indices)),
        positions=component.positions[selector], normals=component.normals[selector],
        e_u=component.e_u[selector], e_v=component.e_v[selector],
        u=component.u[selector], v=component.v[selector],
        tree_edges=remapped_tree, holonomy_edges=remapped_holonomy,
        singularities=(), coherent=True, incoherence_reason=None,
        anchor_seed_type=component.anchor_seed_type,
        raw_positions=raw_positions, projection_displacement=projection_displacement,
        latent_supported=latent_supported,
    )


def _try_chart(
    component: TangentFrameFieldComponent, node_indices: list[int], median_spacing: float,
) -> tuple[TangentFrameFieldComponent, GlobalIntegrationResult, ParametricDomainValidityReport] | None:
    if len(node_indices) < MIN_CHART_SEED_SIZE:
        return None
    restricted = _restrict_component(component, node_indices)
    edges = build_edge_differentials(restricted, median_spacing)
    integration = integrate_global_differential_uv(restricted, edges)
    if not integration.valid:
        return None
    restricted_with_uv = replace(restricted, u=integration.uv[:, 0], v=integration.uv[:, 1])
    report = assess_parametric_domain_validity(restricted_with_uv, integration.uv, median_spacing)
    if not report.valid:
        return None
    return restricted_with_uv, integration, report


def _grow_maximal_chart(
    component: TangentFrameFieldComponent, adjacency: dict[int, list[int]], anchor: int, median_spacing: float,
) -> Chart | None:
    """Ring-by-ring growth from ``anchor``: keep expanding the BFS
    hop-count ring while the candidate chart integrates and validates;
    stop and keep the last valid ring at the first ring that fails either
    step. Never informed by any downstream fit/held-out outcome."""

    hop_distance = _hop_distances(adjacency, {anchor})
    max_ring = max(hop_distance.values()) if hop_distance else 0

    best: tuple[list[int], TangentFrameFieldComponent, GlobalIntegrationResult, ParametricDomainValidityReport, int] | None = None
    for ring in range(0, max_ring + 1):
        candidate_nodes = sorted(node for node, distance in hop_distance.items() if distance <= ring)
        if best is not None and len(candidate_nodes) == len(best[0]):
            continue  # ring did not add any node (disconnected remainder) -- no point re-trying
        if len(candidate_nodes) < MIN_CHART_SEED_SIZE:
            continue  # too small to attempt integration yet -- keep growing, this is not a validity failure
        result = _try_chart(component, candidate_nodes, median_spacing)
        if result is None:
            break
        restricted, integration, report = result
        best = (candidate_nodes, restricted, integration, report, ring)

    if best is None:
        return None
    node_indices, restricted, integration, report, ring = best
    return Chart(
        chart_id=-1,  # assigned by the caller
        anchor_node_index=anchor, node_indices=tuple(node_indices),
        component=restricted, integration=integration, domain_report=report, ring_reached=ring,
    )


def _farthest_uncovered(
    adjacency: dict[int, list[int]], covered: set[int], all_nodes: set[int], excluded: set[int] = frozenset(),
) -> int | None:
    """Deterministic next anchor for atlas coverage: the UNCOVERED,
    not-yet-retired node with maximum BFS graph distance from the current
    coverage. ``excluded`` (already-retired ``unchartable_seeds``) is
    subtracted so a definitively failed anchor is never re-selected -- used
    only to pick where to grow the next chart, never recorded as a physical
    boundary/crease/observation-frontier/structural curve seed."""

    uncovered = all_nodes - covered - excluded
    if not uncovered:
        return None
    if not covered:
        # No coverage yet: fall back to the lowest-index node for full
        # determinism (matches this module's own first-anchor convention).
        return min(uncovered)
    distances = _hop_distances(adjacency, covered)
    reachable_uncovered = {node: distances[node] for node in uncovered if node in distances}
    if not reachable_uncovered:
        # Disconnected remainder relative to `covered` -- pick the
        # lowest-index node in it, deterministic.
        return min(uncovered)
    max_distance = max(reachable_uncovered.values())
    candidates = [node for node, distance in reachable_uncovered.items() if distance == max_distance]
    return min(candidates)


def _connected_size(adjacency: dict[int, list[int]], seed: int, allowed: set[int]) -> int:
    from collections import deque

    visited = {seed}
    frontier = deque([seed])
    while frontier:
        current = frontier.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor in allowed and neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited)


def build_local_chart_atlas(component: TangentFrameFieldComponent, median_spacing: float) -> AtlasResult:
    """Deterministically decompose ``component`` into a maximal atlas of
    overlapping, individually-valid local charts. Never uses fit/held-out
    error to create, resize, merge, or split a chart."""

    torch = require_torch()
    adjacency = _source_graph_adjacency(component)
    all_nodes = set(range(int(component.positions.shape[0])))

    charts: list[Chart] = []
    covered: set[int] = set()
    coverage_count: dict[int, int] = {}
    unchartable_seeds: set[int] = set()

    guard = 0
    while True:
        guard += 1
        if guard > len(all_nodes) + 1:
            break  # defensive -- coverage strictly grows or an anchor is retired every iteration
        anchor = _farthest_uncovered(adjacency, covered, all_nodes, unchartable_seeds)
        if anchor is None:
            break

        remaining_connected_size = _connected_size(adjacency, anchor, all_nodes - unchartable_seeds)
        if remaining_connected_size < MIN_REMAINING_COVERAGE_SIZE:
            unchartable_seeds.add(anchor)
            continue

        chart = _grow_maximal_chart(component, adjacency, anchor, median_spacing)
        if chart is None:
            # Not even the minimal seed around this anchor integrates/
            # validates -- this node cannot anchor a chart. Retire it so
            # coverage always progresses, and report it explicitly.
            unchartable_seeds.add(anchor)
            continue

        chart = replace(chart, chart_id=len(charts))
        charts.append(chart)
        for node in chart.node_indices:
            coverage_count[node] = coverage_count.get(node, 0) + 1
            covered.add(node)

    multiply_covered = frozenset(node for node, count in coverage_count.items() if count > 1)
    # A node retired as a failed ANCHOR can still end up covered as a
    # member of a valid chart grown from a different anchor -- membership
    # (covered) always takes precedence over anchor-failure history when
    # reporting final categories, so the three sets stay a strict partition.
    unchartable_seeds -= covered
    uncovered = frozenset(all_nodes - covered - unchartable_seeds)

    # Seam edges: any supported edge of the FULL component graph whose
    # endpoints are not BOTH members of the SAME chart (spans two charts,
    # or a charted node to an uncovered/unchartable one). Parametric
    # semantics only.
    chart_membership: dict[int, set[int]] = {}
    for chart in charts:
        for node in chart.node_indices:
            chart_membership.setdefault(node, set()).add(chart.chart_id)

    seam_edges: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge_key = tuple(sorted((node, neighbor)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            charts_a = chart_membership.get(edge_key[0], set())
            charts_b = chart_membership.get(edge_key[1], set())
            if not (charts_a & charts_b):
                seam_edges.append(edge_key)

    return AtlasResult(
        charts=tuple(charts),
        covered_node_indices=frozenset(covered),
        multiply_covered_node_indices=multiply_covered,
        uncovered_node_indices=uncovered,
        unchartable_seed_node_indices=frozenset(unchartable_seeds),
        seam_edges=tuple(seam_edges),
    )
