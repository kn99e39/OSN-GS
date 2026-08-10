from __future__ import annotations

"""Worklog 86: partition-seam parametric chart-domain contract.

Worklog 85 closed the STRICT physical-perimeter reconstruction path:
requiring a chart's entire boundary to be a closed loop of observed
boundary-support Gaussians left 83% of coherent, chart-scale-assembled
evidence (Worklog 82/83/84) with no valid manifold topology, because real
admitted boundary candidates fragment into many small closed loops and open
paths rather than one dominant perimeter -- not an algorithm defect (the
graph mechanism was verified sound on synthetic data both before and after a
self-caught search-pool bug fix), but a property of the evidence itself.

This module does not weaken or re-run that physical reconstruction -- it is
reused completely unchanged (`materialize_chart_unit_boundary`, Worklog 85).
It instead corrects a conflation the whole prior chart-boundary line shared:
physical surface termination and parametric chart boundary are NOT the same
thing. A parametric chart is a regular fittable domain; its boundary may
legitimately include an evidence-backed PARTITION SEAM -- an internal cut
through the unit's own coherent interior -- alongside physical_termination /
crease / observation_frontier segments, exactly as standard surface
parameterization cuts a topologically complex patch to make it disk-
shaped, provided the cut is intrinsic (evidence/topology-derived), not
invented or fit-driven.

Contract, in order:

  1. Try Worklog 85's physical-only reconstruction FIRST, completely
     unchanged. If it already materializes a closed loop, that IS the chart
     boundary -- no seam is ever considered (`boundary_composition =
     "physical_only"`).
  2. Only when physical reconstruction finds NO closed loop at all, but
     finds EXACTLY ONE genuine open fragment (a path of physical boundary
     candidates -- itself real evidence, just not yet closed), a partition
     seam is attempted: the shortest path through the unit's OWN interior
     local-2-manifold adjacency graph (`build_same_surface_adjacency`,
     Worklog 82's own INTERIOR-mesh defaults k=8/cap=12, unmodified -- the
     appropriate graph for this 2D-interior use, as opposed to Worklog 85's
     curve-specific unrestricted-pool/degree-2-cap graph) connecting the
     fragment's two loose ends. Every edge of a found seam is, by
     construction, a real same_surface adjacency between two observed
     points -- never a chord across unclassified space. Other boundary
     candidates are excluded as seam intermediates (a seam routes through
     genuine interior evidence, not through stray unclaimed boundary
     points), and the SAME typed-crease veto already used everywhere else
     applies here too, so a seam can never cross an existing physical crease
     -- physical/crease/frontier evidence remains a hard constraint.
  3. If two or more disjoint open fragments exist, no seam is attempted --
     multi-fragment stitching is combinatorially ambiguous and this module
     does not force a choice; the unit is disclosed
     `STATE_MULTI_FRAGMENT_UNRESOLVED`, not guessed at.
  4. If a seam is found, the combined loop (physical fragment + seam,
     forming exactly one cycle since both are open paths sharing the same
     two endpoints) is validated by the SAME two independent safety checks
     Worklog 85 already uses -- self-intersection
     (`evaluate_closed_loop_geometry`) and observed-support occupancy
     (`measure_edge_support_occupancy`) -- then the Worklog 79 coverage
     contract, all unmodified.

Nothing here ever reads fit error, held-out p95, NURBS quality, or desired
patch count -- the seam decision is made and validated entirely before any
parameterization or fitting is attempted. Physical and partition-seam
segments are always reported distinctly (`ChartUnitDomainSegment.is_partition_
seam`) so a materialized domain never silently claims physical evidence it
does not have.

No hull, no PCA rectangle, no bounding box, no alpha shape, no forced
closure, no unsupported gap bridging, no region merge, no centroid-angle
sort, no threshold tuned toward patch counts.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_boundary_support_spacing import measure_edge_support_occupancy
from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import (
    STATE_AMBIGUOUS_OR_OVER_MERGED,
    STATE_COVERAGE_FAILED,
    STATE_MATERIALIZED,
    STATE_NO_DENSE_SUPPORT,
    STATE_NO_VALID_LOOP_TOPOLOGY,
    STATE_SELF_INTERSECTING,
    STATE_UNSUPPORTED_CLOSURE,
    BOUNDARY_CURVE_MAX_DEGREE,
    ChartUnitCoherence,
    assess_chart_unit_coherence,
    materialize_chart_unit_boundary,
)
from osn_gs.surface.torch_dense_surface_consistency_components import (
    DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
    DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
    DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND,
    _nearest_arc_side,
    build_same_surface_adjacency,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_region_owned_full_evidence import (
    MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
    evidence_outside_chart_domain_fraction,
)
from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import evaluate_closed_loop_geometry
from osn_gs.utils.torch_ops import require_torch

PHYSICAL_ONLY = "physical_only"
PHYSICAL_PLUS_PARTITION_SEAM = "physical_plus_partition_seam"

STATE_MULTI_FRAGMENT_UNRESOLVED = "chart_unit_domain_multi_fragment_unresolved"
STATE_SEAM_NOT_FOUND = "chart_unit_domain_seam_not_found"


@dataclass(frozen=True)
class ChartUnitDomainSegment:
    stable_id_a: Any
    stable_id_b: Any
    segment_kind: str  # nearest sparse-arc label, "" if uncovered, or "partition_seam"
    is_partition_seam: bool
    crease_inconsistent: bool


@dataclass(frozen=True)
class ChartUnitDomainResult:
    state: str
    coherence: ChartUnitCoherence | None
    ordered_stable_ids: tuple[Any, ...]
    ordered_positions: Any | None
    segments: tuple[ChartUnitDomainSegment, ...]
    evidence_outside_domain_fraction: float | None
    boundary_composition: str  # PHYSICAL_ONLY | PHYSICAL_PLUS_PARTITION_SEAM | ""
    physical_segment_count: int
    partition_seam_segment_count: int
    open_fragment_component_count: int
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


def _trace_open_path(component: set[int], adjacency: list[set[int]]) -> list[int]:
    """Walk a connected component with exactly two degree-1 endpoints (and
    every other member at degree 2) into its unique path order."""

    endpoints = sorted(node for node in component if len(adjacency[node] & component) == 1)
    start = endpoints[0]
    order = [start]
    prev = None
    current = start
    while True:
        neighbors = [v for v in adjacency[current] if v in component and v != prev]
        if not neighbors:
            break
        prev, current = current, neighbors[0]
        order.append(current)
    return order


def _find_open_paths(candidate_count: int, adjacency: list[set[int]]) -> list[list[int]]:
    """Genuine open paths (>=2 members, exactly two degree-1 endpoints, no
    branching) among the candidate adjacency graph's connected components --
    each a real physical boundary FRAGMENT that did not close into a loop."""

    visited: set[int] = set()
    paths: list[list[int]] = []
    for start in range(candidate_count):
        if start in visited:
            continue
        stack, component_list = [start], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_list.append(node)
            stack.extend(adjacency[node] - visited)
        component = set(component_list)
        if len(component) < 2:
            continue
        degrees = {node: len(adjacency[node] & component) for node in component}
        endpoint_count = sum(1 for d in degrees.values() if d == 1)
        interior_ok = all(d in (1, 2) for d in degrees.values())
        if endpoint_count == 2 and interior_ok:
            paths.append(_trace_open_path(component, adjacency))
    return paths


def _find_partition_seam(
    full_positions: Any,
    full_normals: Any,
    full_arc_side: list[str] | None,
    excluded_full_indices: set[int],
    endpoint_a: int,
    endpoint_b: int,
) -> list[int] | None:
    """Shortest path through the unit's OWN interior local-2-manifold
    adjacency graph (Worklog 82's interior-mesh defaults, unmodified)
    connecting two physical-fragment endpoints, routed through genuine
    interior evidence only (other boundary candidates excluded as
    intermediates). Returns full-unit-index path (endpoints included) or
    ``None`` if the two points are not evidence-connected at all."""

    _edges, adjacency, _crease_vetoed = build_same_surface_adjacency(
        full_positions, full_normals, arc_side=full_arc_side,
        candidate_neighbor_count=DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
        max_candidate_count_per_node=DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
        same_surface_min_normal_alignment=DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
        same_surface_max_mutual_residual=DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    )
    blocked = excluded_full_indices - {endpoint_a, endpoint_b}
    frontier = [endpoint_a]
    came_from: dict[int, int] = {endpoint_a: -1}
    while frontier:
        next_frontier = []
        for node in frontier:
            if node == endpoint_b:
                path = [node]
                while came_from[path[-1]] != -1:
                    path.append(came_from[path[-1]])
                return list(reversed(path))
            for neighbor in sorted(adjacency[node]):
                if neighbor in blocked or neighbor in came_from:
                    continue
                came_from[neighbor] = node
                next_frontier.append(neighbor)
        frontier = next_frontier
    return None


def materialize_chart_unit_domain(
    positions: Any,
    covariance: Any,
    stable_ids: Sequence[Any],
    full_evidence_positions: Any,
    *,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
    max_evidence_outside_domain_fraction: float = MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
) -> ChartUnitDomainResult:
    """Recover a parametric chart DOMAIN for a coherent assembled unit --
    physical boundary when it closes on its own (Worklog 85, unchanged), a
    physical fragment plus an evidence-backed partition seam otherwise, or a
    typed fail-closed disclosure. Never touches fit quality."""

    torch = require_torch()
    n = int(positions.shape[0])

    physical = materialize_chart_unit_boundary(
        positions, covariance, stable_ids, full_evidence_positions,
        arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds,
        max_evidence_outside_domain_fraction=max_evidence_outside_domain_fraction,
    )

    def _pass_through(state: str, *, boundary_composition: str = "") -> ChartUnitDomainResult:
        segments = tuple(
            ChartUnitDomainSegment(s.stable_id_a, s.stable_id_b, s.segment_kind, False, s.crease_inconsistent)
            for s in physical.segments
        )
        return ChartUnitDomainResult(
            state, physical.coherence, physical.ordered_stable_ids, physical.ordered_positions, segments,
            physical.evidence_outside_domain_fraction, boundary_composition, len(segments), 0,
            physical.open_fragment_component_count, physical.reasons,
        )

    if physical.materialized:
        return _pass_through(STATE_MATERIALIZED, boundary_composition=PHYSICAL_ONLY)
    if physical.state != STATE_NO_VALID_LOOP_TOPOLOGY:
        # Coherence/admission failure, self-intersection, unsupported-closure,
        # or coverage failure on the largest PHYSICAL loop attempt -- Worklog
        # 85's own fail-closed states are reused verbatim, no seam attempted.
        return _pass_through(physical.state)

    def _fail(state: str, *reasons: str) -> ChartUnitDomainResult:
        return ChartUnitDomainResult(
            state, physical.coherence, (), None, (), None, "", 0, 0,
            physical.open_fragment_component_count, tuple(reasons),
        )

    # Re-derive the candidate graph (Worklog 85's own construction,
    # unmodified parameters) purely to access the open-path list itself --
    # `materialize_chart_unit_boundary` only exposes counts.
    normals = extract_covariance_frame(covariance).normal_candidate
    support = extract_dense_boundary_support(positions, normals, list(stable_ids))
    candidate_ids = [c.stable_id for c in support.candidates]
    candidate_positions = torch.stack(
        [torch.tensor(c.position, dtype=positions.dtype, device=positions.device) for c in support.candidates], dim=0,
    )
    candidate_normals = torch.stack(
        [torch.tensor(c.normal, dtype=positions.dtype, device=positions.device) for c in support.candidates], dim=0,
    )
    candidate_arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        candidate_arc_side = _nearest_arc_side(candidate_positions, arc_starts, arc_ends, arc_kinds)
    _edges, candidate_adjacency, _crease_vetoed = build_same_surface_adjacency(
        candidate_positions, candidate_normals, arc_side=candidate_arc_side,
        candidate_neighbor_count=max(1, len(support.candidates) - 1),
        max_candidate_count_per_node=BOUNDARY_CURVE_MAX_DEGREE,
        same_surface_min_normal_alignment=DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
        same_surface_max_mutual_residual=DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    )
    open_paths = _find_open_paths(len(support.candidates), candidate_adjacency)
    if len(open_paths) != 1:
        return _fail(
            STATE_MULTI_FRAGMENT_UNRESOLVED if len(open_paths) > 1 else STATE_NO_VALID_LOOP_TOPOLOGY,
            f"open_path_count={len(open_paths)}",
        )

    fragment = open_paths[0]
    fragment_ids = [candidate_ids[i] for i in fragment]
    endpoint_a_id, endpoint_b_id = fragment_ids[0], fragment_ids[-1]
    id_to_full_index = {sid: i for i, sid in enumerate(stable_ids)}
    if endpoint_a_id not in id_to_full_index or endpoint_b_id not in id_to_full_index:
        return _fail(STATE_SEAM_NOT_FOUND, "fragment_endpoint_not_in_full_unit")

    full_normals = normals
    full_arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        full_arc_side = _nearest_arc_side(positions, arc_starts, arc_ends, arc_kinds)
    other_candidate_full_indices = {
        id_to_full_index[sid] for sid in candidate_ids if sid in id_to_full_index and sid not in (endpoint_a_id, endpoint_b_id)
    }
    seam_full_indices = _find_partition_seam(
        positions, full_normals, full_arc_side, other_candidate_full_indices,
        id_to_full_index[endpoint_a_id], id_to_full_index[endpoint_b_id],
    )
    if seam_full_indices is None:
        return _fail(STATE_SEAM_NOT_FOUND, "no_interior_adjacency_path_between_fragment_endpoints")

    # Combined loop: physical fragment (endpoint_a -> ... -> endpoint_b) then
    # the seam back (endpoint_b -> ... -> endpoint_a), excluding the shared
    # endpoints from the seam's own middle so each vertex appears once.
    seam_stable_ids = [stable_ids[i] for i in seam_full_indices]
    combined_ids = fragment_ids + seam_stable_ids[1:-1]
    fragment_positions = [candidate_positions[i] for i in fragment]
    seam_positions = [positions[i] for i in seam_full_indices[1:-1]]
    ordered_positions = torch.stack(fragment_positions + seam_positions, dim=0)

    geometry = evaluate_closed_loop_geometry(
        [tuple(float(v) for v in row) for row in ordered_positions.detach().cpu().tolist()]
    )
    if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
        return _fail(STATE_SELF_INTERSECTING, f"proper_crossing_count={geometry.proper_crossing_count}")

    k = len(combined_ids)
    edge_pairs = [(i, (i + 1) % k) for i in range(k)]
    occupancy = measure_edge_support_occupancy(
        edge_pairs, ordered_positions, positions, full_evidence_spacing=support.full_evidence_scale,
    )
    if occupancy["edges_with_empty_interior_bin"] > 0:
        return _fail(
            STATE_UNSUPPORTED_CLOSURE,
            f"edges_with_empty_interior_bin={occupancy['edges_with_empty_interior_bin']}/{occupancy['edge_count']}",
        )

    physical_edge_count = len(fragment) - 1
    arc_side_ordered = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        arc_side_ordered = _nearest_arc_side(ordered_positions, arc_starts, arc_ends, arc_kinds)

    segments = []
    partition_count = 0
    physical_count = 0
    for i in range(k):
        is_seam_edge = i >= physical_edge_count  # last physical->seam bridge and every seam-internal edge
        kind_a = arc_side_ordered[i] if arc_side_ordered is not None else ""
        kind_b = arc_side_ordered[(i + 1) % k] if arc_side_ordered is not None else ""
        inconsistent = bool(kind_a and kind_b and kind_a != kind_b and ("crease" in kind_a or "crease" in kind_b))
        if is_seam_edge:
            partition_count += 1
            segments.append(ChartUnitDomainSegment(combined_ids[i], combined_ids[(i + 1) % k], "partition_seam", True, inconsistent))
        else:
            physical_count += 1
            segments.append(ChartUnitDomainSegment(combined_ids[i], combined_ids[(i + 1) % k], kind_a or kind_b, False, inconsistent))

    outside = evidence_outside_chart_domain_fraction(ordered_positions, full_evidence_positions)
    if outside is not None and outside > max_evidence_outside_domain_fraction:
        return ChartUnitDomainResult(
            STATE_COVERAGE_FAILED, physical.coherence, tuple(combined_ids), ordered_positions, tuple(segments),
            outside, PHYSICAL_PLUS_PARTITION_SEAM, physical_count, partition_count,
            physical.open_fragment_component_count,
            (f"evidence_outside_chart_domain_fraction={outside:.4f}>{max_evidence_outside_domain_fraction}",),
        )

    return ChartUnitDomainResult(
        STATE_MATERIALIZED, physical.coherence, tuple(combined_ids), ordered_positions, tuple(segments),
        outside, PHYSICAL_PLUS_PARTITION_SEAM, physical_count, partition_count,
        physical.open_fragment_component_count, (),
    )
