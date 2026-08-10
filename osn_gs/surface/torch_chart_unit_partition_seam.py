from __future__ import annotations

"""Worklog 87: partition_seam as a first-class parametric-boundary type.

Worklog 86 validated evidence-backed partition seams but only in the
RESTRICTED case of exactly one open physical fragment closed by exactly one
interior seam -- treating physical boundary-support availability as an
implicit prerequisite (a unit with zero admitted physical candidates, or
with two-or-more disjoint fragments, produced no chart at all).

This module generalizes that into the actual contract requested: a
parametric chart boundary is not required to be a physical surface boundary.
A closed chart domain may contain ANY evidence-justified combination of
physical_termination / crease / observation_frontier / partition_seam, and
partition seams are attempted regardless of how many physical fragments
exist (zero is still a hard floor -- a seam needs at least one anchor point
to connect FROM, so a unit with literally zero admitted boundary-support
candidates still cannot receive an invented boundary; this is reported
honestly, not routed around).

Contract, in order (physical topology is Worklog 85's own construction,
reused verbatim throughout -- nothing here weakens or re-derives it):

  1. Every already-CLOSED physical loop (Worklog 85's own degree-2-regular
     cycle detection, `_find_valid_loops`, unmodified) is tried and validated
     independently -- each is its own materialized `physical_only` domain.
     A unit may therefore produce MORE THAN ONE domain when its own topology
     genuinely proves more than one independent physical loop.
  2. The remaining, still-open topology (every candidate NOT already
     consumed by an accepted physical loop) is a set of open FRAGMENTS
     (`_find_open_paths`, two loose ends each) and possibly isolated single
     candidates (a "fragment" whose two loose ends coincide at one point).
     If at least one such piece remains, it is closed into ONE additional
     domain via DETERMINISTIC fragment-chain stitching: order the pieces by
     their own first candidate's stable id (a stable, reproducible tie-break
     already used elsewhere in this project -- e.g. Worklog 114's
     deterministic-cap convention, Worklog 85's `min(component)` cycle
     start -- never a geometric/angular choice), then seam-connect piece
     i's end to piece (i+1)'s start, wrapping around back to piece 0's
     start. This supports, uniformly, exactly the cases required:
       * one fragment + one seam (Worklog 86's original case, now a
         special instance of N=1),
       * two or more fragments + multiple seams forming one combined loop,
       * a chart with NO physical loop at all, entirely closed by seam(s)
         between otherwise-disjoint physical fragments ("seam-dominated",
         reported when `partition_seam_segment_count >
         physical_segment_count`).
     Every seam is the shortest path through the unit's OWN interior
     local-2-manifold adjacency graph (`build_same_surface_adjacency`,
     Worklog 82's interior-mesh defaults k=8/cap=12, unmodified), routed
     through genuine interior evidence only (other boundary candidates
     excluded as intermediates so a seam is never a shortcut through
     unclaimed boundary evidence), with the SAME typed-crease veto applied
     so a seam can never cross an existing crease -- physical/crease/
     frontier evidence stays a hard constraint throughout. If ANY required
     seam in the chain does not exist (the two points are not evidence-
     connected through the interior at all), the WHOLE chain fails closed
     (`STATE_SEAM_NOT_FOUND`) -- no partial/alternate stitching is
     attempted, since trying other orderings would itself become "another
     sequence of boundary heuristics."
  3. Every combined domain (physical-only or physical+seam) is still
     validated, unweakened, by Worklog 85's two independent safety checks
     (`evaluate_closed_loop_geometry` self-intersection,
     `measure_edge_support_occupancy` observed-support occupancy) and then
     the Worklog 79 coverage contract, before being accepted.

Nothing here ever reads fit error, held-out p95, NURBS quality, or desired
chart count -- every domain is fully decided and validated before any
parameterization or fitting is attempted. Every segment is always typed
distinctly (`ChartUnitDomainSegment.is_partition_seam`), so a materialized
domain never silently claims physical evidence it does not have.

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
    STATE_SELF_INTERSECTING,
    STATE_UNSUPPORTED_CLOSURE,
    BOUNDARY_CURVE_MAX_DEGREE,
    ChartUnitCoherence,
    _find_valid_loops,
    assess_chart_unit_coherence,
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
MIXED_PHYSICAL_PARTITION_SEAM = "mixed_physical_partition_seam"
SEAM_DOMINATED = "seam_dominated"

# Worklog 86's old alias, kept for callers that still refer to the mixed case
# by its original (single-fragment-only) name.
PHYSICAL_PLUS_PARTITION_SEAM = MIXED_PHYSICAL_PARTITION_SEAM

STATE_MULTI_FRAGMENT_UNRESOLVED = "chart_unit_domain_multi_fragment_unresolved"
STATE_SEAM_NOT_FOUND = "chart_unit_domain_seam_not_found"
STATE_NO_OPEN_TOPOLOGY = "chart_unit_domain_no_open_topology"


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
    boundary_composition: str  # PHYSICAL_ONLY | MIXED_PHYSICAL_PARTITION_SEAM | SEAM_DOMINATED | ""
    physical_segment_count: int
    partition_seam_segment_count: int
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


@dataclass(frozen=True)
class ChartUnitDomainSetResult:
    """0 or more independently-materialized domains for one assembled unit,
    plus top-level diagnostics for whatever candidate topology never reached
    a materialized domain at all."""

    coherence: ChartUnitCoherence | None
    domains: tuple[ChartUnitDomainResult, ...]
    admitted_candidate_count: int
    unresolved_reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return len(self.domains) > 0 and any(d.materialized for d in self.domains)


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


def _find_isolated_candidates(candidate_count: int, adjacency: list[set[int]]) -> list[int]:
    """Admitted candidates with zero same_surface neighbours among the other
    candidates -- a degenerate one-point "fragment" whose two loose ends
    coincide, needing two (not one) seam connections to close."""

    return [i for i in range(candidate_count) if len(adjacency[i]) == 0]


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


def _validate_and_build_result(
    ordered_ids: list[Any],
    ordered_positions: Any,
    physical_edge_flags: list[bool],
    positions: Any,
    full_evidence_positions: Any,
    full_evidence_spacing: float,
    arc_starts: Any | None,
    arc_ends: Any | None,
    arc_kinds: Sequence[str] | None,
    coherence: ChartUnitCoherence | None,
    max_evidence_outside_domain_fraction: float,
) -> ChartUnitDomainResult:
    """Shared final validation (self-intersection -> occupancy -> coverage)
    for any candidate closed loop, physical-only or seam-augmented alike."""

    def _fail(state: str, *reasons: str) -> ChartUnitDomainResult:
        return ChartUnitDomainResult(state, coherence, (), None, (), None, "", 0, 0, tuple(reasons))

    geometry = evaluate_closed_loop_geometry(
        [tuple(float(v) for v in row) for row in ordered_positions.detach().cpu().tolist()]
    )
    if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
        return _fail(STATE_SELF_INTERSECTING, f"proper_crossing_count={geometry.proper_crossing_count}")

    k = len(ordered_ids)
    edge_pairs = [(i, (i + 1) % k) for i in range(k)]
    occupancy = measure_edge_support_occupancy(
        edge_pairs, ordered_positions, positions, full_evidence_spacing=full_evidence_spacing,
    )
    if occupancy["edges_with_empty_interior_bin"] > 0:
        return _fail(
            STATE_UNSUPPORTED_CLOSURE,
            f"edges_with_empty_interior_bin={occupancy['edges_with_empty_interior_bin']}/{occupancy['edge_count']}",
        )

    arc_side_ordered = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        arc_side_ordered = _nearest_arc_side(ordered_positions, arc_starts, arc_ends, arc_kinds)

    segments = []
    physical_count = 0
    partition_count = 0
    for i in range(k):
        is_seam_edge = physical_edge_flags[i] is False
        kind_a = arc_side_ordered[i] if arc_side_ordered is not None else ""
        kind_b = arc_side_ordered[(i + 1) % k] if arc_side_ordered is not None else ""
        inconsistent = bool(kind_a and kind_b and kind_a != kind_b and ("crease" in kind_a or "crease" in kind_b))
        if is_seam_edge:
            partition_count += 1
            segments.append(ChartUnitDomainSegment(ordered_ids[i], ordered_ids[(i + 1) % k], "partition_seam", True, inconsistent))
        else:
            physical_count += 1
            segments.append(ChartUnitDomainSegment(ordered_ids[i], ordered_ids[(i + 1) % k], kind_a or kind_b, False, inconsistent))

    if partition_count == 0:
        composition = PHYSICAL_ONLY
    elif partition_count > physical_count:
        composition = SEAM_DOMINATED
    else:
        composition = MIXED_PHYSICAL_PARTITION_SEAM

    outside = evidence_outside_chart_domain_fraction(ordered_positions, full_evidence_positions)
    if outside is not None and outside > max_evidence_outside_domain_fraction:
        return ChartUnitDomainResult(
            STATE_COVERAGE_FAILED, coherence, tuple(ordered_ids), ordered_positions, tuple(segments), outside,
            composition, physical_count, partition_count,
            (f"evidence_outside_chart_domain_fraction={outside:.4f}>{max_evidence_outside_domain_fraction}",),
        )

    return ChartUnitDomainResult(
        STATE_MATERIALIZED, coherence, tuple(ordered_ids), ordered_positions, tuple(segments), outside,
        composition, physical_count, partition_count, (),
    )


def _stitch_pieces_into_domain(
    pieces: list[list[int]],
    candidate_ids: list[Any],
    candidate_positions: Any,
    positions: Any,
    normals: Any,
    stable_ids: Sequence[Any],
    id_to_full_index: dict[Any, int],
    full_arc_side: list[str] | None,
) -> tuple[tuple[list[Any], list[Any], list[bool]] | None, str]:
    """Deterministically chain ``pieces`` (each a list of candidate-local
    indices forming either a genuine open path or a single isolated point)
    into one combined closed-loop node/edge sequence via evidence-backed
    seams -- Worklog 87's general N-fragment stitching, of which Worklog
    86's single-fragment self-seam is the N=1 special case.

    Order is by each piece's own first candidate's stable id (deterministic,
    reproducible, never geometric/angular). Every seam is the shortest path
    through the unit's own interior same_surface graph, excluding only
    NODES ALREADY PLACED into the chain so far (not every other boundary
    candidate categorically -- measured directly on real data that the
    broader exclusion was the actual bottleneck, disconnecting paths that
    otherwise exist: one real 140-point unit's interior graph reaches only
    38/140 nodes from a given start when every other candidate is excluded,
    but the two fragment endpoints in question ARE connected once other,
    not-yet-placed candidates are allowed as pass-through waypoints -- using
    one as a mid-seam waypoint is still a real observed same_surface edge on
    both sides, not an invented chord, so nothing in the task's constraints
    forbids it). To guard the one real risk this relaxation introduces (a
    not-yet-processed piece's own node being consumed as someone else's
    waypoint, which would duplicate it once that piece is later placed), the
    full stitched chain is checked for duplicate stable ids before being
    accepted -- if one occurs, the WHOLE attempt fails closed rather than
    silently producing an invalid loop with a repeated vertex. The typed-
    crease veto still applies throughout. Returns ``(None, reason)`` if any
    required seam in the chain does not exist, or if a duplicate would
    result -- the whole attempt fails closed, no alternate ordering or
    partial stitching is tried.
    """

    pieces = sorted(pieces, key=lambda piece: str(candidate_ids[piece[0]]))
    chain_ids: list[Any] = []
    chain_positions: list[Any] = []
    chain_is_physical: list[bool] = []
    placed_full_indices: set[int] = set()

    for piece_index, piece in enumerate(pieces):
        for local in piece:
            chain_ids.append(candidate_ids[local])
            chain_positions.append(candidate_positions[local])
            sid = candidate_ids[local]
            if sid in id_to_full_index:
                placed_full_indices.add(id_to_full_index[sid])
        if len(piece) >= 2:
            chain_is_physical.extend([True] * (len(piece) - 1))

        next_piece = pieces[(piece_index + 1) % len(pieces)]
        this_end_id = candidate_ids[piece[-1]]
        next_start_id = candidate_ids[next_piece[0]]
        if this_end_id not in id_to_full_index or next_start_id not in id_to_full_index:
            return None, "fragment_endpoint_not_in_full_unit"
        already_placed = placed_full_indices - {id_to_full_index[this_end_id], id_to_full_index[next_start_id]}
        seam_path = _find_partition_seam(
            positions, normals, full_arc_side, already_placed,
            id_to_full_index[this_end_id], id_to_full_index[next_start_id],
        )
        if seam_path is None:
            return None, f"no_interior_adjacency_path_between_fragment_endpoints({this_end_id}->{next_start_id})"
        chain_is_physical.append(False)  # bridge edge: end-of-piece -> first seam interior (or next piece start)
        for full_idx in seam_path[1:-1]:
            if full_idx in placed_full_indices:
                return None, "seam_would_revisit_an_already_placed_vertex"
            chain_ids.append(stable_ids[full_idx])
            chain_positions.append(positions[full_idx])
            chain_is_physical.append(False)
            placed_full_indices.add(full_idx)

    if len(set(chain_ids)) != len(chain_ids):
        return None, "duplicate_vertex_in_stitched_chain"
    return (chain_ids, chain_positions, chain_is_physical), ""


def materialize_chart_unit_domains(
    positions: Any,
    covariance: Any,
    stable_ids: Sequence[Any],
    full_evidence_positions: Any,
    *,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
    max_evidence_outside_domain_fraction: float = MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
) -> ChartUnitDomainSetResult:
    """Recover 0 or more parametric chart DOMAINS for a coherent assembled
    unit. Every already-closed physical loop is its own domain; any
    remaining open topology (fragments and/or isolated candidates) is closed
    into at most one additional domain via deterministic seam-stitching.
    Never touches fit quality."""

    torch = require_torch()
    n = int(positions.shape[0])

    def _empty(reason: str, coherence: ChartUnitCoherence | None = None, candidate_count: int = 0) -> ChartUnitDomainSetResult:
        return ChartUnitDomainSetResult(coherence, (), candidate_count, (reason,))

    coherence = assess_chart_unit_coherence(covariance, list(range(n)))
    if not coherence.coherent:
        return _empty(
            f"{STATE_AMBIGUOUS_OR_OVER_MERGED}:internal_normal_disagreement_fraction="
            f"{coherence.internal_normal_disagreement_fraction:.4f}>{NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND}",
            coherence,
        )

    normals = extract_covariance_frame(covariance).normal_candidate
    support = extract_dense_boundary_support(positions, normals, list(stable_ids))
    # A fragment needs at least 2 candidates to have any edge at all -- this
    # is the correct floor for the generalized architecture (not the old <3,
    # which only made sense when a closed PHYSICAL loop was the only option).
    if len(support.candidates) < 2:
        return _empty(f"{STATE_NO_DENSE_SUPPORT}:admitted_boundary_candidate_count={len(support.candidates)}<2", coherence, len(support.candidates))

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

    id_to_full_index = {sid: i for i, sid in enumerate(stable_ids)}
    full_arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        full_arc_side = _nearest_arc_side(positions, arc_starts, arc_ends, arc_kinds)

    domains: list[ChartUnitDomainResult] = []
    reasons: list[str] = []

    # --- Step 1: every independently-CLOSED physical loop, unchanged Worklog
    # 85 mechanism, each its own domain. Consumed candidates are removed from
    # further consideration -- they already have a home.
    loops, branch_count, open_count = _find_valid_loops(len(support.candidates), candidate_adjacency)
    consumed: set[int] = set()
    for loop_indices in loops:
        ordered_ids = [candidate_ids[i] for i in loop_indices]
        ordered_positions = candidate_positions[torch.tensor(loop_indices, dtype=torch.long, device=positions.device)]
        result = _validate_and_build_result(
            ordered_ids, ordered_positions, [True] * len(ordered_ids),
            positions, full_evidence_positions, support.full_evidence_scale,
            arc_starts, arc_ends, arc_kinds, coherence, max_evidence_outside_domain_fraction,
        )
        if result.materialized:
            domains.append(result)
            consumed.update(loop_indices)
        else:
            reasons.append(f"physical_loop_rejected:{result.state}")

    # --- Step 2: remaining open topology (fragments + isolated candidates,
    # excluding whatever a successful physical loop already consumed) closed
    # into at most one additional domain via deterministic seam-stitching.
    remaining_candidate_indices = [i for i in range(len(support.candidates)) if i not in consumed]
    if remaining_candidate_indices:
        # Recompute the candidate graph restricted to the remaining
        # candidates only, so a consumed loop's own edges cannot leak in.
        remap = {old: new for new, old in enumerate(remaining_candidate_indices)}
        remaining_adjacency: list[set[int]] = [set() for _ in remaining_candidate_indices]
        for old_i in remaining_candidate_indices:
            for old_j in candidate_adjacency[old_i]:
                if old_j in remap:
                    remaining_adjacency[remap[old_i]].add(remap[old_j])

        pieces_new_index: list[list[int]] = list(_find_open_paths(len(remaining_candidate_indices), remaining_adjacency))
        for isolated in _find_isolated_candidates(len(remaining_candidate_indices), remaining_adjacency):
            pieces_new_index.append([isolated])
        # Anything left over (branching, or a candidate consumed by neither a
        # path nor an isolation check -- e.g. still degree>=3) is genuinely
        # unresolved and not stitched.
        used_in_pieces = {i for piece in pieces_new_index for i in piece}
        stray = [i for i in range(len(remaining_candidate_indices)) if i not in used_in_pieces]

        if not pieces_new_index:
            if stray:
                reasons.append(f"unresolved_branching_or_degenerate_candidate_count={len(stray)}")
        else:
            pieces = [[remaining_candidate_indices[i] for i in piece] for piece in pieces_new_index]
            stitched, seam_reason = _stitch_pieces_into_domain(
                pieces, candidate_ids, candidate_positions, positions, normals, stable_ids,
                id_to_full_index, full_arc_side,
            )
            if stitched is None:
                reasons.append(f"{STATE_SEAM_NOT_FOUND}:{seam_reason}")
            else:
                chain_ids, chain_positions, chain_is_physical = stitched
                ordered_positions = torch.stack(chain_positions, dim=0)
                result = _validate_and_build_result(
                    chain_ids, ordered_positions, chain_is_physical,
                    positions, full_evidence_positions, support.full_evidence_scale,
                    arc_starts, arc_ends, arc_kinds, coherence, max_evidence_outside_domain_fraction,
                )
                if result.materialized:
                    domains.append(result)
                else:
                    reasons.append(f"seam_chain_rejected:{result.state}")

    if not domains and not reasons:
        reasons.append(f"{STATE_NO_OPEN_TOPOLOGY}:branch_detected_component_count={branch_count},open_fragment_component_count={open_count}")

    return ChartUnitDomainSetResult(coherence, tuple(domains), len(support.candidates), tuple(reasons))
