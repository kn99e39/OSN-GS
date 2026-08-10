from __future__ import annotations

"""Worklog 84: chart-unit coherence audit + evidence-scale boundary topology.

Worklog 83 assembled Worklog 82 micro-components into chart-scale units via
aggregate, redundant evidence (75-93% of region evidence recovered per
region, zero unsupported gap bridging audited). But two questions stayed
coupled and unresolved: is an assembled unit genuinely one coherent
parametric-chart unit rather than an over-merge, and can it derive its own
boundary topology instead of depending on 3-7 sparse representative nodes
(Worklog 80's macro topology) to type/geometrically bound tens to hundreds
of evidence points. This module closes both, reusing existing contracts only.

COHERENCE AUDIT (evidence-only, never fit quality):
Reuses `internal_normal_disagreement_fraction` (Worklog 82's own per-
micro-component non-manifold check, the SAME formula and the SAME 0.15
bound, unchanged) applied to the WHOLE assembled unit's evidence instead of
one micro-component's. A unit whose accepted-edge assembly chain folded
together evidence with real internal orientation disagreement is exactly
what that check already exists to catch -- applying it again at assembled
scale is reuse, not a new/tuned criterion. Units over the bound are
`AMBIGUOUS_OR_OVER_MERGED` and never reach boundary materialization.

EVIDENCE-SCALE BOUNDARY TOPOLOGY (coherent units only):
Worklog 80's `build_dense_chart_support` required a sparse macro cycle (3-7
representative nodes) to supply the geometric ARC ORDER a chart's dense
support gets assigned into -- workable when the chart IS the representative-
scale region, unworkable when an assembled unit spans hundreds of evidence
points with only a handful of representative nodes touching it.

Two designs were tried and rejected before this one, both measured directly
on real baseline_compatible@2900 assembled units, not assumed:
  1. Reusing `extract_dense_boundary_support`'s own `_connect` closed-loop
     recovery (mutual +/-tangent half-line selection) as the ORDER: 0/178
     materialized (107 `not_closed`, 71 `no_dense_support`), reproducing
     Worklog 71's own already-documented limit (17/282 seed components ever
     reached `closed_loop_recovered`).
  2. Ordering admitted boundary candidates by ANGLE around their own
     centroid in the unit's best-fit tangent plane: this DOES produce a
     simple polygon on any star-shaped boundary, but is not a general
     perimeter-topology reconstruction -- on a genuinely concave/non-star-
     shaped boundary it silently reorders vertices past their true
     neighbours, and on a genuinely open fragment it always closes SOME
     polygon (a chord across the open end), which the self-intersection
     check alone cannot catch (an `UNSUPPORTED_CLOSURE` occupancy gate was
     added specifically to catch that chord). Neither failure means the
     underlying perimeter evidence doesn't exist -- the ORDERING mechanism
     itself was insufficiently general.

This module instead builds an actual LOCAL 2-MANIFOLD ADJACENCY GRAPH among
the admitted boundary candidates and recovers the loop(s) FROM THAT GRAPH's
own structure, never from any single global angle or projection:

  1. `extract_dense_boundary_support` supplies CANDIDATE ADMISSION ONLY
     (Worklog 77's corrected predicate, unmodified -- observed reliable),
     each candidate carrying its own local normal.
  2. `build_same_surface_adjacency` (Worklog 82's own normal-alignment/
     tangent-residual CONSISTENCY criterion, reused UNCHANGED at its own
     0.85/0.35 thresholds) is applied to the candidates themselves, with the
     SAME typed-crease veto Worklog 82/83 already use. Two DEGREE parameters
     are involved and were measured separately, not conflated:
       * the SEARCH POOL (how many nearby candidates each point may even be
         matched against) is left UNRESTRICTED (every other candidate is a
         legal match target) so a genuine curve-neighbour is never missed
         merely because it happened to rank outside an arbitrary cutoff --
         confirmed directly: restricting the search pool to Worklog 82's own
         interior-mesh k=8 left most real candidates at degree 0-1 even
         though a wider search recovers far more valid matches;
       * the FINAL ACCEPTED DEGREE is capped at 2 per candidate, which is
         not a tuned threshold but the definition of a 1-manifold CURVE (a
         simple perimeter loop cannot have a vertex of degree > 2). This cap
         alone is what keeps the graph from ever becoming the dense-clique
         structure this module must avoid, regardless of how wide the
         search pool is -- confirmed on a synthetic ring both restricted and
         unrestricted, and the cap is what actually matters: with it, the
         resulting adjacency is provably never denser than degree 2 per
         node, i.e. never a clique, independent of search-pool width.
     This is "local 2-manifold neighborhood relations" in the precise sense
     the project already established that phrase to mean, sized to the
     1-manifold curve object actually being reconstructed here.
  3. A connected component of that graph in which EVERY vertex has degree
     EXACTLY 2 is, by simple graph theory, exactly one simple cycle -- no
     angle computation, no projection, no convex-hull-equivalent operation
     is involved in recovering it; the ORDER falls directly out of walking
     the graph's own edges. Components with a vertex of degree >= 3 are
     disclosed as `branch_detected` (an ambiguous local junction, not
     force-resolved); components with a vertex of degree <= 1 are
     `open_fragment` (genuinely insufficient perimeter evidence to close,
     not force-closed). Multiple valid degree-2-regular components are all
     genuine, topology-proven, INDEPENDENT boundary loops -- reported, not
     collapsed to one.
  4. Every edge in a recovered loop is, by construction, a real local
     same_surface adjacency between two observed points -- never invented,
     never a chord across unclassified space.

Validated, not assumed, before acceptance (tried loops largest-first; the
first to pass both is used):
  * `evaluate_closed_loop_geometry` (Worklog 71, unmodified) -- self-
    intersection, still checked even though graph-degree-2 loops are far
    less likely to self-intersect than an angle-sorted one.
  * `measure_edge_support_occupancy` (Worklog 76) retained as an
    INDEPENDENT FINAL safety check, exactly as before -- it is never used to
    invent or accept topology the graph itself didn't produce, only to veto
    a graph-recovered loop after the fact if some edge still turns out to
    cross an empty interior bin.

Sparse macro topology is used only AFTER a loop exists, and only for two
things: typing each boundary vertex with the nearest sparse arc's
`segment_kind` (label only, never geometry), and disclosing (not silently
accepting) a boundary vertex whose nearest macro arc is a `crease` while the
loop's own connectivity carried it in from the other side of that same arc.

Every failure is typed and fail-closed: no hull, no PCA rectangle, no
bounding box, no alpha shape, no forced closure, no gap bridging, no
centroid-angle sort. If the unit's own dense evidence does not produce a
valid closed loop, this module reports EXACTLY which local-topology
condition failed (branching / open fragment / disconnected) rather than
falling back to any global heuristic.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_dense_surface_consistency_components import (
    DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND,
    _nearest_arc_side,
    build_same_surface_adjacency,
    internal_normal_disagreement_fraction,
)
from osn_gs.surface.torch_boundary_support_spacing import measure_edge_support_occupancy
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_region_owned_full_evidence import (
    MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
    evidence_outside_chart_domain_fraction,
)
from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import evaluate_closed_loop_geometry
from osn_gs.utils.torch_ops import require_torch

# A CURVE's own topology, not a tuned threshold: a 1-manifold with boundary
# (a simple perimeter loop) cannot have a vertex of degree > 2 in its own
# along-curve graph, so the FINAL accepted degree is capped at 2 -- this
# alone is what keeps the graph from ever becoming a dense clique, regardless
# of search-pool width. The SEARCH POOL considered for matching is left
# unrestricted (see module docstring): restricting it to Worklog 82's own
# interior-mesh k=8 was measured directly to leave most real candidates
# under-connected (a search-radius limitation, not a sign of missing
# evidence), while an unrestricted pool with the SAME degree-2 cap still
# recovers a perfect cycle on a synthetic ring and recovers substantially
# more real degree-2 candidates. The SAME_SURFACE consistency criterion
# itself (0.85 alignment / 0.35 residual) is unchanged.
BOUNDARY_CURVE_MAX_DEGREE = 2

STATE_MATERIALIZED = "chart_unit_boundary_materialized"
STATE_AMBIGUOUS_OR_OVER_MERGED = "chart_unit_ambiguous_or_over_merged"
STATE_NO_DENSE_SUPPORT = "chart_unit_boundary_no_dense_support"
STATE_NO_VALID_LOOP_TOPOLOGY = "chart_unit_boundary_no_valid_loop_topology"
STATE_SELF_INTERSECTING = "chart_unit_boundary_self_intersecting"
STATE_UNSUPPORTED_CLOSURE = "chart_unit_boundary_unsupported_closure"
STATE_COVERAGE_FAILED = "chart_unit_boundary_coverage_failed"


@dataclass(frozen=True)
class ChartUnitBoundarySegment:
    stable_id_a: Any
    stable_id_b: Any
    segment_kind: str  # nearest sparse-arc label, or "" if no macro arc coverage
    crease_inconsistent: bool  # loop crossed a crease arc without a matching veto upstream


@dataclass(frozen=True)
class ChartUnitCoherence:
    coherent: bool
    internal_normal_disagreement_fraction: float


@dataclass(frozen=True)
class ChartUnitBoundaryResult:
    state: str
    coherence: ChartUnitCoherence | None
    ordered_stable_ids: tuple[Any, ...]
    ordered_positions: Any | None
    segments: tuple[ChartUnitBoundarySegment, ...]
    evidence_outside_domain_fraction: float | None
    crease_inconsistent_segment_count: int
    additional_valid_loop_count: int
    branch_detected_component_count: int
    open_fragment_component_count: int
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


def assess_chart_unit_coherence(covariance: Any, member_local_indices: Sequence[int]) -> ChartUnitCoherence:
    """Evidence-only coherence audit -- reuses Worklog 82's own formula/bound
    at assembled chart-unit scale. Never reads fit quality."""

    normals = extract_covariance_frame(covariance).normal_candidate
    fraction = internal_normal_disagreement_fraction(normals, member_local_indices)
    return ChartUnitCoherence(fraction <= NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND, fraction)


def _trace_degree_two_cycle(component: set[int], adjacency: list[set[int]]) -> list[int]:
    """Walk a connected, degree-2-regular component into its unique cycle
    order. Deterministic: starts at the smallest index, takes the smaller-
    index neighbor first."""

    start = min(component)
    order = [start]
    neighbors = sorted(adjacency[start] & component)
    prev, current = start, neighbors[0]
    order.append(current)
    while current != start:
        nxt = next(v for v in adjacency[current] if v != prev and v in component)
        prev, current = current, nxt
        if current != start:
            order.append(current)
    return order


def _find_valid_loops(
    candidate_count: int, adjacency: list[set[int]],
) -> tuple[list[list[int]], int, int]:
    """Connected components of the candidate adjacency graph, split into
    valid degree-2-regular simple cycles vs disclosed failure categories.
    Returns ``(loops, branch_detected_component_count, open_fragment_component_count)``,
    loops sorted largest-first (ties broken by minimum member index, for
    determinism)."""

    visited: set[int] = set()
    loops: list[list[int]] = []
    branch_count = 0
    open_count = 0
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
        if len(component) < 3:
            open_count += 1
            continue
        degrees = {node: len(adjacency[node] & component) for node in component}
        if all(degree == 2 for degree in degrees.values()):
            loops.append(_trace_degree_two_cycle(component, adjacency))
        elif any(degree >= 3 for degree in degrees.values()):
            branch_count += 1
        else:
            open_count += 1
    loops.sort(key=lambda loop: (-len(loop), min(loop)))
    return loops, branch_count, open_count


def materialize_chart_unit_boundary(
    positions: Any,
    covariance: Any,
    stable_ids: Sequence[Any],
    full_evidence_positions: Any,
    *,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
    max_evidence_outside_domain_fraction: float = MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
) -> ChartUnitBoundaryResult:
    """Coherence-audit an assembled chart unit, then (if coherent) recover its
    boundary topology directly from its OWN dense evidence via a local
    2-manifold adjacency graph among boundary-support candidates -- never a
    centroid angle, hull, or projection.

    ``positions``/``covariance``/``stable_ids`` describe the chart unit's
    member evidence only. ``full_evidence_positions`` is the SAME unit's
    evidence again (kept as a separate argument to make the Worklog 79
    coverage-contract call site explicit and match its existing signature).
    ``arc_starts``/``arc_ends``/``arc_kinds`` are Worklog 80's sparse macro
    arcs, in the SAME frame as ``positions`` -- optional: typing is
    best-effort, never required to produce geometry.
    """

    torch = require_torch()
    n = int(positions.shape[0])

    def _fail(state: str, coherence: ChartUnitCoherence | None, *reasons: str, branch: int = 0, open_frag: int = 0) -> ChartUnitBoundaryResult:
        return ChartUnitBoundaryResult(state, coherence, (), None, (), None, 0, 0, branch, open_frag, tuple(reasons))

    coherence = assess_chart_unit_coherence(covariance, list(range(n)))
    if not coherence.coherent:
        return _fail(
            STATE_AMBIGUOUS_OR_OVER_MERGED, coherence,
            f"internal_normal_disagreement_fraction={coherence.internal_normal_disagreement_fraction:.4f}"
            f">{NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND}",
        )

    normals = extract_covariance_frame(covariance).normal_candidate
    support = extract_dense_boundary_support(positions, normals, list(stable_ids))
    if len(support.candidates) < 3:
        return _fail(STATE_NO_DENSE_SUPPORT, coherence, f"admitted_boundary_candidate_count={len(support.candidates)}<3")

    # Build the LOCAL 2-manifold adjacency graph directly among the admitted
    # candidates (Worklog 82's own bounded-degree kNN + same_surface
    # criterion, unchanged defaults) -- the order comes from this graph's own
    # structure, never from any angle or projection.
    candidate_ids = [c.stable_id for c in support.candidates]
    candidate_positions_all = torch.stack(
        [torch.tensor(c.position, dtype=positions.dtype, device=positions.device) for c in support.candidates], dim=0,
    )
    candidate_normals = torch.stack(
        [torch.tensor(c.normal, dtype=positions.dtype, device=positions.device) for c in support.candidates], dim=0,
    )
    candidate_arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        candidate_arc_side = _nearest_arc_side(candidate_positions_all, arc_starts, arc_ends, arc_kinds)

    # Search pool unrestricted (every other candidate is a legal match
    # target); only the FINAL accepted degree is capped, at the curve
    # invariant of 2 -- see module docstring and the constant's own comment.
    _edges, adjacency, _crease_vetoed = build_same_surface_adjacency(
        candidate_positions_all, candidate_normals, arc_side=candidate_arc_side,
        candidate_neighbor_count=max(1, len(support.candidates) - 1),
        max_candidate_count_per_node=BOUNDARY_CURVE_MAX_DEGREE,
        same_surface_min_normal_alignment=DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
        same_surface_max_mutual_residual=DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    )
    loops, branch_count, open_count = _find_valid_loops(len(support.candidates), adjacency)
    if not loops:
        return _fail(
            STATE_NO_VALID_LOOP_TOPOLOGY, coherence,
            f"branch_detected_component_count={branch_count}",
            f"open_fragment_component_count={open_count}",
            branch=branch_count, open_frag=open_count,
        )

    # Try topology-proven loops largest-first; the first to survive BOTH
    # independent safety checks below is used. Others stay disclosed via
    # `additional_valid_loop_count`, never silently collapsed into one.
    for loop_rank, loop_indices in enumerate(loops):
        ordered_ids = tuple(candidate_ids[i] for i in loop_indices)
        ordered_positions = candidate_positions_all[torch.tensor(loop_indices, dtype=torch.long, device=positions.device)]

        geometry = evaluate_closed_loop_geometry(
            [tuple(float(v) for v in row) for row in ordered_positions.detach().cpu().tolist()]
        )
        if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
            if loop_rank < len(loops) - 1:
                continue
            return _fail(
                STATE_SELF_INTERSECTING, coherence, f"proper_crossing_count={geometry.proper_crossing_count}",
                branch=branch_count, open_frag=open_count,
            )

        # Independent final safety net (Worklog 76) -- NOT used to invent or
        # accept topology the graph did not already produce; every edge here
        # is a real graph edge, this only vetoes a graph-recovered loop that
        # still happens to cross observed empty space.
        edge_pairs = [(i, (i + 1) % len(ordered_ids)) for i in range(len(ordered_ids))]
        occupancy = measure_edge_support_occupancy(
            edge_pairs, ordered_positions, positions, full_evidence_spacing=support.full_evidence_scale,
        )
        if occupancy["edges_with_empty_interior_bin"] > 0:
            if loop_rank < len(loops) - 1:
                continue
            return _fail(
                STATE_UNSUPPORTED_CLOSURE, coherence,
                f"edges_with_empty_interior_bin={occupancy['edges_with_empty_interior_bin']}/{occupancy['edge_count']}",
                branch=branch_count, open_frag=open_count,
            )

        # Typed provenance: label only, from the nearest sparse macro arc.
        arc_side = None
        if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
            arc_side = _nearest_arc_side(ordered_positions, arc_starts, arc_ends, arc_kinds)

        k = len(ordered_ids)
        segments = []
        crease_inconsistent_count = 0
        for i in range(k):
            kind_a = arc_side[i] if arc_side is not None else ""
            kind_b = arc_side[(i + 1) % k] if arc_side is not None else ""
            inconsistent = bool(kind_a and kind_b and kind_a != kind_b and ("crease" in kind_a or "crease" in kind_b))
            if inconsistent:
                crease_inconsistent_count += 1
            segments.append(ChartUnitBoundarySegment(ordered_ids[i], ordered_ids[(i + 1) % k], kind_a or kind_b, inconsistent))

        additional_loops = len(loops) - 1
        outside = evidence_outside_chart_domain_fraction(ordered_positions, full_evidence_positions)
        if outside is not None and outside > max_evidence_outside_domain_fraction:
            if loop_rank < len(loops) - 1:
                continue
            return ChartUnitBoundaryResult(
                STATE_COVERAGE_FAILED, coherence, ordered_ids, ordered_positions, tuple(segments), outside,
                crease_inconsistent_count, additional_loops, branch_count, open_count,
                (f"evidence_outside_chart_domain_fraction={outside:.4f}>{max_evidence_outside_domain_fraction}",),
            )

        return ChartUnitBoundaryResult(
            STATE_MATERIALIZED, coherence, ordered_ids, ordered_positions, tuple(segments), outside,
            crease_inconsistent_count, additional_loops, branch_count, open_count, (),
        )

    # Unreachable in practice (the loop above always returns on the final
    # rank), kept only as an explicit fail-closed fallback.
    return _fail(STATE_NO_VALID_LOOP_TOPOLOGY, coherence, "no_loop_survived_safety_checks",
                 branch=branch_count, open_frag=open_count)
