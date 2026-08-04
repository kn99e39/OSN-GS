from __future__ import annotations

"""Directed boundary ordering: deterministic one-in/one-out cycle recovery
over the local directed compatibility graph (worklog 35).

Worklog 33/34 established that candidate scarcity is NOT the failure mode
here -- a well-formed post-ADC region (box_face, 19 genuine candidates, same-
region partner median 18) still fragmented into 6 open chains under the
previous mutual-agreement + greedy-augmentation heuristic. Worklog 35 traced
the exact cause: 13/19 nodes have more than one directionally-compatible
successor, and greedy per-node score maximization is only LOCALLY optimal --
two neighboring nodes can each prefer a THIRD node's slot, so mutual
agreement (forward pick == backward pick) held for only 11/19 pairs, and the
remaining 8 were swept up by score-ordered greedy augmentation with no
mechanism to keep them on a single ring, splitting one 19-node loop into 6
fragments.

This module replaces the heuristic with an exact maximum-weight one-in/one-
out MATCHING (Hungarian algorithm) restricted to the directed compatibility
edges of one region at a time (never a global cross-region graph, never an
unbounded all-pairs graph -- region size is already bounded by the upstream
representative cap). A perfect one-in/one-out matching under these edges
decomposes uniquely into disjoint directed simple cycles and open paths --
exactly the "closed loop OR left honestly open" contract required.  Nodes
with zero compatible edges are never forced into the matching (no
interpolation of missing evidence); a node matched at most once as source and
once as target follows directly from bipartite matching, so in-degree<=1 and
out-degree<=1 for every admitted node are structural guarantees, not
heuristics.
"""

from dataclasses import dataclass
from typing import Sequence

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate

# Hard cap on the number of genuine candidates handled by exact bipartite
# matching within a single region-component in one ordering pass. Real
# long-horizon snapshots (worklog 30-34) have region member counts in the
# tens (largest observed positive control: 32); this is O(n^3) and measured
# at ~0.4s for n=100, ~3.2s for n=200 (each call runs the matching TWICE --
# forward and reversed-tangent candidates). This exists only as a defensive
# ceiling against a pathological future region, not a tuned performance
# knob -- exceeding it falls back to a deterministic greedy heuristic for
# that one region rather than paying unbounded assignment cost.
_EXACT_MATCHING_MAX_CANDIDATES_PER_REGION = 150

_NEGATIVE_INFINITY = float("-inf")


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
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _norm(a): return max(sum(x * x for x in a) ** .5, 1e-12)


def _build_accepted_adjacency(accepted_topology: Sequence[tuple[object, object]]) -> dict:
    """Undirected adjacency over the region's accepted-topology edges, used
    as a bounded support certificate for boundary adjacency (worklog 39)."""
    adjacency: dict = {}
    for left, right in accepted_topology:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _has_region_topology_support(
    source_id, target_id, accepted_pairs: set, accepted_adjacency: dict | None,
    boundary_candidate_ids: frozenset | None = None,
) -> bool:
    """Direct accepted edge, or a bounded TWO-hop path through the accepted
    region graph via a shared neighbour that is ITSELF NOT a boundary
    candidate.

    Two hops only -- enough to bridge a single bounded-kNN dropout between
    perimeter-consecutive candidates (measured: all 45 such box pairs are
    2-hop reachable) -- while still requiring the region's own topology to
    vouch for the pair. Never crosses regions: `accepted_pairs` and
    `accepted_adjacency` are built from region-internal accepted edges only.

    The shared neighbour must be a NON-candidate interior node. Worklog 39
    measured the failure this prevents: with any shared neighbour allowed, a
    Y-junction stub sitting INSIDE a ring (radius 0.6 against the ring's 1.0)
    with a single accepted edge to ring node 0 becomes "adjacent" to ring
    node 1 via that node, and the matching splices it into the loop -- a
    13-node cycle through a non-perimeter node, i.e. a fabricated physical
    boundary. Two candidates that are genuinely consecutive on a perimeter
    are separated by interior surface, so their bridging evidence is an
    interior node; a branch stub's only route to the ring is through another
    CANDIDATE, which this rejects.
    """
    if frozenset((source_id, target_id)) in accepted_pairs:
        return True
    if not accepted_adjacency:
        return False
    source_neighbors = accepted_adjacency.get(source_id)
    target_neighbors = accepted_adjacency.get(target_id)
    if not source_neighbors or not target_neighbors:
        return False
    shared = source_neighbors & target_neighbors
    if not shared:
        return False
    if boundary_candidate_ids is None:
        return True
    return bool(shared - boundary_candidate_ids)

# Worklog 36 (task section 6): a score margin below this fraction of the
# winning score is treated as "genuinely ambiguous" -- the matching had to
# pick ONE of two (or more) similarly-plausible successors/predecessors,
# which is the signature of an unresolved branch rather than a clearly
# dominant physical continuation. Not a canonical-final constant; a
# conservative diagnostic threshold, analogous in spirit to the existing
# `near_threshold_margin_ratio` pattern elsewhere in this codebase.
_BRANCH_AMBIGUOUS_SCORE_MARGIN_RATIO = 0.05


def _diagnose_branch_ambiguity(
    region_candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    edges: dict[tuple[str, str], DirectedBoundarySuccessor],
    matched: dict[str, str],
) -> set[str]:
    """Pre-admission non-manifold diagnostic (worklog 36 section 6).

    A one-in/one-out MATCHING only constrains the SELECTED result's degree --
    it does not erase the fact that the underlying compatibility graph may
    have offered a node 3+ directionally-compatible partners in either
    direction (undirected compatibility degree > 2). Two situations produce
    this, and only one is a genuine ambiguity worth flagging:

      1. Normal sample density: node has >2 compatible candidates, but one
         forward/backward choice clearly dominates by score margin -- the
         matching's choice is well-supported, not really "branching".
      2. Genuine Y-junction / non-manifold boundary: multiple candidates are
         near-tied in score -- the matching had to arbitrarily break a tie
         that real geometry does not actually resolve.

    A node is flagged as an unresolved branch only when its RAW compatibility
    degree (in either direction, over ALL region-local candidates, not just
    the matched pair) exceeds 2 AND the matched edge's score does not clearly
    dominate the next-best alternative in the same direction. This does NOT
    reject purely because degree>2 (dense sampling is normal), and does NOT
    assume the matching already resolved ambiguity just because it selected
    one edge (a forced tie-break is still an unresolved ambiguity).
    """
    forward_options: dict[str, list[float]] = {}
    backward_options: dict[str, list[float]] = {}
    for (source_id, target_id), edge in edges.items():
        forward_options.setdefault(source_id, []).append(edge.score)
        backward_options.setdefault(target_id, []).append(edge.score)

    branch_nodes: set[str] = set()
    for candidate in region_candidates:
        node_id = candidate.half_edge_id
        undirected_degree = len(forward_options.get(node_id, [])) + len(backward_options.get(node_id, []))
        if undirected_degree <= 2:
            continue
        matched_out = matched.get(node_id)
        if matched_out is not None:
            matched_score = edges[(node_id, matched_out)].score
            other_scores = sorted((edges[(node_id, t)].score for t in [t for (s, t) in edges if s == node_id and t != matched_out]), reverse=True)
            if other_scores:
                runner_up = other_scores[0]
                margin = matched_score - runner_up
                if margin < _BRANCH_AMBIGUOUS_SCORE_MARGIN_RATIO * max(abs(matched_score), 1e-6):
                    branch_nodes.add(node_id)
        matched_in = next((s for s, t in matched.items() if t == node_id), None)
        if matched_in is not None:
            matched_score = edges[(matched_in, node_id)].score
            other_scores = sorted((edges[(s, node_id)].score for (s, t) in edges if t == node_id and s != matched_in), reverse=True)
            if other_scores:
                runner_up = other_scores[0]
                margin = matched_score - runner_up
                if margin < _BRANCH_AMBIGUOUS_SCORE_MARGIN_RATIO * max(abs(matched_score), 1e-6):
                    branch_nodes.add(node_id)
    return branch_nodes


def _compatible_directed_edges(
    region_candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    accepted_pairs: set,
    local_spacing: float,
    accepted_adjacency: dict | None = None,
    boundary_candidate_ids: frozenset | None = None,
) -> dict[tuple[str, str], DirectedBoundarySuccessor]:
    """All directionally-compatible (source, target) edges within one region
    -- the full compatibility graph, NOT one selection per source. Identical
    geometric gates to the previous implementation (forward>0,
    distance<=1.6*local_spacing, lateral<=0.9*local_spacing,
    tangent>=-0.15, normal>=0.45); only the downstream selection differs.

    Worklog 39 (task section 11/12, Case D): the topology gate used to
    require a DIRECT ``accepted_core_pair``. That set is
    ``internal_accepted_edge_ids`` -- REGION-topology evidence built from the
    bounded-kNN affinity graph -- so it answers "are these two Gaussians
    linked in the region's own connectivity", not "are these two boundary
    candidates consecutive along the perimeter". Two candidates genuinely
    adjacent on the physical perimeter routinely have no direct affinity edge
    purely because bounded-k dropped it.

    Measured on the box fixture: every face loses 5-11 of its
    perimeter-adjacent pairs to this gate, leaving 10-17 compatible edges
    where an N-node ring needs N, so no face can close. All 45 such rejected
    pairs across the six faces are reachable by a 2-hop path in the region's
    accepted graph (0 needed 3 hops, 0 had no path at all) -- they are
    same-surface by the region's own evidence, just not directly linked. By
    contrast box_face, which does close, loses zero adjacent pairs here.

    The gate is therefore widened from "direct accepted edge" to "direct
    accepted edge OR a bounded 2-hop path through the region's accepted
    graph". Accepted topology is still required as a support certificate --
    this never admits a pair with no region-graph support, never crosses
    regions, and never bypasses any geometric gate below.
    """
    max_distance, max_lateral = local_spacing * 1.6, local_spacing * 0.9
    edges: dict[tuple[str, str], DirectedBoundarySuccessor] = {}
    for source in region_candidates:
        for target in region_candidates:
            if target.half_edge_id == source.half_edge_id:
                continue
            if not _has_region_topology_support(
                source.source_gaussian_id, target.source_gaussian_id,
                accepted_pairs, accepted_adjacency, boundary_candidate_ids,
            ):
                continue
            delta = _sub(target.world_position, source.world_position)
            distance = _norm(delta)
            forward = _dot(delta, source.boundary_direction)
            if forward <= 1e-8 or distance > max_distance:
                continue
            lateral = (max(distance * distance - forward * forward, 0.0)) ** .5
            raw_tangent_alignment = _dot(source.boundary_direction, target.boundary_direction)
            # The source tangent chooses directed traversal through the
            # forward/lateral gates above. The target tangent is a local
            # boundary line orientation, so its sign can flip at corners or
            # under equivalent local frame transport without changing the
            # physical successor relation.
            tan_align = abs(raw_tangent_alignment)
            normal_align = abs(_dot(source.local_normal, target.local_normal))
            outward_align = normal_align * tan_align
            if lateral > max_lateral or normal_align < .45:
                continue
            score = forward / distance + tan_align + normal_align + outward_align - lateral / max_lateral
            edges[(source.half_edge_id, target.half_edge_id)] = DirectedBoundarySuccessor(
                source.half_edge_id, target.half_edge_id, forward, lateral, distance / max_distance,
                tan_align, normal_align, outward_align, score, "compatible_directed_edge",
            )
    return edges


def _solve_one_in_one_out_assignment(
    node_ids: Sequence[str],
    edges: dict[tuple[str, str], DirectedBoundarySuccessor],
    forbidden: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, str]:
    """Exact maximum-total-score one-in/one-out assignment via the Hungarian
    algorithm (Jonker-Volgenant style, O(n^3)), restricted to feasible edges
    only, minus any pair explicitly listed in ``forbidden`` (worklog 53 --
    used to exclude one direction of a mutual pair once a 2-cycle is
    detected; never used to add an edge that failed compatibility). A "no
    edge" pair is infeasible (never matched), not a large finite cost -- so a
    node with zero compatible successors is simply left unmatched rather
    than forced onto a spurious best-available slot.
    """
    n = len(node_ids)
    if n == 0:
        return {}
    index_of = {node_id: i for i, node_id in enumerate(node_ids)}
    # cost[i][j] = -score if (i -> j) feasible else +inf (infeasible, never chosen)
    inf = float("inf")
    cost = [[inf] * n for _ in range(n)]
    for (source_id, target_id), edge in edges.items():
        if (source_id, target_id) in forbidden:
            continue
        i, j = index_of[source_id], index_of[target_id]
        cost[i][j] = -edge.score

    # Pad with dummy "unmatched" slots so nodes with no feasible outgoing
    # edge, or nodes that lose out in the optimal assignment, can be left
    # unmatched at zero cost instead of forcing an infeasible pairing.
    size = 2 * n
    padded = [[0.0] * size for _ in range(size)]
    big = 10.0 * (max((-c for row in cost for c in row if c != inf), default=1.0) + 1.0)
    for i in range(n):
        for j in range(n):
            padded[i][j] = cost[i][j] if cost[i][j] != inf else big
        for j in range(n, size):
            padded[i][j] = 0.0  # source i left unmatched
    for i in range(n, size):
        for j in range(size):
            padded[i][j] = 0.0  # dummy source, any target free

    assignment = _hungarian_min_cost(padded)
    matched: dict[str, str] = {}
    for i in range(n):
        j = assignment[i]
        if j < n and cost[i][j] != inf:
            matched[node_ids[i]] = node_ids[j]
    return matched


def _find_two_cycle(matched: dict[str, str]) -> tuple[str, str] | None:
    """Deterministic (lexicographically smallest) mutual pair ``a->b``,
    ``b->a`` both present in ``matched``, or ``None``."""
    for source_id in sorted(matched):
        target_id = matched[source_id]
        if matched.get(target_id) == source_id and source_id < target_id:
            return source_id, target_id
    return None


# Worklog 53: bounded defensive ceiling on 2-cycle-elimination branching, the
# same "explicit fail-closed beyond a bound" philosophy as
# `_EXACT_MATCHING_MAX_CANDIDATES_PER_REGION`. Every branch forbids exactly
# one more (source, target) pair than its parent, so this bounds total
# Hungarian re-solves, not candidate count; 2-cycles are rare (a mutual pair
# of near-identical-quality compatible edges), and each branch step strictly
# shrinks the feasible edge set, so real regions terminate in 1-2 steps.
_MAX_TWO_CYCLE_BRANCH_EXPANSIONS = 16


def _matching_score(matched: dict[str, str], edges: dict[tuple[str, str], DirectedBoundarySuccessor]) -> float:
    return sum(edges[(source_id, target_id)].score for source_id, target_id in matched.items())


def _closed_cycle_count(node_ids: Sequence[str], matched: dict[str, str]) -> int:
    cycles, _paths, _isolated = _decompose_into_paths_and_cycles(matched, node_ids)
    return len(cycles)


def _cycles_are_safe(
    node_ids: Sequence[str],
    matched: dict[str, str],
    world_position_by_id: dict[str, tuple[float, float, float]] | None,
) -> bool:
    """Worklog 53: every closed (length >= 3) cycle in ``matched`` must pass
    the same simple-closed-loop check materialization already applies
    downstream (``validate_simple_closed_loop``). ``world_position_by_id is
    None`` means the caller did not provide geometry (kept optional so this
    module's other callers/tests are unaffected) -- treated as "unknown,
    assume safe" rather than blocking the 2-cycle exchange outright.
    """
    if world_position_by_id is None:
        return True
    cycles, _paths, _isolated = _decompose_into_paths_and_cycles(matched, node_ids)
    for cycle in cycles:
        points = [world_position_by_id[node_id] for node_id in cycle]
        if not validate_simple_closed_loop(points).is_simple_polygon:
            return False
    return True


def _matching_forbidding_two_cycles(
    node_ids: Sequence[str],
    edges: dict[tuple[str, str], DirectedBoundarySuccessor],
    forbidden: frozenset[tuple[str, str]],
    budget: list[int],
    world_position_by_id: dict[str, tuple[float, float, float]] | None,
    baseline: dict[str, str],
    max_branch_expansions: int,
) -> tuple[dict[str, str], bool]:
    """Returns ``(matched, budget_exhausted)``. ``budget_exhausted`` is True
    only when the branch-expansion ceiling was hit WHILE a 2-cycle was still
    unresolved in the returned assignment (worklog 54) -- i.e. the safety
    exploration could not be completed, not merely that some node happened to
    have zero compatible edges.
    """
    matched = _solve_one_in_one_out_assignment(node_ids, edges, forbidden)
    two_cycle = _find_two_cycle(matched)
    if two_cycle is None:
        return matched, False
    if budget[0] >= max_branch_expansions:
        return matched, True
    source_id, target_id = two_cycle
    budget[0] += 1
    option_forward_forbidden, exhausted_forward = _matching_forbidding_two_cycles(
        node_ids, edges, forbidden | {(source_id, target_id)}, budget, world_position_by_id, baseline, max_branch_expansions,
    )
    option_backward_forbidden, exhausted_backward = _matching_forbidding_two_cycles(
        node_ids, edges, forbidden | {(target_id, source_id)}, budget, world_position_by_id, baseline, max_branch_expansions,
    )
    # Worklog 53 (safety first): excluding a 2-cycle frees capacity that the
    # Hungarian solver can then spend absorbing OTHER nodes into a larger
    # cycle -- measured on the thin_slab fixture, this reshaped a previously
    # safe closed loop into a self-intersecting one (a proper regression,
    # not a defect fix). Only prefer a 2-cycle-free branch when its own
    # closed cycles are all simple polygons; if neither exchange direction
    # is safe, keep the ORIGINAL (2-cycle-tolerant) baseline -- it wastes
    # two candidates on a downstream-discarded pair, but that is strictly
    # better than trading it for a self-intersecting "closed" loop
    # materialization would reject anyway.
    safe_forward = _cycles_are_safe(node_ids, option_forward_forbidden, world_position_by_id)
    safe_backward = _cycles_are_safe(node_ids, option_backward_forbidden, world_position_by_id)
    if safe_forward != safe_backward:
        return (option_forward_forbidden, exhausted_forward) if safe_forward else (option_backward_forbidden, exhausted_backward)
    if not safe_forward and not safe_backward:
        return baseline, False
    score_forward = _matching_score(option_forward_forbidden, edges)
    score_backward = _matching_score(option_backward_forbidden, edges)
    if score_forward != score_backward:
        return (option_forward_forbidden, exhausted_forward) if score_forward > score_backward else (option_backward_forbidden, exhausted_backward)
    # Worklog 53: on a genuine score tie, deterministically prefer whichever
    # branch closes strictly more valid (length >= 3) cycles -- reusing the
    # same principle worklog 52 audited and confirmed as a safe, evidence-
    # neutral tie-break (never overrides an actual score difference, only
    # breaks a tie between equally-scored feasible options).
    closed_forward = _closed_cycle_count(node_ids, option_forward_forbidden)
    closed_backward = _closed_cycle_count(node_ids, option_backward_forbidden)
    if closed_forward != closed_backward:
        return (option_forward_forbidden, exhausted_forward) if closed_forward > closed_backward else (option_backward_forbidden, exhausted_backward)
    return option_forward_forbidden, exhausted_forward


def _max_weight_one_in_one_out_matching_with_diagnostics(
    node_ids: Sequence[str],
    edges: dict[tuple[str, str], DirectedBoundarySuccessor],
    world_position_by_id: dict[str, tuple[float, float, float]] | None = None,
    *,
    max_branch_expansions: int | None = None,
) -> tuple[dict[str, str], bool]:
    """Worklog 54: same result as :func:`_max_weight_one_in_one_out_matching`,
    plus whether the 2-cycle branch-elimination budget was exhausted while a
    2-cycle remained unresolved. ``max_branch_expansions`` is overridable
    (production leaves it ``None``, which reads the module ceiling FRESH on
    every call -- not bound at import time -- so tests can patch
    ``_MAX_TWO_CYCLE_BRANCH_EXPANSIONS`` and force exhaustion deterministically
    without waiting for a pathological region).
    """
    if max_branch_expansions is None:
        max_branch_expansions = _MAX_TWO_CYCLE_BRANCH_EXPANSIONS
    baseline = _solve_one_in_one_out_assignment(node_ids, edges, frozenset())
    return _matching_forbidding_two_cycles(node_ids, edges, frozenset(), [0], world_position_by_id, baseline, max_branch_expansions)


def _max_weight_one_in_one_out_matching(
    node_ids: Sequence[str],
    edges: dict[tuple[str, str], DirectedBoundarySuccessor],
    world_position_by_id: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, str]:
    """Exact maximum-total-score one-in/one-out matching via the Hungarian
    algorithm, restricted to feasible edges only, with 2-cycles excluded from
    the feasible result (worklog 53).

    This is the deterministic replacement for greedy/mutual-agreement
    selection: the SAME set of feasible edges as before, but the globally
    optimal one-in/one-out assignment over them instead of a per-node local
    optimum followed by ad hoc greedy patch-up.

    Worklog 36 (task section 5) evaluated a cardinality-first lexicographic
    objective (B: maximize matched-node count first, score second) as a
    candidate fix for box_face(cap=27)'s one stranded node (node 21, 3
    in/3 out compatible edges, unmatched under max-score-only). Objective B
    DID recover that node (15-node loop -> 16-node loop, isolated count 1->0)
    -- but on the cylinder positive control, it REGRESSED the side wall from
    1 clean 88-candidate closed loop to 3 fragmented `ambiguous_ordering`
    pieces (10/18/26 nodes) plus 2 stranded isolated nodes: forcing marginal
    low-score nodes into the matching at a larger candidate count broke an
    otherwise-coherent large cycle's internal consistency. This is exactly
    the "improves one fixture, breaks another" failure mode the task
    explicitly warned against -- max-score-only (this function's actual
    behavior) was kept as the production objective specifically because it
    is the only one of the two tested that does not regress an existing,
    previously-correct positive control. See worklog 36 for the full
    A/B comparison; the box_face residual (node 21, and the 3-node open
    path) is disclosed as unresolved, not silently accepted.

    Worklog 53: `_decompose_into_paths_and_cycles` has always required
    length >= 3 for a valid closed loop (a 2-node mutual pair is downstream-
    useless, converted to an "ambiguous_ordering" path). Plain max-score
    assignment does not know this -- a mutual pair (source->target AND
    target->source both compatible and both strong) can win BOTH directions
    simultaneously, consuming both nodes' entire in/out capacity for a
    component the decomposition step immediately discards. Measured on real
    3k region 56: the unconstrained optimum selects exactly such a mutual
    pair (1039800<->819956, combined score 5.89) plus an unrelated 2-node
    path (278207->1110285, 2.05) -- raw sum 7.94, but its DOWNSTREAM value is
    zero closed loops from four real candidates. Excluding 2-cycles from the
    feasible result (by construction, not by a closure bonus -- every edge
    used must still pass the same compatibility gate as before) yields the
    true best valid alternative: one single 4-node open path using all four
    candidates, score 7.224 -- still open (this region's candidates do not
    admit a higher-scoring closed alternative; forcing the closed 3-cycle
    over this valid, still-higher-scoring path would itself be an
    unauthorized closure bonus), but no longer wasting two candidates on a
    structurally meaningless pair.

    Worklog 53 (safety): a 2-cycle-free exchange is only taken when its own
    closed cycles are simple polygons (``world_position_by_id``, when
    supplied); otherwise the original 2-cycle-tolerant assignment is kept
    rather than trading a harmlessly-discarded pair for a self-intersecting
    "closed" loop materialization would reject anyway.
    """
    matched, _exhausted = _max_weight_one_in_one_out_matching_with_diagnostics(node_ids, edges, world_position_by_id)
    return matched


def _hungarian_min_cost(cost: list[list[float]]) -> list[int]:
    """Standard O(n^3) Hungarian algorithm (Jonker-Volgenant potentials
    formulation) for a square cost matrix. Returns assignment[i] = j."""
    n = len(cost)
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = row assigned to column j (1-indexed columns)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            result[p[j] - 1] = j - 1
    return result


def _decompose_into_paths_and_cycles(
    matched: dict[str, str], all_node_ids: Sequence[str] = (),
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Decompose a one-in/one-out partial function into disjoint simple
    cycles, open paths, and fully-isolated nodes. Every node has out-degree<=1
    (from `matched`) and in-degree<=1 (a matching target is used by at most
    one source, enforced by the assignment being a bipartite matching) -- so
    this decomposition is always well-defined and never revisits an edge.

    Worklog 36: a node that is NEITHER a matched source NOR a matched target
    (zero compatible edges survived the matching in either direction) was
    previously dropped from BOTH the cycles and paths output entirely --
    silently missing from 100% candidate accounting (box_face cap=27: 19
    genuine candidates, 15+3=18 reported, 1 silently unaccounted). `all_node_ids`
    lets the caller recover these as explicit isolated singleton "paths" so
    every input node receives exactly one final state.
    """
    in_degree_source = set(matched.keys())
    targets = set(matched.values())
    cycles: list[list[str]] = []
    paths: list[list[str]] = []
    isolated: list[str] = []

    # Cycles: start from any node that is both matched-from and reachable
    # back to itself; standard union-find-free walk with visited tracking.
    visited_globally: set[str] = set()
    for start in sorted(in_degree_source):
        if start in visited_globally:
            continue
        chain = [start]
        seen_in_chain = {start}
        current = start
        closed = False
        while current in matched:
            nxt = matched[current]
            if nxt == start:
                closed = True
                break
            if nxt in seen_in_chain:
                break  # defensive: cannot happen for a valid matching, but never loop forever
            chain.append(nxt)
            seen_in_chain.add(nxt)
            current = nxt
        for node in chain:
            visited_globally.add(node)
        if closed and len(chain) >= 3:
            cycles.append(chain)
        elif closed:
            # 1- or 2-node closed loop is not a valid boundary cycle; leave as an
            # (unclosed) short path so it surfaces as ambiguous, never silently dropped.
            paths.append(chain)
        else:
            # Only record as a standalone open path if it did not start
            # mid-chain (i.e. no other node maps onto `start`) -- otherwise
            # it will already be captured when walked from its true head.
            if start not in targets:
                paths.append(chain)

    accounted = set()
    for chain in cycles:
        accounted.update(chain)
    for chain in paths:
        accounted.update(chain)
    # Any input node that is neither a matched source nor a matched target
    # (compatibility_degree_zero in either direction after matching) is
    # explicitly recorded, not dropped.
    for node_id in sorted(all_node_ids):
        if node_id not in accounted and node_id not in in_degree_source and node_id not in targets:
            isolated.append(node_id)
            accounted.add(node_id)

    return cycles, paths, isolated


def _recover_directed_boundary_components(
    candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    accepted_topology: Sequence[tuple[object, object]] = (),
) -> tuple[tuple[DirectedBoundarySuccessor, ...], tuple[OrderedBoundaryComponent, ...]]:
    """Deterministic one-in/one-out cycle recovery per region (worklog 35)."""
    candidates = tuple(item for item in candidates if item.boundary_reason == "observed_support_termination")
    by_id = {item.half_edge_id: item for item in candidates}
    accepted_pairs = {frozenset(pair) for pair in accepted_topology}
    accepted_adjacency = _build_accepted_adjacency(accepted_topology)

    by_region: dict[int, list[WorldSpaceBoundaryHalfEdgeCandidate]] = {}
    for item in candidates:
        by_region.setdefault(item.source_region_id, []).append(item)

    all_edges: list[DirectedBoundarySuccessor] = []
    output: list[OrderedBoundaryComponent] = []

    for region_id in sorted(by_region):
        region_candidates = by_region[region_id]
        nearest = [
            _norm(_sub(source.world_position, target.world_position))
            for source in region_candidates for target in region_candidates
            if target.half_edge_id != source.half_edge_id
        ]
        local_spacing = sorted(nearest)[len(nearest) // 2] if nearest else 1.0
        edges = _compatible_directed_edges(
            region_candidates, accepted_pairs, local_spacing, accepted_adjacency,
            frozenset(item.source_gaussian_id for item in region_candidates),
        )
        all_edges.extend(edges.values())

        node_ids = sorted(item.half_edge_id for item in region_candidates)
        if len(node_ids) > _EXACT_MATCHING_MAX_CANDIDATES_PER_REGION:
            # Worklog 36 (task section 7): worklog 35 silently fell back to a
            # deterministic greedy heuristic above this cap -- a SILENT
            # correctness-contract change keyed purely on candidate count
            # (exact one-in/one-out optimality below the cap, ad hoc local
            # greedy above it), with no state marker distinguishing the two.
            # No currently-observed real or synthetic scene reaches this cap
            # (largest observed: 88, cylinder side, well within bounds), but
            # relying on that as a standing assumption is exactly what this
            # task section forbids. Replaced with explicit fail-closed
            # (Case C): every candidate in an oversized region gets an
            # honest `ordering_capacity_exceeded` state, never a
            # same-confidence approximate ordering silently presented as
            # equivalent to the exact solver's result.
            for candidate in region_candidates:
                output.append(OrderedBoundaryComponent(
                    f"region:{region_id}:directed:{candidate.half_edge_id}", region_id, (candidate.half_edge_id,),
                    (candidate.source_gaussian_id,), "ordering_capacity_exceeded", False, (),
                    {"observed_support_termination": 1}, 0.0, "unresolved_boundary_role",
                    "reliable_core_only", False, ("region_candidate_count_exceeds_exact_matching_capacity",),
                ))
            continue

        world_position_by_id = {item.half_edge_id: item.world_position for item in region_candidates}
        matched, budget_exhausted = _max_weight_one_in_one_out_matching_with_diagnostics(node_ids, edges, world_position_by_id)
        cycles, paths, isolated = _decompose_into_paths_and_cycles(matched, node_ids)
        for chain in cycles:
            source = by_id[chain[0]]
            # Worklog 54: if the 2-cycle branch-elimination budget was
            # exhausted while resolving this region, the safety exploration
            # (worklog 53) could not run to completion -- a returned
            # `ordered_closed_loop` here has not been fully vetted against
            # the alternative branches that would have been tried. Fail
            # closed by tagging it rather than trusting it silently; the
            # region-status layer (worklog 54) routes any tagged component to
            # `rejected_unsafe` regardless of `ordering_state`.
            output.append(OrderedBoundaryComponent(
                f"region:{region_id}:directed:{min(chain)}", region_id, tuple(chain),
                tuple(by_id[x].source_gaussian_id for x in chain), "ordered_closed_loop", True, (),
                {"observed_support_termination": len(chain)}, .7, "outer_boundary_candidate",
                "reliable_core_only", False, ("two_cycle_branch_budget_exhausted",) if budget_exhausted else (),
            ))
        for chain in paths:
            if len(chain) < 1:
                continue
            source = by_id[chain[0]]
            reasons = ("one_in_one_out_cycle_not_closed",)
            if budget_exhausted:
                reasons = reasons + ("two_cycle_branch_budget_exhausted",)
            output.append(OrderedBoundaryComponent(
                f"region:{region_id}:directed:{min(chain)}", region_id, tuple(chain),
                tuple(by_id[x].source_gaussian_id for x in chain), "ambiguous_ordering", False, (),
                {"observed_support_termination": len(chain)}, .7, "unresolved_boundary_role",
                "reliable_core_only", False, reasons,
            ))
        for node_id in isolated:
            # Worklog 36: explicit final state for a candidate whose
            # compatibility degree collapsed to zero in BOTH directions after
            # matching (either it had no feasible edge at all, or it lost
            # every edge to a higher-scoring competitor) -- never silently
            # dropped from accounting.
            output.append(OrderedBoundaryComponent(
                f"region:{region_id}:directed:{node_id}", region_id, (node_id,),
                (by_id[node_id].source_gaussian_id,), "isolated_boundary_candidate", False, (),
                {"observed_support_termination": 1}, .7, "unresolved_boundary_role",
                "reliable_core_only", False, ("compatibility_degree_zero_after_matching",),
            ))

    return tuple(all_edges), tuple(output)


def recover_directed_boundary_components(
    candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    accepted_topology: Sequence[tuple[object, object]] = (),
) -> tuple[tuple[DirectedBoundarySuccessor, ...], tuple[OrderedBoundaryComponent, ...]]:
    """Recover ordering while treating a region-wide tangent reversal as equivalent."""
    from dataclasses import replace
    direct = _recover_directed_boundary_components(candidates, accepted_topology)
    # Covariance normals define an unoriented surface. Reversing every tangent
    # in a region is therefore the same loop with the opposite traversal.
    reversed_candidates = tuple(
        replace(item, boundary_direction=tuple(-value for value in item.boundary_direction), local_tangent_direction=tuple(-value for value in item.local_tangent_direction))
        for item in candidates
    )
    reverse = _recover_directed_boundary_components(reversed_candidates, accepted_topology)

    # Worklog 53 (safety): world position is unaffected by the tangent
    # reversal, so one lookup serves both orientations' quality checks.
    world_position_by_id = {item.half_edge_id: item.world_position for item in candidates}

    def _safe_closed_loop(component: OrderedBoundaryComponent) -> bool:
        if component.ordering_state != "ordered_closed_loop":
            return False
        points = [world_position_by_id[half_edge_id] for half_edge_id in component.ordered_half_edge_ids]
        return validate_simple_closed_loop(points).is_simple_polygon

    def quality(result):
        # A self-intersecting "closed loop" is never a real win: it fails
        # materialization's own self-intersection check downstream anyway,
        # so counting it here can make an ORIENTATION with an unrelated
        # self-intersecting component look artificially better than one
        # without it -- measured on the thin_slab fixture, where fixing a
        # genuine 2-cycle in one region of the reversed-tangent candidate
        # set raised that orientation's closed-loop count just enough to win
        # the comparison, silently dragging in an unrelated, pre-existing
        # self-intersecting loop from a different region of that same
        # orientation. Only a component that is BOTH `ordered_closed_loop`
        # AND passes the same downstream simple-polygon check counts here.
        edges, components = result
        safe_closed = sum(_safe_closed_loop(component) for component in components)
        return (safe_closed, -sum(component.ordering_state != "ordered_closed_loop" for component in components), len(edges))

    return reverse if quality(reverse) > quality(direct) else direct
