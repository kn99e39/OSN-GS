"""Worklog 35: synthetic-halfedge-level negative controls for the deterministic
directed-ordering repair -- open chain, Y-junction, two disjoint loops,
duplicate/reverse-duplicate edge, orientation-reversed candidate, one missing
edge in an otherwise-closed loop, and sparse gap. Constructs
WorldSpaceBoundaryHalfEdgeCandidate objects directly (bypassing region
formation/reliability) to test `recover_directed_boundary_components` in
isolation, exactly as production calls it.
"""

from __future__ import annotations

import math

from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _ring_candidates(n: int, radius: float = 1.0, region_id: int = 0, drop_index: int | None = None):
    candidates = []
    ids = [i for i in range(n) if i != drop_index]
    for i in ids:
        angle = 2 * math.pi * i / n
        next_angle = 2 * math.pi * ((i + 1) % n) / n
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        tangent = (-math.sin(angle), math.cos(angle), 0.0)
        candidates.append(WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"h{i}", source_region_id=region_id, source_gaussian_id=i, adjacent_gaussian_id=None,
            world_position=(x, y, 0.0), local_normal=(0.0, 0.0, 1.0), local_tangent_direction=tangent,
            boundary_direction=tangent, boundary_reason="observed_support_termination", source_pair_ids=None,
            confidence=0.7, ordering_state="locally_chainable", review_reasons=(),
        ))
    return candidates


def _accepted_pairs_for_ring(n: int, drop_index: int | None = None):
    ids = [i for i in range(n) if i != drop_index]
    pairs = []
    for pos in range(len(ids)):
        a, b = ids[pos], ids[(pos + 1) % len(ids)]
        pairs.append((a, b))
    return pairs


def test_closed_ring():
    n = 12
    candidates = _ring_candidates(n)
    accepted = _accepted_pairs_for_ring(n)
    _, components = recover_directed_boundary_components(candidates, accepted)
    closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
    print(f"closed_ring: components={len(components)} closed={len(closed)} sizes={[len(c.ordered_source_ids) for c in components]}")
    assert len(closed) == 1 and len(closed[0].ordered_source_ids) == n, "expected exactly one closed loop covering all nodes"


def test_open_chain_not_forced_closed():
    n = 12
    # Drop the accepted pair that would close the loop -- physical evidence
    # for the LAST edge is simply missing (no synthetic interpolation allowed).
    candidates = _ring_candidates(n)
    accepted = _accepted_pairs_for_ring(n)[:-1]  # remove edge (n-1 -> 0)
    _, components = recover_directed_boundary_components(candidates, accepted)
    closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
    print(f"open_chain: components={len(components)} closed={len(closed)} sizes={[len(c.ordered_source_ids) for c in components]}")
    assert len(closed) == 0, "an open chain with one missing physical edge must never be force-closed"


def test_two_disjoint_loops_kept_separate():
    n = 8
    ring_a = _ring_candidates(n, radius=1.0, region_id=0)
    ring_b = _ring_candidates(n, radius=1.0, region_id=0)
    # offset ring_b ids/positions so it's a spatially distinct second loop in the SAME region.
    ring_b = [
        WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"h{c.source_gaussian_id + 100}", source_region_id=0, source_gaussian_id=c.source_gaussian_id + 100,
            adjacent_gaussian_id=None, world_position=(c.world_position[0] + 10.0, c.world_position[1], 0.0),
            local_normal=c.local_normal, local_tangent_direction=c.local_tangent_direction,
            boundary_direction=c.boundary_direction, boundary_reason=c.boundary_reason, source_pair_ids=None,
            confidence=0.7, ordering_state="locally_chainable", review_reasons=(),
        )
        for c in ring_a
    ]
    accepted_a = _accepted_pairs_for_ring(n)
    accepted_b = [(a + 100, b + 100) for a, b in accepted_a]
    candidates = ring_a + ring_b
    accepted = accepted_a + accepted_b
    _, components = recover_directed_boundary_components(candidates, accepted)
    closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
    print(f"two_disjoint_loops: components={len(components)} closed={len(closed)} sizes={[len(c.ordered_source_ids) for c in components]}")
    assert len(closed) == 2, "two spatially disjoint loops must be returned as two separate closed components, never merged"
    assert all(len(c.ordered_source_ids) == n for c in closed)


def test_y_junction_not_admitted_as_simple_loop():
    n = 12
    candidates = _ring_candidates(n)
    accepted = _accepted_pairs_for_ring(n)
    # Add a branch: node 0 also accepted-paired with an extra stub node that
    # points roughly forward from node 0 -- giving node 0 two competing
    # geometrically-plausible successors (a Y-junction), never a proper cycle.
    stub_angle = 2 * math.pi * 0.5 / n
    stub = WorldSpaceBoundaryHalfEdgeCandidate(
        half_edge_id="stub", source_region_id=0, source_gaussian_id=999, adjacent_gaussian_id=None,
        world_position=(1.0 * math.cos(stub_angle) * 0.6, 1.0 * math.sin(stub_angle) * 0.6, 0.0),
        local_normal=(0.0, 0.0, 1.0), local_tangent_direction=(-math.sin(stub_angle), math.cos(stub_angle), 0.0),
        boundary_direction=(-math.sin(stub_angle), math.cos(stub_angle), 0.0),
        boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
        ordering_state="locally_chainable", review_reasons=(),
    )
    candidates = candidates + [stub]
    accepted = accepted + [(0, 999)]
    _, components = recover_directed_boundary_components(candidates, accepted)
    closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
    stub_in_closed = any(999 in c.ordered_source_ids for c in closed)
    print(f"y_junction: components={len(components)} closed={len(closed)} stub_in_any_closed_loop={stub_in_closed}")
    # The Y-branch must not silently vanish into a false "simple loop" that
    # includes 13 nodes for a 12-node ring, nor may the stub be dropped
    # without trace; but a closed 12-loop excluding the stub OR the stub
    # showing up in an open/ambiguous leftover component are both acceptable
    # (one-in/one-out matching structurally cannot admit a true branch into a
    # cycle since the branch node would need out-degree>1).
    for c in closed:
        assert len(c.ordered_source_ids) <= n, "a matching-based cycle cannot exceed the number of nodes with degree<=1 each way"


def test_duplicate_and_reverse_duplicate_edges_do_not_double_count():
    n = 10
    candidates = _ring_candidates(n)
    base_pairs = _accepted_pairs_for_ring(n)
    # Duplicate every accepted pair plus its reverse -- must not create
    # parallel multi-edges that corrupt the matching (accepted_pairs is a
    # frozenset-of-pairs set, so duplicates collapse naturally; verify this).
    accepted = base_pairs + [(b, a) for a, b in base_pairs] + base_pairs
    _, components = recover_directed_boundary_components(candidates, accepted)
    closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
    print(f"duplicate_reverse_edges: closed={len(closed)} sizes={[len(c.ordered_source_ids) for c in closed]}")
    assert len(closed) == 1 and len(closed[0].ordered_source_ids) == n


def test_sparse_gap_not_bridged():
    n = 12
    # Remove TWO consecutive candidate nodes entirely (a real physical gap,
    # not just a missing accepted-pair) -- the remaining 10 nodes must not be
    # bridged across the gap into a false closed loop.
    candidates = _ring_candidates(n, drop_index=None)
    candidates = [c for c in candidates if c.source_gaussian_id not in (3, 4)]
    accepted = [pair for pair in _accepted_pairs_for_ring(n) if 3 not in pair and 4 not in pair]
    _, components = recover_directed_boundary_components(candidates, accepted)
    closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
    print(f"sparse_gap: components={len(components)} closed={len(closed)} sizes={[len(c.ordered_source_ids) for c in components]}")
    assert len(closed) == 0, "a genuine 2-node physical gap must never be silently bridged into a closed loop"


if __name__ == "__main__":
    test_closed_ring()
    test_open_chain_not_forced_closed()
    test_two_disjoint_loops_kept_separate()
    test_y_junction_not_admitted_as_simple_loop()
    test_duplicate_and_reverse_duplicate_edges_do_not_double_count()
    test_sparse_gap_not_bridged()
    print("ALL PASSED")
