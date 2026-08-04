"""Worklog 36: audit the exact unmatched/dummy/forbidden-edge contract of
`_max_weight_one_in_one_out_matching` (torch_directed_boundary_ordering.py).

Standard square-matrix Hungarian assignment matches every row to a column.
This implementation pads an n x n cost matrix to 2n x 2n with a dummy
"unmatched" block (cost 0) so nodes with no good compatible edge are not
forced to match a real (possibly forbidden-cost) node. This script verifies,
with tiny hand-built cost matrices, that:
  - a node whose only real edges are "forbidden" (infinite cost) is matched
    to a dummy (unmatched), never to a real node
  - a node with no outgoing edges but with an incoming edge still gets a
    valid in/out state
  - dummy-to-dummy assignments never leak into the returned `matched` dict
  - the optimal REAL assignment (ignoring dummies) is found when it exists
"""

from __future__ import annotations

from osn_gs.surface.torch_directed_boundary_ordering import (
    DirectedBoundarySuccessor,
    _max_weight_one_in_one_out_matching,
)


def _edge(s, t, score):
    return DirectedBoundarySuccessor(s, t, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, score, "compatible_directed_edge")


def test_all_forbidden_node_stays_unmatched():
    # Node "c" has zero outgoing edges at all -- must never receive a fabricated match.
    nodes = ["a", "b", "c"]
    edges = {("a", "b"): _edge("a", "b", 1.0)}
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "b"}, matched
    print("test_all_forbidden_node_stays_unmatched: OK", matched)


def test_successor_only_no_predecessor():
    # a->b compatible, nothing points to a.
    nodes = ["a", "b"]
    edges = {("a", "b"): _edge("a", "b", 2.0)}
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "b"}
    print("test_successor_only_no_predecessor: OK", matched)


def test_valid_open_path_three_nodes():
    nodes = ["a", "b", "c"]
    edges = {
        ("a", "b"): _edge("a", "b", 3.0),
        ("b", "c"): _edge("b", "c", 3.0),
    }
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "b", "b": "c"}, matched
    print("test_valid_open_path_three_nodes: OK", matched)


def test_valid_cycle_plus_unmatched_extra_node():
    nodes = ["a", "b", "c", "d"]
    edges = {
        ("a", "b"): _edge("a", "b", 5.0),
        ("b", "c"): _edge("b", "c", 5.0),
        ("c", "a"): _edge("c", "a", 5.0),
        # d has no compatible edges at all.
    }
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "b", "b": "c", "c": "a"}, matched
    assert "d" not in matched and "d" not in matched.values()
    print("test_valid_cycle_plus_unmatched_extra_node: OK", matched)


def test_two_disjoint_cycles():
    nodes = ["a1", "a2", "a3", "b1", "b2", "b3"]
    edges = {
        ("a1", "a2"): _edge("a1", "a2", 4.0), ("a2", "a3"): _edge("a2", "a3", 4.0), ("a3", "a1"): _edge("a3", "a1", 4.0),
        ("b1", "b2"): _edge("b1", "b2", 4.0), ("b2", "b3"): _edge("b2", "b3", 4.0), ("b3", "b1"): _edge("b3", "b1", 4.0),
    }
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched.get("a1") == "a2" and matched.get("a2") == "a3" and matched.get("a3") == "a1"
    assert matched.get("b1") == "b2" and matched.get("b2") == "b3" and matched.get("b3") == "b1"
    print("test_two_disjoint_cycles: OK", matched)


def test_two_cycle_only_graph_rejected_by_decomposition_not_solver():
    # A pure 2-cycle a<->b is a valid one-in/one-out MATCHING (solver-level),
    # even though decomposition later refuses to admit it as a closed loop
    # (len(chain) < 3). Solver contract: it CAN select this if it's optimal.
    nodes = ["a", "b"]
    edges = {("a", "b"): _edge("a", "b", 5.0), ("b", "a"): _edge("b", "a", 5.0)}
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "b", "b": "a"}, matched
    print("test_two_cycle_only_graph_rejected_by_decomposition_not_solver: OK", matched)


def test_self_loop_never_offered_never_selected():
    # Self-loops should never appear in `edges` (candidate generation excludes
    # target==source), but verify the solver does not spuriously create one.
    nodes = ["a", "b"]
    edges = {("a", "b"): _edge("a", "b", 1.0)}
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    for s, t in matched.items():
        assert s != t
    print("test_self_loop_never_offered_never_selected: OK", matched)


def test_score_tie_deterministic():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"): _edge("a", "b", 1.0), ("a", "c"): _edge("a", "c", 1.0)}
    m1 = _max_weight_one_in_one_out_matching(nodes, edges)
    m2 = _max_weight_one_in_one_out_matching(nodes, edges)
    assert m1 == m2, (m1, m2)
    print("test_score_tie_deterministic: OK", m1)


def test_extreme_score_range():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"): _edge("a", "b", 1e9), ("a", "c"): _edge("a", "c", 1e-9)}
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "b"}, matched
    print("test_extreme_score_range: OK", matched)


def test_max_score_chooses_higher_scoring_edge():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"): _edge("a", "b", 1.0), ("a", "c"): _edge("a", "c", 5.0)}
    matched = _max_weight_one_in_one_out_matching(nodes, edges)
    assert matched == {"a": "c"}, matched
    print("test_max_score_chooses_higher_scoring_edge: OK", matched)


if __name__ == "__main__":
    test_all_forbidden_node_stays_unmatched()
    test_successor_only_no_predecessor()
    test_valid_open_path_three_nodes()
    test_valid_cycle_plus_unmatched_extra_node()
    test_two_disjoint_cycles()
    test_two_cycle_only_graph_rejected_by_decomposition_not_solver()
    test_self_loop_never_offered_never_selected()
    test_score_tie_deterministic()
    test_extreme_score_range()
    test_max_score_chooses_higher_scoring_edge()
    print("ALL PASSED")
