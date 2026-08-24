from __future__ import annotations

import torch

from scripts.devtools.camera_induced_representative_backbone_audit import (
    _component_size_stats,
    _distribution,
    _find_bridges,
    _scatter_or,
)


def test_find_bridges_on_a_simple_path():
    """0-1-2-3: every edge is a bridge."""

    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    bridges = _find_bridges(4, edges)
    bridge_set = {frozenset(pair) for pair in bridges}
    assert bridge_set == {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3))}


def test_find_bridges_on_a_cycle_with_a_pendant():
    """Triangle 0-1-2-0 plus a pendant edge 2-3: only (2,3) is a bridge."""

    edges = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3]])
    bridges = _find_bridges(4, edges)
    bridge_set = {frozenset(pair) for pair in bridges}
    assert bridge_set == {frozenset((2, 3))}


def test_find_bridges_on_a_fully_connected_component_has_none():
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]])
    bridges = _find_bridges(3, edges)
    assert bridges == []


def test_find_bridges_handles_disconnected_nodes():
    """An isolated node (no edges) must not crash bridge-finding."""

    edges = torch.tensor([[0, 1]])
    bridges = _find_bridges(3, edges)  # node 2 is isolated
    assert {frozenset(pair) for pair in bridges} == {frozenset((0, 1))}


def test_distribution_matches_known_values():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = _distribution(values)
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["median"] == 3.0
    assert stats["mean"] == 3.0


def test_distribution_empty_input_is_well_defined():
    stats = _distribution(torch.zeros((0,)))
    assert stats["min"] == 0.0 and stats["max"] == 0.0


def test_component_size_stats_delegates_to_distribution():
    sizes = torch.tensor([1, 1, 1, 5, 10])
    stats = _component_size_stats(sizes)
    assert stats["max"] == 10.0
    assert stats["min"] == 1.0


def test_scatter_or_marks_both_endpoints():
    edges = torch.tensor([[0, 2], [1, 3]])
    flag = torch.tensor([True, False])
    node_flag = _scatter_or(4, edges, flag, torch.device("cpu"))
    assert node_flag.tolist() == [True, False, True, False]


def test_scatter_or_empty_edges_returns_all_false():
    edges = torch.zeros((0, 2), dtype=torch.int64)
    flag = torch.zeros((0,), dtype=torch.bool)
    node_flag = _scatter_or(3, edges, flag, torch.device("cpu"))
    assert node_flag.tolist() == [False, False, False]
