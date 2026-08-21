from __future__ import annotations

import math

import torch

from osn_gs.surface.torch_positive_visible_adjacency import PositiveVisibleAdjacencyConfig, partition_positive_visible_adjacency
from osn_gs.surface.torch_primitive_ownership_visible_topology_separation import (
    derive_primitive_vs_visible_topology_accounting,
)
from tests.test_maximal_visible_connectivity import _LOCAL_CONFIG, _Orientation, _cylinder_band, _wall_pair_with_gap


def test_all_singletons_are_owned_but_not_structural_topology_members():
    """No camera at all -> every surfel is its own singleton component
    (see test_positive_visible_adjacency.py::test_no_positive_observation_never_creates_adjacency).
    Primitive ownership must still be 100%, but visible-topology membership
    must be 0% -- exactly the representational separation this batch adds."""

    positions = torch.stack([torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.full((10,), 2.0)], dim=1)
    orientation = _Orientation(positions, torch.tensor([[0.0, 0.0, -1.0]]).repeat(10, 1), torch.tensor([[1.0, 0.0, 0.0]]).repeat(10, 1), torch.tensor([[0.0, 1.0, 0.0]]).repeat(10, 1))
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, None, config)

    primitive, topology = derive_primitive_vs_visible_topology_accounting(result)
    assert primitive.total_surfels == 10
    assert primitive.retained_surfels == 10
    assert primitive.payload()["ownership_complete"] is True

    assert topology.structural_visible_surfel_count == 0
    assert topology.structural_visible_surfel_fraction == 0.0
    assert topology.non_visible_topology_owned_surfel_count == 10
    assert topology.non_visible_topology_owned_surfel_fraction == 1.0
    assert bool(topology.structural_membership_mask.any()) is False


def test_wall_pair_with_occluder_has_structural_membership_on_both_sides():
    """A genuine multi-member component (each wall side, tied together by
    real positive edges) must be counted as structural visible-topology
    membership -- the separation must not zero out real components."""

    from tests.test_positive_visible_adjacency import _wall_gap_wall_camera_evidence

    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)

    primitive, topology = derive_primitive_vs_visible_topology_accounting(result)
    assert primitive.total_surfels == len(result)
    assert primitive.retained_surfels == primitive.total_surfels
    assert topology.structural_visible_surfel_count > 0
    assert topology.structural_component_count >= 2  # at least the two wall-side components
    # coverage identity (ownership) must hold regardless of topology split
    assert topology.structural_visible_surfel_count + topology.non_visible_topology_owned_surfel_count == primitive.total_surfels


def test_empty_input_stays_well_defined():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)))
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(empty, None, config)
    primitive, topology = derive_primitive_vs_visible_topology_accounting(result)
    assert primitive.total_surfels == 0
    assert topology.structural_visible_surfel_count == 0
    assert topology.structural_component_count == 0


def test_ownership_never_discards_non_structural_surfels():
    """The central non-negotiable contract (directive section 6, Branch A):
    non-visible/unresolved surfels are retained, never discarded. Ownership
    count must equal the ORIGINAL input count regardless of how fragmented
    the visible topology is."""

    orientation = _cylinder_band(n_theta=10, n_y=2, theta_span=math.pi / 2.0)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, None, config)
    primitive, topology = derive_primitive_vs_visible_topology_accounting(result)
    assert primitive.retained_surfels == int(orientation.positions.shape[0])
    assert topology.structural_visible_surfel_count + topology.non_visible_topology_owned_surfel_count == primitive.retained_surfels
