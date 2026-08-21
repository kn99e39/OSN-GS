from __future__ import annotations

import torch

from osn_gs.surface.torch_node_level_observability_accounting import (
    CATEGORY_A_NEVER_POSITIVELY_OBSERVED,
    CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE,
    CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE,
    CATEGORY_D_OBSERVED_CONFLICT_ONLY,
    NODE_OBSERVABILITY_CATEGORIES,
    SINGLETON_CAUSE_CATEGORIES,
    SINGLETON_COOBSERVED_CORRIDOR_FAILS,
    SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE,
    SINGLETON_NODE_VISIBLE_NO_COOBSERVED_EDGE,
    SINGLETON_OBSERVATION_CONFLICT,
    SINGLETON_POSITIVE_BUT_GEOMETRIC_CUT,
    NodeViewObservability,
    classify_node_observability,
    classify_singleton_causes,
    compute_node_view_observability,
    node_observability_accounting,
)
from tests.test_maximal_visible_connectivity import (
    _build_camera,
    _evidence_for,
    _grid,
    _lookat_world_view,
)
from tests.test_positive_visible_adjacency import _splatted_evidence


def _observability(count: int, on_observed) -> NodeViewObservability:
    on_observed_t = torch.as_tensor(on_observed, dtype=torch.int32)
    assert int(on_observed_t.shape[0]) == count
    return NodeViewObservability(
        on_observed_surface_view_count=on_observed_t,
        in_bounds_view_count=torch.zeros((count,), dtype=torch.int32),
        projectable_view_count=None,
        total_views=3,
    )


# --------------------------------------------------------------------------
# classify_node_observability: A/B/C/D (directive section 2)
# --------------------------------------------------------------------------


def test_never_observed_node_is_category_a():
    obs = _observability(3, [0, 2, 0])
    category = classify_node_observability(obs, torch.tensor([True, True, True]), torch.tensor([True, True, True]))
    assert NODE_OBSERVABILITY_CATEGORIES[int(category[0])] == CATEGORY_A_NEVER_POSITIVELY_OBSERVED
    assert NODE_OBSERVABILITY_CATEGORIES[int(category[2])] == CATEGORY_A_NEVER_POSITIVELY_OBSERVED


def test_observed_node_without_positive_edge_is_category_b():
    obs = _observability(1, [4])
    category = classify_node_observability(obs, torch.tensor([False]), torch.tensor([False]))
    assert NODE_OBSERVABILITY_CATEGORIES[int(category[0])] == CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE


def test_observed_node_with_positive_edge_is_category_c():
    obs = _observability(1, [4])
    category = classify_node_observability(obs, torch.tensor([True]), torch.tensor([False]))
    assert NODE_OBSERVABILITY_CATEGORIES[int(category[0])] == CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE


def test_observed_node_with_conflict_only_is_category_d():
    obs = _observability(1, [4])
    category = classify_node_observability(obs, torch.tensor([False]), torch.tensor([True]))
    assert NODE_OBSERVABILITY_CATEGORIES[int(category[0])] == CATEGORY_D_OBSERVED_CONFLICT_ONLY


def test_positive_edge_takes_priority_over_conflict_when_both_present():
    obs = _observability(1, [4])
    category = classify_node_observability(obs, torch.tensor([True]), torch.tensor([True]))
    assert NODE_OBSERVABILITY_CATEGORIES[int(category[0])] == CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE


# --------------------------------------------------------------------------
# node_observability_accounting: primitive vs visible-topology masking
# --------------------------------------------------------------------------


def test_node_observability_accounting_mask_restricts_to_singletons():
    category = torch.tensor([0, 0, 1, 2, 3])  # A, A, B, C, D
    all_accounting = node_observability_accounting(category)
    assert all_accounting["total"] == 5
    assert all_accounting["counts"][NODE_OBSERVABILITY_CATEGORIES[0]] == 2

    singleton_mask = torch.tensor([True, False, True, False, False])
    singleton_accounting = node_observability_accounting(category, mask=singleton_mask)
    assert singleton_accounting["total"] == 2
    assert singleton_accounting["counts"][NODE_OBSERVABILITY_CATEGORIES[0]] == 1
    assert singleton_accounting["counts"][NODE_OBSERVABILITY_CATEGORIES[1]] == 1
    assert singleton_accounting["fractions"][NODE_OBSERVABILITY_CATEGORIES[0]] == 0.5


# --------------------------------------------------------------------------
# classify_singleton_causes: 6-way exclusive attribution (directive section 5)
# --------------------------------------------------------------------------


def test_singleton_never_visible_node():
    obs = _observability(1, [0])
    category = classify_singleton_causes(
        torch.tensor([True]), obs, torch.tensor([True]), torch.tensor([True]), torch.tensor([False]), torch.tensor([False])
    )
    assert SINGLETON_CAUSE_CATEGORIES[int(category[0])] == SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE


def test_singleton_visible_but_no_coobserved_edge():
    obs = _observability(1, [3])
    category = classify_singleton_causes(
        torch.tensor([True]), obs, torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), torch.tensor([False])
    )
    assert SINGLETON_CAUSE_CATEGORIES[int(category[0])] == SINGLETON_NODE_VISIBLE_NO_COOBSERVED_EDGE


def test_singleton_coobserved_but_corridor_fails():
    obs = _observability(1, [3])
    category = classify_singleton_causes(
        torch.tensor([True]), obs, torch.tensor([True]), torch.tensor([False]), torch.tensor([False]), torch.tensor([False])
    )
    assert SINGLETON_CAUSE_CATEGORIES[int(category[0])] == SINGLETON_COOBSERVED_CORRIDOR_FAILS


def test_singleton_positive_but_geometric_gate_cuts():
    obs = _observability(1, [3])
    category = classify_singleton_causes(
        torch.tensor([True]), obs, torch.tensor([True]), torch.tensor([True]), torch.tensor([True]), torch.tensor([False])
    )
    assert SINGLETON_CAUSE_CATEGORIES[int(category[0])] == SINGLETON_POSITIVE_BUT_GEOMETRIC_CUT


def test_singleton_observation_conflict():
    obs = _observability(1, [3])
    category = classify_singleton_causes(
        torch.tensor([True]), obs, torch.tensor([True]), torch.tensor([False]), torch.tensor([False]), torch.tensor([True])
    )
    assert SINGLETON_CAUSE_CATEGORIES[int(category[0])] == SINGLETON_OBSERVATION_CONFLICT


def test_non_singleton_entries_are_ignored_by_mask_not_by_value():
    """The category tensor is computed for every node (vectorized), but
    callers must restrict reporting via `mask` -- verify a non-singleton
    node's category value does not leak into a singleton-only accounting
    call even though it was computed densely."""

    obs = _observability(2, [0, 3])
    category = classify_singleton_causes(
        torch.tensor([True, False]), obs, torch.tensor([True, True]), torch.tensor([True, True]),
        torch.tensor([False, False]), torch.tensor([False, False]),
    )
    accounting = node_observability_accounting(category, categories=SINGLETON_CAUSE_CATEGORIES, mask=torch.tensor([True, False]))
    assert accounting["total"] == 1
    assert accounting["counts"][SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE] == 1


# --------------------------------------------------------------------------
# compute_node_view_observability: real per-view canonical classification
# --------------------------------------------------------------------------


def test_compute_node_view_observability_counts_match_camera_coverage():
    positions = _grid((-0.3, 0.3), (-0.3, 0.3), 2.0, 0.1)
    camera_front = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    camera_blind = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    evidence_front = _evidence_for(positions, [camera_front])
    from osn_gs.surface.torch_observation_evidence import ObservationEvidence
    from dataclasses import replace as _dc_replace

    view_blind = _dc_replace(evidence_front.views[0], camera_index=1, view_depth=torch.zeros_like(evidence_front.views[0].view_depth), valid_depth_mask=torch.zeros_like(evidence_front.views[0].valid_depth_mask), world_view_transform=camera_blind.world_view_transform, full_proj_transform=camera_blind.full_proj_transform)
    combined = ObservationEvidence(views=[evidence_front.views[0], view_blind], near=evidence_front.near, far=evidence_front.far, depth_epsilon=evidence_front.depth_epsilon, topology_version="test", camera_set_version="test")

    result = compute_node_view_observability(positions, combined)
    assert int(result.on_observed_surface_view_count.max()) <= 2
    assert bool((result.on_observed_surface_view_count > 0).any())
    # the blind camera contributes 0 to on_observed_surface everywhere
    assert int(result.on_observed_surface_view_count.sum()) <= int(positions.shape[0])


def test_compute_node_view_observability_projectable_count_uses_supplied_radii():
    positions = _grid((-0.3, 0.3), (-0.3, 0.3), 2.0, 0.1)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    evidence = _splatted_evidence(camera, 256, 256, positions)
    count = int(positions.shape[0])
    radii = torch.zeros((count,))
    radii[: count // 2] = 5.0  # only half the surfels "projectable" in this view
    result = compute_node_view_observability(positions, evidence, radii_per_view={0: radii})
    assert result.projectable_view_count is not None
    assert int((result.projectable_view_count > 0).sum()) == count // 2
