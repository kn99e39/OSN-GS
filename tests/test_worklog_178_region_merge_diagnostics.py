from __future__ import annotations

import numpy as np
import torch

from osn_gs.surface.torch_region_coherent_surfel_partition import (
    PARTITION_ROLES,
    ROLE_STRUCTURAL_CORE,
    RegionCoherenceConfig,
    partition_surfels_region_coherent,
)
from osn_gs.surface.worklog_178_region_merge_diagnostics import (
    diagnostic_contract_summary,
    deterministic_shortest_path,
    graph_support_diagnostic,
    path_diagnostic,
    plain_w96_subset_ids,
    replay_w97_region_growth,
)
from scripts.devtools.worklog_178_region_level_anti_chaining_contract_discovery import (
    Orientation,
    fixtures,
)


def test_diagnostic_replay_does_not_change_w97_subset_ids():
    config = RegionCoherenceConfig()
    for fixture in fixtures():
        orientation = Orientation(fixture.positions, fixture.normals)
        baseline = partition_surfels_region_coherent(orientation, config)
        chronology = replay_w97_region_growth(orientation, config, graph=baseline.graph)
        summary = diagnostic_contract_summary(baseline.subset_ids, baseline.subset_ids.clone(), chronology)
        assert summary["diagnostic_enabled_w97_subset_ids_equal_historical_baseline"] is True


def test_replayed_merge_results_match_w97_rejected_merge_mask():
    """The read-only chronology must expose W97's existing decision, not a
    numerically similar replacement rule."""

    config = RegionCoherenceConfig()
    for fixture in fixtures():
        orientation = Orientation(fixture.positions, fixture.normals)
        baseline = partition_surfels_region_coherent(orientation, config)
        chronology = replay_w97_region_growth(orientation, config, graph=baseline.graph)
        rejected_mask = baseline.rejected_merge_mask.detach().cpu().numpy()
        for event in chronology.events:
            assert (event.result == "rejected_region_concentration") == bool(
                rejected_mask[event.candidate_edge_index]
            )

def test_merge_chronology_is_deterministic():
    fixture = fixtures()[1]
    orientation = Orientation(fixture.positions, fixture.normals)
    config = RegionCoherenceConfig()
    baseline = partition_surfels_region_coherent(orientation, config)
    first = replay_w97_region_growth(orientation, config, graph=baseline.graph)
    second = replay_w97_region_growth(orientation, config, graph=baseline.graph)
    assert [event.to_dict() for event in first.events] == [event.to_dict() for event in second.events]
    assert np.array_equal(first.final_component_roots, second.final_component_roots)


def test_scatter_spectrum_accounting_has_three_normalized_values():
    fixture = fixtures()[2]
    orientation = Orientation(fixture.positions, fixture.normals)
    partition = partition_surfels_region_coherent(orientation)
    chronology = replay_w97_region_growth(orientation, graph=partition.graph)
    assert chronology.events
    for event in chronology.events:
        assert len(event.hypothetical_post_spectrum) == 3
        assert abs(sum(event.hypothetical_post_spectrum) - 1.0) < 1e-6
        assert 0.0 <= event.local_pairwise_angle_degrees <= 90.0


def test_path_diagnostic_uses_deterministic_existing_graph_shortest_path():
    fixture = fixtures()[1]
    orientation = Orientation(fixture.positions, fixture.normals)
    partition = partition_surfels_region_coherent(orientation)
    first = deterministic_shortest_path(partition.graph, *fixture.path_indices)
    second = deterministic_shortest_path(partition.graph, *fixture.path_indices)
    assert first == second
    assert first is not None
    diagnostic = path_diagnostic(orientation, first)
    assert diagnostic["available"] is True
    assert diagnostic["path_length_edges"] == len(first) - 1
    assert diagnostic["cumulative_unsigned_local_rotation_degrees"] >= diagnostic["direct_endpoint_unsigned_normal_angle_degrees"]


def test_graph_support_is_diagnostic_only_and_has_no_membership_result():
    fixture = fixtures()[3]
    orientation = Orientation(fixture.positions, fixture.normals)
    partition = partition_surfels_region_coherent(orientation)
    chronology = replay_w97_region_growth(orientation, graph=partition.graph)
    support = graph_support_diagnostic(partition.graph, chronology, chronology.first_rejection())
    assert support["available"] is True
    assert support["diagnostic_only"] is True
    assert support["articulation_like"] in (True, False)
    assert "threshold" not in str(support).lower()


def test_synthetic_contracts_cover_required_positive_and_negative_cases():
    results = {}
    config = RegionCoherenceConfig()
    for fixture in fixtures():
        orientation = Orientation(fixture.positions, fixture.normals)
        partition = partition_surfels_region_coherent(orientation, config)
        plain = plain_w96_subset_ids(partition.graph, config.local)
        chronology = replay_w97_region_growth(orientation, config, graph=partition.graph)
        results[fixture.category] = (int(np.unique(plain).size), int(partition.subset_count), len(chronology.rejected_events))

    assert results["PLANAR_POSITIVE"][1] == 1
    assert results["SMOOTH_CURVED_POSITIVE"][0] == 1
    assert results["SMOOTH_CURVED_POSITIVE"][1] > 1
    assert results["STRONGLY_BENT_CONTINUOUS_POSITIVE"][0] == 1
    assert results["STRONGLY_BENT_CONTINUOUS_POSITIVE"][1] > 1
    assert results["PATHOLOGICAL_CHAIN_NEGATIVE"][0] == 1
    assert results["PATHOLOGICAL_CHAIN_NEGATIVE"][1] > 1
    assert results["PATHOLOGICAL_CHAIN_NEGATIVE"][2] > 0
    assert results["PARALLEL_SHORTCUT_NEGATIVE"][1] == 1


def test_stable_ids_are_carried_without_reconstruction_in_synthetic_controls():
    fixture = fixtures()[0]
    orientation = Orientation(fixture.positions, fixture.normals)
    assert torch.equal(orientation.gaussian_ids, torch.arange(len(fixture.positions), dtype=torch.int64))
    partition = partition_surfels_region_coherent(orientation)
    chronology = replay_w97_region_growth(orientation, graph=partition.graph)
    for event in chronology.events:
        assert event.endpoint_stable_ids == event.endpoint_indices


def test_structural_core_final_membership_matches_replay_components():
    fixture = fixtures()[3]
    orientation = Orientation(fixture.positions, fixture.normals)
    partition = partition_surfels_region_coherent(orientation)
    chronology = replay_w97_region_growth(orientation, graph=partition.graph)
    core = partition.partition_role == PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)
    replay_roots = chronology.final_component_roots
    subset = partition.subset_ids.detach().cpu().numpy()
    core_indices = torch.nonzero(core, as_tuple=False).reshape(-1).tolist()
    for left in core_indices:
        for right in core_indices:
            assert (replay_roots[left] == replay_roots[right]) == (subset[left] == subset[right])
