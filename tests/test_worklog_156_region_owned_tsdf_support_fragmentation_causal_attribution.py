import numpy as np

from devtools.demo.worklog_156_region_owned_tsdf_support_fragmentation_causal_attribution import (
    CATEGORY_CODE,
    CATEGORIES,
    INTERNAL_SAME_REGION,
    _classify_observed_state,
    _classify_target_frontier,
    _encode_cells,
    _lookup_sorted,
    _synthetic_contracts,
)


def test_scalar_frontier_classifier_covers_all_mutually_exclusive_categories() -> None:
    cases = {
        "OUTSIDE_AUTHORITATIVE_FIELD": dict(authoritative=False, outside_bounds=False, zero_surface=False, sample_exists=False, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0),
        "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE": dict(authoritative=True, outside_bounds=False, zero_surface=False, sample_exists=True, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0),
        "ZERO_SURFACE_DIFFERENT_REGION": dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=True, neighbor_region=8, target_region=7, neighbor_component=1, current_component=0),
        "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS": dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=False, neighbor_region=7, target_region=7, neighbor_component=-1, current_component=0),
        "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED": dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=True, neighbor_region=7, target_region=7, neighbor_component=1, current_component=0),
        "FIELD_REPLAY_VOLUME_BOUNDARY": dict(authoritative=False, outside_bounds=True, zero_surface=False, sample_exists=False, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0),
        "OTHER_EXISTING_CONTRACT_REASON": dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=False, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0),
    }
    observed = {name: _classify_observed_state(**kwargs) for name, kwargs in cases.items()}
    assert observed == {
        "OUTSIDE_AUTHORITATIVE_FIELD": "OUTSIDE_AUTHORITATIVE_FIELD",
        "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE": "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE",
        "ZERO_SURFACE_DIFFERENT_REGION": "ZERO_SURFACE_DIFFERENT_REGION",
        "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS": "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS",
        "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED": "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED",
        "FIELD_REPLAY_VOLUME_BOUNDARY": "FIELD_REPLAY_VOLUME_BOUNDARY",
        "OTHER_EXISTING_CONTRACT_REASON": "OTHER_EXISTING_CONTRACT_REASON",
    }


def test_synthetic_contracts_pass_without_claiming_real_scene_success() -> None:
    result = _synthetic_contracts()
    assert result["all_pass"] is True
    assert result["synthetic_accounting_only_not_architecture_success"] is True
    assert {case["name"][0] for case in result["cases"]} == set("ABCDEFG")


def test_exact_cell_encoding_and_sorted_lookup_are_reversible() -> None:
    cells = np.asarray([[-2, 4, 1], [0, 0, 0], [2, -1, 3]], dtype=np.int64)
    keys = np.sort(_encode_cells(cells))
    positions, present = _lookup_sorted(keys, keys)
    assert present.tolist() == [True, True, True]
    assert positions.tolist() == [0, 1, 2]


def test_face_adjacent_same_component_is_not_reported_as_false_fragmentation() -> None:
    current_cells = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    sample_cells = np.concatenate([current_cells, np.asarray([[2, 0, 0]], dtype=np.int64)], axis=0)
    sample_keys = _encode_cells(sample_cells)
    order = np.argsort(sample_keys)
    sample_keys = sample_keys[order]
    sample_cells = sample_cells[order]

    field_cells = np.asarray(
        [[x, y, z] for x in range(-2, 5) for y in range(-2, 3) for z in range(-2, 3)],
        dtype=np.int64,
    )
    field_keys = np.sort(_encode_cells(field_cells))
    result = _classify_target_frontier(
        {
            "sample_keys": sample_keys,
            "sample_cells": sample_cells,
            "sample_xyz": sample_cells.astype(np.float32),
            "nearest_region": np.asarray([7, 7, 7], dtype=np.int64)[order],
            "owned_region": np.asarray([7, 7, 7], dtype=np.int64)[order],
            "accepted": np.asarray([True, True, True], dtype=bool)[order],
            "component_id": np.asarray([0, 0, 1], dtype=np.int64)[order],
            "component_count": 2,
            "field_keys": field_keys,
            "field_values": np.zeros((len(field_keys),), dtype=np.float32),
            "field_support": np.ones((len(field_keys),), dtype=np.int32),
            "field_min": field_cells.min(axis=0),
            "field_max": field_cells.max(axis=0),
            "nearest_status": np.zeros((3,), dtype=np.int8)[order],
        },
        7,
        chunk_size=2,
    )
    assert result["category_details"]["SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED"]["frontier_face_count"] == 2
    assert result["total_frontier_faces"] > 0
    assert set(CATEGORIES) == set(CATEGORY_CODE)
    assert INTERNAL_SAME_REGION not in result["category_details"]
