from __future__ import annotations

import numpy as np
import pytest

from devtools.demo.worklog_164_canonical_renderer_contributor_primitive_observability_historical_candidate_b_ab import (
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    _apply_contributor_override,
    _contributor_count_bins,
    _population_accounting,
    _transition_matrix,
    synthetic_contracts,
)


def test_synthetic_a_to_f_contracts_pass() -> None:
    result = synthetic_contracts()
    assert result["all_pass"] is True
    assert len(result["cases"]) == 6


def test_positive_contributor_override_only_changes_observation_state() -> None:
    point_query = np.asarray([STATE_OCCLUDED, STATE_OBSERVED, STATE_UNRESOLVED, STATE_OCCLUDED], dtype=np.int8)
    contributed = np.asarray([True, False, False, True], dtype=bool)
    candidate = _apply_contributor_override(point_query, contributed)
    np.testing.assert_array_equal(point_query, [STATE_OCCLUDED, STATE_OBSERVED, STATE_UNRESOLVED, STATE_OCCLUDED])
    np.testing.assert_array_equal(candidate, [STATE_OBSERVED, STATE_OBSERVED, STATE_UNRESOLVED, STATE_OBSERVED])


def test_transition_and_region_accounting_preserve_historical_occluded_breakdown() -> None:
    baseline = np.asarray([STATE_OBSERVED, STATE_OCCLUDED, STATE_OCCLUDED, STATE_UNRESOLVED], dtype=np.int8)
    candidate = np.asarray([STATE_OBSERVED, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED], dtype=np.int8)
    counts = np.asarray([0, 1, 6, 21], dtype=np.int32)
    assert _transition_matrix(baseline, candidate)["OCCLUDED"] == {"OBSERVED": 1, "OCCLUDED": 1, "UNRESOLVED": 0}
    result = _population_accounting(np.arange(4), baseline, candidate, counts)
    assert result["historical_global_occluded_to_candidate_global_observed"] == 1
    assert result["historical_global_occluded_to_candidate_global_unresolved"] == 0
    assert result["historical_global_occluded_unchanged_global_occluded"] == 1
    assert result["contributor_camera_count_distribution_per_primitive"]["bins"] == {
        "zero_cameras": 1,
        "exactly_1_camera": 1,
        "2_to_5_cameras": 0,
        "6_to_10_cameras": 1,
        "11_to_20_cameras": 0,
        "greater_than_20_cameras": 1,
    }


def test_contributor_camera_bins_are_explicit_and_non_overlapping() -> None:
    counts = np.asarray([0, 1, 2, 5, 6, 10, 11, 20, 21], dtype=np.int32)
    bins = _contributor_count_bins(counts)
    assert sum(bins.values()) == counts.size


def test_arbitrary_xyz_without_primitive_identity_is_refused() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        _apply_contributor_override(np.asarray([STATE_OCCLUDED]), np.asarray([True, False]))
