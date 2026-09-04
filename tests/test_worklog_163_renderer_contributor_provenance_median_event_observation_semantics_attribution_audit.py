from __future__ import annotations

import numpy as np

from devtools.demo.worklog_163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit import (
    AFTER_MEDIAN_EVENT,
    AT_MEDIAN_EVENT,
    BEFORE_MEDIAN_EVENT,
    MEDIAN_DIFFERENT_REGION,
    MEDIAN_IDENTITY_UNAVAILABLE,
    MEDIAN_SAME_GAUSSIAN,
    MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN,
    QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE,
    QUERY_IS_EXACT_CONTRIBUTOR,
    QUERY_NOT_CONTRIBUTOR,
    _median_identity,
    _order_relation,
    _query_participation,
    synthetic_contracts,
)


def test_synthetic_a_to_f_contracts_pass() -> None:
    result = synthetic_contracts()
    assert result["all_pass"] is True
    assert len(result["cases"]) == 6
    assert result["no_contribution_threshold"] is True


def test_query_participation_is_exact_only_when_captured_or_untruncated_absence() -> None:
    assert _query_participation(7, [2, 7], 99) == QUERY_IS_EXACT_CONTRIBUTOR
    assert _query_participation(7, [2, 3], 2) == QUERY_NOT_CONTRIBUTOR
    assert _query_participation(7, [2, 3], 17) == QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE


def test_median_identity_uses_stable_row_and_existing_region_only() -> None:
    assert _median_identity(4, 1, 4, 1) == MEDIAN_SAME_GAUSSIAN
    assert _median_identity(4, 1, 9, 1) == MEDIAN_SAME_REGION_DIFFERENT_GAUSSIAN
    assert _median_identity(4, 1, 9, 0) == MEDIAN_DIFFERENT_REGION
    assert _median_identity(4, 1, -1, -1) == MEDIAN_IDENTITY_UNAVAILABLE


def test_order_relation_has_no_epsilon_or_depth_threshold() -> None:
    assert _order_relation(4, 9, 3, 0) == BEFORE_MEDIAN_EVENT
    assert _order_relation(4, 4, 3, 0) == AT_MEDIAN_EVENT
    assert _order_relation(4, 9, 3, 1) == AFTER_MEDIAN_EVENT
    assert _order_relation(4, 9, -1, -1) != AFTER_MEDIAN_EVENT


def test_synthetic_labels_are_not_numeric_state_votes() -> None:
    values = np.asarray([QUERY_NOT_CONTRIBUTOR, QUERY_IS_EXACT_CONTRIBUTOR, QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE])
    assert values.tolist() == [QUERY_NOT_CONTRIBUTOR, QUERY_IS_EXACT_CONTRIBUTOR, QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE]
