from __future__ import annotations

import numpy as np

from devtools.demo.worklog_162_renderer_median_event_direct_observation_semantic_validity_audit import (
    TABLETOP_REGION_ID,
    WORLD_CLASS_A,
    WORLD_CLASS_B,
    WORLD_CLASS_C,
    _world_classification,
    synthetic_contracts,
)


def test_synthetic_a_to_e_contracts_pass() -> None:
    result = synthetic_contracts()
    assert result["all_pass"] is True
    assert len(result["cases"]) == 5


def test_world_attribution_uses_existing_region_and_membership_only() -> None:
    assert TABLETOP_REGION_ID == 1
    assert _world_classification(TABLETOP_REGION_ID, 0) == WORLD_CLASS_B
    assert _world_classification(0, 0) == WORLD_CLASS_A
    assert _world_classification(TABLETOP_REGION_ID, 2) == WORLD_CLASS_C
    assert _world_classification(TABLETOP_REGION_ID, 4) == WORLD_CLASS_C


def test_world_attribution_does_not_depend_on_a_new_distance_threshold() -> None:
    # The helper accepts only the frozen W155 IDs/status values.  XYZ or a
    # radius is intentionally not part of this decision function.
    assert np.asarray([_world_classification(1, 1), _world_classification(99, 1)]).tolist() == [WORLD_CLASS_B, WORLD_CLASS_A]
