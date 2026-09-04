from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "devtools" / "demo" / "worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit.py"
SPEC = importlib.util.spec_from_file_location("worklog160_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_synthetic_a_to_h_contracts_pass() -> None:
    result = MODULE.synthetic_contracts()
    assert result["all_pass"] is True
    assert len(result["cases"]) == 8
    assert all(case["pass"] for case in result["cases"])


def test_projective_signed_distance_keeps_surface_alignment_separate() -> None:
    aligned = MODULE.classify_projective_sdf_evidence(
        relevant=True, query_depth=2.0, median_depth=2.0
    )
    reachable = MODULE.classify_projective_sdf_evidence(
        relevant=True, query_depth=1.0, median_depth=2.0
    )
    occluded = MODULE.classify_projective_sdf_evidence(
        relevant=True, query_depth=3.0, median_depth=2.0
    )
    assert aligned["state"] == MODULE.STATE_OBSERVED
    assert aligned["surface_aligned"] is True
    assert reachable["state"] == MODULE.STATE_OBSERVED
    assert reachable["surface_aligned"] is False
    assert occluded["state"] == MODULE.STATE_OCCLUDED


def test_invalid_or_irrelevant_camera_is_not_an_occluded_vote() -> None:
    unresolved = MODULE.classify_projective_sdf_evidence(
        relevant=True, query_depth=2.0, median_depth=None
    )
    non_relevant = MODULE.classify_projective_sdf_evidence(
        relevant=False, query_depth=3.0, median_depth=2.0
    )
    assert unresolved["state"] == MODULE.STATE_UNRESOLVED
    assert non_relevant["state"] == MODULE.STATE_NON_RELEVANT
    states = np.asarray([[non_relevant["state"], MODULE.STATE_OCCLUDED]], dtype=np.int8)
    assert MODULE.aggregate_persistent_states(states)[0] == MODULE.STATE_OCCLUDED


def test_all_relevant_views_are_required_for_global_occlusion() -> None:
    all_occluded = np.asarray(
        [[MODULE.STATE_OCCLUDED, MODULE.STATE_NON_RELEVANT, MODULE.STATE_OCCLUDED]], dtype=np.int8
    )
    mixed = np.asarray(
        [[MODULE.STATE_OCCLUDED, MODULE.STATE_OBSERVED, MODULE.STATE_OCCLUDED]], dtype=np.int8
    )
    unresolved = np.asarray(
        [[MODULE.STATE_OCCLUDED, MODULE.STATE_UNRESOLVED, MODULE.STATE_NON_RELEVANT]], dtype=np.int8
    )
    no_relevant = np.asarray(
        [[MODULE.STATE_NON_RELEVANT, MODULE.STATE_NON_RELEVANT]], dtype=np.int8
    )
    assert MODULE.aggregate_persistent_states(all_occluded)[0] == MODULE.STATE_OCCLUDED
    assert MODULE.aggregate_persistent_states(mixed)[0] == MODULE.STATE_OBSERVED
    assert MODULE.aggregate_persistent_states(unresolved)[0] == MODULE.STATE_UNRESOLVED
    assert MODULE.aggregate_persistent_states(no_relevant)[0] == MODULE.STATE_UNRESOLVED
