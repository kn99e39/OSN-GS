"""Worklog 179 -- renderer-native three-state contract gate tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osn_gs.render.torch_observation_contract_audit import (
    RAW_REACHED,
    RAW_TERMINATED,
    RAW_UNRESOLVED,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    aggregate_raw_query_events,
    classify_raw_query_event,
    evaluate_contract_gate,
    existing_evidence_subject_audit,
    primitive_observation_outcome,
    validate_audit_schema,
)


def test_subject_type_audit_keeps_primitive_and_point_query_evidence_separate() -> None:
    audits = {item.name: item for item in existing_evidence_subject_audit()}
    assert audits["forward_accepted"].subject_type == "primitive-camera pair"
    assert audits["forward_accepted"].semantic_type == "Primitive Observation Evidence"
    assert audits["forward_accepted"].positive_observed_witness is True
    assert audits["forward_accepted"].positive_blocker_witness is False
    assert audits["forward_accepted"].supports_arbitrary_world_point is False

    assert audits["qdepth query event"].subject_type == "camera-ray point query"
    assert audits["qdepth query event"].semantic_type == "Point-Query Occlusion Evidence candidate"
    assert audits["qdepth query event"].positive_blocker_witness is True
    assert audits["qdepth query event"].positive_observed_witness is False
    assert audits["qdepth query event"].supports_arbitrary_world_point is True


def test_raw_qdepth_partition_is_exhaustive_and_mutually_exclusive() -> None:
    assert classify_raw_query_event(1, 0, query_resolution_depth=2.0) == RAW_TERMINATED
    assert classify_raw_query_event(0, 1, query_resolution_depth=2.0) == RAW_REACHED
    assert classify_raw_query_event(0, 0) == RAW_UNRESOLVED
    with pytest.raises(ValueError, match="both be 1"):
        classify_raw_query_event(1, 1)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        classify_raw_query_event(2, 0)
    with pytest.raises(ValueError, match="resolution_depth"):
        classify_raw_query_event(0, 0, query_resolution_depth=3.0)


def test_raw_global_aggregation_is_order_invariant_and_not_majority_vote() -> None:
    assert aggregate_raw_query_events([RAW_TERMINATED, RAW_TERMINATED]) == RAW_TERMINATED
    assert aggregate_raw_query_events([RAW_TERMINATED, RAW_UNRESOLVED]) == RAW_UNRESOLVED
    assert aggregate_raw_query_events([RAW_UNRESOLVED, RAW_TERMINATED]) == RAW_UNRESOLVED
    assert aggregate_raw_query_events([RAW_REACHED, RAW_TERMINATED, RAW_UNRESOLVED]) == RAW_REACHED
    assert aggregate_raw_query_events([RAW_TERMINATED, RAW_REACHED]) == RAW_REACHED
    assert aggregate_raw_query_events([]) == RAW_UNRESOLVED


def test_primitive_noncontribution_never_becomes_occluded() -> None:
    assert primitive_observation_outcome(1) == STATE_OBSERVED
    assert primitive_observation_outcome(0) == STATE_UNRESOLVED
    assert primitive_observation_outcome(0) != STATE_OCCLUDED


def test_gate_stops_without_promoting_an_f() -> None:
    report = evaluate_contract_gate()
    assert report["verdict"] == "NO_RENDERER_NATIVE_THREE_STATE_CONTRACT"
    assert report["f_implemented"] is False
    assert "SUBJECT_TYPE_MISMATCH" in report["stop_conditions"]
    assert "OCCLUDED_NOT_POSITIVELY_IDENTIFIABLE" in report["stop_conditions"]
    assert report["point_query"]["raw_event_partition"] == sorted(
        [RAW_REACHED, RAW_TERMINATED, RAW_UNRESOLVED]
    )


def test_audit_schema_is_explicit() -> None:
    for item in existing_evidence_subject_audit():
        validate_audit_schema(item.as_dict())


def test_audit_module_has_no_forbidden_surface_dependencies() -> None:
    path = Path(__file__).resolve().parents[1] / "osn_gs" / "render" / "torch_observation_contract_audit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("osn_gs.surface") for name in imported)
    assert not any(name.startswith("osn_gs.core") for name in imported)

