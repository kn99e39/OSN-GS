"""Focused WL151 compatibility-gate tests."""

from __future__ import annotations

import json

from devtools.demo import renderer_event_canonical_surface_compatibility_audit as wl151


def test_wl150_baseline_is_reconciled_before_compatibility() -> None:
    audit = wl151.build_audit()
    baseline = audit["baseline_reconciliation"]["wl149"]
    assert audit["baseline_reconciliation"]["exact"] is True
    assert baseline["event_count"] == 1586
    assert baseline["event_union_sha256"] == "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78"
    assert baseline["event_1527_influence_row"]["event_id"] == 1527
    assert baseline["event_1527_influence_row"]["fixed_is_v_min_owner"] is True
    assert baseline["representative_shape"] == [3840, 3]
    assert baseline["support_vertices"] == 314
    assert baseline["support_mask_sha256"] == "23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9"
    assert baseline["fully_supported_cells"] == 211


def test_canonical_contract_is_extracted_and_renderer_gap_is_explicit() -> None:
    audit = wl151.build_audit()
    local = audit["canonical_local_surface_contract"]
    boundary = audit["canonical_boundary_first_contract"]
    matrix = audit["compatibility_matrix"]
    assert local["source_evidence"]["canonical_constructor"]["status"] == "PRESENT"
    assert local["source_evidence"]["canonical_region_formation"]["status"] == "PRESENT"
    assert boundary["source_evidence"]["canonical_boundary_adapter"]["status"] == "PRESENT"
    classifications = {row["classification"] for row in matrix}
    assert "INCOMPATIBLE_MISSING" in classifications
    assert "SEMANTICALLY_DIFFERENT" in classifications
    missing_requirements = {row["canonical_requirement"] for row in matrix if row["classification"] == "INCOMPATIBLE_MISSING"}
    assert "region ownership" in missing_requirements
    assert "boundary adjacency/order" in missing_requirements
    assert "manifold affinity graph" in missing_requirements


def test_stop_condition_a_prevents_candidate_c_and_new_semantics() -> None:
    audit = wl151.build_audit()
    gate = audit["adapter_eligibility_verdict"]
    assert gate["status"] == "CONTRACT_GAP"
    assert gate["eligible"] is False
    assert gate["stop_condition"] == "A"
    assert gate["candidate_c_not_implemented"] is True
    assert gate["synthetic_contracts_not_run"] is True
    assert gate["real_scene_replay_not_run"] is True
    assert audit["architecture_verdict"]["verdict"] == "A. CONTRACT_GAP"
    assert audit["renderer_event_evidence_contract"]["event_1527_human_update"] == "HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE"


def test_numbered_export_is_deterministic_and_preserves_event_1527() -> None:
    result = wl151.write_audit()
    assert result["verdict"] == "A. CONTRACT_GAP"
    assert (wl151.TEMP_ROOT / "compatibility_matrix.md").exists()
    report = json.loads((wl151.TEMP_ROOT / "renderer_event_canonical_surface_compatibility_audit_report.json").read_text(encoding="utf-8"))
    assert report["13. ARCHITECTURE VERDICT"] == "A. CONTRACT_GAP"
    assert report["2. WL150 BASELINE RECONCILIATION"]["event_1527"]["event_id"] == 1527
    assert report["INTENT ALIGNMENT"]["event_1527_removed"] is False
    assert report["ARCHITECTURE RESULT"]["conditional_sections_skipped"] == ["Candidate C", "synthetic contracts", "real-scene replay", "qualitative Candidate C exports"]
