"""Focused WL152 Raw Visible Surface carrier-contract tests."""

from __future__ import annotations

import json

from devtools.demo import visible_surface_carrier_contract_audit as wl152


def test_historical_baseline_and_raw_artifact_are_reconciled() -> None:
    audit = wl152.build_audit()
    baseline = audit["baseline_reconciliation"]
    assert baseline["wl149_wl151_exact"] is True
    assert baseline["event_count"] == 1586
    assert baseline["event_union_sha256"] == "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78"
    assert baseline["event_1527"]["event_id"] == 1527
    assert baseline["event_1527"]["v_min_owner"] is True
    assert baseline["representative_shape"] == [3840, 3]
    assert baseline["support_vertices"] == 314
    assert baseline["all_four_supported_cells"] == 211
    assert baseline["raw_visible_surface_artifact"]["elements"] == {"vertex": 1212365}
    assert baseline["raw_visible_surface_exact_replay"]["status"] == "AVAILABLE_POINT_CLOUD_ONLY_TSDF_MESH_REPLAY_UNAVAILABLE"


def test_raw_surface_contract_does_not_promote_point_cloud_to_mesh_topology() -> None:
    audit = wl152.build_audit()
    contract = audit["raw_visible_surface_contract"]["property_contract"]
    assert contract["XYZ"]["status"] == "DIRECTLY_PRESENT"
    assert contract["mesh_face"]["status"] == "ABSENT"
    assert contract["connectivity"]["status"] == "ABSENT"
    assert contract["source_TSDF_cell"]["status"] == "ABSENT"
    assert contract["source_camera_evidence_provenance"]["status"] == "ABSENT"
    assert contract["ownership"]["status"] == "ABSENT"
    assert audit["connectivity_diagnostic"]["connected_components"] == "NOT_DEFINED_WITHOUT_FACES"


def test_provenance_and_membership_gaps_are_explicit() -> None:
    audit = wl152.build_audit()
    provenance = audit["evidence_to_raw_surface_provenance"]
    assert provenance["event_1527_trace"]["status"] == "NOT_MAPPABLE_UNDER_EXISTING_CONTRACT"
    assert provenance["event_1527_trace"]["blacklist"] is False
    assert provenance["mesh_element_to_renderer_event"]["status"] == "ABSENT"
    assert audit["physical_sheet_connectivity_result"]["outcome"] == "E. INSUFFICIENT_EVIDENCE"
    assert audit["topology_carrier_eligibility"]["architecture_verdict"] == "INELIGIBLE_CARRIER"


def test_candidate_d_is_not_run_after_eligibility_stop() -> None:
    result = wl152.write_audit()
    assert result["verdict"] == "INELIGIBLE_CARRIER"
    assert (wl152.TEMP_ROOT / "carrier_verdict.md").exists()
    report = json.loads((wl152.TEMP_ROOT / "visible_surface_carrier_contract_audit_report.json").read_text(encoding="utf-8"))
    assert report["15. ARCHITECTURE VERDICT"] == "INELIGIBLE_CARRIER"
    assert report["10. CONDITIONAL CANDIDATE D"]["status"] == "NOT_RUN"
    assert report["INTENT ALIGNMENT"]["event_1527_removed"] is False
    assert report["IMPLEMENTATION FIDELITY"]["canonical_source_modified"] is False
