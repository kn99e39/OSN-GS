"""Focused WL150 source/history and frozen-baseline contract tests."""

from __future__ import annotations

import json

from devtools.demo import boundary_first_local_surface_decomposition_contract_trace as wl150


def test_exact_wl149_baseline_reconciliation() -> None:
    baseline = wl150._load_wl149_baseline()
    assert baseline["event_count"] == 1586
    assert baseline["event_union_sha256"] == "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78"
    assert baseline["camera_counts"] == {"DSC08043.JPG": 754, "DSC07960.JPG": 330, "DSC08003.JPG": 502}
    assert baseline["representative_shape"] == [3840, 3]
    assert baseline["support_vertices"] == 314
    assert baseline["fully_supported_cells"] == 211
    assert baseline["all_event_ids_exact"] is True
    assert baseline["preservation_assertions"]["event_1527_preserved"] is True


def test_event_1527_trace_is_deterministic_and_human_scoped() -> None:
    first = wl150.build_trace()["event_1527_trace"]
    second = wl150.build_trace()["event_1527_trace"]
    assert json.dumps(first, sort_keys=True, default=wl150._json_default) == json.dumps(second, sort_keys=True, default=wl150._json_default)
    assert first["event_id"] == 1527
    assert first["source_camera"] == "DSC08003.JPG"
    assert first["source_pixel"] == [259, 169]
    assert first["pca_coordinates"]["v"] == -0.5984138975738875
    assert first["v_min_ownership"]["is_owner"] is True
    assert first["human_review_fact"] == "HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE"
    assert "No general rejection" in first["other_sparse_events"]


def test_canonical_contract_and_actual_bypass_are_distinguishable() -> None:
    intended = wl150._intended_boundary_first_contract()
    local = wl150._intended_local_surface_decomposition_contract()
    actual = wl150._actual_dataflow(wl150._load_wl149_baseline(), wl150._load_event_1527(wl150._load_wl149_baseline()))
    constructor = intended["source_evidence"][0]
    region = local["source_evidence"][0]
    assert constructor["status"] == "PRESENT"
    assert "form_surface_regions" in constructor["calls"]
    assert "materialize_visible_boundary_component" in constructor["calls"]
    assert region["status"] == "PRESENT"
    assert actual["source_evidence"]["union_and_pca"]["status"] == "PRESENT"
    assert actual["source_evidence"]["union_and_pca"]["matched_source_lines"]
    assert actual["source_evidence"]["wl139_fitter"]["status"] == "PRESENT"
    assert all(stage.get("region_ids_before_union", "ABSENT") == "ABSENT" for stage in actual["stages"][:2])


def test_contract_path_export_is_deterministic_and_non_repairing() -> None:
    result = wl150.write_trace()
    assert result["verdict"] == "B. ARCHITECTURE_BYPASS"
    assert "event_1527_trace.json" in result["files"]
    assert (wl150.TEMP_ROOT / "event_1527_trace.json").exists()
    report = json.loads((wl150.TEMP_ROOT / "boundary_first_local_surface_decomposition_contract_trace_report.json").read_text(encoding="utf-8"))
    assert report["11. ARCHITECTURE VERDICT"] == "B. ARCHITECTURE_BYPASS"
    assert report["7. PRE-FIT VS POST-FIT DOMAIN CONTROL"]["required_statement"] == "WL148 B does NOT by itself restore Boundary First semantics."
    assert report["IMPLEMENTATION FIDELITY"]["canonical_source_modified"] is False
    assert report["IMPLEMENTATION FIDELITY"]["new_heuristic_added"] is False
