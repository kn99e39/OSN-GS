from pathlib import Path

from devtools.demo.worklog_177_intrinsic_structural_dependency_audit import CLASSIFICATIONS, build_report


ROOT = Path(__file__).resolve().parents[1]


def test_clean_composition_gate_stops_before_candidate_c():
    report = build_report(ROOT)

    assert report["verdict"] == "NO_CLEAN_NORMAL_SUBSTITUTION"
    assert report["clean_composition_gate"]["passed"] is False
    assert report["candidate_c"]["status"] == "NOT_IMPLEMENTED_CLEAN_GATE_FAILED"
    assert report["candidate_c"]["separate_module_created"] is False
    assert report["production_behavior_modified"] is False


def test_w154_w97_and_w150_paths_are_distinct_and_baseline_b_is_not_comparable():
    report = build_report(ROOT)

    assert report["baseline_a_w97"]["exact_call_lines"] == [449]
    assert report["baseline_b_w150"]["form_surface_regions_call_lines"] == [207]
    assert report["baseline_b_w150"]["status"] == "NOT_COMPARABLE"
    assert report["source_assertions"]["w154_w150_call_detected"] == []


def test_dependency_audit_covers_requested_quantities_and_exact_blockers():
    report = build_report(ROOT)
    quantities = {row["quantity"] for row in report["dependency_audit"]}
    assert quantities == {
        "normal",
        "tangent axes",
        "spatial relation",
        "tangent-plane displacement",
        "normal separation",
        "reliability",
        "affinity",
        "parallel conflict",
        "shared-neighbor consensus",
        "seed classification",
        "component-pair support",
        "bridge veto",
        "path/tangent transport",
        "ambiguity",
    }
    assert all(row["classification"] in CLASSIFICATIONS for row in report["dependency_audit"])
    assert report["source_assertions"]["frame_has_covariance_tangent_and_scale_fields"] is True
    assert report["source_assertions"]["reliability_uses_frame_scales"] is True
    assert report["source_assertions"]["affinity_uses_frame_scales"] is True
    assert "parallel conflict" in report["clean_composition_gate"]["blocking_dependencies"][3] or report["clean_composition_gate"]["blocking_dependencies"]


def test_historical_witness_and_tabletop_are_baseline_only():
    report = build_report(ROOT)

    witness = report["w174_witness"]
    assert witness["row_4043"]["stable_gaussian_id"] == 4937175
    assert witness["row_4051"]["stable_gaussian_id"] == 3929355
    assert witness["row_4043"]["path_subset_crossing"] == 0
    assert witness["row_4051"]["path_subset_crossing"] == 0
    assert report["tabletop_fixed_aabb"]["support_subset_count"] == 1
    assert report["tabletop_fixed_aabb"]["candidate_c"] == "NOT_RUN_CLEAN_GATE_FAILED"
    assert report["zero_set_projection"]["status"] == "NOT_RUN_NO_CANDIDATE"


def test_visualization_is_not_claimed_when_gate_fails():
    report = build_report(ROOT)

    assert report["visualization"]["status"] == "NOT_GENERATED_CLEAN_GATE_FAILED"
    assert report["visualization"]["tsdf_component_coloring"] is False
    assert report["historical_lineages"]["curved_real_positive_control"] == (
        "REAL_CURVED_POSITIVE_CONTROL_UNAVAILABLE (W171 curved/vase invalid; synthetic historical controls only)"
    )
