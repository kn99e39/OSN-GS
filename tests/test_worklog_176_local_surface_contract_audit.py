from pathlib import Path

from devtools.demo.worklog_176_local_surface_contract_audit import build_report


ROOT = Path(__file__).resolve().parents[1]


def test_worklog_176_reconstructs_split_contract_without_running_control():
    report = build_report(ROOT)

    assert report["architecture_result"] == "CONTRACT_SPLIT_ACROSS_MULTIPLE_HISTORICAL_BRANCHES"
    assert report["production_behavior_modified"] is False
    assert report["conditional_same_checkpoint_control"]["status"] == "NOT_RUN"
    assert report["conditional_same_checkpoint_control"]["checkpoint_preserved"] is True


def test_w154_active_path_is_w97_and_w150_is_not_called():
    report = build_report(ROOT)
    choice = report["why_w154_uses_w97"]

    assert choice["w154_w97_call_lines"] == [449]
    assert choice["w154_form_surface_regions_call_lines"] == []
    assert choice["w150_constructor_call_lines"] == [207]
    assert choice["determination"].startswith("EXPLICIT_PATH_DIFFERENCE")


def test_matrix_covers_requested_fields_and_uses_declared_statuses():
    report = build_report(ROOT)
    matrix = report["decomposition_contract_matrix"]
    expected = {
        "gaussian_normal_source",
        "local_geometric_neighborhood",
        "knn_definition",
        "distance_locality_gate",
        "sign_independent_normal_consistency",
        "positional_continuity",
        "normal_direction_separation",
        "tangent_mutual_tangent_consistency",
        "discontinuity_first_cuts",
        "shared_neighbor_multi_edge_consensus",
        "region_orientation_concentration",
        "interface_consistency",
        "bridge_veto",
        "path_tangent_transport_consistency",
        "singleton_propagation",
        "ambiguity_handling",
        "deterministic_tie_breaking",
    }
    assert {row["candidate"] for row in matrix} == {
        "W96 intrinsic-normal coverage-first",
        "W97 region-coherent intrinsic-normal partition",
        "W98 discontinuity-first intrinsic-normal partition",
        "W99 interface-coherent intrinsic-normal merge",
        "W100 bilateral region-conditioned intrinsic-normal merge",
        "W10/W31-W38/W150 form_surface_regions",
    }
    for row in matrix:
        assert set(row["contract"]) == expected
        assert all(value in {"IMPLEMENTED", "OPTIONAL", "ABSENT", "DIAGNOSTIC_ONLY"} for value in row["contract"].values())


def test_w174_witness_is_baseline_only_when_no_approved_comparison_exists():
    report = build_report(ROOT)
    witness = report["w174_witness_result"]
    assert witness["classification"] == "NOT_COMPARABLE"
    assert witness["w97_baseline"]["row_4043"]["subset_id"] == 1
    assert witness["w97_baseline"]["row_4051"]["subset_id"] == 1
    assert witness["w97_baseline"]["path_subset_boundary_crossing"] == 0


def test_no_visual_comparison_is_claimed_for_a_split_contract():
    report = build_report(ROOT)
    assert report["visual_review"]["status"] == "NOT_GENERATED_NO_COMPARABLE_APPROVED_CONTRACT"
    assert report["source_assertions"]["w154_active_path_detected"] is True
    assert report["source_assertions"]["w175_preserves_w150_w97_distinction"] is True
