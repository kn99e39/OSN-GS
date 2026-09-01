"""Focused contracts for Worklog 139's isolated physical-chart graph fit."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import devtools.demo.physical_chart_surface_representative as demo


def _graph_points(*, include_withheld: bool = False, second_mode: bool = False) -> np.ndarray:
    config = demo.CURVED_RIM_CASE
    u1 = config.u_bounds[1] if include_withheld else config.u_cut
    u = np.linspace(config.u_bounds[0], u1, 70)
    v = np.linspace(config.v_bounds[0], config.v_bounds[1], 52)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    n = 1.34 + 0.06 * np.sin(2.2 * (uu - config.u_bounds[0])) + 0.025 * (vv - config.v_bounds[0])
    axis_u = demo._normalised_axis(config.u_axis)
    axis_v = demo._normalised_axis(config.v_axis)
    axis_n = demo._normalised_axis(config.n_axis)
    points = uu[..., None] * axis_u + vv[..., None] * axis_v + n[..., None] * axis_n
    flat = points.reshape(-1, 3)
    if second_mode:
        flat = np.concatenate([flat, flat + 0.08 * axis_n[None, :]], axis=0)
    return flat


def test_wl138_module_and_historical_output_path_remain_untouched() -> None:
    before = demo._file_sha256(demo.WL138_MODULE)
    _ = demo.audit_raw_graphness(_graph_points(), demo.CURVED_RIM_CASE, h=0.01)
    after = demo._file_sha256(demo.WL138_MODULE)
    assert before == after
    source = inspect.getsource(demo.run_demo)
    assert "output/scale_separated_visible_surface_representative" not in source
    assert demo.OUTPUT_ROOT.name == "139_physical_chart_surface_representative"


def test_graphness_uses_retained_points_and_detects_separated_modes() -> None:
    graph = demo.audit_raw_graphness(_graph_points(), demo.CURVED_RIM_CASE, h=0.01)
    multimode = demo.audit_raw_graphness(_graph_points(second_mode=True), demo.CURVED_RIM_CASE, h=0.01)
    assert graph.status == "PASS_GRAPH_LIKE"
    assert graph.multimode_bins == 0
    assert multimode.status == "FAIL_MATERIALLY_MULTIVALUED"
    assert multimode.multimode_fraction > demo.GRAPH_MAX_MULTIMODE_FRACTION
    assert "withheld" not in inspect.signature(demo.audit_raw_graphness).parameters


def test_retained_graph_fit_receives_only_explicit_rows_and_is_deterministic() -> None:
    points = _graph_points()
    first = demo.fit_physical_chart_surface(
        points,
        demo.CURVED_RIM_CASE,
        role="retained_construction",
        max_fit_points=1800,
        device_name="cpu",
    )
    second = demo.fit_physical_chart_surface(
        points,
        demo.CURVED_RIM_CASE,
        role="retained_construction",
        max_fit_points=1800,
        device_name="cpu",
    )
    expected = points[demo.deterministic_indices(len(points), 1800)]
    assert first.fit_input_sha256 == demo._sha256_rows(expected)
    assert np.array_equal(first.scalar_control_grid, second.scalar_control_grid)
    source = inspect.getsource(demo.fit_physical_chart_surface)
    assert "withheld" not in source
    assert "evaluate_prediction" not in source


def test_physical_uv_chart_is_fixed_monotonic_and_orientation_preserving() -> None:
    representative = demo.fit_physical_chart_surface(
        _graph_points(),
        demo.CURVED_RIM_CASE,
        role="retained_construction",
        max_fit_points=1800,
        device_name="cpu",
    )
    assert np.max(representative.physical_u_precision_error) < 1.0e-5
    assert np.max(representative.physical_v_precision_error) < 1.0e-5
    topology = demo.audit_representative_topology(
        representative.surface,
        demo.CURVED_RIM_CASE,
        domain_u=representative.domain_u,
        domain_v=representative.domain_v,
        h=0.01,
    )
    assert topology["physical_u_reversal_count"] == 0
    assert topology["physical_v_reversal_count"] == 0
    assert topology["jacobian_orientation_flip_count"] == 0
    assert topology["duplicate_multivalued_chart_bins"] == 0
    assert topology["topology_contract_valid"] is True


def test_full_representative_is_evaluation_only_and_rejected_by_continuation() -> None:
    full = demo.fit_physical_chart_surface(
        _graph_points(include_withheld=True),
        demo.CURVED_RIM_CASE,
        role="full_evaluation_only",
        max_fit_points=1800,
        device_name="cpu",
    )
    assert full.role == "full_evaluation_only"
    with pytest.raises(AssertionError, match="retained_construction"):
        demo.build_chart_continuation(None, full)  # type: ignore[arg-type]


def test_holdout_selection_uses_world_physical_coordinate_not_parameter_u() -> None:
    points = _graph_points(include_withheld=True)
    mask = demo.select_physical_heldout_samples(points, demo.CURVED_RIM_CASE)
    physical_u = demo._case_coordinates(points, demo.CURVED_RIM_CASE)[:, 0]
    assert np.array_equal(mask, physical_u > demo.CURVED_RIM_CASE.u_cut + 1.0e-8)
    source = inspect.getsource(demo.select_physical_heldout_samples)
    assert "_case_coordinates" in source
    assert "uv" not in inspect.signature(demo.select_physical_heldout_samples).parameters
    assert "physical_u > float(config.u_cut)" in source


def test_continuation_accepts_retained_representative_only_and_has_no_metric_feedback() -> None:
    source = inspect.getsource(demo.build_chart_continuation)
    assert "retained_construction" in source
    assert "evaluate_prediction" not in source
    assert "withheld" not in source
    assert "u_bounds[1] - holdout.u_cut" in source
    assert "second" not in source.lower()


def test_fixed_configuration_matches_wl138_without_uv_correction() -> None:
    assert demo.GRAPH_RESOLUTION_U == demo.FIXED_FITTER_CONFIG["resolution_u"] == 8
    assert demo.GRAPH_RESOLUTION_V == demo.FIXED_FITTER_CONFIG["resolution_v"] == 4
    assert demo.GRAPH_DEGREE_U == demo.FIXED_FITTER_CONFIG["degree_u"] == 2
    assert demo.GRAPH_DEGREE_V == demo.FIXED_FITTER_CONFIG["degree_v"] == 2
    assert demo.GRAPH_SMOOTHNESS_LAMBDA == demo.FIXED_FITTER_CONFIG["smoothness_lambda"] == 1.0e-4
    assert demo.GRAPH_TIKHONOV_LAMBDA == demo.FIXED_FITTER_CONFIG["tikhonov_lambda"] == 1.0e-4
    source = inspect.getsource(demo.fit_physical_chart_surface)
    assert "project_torch_points_to_nurbs" not in source


def test_raw_visualization_is_near_opaque_and_display_thinning_is_isolated() -> None:
    assert 0.95 <= demo.DISPLAY_POINT_ALPHA <= 1.0
    assert 0.95 <= demo.DISPLAY_REFERENCE_ALPHA <= 1.0
    assert demo.DISPLAY_POINT_SIZE >= 3.0
    assert demo.DISPLAY_VOXEL_WORLD not in {
        demo.GRAPH_SMOOTHNESS_LAMBDA,
        demo.GRAPH_TIKHONOV_LAMBDA,
        demo.GRAPH_BIN_SCALE_H,
    }


def test_confirmed_wl138_loader_is_exact_and_read_only() -> None:
    if not demo.WL138_CONFIRMED_ROOT.exists():
        pytest.skip("confirmed WL138 artifact is not available")
    before = demo._artifact_manifest(demo.WL138_CONFIRMED_ROOT)
    frozen = demo._load_frozen_wl138_case(
        demo.WL138_CONFIRMED_ROOT,
        demo.CURVED_RIM_CASE,
        max_fit_points=12000,
        device_name="cpu",
    )
    after = demo._artifact_manifest(demo.WL138_CONFIRMED_ROOT)
    assert before == after
    assert len(frozen["retained"]) == 13443
    assert len(frozen["withheld"]) == 10557
    assert frozen["baseline"].fit_input_sha256 == frozen["historical_stats"]["fit_input_sha256"]
    assert max(frozen["replay_validation"].values()) < 1.0e-4


def test_macro_reference_validity_is_symmetric_and_true_gate_requires_candidate_b() -> None:
    macro_source = inspect.getsource(demo._full_macro_reference_contract)
    assert 'raw_to_macro["p95_over_h"] <= 12.0' in macro_source
    assert 'macro_to_raw["p95_over_h"] <= 12.0' in macro_source
    run_source = inspect.getsource(demo.run_demo)
    assert "CANDIDATE_B_ARCHIVE.exists()" in run_source
    assert "NOT_EXECUTED_CANDIDATE_B_ARCHIVE_UNAVAILABLE" in run_source
