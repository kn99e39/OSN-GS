"""Focused contracts for Worklog 140's frozen real-scene evaluation path."""

from __future__ import annotations

import inspect

import numpy as np

import devtools.demo.real_gaussian_scene_surface_validation as demo


class _Camera:
    def __init__(self, name: str, center: tuple[float, float, float]) -> None:
        self.image_name = name
        self.image_width = 100
        self.image_height = 100
        self.world_view_transform = np.eye(4, dtype=np.float64)
        self.full_proj_transform = np.eye(4, dtype=np.float64)
        self.camera_center = np.asarray(center, dtype=np.float64)


def test_review_set_is_fixed_and_contains_required_semantic_classes() -> None:
    regions = demo.frozen_review_regions()
    assert len(regions) == 7
    by_name = {region.config.name: region for region in regions}
    assert by_name["curved_table_rim"].config.roi_box.as_json() != demo.CURVED_RIM_CASE.roi_box.as_json()
    assert by_name["historical_wl139_curved_rim_alignment_control"].config.roi_box.as_json() == demo.CURVED_RIM_CASE.roi_box.as_json()
    assert by_name["wl136_leg_brace"].config.roi_box.as_json() == demo.LEG_CASE.roi_box.as_json()
    classes = {region.semantic_class for region in regions}
    assert "BROAD_PLANAR_SURFACE" in classes
    assert "BROAD_CURVED_SURFACE_POSITIVE_CONTROL" in classes
    assert "BROAD_CURVED_SURFACE_SECOND_CASE" in classes
    assert "THIN_MULTI_SHEET_STRUCTURE" in classes
    assert "COMPLEX_BACKGROUND_FOLIAGE_LIKE_CONTROL" in classes
    assert "HISTORICAL_COORDINATE_ALIGNMENT_CONTROL" in classes
    for region in regions:
        assert region.config.u_cut == region.config.u_bounds[1]


def test_camera_selection_is_deterministic_and_has_no_representative_input() -> None:
    points = np.asarray([[-0.5, -0.5, 0.5], [0.0, 0.0, 0.7], [0.5, 0.5, 0.9]] * 20, dtype=np.float64)
    cameras = [_Camera("cam_b", (0.0, 0.0, 2.0)), _Camera("cam_a", (1.0, 0.0, 2.0)), _Camera("cam_c", (-1.0, 0.0, 2.0))]
    first = demo.select_review_cameras(points, cameras, count=3)
    second = demo.select_review_cameras(points, cameras, count=3)
    assert [item["camera_name"] for item in first] == [item["camera_name"] for item in second]
    assert "representative" not in str(inspect.signature(demo.select_review_cameras)).lower()
    assert "points" in str(inspect.signature(demo.select_review_cameras))


def test_graphness_gate_precedes_fit_and_failed_regions_are_not_forced() -> None:
    source = inspect.getsource(demo.run_validation)
    assert source.index("audit_raw_graphness") < source.index("fit_physical_chart_surface")
    assert "if audit.status == \"PASS_GRAPH_LIKE\":" in source
    assert "OUT_OF_DOMAIN_GRAPHNESS_FAIL" in source
    assert "build_chart_continuation" not in source
    assert "build_self_continuation" not in source


def test_projection_matches_expected_identity_camera_coordinates() -> None:
    camera = _Camera("identity", (0.0, 0.0, 0.0))
    projected = demo.project_world_points(np.asarray([[0.0, 0.0, 0.5], [1.0, -1.0, 0.5], [2.0, 0.0, 0.5]]), camera)
    assert np.array_equal(projected["valid"], np.asarray([True, True, False]))
    assert np.allclose(projected["x"][:2], [49.5, 99.0])
    assert np.allclose(projected["y"][:2], [49.5, 99.0])


def test_visual_display_subsampling_does_not_mutate_metric_population() -> None:
    points = np.arange(300, dtype=np.float64).reshape(-1, 3)
    original = points.copy()
    selected = demo._display_points(points, max_points=20)
    assert len(selected) == 20
    assert np.array_equal(points, original)
    assert demo.REVIEW_POINT_ALPHA >= 0.95
    assert demo.REVIEW_POINT_SIZE >= 3.0


def test_frozen_wl139_settings_are_reused_and_occluded_paths_are_disabled() -> None:
    assert demo.GRAPH_RESOLUTION_U == demo.FIXED_FITTER_CONFIG["resolution_u"] == 8
    assert demo.GRAPH_RESOLUTION_V == demo.FIXED_FITTER_CONFIG["resolution_v"] == 4
    assert demo.GRAPH_DEGREE_U == demo.FIXED_FITTER_CONFIG["degree_u"] == 2
    assert demo.GRAPH_DEGREE_V == demo.FIXED_FITTER_CONFIG["degree_v"] == 2
    source = inspect.getsource(demo.run_validation)
    assert "role=\"full_evaluation_only\"" in source
    assert "build_chart_continuation" not in source
    assert "build_self_continuation" not in source
    assert "Candidate B" not in source
    assert "occluded_surface_executed" in source
