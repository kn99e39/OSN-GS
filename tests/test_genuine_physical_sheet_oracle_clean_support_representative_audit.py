"""Focused WL145 contract tests; no renderer or CUDA execution is required."""

from __future__ import annotations

import numpy as np

from devtools.demo import genuine_physical_sheet_oracle_clean_support_representative_audit as wl145


def test_manual_controls_are_deterministic_and_use_three_source_views() -> None:
    first = [control.as_json() for control in wl145._manual_controls()]
    second = [control.as_json() for control in wl145._manual_controls()]
    assert first == second
    assert all(len(control["masks"]) == 3 for control in first)
    assert all(
        {mask["camera_name"] for mask in control["masks"]} == set(wl145.CAMERAS)
        for control in first
    )
    assert all(control["historical_wl141_masks_used"] is False for control in first)


def test_post_validation_chart_is_deterministic_and_not_membership_selection() -> None:
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.1],
            [1.0, 2.0, -0.1],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    config_a, provenance_a = wl145._pca_chart_config(points, "case", "sheet")
    config_b, provenance_b = wl145._pca_chart_config(points, "case", "sheet")
    assert config_a.as_json() == config_b.as_json()
    np.testing.assert_allclose(provenance_a["axes_columns_world_xyz"], provenance_b["axes_columns_world_xyz"])
    assert provenance_a["used_for_oracle_membership"] is False
    assert provenance_a["full_reference_xyz_used"] is False


def test_frozen_wl139_settings_and_fixed_scales_are_not_replaced_by_demo_values() -> None:
    assert wl145.FIXED_FITTER_CONFIG == {
        "resolution_u": 8,
        "resolution_v": 4,
        "degree_u": 2,
        "degree_v": 2,
        "smoothness_lambda": 1.0e-4,
        "tikhonov_lambda": 1.0e-4,
        "correction_rounds": 2,
        "chunk_size": 8192,
        "projection_iterations": 2,
    }
    assert wl145.DISPLAY_MAX_POINTS == 20000
    assert all(control.review_classification in {
        "CLEAR_PHYSICAL_SHEET_ORACLE",
        "PARTIAL / MIXED",
        "DIFFERENT_SURFACES",
        "AMBIGUOUS",
        "INSUFFICIENT_EVIDENCE",
    } for control in wl145._manual_controls())


def test_reported_provenance_identifies_control_and_renderer_depth() -> None:
    mask = wl145.ManualCameraMask("DSC08043.JPG", ((1, 1), (4, 1), (4, 4)), "test")
    cloud = wl145.PerViewEventCloud(
        camera_name="DSC08043.JPG",
        polygon=mask,
        pixel_x=np.array([1, 2], dtype=np.int64),
        pixel_y=np.array([1, 2], dtype=np.int64),
        median_depth=np.array([1.0, 1.1], dtype=np.float64),
        points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float64),
        local_normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    )
    summary = wl145._event_cloud_summary(cloud, mask, "control_id")
    assert summary["provenance"]["physical_sheet_control_id"] == "control_id"
    assert summary["provenance"]["renderer_depth"] == "median_depth array in the per-view NPZ"
    assert summary["provenance"]["source_camera"] == "DSC08043.JPG"
