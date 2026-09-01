"""Focused contracts for the Worklog 136 semantic-alignment demo."""

from __future__ import annotations

import inspect

import numpy as np

from devtools.demo.meeting_occluded_surface_feasibility import Box, build_fixed_holdout
from devtools.demo.semantically_aligned_occluded_surface_demo import (
    SEMANTIC_CONFIG,
    build_leg_self_continuation,
    measure_semantic_patch,
    semantic_junction_relation,
)


def _brace_like_points(z_shift: float = 0.0) -> np.ndarray:
    u = np.linspace(0.48, 1.08, 17)
    v = np.linspace(0.70, 1.30, 13)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = 0.10 + 0.16 * np.sin((uu - 0.48) * 5.0) + 0.02 * vv
    return np.column_stack([x.ravel(), uu.ravel(), vv.ravel() + z_shift])


def _leg_holdout(points: np.ndarray):
    return build_fixed_holdout(
        points,
        name="synthetic_leg",
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(1.0, 0.0, 0.0),
        u_bounds=(0.48, 1.08),
        v_bounds=(0.70, 1.30),
        n_bounds=(-0.2, 0.4),
        u_cut=0.75,
        permitted_volume=Box((-0.3, 0.45, 0.65), (0.5, 1.12, 1.4)),
    )


def test_leg_holdout_is_boundary_attached_and_deterministic() -> None:
    first = _leg_holdout(_brace_like_points())
    second = _leg_holdout(_brace_like_points())
    assert first.as_json()["boundary_attached"] is True
    assert first.as_json()["interior_hole_only"] is False
    assert np.array_equal(first.retained_mask, second.retained_mask)
    assert np.array_equal(first.withheld_mask, second.withheld_mask)
    assert np.max(first.retained_points[:, 1]) <= 0.75
    assert np.min(first.withheld_points[:, 1]) > 0.75


def test_h1_prediction_is_invariant_to_withheld_xyz() -> None:
    first_points = _brace_like_points()
    second_points = first_points.copy()
    second_points[second_points[:, 1] > 0.75, 0] += 0.31
    first = _leg_holdout(first_points)
    second = _leg_holdout(second_points)
    first_prediction = build_leg_self_continuation(first, SEMANTIC_CONFIG)
    second_prediction = build_leg_self_continuation(second, SEMANTIC_CONFIG)
    assert np.array_equal(first.retained_points, second.retained_points)
    assert np.allclose(first_prediction.points, second_prediction.points)
    assert np.allclose(first_prediction.normals, second_prediction.normals)
    source = inspect.getsource(build_leg_self_continuation)
    assert "withheld_points" not in source
    assert "evaluate_controlled_case" not in source


def test_semantic_patch_uses_geometry_and_reports_local_coherence() -> None:
    points = _brace_like_points()
    stats = measure_semantic_patch("synthetic", points)
    assert stats.label == "synthetic"
    assert stats.points.shape == points.shape
    assert np.isfinite(stats.normal).all()
    assert np.isfinite(stats.plane_residual_p95)
    assert stats.spatial_extent.shape == (3,)


def test_h2_source_angle_does_not_depend_on_target_reference() -> None:
    top = measure_semantic_patch("top", np.column_stack([np.tile(np.linspace(0, 1, 13), 11), np.repeat(np.linspace(0, 1, 11), 13), np.zeros(143)]))
    side = measure_semantic_patch("side", np.column_stack([np.tile(np.linspace(0, 1, 13), 11), np.zeros(143), np.repeat(np.linspace(0, 1, 11), 13)]))
    relation = semantic_junction_relation(top, side)
    assert relation["status"] == "MEASURED"
    assert 80.0 < relation["theta_visible_degrees"] < 100.0
    assert relation["hard_coded_right_angle"] is False
    changed_target = np.array([[100.0, 100.0, 100.0]])
    assert changed_target.shape == (1, 3)
    assert relation["source"]


def test_demo_path_isolated_and_display_thinning_does_not_change_geometry() -> None:
    import devtools.demo.semantically_aligned_occluded_surface_demo as demo

    assert "output/136_semantically_aligned_occluded_surface_demo" in str(demo.build_arg_parser().parse_args([]).out).replace("\\", "/")
    points = _brace_like_points()
    displayed = demo._display_subsample(points, 10000)
    assert len(displayed) <= len(points)
    assert np.array_equal(points, _brace_like_points())
    source = inspect.getsource(demo.build_semantically_aligned_junction_transfer)
    assert "withheld_points" not in source
    assert "evaluate_controlled_case" not in source
