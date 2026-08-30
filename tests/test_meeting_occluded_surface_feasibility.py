"""Focused contract tests for the isolated Worklog 134 meeting demo."""

from __future__ import annotations

import inspect

import numpy as np

from devtools.demo.meeting_occluded_surface_feasibility import (
    Box,
    FreeSpaceProxy,
    Prediction,
    build_fixed_holdout,
    build_self_continuation,
    evaluate_controlled_case,
    validate_branch,
)


def _plane_points(*, z_shift: float = 0.0) -> np.ndarray:
    u = np.linspace(0.0, 1.0, 13)
    v = np.linspace(0.0, 1.0, 11)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.column_stack([uu.ravel(), vv.ravel(), np.full(uu.size, z_shift)])


def _holdout(points: np.ndarray):
    return build_fixed_holdout(
        points,
        name="synthetic_boundary_holdout",
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        n_axis=(0.0, 0.0, 1.0),
        u_bounds=(0.0, 1.0),
        v_bounds=(0.0, 1.0),
        n_bounds=(-1.0, 1.0),
        u_cut=0.5,
        permitted_volume=Box((0.5, -0.1, -0.1), (1.05, 1.1, 0.1)),
    )


def _prediction(points_grid: np.ndarray) -> Prediction:
    normals_grid = np.zeros_like(points_grid)
    normals_grid[..., 2] = 1.0
    frontier = points_grid[0]
    return Prediction(
        name="synthetic",
        status="VALID",
        points_grid=points_grid,
        normals_grid=normals_grid,
        frontier_points=frontier,
        frontier_tangent=np.array([0.0, 1.0, 0.0]),
        continuation_direction=np.array([1.0, 0.0, 0.0]),
        l_values=np.linspace(0.0, 0.5, points_grid.shape[0]),
        branch_diagnostics={},
    )


def test_holdout_is_deterministic_and_boundary_attached() -> None:
    holdout = _holdout(_plane_points())
    repeated = _holdout(_plane_points())
    assert holdout.as_json()["boundary_attached"] is True
    assert holdout.as_json()["interior_hole_only"] is False
    assert np.array_equal(holdout.retained_mask, repeated.retained_mask)
    assert np.array_equal(holdout.withheld_mask, repeated.withheld_mask)
    assert np.max(holdout.retained_points[:, 0]) <= 0.5
    assert np.min(holdout.withheld_points[:, 0]) > 0.5


def test_prediction_does_not_depend_on_withheld_xyz() -> None:
    retained = _plane_points()[_plane_points()[:, 0] <= 0.5]
    first = _holdout(np.vstack([retained, _plane_points(z_shift=0.31)[_plane_points(z_shift=0.31)[:, 0] > 0.5]]))
    second = _holdout(np.vstack([retained, _plane_points(z_shift=-0.47)[_plane_points(z_shift=-0.47)[:, 0] > 0.5]]))
    prediction_first = build_self_continuation(first, frontier_bins=8, continuation_samples=9)
    prediction_second = build_self_continuation(second, frontier_bins=8, continuation_samples=9)
    assert np.array_equal(first.retained_points, second.retained_points)
    assert np.allclose(prediction_first.points, prediction_second.points)
    assert np.allclose(prediction_first.normals, prediction_second.normals)
    assert "holdout.withheld_points" not in inspect.getsource(build_self_continuation)
    assert "evaluate_controlled_case" not in inspect.getsource(build_self_continuation)


def test_evaluation_uses_withheld_region_only_and_does_not_rebuild_prediction() -> None:
    first = _holdout(_plane_points())
    second = _holdout(np.vstack([first.retained_points, _plane_points(z_shift=0.2)[_plane_points(z_shift=0.2)[:, 0] > 0.5]]))
    prediction = build_self_continuation(first, frontier_bins=8, continuation_samples=9)
    before = prediction.points.copy()
    first_metrics = evaluate_controlled_case(first, prediction, h=0.01)
    second_metrics = evaluate_controlled_case(second, prediction, h=0.01)
    assert np.array_equal(before, prediction.points)
    assert first_metrics["evaluation_population"] == "held-out reconstructed visible-surface reference only"
    assert first_metrics["source_reference_used_for_prediction"] is False
    assert first_metrics["metric_fed_back_into_construction"] is False
    assert second_metrics["point_to_surface"]["median_over_h"] != first_metrics["point_to_surface"]["median_over_h"]


def test_free_space_proxy_veto_and_volume_containment() -> None:
    grid = np.array([
        [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]],
        [[0.7, 0.0, 0.0], [0.7, 1.0, 0.0]],
    ])
    prediction = _prediction(grid)
    source_frontier = grid[0]
    free = FreeSpaceProxy("synthetic free-space", np.array([[0.7, 0.0, 0.0]]), radius=1e-6)
    diagnostics = validate_branch(
        prediction,
        permitted_volume=Box((0.5, -0.1, -0.1), (0.8, 1.1, 0.1)),
        free_space=free,
        source_normal=np.array([0.0, 0.0, 1.0]),
        source_frontier=source_frontier,
        u_axis=np.array([1.0, 0.0, 0.0]),
    )
    assert diagnostics["known_free_space_violation_point_count"] > 0
    assert diagnostics["valid"] is False


def test_construction_source_is_reference_free_and_h2_selection_contract_is_explicit() -> None:
    source = inspect.getsource(build_self_continuation)
    validator = inspect.getsource(validate_branch)
    assert "full_points" not in source
    assert "withheld_points" not in source
    assert "full_points" not in validator
    assert "withheld_points" not in validator
    assert "second_order" in source
    assert "free_space" in validator
