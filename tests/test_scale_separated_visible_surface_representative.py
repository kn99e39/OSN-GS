"""Focused contracts for Worklog 138's isolated representative audit."""

from __future__ import annotations

import inspect

import numpy as np

import devtools.demo.scale_separated_visible_surface_representative as demo


def _leg_grid() -> np.ndarray:
    y = np.linspace(0.48, 1.08, 13)
    z = np.linspace(0.70, 1.30, 11)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    x = 0.08 + 0.08 * np.sin((yy - 0.48) * 5.0)
    return np.column_stack([x.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def test_worklog_138_holdout_is_boundary_attached_and_matches_wl136() -> None:
    holdout = demo._build_case_holdout(_leg_grid(), demo.LEG_CASE)
    assert holdout.as_json()["boundary_attached"] is True
    assert holdout.as_json()["interior_hole_only"] is False
    assert holdout.u_cut == demo.WL136_SEMANTIC_CONFIG.leg_u_cut
    assert np.max(holdout.retained_points @ holdout.u_axis) <= holdout.u_cut
    assert np.min(holdout.withheld_points @ holdout.u_axis) > holdout.u_cut


def test_retained_fit_receives_only_explicit_retained_points(monkeypatch) -> None:
    import torch

    captured: list[np.ndarray] = []

    def fake_fitter(points, **kwargs):
        captured.append(points.detach().cpu().numpy().copy())
        control = torch.zeros((8, 4, 3), dtype=torch.float32, device=points.device)
        control[:, :, 0] = torch.linspace(-0.1, 0.1, 8, device=points.device)[:, None]
        control[:, :, 1] = 0.7
        control[:, :, 2] = 1.0
        surface = demo.TorchNURBSSurface(
            control_grid=control,
            weights=torch.ones((8, 4), dtype=torch.float32, device=points.device),
            degree_u=2,
            degree_v=2,
            observed_v_max=1.0,
        )
        return surface, kwargs["initial_uv"]

    monkeypatch.setattr(demo, "fit_torch_visible_surface_lsq", fake_fitter)
    points = _leg_grid()[:32]
    representative = demo.fit_existing_nurbs(
        points,
        demo.LEG_CASE,
        h=0.01,
        retained_domain=True,
        max_fit_points=1000,
        device_name="cpu",
    )
    assert len(captured) == 1
    assert np.allclose(captured[0], points)
    assert representative.fit_input_sha256 == demo._sha256_rows(points)
    source = inspect.getsource(demo.fit_existing_nurbs)
    assert "withheld" not in source


def test_full_and_retained_representatives_share_one_frozen_configuration() -> None:
    assert demo.FIXED_FITTER_CONFIG == {
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
    source = inspect.getsource(demo.run_demo)
    assert "fit_existing_nurbs(full" in source
    assert "fit_existing_nurbs(retained" in source


def test_continuation_maps_physical_frontier_and_never_uses_uv_edge() -> None:
    source = inspect.getsource(demo._representative_frontier_continuation)
    assert "frozen.frontier_points" in source
    assert "project_torch_points_to_nurbs" in source
    assert "control_grid[-1]" not in source
    assert "frozen physical" in source
    assert "NURBS UV edge" in inspect.getsource(demo._frontier_mapping_report)


def test_visualization_is_near_opaque_and_display_thinning_is_not_metric_thinning() -> None:
    assert demo.DISPLAY_POINT_ALPHA >= 0.95
    assert demo.DISPLAY_REFERENCE_ALPHA >= 0.95
    assert demo.DISPLAY_POINT_SIZE >= 2.0
    assert demo.DISPLAY_VOXEL_WORLD == 0.02
    assert demo.DISPLAY_VOXEL_WORLD not in demo.FIXED_FITTER_CONFIG.values()


def test_continuation_assessment_is_post_evaluation_only() -> None:
    source = inspect.getsource(demo._continuation_assessment)
    assert "used_to_construct_or_tune_prediction" in source
    assert "continuation.points_grid" not in source
