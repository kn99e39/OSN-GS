"""Focused contracts for Worklog 131 explicit geometric termination."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from devtools.demo.explicit_geometric_termination_continuation import (
    ROOT_RESIDUAL_TOLERANCE,
    TerminationCurve,
    build_explicit_prediction,
    physical_first_order_direction,
    solve_plane_intersections,
    termination_plane,
)
from devtools.demo.parametric_surface_continuation import PRIMARY_ROI
from osn_gs.surface.torch_nurbs import TorchNURBSSurface


def _plane_surface() -> TorchNURBSSurface:
    u = torch.linspace(0.0, 1.0, 2)
    v = torch.linspace(0.0, 1.0, 2)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    x = PRIMARY_ROI.u_bounds[0] + (PRIMARY_ROI.u_bounds[1] - PRIMARY_ROI.u_bounds[0]) * uu
    z = PRIMARY_ROI.v_bounds[0] + (PRIMARY_ROI.v_bounds[1] - PRIMARY_ROI.v_bounds[0]) * vv
    y = torch.full_like(x, 1.35)
    grid = torch.stack([x, y, z], dim=-1)
    return TorchNURBSSurface(grid, torch.ones((2, 2)), degree_u=1, degree_v=1)


def test_termination_plane_uses_only_fixed_roi_contract():
    plane = termination_plane(PRIMARY_ROI)
    expected = PRIMARY_ROI.u_bounds[0] + PRIMARY_ROI.holdout_u_cut * (PRIMARY_ROI.u_bounds[1] - PRIMARY_ROI.u_bounds[0])
    assert plane["local_u_world_coordinate"] == expected
    assert plane["withheld_xyz_used"] is False
    assert plane["source"].startswith("frozen ROI")


def test_plane_intersection_recovers_known_gamma_without_target_input():
    surface = _plane_surface()
    plane = termination_plane(PRIMARY_ROI)
    rows, v_values = solve_plane_intersections(surface, plane, samples_v=9)
    expected_u = PRIMARY_ROI.holdout_u_cut
    assert len(rows) == 9
    assert np.allclose(v_values, np.linspace(0.0, 1.0, 9))
    assert all(len(row.roots_u) == 1 for row in rows)
    assert all(abs(row.selected_u - expected_u) < 1e-5 for row in rows)
    assert all(residual <= ROOT_RESIDUAL_TOLERANCE for row in rows for residual in row.root_residuals)


def test_physical_direction_normalizes_local_u_derivative_to_one():
    surface = _plane_surface()
    plane = termination_plane(PRIMARY_ROI)
    uv = np.asarray([[PRIMARY_ROI.holdout_u_cut, 0.0], [PRIMARY_ROI.holdout_u_cut, 0.5], [PRIMARY_ROI.holdout_u_cut, 1.0]])
    _points, _duv, directions, local_derivative = physical_first_order_direction(surface, uv, plane)
    assert np.all(np.isfinite(directions))
    assert np.allclose(local_derivative, 1.0, atol=1e-5)
    assert np.all(directions[:, 0] > 0.0)


def test_explicit_prediction_is_fixed_first_order_and_uses_frozen_horizon():
    v = np.linspace(0.0, 1.0, 5)
    l_max = (PRIMARY_ROI.u_bounds[1] - PRIMARY_ROI.u_bounds[0]) * (1.0 - PRIMARY_ROI.holdout_u_cut)
    curve = TerminationCurve(
        gamma_uv=np.column_stack([np.full(5, PRIMARY_ROI.holdout_u_cut), v]),
        gamma_points=np.column_stack([np.full(5, -5.498), np.full(5, 1.35), np.linspace(3.0, 4.2, 5)]),
        derivatives_uv=np.zeros((5, 2)),
        physical_direction=np.tile(np.asarray([[1.0, 0.0, 0.0]]), (5, 1)),
        local_u_derivative=np.ones(5),
        support_fit_distance=np.zeros(5),
        support_mesh_distance=np.zeros(5),
        root_rows=[],
        valid_mask=np.ones(5, dtype=bool),
    )
    prediction = build_explicit_prediction(curve, PRIMARY_ROI, 0.012)
    assert prediction.points.shape == (96 * 5, 3)
    assert prediction.l_values[-1] == l_max
    assert np.isclose(np.ptp(prediction.points[:, 0]), l_max)
    assert np.allclose(prediction.normals[:, 0], 0.0, atol=1e-6)


def test_no_second_order_geometry_or_target_selected_root_is_present():
    module_source = Path("devtools/demo/explicit_geometric_termination_continuation.py").read_text(encoding="utf-8")
    assert "evaluate_with_second_derivatives" not in module_source
    assert "S_uu" not in module_source
    assert list(inspect.signature(solve_plane_intersections).parameters) == ["surface", "plane", "samples_v"]


def test_historical_inputs_are_referenced_as_read_only_artifacts():
    assert Path("devtools/demo/parametric_surface_continuation.py").exists()
    assert Path("devtools/demo/corrected_first_order_parametric_continuation.py").exists()
    assert Path("devtools/demo/parametric_continuation_attribution.py").exists()
    assert "GEOMETRIC_TERMINATION_CURVE" in Path("devtools/demo/explicit_geometric_termination_continuation.py").read_text(encoding="utf-8")
