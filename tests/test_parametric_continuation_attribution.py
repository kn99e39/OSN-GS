"""Focused contracts for Worklog 129 attribution-only diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from devtools.demo.corrected_first_order_parametric_continuation import FrozenCase
from devtools.demo.parametric_continuation_attribution import (
    FIT_KWARGS,
    PRIMARY_ROI,
    ReplayedFit,
    _fit_input_and_initial_uv,
    _inversion_count,
    _iter_npy_rows_from_npz,
    _roi_vertex_contract,
    boundary_support_report,
    geometric_interface_report,
    parameterization_report,
    replay_frozen_fit,
    target_coherence_report,
    trace_mesh_interface,
)
from osn_gs.surface.torch_nurbs import TorchNURBSSurface


def _tiny_reference() -> np.ndarray:
    xs = np.asarray([-6.6, -5.3, -4.7], dtype=np.float64)
    zs = np.asarray([3.0, 3.6, 4.2], dtype=np.float64)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    yy = 1.3 + 0.02 * (xx + 6.6) + 0.04 * (zz - 3.0)
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def _tiny_case(points: np.ndarray, grid: np.ndarray) -> FrozenCase:
    partition = __import__("devtools.demo.parametric_surface_continuation", fromlist=["build_holdout_partition"]).build_holdout_partition(points, PRIMARY_ROI)
    observed = points[partition.observed_mask]
    withheld = points[partition.withheld_mask]
    return FrozenCase(
        config=PRIMARY_ROI,
        full_points=points,
        observed_points=observed,
        withheld_points=withheld,
        reference_eval_points=withheld,
        historical_predicted_points=withheld[:1],
        historical_predicted_normals=np.zeros((1, 3), dtype=np.float64),
        historical_distances=np.zeros((len(withheld),), dtype=np.float64),
        control_grid=grid,
        historical_control_grid=grid,
        historical_metrics={"point_to_predicted_surface_distance": {"median_over_h": 0.0, "p95_over_h": 0.0}, "withheld_reference_coverage": {"fraction_le_h": 1.0, "fraction_le_2h": 1.0}},
    )


def test_inversion_count_is_pairwise_and_deterministic():
    count, pairs = _inversion_count(np.asarray([3.0, 1.0, 2.0]))
    assert count == 2
    assert pairs == 3


def test_npy_face_stream_does_not_change_rows(tmp_path):
    path = tmp_path / "mesh.npz"
    vertices = np.arange(30, dtype=np.float64).reshape(10, 3)
    faces = np.arange(27, dtype=np.int64).reshape(9, 3)
    np.savez(path, vertices=vertices, faces=faces)
    chunks = list(_iter_npy_rows_from_npz(path, "faces.npy", rows_per_chunk=4))
    assert [start for start, _rows in chunks] == [0, 4, 8]
    np.testing.assert_array_equal(np.concatenate([rows for _start, rows in chunks]), faces)


def test_frozen_fit_replay_compares_control_grid_and_captures_final_uv(monkeypatch):
    import torch

    points = _tiny_reference()
    grid = torch.zeros((8, 4, 3), dtype=torch.float32)
    grid[:, :, 0] = torch.linspace(-6.6, -4.7, 8)[:, None]
    grid[:, :, 1] = 1.3
    grid[:, :, 2] = torch.linspace(3.0, 4.2, 4)[None, :]
    case = _tiny_case(points, grid.numpy())
    _fit_points, initial = _fit_input_and_initial_uv(case)

    def fake_fitter(points, *, initial_uv, collect_diagnostics, **kwargs):
        assert kwargs == FIT_KWARGS
        assert collect_diagnostics is True
        surface = TorchNURBSSurface(grid.clone(), torch.ones((8, 4)))
        diagnostics = SimpleNamespace(rounds=[SimpleNamespace(uv=initial_uv.clone())])
        return surface, initial_uv + 0.03, diagnostics

    import devtools.demo.parametric_continuation_attribution as module

    monkeypatch.setattr(module, "fit_torch_visible_surface_lsq", fake_fitter)
    replay = replay_frozen_fit(case, "cpu")
    assert replay.diagnostics["sufficiently_identical"] is True
    assert np.allclose(replay.final_uv, replay.initial_uv + 0.03)
    report = parameterization_report(replay)
    assert report["u"]["final_max"] > report["u"]["initial_max"]


def test_face_interface_and_target_components_are_fixed_and_deterministic(tmp_path):
    points = _tiny_reference()
    faces = np.asarray([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ], dtype=np.int64)
    path = tmp_path / "mesh.npz"
    np.savez(path, vertices=points, faces=faces)
    vertex_roi = _roi_vertex_contract(points, PRIMARY_ROI)
    first = trace_mesh_interface(path, points, vertex_roi)
    second = trace_mesh_interface(path, points, vertex_roi)
    assert len(first.interface_points) > 0
    np.testing.assert_allclose(first.interface_points, second.interface_points)
    np.testing.assert_array_equal(first.withheld_component, second.withheld_component)
    assert first.withheld_face_count > 0
    assert target_coherence_report(
        _tiny_case(points, np.zeros((8, 4, 3), dtype=np.float64)), vertex_roi, first, 0.1
    )["interface_connected_component_count"] >= 1


def test_boundary_support_uses_observed_points_and_final_footpoints_only():
    points = _tiny_reference()
    vertex_roi = _roi_vertex_contract(points, PRIMARY_ROI)
    replay = SimpleNamespace(
        fit_points=vertex_roi.observed_points,
        final_uv=np.column_stack([
            np.full(len(vertex_roi.observed_points), 0.97),
            np.linspace(0.0, 1.0, len(vertex_roi.observed_points)),
        ]),
    )
    boundary = vertex_roi.observed_points[: min(8, len(vertex_roi.observed_points))]
    report = boundary_support_report(replay, vertex_roi, boundary, 0.1)
    assert report["observed_geometry_only"] is True
    assert report["unsupported_v_bin_fraction"]["no_footpoint_u_ge_0.95"] < 1.0


def test_geometric_interface_report_does_not_use_withheld_target_for_fit():
    # This is a contract-level smoke check: interface agreement is evaluated
    # against a face-derived interface and a supplied fitted boundary only.
    points = _tiny_reference()
    faces = np.asarray([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ], dtype=np.int64)
    path_points = points
    # The full geometry is enough for the deterministic ROI/face audit; use a
    # minimal synthetic trace with the actual face-derived interface.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = __import__("pathlib").Path(directory) / "mesh.npz"
        np.savez(path, vertices=path_points, faces=faces)
        vertex_roi = _roi_vertex_contract(points, PRIMARY_ROI)
        trace = trace_mesh_interface(path, points, vertex_roi)
    boundary = trace.interface_points[: min(16, len(trace.interface_points))]
    tangent = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (len(boundary), 1))
    normals = np.tile(np.asarray([[0.0, 1.0, 0.0]]), (len(boundary), 1))
    replay = SimpleNamespace(surface=None)
    result = geometric_interface_report(replay, vertex_roi, trace, boundary, tangent, normals, 0.1)
    assert result["interface_definition"].startswith("mesh face edges")
    assert "fitted_boundary_to_interface_distance_over_h" in result
