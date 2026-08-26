from __future__ import annotations

"""Worklog 119 -- Visible-NURBS Geometry / UV Control Correction focused tests.

Covers: equal solve-count UV A/B, ARM B never reprojecting UV, common
geometric/camera-UV evaluation not mutating either surface, direct
surfel-intersection world-point reconstruction, rho3d/rho2d branch
classification, pixel<->raster-index alignment used for pixel-level D
attribution, and deterministic replay of the corrected synthetic controls.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from visible_nurbs_geometry_uv_control_correction import (  # noqa: E402
    _build_pixel_records_vectorized,
    camera_correspondence_error,
    classify_median_event_branch,
    fit_fixed_uv_equal_solves,
    geometric_point_to_surface_error,
    reconstruct_direct_surfel_intersection_world_point,
    run_equal_count_synthetic_contracts_corrected,
)
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points  # noqa: E402


def _synthetic_curved_points(n: int = 200, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand((n,), generator=generator)
    v = torch.rand((n,), generator=generator)
    x = u * 2.0
    y = v * 1.0
    z = 0.3 * torch.sin(torch.pi * u) * torch.sin(torch.pi * v)
    return torch.stack([x, y, z], dim=-1)


def _legacy_pixel_records(
    chart_id: int,
    view_index: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    residual: torch.Tensor,
    rep_id: torch.Tensor,
    rho3d: torch.Tensor,
    rho2d: torch.Tensor,
    branch: torch.Tensor,
    s_magnitude: torch.Tensor,
    depth: torch.Tensor,
    g0_g2_distance: torch.Tensor,
    g2_finite: torch.Tensor,
) -> list[dict[str, object]]:
    """Reference for the pre-fix per-element extraction contract."""

    records = []
    for i in range(int(rows.shape[0])):
        branch_value = int(branch[i])
        records.append({
            "chart_id": chart_id, "view_index": view_index, "row": int(rows[i]), "col": int(cols[i]),
            "residual_camera_correspondence_arm_a": float(residual[i]),
            "representative_id_full": int(rep_id[i]),
            "rho3d": float(rho3d[i]), "rho2d": float(rho2d[i]),
            "branch": "rho3d" if branch_value == 1 else ("rho2d" if branch_value == 0 else "none"),
            "s_magnitude": float(s_magnitude[i]),
            "depth": float(depth[i]),
            "g0_vs_g2_distance": float(g0_g2_distance[i]) if bool(g2_finite[i]) else None,
            "region_label": None,
        })
    return records


class TestPixelRecordSerializationEquivalence:
    def test_bulk_numpy_builder_matches_legacy_scalar_contract_exactly(self):
        rows = torch.tensor([0, 3, 7, 10], dtype=torch.int64)
        cols = torch.tensor([2, 1, 9, 0], dtype=torch.int64)
        residual = torch.tensor([0.0, 1.25, float("nan"), float("inf")], dtype=torch.float32)
        rep_id = torch.tensor([17, 4, 0, 999], dtype=torch.int64)
        rho3d = torch.tensor([0.5, -1.0, 2.0, float("inf")], dtype=torch.float32)
        rho2d = torch.tensor([0.75, -1.0, 1.0, float("nan")], dtype=torch.float32)
        branch = torch.tensor([1, -1, 0, 1], dtype=torch.int8)
        s_magnitude = torch.tensor([0.1, 0.2, float("nan"), float("inf")], dtype=torch.float32)
        depth = torch.tensor([2.0, 3.0, 4.0, float("nan")], dtype=torch.float32)
        g0_g2_distance = torch.tensor([0.25, float("nan"), 1.5, float("inf")], dtype=torch.float32)
        g2_finite = torch.tensor([True, False, True, False])

        old_records = _legacy_pixel_records(
            42, 7, rows, cols, residual, rep_id, rho3d, rho2d, branch,
            s_magnitude, depth, g0_g2_distance, g2_finite,
        )
        new_records = _build_pixel_records_vectorized(
            42, 7, *(array.numpy() for array in (
                rows, cols, residual, rep_id, rho3d, rho2d, branch,
                s_magnitude, depth, g0_g2_distance, g2_finite,
            ))
        )

        assert len(new_records) == len(old_records)
        assert [record.keys() for record in new_records] == [record.keys() for record in old_records]
        for new_record, old_record in zip(new_records, old_records):
            for key in new_record:
                new_value, old_value = new_record[key], old_record[key]
                if isinstance(new_value, float) and isinstance(old_value, float):
                    if np.isnan(new_value) and np.isnan(old_value):
                        continue
                    assert np.asarray(new_value, dtype=np.float64).view(np.uint64).item() == np.asarray(old_value, dtype=np.float64).view(np.uint64).item()
                else:
                    assert new_value == old_value


class TestEqualSolveCountUVControl:
    def test_arm_b_never_changes_uv(self):
        points = _synthetic_curved_points()
        uv = pca_parameterize_points(points)
        uv_before = uv.clone()
        _surface, uv_returned = fit_fixed_uv_equal_solves(points, uv, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, correction_rounds=2)
        assert torch.equal(uv_returned, uv_before)

    def test_arm_b_performs_the_same_solve_count_as_requested(self, monkeypatch):
        points = _synthetic_curved_points()
        uv = pca_parameterize_points(points)
        call_count = {"n": 0}
        import visible_nurbs_geometry_uv_control_correction as module

        original = module._solve_control_grid_lsq

        def _counting_solve(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_solve_control_grid_lsq", _counting_solve)
        module.fit_fixed_uv_equal_solves(points, uv, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, correction_rounds=2)
        assert call_count["n"] == 2

    def test_arm_a_and_arm_b_use_the_same_solve_count(self, monkeypatch):
        points = _synthetic_curved_points()
        uv = pca_parameterize_points(points)
        import visible_nurbs_geometry_uv_control_correction as module
        import osn_gs.surface.torch_nurbs as nurbs_module

        call_count = {"n": 0}
        original = nurbs_module._solve_control_grid_lsq

        def _counting_solve(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        # ARM A's solve loop lives in `fit_torch_visible_surface_lsq`, which
        # calls `_solve_control_grid_lsq` from ITS OWN module namespace
        # (`osn_gs.surface.torch_nurbs`), not the copy imported into the
        # devtools script -- patch that namespace for this arm.
        monkeypatch.setattr(nurbs_module, "_solve_control_grid_lsq", _counting_solve)
        fit_torch_visible_surface_lsq(points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=uv, correction_rounds=2, projection_iterations=3)
        arm_a_solves = call_count["n"]

        # ARM B's solve loop lives in the devtools script and calls
        # `_solve_control_grid_lsq` via ITS OWN imported name -- patch that
        # namespace for this arm.
        call_count["n"] = 0
        monkeypatch.setattr(module, "_solve_control_grid_lsq", _counting_solve)
        module.fit_fixed_uv_equal_solves(points, uv, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, correction_rounds=2)
        assert call_count["n"] == arm_a_solves


class TestCommonEvaluationMetrics:
    def test_geometric_evaluation_does_not_mutate_surface(self):
        points = _synthetic_curved_points()
        uv = pca_parameterize_points(points)
        surface, _uv_footpoint = fit_torch_visible_surface_lsq(points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=uv, correction_rounds=2, projection_iterations=3)
        control_before = surface.control_grid.clone()
        residual, uv_eval = geometric_point_to_surface_error(surface, points)
        assert torch.equal(surface.control_grid, control_before)
        assert residual.shape[0] == points.shape[0]
        assert uv_eval.shape == (points.shape[0], 2)

    def test_camera_correspondence_evaluation_does_not_mutate_surface(self):
        points = _synthetic_curved_points()
        uv = pca_parameterize_points(points)
        surface, _uv_footpoint = fit_torch_visible_surface_lsq(points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=uv, correction_rounds=2, projection_iterations=3)
        control_before = surface.control_grid.clone()
        residual = camera_correspondence_error(surface, points, uv)
        assert torch.equal(surface.control_grid, control_before)
        assert residual.shape[0] == points.shape[0]

    def test_geometric_error_is_never_worse_than_camera_correspondence_error(self):
        # Closest-point projection can only match or improve on a fixed-UV residual.
        points = _synthetic_curved_points()
        uv = pca_parameterize_points(points)
        surface, _uv_footpoint = fit_torch_visible_surface_lsq(points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=uv, correction_rounds=2, projection_iterations=3)
        residual_g, _ = geometric_point_to_surface_error(surface, points)
        residual_c = camera_correspondence_error(surface, points, uv)
        assert float(residual_g.mean()) <= float(residual_c.mean()) + 1e-4


class TestDirectSurfelIntersectionReconstruction:
    def test_reconstruction_matches_hand_computed_point(self):
        positions = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        tangent_u = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tangent_v = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        scale_u = torch.tensor([2.0, 3.0])
        scale_v = torch.tensor([0.5, 4.0])
        representative_id = torch.tensor([[0, 1], [-1, 0]])
        s_u = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
        s_v = torch.tensor([[2.0, -1.0], [0.0, 0.5]])

        world, valid = reconstruct_direct_surfel_intersection_world_point(
            positions, tangent_u, tangent_v, scale_u, scale_v, representative_id, s_u, s_v
        )
        assert valid.tolist() == [[True, True], [False, True]]
        # pixel (0,0): surfel 0, center (1,2,3), s_u=1*scale_u=2 along (1,0,0), s_v=2*scale_v=0.5*2=1 along (0,1,0)
        expected_00 = torch.tensor([1.0 + 1.0 * 2.0, 2.0 + 2.0 * 0.5, 3.0])
        torch.testing.assert_close(world[0, 0], expected_00)
        # pixel (1,1): surfel 0 again, s_u=-1*2=-2 along (1,0,0), s_v=0.5*0.5=0.25 along (0,1,0)
        expected_11 = torch.tensor([1.0 - 1.0 * 2.0, 2.0 + 0.5 * 0.5, 3.0])
        torch.testing.assert_close(world[1, 1], expected_11)

    def test_invalid_representative_is_flagged_but_does_not_crash(self):
        positions = torch.zeros((1, 3))
        tangent_u = torch.tensor([[1.0, 0.0, 0.0]])
        tangent_v = torch.tensor([[0.0, 1.0, 0.0]])
        scale_u = torch.tensor([1.0])
        scale_v = torch.tensor([1.0])
        representative_id = torch.tensor([[-1]])
        s_u = torch.tensor([[5.0]])
        s_v = torch.tensor([[5.0]])
        world, valid = reconstruct_direct_surfel_intersection_world_point(
            positions, tangent_u, tangent_v, scale_u, scale_v, representative_id, s_u, s_v
        )
        assert valid.item() is False
        assert torch.isfinite(world).all()  # clamped to a valid (if meaningless) gather, no NaN/crash


class TestBranchClassification:
    def test_rho3d_dominated_when_smaller_or_equal(self):
        rho3d = torch.tensor([1.0, 2.0, 3.0, -1.0])
        rho2d = torch.tensor([2.0, 1.0, 3.0, 5.0])
        branch = classify_median_event_branch(rho3d, rho2d)
        assert branch.tolist() == [1, 0, 1, -1]

    def test_no_event_is_minus_one(self):
        rho3d = torch.tensor([-1.0, -1.0])
        rho2d = torch.tensor([0.5, -1.0])
        branch = classify_median_event_branch(rho3d, rho2d)
        assert branch.tolist() == [-1, -1]


class TestPixelRasterAlignment:
    def test_nonzero_row_major_order_matches_boolean_flatten_order(self):
        # Mirrors the alignment the main script relies on between
        # `np.nonzero(blob_mask)` and the pixel order `build_view_chart_pixel_samples`
        # produces via `valid = blob_labels >= 0` boolean-indexing on the same 2D grid.
        mask = np.array([
            [False, True, False],
            [True, False, True],
            [False, False, True],
        ])
        rows, cols = np.nonzero(mask)
        flat_valid_order = np.argwhere(mask.reshape(-1)).reshape(-1)
        recovered = rows * mask.shape[1] + cols
        assert np.array_equal(recovered, flat_valid_order)


class TestDeterministicReplay:
    def test_synthetic_contracts_are_deterministic(self):
        first = run_equal_count_synthetic_contracts_corrected()
        second = run_equal_count_synthetic_contracts_corrected()
        assert first["planar"]["removed_count_in_B_and_C"] == second["planar"]["removed_count_in_B_and_C"]
        assert first["curved"]["B_enclosed_hole_footpoint"]["residual"]["median"] == pytest.approx(
            second["curved"]["B_enclosed_hole_footpoint"]["residual"]["median"]
        )
        assert first["curved"]["D_boundary_notch_footpoint"]["residual"]["median"] == pytest.approx(
            second["curved"]["D_boundary_notch_footpoint"]["residual"]["median"]
        )

    def test_boundary_notch_has_a_comparable_removed_count_to_hole_and_dispersed(self):
        results = run_equal_count_synthetic_contracts_corrected()
        for label in ("planar", "curved"):
            removed_bc = results[label]["removed_count_in_B_and_C"]
            removed_d = results[label]["notch_removed_count_in_D"]
            assert abs(removed_bc - removed_d) <= 24  # within one row's worth (cols=24)
