from __future__ import annotations

"""Worklog 117 -- Holey-Chart Fitting-Coupling Attribution.

Pure-logic focused tests for the diagnostic-only helpers added in
`scripts/devtools/holey_chart_fitting_coupling_attribution.py`: mask
hole/edge decomposition, distance-to-unsupported/hole accounting, within-
chart correlation/near-far statistics, and the synthetic full-vs-hole
ground-truth contracts. No production `osn_gs` code is touched by this
batch -- these tests exercise the new devtools-only functions directly, plus
one real-fitter contract proving support-mask assignment never alters an
already-computed fit.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from holey_chart_fitting_coupling_attribution import (  # noqa: E402
    _make_curved_grid,
    _make_planar_grid,
    distance_to_hole_grid,
    distance_to_unsupported_grid,
    hole_and_edge_masks,
    near_far_median_split,
    run_synthetic_contracts,
    sample_uv_to_cell,
    within_chart_distance_correlation,
)
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline  # noqa: E402
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq  # noqa: E402


class HoleAndEdgeMasksTest(unittest.TestCase):
    def test_solid_rectangle_has_no_holes_no_edge_unsupported(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        holes, edge = hole_and_edge_masks(mask)
        self.assertFalse(holes.any())
        self.assertFalse(edge.any())

    def test_ring_shape_has_one_enclosed_hole(self) -> None:
        mask = np.ones((20, 20), dtype=bool)
        mask[8:12, 8:12] = False
        holes, edge = hole_and_edge_masks(mask)
        self.assertTrue(holes[9, 9])
        self.assertFalse(edge[9, 9])

    def test_corner_gap_is_edge_unsupported_not_a_hole(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        mask[0:3, 0:3] = False  # touches the border -> not enclosed
        holes, edge = hole_and_edge_masks(mask)
        self.assertFalse(holes.any())
        self.assertTrue(edge[0, 0])

    def test_two_separate_holes_both_detected(self) -> None:
        mask = np.ones((20, 30), dtype=bool)
        mask[4:8, 4:8] = False
        mask[10:14, 20:24] = False
        holes, _edge = hole_and_edge_masks(mask)
        from scipy.ndimage import label
        _labels, count = label(holes)
        self.assertEqual(count, 2)


class DistanceAccountingTest(unittest.TestCase):
    def test_distance_to_unsupported_is_zero_at_the_boundary(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        mask[:, 5:] = False
        dist = distance_to_unsupported_grid(mask)
        self.assertAlmostEqual(dist[0, 4], 1.0)  # adjacent to unsupported column 5
        self.assertGreater(dist[0, 0], dist[0, 4])  # farther from the boundary -> larger distance

    def test_distance_to_hole_is_none_without_holes(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        holes, _edge = hole_and_edge_masks(mask)
        self.assertIsNone(distance_to_hole_grid(holes))

    def test_distance_to_hole_increases_away_from_the_hole(self) -> None:
        mask = np.ones((20, 20), dtype=bool)
        mask[9:11, 9:11] = False
        holes, _edge = hole_and_edge_masks(mask)
        dist = distance_to_hole_grid(holes)
        self.assertIsNotNone(dist)
        self.assertLess(dist[8, 9], dist[0, 0])

    def test_sample_uv_to_cell_matches_occupancy_mask_binning(self) -> None:
        uv = torch.tensor([[0.05, 0.95], [0.5, 0.5]])
        cell_u, cell_v = sample_uv_to_cell(uv, resolution=10)
        mask = TorchOSNGSPipeline._uv_occupancy_mask(uv, 10, 0)
        self.assertTrue(bool(mask[cell_u[0], cell_v[0]]))
        self.assertTrue(bool(mask[cell_u[1], cell_v[1]]))


class WithinChartStatisticsTest(unittest.TestCase):
    def test_correlation_detects_perfect_negative_relationship(self) -> None:
        distance = np.linspace(0.0, 1.0, 50)
        residual = 1.0 - distance  # perfectly negatively correlated with distance
        corr = within_chart_distance_correlation(residual, distance)
        self.assertLess(corr, -0.99)

    def test_correlation_is_nan_for_constant_distance(self) -> None:
        distance = np.full((10,), 0.5)
        residual = np.random.rand(10)
        corr = within_chart_distance_correlation(residual, distance)
        self.assertTrue(np.isnan(corr))

    def test_near_far_split_orders_correctly_for_b2_consistent_data(self) -> None:
        distance = np.linspace(0.0, 1.0, 30)
        residual = 1.0 - distance  # near (small distance) -> high residual
        near, far = near_far_median_split(residual, distance)
        self.assertGreater(near, far)


class SyntheticContractGeometryTest(unittest.TestCase):
    def test_planar_hole_grid_removes_only_the_central_region(self) -> None:
        full_points, full_uv = _make_planar_grid(20, 20, hole=False)
        hole_points, hole_uv = _make_planar_grid(20, 20, hole=True)
        self.assertGreater(int(full_points.shape[0]), int(hole_points.shape[0]))
        # every retained UV must lie outside the cut region
        cut = (hole_uv[:, 0] > 0.35) & (hole_uv[:, 0] < 0.65) & (hole_uv[:, 1] > 0.35) & (hole_uv[:, 1] < 0.65)
        self.assertFalse(bool(cut.any()))

    def test_curved_hole_grid_shares_identical_geometry_outside_the_hole(self) -> None:
        full_points, full_uv = _make_curved_grid(16, 16, hole=False)
        hole_points, hole_uv = _make_curved_grid(16, 16, hole=True)
        full_map = {(round(float(u), 6), round(float(v), 6)): full_points[i] for i, (u, v) in enumerate(full_uv.tolist())}
        for i, (u, v) in enumerate(hole_uv.tolist()):
            self.assertTrue(torch.allclose(full_map[(round(u, 6), round(v, 6))], hole_points[i]))

    def test_run_synthetic_contracts_produces_planar_and_curved_results(self) -> None:
        results = run_synthetic_contracts()
        self.assertIn("planar", results)
        self.assertIn("curved", results)
        for label in ("planar", "curved"):
            self.assertGreater(results[label]["full_point_count"], results[label]["hole_point_count"])


class SupportMaskDoesNotAlterFitTest(unittest.TestCase):
    def test_assigning_uv_support_mask_never_changes_control_grid_or_evaluate(self) -> None:
        points, uv = _make_curved_grid(12, 12, hole=False)
        with torch.no_grad():
            surface, fit_uv = fit_torch_visible_surface_lsq(
                points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
                initial_uv=uv, correction_rounds=2, projection_iterations=3,
            )
            control_before = surface.control_grid.detach().clone()
            fitted_before = surface.evaluate(fit_uv)

            mask = TorchOSNGSPipeline._uv_occupancy_mask(fit_uv.detach(), 24, 1)
            surface.uv_support_mask = mask

            control_after = surface.control_grid.detach()
            fitted_after = surface.evaluate(fit_uv)

        self.assertTrue(torch.equal(control_before, control_after))
        self.assertTrue(torch.equal(fitted_before, fitted_after))


if __name__ == "__main__":
    unittest.main()
