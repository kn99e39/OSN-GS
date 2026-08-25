from __future__ import annotations

"""Worklog 113 -- Chart Representation Contract Diagnostic.

Pure-logic focused tests for the diagnostic-only helpers added in
`scripts/devtools/chart_representation_contract_diagnostic.py`: raster
blob domain-shape accounting (bbox / occupancy / holes, directive section
3), fixed-NURBS design-matrix rank/conditioning (directive section 6), and
quantile-bin/distribution utilities (directive section 5). No production
`osn_gs` code is touched by this batch -- these tests exercise the new
devtools-only functions directly.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from chart_representation_contract_diagnostic import (  # noqa: E402
    _bin_by_quantile,
    _distribution,
    blob_domain_shape,
    design_matrix_rank_diagnostics,
)
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq  # noqa: E402


class BlobDomainShapeTest(unittest.TestCase):
    def test_solid_rectangle_has_full_occupancy_and_no_holes(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 3:15] = True
        result = blob_domain_shape(mask)
        self.assertEqual(result["bbox_h"], 6)
        self.assertEqual(result["bbox_w"], 12)
        self.assertEqual(result["pixel_count"], 6 * 12)
        self.assertAlmostEqual(result["occupancy_ratio"], 1.0)
        self.assertEqual(result["hole_count"], 0)
        self.assertAlmostEqual(result["aspect_ratio"], 12 / 6)

    def test_ring_shape_has_low_occupancy_and_one_hole(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True
        mask[6:14, 6:14] = False  # punch out an interior hole
        result = blob_domain_shape(mask)
        self.assertLess(result["occupancy_ratio"], 1.0)
        self.assertEqual(result["hole_count"], 1)
        self.assertEqual(result["hole_area"], 8 * 8)

    def test_l_shape_has_partial_bbox_occupancy_no_holes(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:5, 0:2] = True
        mask[3:5, 0:8] = True
        result = blob_domain_shape(mask)
        self.assertEqual(result["hole_count"], 0)
        self.assertLess(result["occupancy_ratio"], 1.0)

    def test_two_separate_holes_counted_independently(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:18, 2:28] = True
        mask[4:8, 4:8] = False
        mask[10:14, 20:24] = False
        result = blob_domain_shape(mask)
        self.assertEqual(result["hole_count"], 2)
        self.assertEqual(result["hole_area"], 4 * 4 + 4 * 4)

    def test_empty_mask_returns_zeroed_record(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        result = blob_domain_shape(mask)
        self.assertEqual(result["pixel_count"], 0)
        self.assertEqual(result["occupancy_ratio"], 0.0)


class DesignMatrixRankDiagnosticsTest(unittest.TestCase):
    def _planar_chart(self, n: int = 64) -> tuple:
        rng = torch.Generator().manual_seed(0)
        u = torch.rand((n,), generator=rng)
        v = torch.rand((n,), generator=rng)
        uv = torch.stack([u, v], dim=1)
        points = torch.stack([u * 2.0, v * 2.0, torch.zeros_like(u)], dim=1)
        return points, uv

    def test_well_supported_planar_chart_reaches_full_rank(self) -> None:
        points, uv = self._planar_chart(n=200)
        surface, fitted_uv = fit_torch_visible_surface_lsq(
            points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
            initial_uv=uv, correction_rounds=2, projection_iterations=3,
        )
        result = design_matrix_rank_diagnostics(surface, fitted_uv, seed=0)
        self.assertEqual(result["full_capacity"], 8 * 4)
        self.assertGreater(result["rank"], 0)
        self.assertLessEqual(result["rank"], result["full_capacity"])

    def test_degenerate_single_point_chart_is_severely_rank_deficient(self) -> None:
        # 4 nearly-identical points can never determine a 32-control-point surface.
        points = torch.tensor([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0], [1e-6, 1e-6, 0.0]])
        uv = torch.tensor([[0.1, 0.1], [0.11, 0.1], [0.1, 0.11], [0.11, 0.11]])
        surface, fitted_uv = fit_torch_visible_surface_lsq(
            points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
            initial_uv=uv, correction_rounds=1, projection_iterations=1,
        )
        result = design_matrix_rank_diagnostics(surface, fitted_uv, seed=0)
        self.assertLess(result["rank"], result["full_capacity"])

    def test_subsampling_is_deterministic_for_same_seed(self) -> None:
        points, uv = self._planar_chart(n=5000)
        surface, fitted_uv = fit_torch_visible_surface_lsq(
            points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
            initial_uv=uv, correction_rounds=1, projection_iterations=2,
        )
        result_a = design_matrix_rank_diagnostics(surface, fitted_uv, seed=42, subsample_cap=256)
        result_b = design_matrix_rank_diagnostics(surface, fitted_uv, seed=42, subsample_cap=256)
        self.assertEqual(result_a["rank"], result_b["rank"])
        self.assertAlmostEqual(result_a["cond_number"], result_b["cond_number"], places=5)


class DistributionAndBinningTest(unittest.TestCase):
    def test_distribution_matches_known_quantiles(self) -> None:
        values = np.arange(1, 101, dtype=np.float64)  # 1..100
        result = _distribution(values)
        self.assertEqual(result["count"], 100)
        self.assertAlmostEqual(result["min"], 1.0)
        self.assertAlmostEqual(result["max"], 100.0)
        self.assertAlmostEqual(result["median"], values[50])  # round(0.5 * 99) == 50

    def test_empty_distribution_is_all_zero(self) -> None:
        result = _distribution(np.zeros((0,)))
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["median"], 0.0)

    def test_bin_by_quantile_produces_requested_bin_count(self) -> None:
        values = np.arange(100, dtype=np.float64)
        bins = _bin_by_quantile(values, n_bins=4)
        self.assertEqual(len(set(bins.tolist())), 4)

    def test_bin_by_quantile_groups_low_and_high_values_separately(self) -> None:
        values = np.concatenate([np.zeros(50), np.full(50, 1000.0)])
        bins = _bin_by_quantile(values, n_bins=4)
        self.assertNotEqual(bins[0], bins[-1])


if __name__ == "__main__":
    unittest.main()
