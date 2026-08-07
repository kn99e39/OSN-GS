"""Worklog 69: single-chart PCA-UV parameterization validity diagnostics."""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_single_chart_uv_validity import (
    accepted_edge_uv_crossings,
    interior_within_boundary,
    neighborhood_preservation,
    parallel_sheet_suspicion,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)


class UVDuplicateDiagnosticsTest(unittest.TestCase):
    def test_regular_grid_has_no_duplicates(self):
        axis = torch.linspace(0.0, 1.0, 6)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        uv = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
        result = uv_duplicate_diagnostics(uv)
        self.assertEqual(result["uv_duplicate_count"], 0)

    def test_injected_duplicate_is_detected(self):
        uv = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.5, 0.5 + 1e-9]])
        result = uv_duplicate_diagnostics(uv)
        self.assertGreaterEqual(result["uv_duplicate_count"], 1)


class NeighborhoodPreservationTest(unittest.TestCase):
    def test_flat_identity_projection_preserves_neighbors(self):
        axis = torch.linspace(0.0, 1.0, 6)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((gx.reshape(-1), gy.reshape(-1), torch.zeros(36)), dim=1)
        uv = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
        result = neighborhood_preservation(positions, uv, k=4)
        self.assertGreater(result["neighborhood_preservation_mean"], 0.9)

    def test_folded_mapping_breaks_neighborhood_preservation(self):
        axis = torch.linspace(0.0, 1.0, 6)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((gx.reshape(-1), gy.reshape(-1), torch.zeros(36)), dim=1)
        # UV mapping scrambles point identity entirely (worst case fold).
        perm = torch.randperm(36, generator=torch.Generator().manual_seed(0))
        uv = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)[perm]
        result = neighborhood_preservation(positions, uv, k=4)
        self.assertLess(result["neighborhood_preservation_mean"], 0.9)


class AcceptedEdgeUVCrossingsTest(unittest.TestCase):
    def test_square_edges_do_not_cross(self):
        uv_by_id = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        result = accepted_edge_uv_crossings(uv_by_id, edges)
        self.assertEqual(result["accepted_edge_uv_crossing_count"], 0)

    def test_bowtie_edges_cross(self):
        uv_by_id = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
        edges = [(0, 1), (2, 3)]  # two diagonals of a square -- cross at center
        result = accepted_edge_uv_crossings(uv_by_id, edges)
        self.assertEqual(result["accepted_edge_uv_crossing_count"], 1)


class UVTriangulationDiagnosticsTest(unittest.TestCase):
    def test_flat_regular_grid_has_no_folds_and_low_distortion(self):
        axis = torch.linspace(0.0, 1.0, 6)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((gx.reshape(-1), gy.reshape(-1), torch.zeros(36)), dim=1)
        uv = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
        result = uv_triangulation_diagnostics(positions, uv)
        self.assertEqual(result["triangle_fold_count"], 0)
        self.assertAlmostEqual(result["area_distortion_median"], 1.0, places=3)

    def test_steep_ridge_folded_flat_by_naive_uv_produces_folds(self):
        """A steep tent/ridge (two flat half-planes meeting at x=0 with a
        sharp dihedral angle) naively projected to UV=(x, y) (never
        unfolding the ridge) -- UV-adjacent triangles straddling the ridge
        have near-opposite 3D normals, a genuine single-chart fold. Flat/
        coplanar points can NEVER produce this signal (every triangle's
        normal is trivially +-z regardless of layout), so the fixture must
        have real 3D non-planarity, not just a scrambled 2D<->3D mapping."""

        x = torch.linspace(-1.0, 1.0, 9)
        y = torch.linspace(0.0, 1.0, 5)
        gx, gy = torch.meshgrid(x, y, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        k = 50.0
        gz = -k * gx.abs()
        positions = torch.stack((gx, gy, gz), dim=1)
        uv = torch.stack((gx, gy), dim=1)  # naive projection -- never unfolds the ridge
        result = uv_triangulation_diagnostics(positions, uv)
        self.assertGreater(result["triangle_fold_count"], 0)


class InteriorWithinBoundaryTest(unittest.TestCase):
    def test_interior_point_inside_square_is_counted_inside(self):
        boundary = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        interior = torch.tensor([[0.5, 0.5]])
        result = interior_within_boundary(interior, boundary)
        self.assertEqual(result["interior_outside_boundary_count"], 0)

    def test_point_outside_square_is_counted_outside(self):
        boundary = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        interior = torch.tensor([[2.0, 2.0]])
        result = interior_within_boundary(interior, boundary)
        self.assertEqual(result["interior_outside_boundary_count"], 1)


class ParallelSheetSuspicionTest(unittest.TestCase):
    def test_single_coherent_plane_is_not_suspected(self):
        axis = torch.linspace(0.0, 1.0, 10)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((gx.reshape(-1), gy.reshape(-1), torch.zeros(100)), dim=1)
        normal = torch.tensor([0.0, 0.0, 1.0])
        result = parallel_sheet_suspicion(positions, normal)
        self.assertFalse(result["parallel_sheet_suspected"])

    def test_single_outlier_point_is_not_mistaken_for_a_second_sheet(self):
        """A lone routine outlier Gaussian (common in real ADC-trained data)
        must not trigger this check by itself -- the gap it creates is real
        but only borders 1 point, far below any meaningful cluster size."""

        axis_lin = torch.linspace(0.0, 1.0, 10)
        gx, gy = torch.meshgrid(axis_lin, axis_lin, indexing="ij")
        jitter = torch.linspace(-0.01, 0.01, 100)
        main_sheet = torch.stack((gx.reshape(-1), gy.reshape(-1), jitter), dim=1)
        outlier = torch.tensor([[0.5, 0.5, 50.0]])  # one extreme outlier far along the normal axis
        positions = torch.cat((main_sheet, outlier), dim=0)
        normal = torch.tensor([0.0, 0.0, 1.0])
        result = parallel_sheet_suspicion(positions, normal)
        self.assertFalse(result["parallel_sheet_suspected"])

    def test_two_well_separated_parallel_sheets_are_suspected(self):
        axis = torch.linspace(0.0, 1.0, 10)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        # Slight per-point z jitter within each sheet -- realistic Gaussian
        # evidence is never perfectly coplanar, and the gap-ratio test
        # requires nonzero within-sheet spacing along the normal axis to be
        # well-defined (a genuinely zero-variance sheet is a degenerate
        # edge case this synthetic fixture must avoid, not something the
        # function needs to special-case).
        jitter = torch.linspace(-0.01, 0.01, 100)
        sheet_a = torch.stack((gx.reshape(-1), gy.reshape(-1), jitter), dim=1)
        sheet_b = sheet_a.clone()
        sheet_b[:, 2] += 5.0  # far offset along the normal axis
        positions = torch.cat((sheet_a, sheet_b), dim=0)
        normal = torch.tensor([0.0, 0.0, 1.0])
        result = parallel_sheet_suspicion(positions, normal)
        self.assertTrue(result["parallel_sheet_suspected"])


if __name__ == "__main__":
    unittest.main()
