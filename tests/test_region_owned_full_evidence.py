"""Worklog 67: region-owned full-cloud evidence for NURBS fitting/fidelity
validation only -- see osn_gs/surface/torch_region_owned_full_evidence.py.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_region_owned_full_evidence import (
    STATE_MATERIALIZED,
    STATE_UNDER_SUPPORTED,
    collect_region_owned_evidence,
    fit_region_owned_full_evidence_patch,
)


class CollectRegionOwnedEvidenceTest(unittest.TestCase):
    def test_filters_by_patch_id(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        stable_ids = [10, 11, 12, 13]
        propagated = torch.tensor([0, 1, 0, -1])
        owned_points, owned_ids = collect_region_owned_evidence(points, stable_ids, propagated, patch_id=0)
        self.assertEqual(owned_ids, (10, 12))
        self.assertTrue(torch.equal(owned_points, points[[0, 2]]))

    def test_empty_when_no_point_matches(self):
        points = torch.zeros((3, 3))
        stable_ids = [0, 1, 2]
        propagated = torch.tensor([-1, -1, -1])
        owned_points, owned_ids = collect_region_owned_evidence(points, stable_ids, propagated, patch_id=0)
        self.assertEqual(owned_ids, ())
        self.assertEqual(int(owned_points.shape[0]), 0)

    def test_deduplicates_exact_duplicate_stable_ids(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        stable_ids = [5, 5, 6]  # same stable id appearing twice (defensive case)
        propagated = torch.tensor([0, 0, 0])
        owned_points, owned_ids = collect_region_owned_evidence(points, stable_ids, propagated, patch_id=0)
        self.assertEqual(owned_ids, (5, 6))
        self.assertEqual(int(owned_points.shape[0]), 2)


class FitRegionOwnedFullEvidencePatchTest(unittest.TestCase):
    def _planar_grid(self, n: int = 6) -> torch.Tensor:
        axis = torch.linspace(-0.5, 0.5, n)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(n * n)), dim=1)

    def test_under_supported_when_full_evidence_below_minimum(self):
        boundary = self._planar_grid(3)[:4]
        evidence = boundary[:3]  # 3 < MIN_FULL_EVIDENCE_SUPPORT(4)
        fit = fit_region_owned_full_evidence_patch(
            "parametric", 7, boundary, evidence, (1, 2, 3), representative_support_count=3,
        )
        self.assertEqual(fit.state, STATE_UNDER_SUPPORTED)
        self.assertIsNone(fit.surface)
        self.assertEqual(fit.full_evidence_support_count, 3)
        self.assertIn("full_evidence_support_count=3<4", fit.reasons[0])

    def test_materializes_with_a_well_supported_planar_patch(self):
        grid = self._planar_grid(6)
        boundary = grid[:4]
        fit = fit_region_owned_full_evidence_patch(
            "physical", 3, boundary, grid, tuple(range(grid.shape[0])), representative_support_count=4,
        )
        self.assertEqual(fit.state, STATE_MATERIALIZED)
        self.assertIsNotNone(fit.surface)
        self.assertEqual(fit.full_evidence_support_count, grid.shape[0])
        self.assertGreater(fit.full_evidence_support_count, fit.representative_support_count)
        self.assertEqual(fit.jacobian_near_degenerate_count, 0)
        self.assertIsNotNone(fit.boundary_residual)
        self.assertIsNotNone(fit.full_evidence_interior_residual)

    def test_representative_and_full_evidence_counts_recorded_separately(self):
        grid = self._planar_grid(6)
        boundary = grid[:4]
        fit = fit_region_owned_full_evidence_patch(
            "physical", 3, boundary, grid, tuple(range(grid.shape[0])), representative_support_count=4,
        )
        self.assertEqual(fit.representative_support_count, 4)
        self.assertEqual(fit.full_evidence_support_count, 36)
        self.assertNotEqual(fit.representative_support_count, fit.full_evidence_support_count)


def _pipeline(max_points: int) -> TorchOSNGSPipeline:
    return TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=max_points), device="cpu")


class PipelineWiringTest(unittest.TestCase):
    """Real end-to-end: downsampled construction recovers MORE evidence per
    materialized region than the representative-only fit ever saw, while
    leaving region/boundary/eligibility counts completely unaffected."""

    def _dense_box_scene(self):
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_density_sweep_scene
        return make_gaussian_density_sweep_scene("box", 4, seed=0)

    def test_downsampled_construction_populates_region_owned_fits(self):
        scene = self._dense_box_scene()
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))

        pipeline = _pipeline(max_points=200)
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)

        self.assertLess(200, int(positions.shape[0]))  # sanity: scene really is denser than the budget
        materialized_keys = {
            ("physical", item.input.source_region_id)
            for item in bundle.construction.materialized_visible_nurbs_surfaces if item.surface is not None
        } | {
            ("parametric", item.input.source_region_id)
            for item in bundle.construction.materialized_parametric_chart_surfaces if item.surface is not None
        }
        self.assertTrue(materialized_keys, "fixture must materialize at least one patch for this test to mean anything")
        self.assertEqual(set(bundle.region_owned_full_evidence_fits.keys()), materialized_keys)

        recovered_more_evidence = False
        for key, fit in bundle.region_owned_full_evidence_fits.items():
            self.assertIn(fit.state, ("materialized", "under_supported", "unsafe_geometry", "fit_failed"))
            if fit.full_evidence_support_count > fit.representative_support_count:
                recovered_more_evidence = True
        self.assertTrue(recovered_more_evidence, "expected at least one region to recover MORE than representative-only evidence")

    def test_non_downsampled_construction_leaves_fits_empty(self):
        """Small scene where the representative budget already covers every
        Gaussian: there is no extra full-cloud evidence to recover."""
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
        scene = make_gaussian_reliability_scene("box_face")
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))

        pipeline = _pipeline(max_points=max(2048, int(positions.shape[0]) + 1))
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        self.assertEqual(bundle.region_owned_full_evidence_fits, {})

    def test_region_formation_and_boundary_are_unaffected_by_full_evidence_recovery(self):
        """Same construction, same region/materialized counts regardless of
        this new step -- it is additive-only by construction (never writes
        to `bundle.construction`), verified here as a regression guard."""
        scene = self._dense_box_scene()
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))

        pipeline = _pipeline(max_points=200)
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        region_count = bundle.construction.diagnostic_summary["region_count"]
        materialized_count = bundle.construction.diagnostic_summary["materialized_surface_count"]
        parametric_count = bundle.construction.diagnostic_summary["parametric_chart_materialized_surface_count"]

        # Calling the fit collector again (idempotent, read-only over `construction`) must not change anything.
        refits = pipeline._collect_region_owned_full_evidence_fits(positions, covariance, stable_ids, bundle)
        self.assertEqual(bundle.construction.diagnostic_summary["region_count"], region_count)
        self.assertEqual(bundle.construction.diagnostic_summary["materialized_surface_count"], materialized_count)
        self.assertEqual(bundle.construction.diagnostic_summary["parametric_chart_materialized_surface_count"], parametric_count)
        self.assertEqual(set(refits.keys()), set(bundle.region_owned_full_evidence_fits.keys()))


if __name__ == "__main__":
    unittest.main()
