from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch
import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_curvature_sweep_scene, make_gaussian_reliability_scene
import osn_gs.surface.torch_visible_surface_construction as construction


class VisibleSurfaceConstructionInvarianceTest(unittest.TestCase):
    @staticmethod
    def _sample(surface):
        import torch
        grid = torch.linspace(0.04, 0.96, 11, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
        u, v = torch.meshgrid(grid, grid, indexing="ij")
        uv = torch.stack((u.reshape(-1), v.reshape(-1)), dim=1)
        return surface.evaluate(uv), surface.normals(uv)

    def _assert_equivalent_surface(self, baseline, variant, inverse, metric_scale):
        import torch
        baseline_attempt = baseline.materialized_visible_nurbs_surfaces[0]
        variant_attempt = variant.materialized_visible_nurbs_surfaces[0]
        base_points, base_normals = self._sample(baseline_attempt.surface)
        variant_points, variant_normals = self._sample(variant_attempt.surface)
        variant_points = inverse(variant_points)
        variant_normals = torch.nn.functional.normalize(inverse(variant_normals), dim=1)

        distances = torch.cdist(base_points, variant_points)
        chamfer = float(0.5 * (distances.min(dim=0).values.mean() + distances.min(dim=1).values.mean()))
        self.assertLess(chamfer, 0.035, "inverse-transformed sampled surfaces diverged")
        nearest = distances.argmin(dim=1)
        normal_alignment = float((base_normals * variant_normals[nearest]).sum(dim=1).abs().mean())
        self.assertGreater(normal_alignment, 0.94, "inverse-transformed surface normals diverged")

        # Residuals are world-distance metrics; compare in original scale.
        self.assertIsNotNone(baseline_attempt.boundary_residual)
        self.assertIsNotNone(variant_attempt.boundary_residual)
        self.assertIsNotNone(baseline_attempt.interior_residual)
        self.assertIsNotNone(variant_attempt.interior_residual)
        self.assertLess(abs(variant_attempt.boundary_residual / metric_scale - baseline_attempt.boundary_residual), 0.02)
        self.assertLess(abs(variant_attempt.interior_residual / metric_scale - baseline_attempt.interior_residual), 0.02)

    def test_box_face_and_curved_patch_survive_rigid_scale_and_order_variants(self):
        import torch
        rotations = (
            torch.tensor([[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]]),
            torch.tensor([[0.60, 0.64, 0.48], [-0.80, 0.48, 0.36], [0.00, -0.60, 0.80]]),
            torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        for scene in (make_gaussian_reliability_scene("box_face"), make_curvature_sweep_scene(10.0)):
            ids = tuple(range(len(scene.positions)))
            baseline = construction.construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=ids)
            expected_members = {frozenset(region.member_ids) for region in baseline.surface_regions.regions}
            expected_boundary = {frozenset(component.ordered_source_ids) for component in baseline.ordered_boundary_components}
            order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(712))
            variants = [
                (scene.positions @ rotation.T, rotation @ scene.covariances @ rotation.T, ids, lambda values, r=rotation: values @ r, 1.0)
                for rotation in rotations
            ]
            variants += [
                (scene.positions * scale, scene.covariances * scale ** 2, ids, lambda values, s=scale: values / s, scale)
                for scale in (0.31, 4.2)
            ]
            variants += [(scene.positions[order], scene.covariances[order], tuple(ids[item] for item in order.tolist()), lambda values: values, 1.0)]
            for positions, covariance, variant_ids, inverse, metric_scale in variants:
                result = construction.construct_visible_nurbs_from_gaussians(positions, covariance=covariance, stable_ids=variant_ids)
                self.assertEqual(result.construction_state, "constructed")
                self.assertEqual(len(result.materialized_visible_nurbs_surfaces), 1)
                self.assertEqual({frozenset(region.member_ids) for region in result.surface_regions.regions}, expected_members)
                self.assertEqual({frozenset(component.ordered_source_ids) for component in result.ordered_boundary_components}, expected_boundary)
                self._assert_equivalent_surface(baseline, result, inverse, metric_scale)

    def test_end_to_end_covariance_eigenvector_sign_equivalence(self):
        import torch
        for scene in (make_gaussian_reliability_scene("box_face"), make_curvature_sweep_scene(10.0)):
            ids = tuple(range(len(scene.positions)))
            baseline = construction.construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=ids)
            extracted = construction.extract_covariance_frame(scene.covariances)
            sign = torch.full((len(ids), 1), -1.0, dtype=scene.positions.dtype, device=scene.positions.device)
            # This simulates the sign-equivalent eigensolver representation
            # directly at the frame boundary; covariance itself cannot encode it.
            sign_flipped = replace(
                extracted,
                tangent_u=extracted.tangent_u * sign,
                tangent_v=extracted.tangent_v * -sign,
                normal_candidate=extracted.normal_candidate * sign,
            )
            with patch.object(construction, "extract_covariance_frame", return_value=sign_flipped):
                result = construction.construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=ids)
            self.assertEqual(result.construction_state, "constructed")
            self.assertEqual({frozenset(region.member_ids) for region in result.surface_regions.regions}, {frozenset(region.member_ids) for region in baseline.surface_regions.regions})
            self.assertEqual({frozenset(component.ordered_source_ids) for component in result.ordered_boundary_components}, {frozenset(component.ordered_source_ids) for component in baseline.ordered_boundary_components})
            self._assert_equivalent_surface(baseline, result, lambda values: values, 1.0)


if __name__ == "__main__":
    unittest.main()
