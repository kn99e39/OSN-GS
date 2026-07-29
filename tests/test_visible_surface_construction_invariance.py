from __future__ import annotations

import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


class VisibleSurfaceConstructionInvarianceTest(unittest.TestCase):
    def test_plane_and_curved_sheet_survive_rigid_scale_and_order_variants(self):
        import torch
        rotations = (
            torch.tensor([[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]]),
            torch.tensor([[0.60, 0.64, 0.48], [-0.80, 0.48, 0.36], [0.00, -0.60, 0.80]]),
            torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        for name in ("plane", "smooth_curved_sheet"):
            scene = make_gaussian_reliability_scene(name)
            ids = tuple(range(len(scene.positions)))
            baseline = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=ids)
            expected_members = {frozenset(region.member_ids) for region in baseline.surface_regions.regions}
            expected_boundary = {frozenset(component.ordered_source_ids) for component in baseline.ordered_boundary_components}
            order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(712))
            variants = [
                (scene.positions @ rotation.T, rotation @ scene.covariances @ rotation.T, ids)
                for rotation in rotations
            ]
            variants += [
                (scene.positions * scale, scene.covariances * scale ** 2, ids)
                for scale in (0.31, 4.2)
            ]
            variants += [
                (scene.positions[order], scene.covariances[order], tuple(ids[item] for item in order.tolist())),
                # Covariance is sign-equivalent by construction: eigenvector
                # signs are not part of the covariance representation.
                (scene.positions, scene.covariances.clone(), ids),
            ]
            for positions, covariance, variant_ids in variants:
                result = construct_visible_nurbs_from_gaussians(positions, covariance=covariance, stable_ids=variant_ids)
                self.assertEqual(result.construction_state, "constructed")
                self.assertEqual(len(result.materialized_visible_nurbs_surfaces), 1)
                self.assertEqual({frozenset(region.member_ids) for region in result.surface_regions.regions}, expected_members)
                self.assertEqual({frozenset(component.ordered_source_ids) for component in result.ordered_boundary_components}, expected_boundary)


if __name__ == "__main__":
    unittest.main()

