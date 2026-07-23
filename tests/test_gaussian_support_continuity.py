from __future__ import annotations

"""Stage 3-R Gaussian-native continuity diagnostics tests."""

import json
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class GaussianSupportContinuityTest(unittest.TestCase):
    def _evaluate(self, means, covariances, region_a, region_b, opacities=None, config=None):
        from osn_gs.surface.torch_gaussian_support_continuity import (
            evaluate_gaussian_support_continuity,
        )

        if opacities is None:
            opacities = torch.ones(len(means))
        return evaluate_gaussian_support_continuity(
            region_a,
            region_b,
            torch.as_tensor(means, dtype=torch.float64),
            torch.as_tensor(covariances, dtype=torch.float64),
            torch.as_tensor(opacities, dtype=torch.float64),
            config,
        )

    @staticmethod
    def _two_patches(offset=(1.0, 0.0, 0.0)):
        base = torch.tensor(
            [[0.0, -0.05, 0.0], [0.0, 0.0, 0.0], [0.0, 0.05, 0.0]],
            dtype=torch.float64,
        )
        other = base + torch.tensor(offset, dtype=torch.float64)
        return torch.cat([base, other], dim=0), [0, 1, 2], [3, 4, 5]

    def test_isotropic_covariance_distance(self):
        means, region_a, region_b = self._two_patches()
        covariance = torch.eye(3, dtype=torch.float64).repeat(6, 1, 1) * 0.25
        result = self._evaluate(means, covariance, region_a, region_b)
        self.assertAlmostEqual(result.mahalanobis["one_sided_a"]["minimum"], 2.0, places=6)
        self.assertAlmostEqual(result.mahalanobis["pooled"]["minimum"], 2.0**0.5, places=6)

    def test_anisotropic_tangent_reach(self):
        means, region_a, region_b = self._two_patches()
        isotropic = torch.eye(3, dtype=torch.float64).repeat(6, 1, 1) * 0.01
        elongated = isotropic.clone()
        elongated[:, 0, 0] = 0.25
        iso = self._evaluate(means, isotropic, region_a, region_b)
        tangent = self._evaluate(means, elongated, region_a, region_b)
        self.assertLess(
            tangent.projected_reach["center_gap_over_directional_reach"]["minimum"],
            iso.projected_reach["center_gap_over_directional_reach"]["minimum"],
        )

    def test_normal_direction_separation(self):
        tangent_means, region_a, region_b = self._two_patches((0.5, 0.0, 0.0))
        normal_means, _, _ = self._two_patches((0.0, 0.0, 0.5))
        covariance = torch.diag(torch.tensor([0.04, 0.04, 0.0004], dtype=torch.float64)).repeat(6, 1, 1)
        tangent = self._evaluate(tangent_means, covariance, region_a, region_b)
        normal = self._evaluate(normal_means, covariance, region_a, region_b)
        self.assertGreater(
            normal.projected_reach["normal_reach_ratio"]["median"],
            tangent.projected_reach["normal_reach_ratio"]["median"],
        )

    def test_ellipsoid_overlap_and_non_overlap(self):
        close, region_a, region_b = self._two_patches((0.2, 0.0, 0.0))
        far, _, _ = self._two_patches((1.0, 0.0, 0.0))
        covariance = torch.eye(3, dtype=torch.float64).repeat(6, 1, 1) * 0.01
        overlap = self._evaluate(close, covariance, region_a, region_b)
        separated = self._evaluate(far, covariance, region_a, region_b)
        self.assertGreaterEqual(overlap.ellipsoid_overlap["k1"]["overlap_fraction"], 1.0)
        self.assertEqual(separated.ellipsoid_overlap["k1"]["overlap_fraction"], 0.0)

    def test_bridge_density_continuous_case(self):
        means = torch.tensor(
            [[-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        covariance = torch.diag(torch.tensor([0.09, 0.0025, 0.0025], dtype=torch.float64)).repeat(5, 1, 1)
        result = self._evaluate(means, covariance, [0, 1, 2], [3, 4])
        ratio = result.bridge_density["unweighted"]["endpoint_minimum_ratio"]["median"]
        self.assertGreater(ratio, 0.7)

    def test_bridge_density_disconnected_case(self):
        means = torch.tensor(
            [[-1.0, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        covariance = torch.eye(3, dtype=torch.float64).repeat(4, 1, 1) * 0.0025
        result = self._evaluate(means, covariance, [0, 1], [2, 3])
        ratio = result.bridge_density["unweighted"]["endpoint_minimum_ratio"]["median"]
        self.assertLess(ratio, 0.1)

    def test_opacity_weighting_is_recorded_independently(self):
        means = torch.tensor(
            [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        covariance = torch.diag(torch.tensor([0.04, 0.0025, 0.0025], dtype=torch.float64)).repeat(4, 1, 1)
        result = self._evaluate(means, covariance, [0, 1], [2, 3], [1.0, 0.01, 1.0, 1.0])
        self.assertIn("unweighted", result.bridge_density)
        self.assertIn("opacity_weighted", result.bridge_density)
        self.assertNotEqual(
            result.bridge_density["unweighted"]["minimum"]["median"],
            result.bridge_density["opacity_weighted"]["minimum"]["median"],
        )

    def test_degenerate_covariance_handling(self):
        means, region_a, region_b = self._two_patches()
        covariance = torch.zeros((6, 3, 3), dtype=torch.float64)
        result = self._evaluate(means, covariance, region_a, region_b)
        self.assertTrue(result.validity_flags["valid"])
        self.assertEqual(result.validity_flags["covariance_eigenvalue_floored_count"], 6)
        self.assertTrue(result.validity_flags["principal_axis_ambiguous_count"] > 0)

    def test_deterministic_output(self):
        means, region_a, region_b = self._two_patches()
        covariance = torch.eye(3, dtype=torch.float64).repeat(6, 1, 1) * 0.01
        first = self._evaluate(means, covariance, region_a, region_b).payload()
        second = self._evaluate(means, covariance, list(reversed(region_a)), list(reversed(region_b))).payload()
        self.assertEqual(
            json.dumps(first, sort_keys=True, allow_nan=False),
            json.dumps(second, sort_keys=True, allow_nan=False),
        )

    def test_scale_rotation_covariance_matches_3dgs_convention(self):
        from osn_gs.surface.torch_gaussian_support_continuity import (
            covariance_from_scale_rotation,
        )

        scales = torch.tensor([[2.0, 1.0, 0.5]])
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        covariance = covariance_from_scale_rotation(scales, identity)
        self.assertTrue(torch.allclose(covariance[0], torch.diag(torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64))))

    def test_production_component_membership_is_not_touched(self):
        from nurbs_constructor_benchmark.scenes import make_scene
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        scene = make_scene("plane", 120, 0)
        hierarchy = build_voxel_gaussian_hierarchy(
            scene.points,
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=50,
            voxel_max_depth=4,
        )
        before = build_surface_components(hierarchy, scene.points)
        leaves = [
            leaf
            for leaf in hierarchy.leaves()
            if leaf.gaussian_indices is not None and int(leaf.gaussian_indices.numel()) > 0
        ]
        covariance = torch.eye(3).repeat(scene.points.shape[0], 1, 1) * 0.01
        self._evaluate(
            scene.points,
            covariance,
            leaves[0].gaussian_indices,
            leaves[1].gaussian_indices,
        )
        after = build_surface_components(hierarchy, scene.points)
        self.assertEqual(before.leaf_component_id, after.leaf_component_id)
        self.assertEqual(before.component_count(), after.component_count())


if __name__ == "__main__":
    unittest.main()
