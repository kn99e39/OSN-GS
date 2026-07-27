import unittest

from nurbs_constructor_benchmark.boundary_first_support import construct_boundary_first_support
from nurbs_constructor_benchmark.scenes import make_scene
from osn_gs.surface.torch_boundary_surface_quality import measure_boundary_first_surface_quality


class BoundaryFirstSurfaceQualityTest(unittest.TestCase):
    def test_curved_annulus_boundary_samples_and_cyclic_seams_are_exact(self):
        support = construct_boundary_first_support(
            make_scene("curved_annulus", 600, seed=0), curve_count=8, samples_per_curve=8
        )
        visible = support.visible_results[0]
        quality = measure_boundary_first_surface_quality(visible.surface_result)
        self.assertTrue(quality.finite)
        self.assertEqual(quality.patch_count, 8)
        self.assertLessEqual(quality.boundary_sample_max_error, 1e-6)
        self.assertLessEqual(quality.seam_max_error, 1e-6)
        self.assertGreater(quality.minimum_jacobian_norm, 1e-6)

    def test_measurement_is_deterministic(self):
        def payload():
            support = construct_boundary_first_support(
                make_scene("curved_annulus", 600, seed=0), curve_count=8, samples_per_curve=8
            )
            return measure_boundary_first_surface_quality(support.visible_results[0].surface_result).payload()

        self.assertEqual(payload(), payload())


if __name__ == "__main__":
    unittest.main()
