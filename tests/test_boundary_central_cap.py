import unittest

from nurbs_constructor_benchmark.boundary_first_support import construct_boundary_first_support
from nurbs_constructor_benchmark.scenes import make_scene


class BoundaryCentralCapTest(unittest.TestCase):
    def test_triangle_uses_observed_anchor_central_cap(self):
        result = construct_boundary_first_support(make_scene('triangle', 600, seed=0))
        visible = result.visible_results[0]
        self.assertEqual(visible.state, 'constructed')
        self.assertEqual(visible.topology, 'boundary_role_network')
        self.assertEqual(visible.provenance['boundary_roles'], ['outer_boundary', 'interior_anchor'])
        self.assertEqual(visible.provenance['anchor']['source_kind'], 'observed_max_support_clearance')
        self.assertGreaterEqual(visible.provenance['anchor_ray_support_coverage'], 0.99)
        self.assertTrue(visible.provenance['pole_singularity'].startswith('explicit_'))

    def test_central_cap_reports_pole_aware_regularity(self):
        from osn_gs.surface.torch_boundary_surface_quality import measure_boundary_first_surface_quality
        result = construct_boundary_first_support(make_scene('triangle', 600, seed=0))
        quality = measure_boundary_first_surface_quality(result.visible_results[0].surface_result)
        self.assertEqual(quality.minimum_jacobian_norm, 0.0)
        self.assertIsNotNone(quality.pole_excluded_minimum_jacobian_norm)
        self.assertGreater(quality.pole_excluded_minimum_jacobian_norm, 0.0)
    def test_concave_u_shape_is_not_filled_by_central_fan(self):
        result = construct_boundary_first_support(make_scene('u_shape', 600, seed=0))
        visible = result.visible_results[0]
        self.assertEqual(visible.state, 'unsupported')
        self.assertEqual(visible.reason, 'interior_support_crosses_unobserved_region')
        self.assertLess(visible.provenance['anchor_ray_support_coverage'], 0.99)


if __name__ == '__main__':
    unittest.main()