import math
import unittest

import torch

from nurbs_constructor_benchmark.boundary_first_support import construct_boundary_first_support
from nurbs_constructor_benchmark.scenes import make_scene
from osn_gs.surface.torch_boundary_central_cap import (
    ObservedInteriorAnchor,
    _resample_closed_by_angle,
    build_boundary_central_cap,
    validate_star_shaped_boundary,
)
from osn_gs.surface.torch_boundary_support_network import ObservedBoundaryCurve


def _circle(radius: float, count: int = 64, center: tuple = (0.0, 0.0, 0.0)) -> torch.Tensor:
    cx, cy, cz = center
    return torch.tensor(
        [
            [cx + radius * math.cos(2 * math.pi * k / count), cy + radius * math.sin(2 * math.pi * k / count), cz]
            for k in range(count)
        ]
    )


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


class StarShapedBoundaryCorrespondenceTest(unittest.TestCase):
    """worklog 112: equal-angle, star-shape-validated boundary correspondence."""

    def test_centered_circle_is_star_shaped_and_resamples_at_even_angles(self):
        boundary = _circle(2.0, count=64)
        pole = torch.zeros(3)
        result = validate_star_shaped_boundary(boundary, pole)
        self.assertTrue(result['is_valid'])
        self.assertGreaterEqual(result['monotonicity_ratio'], 0.99)
        samples = _resample_closed_by_angle(boundary, pole, 8, direction=result['direction'])
        relative = samples - pole
        angles = torch.atan2(relative[:, 1], relative[:, 0])
        diffs = angles[1:] - angles[:-1]
        wrapped = (diffs + math.pi) % (2 * math.pi) - math.pi
        for step in wrapped:
            self.assertAlmostEqual(float(step.abs()), math.radians(45.0), delta=0.05)

    def test_anchor_outside_loop_is_rejected_as_non_star_shaped(self):
        # A small circle far from the pole subtends only a narrow angular
        # wedge -- walking it never sweeps anywhere near a full turn around
        # the anchor, so this must fail on total angular sweep, not be
        # silently angle-sorted into something misleading.
        boundary = _circle(1.0, count=64, center=(5.0, 0.0, 0.0))
        pole = torch.zeros(3)
        result = validate_star_shaped_boundary(boundary, pole)
        self.assertFalse(result['is_valid'])
        self.assertLess(result['total_angular_sweep_degrees'], result['thresholds']['min_total_sweep_degrees'])

    def test_validation_is_deterministic(self):
        boundary = _circle(2.0, count=48)
        pole = torch.zeros(3)
        first = validate_star_shaped_boundary(boundary, pole)
        second = validate_star_shaped_boundary(boundary, pole)
        self.assertEqual(first, second)

    def test_non_star_shaped_anchor_boundary_pair_is_rejected_without_synthetic_fallback(self):
        boundary = _circle(1.0, count=64, center=(5.0, 0.0, 0.0))
        outer = ObservedBoundaryCurve('outer', boundary, True, 'observed_support', {})
        anchor = ObservedInteriorAnchor(torch.zeros(3), 0, {'source_kind': 'observed_max_support_clearance'})
        cap = build_boundary_central_cap(outer, anchor, segment_count=8)
        self.assertEqual(cap.state, 'unsupported')
        self.assertEqual(cap.reason, 'insufficient_observed_interior_support')
        self.assertEqual(cap.surfaces, ())
        self.assertIn('star_shape_validation', cap.provenance)
        self.assertFalse(cap.provenance['star_shape_validation']['is_valid'])

    def test_star_shaped_pair_materializes_with_equal_angle_correspondence_provenance(self):
        boundary = _circle(2.0, count=64)
        outer = ObservedBoundaryCurve('outer', boundary, True, 'observed_support', {})
        anchor = ObservedInteriorAnchor(torch.zeros(3), 0, {'source_kind': 'observed_max_support_clearance'})
        cap = build_boundary_central_cap(outer, anchor, segment_count=8)
        self.assertEqual(cap.state, 'constructed_central_cap')
        self.assertEqual(len(cap.surfaces), 8)
        self.assertEqual(cap.provenance['boundary_correspondence'], 'equal_angle_star_shaped')
        self.assertTrue(cap.provenance['star_shape_validation']['is_valid'])


if __name__ == '__main__':
    unittest.main()