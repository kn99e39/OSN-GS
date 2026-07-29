from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_boundary_review_geometry import (
    REPRESENTATION_CONTROL_POLYGON,
    REPRESENTATION_EVALUATED_CURVE,
    CROSSING_NEAR_TOUCHING,
    CROSSING_NO_CROSSING,
    CROSSING_NOT_CHECKED,
    CROSSING_OVERLAPPING_SUPPORT_PATH,
    CROSSING_SCOPE_REPRESENTATIVE_BUNDLE,
    CROSSING_TRANSVERSAL_INTERSECTION,
    CROSSING_VALID_SHARED_ENDPOINT,
    CROSSING_VALID_SHARED_POLE,
    NOT_YET_CHECKED_CROSSING_CATEGORIES,
    ReviewGeometryEntity,
    classify_support_curve_pair,
    combine_ordered_patch_boundary,
    control_polygon_entity,
    detect_support_curve_crossings,
    evaluate_interior_iso_curve,
    evaluate_iso_edge,
)
from osn_gs.surface.torch_nurbs import TorchNURBSSurface


def _cubic_u_surface() -> TorchNURBSSurface:
    """A patch whose U-direction (degree 3) control polygon is a visible S-curve."""
    curved_row = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, -2.0, 0.0], [3.0, 0.0, 0.0]])
    grid = torch.stack((curved_row, curved_row + torch.tensor([0.0, 0.0, 1.0])), dim=1)
    weights = torch.ones((4, 2))
    return TorchNURBSSurface(control_grid=grid, weights=weights, degree_u=3, degree_v=1)


def _pole_fan_surface(angle: float) -> TorchNURBSSurface:
    """A degenerate degree-1 spoke surface: any fixed U gives the same pole (v=0) -> tip (v=1) line."""
    pole = torch.tensor([0.0, 0.0, 0.0])
    tip = torch.tensor([2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0])
    v0_row = torch.stack((pole, pole))
    v1_row = torch.stack((tip, tip))
    grid = torch.stack((v0_row, v1_row), dim=1)
    weights = torch.ones((2, 2))
    return TorchNURBSSurface(control_grid=grid, weights=weights, degree_u=1, degree_v=1)


class ControlVsEvaluatedGeometryTest(unittest.TestCase):
    def test_cubic_control_polygon_and_evaluated_curve_are_not_the_same_geometry(self):
        surface = _cubic_u_surface()
        control = control_polygon_entity(surface, "v0", patch_id=0, role="outer_boundary", entity_id="p0:control")
        evaluated = evaluate_iso_edge(surface, "v0", samples=5, patch_id=0, role="outer_boundary", entity_id="p0:evaluated")
        self.assertEqual(control.representation_kind, REPRESENTATION_CONTROL_POLYGON)
        self.assertEqual(evaluated.representation_kind, REPRESENTATION_EVALUATED_CURVE)
        # Endpoints are shared by any clamped NURBS curve regardless of degree.
        torch.testing.assert_close(evaluated.points[0], control.points[0])
        torch.testing.assert_close(evaluated.points[-1], control.points[-1])
        # The cubic interior is NOT any raw interior control point -- proves the
        # exporter cannot treat a degree>=2 control polygon as the actual curve.
        midpoint = evaluated.points[2]
        self.assertFalse(torch.allclose(midpoint, control.points[1], atol=1e-4))
        self.assertFalse(torch.allclose(midpoint, control.points[2], atol=1e-4))
        expected_midpoint = torch.tensor([1.5, 0.0, 0.0])
        torch.testing.assert_close(midpoint, expected_midpoint, atol=1e-4, rtol=0.0)

    def test_linear_axis_evaluated_edge_matches_control_points_at_shared_parameters(self):
        surface = _cubic_u_surface()
        # v is degree 1 with only two control rows, so a 2-sample evaluation of a
        # fixed-u edge reduces exactly to the two control points on that column.
        control = control_polygon_entity(surface, "u0", patch_id=0, role="support_curve", entity_id="p0:control")
        evaluated = evaluate_iso_edge(surface, "u0", samples=2, patch_id=0, role="support_curve", entity_id="p0:evaluated")
        torch.testing.assert_close(evaluated.points, control.points, atol=1e-5, rtol=0.0)

    def test_combine_ordered_patch_boundary_drops_shared_junction_and_orders_patches(self):
        entity_a = ReviewGeometryEntity(
            entity_id="a", representation_kind=REPRESENTATION_EVALUATED_CURVE, role="outer_boundary",
            points=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]), patch_ids=(0,),
        )
        entity_b = ReviewGeometryEntity(
            entity_id="b", representation_kind=REPRESENTATION_EVALUATED_CURVE, role="outer_boundary",
            points=torch.tensor([[2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]]), patch_ids=(1,),
        )
        combined = combine_ordered_patch_boundary([entity_a, entity_b], entity_id="combined", role="outer_boundary", closed=False)
        self.assertEqual(combined.patch_ids, (0, 1))
        torch.testing.assert_close(
            combined.points,
            torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]]),
        )

    def test_combine_ordered_patch_boundary_dedups_closed_loop_wraparound(self):
        entity_a = ReviewGeometryEntity(
            entity_id="a", representation_kind=REPRESENTATION_EVALUATED_CURVE, role="outer_boundary",
            points=torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]), patch_ids=(0,),
        )
        entity_b = ReviewGeometryEntity(
            entity_id="b", representation_kind=REPRESENTATION_EVALUATED_CURVE, role="outer_boundary",
            points=torch.tensor([[1.0, 0, 0], [0.0, 0, 0]]), patch_ids=(1,),
        )
        combined = combine_ordered_patch_boundary([entity_a, entity_b], entity_id="combined", role="outer_boundary", closed=True)
        torch.testing.assert_close(combined.points, torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]))

    def test_combine_ordered_patch_boundary_rejects_control_polygon_input(self):
        control = ReviewGeometryEntity(
            entity_id="c", representation_kind=REPRESENTATION_CONTROL_POLYGON, role="outer_boundary",
            points=torch.zeros((4, 3)), patch_ids=(0,),
        )
        with self.assertRaises(ValueError):
            combine_ordered_patch_boundary([control], entity_id="x", role="outer_boundary", closed=False)


class SupportCurveCrossingTest(unittest.TestCase):
    def test_valid_shared_pole_is_not_flagged(self):
        pole = [0.0, 0.0, 0.0]
        curve_a = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        curve_b = [[0.0, 0, 0], [0.0, 1, 0], [0.0, 2, 0]]
        result = detect_support_curve_crossings([curve_a, curve_b], expected_shared_point=pole, expected_shared_kind="pole")
        self.assertEqual(result["state"], "checked")
        self.assertEqual(result["scope"], CROSSING_SCOPE_REPRESENTATIVE_BUNDLE)
        self.assertEqual(result["not_checked_categories"], list(NOT_YET_CHECKED_CROSSING_CATEGORIES))
        self.assertFalse(result["has_invalid_crossing"])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_VALID_SHARED_POLE)

    def test_valid_shared_boundary_endpoint_is_not_flagged(self):
        curve_a = [[5.0, 5, 5], [1.0, 0, 0], [0.0, 0, 0]]
        curve_b = [[0.0, 0, 0], [1.0, 1, 1], [2.0, 2, 2]]
        result = classify_support_curve_pair(
            curve_a, curve_b, tolerance_scale=1.0, expected_shared_point=[0.0, 0.0, 0.0], expected_shared_kind="boundary_endpoint",
        )
        self.assertEqual(result["classification"], CROSSING_VALID_SHARED_ENDPOINT)

    def test_transversal_intersection_is_detected(self):
        curve_a = [[0.0, 0, 0], [1.0, 1, 0], [2.0, 2, 0]]
        curve_b = [[0.0, 2, 0], [1.0, 1, 0], [2.0, 0, 0]]
        result = detect_support_curve_crossings([curve_a, curve_b])
        self.assertTrue(result["has_invalid_crossing"])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_TRANSVERSAL_INTERSECTION)

    def test_overlapping_support_path_is_distinguished_from_a_point_crossing(self):
        # Two curves that run alongside each other over the whole interior
        # (the real `plane`-scene defect: segments nearly coincide, not a
        # single transversal crossing point).
        curve_a = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]]
        curve_b = [[0.0, 0.01, 0], [1.0, 0.01, 0], [2.0, 0.01, 0], [3.0, 0.01, 0], [4.0, 0.01, 0]]
        result = detect_support_curve_crossings([curve_a, curve_b])
        self.assertTrue(result["has_invalid_crossing"])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_OVERLAPPING_SUPPORT_PATH)

    def test_too_short_to_distinguish_stays_ambiguous_not_invalid(self):
        # Only one interior sample on each side of a shared pole -- not enough
        # resolution to tell a point-crossing from an overlapping run, so this
        # must stay ambiguous (non-blocking) rather than guess "invalid".
        pole = [0.0, 0.0, 0.0]
        curve_a = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        curve_b = [[0.0, 0, 0], [1.0, 1e-4, 0], [2.0, 1, 0]]
        result = classify_support_curve_pair(
            curve_a, curve_b, tolerance_scale=1.0, expected_shared_point=pole, expected_shared_kind="pole",
        )
        self.assertEqual(result["classification"], CROSSING_NEAR_TOUCHING)

    def test_near_touching_curves_are_ambiguous_not_silently_accepted(self):
        curve_a = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        curve_b = [[0.0, 0.1, 0], [1.0, 0.1, 0], [2.0, 0.1, 0]]
        result = detect_support_curve_crossings([curve_a, curve_b])
        self.assertFalse(result["has_invalid_crossing"])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_NEAR_TOUCHING)

    def test_far_apart_curves_report_no_crossing(self):
        curve_a = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        curve_b = [[0.0, 5, 0], [1.0, 5, 0], [2.0, 5, 0]]
        result = detect_support_curve_crossings([curve_a, curve_b])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_NO_CROSSING)
        self.assertFalse(result["has_invalid_crossing"])

    def test_fewer_than_two_curves_is_not_checked(self):
        result = detect_support_curve_crossings([[[0.0, 0, 0], [1.0, 0, 0]]])
        self.assertEqual(result["state"], CROSSING_NOT_CHECKED)
        self.assertEqual(result["not_checked_categories"], list(NOT_YET_CHECKED_CROSSING_CATEGORIES))

    def test_explicit_tolerance_scale_is_resolution_independent(self):
        """Classification must not flip just because curves were sampled finer.

        Uses the SAME fixed geometric tolerance_scale (never derived from the
        curves' own sample spacing) while re-evaluating the actual surfaces at
        several resolutions -- the whole point of separating tolerance from
        sampling accuracy.
        """
        surface_a = _pole_fan_surface(0.0)
        surface_b = _pole_fan_surface(0.05)  # a near-overlapping spoke, close to surface_a's direction
        fixed_scale = 2.0
        classifications = set()
        for samples in (8, 16, 32, 64):
            curve_a = evaluate_interior_iso_curve(
                surface_a, fixed_direction="u", fixed_value=0.0, samples=samples,
                patch_id=0, role="support_curve", entity_id="a",
            ).points
            curve_b = evaluate_interior_iso_curve(
                surface_b, fixed_direction="u", fixed_value=0.0, samples=samples,
                patch_id=1, role="support_curve", entity_id="b",
            ).points
            result = classify_support_curve_pair(
                curve_a, curve_b, tolerance_scale=fixed_scale,
                expected_shared_point=[0.0, 0.0, 0.0], expected_shared_kind="pole",
            )
            classifications.add(result["classification"])
        self.assertEqual(len(classifications), 1, f"classification changed across sampling resolutions: {classifications}")


if __name__ == "__main__":
    unittest.main()
