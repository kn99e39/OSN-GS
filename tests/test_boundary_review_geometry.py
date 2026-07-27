from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_boundary_review_geometry import (
    REPRESENTATION_CONTROL_POLYGON,
    REPRESENTATION_EVALUATED_CURVE,
    CROSSING_INVALID_INTERIOR,
    CROSSING_NEAR_TOUCHING,
    CROSSING_NO_CROSSING,
    CROSSING_VALID_SHARED_ENDPOINT,
    CROSSING_VALID_SHARED_POLE,
    ReviewGeometryEntity,
    classify_support_curve_pair,
    combine_ordered_patch_boundary,
    control_polygon_entity,
    detect_support_curve_crossings,
    evaluate_iso_edge,
)
from osn_gs.surface.torch_nurbs import TorchNURBSSurface


def _cubic_u_surface() -> TorchNURBSSurface:
    """A patch whose U-direction (degree 3) control polygon is a visible S-curve."""
    curved_row = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, -2.0, 0.0], [3.0, 0.0, 0.0]])
    grid = torch.stack((curved_row, curved_row + torch.tensor([0.0, 0.0, 1.0])), dim=1)
    weights = torch.ones((4, 2))
    return TorchNURBSSurface(control_grid=grid, weights=weights, degree_u=3, degree_v=1)


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
        self.assertFalse(result["has_invalid_crossing"])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_VALID_SHARED_POLE)

    def test_valid_shared_boundary_endpoint_is_not_flagged(self):
        curve_a = [[5.0, 5, 5], [1.0, 0, 0], [0.0, 0, 0]]
        curve_b = [[0.0, 0, 0], [1.0, 1, 1], [2.0, 2, 2]]
        result = classify_support_curve_pair(
            curve_a, curve_b, scale=1.0, expected_shared_point=[0.0, 0.0, 0.0], expected_shared_kind="boundary_endpoint",
        )
        self.assertEqual(result["classification"], CROSSING_VALID_SHARED_ENDPOINT)

    def test_invalid_interior_crossing_is_detected(self):
        curve_a = [[0.0, 0, 0], [1.0, 1, 0], [2.0, 2, 0]]
        curve_b = [[0.0, 2, 0], [1.0, 1, 0], [2.0, 0, 0]]
        result = detect_support_curve_crossings([curve_a, curve_b])
        self.assertTrue(result["has_invalid_crossing"])
        self.assertEqual(result["pairs"][0]["classification"], CROSSING_INVALID_INTERIOR)

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
        self.assertEqual(result["state"], "not_checked")


if __name__ == "__main__":
    unittest.main()
