from __future__ import annotations

"""Phase A patch-boundary data-contract tests."""

import json
import unittest

import torch

from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.surface.torch_patch_boundary import (
    BOUNDARY_RECONCILED_INTERNAL,
    build_rectangular_patch_edge,
    extract_trimmed_patch_boundaries,
    patch_boundaries_payload,
)


def _plane_surface(resolution: int = 6) -> TorchNURBSSurface:
    u = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    v = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    grid = torch.stack(
        (
            u[:, None].expand(resolution, resolution),
            v[None, :].expand(resolution, resolution),
            torch.zeros((resolution, resolution), dtype=torch.float64),
        ),
        dim=2,
    )
    return TorchNURBSSurface(grid, torch.ones((resolution, resolution), dtype=torch.float64))


class PatchBoundaryContractTest(unittest.TestCase):
    def test_trim_mask_becomes_ordered_oriented_loops_with_inner_isocurves(self):
        surface = _plane_surface()
        mask = torch.zeros((12, 12), dtype=torch.bool)
        mask[1:11, 1:11] = True
        mask[4:8, 4:8] = False
        surface.uv_support_mask = mask

        boundaries = extract_trimmed_patch_boundaries(7, surface)
        self.assertEqual([item.boundary_id for item in boundaries], ["p7:trim:0", "p7:trim:1"])
        self.assertEqual({item.orientation for item in boundaries}, {"ccw", "cw"})
        for boundary in boundaries:
            self.assertTrue(boundary.closed)
            self.assertEqual(boundary.interior_side, "left")
            torch.testing.assert_close(boundary.uv[0], boundary.uv[-1])
            torch.testing.assert_close(boundary.inner_uv[0], boundary.inner_uv[-1])
            self.assertTrue(bool(surface.support(boundary.inner_uv).all()))
            self.assertGreater(boundary.confidence["inner_distance_median"], 0.0)
            self.assertGreater(boundary.confidence["jacobian_min"], 0.0)

        repeated = extract_trimmed_patch_boundaries(7, surface)
        for left, right in zip(boundaries, repeated):
            self.assertEqual(left.boundary_id, right.boundary_id)
            torch.testing.assert_close(left.uv, right.uv)
            torch.testing.assert_close(left.inner_uv, right.inner_uv)

    def test_rectangular_edge_orientation_keeps_interior_on_left(self):
        surface = _plane_surface()
        u0 = build_rectangular_patch_edge(2, surface, "u0")
        u1 = build_rectangular_patch_edge(
            2,
            surface,
            "u1",
            state=BOUNDARY_RECONCILED_INTERNAL,
            adjacent_patch_id=3,
            adjacent_boundary_id="p3:edge:u0",
        )
        self.assertTrue(bool((u0.inner_uv[:, 0] > u0.uv[:, 0]).all()))
        self.assertTrue(bool((u1.inner_uv[:, 0] < u1.uv[:, 0]).all()))
        self.assertLess(float(u0.uv[0, 1]), 1.01)
        self.assertGreater(float(u0.uv[0, 1]), float(u0.uv[-1, 1]))
        self.assertEqual(u1.state, BOUNDARY_RECONCILED_INTERNAL)
        self.assertEqual(u1.adjacent_boundary_id, "p3:edge:u0")

    def test_boundary_first_state_preserves_trimmed_boundaries_and_knots(self):
        from nurbs_constructor_benchmark.boundary_first import construct_boundary_first, renderer_payload
        from nurbs_constructor_benchmark.scenes import make_scene

        state, patches = construct_boundary_first(make_scene("plane", 600, seed=0))
        self.assertEqual(len(state.component_boundaries), state.component_count)
        self.assertGreaterEqual(len(state.patch_boundaries), 1)
        self.assertIn("knots_u", patches[0])
        self.assertIn("knots_v", patches[0])
        self.assertEqual(
            patches[0]["boundary_ids"],
            sorted(boundary.boundary_id for boundary in state.patch_boundaries if boundary.patch_id == 0),
        )
        payload = renderer_payload(
            "plane", patches, [boundary.payload() for boundary in state.patch_boundaries]
        )
        self.assertEqual(payload["metadata"]["patch_boundary_count"], len(state.patch_boundaries))
        json.dumps(payload, allow_nan=False)

    def test_annulus_artificial_seams_are_explicitly_reconciled(self):
        from nurbs_constructor_benchmark.boundary_first import construct_boundary_first
        from nurbs_constructor_benchmark.scenes import make_scene

        state, _ = construct_boundary_first(make_scene("planar_hole", 600, seed=0))
        seam_records = [
            boundary for boundary in state.patch_boundaries
            if boundary.source_kind == "annulus_artificial_seam"
        ]
        self.assertEqual(len(seam_records), 2 * len(state.surface_patches))
        self.assertTrue(seam_records)
        self.assertTrue(all(boundary.state == BOUNDARY_RECONCILED_INTERNAL for boundary in seam_records))
        by_id = {boundary.boundary_id: boundary for boundary in seam_records}
        for boundary in seam_records:
            self.assertIn(boundary.adjacent_boundary_id, by_id)
            partner = by_id[boundary.adjacent_boundary_id]
            self.assertEqual(partner.adjacent_boundary_id, boundary.boundary_id)
    def test_boundary_payload_is_json_serializable(self):
        surface = _plane_surface()
        surface.uv_support_mask = torch.ones((8, 8), dtype=torch.bool)
        payload = patch_boundaries_payload(extract_trimmed_patch_boundaries(0, surface))
        encoded = json.dumps(payload, allow_nan=False)
        self.assertIn("p0:trim:0", encoded)
        self.assertIn("inner_uv", encoded)


if __name__ == "__main__":
    unittest.main()