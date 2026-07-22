from __future__ import annotations

"""Phase 2 Component-Level Boundary Extraction unit tests
(docs/Urgent_Work/OSN_GS_Final_Boundary_First_NURBS_Direction.md §Phase 2)."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _plane_with_hole(count: int = 800, hole_radius: float = 0.35, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    accepted = []
    remaining = count
    while remaining > 0:
        xy = torch.rand((count * 2, 2), generator=generator) * 2.0 - 1.0
        keep = xy.square().sum(dim=1).sqrt() >= hole_radius
        xy = xy[keep][:remaining]
        accepted.append(xy)
        remaining -= int(xy.shape[0])
    xy = torch.cat(accepted, dim=0)
    z = torch.zeros((xy.shape[0], 1))
    return torch.cat([xy, z], dim=1)


def _flat_plane(count: int = 800, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand((count, 2), generator=generator) * 2.0 - 1.0
    return torch.cat([xy, torch.zeros((count, 1))], dim=1)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ComponentBoundaryExtractionTest(unittest.TestCase):
    def _extract(self, points, **overrides):
        from osn_gs.surface.torch_component_boundary import extract_component_boundary
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        hierarchy = build_voxel_gaussian_hierarchy(
            points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6
        )
        component_set = build_surface_components(hierarchy, points)
        self.assertEqual(component_set.component_count(), 1, "fixture must yield a single component")
        component = component_set.components[0]
        kwargs = dict(resolution=64)
        kwargs.update(overrides)
        return extract_component_boundary(component, hierarchy, points, **kwargs)

    def test_flat_plane_has_no_hole_or_tiny_artifact(self):
        result = self._extract(_flat_plane())
        self.assertEqual(result.topology["outer_loop_count"], 1)
        self.assertEqual(result.topology["hole_count"], 0)
        self.assertEqual(result.topology["tiny_artifact_loop_count"], 0)

    def test_hole_is_recovered_as_single_significant_loop(self):
        result = self._extract(_plane_with_hole())
        self.assertEqual(result.topology["outer_loop_count"], 1)
        self.assertEqual(result.topology["hole_count"], 1)
        self.assertEqual(result.topology["tiny_artifact_loop_count"], 0)
        self.assertGreater(result.hole_loops[0].area_cells, 50)
        # The hole must be nested inside the single outer loop.
        self.assertEqual(result.hole_loops[0].nested_in_outer_label, result.outer_loops[0].label)

    def test_refined_support_is_subset_of_dilated_coarse_support(self):
        # refined_mask is ANDed against the gap-closed (dilated) coarse mask,
        # not the raw one -- that dilation is exactly what closes the
        # curved-component reprojection seams (see the function docstring).
        result = self._extract(_plane_with_hole())
        self.assertFalse(bool((result.refined_mask & ~result.coarse_mask_dilated).any()))

    def test_coarse_gap_closing_cells_zero_falls_back_to_raw_coarse_mask(self):
        # filter_boundary_leaf_eligibility=False isolates the PLAIN-mask
        # dilation toggle this test targets -- with eligibility filtering on
        # (the default since worklog 45-49), coarse_gap_closing_cells=0
        # alone would not disable dilation, since the eligibility path uses
        # its own separate eligibility_gap_closing_cells (default 1, needed
        # to keep hole/ring topology intact -- see extract_component_
        # boundary's docstring and worklog 49).
        result = self._extract(_plane_with_hole(), coarse_gap_closing_cells=0, filter_boundary_leaf_eligibility=False)
        self.assertTrue(torch.equal(result.coarse_mask, result.coarse_mask_dilated))
        self.assertFalse(bool((result.refined_mask & ~result.coarse_mask).any()))

    def test_false_fill_matches_manual_computation(self):
        result = self._extract(_plane_with_hole())
        manual = int((result.coarse_mask & ~result.refined_mask).sum())
        self.assertEqual(manual, result.topology["false_fill_cells"])

    def test_coarse_gap_closing_does_not_alter_density_field(self):
        # The dilation only touches the intermediate coarse->AND step; the
        # exported density_grid/threshold_field must be identical regardless
        # of coarse_gap_closing_cells.
        closed = self._extract(_plane_with_hole(), coarse_gap_closing_cells=2)
        open_ = self._extract(_plane_with_hole(), coarse_gap_closing_cells=0)
        self.assertTrue(torch.allclose(closed.density_grid, open_.density_grid))
        self.assertTrue(torch.equal(closed.threshold_field, open_.threshold_field))

    def test_hole_shrinks_smoothly_as_bandwidth_grows(self):
        # Sanity check that the threshold/bandwidth knobs actually move the
        # refined support monotonically for this scene (used during tuning).
        areas = []
        for bandwidth in (1.5, 2.0, 3.0):
            result = self._extract(_plane_with_hole(), density_bandwidth_multiplier=bandwidth)
            areas.append(result.topology["refined_support_cells"])
        self.assertEqual(areas, sorted(areas))

    def test_payload_serializes(self):
        import json

        from osn_gs.surface.torch_component_boundary import component_boundary_payload

        result = self._extract(_plane_with_hole())
        payload = component_boundary_payload(result)
        json.dumps(payload)  # must not raise
        self.assertEqual(payload["topology"]["hole_count"], 1)

    def test_boundary_world_points_lie_near_true_hole_radius(self):
        result = self._extract(_plane_with_hole(hole_radius=0.35))
        points = torch.tensor(result.hole_loops[0].boundary_world_points)
        radii = points[:, :2].norm(dim=1)
        # Grid resolution 64 over roughly [-1, 1]^2 gives ~0.03 cell size;
        # allow a generous tolerance for boundary-cell quantization.
        self.assertTrue(bool(((radii - 0.35).abs() < 0.15).all()))


if __name__ == "__main__":
    unittest.main()
