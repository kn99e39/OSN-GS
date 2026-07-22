from __future__ import annotations

"""Phase 3 Trimmed Component Correctness Baseline unit tests
(docs/Urgent_Work/OSN_GS_Final_Boundary_First_NURBS_Direction.md §Phase 3)."""

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
class TrimmedComponentFitterTest(unittest.TestCase):
    def _fit(self, points, **overrides):
        from osn_gs.surface.torch_component_boundary import extract_component_boundary
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_trimmed_component_fitter import fit_trimmed_component
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        hierarchy = build_voxel_gaussian_hierarchy(
            points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6
        )
        component_set = build_surface_components(hierarchy, points)
        self.assertEqual(component_set.component_count(), 1, "fixture must yield a single component")
        component = component_set.components[0]
        boundary = extract_component_boundary(component, hierarchy, points, resolution=64, density_threshold=3.0)
        kwargs = dict(resolution_u=12, resolution_v=12)
        kwargs.update(overrides)
        return fit_trimmed_component(component, points, boundary.frame, boundary.refined_mask, **kwargs)

    def test_fits_flat_plane_with_low_residual(self):
        result = self._fit(_flat_plane())
        self.assertLess(result.fit_metrics["point_to_surface_rms"], 0.01)
        self.assertFalse(result.fit_metrics["control_grid_collapsed"])
        self.assertGreater(result.fit_metrics["jacobian_min"], 0.0)
        self.assertEqual(result.fit_metrics["degenerate_fraction"], 0.0)

    def test_control_grid_crosses_the_hole_but_render_stays_trimmed(self):
        # Section 3.3: the control grid MAY cross the hole; topology comes
        # entirely from the trim mask, not from control-grid structure.
        result = self._fit(_plane_with_hole())
        n_u, n_v, _ = result.surface.control_grid.shape
        self.assertEqual(n_u * n_v, 12 * 12)  # untouched rectangular grid
        self.assertIsNotNone(result.surface.uv_support_mask)
        self.assertFalse(bool(result.surface.uv_support_mask.all()))  # some UV IS trimmed away

    def test_mask_hit_rate_is_high_after_footpoint_correction(self):
        result = self._fit(_plane_with_hole())
        self.assertGreater(result.fit_metrics["mask_hit_rate"], 0.9)

    def test_uv_support_mask_matches_phase2_refined_mask(self):
        from osn_gs.surface.torch_component_boundary import extract_component_boundary
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_trimmed_component_fitter import fit_trimmed_component
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        points = _plane_with_hole()
        hierarchy = build_voxel_gaussian_hierarchy(
            points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6
        )
        component_set = build_surface_components(hierarchy, points)
        component = component_set.components[0]
        boundary = extract_component_boundary(component, hierarchy, points, resolution=64, density_threshold=3.0)
        result = fit_trimmed_component(component, points, boundary.frame, boundary.refined_mask)
        self.assertTrue(torch.equal(result.surface.uv_support_mask, boundary.refined_mask))

    def test_jacobian_metrics_detect_a_healthy_flat_fit(self):
        from osn_gs.surface.torch_trimmed_component_fitter import _jacobian_metrics

        result = self._fit(_flat_plane())
        metrics = _jacobian_metrics(result.surface)
        self.assertGreater(metrics["jacobian_min"], 0.0)
        self.assertEqual(metrics["degenerate_fraction"], 0.0)

    def test_control_grid_metrics_report_extent_and_edges(self):
        from osn_gs.surface.torch_trimmed_component_fitter import _control_grid_metrics

        result = self._fit(_flat_plane())
        metrics = _control_grid_metrics(result.surface.control_grid)
        self.assertGreater(metrics["extent"], 0.0)
        self.assertGreater(metrics["edge_median"], 0.0)
        self.assertFalse(metrics["collapsed"])


if __name__ == "__main__":
    unittest.main()
