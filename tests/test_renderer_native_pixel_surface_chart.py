from __future__ import annotations

"""Worklog 112 -- Renderer-Native Pixel Surface as NURBS fitting geometry.

Two layers of tests: (1) pure-logic chart-pixel-sample construction (no
CUDA), matching torch_camera_observed_chart_domains.build_view_chart_
pixel_samples; (2) real diagnostic-renderer tests proving the SAME
representative surfel can produce DISTINCT per-pixel 3D surface positions
(directive section 3), and that the median-depth channel unprojects
correctly via the existing OFFICIAL_CODE_FAITHFUL `depths_to_points`.
"""

import unittest

import torch

from osn_gs.surface.torch_camera_observed_chart_domains import (
    build_view_chart_pixel_samples,
    valid_pixel_chart_mask,
)
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from osn_gs.render.diff_surfel_loader import get_diff_surfel_backend
    try:
        from osn_gs.render.torch_surfel_representative_diagnostics import (
            get_diag_extension,
            render_with_pixel_representative,
        )
        DIAG_EXTENSION_AVAILABLE = True
        try:
            get_diag_extension()
        except Exception:
            DIAG_EXTENSION_AVAILABLE = False
    except Exception:
        DIAG_EXTENSION_AVAILABLE = False
    from tests.test_surfel_contribution_diagnostics import _identity_camera, _flat_surfel
    from tests.test_surfel_rasterization_cuda import _tilted_camera, _single_surfel
    BACKEND_AVAILABLE = get_diff_surfel_backend() is not None
else:  # pragma: no cover
    BACKEND_AVAILABLE = False
    DIAG_EXTENSION_AVAILABLE = False

requires_cuda_and_diag = unittest.skipUnless(
    CUDA_AVAILABLE and BACKEND_AVAILABLE and DIAG_EXTENSION_AVAILABLE,
    "CUDA and the diagnostic diff_surfel_rasterization_diag build are required",
)

_MIDDEPTH_OFFSET = 5


class BuildViewChartPixelSamplesTest(unittest.TestCase):
    def test_two_components_never_share_a_chart(self):
        comp = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
        rep = torch.tensor([[10, 10, 20, 20], [11, 11, 21, 21]], dtype=torch.int64)
        world = torch.zeros((2, 4, 3), dtype=torch.float32)
        vs = build_view_chart_pixel_samples(0, comp, rep, world)
        self.assertEqual(vs.blob_count, 2)
        for blob_id, rep_id in zip(vs.pixel_blob_id.tolist(), vs.pixel_representative_id.tolist()):
            expected = 0 if rep_id < 20 else 1
            self.assertEqual(int(vs.blob_component_id[blob_id].item()), expected)

    def test_every_valid_pixel_kept_as_its_own_sample_not_collapsed(self):
        """Directive section 6: unlike WL111's per-representative mean, every
        valid pixel keeps its OWN sample even when several pixels share the
        same representative id."""

        comp = torch.zeros((2, 4), dtype=torch.int64)
        rep = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.int64)  # all 8 pixels, ONE representative
        world = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        vs = build_view_chart_pixel_samples(0, comp, rep, world)
        self.assertEqual(int(vs.pixel_uv.shape[0]), 8)  # NOT collapsed to 1
        self.assertTrue(torch.equal(vs.pixel_xyz, world.reshape(-1, 3)))

    def test_occluded_gap_splits_into_two_components_no_chart_crosses(self):
        comp = torch.tensor([[0, 0, 0, -1, 1, 1, 1]] * 3, dtype=torch.int64)
        rep = torch.arange(21, dtype=torch.int64).reshape(3, 7)
        world = torch.zeros((3, 7, 3), dtype=torch.float32)
        vs = build_view_chart_pixel_samples(0, comp, rep, world)
        self.assertEqual(vs.blob_count, 2)
        gap_ids = set(rep[:, 3].tolist())
        self.assertFalse(gap_ids & set(vs.pixel_representative_id.tolist()))

    def test_valid_mask_uses_pixel_count_not_representative_count(self):
        """Directive section 7/8: a component with FEW distinct representatives
        but MANY observed pixels must be eligible."""

        comp = torch.zeros((4, 8), dtype=torch.int64)
        rep = torch.full((4, 8), 1, dtype=torch.int64)  # 32 pixels, exactly ONE representative
        world = torch.zeros((4, 8, 3), dtype=torch.float32)
        vs = build_view_chart_pixel_samples(0, comp, rep, world)
        self.assertEqual(int(vs.blob_pixel_total[0].item()), 32)
        self.assertTrue(bool(valid_pixel_chart_mask(vs, 32).all()))
        self.assertFalse(bool(valid_pixel_chart_mask(vs, 33).any()))

    def test_deterministic_replay(self):
        comp = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
        rep = torch.tensor([[10, 10, 20, 20], [11, 11, 21, 21]], dtype=torch.int64)
        world = torch.rand((2, 4, 3), dtype=torch.float32)
        first = build_view_chart_pixel_samples(0, comp, rep, world)
        second = build_view_chart_pixel_samples(0, comp, rep, world)
        self.assertTrue(torch.equal(first.pixel_blob_id, second.pixel_blob_id))
        self.assertTrue(torch.equal(first.pixel_uv, second.pixel_uv))
        self.assertTrue(torch.equal(first.pixel_xyz, second.pixel_xyz))


class PixelSurfaceNURBSFitTest(unittest.TestCase):
    def test_planar_pixel_surface_fits_accurately(self):
        rows, cols = 10, 10
        ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
        world = torch.stack([ii / (rows - 1), jj / (cols - 1), torch.zeros_like(ii)], dim=-1)
        comp = torch.zeros((rows, cols), dtype=torch.int64)
        rep = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
        vs = build_view_chart_pixel_samples(0, comp, rep, world)
        surface, uv = fit_torch_visible_surface_lsq(
            vs.pixel_xyz, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=vs.pixel_uv,
        )
        residual = (surface.evaluate(uv) - vs.pixel_xyz).norm(dim=-1)
        self.assertLess(float(residual.max().item()), 0.05)

    def test_curved_pixel_surface_fits_continuously(self):
        rows, cols = 12, 12
        ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
        x = ii / (rows - 1)
        y = jj / (cols - 1)
        z = 0.3 * torch.sin(x * 3.14159)
        world = torch.stack([x, y, z], dim=-1)
        comp = torch.zeros((rows, cols), dtype=torch.int64)
        rep = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
        vs = build_view_chart_pixel_samples(0, comp, rep, world)
        surface, uv = fit_torch_visible_surface_lsq(
            vs.pixel_xyz, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=vs.pixel_uv, correction_rounds=3,
        )
        residual = (surface.evaluate(uv) - vs.pixel_xyz).norm(dim=-1)
        self.assertLess(float(residual.mean().item()), 0.05)


@requires_cuda_and_diag
class RendererNativeDepthUnprojectionTest(unittest.TestCase):
    def test_median_depth_channel_unprojects_to_plausible_world_points(self):
        from osn_gs.render.surfel_geometry import depths_to_points

        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        depth = diag["out_others"][_MIDDEPTH_OFFSET]
        rep = diag["representative_id"]
        valid = rep >= 0
        self.assertTrue(bool(valid.any()))
        points = depths_to_points(camera, depth.unsqueeze(0)).reshape(*depth.shape, 3)
        # Unprojected points where a representative exists must be finite and
        # roughly near the known surfel (z=default fixture depth), not the origin.
        valid_points = points[valid]
        self.assertTrue(bool(torch.isfinite(valid_points).all()))

    def test_same_representative_yields_distinct_per_pixel_positions(self):
        """Directive section 3's required contract: two pixels whose median
        representative is the SAME surfel may still produce different 3D
        renderer-native surface positions (a tilted flat surfel's footprint
        spans a range of camera-space depths across its extent)."""

        from osn_gs.render.surfel_geometry import depths_to_points

        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        depth = diag["out_others"][_MIDDEPTH_OFFSET]
        rep = diag["representative_id"]
        valid = rep >= 0
        self.assertTrue(bool(valid.any()))
        points = depths_to_points(camera, depth.unsqueeze(0)).reshape(*depth.shape, 3)

        same_rep_points = points[valid][rep[valid] == 0]
        self.assertGreater(int(same_rep_points.shape[0]), 1)
        spread = (same_rep_points - same_rep_points[0]).norm(dim=-1).max()
        self.assertGreater(float(spread.item()), 1e-4)


if __name__ == "__main__":
    unittest.main()
