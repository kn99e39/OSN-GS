from __future__ import annotations

import unittest

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from osn_gs.render.diff_surfel_loader import get_diff_surfel_backend
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
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
    from tests.test_surfel_contribution_diagnostics import _identity_camera, _flat_surfel, _multi_flat_surfel
    from tests.test_surfel_rasterization_cuda import _single_surfel, _tilted_camera

    BACKEND_AVAILABLE = get_diff_surfel_backend() is not None
else:  # pragma: no cover - environment dependent
    BACKEND_AVAILABLE = False
    DIAG_EXTENSION_AVAILABLE = False

requires_cuda_and_diag = unittest.skipUnless(
    CUDA_AVAILABLE and BACKEND_AVAILABLE and DIAG_EXTENSION_AVAILABLE,
    "CUDA, the vendored diff_surfel_rasterization extension, and the diagnostic "
    "diff_surfel_rasterization_diag build (scripts/build_surfel_extension_diag.bat) are required",
)


@requires_cuda_and_diag
class SurfelRepresentativeDiagnosticsTest(unittest.TestCase):
    def test_diagnostic_rendering_matches_canonical(self):
        """Directive section 5/3: the diagnostic build must not change the
        rendered image."""

        camera = _tilted_camera()
        model = _single_surfel()
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        canonical = rasterizer.render(camera, model)
        diag = render_with_pixel_representative(camera, model)
        torch.testing.assert_close(diag["render"], canonical["render"].detach())

    def test_single_surfel_is_its_own_representative_where_covered(self):
        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        covered = rep >= 0
        self.assertTrue(bool(covered.any()))
        self.assertTrue(bool((rep[covered] == 0).all()))

    def test_uncovered_pixels_have_no_representative(self):
        camera = _tilted_camera()
        model = _single_surfel(scale=0.3)  # small footprint, most of the frame uncovered
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        self.assertTrue(bool((rep == -1).any()))

    def test_occluded_surfel_never_becomes_the_representative(self):
        """A near, fully-covering, near-opaque surfel occludes a far one --
        the far surfel must never appear as any pixel's representative."""

        model = _multi_flat_surfel([(2.0, 0.99, 4.0), (5.0, 0.99, 4.0)])
        camera = _identity_camera()
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        self.assertTrue(bool((rep == 0).any()))
        self.assertFalse(bool((rep == 1).any()))

    def test_representative_id_is_deterministic(self):
        camera = _identity_camera()
        model = _flat_surfel(z=2.0, opacity=0.9, scale=3.0)
        first = render_with_pixel_representative(camera, model)["representative_id"]
        second = render_with_pixel_representative(camera, model)["representative_id"]
        self.assertTrue(torch.equal(first, second))

    def test_primitive_tensors_remain_unchanged(self):
        camera = _identity_camera()
        model = _flat_surfel(z=2.0, opacity=0.9, scale=3.0)
        xyz_before = model.get_xyz.detach().clone()
        render_with_pixel_representative(camera, model)
        torch.testing.assert_close(model.get_xyz.detach(), xyz_before)
        self.assertIsNone(model._xyz.grad)
