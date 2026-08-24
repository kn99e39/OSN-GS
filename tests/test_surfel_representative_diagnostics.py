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

    def test_diagnostic_rendering_still_matches_canonical_after_forward_accepted_addition(self):
        """Worklog 108 continuation: `out_forward_accepted` was threaded
        through the same forward.cu kernel that produces the rendered
        image. Re-verify render invariance now that the diagnostic build
        has been extended a second time."""

        camera = _tilted_camera()
        model = _single_surfel()
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        canonical = rasterizer.render(camera, model)
        diag = render_with_pixel_representative(camera, model)
        torch.testing.assert_close(diag["render"], canonical["render"].detach())

    def test_visible_surfel_is_forward_accepted(self):
        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        self.assertEqual(int(diag["forward_accepted"][0].item()), 1)

    def test_occluded_but_weakly_transmitted_surfel_can_still_be_forward_accepted(self):
        """Direct, same-execution evidence for CAVEAT 1: `forward_accepted`
        is a strictly WEAKER condition than "became the representative".
        With opacity 0.99 (not fully opaque), the near surfel leaves a
        small residual transmittance (test_T ~ 0.02, still >= 0.0001), so
        the far, visually-occluded surfel still passes every forward
        acceptance check and weakly accumulates color (w = alpha*T > 0) --
        this is the exact `dchannel_dcolor = alpha*T > 0` condition worklog
        105's backward-gradient diagnostic already exploited. It is
        `forward_accepted == 1` here even though worklog 107's own
        `test_occluded_surfel_never_becomes_the_representative` (same
        fixture) confirms it never becomes the representative. This is
        real renderer semantics, not a fixture defect: many primitives can
        be forward-accepted contributors without ever being anyone's
        median-depth representative -- the asymmetry runs in that
        direction, never the reverse (see
        `test_representative_never_occurs_without_forward_accepted_same_execution`)."""

        model = _multi_flat_surfel([(2.0, 0.99, 4.0), (5.0, 0.99, 4.0)])
        camera = _identity_camera()
        diag = render_with_pixel_representative(camera, model)
        forward_accepted = diag["forward_accepted"]
        self.assertEqual(int(forward_accepted[0].item()), 1)
        self.assertEqual(int(forward_accepted[1].item()), 1)

    def test_representative_never_occurs_without_forward_accepted_same_execution(self):
        """CAVEAT 1's central question, checked directly on a fixture: can
        MEDIAN_SURFACE_REPRESENTATIVE occur without FORWARD_ACCEPTED_CONTRIBUTOR
        in the same forward execution? By construction (the acceptance flag
        is set unconditionally earlier in the same branch that conditionally
        sets the representative) it must not -- verified here, and again at
        real-scene scale in the audit script."""

        for camera, model in (
            (_tilted_camera(), _single_surfel()),
            (_identity_camera(), _multi_flat_surfel([(2.0, 0.99, 4.0), (5.0, 0.99, 4.0)])),
            (_identity_camera(), _flat_surfel(z=2.0, opacity=0.9, scale=3.0)),
        ):
            diag = render_with_pixel_representative(camera, model)
            rep = diag["representative_id"]
            forward_accepted = diag["forward_accepted"]
            represented_ids = torch.unique(rep[rep >= 0])
            for surfel_id in represented_ids.tolist():
                self.assertEqual(
                    int(forward_accepted[surfel_id].item()), 1,
                    f"surfel {surfel_id} is a representative but was not forward_accepted",
                )
