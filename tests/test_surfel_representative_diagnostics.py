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

    def test_diagnostic_rendering_still_matches_canonical_after_contrib_provenance_addition(self):
        """Worklog 110: `out_contrib_ids`/`out_contrib_post_median`/
        `out_contrib_count` were threaded through the same forward.cu
        kernel a third time. Re-verify render invariance again."""

        camera = _tilted_camera()
        model = _single_surfel()
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        canonical = rasterizer.render(camera, model)
        diag = render_with_pixel_representative(camera, model)
        torch.testing.assert_close(diag["render"], canonical["render"].detach())

    def test_single_visible_surfel_is_pre_or_at_median_never_post(self):
        """A single surfel covering a pixel is trivially its own (only)
        accepted contributor there, so it is always at-or-before the
        median crossing (contrib_post_median=0) -- there is no earlier
        contributor to have already crossed T=0.5."""

        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        contrib_ids = diag["contrib_ids"]
        contrib_post_median = diag["contrib_post_median"]
        valid = contrib_ids >= 0
        self.assertTrue(bool(valid.any()))
        self.assertTrue(bool((contrib_ids[valid] == 0).all()))
        self.assertTrue(bool((contrib_post_median[valid] == 0).all()))

    def test_occluded_weakly_transmitted_contributor_is_post_median(self):
        """Direct traversal-order evidence for the same fixture as
        `test_occluded_but_weakly_transmitted_surfel_can_still_be_forward_
        accepted`: the near surfel (id 0) is accepted at-or-before the
        median (it crosses/is the median itself, contrib_post_median=0);
        the far, visually-occluded surfel (id 1) is accepted STRICTLY
        AFTER the median crossing (contrib_post_median=1) -- exactly the
        renderer-order distinction worklog 110 exists to make explicit,
        not surface-representative status."""

        model = _multi_flat_surfel([(2.0, 0.99, 4.0), (5.0, 0.99, 4.0)])
        camera = _identity_camera()
        diag = render_with_pixel_representative(camera, model)
        contrib_ids = diag["contrib_ids"]
        contrib_post_median = diag["contrib_post_median"]

        near_valid = contrib_ids == 0
        far_valid = contrib_ids == 1
        self.assertTrue(bool(near_valid.any()))
        self.assertTrue(bool(far_valid.any()))
        self.assertTrue(bool((contrib_post_median[near_valid] == 0).all()))
        self.assertTrue(bool((contrib_post_median[far_valid] == 1).all()))

    def test_contrib_count_matches_number_of_distinct_accepted_contributors(self):
        """`contrib_count` is the true, uncapped per-pixel accepted count --
        for the 2-surfel occlusion fixture (both accepted at every covered
        pixel), it must be exactly 2 wherever both surfels cover the pixel."""

        model = _multi_flat_surfel([(2.0, 0.99, 4.0), (5.0, 0.99, 4.0)])
        camera = _identity_camera()
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        contrib_count = diag["contrib_count"]
        covered = rep >= 0
        self.assertTrue(bool(covered.any()))
        self.assertTrue(bool((contrib_count[covered] == 2).all()))

    def test_diagnostic_rendering_still_matches_canonical_after_median_lowpass_provenance_addition(self):
        """Worklog 118: `median_rho3d`/`median_rho2d`/`median_s_u`/
        `median_s_v` were threaded through the same forward.cu kernel a
        fourth time, at the exact same T>0.5 site. Re-verify render
        invariance again."""

        camera = _tilted_camera()
        model = _single_surfel()
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        canonical = rasterizer.render(camera, model)
        diag = render_with_pixel_representative(camera, model)
        torch.testing.assert_close(diag["render"], canonical["render"].detach())

    def test_median_lowpass_fields_set_only_where_representative_exists(self):
        """`median_rho3d`/`median_rho2d` use the same -1 sentinel convention
        as `representative_id`: set at covered pixels, -1 at uncovered ones
        (never a stale/garbage value at pixels with no median event)."""

        camera = _tilted_camera()
        model = _single_surfel(scale=0.3)  # small footprint, most of the frame uncovered
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        rho3d = diag["median_rho3d"]
        rho2d = diag["median_rho2d"]
        covered = rep >= 0
        self.assertTrue(bool(covered.any()))
        self.assertTrue(bool((~covered).any()))
        self.assertTrue(bool((rho3d[~covered] == -1).all()))
        self.assertTrue(bool((rho2d[~covered] == -1).all()))
        self.assertTrue(bool((rho3d[covered] >= 0).all()))
        self.assertTrue(bool((rho2d[covered] >= 0).all()))

    def test_median_s_matches_the_depth_formula_at_the_same_event(self):
        """The captured surfel-local intersection `(s_u, s_v)` must be
        exactly the pair the SAME event's `median_depth` is derived from --
        for a surfel lying flat in a plane parallel to the image (the
        `_single_surfel` fixture, oriented via `_tilted_camera`), depth is
        an affine function of `(s_u, s_v)`; this test only asserts internal
        consistency (finite, matches rho3d = s_u^2 + s_v^2 within floating
        tolerance), not a specific numeric depth (fixture-geometry-
        dependent and already covered by other tests)."""

        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        rho3d = diag["median_rho3d"]
        s_u = diag["median_s_u"]
        s_v = diag["median_s_v"]
        covered = rep >= 0
        self.assertTrue(bool(covered.any()))
        reconstructed_rho3d = s_u[covered] ** 2 + s_v[covered] ** 2
        torch.testing.assert_close(reconstructed_rho3d, rho3d[covered], atol=1e-4, rtol=1e-3)
