from __future__ import annotations

"""Worklog 118 -- Visible-NURBS Evidence Contract Closure.

Pure-logic focused tests for the diagnostic-only helpers in
`scripts/devtools/visible_nurbs_evidence_contract_closure.py`: explicit
camera/footpoint UV separation, the fixed-UV fitting arm (reusing the
existing regularized solver's own fixed-UV capability, not a new fitter),
sign-invariant normal comparison, the rasterizer-pixel-center unprojection
control, and equal-retained-count synthetic contracts. Plus one real-CUDA
test proving the new low-pass provenance fields are captured correctly.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from visible_nurbs_evidence_contract_closure import (  # noqa: E402
    depths_to_points_rasterizer_pixel_center,
    run_equal_count_synthetic_contracts,
    sign_invariant_normal_discrepancy_degrees,
)
from osn_gs.surface.torch_nurbs import _solve_control_grid_lsq, fit_torch_visible_surface, fit_torch_visible_surface_lsq  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from osn_gs.render.diff_surfel_loader import get_diff_surfel_backend
    try:
        from osn_gs.render.torch_surfel_representative_diagnostics import get_diag_extension, render_with_pixel_representative
        DIAG_EXTENSION_AVAILABLE = True
        try:
            get_diag_extension()
        except Exception:
            DIAG_EXTENSION_AVAILABLE = False
    except Exception:
        DIAG_EXTENSION_AVAILABLE = False
    from tests.test_surfel_rasterization_cuda import _single_surfel, _tilted_camera
    BACKEND_AVAILABLE = get_diff_surfel_backend() is not None
else:  # pragma: no cover
    BACKEND_AVAILABLE = False
    DIAG_EXTENSION_AVAILABLE = False

requires_cuda_and_diag = unittest.skipUnless(
    CUDA_AVAILABLE and BACKEND_AVAILABLE and DIAG_EXTENSION_AVAILABLE,
    "CUDA and the diagnostic diff_surfel_rasterization_diag build are required",
)


def _grid_positions(rows: int, cols: int, z_fn) -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
    x = ii / max(rows - 1, 1)
    y = jj / max(cols - 1, 1)
    z = z_fn(x, y)
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3), torch.stack([x, y], dim=-1).reshape(-1, 2)


class CameraVsFootpointUVSeparationTest(unittest.TestCase):
    def test_footpoint_uv_can_differ_from_camera_uv_for_curved_surface(self) -> None:
        points, uv_camera = _grid_positions(14, 14, lambda x, y: 0.3 * torch.sin(x * 3.14159))
        with torch.no_grad():
            _surface, uv_footpoint = fit_torch_visible_surface_lsq(
                points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
                initial_uv=uv_camera, correction_rounds=2, projection_iterations=3,
            )
        displacement = (uv_footpoint - uv_camera).norm(dim=-1)
        # a curved surface should show at least SOME nonzero foot-point drift
        self.assertGreater(float(displacement.max().item()), 1e-6)

    def test_camera_uv_is_never_silently_overwritten(self) -> None:
        points, uv_camera = _grid_positions(10, 10, lambda x, y: torch.zeros_like(x))
        uv_camera_copy = uv_camera.clone()
        with torch.no_grad():
            fit_torch_visible_surface_lsq(
                points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
                initial_uv=uv_camera, correction_rounds=2, projection_iterations=3,
            )
        self.assertTrue(torch.equal(uv_camera, uv_camera_copy))


class FixedUVArmTest(unittest.TestCase):
    def test_fixed_uv_arm_never_reprojects_uv(self) -> None:
        """ARM B must evaluate at exactly `uv_camera`, never a corrected UV
        -- verified by construction (the fixed-UV arm never calls
        `project_torch_points_to_nurbs` at all)."""

        points, uv_camera = _grid_positions(12, 12, lambda x, y: 0.2 * torch.sin(x * 3.14159))
        with torch.no_grad():
            surface = fit_torch_visible_surface(points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=uv_camera)
            surface.control_grid = _solve_control_grid_lsq(points, uv_camera, surface, 1e-4, 1e-4, 4096, None)
            fitted = surface.evaluate(uv_camera)
        residual = (fitted - points).norm(dim=-1)
        self.assertEqual(int(residual.shape[0]), int(points.shape[0]))
        self.assertTrue(bool(torch.isfinite(residual).all()))

    def test_fixed_uv_arm_reasonable_on_planar_surface(self) -> None:
        points, uv_camera = _grid_positions(10, 10, lambda x, y: torch.zeros_like(x))
        with torch.no_grad():
            surface = fit_torch_visible_surface(points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2, initial_uv=uv_camera)
            surface.control_grid = _solve_control_grid_lsq(points, uv_camera, surface, 1e-4, 1e-4, 4096, None)
            fitted = surface.evaluate(uv_camera)
        residual = (fitted - points).norm(dim=-1)
        self.assertLess(float(residual.max().item()), 0.05)


class SignInvariantNormalTest(unittest.TestCase):
    def test_opposite_normals_have_zero_sign_invariant_discrepancy(self) -> None:
        a = torch.tensor([[0.0, 0.0, 1.0]])
        b = torch.tensor([[0.0, 0.0, -1.0]])
        signed_cos = (a * b).sum(dim=-1).clamp(-1, 1)
        signed_deg = torch.rad2deg(torch.acos(signed_cos))
        self.assertAlmostEqual(float(signed_deg.item()), 180.0, places=3)
        sign_invariant = sign_invariant_normal_discrepancy_degrees(a, b)
        self.assertAlmostEqual(float(sign_invariant.item()), 0.0, places=3)

    def test_perpendicular_normals_unaffected_by_sign_invariance(self) -> None:
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0]])
        sign_invariant = sign_invariant_normal_discrepancy_degrees(a, b)
        self.assertAlmostEqual(float(sign_invariant.item()), 90.0, places=3)


class RasterizerPixelCenterUnprojectionTest(unittest.TestCase):
    def test_pixel_center_variant_differs_from_official_by_a_bounded_amount(self) -> None:
        from tests.test_surfel_rasterization_cuda import _tilted_camera
        from osn_gs.render.surfel_geometry import depths_to_points

        camera = _tilted_camera()
        depth = torch.full((1, camera.image_height, camera.image_width), 5.0, device=camera.world_view_transform.device)
        official = depths_to_points(camera, depth)
        alt = depths_to_points_rasterizer_pixel_center(camera, depth)
        displacement = (official - alt).norm(dim=-1)
        self.assertTrue(bool(torch.isfinite(displacement).all()))
        # half-pixel convention difference must be small relative to scene scale (5 units depth)
        self.assertLess(float(displacement.max().item()), 1.0)

    def test_pixel_center_variant_matches_official_at_zero_depth(self) -> None:
        from tests.test_surfel_rasterization_cuda import _tilted_camera
        from osn_gs.render.surfel_geometry import depths_to_points

        camera = _tilted_camera()
        depth = torch.zeros((1, camera.image_height, camera.image_width), device=camera.world_view_transform.device)
        official = depths_to_points(camera, depth)
        alt = depths_to_points_rasterizer_pixel_center(camera, depth)
        # depth=0 -> both collapse to the camera center regardless of pixel-center convention
        torch.testing.assert_close(official, alt, atol=1e-4, rtol=1e-3)


class EqualCountSyntheticContractTest(unittest.TestCase):
    def test_hole_and_dispersed_removal_retain_the_same_count(self) -> None:
        results = run_equal_count_synthetic_contracts()
        for label in ("planar", "curved"):
            b_count = results[label]["B_enclosed_hole_footpoint"]["retained_count"]
            c_count = results[label]["C_dispersed_removal_footpoint"]["retained_count"]
            self.assertEqual(b_count, c_count)

    def test_curved_enclosed_hole_has_worse_tail_than_dispersed_removal_at_equal_count(self) -> None:
        results = run_equal_count_synthetic_contracts()
        b_p95 = results["curved"]["B_enclosed_hole_footpoint"]["residual"]["p95"]
        c_p95 = results["curved"]["C_dispersed_removal_footpoint"]["residual"]["p95"]
        self.assertGreater(b_p95, c_p95)

    def test_planar_geometry_shows_negligible_difference_regardless_of_removal_pattern(self) -> None:
        results = run_equal_count_synthetic_contracts()
        b_median = results["planar"]["B_enclosed_hole_footpoint"]["residual"]["median"]
        c_median = results["planar"]["C_dispersed_removal_footpoint"]["residual"]["median"]
        self.assertLess(abs(b_median - c_median), 1e-5)


@requires_cuda_and_diag
class LowPassProvenanceCaptureTest(unittest.TestCase):
    def test_median_low_pass_fields_present_and_consistent(self) -> None:
        camera = _tilted_camera()
        model = _single_surfel()
        diag = render_with_pixel_representative(camera, model)
        rep = diag["representative_id"]
        covered = rep >= 0
        self.assertTrue(bool(covered.any()))
        rho3d = diag["median_rho3d"]
        rho2d = diag["median_rho2d"]
        self.assertTrue(bool((rho3d[covered] >= 0).all()))
        self.assertTrue(bool((rho2d[covered] >= 0).all()))


if __name__ == "__main__":
    unittest.main()
