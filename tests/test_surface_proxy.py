from __future__ import annotations

"""Stage 1 diagnostics-only quadratic surface proxy tests."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _grid(x0: float, x1: float, nx: int = 11, ny: int = 9):
    x = torch.linspace(x0, x1, nx)
    y = torch.linspace(-0.6, 0.6, ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def _paraboloid(x0: float, x1: float):
    x, y = _grid(x0, x1)
    z = 0.18 * x.square() + 0.07 * x * y - 0.04 * y.square()
    return torch.stack([x, y, z], dim=1)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class QuadraticSurfaceProxyTest(unittest.TestCase):
    def test_exact_plane_has_near_zero_proxy_error(self):
        from osn_gs.surface.torch_surface_proxy import fit_quadratic_surface_proxy

        x, y = _grid(-1.0, 1.0)
        points = torch.stack([x, y, 0.2 * x - 0.1 * y + 0.3], dim=1)
        proxy = fit_quadratic_surface_proxy(points)
        self.assertTrue(proxy.valid)
        self.assertLess(proxy.world_rms_residual, 1e-7)
        self.assertLess(proxy.local_curvature_proxy, 1e-5)

    def test_exact_paraboloid_is_recovered_and_improves_over_plane(self):
        from osn_gs.surface.torch_surface_proxy import fit_quadratic_surface_proxy

        proxy = fit_quadratic_surface_proxy(_paraboloid(-1.0, 1.0))
        self.assertTrue(proxy.valid)
        self.assertLess(proxy.world_rms_residual, 2e-6)
        self.assertGreater(proxy.plane_world_rms_residual, proxy.world_rms_residual * 100.0)
        self.assertGreater(proxy.local_curvature_proxy, 0.01)

    def test_noisy_curved_sheet_stays_finite(self):
        from osn_gs.surface.torch_surface_proxy import fit_quadratic_surface_proxy

        generator = torch.Generator().manual_seed(7)
        points = _paraboloid(-1.0, 1.0)
        points[:, 2] += torch.randn(points.shape[0], generator=generator) * 0.002
        proxy = fit_quadratic_surface_proxy(points)
        self.assertTrue(proxy.valid)
        self.assertTrue(torch.isfinite(proxy.coefficients).all())
        self.assertLess(proxy.world_rms_residual, proxy.plane_world_rms_residual)

    def test_line_like_support_is_invalid(self):
        from osn_gs.surface.torch_surface_proxy import fit_quadratic_surface_proxy

        x = torch.linspace(-1.0, 1.0, 20)
        points = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
        proxy = fit_quadratic_surface_proxy(points)
        self.assertFalse(proxy.valid)
        self.assertEqual(proxy.invalid_reason, "degenerate_tangent_support")

    def test_evaluation_and_signed_residual(self):
        from osn_gs.surface.torch_surface_proxy import (
            evaluate_quadratic_proxy,
            fit_quadratic_surface_proxy,
            quadratic_proxy_signed_residuals,
        )

        points = _paraboloid(-1.0, 1.0)
        proxy = fit_quadratic_surface_proxy(points)
        projected = evaluate_quadratic_proxy(proxy, points)
        residual = quadratic_proxy_signed_residuals(proxy, points)
        self.assertEqual(tuple(projected.shape), tuple(points.shape))
        self.assertLess(float(residual.square().mean().sqrt()), 2e-6)

    def test_fit_is_deterministic(self):
        from osn_gs.surface.torch_surface_proxy import fit_quadratic_surface_proxy

        points = _paraboloid(-0.9, 0.8)
        first = fit_quadratic_surface_proxy(points)
        second = fit_quadratic_surface_proxy(points.clone())
        self.assertTrue(torch.equal(first.origin, second.origin))
        self.assertTrue(torch.equal(first.coefficients, second.coefficients))
        self.assertEqual(first.payload(), second.payload())


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ProxyMergeDiagnosticsTest(unittest.TestCase):
    def test_smooth_curved_pair_has_negligible_merge_cost(self):
        from osn_gs.surface.torch_surface_proxy import merge_proxy_diagnostics

        diagnostics = merge_proxy_diagnostics(
            _paraboloid(-1.0, 0.0), _paraboloid(0.0, 1.0)
        )
        self.assertTrue(diagnostics.valid)
        self.assertLess(diagnostics.normalized_error_increase, 1e-8)
        self.assertLess(diagnostics.scale_normalized_support_gap, 1.5)

    def test_crease_has_more_merge_error_than_smooth_pair(self):
        from osn_gs.surface.torch_surface_proxy import merge_proxy_diagnostics

        smooth = merge_proxy_diagnostics(
            _paraboloid(-1.0, 0.0), _paraboloid(0.0, 1.0)
        )
        xa, ya = _grid(-1.0, 0.0)
        xb, yb = _grid(0.0, 1.0)
        crease_a = torch.stack([xa, ya, -0.45 * xa], dim=1)
        crease_b = torch.stack([xb, yb, 0.45 * xb], dim=1)
        crease = merge_proxy_diagnostics(crease_a, crease_b)
        self.assertTrue(crease.valid)
        self.assertGreater(crease.normalized_error_increase, smooth.normalized_error_increase + 1e-5)
        self.assertGreater(crease.normal_angle_degrees, 20.0)

    def test_parallel_layers_are_exposed_by_layer_score(self):
        from osn_gs.surface.torch_surface_proxy import merge_proxy_diagnostics

        x, y = _grid(-1.0, 1.0)
        upper = torch.stack([x, y, torch.full_like(x, 0.06)], dim=1)
        lower = torch.stack([x, y, torch.full_like(x, -0.06)], dim=1)
        diagnostics = merge_proxy_diagnostics(upper, lower)
        self.assertTrue(diagnostics.valid)
        self.assertGreater(diagnostics.layer_separation_score, 0.5)
        self.assertGreater(diagnostics.normalized_error_increase, 1e-4)

    def test_disconnected_coplanar_patches_need_support_gap_signal(self):
        from osn_gs.surface.torch_surface_proxy import merge_proxy_diagnostics

        xa, ya = _grid(-1.0, -0.45)
        xb, yb = _grid(0.45, 1.0)
        left = torch.stack([xa, ya, torch.zeros_like(xa)], dim=1)
        right = torch.stack([xb, yb, torch.zeros_like(xb)], dim=1)
        diagnostics = merge_proxy_diagnostics(left, right)
        self.assertTrue(diagnostics.valid)
        self.assertLess(diagnostics.normalized_error_increase, 1e-10)
        self.assertGreater(diagnostics.scale_normalized_support_gap, 4.0)


if __name__ == "__main__":
    unittest.main()
