from __future__ import annotations

"""NURBS parametric-representation tests: derivatives, normals, foot-point projection."""

import unittest
from unittest import mock

import torch

import osn_gs.surface.torch_nurbs as torch_nurbs_module

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_nurbs import (
    SharedBoundaryConstraint,
    TorchNURBSSurface,
    boundary_control_indices,
    fit_coupled_patch_graph_lsq,
    fit_coupled_wedge_ring_lsq,
    fit_torch_visible_surface,
    fit_torch_visible_surface_lsq,
    project_torch_points_to_nurbs,
)


def _random_surface(
    n_u: int = 6,
    n_v: int = 5,
    degree_u: int = 2,
    degree_v: int = 2,
    dtype=torch.float64,
    seed: int = 3,
) -> TorchNURBSSurface:
    torch.manual_seed(seed)
    base_u = torch.linspace(0.0, 1.0, n_u, dtype=dtype)
    base_v = torch.linspace(0.0, 1.0, n_v, dtype=dtype)
    grid = torch.stack(
        [
            base_u[:, None].expand(n_u, n_v),
            base_v[None, :].expand(n_u, n_v),
            0.3 * torch.sin(base_u[:, None] * 4.0) * torch.cos(base_v[None, :] * 3.0),
        ],
        dim=-1,
    )
    grid = grid + 0.02 * torch.randn(n_u, n_v, 3, dtype=dtype)
    weights = 0.5 + torch.rand(n_u, n_v, dtype=dtype)
    return TorchNURBSSurface(
        control_grid=grid, weights=weights, degree_u=degree_u, degree_v=degree_v
    )


class NURBSDerivativeTest(unittest.TestCase):
    def test_knot_vectors_are_cached_and_invalidated_by_surface_structure(self):
        surface = _random_surface(n_u=6, n_v=5)
        uv = torch.rand(16, 2, dtype=torch.float64)

        with mock.patch.object(
            torch_nurbs_module,
            "_clamped_knot_vector",
            wraps=torch_nurbs_module._clamped_knot_vector,
        ) as knot_builder:
            expected = surface.evaluate(uv)
            self.assertEqual(knot_builder.call_count, 2)
            torch.testing.assert_close(surface.evaluate(uv), expected)
            self.assertEqual(knot_builder.call_count, 2)

            surface.degree_u = 3
            self.assertTrue(torch.isfinite(surface.evaluate(uv)).all())
            self.assertEqual(knot_builder.call_count, 4)

            surface.control_grid = surface.control_grid[:-1]
            surface.weights = surface.weights[:-1]
            self.assertTrue(torch.isfinite(surface.evaluate(uv)).all())
            self.assertEqual(knot_builder.call_count, 6)

    def test_cached_knot_vectors_preserve_autograd(self):
        surface = _random_surface()
        surface.control_grid.requires_grad_(True)
        surface.weights.requires_grad_(True)
        uv = torch.rand(32, 2, dtype=torch.float64, requires_grad=True)

        loss = surface.evaluate(uv).square().mean() + surface.evaluate(uv).abs().mean()
        loss.backward()
        self.assertIsNotNone(surface.control_grid.grad)
        self.assertIsNotNone(surface.weights.grad)
        self.assertIsNotNone(uv.grad)
        self.assertTrue(torch.isfinite(surface.control_grid.grad).all())
        self.assertTrue(torch.isfinite(surface.weights.grad).all())
        self.assertTrue(torch.isfinite(uv.grad).all())
        self.assertFalse(surface._cached_knots_u.requires_grad)
        self.assertFalse(surface._cached_knots_v.requires_grad)

    def test_analytic_derivatives_match_finite_differences(self):
        surface = _random_surface()
        torch.manual_seed(11)
        uv = 0.05 + 0.9 * torch.rand(64, 2, dtype=torch.float64)
        point, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
        self.assertTrue(torch.allclose(point, surface.evaluate(uv)))

        step = 1e-6
        offset_u = torch.tensor([step, 0.0], dtype=torch.float64)
        offset_v = torch.tensor([0.0, step], dtype=torch.float64)
        fd_u = (surface.evaluate(uv + offset_u) - surface.evaluate(uv - offset_u)) / (2 * step)
        fd_v = (surface.evaluate(uv + offset_v) - surface.evaluate(uv - offset_v)) / (2 * step)
        self.assertTrue(torch.allclose(deriv_u, fd_u, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(deriv_v, fd_v, atol=1e-4, rtol=1e-4))

    def test_public_knot_snapshots_cannot_mutate_cached_evaluation_knots(self):
        surface = _random_surface()
        expected = surface.evaluate(torch.rand(8, 2, dtype=torch.float64))
        public_u, public_v = surface.knot_vectors()
        public_u.fill_(0.25)
        public_v.fill_(0.75)
        self.assertEqual(float(surface.knots_u[0]), 0.0)
        self.assertEqual(float(surface.knots_u[-1]), 1.0)
        self.assertEqual(float(surface.knots_v[0]), 0.0)
        self.assertEqual(float(surface.knots_v[-1]), 1.0)
        self.assertTrue(torch.isfinite(expected).all())

    def test_analytic_second_derivatives_match_finite_differences(self):
        surface = _random_surface(degree_u=3, degree_v=3)
        torch.manual_seed(29)
        uv = 0.12 + 0.76 * torch.rand(48, 2, dtype=torch.float64)
        point, deriv_u, deriv_v, deriv_uu, deriv_uv, deriv_vv = surface.evaluate_with_second_derivatives(uv)
        torch.testing.assert_close(point, surface.evaluate(uv))

        step = 2e-5
        offset_u = torch.tensor([step, 0.0], dtype=torch.float64)
        offset_v = torch.tensor([0.0, step], dtype=torch.float64)
        _, plus_u_du, plus_u_dv = surface.evaluate_with_derivatives(uv + offset_u)
        _, minus_u_du, minus_u_dv = surface.evaluate_with_derivatives(uv - offset_u)
        _, plus_v_du, plus_v_dv = surface.evaluate_with_derivatives(uv + offset_v)
        _, minus_v_du, minus_v_dv = surface.evaluate_with_derivatives(uv - offset_v)
        fd_uu = (plus_u_du - minus_u_du) / (2.0 * step)
        fd_uv_from_u = (plus_u_dv - minus_u_dv) / (2.0 * step)
        fd_uv_from_v = (plus_v_du - minus_v_du) / (2.0 * step)
        fd_vv = (plus_v_dv - minus_v_dv) / (2.0 * step)
        torch.testing.assert_close(deriv_u, surface.evaluate_with_derivatives(uv)[1])
        torch.testing.assert_close(deriv_v, surface.evaluate_with_derivatives(uv)[2])
        torch.testing.assert_close(deriv_uu, fd_uu, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(deriv_uv, fd_uv_from_u, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(deriv_uv, fd_uv_from_v, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(deriv_vv, fd_vv, atol=2e-3, rtol=2e-3)
    def test_degree_zero_axis_returns_zero_derivative(self):
        grid = torch.rand(4, 1, 3, dtype=torch.float64)
        surface = TorchNURBSSurface(
            control_grid=grid, weights=torch.ones(4, 1, dtype=torch.float64)
        )
        uv = torch.rand(8, 2, dtype=torch.float64)
        _, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
        self.assertTrue(torch.isfinite(deriv_u).all())
        self.assertTrue(torch.equal(deriv_v, torch.zeros_like(deriv_v)))

    def test_planar_surface_normals_are_z_aligned(self):
        base_u = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        base_v = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
        grid = torch.stack(
            [
                base_u[:, None].expand(5, 4),
                base_v[None, :].expand(5, 4),
                torch.zeros(5, 4, dtype=torch.float64),
            ],
            dim=-1,
        )
        surface = TorchNURBSSurface(
            control_grid=grid,
            weights=torch.ones(5, 4, dtype=torch.float64),
            degree_u=2,
            degree_v=2,
        )
        normals = surface.normals(torch.rand(16, 2, dtype=torch.float64))
        self.assertTrue(torch.allclose(normals.abs()[:, 2], torch.ones(16, dtype=torch.float64), atol=1e-6))
        self.assertTrue(torch.allclose(normals[:, :2], torch.zeros(16, 2, dtype=torch.float64), atol=1e-6))


class FootPointProjectionTest(unittest.TestCase):
    def test_on_surface_points_project_back_onto_surface(self):
        surface = _random_surface()
        torch.manual_seed(5)
        uv_true = 0.05 + 0.9 * torch.rand(128, 2, dtype=torch.float64)
        points = surface.evaluate(uv_true)
        uv = project_torch_points_to_nurbs(points, surface, iterations=6)
        distance = (surface.evaluate(uv) - points).norm(dim=1)
        self.assertLess(float(distance.max()), 1e-5)

    def test_gauss_newton_never_regresses_from_grid_seed(self):
        surface = _random_surface(seed=9)
        torch.manual_seed(21)
        points = surface.evaluate(torch.rand(64, 2, dtype=torch.float64))
        points = points + 0.05 * torch.randn(64, 3, dtype=torch.float64)
        uv_seed = project_torch_points_to_nurbs(points, surface, iterations=0)
        uv_refined = project_torch_points_to_nurbs(points, surface, iterations=6)
        dist_seed = (surface.evaluate(uv_seed) - points).norm(dim=1)
        dist_refined = (surface.evaluate(uv_refined) - points).norm(dim=1)
        self.assertTrue(bool((dist_refined <= dist_seed + 1e-12).all()))
        self.assertLess(float(dist_refined.mean()), float(dist_seed.mean()))


def _sine_sheet(count: int = 1200, seed: int = 2):
    """Scattered samples of an analytic smooth sheet z = f(x, y)."""

    torch.manual_seed(seed)
    xy = torch.rand(count, 2) * 2.0 - 1.0
    z = 0.25 * torch.sin(xy[:, 0] * 3.0) * torch.cos(xy[:, 1] * 2.0)
    return torch.cat([xy, z.unsqueeze(1)], dim=1)


def _rms_surface_distance(points, surface) -> float:
    uv = project_torch_points_to_nurbs(points, surface, iterations=6)
    return float((surface.evaluate(uv) - points).norm(dim=1).square().mean().sqrt())


class LeastSquaresFitTest(unittest.TestCase):
    def test_lsq_fit_beats_idw_seed_on_analytic_sheet(self):
        points = _sine_sheet()
        idw = fit_torch_visible_surface(points, resolution_u=10, resolution_v=8)
        lsq, _ = fit_torch_visible_surface_lsq(points, resolution_u=10, resolution_v=8)
        extent = float((points.amax(dim=0) - points.amin(dim=0)).norm())
        rms_idw = _rms_surface_distance(points, idw)
        rms_lsq = _rms_surface_distance(points, lsq)
        self.assertLess(rms_lsq, rms_idw)
        self.assertLess(rms_lsq / extent, 0.005)

    def test_lsq_returns_foot_point_uv_for_input_points(self):
        points = _sine_sheet(count=600, seed=8)
        surface, uv = fit_torch_visible_surface_lsq(points, resolution_u=8, resolution_v=6)
        anchors = surface.evaluate(uv)
        extent = float((points.amax(dim=0) - points.amin(dim=0)).norm())
        mean_ratio = float((anchors - points).norm(dim=1).mean()) / extent
        self.assertLess(mean_ratio, 0.01)

    def test_lsq_handles_more_controls_than_points(self):
        points = _sine_sheet(count=12, seed=4)
        surface, uv = fit_torch_visible_surface_lsq(points, resolution_u=8, resolution_v=6)
        self.assertTrue(torch.isfinite(surface.control_grid).all())
        self.assertTrue(torch.isfinite(surface.evaluate(uv)).all())

    def test_pipeline_config_controls_degrees_and_mode(self):
        points = _sine_sheet(count=300, seed=6)
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(
                voxel_grid_resolution=4,
                visible_surface_resolution_u=6,
                visible_surface_resolution_v=4,
                surface_degree_u=3,
                surface_degree_v=2,
            ),
            device="cpu",
        )
        state = pipeline.initialize(points, torch.rand(points.shape[0], 3))
        for patch in state.surface_patches:
            self.assertEqual(patch.degree_u, 3)
            self.assertEqual(patch.degree_v, 2)


class CoupledPatchGraphFitTest(unittest.TestCase):
    def _patches(self):
        torch.manual_seed(31)
        uv_a = torch.rand(240, 2)
        uv_b = torch.rand(240, 2)
        points_a = torch.stack((uv_a[:, 0], uv_a[:, 1], 0.08 * uv_a[:, 0] * uv_a[:, 1]), dim=1)
        # Same world sheet on [1, 2] x [0, 1], but v is reversed locally.
        points_b = torch.stack((1.0 + uv_b[:, 0], 1.0 - uv_b[:, 1], 0.08 * (1.0 + uv_b[:, 0]) * (1.0 - uv_b[:, 1])), dim=1)
        return [points_a, points_b], [uv_a, uv_b]

    def test_reversed_full_edge_is_one_joint_variable_sequence(self):
        points, uv = self._patches()
        n_u, n_v = 6, 5
        constraint = SharedBoundaryConstraint(
            0,
            boundary_control_indices(n_u, n_v, "u1"),
            1,
            boundary_control_indices(n_u, n_v, "u0"),
            reverse=True,
            constraint_id="shared",
        )
        results = fit_coupled_patch_graph_lsq(
            points, uv, [constraint], resolution_u=n_u, resolution_v=n_v
        )
        torch.testing.assert_close(
            results[0][0].control_grid[-1],
            results[1][0].control_grid[0].flip(0),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_partial_edge_constraint_only_shares_selected_controls(self):
        points, uv = self._patches()
        n_u, n_v = 6, 5
        selected_a = boundary_control_indices(n_u, n_v, "u1", 1, 4)
        selected_b = boundary_control_indices(n_u, n_v, "u0", 1, 4)
        constraint = SharedBoundaryConstraint(0, selected_a, 1, selected_b, reverse=True)
        results = fit_coupled_patch_graph_lsq(
            points, uv, [constraint], resolution_u=n_u, resolution_v=n_v
        )
        edge_a = results[0][0].control_grid[-1]
        edge_b = results[1][0].control_grid[0].flip(0)
        torch.testing.assert_close(edge_a[1:4], edge_b[1:4], atol=1e-6, rtol=1e-6)
        self.assertGreater(float((edge_a[[0, 4]] - edge_b[[0, 4]]).abs().max()), 1e-7)

class CoupledWedgeRingFitTest(unittest.TestCase):
    """Phase 5 Step 5-A (docs/worklogs/55): white-box tests on
    ``fit_coupled_wedge_ring_lsq`` -- the core claim is that adjacent
    wedges' shared seam boundary column is the exact same joint variable,
    not two independently-fit columns that merely end up close."""

    def test_shared_seam_columns_are_exactly_equal(self):
        torch.manual_seed(0)
        segments = 4
        wedge_points = []
        wedge_uv = []
        for k in range(segments):
            uv = torch.rand(200, 2)
            pts = torch.stack(
                [uv[:, 0] + k, uv[:, 1], 0.05 * torch.sin(uv[:, 0] * 3.0 + k)], dim=1
            )
            wedge_points.append(pts)
            wedge_uv.append(uv)

        results = fit_coupled_wedge_ring_lsq(
            wedge_points, wedge_uv, resolution_u=6, resolution_v=4, degree_u=2, degree_v=2
        )
        self.assertEqual(len(results), segments)
        surfaces = [r[0] for r in results]
        for k in range(segments):
            this_last_column = surfaces[k].control_grid[-1]
            next_first_column = surfaces[(k + 1) % segments].control_grid[0]
            torch.testing.assert_close(this_last_column, next_first_column, atol=1e-5, rtol=1e-5)
        for surface, uv in results:
            self.assertTrue(torch.isfinite(surface.control_grid).all())
            self.assertTrue(torch.isfinite(surface.evaluate(uv)).all())

    def test_handles_minimal_two_wedge_ring_without_error(self):
        # Degenerate/robustness check: the smallest possible ring (2 wedges,
        # each sharing two distinct seams with the other) must still solve.
        torch.manual_seed(1)
        segments = 2
        wedge_points = [torch.rand(100, 3) for _ in range(segments)]
        wedge_uv = [torch.rand(100, 2) for _ in range(segments)]
        results = fit_coupled_wedge_ring_lsq(wedge_points, wedge_uv, resolution_u=5, resolution_v=3)
        self.assertEqual(len(results), segments)
        for surface, uv in results:
            self.assertTrue(torch.isfinite(surface.control_grid).all())

    def test_collect_diagnostics_returns_per_wedge_diagnostics(self):
        torch.manual_seed(2)
        segments = 3
        wedge_points = [torch.rand(150, 3) for _ in range(segments)]
        wedge_uv = [torch.rand(150, 2) for _ in range(segments)]
        results = fit_coupled_wedge_ring_lsq(
            wedge_points, wedge_uv, resolution_u=5, resolution_v=3, collect_diagnostics=True
        )
        self.assertEqual(len(results), segments)
        for surface, uv, diagnostics in results:
            self.assertTrue(torch.isfinite(diagnostics.final_control_grid).all())
            self.assertGreaterEqual(len(diagnostics.rounds), 1)


class MaintenanceUVRefreshTest(unittest.TestCase):
    def _state(self, count: int = 64):
        torch.manual_seed(13)
        xy = torch.rand(count, 2)
        z = 0.2 * torch.sin(xy[:, 0] * 4.0)
        points = torch.cat([xy, z.unsqueeze(1)], dim=1)
        colors = torch.rand(count, 3)
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(
                voxel_grid_resolution=4,
                visible_surface_resolution_u=5,
                visible_surface_resolution_v=4,
            ),
            device="cpu",
        )
        return pipeline, pipeline.initialize(points, colors)

    def test_refresh_updates_bindings_and_tightens_residual(self):
        pipeline, state = self._state()
        torch.manual_seed(17)
        state.model.surface_uv.copy_(torch.rand_like(state.model.surface_uv))
        scrambled_uv = state.model.surface_uv.detach().clone()

        stale = pipeline.maintain_surface_from_certain(
            state, residual_patience=1000, refresh_uv=False, enable_local_correction=False
        )
        self.assertTrue(torch.equal(state.model.surface_uv, scrambled_uv))

        refreshed = pipeline.maintain_surface_from_certain(
            state, residual_patience=1000, refresh_uv=True, enable_local_correction=False
        )
        self.assertGreater(refreshed["uv_refreshed"], 0)
        self.assertFalse(torch.equal(state.model.surface_uv, scrambled_uv))
        self.assertLessEqual(
            refreshed["max_residual_ratio"], stale["max_residual_ratio"] + 1e-9
        )


if __name__ == "__main__":
    unittest.main()
