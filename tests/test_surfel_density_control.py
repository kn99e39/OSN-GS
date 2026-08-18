"""2DGS-compatible adaptive density control.

Verifies the audit recorded in
`osn_gs/gaussian/torch_surfel_density_control.py` against the official
`scene/gaussian_model.py::densify_and_prune`, with particular attention to the
one place volumetric 3DGS semantics would be wrong: split children of a planar
primitive must be sampled strictly inside the parent's tangent plane.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.gaussian.torch_density_control import (
    TorchDensityControlConfig,
    _shape_transaction_candidates,
    apply_adaptive_density_control,
    should_run_adc,
)
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surfel_density_control import (
    OFFICIAL_DENSIFY_GRAD_THRESHOLD,
    OFFICIAL_OPACITY_CULL,
    surfel_density_control_config,
)
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel, build_rotation


def _model(cls, count: int, scale: float, seed: int = 11):
    torch.manual_seed(seed)
    model = cls(sh_degree=1, device="cpu")
    scale_columns = cls.scale_dim
    model.initialize(
        positions=torch.rand((count, 3)),
        colors=torch.rand((count, 3)),
        opacities=torch.full((count, 1), 0.5),
        scales=torch.full((count, scale_columns), scale),
        rotations=torch.rand((count, 4)),
    )
    model.xyz_gradient_accum = torch.full((count, 1), 1.0)
    model.denom = torch.ones((count, 1))
    return model


class SurfelSplitSamplingTest(unittest.TestCase):
    """Audit item 3: children are sampled with a ZERO normal-direction std."""

    def test_split_children_stay_in_the_parent_tangent_plane(self):
        model = _model(TorchGaussianSurfelModel, 8, scale=0.2)
        samples = 2
        split_idx = torch.arange(8)
        candidate = _shape_transaction_candidates(
            model, torch.zeros((0,), dtype=torch.long), split_idx, samples
        )
        children = candidate["xyz"][8:]
        parent_centers = model.get_xyz.detach().repeat_interleave(samples, dim=0)
        parent_normals = model.get_normal.detach().repeat_interleave(samples, dim=0)
        residual = ((children - parent_centers) * parent_normals).sum(dim=1).abs()
        # Not "small": exactly zero up to float round-off in the rotation.
        self.assertLess(residual.max().item(), 1e-6)
        # And the children are genuinely displaced within the plane, so the
        # zero above is a plane constraint rather than a no-op.
        self.assertGreater((children - parent_centers).norm(dim=1).max().item(), 1e-3)

    def test_volumetric_split_children_leave_the_tangent_plane(self):
        """Control: the 3D Gaussian arm keeps full 3-axis sampling."""

        model = _model(TorchGaussianModel, 8, scale=0.2)
        samples = 2
        split_idx = torch.arange(8)
        candidate = _shape_transaction_candidates(
            model, torch.zeros((0,), dtype=torch.long), split_idx, samples
        )
        children = candidate["xyz"][8:]
        parent_centers = model.get_xyz.detach().repeat_interleave(samples, dim=0)
        third_axis = build_rotation(model.get_rotation.detach())[:, :, 2].repeat_interleave(
            samples, dim=0
        )
        residual = ((children - parent_centers) * third_axis).sum(dim=1).abs()
        self.assertGreater(residual.max().item(), 1e-3)

    def test_split_children_inherit_the_tangent_frame_and_shrunk_scales(self):
        model = _model(TorchGaussianSurfelModel, 6, scale=0.2)
        samples = 2
        split_idx = torch.arange(6)
        candidate = _shape_transaction_candidates(
            model, torch.zeros((0,), dtype=torch.long), split_idx, samples
        )
        parent_rotation = model._rotation.detach().repeat_interleave(samples, dim=0)
        torch.testing.assert_close(candidate["rotation"][6:], parent_rotation)
        # Official: new_scaling = log(get_scaling.repeat(N,1) / (0.8*N)).
        expected = torch.log(
            model.get_scaling.detach().repeat_interleave(samples, dim=0) / (0.8 * samples)
        )
        torch.testing.assert_close(candidate["scaling"][6:], expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(candidate["scaling"].shape[1], 2)

    def test_clone_copies_the_parent_verbatim(self):
        model = _model(TorchGaussianSurfelModel, 5, scale=0.001)
        clone_idx = torch.arange(5)
        candidate = _shape_transaction_candidates(
            model, clone_idx, torch.zeros((0,), dtype=torch.long), 2
        )
        for key in ("xyz", "scaling", "rotation", "opacity"):
            torch.testing.assert_close(candidate[key][5:], candidate[key][:5])


class SurfelDensityControlIntegrationTest(unittest.TestCase):
    def test_adc_preserves_the_two_column_scaling(self):
        model = _model(TorchGaussianSurfelModel, 32, scale=0.2)
        config = surfel_density_control_config(densify_until_iter=1000, densification_interval=1)
        report = apply_adaptive_density_control(model, config, scene_extent=1.0, iteration=100)
        self.assertGreater(report.split + report.cloned, 0)
        self.assertEqual(model._scaling.shape[1], 2)
        self.assertEqual(model.get_scaling.shape[1], 2)

    def test_adc_keeps_stable_ids_unique_and_preserves_survivors(self):
        model = _model(TorchGaussianSurfelModel, 32, scale=0.0005)
        before = set(model.stable_gaussian_ids.tolist())
        config = surfel_density_control_config(densify_until_iter=1000, densification_interval=1)
        apply_adaptive_density_control(model, config, scene_extent=1.0, iteration=100)
        after = model.stable_gaussian_ids.tolist()
        self.assertEqual(len(after), len(set(after)))
        # A pure clone pass keeps every original row alive.
        self.assertTrue(before.issubset(set(after)))

    def test_opacity_reset_cannot_touch_scale(self):
        """Audit item 7: the reset is opacity-only."""

        model = _model(TorchGaussianSurfelModel, 10, scale=0.2)
        before = model._scaling.detach().clone()
        model.reset_opacity()
        torch.testing.assert_close(model._scaling.detach(), before)
        self.assertEqual(model._scaling.shape[1], 2)


class SurfelDensityControlConfigTest(unittest.TestCase):
    def test_official_2dgs_parameters(self):
        config = surfel_density_control_config()
        # Paper sec. 6.1 / official OptimizationParams.
        self.assertAlmostEqual(config.prune_opacity_threshold, OFFICIAL_OPACITY_CULL)
        self.assertAlmostEqual(config.densify_grad_threshold, OFFICIAL_DENSIFY_GRAD_THRESHOLD)
        self.assertAlmostEqual(config.percent_dense, 0.01)
        self.assertEqual(config.densify_from_iter, 500)
        self.assertEqual(config.densify_until_iter, 15_000)
        self.assertEqual(config.densification_interval, 100)
        self.assertEqual(config.opacity_reset_interval, 3000)
        self.assertAlmostEqual(config.max_screen_size, 20.0)
        self.assertAlmostEqual(config.max_scale_ratio, 0.1)

    def test_2dgs_opacity_cull_differs_from_the_3dgs_default(self):
        """0.05, not 3DGS's 0.005 -- a real methodology difference."""

        self.assertAlmostEqual(TorchDensityControlConfig().prune_opacity_threshold, 0.005)
        self.assertAlmostEqual(surfel_density_control_config().prune_opacity_threshold, 0.05)

    def test_screen_size_gate_follows_opacity_reset_interval(self):
        """Official: `size_threshold = 20 if iteration > opacity_reset_interval`."""

        for interval in (3000, 5000):
            config = surfel_density_control_config(opacity_reset_interval=interval)
            self.assertEqual(config.screen_size_prune_from_iter, interval)

    def test_official_densification_window(self):
        config = surfel_density_control_config()
        self.assertFalse(should_run_adc(500, config))
        self.assertTrue(should_run_adc(600, config))
        self.assertFalse(should_run_adc(15_000, config))


if __name__ == "__main__":
    unittest.main()
