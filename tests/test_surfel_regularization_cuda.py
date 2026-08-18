"""CUDA behaviour tests for the two 2DGS geometric regularizers.

`tests/test_surfel_losses.py` pins the formulations; these tests establish
that, driven through the real rasterizer, each term actually does what the
paper says it does -- depth distortion concentrates intersections in depth,
normal consistency reorients surfels -- and that densification under real
gradients keeps producing valid planar surfels.
"""

from __future__ import annotations

import math
import unittest

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from osn_gs.gaussian.torch_density_control import (
        add_densification_stats,
        apply_adaptive_density_control,
        update_max_radii,
    )
    from osn_gs.gaussian.torch_model import GaussianParameterGroups
    from osn_gs.gaussian.torch_surfel_density_control import surfel_density_control_config
    from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
    from osn_gs.losses.torch_surfel_losses import (
        depth_distortion_loss,
        normal_consistency_loss,
    )
    from osn_gs.render.diff_surfel_loader import get_diff_surfel_backend
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    BACKEND_AVAILABLE = get_diff_surfel_backend() is not None
    from tests.test_surfel_rasterization_cuda import _camera
else:  # pragma: no cover - environment dependent
    BACKEND_AVAILABLE = False

requires_cuda = unittest.skipUnless(
    CUDA_AVAILABLE and BACKEND_AVAILABLE,
    "CUDA and the vendored diff_surfel_rasterization extension are required",
)


def _front_camera():
    """Looks down +z from the origin."""

    return _camera(torch.eye(3), torch.zeros(3))


def _facing_quaternion(count: int, jitter: float, seed: int) -> torch.Tensor:
    """Quaternions near identity (tangent plane ~ xy, normal ~ +z) with jitter."""

    generator = torch.Generator().manual_seed(seed)
    axis = torch.nn.functional.normalize(
        torch.rand((count, 3), generator=generator) - 0.5, dim=1
    )
    angle = jitter * (torch.rand((count, 1), generator=generator) * 2 - 1)
    return torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=1)


def _slab_model(layers: int = 6, per_layer: int = 5, spread: float = 1.2, jitter: float = 0.0,
                opacity: float = 0.35, scale: float = 0.35, seed: int = 3):
    """Surfels spread through a depth SLAB in front of the camera.

    All layers project onto the same pixels, so a pixel ray intersects every
    one of them -- the configuration depth distortion is meant to collapse.
    """

    torch.manual_seed(seed)
    depths = torch.linspace(4.0 - spread, 4.0 + spread, layers)
    axis = torch.linspace(-0.25, 0.25, per_layer)
    positions = torch.stack(
        [
            torch.tensor([float(x), float(y), float(z)])
            for z in depths
            for x in axis
            for y in axis
        ]
    )
    count = positions.shape[0]
    model = TorchGaussianSurfelModel(sh_degree=0, device="cuda")
    model.initialize(
        positions=positions,
        colors=torch.rand((count, 3)),
        opacities=torch.full((count, 1), opacity),
        scales=torch.full((count, 2), scale),
        rotations=_facing_quaternion(count, jitter, seed),
    )
    return model


@requires_cuda
class DepthDistortionBehaviourTest(unittest.TestCase):
    """Paper eq. 13: concentrate the contributing intersections in depth."""

    def test_a_depth_slab_scores_higher_than_a_concentrated_sheet(self):
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig(depth_ratio=1.0))
        camera = _front_camera()
        spread_loss = float(
            depth_distortion_loss(rasterizer.render(camera, _slab_model(spread=1.2)))
        )
        tight_loss = float(
            depth_distortion_loss(rasterizer.render(camera, _slab_model(spread=0.01)))
        )
        self.assertGreater(spread_loss, 10.0 * tight_loss)

    def test_optimizing_the_term_concentrates_the_contributing_intersections(self):
        """The paper's stated purpose, measured on the rendered rays.

        The observable is the per-pixel gap between the EXPECTED (alpha-
        weighted mean) intersection depth and the MEDIAN intersection depth.
        That gap is zero exactly when a ray's contributions sit at one depth
        and grows as they spread, so it reports the ray-level concentration
        eq. 13 targets without depending on how individual primitives happen
        to move. Only the depth coordinate is optimized, so the term cannot
        "win" by sliding surfels laterally out of the shared rays.
        """

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig(depth_ratio=1.0))
        camera = _front_camera()
        model = _slab_model(spread=1.2)
        optimizer = torch.optim.Adam([model._xyz], lr=0.02)

        def ray_depth_gap() -> float:
            with torch.no_grad():
                package = rasterizer.render(camera, model)
                mask = package["rend_alpha"][0] > 0.5
                gap = (package["depth_expected"][0] - package["depth_median"][0]).abs()
                return float(gap[mask].mean())

        initial_gap = ray_depth_gap()
        initial_loss = float(depth_distortion_loss(rasterizer.render(camera, model)))
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            # 2DGS weights this term by alpha (1000 bounded / 100 unbounded);
            # the scale only sets the step size here.
            (100.0 * depth_distortion_loss(rasterizer.render(camera, model))).backward()
            # Depth-only motion: the lateral escape route is closed off so the
            # measurement is about depth concentration and nothing else.
            model._xyz.grad[:, :2] = 0.0
            optimizer.step()
        final_gap = ray_depth_gap()
        final_loss = float(depth_distortion_loss(rasterizer.render(camera, model)))

        self.assertLess(final_loss, initial_loss)
        self.assertGreater(initial_gap, 0.05, "the fixture must start spread out")
        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_the_term_is_finite_and_non_negative(self):
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig(depth_ratio=1.0))
        package = rasterizer.render(_front_camera(), _slab_model())
        distortion = package["rend_dist"]
        self.assertTrue(bool(torch.isfinite(distortion).all()))
        # The CUDA recursion accumulates in float32, so an all-but-zero pixel
        # can land a few 1e-8 below zero.
        self.assertGreaterEqual(distortion.min().item(), -1e-6)


@requires_cuda
class NormalConsistencyBehaviourTest(unittest.TestCase):
    """Paper eqs. 14-15: align surfel normals with the rendered surface."""

    def test_optimizing_the_term_reorients_surfels_toward_the_surface(self):
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig(depth_ratio=1.0))
        camera = _front_camera()
        # A single depth sheet whose surfels are randomly mis-oriented.
        model = _slab_model(layers=1, per_layer=9, spread=0.0, jitter=0.7, opacity=0.9, scale=0.05)
        surface_normal = torch.tensor([0.0, 0.0, 1.0], device="cuda")

        initial_alignment = float((model.get_normal.detach() @ surface_normal).abs().mean())
        initial_loss = float(normal_consistency_loss(rasterizer.render(camera, model)))

        optimizer = torch.optim.Adam([model._rotation], lr=0.02)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            normal_consistency_loss(rasterizer.render(camera, model)).backward()
            optimizer.step()

        final_alignment = float((model.get_normal.detach() @ surface_normal).abs().mean())
        final_loss = float(normal_consistency_loss(rasterizer.render(camera, model)))

        self.assertLess(final_loss, initial_loss)
        self.assertGreater(final_alignment, initial_alignment)
        self.assertLess(initial_alignment, 0.98, "the fixture must start mis-oriented")

    def test_gradient_reaches_rotation_but_the_normal_stays_intrinsic(self):
        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig(depth_ratio=1.0))
        model = _slab_model(layers=1, per_layer=9, spread=0.0, jitter=0.7, opacity=0.9, scale=0.05)
        normal_consistency_loss(rasterizer.render(_front_camera(), model)).backward()
        self.assertGreater(model._rotation.grad.abs().max().item(), 0.0)
        # The normal is still exactly t_u x t_v after the update path exists.
        expected = torch.cross(model.get_tangent_u, model.get_tangent_v, dim=1)
        torch.testing.assert_close(model.get_normal, expected, atol=1e-5, rtol=1e-5)


@requires_cuda
class DensificationUnderRealGradientsTest(unittest.TestCase):
    """Densification driven by the real renderer keeps producing planar surfels."""

    def test_adc_after_real_render_and_backward_preserves_planarity(self):
        rasterizer = OSNSurfelRasterizer()
        camera = _front_camera()
        model = _slab_model(layers=3, per_layer=7, spread=0.4, opacity=0.6, scale=0.25)
        model.spatial_lr_scale = 1.0
        model.training_setup(GaussianParameterGroups())
        target = torch.rand((3, camera.image_height, camera.image_width), device="cuda")

        config = surfel_density_control_config(densify_until_iter=1000, densification_interval=1)
        grew = False
        for _ in range(4):
            package = rasterizer.render(camera, model)
            model.optimizer.zero_grad(set_to_none=True)
            (package["render"] - target).abs().mean().backward()
            update_max_radii(model, package["radii"], package["visibility_filter"])
            add_densification_stats(model, package["viewspace_points"], package["visibility_filter"])
            model.optimizer.step()
            before = len(model)
            report = apply_adaptive_density_control(model, config, scene_extent=2.0, iteration=100)
            grew = grew or (report.cloned + report.split) > 0
            self.assertEqual(model._scaling.shape[1], 2)
            self.assertEqual(model.get_scaling.shape[1], 2)
            self.assertTrue(bool(torch.isfinite(model._scaling).all()))
            self.assertTrue(bool(torch.isfinite(model._rotation).all()))
            identifiers = model.stable_gaussian_ids.tolist()
            self.assertEqual(len(identifiers), len(set(identifiers)))
            # `report.pruned` counts opacity/screen/world prunes only; split
            # parents are removed separately and reported as `split_parents`.
            self.assertEqual(
                len(model),
                before + report.cloned + report.split - report.split_parents - report.pruned,
            )

        self.assertTrue(grew, "the fixture must actually densify")
        # Every surviving surfel still has an exactly intrinsic unit normal.
        normals = model.get_normal.detach()
        torch.testing.assert_close(
            normals.norm(dim=1), torch.ones(len(model), device="cuda"), atol=1e-4, rtol=1e-4
        )
        # And it still renders.
        self.assertGreater(int((rasterizer.render(camera, model)["radii"] > 0).sum()), 0)

    def test_densification_statistics_come_from_the_screen_space_gradient(self):
        rasterizer = OSNSurfelRasterizer()
        camera = _front_camera()
        model = _slab_model(layers=2, per_layer=5, spread=0.3, opacity=0.6, scale=0.2)
        package = rasterizer.render(camera, model)
        target = torch.rand_like(package["render"])
        (package["render"] - target).abs().mean().backward()
        add_densification_stats(model, package["viewspace_points"], package["visibility_filter"])
        self.assertEqual(model.density_gradient_sources["screen_space"], 1)
        self.assertEqual(model.density_gradient_sources["xyz_fallback"], 0)
        self.assertGreater(model.xyz_gradient_accum.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
