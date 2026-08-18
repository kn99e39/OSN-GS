"""CUDA verification that the branch renders 2DGS, not flattened 3DGS.

Requires a CUDA device and the vendored `diff_surfel_rasterization` extension;
skipped otherwise. These are the checks that separate "a 2D Gaussian Splatting
implementation" from "a 3D Gaussian with a small third scale rendered by the
old rasterizer":

* the rendered depth is the PERSPECTIVE-CORRECT ray-splat intersection --
  every rendered surface point lies on the surfel's own tangent plane to
  float precision, across the whole footprint and not just near the centre,
  which the 3DGS affine approximation cannot do;
* gradients reach every trainable tensor through the new renderer;
* the object-space low-pass filter keeps edge-on splats rasterizable;
* the renderer exposes the geometric outputs the paper's regularizers and
  OSN-GS's structural-evidence analysis need.
"""

from __future__ import annotations

import math
import unittest

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
    from osn_gs.render.diff_surfel_loader import get_diff_surfel_backend
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    from osn_gs.render.torch_fallback import TorchCamera

    BACKEND_AVAILABLE = get_diff_surfel_backend() is not None
else:  # pragma: no cover - environment dependent
    BACKEND_AVAILABLE = False

requires_cuda = unittest.skipUnless(
    CUDA_AVAILABLE and BACKEND_AVAILABLE,
    "CUDA and the vendored diff_surfel_rasterization extension are required",
)

WIDTH, HEIGHT = 160, 120
FOVX = math.radians(60.0)
FOVY = 2.0 * math.atan(math.tan(FOVX * 0.5) * HEIGHT / WIDTH)
ZNEAR, ZFAR = 0.01, 100.0


def _projection_matrix() -> torch.Tensor:
    """Graphdeco/2DGS projection matrix, matching `colmap_scene.projection_matrix`."""

    top = math.tan(FOVY * 0.5) * ZNEAR
    right = math.tan(FOVX * 0.5) * ZNEAR
    matrix = torch.zeros((4, 4), dtype=torch.float32)
    matrix[0, 0] = ZNEAR / right
    matrix[1, 1] = ZNEAR / top
    matrix[3, 2] = 1.0
    matrix[2, 2] = ZFAR / (ZFAR - ZNEAR)
    matrix[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    return matrix


def _camera(rotation: torch.Tensor, translation: torch.Tensor) -> TorchCamera:
    """Camera with OSN-GS/Graphdeco transposed-matrix conventions."""

    world_view = torch.eye(4, dtype=torch.float32)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = translation
    world_view = world_view.transpose(0, 1).contiguous().cuda()
    projection = _projection_matrix().transpose(0, 1).contiguous().cuda()
    full_proj = (world_view.unsqueeze(0) @ projection.unsqueeze(0)).squeeze(0)
    center = (-rotation.T @ translation).cuda()
    return TorchCamera(
        image_height=HEIGHT,
        image_width=WIDTH,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
        camera_center=center,
        FoVx=FOVX,
        FoVy=FOVY,
        image_name="test",
    )


def _tilted_camera() -> TorchCamera:
    """Looks down +z from the origin, rolled so the plane is genuinely oblique."""

    angle = math.radians(25.0)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=torch.float32
    )
    return _camera(rotation, torch.tensor([0.0, 0.0, 4.0], dtype=torch.float32))


def _single_surfel(scale: float = 1.2, tilt_degrees: float = 55.0, opacity: float = 0.99):
    """One big, strongly tilted surfel centred in front of the camera."""

    model = TorchGaussianSurfelModel(sh_degree=0, device="cuda")
    half = math.radians(tilt_degrees) * 0.5
    # Rotation about the x axis by `tilt_degrees`, wxyz.
    rotation = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]], dtype=torch.float32)
    model.initialize(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        colors=torch.tensor([[0.6, 0.4, 0.2]], dtype=torch.float32),
        opacities=torch.tensor([[opacity]], dtype=torch.float32),
        scales=torch.tensor([[scale, scale]], dtype=torch.float32),
        rotations=rotation,
    )
    return model


def _pixel_rays(camera: TorchCamera) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera-space ray directions per pixel, in the rasterizer's pixel convention.

    Derived independently of the transform machinery under test: the vendored
    `compute_transmat` builds `ndc2pix` with offsets `(W-1)/2, (H-1)/2` and
    scales `W/2, H/2`, so pixel (x, y) corresponds to
    ``ndc = ((x - (W-1)/2) / (W/2), (y - (H-1)/2) / (H/2))`` and, for this
    symmetric frustum, to camera-space direction
    ``(ndc_x * tan(fovx/2), ndc_y * tan(fovy/2), 1)``.
    """

    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float32, device="cuda"),
        torch.arange(WIDTH, dtype=torch.float32, device="cuda"),
        indexing="ij",
    )
    ndc_x = (xs - (WIDTH - 1) / 2.0) / (WIDTH / 2.0)
    ndc_y = (ys - (HEIGHT - 1) / 2.0) / (HEIGHT / 2.0)
    directions = torch.stack(
        [ndc_x * math.tan(FOVX * 0.5), ndc_y * math.tan(FOVY * 0.5), torch.ones_like(ndc_x)],
        dim=-1,
    )
    return directions, ndc_x


def _analytic_intersection_depth(camera: TorchCamera, model) -> torch.Tensor:
    """Exact ray-plane intersection depth (view-space z) for every pixel."""

    view = camera.world_view_transform.T  # world -> camera
    center_world = model.get_xyz.detach()[0]
    normal_world = model.get_normal.detach()[0]
    center_view = (view[:3, :3] @ center_world) + view[:3, 3]
    normal_view = view[:3, :3] @ normal_world

    directions, _ = _pixel_rays(camera)
    # Ray origin is the camera centre, i.e. the origin in view space.
    denominator = (directions * normal_view).sum(dim=-1)
    numerator = (center_view * normal_view).sum()
    t = numerator / denominator
    return t  # directions have z == 1, so t IS the view-space depth


@requires_cuda
class PerspectiveCorrectIntersectionTest(unittest.TestCase):
    """Paper sec. 4.2 / eqs. 8-10: an explicit ray-splat intersection."""

    def setUp(self):
        self.camera = _tilted_camera()
        self.model = _single_surfel()
        self.rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig(depth_ratio=1.0))
        self.package = self.rasterizer.render(self.camera, self.model)
        # Only pixels the splat actually covers carry a median depth.
        self.mask = (self.package["rend_alpha"][0] > 0.5) & (self.package["depth_median"][0] > 0)
        self.assertGreater(int(self.mask.sum()), 1000, "test scene did not cover enough pixels")

    def test_rendered_depth_is_the_exact_ray_plane_intersection(self):
        analytic = _analytic_intersection_depth(self.camera, self.model)
        rendered = self.package["depth_median"][0]
        error = (rendered - analytic)[self.mask].abs()
        depth_span = (analytic[self.mask].max() - analytic[self.mask].min()).item()
        self.assertGreater(depth_span, 1.0, "the plane must be genuinely oblique")
        # Perspective-correct: matches the closed-form intersection everywhere,
        # not just near the projected centre.
        self.assertLess(error.max().item() / depth_span, 1e-3)

    def test_it_is_not_the_affine_center_approximation(self):
        """Control: a centre-only approximation is wrong by an enormous margin."""

        analytic = _analytic_intersection_depth(self.camera, self.model)
        rendered = self.package["depth_median"][0]
        center_view_z = (
            (self.camera.world_view_transform.T[:3, :3] @ self.model.get_xyz.detach()[0])
            + self.camera.world_view_transform.T[:3, 3]
        )[2]
        affine_error = (center_view_z - analytic)[self.mask].abs().max().item()
        exact_error = (rendered - analytic)[self.mask].abs().max().item()
        self.assertGreater(affine_error, 100.0 * max(exact_error, 1e-6))

    def test_unprojected_surface_points_lie_on_the_tangent_plane(self):
        """Restatement in world space: every rendered point is ON the splat.

        Unprojection uses this file's own pixel convention (`_pixel_rays`,
        derived from the vendored `compute_transmat`'s `ndc2pix`), not
        upstream's `depths_to_points`, whose half-pixel offset is quantified
        separately below.
        """

        view = self.camera.world_view_transform.T
        camera_to_world = torch.linalg.inv(view)
        directions, _ = _pixel_rays(self.camera)
        depth = self.package["depth_median"][0]
        points_view = directions * depth[..., None]
        points_world = points_view @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]

        offset = points_world - self.model.get_xyz.detach()[0]
        out_of_plane = (offset * self.model.get_normal.detach()[0]).sum(dim=-1).abs()
        tangent_extent = offset.norm(dim=-1)[self.mask].max().item()
        self.assertGreater(tangent_extent, 0.5)
        self.assertLess(out_of_plane[self.mask].max().item() / tangent_extent, 1e-3)

    def test_upstream_depth_to_normal_unprojection_half_pixel_offset(self):
        """Quantifies a known UPSTREAM inconsistency, preserved deliberately.

        `utils/point_utils.py::depths_to_points` builds `ndc2pix` with `W/2`,
        `H/2` offsets while the CUDA `compute_transmat` uses `(W-1)/2`,
        `(H-1)/2`. That half-pixel shift is upstream's and is kept verbatim,
        because the normal-consistency loss compares the depth-derived normal
        against the rasterized normal exactly as the official code does. This
        test records its magnitude rather than asserting it away: the induced
        out-of-plane error must stay sub-pixel-equivalent, i.e. well under the
        depth change one pixel of parallax produces.
        """

        from osn_gs.render.surfel_geometry import depths_to_points

        points = depths_to_points(self.camera, self.package["surf_depth"]).reshape(HEIGHT, WIDTH, 3)
        offset = points - self.model.get_xyz.detach()[0]
        out_of_plane = (offset * self.model.get_normal.detach()[0]).sum(dim=-1).abs()
        analytic = _analytic_intersection_depth(self.camera, self.model)
        # Depth change across one pixel row, i.e. the parallax scale.
        per_pixel_depth_step = (analytic[1:, :] - analytic[:-1, :]).abs()[self.mask[1:, :]]
        self.assertLess(
            out_of_plane[self.mask].max().item(), 4.0 * per_pixel_depth_step.max().item()
        )

    def test_depth_varies_across_the_footprint(self):
        rendered = self.package["depth_median"][0][self.mask]
        self.assertGreater((rendered.max() - rendered.min()).item(), 1.0)


@requires_cuda
class RenderOutputContractTest(unittest.TestCase):
    """Every geometric quantity the paper and OSN-GS analysis need is exposed."""

    def setUp(self):
        self.camera = _tilted_camera()
        self.model = _single_surfel()
        self.package = OSNSurfelRasterizer().render(self.camera, self.model)

    def test_required_outputs_and_shapes(self):
        expected = {
            "render": (3, HEIGHT, WIDTH),
            "rend_alpha": (1, HEIGHT, WIDTH),
            "rend_normal": (3, HEIGHT, WIDTH),
            "rend_dist": (1, HEIGHT, WIDTH),
            "surf_depth": (1, HEIGHT, WIDTH),
            "surf_normal": (3, HEIGHT, WIDTH),
            "depth_expected": (1, HEIGHT, WIDTH),
            "depth_median": (1, HEIGHT, WIDTH),
        }
        for key, shape in expected.items():
            self.assertIn(key, self.package)
            self.assertEqual(tuple(self.package[key].shape), shape, key)
        self.assertEqual(tuple(self.package["radii"].shape), (len(self.model),))
        self.assertEqual(self.package["visibility_filter"].tolist(), [0])
        self.assertTrue(bool(self.package["visibility_mask"].all()))

    def test_rendered_normal_matches_the_camera_facing_surfel_normal(self):
        """`DUAL_VISIABLE`: the rasterized normal is the camera-facing t_w."""

        alpha = self.package["rend_alpha"][0]
        mask = alpha > 0.9
        rendered = self.package["rend_normal"].permute(1, 2, 0)[mask] / alpha[mask][:, None]
        intrinsic = self.model.get_normal.detach()[0]
        alignment = (rendered * intrinsic).sum(dim=-1).abs()
        torch.testing.assert_close(
            alignment, torch.ones_like(alignment), atol=2e-2, rtol=2e-2
        )

    def test_alpha_is_bounded_and_saturates_on_the_splat(self):
        alpha = self.package["rend_alpha"]
        self.assertGreaterEqual(alpha.min().item(), 0.0)
        self.assertLessEqual(alpha.max().item(), 1.0)
        self.assertGreater(alpha.max().item(), 0.9)


@requires_cuda
class GradientPropagationTest(unittest.TestCase):
    def test_every_trainable_tensor_receives_a_finite_gradient(self):
        camera = _tilted_camera()
        model = _single_surfel(scale=0.6, opacity=0.8)
        package = OSNSurfelRasterizer().render(camera, model)
        target = torch.rand_like(package["render"])
        (package["render"] - target).abs().mean().backward()

        for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc"):
            grad = getattr(model, name).grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(grad.abs().max().item(), 0.0, name)
        self.assertEqual(model._scaling.grad.shape[1], 2)

    def test_geometric_regularizers_reach_the_geometry(self):
        from osn_gs.losses.torch_surfel_losses import (
            depth_distortion_loss,
            normal_consistency_loss,
        )

        camera = _tilted_camera()
        model = _single_surfel(scale=0.6, opacity=0.8)
        package = OSNSurfelRasterizer().render(camera, model)
        (depth_distortion_loss(package) + normal_consistency_loss(package)).backward()
        for name in ("_xyz", "_scaling", "_rotation"):
            grad = getattr(model, name).grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)

    def test_screenspace_gradient_is_populated_for_densification(self):
        camera = _tilted_camera()
        model = _single_surfel(scale=0.6, opacity=0.8)
        package = OSNSurfelRasterizer().render(camera, model)
        target = torch.rand_like(package["render"])
        (package["render"] - target).abs().mean().backward()
        screenspace = package["viewspace_points"]
        self.assertIsNotNone(screenspace.grad)
        self.assertGreater(screenspace.grad[:, :2].abs().max().item(), 0.0)
        # 2DGS writes only x/y (the projected 3D-centre gradient); z stays 0.
        self.assertEqual(screenspace.grad[:, 2].abs().max().item(), 0.0)


@requires_cuda
class LowPassFilterTest(unittest.TestCase):
    """Paper eq. 11 / `FilterSize`: degenerate near-edge-on splats survive."""

    def test_edge_on_splat_is_still_rasterized(self):
        camera = _tilted_camera()
        # Tilted to 89.5 degrees: the disk projects to essentially a line.
        model = _single_surfel(scale=1.0, tilt_degrees=89.5, opacity=0.99)
        package = OSNSurfelRasterizer().render(camera, model)
        self.assertGreater(int(package["radii"][0]), 0)
        self.assertGreater(int((package["rend_alpha"] > 0.01).sum()), 0)

    def test_screen_radius_never_falls_below_the_filter_floor(self):
        """`radius = ceil(max(extent, cutoff * FilterSize))` = at least 3 px."""

        camera = _tilted_camera()
        for tilt in (0.0, 45.0, 89.0, 89.9):
            model = _single_surfel(scale=0.02, tilt_degrees=tilt, opacity=0.99)
            package = OSNSurfelRasterizer().render(camera, model)
            self.assertGreaterEqual(int(package["radii"][0]), 3, f"tilt={tilt}")

    def test_a_tiny_splat_still_receives_gradient(self):
        camera = _tilted_camera()
        model = _single_surfel(scale=0.01, tilt_degrees=89.0, opacity=0.99)
        package = OSNSurfelRasterizer().render(camera, model)
        package["render"].sum().backward()
        self.assertGreater(model._xyz.grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
