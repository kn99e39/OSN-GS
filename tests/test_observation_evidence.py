from __future__ import annotations

"""Phase C isolated observation-evidence / free-space-query tests."""

import unittest

import torch

from osn_gs.data.colmap_scene import projection_matrix
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.diff_gaussian_loader import get_diff_gaussian_backend
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.surface.torch_observation_evidence import (
    STATUS_CONFLICTING_EVIDENCE,
    STATUS_KNOWN_FREE_SPACE,
    STATUS_OCCLUDED_CANDIDATE,
    STATUS_ON_OBSERVED_SURFACE,
    STATUS_OUTSIDE_VALID_VIEW,
    STATUS_UNOBSERVED,
    VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE,
    VIEW_STATUS_KNOWN_FREE_SPACE,
    VIEW_STATUS_ON_OBSERVED_SURFACE,
    VOXEL_SUPPORT_NONE_OBSERVED,
    build_observation_evidence,
    classify_world_samples,
    evidence_cache_key,
    query_empty_voxel_support,
)
from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy


def _lookat_world_view(eye, target, up):
    eye_t = torch.tensor(eye, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.float32)
    up_t = torch.tensor(up, dtype=torch.float32)
    forward = torch.nn.functional.normalize(target_t - eye_t, dim=0, eps=1e-8)
    right = torch.nn.functional.normalize(torch.cross(forward, up_t, dim=0), dim=0, eps=1e-8)
    true_up = torch.cross(right, forward, dim=0)
    rotation = torch.stack([right, true_up, forward], dim=0)
    translation = -(rotation @ eye_t)
    world_view = torch.eye(4, dtype=torch.float32)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = translation
    return world_view.transpose(0, 1).contiguous(), eye_t


def _build_camera(world_view_and_center, fovx=0.9, fovy=0.9, height=64, width=64):
    world_view, camera_center = world_view_and_center
    projection = projection_matrix(0.01, 100.0, fovx, fovy, device="cpu").transpose(0, 1).contiguous()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return TorchCamera(
        image_height=height,
        image_width=width,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
        camera_center=camera_center,
        FoVx=fovx,
        FoVy=fovy,
    )


def _grid_points(x_range, y_range, z, steps):
    xs = torch.linspace(x_range[0], x_range[1], steps)
    ys = torch.linspace(y_range[0], y_range[1], steps)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    zz = torch.full_like(xx, z)
    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)


def _build_model(positions, device="cpu", opacities=None):
    positions = positions.to(device=device)
    colors = torch.full((positions.shape[0], 3), 0.5, device=device)
    scales = torch.full((positions.shape[0], 3), 0.03, device=device)
    model = TorchGaussianModel(sh_degree=0, device=device)
    model.initialize(positions=positions, colors=colors, scales=scales, opacities=opacities)
    return model


def _wall_with_gap_scene(*, include_conflict_cameras: bool = False):
    """Wall A (observed) at z=2, an empty gap in front and a potential-occlusion
    region behind it, a far off-frustum cluster, and camera 2 pointed away for
    the outside_valid_view case. When ``include_conflict_cameras`` is set, adds
    camera 3 + a small Wall B patch positioned so camera 3 sees past Wall A's
    finite extent to Wall B, giving genuine multi-camera disagreement for the
    sample point at (0, 0, 3) used by both the behind-surface and conflict tests.
    """

    wall_a = _grid_points((-0.5, 0.5), (-0.5, 0.5), 2.0, 11)
    outside_cluster = torch.tensor(
        [[50.0, 0.0, 1.0], [50.1, 0.0, 1.0], [49.9, 0.1, 1.0]], dtype=torch.float32
    )
    points = [wall_a, outside_cluster]

    camera_1 = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    camera_2 = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    cameras = [camera_1, camera_2]

    if include_conflict_cameras:
        # Positioned so its ray to (0, 0, 3) misses Wall A's [-0.5, 0.5] extent
        # at z=2 (crosses x=1.667 there) but resolves Wall B farther along the
        # same ray -- verified numerically, see docs/worklogs/76.
        wall_b = _grid_points((-6.0, -4.0), (-1.0, 1.0), 6.0, 11)
        points.append(wall_b)
        camera_3 = _build_camera(_lookat_world_view((5.0, 0.0, 0.0), (0.0, 0.0, 3.0), (0.0, 1.0, 0.0)))
        cameras.append(camera_3)

    positions = torch.cat(points, dim=0)
    model = _build_model(positions)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))
    evidence = build_observation_evidence(cameras, model, rasterizer, depth_epsilon=1e-2)
    return evidence, positions, model


def _on_surface_multi_view_scene(*, include_free_camera: bool = False):
    """Wall A at z=2 (sample (0,0,2) sits exactly on it, giving camera A
    ``on_observed_surface``). Camera B is offset and aimed at the sample point
    directly; a small high-opacity occluder ("Wall G") sits nearer along that
    same ray, so camera B resolves ``behind_first_observed_surface`` there
    instead (verified numerically: obs_depth ~3.67 < sample_depth ~5.39).
    High opacity (0.95) on Wall G is required so its contribution dominates
    the naive screen-proximity-weighted blend over Wall A's own single exact
    center-point Gaussian, which also projects to camera B's same pixel (the
    fallback renderer has no real depth-ordered occlusion -- documented as a
    known limitation in worklog 78). When ``include_free_camera`` is set, adds
    camera C on the opposite side aimed at the same sample point, with a farther surface
    ("Wall H") beyond it along that ray, giving a clean ``known_free_space``
    reading for the same sample and completing the free+behind+on-surface
    triple combination.
    """

    wall_a = _grid_points((-0.5, 0.5), (-0.5, 0.5), 2.0, 11)
    wall_g = _grid_points((2.4, 2.6), (-0.1, 0.1), 1.0, 3)
    points = [wall_a, wall_g]
    opacities = [torch.full((wall_a.shape[0], 1), 0.1), torch.full((wall_g.shape[0], 1), 0.95)]

    camera_a = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    camera_b = _build_camera(_lookat_world_view((5.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.0, 1.0, 0.0)))
    cameras = [camera_a, camera_b]

    if include_free_camera:
        wall_h = _grid_points((4.9, 5.1), (-0.1, 0.1), 4.0, 3)
        points.append(wall_h)
        opacities.append(torch.full((wall_h.shape[0], 1), 0.5))
        camera_c = _build_camera(_lookat_world_view((-5.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.0, 1.0, 0.0)))
        cameras.append(camera_c)

    positions = torch.cat(points, dim=0)
    model = _build_model(positions, opacities=torch.cat(opacities, dim=0))
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))
    evidence = build_observation_evidence(cameras, model, rasterizer, depth_epsilon=1e-2)
    return evidence


class ObservationEvidenceTest(unittest.TestCase):
    def test_sample_in_front_of_camera_is_known_free_space(self):
        evidence, _, _ = _wall_with_gap_scene()
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 1.0]]))[0]
        self.assertEqual(result.status, STATUS_KNOWN_FREE_SPACE)

    def test_sample_behind_observed_surface_is_flagged(self):
        evidence, _, _ = _wall_with_gap_scene()
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 3.0]]))[0]
        self.assertEqual(result.status, STATUS_OCCLUDED_CANDIDATE)

    def test_sample_outside_every_camera_view_is_outside_valid_view(self):
        evidence, _, _ = _wall_with_gap_scene()
        result = classify_world_samples(evidence, torch.tensor([[50.0, 0.0, 1.0]]))[0]
        self.assertEqual(result.status, STATUS_OUTSIDE_VALID_VIEW)

    def test_on_surface_only_aggregates_as_on_observed_surface(self):
        # Only camera 1 resolves this sample at all (on Wall A's own surface);
        # camera 2 is pointed away and contributes outside_valid_view, which
        # must not change the on-surface-only aggregate (Gate C round 2).
        evidence, _, _ = _wall_with_gap_scene()
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 2.0]]))[0]
        on_surface_views = [v for v in result.per_view if v.status == VIEW_STATUS_ON_OBSERVED_SURFACE]
        self.assertGreaterEqual(len(on_surface_views), 1)
        self.assertNotEqual(result.status, STATUS_KNOWN_FREE_SPACE)
        self.assertNotEqual(result.status, STATUS_OCCLUDED_CANDIDATE)
        self.assertEqual(result.status, STATUS_ON_OBSERVED_SURFACE)

    def test_multi_view_conflict_is_preserved_not_collapsed(self):
        evidence, _, _ = _wall_with_gap_scene(include_conflict_cameras=True)
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 3.0]]))[0]
        by_camera = {v.camera_index: v.status for v in result.per_view}
        # Explicit preconditions: prove the fixture actually creates disagreement.
        self.assertEqual(by_camera[0], VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE)
        self.assertEqual(by_camera[2], VIEW_STATUS_KNOWN_FREE_SPACE)
        self.assertEqual(result.status, STATUS_CONFLICTING_EVIDENCE)
        self.assertIn(2, result.free_space_confirmed_by)
        self.assertIn(0, result.behind_surface_in)

    def test_multi_view_on_surface_vetoes_known_free_space(self):
        evidence, _, _ = _wall_with_gap_scene(include_conflict_cameras=True)
        # Sample sits exactly on Wall A's surface for camera 1, but camera 3
        # (aimed past Wall A's edge at Wall B) resolves a farther surface here.
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 2.0]]))[0]
        by_camera = {v.camera_index: v.status for v in result.per_view}
        # Explicit preconditions: prove the fixture actually creates disagreement.
        self.assertEqual(by_camera[0], VIEW_STATUS_ON_OBSERVED_SURFACE)
        self.assertEqual(by_camera[2], VIEW_STATUS_KNOWN_FREE_SPACE)
        self.assertNotEqual(result.status, STATUS_KNOWN_FREE_SPACE)
        self.assertEqual(result.status, STATUS_CONFLICTING_EVIDENCE)
        self.assertIn(2, result.free_space_confirmed_by)
        self.assertIn(0, result.on_surface_in)

    def test_multi_view_on_surface_and_behind_are_conflicting(self):
        # Camera A resolves the sample as exactly on Wall A's own surface;
        # camera B, aimed at the same point, resolves a nearer occluder
        # (Wall G) first -- neither camera reports free space here.
        evidence = _on_surface_multi_view_scene()
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 2.0]]))[0]
        by_camera = {v.camera_index: v.status for v in result.per_view}
        self.assertEqual(by_camera[0], VIEW_STATUS_ON_OBSERVED_SURFACE)
        self.assertEqual(by_camera[1], VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE)
        self.assertNotEqual(result.status, STATUS_KNOWN_FREE_SPACE)
        self.assertNotEqual(result.status, STATUS_OCCLUDED_CANDIDATE)
        self.assertNotEqual(result.status, STATUS_ON_OBSERVED_SURFACE)
        self.assertEqual(result.status, STATUS_CONFLICTING_EVIDENCE)
        self.assertEqual(result.reason, "behind_and_on_surface_conflict")
        self.assertIn(0, result.on_surface_in)
        self.assertIn(1, result.behind_surface_in)

    def test_multi_view_free_behind_and_on_surface_triple_conflict(self):
        # Adds a third camera (aimed at the same point, with a farther Wall H
        # beyond it) so all three per-view evidence kinds appear for one sample.
        evidence = _on_surface_multi_view_scene(include_free_camera=True)
        result = classify_world_samples(evidence, torch.tensor([[0.0, 0.0, 2.0]]))[0]
        by_camera = {v.camera_index: v.status for v in result.per_view}
        self.assertEqual(by_camera[0], VIEW_STATUS_ON_OBSERVED_SURFACE)
        self.assertEqual(by_camera[1], VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE)
        self.assertEqual(by_camera[2], VIEW_STATUS_KNOWN_FREE_SPACE)
        self.assertEqual(result.status, STATUS_CONFLICTING_EVIDENCE)
        self.assertEqual(result.reason, "free_space_behind_and_on_surface_conflict")
        self.assertIn(0, result.on_surface_in)
        self.assertIn(1, result.behind_surface_in)
        self.assertIn(2, result.free_space_confirmed_by)

    def test_evidence_cache_key_reacts_to_appearance_camera_and_config_changes(self):
        wall_a = _grid_points((-0.5, 0.5), (-0.5, 0.5), 2.0, 11)
        model = _build_model(wall_a)
        camera_1 = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))

        base = build_observation_evidence([camera_1], model, rasterizer, depth_epsilon=1e-2)
        base_key = evidence_cache_key(base)

        # Identical rebuild -> identical key (determinism control).
        repeat = build_observation_evidence([camera_1], model, rasterizer, depth_epsilon=1e-2)
        self.assertEqual(base_key, evidence_cache_key(repeat))

        # Scale-only change.
        scaled_model = _build_model(wall_a)
        with torch.no_grad():
            scaled_model._scaling.add_(0.5)
        scaled = build_observation_evidence([camera_1], scaled_model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(base_key, evidence_cache_key(scaled))

        # Rotation-only change.
        rotated_model = _build_model(wall_a)
        with torch.no_grad():
            rotated_model._rotation[:, 1].add_(0.1)
        rotated = build_observation_evidence([camera_1], rotated_model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(base_key, evidence_cache_key(rotated))

        # Opacity-only change.
        opaque_model = _build_model(wall_a)
        with torch.no_grad():
            opaque_model._opacity.add_(2.0)
        opaque = build_observation_evidence([camera_1], opaque_model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(base_key, evidence_cache_key(opaque))

        # Camera projection (FoV) change -- full_proj_transform differs.
        wide_camera = _build_camera(
            _lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), fovx=1.4, fovy=1.4
        )
        wide = build_observation_evidence([wide_camera], model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(base_key, evidence_cache_key(wide))

        # Query-config-only change.
        tighter_epsilon = build_observation_evidence([camera_1], model, rasterizer, depth_epsilon=1e-4)
        self.assertNotEqual(base_key, evidence_cache_key(tighter_epsilon))

    def test_evidence_cache_key_reacts_to_resolution_only_change(self):
        wall_a = _grid_points((-0.5, 0.5), (-0.5, 0.5), 2.0, 11)
        model = _build_model(wall_a)
        rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))

        camera_1 = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        base = build_observation_evidence([camera_1], model, rasterizer, depth_epsilon=1e-2)

        # Same pose/FoV, only image_height/image_width differ. world_view_transform
        # and full_proj_transform are unaffected by resolution alone, so this
        # isolates the height/width fields specifically (Gate C round 2).
        higher_res_camera = _build_camera(
            _lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=128, width=128
        )
        higher_res = build_observation_evidence([higher_res_camera], model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(evidence_cache_key(base), evidence_cache_key(higher_res))

    def test_evidence_cache_key_reacts_to_identity_only_change(self):
        wall_a = _grid_points((-0.5, 0.5), (-0.5, 0.5), 2.0, 11)
        model = _build_model(wall_a)
        rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))

        camera_1 = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        camera_1.image_name = "front_camera"
        base = build_observation_evidence([camera_1], model, rasterizer, depth_epsilon=1e-2)

        # Byte-identical transforms/dims, only image_name differs.
        camera_1_renamed = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        camera_1_renamed.image_name = "other_camera"
        renamed = build_observation_evidence([camera_1_renamed], model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(evidence_cache_key(base), evidence_cache_key(renamed))

        # Two cameras sharing the default image_name ("camera") in a different
        # list order must still get different keys -- proves the explicit
        # camera_index identity component (not just image_name) is load-bearing.
        camera_x = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        camera_y = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
        order_ab = build_observation_evidence([camera_x, camera_y], model, rasterizer, depth_epsilon=1e-2)
        order_ba = build_observation_evidence([camera_y, camera_x], model, rasterizer, depth_epsilon=1e-2)
        self.assertNotEqual(evidence_cache_key(order_ab), evidence_cache_key(order_ba))

    @unittest.skipUnless(
        torch.cuda.is_available() and get_diff_gaussian_backend() is not None,
        "CUDA rasterizer is required",
    )
    def test_cuda_fallback_depth_parity(self):
        # Only guaranteed for this low-depth-variance fixture (a single frontal
        # wall on-axis) -- see the module docstring's E[1/z] vs 1/E[z] caveat.
        wall_a = _grid_points((-0.5, 0.5), (-0.5, 0.5), 2.0, 11)
        model = _build_model(wall_a, device="cuda")
        camera_1 = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        camera_1.world_view_transform = camera_1.world_view_transform.cuda()
        camera_1.full_proj_transform = camera_1.full_proj_transform.cuda()
        camera_1.camera_center = camera_1.camera_center.cuda()

        cuda_rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True))
        fallback_rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))
        cuda_evidence = build_observation_evidence([camera_1], model, cuda_rasterizer, depth_epsilon=1e-2)
        fallback_evidence = build_observation_evidence([camera_1], model, fallback_rasterizer, depth_epsilon=1e-2)

        sample = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
        cuda_results = classify_world_samples(cuda_evidence, sample)
        fallback_results = classify_world_samples(fallback_evidence, sample)
        for cuda_result, fallback_result in zip(cuda_results, fallback_results):
            self.assertEqual(cuda_result.status, fallback_result.status)
        cuda_depth = cuda_evidence.views[0].view_depth.cpu()
        fallback_depth = fallback_evidence.views[0].view_depth.cpu()
        finite_mask = torch.isfinite(cuda_depth) & torch.isfinite(fallback_depth)
        torch.testing.assert_close(
            cuda_depth[finite_mask], fallback_depth[finite_mask], atol=1e-2, rtol=1e-2
        )

    def test_empty_voxel_query_never_exceeds_no_observed_support(self):
        _, positions, _ = _wall_with_gap_scene()
        hierarchy = build_voxel_gaussian_hierarchy(positions, voxel_max_gaussian_count=20)
        result = query_empty_voxel_support(
            hierarchy, torch.tensor([-0.1, -0.1, 0.9]), torch.tensor([0.1, 0.1, 1.1])
        )
        self.assertEqual(result.support, VOXEL_SUPPORT_NONE_OBSERVED)
        self.assertEqual(
            set(vars(result).keys()), {"query_min", "query_max", "overlapping_empty_leaf_ids", "support"}
        )

    def test_free_space_never_asserted_for_occluded_candidate_point(self):
        evidence, _, _ = _wall_with_gap_scene()
        zs = torch.linspace(2.01, 5.0, 30)
        sweep = torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], dim=1)
        results = classify_world_samples(evidence, sweep)
        false_accept_count = sum(1 for result in results if result.status == STATUS_KNOWN_FREE_SPACE)
        self.assertEqual(false_accept_count, 0)


if __name__ == "__main__":
    unittest.main()
