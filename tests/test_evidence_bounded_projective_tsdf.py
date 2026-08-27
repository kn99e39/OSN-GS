"""Worklog 127 -- focused contracts for the evidence-bounded projective TSDF.

Directive section 23's required list, one class per contract. Everything here
targets the isolated experimental module family; no production or shared
behaviour is touched by the code under test.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for path in (str(DEVTOOLS), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evidence_bounded_tsdf import extraction, field as tsdf_field, mesh_ops, scale  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------
class TestProjectiveSignedDistanceSign:
    def test_sign_convention_matches_the_directive(self):
        depth = torch.tensor([3.0, 4.0, 5.0])
        median = torch.tensor([4.0, 4.0, 4.0])
        signed = tsdf_field.projective_signed_distance(depth, median)
        assert float(signed[0]) > 0, "camera-facing side must be positive"
        assert float(signed[1]) == 0.0, "on the renderer median surface must be exactly zero"
        assert float(signed[2]) < 0, "behind the renderer median surface must be negative"

    def test_truncation_clamps_only_outside_the_band(self):
        mu = 0.3
        signed = torch.tensor([-0.9, -0.3, -0.15, 0.0, 0.15, 0.3, 0.9])
        phi = tsdf_field.truncated_phi(signed, mu)
        assert torch.allclose(phi, torch.tensor([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0]))

    def test_only_the_band_carries_authority(self):
        mu = 0.3
        assert abs(-0.30001) > mu and abs(0.29999) <= mu


class TestFrozenCameraDepthSemantics:
    """The candidate must consume the SAME camera-space query-depth quantity the
    frozen worklog 120-123 classifier does."""

    def test_middepth_offset_matches_candidate_b(self):
        from observed_occluded import candidate_b_median_depth as candidate_b

        assert tsdf_field.MIDDEPTH_OFFSET == candidate_b.MIDDEPTH_OFFSET

    def test_near_plane_matches_frozen_shared_module(self):
        from observed_occluded.shared import CANONICAL_NEAR_N

        assert tsdf_field.CANONICAL_NEAR_N == CANONICAL_NEAR_N

    def test_projection_is_bitwise_identical_to_the_frozen_one(self):
        from observed_occluded.shared import project_queries

        torch.manual_seed(0)
        width, height = 37, 23
        world_view = torch.eye(4)
        world_view[3, :3] = torch.tensor([0.1, -0.2, 4.0])
        from osn_gs.data.colmap_scene import projection_matrix

        projection = projection_matrix(0.01, 100.0, 0.7, 0.7, device="cpu").transpose(0, 1).contiguous()
        full_proj = world_view @ projection

        class _Camera:
            image_width = width
            image_height = height
            world_view_transform = world_view
            full_proj_transform = full_proj

        points = (torch.rand(4096, 3) - 0.5) * 6.0
        frozen = project_queries(_Camera(), points)
        ours = tsdf_field.project_world_points(points, world_view, full_proj, width, height)
        assert torch.equal(frozen.depth, ours.depth)
        assert torch.equal(frozen.pixel_index, ours.pixel_index)
        assert torch.equal(frozen.relevant, ours.relevant)


class TestScaleDerivation:
    def test_footprint_is_depth_over_geometric_mean_focal(self):
        class _Camera:
            image_width, image_height = 640, 480
            FoVx, FoVy = 1.0, 0.8

        depth = torch.tensor([0.0, 2.0, 4.0])
        footprints = scale.view_footprints(_Camera(), depth)
        fx, fy = scale.camera_focal_lengths(_Camera())
        assert footprints.shape[0] == 2, "zero (no median event) must be excluded"
        assert math.isclose(float(footprints[0]), 2.0 / math.sqrt(fx * fy), rel_tol=1e-6)

    def test_h_is_the_global_median_and_mu_is_exactly_three_h(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
        derived = scale.derive_canonical_scale(values)
        assert derived.h == pytest.approx(3.5)          # median of six values
        assert derived.mu == pytest.approx(3.0 * derived.h)
        assert scale.TRUNCATION_RATIO == 3.0

    def test_h_is_insensitive_to_the_far_field_tail(self):
        base = torch.arange(1.0, 1001.0)
        with_tail = torch.cat([base, torch.tensor([1e6, 1e7, 1e8])])
        assert scale.derive_canonical_scale(base).h == pytest.approx(500.5)
        assert scale.derive_canonical_scale(with_tail).h == pytest.approx(502.0)


class TestVoxelKeys:
    def test_encode_decode_round_trip(self):
        index = torch.tensor([[0, 0, 0], [1, -1, 5], [-70000, 3, 90000]], dtype=torch.int64)
        keys, dropped = tsdf_field.encode_keys(index)
        assert dropped == 0
        assert torch.equal(tsdf_field.decode_keys(keys), index)

    def test_keys_sort_consistently_and_strides_are_pure_offsets(self):
        index = torch.tensor([[3, 4, 5]], dtype=torch.int64)
        keys, _ = tsdf_field.encode_keys(index)
        for stride, axis in ((tsdf_field.STRIDE_X, 0), (tsdf_field.STRIDE_Y, 1), (tsdf_field.STRIDE_Z, 2)):
            shifted = tsdf_field.decode_keys(keys + stride)
            expected = index.clone()
            expected[0, axis] += 1
            assert torch.equal(shifted, expected)

    def test_margin_drops_voxels_that_could_wrap(self):
        index = torch.tensor([[tsdf_field.KEY_BOUND - 2, 0, 0]], dtype=torch.int64)
        _keys, dropped = tsdf_field.encode_keys(index, margin=4)
        assert dropped == 1

    def test_voxel_centre_is_the_cell_centre(self):
        keys, _ = tsdf_field.encode_keys(torch.tensor([[0, 1, -1]], dtype=torch.int64))
        centres = tsdf_field.voxel_centers(keys, 2.0)
        assert torch.allclose(centres, torch.tensor([[1.0, 3.0, -1.0]]))

    def test_dilation_is_the_exact_linf_ball(self):
        keys, _ = tsdf_field.encode_keys(torch.tensor([[0, 0, 0]], dtype=torch.int64))
        for radius in (1, 2, 3):
            grown = tsdf_field.dilate_linf(keys, radius)
            assert int(grown.numel()) == (2 * radius + 1) ** 3
            decoded = tsdf_field.decode_keys(grown)
            assert int(decoded.abs().max()) == radius

    def test_neighbour_shell_excludes_the_input(self):
        keys, _ = tsdf_field.encode_keys(torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int64))
        shell = tsdf_field.neighbour_shell(keys, radius=1)
        assert not bool(torch.isin(shell, keys).any())
        assert int(shell.numel()) == int(tsdf_field.dilate_linf(keys, 1).numel()) - 2


# --------------------------------------------------------------------------
def _flat_camera(width=48, height=48, distance=4.0, device="cpu"):
    from osn_gs.data.colmap_scene import projection_matrix
    from osn_gs.render.torch_fallback import TorchCamera

    world_view = torch.eye(4, dtype=torch.float32)
    world_view[:3, 3] = torch.tensor([0.0, 0.0, distance])
    world_view = world_view.transpose(0, 1).contiguous().to(device)
    projection = projection_matrix(0.01, 100.0, 0.7, 0.7, device=device).transpose(0, 1).contiguous()
    return TorchCamera(
        image_height=height, image_width=width, world_view_transform=world_view,
        full_proj_transform=world_view @ projection,
        camera_center=torch.tensor([0.0, 0.0, -distance], device=device), FoVx=0.7, FoVy=0.7, image_name="flat",
    )


def _plane_depth_map(camera, plane_z=0.0):
    """Analytic median depth of the plane z = plane_z, for every pixel."""

    width, height = int(camera.image_width), int(camera.image_height)
    rows = torch.arange(height, dtype=torch.float32).reshape(-1, 1).expand(height, width)
    cols = torch.arange(width, dtype=torch.float32).reshape(1, -1).expand(height, width)
    ndc_x = (2.0 * cols + 1.0) / width - 1.0
    ndc_y = (2.0 * rows + 1.0) / height - 1.0
    tan = math.tan(0.7 * 0.5)
    origin_z = float(camera.camera_center[2])
    depth = torch.full((height, width), plane_z - origin_z, dtype=torch.float32)
    _ = ndc_x, ndc_y, tan
    return depth.reshape(-1)


class TestUnknownPreservationAndFusion:
    def test_unknown_voxels_are_absent_never_filled(self):
        camera = _flat_camera()
        depth = _plane_depth_map(camera)
        h = 0.05
        mu = 3 * h
        far = torch.tensor([[0.0, 0.0, 40.0]])
        keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(far, h))
        field = tsdf_field.fuse_views(keys, [(camera, depth)], h=h, mu=mu)
        assert len(field) == 0, "a voxel outside every truncation band must not exist in the store"
        value, count, found = field.lookup(keys)
        assert not bool(found.any())
        assert bool(torch.isnan(value).all()), "UNKNOWN must read back as NaN, not +1/-1/0"
        assert int(count.sum()) == 0

    def test_fusion_weight_is_exactly_one_per_view(self):
        camera = _flat_camera()
        depth = _plane_depth_map(camera)
        h = 0.05
        mu = 3 * h
        # two IDENTICAL views: the mean must be the same value, the count must double
        on_surface = torch.tensor([[0.0, 0.0, -0.5 * h]])
        keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(on_surface, h))
        one = tsdf_field.fuse_views(keys, [(camera, depth)], h=h, mu=mu)
        two = tsdf_field.fuse_views(keys, [(camera, depth), (camera, depth)], h=h, mu=mu)
        assert len(one) == len(two) == 1
        assert float(one.value[0]) == pytest.approx(float(two.value[0]), abs=1e-6)
        assert int(one.support_count[0]) == 1 and int(two.support_count[0]) == 2

    def test_a_single_observation_is_enough(self):
        camera = _flat_camera()
        depth = _plane_depth_map(camera)
        h = 0.05
        point = torch.tensor([[0.0, 0.0, -h]])
        keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(point, h))
        field = tsdf_field.fuse_views(keys, [(camera, depth)], h=h, mu=3 * h)
        assert len(field) == 1 and int(field.support_count[0]) == 1

    def test_support_count_accounting_matches_manual_authority(self):
        cameras = [_flat_camera(distance=4.0), _flat_camera(distance=5.0)]
        depths = [_plane_depth_map(c) for c in cameras]
        h = 0.05
        mu = 3 * h
        points = torch.tensor([[0.0, 0.0, -h], [0.0, 0.0, 10.0 * h], [0.02, -0.01, h]])
        keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(points, h))
        keys = torch.sort(keys).values
        field = tsdf_field.fuse_views(keys, list(zip(cameras, depths)), h=h, mu=mu)
        centres = tsdf_field.voxel_centers(keys, h)
        expected = torch.zeros(3, dtype=torch.int32)
        for camera, depth in zip(cameras, depths):
            authoritative, _signed = tsdf_field.view_authority(centres, camera, depth, mu)
            expected += authoritative.to(torch.int32)
        _value, count, _found = field.lookup(keys)
        assert torch.equal(count, expected)

    def test_no_minimum_view_threshold_exists_in_the_source(self):
        source = (DEVTOOLS / "evidence_bounded_tsdf" / "field.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fuse = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fuse_views"
        )
        # every comparison that touches a count must be `<count> > 0`, i.e.
        # "has ANY authority" -- never `>= k` for some minimum view count.
        count_names = {"counts", "chunk_count", "support_count"}
        checked = 0
        for node in ast.walk(fuse):
            if not isinstance(node, ast.Compare):
                continue
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if not (names & count_names):
                continue
            checked += 1
            assert len(node.ops) == 1 and isinstance(node.ops[0], ast.Gt), ast.dump(node)
            comparator = node.comparators[0]
            assert isinstance(comparator, ast.Constant) and comparator.value == 0, ast.dump(node)
        assert checked >= 1, "expected at least one count comparison to inspect"


class TestMaskedCellExtraction:
    def _plane_field(self, h=0.1, side=6):
        """phi = -(x index - 2.5) so the zero set is a plane between x = 2 and 3."""

        indices = []
        values = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    indices.append((i, j, k))
                    values.append(float(2.5 - i) / 3.0)
        index = torch.tensor(indices, dtype=torch.int64)
        keys, _ = tsdf_field.encode_keys(index)
        order = torch.argsort(keys)
        return tsdf_field.SparseProjectiveTSDF(
            keys=keys[order], value=torch.tensor(values, dtype=torch.float32)[order],
            support_count=torch.ones(len(values), dtype=torch.int32), h=h, mu=3 * h,
        )

    def test_extracts_the_plane(self):
        field = self._plane_field()
        surface = extraction.extract_zero_level_set(field, block=8, batch_blocks=2)
        assert surface.faces.shape[0] > 0
        assert np.allclose(surface.vertices[:, 0], 2.5 * field.h + 0.5 * field.h, atol=1e-6)

    def test_no_triangle_comes_from_a_cell_with_an_unknown_corner(self):
        field = self._plane_field()
        # remove one corner that a crossing cell needs
        drop = tsdf_field.encode_keys(torch.tensor([[2, 2, 2]], dtype=torch.int64))[0][0]
        keep = field.keys != drop
        holed = tsdf_field.SparseProjectiveTSDF(
            keys=field.keys[keep], value=field.value[keep],
            support_count=field.support_count[keep], h=field.h, mu=field.mu,
        )
        full = extraction.extract_zero_level_set(field, block=8, batch_blocks=2)
        holed_surface = extraction.extract_zero_level_set(holed, block=8, batch_blocks=2)
        assert holed_surface.stats["eligible_cells_authoritative_and_sign_changing"] < \
            full.stats["eligible_cells_authoritative_and_sign_changing"]
        # every cell that touches the removed voxel must be gone
        centroids = holed_surface.vertices[holed_surface.faces].mean(axis=1) / field.h - 0.5
        cell = np.floor(centroids).astype(np.int64)
        touching = np.all((cell >= 1) & (cell <= 2), axis=1)
        assert int(touching.sum()) == 0

    def test_sentinel_choice_cannot_change_kept_triangles(self):
        """Marching cubes is per-cell, so filling UNKNOWN with ANY sentinel and
        then discarding ineligible cells is provably the masked extraction."""

        field = self._plane_field()
        drop = tsdf_field.encode_keys(torch.tensor([[2, 2, 2], [3, 3, 3]], dtype=torch.int64))[0]
        keep = ~torch.isin(field.keys, drop)
        holed = tsdf_field.SparseProjectiveTSDF(
            keys=field.keys[keep], value=field.value[keep],
            support_count=field.support_count[keep], h=field.h, mu=field.mu,
        )
        positive = extraction.extract_zero_level_set(holed, block=8, batch_blocks=2, sentinel=+7.0)
        negative = extraction.extract_zero_level_set(holed, block=8, batch_blocks=2, sentinel=-7.0)
        assert positive.faces.shape == negative.faces.shape
        assert np.allclose(np.sort(positive.vertices, axis=0), np.sort(negative.vertices, axis=0))

    def test_eligibility_needs_both_authority_and_a_sign_change(self):
        h = 0.1
        index = torch.tensor(
            [(i, j, k) for i in range(3) for j in range(3) for k in range(3)], dtype=torch.int64
        )
        keys, _ = tsdf_field.encode_keys(index)
        order = torch.argsort(keys)
        all_positive = tsdf_field.SparseProjectiveTSDF(
            keys=keys[order], value=torch.full((27,), 0.4), support_count=torch.ones(27, dtype=torch.int32),
            h=h, mu=3 * h,
        )
        surface = extraction.extract_zero_level_set(all_positive, block=4, batch_blocks=1)
        assert surface.stats["cells_with_all_eight_authoritative_corners"] > 0
        assert surface.stats["eligible_cells_authoritative_and_sign_changing"] == 0
        assert surface.faces.shape[0] == 0

    def test_seam_weld_never_merges_distinct_geometry(self):
        vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 2, 3], [1, 2, 3]], dtype=np.int64)
        supports = np.array([1, 1, 2, 3], dtype=np.int32)
        values = np.zeros(4, dtype=np.float32)
        welded_v, welded_f, welded_s, _ = extraction.weld_block_seams(vertices, faces, supports, values, 1.0)
        assert welded_v.shape[0] == 3, "the duplicated seam vertex merges"
        assert welded_f.max() == 2
        assert set(welded_s.tolist()) == {1, 2, 3}


class TestConstructionIsolation:
    """Directive section 8: the control experiment IS this isolation."""

    FORBIDDEN = (
        "torch_camera_induced_visible_adjacency", "torch_coverage_first_subset_partition",
        "torch_camera_observed_chart_domains", "torch_nurbs", "torch_surface_candidate_graph",
        "torch_boundary", "torch_chart", "torch_region", "torch_voxel_regions",
        "torch_maximal_visible_connectivity", "torch_positive_visible_adjacency",
        "torch_exact_knn_performance", "torch_gaussian_manifold_affinity",
        "observed_occluded", "torch_occluded", "trust",
    )

    @pytest.mark.parametrize("module", ["scale", "field", "extraction", "mesh_ops"])
    def test_construction_modules_import_nothing_historical(self, module):
        source = (DEVTOOLS / "evidence_bounded_tsdf" / f"{module}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            for forbidden in self.FORBIDDEN:
                assert forbidden not in name, f"{module}.py imports {name}"

    def test_synthetic_module_imports_nothing_historical_either(self):
        source = (DEVTOOLS / "evidence_bounded_tsdf" / "synthetic.py").read_text(encoding="utf-8")
        for forbidden in self.FORBIDDEN:
            assert forbidden not in source, f"synthetic.py mentions {forbidden}"

    def test_attribution_is_the_only_module_allowed_to_read_history(self):
        from evidence_bounded_tsdf import CONSTRUCTION_MODULES

        assert "attribution" not in CONSTRUCTION_MODULES
        source = (DEVTOOLS / "evidence_bounded_tsdf" / "attribution.py").read_text(encoding="utf-8")
        assert "observed_occluded" in source, "attribution is the module that reads the frozen candidates"


class TestMeshOps:
    def test_point_triangle_distance_matches_hand_computed_cases(self):
        triangle = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).expand(5, 3, 3).contiguous()
        points = torch.tensor([
            [0.25, 0.25, 0.7],     # above the interior
            [0.0, 0.0, 0.0],       # on vertex a
            [-1.0, 0.0, 0.0],      # beyond vertex a along -x
            [0.5, -0.5, 0.0],      # off edge ab
            [2.0, 2.0, 0.0],       # beyond edge bc
        ])
        distance = mesh_ops.point_triangle_distance(points, triangle)
        expected = torch.tensor([0.7, 0.0, 1.0, 0.5, math.hypot(1.5, 1.5)])
        assert torch.allclose(distance, expected, atol=1e-5)

    def test_triangle_area_and_components(self):
        vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        assert float(mesh_ops.triangle_areas(vertices, faces).sum()) == pytest.approx(1.0)
        _labels, count = mesh_ops.connected_components(6, faces)
        assert count == 2

    def test_nearest_surface_distance_reports_infinity_when_nothing_is_near(self):
        vertices = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        index = mesh_ops.build_triangle_cell_index(vertices, faces, 0.5)
        near = mesh_ops.nearest_surface_distance(torch.tensor([[0.2, 0.2, 0.05]]), index, max_radius=3)
        far = mesh_ops.nearest_surface_distance(torch.tensor([[0.2, 0.2, 40.0]]), index, max_radius=3)
        assert float(near[0]) == pytest.approx(0.05, abs=1e-5)
        assert not bool(torch.isfinite(far[0])), "no local surface must stay +inf, never a filled-in number"

    def test_rasterized_raycast_reproduces_a_known_plane_depth(self):
        camera = _flat_camera(width=32, height=32, distance=4.0)
        vertices = torch.tensor([
            [-3.0, -3.0, 0.0], [3.0, -3.0, 0.0], [3.0, 3.0, 0.0], [-3.0, 3.0, 0.0],
        ], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        depth, stats = mesh_ops.rasterize_mesh_depth(camera, vertices, faces)
        finite = torch.isfinite(depth)
        assert bool(finite.any())
        assert torch.allclose(depth[finite], torch.full_like(depth[finite], 4.0), atol=1e-4)
        assert stats["triangles_beyond_largest_tier"] == 0

    def test_raycast_keeps_the_nearest_of_two_layers(self):
        camera = _flat_camera(width=32, height=32, distance=4.0)
        vertices = torch.tensor([
            [-3.0, -3.0, 0.0], [3.0, -3.0, 0.0], [3.0, 3.0, 0.0],
            [-3.0, -3.0, -1.0], [3.0, -3.0, -1.0], [3.0, 3.0, -1.0],
        ], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        depth, _stats = mesh_ops.rasterize_mesh_depth(camera, vertices, faces)
        finite = torch.isfinite(depth)
        assert torch.allclose(depth[finite], torch.full_like(depth[finite], 3.0), atol=1e-4)


class TestDeterminism:
    def test_reconstruction_planning_is_deterministic(self):
        camera = _flat_camera()
        depth = _plane_depth_map(camera)
        h = 0.05
        seeds = []
        for _ in range(2):
            valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)
            world = tsdf_field.unproject_pixels(camera, valid, depth[valid])
            keys, _dropped = tsdf_field.encode_keys(tsdf_field.voxel_index_of(world, h), margin=8)
            seeds.append(torch.unique(keys))
        assert torch.equal(seeds[0], seeds[1])

    def test_fusion_is_chunk_size_invariant(self):
        camera = _flat_camera()
        depth = _plane_depth_map(camera)
        h = 0.05
        index = torch.tensor(
            [(i, j, k) for i in range(-4, 4) for j in range(-4, 4) for k in range(-4, 4)], dtype=torch.int64
        )
        keys, _ = tsdf_field.encode_keys(index)
        keys = torch.unique(keys)
        small = tsdf_field.fuse_views(keys, [(camera, depth)], h=h, mu=3 * h, chunk_size=7)
        large = tsdf_field.fuse_views(keys, [(camera, depth)], h=h, mu=3 * h, chunk_size=10 ** 9)
        assert torch.equal(small.keys, large.keys)
        assert torch.equal(small.value, large.value)
        assert torch.equal(small.support_count, large.support_count)

    def test_frontier_closure_equals_full_shell_closure(self):
        """The closure loop tests only the shell of the PREVIOUS round's new
        voxels. That is exact -- a voxel the truncation rule already rejected can
        never become authoritative later -- and this asserts it reaches the same
        fixed point as growing the whole field's shell every round."""

        camera_a = _flat_camera(distance=4.0)
        camera_b = _flat_camera(distance=4.6)
        views = [(camera_a, _plane_depth_map(camera_a)), (camera_b, _plane_depth_map(camera_b))]
        h = 0.05
        mu = 3 * h
        valid = torch.nonzero(views[0][1] > 0, as_tuple=False).reshape(-1)
        world = tsdf_field.unproject_pixels(camera_a, valid, views[0][1][valid])
        seeds, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(world, h), margin=64)
        seeds = torch.unique(seeds)

        fast, report = tsdf_field.grow_field_to_closure(seeds, views, h=h, mu=mu, max_rounds=40)

        reference = tsdf_field.fuse_views(seeds, views, h=h, mu=mu)
        for _ in range(40):
            shell = tsdf_field.neighbour_shell(reference.keys, radius=1)
            if shell.numel() == 0:
                break
            grown = tsdf_field.fuse_views(shell, views, h=h, mu=mu)
            if len(grown) == 0:
                break
            merged = torch.cat([reference.keys, grown.keys])
            order = torch.argsort(merged)
            reference = tsdf_field.SparseProjectiveTSDF(
                keys=merged[order], value=torch.cat([reference.value, grown.value])[order],
                support_count=torch.cat([reference.support_count, grown.support_count])[order], h=h, mu=mu,
            )
        assert report["closed"]
        assert torch.equal(fast.keys, reference.keys)
        assert torch.equal(fast.value, reference.value)
        assert torch.equal(fast.support_count, reference.support_count)

    def test_closure_is_a_fixed_point_when_it_reports_closed(self):
        camera = _flat_camera()
        depth = _plane_depth_map(camera)
        h = 0.05
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)
        world = tsdf_field.unproject_pixels(camera, valid, depth[valid])
        seeds, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(world, h), margin=64)
        seeds = torch.unique(seeds)
        field, report = tsdf_field.grow_field_to_closure(seeds, [(camera, depth)], h=h, mu=3 * h, max_rounds=30)
        if report["closed"]:
            shell = tsdf_field.neighbour_shell(field.keys, radius=1)
            grown = tsdf_field.fuse_views(shell, [(camera, depth)], h=h, mu=3 * h)
            assert len(grown) == 0, "a closed report must mean the whole shell was rejected"


@CUDA
class TestSyntheticContracts:
    """Directive section 9. S2 and S7 are STOP contracts."""

    @pytest.fixture(scope="class")
    def results(self):
        from evidence_bounded_tsdf import synthetic

        return {
            "S1": synthetic.s1_single_open_plane_patch("cuda").metrics,
            "S2": synthetic.s2_two_coplanar_patches_with_gap("cuda").metrics,
            "S3": synthetic.s3_curved_open_sheet("cuda").metrics,
            "S4": synthetic.s4_two_distinct_depth_layers("cuda").metrics,
            "S5": synthetic.s5_cross_view_disocclusion("cuda").metrics,
            "S6": synthetic.s6_thin_structure("cuda").metrics,
            "S7": synthetic.s7_true_occluded_gap("cuda").metrics,
        }

    def test_s1_open_patch_is_recovered_and_stays_open(self, results):
        s1 = results["S1"]
        assert s1["known_surface_coverage_within_h"] > 0.99
        assert s1["point_to_surface_median_over_h"] < 0.1
        assert s1["cap_like_area_fraction"] == 0.0, "no cap may be created at the open boundary"
        assert s1["extent_beyond_patch_x_over_h"] <= 1.0, "no continuation beyond the supported footprint"
        assert 0.9 < s1["reconstructed_area_over_ground_truth"] < 1.1

    def test_s2_unsupported_gap_is_never_bridged(self, results):
        s2 = results["S2"]
        assert s2["gap_half_width_over_mu"] > 3.0, "the fixture must present a genuinely wide gap"
        assert s2["gap_bridging_triangle_count"] == 0
        assert s2["gap_bridging_surface_area"] == 0.0
        assert s2["component_count"] >= 2

    def test_s3_curved_sheet_is_recovered_without_closing(self, results):
        s3 = results["S3"]
        assert s3["radius_error_median_over_h"] < 0.25
        assert s3["beyond_arc_triangle_count"] == 0
        assert s3["opposite_hemisphere_triangle_count"] == 0

    def test_s4_layers_are_separate(self, results):
        s4 = results["S4"]
        assert s4["front_layer_triangles"] > 0 and s4["rear_layer_triangles"] > 0
        assert s4["connecting_sheet_triangles"] == 0
        assert s4["known_surface_coverage_within_h"] > 0.95

    def test_s5_disocclusion_is_recovered_globally(self, results):
        assert results["S5"]["disocclusion_recovered"] is True
        assert results["S5"]["rear_plane_triangles_behind_the_blocker"] > 0

    def test_s6_thin_structure_is_reported_not_rescued(self, results):
        columns = results["S6"]["per_column"]
        assert len(columns) == 4
        assert all("preserved" in c for c in columns)
        assert results["S6"]["note"].startswith("resolution NOT tuned")

    def test_s7_true_occluded_gap_is_verified_and_never_crossed(self, results):
        s7 = results["S7"]
        assert s7["never_observed_samples_verified"] is True, "the fixture must actually occlude"
        assert s7["strip_probe_samples_directly_observed"] == 0
        assert s7["gap_bridging_triangle_count"] == 0

    def test_every_fixture_reports_zero_unsupported_triangles(self, results):
        for name, metrics in results.items():
            assert metrics["unsupported_triangle_count"] == 0, name
            assert metrics["unsupported_surface_area"] == 0.0, name
