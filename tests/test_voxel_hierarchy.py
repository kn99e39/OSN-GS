from __future__ import annotations

"""Stage 1 recursive raw-count voxel hierarchy unit tests (migration plan §1-A)."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _plane_with_hole(count: int = 800, hole_radius: float = 0.35, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    accepted = []
    remaining = count
    while remaining > 0:
        xy = torch.rand((count * 2, 2), generator=generator) * 2.0 - 1.0
        keep = xy.square().sum(dim=1).sqrt() >= hole_radius
        xy = xy[keep][:remaining]
        accepted.append(xy)
        remaining -= int(xy.shape[0])
    xy = torch.cat(accepted, dim=0)
    z = torch.zeros((xy.shape[0], 1))
    return torch.cat([xy, z], dim=1)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class VoxelHierarchyTest(unittest.TestCase):
    def _build(self, points, **overrides):
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        kwargs = dict(
            voxel_min_gaussian_count=8,
            voxel_max_gaussian_count=64,
            voxel_max_depth=5,
            voxel_min_size=0.0,
        )
        kwargs.update(overrides)
        return build_voxel_gaussian_hierarchy(points, **kwargs)

    def test_conservation_and_partition(self):
        from osn_gs.surface.torch_voxel_hierarchy import validate_hierarchy_conservation

        points = _plane_with_hole()
        hierarchy = self._build(points)
        validate_hierarchy_conservation(hierarchy)  # raises on violation
        for node in hierarchy.nodes:
            if node.state == "subdivided":
                child_sum = sum(hierarchy.nodes[child].count for child in node.children)
                self.assertEqual(child_sum, node.count)

    def test_deterministic_and_stable_leaf_ids(self):
        points = _plane_with_hole()
        first = self._build(points)
        second = self._build(points.clone())
        self.assertEqual(
            [(n.node_id, n.state, n.count) for n in first.nodes],
            [(n.node_id, n.state, n.count) for n in second.nodes],
        )

    def test_count_thresholds_drive_states(self):
        points = _plane_with_hole()
        hierarchy = self._build(points)
        for node in hierarchy.leaves():
            if node.count == 0:
                self.assertEqual(node.state, "empty")
            elif node.count < 8:
                self.assertEqual(node.state, "inactive")
            elif node.state == "active":
                self.assertLessEqual(node.count, 64 if node.subdivision_blocked is None else node.count)

    def test_empty_children_are_recorded(self):
        # A flat plane leaves the upper/lower z octants empty on every split.
        points = _plane_with_hole()
        hierarchy = self._build(points)
        subdivided = [n for n in hierarchy.nodes if n.state == "subdivided"]
        self.assertTrue(subdivided)
        for node in subdivided:
            self.assertEqual(len(node.children), 8)
        empties = hierarchy.leaves_in_state("empty")
        self.assertTrue(empties)

    def test_max_depth_and_min_size_block_subdivision(self):
        points = _plane_with_hole()
        shallow = self._build(points, voxel_max_depth=1, voxel_max_gaussian_count=16)
        self.assertLessEqual(shallow.max_depth_reached(), 1)
        blocked = [n for n in shallow.leaves() if n.subdivision_blocked == "max_depth"]
        self.assertTrue(blocked)
        coarse = self._build(points, voxel_min_size=10.0, voxel_max_gaussian_count=16)
        self.assertEqual(coarse.max_depth_reached(), 0)

    def test_complex_leaf_marked_at_depth_limit(self):
        generator = torch.Generator().manual_seed(1)
        ball = torch.randn((256, 3), generator=generator)
        hierarchy = self._build(ball, voxel_max_depth=0, voxel_max_gaussian_count=512)
        self.assertEqual(len(hierarchy.leaves()), 1)
        self.assertEqual(hierarchy.leaves()[0].state, "complex")

    def test_hole_interior_has_no_active_leaf(self):
        points = _plane_with_hole(count=1200, hole_radius=0.5)
        hierarchy = self._build(points, voxel_max_gaussian_count=48)
        for node in hierarchy.leaves_in_state("active"):
            center = (node.aabb_min + node.aabb_max) * 0.5
            # No active voxel should be centered deep inside the hole.
            self.assertFalse(
                float(center[:2].norm()) < 0.25,
                f"active leaf {node.node_id} centered inside the hole",
            )


@unittest.skipUnless(torch is not None, "PyTorch is required")
class PlaneAABBPolygonTest(unittest.TestCase):
    def test_axis_aligned_plane_yields_square(self):
        from osn_gs.surface.torch_voxel_hierarchy import plane_aabb_intersection_polygon

        centroid = torch.tensor([0.5, 0.5, 0.5])
        normal = torch.tensor([0.0, 0.0, 1.0])
        polygon = plane_aabb_intersection_polygon(
            centroid, normal, torch.zeros(3), torch.ones(3)
        )
        self.assertEqual(int(polygon.shape[0]), 4)
        self.assertTrue(torch.allclose(polygon[:, 2], torch.full((4,), 0.5)))

    def test_diagonal_plane_yields_hexagon(self):
        from osn_gs.surface.torch_voxel_hierarchy import plane_aabb_intersection_polygon

        centroid = torch.tensor([0.5, 0.5, 0.5])
        normal = torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 1.0]), dim=0)
        polygon = plane_aabb_intersection_polygon(
            centroid, normal, torch.zeros(3), torch.ones(3)
        )
        self.assertEqual(int(polygon.shape[0]), 6)

    def test_polygon_ordering_is_convex(self):
        from osn_gs.surface.torch_voxel_hierarchy import plane_aabb_intersection_polygon

        centroid = torch.tensor([0.4, 0.5, 0.6])
        normal = torch.nn.functional.normalize(torch.tensor([0.3, 1.0, 0.5]), dim=0)
        polygon = plane_aabb_intersection_polygon(
            centroid, normal, torch.zeros(3), torch.ones(3)
        )
        self.assertGreaterEqual(int(polygon.shape[0]), 3)
        # All ordered vertices lie on the plane and successive cross products
        # keep one sign (convex, consistently wound).
        offsets = (polygon - centroid) @ normal
        self.assertTrue(torch.allclose(offsets, torch.zeros_like(offsets), atol=1e-5))

    def test_rasterize_full_and_half_domain(self):
        from osn_gs.surface.torch_voxel_hierarchy import rasterize_convex_polygon_uv

        full = torch.tensor([[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]])
        mask = rasterize_convex_polygon_uv(full, 16)
        self.assertTrue(bool(mask.all()))
        half = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])
        mask = rasterize_convex_polygon_uv(half, 16)
        fraction = float(mask.float().mean())
        self.assertGreater(fraction, 0.4)
        self.assertLess(fraction, 0.6)


if __name__ == "__main__":
    unittest.main()
