from __future__ import annotations

"""Phase 1 Surface-Cell Component Builder unit tests
(OSN_GS_Final_Boundary_First_NURBS_Direction.md §Phase 1)."""

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


def _crease(count: int = 800, slope: float = 0.45, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand((count, 2), generator=generator) * 2.0 - 1.0
    z = slope * xy[:, 0].abs()
    return torch.cat([xy, z.reshape(-1, 1)], dim=1)


def _two_parallel_sheets(count: int = 800, gap: float = 0.12, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    half = count // 2
    xy_top = torch.rand((half, 2), generator=generator) * 2.0 - 1.0
    xy_bottom = torch.rand((count - half, 2), generator=generator) * 2.0 - 1.0
    top = torch.cat([xy_top, torch.full((half, 1), gap * 0.5)], dim=1)
    bottom = torch.cat([xy_bottom, torch.full((count - half, 1), -gap * 0.5)], dim=1)
    return torch.cat([top, bottom], dim=0)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class SurfaceComponentBuilderTest(unittest.TestCase):
    def _hierarchy(self, points, **overrides):
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        kwargs = dict(
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=150,
            voxel_max_depth=6,
            voxel_min_size=0.0,
        )
        kwargs.update(overrides)
        return build_voxel_gaussian_hierarchy(points, **kwargs)

    def _components(self, points, **overrides):
        from osn_gs.surface.torch_surface_components import build_surface_components

        hierarchy = self._hierarchy(points)
        kwargs: dict = {}
        kwargs.update(overrides)
        return hierarchy, build_surface_components(hierarchy, points, **kwargs)

    def test_flat_plane_with_hole_is_one_component(self):
        points = _plane_with_hole()
        _, component_set = self._components(points)
        self.assertEqual(component_set.component_count(), 1)
        component = component_set.components[0]
        self.assertTrue(component.boundary_leaf_ids)
        # Every boundary face of the single component must be a support face
        # (the hole and outer silhouette) -- there is nothing else to merge.
        for leaf_id in component.boundary_leaf_ids:
            kinds = {face["kind"] for face in component_set.component_boundary_faces[leaf_id]}
            self.assertTrue(kinds <= {"support"})

    def test_crease_splits_into_two_components(self):
        points = _crease()
        _, component_set = self._components(points)
        self.assertEqual(component_set.component_count(), 2)
        # At least one edge decision must be an incompatible normal (the ridge).
        reasons = component_set.edge_reason_counts()
        self.assertIn("normal", reasons)
        self.assertGreater(reasons["normal"], 0)
        # The split edges must surface as "crease" boundary faces.
        crease_faces = [
            face
            for faces in component_set.component_boundary_faces.values()
            for face in faces
            if face["kind"] == "crease"
        ]
        self.assertTrue(crease_faces)

    def test_close_parallel_sheets_stay_separate(self):
        points = _two_parallel_sheets()
        _, component_set = self._components(points)
        # Each sheet must land in its own component (no cross-gap merge).
        self.assertGreaterEqual(component_set.component_count(), 2)
        for component in component_set.components:
            z = points[component.gaussian_indices][:, 2]
            # A correctly-separated component's z values must not straddle 0.
            self.assertTrue(bool((z > 0).all()) or bool((z < 0).all()))

    def test_offset_incompatibility_marks_crease_reason(self):
        _, component_set = self._components(_two_parallel_sheets())
        reasons = component_set.edge_reason_counts()
        # The two sheets are face-adjacent at the z=0 midplane, so at least
        # one candidate edge must be rejected for being non-coplanar.
        self.assertIn("offset", reasons)

    def test_deterministic_across_runs(self):
        points = _plane_with_hole()
        _, first = self._components(points)
        _, second = self._components(points.clone())
        self.assertEqual(first.leaf_component_id, second.leaf_component_id)
        self.assertEqual(
            [c.member_leaf_ids for c in first.components],
            [c.member_leaf_ids for c in second.components],
        )

    def test_gaussian_indices_partition_mergeable_leaves(self):
        points = _plane_with_hole()
        hierarchy, component_set = self._components(points)
        from osn_gs.surface.torch_voxel_hierarchy import STATE_ACTIVE, STATE_COMPLEX

        expected_total = sum(
            leaf.count for leaf in hierarchy.leaves() if leaf.state in (STATE_ACTIVE, STATE_COMPLEX)
        )
        covered = sum(int(c.gaussian_indices.numel()) for c in component_set.components)
        self.assertEqual(covered, expected_total)
        all_indices = torch.cat([c.gaussian_indices for c in component_set.components])
        self.assertEqual(int(torch.unique(all_indices).numel()), int(all_indices.numel()))

    def test_active_active_shared_face_is_never_a_boundary_within_component(self):
        points = _plane_with_hole()
        _, component_set = self._components(points)
        for component in component_set.components:
            member_set = set(component.member_leaf_ids)
            for leaf_id in component.member_leaf_ids:
                for face in component_set.component_boundary_faces.get(leaf_id, []):
                    if face["neighbor_id"] is not None:
                        self.assertNotIn(face["neighbor_id"], member_set)

    def test_no_mergeable_leaves_returns_empty_set(self):
        from osn_gs.surface.torch_surface_components import build_surface_components

        # A tiny point cloud below voxel_min_gaussian_count yields no active leaves.
        points = torch.rand((3, 3))
        hierarchy = self._hierarchy(points, voxel_min_gaussian_count=10)
        component_set = build_surface_components(hierarchy, points)
        self.assertEqual(component_set.component_count(), 0)

    def test_payload_serializes(self):
        import json

        from osn_gs.surface.torch_surface_components import surface_component_set_payload

        points = _crease()
        _, component_set = self._components(points)
        payload = surface_component_set_payload(component_set)
        json.dumps(payload)  # must not raise
        self.assertEqual(payload["component_count"], component_set.component_count())


if __name__ == "__main__":
    unittest.main()
