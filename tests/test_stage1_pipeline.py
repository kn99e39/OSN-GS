from __future__ import annotations

"""Stage 1 voxel-per-patch pipeline tests (migration plan §1-B / §1-C)."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _annulus_points(count: int = 900, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    accepted = []
    remaining = count
    while remaining > 0:
        xy = torch.rand((count * 3, 2), generator=generator) * 2.0 - 1.0
        r = xy.square().sum(dim=1).sqrt()
        xy = xy[(r >= 0.32) & (r <= 0.9)][:remaining]
        accepted.append(xy)
        remaining -= int(xy.shape[0])
    xy = torch.cat(accepted, dim=0)
    points = torch.cat([xy, torch.zeros((xy.shape[0], 1))], dim=1)
    colors = torch.rand((xy.shape[0], 3), generator=generator)
    return points, colors


@unittest.skipUnless(torch is not None, "PyTorch is required")
class Stage1PipelineTest(unittest.TestCase):
    def _stage1_config(self, **overrides):
        from osn_gs.core.torch_pipeline import TorchPipelineConfig

        kwargs = dict(
            nurbs_constructor_mode="voxel_patch_stage1",
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=120,
            voxel_max_depth=4,
            visible_surface_resolution_u=8,
            visible_surface_resolution_v=4,
        )
        kwargs.update(overrides)
        return TorchPipelineConfig(**kwargs)

    def test_stage1_builds_one_patch_per_active_leaf(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline

        points, colors = _annulus_points()
        pipeline = TorchOSNGSPipeline(self._stage1_config(), device="cpu")
        state = pipeline.initialize(points, colors)

        self.assertIsNotNone(state.voxel_hierarchy)
        active = state.voxel_hierarchy.leaves_in_state("active")
        complex_leaves = state.voxel_hierarchy.leaves_in_state("complex")
        self.assertEqual(len(state.surface_patches), len(active) + len(complex_leaves))
        self.assertGreater(len(state.surface_patches), 1)
        self.assertEqual(len(state.stage1_patch_provenance), len(state.surface_patches))

        # Every patch traces back to exactly one leaf, and its Gaussians carry
        # that patch id.
        for provenance in state.stage1_patch_provenance:
            patch_id = provenance["patch_id"]
            assigned = state.model.cluster_ids == patch_id
            self.assertEqual(int(assigned.sum()), provenance["gaussian_count"])
            self.assertGreater(provenance["observations_per_control"], 0.0)

    def test_stage1_voxel_support_masks_are_polygon_based(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline

        points, colors = _annulus_points()
        pipeline = TorchOSNGSPipeline(self._stage1_config(), device="cpu")
        state = pipeline.initialize(points, colors)
        masked = [p for p in state.surface_patches if p.uv_support_mask is not None]
        self.assertEqual(len(masked), len(state.surface_patches))
        for provenance in state.stage1_patch_provenance:
            self.assertIsNotNone(provenance["support_polygon_world"])
            self.assertIsNotNone(provenance["support_polygon_uv"])
            self.assertFalse(provenance["support_mask_empty"])

    def test_stage1_support_mode_none_leaves_charts_untrimmed(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline

        points, colors = _annulus_points()
        pipeline = TorchOSNGSPipeline(
            self._stage1_config(stage1_support_mode="none"), device="cpu"
        )
        state = pipeline.initialize(points, colors)
        for patch in state.surface_patches:
            self.assertIsNone(patch.uv_support_mask)

    def test_stage1_payload_contains_hierarchy_and_provenance(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, nurbs_intermediate_payload

        points, colors = _annulus_points()
        pipeline = TorchOSNGSPipeline(self._stage1_config(), device="cpu")
        state = pipeline.initialize(points, colors)
        payload = nurbs_intermediate_payload(state)
        self.assertIn("voxel_hierarchy", payload)
        self.assertEqual(payload["metadata"]["constructor_mode"], "voxel_patch_stage1")
        self.assertEqual(
            payload["voxel_hierarchy"]["point_count"], int(points.shape[0])
        )
        for entry in payload["patches"]:
            self.assertIn("stage1", entry)
            self.assertIn("source_leaf_voxel_id", entry["stage1"])

    def test_legacy_mode_is_untouched_by_stage1_config_fields(self):
        from osn_gs.core.torch_pipeline import (
            TorchOSNGSPipeline,
            TorchPipelineConfig,
            nurbs_intermediate_payload,
        )

        points, colors = _annulus_points(count=400)
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(voxel_grid_resolution=6, base_curve_count=4),
            device="cpu",
        )
        state = pipeline.initialize(points, colors)
        self.assertIsNone(state.voxel_hierarchy)
        self.assertEqual(state.stage1_patch_provenance, [])
        payload = nurbs_intermediate_payload(state)
        self.assertNotIn("voxel_hierarchy", payload)
        self.assertNotIn("constructor_mode", payload["metadata"])
        self.assertNotIn("stage1", payload["patches"][0])

    def test_leaf_face_adjacency_classification(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline
        from osn_gs.surface.torch_voxel_hierarchy import compute_leaf_face_adjacency

        points, colors = _annulus_points()
        pipeline = TorchOSNGSPipeline(self._stage1_config(), device="cpu")
        state = pipeline.initialize(points, colors)
        adjacency = compute_leaf_face_adjacency(state.voxel_hierarchy)
        self.assertEqual(adjacency, state.stage1_leaf_adjacency)
        actives = {n.node_id for n in state.voxel_hierarchy.leaves_in_state("active")}
        boundary_count = 0
        for leaf_id, entry in adjacency.items():
            if leaf_id not in actives:
                continue
            self.assertTrue(entry["contacts"], f"active leaf {leaf_id} has no contacts")
            for contact in entry["contacts"]:
                if contact["neighbor_state"] in ("active",):
                    self.assertEqual(contact["classification"], "interior")
                if contact["neighbor_state"] in ("inactive", "empty", "outside"):
                    self.assertEqual(contact["classification"], "exterior_support")
            if entry["is_boundary_leaf"]:
                boundary_count += 1
                self.assertTrue(entry["boundary_faces"])
        # The annulus plane's active leaves all touch empty space above/below,
        # so every active leaf must be a boundary leaf here.
        self.assertEqual(boundary_count, len(actives))

    def test_marching_squares_circle_contour(self):
        from osn_gs.surface.torch_boundary_refinement import (
            contour_length_uv,
            marching_squares,
        )

        resolution = 64
        centers = (torch.arange(resolution) + 0.5) / resolution
        u, v = torch.meshgrid(centers, centers, indexing="ij")
        # Signed field whose 0.5-level set is a circle of radius 0.3.
        field = 0.5 + (0.3 - ((u - 0.5) ** 2 + (v - 0.5) ** 2).sqrt())
        segments = marching_squares(field, 0.5)
        self.assertTrue(segments)
        length = contour_length_uv(segments)
        expected = 2.0 * 3.14159265 * 0.3
        self.assertAlmostEqual(length, expected, delta=expected * 0.05)
        # Every contour point lies close to the true circle (sub-cell accuracy).
        for a, b in segments:
            for point in (a, b):
                radius = ((point[0] - 0.5) ** 2 + (point[1] - 0.5) ** 2) ** 0.5
                self.assertLess(abs(radius - 0.3), 1.5 / resolution)

    def test_density_refinement_subset_and_provenance(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline

        points, colors = _annulus_points()
        polygon_state = TorchOSNGSPipeline(
            self._stage1_config(stage1_support_mode="voxel"), device="cpu"
        ).initialize(points, colors)
        refined_state = TorchOSNGSPipeline(
            self._stage1_config(stage1_support_mode="voxel_density"), device="cpu"
        ).initialize(points, colors)
        self.assertEqual(
            len(polygon_state.surface_patches), len(refined_state.surface_patches)
        )
        strictly_smaller = 0
        for polygon_patch, refined_patch, provenance in zip(
            polygon_state.surface_patches,
            refined_state.surface_patches,
            refined_state.stage1_patch_provenance,
        ):
            refined_mask = refined_patch.uv_support_mask
            polygon_mask = polygon_patch.uv_support_mask
            # Refined support must stay inside the source voxel polygon.
            self.assertFalse(bool((refined_mask & ~polygon_mask).any()))
            self.assertTrue(bool(refined_mask.any()))
            if provenance["is_boundary_leaf"]:
                self.assertIsNotNone(provenance["density_refinement"])
                refinement = provenance["density_refinement"]
                self.assertGreater(refinement["bandwidth_uv"], 0.0)
                self.assertGreaterEqual(
                    refinement["coarse_cells"], refinement["refined_cells"]
                )
                if refinement["refined_cells"] < refinement["coarse_cells"]:
                    strictly_smaller += 1
            else:
                self.assertIsNone(provenance["density_refinement"])
        self.assertGreater(strictly_smaller, 0)
        # Coarse masks are preserved for coarse-vs-refined metrics.
        self.assertEqual(
            len(refined_state.stage1_coarse_masks), len(refined_state.surface_patches)
        )

    def test_uv_frame_matches_pca_parameterization(self):
        from osn_gs.surface.torch_nurbs import (
            pca_parameterize_points,
            uv_frame_from_axes,
        )

        points, _ = _annulus_points(count=300)
        centered = points - points.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        frame = uv_frame_from_axes(points, points.mean(dim=0), vh[0], vh[1])
        uv_frame = frame.apply(points)
        uv_pca = pca_parameterize_points(points)
        self.assertTrue(torch.allclose(uv_frame, uv_pca, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
