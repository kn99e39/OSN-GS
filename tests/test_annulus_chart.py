from __future__ import annotations

"""Phase 4 Boundary-Conforming Chart Generator unit tests
(OSN_GS_Final_Boundary_First_NURBS_Direction.md §Phase 4)."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _annulus(count: int = 900, inner: float = 0.32, outer: float = 0.9, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    accepted = []
    remaining = count
    while remaining > 0:
        xy = torch.rand((count * 2, 2), generator=generator) * 2.0 - 1.0
        r = xy.square().sum(dim=1).sqrt()
        xy = xy[(r >= inner) & (r <= outer)][:remaining]
        accepted.append(xy)
        remaining -= int(xy.shape[0])
    xy = torch.cat(accepted, dim=0)
    return torch.cat([xy, torch.zeros((xy.shape[0], 1))], dim=1)


def _flat_plane(count: int = 800, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand((count, 2), generator=generator) * 2.0 - 1.0
    return torch.cat([xy, torch.zeros((count, 1))], dim=1)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ChartTopologyClassifierTest(unittest.TestCase):
    def test_disk_like(self):
        from osn_gs.surface.torch_chart_topology import classify_component_topology

        self.assertEqual(
            classify_component_topology({"outer_loop_count": 1, "hole_count": 0}), "disk_like"
        )

    def test_annulus(self):
        from osn_gs.surface.torch_chart_topology import classify_component_topology

        self.assertEqual(
            classify_component_topology({"outer_loop_count": 1, "hole_count": 1}), "annulus"
        )

    def test_multi_hole(self):
        from osn_gs.surface.torch_chart_topology import classify_component_topology

        self.assertEqual(
            classify_component_topology({"outer_loop_count": 1, "hole_count": 2}), "multi_hole"
        )

    def test_complex_and_non_chartable(self):
        from osn_gs.surface.torch_chart_topology import classify_component_topology

        self.assertEqual(
            classify_component_topology({"outer_loop_count": 2, "hole_count": 0}), "complex"
        )
        self.assertEqual(
            classify_component_topology({"outer_loop_count": 0, "hole_count": 0}), "non_chartable"
        )

    def test_tiny_hole_relative_to_outer_area_falls_back_to_disk_like(self):
        # Regression guard for the density_gradient finding: a "hole" that is
        # a negligible fraction of the outer loop's own area (a sparse-
        # sampling density artifact, not a real hole) must NOT route to the
        # O-grid annulus chart.
        from osn_gs.surface.torch_chart_topology import classify_component_topology

        artifact_topology = {
            "outer_loop_count": 1, "hole_count": 1,
            "hole_loop_areas_cells": [17], "outer_loop_area_cells": 2468,
        }
        self.assertEqual(classify_component_topology(artifact_topology), "disk_like")

        real_hole_topology = {
            "outer_loop_count": 1, "hole_count": 1,
            "hole_loop_areas_cells": [262], "outer_loop_area_cells": 3202,
        }
        self.assertEqual(classify_component_topology(real_hole_topology), "annulus")


@unittest.skipUnless(torch is not None, "PyTorch is required")
class AnnulusOGridChartTest(unittest.TestCase):
    def _build(self, points, **overrides):
        from osn_gs.surface.torch_annulus_chart import build_annulus_chart
        from osn_gs.surface.torch_chart_topology import classify_boundary_result
        from osn_gs.surface.torch_component_boundary import extract_component_boundary
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        hierarchy = build_voxel_gaussian_hierarchy(
            points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6
        )
        component_set = build_surface_components(hierarchy, points)
        self.assertEqual(component_set.component_count(), 1)
        component = component_set.components[0]
        boundary = extract_component_boundary(component, hierarchy, points, resolution=64, density_threshold=3.0)
        self.assertEqual(classify_boundary_result(boundary), "annulus")
        kwargs = dict(segments=8)
        kwargs.update(overrides)
        return build_annulus_chart(
            component, points, boundary.frame, boundary.refined_mask,
            boundary.hole_loops[0].boundary_world_points, **kwargs,
        )

    def test_annulus_classified_and_no_jacobian_fold(self):
        result = self._build(_annulus())
        self.assertEqual(len(result.slices), 8)
        self.assertEqual(result.topology_checks["jacobian_fold_count"], 0)
        for s in result.slices:
            self.assertGreater(s.fit_metrics["jacobian_min"], 0.0)

    def test_uv_domains_do_not_overlap(self):
        result = self._build(_annulus())
        # Angle ranges partition [0, 2pi) exactly by construction.
        ranges = sorted(s.angle_range for s in result.slices)
        for (_, hi), (lo2, _) in zip(ranges, ranges[1:]):
            self.assertAlmostEqual(hi, lo2, places=5)
        self.assertFalse(result.topology_checks["uv_overlap"])

    def test_seam_gaps_are_small_relative_to_domain_scale(self):
        result = self._build(_annulus())
        # Domain spans roughly [-1, 1]; seam gaps should be a small fraction
        # of that, not comparable to the hole/outer radius scale.
        for seam in result.seams:
            self.assertLess(seam.mean_gap, 0.05)

    def test_zero_overlap_beats_positive_overlap_on_seam_and_fit(self):
        # Regression guard for the overlap-clamping-pileup finding: overlap=0
        # must not be worse than a positive overlap on this scene.
        zero = self._build(_annulus(), angular_overlap_fraction=0.0)
        widened = self._build(_annulus(), angular_overlap_fraction=0.25)
        zero_gap = sum(s.mean_gap for s in zero.seams) / len(zero.seams)
        widened_gap = sum(s.mean_gap for s in widened.seams) / len(widened.seams)
        self.assertLess(zero_gap, widened_gap)

    def test_each_slice_has_enough_points(self):
        result = self._build(_annulus())
        for s in result.slices:
            self.assertGreaterEqual(s.fit_metrics["point_count"], 4)

    def test_payload_serializes(self):
        import json

        from osn_gs.surface.torch_annulus_chart import annulus_chart_payload

        result = self._build(_annulus())
        payload = annulus_chart_payload(result)
        json.dumps(payload)
        self.assertEqual(len(payload["slices"]), 8)
        iso = payload["iso_lines"]
        self.assertEqual(iso["coordinate_semantics"]["v"], "radial coordinate from the inner boundary (0) to the outer boundary (1)")
        self.assertEqual(len(iso["slices"]), 8)
        self.assertEqual(len(iso["slices"][0]["u_lines"]), 5)
        self.assertEqual(len(iso["slices"][0]["v_lines"]), 5)
        self.assertEqual(len(iso["slices"][0]["u_lines"][0]["points"]), 17)

    def test_rejects_too_few_segments(self):
        from osn_gs.surface.torch_annulus_chart import build_annulus_chart
        from osn_gs.surface.torch_component_boundary import extract_component_boundary
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        points = _annulus()
        hierarchy = build_voxel_gaussian_hierarchy(
            points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6
        )
        component_set = build_surface_components(hierarchy, points)
        component = component_set.components[0]
        boundary = extract_component_boundary(component, hierarchy, points, resolution=64, density_threshold=3.0)
        with self.assertRaises(ValueError):
            build_annulus_chart(
                component, points, boundary.frame, boundary.refined_mask,
                boundary.hole_loops[0].boundary_world_points, segments=2,
            )


if __name__ == "__main__":
    unittest.main()
