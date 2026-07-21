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
        kwargs = dict(
            segments=8,
            outer_boundary_world_points=boundary.outer_loops[0].boundary_world_points if boundary.outer_loops else None,
        )
        kwargs.update(overrides)
        return build_annulus_chart(
            component, points, boundary.frame, boundary.refined_mask,
            boundary.hole_loops[0].boundary_world_points, **kwargs,
        )

    def test_annulus_classified_and_no_jacobian_fold(self):
        result = self._build(_annulus())
        self.assertEqual(len(result.slices), 8)
        self.assertEqual(result.topology_checks["near_degenerate_slice_count"], 0)
        self.assertEqual(result.topology_checks["total_orientation_flip_samples"], 0)
        for s in result.slices:
            self.assertGreater(s.fit_metrics["min_area_jacobian"], 0.0)
            self.assertGreater(s.fit_metrics["min_jacobian_singular_value"], 0.0)

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

    def test_new_diagnostic_fields_present_and_sane_on_a_healthy_build(self):
        # Step 1's new fields, on a build with no known pathology: presence
        # and basic sanity, not tight numeric bounds (those come from the
        # multi-scene baseline in Step 3).
        result = self._build(_annulus())
        for s in result.slices:
            fm = s.fit_metrics
            for key in (
                "min_area_jacobian", "min_jacobian_singular_value", "jacobian_condition_mean",
                "jacobian_condition_p95", "max_jacobian_condition", "orientation_flip_count",
                "near_degenerate_count",
            ):
                self.assertIn(key, fm)
            self.assertGreaterEqual(fm["max_jacobian_condition"], fm["jacobian_condition_mean"])
            pq = fm["parameter_quality"]
            for key in ("cv_v_along_u_line_mean", "cv_u_along_v_line_mean", "anisotropy_mean", "orthogonality_mean"):
                self.assertIn(key, pq)
            self.assertGreaterEqual(pq["anisotropy_mean"], 0.0)
            self.assertLessEqual(pq["anisotropy_mean"], 1.0)
        for seam in result.seams:
            for value in (
                seam.seam_tangent_angle_deg_mean, seam.seam_cross_derivative_angle_deg_mean,
                seam.seam_normal_angle_deg_mean,
            ):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 180.0)
        cq = result.chart_quality
        self.assertIn("jacobian", cq)
        self.assertIn("seams", cq)
        self.assertIn("parameter_quality", cq)
        self.assertIn("phase2_boundary_conformance", cq)
        self.assertIsNotNone(cq["phase2_boundary_conformance"]["inner"])
        self.assertIsNotNone(cq["phase2_boundary_conformance"]["outer"])

    def test_known_bad_seed_reproduces_inner_corner_degeneracy(self):
        # Regression/detection guard: this exact scene (test _annulus fixture,
        # seed=14) reproduces the O-grid inner-pole degeneracy documented in
        # OSN_GS_Phase4_Hardening_Plan.md -- 8 samples with an orientation-
        # flipped in-plane Jacobian, confined to the (u~=0, v~=0) corner
        # nearest the hole/seam. This proves the new metrics actually detect
        # a REAL failure mode, not just synthetic constructions. When a
        # future Step 4 fix (e.g. arc-length reparameterization) addresses
        # this, this test's expected count should be revisited -- it is a
        # detection guard, not a claim that 8 is an acceptable steady state.
        result = self._build(_annulus(seed=14))
        self.assertEqual(result.topology_checks["total_orientation_flip_samples"], 8)
        flipped_slices = [s for s in result.slices if s.fit_metrics["orientation_flip_count"] > 0]
        self.assertTrue(flipped_slices)
        for s in flipped_slices:
            # Confined to samples very near the inner boundary -- a healthy
            # slice's own min singular value should be far from this one's.
            self.assertLess(s.fit_metrics["min_jacobian_singular_value"], 0.05)

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


def _flat_grid_surface(nu=4, nv=4, degree_u=1, degree_v=1, v_scale=1.0):
    """A regular flat (z=0) bilinear control grid spanning x,y in [0,1]x[0,v_scale]."""

    from osn_gs.surface.torch_nurbs import TorchNURBSSurface

    cg = torch.zeros((nu, nv, 3))
    for i in range(nu):
        for j in range(nv):
            cg[i, j] = torch.tensor([i / (nu - 1), (j / (nv - 1)) * v_scale, 0.0])
    return TorchNURBSSurface(control_grid=cg, weights=torch.ones((nu, nv)), degree_u=degree_u, degree_v=degree_v)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class JacobianDiagnosticsUnitTest(unittest.TestCase):
    """White-box tests on ``_jacobian_diagnostics`` with hand-built control
    grids, per the Phase 4 hardening plan's requirement that new metrics be
    validated against deliberately injected failures, not just real fits."""

    def test_healthy_flat_grid(self):
        from osn_gs.surface.torch_annulus_chart import _jacobian_diagnostics

        d = _jacobian_diagnostics(_flat_grid_surface())
        self.assertAlmostEqual(d["min_area_jacobian"], 1.0, places=4)
        self.assertAlmostEqual(d["min_jacobian_singular_value"], 1.0, places=4)
        self.assertAlmostEqual(d["max_jacobian_condition"], 1.0, places=4)
        self.assertEqual(d["orientation_flip_count"], 0)
        self.assertEqual(d["near_degenerate_count"], 0)

    def test_collapsed_radial_extent_is_near_degenerate(self):
        # Injected failure: v (radial) extent collapsed to ~0 -- the
        # "degenerate radial strip" case from the plan review.
        from osn_gs.surface.torch_annulus_chart import _jacobian_diagnostics

        d = _jacobian_diagnostics(_flat_grid_surface(v_scale=1e-6))
        self.assertLess(d["min_area_jacobian"], 1e-4)
        self.assertGreater(d["near_degenerate_count"], 0)

    def test_twisted_grid_flags_orientation_flip_without_degeneracy(self):
        # Injected failure: a "bowtie" control grid where one row's v
        # direction is reversed relative to its neighbor, producing a real
        # local self-intersection (orientation reversal) that is NOT a
        # degeneracy (area stays healthy) -- proves the two conditions are
        # correctly distinguished, per the plan review's point that
        # ``jacobian_min <= 0`` alone conflates them.
        from osn_gs.surface.torch_nurbs import TorchNURBSSurface
        from osn_gs.surface.torch_annulus_chart import _jacobian_diagnostics

        cg = torch.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                cg[i, j] = torch.tensor([float(i), float(j), 0.0])
        cg[2] = torch.tensor([[2.0, 2.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        surf = TorchNURBSSurface(control_grid=cg, weights=torch.ones((3, 3)), degree_u=1, degree_v=1)
        d = _jacobian_diagnostics(surf, resolution=20)
        self.assertGreater(d["orientation_flip_count"], 0)
        self.assertEqual(d["near_degenerate_count"], 0)
        self.assertGreater(d["min_area_jacobian"], 1e-3)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class SeamMetricsUnitTest(unittest.TestCase):
    """White-box tests on ``_measure_seams`` with hand-built adjacent slices."""

    @staticmethod
    def _slice(idx, u_origin, v_reverse=False, mirror_y=False, translate=(0.0, 0.0, 0.0), nu=4, nv=4):
        from osn_gs.surface.torch_nurbs import TorchNURBSSurface
        from osn_gs.surface.torch_annulus_chart import AnnulusChartSlice

        cg = torch.zeros((nu, nv, 3))
        for i in range(nu):
            for j in range(nv):
                vv = j / (nv - 1)
                if v_reverse:
                    vv = 1.0 - vv
                y = -vv if mirror_y else vv
                cg[i, j] = torch.tensor([u_origin + i / (nu - 1), y, 0.0])
        cg = cg + torch.tensor(translate)
        surf = TorchNURBSSurface(control_grid=cg, weights=torch.ones((nu, nv)), degree_u=1, degree_v=1)
        return AnnulusChartSlice(
            slice_index=idx, angle_range=(0.0, 1.0), inner_radius=0.0, outer_radius=1.0,
            gaussian_indices=None, surface=surf, uv=None, diagnostics=None, fit_metrics={},
        )

    def test_perfect_match_has_zero_gap_and_zero_angles(self):
        from osn_gs.surface.torch_annulus_chart import _measure_seams

        a, b = self._slice(0, u_origin=0.0), self._slice(1, u_origin=1.0)
        seam = _measure_seams([a, b], 9)[0]
        self.assertLess(seam.mean_gap, 1e-5)
        self.assertLess(seam.seam_tangent_angle_deg_mean, 1e-3)
        self.assertLess(seam.seam_cross_derivative_angle_deg_mean, 1e-3)
        self.assertLess(seam.seam_normal_angle_deg_mean, 1e-3)
        self.assertAlmostEqual(seam.seam_derivative_ratio_mean, 1.0, places=4)

    def test_seam_translation_only_moves_position_gap(self):
        # Injected failure: a pure rigid offset changes position gap but NOT
        # tangent/normal direction -- proves the metrics are independent,
        # not accidentally coupled.
        from osn_gs.surface.torch_annulus_chart import _measure_seams

        a = self._slice(0, u_origin=0.0)
        b = self._slice(1, u_origin=1.0, translate=(0.0, 0.0, 0.5))
        seam = _measure_seams([a, b], 9)[0]
        self.assertAlmostEqual(seam.mean_gap, 0.5, places=4)
        self.assertLess(seam.seam_tangent_angle_deg_mean, 1e-3)
        self.assertLess(seam.seam_cross_derivative_angle_deg_mean, 1e-3)

    def test_tangent_reversal_is_detected(self):
        # Injected failure: B's radial (v) direction reversed relative to A
        # -- the along-seam tangent must show ~180 degrees mismatch.
        from osn_gs.surface.torch_annulus_chart import _measure_seams

        a = self._slice(0, u_origin=0.0)
        b = self._slice(1, u_origin=1.0, v_reverse=True)
        seam = _measure_seams([a, b], 9)[0]
        self.assertAlmostEqual(seam.seam_tangent_angle_deg_mean, 180.0, places=1)

    def test_mirrored_slice_flips_normal_and_tangent(self):
        # Injected failure: B mirrored across its own u-axis. For a flat
        # (z=0) surface this necessarily reverses both the tangent AND the
        # normal together (normal = Su x Sv is fully determined by both
        # tangent directions) -- documenting that these are coupled for a
        # planar surface, not independently constructible failure modes.
        from osn_gs.surface.torch_annulus_chart import _measure_seams

        a = self._slice(0, u_origin=0.0)
        b = self._slice(1, u_origin=1.0, mirror_y=True)
        seam = _measure_seams([a, b], 9)[0]
        self.assertAlmostEqual(seam.seam_normal_angle_deg_mean, 180.0, places=1)
        self.assertAlmostEqual(seam.seam_tangent_angle_deg_mean, 180.0, places=1)

    def test_periodic_closure_wraps_last_to_first(self):
        # The seam loop already wraps (k+1) % n (module docstring claim) --
        # verify the wrap-around seam is actually measured, not skipped.
        from osn_gs.surface.torch_annulus_chart import _measure_seams

        a, b, c = self._slice(0, u_origin=0.0), self._slice(1, u_origin=1.0), self._slice(2, u_origin=2.0)
        seams = _measure_seams([a, b, c], 9)
        self.assertEqual(len(seams), 3)
        self.assertEqual((seams[-1].slice_a, seams[-1].slice_b), (2, 0))


@unittest.skipUnless(torch is not None, "PyTorch is required")
class BoundaryConformanceUnitTest(unittest.TestCase):
    """White-box tests on ``_boundary_conformance``."""

    def test_perfect_match(self):
        from osn_gs.surface.torch_annulus_chart import _boundary_conformance

        ref = torch.stack([torch.linspace(0.0, 1.0, 50), torch.zeros(50), torch.zeros(50)], dim=1)
        d = _boundary_conformance(ref.clone(), ref, coverage_tolerance=0.05)
        self.assertLess(d["symmetric_chamfer"], 1e-6)
        self.assertEqual(d["boundary_coverage_ratio"], 1.0)

    def test_uniform_offset_reduces_coverage(self):
        # Injected failure: boundary offset -- both directions should grow
        # together for a uniform shift.
        from osn_gs.surface.torch_annulus_chart import _boundary_conformance

        ref = torch.stack([torch.linspace(0.0, 1.0, 50), torch.zeros(50), torch.zeros(50)], dim=1)
        edge = ref.clone()
        edge[:, 1] += 0.2
        d = _boundary_conformance(edge, ref, coverage_tolerance=0.05)
        self.assertAlmostEqual(d["edge_to_reference_mean"], 0.2, places=4)
        self.assertAlmostEqual(d["reference_to_edge_mean"], 0.2, places=4)
        self.assertEqual(d["boundary_coverage_ratio"], 0.0)

    def test_collapsed_edge_is_caught_by_symmetric_metric_not_one_directional(self):
        # This is the exact failure the plan review warned about: a chart
        # edge collapsed onto a single point of the true boundary looks
        # PERFECT under a one-directional (chart->reference) distance alone
        # (every collapsed sample sits exactly on a reference point), but a
        # true collapse -- most of the real boundary now has nothing nearby.
        from osn_gs.surface.torch_annulus_chart import _boundary_conformance

        ref = torch.stack([torch.linspace(0.0, 1.0, 50), torch.zeros(50), torch.zeros(50)], dim=1)
        collapsed_edge = ref[24:25].repeat(50, 1)  # every chart sample is the SAME single reference point
        d = _boundary_conformance(collapsed_edge, ref, coverage_tolerance=0.05)
        self.assertLess(d["edge_to_reference_mean"], 1e-6)  # one-directional metric alone: looks perfect
        self.assertGreater(d["reference_to_edge_mean"], 0.1)  # symmetric direction: reveals the collapse
        self.assertLess(d["boundary_coverage_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
