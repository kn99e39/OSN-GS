from __future__ import annotations

"""Phase 2 outer-boundary bias analysis unit tests
(OSN_GS_Phase4_Hardening_Plan.md cross-reference: this diagnostic module is
upstream of, but distinct from, the Phase 4 annulus-chart hardening).

Per the project's standing "validate against injected failures, not just
real fits" discipline: every metric is checked against a hand-constructed
case with a KNOWN analytic bias, not just run on real data and eyeballed.
"""

import math
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class BiasMetricsUnitTest(unittest.TestCase):
    """White-box tests on ``compute_bias_metrics`` with a synthetic,
    exactly-known ``r_stage`` (not derived from the real pipeline)."""

    @staticmethod
    def _scene(a=0.75, b=0.75, center=(0.0, 0.0)):
        from nurbs_constructor_benchmark.boundary_bias_analysis import BoundaryBiasScene

        return BoundaryBiasScene(
            name="test", points=torch.zeros((1, 3)), center=torch.tensor(center), a=a, b=b, description="test",
        )

    def test_perfect_match_has_zero_bias(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_bias_metrics
        from nurbs_constructor_benchmark.support_domains import ellipse_radius_at_angle

        scene = self._scene()
        theta = torch.linspace(0.0, 2 * math.pi * 143 / 144, 144)
        r_stage = ellipse_radius_at_angle(theta, scene.a, scene.b)
        m = compute_bias_metrics(theta, r_stage, scene)
        self.assertAlmostEqual(m["signed_distance_mean"], 0.0, places=5)
        self.assertAlmostEqual(m["symmetric_chamfer"], 0.0, places=4)
        self.assertAlmostEqual(m["area_error_relative"], 0.0, places=4)
        self.assertAlmostEqual(m["coverage"], 1.0, places=4)
        for bias in m["sector_bias"]:
            self.assertAlmostEqual(bias, 0.0, places=4)

    def test_known_constant_outward_offset_is_reported_exactly(self):
        # Injected failure: r_stage = r_gt + 0.05 everywhere -- the exact
        # known bias must come back out of the metric, not an approximation.
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_bias_metrics
        from nurbs_constructor_benchmark.support_domains import ellipse_radius_at_angle

        scene = self._scene()
        theta = torch.linspace(0.0, 2 * math.pi * 143 / 144, 144)
        offset = 0.05
        r_stage = ellipse_radius_at_angle(theta, scene.a, scene.b) + offset
        m = compute_bias_metrics(theta, r_stage, scene)
        self.assertAlmostEqual(m["signed_distance_mean"], offset, places=3)
        self.assertGreater(m["area_error"], 0.0)  # outward offset -> larger enclosed area
        self.assertGreater(m["false_fill_area"], 0.0)
        for bias in m["sector_bias"]:
            self.assertAlmostEqual(bias, offset, places=3)

    def test_known_constant_inward_offset_is_reported_exactly(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_bias_metrics
        from nurbs_constructor_benchmark.support_domains import ellipse_radius_at_angle

        scene = self._scene()
        theta = torch.linspace(0.0, 2 * math.pi * 143 / 144, 144)
        offset = -0.05
        r_stage = ellipse_radius_at_angle(theta, scene.a, scene.b) + offset
        m = compute_bias_metrics(theta, r_stage, scene)
        self.assertAlmostEqual(m["signed_distance_mean"], offset, places=3)
        self.assertLess(m["area_error"], 0.0)
        self.assertAlmostEqual(m["false_fill_area"], 0.0, places=4)  # never exceeds GT -> no false fill

    def test_one_sided_bulge_is_isolated_to_its_own_sector(self):
        # Injected failure: bias only in a narrow angular window around
        # theta=0 -- must show up in that sector's bias and NOT the others.
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_bias_metrics
        from nurbs_constructor_benchmark.support_domains import ellipse_radius_at_angle

        scene = self._scene()
        theta = torch.linspace(0.0, 2 * math.pi * 143 / 144, 144)
        bulge = torch.where(theta.abs() < 0.2, torch.full_like(theta, 0.1), torch.zeros_like(theta))
        r_stage = ellipse_radius_at_angle(theta, scene.a, scene.b) + bulge
        m = compute_bias_metrics(theta, r_stage, scene, sectors=8)
        self.assertGreater(m["sector_bias"][0], 0.01)  # sector containing theta=0
        for bias in m["sector_bias"][2:6]:  # sectors far from theta=0
            self.assertAlmostEqual(bias, 0.0, places=3)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class PipelineExtractionSmokeTest(unittest.TestCase):
    """Integration smoke test: the real Stage 1/Phase 1/Phase 2 pipeline
    produces finite, sane radius profiles for a circle scene at every stage."""

    def test_circle_scene_all_stages_finite_and_near_true_radius(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import analyze_scene, generate_boundary_bias_scene

        scene = generate_boundary_bias_scene("boundary_bias_circle", count=500, seed=0)
        self.assertAlmostEqual(scene.a, scene.b, places=6)  # sanity: circle has equal semi-axes
        result = analyze_scene(scene)
        self.assertEqual(len(result), 7)
        for stage, metrics in result.items():
            self.assertTrue(math.isfinite(metrics["signed_distance_mean"]), stage)
            # No stage should be off by more than ~30% of the true radius --
            # a bug (e.g. computing angle/radius in normalized UV space
            # instead of world space) previously produced ~-30% errors here.
            self.assertLess(abs(metrics["signed_distance_mean"]), 0.3 * scene.a, stage)


if __name__ == "__main__":
    unittest.main()
