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


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ConvexHullUnitTest(unittest.TestCase):
    """White-box test on ``_convex_hull_2d`` with a known point set."""

    def test_square_with_interior_points_hull_is_the_four_corners(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import _convex_hull_2d, _polygon_area_2d

        points = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # corners
            [0.5, 0.5], [0.3, 0.4], [0.6, 0.2],  # interior, must be excluded
        ])
        hull = _convex_hull_2d(points)
        self.assertEqual(hull.shape[0], 4)
        self.assertAlmostEqual(_polygon_area_2d(hull), 1.0, places=5)

    def test_triangle_hull_area_matches_shoelace(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import _convex_hull_2d, _polygon_area_2d

        points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        hull = _convex_hull_2d(points)
        self.assertAlmostEqual(_polygon_area_2d(hull), 3.0, places=5)  # 0.5*base*height = 0.5*2*3


@unittest.skipUnless(torch is not None, "PyTorch is required")
class SutherlandHodgmanClipUnitTest(unittest.TestCase):
    """White-box test on the convex-convex polygon clip."""

    def test_clipping_a_square_by_a_smaller_centered_square(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import _polygon_area_2d, _sutherland_hodgman_clip

        subject = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])  # area 4
        clip = torch.tensor([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])  # area 1, inside subject
        result = _sutherland_hodgman_clip(subject, clip)
        self.assertAlmostEqual(_polygon_area_2d(result), 1.0, places=5)

    def test_clip_never_expands_beyond_the_smaller_polygon(self):
        # A clip larger than the subject must not expand the subject.
        from nurbs_constructor_benchmark.boundary_bias_analysis import _polygon_area_2d, _sutherland_hodgman_clip

        subject = torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])  # area 0.04
        clip = torch.tensor([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])  # huge
        result = _sutherland_hodgman_clip(subject, clip)
        self.assertAlmostEqual(_polygon_area_2d(result), 0.04, places=5)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class BoundaryLeafClippingIntegrationTest(unittest.TestCase):
    """Integration tests on the real Phase 1/Phase 2 pipeline (unmodified) +
    the Step B prototype clip, per the plan's core invariants."""

    def test_interior_leaf_polygon_never_clipped(self):
        # Core invariant: whatever the clipping does to boundary leaves, an
        # interior leaf's polygon must stay byte-identical to the
        # unmodified plane-AABB polygon -- "existing voxel hierarchy/
        # component logic unchanged" for the majority of leaves.
        from nurbs_constructor_benchmark.boundary_bias_analysis import (
            build_boundary_leaf_records, generate_boundary_bias_scene,
        )
        from osn_gs.surface.torch_component_boundary import extract_component_boundary
        from osn_gs.surface.torch_surface_components import build_surface_components
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        # A denser, larger scene is more likely to produce at least one
        # fully-interior leaf (surrounded on all sides by other active leaves).
        scene = generate_boundary_bias_scene("boundary_bias_circle", count=2000, seed=1)
        hierarchy = build_voxel_gaussian_hierarchy(scene.points, voxel_min_gaussian_count=10, voxel_max_gaussian_count=150, voxel_max_depth=6)
        component_set = build_surface_components(hierarchy, scene.points)
        component = component_set.components[0]
        boundary = extract_component_boundary(component, hierarchy, scene.points, resolution=64)
        records = build_boundary_leaf_records(component, hierarchy, scene.points, boundary.frame, scene.center)
        interior = [r for r in records if not r.is_boundary]
        if not interior:
            self.skipTest("no interior leaf in this scene/seed -- structural, not a failure of the invariant")
        for r in interior:
            self.assertTrue(torch.equal(r.plane_aabb_polygon_uv, r.clipped_polygon_uv))
            self.assertEqual(r.plane_aabb_area, r.clipped_area)

    def test_clipping_reduces_outward_bias_on_ellipse(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import analyze_scene_with_clipping, generate_boundary_bias_scene

        scene = generate_boundary_bias_scene("boundary_bias_ellipse", count=600, seed=0)
        result = analyze_scene_with_clipping(scene)
        before, after = result["before"], result["after"]
        self.assertLess(abs(after["signed_distance_mean"]), abs(before["signed_distance_mean"]))
        self.assertLess(after["false_fill_area"], before["false_fill_area"])
        # The clip is conservative (never expands), so some legitimate
        # coverage is expected to be lost -- document, don't hide, the trade-off.
        self.assertGreaterEqual(after["under_coverage"], before["under_coverage"])


class _FakeFrame:
    """Minimal stand-in for UVFrame: identity projection onto XY, no clamp."""

    def apply(self, points, clamp=False):
        return points[:, :2]


class _FakePlane:
    def __init__(self, centroid, normal):
        self.centroid = centroid
        self.normal = normal


class _FakeLeaf:
    def __init__(self, node_id, state, centroid, normal, gaussian_indices):
        self.node_id = node_id
        self.state = state
        self.plane = _FakePlane(centroid, normal)
        self.gaussian_indices = gaussian_indices


@unittest.skipUnless(torch is not None, "PyTorch is required")
class EligibilityClassifierUnitTest(unittest.TestCase):
    """Deterministic rule-correctness tests on ``compute_leaf_eligibility``
    with hand-built leaf/adjacency inputs -- no real hierarchy needed."""

    def setUp(self):
        from osn_gs.surface.torch_voxel_hierarchy import STATE_ACTIVE, STATE_COMPLEX, STATE_INACTIVE

        self.STATE_ACTIVE = STATE_ACTIVE
        self.STATE_COMPLEX = STATE_COMPLEX
        self.STATE_INACTIVE = STATE_INACTIVE
        self.frame = _FakeFrame()
        # A leaf's own points, all points, live within a unit-ish cell.
        self.polygon_uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def _good_neighbor_adjacency(self, leaf_id, n_active=3, n_total=4):
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility  # noqa: F401

        contacts = []
        node_by_id = {}
        for i in range(n_total):
            neighbor_id = f"n{i}"
            state = self.STATE_ACTIVE if i < n_active else self.STATE_INACTIVE
            contacts.append({"neighbor_id": neighbor_id, "classification": "interior", "neighbor_state": state})
            node_by_id[neighbor_id] = _FakeLeaf(neighbor_id, state, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), None)
        return {leaf_id: {"contacts": contacts}}, node_by_id

    def test_dense_planar_consistent_connected_is_active_observed(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility

        all_points = torch.rand((300, 3)) * torch.tensor([1.0, 1.0, 0.0])  # dense grid, exactly on z=0 plane
        leaf = _FakeLeaf("L", self.STATE_ACTIVE, torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.arange(300))
        adjacency, node_by_id = self._good_neighbor_adjacency("L", n_active=3, n_total=4)
        result = compute_leaf_eligibility(leaf, all_points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        self.assertEqual(result.final_class, "ACTIVE_OBSERVED")
        self.assertEqual(result.primary_spacing_class, "active_candidate")

    def test_sparse_but_otherwise_consistent_is_uncertain_not_inactive(self):
        # Sparse spacing alone must NOT be enough for INACTIVE (plan review
        # point 9: conservative default is UNCERTAIN unless corroborated).
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility

        few_points = torch.tensor([[0.1, 0.1, 0.0], [0.9, 0.9, 0.0], [0.1, 0.9, 0.0]])
        leaf = _FakeLeaf("L", self.STATE_ACTIVE, torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.arange(3))
        adjacency, node_by_id = self._good_neighbor_adjacency("L", n_active=3, n_total=4)
        result = compute_leaf_eligibility(leaf, few_points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        self.assertIn(result.primary_spacing_class, ("inactive_candidate", "uncertain_candidate"))
        self.assertEqual(result.final_class, "UNCERTAIN")
        self.assertNotEqual(result.final_class, "INACTIVE")

    def test_sparse_and_corroborated_is_inactive(self):
        # Sparse spacing + bad plane residual + no active neighbors together
        # -> INACTIVE (multiple corroborating signals, per point 9).
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility

        # Only 2 points, near-opposite corners of the cell (maximal NN
        # spacing relative to cell size) and off the nominal z=0 plane.
        few_points = torch.tensor([[0.05, 0.05, 0.4], [0.95, 0.95, -0.4]])
        leaf = _FakeLeaf("L", self.STATE_ACTIVE, torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.arange(2))
        # No active neighbors at all -> bad neighbor continuity too.
        adjacency, node_by_id = self._good_neighbor_adjacency("L", n_active=0, n_total=4)
        result = compute_leaf_eligibility(leaf, few_points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        self.assertEqual(result.primary_spacing_class, "inactive_candidate")
        self.assertEqual(result.final_class, "INACTIVE")

    def test_complex_state_short_circuits_to_complex_regardless_of_features(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility

        dense_points = torch.rand((300, 3)) * torch.tensor([1.0, 1.0, 0.0])
        leaf = _FakeLeaf("L", self.STATE_COMPLEX, torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.arange(300))
        adjacency, node_by_id = self._good_neighbor_adjacency("L", n_active=3, n_total=4)
        result = compute_leaf_eligibility(leaf, dense_points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        self.assertEqual(result.final_class, "COMPLEX")
        self.assertEqual(result.class_transition_reason, "phase1_complex_state")

    def test_normal_consistency_uses_absolute_dot_for_sign_ambiguity(self):
        # A neighbor plane normal pointing the OPPOSITE way (same physical
        # plane, flipped sign) must NOT be treated as inconsistent.
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility

        dense_points = torch.rand((300, 3)) * torch.tensor([1.0, 1.0, 0.0])
        leaf = _FakeLeaf("L", self.STATE_ACTIVE, torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.arange(300))
        contacts = [{"neighbor_id": "n0", "classification": "interior", "neighbor_state": self.STATE_ACTIVE}]
        node_by_id = {"n0": _FakeLeaf("n0", self.STATE_ACTIVE, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]), None)}
        adjacency = {"L": {"contacts": contacts}}
        result = compute_leaf_eligibility(leaf, dense_points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        self.assertAlmostEqual(result.normal_consistency, 1.0, places=5)
        self.assertEqual(result.normal_consistency_vote, "good")

    def test_deterministic_repeat_calls_give_identical_result(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import compute_leaf_eligibility

        points = torch.rand((300, 3), generator=torch.Generator().manual_seed(0)) * torch.tensor([1.0, 1.0, 0.0])
        leaf = _FakeLeaf("L", self.STATE_ACTIVE, torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.arange(300))
        adjacency, node_by_id = self._good_neighbor_adjacency("L", n_active=3, n_total=4)
        r1 = compute_leaf_eligibility(leaf, points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        r2 = compute_leaf_eligibility(leaf, points, self.frame, self.polygon_uv, adjacency, node_by_id, {"L"})
        self.assertEqual(r1.final_class, r2.final_class)
        self.assertEqual(r1.spacing_ratio, r2.spacing_ratio)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class EligibilityMaskStructuralInvariantTest(unittest.TestCase):
    """Structural contract tests on ``rasterize_eligibility_masks`` (plan
    review point 11): mask disjointness and cumulative-view correctness,
    using directly-constructed records -- not scene-level number assertions."""

    @staticmethod
    def _record(polygon, final_class):
        from nurbs_constructor_benchmark.boundary_bias_analysis import BoundaryLeafRecord, LeafEligibilityResult, LeafBoundaryProvenance

        eligibility = LeafEligibilityResult(
            leaf_id="x", spacing_ratio=0.0, rho_u=0.0, rho_v=0.0, plane_residual_world=0.0,
            plane_residual_normalized=0.0, normal_consistency=1.0, normal_neighbor_count=1,
            neighbor_phase1_active_ratio=1.0, primary_spacing_class="active_candidate",
            plane_residual_vote="good", normal_consistency_vote="good", neighbor_continuity_vote="good",
            final_class=final_class, class_transition_reason="test", provenance=LeafBoundaryProvenance(False, False, False),
        )
        record = BoundaryLeafRecord(
            leaf_id="x", is_boundary=True, point_count=10, center_angle=0.0,
            plane_aabb_polygon_uv=polygon, clipped_polygon_uv=polygon, plane_aabb_area=1.0, clipped_area=1.0, occupancy_ratio=1.0,
        )
        return record, eligibility

    def test_inactive_leaf_contributes_to_no_mask(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import rasterize_eligibility_masks

        polygon = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        masks = rasterize_eligibility_masks([self._record(polygon, "INACTIVE")], resolution=16)
        for key in ("active", "uncertain", "complex", "active_only", "active_plus_uncertain", "active_plus_uncertain_plus_complex"):
            self.assertFalse(bool(masks[key].any()), key)

    def test_complex_leaf_excluded_from_active_but_present_in_full_union(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import rasterize_eligibility_masks

        polygon = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        masks = rasterize_eligibility_masks([self._record(polygon, "COMPLEX")], resolution=16)
        self.assertFalse(bool(masks["active"].any()))
        self.assertFalse(bool(masks["active_only"].any()))
        self.assertFalse(bool(masks["active_plus_uncertain"].any()))
        self.assertTrue(bool(masks["complex"].any()))
        self.assertTrue(bool(masks["active_plus_uncertain_plus_complex"].any()))

    def test_active_and_uncertain_leaves_land_in_disjoint_own_masks(self):
        from nurbs_constructor_benchmark.boundary_bias_analysis import rasterize_eligibility_masks

        active_polygon = torch.tensor([[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]])
        uncertain_polygon = torch.tensor([[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]])
        records = [self._record(active_polygon, "ACTIVE_OBSERVED"), self._record(uncertain_polygon, "UNCERTAIN")]
        masks = rasterize_eligibility_masks(records, resolution=16)
        self.assertFalse(bool((masks["active"] & masks["uncertain"]).any()))
        self.assertTrue(bool(masks["active"].any()))
        self.assertTrue(bool(masks["uncertain"].any()))
        # active_only must NOT include the uncertain leaf's contribution.
        self.assertTrue(bool((masks["active_only"] == masks["active"]).all()))
        # active_plus_uncertain must be a strict superset.
        self.assertTrue(bool(((masks["active_plus_uncertain"] & ~masks["active_only"]) == masks["uncertain"]).all()))


if __name__ == "__main__":
    unittest.main()
