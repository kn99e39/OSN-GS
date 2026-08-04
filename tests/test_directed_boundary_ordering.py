"""Worklog 35: deterministic one-in/one-out directed boundary ordering (C11).

Worklog 33/34 established that a well-formed post-ADC region (box_face, 19
genuine candidates, same-region partner median 18) fragmented into 6 open
chains under the previous mutual-agreement + greedy-augmentation heuristic --
NOT a candidate-scarcity problem. Traced (worklog 35) to 13/19 nodes having
more than one directionally-compatible successor: greedy per-node score
maximization is only locally optimal, so mutual agreement held for only
11/19 pairs and the remainder fragmented under greedy augmentation.

This file tests the replacement: an exact maximum-weight one-in/one-out
matching (Hungarian algorithm) over the SAME compatibility edges, restricted
per region, which structurally guarantees in-degree<=1/out-degree<=1 and
decomposes into disjoint simple cycles/paths with no heuristic patch-up.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _ring_candidates(n: int, radius: float = 1.0, region_id: int = 0, id_offset: int = 0):
    candidates = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        tangent = (-math.sin(angle), math.cos(angle), 0.0)
        candidates.append(WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"h{i + id_offset}", source_region_id=region_id, source_gaussian_id=i + id_offset,
            adjacent_gaussian_id=None, world_position=(x, y, 0.0), local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=tangent, boundary_direction=tangent,
            boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
            ordering_state="locally_chainable", review_reasons=(),
        ))
    return candidates


def _ring_accepted_pairs(n: int, id_offset: int = 0):
    return [(i + id_offset, (i + 1) % n + id_offset) for i in range(n)]


class SimpleCycleRecoveryTest(unittest.TestCase):
    def test_closed_ring_recovered_as_single_simple_cycle(self):
        n = 12
        candidates = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)
        _, components = recover_directed_boundary_components(candidates, accepted)
        closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0].ordered_source_ids), n)
        self.assertEqual(set(closed[0].ordered_source_ids), set(range(n)))

    def test_box_face_downsampled_fragmentation_is_recovered(self):
        """Exact reproduction of worklog 33/34's box_face fragmentation
        (27-member region, 19 genuine candidates, previously 6 open chains of
        sizes [1,2,3,5,7,1]) -- must now recover a large closed loop instead."""
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=27)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        summary = bundle.construction.diagnostic_summary
        self.assertGreaterEqual(summary["boundary_component_closed_count"], 1)
        self.assertGreaterEqual(summary["materialized_surface_count"], 1)

    def test_multiple_disjoint_loops_kept_separate(self):
        n = 8
        ring_a = _ring_candidates(n, region_id=0, id_offset=0)
        ring_b_base = _ring_candidates(n, region_id=0, id_offset=100)
        ring_b = [replace(c, world_position=(c.world_position[0] + 10.0, c.world_position[1], 0.0)) for c in ring_b_base]
        candidates = ring_a + ring_b
        accepted = _ring_accepted_pairs(n, id_offset=0) + _ring_accepted_pairs(n, id_offset=100)
        _, components = recover_directed_boundary_components(candidates, accepted)
        closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
        self.assertEqual(len(closed), 2)
        self.assertTrue(all(len(c.ordered_source_ids) == n for c in closed))


class OpenAndBrokenTopologyRejectionTest(unittest.TestCase):
    def test_target_tangent_sign_flip_does_not_reject_supported_corner_successor(self):
        source = WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id="h0", source_region_id=0, source_gaussian_id=0,
            adjacent_gaussian_id=None, world_position=(0.0, 0.0, 0.0),
            local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=(1.0, 0.0, 0.0), boundary_direction=(1.0, 0.0, 0.0),
            boundary_reason="observed_support_termination", source_pair_ids=None,
            confidence=0.7, ordering_state="locally_chainable", review_reasons=(),
        )
        target = WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id="h1", source_region_id=0, source_gaussian_id=1,
            adjacent_gaussian_id=None, world_position=(1.0, 0.0, 0.0),
            local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=(-1.0, 0.0, 0.0), boundary_direction=(-1.0, 0.0, 0.0),
            boundary_reason="observed_support_termination", source_pair_ids=None,
            confidence=0.7, ordering_state="locally_chainable", review_reasons=(),
        )
        from osn_gs.surface.torch_directed_boundary_ordering import _compatible_directed_edges

        edges = _compatible_directed_edges(
            (source, target), {frozenset((0, 1))}, 1.0, {0: {1}, 1: {0}}, frozenset((0, 1))
        )
        self.assertIn(("h0", "h1"), edges)
        self.assertEqual(edges[("h0", "h1")].tangent_alignment, 1.0)


    def test_open_chain_never_force_closed(self):
        n = 12
        candidates = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)[:-1]
        _, components = recover_directed_boundary_components(candidates, accepted)
        closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
        self.assertEqual(len(closed), 0)

    def test_two_missing_nodes_sparse_gap_not_bridged(self):
        n = 12
        candidates = [c for c in _ring_candidates(n) if c.source_gaussian_id not in (3, 4)]
        accepted = [pair for pair in _ring_accepted_pairs(n) if 3 not in pair and 4 not in pair]
        _, components = recover_directed_boundary_components(candidates, accepted)
        closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
        self.assertEqual(len(closed), 0)

    def test_y_junction_branch_never_admitted_into_a_cycle(self):
        n = 12
        candidates = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)
        stub_angle = 2 * math.pi * 0.5 / n
        stub = WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id="stub", source_region_id=0, source_gaussian_id=999, adjacent_gaussian_id=None,
            world_position=(math.cos(stub_angle) * 0.6, math.sin(stub_angle) * 0.6, 0.0),
            local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=(-math.sin(stub_angle), math.cos(stub_angle), 0.0),
            boundary_direction=(-math.sin(stub_angle), math.cos(stub_angle), 0.0),
            boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
            ordering_state="locally_chainable", review_reasons=(),
        )
        _, components = recover_directed_boundary_components(candidates + [stub], accepted + [(0, 999)])
        for component in components:
            if component.ordering_state == "ordered_closed_loop":
                self.assertNotIn(999, component.ordered_source_ids)

    def test_duplicate_and_reverse_duplicate_accepted_pairs_do_not_corrupt_matching(self):
        n = 10
        candidates = _ring_candidates(n)
        base_pairs = _ring_accepted_pairs(n)
        accepted = base_pairs + [(b, a) for a, b in base_pairs] + base_pairs
        _, components = recover_directed_boundary_components(candidates, accepted)
        closed = [c for c in components if c.ordering_state == "ordered_closed_loop"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0].ordered_source_ids), n)


class RigidTransformInvarianceTest(unittest.TestCase):
    """Frozen synthetic candidate set (worklog 33's "Test A" pattern applied
    to boundary candidates instead of representatives): rotation/translation/
    uniform-scale must reproduce EXACTLY the same closed-loop stable-ID sets."""

    @staticmethod
    def _rotate(vector, axis, angle):
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / norm, ay / norm, az / norm
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y, z = vector
        dot = x * ax + y * ay + z * az
        cross = (ay * z - az * y, az * x - ax * z, ax * y - ay * x)
        return (
            x * cos_a + cross[0] * sin_a + ax * dot * (1 - cos_a),
            y * cos_a + cross[1] * sin_a + ay * dot * (1 - cos_a),
            z * cos_a + cross[2] * sin_a + az * dot * (1 - cos_a),
        )

    def _transform(self, candidates, *, rotate_angle=0.0, axis=(0.3, 0.7, 0.5), translate=(0.0, 0.0, 0.0), scale=1.0):
        output = []
        for c in candidates:
            pos = tuple(v * scale for v in self._rotate(c.world_position, axis, rotate_angle))
            pos = tuple(p + t for p, t in zip(pos, translate))
            normal = self._rotate(c.local_normal, axis, rotate_angle)
            tangent = self._rotate(c.local_tangent_direction, axis, rotate_angle)
            boundary_dir = self._rotate(c.boundary_direction, axis, rotate_angle)
            output.append(replace(c, world_position=pos, local_normal=normal, local_tangent_direction=tangent, boundary_direction=boundary_dir))
        return output

    def test_rotation_translation_scale_exact_invariance(self):
        n = 14
        base = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)
        _, base_components = recover_directed_boundary_components(base, accepted)
        base_closed = sorted(tuple(sorted(c.ordered_source_ids)) for c in base_components if c.ordering_state == "ordered_closed_loop")
        self.assertEqual(base_closed, [tuple(range(n))])

        for kwargs in [
            dict(rotate_angle=0.77),
            dict(translate=(5.0, -3.0, 2.0)),
            dict(scale=3.5),
            dict(rotate_angle=1.1, translate=(1.0, 2.0, -1.0), scale=0.4),
        ]:
            transformed = self._transform(base, **kwargs)
            _, components = recover_directed_boundary_components(transformed, accepted)
            closed = sorted(tuple(sorted(c.ordered_source_ids)) for c in components if c.ordering_state == "ordered_closed_loop")
            self.assertEqual(closed, base_closed, kwargs)


class MaterializationBoundaryAccuracyTest(unittest.TestCase):
    def test_box_face_materialized_surface_has_low_boundary_residual(self):
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=27)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        materialized = [a for a in bundle.construction.materialization_attempts if a.state == "materialized"]
        self.assertGreaterEqual(len(materialized), 1)
        for attempt in materialized:
            self.assertIsNotNone(attempt.boundary_residual)
            self.assertLess(attempt.boundary_residual, 0.2)

    def test_box_face_closed_loop_is_simple_non_self_intersecting(self):
        """Winding-number sanity: a correctly-recovered simple closed loop's
        cumulative signed turning angle (viewed from the surface normal) is
        +-2*pi, not a multiple thereof (self-crossing) or near zero (degenerate)."""
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=27)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        closed = [c for c in bundle.construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
        self.assertGreaterEqual(len(closed), 1)
        for component in closed:
            idx = torch.tensor(list(component.ordered_source_ids))
            pts = scene.positions[idx]
            centroid = pts.mean(dim=0)
            angles = torch.atan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            diffs = (angles.roll(-1) - angles)
            diffs = (diffs + torch.pi) % (2 * torch.pi) - torch.pi
            total_turning = float(diffs.sum())
            self.assertAlmostEqual(abs(total_turning), 2 * math.pi, delta=0.05)


class NegativeControlNoFalseMergeTest(unittest.TestCase):
    def _region_summary(self, scene_name: str):
        scene = make_gaussian_reliability_scene(scene_name, seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=2048)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        return bundle.construction

    def test_thin_slab_front_and_back_stay_separate(self):
        construction = self._region_summary("thin_slab")
        self.assertEqual(construction.diagnostic_summary["region_count"], 2)

    def test_floater_never_enters_a_closed_loop(self):
        construction = self._region_summary("box_isolated_floater")
        labels = ("face",) * 81 + ("floater",)
        closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
        for component in closed:
            for sid in component.ordered_source_ids:
                if sid < len(labels):
                    self.assertNotEqual(labels[sid], "floater")


class CandidateAccountingTest(unittest.TestCase):
    """Worklog 36 task section 3: every genuine candidate must receive
    exactly one final state; total across all component rows must equal the
    input candidate count."""

    def test_box_face_downsampled_all_19_candidates_accounted_for(self):
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=27)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        construction = bundle.construction
        genuine = [h for h in construction.boundary_halfedge_candidates if h.boundary_reason == "observed_support_termination"]
        total_accounted = sum(len(c.ordered_source_ids) for c in construction.ordered_boundary_components)
        self.assertEqual(total_accounted, len(genuine))
        all_final_states = {"ordered_closed_loop", "ambiguous_ordering", "isolated_boundary_candidate", "ordering_capacity_exceeded"}
        for component in construction.ordered_boundary_components:
            self.assertIn(component.ordering_state, all_final_states)

    def test_isolated_node_zero_compatibility_after_matching_gets_explicit_state(self):
        n = 5
        candidates = _ring_candidates(n)
        # Remove all accepted pairs touching node 2 -- it becomes fully isolated.
        accepted = [p for p in _ring_accepted_pairs(n) if 2 not in p]
        _, components = recover_directed_boundary_components(candidates, accepted)
        total_accounted = sum(len(c.ordered_source_ids) for c in components)
        self.assertEqual(total_accounted, n)
        isolated = [c for c in components if c.ordering_state == "isolated_boundary_candidate"]
        self.assertTrue(any(2 in c.ordered_source_ids for c in isolated))


class HungarianSolverContractTest(unittest.TestCase):
    """Worklog 36 task section 4: unmatched/dummy/forbidden-edge contract."""

    def test_node_with_no_compatible_edges_stays_unmatched_not_forced(self):
        from osn_gs.surface.torch_directed_boundary_ordering import _max_weight_one_in_one_out_matching, DirectedBoundarySuccessor

        def edge(s, t, score):
            return DirectedBoundarySuccessor(s, t, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, score, "compatible_directed_edge")

        nodes = ["a", "b", "c"]
        edges = {("a", "b"): edge("a", "b", 1.0)}
        matched = _max_weight_one_in_one_out_matching(nodes, edges)
        self.assertEqual(matched, {"a": "b"})
        self.assertNotIn("c", matched)
        self.assertNotIn("c", matched.values())

    def test_deterministic_across_repeated_calls(self):
        from osn_gs.surface.torch_directed_boundary_ordering import _max_weight_one_in_one_out_matching, DirectedBoundarySuccessor

        def edge(s, t, score):
            return DirectedBoundarySuccessor(s, t, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, score, "compatible_directed_edge")

        nodes = ["a", "b", "c"]
        edges = {("a", "b"): edge("a", "b", 1.0), ("a", "c"): edge("a", "c", 1.0)}
        results = [_max_weight_one_in_one_out_matching(nodes, edges) for _ in range(5)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_max_score_edge_selected_over_lower_score_alternative(self):
        from osn_gs.surface.torch_directed_boundary_ordering import _max_weight_one_in_one_out_matching, DirectedBoundarySuccessor

        def edge(s, t, score):
            return DirectedBoundarySuccessor(s, t, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, score, "compatible_directed_edge")

        nodes = ["a", "b", "c"]
        edges = {("a", "b"): edge("a", "b", 1.0), ("a", "c"): edge("a", "c", 9.0)}
        matched = _max_weight_one_in_one_out_matching(nodes, edges)
        self.assertEqual(matched, {"a": "c"})


class CapacityLimitTest(unittest.TestCase):
    """Worklog 36 task section 7: capacity limit must fail closed explicitly,
    never silently degrade to a different-confidence approximate result."""

    def test_oversized_region_fails_closed_with_explicit_state(self):
        n = 200  # above _EXACT_MATCHING_MAX_CANDIDATES_PER_REGION
        candidates = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)
        _, components = recover_directed_boundary_components(candidates, accepted)
        states = {c.ordering_state for c in components}
        self.assertEqual(states, {"ordering_capacity_exceeded"})
        total_accounted = sum(len(c.ordered_source_ids) for c in components)
        self.assertEqual(total_accounted, n)
        for component in components:
            self.assertIn("region_candidate_count_exceeds_exact_matching_capacity", component.unresolved_reasons)


class SelfIntersectionValidationTest(unittest.TestCase):
    """Worklog 36 task section 9."""

    def test_rectangle_is_simple(self):
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        report = validate_simple_closed_loop([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
        self.assertTrue(report.is_simple_polygon)
        self.assertEqual(report.proper_intersection_count, 0)

    def test_concave_simple_loop_is_simple(self):
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        report = validate_simple_closed_loop([(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 1, 0), (0, 2, 0)])
        self.assertTrue(report.is_simple_polygon)

    def test_bow_tie_self_intersection_detected(self):
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        report = validate_simple_closed_loop([(0, 0, 0), (2, 2, 0), (2, 0, 0), (0, 2, 0)])
        self.assertFalse(report.is_simple_polygon)
        self.assertGreater(report.proper_intersection_count, 0)
        self.assertIn("proper_self_intersection", report.reasons)

    def test_repeated_vertex_detected(self):
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        report = validate_simple_closed_loop([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0), (0.5, 1, 0)])
        self.assertFalse(report.is_simple_polygon)
        self.assertGreater(report.repeated_vertex_count, 0)

    def test_near_touching_loop_stays_simple(self):
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        report = validate_simple_closed_loop([(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1e-6, 0), (0, 1, 0)])
        self.assertTrue(report.is_simple_polygon)

    def test_figure_eight_rejected(self):
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        report = validate_simple_closed_loop([(0, 0, 0), (1, 1, 0), (2, 0, 0), (1, -1, 0), (0, 0, 0), (-1, 1, 0), (-2, 0, 0), (-1, -1, 0)])
        self.assertFalse(report.is_simple_polygon)

    def test_box_face_recovered_loop_passes_self_intersection_check(self):
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians
        from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
        construction = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=stable_ids)
        closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
        self.assertGreaterEqual(len(closed), 1)
        id_to_pos = {sid: tuple(scene.positions[i].tolist()) for i, sid in enumerate(stable_ids)}
        for component in closed:
            pts = [id_to_pos[sid] for sid in component.ordered_source_ids]
            report = validate_simple_closed_loop(pts)
            self.assertTrue(report.is_simple_polygon, report.reasons)

    def test_self_intersecting_loop_fails_materialization(self):
        from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent
        from osn_gs.surface.torch_visible_boundary_materialization_adapter import materialize_visible_boundary_component

        # Bow-tie ordering: geometrically self-intersecting even though it IS
        # a structurally valid one-in/one-out closed cycle.
        component = OrderedBoundaryComponent(
            "region:0:directed:test", 0, ("h0", "h1", "h2", "h3"), (0, 1, 2, 3),
            "ordered_closed_loop", True, (), {"observed_support_termination": 4}, 0.7,
            "outer_boundary_candidate", "reliable_core_only", False, (),
        )
        boundary_points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        result = materialize_visible_boundary_component(
            component, boundary_points, boundary_points, boundary_ids=(0, 1, 2, 3), interior_ids=(0, 1, 2, 3),
        )
        self.assertNotEqual(result.state, "materialized")
        self.assertTrue(any("self_intersection" in r for r in result.review_reasons))


class PlanarityPreconditionTest(unittest.TestCase):
    """Worklog 37 task section 3: the PCA-plane self-intersection projection
    is only valid when the loop is close enough to planar."""

    def test_flat_rectangle_is_planar_enough(self):
        from osn_gs.surface.torch_boundary_self_intersection import PLANAR_ENOUGH, compute_planarity
        report = compute_planarity([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
        self.assertEqual(report.planarity_class, PLANAR_ENOUGH)

    def test_box_face_real_loop_is_planar_enough(self):
        from osn_gs.surface.torch_boundary_self_intersection import PLANAR_ENOUGH, compute_planarity
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        construction = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=stable_ids)
        closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
        self.assertGreaterEqual(len(closed), 1)
        id_to_pos = {sid: tuple(scene.positions[i].tolist()) for i, sid in enumerate(stable_ids)}
        for component in closed:
            pts = [id_to_pos[sid] for sid in component.ordered_source_ids]
            report = compute_planarity(pts)
            self.assertEqual(report.planarity_class, PLANAR_ENOUGH)

    def test_genuinely_nonplanar_loop_fails_closed(self):
        import math
        from osn_gs.surface.torch_boundary_self_intersection import NONPLANAR_AMBIGUOUS, validate_simple_closed_loop
        n = 12
        pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), 0.5 * math.sin(2 * (2 * math.pi * i / n))) for i in range(n)]
        report = validate_simple_closed_loop(pts)
        self.assertFalse(report.is_simple_polygon)
        self.assertIn("self_intersection_not_checked_nonplanar", report.reasons)
        self.assertIsNotNone(report.planarity)
        self.assertEqual(report.planarity.planarity_class, NONPLANAR_AMBIGUOUS)

    def test_cylinder_boundary_ring_loops_are_planar(self):
        """A cylinder's closed boundary loops (top/bottom rings, or the side
        wall's own edge rings) are geometrically flat circles even though
        the SURFACE they bound is curved -- the boundary curve itself is
        planar, so PCA projection remains a valid proxy."""
        from osn_gs.surface.torch_boundary_self_intersection import PLANAR_ENOUGH, compute_planarity
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians
        scene = make_gaussian_reliability_scene("cylinder", seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        construction = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=stable_ids)
        closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
        self.assertGreaterEqual(len(closed), 1)
        id_to_pos = {sid: tuple(scene.positions[i].tolist()) for i, sid in enumerate(stable_ids)}
        for component in closed:
            pts = [id_to_pos[sid] for sid in component.ordered_source_ids]
            report = compute_planarity(pts)
            self.assertEqual(report.planarity_class, PLANAR_ENOUGH)


if __name__ == "__main__":
    unittest.main()
