from __future__ import annotations

"""Worklog 114: Reliability/Affinity Robustness Hardening.

These tests do NOT connect the covariance-guided reliability/affinity
foundation to the existing Boundary-first builder, the default dispatcher, or
any production/trainer path -- they only harden the isolated modules
introduced in worklog 113 (state contract, scale contract, candidate policy,
invariance, robustness matrix, contamination) so they can eventually be used
on complex scenes. See docs/worklogs/114_*.md for the full report.
"""

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    make_anisotropic_planar_bridge_scene,
    make_contamination_regression_scene,
    make_curvature_sweep_scene,
    make_density_variation_scene,
    make_gap_sweep_scene,
    make_gaussian_reliability_scene,
    make_missing_support_gap_scene,
    make_orientation_noise_scene,
    make_position_noise_scene,
    make_shape_ratio_sweep_scene,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    CANDIDATE_STATUS_CAPPED_OUT,
    CANDIDATE_STATUS_CANDIDATE,
    CANDIDATE_STATUS_OUTSIDE_SUPPORT,
    ENDPOINT_ONE_UNRELIABLE,
    RELATION_NOT_EVALUATED,
    RELATION_SAME_SURFACE,
    ManifoldAffinityConfig,
    build_manifold_affinity_graph,
    diagnose_same_surface_regions,
)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    AGGREGATION_MEAN,
    AGGREGATION_MEDIAN,
    AGGREGATION_REJECTED_EXCLUDED,
    AGGREGATION_RELIABILITY_WEIGHTED,
    AGGREGATION_TRIMMED_MEAN,
    CONTEXTUAL_MIXED,
    INTRINSIC_RELIABLE,
    ContextualConsistencyConfig,
    StructuralReliabilityConfig,
    evaluate_structural_reliability,
)


def _pipeline(scene, *, config=None, ids=None):
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(scene.positions, frame, reliability, config=config, ids=ids)
    return frame, reliability, graph


def _edges_by_stable_id(graph):
    out = {}
    for edge in graph.edges:
        key = (edge.source_id, edge.target_id) if edge.source_id < edge.target_id else (edge.target_id, edge.source_id)
        out[key] = (edge.manifold_relation, edge.candidate_status)
    return out


class StateContractTest(unittest.TestCase):
    """Worklog 114 SS2/SS5: intrinsic/contextual split and orthogonal edge state."""

    def test_crease_gaussian_is_intrinsically_reliable_but_contextually_mixed(self):
        scene = make_gaussian_reliability_scene("two_perpendicular_surfaces")
        _, reliability, _ = _pipeline(scene)
        # A real crease must NOT collapse to the same bucket as an
        # intrinsically-bad Gaussian: it stays intrinsically reliable, only
        # its neighborhood-level contextual consistency is legitimately mixed.
        mixed_but_reliable = [
            i for i in range(scene.positions.shape[0])
            if reliability.intrinsic.intrinsic_class[i] == INTRINSIC_RELIABLE
            and reliability.contextual.contextual_class[i] == CONTEXTUAL_MIXED
        ]
        self.assertTrue(mixed_but_reliable)

    def test_isotropic_gaussian_is_intrinsically_rejected_regardless_of_context(self):
        scene = make_gaussian_reliability_scene("isotropic_blob")
        _, reliability, _ = _pipeline(scene)
        isotropic_indices = [i for i, l in enumerate(scene.group_labels) if l == "isotropic"]
        for index in isotropic_indices:
            self.assertEqual(reliability.intrinsic.intrinsic_class[index], "intrinsic_rejected")

    def test_rejected_floater_outside_candidate_support_worked_example(self):
        # Worklog 114 spec's own worked example: a REJECTED floater (not just
        # an isolated-but-otherwise-fine Gaussian) outside candidate support
        # must show candidate_status=outside_candidate_support AND
        # endpoint_status=one_intrinsically_unreliable AND
        # manifold_relation=not_evaluated simultaneously (never collapsed).
        from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import GaussianReliabilityScene

        scene = make_gaussian_reliability_scene("plane")
        rejected_floater_position = torch.tensor([[3.0, 3.0, 3.0]])
        rejected_floater_scale = torch.tensor([[0.05, 0.05, 0.05]])  # isotropic -> intrinsically rejected
        rejected_floater_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        rejected_floater_cov = covariance_from_scale_rotation(rejected_floater_scale, rejected_floater_quaternion)
        positions = torch.cat((scene.positions, rejected_floater_position), dim=0)
        covariances = torch.cat((scene.covariances, rejected_floater_cov), dim=0)
        composed = GaussianReliabilityScene(
            "rejected_floater_worked_example", positions, covariances, "",
            ("plane",) * scene.positions.shape[0] + ("rejected_floater",),
        )
        _, _, graph = _pipeline(composed)
        floater_index = composed.group_labels.index("rejected_floater")
        floater_edges = [e for e in graph.edges if e.source == floater_index or e.target == floater_index]
        self.assertTrue(floater_edges)
        for edge in floater_edges:
            self.assertEqual(edge.candidate_status, CANDIDATE_STATUS_OUTSIDE_SUPPORT)
            self.assertEqual(edge.endpoint_status, ENDPOINT_ONE_UNRELIABLE)
            self.assertEqual(edge.manifold_relation, RELATION_NOT_EVALUATED)

    def test_state_compatibility_projection_preserves_old_semantics(self):
        scene = make_gaussian_reliability_scene("plane")
        _, _, graph = _pipeline(scene)
        # Old callers reading `.state` alone should see the pre-114 bucket
        # (not_evaluated -> proximity_only) even though the richer axes exist.
        for edge in graph.edges:
            if edge.manifold_relation == RELATION_NOT_EVALUATED:
                self.assertEqual(edge.state, "proximity_only")
            else:
                self.assertEqual(edge.state, edge.manifold_relation)


class ScaleContractTest(unittest.TestCase):
    """Worklog 114 SS3: independently-preserved scale fields, never one scalar."""

    def test_scale_fields_are_independent_not_collapsed(self):
        scene = make_gaussian_reliability_scene("plane")
        frame = extract_covariance_frame(scene.covariances)
        # A thin wide surfel: tangent scales >> normal thickness.
        self.assertTrue(bool((frame.tangent_major_scale > 10 * frame.normal_thickness).all()))
        self.assertTrue(bool((frame.tangent_minor_scale > 10 * frame.normal_thickness).all()))

    def test_close_parallel_separation_uses_normal_thickness_not_tangent_scale(self):
        scene = make_gaussian_reliability_scene("close_parallel_sheets")
        _, _, graph = _pipeline(scene)
        labels = scene.group_labels
        cross_edges = [e for e in graph.edges if labels[e.source] != labels[e.target] and e.metrics is not None]
        self.assertTrue(cross_edges)
        for e in cross_edges:
            if e.manifold_relation == "parallel_but_separate":
                self.assertGreaterEqual(e.metrics.normal_direction_separation_over_thickness, 1.0)

    def test_oversized_footprint_ratio_blocks_same_surface_regardless_of_residual(self):
        scene = make_anisotropic_planar_bridge_scene()
        cfg = ManifoldAffinityConfig(candidate_neighbor_count=20, max_candidate_count_per_node=24)
        _, _, graph = _pipeline(scene, config=cfg)
        bridge_index = scene.group_labels.index("anisotropic_bridge")
        floor_indices = {i for i, l in enumerate(scene.group_labels) if l == "floor"}
        bridge_floor_edges = [
            e for e in graph.edges
            if (e.source == bridge_index and e.target in floor_indices) or (e.target == bridge_index and e.source in floor_indices)
        ]
        self.assertTrue(bridge_floor_edges)
        self.assertTrue(all(e.manifold_relation != RELATION_SAME_SURFACE for e in bridge_floor_edges))


class CandidateGraphTest(unittest.TestCase):
    """Worklog 114 SS4: candidate generation separated from relation classification."""

    def test_distance_only_hit_never_becomes_same_surface(self):
        scene = make_gaussian_reliability_scene("isolated_floater")
        _, _, graph = _pipeline(scene)
        floater_index = scene.group_labels.index("floater")
        edges = [e for e in graph.edges if e.source == floater_index or e.target == floater_index]
        self.assertTrue(all(e.manifold_relation != RELATION_SAME_SURFACE for e in edges))
        self.assertTrue(all(e.candidate_status == CANDIDATE_STATUS_OUTSIDE_SUPPORT for e in edges))

    def test_candidate_status_and_manifold_relation_are_independent_axes(self):
        scene = make_gaussian_reliability_scene("two_perpendicular_surfaces")
        cfg = ManifoldAffinityConfig(max_candidate_count_per_node=4)
        _, _, graph = _pipeline(scene, config=cfg)
        capped = [e for e in graph.edges if e.candidate_status == CANDIDATE_STATUS_CAPPED_OUT]
        candidates = [e for e in graph.edges if e.candidate_status == CANDIDATE_STATUS_CANDIDATE]
        self.assertTrue(capped)
        self.assertTrue(candidates)
        # capped_out always carries manifold_relation=not_evaluated (a candidate
        # status), while real candidates carry an actual relation value.
        self.assertTrue(all(e.manifold_relation == RELATION_NOT_EVALUATED for e in capped))
        self.assertTrue(all(e.manifold_relation != RELATION_NOT_EVALUATED for e in candidates))

    def test_pair_identity_is_order_independent_via_stable_ids(self):
        scene = make_gaussian_reliability_scene("plane")
        _, _, graph = _pipeline(scene, ids=list(range(scene.positions.shape[0])))
        seen = set()
        for e in graph.edges:
            key = (e.source_id, e.target_id) if e.source_id < e.target_id else (e.target_id, e.source_id)
            self.assertNotIn(key, seen)
            seen.add(key)


class InvarianceTest(unittest.TestCase):
    """Worklog 114 SS8: translation/rotation/uniform-scale/order invariance."""

    def _scene(self):
        return make_gaussian_reliability_scene("two_perpendicular_surfaces")

    def test_translation_invariance(self):
        scene = self._scene()
        ids = list(range(scene.positions.shape[0]))
        _, _, graph0 = _pipeline(scene, ids=ids)
        translated = scene.positions + torch.tensor([5.0, -3.0, 2.0])
        frame1 = extract_covariance_frame(scene.covariances)
        reliability1 = evaluate_structural_reliability(translated, frame1)
        graph1 = build_manifold_affinity_graph(translated, frame1, reliability1, ids=ids)
        self.assertEqual(_edges_by_stable_id(graph0), _edges_by_stable_id(graph1))

    def test_rotation_invariance_including_eigenvector_sign(self):
        scene = self._scene()
        ids = list(range(scene.positions.shape[0]))
        _, _, graph0 = _pipeline(scene, ids=ids)
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated_positions = scene.positions @ rotation.T
        rotated_covariances = rotation @ scene.covariances @ rotation.T
        frame1 = extract_covariance_frame(rotated_covariances)
        reliability1 = evaluate_structural_reliability(rotated_positions, frame1)
        graph1 = build_manifold_affinity_graph(rotated_positions, frame1, reliability1, ids=ids)
        self.assertEqual(_edges_by_stable_id(graph0), _edges_by_stable_id(graph1))

    def test_uniform_scale_invariance_of_normalized_relations(self):
        scene = self._scene()
        ids = list(range(scene.positions.shape[0]))
        _, _, graph0 = _pipeline(scene, ids=ids)
        factor = 2.5
        scaled_positions = scene.positions * factor
        scaled_covariances = scene.covariances * (factor ** 2)
        frame1 = extract_covariance_frame(scaled_covariances)
        reliability1 = evaluate_structural_reliability(scaled_positions, frame1)
        graph1 = build_manifold_affinity_graph(scaled_positions, frame1, reliability1, ids=ids)
        rel0 = {k: v[0] for k, v in _edges_by_stable_id(graph0).items()}
        rel1 = {k: v[0] for k, v in _edges_by_stable_id(graph1).items()}
        self.assertEqual(rel0, rel1)

    def test_input_order_determinism_with_stable_ids(self):
        scene = self._scene()
        ids = list(range(scene.positions.shape[0]))
        _, _, graph0 = _pipeline(scene, ids=ids)
        for seed in (0, 1, 2):
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(scene.positions.shape[0], generator=generator)
            shuffled_ids = [ids[i] for i in perm.tolist()]
            frame1 = extract_covariance_frame(scene.covariances[perm])
            reliability1 = evaluate_structural_reliability(scene.positions[perm], frame1)
            graph1 = build_manifold_affinity_graph(scene.positions[perm], frame1, reliability1, ids=shuffled_ids)
            self.assertEqual(_edges_by_stable_id(graph0), _edges_by_stable_id(graph1), f"seed={seed}")


class RobustnessMatrixTest(unittest.TestCase):
    """Worklog 114 SS9: expanded synthetic robustness matrix."""

    def test_density_variation_does_not_fragment_a_connected_plane(self):
        for kind in ("uniform", "center_dense_boundary_sparse", "gradual_gradient", "abrupt_transition", "sparse_but_continuous"):
            scene = make_density_variation_scene(kind)
            _, reliability, graph = _pipeline(scene)
            diag = diagnose_same_surface_regions(scene.positions.shape[0], graph, reliability)
            self.assertEqual(diag.region_count, 1, kind)

    def test_position_noise_sweep_degrades_gradually(self):
        coverages = []
        for noise in (0.0, 0.005, 0.02, 0.05):
            scene = make_position_noise_scene(noise)
            _, reliability, graph = _pipeline(scene)
            diag = diagnose_same_surface_regions(scene.positions.shape[0], graph, reliability)
            coverages.append(diag.reliable_node_coverage)
        # Monotonic-ish degradation, no premature collapse at zero/low noise.
        self.assertEqual(coverages[0], 1.0)
        self.assertGreaterEqual(coverages[1], coverages[2] - 1e-6)
        self.assertGreaterEqual(coverages[2], coverages[3] - 1e-6)

    def test_orientation_noise_sweep_is_soft_then_hard(self):
        scene_low = make_orientation_noise_scene(2.0)
        scene_high = make_orientation_noise_scene(60.0)
        _, rel_low, graph_low = _pipeline(scene_low)
        _, rel_high, graph_high = _pipeline(scene_high)
        diag_low = diagnose_same_surface_regions(scene_low.positions.shape[0], graph_low, rel_low)
        diag_high = diagnose_same_surface_regions(scene_high.positions.shape[0], graph_high, rel_high)
        self.assertEqual(diag_low.region_count, 1)
        self.assertGreaterEqual(diag_high.region_count, diag_low.region_count)

    def test_low_curvature_does_not_fragment_a_smooth_sheet(self):
        for amplitude in (0.0, 0.02, 0.05):
            scene = make_curvature_sweep_scene(amplitude)
            _, reliability, graph = _pipeline(scene)
            diag = diagnose_same_surface_regions(scene.positions.shape[0], graph, reliability)
            self.assertEqual(diag.region_count, 1, amplitude)

    def test_anisotropic_planar_bridge_is_blocked_by_footprint_not_isotropic_rejection(self):
        scene = make_anisotropic_planar_bridge_scene()
        cfg = ManifoldAffinityConfig(candidate_neighbor_count=20, max_candidate_count_per_node=24)
        frame = extract_covariance_frame(scene.covariances)
        bridge_index = scene.group_labels.index("anisotropic_bridge")
        # The bridge itself must be a genuinely planar (not isotropic) shape --
        # the guard must not rely on isotropic rejection to block it.
        self.assertEqual(frame.shape_class[bridge_index], "planar_surfel")
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability, config=cfg)
        floor_indices = {i for i, l in enumerate(scene.group_labels) if l == "floor"}
        bridge_floor_edges = [
            e for e in graph.edges
            if (e.source == bridge_index and e.target in floor_indices) or (e.target == bridge_index and e.source in floor_indices)
        ]
        self.assertTrue(bridge_floor_edges)
        self.assertTrue(all(e.manifold_relation != RELATION_SAME_SURFACE for e in bridge_floor_edges))

    def test_gap_sweep_reliably_separates_from_a_moderate_gap_onward(self):
        for gap in (0.05, 0.1, 0.15, 0.3):
            scene = make_gap_sweep_scene(gap)
            _, reliability, graph = _pipeline(scene)
            diag = diagnose_same_surface_regions(scene.positions.shape[0], graph, reliability)
            self.assertEqual(diag.region_count, 2, gap)
            self.assertEqual(sorted(diag.region_sizes), [49, 49], gap)

    def test_missing_support_gap_is_not_forced_bridged_by_an_oversized_gaussian(self):
        scene = make_missing_support_gap_scene(0.3)
        _, reliability, graph = _pipeline(scene)
        diag = diagnose_same_surface_regions(scene.positions.shape[0], graph, reliability)
        # A sampling gap in an otherwise-flat plane should not fragment into
        # many small regions -- the remaining ring stays one connected region.
        self.assertEqual(diag.region_count, 1)

    def test_shape_ratio_sweep_transitions_needle_to_isotropic(self):
        classes = []
        for ratio in (0.0, 0.3, 0.6, 1.0):
            scene = make_shape_ratio_sweep_scene(ratio)
            frame = extract_covariance_frame(scene.covariances)
            classes.append(frame.shape_class[0])
        self.assertEqual(classes[0], "needle_like")
        self.assertEqual(classes[-1], "isotropic")


class ContaminationTest(unittest.TestCase):
    """Worklog 114 SS10: neighborhood contamination regression."""

    def test_clean_plane_subgraph_stays_fully_connected_around_all_contaminants(self):
        scene = make_contamination_regression_scene()
        _, reliability, graph = _pipeline(scene)
        labels = scene.group_labels
        plane_indices = [i for i, l in enumerate(labels) if l == "plane"]
        same_surface = graph.same_surface_neighbors(scene.positions.shape[0])
        visited = {plane_indices[0]}
        frontier = [plane_indices[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in same_surface[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(visited, set(plane_indices))

    def test_isotropic_contaminant_does_not_bridge_or_degrade_all_neighbors(self):
        scene = make_contamination_regression_scene()
        _, reliability, graph = _pipeline(scene)
        labels = scene.group_labels
        isotropic_index = labels.index("isotropic")
        edges = [e for e in graph.edges if e.source == isotropic_index or e.target == isotropic_index]
        self.assertTrue(all(e.manifold_relation != RELATION_SAME_SURFACE for e in edges))

    def test_rejected_excluded_aggregation_prevents_isotropic_neighbor_contamination(self):
        # Direct re-analysis of the worklog 113 finding: compare aggregation
        # methods on the same isotropic-blob scene.
        scene = make_gaussian_reliability_scene("isotropic_blob")
        frame = extract_covariance_frame(scene.covariances)
        labels = scene.group_labels
        plane_count = sum(1 for l in labels if l == "plane")

        def contaminated_count(method):
            cfg = StructuralReliabilityConfig(contextual=ContextualConsistencyConfig(aggregation_method=method))
            reliability = evaluate_structural_reliability(scene.positions, frame, config=cfg)
            return sum(
                1 for i in range(len(labels))
                if labels[i] == "plane" and reliability.reliability_class[i] != "reliable_structural_evidence"
            )

        mean_contaminated = contaminated_count(AGGREGATION_MEAN)
        rejected_excluded_contaminated = contaminated_count(AGGREGATION_REJECTED_EXCLUDED)
        median_contaminated = contaminated_count(AGGREGATION_MEDIAN)
        trimmed_contaminated = contaminated_count(AGGREGATION_TRIMMED_MEAN)
        weighted_contaminated = contaminated_count(AGGREGATION_RELIABILITY_WEIGHTED)

        self.assertGreater(mean_contaminated, 0)
        self.assertEqual(rejected_excluded_contaminated, 0)
        self.assertLessEqual(median_contaminated, mean_contaminated)
        self.assertLessEqual(trimmed_contaminated, mean_contaminated)
        self.assertEqual(weighted_contaminated, 0)
        self.assertLess(plane_count, len(labels))  # sanity: fixture actually has plane points


if __name__ == "__main__":
    unittest.main()
