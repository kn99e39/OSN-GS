from __future__ import annotations

"""Stage 3 diagnostics-only merge-only agglomeration tests."""

import hashlib
import json
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ProxySurfaceDecompositionTest(unittest.TestCase):
    def _case(self, name: str, count: int = 600, seed: int = 0):
        from nurbs_constructor_benchmark.scenes import make_scene
        from osn_gs.surface.torch_surface_candidate_graph import (
            build_surface_cell_candidate_graph,
        )
        from osn_gs.surface.torch_voxel_hierarchy import (
            build_voxel_gaussian_hierarchy,
        )

        scene = make_scene(name, count=count, seed=seed)
        hierarchy = build_voxel_gaussian_hierarchy(
            scene.points,
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=150,
            voxel_max_depth=6,
        )
        graph = build_surface_cell_candidate_graph(
            hierarchy, scene.points, radius_factor=0.25, max_neighbors=0
        )
        return scene, hierarchy, graph

    @staticmethod
    def _result(hierarchy, points, graph=None, config=None):
        from osn_gs.surface.torch_surface_decomposition import (
            build_proxy_surface_components_diagnostics,
        )

        return build_proxy_surface_components_diagnostics(
            hierarchy, points, candidate_graph=graph, config=config
        )

    @staticmethod
    def _leaf_majority_labels(hierarchy, leaf_ids, labels):
        lookup = {node.node_id: node for node in hierarchy.leaves()}
        result = {}
        for leaf_id in leaf_ids:
            values, counts = torch.unique(
                labels[lookup[leaf_id].gaussian_indices].long(), return_counts=True
            )
            result[leaf_id] = int(values[int(torch.argmax(counts))])
        return result

    def _assert_regions_do_not_mix_labels(self, result, leaf_labels):
        for region in result.final_regions:
            self.assertEqual(
                len({leaf_labels[item] for item in region.member_leaf_ids}), 1
            )

    def test_curved_annulus_recovers_one_diagnostic_region_only(self):
        from osn_gs.surface.torch_surface_components import build_surface_components

        scene, hierarchy, graph = self._case("curved_annulus")
        production_before = build_surface_components(hierarchy, scene.points)
        result = self._result(hierarchy, scene.points, graph)
        production_after = build_surface_components(hierarchy, scene.points)

        self.assertEqual(production_before.component_count(), 2)
        self.assertEqual(result.component_count(), 1)
        self.assertEqual(production_after.component_count(), 2)
        self.assertEqual(
            production_before.leaf_component_id, production_after.leaf_component_id
        )
        self.assertFalse(result.payload()["production_membership_changed"])

    def test_crease_parallel_and_disconnected_controls_stay_split(self):
        from scripts.devtools.analyze_surface_candidate_graph import (
            _disconnected_close_points,
        )
        from osn_gs.surface.torch_surface_candidate_graph import (
            build_surface_cell_candidate_graph,
        )
        from osn_gs.surface.torch_voxel_hierarchy import (
            build_voxel_gaussian_hierarchy,
        )

        for name in ("crease", "close_parallel_sheets"):
            with self.subTest(name=name):
                scene, hierarchy, graph = self._case(name)
                labels = (
                    scene.gt_patch_label(scene.points[:, :2])
                    if name == "crease"
                    else scene.gt_patch_label(scene.points)
                )
                result = self._result(hierarchy, scene.points, graph)
                leaf_labels = self._leaf_majority_labels(
                    hierarchy, result.initial_leaf_ids, labels
                )
                self.assertEqual(result.component_count(), 2)
                self._assert_regions_do_not_mix_labels(result, leaf_labels)

        parallel_scene, _, _ = self._case("close_parallel_sheets")
        parallel_labels = parallel_scene.gt_patch_label(parallel_scene.points)
        close_points = parallel_scene.points.clone()
        close_points[:, 2] = torch.where(
            parallel_labels == 0,
            torch.full((600,), 0.015),
            torch.full((600,), -0.015),
        )
        parallel_hierarchy = build_voxel_gaussian_hierarchy(
            close_points,
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=150,
            voxel_max_depth=6,
        )
        close_graph = build_surface_cell_candidate_graph(
            parallel_hierarchy, close_points
        )
        close_result = self._result(
            parallel_hierarchy, close_points, close_graph
        )
        close_leaf_labels = self._leaf_majority_labels(
            parallel_hierarchy, close_result.initial_leaf_ids, parallel_labels
        )
        self.assertEqual(close_result.component_count(), 2)
        self._assert_regions_do_not_mix_labels(close_result, close_leaf_labels)

        points, labels = _disconnected_close_points(600, 0, 0.1)
        hierarchy = build_voxel_gaussian_hierarchy(
            points,
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=150,
            voxel_max_depth=6,
        )
        graph = build_surface_cell_candidate_graph(hierarchy, points)
        result = self._result(hierarchy, points, graph)
        leaf_labels = self._leaf_majority_labels(
            hierarchy, result.initial_leaf_ids, labels
        )
        self.assertEqual(result.component_count(), 2)
        self._assert_regions_do_not_mix_labels(result, leaf_labels)

    def test_planar_and_density_gradient_regression_controls_reach_one_region(self):
        for name in (
            "plane",
            "planar_hole",
            "planar_hole_offcenter",
            "planar_hole_elliptical",
            "planar_hole_density_gradient",
            "density_gradient",
        ):
            with self.subTest(name=name):
                scene, hierarchy, graph = self._case(name)
                result = self._result(hierarchy, scene.points, graph)
                self.assertEqual(result.component_count(), 1)

    def test_known_high_curvature_quadratic_control_is_not_layer_rejected(self):
        from osn_gs.surface.torch_voxel_hierarchy import (
            build_voxel_gaussian_hierarchy,
        )

        axis = torch.linspace(-1.0, 1.0, 24)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack(
            [
                xx.flatten(),
                yy.flatten(),
                0.55 * xx.flatten().square() + 0.25 * yy.flatten().square(),
            ],
            dim=1,
        )
        hierarchy = build_voxel_gaussian_hierarchy(
            points,
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=150,
            voxel_max_depth=6,
        )
        result = self._result(hierarchy, points)
        self.assertEqual(result.component_count(), 1)

    def test_repeat_and_shuffled_candidate_order_are_hash_identical(self):
        from osn_gs.surface.torch_surface_candidate_graph import SurfaceCandidateGraph

        scene, hierarchy, graph = self._case("curved_annulus")
        first = self._result(hierarchy, scene.points, graph)
        repeated = self._result(hierarchy, scene.points, graph)
        shuffled = SurfaceCandidateGraph(
            node_ids=list(reversed(graph.node_ids)),
            edges=list(reversed(graph.edges)),
            config=dict(graph.config),
        )
        reordered = self._result(hierarchy, scene.points, shuffled)

        def digest(result):
            encoded = json.dumps(
                result.payload(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        self.assertEqual(digest(first), digest(repeated))
        self.assertEqual(digest(first), digest(reordered))

    def test_stale_queue_provenance_and_merge_history_invariants(self):
        scene, hierarchy, graph = self._case("curved_annulus")
        result = self._result(hierarchy, scene.points, graph)
        flattened = [
            leaf_id
            for region in result.final_regions
            for leaf_id in region.member_leaf_ids
        ]
        self.assertEqual(sorted(flattened), result.initial_leaf_ids)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            len(result.merge_history),
            len(result.initial_leaf_ids) - result.component_count(),
        )
        self.assertGreater(result.stale_queue_entry_count, 0)
        self.assertEqual(result.termination_reason, "one_region")

    def test_all_gate_results_are_preserved_before_reason_selection(self):
        from osn_gs.surface.torch_surface_decomposition import (
            GATE_ORDER,
            ProxySurfaceDecompositionConfig,
        )

        scene, hierarchy, graph = self._case("curved_annulus")
        config = ProxySurfaceDecompositionConfig(
            max_normalized_proxy_rms=0.0,
            max_normalized_error_increase=0.0,
            max_support_gap_over_spacing=0.0,
            max_layer_separation=0.0,
            min_layer_rms_ratio=0.0,
            min_layer_normalized_error_increase=0.0,
            minimum_support=1000,
        )
        result = self._result(hierarchy, scene.points, graph, config)
        self.assertEqual(result.termination_reason, "no_admissible_candidate_pairs")
        self.assertTrue(result.pair_evaluations)
        for evaluation in result.pair_evaluations:
            self.assertEqual(tuple(evaluation.gate_results), GATE_ORDER)
            self.assertTrue(evaluation.failed_gates)
            self.assertEqual(evaluation.decision_reason, evaluation.failed_gates[0])

    def test_candidate_config_mismatch_is_rejected(self):
        from osn_gs.surface.torch_surface_decomposition import (
            ProxySurfaceDecompositionConfig,
        )

        scene, hierarchy, graph = self._case("plane", count=300)
        with self.assertRaises(ValueError):
            self._result(
                hierarchy,
                scene.points,
                graph,
                ProxySurfaceDecompositionConfig(candidate_radius_factor=0.5),
            )

    def test_invalid_config_is_rejected(self):
        from osn_gs.surface.torch_surface_decomposition import (
            ProxySurfaceDecompositionConfig,
        )

        with self.assertRaises(ValueError):
            ProxySurfaceDecompositionConfig(max_normalized_proxy_rms=-1.0)
        with self.assertRaises(ValueError):
            ProxySurfaceDecompositionConfig(minimum_support=0)
        with self.assertRaises(ValueError):
            ProxySurfaceDecompositionConfig(support_gap_quantile=1.1)


if __name__ == "__main__":
    unittest.main()
