from __future__ import annotations

"""Stage 2 diagnostics-only spatial candidate graph tests."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class SurfaceCandidateGraphTest(unittest.TestCase):
    def _scene_graph(self, scene_name: str = "curved_annulus", **graph_kwargs):
        from nurbs_constructor_benchmark.scenes import make_scene
        from osn_gs.surface.torch_surface_candidate_graph import (
            build_surface_cell_candidate_graph,
        )
        from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy

        scene = make_scene(scene_name, count=600, seed=0)
        hierarchy = build_voxel_gaussian_hierarchy(
            scene.points,
            voxel_min_gaussian_count=10,
            voxel_max_gaussian_count=150,
            voxel_max_depth=6,
        )
        graph = build_surface_cell_candidate_graph(
            hierarchy, scene.points, **graph_kwargs
        )
        return scene, hierarchy, graph

    def test_curved_annulus_known_missing_pairs_are_candidates(self):
        _, _, graph = self._scene_graph(radius_factor=0.25)
        pairs = graph.edge_pairs()
        for pair in (("r07", "r52"), ("r05", "r50"), ("r05", "r52"), ("r07", "r50")):
            self.assertIn(pair, pairs)

    def test_existing_face_smooth_pair_recall_is_one(self):
        from osn_gs.surface.torch_surface_candidate_graph import candidate_graph_payload
        from osn_gs.surface.torch_surface_components import build_surface_components

        scene, hierarchy, graph = self._scene_graph("mild_curved_sheet")
        components = build_surface_components(hierarchy, scene.points)
        smooth_pairs = [
            (edge.leaf_a, edge.leaf_b)
            for edge in components.edge_decisions
            if edge.reason == "merged"
        ]
        payload = candidate_graph_payload(
            graph, reference_pairs={"existing_face_smooth": smooth_pairs}
        )
        self.assertEqual(
            payload["reference_recall"]["existing_face_smooth"]["recall"], 1.0
        )

    def test_ordering_duplicates_and_repeatability(self):
        _, _, first = self._scene_graph(radius_factor=0.25)
        _, _, second = self._scene_graph(radius_factor=0.25)
        first_pairs = [edge.pair for edge in first.edges]
        self.assertEqual(first_pairs, sorted(first_pairs))
        self.assertEqual(len(first_pairs), len(set(first_pairs)))
        self.assertEqual(first_pairs, [edge.pair for edge in second.edges])
        self.assertEqual(
            [edge.payload() for edge in first.edges],
            [edge.payload() for edge in second.edges],
        )

    def test_positive_neighbor_cap_is_deterministic_and_bounded(self):
        _, _, graph = self._scene_graph(radius_factor=1.0, max_neighbors=3)
        self.assertLessEqual(max(graph.degree_by_node().values()), 3)
        _, _, repeated = self._scene_graph(radius_factor=1.0, max_neighbors=3)
        self.assertEqual(
            [edge.pair for edge in graph.edges],
            [edge.pair for edge in repeated.edges],
        )

    def test_contact_relation_is_diagnostic_only(self):
        _, _, touching_only = self._scene_graph(radius_factor=0.0)
        relations = {edge.contact_relation for edge in touching_only.edges}
        self.assertTrue(relations <= {"overlap", "face", "edge", "corner"})
        # Edge/corner contacts remain candidates even though legacy Phase 1
        # generated only face-adjacent pairs.
        self.assertTrue(relations & {"edge", "corner"})

    def test_payload_reports_degree_source_and_recall(self):
        from osn_gs.surface.torch_surface_candidate_graph import candidate_graph_payload

        _, _, graph = self._scene_graph(radius_factor=0.25)
        payload = candidate_graph_payload(
            graph,
            reference_pairs={"known_missing": [("r07", "r52")]},
            diagnostic_tags={("r07", "r52"): ["known_missing_smooth"]},
        )
        self.assertEqual(payload["reference_recall"]["known_missing"]["recall"], 1.0)
        self.assertEqual(sum(payload["degree"]["histogram"].values()), payload["node_count"])
        self.assertEqual(sum(payload["candidate_source_counts"].values()), payload["edge_count"])
        tagged = [edge for edge in payload["edges"] if edge["diagnostic_tags"]]
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0]["pair_id"], "r07|r52")
        self.assertGreaterEqual(tagged[0]["support_gap"], 0.0)
        self.assertGreaterEqual(tagged[0]["scale_normalized_gap"], 0.0)

    def test_boundary_first_diagnostics_do_not_change_membership_or_fit(self):
        from nurbs_constructor_benchmark.boundary_first import construct_boundary_first
        from nurbs_constructor_benchmark.scenes import make_scene

        scene = make_scene("plane", count=300, seed=3)
        baseline, baseline_patches = construct_boundary_first(scene)
        diagnosed, diagnosed_patches = construct_boundary_first(
            scene, candidate_graph_diagnostics=True
        )
        self.assertTrue(torch.equal(baseline.model.cluster_ids, diagnosed.model.cluster_ids))
        self.assertTrue(torch.equal(baseline.model.surface_uv, diagnosed.model.surface_uv))
        self.assertEqual(baseline.per_component, diagnosed.per_component)
        self.assertEqual(baseline_patches, diagnosed_patches)
        self.assertIsNone(baseline.candidate_graph)
        self.assertIsNotNone(diagnosed.candidate_graph)

    def test_curved_annulus_diagnostics_preserve_two_production_components(self):
        from nurbs_constructor_benchmark.boundary_first import construct_boundary_first
        from nurbs_constructor_benchmark.scenes import make_scene

        scene = make_scene("curved_annulus", count=600, seed=0)
        baseline, _ = construct_boundary_first(scene)
        diagnosed, _ = construct_boundary_first(
            scene, candidate_graph_diagnostics=True
        )
        self.assertEqual(baseline.component_count, 2)
        self.assertEqual(diagnosed.component_count, 2)
        self.assertTrue(torch.equal(baseline.model.cluster_ids, diagnosed.model.cluster_ids))
        pairs = {
            (edge["cell_a"], edge["cell_b"])
            for edge in diagnosed.candidate_graph["edges"]
        }
        self.assertIn(("r07", "r52"), pairs)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            self._scene_graph(radius_factor=-0.1)
        with self.assertRaises(ValueError):
            self._scene_graph(max_neighbors=-1)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class AABBContactClassificationTest(unittest.TestCase):
    def test_face_edge_corner_and_disjoint(self):
        from osn_gs.surface.torch_surface_candidate_graph import classify_aabb_contact

        lo = torch.tensor([0.0, 0.0, 0.0])
        hi = torch.tensor([1.0, 1.0, 1.0])
        self.assertEqual(
            classify_aabb_contact(lo, hi, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0, 1.0, 1.0]), 1e-8),
            "face",
        )
        self.assertEqual(
            classify_aabb_contact(lo, hi, torch.tensor([1.0, 1.0, 0.0]), torch.tensor([2.0, 2.0, 1.0]), 1e-8),
            "edge",
        )
        self.assertEqual(
            classify_aabb_contact(lo, hi, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0]), 1e-8),
            "corner",
        )
        self.assertEqual(
            classify_aabb_contact(lo, hi, torch.tensor([1.1, 0.0, 0.0]), torch.tensor([2.1, 1.0, 1.0]), 1e-8),
            "disjoint",
        )

    def test_shared_degenerate_axis_does_not_inflate_contact_dimension(self):
        from osn_gs.surface.torch_surface_candidate_graph import classify_aabb_contact

        self.assertEqual(
            classify_aabb_contact(
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0]),
                torch.tensor([2.0, 1.0, 0.0]),
                1e-8,
            ),
            "face",
        )


if __name__ == "__main__":
    unittest.main()
