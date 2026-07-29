from __future__ import annotations

import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_curvature_sweep_scene
from nurbs_constructor_benchmark.surface_region_adversarial_scenes import make_genuine_narrow_connection_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import ManifoldAffinityConfig, build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationConfig, form_surface_regions


def _form(scene, *, broad=False):
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(
        scene.positions, frame, reliability,
        config=ManifoldAffinityConfig(candidate_neighbor_count=20, max_candidate_count_per_node=24) if broad else None,
        ids=list(range(scene.positions.shape[0])),
    )
    result = form_surface_regions(
        scene.positions, frame, reliability, graph,
        config=RegionFormationConfig(nonlocal_shortcut_mode="auto"),
        ids=list(range(scene.positions.shape[0])),
    )
    return graph, result


class PhaseAliasAndNarrowConnectionTest(unittest.TestCase):
    def test_auto_policy_excludes_broad_curved_shortcuts_from_accepted_topology(self):
        scene = make_curvature_sweep_scene(0.05)
        graph, result = _form(scene, broad=True)
        shortcuts = {
            tuple(sorted((edge.source, edge.target)))
            for edge in graph.edges
            if edge.manifold_relation == "same_surface" and edge.metrics.normalized_distance > 4.0
        }
        accepted = {tuple(sorted(edge)) for region in result.regions for edge in region.internal_accepted_edge_ids}
        self.assertTrue(shortcuts)
        self.assertFalse(shortcuts & accepted)
        self.assertEqual(len(result.regions), 1)

    def test_genuine_multi_edge_neck_remains_one_region(self):
        _, result = _form(make_genuine_narrow_connection_scene())
        self.assertEqual(len(result.regions), 1)


if __name__ == "__main__":
    unittest.main()
