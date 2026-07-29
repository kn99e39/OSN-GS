from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    EDGE_AMBIGUOUS,
    EDGE_CREASE,
    EDGE_PARALLEL_SEPARATE,
    EDGE_PROXIMITY_ONLY,
    EDGE_REJECTED,
    EDGE_SAME_SURFACE,
    build_manifold_affinity_graph,
    classify_node_boundary_status,
)
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability


def _graph_for(name: str):
    scene = make_gaussian_reliability_scene(name)
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
    return scene, frame, reliability, graph


def _cross_label_states(scene, graph):
    labels = scene.group_labels
    states = set()
    for edge in graph.edges:
        if labels[edge.source] != labels[edge.target]:
            states.add(edge.state)
    return states


class GaussianManifoldAffinityTest(unittest.TestCase):
    def test_same_plane_neighbors_are_same_surface(self):
        scene, _, _, graph = _graph_for("plane")
        states = {edge.state for edge in graph.edges}
        self.assertIn(EDGE_SAME_SURFACE, states)
        self.assertTrue(all(edge.state in (EDGE_SAME_SURFACE, EDGE_PROXIMITY_ONLY) for edge in graph.edges))

    def test_perpendicular_wall_floor_contact_is_a_crease(self):
        scene, _, _, graph = _graph_for("two_perpendicular_surfaces")
        cross_states = _cross_label_states(scene, graph)
        self.assertIn(EDGE_CREASE, cross_states)
        self.assertNotIn(EDGE_SAME_SURFACE, cross_states)

    def test_close_parallel_sheets_are_not_connected_as_same_surface(self):
        scene, _, _, graph = _graph_for("close_parallel_sheets")
        cross_states = _cross_label_states(scene, graph)
        self.assertNotIn(EDGE_SAME_SURFACE, cross_states)
        # Aligned normals but a real tangent-plane offset -> recognized as
        # parallel-but-separate, not silently dropped or force-merged.
        self.assertIn(EDGE_PARALLEL_SEPARATE, cross_states)

    def test_smooth_curved_sheet_local_neighbors_can_be_same_surface(self):
        _, _, _, graph = _graph_for("smooth_curved_sheet")
        states = {edge.state for edge in graph.edges}
        self.assertIn(EDGE_SAME_SURFACE, states)

    def test_aligned_normal_with_large_mutual_residual_is_not_same_surface(self):
        from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame as _extract
        from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph as _build
        from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability as _evaluate
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import _flat_grid

        lower, lower_cov = _flat_grid(6, 0.1, origin=(0.0, 0.0, 0.0))
        upper, upper_cov = _flat_grid(6, 0.1, origin=(0.0, 0.0, 5.0))  # far offset along the shared normal
        positions = torch.cat((lower, upper), dim=0)
        covariances = torch.cat((lower_cov, upper_cov), dim=0)
        frame = _extract(covariances)
        reliability = _evaluate(positions, frame)
        graph = _build(positions, frame, reliability)
        labels = ("lower",) * lower.shape[0] + ("upper",) * upper.shape[0]
        cross_states = {edge.state for edge in graph.edges if labels[edge.source] != labels[edge.target]}
        self.assertNotIn(EDGE_SAME_SURFACE, cross_states)

    def test_one_ambiguous_endpoint_with_inconclusive_geometry_is_not_a_hard_boundary(self):
        # Borderline normal alignment (neither same_surface nor a clean crease)
        # combined with a non-reliable endpoint must stay "ambiguous", never
        # forced into a confident state.
        scene = make_gaussian_reliability_scene("isotropic_blob")
        frame = extract_covariance_frame(scene.covariances)
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
        isotropic_indices = {i for i, label in enumerate(scene.group_labels) if label == "isotropic"}
        touching_isotropic = [e for e in graph.edges if e.source in isotropic_indices or e.target in isotropic_indices]
        self.assertTrue(touching_isotropic)
        self.assertTrue(all(edge.state in (EDGE_REJECTED, EDGE_AMBIGUOUS) for edge in touching_isotropic))

    def test_pure_euclidean_proximity_alone_does_not_create_an_edge_state(self):
        scene, _, _, graph = _graph_for("isolated_floater")
        floater_index = scene.group_labels.index("floater")
        floater_edges = [e for e in graph.edges if e.source == floater_index or e.target == floater_index]
        self.assertTrue(floater_edges)
        self.assertTrue(all(edge.state == EDGE_PROXIMITY_ONLY for edge in floater_edges))


class GaussianSceneGraphTest(unittest.TestCase):
    def test_connected_wall_and_floor_split_into_separate_surface_regions(self):
        scene, _, _, graph = _graph_for("two_perpendicular_surfaces")
        same_surface = graph.same_surface_neighbors(scene.positions.shape[0])
        floor_indices = {i for i, label in enumerate(scene.group_labels) if label == "floor"}
        wall_indices = {i for i, label in enumerate(scene.group_labels) if label == "wall"}
        for index in floor_indices:
            self.assertTrue(same_surface[index] <= floor_indices, (index, same_surface[index]))
        for index in wall_indices:
            self.assertTrue(same_surface[index] <= wall_indices, (index, same_surface[index]))

    def test_oversized_gaussian_does_not_bridge_two_surfaces(self):
        scene, _, _, graph = _graph_for("oversized_bridge")
        same_surface = graph.same_surface_neighbors(scene.positions.shape[0])
        bridge_index = scene.group_labels.index("bridge")
        self.assertEqual(same_surface[bridge_index], set())
        floor_indices = {i for i, label in enumerate(scene.group_labels) if label == "floor"}
        wall_indices = {i for i, label in enumerate(scene.group_labels) if label == "wall"}
        for index in floor_indices:
            self.assertTrue(same_surface[index].isdisjoint(wall_indices))

    def test_unreliable_gaussian_removal_does_not_needlessly_fragment_a_reliable_region(self):
        scene, _, _, graph = _graph_for("isolated_floater")
        same_surface = graph.same_surface_neighbors(scene.positions.shape[0])
        plane_indices = [i for i, label in enumerate(scene.group_labels) if label == "plane"]
        # BFS connectivity check within the plane's own same_surface subgraph.
        visited = {plane_indices[0]}
        frontier = [plane_indices[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in same_surface[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(visited, set(plane_indices))

    def test_output_ordering_and_ids_are_deterministic(self):
        scene, frame, reliability, graph_first = _graph_for("two_perpendicular_surfaces")
        graph_second = build_manifold_affinity_graph(scene.positions, frame, reliability)
        self.assertEqual([e.payload() for e in graph_first.edges], [e.payload() for e in graph_second.edges])

    def test_node_boundary_status_is_experimental_diagnostic_only(self):
        scene, _, reliability, graph = _graph_for("plane")
        statuses = classify_node_boundary_status(scene.positions.shape[0], graph, reliability)
        self.assertEqual(len(statuses), scene.positions.shape[0])
        # Interior of a clean flat plane should mostly self-report as continuing interior.
        self.assertGreater(sum(1 for s in statuses if s == "interior_continuation"), scene.positions.shape[0] // 2)


if __name__ == "__main__":
    unittest.main()
