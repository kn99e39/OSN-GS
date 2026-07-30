from __future__ import annotations

import unittest

from nurbs_constructor_benchmark.surface_region_adversarial_scenes import (
    make_cylinder_phase_alias_scene,
    make_genuine_narrow_connection_scene,
)
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
    def test_auto_policy_excludes_broad_cylinder_ring_shortcuts_from_accepted_topology(self):
        # A cylinder's side wall is periodic: opposite-side points have
        # anti-parallel (hence abs(dot)-aligned) normals, a genuine
        # volumetric analogue of the earlier ad hoc sine-sheet phase-alias
        # fixture. A wide candidate radius surfaces these as long-range
        # same_surface candidates that must never enter accepted topology.
        scene = make_cylinder_phase_alias_scene()
        graph, result = _form(scene, broad=True)
        shortcuts = {
            tuple(sorted((edge.source, edge.target)))
            for edge in graph.edges
            if edge.manifold_relation == "same_surface" and edge.metrics.normalized_distance > 4.0
        }
        accepted = {tuple(sorted(edge)) for region in result.regions for edge in region.internal_accepted_edge_ids}
        self.assertTrue(shortcuts)
        self.assertFalse(shortcuts & accepted)
        # No region may ever mix "side" wall members with either end cap --
        # that would mean a phase-alias shortcut (or the circular crease)
        # was used as real same-surface evidence. The wide "broad" candidate
        # radius used here is itself allowed to split the side wall into
        # several conservative pieces; only cross-face mixing is a failure.
        labels = scene.group_labels
        for region in result.regions:
            member_labels = {labels[index] for index in region.member_ids}
            self.assertLessEqual(len(member_labels), 1, member_labels)

    def test_genuine_multi_edge_neck_remains_one_region(self):
        _, result = _form(make_genuine_narrow_connection_scene())
        self.assertEqual(len(result.regions), 1)


if __name__ == "__main__":
    unittest.main()
