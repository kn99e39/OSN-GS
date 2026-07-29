from __future__ import annotations

"""Regression tests for the isolated consensus-aware region foundation.

These tests deliberately consume only the Worklog 113/115 covariance,
reliability, and affinity contract.  They do not create boundaries, NURBS
patches, or production-facing objects.
"""

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    make_anisotropic_planar_bridge_scene,
    make_curvature_sweep_scene,
    make_density_variation_scene,
    make_gap_sweep_scene,
    make_gaussian_reliability_scene,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    MEMBER_REJECTED,
    REGION_CORE,
    REGION_SMALL_REVIEW,
    REGION_STABLE,
    form_surface_regions,
)


def _form(scene, *, ids=None):
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(scene.positions, frame, reliability, ids=ids)
    return form_surface_regions(scene.positions, frame, reliability, graph, ids=ids)


def _member_sets(result):
    return sorted(tuple(sorted(region.member_ids)) for region in result.regions)


class CoreAndBridgeRegressionTest(unittest.TestCase):
    def test_clean_plane_is_one_stable_core_region(self):
        result = _form(make_gaussian_reliability_scene("plane"))
        self.assertEqual(len(result.regions), 1)
        self.assertEqual(len(result.regions[0].member_ids), 81)
        self.assertIn(result.regions[0].region_state, (REGION_CORE, REGION_STABLE))

    def test_an_isolated_same_surface_pair_cannot_seed_a_core_region(self):
        scene = make_gaussian_reliability_scene("plane")
        two_node_scene = type(scene)(
            "two_node_same_surface_pair", scene.positions[:2], scene.covariances[:2], "isolated pair",
        )
        result = _form(two_node_scene)
        self.assertEqual(result.regions, ())

    def test_rejected_nodes_are_not_members_of_any_region(self):
        scene = make_gaussian_reliability_scene("isotropic_blob")
        result = _form(scene)
        rejected = [i for i, label in enumerate(scene.group_labels) if label == "isotropic"]
        self.assertTrue(rejected)
        for index in rejected:
            self.assertEqual(result.node_membership_state[index], MEMBER_REJECTED)
            self.assertEqual(result.node_region_id[index], -1)

    def test_gap_point_zero_two_does_not_false_merge_parallel_sheets(self):
        scene = make_gap_sweep_scene(0.02)
        result = _form(scene)
        groups = []
        for region in result.regions:
            labels = {scene.group_labels[i] for i in region.member_ids}
            groups.append(labels)
        self.assertTrue(groups)
        self.assertTrue(all(len(labels) == 1 for labels in groups))
        self.assertGreaterEqual(len(groups), 2)

    def test_perpendicular_surfaces_and_oversized_bridge_do_not_merge(self):
        for scene in (
            make_gaussian_reliability_scene("two_perpendicular_surfaces"),
            make_anisotropic_planar_bridge_scene(),
        ):
            result = _form(scene)
            for region in result.regions:
                labels = {scene.group_labels[i] for i in region.member_ids}
                self.assertFalse({"floor", "wall"}.issubset(labels), scene.name)


class SmoothnessAndDeterminismRegressionTest(unittest.TestCase):
    def test_smooth_curvature_and_density_controls_remain_unfragmented(self):
        scenes = (
            make_curvature_sweep_scene(0.05),
            make_density_variation_scene("gradual_gradient"),
            make_density_variation_scene("sparse_but_continuous"),
        )
        for scene in scenes:
            result = _form(scene)
            self.assertEqual(len(result.regions), 1, scene.name)
            self.assertNotEqual(result.regions[0].region_state, REGION_SMALL_REVIEW, scene.name)

    def test_stable_ids_make_membership_input_order_deterministic(self):
        scene = make_gaussian_reliability_scene("two_perpendicular_surfaces")
        ids = list(range(scene.positions.shape[0]))
        baseline = _member_sets(_form(scene, ids=ids))
        for seed in range(3):
            permutation = torch.randperm(scene.positions.shape[0], generator=torch.Generator().manual_seed(seed))
            shuffled = type(scene)(
                scene.name,
                scene.positions[permutation],
                scene.covariances[permutation],
                scene.description,
                tuple(scene.group_labels[i] for i in permutation.tolist()),
            )
            shuffled_ids = [ids[i] for i in permutation.tolist()]
            self.assertEqual(_member_sets(_form(shuffled, ids=shuffled_ids)), baseline, seed)


if __name__ == "__main__":
    unittest.main()
