from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions


def _signature(positions, covariances, ids):
    frame = extract_covariance_frame(covariances)
    reliability = evaluate_structural_reliability(positions, frame)
    graph = build_manifold_affinity_graph(positions, frame, reliability, ids=ids)
    result = form_surface_regions(positions, frame, reliability, graph, ids=ids)
    return sorted((tuple(sorted(r.member_ids)), tuple(sorted(r.core_member_ids)), r.region_state) for r in result.regions)


class SurfaceRegionInvarianceTest(unittest.TestCase):
    def test_region_membership_is_stable_under_rigid_scale_and_order_changes(self):
        scene = make_gaussian_reliability_scene("two_perpendicular_surfaces")
        ids = list(range(scene.positions.shape[0]))
        baseline = _signature(scene.positions, scene.covariances, ids)
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transformed_positions = (scene.positions @ rotation.T) * 2.5 + torch.tensor([5.0, -3.0, 2.0])
        transformed_covariances = rotation @ (scene.covariances * (2.5 ** 2)) @ rotation.T
        self.assertEqual(_signature(transformed_positions, transformed_covariances, ids), baseline)
        perm = torch.randperm(scene.positions.shape[0], generator=torch.Generator().manual_seed(7))
        shuffled_ids = [ids[index] for index in perm.tolist()]
        self.assertEqual(_signature(scene.positions[perm], scene.covariances[perm], shuffled_ids), baseline)


if __name__ == "__main__":
    unittest.main()
