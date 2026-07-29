from __future__ import annotations

"""Worklog 117 adversarial regression: still no boundary construction."""

import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gap_sweep_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions


def _form(scene):
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
    return graph, form_surface_regions(scene.positions, frame, reliability, graph)


class GapSweepAdversarialTest(unittest.TestCase):
    def test_gap_point_zero_two_and_larger_never_create_a_mixed_region(self):
        # Extremely tiny gaps may remain evidence-insufficient, but the
        # threshold-sensitive 0.02 case that previously had 19 false pairwise
        # same_surface edges must not become a giant mixed region.
        for gap in (0.02, 0.025, 0.03, 0.05, 0.1):
            scene = make_gap_sweep_scene(gap)
            _, result = _form(scene)
            for region in result.regions:
                labels = {scene.group_labels[index] for index in region.member_ids}
                self.assertLessEqual(len(labels), 1, gap)


if __name__ == "__main__":
    unittest.main()
