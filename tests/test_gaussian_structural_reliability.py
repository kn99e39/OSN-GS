from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_structural_reliability import (
    AMBIGUOUS,
    RELIABLE,
    REJECTED,
    evaluate_structural_reliability,
)


def _reliability_for(name: str):
    scene = make_gaussian_reliability_scene(name)
    frame = extract_covariance_frame(scene.covariances)
    result = evaluate_structural_reliability(scene.positions, frame)
    return scene, result


class GaussianStructuralReliabilityTest(unittest.TestCase):
    def test_clean_box_face_gaussians_are_reliable(self):
        _, result = _reliability_for("box_face")
        self.assertTrue(all(c == RELIABLE for c in result.reliability_class))

    def test_isolated_floater_is_ambiguous_or_rejected(self):
        scene, result = _reliability_for("box_isolated_floater")
        floater_index = scene.group_labels.index("floater")
        self.assertIn(result.reliability_class[floater_index], (AMBIGUOUS, REJECTED))
        self.assertTrue(len(result.reasons[floater_index]) > 0)

    def test_oversized_bridge_gaussian_is_rejected_or_low_confidence(self):
        scene, result = _reliability_for("box_with_bridge")
        bridge_index = scene.group_labels.index("bridge")
        self.assertIn(result.reliability_class[bridge_index], (AMBIGUOUS, REJECTED))

    def test_local_normal_disagreement_lowers_reliability(self):
        scene, result = _reliability_for("box_isotropic_contamination")
        isotropic_indices = [i for i, label in enumerate(scene.group_labels) if label == "isotropic"]
        for index in isotropic_indices:
            self.assertNotEqual(result.reliability_class[index], RELIABLE)

    def test_deterministic_across_repeated_calls(self):
        scene, first = _reliability_for("box_face")
        frame = extract_covariance_frame(scene.covariances)
        second = evaluate_structural_reliability(scene.positions, frame)
        self.assertEqual(first.reliability_class, second.reliability_class)
        torch.testing.assert_close(first.planarity_score, second.planarity_score)

    def test_score_components_and_reasons_are_preserved_in_provenance(self):
        _, result = _reliability_for("box_isolated_floater")
        payload = result.payload()
        row = payload[-1]  # the floater
        for key in (
            "planarity_score", "neighbor_normal_agreement", "mutual_tangent_residual",
            "scale_consistency", "local_support_score", "final_reliability_class", "reasons",
        ):
            self.assertIn(key, row)
        self.assertIsInstance(row["reasons"], list)
        self.assertTrue(len(row["reasons"]) > 0)


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the CUDA device-index regression")
    def test_cuda_contextual_masks_follow_neighbor_index_device(self):
        scene = make_gaussian_reliability_scene("box_face")
        positions = scene.positions.to("cuda")
        frame = extract_covariance_frame(scene.covariances.to("cuda"))
        result = evaluate_structural_reliability(positions, frame)
        self.assertEqual(len(result.reliability_class), len(scene.positions))
        self.assertTrue(all(value == RELIABLE for value in result.reliability_class))


if __name__ == "__main__":
    unittest.main()
