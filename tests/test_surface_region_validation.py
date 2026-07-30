from __future__ import annotations

import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions
from osn_gs.surface.torch_surface_region_validation import (
    READINESS_READY,
    READINESS_REVIEW,
    diagnose_boundary_input_readiness,
)


class BoundaryInputReadinessTest(unittest.TestCase):
    def test_clean_plane_exposes_readiness_without_constructing_a_boundary(self):
        scene = make_gaussian_reliability_scene("box_face")
        frame = extract_covariance_frame(scene.covariances)
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
        regions = form_surface_regions(scene.positions, frame, reliability, graph)
        reports = diagnose_boundary_input_readiness(regions, graph)
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].stable_core_present)
        self.assertIn(reports[0].boundary_extraction_readiness, (READINESS_READY, READINESS_REVIEW))


if __name__ == "__main__":
    unittest.main()
