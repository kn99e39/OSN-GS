from __future__ import annotations

import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions
from osn_gs.surface.torch_world_space_boundary_halfedges import extract_world_space_boundary_halfedge_candidates


class WorldSpaceBoundaryHalfedgeTest(unittest.TestCase):
    def test_perpendicular_surfaces_expose_crease_candidates_without_boundary_loop(self):
        scene = make_gaussian_reliability_scene("box")
        frame = extract_covariance_frame(scene.covariances)
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
        regions = form_surface_regions(scene.positions, frame, reliability, graph)
        candidates = extract_world_space_boundary_halfedge_candidates(scene.positions, frame.normal_candidate, regions, graph)
        self.assertTrue(any(item.boundary_reason == "crease_discontinuity" for item in candidates))
        self.assertTrue(all("not_ordered_boundary" in item.review_reasons[0] for item in candidates))


if __name__ == "__main__":
    unittest.main()
