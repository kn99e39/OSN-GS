"""Worklog 39: accepted-core-pair vs boundary-adjacency semantic separation.

`internal_accepted_edge_ids` is REGION-topology evidence built from the
bounded-kNN affinity graph -- it answers "are these two Gaussians linked in
the region's connectivity", not "are these two boundary candidates
consecutive along the perimeter". Requiring a DIRECT accepted edge between
boundary candidates therefore rejected pairs that are genuinely adjacent on
the physical perimeter but lost their direct affinity edge to bounded-k
sparsity.

Measured (worklog 39): every box face loses 5-11 perimeter-adjacent pairs to
that gate, leaving 10-17 compatible edges where an N-node ring needs N, so no
face closes. All 45 such rejected pairs across the six faces are reachable by
a 2-hop path in the region's own accepted graph. box_face, which does close,
loses zero adjacent pairs.

The gate is widened to "direct accepted edge OR a 2-hop path through a shared
NON-candidate interior node". The non-candidate requirement is load-bearing:
allowing any shared neighbour lets a Y-junction stub splice itself into a ring.
"""

from __future__ import annotations

import math
import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_directed_boundary_ordering import (
    _build_accepted_adjacency,
    _has_region_topology_support,
    recover_directed_boundary_components,
)
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _construct(scene_name: str):
    scene = make_gaussian_reliability_scene(scene_name, seed=0)
    ids = tuple(range(scene.positions.shape[0]))
    result = construct_visible_nurbs_from_gaussians(
        scene.positions, covariance=scene.covariances, stable_ids=ids,
    )
    return scene, ids, result


def _closed_loops(result):
    return [c for c in result.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]


class RegionTopologySupportContractTest(unittest.TestCase):
    def test_direct_accepted_edge_is_supported(self):
        accepted_pairs = {frozenset((1, 2))}
        adjacency = _build_accepted_adjacency([(1, 2)])
        self.assertTrue(_has_region_topology_support(1, 2, accepted_pairs, adjacency, frozenset()))

    def test_two_hop_via_non_candidate_interior_node_is_supported(self):
        # 1 -- 9 -- 2, where 9 is an interior (non-candidate) node.
        accepted_pairs = {frozenset((1, 9)), frozenset((9, 2))}
        adjacency = _build_accepted_adjacency([(1, 9), (9, 2)])
        candidates = frozenset({1, 2})
        self.assertTrue(_has_region_topology_support(1, 2, accepted_pairs, adjacency, candidates))

    def test_two_hop_via_another_candidate_is_rejected(self):
        # 1 -- 3 -- 2 where 3 is ITSELF a boundary candidate: this is the
        # Y-junction/branch route, which must not create adjacency.
        accepted_pairs = {frozenset((1, 3)), frozenset((3, 2))}
        adjacency = _build_accepted_adjacency([(1, 3), (3, 2)])
        candidates = frozenset({1, 2, 3})
        self.assertFalse(_has_region_topology_support(1, 2, accepted_pairs, adjacency, candidates))

    def test_no_path_is_rejected(self):
        accepted_pairs = {frozenset((1, 9))}
        adjacency = _build_accepted_adjacency([(1, 9)])
        self.assertFalse(_has_region_topology_support(1, 2, accepted_pairs, adjacency, frozenset()))

    def test_three_hop_is_not_enough(self):
        accepted_pairs = {frozenset((1, 8)), frozenset((8, 9)), frozenset((9, 2))}
        adjacency = _build_accepted_adjacency([(1, 8), (8, 9), (9, 2)])
        self.assertFalse(_has_region_topology_support(1, 2, accepted_pairs, adjacency, frozenset()))


class YJunctionStillRejectedTest(unittest.TestCase):
    """The widened gate must not let a branch stub into a boundary loop."""

    def test_interior_stub_never_enters_a_closed_loop(self):
        n = 12
        candidates = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            tangent = (-math.sin(angle), math.cos(angle), 0.0)
            candidates.append(WorldSpaceBoundaryHalfEdgeCandidate(
                half_edge_id=f"h{i}", source_region_id=0, source_gaussian_id=i, adjacent_gaussian_id=None,
                world_position=(math.cos(angle), math.sin(angle), 0.0), local_normal=(0.0, 0.0, 1.0),
                local_tangent_direction=tangent, boundary_direction=tangent,
                boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
                ordering_state="locally_chainable", review_reasons=(),
            ))
        accepted = [(i, (i + 1) % n) for i in range(n)]
        stub_angle = 2 * math.pi * 0.5 / n
        stub = WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id="stub", source_region_id=0, source_gaussian_id=999, adjacent_gaussian_id=None,
            world_position=(math.cos(stub_angle) * 0.6, math.sin(stub_angle) * 0.6, 0.0),
            local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=(-math.sin(stub_angle), math.cos(stub_angle), 0.0),
            boundary_direction=(-math.sin(stub_angle), math.cos(stub_angle), 0.0),
            boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
            ordering_state="locally_chainable", review_reasons=(),
        )
        _, components = recover_directed_boundary_components(candidates + [stub], accepted + [(0, 999)])
        for component in components:
            if component.ordering_state == "ordered_closed_loop":
                self.assertNotIn(999, component.ordered_source_ids)


class AnalyticBoundaryRecoveryTest(unittest.TestCase):
    def test_cylinder_recovers_side_plus_two_caps(self):
        _scene, _ids, result = _construct("cylinder")
        self.assertEqual(len(_closed_loops(result)), 3)

    def test_box_faces_recover_closed_loops(self):
        _scene, _ids, result = _construct("box")
        self.assertGreaterEqual(len(_closed_loops(result)), 5)

    def test_box_face_still_closes(self):
        _scene, _ids, result = _construct("box_face")
        self.assertEqual(len(_closed_loops(result)), 1)

    def test_sphere_never_manufactures_a_closed_boundary(self):
        _scene, _ids, result = _construct("sphere")
        self.assertEqual(len(_closed_loops(result)), 0)


class NoFalseCrossSurfaceConnectionTest(unittest.TestCase):
    def _assert_loops_are_planar_and_single_face(self, scene, result, max_thickness=0.05):
        for component in _closed_loops(result):
            points = scene.positions[list(component.ordered_source_ids)]
            extent = points.max(dim=0).values - points.min(dim=0).values
            self.assertLess(
                float(extent.min()), max_thickness,
                "a closed loop must stay on a single planar face",
            )

    def test_box_loops_stay_on_one_face_each(self):
        scene, _ids, result = _construct("box")
        self._assert_loops_are_planar_and_single_face(scene, result)

    def test_bridge_contamination_loops_stay_on_one_face_each(self):
        scene, _ids, result = _construct("box_with_bridge")
        self._assert_loops_are_planar_and_single_face(scene, result)

    def test_thin_slab_loops_do_not_span_both_sides(self):
        scene, _ids, result = _construct("thin_slab")
        loops = _closed_loops(result)
        self.assertEqual(len(loops), 2)
        for component in loops:
            zs = [float(scene.positions[i][2]) for i in component.ordered_source_ids]
            self.assertTrue(all(z > 0 for z in zs) or all(z < 0 for z in zs))

    def test_every_recovered_loop_is_a_simple_polygon(self):
        for scene_name in ("box", "cylinder", "box_face", "thin_slab"):
            scene, ids, result = _construct(scene_name)
            positions = {sid: tuple(scene.positions[i].tolist()) for i, sid in enumerate(ids)}
            for component in _closed_loops(result):
                report = validate_simple_closed_loop([positions[s] for s in component.ordered_source_ids])
                self.assertTrue(report.is_simple_polygon, (scene_name, report.reasons))


if __name__ == "__main__":
    unittest.main()
