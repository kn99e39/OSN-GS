"""Full-region face topology followed by chart-unit membership incidence."""

import unittest

import torch

from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import (
    ROLE_INNER_BOUNDARY,
    ROLE_OUTER_BOUNDARY,
    SEGMENT_PARTITION_SEAM,
    STATE_INNER_BOUNDARY_REVIEW_REQUIRED,
    STATE_UNTYPED_PHYSICAL_BOUNDARY,
    ChartUnitFaceTopologyContext,
    materialize_chart_unit_face_incidence_boundaries,
)
from osn_gs.surface.torch_full_region_surface_face_topology import (
    build_full_region_surface_face_topology,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame


def _flat_covariance(count: int) -> torch.Tensor:
    covariance = torch.zeros((count, 3, 3), dtype=torch.float32)
    covariance[:, 0, 0] = 0.02
    covariance[:, 1, 1] = 0.02
    covariance[:, 2, 2] = 0.0001
    return covariance


def _adjacency(count: int, triangles: list[tuple[int, int, int]]) -> tuple[frozenset[int], ...]:
    rows = [set() for _ in range(count)]
    for triangle in triangles:
        for offset, a in enumerate(triangle):
            for b in triangle[offset + 1 :]:
                rows[a].add(b)
                rows[b].add(a)
    return tuple(frozenset(row) for row in rows)


def _context(
    positions: torch.Tensor,
    triangles: list[tuple[int, int, int]],
    physical_candidate_ids=(),
) -> ChartUnitFaceTopologyContext:
    covariance = _flat_covariance(len(positions))
    stable_ids = tuple(range(len(positions)))
    adjacency = _adjacency(len(positions), triangles)
    faces = build_full_region_surface_face_topology(positions, covariance, stable_ids, adjacency)
    return ChartUnitFaceTopologyContext(
        positions=positions,
        covariance=covariance,
        stable_ids=stable_ids,
        normals=extract_covariance_frame(covariance).normal_candidate,
        arc_side=None,
        same_surface_adjacency=adjacency,
        surface_faces=faces,
        full_region_physical_candidate_ids=frozenset(physical_candidate_ids),
        full_evidence_spacing=1.0,
    )


class FullRegionLocalRotationTest(unittest.TestCase):
    def test_recovers_faces_before_membership_from_actual_local_neighbors(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        covariance = _flat_covariance(4)
        adjacency = _adjacency(4, [(0, 1, 2), (0, 2, 3)])
        topology = build_full_region_surface_face_topology(
            positions, covariance, [40, 10, 30, 20], adjacency,
        )
        self.assertEqual(
            {frozenset(face.ordered_region_indices) for face in topology.observed_faces},
            {frozenset((0, 1, 2)), frozenset((0, 2, 3))},
        )
        self.assertEqual(len(topology.face_incidence_by_edge[(0, 2)]), 2)
        self.assertEqual(len(topology.face_incidence_by_edge[(0, 1)]), 1)
        self.assertFalse(topology.invalid_topology_nodes)

    def test_rigidly_rotated_local_frames_preserve_face_incidence(self):
        base = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        angle = torch.tensor(0.71)
        rotation = torch.tensor(
            [
                [torch.cos(angle), 0.0, torch.sin(angle)],
                [0.0, 1.0, 0.0],
                [-torch.sin(angle), 0.0, torch.cos(angle)],
            ]
        )
        covariance = _flat_covariance(4)
        rotated_positions = base @ rotation.T
        rotated_covariance = rotation[None] @ covariance @ rotation.T[None]
        adjacency = _adjacency(4, [(0, 1, 2), (0, 2, 3)])
        topology = build_full_region_surface_face_topology(
            rotated_positions, rotated_covariance, range(4), adjacency,
        )
        self.assertEqual(set(topology.face_incidence_by_edge), {(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)})
        self.assertEqual(len(topology.face_incidence_by_edge[(0, 2)]), 2)


class MembershipFaceIncidenceTest(unittest.TestCase):
    def test_continuous_full_surface_cut_is_partition_seam(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        # Candidate 1 types only the two exterior edges of unit face 0-1-2;
        # the diagonal 2-0 has two full-region incident faces and is a seam.
        context = _context(positions, [(0, 1, 2), (0, 2, 3)], physical_candidate_ids=(1,))
        result = materialize_chart_unit_face_incidence_boundaries(context, [0, 1, 2])
        self.assertEqual(result.admitted_boundary_candidate_count, 1)
        self.assertEqual(result.membership_crossing_edge_count, 1)
        self.assertEqual(len(result.boundary_loops), 1)
        self.assertEqual(result.boundary_loops[0].role, ROLE_OUTER_BOUNDARY)
        self.assertEqual(
            sum(segment.segment_kind == SEGMENT_PARTITION_SEAM for segment in result.boundary_loops[0].segments),
            1,
        )

    def test_zero_candidate_unit_is_evaluated_without_candidate_anchor(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        context = _context(positions, [(0, 1, 2), (0, 2, 3)])
        result = materialize_chart_unit_face_incidence_boundaries(context, [0, 1, 2])
        self.assertEqual(result.admitted_boundary_candidate_count, 0)
        # It reaches face incidence. The two true exterior edges are not
        # invented as seams, so the chart then fails closed on provenance.
        self.assertEqual(result.unit_supported_face_count, 1)
        self.assertTrue(any(STATE_UNTYPED_PHYSICAL_BOUNDARY in reason for reason in result.unresolved_reasons))

    def test_all_independent_boundary_loops_are_preserved(self):
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0],
            ]
        )
        context = _context(
            positions, [(0, 1, 2), (3, 4, 5)], physical_candidate_ids=range(6),
        )
        result = materialize_chart_unit_face_incidence_boundaries(context, range(6))
        self.assertEqual(len(result.boundary_loops), 2)
        self.assertTrue(all(loop.role == ROLE_OUTER_BOUNDARY for loop in result.boundary_loops))

    def test_inner_loop_is_preserved_and_untrimmed_domain_fails_closed(self):
        positions = torch.tensor(
            [
                [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0],
                [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        triangles = [
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
            (8, 4, 5), (8, 5, 6), (8, 6, 7), (8, 7, 4),
        ]
        context = _context(positions, triangles, physical_candidate_ids=range(8))
        result = materialize_chart_unit_face_incidence_boundaries(context, range(8))
        self.assertEqual({loop.role for loop in result.boundary_loops}, {ROLE_OUTER_BOUNDARY, ROLE_INNER_BOUNDARY})
        self.assertIn(STATE_INNER_BOUNDARY_REVIEW_REQUIRED, result.unresolved_reasons)
        self.assertFalse(result.materialized)

    def test_every_boundary_segment_is_an_actual_same_surface_edge(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        context = _context(positions, [(0, 1, 2), (0, 2, 3)], physical_candidate_ids=(1,))
        result = materialize_chart_unit_face_incidence_boundaries(context, [0, 1, 2])
        for loop in result.boundary_loops:
            for segment in loop.segments:
                self.assertIn(segment.stable_id_b, context.same_surface_adjacency[segment.stable_id_a])


if __name__ == "__main__":
    unittest.main()
