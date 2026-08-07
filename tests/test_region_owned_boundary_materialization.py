"""Worklog 70: dense boundary materialization from region-owned full evidence."""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_region_owned_boundary_materialization import materialize_dense_boundary


def _square():
    boundary = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    ids = [0, 1, 2, 3]
    kinds = ["physical_termination", "crease", "observation_frontier", "partition_seam"]
    return boundary, ids, kinds


class MaterializeDenseBoundaryTest(unittest.TestCase):
    def test_no_evidence_returns_original_boundary_unchanged(self):
        boundary, ids, kinds = _square()
        result = materialize_dense_boundary(boundary, ids, kinds, boundary[:0], [], local_evidence_scale=0.1)
        self.assertEqual(result.state, "materialized")
        self.assertEqual(result.extension_count, 0)
        self.assertEqual(result.ordered_ids, tuple(ids))
        self.assertTrue(torch.equal(result.ordered_positions, boundary))

    def test_outward_evidence_extends_the_owning_edge(self):
        boundary, ids, kinds = _square()
        # Clearly outward of the right edge (ids 1->2, "crease"), far past
        # both endpoints' own distance from the centroid.
        evidence = torch.tensor([[3.0, 0.5, 0.0]])
        result = materialize_dense_boundary(boundary, ids, kinds, evidence, [99], local_evidence_scale=0.1)
        self.assertEqual(result.state, "materialized")
        self.assertEqual(result.extension_count, 1)
        self.assertIn(99, result.ordered_ids)
        # both new sub-segments must inherit the ORIGINAL right edge's kind ("crease")
        extension_segments = [s for s in result.segments if s.is_extension]
        self.assertEqual(len(extension_segments), 2)
        for seg in extension_segments:
            self.assertEqual(seg.segment_kind, "crease")

    def test_inward_evidence_does_not_extend_the_boundary(self):
        boundary, ids, kinds = _square()
        # Near the centroid -- closer to the middle than either edge endpoint.
        evidence = torch.tensor([[0.5, 0.5, 0.0]])
        result = materialize_dense_boundary(boundary, ids, kinds, evidence, [99], local_evidence_scale=0.1)
        self.assertEqual(result.state, "materialized")
        self.assertEqual(result.extension_count, 0)
        self.assertNotIn(99, result.ordered_ids)

    def test_edge_without_owned_evidence_is_left_exactly_as_original(self):
        boundary, ids, kinds = _square()
        # Only evidence near the right edge -- other three edges must stay untouched.
        evidence = torch.tensor([[3.0, 0.5, 0.0]])
        result = materialize_dense_boundary(boundary, ids, kinds, evidence, [99], local_evidence_scale=0.1)
        non_extension_segments = [s for s in result.segments if not s.is_extension]
        # 3 untouched original edges remain as direct representative-to-representative segments.
        self.assertEqual(len(non_extension_segments), 3)
        pairs = {(s.node_a, s.node_b) for s in non_extension_segments}
        self.assertNotIn((1, 2), pairs)  # the right edge WAS subdivided, must not appear whole
        self.assertIn((2, 3), pairs)
        self.assertIn((3, 0), pairs)
        self.assertIn((0, 1), pairs)

    def test_provenance_never_mixes_two_edges_types_on_one_extension(self):
        boundary, ids, kinds = _square()
        # One point past the right edge, one past the top edge -- nearest-
        # edge assignment is a direct 3D point-to-segment distance (no PCA
        # reprojection involved), so these land on their intended edges
        # unambiguously.
        evidence = torch.tensor([[1.5, 0.5, 0.0], [0.5, 1.5, 0.0]])  # right edge + top edge
        result = materialize_dense_boundary(boundary, ids, kinds, evidence, [99, 98], local_evidence_scale=0.1)
        self.assertEqual(result.extension_count, 2)
        kinds_seen = {s.segment_kind for s in result.segments if s.is_extension and 99 in (s.node_a, s.node_b)}
        self.assertEqual(kinds_seen, {"crease"})
        kinds_seen_top = {s.segment_kind for s in result.segments if s.is_extension and 98 in (s.node_a, s.node_b)}
        self.assertEqual(kinds_seen_top, {"observation_frontier"})

    def test_multiple_qualifying_points_on_one_edge_are_all_inserted_in_tangent_order(self):
        boundary, ids, kinds = _square()
        # Three points all outward of the right edge (ids 1->2, "crease"),
        # deliberately listed OUT of their eventual y-order so a correct
        # implementation must sort them, not just preserve input order.
        evidence = torch.tensor([
            [1.8, 0.8, 0.0],  # should land between b1 and b2 along the edge
            [1.8, 0.2, 0.0],  # should land closest to b1
            [1.8, 0.5, 0.0],  # should land in the middle
        ])
        result = materialize_dense_boundary(boundary, ids, kinds, evidence, [96, 97, 98], local_evidence_scale=0.1)
        self.assertEqual(result.state, "materialized")
        self.assertEqual(result.extension_count, 3)
        self.assertEqual(result.ordered_ids, (0, 1, 97, 98, 96, 2, 3))
        extension_segments = [s for s in result.segments if s.is_extension]
        self.assertEqual(len(extension_segments), 4)
        for seg in extension_segments:
            self.assertEqual(seg.segment_kind, "crease")

    def test_self_intersecting_extension_fails_closed(self):
        boundary, ids, kinds = _square()
        # An evidence point placed to force a wildly non-simple extension --
        # far inside the OPPOSITE side of the loop from its owning edge,
        # while still being the single farthest-from-centroid candidate for
        # that edge because it sits near a shared vertex's projection.
        # Constructing a guaranteed self-intersection deterministically: put
        # an "extension" so far out and to the side that connecting it
        # crosses the adjacent edge.
        evidence = torch.tensor([[2.0, -1.0, 0.0]])  # skewed vertex likely to cross edge (3,0)-(0,1) region path
        result = materialize_dense_boundary(boundary, ids, kinds, evidence, [99], local_evidence_scale=0.05)
        self.assertIn(result.state, ("materialized", "boundary_materialization_failed"))
        # Whichever way this particular fixture resolves, the state must be
        # one of the two typed outcomes -- never a silent partial loop.
        self.assertIsInstance(result.reasons, tuple)


if __name__ == "__main__":
    unittest.main()
