"""Worklog 61: region-local parametric chart boundary construction.

Tests the leftmost-turn planar-graph boundary trace directly (a non-convex
shape with an interior chord must never be cut through), the segment-kind
classification (physical_termination > crease > observation_frontier >
partition_seam precedence), and typed rejection for insufficient/open/
self-intersecting topology -- independent of, and without touching, the
`eligible_closed_boundary` physical-termination contract.
"""

from __future__ import annotations

import unittest

from osn_gs.surface.torch_region_parametric_chart_boundary import (
    SEGMENT_CREASE,
    SEGMENT_OBSERVATION_FRONTIER,
    SEGMENT_PARTITION_SEAM,
    SEGMENT_PHYSICAL_TERMINATION,
    STATUS_INSUFFICIENT_TOPOLOGY,
    STATUS_TOPOLOGY_OPEN_OR_BRANCHING,
    _segment_kind_for_endpoints,
    _trace_leftmost_turn_boundary,
)


class LeftmostTurnTraceTest(unittest.TestCase):
    def test_concave_l_shape_with_interior_chord_traces_the_outer_face_only(self):
        # Hexagonal L-shape plus an interior chord A-D across the concave
        # corner -- a convex-hull-based method would cut straight across;
        # this trace must follow the real (concave) outer boundary only.
        uv = {
            "A": (0.0, 0.0), "B": (2.0, 0.0), "C": (2.0, 1.0),
            "D": (1.0, 1.0), "E": (1.0, 2.0), "F": (0.0, 2.0),
        }
        adjacency = {
            "A": {"B", "F", "D"}, "B": {"A", "C"}, "C": {"B", "D"},
            "D": {"C", "E", "A"}, "E": {"D", "F"}, "F": {"E", "A"},
        }
        result = _trace_leftmost_turn_boundary(uv, adjacency)
        self.assertIsNotNone(result)
        self.assertEqual(set(result), set(uv))
        self.assertEqual(len(result), 6)
        # The interior chord A-D must never appear as a consecutive pair.
        pairs = {frozenset((result[i], result[(i + 1) % 6])) for i in range(6)}
        self.assertNotIn(frozenset({"A", "D"}), pairs)

    def test_triangle_closes(self):
        uv = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.5, 1.0)}
        adjacency = {"A": {"B", "C"}, "B": {"A", "C"}, "C": {"A", "B"}}
        result = _trace_leftmost_turn_boundary(uv, adjacency)
        self.assertEqual(set(result), {"A", "B", "C"})

    def test_open_chain_never_closes(self):
        uv = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}
        adjacency = {"A": {"B"}, "B": {"A", "C"}, "C": {"B"}}
        self.assertIsNone(_trace_leftmost_turn_boundary(uv, adjacency))

    def test_fewer_than_three_nodes_returns_none(self):
        uv = {"A": (0.0, 0.0), "B": (1.0, 0.0)}
        adjacency = {"A": {"B"}, "B": {"A"}}
        self.assertIsNone(_trace_leftmost_turn_boundary(uv, adjacency))


class SegmentKindClassificationTest(unittest.TestCase):
    def test_physical_termination_wins_over_frontier(self):
        reason_by_node = {"A": ("reliability_frontier",), "B": ("observed_support_termination",)}
        self.assertEqual(_segment_kind_for_endpoints("A", "B", reason_by_node), SEGMENT_PHYSICAL_TERMINATION)

    def test_crease_wins_over_frontier(self):
        reason_by_node = {"A": ("unresolved_sampling_gap",), "B": ("crease_discontinuity",)}
        self.assertEqual(_segment_kind_for_endpoints("A", "B", reason_by_node), SEGMENT_CREASE)

    def test_frontier_when_only_soft_evidence_present(self):
        reason_by_node = {"A": ("parallel_sheet_conflict",), "B": ()}
        self.assertEqual(_segment_kind_for_endpoints("A", "B", reason_by_node), SEGMENT_OBSERVATION_FRONTIER)

    def test_partition_seam_when_no_typed_evidence_at_either_endpoint(self):
        reason_by_node: dict = {}
        self.assertEqual(_segment_kind_for_endpoints("A", "B", reason_by_node), SEGMENT_PARTITION_SEAM)


if __name__ == "__main__":
    unittest.main()
