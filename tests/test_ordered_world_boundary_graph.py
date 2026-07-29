from __future__ import annotations

import unittest

from osn_gs.surface.torch_ordered_world_boundary_graph import build_boundary_compatibility, recover_ordered_boundary_components
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _candidate(index: int, reason: str = "observed_support_termination"):
    return WorldSpaceBoundaryHalfEdgeCandidate(f"h{index}", 0, index, None, (float(index) * 0.1, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), reason, None, 0.8, "locally_chainable", ())


class OrderedBoundaryGraphTest(unittest.TestCase):
    def test_open_chain_is_not_forced_closed(self):
        candidates = tuple(_candidate(i) for i in range(3))
        components = recover_ordered_boundary_components(candidates, build_boundary_compatibility(candidates))
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].ordering_state, "ordered_open_chain")
        self.assertFalse(components[0].closed)


if __name__ == "__main__":
    unittest.main()
