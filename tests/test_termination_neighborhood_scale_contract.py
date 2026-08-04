from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_termination_neighborhood_scale import (
    resolve_termination_neighborhood_scale,
)


class TerminationNeighborhoodScaleContractTest(unittest.TestCase):
    def test_explicit_candidate_scale_is_preserved_by_identity(self):
        candidate_scale = torch.tensor([0.4, 0.8, 1.6])
        footprint_like_scale = torch.tensor([0.02, 0.03, 0.04])
        resolved = resolve_termination_neighborhood_scale(
            candidate_scale=candidate_scale,
            tangent_major_scale=footprint_like_scale,
        )
        self.assertIs(resolved, candidate_scale)
        self.assertFalse(torch.equal(resolved, footprint_like_scale))

    def test_legacy_default_matches_affinity_graph_default(self):
        tangent_major_scale = torch.tensor([0.4, 0.8, 1.6])
        resolved = resolve_termination_neighborhood_scale(
            candidate_scale=None,
            tangent_major_scale=tangent_major_scale,
        )
        self.assertIs(resolved, tangent_major_scale)


if __name__ == '__main__':
    unittest.main()
