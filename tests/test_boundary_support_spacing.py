"""Worklog 76: explicit boundary_support_spacing contract."""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_boundary_support_spacing import (
    SPACING_MODE_FULL_EVIDENCE,
    SPACING_MODE_LOCAL_BOUNDARY_SUPPORT,
    SPACING_MODE_REGION_BOUNDARY_SUPPORT,
    candidate_nearest_neighbour_spacing,
    measure_edge_support_occupancy,
    resolve_boundary_support_spacing,
)
from osn_gs.surface.torch_region_owned_dense_boundary_support import (
    DenseBoundarySupportCandidate,
    _connect,
)


def _ring(count: int = 8, radius: float = 1.0) -> torch.Tensor:
    angles = torch.arange(count, dtype=torch.float32) / count * 2 * torch.pi
    return torch.stack((radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros(count)), dim=1)


class ResolveSpacingTest(unittest.TestCase):
    def test_full_evidence_mode_returns_the_baseline_for_every_candidate(self):
        positions = _ring()
        resolved = resolve_boundary_support_spacing(
            SPACING_MODE_FULL_EVIDENCE, positions, full_evidence_spacing=0.05, representative_spacing=0.3,
        )
        self.assertEqual(set(resolved.per_candidate_scale), {0.05})
        self.assertEqual(resolved.full_evidence_spacing, 0.05)
        # representative spacing is carried but must never become the scale
        self.assertEqual(resolved.representative_spacing, 0.3)
        self.assertNotIn(0.3, set(resolved.per_candidate_scale))

    def test_region_mode_uses_candidate_spacing_not_full_evidence_spacing(self):
        positions = _ring(count=8, radius=1.0)
        # ring chord for 8 points at r=1 is ~0.765, far above the 0.05 baseline
        resolved = resolve_boundary_support_spacing(
            SPACING_MODE_REGION_BOUNDARY_SUPPORT, positions, full_evidence_spacing=0.05,
        )
        scales = set(resolved.per_candidate_scale)
        self.assertEqual(len(scales), 1)
        self.assertGreater(scales.pop(), 0.7)
        self.assertIsNotNone(resolved.boundary_support_spacing)

    def test_local_mode_gives_a_smaller_scale_in_the_dense_half(self):
        # Left half tightly packed, right half sparse -- a single region-level
        # number cannot describe both, which is what local mode exists for.
        dense = torch.stack((torch.linspace(0.0, 0.3, 8), torch.zeros(8), torch.zeros(8)), dim=1)
        sparse = torch.stack((torch.linspace(2.0, 6.0, 5), torch.zeros(5), torch.zeros(5)), dim=1)
        positions = torch.cat((dense, sparse), dim=0)
        resolved = resolve_boundary_support_spacing(
            SPACING_MODE_LOCAL_BOUNDARY_SUPPORT, positions, full_evidence_spacing=0.01,
        )
        scales = list(resolved.per_candidate_scale)
        self.assertLess(max(scales[:8]), min(scales[8:]))

    def test_degenerate_candidate_set_falls_back_to_full_evidence_spacing(self):
        positions = torch.zeros((1, 3))
        for mode in (SPACING_MODE_REGION_BOUNDARY_SUPPORT, SPACING_MODE_LOCAL_BOUNDARY_SUPPORT):
            resolved = resolve_boundary_support_spacing(mode, positions, full_evidence_spacing=0.07)
            self.assertEqual(resolved.per_candidate_scale, (0.07,))
            self.assertTrue(resolved.diagnostics["fell_back_to_full_evidence"])

    def test_no_candidates_is_handled_without_raising(self):
        resolved = resolve_boundary_support_spacing(
            SPACING_MODE_REGION_BOUNDARY_SUPPORT, torch.zeros((0, 3)), full_evidence_spacing=0.02,
        )
        self.assertEqual(resolved.per_candidate_scale, ())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_boundary_support_spacing("made_up_mode", _ring(), full_evidence_spacing=0.1)

    def test_candidate_nn_spacing_matches_a_known_ring_chord(self):
        positions = _ring(count=4, radius=1.0)  # square, side = sqrt(2)
        spacing = candidate_nearest_neighbour_spacing(positions)
        self.assertAlmostEqual(float(spacing.median()), 2.0 ** 0.5, places=4)


class ConnectScaleOverrideTest(unittest.TestCase):
    def _candidates(self, positions, scale):
        return tuple(
            DenseBoundarySupportCandidate(
                stable_id=i, position=tuple(float(x) for x in row), normal=(0.0, 0.0, 1.0),
                tangent=(1.0, 0.0, 0.0), boundary_reason="observed_support_termination",
                full_evidence_scale=scale,
            )
            for i, row in enumerate(positions)
        )

    def test_none_override_is_identical_to_the_previous_behaviour(self):
        positions = _ring(count=6, radius=0.5)
        candidates = self._candidates(positions, 0.4)
        baseline = _connect(candidates, None)
        explicit = _connect(candidates, None, [0.4] * len(candidates))
        self.assertEqual(baseline.rejection_counts, explicit.rejection_counts)
        self.assertEqual(
            [c.stable_ids for c in baseline.components], [c.stable_ids for c in explicit.components],
        )

    def test_a_larger_scale_admits_pairs_the_baseline_rejected_on_distance(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        candidates = self._candidates(positions, 0.1)  # 2.5*0.1 = 0.25 < 1.0 spacing
        tight = _connect(candidates, None)
        loose = _connect(candidates, None, [1.0] * 3)  # 2.5*1.0 = 2.5 > 1.0
        self.assertGreater(
            tight.rejection_counts.get("distance_local_scale", 0),
            loose.rejection_counts.get("distance_local_scale", 0),
        )


class EdgeSupportOccupancyTest(unittest.TestCase):
    def test_edge_running_along_dense_evidence_is_fully_supported(self):
        candidates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        evidence = torch.stack((torch.linspace(0.0, 1.0, 40), torch.zeros(40), torch.zeros(40)), dim=1)
        report = measure_edge_support_occupancy([(0, 1)], candidates, evidence, full_evidence_spacing=0.05)
        self.assertEqual(report["edges_with_empty_interior_bin"], 0)
        self.assertEqual(report["unsupported_edge_fraction"], 0.0)

    def test_edge_spanning_a_real_gap_is_disclosed(self):
        candidates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # evidence only near the two ends; the middle is observed empty space
        ends = torch.cat((torch.linspace(0.0, 0.2, 10), torch.linspace(0.8, 1.0, 10)))
        evidence = torch.stack((ends, torch.zeros(20), torch.zeros(20)), dim=1)
        report = measure_edge_support_occupancy([(0, 1)], candidates, evidence, full_evidence_spacing=0.05)
        self.assertEqual(report["edges_with_empty_interior_bin"], 1)
        self.assertGreater(report["max_unsupported_run_ratio"], 0.4)

    def test_evidence_far_off_axis_does_not_count_as_support(self):
        candidates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        evidence = torch.stack((torch.linspace(0.0, 1.0, 40), torch.full((40,), 5.0), torch.zeros(40)), dim=1)
        report = measure_edge_support_occupancy([(0, 1)], candidates, evidence, full_evidence_spacing=0.05)
        self.assertEqual(report["edges_with_empty_interior_bin"], 1)

    def test_no_edges_is_handled(self):
        report = measure_edge_support_occupancy([], torch.zeros((0, 3)), torch.zeros((0, 3)), full_evidence_spacing=0.1)
        self.assertEqual(report["edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
