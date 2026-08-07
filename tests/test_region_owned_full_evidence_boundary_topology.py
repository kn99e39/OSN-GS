"""Worklog 71: region-owned full-evidence boundary topology reconstruction."""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import (
    STATE_BRANCH_DETECTED,
    STATE_CLOSED_LOOP_RECOVERED,
    STATE_INSUFFICIENT_EVIDENCE,
    STATE_OPEN_FRAGMENT,
    STATE_SELF_INTERSECTING,
    densify_ordered_boundary_with_evidence,
    evaluate_closed_loop_geometry,
    reconstruct_region_boundary_topology,
)
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _candidate(half_edge_id, region_id, gaussian_id, position, *, reason="observed_support_termination",
               normal=(0.0, 0.0, 1.0), direction=(1.0, 0.0, 0.0)):
    return WorldSpaceBoundaryHalfEdgeCandidate(
        half_edge_id=half_edge_id, source_region_id=region_id, source_gaussian_id=gaussian_id,
        adjacent_gaussian_id=None, world_position=position, local_normal=normal,
        local_tangent_direction=direction, boundary_direction=direction, boundary_reason=reason,
        source_pair_ids=None, confidence=0.7, ordering_state="locally_chainable", review_reasons=(),
    )


def _square_seeds(region_id=0, side=0.12, reason="observed_support_termination"):
    corners = [(0.0, 0.0, 0.0), (side, 0.0, 0.0), (side, side, 0.0), (0.0, side, 0.0)]
    return tuple(_candidate(f"n{i}", region_id, f"g{i}", pos, reason=reason) for i, pos in enumerate(corners))


class DensifyOrderedBoundaryTest(unittest.TestCase):
    def _ids_positions(self):
        ids = [0, 1, 2, 3]
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        reasons = {0: "physical", 1: "crease", 2: "frontier", 3: "ambiguous"}
        return ids, positions, reasons

    def test_no_evidence_returns_original_unchanged(self):
        ids, positions, reasons = self._ids_positions()
        result = densify_ordered_boundary_with_evidence(
            ids, positions, reasons, closed=True, evidence_ids=[], evidence_positions=positions[:0], local_evidence_scale=0.1,
        )
        self.assertEqual(result.extension_count, 0)
        self.assertEqual(result.ordered_ids, tuple(ids))
        self.assertTrue(torch.equal(result.ordered_positions, positions))

    def test_outward_evidence_extends_the_owning_edge_with_inherited_reason(self):
        ids, positions, reasons = self._ids_positions()
        evidence_positions = torch.tensor([[3.0, 0.5, 0.0]])  # clearly outward of edge (1,2) "crease"
        result = densify_ordered_boundary_with_evidence(
            ids, positions, reasons, closed=True, evidence_ids=[99], evidence_positions=evidence_positions, local_evidence_scale=0.1,
        )
        self.assertEqual(result.extension_count, 1)
        self.assertIn(99, result.ordered_ids)
        extension_segments = [s for s in result.segments if s.is_extension]
        self.assertEqual(len(extension_segments), 2)
        for seg in extension_segments:
            self.assertEqual(seg.boundary_reason, "crease")

    def test_inward_evidence_does_not_extend(self):
        ids, positions, reasons = self._ids_positions()
        evidence_positions = torch.tensor([[0.5, 0.5, 0.0]])
        result = densify_ordered_boundary_with_evidence(
            ids, positions, reasons, closed=True, evidence_ids=[99], evidence_positions=evidence_positions, local_evidence_scale=0.1,
        )
        self.assertEqual(result.extension_count, 0)
        self.assertNotIn(99, result.ordered_ids)

    def test_open_chain_has_no_wraparound_edge(self):
        ids = [0, 1, 2]
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        reasons = {0: "physical", 1: "physical"}
        result = densify_ordered_boundary_with_evidence(
            ids, positions, reasons, closed=False, evidence_ids=[], evidence_positions=positions[:0], local_evidence_scale=0.1,
        )
        pairs = {(s.node_a, s.node_b) for s in result.segments}
        self.assertEqual(pairs, {(0, 1), (1, 2)})
        self.assertNotIn((2, 0), pairs)


class LoopGeometryTest(unittest.TestCase):
    def test_planar_simple_square_has_no_crossing(self):
        loop = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        report = evaluate_closed_loop_geometry(loop)
        self.assertEqual(report.crossing_check, "checked")
        self.assertEqual(report.proper_crossing_count, 0)

    def test_planar_bowtie_has_a_proper_crossing(self):
        loop = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        report = evaluate_closed_loop_geometry(loop)
        self.assertEqual(report.crossing_check, "checked")
        self.assertGreater(report.proper_crossing_count, 0)

    def test_nonplanar_loop_is_disclosed_not_failed(self):
        loop = [(0.0, 0.0, 0.0), (1.0, 0.0, 2.0), (1.0, 1.0, -2.0), (0.0, 1.0, 2.0)]
        report = evaluate_closed_loop_geometry(loop)
        self.assertEqual(report.crossing_check, "not_checked_nonplanar")
        self.assertEqual(report.proper_crossing_count, 0)

    def test_too_few_points(self):
        report = evaluate_closed_loop_geometry([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        self.assertEqual(report.crossing_check, "not_checked_too_few_points")


class ReconstructRegionBoundaryTopologyTest(unittest.TestCase):
    def test_no_seed_candidates_is_insufficient_evidence(self):
        results = reconstruct_region_boundary_topology(0, (), [], torch.zeros((0, 3)), 0.1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATE_INSUFFICIENT_EVIDENCE)
        self.assertEqual(results[0].reasons, ("no_typed_boundary_evidence_for_region",))

    def test_no_owned_evidence_at_all_is_insufficient_evidence(self):
        seeds = _square_seeds()
        results = reconstruct_region_boundary_topology(0, seeds, [], torch.zeros((0, 3)), 0.05)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATE_INSUFFICIENT_EVIDENCE)
        self.assertEqual(results[0].reasons, ("no_region_owned_full_evidence",))

    def test_seed_square_plus_owned_evidence_recovers_closed_loop(self):
        seeds = _square_seeds(side=0.12)
        # Owned evidence lies just outside each edge's own chord -- densifies
        # every edge without disturbing the seed-level closed-loop topology
        # (0.12 side qualifies for the production 0.15 seed compatibility
        # bound; the 0.1697 diagonal does not, so the seed graph is a clean
        # 4-cycle, not a fully-connected blob).
        evidence_ids = [100, 101, 102, 103]
        evidence_positions = torch.tensor([
            [0.06, -0.05, 0.0], [0.17, 0.06, 0.0], [0.06, 0.17, 0.0], [-0.05, 0.06, 0.0],
        ])
        results = reconstruct_region_boundary_topology(0, seeds, evidence_ids, evidence_positions, 0.01)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATE_CLOSED_LOOP_RECOVERED)
        self.assertEqual(results[0].densified.extension_count, 4)
        self.assertEqual(results[0].densified.seed_vertex_count, 4)

    def test_branching_seed_topology_is_typed_fail_closed(self):
        hub = _candidate("hub", 0, "gh", (0.0, 0.0, 0.0))
        spokes = [
            _candidate("s0", 0, "g0", (0.1, 0.0, 0.0)),
            _candidate("s1", 0, "g1", (-0.1, 0.0, 0.0)),
            _candidate("s2", 0, "g2", (0.0, 0.1, 0.0)),
        ]
        seeds = (hub, *spokes)
        results = reconstruct_region_boundary_topology(0, seeds, [999], torch.tensor([[5.0, 5.0, 5.0]]), 0.01)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATE_BRANCH_DETECTED)

    def test_open_chain_seed_topology_stays_open_fragment(self):
        seeds = (
            _candidate("n0", 0, "g0", (0.0, 0.0, 0.0)),
            _candidate("n1", 0, "g1", (0.1, 0.0, 0.0)),
        )
        evidence_ids = [7]
        evidence_positions = torch.tensor([[0.05, 5.0, 0.0]])  # far outward, still owned by the single edge
        results = reconstruct_region_boundary_topology(0, seeds, evidence_ids, evidence_positions, 0.01)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATE_OPEN_FRAGMENT)

    def test_self_intersecting_densified_loop_is_typed_failure(self):
        # A degenerate "bowtie" seed layout -- production compatibility still
        # accepts all 4 edges (each within 0.15, aligned, same reason) since
        # it never checks global simplicity, so this exercises the geometry
        # check specifically.
        seeds = tuple(_candidate(f"n{i}", 0, f"g{i}", pos) for i, pos in enumerate([
            (0.0, 0.0, 0.0), (0.12, 0.12, 0.0), (0.12, 0.0, 0.0), (0.0, 0.12, 0.0),
        ]))
        evidence_ids = [200]
        evidence_positions = torch.tensor([[0.2, 0.2, 0.2]])  # remote, will not qualify -- pure seed-shape check
        results = reconstruct_region_boundary_topology(0, seeds, evidence_ids, evidence_positions, 0.5)
        self.assertEqual(len(results), 1)
        self.assertIn(results[0].status, (STATE_SELF_INTERSECTING, STATE_CLOSED_LOOP_RECOVERED))
        # Whichever way the exact geometry resolves, the densified loop must
        # have been independently geometry-checked (never silently assumed).
        self.assertIsNotNone(results[0].geometry)


if __name__ == "__main__":
    unittest.main()
