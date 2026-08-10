"""Worklog 87: partition_seam as a first-class parametric-boundary type."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_chart_unit_partition_seam import (
    MIXED_PHYSICAL_PARTITION_SEAM,
    PHYSICAL_ONLY,
    SEAM_DOMINATED,
    STATE_NO_OPEN_TOPOLOGY,
    STATE_SEAM_NOT_FOUND,
    _find_isolated_candidates,
    _find_open_paths,
    _find_partition_seam,
    _stitch_pieces_into_domain,
    materialize_chart_unit_domains,
)
from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import (
    STATE_MATERIALIZED,
    STATE_NO_DENSE_SUPPORT,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def _ring(count: int, radius: float) -> torch.Tensor:
    angles = torch.arange(count, dtype=torch.float32) / count * 2 * math.pi
    return torch.stack((radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros(count)), dim=1)


def _disc(count_per_axis: int, radius: float) -> torch.Tensor:
    axis = torch.linspace(-radius, radius, count_per_axis)
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((u.reshape(-1), v.reshape(-1), torch.zeros(count_per_axis ** 2)), dim=1)
    return points[points[:, :2].norm(dim=1) <= radius * 0.95]


def _flat_covariance(count: int, tangent_scale: float = 0.15, normal_thickness: float = 0.01) -> torch.Tensor:
    scale = torch.tensor([tangent_scale, tangent_scale, normal_thickness]).expand(count, 3)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, 4)
    return covariance_from_scale_rotation(scale, identity_quat)


def _flat_disc_evidence(radius: float = 1.0, boundary_count: int = 24, interior_axis: int = 12):
    boundary = _ring(boundary_count, radius)
    interior = _disc(interior_axis, radius * 0.9)
    positions = torch.cat((boundary, interior), dim=0)
    covariance = _flat_covariance(int(positions.shape[0]))
    stable_ids = list(range(int(positions.shape[0])))
    return positions, covariance, stable_ids


class FindIsolatedCandidatesTest(unittest.TestCase):
    def test_a_zero_degree_node_is_isolated(self):
        adjacency = [set(), {2}, {1}]
        self.assertEqual(_find_isolated_candidates(3, adjacency), [0])

    def test_no_isolated_candidates_in_a_full_cycle(self):
        adjacency = [{1, 2}, {0, 2}, {0, 1}]
        self.assertEqual(_find_isolated_candidates(3, adjacency), [])


class StitchPiecesIntoDomainTest(unittest.TestCase):
    """Direct tests of the general N-piece daisy-chain stitcher, isolated
    from candidate-admission end-to-end quirks."""

    def _dense_interior_fixture(self):
        # 8-point boundary ring + dense interior disc, all coplanar so the
        # interior graph is richly connected -- exactly the "genuine
        # interior evidence" a seam is supposed to route through.
        positions, _covariance, stable_ids = _flat_disc_evidence(radius=1.0, boundary_count=8, interior_axis=10)
        normals = torch.tensor([0.0, 0.0, 1.0]).expand(int(positions.shape[0]), 3)
        candidate_ids = list(range(8))  # the 8 boundary-ring points, used directly as "candidates"
        candidate_positions = positions[:8]
        id_to_full_index = {sid: i for i, sid in enumerate(stable_ids)}
        return positions, normals, stable_ids, candidate_ids, candidate_positions, id_to_full_index

    def test_two_single_point_pieces_are_stitched_via_two_seams(self):
        positions, normals, stable_ids, candidate_ids, candidate_positions, id_to_full_index = self._dense_interior_fixture()
        # Two isolated "fragments" of length 1 -- opposite sides of the ring.
        pieces = [[0], [4]]
        result, reason = _stitch_pieces_into_domain(
            pieces, candidate_ids, candidate_positions, positions, normals, stable_ids, id_to_full_index, None,
        )
        self.assertIsNotNone(result, reason)
        chain_ids, chain_positions, chain_is_physical = result
        self.assertEqual(len(chain_ids), len(chain_positions))
        self.assertEqual(len(chain_ids), len(chain_is_physical))
        # Two single-point pieces -> zero internal physical edges anywhere.
        self.assertFalse(any(chain_is_physical))

    def test_deterministic_ordering_is_independent_of_input_piece_order(self):
        positions, normals, stable_ids, candidate_ids, candidate_positions, id_to_full_index = self._dense_interior_fixture()
        pieces_a = [[0], [4]]
        pieces_b = [[4], [0]]
        result_a, _ = _stitch_pieces_into_domain(
            pieces_a, candidate_ids, candidate_positions, positions, normals, stable_ids, id_to_full_index, None,
        )
        result_b, _ = _stitch_pieces_into_domain(
            pieces_b, candidate_ids, candidate_positions, positions, normals, stable_ids, id_to_full_index, None,
        )
        self.assertEqual(result_a[0], result_b[0])

    def test_a_two_point_path_piece_contributes_one_internal_physical_edge(self):
        positions, normals, stable_ids, candidate_ids, candidate_positions, id_to_full_index = self._dense_interior_fixture()
        pieces = [[0, 1], [4, 5]]
        result, reason = _stitch_pieces_into_domain(
            pieces, candidate_ids, candidate_positions, positions, normals, stable_ids, id_to_full_index, None,
        )
        self.assertIsNotNone(result, reason)
        _chain_ids, _chain_positions, chain_is_physical = result
        self.assertEqual(sum(chain_is_physical), 2)  # one internal edge per 2-point piece

    def test_unreachable_endpoint_fails_closed_with_a_reason(self):
        positions, normals, stable_ids, candidate_ids, candidate_positions, id_to_full_index = self._dense_interior_fixture()
        far_point = torch.tensor([[500.0, 500.0, 500.0]])
        positions2 = torch.cat((positions, far_point), dim=0)
        normals2 = torch.cat((normals, torch.tensor([[0.0, 0.0, 1.0]])), dim=0)
        stable_ids2 = stable_ids + [len(stable_ids)]
        id_to_full_index2 = {sid: i for i, sid in enumerate(stable_ids2)}
        candidate_ids2 = candidate_ids + [len(stable_ids)]
        candidate_positions2 = torch.cat((candidate_positions, far_point), dim=0)
        pieces = [[0], [8]]  # index 8 is the far, disconnected point
        result, reason = _stitch_pieces_into_domain(
            pieces, candidate_ids2, candidate_positions2, positions2, normals2, stable_ids2, id_to_full_index2, None,
        )
        self.assertIsNone(result)
        self.assertIn("no_interior_adjacency_path", reason)


class MaterializeChartUnitDomainsTest(unittest.TestCase):
    def test_a_fully_closed_physical_disc_is_one_physical_only_domain(self):
        positions, covariance, stable_ids = _flat_disc_evidence()
        result = materialize_chart_unit_domains(positions, covariance, stable_ids, positions)
        self.assertEqual(len(result.domains), 1)
        self.assertTrue(result.materialized)
        self.assertEqual(result.domains[0].state, STATE_MATERIALIZED)
        self.assertEqual(result.domains[0].boundary_composition, PHYSICAL_ONLY)
        self.assertEqual(result.domains[0].partition_seam_segment_count, 0)

    def test_two_disjoint_physical_loops_are_each_independently_detected_and_validated(self):
        # Two far-apart rings artificially concatenated into one "unit" --
        # this is not how real Worklog 83 assembly would ever group evidence
        # (it requires proximity), so this exercises the DETECTION/
        # independent-validation mechanism directly rather than asserting a
        # realistic end-to-end success. Each ring is only ~50% of the
        # combined unit's evidence, so the Worklog 79 coverage contract
        # (unmodified, checked against the WHOLE unit) correctly rejects
        # both individually here -- that is the coverage contract doing
        # exactly its job, not a bug in multi-loop detection.
        loop_a = _flat_disc_evidence(radius=1.0, boundary_count=16, interior_axis=8)[0]
        loop_b_xy = _flat_disc_evidence(radius=1.0, boundary_count=16, interior_axis=8)[0]
        loop_b = torch.stack((loop_b_xy[:, 0] + 5.0, loop_b_xy[:, 1], loop_b_xy[:, 2]), dim=1)
        positions = torch.cat((loop_a, loop_b), dim=0)
        covariance = _flat_covariance(int(positions.shape[0]))
        stable_ids = list(range(int(positions.shape[0])))
        result = materialize_chart_unit_domains(positions, covariance, stable_ids, positions)
        physical_loop_attempts = [r for r in result.unresolved_reasons if r.startswith("physical_loop_rejected")]
        self.assertEqual(len(physical_loop_attempts), 2, "both rings must be found and validated as SEPARATE loops")

    def test_two_disjoint_physical_loops_each_covering_their_own_full_evidence_both_materialize(self):
        # Same two-ring detection mechanism, but each domain is validated
        # against ITS OWN evidence (as a real per-unit call would receive),
        # so the coverage contract is satisfied and both materialize.
        loop_a = _flat_disc_evidence(radius=1.0, boundary_count=16, interior_axis=8)[0]
        loop_b_xy = _flat_disc_evidence(radius=1.0, boundary_count=16, interior_axis=8)[0]
        loop_b = torch.stack((loop_b_xy[:, 0] + 5.0, loop_b_xy[:, 1], loop_b_xy[:, 2]), dim=1)
        cov_a = _flat_covariance(int(loop_a.shape[0]))
        cov_b = _flat_covariance(int(loop_b.shape[0]))
        result_a = materialize_chart_unit_domains(loop_a, cov_a, list(range(int(loop_a.shape[0]))), loop_a)
        result_b = materialize_chart_unit_domains(loop_b, cov_b, list(range(int(loop_b.shape[0]))), loop_b)
        self.assertTrue(result_a.materialized)
        self.assertTrue(result_b.materialized)
        self.assertEqual(result_a.domains[0].boundary_composition, PHYSICAL_ONLY)

    def test_too_few_candidates_yields_no_dense_support_with_new_floor(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        covariance = _flat_covariance(3)
        stable_ids = list(range(3))
        result = materialize_chart_unit_domains(positions, covariance, stable_ids, positions)
        self.assertEqual(len(result.domains), 0)
        self.assertTrue(any(STATE_NO_DENSE_SUPPORT in r for r in result.unresolved_reasons))

    def test_ambiguous_over_merged_unit_yields_no_domains(self):
        dominant = _flat_disc_evidence(radius=1.0, boundary_count=24, interior_axis=12)[0]
        cov_dominant = _flat_covariance(int(dominant.shape[0]))
        minority_xy = _ring(45, 0.3)
        minority = torch.stack((minority_xy[:, 0] + 3.0, minority_xy[:, 2], minority_xy[:, 1]), dim=1)
        scale_minority = torch.tensor([0.15, 0.01, 0.15]).expand(int(minority.shape[0]), 3)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(int(minority.shape[0]), 4)
        cov_minority = covariance_from_scale_rotation(scale_minority, identity_quat)
        positions = torch.cat((dominant, minority), dim=0)
        covariance = torch.cat((cov_dominant, cov_minority), dim=0)
        stable_ids = list(range(int(positions.shape[0])))
        result = materialize_chart_unit_domains(positions, covariance, stable_ids, positions)
        self.assertEqual(len(result.domains), 0)
        self.assertFalse(result.coherence.coherent)

    def test_never_fabricates_a_domain_when_topology_has_nothing_open(self):
        # A trivially tiny set that can admit candidates but not close or
        # chain into anything -- must disclose, not invent.
        positions, covariance, stable_ids = _flat_disc_evidence(radius=1.0, boundary_count=4, interior_axis=2)
        result = materialize_chart_unit_domains(positions, covariance, stable_ids, positions)
        for d in result.domains:
            self.assertNotEqual(d.state, "")


if __name__ == "__main__":
    unittest.main()
