"""Worklog 84: chart-unit coherence audit + evidence-scale boundary topology."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import (
    STATE_AMBIGUOUS_OR_OVER_MERGED,
    STATE_COVERAGE_FAILED,
    STATE_MATERIALIZED,
    STATE_NO_DENSE_SUPPORT,
    STATE_SELF_INTERSECTING,
    assess_chart_unit_coherence,
    materialize_chart_unit_boundary,
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


class CoherenceAuditTest(unittest.TestCase):
    def test_flat_disc_is_coherent(self):
        positions, covariance, _ = _flat_disc_evidence()
        result = assess_chart_unit_coherence(covariance, list(range(int(positions.shape[0]))))
        self.assertTrue(result.coherent)
        self.assertLessEqual(result.internal_normal_disagreement_fraction, 0.15)

    def test_minority_orthogonal_cluster_makes_the_unit_incoherent(self):
        # A dominant flat disc (its normal sets the unit's own mean
        # reference) plus a smaller orthogonally-oriented cluster (~30% of
        # the population -- large enough to exceed the 0.15 disagreement
        # bound, small enough that it does not itself dominate the mean and
        # produce a degenerate symmetric compromise direction).
        dominant = _flat_disc_evidence(radius=1.0, boundary_count=24, interior_axis=12)[0]
        cov_dominant = _flat_covariance(int(dominant.shape[0]))
        minority_xy = _ring(45, 0.3)
        minority = torch.stack((minority_xy[:, 0] + 3.0, minority_xy[:, 2], minority_xy[:, 1]), dim=1)
        scale_minority = torch.tensor([0.15, 0.01, 0.15]).expand(int(minority.shape[0]), 3)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(int(minority.shape[0]), 4)
        cov_minority = covariance_from_scale_rotation(scale_minority, identity_quat)
        covariance = torch.cat((cov_dominant, cov_minority), dim=0)
        result = assess_chart_unit_coherence(covariance, list(range(int(covariance.shape[0]))))
        self.assertFalse(result.coherent)
        self.assertGreater(result.internal_normal_disagreement_fraction, 0.15)


class EvidenceScaleBoundaryMaterializationTest(unittest.TestCase):
    def test_coherent_flat_disc_materializes_a_closed_boundary_directly_from_evidence(self):
        positions, covariance, stable_ids = _flat_disc_evidence()
        result = materialize_chart_unit_boundary(positions, covariance, stable_ids, positions)
        self.assertEqual(result.state, STATE_MATERIALIZED)
        self.assertGreaterEqual(len(result.ordered_stable_ids), 3)
        # No sparse macro arcs were supplied -- geometry must still succeed.
        self.assertTrue(all(s.segment_kind == "" for s in result.segments))

    def test_boundary_does_not_require_sparse_macro_nodes(self):
        # Directly asserts the architectural requirement: a unit with
        # hundreds of evidence points and ZERO sparse arc coverage still
        # gets a materialized boundary from its own dense evidence alone.
        positions, covariance, stable_ids = _flat_disc_evidence(radius=1.0, boundary_count=40, interior_axis=20)
        result = materialize_chart_unit_boundary(
            positions, covariance, stable_ids, positions,
            arc_starts=None, arc_ends=None, arc_kinds=None,
        )
        self.assertEqual(result.state, STATE_MATERIALIZED)

    def test_typed_arc_labels_boundary_segments_when_supplied(self):
        positions, covariance, stable_ids = _flat_disc_evidence()
        arc_starts = torch.tensor([[0.0, -1.5, 0.0]])
        arc_ends = torch.tensor([[0.0, 1.5, 0.0]])
        result = materialize_chart_unit_boundary(
            positions, covariance, stable_ids, positions,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=["crease"],
        )
        self.assertEqual(result.state, STATE_MATERIALIZED)
        self.assertTrue(any(s.segment_kind == "crease" for s in result.segments))


class FailClosedTest(unittest.TestCase):
    def test_ambiguous_or_over_merged_unit_never_reaches_boundary_stage(self):
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
        result = materialize_chart_unit_boundary(positions, covariance, stable_ids, positions)
        self.assertEqual(result.state, STATE_AMBIGUOUS_OR_OVER_MERGED)
        self.assertEqual(result.ordered_stable_ids, ())
        self.assertIsNotNone(result.coherence)
        self.assertFalse(result.coherence.coherent)

    def test_too_few_points_for_local_evidence_fails_no_dense_support(self):
        # `extract_dense_boundary_support` itself requires >= 4 points to
        # estimate any local angular evidence at all.
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        covariance = _flat_covariance(3)
        stable_ids = list(range(3))
        result = materialize_chart_unit_boundary(positions, covariance, stable_ids, positions)
        self.assertEqual(result.state, STATE_NO_DENSE_SUPPORT)

    def test_never_bridges_to_close_an_open_fragment(self):
        # A half-ring (not a full loop): angular ordering around the
        # centroid still produces a SIMPLE (non-self-intersecting) polygon by
        # closing the open end with a straight chord -- that chord crosses
        # empty space, which the occupancy safety gate must catch.
        angles = torch.linspace(0.0, math.pi, 10)
        half_ring = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros(10)), dim=1)
        covariance = _flat_covariance(10)
        stable_ids = list(range(10))
        result = materialize_chart_unit_boundary(half_ring, covariance, stable_ids, half_ring)
        self.assertNotEqual(result.state, STATE_MATERIALIZED)

    def test_open_fragment_fails_with_unsupported_closure_specifically(self):
        from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import STATE_UNSUPPORTED_CLOSURE

        angles = torch.linspace(0.0, math.pi, 10)
        half_ring = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros(10)), dim=1)
        covariance = _flat_covariance(10)
        stable_ids = list(range(10))
        result = materialize_chart_unit_boundary(half_ring, covariance, stable_ids, half_ring)
        self.assertEqual(result.state, STATE_UNSUPPORTED_CLOSURE)


if __name__ == "__main__":
    unittest.main()
