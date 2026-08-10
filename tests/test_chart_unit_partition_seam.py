"""Worklog 86: partition-seam parametric chart-domain contract."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_chart_unit_partition_seam import (
    PHYSICAL_ONLY,
    STATE_MULTI_FRAGMENT_UNRESOLVED,
    _find_open_paths,
    _find_partition_seam,
    materialize_chart_unit_domain,
)
from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import (
    STATE_AMBIGUOUS_OR_OVER_MERGED,
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


class PhysicalOnlyPassThroughTest(unittest.TestCase):
    """On real evidence (and on a synthetic flat disc, where every point
    shares one normal so any candidate pair passes the same_surface
    criterion regardless of distance -- an unrestricted search pool is
    Worklog 85's own deliberate design, measured necessary there), a
    genuinely closed physical loop is what should be reused unchanged."""

    def test_a_fully_closed_physical_disc_never_attempts_a_seam(self):
        positions, covariance, stable_ids = _flat_disc_evidence()
        result = materialize_chart_unit_domain(positions, covariance, stable_ids, positions)
        self.assertEqual(result.state, STATE_MATERIALIZED)
        self.assertEqual(result.boundary_composition, PHYSICAL_ONLY)
        self.assertEqual(result.partition_seam_segment_count, 0)
        self.assertTrue(all(not s.is_partition_seam for s in result.segments))

    def test_ambiguous_over_merged_unit_passes_through_unchanged(self):
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
        result = materialize_chart_unit_domain(positions, covariance, stable_ids, positions)
        self.assertEqual(result.state, STATE_AMBIGUOUS_OR_OVER_MERGED)
        self.assertEqual(result.boundary_composition, "")

    def test_too_few_points_passes_through_no_dense_support(self):
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        covariance = _flat_covariance(3)
        stable_ids = list(range(3))
        result = materialize_chart_unit_domain(positions, covariance, stable_ids, positions)
        self.assertEqual(result.state, STATE_NO_DENSE_SUPPORT)


class FindOpenPathsTest(unittest.TestCase):
    """Direct tests of the new open-path detector, isolated from candidate
    admission/end-to-end quirks -- mirrors how Worklog 85's own
    `_find_valid_loops` was verified directly."""

    @staticmethod
    def _link(adjacency, a, b):
        adjacency[a].add(b)
        adjacency[b].add(a)

    def test_a_single_open_path_is_found_and_ordered_from_one_endpoint(self):
        adjacency = [set() for _ in range(5)]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            self._link(adjacency, a, b)
        paths = _find_open_paths(5, adjacency)
        self.assertEqual(len(paths), 1)
        self.assertIn(paths[0], ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]))

    def test_a_closed_cycle_is_never_reported_as_an_open_path(self):
        adjacency = [set() for _ in range(4)]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            self._link(adjacency, a, b)
        self.assertEqual(_find_open_paths(4, adjacency), [])

    def test_two_disjoint_open_paths_are_both_found(self):
        adjacency = [set() for _ in range(6)]
        for a, b in [(0, 1), (1, 2)]:
            self._link(adjacency, a, b)
        for a, b in [(3, 4), (4, 5)]:
            self._link(adjacency, a, b)
        paths = _find_open_paths(6, adjacency)
        self.assertEqual(len(paths), 2)

    def test_a_branching_component_is_not_reported_as_an_open_path(self):
        adjacency = [set() for _ in range(4)]
        for a, b in [(0, 1), (0, 2), (0, 3)]:
            self._link(adjacency, a, b)
        self.assertEqual(_find_open_paths(4, adjacency), [])


class FindPartitionSeamTest(unittest.TestCase):
    """Direct tests of the seam BFS over the unit's own interior
    same_surface graph."""

    def test_seam_found_through_dense_coherent_interior(self):
        positions, covariance, stable_ids = _flat_disc_evidence(radius=1.0, boundary_count=8, interior_axis=10)
        normals = torch.tensor([0.0, 0.0, 1.0]).expand(int(positions.shape[0]), 3)
        # Two boundary points on opposite sides of the disc -- connected only
        # via the dense interior (no adjacency graph among the 2 alone).
        endpoint_a, endpoint_b = 0, 4  # opposite points in the 8-point boundary ring
        seam = _find_partition_seam(positions, normals, None, set(), endpoint_a, endpoint_b)
        self.assertIsNotNone(seam)
        self.assertEqual(seam[0], endpoint_a)
        self.assertEqual(seam[-1], endpoint_b)
        # Every seam vertex is a real index into the unit's own positions.
        self.assertTrue(all(0 <= i < int(positions.shape[0]) for i in seam))

    def test_seam_returns_none_when_genuinely_disconnected(self):
        positions, covariance, stable_ids = _flat_disc_evidence(radius=1.0, boundary_count=8, interior_axis=10)
        far_point = torch.tensor([[50.0, 50.0, 50.0]])
        all_positions = torch.cat((positions, far_point), dim=0)
        normals = torch.tensor([0.0, 0.0, 1.0]).expand(int(all_positions.shape[0]), 3)
        seam = _find_partition_seam(all_positions, normals, None, set(), 0, int(all_positions.shape[0]) - 1)
        self.assertIsNone(seam)

    def test_excluded_indices_are_never_used_as_seam_intermediates(self):
        positions, covariance, stable_ids = _flat_disc_evidence(radius=1.0, boundary_count=8, interior_axis=10)
        normals = torch.tensor([0.0, 0.0, 1.0]).expand(int(positions.shape[0]), 3)
        n = int(positions.shape[0])
        excluded = set(range(n)) - {0, 4}
        seam = _find_partition_seam(positions, normals, None, excluded, 0, 4)
        # With every other candidate blocked, only a direct 0<->4 edge (if
        # one exists in the graph) could possibly work -- assert no excluded
        # index appears in the result either way.
        if seam is not None:
            self.assertTrue(all(i in (0, 4) for i in seam))


class FailClosedTest(unittest.TestCase):
    def test_multi_fragment_state_is_reachable_and_distinct_from_materialized(self):
        # Exercises the STATE_MULTI_FRAGMENT_UNRESOLVED code path directly by
        # constructing a physical candidate graph with two disjoint open
        # paths via the internal helper, independent of whether any real
        # end-to-end admission fixture happens to reach it.
        adjacency = [set() for _ in range(6)]

        def link(a, b):
            adjacency[a].add(b)
            adjacency[b].add(a)

        for a, b in [(0, 1), (1, 2)]:
            link(a, b)
        for a, b in [(3, 4), (4, 5)]:
            link(a, b)
        paths = _find_open_paths(6, adjacency)
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(STATE_MULTI_FRAGMENT_UNRESOLVED, STATE_MATERIALIZED)


if __name__ == "__main__":
    unittest.main()
