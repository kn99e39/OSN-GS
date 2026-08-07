"""Worklog 77: discretization-bias correction in the boundary-support predicate.

These pin the specific defect fixed in worklog 77: the angular-gap estimator
measures the empty sector between two flanking neighbour RAYS, which
understates the true empty sector by roughly the local angular sampling
resolution. For a STRAIGHT boundary the true empty sector is exactly pi, so the
uncorrected measurement converges to pi from below and `gap >= pi` is failed in
the limit of perfect sampling.

The threshold (`missing_sector_radians`) is unchanged at pi throughout; what
these tests protect is that the estimate is de-biased by the point's own
sampling resolution, and -- critically -- that this does NOT loosen admission:
interior points and closed manifolds must still be rejected.
"""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support


def _flat_grid(side: int, spacing: float = 1.0) -> torch.Tensor:
    coords = torch.arange(side, dtype=torch.float32) * spacing
    u, v = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((u.reshape(-1), v.reshape(-1), torch.zeros(side * side)), dim=1)


def _perimeter_indices(side: int) -> set[int]:
    return {i for i in range(side * side) if (i // side in (0, side - 1)) or (i % side in (0, side - 1))}


def _admitted(points: torch.Tensor, normals: torch.Tensor | None = None) -> set:
    n = int(points.shape[0])
    if normals is None:
        normals = torch.tensor([[0.0, 0.0, 1.0]] * n, dtype=torch.float32)
    result = extract_dense_boundary_support(points, normals, list(range(n)))
    return {c.stable_id for c in result.candidates}


class StraightBoundaryAdmissionTest(unittest.TestCase):
    def test_entire_flat_grid_perimeter_is_admitted(self):
        side = 7
        points = _flat_grid(side)
        admitted = _admitted(points)
        perimeter = _perimeter_indices(side)
        self.assertEqual(admitted & perimeter, perimeter, "every straight-boundary point must be admitted")

    def test_no_interior_point_is_admitted(self):
        side = 7
        points = _flat_grid(side)
        admitted = _admitted(points)
        interior = set(range(side * side)) - _perimeter_indices(side)
        self.assertEqual(admitted & interior, set(), "the correction must not admit interior points")

    def test_edge_midpoint_the_exact_degenerate_case_is_admitted(self):
        # A point in the middle of a straight edge is the exactly-degenerate
        # case: its true empty sector is exactly pi. This is the case the
        # uncorrected predicate decided by floating-point noise.
        side = 7
        points = _flat_grid(side)
        midpoint = (side // 2) * side  # middle of the col==0 edge
        self.assertIn(midpoint, _admitted(points))

    def test_admission_survives_sampling_noise_on_a_straight_edge(self):
        side = 7
        torch.manual_seed(0)
        points = _flat_grid(side) + 0.01 * torch.randn(side * side, 3)
        admitted = _admitted(points)
        perimeter = _perimeter_indices(side)
        # Noise must not flip genuine straight-boundary points out of admission.
        self.assertGreaterEqual(len(admitted & perimeter), len(perimeter) - 1)


class FailClosedPreservedTest(unittest.TestCase):
    def test_closed_manifold_admits_no_boundary(self):
        # Fibonacci sphere: a closed surface with no boundary anywhere. The
        # correction must not manufacture boundary support here.
        count = 300
        indices = torch.arange(count, dtype=torch.float32)
        phi = torch.acos(1 - 2 * (indices + 0.5) / count)
        theta = math.pi * (1 + 5 ** 0.5) * indices
        points = torch.stack((
            torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi),
        ), dim=1)
        normals = points / points.norm(dim=1, keepdim=True)
        self.assertEqual(_admitted(points, normals), set())

    def test_empty_evidence_still_invents_nothing(self):
        result = extract_dense_boundary_support(torch.zeros((0, 3)), torch.zeros((0, 3)), [])
        self.assertEqual(result.candidates, ())

    def test_dense_interior_is_rejected_more_firmly_as_density_rises(self):
        # The correction is a bias term that VANISHES as sampling density
        # grows, so a denser interior must not drift toward admission.
        for side in (7, 11, 15):
            points = _flat_grid(side, spacing=1.0)
            interior = set(range(side * side)) - _perimeter_indices(side)
            self.assertEqual(_admitted(points) & interior, set(), f"side={side}")


if __name__ == "__main__":
    unittest.main()
