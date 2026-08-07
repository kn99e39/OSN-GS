"""Worklog 75: detached structural normal (positions-only local PCA)."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_region_owned_dense_boundary_support import DenseBoundarySupportCandidate
from osn_gs.surface.torch_structural_normal import (
    compute_structural_normals,
    normal_angular_disagreement_degrees,
    rebuild_candidate_orientation,
)


def _planar_grid(side: int = 7, spacing: float = 0.1) -> torch.Tensor:
    coords = torch.arange(side, dtype=torch.float32) * spacing
    u, v = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((u.reshape(-1), v.reshape(-1), torch.zeros(side * side)), dim=1)


class ComputeStructuralNormalsTest(unittest.TestCase):
    def test_planar_grid_normal_is_the_plane_normal(self):
        points = _planar_grid()
        normals = compute_structural_normals(points, neighbors=8)
        # Plane is z=0, so every structural normal must be +/- z.
        alignment = normals[:, 2].abs()
        self.assertGreater(float(alignment.min()), 0.999)

    def test_tilted_plane_normal_follows_the_plane(self):
        points = _planar_grid()
        # Rotate 30 degrees about the x axis; the normal must rotate with it.
        angle = math.radians(30.0)
        rotation = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ])
        rotated = points @ rotation.T
        expected = rotation @ torch.tensor([0.0, 0.0, 1.0])
        normals = compute_structural_normals(rotated, neighbors=8)
        alignment = (normals * expected[None, :]).sum(dim=-1).abs()
        self.assertGreater(float(alignment.min()), 0.999)

    def test_reads_positions_only_covariance_is_irrelevant(self):
        # The API takes no covariance/scale/rotation/opacity/SH at all; this
        # test pins that contract by construction -- the same positions must
        # give the same normals regardless of any other per-Gaussian state.
        points = _planar_grid()
        first = compute_structural_normals(points, neighbors=8)
        second = compute_structural_normals(points.clone(), neighbors=8)
        self.assertTrue(torch.equal(first, second))

    def test_too_few_points_returns_zeros_without_raising(self):
        points = torch.zeros((2, 3))
        normals = compute_structural_normals(points)
        self.assertEqual(tuple(normals.shape), (2, 3))


class AngularDisagreementTest(unittest.TestCase):
    def test_identical_fields_disagree_by_zero(self):
        a = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        degrees = normal_angular_disagreement_degrees(a, a.clone())
        self.assertLess(float(degrees.max()), 1e-3)

    def test_sign_flip_is_not_counted_as_disagreement(self):
        a = torch.tensor([[0.0, 0.0, 1.0]])
        degrees = normal_angular_disagreement_degrees(a, -a)
        self.assertLess(float(degrees.max()), 1e-3)

    def test_orthogonal_fields_disagree_by_ninety_degrees(self):
        a = torch.tensor([[0.0, 0.0, 1.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        degrees = normal_angular_disagreement_degrees(a, b)
        self.assertAlmostEqual(float(degrees[0]), 90.0, places=3)


class RebuildCandidateOrientationTest(unittest.TestCase):
    def _frozen(self, stable_ids, scale=0.1):
        return tuple(
            DenseBoundarySupportCandidate(
                stable_id=sid, position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
                tangent=(0.0, 1.0, 0.0), boundary_reason="observed_support_termination",
                full_evidence_scale=scale,
            )
            for sid in stable_ids
        )

    def test_identity_frozen_fields_are_carried_through_unchanged(self):
        points = _planar_grid()
        stable_ids = list(range(int(points.shape[0])))
        normals = compute_structural_normals(points, neighbors=8)
        frozen = self._frozen([0, 5, 9])
        rebuilt, diagnostics = rebuild_candidate_orientation(points, normals, stable_ids, frozen, neighbors=8)
        self.assertEqual(len(rebuilt), 3)
        for original, updated in zip(frozen, rebuilt):
            self.assertEqual(original.stable_id, updated.stable_id)
            self.assertEqual(original.position, updated.position)
            self.assertEqual(original.boundary_reason, updated.boundary_reason)
            self.assertEqual(original.full_evidence_scale, updated.full_evidence_scale)
        self.assertEqual(diagnostics["rebuilt_count"], 3)

    def test_tangent_becomes_orthogonal_to_the_new_normal(self):
        points = _planar_grid()
        stable_ids = list(range(int(points.shape[0])))
        normals = compute_structural_normals(points, neighbors=8)
        rebuilt, _ = rebuild_candidate_orientation(points, normals, stable_ids, self._frozen([0, 3, 12]), neighbors=8)
        for candidate in rebuilt:
            normal = torch.tensor(candidate.normal)
            tangent = torch.tensor(candidate.tangent)
            self.assertLess(abs(float(normal @ tangent)), 1e-5)
            self.assertAlmostEqual(float(tangent.norm()), 1.0, places=5)

    def test_candidate_set_stays_frozen_even_when_admission_would_fail(self):
        # Interior grid points have full angular support, so under any normal
        # their largest gap is far below pi -- they would NOT be admitted as
        # candidates. They must still be returned (frozen), and merely counted.
        points = _planar_grid()
        stable_ids = list(range(int(points.shape[0])))
        normals = compute_structural_normals(points, neighbors=8)
        interior = [24, 25, 31]  # middle of a 7x7 grid
        rebuilt, diagnostics = rebuild_candidate_orientation(
            points, normals, stable_ids, self._frozen(interior), neighbors=8,
        )
        self.assertEqual(len(rebuilt), len(interior))
        self.assertEqual(diagnostics["would_fail_sector_admission"], len(interior))

    def test_unknown_stable_id_is_carried_through_not_dropped(self):
        points = _planar_grid()
        stable_ids = list(range(int(points.shape[0])))
        normals = compute_structural_normals(points, neighbors=8)
        frozen = self._frozen(["not_in_this_point_set"])
        rebuilt, _ = rebuild_candidate_orientation(points, normals, stable_ids, frozen, neighbors=8)
        self.assertEqual(rebuilt, frozen)


if __name__ == "__main__":
    unittest.main()
