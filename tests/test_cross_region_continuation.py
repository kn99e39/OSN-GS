"""Worklog 40: cross-region smooth-continuation certificate.

A candidate's angular gap is measured over SAME-REGION accepted adjacency, so
a node on a region frontier reports a "support-free" direction that is in
fact occupied by observed Gaussians from a neighbouring region. Whether that
makes the candidate nonphysical depends on WHAT the neighbouring region is,
and the manifold affinity graph already knows:

  sphere    region pair (0,1): same_surface=12, crease=0  -> smooth continuation
  box       every face pair:   crease=32-33              -> genuine crease boundary
  cylinder  side/cap pairs:    crease=88-90              -> genuine crease boundary
  thin_slab region pair (0,1): parallel_but_separate=57  -> genuine outer edge

Only the crease-free, same_surface-bearing case is reclassified (to the
existing non-physical `reliability_frontier` state, with provenance kept).
Worklog 39 tried the cruder rule -- suppress whenever ANY out-of-region
support occupies the arc -- and it destroyed every genuine candidate on
box (110->0), cylinder (74->0) and thin_slab (48->3).
"""

from __future__ import annotations

import math
import unittest
from collections import Counter

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    _flat_grid,
    make_gaussian_reliability_scene,
)
from osn_gs.surface.torch_boundary_support_termination import classify_cross_region_pairs
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def _construct_scene(name: str):
    scene = make_gaussian_reliability_scene(name, seed=0)
    result = construct_visible_nurbs_from_gaussians(
        scene.positions, covariance=scene.covariances,
        stable_ids=tuple(range(scene.positions.shape[0])),
    )
    return scene, result


def _reason_counts(result) -> Counter:
    return Counter(h.boundary_reason for h in result.boundary_halfedge_candidates)


def _genuine(result) -> int:
    return _reason_counts(result).get("observed_support_termination", 0)


class CrossRegionRelationTaxonomyTest(unittest.TestCase):
    """The region-pair verdict must match the physical truth of each fixture."""

    def test_sphere_region_pair_is_smooth_continuation(self):
        _scene, result = _construct_scene("sphere")
        verdicts = classify_cross_region_pairs(result.surface_regions, result.manifold_affinity)
        self.assertTrue(verdicts, "sphere must fragment into touching regions for this fixture to be meaningful")
        self.assertTrue(all(v == "smooth_continuation" for v in verdicts.values()), verdicts)

    def test_box_region_pairs_are_crease_adjacent(self):
        _scene, result = _construct_scene("box")
        verdicts = classify_cross_region_pairs(result.surface_regions, result.manifold_affinity)
        self.assertTrue(verdicts)
        self.assertTrue(all(v == "crease_adjacent" for v in verdicts.values()), verdicts)

    def test_cylinder_side_cap_pairs_are_crease_adjacent(self):
        _scene, result = _construct_scene("cylinder")
        verdicts = classify_cross_region_pairs(result.surface_regions, result.manifold_affinity)
        self.assertTrue(verdicts)
        self.assertTrue(all(v == "crease_adjacent" for v in verdicts.values()), verdicts)

    def test_thin_slab_region_pair_is_parallel_separate(self):
        _scene, result = _construct_scene("thin_slab")
        verdicts = classify_cross_region_pairs(result.surface_regions, result.manifold_affinity)
        self.assertTrue(verdicts)
        self.assertTrue(all(v == "parallel_separate" for v in verdicts.values()), verdicts)


class SphereFalseBoundaryRemovedTest(unittest.TestCase):
    def test_sphere_emits_no_physical_termination_candidate(self):
        _scene, result = _construct_scene("sphere")
        self.assertEqual(_genuine(result), 0)

    def test_sphere_seam_candidates_are_kept_as_nonphysical_frontier(self):
        """Reclassified, never silently dropped -- provenance is preserved.

        Worklog 41 (task section 6): the typed state is now
        `smooth_cross_region_continuation`. These candidates are not a
        reliability problem -- the surface demonstrably continues into the
        neighbouring region -- so labelling them `reliability_frontier`
        ("support exists but is too ambiguous to trust") was semantically
        wrong.
        """
        _scene, result = _construct_scene("sphere")
        counts = _reason_counts(result)
        self.assertGreater(counts.get("smooth_cross_region_continuation", 0), 0)
        self.assertEqual(counts.get("reliability_frontier", 0), 0)

    def test_sphere_never_closes_a_loop_or_materializes(self):
        _scene, result = _construct_scene("sphere")
        summary = result.diagnostic_summary
        self.assertEqual(summary["boundary_component_closed_count"], 0)
        self.assertEqual(summary["materialized_surface_count"], 0)


class CreaseAndParallelBoundariesPreservedTest(unittest.TestCase):
    """The certificate must not suppress genuine physical terminations."""

    def test_box_keeps_all_genuine_candidates_and_closed_faces(self):
        _scene, result = _construct_scene("box")
        self.assertGreaterEqual(_genuine(result), 100)
        self.assertGreaterEqual(result.diagnostic_summary["boundary_component_closed_count"], 5)

    def test_cylinder_keeps_side_plus_two_caps(self):
        _scene, result = _construct_scene("cylinder")
        self.assertGreaterEqual(_genuine(result), 70)
        self.assertEqual(result.diagnostic_summary["boundary_component_closed_count"], 3)

    def test_thin_slab_keeps_both_outer_edges(self):
        _scene, result = _construct_scene("thin_slab")
        self.assertGreaterEqual(_genuine(result), 40)
        self.assertEqual(result.diagnostic_summary["boundary_component_closed_count"], 2)

    def test_box_face_single_region_is_unaffected(self):
        _scene, result = _construct_scene("box_face")
        self.assertEqual(_genuine(result), 32)
        self.assertEqual(result.diagnostic_summary["boundary_component_closed_count"], 1)


class FoldedSheetCreaseSweepTest(unittest.TestCase):
    """A sharp fold is a crease (keep); a flat sheet is one region (no
    cross-region question at all)."""

    @staticmethod
    def _folded(angle_degrees: float, n: int = 7, spacing: float = 0.12):
        angle = math.radians(angle_degrees)
        first_positions, first_cov = _flat_grid(
            n, spacing, normal=(0.0, 0.0, 1.0),
            origin=(-0.5 * spacing * (n - 1), 0.0, 0.0), seed=0,
        )
        second_positions, second_cov = _flat_grid(
            n, spacing, normal=(math.sin(angle), 0.0, math.cos(angle)),
            origin=(0.5 * spacing * (n - 1) * math.cos(angle), 0.0,
                    0.5 * spacing * (n - 1) * math.sin(angle)), seed=1,
        )
        return (
            torch.cat((first_positions, second_positions)),
            torch.cat((first_cov, second_cov)),
        )

    def test_sharp_fold_is_not_treated_as_smooth_continuation(self):
        positions, covariances = self._folded(90.0)
        result = construct_visible_nurbs_from_gaussians(
            positions, covariance=covariances, stable_ids=tuple(range(positions.shape[0])),
        )
        verdicts = classify_cross_region_pairs(result.surface_regions, result.manifold_affinity)
        for verdict in verdicts.values():
            self.assertNotEqual(verdict, "smooth_continuation")
        self.assertGreater(_genuine(result), 0)

    def test_flat_sheet_forms_one_region_with_genuine_boundary(self):
        positions, covariances = self._folded(180.0)
        result = construct_visible_nurbs_from_gaussians(
            positions, covariance=covariances, stable_ids=tuple(range(positions.shape[0])),
        )
        self.assertEqual(result.diagnostic_summary["region_count"], 1)
        self.assertGreater(_genuine(result), 0)


class AmbiguousCrossRegionIsNotSuppressedTest(unittest.TestCase):
    def test_ambiguous_verdict_never_suppresses_a_candidate(self):
        """A region pair with no crease/parallel/same_surface evidence is
        `ambiguous` and must be left as a physical candidate for review,
        not silently reclassified."""
        positions, covariances = FoldedSheetCreaseSweepTest._folded(120.0)
        result = construct_visible_nurbs_from_gaussians(
            positions, covariance=covariances, stable_ids=tuple(range(positions.shape[0])),
        )
        verdicts = classify_cross_region_pairs(result.surface_regions, result.manifold_affinity)
        if "ambiguous" in verdicts.values():
            self.assertGreater(_genuine(result), 0)


class RotationInvarianceTest(unittest.TestCase):
    @staticmethod
    def _rotate(tensor, angle, axis=(0.3, 0.7, 0.5), covariance=False):
        direction = torch.tensor(axis, dtype=tensor.dtype)
        direction = direction / direction.norm()
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cross = torch.tensor([
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ], dtype=tensor.dtype)
        rotation = torch.eye(3, dtype=tensor.dtype) + sin_a * cross + (1 - cos_a) * (cross @ cross)
        if covariance:
            return rotation @ tensor @ rotation.T
        return tensor @ rotation.T

    def test_candidate_classification_is_rotation_invariant(self):
        for scene_name in ("box_face", "cylinder", "sphere"):
            scene = make_gaussian_reliability_scene(scene_name, seed=0)
            baseline = None
            for angle in (0.0, 0.37, 0.91, 1.57):
                positions = self._rotate(scene.positions, angle)
                covariances = self._rotate(scene.covariances, angle, covariance=True)
                result = construct_visible_nurbs_from_gaussians(
                    positions, covariance=covariances, stable_ids=tuple(range(positions.shape[0])),
                )
                observed = (
                    _genuine(result),
                    result.diagnostic_summary["boundary_component_closed_count"],
                )
                if baseline is None:
                    baseline = observed
                self.assertEqual(observed, baseline, (scene_name, angle))


if __name__ == "__main__":
    unittest.main()
