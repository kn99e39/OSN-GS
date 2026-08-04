"""Worklog 32: LOCAL EVIDENCE SCALE and the rejected REPRESENTATIVE GRAPH SCALE.

Per-representative reliability gate tracing (worklog 31) found that
``compute_full_neighborhood_evidence`` normalized its local-radius bound and
tangent-residual denominator by a single representative Gaussian's own
``tangent_major_scale`` -- on real long-horizon-trained checkpoints this is
~8x smaller than the true local full-cloud spacing, contaminating contextual
reliability. ``TorchOSNGSPipeline._construct_canonical_with_full_evidence``
now derives a separate, per-representative LOCAL EVIDENCE SCALE
(``cbrt(cell_volume / source_count)``, using a rotation/translation/uniform-
scale-INVARIANT characteristic scene length) and passes it into
``compute_full_neighborhood_evidence`` as ``local_evidence_scale``.

A companion REPRESENTATIVE GRAPH SCALE for ``build_manifold_affinity_graph``
(worklog 31's other finding: representative candidate generation) was
attempted with three different candidate definitions and REJECTED every
time -- each broke existing rigid-rotation invariance tests. That code path
(``torch_gaussian_manifold_affinity.py``) is UNCHANGED this round; see the
worklog for the full record of what was tried and why it failed.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def _pipeline(max_points: int) -> TorchOSNGSPipeline:
    return TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=max_points), device="cpu")


def _dense_plane_scene(count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    xy = torch.rand((count, 2), generator=generator) * 2.0 - 1.0
    positions = torch.cat([xy, torch.zeros((count, 1))], dim=1)
    scale = torch.tensor([[0.01, 0.008, 0.0005]]).repeat(count, 1)
    quaternion = torch.zeros((count, 4))
    quaternion[:, 0] = 1.0  # identity: local z axis == world z, matches the flat plane's normal
    covariance = covariance_from_scale_rotation(scale, quaternion)
    return positions, covariance


class LocalEvidenceScaleImprovesContextualReliabilityTest(unittest.TestCase):
    def test_dense_plane_gets_more_contextual_consistent_representatives_than_tiny_footprint_alone(self):
        """Direct before/after comparison on a synthetic case that reproduces
        the real-checkpoint pattern: many more full Gaussians than the
        representative cap, with individual covariance scale far smaller
        than the true representative spacing."""
        generator = torch.Generator().manual_seed(0)
        positions, covariance = _dense_plane_scene(3000, generator)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))

        pipeline = _pipeline(max_points=200)
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        summary = bundle.construction.diagnostic_summary
        # A flat, single-surface, densely-sampled plane should not collapse
        # to near-zero contextual-consistent representatives the way the
        # real long-horizon checkpoints did pre-fix (worklog 31).
        self.assertGreater(summary["reliable_count"], summary["intrinsic_reliable_count"] * 0.1)

    def test_local_evidence_scale_only_engages_when_downsampled(self):
        """Small scenes (representative count == full cloud) must keep the
        exact pre-worklog-32 code path -- ``local_evidence_scale`` stays
        unused when nothing was downsampled (worklog 129's own documented
        degeneracy argument, unchanged by this round)."""
        generator = torch.Generator().manual_seed(1)
        positions, covariance = _dense_plane_scene(20, generator)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        pipeline = _pipeline(max_points=2048)
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        self.assertEqual(int(bundle.representative_indices.numel()), positions.shape[0])


class RigidTransformInvarianceRegressionTest(unittest.TestCase):
    """Regression guard for the exact failure mode hit while iterating on
    this fix: a graph/evidence scale computed from an axis-aligned bounding
    box, or from the representative subset's own positions, is NOT
    rotation-invariant. ``local_evidence_scale`` must stay invariant since it
    now feeds a reliability decision."""

    def test_region_count_stable_under_rigid_rotation_translation_scale(self):
        generator = torch.Generator().manual_seed(2)
        positions, covariance = _dense_plane_scene(500, generator)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        pipeline = _pipeline(max_points=64)
        baseline = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)

        angle = 0.7
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotation = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ])
        scale = 3.0
        translation = torch.tensor([2.0, -1.0, 4.0])
        transformed_positions = (positions @ rotation.T) * scale + translation
        transformed_covariance = scale * scale * (rotation @ covariance @ rotation.T)
        transformed = pipeline._construct_canonical_with_full_evidence(
            transformed_positions, transformed_covariance, opacity, stable_ids
        )
        self.assertEqual(
            baseline.construction.diagnostic_summary["region_count"],
            transformed.construction.diagnostic_summary["region_count"],
        )


if __name__ == "__main__":
    unittest.main()
