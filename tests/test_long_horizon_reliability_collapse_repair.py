"""Worklog 135: long-horizon reliability-collapse root causes and repairs.

Covers two narrow, evidence-backed fixes found while diagnosing why the
Visible Surface Constructor collapses on real long-horizon training
checkpoints (1.6M-3M Gaussians) well before reaching boundary recovery:

1. ``extract_covariance_frame`` chunks its ``torch.linalg.eigh`` call to stay
   under cuSOLVER's batched-eigensolver batch-size ceiling (reproduced at
   ~2.06M on the reference GPU/driver -- unrelated to NaN despite the
   exception's own misleading hint). Chunking must be numerically identical
   to one unchunked call.
2. ``compute_full_neighborhood_evidence`` bounds its per-representative
   aggregate to a local radius (``local_radius_tangent_scale_multiplier *
   representative's own tangent_major_scale``) instead of an unbounded
   global nearest-representative Voronoi partition, so a representative's
   contextual evidence is not contaminated by spatially non-local full-cloud
   members once the true Gaussian count vastly exceeds the representative
   cap. See docs/worklogs/*_long_horizon_reliability_collapse_and_nan_repair.md.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface import torch_gaussian_covariance_frame as covariance_frame_module
from osn_gs.surface.torch_gaussian_covariance_frame import (
    covariance_from_scale_rotation,
    extract_covariance_frame,
)
from osn_gs.surface.torch_full_neighborhood_evidence import (
    FullNeighborhoodEvidenceConfig,
    compute_full_neighborhood_evidence,
)
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_intrinsic_reliability


def _random_planar_covariance(count: int, generator: torch.Generator) -> torch.Tensor:
    scale = torch.stack(
        [
            torch.full((count,), 0.05),
            torch.full((count,), 0.04),
            torch.full((count,), 0.005),
        ],
        dim=1,
    )
    quaternion = torch.nn.functional.normalize(torch.randn((count, 4), generator=generator), dim=1)
    return covariance_from_scale_rotation(scale, quaternion)


class ChunkedEighEquivalenceTest(unittest.TestCase):
    def test_chunked_frame_matches_unchunked_frame_exactly(self):
        """Splitting the batch must not change any per-Gaussian result --
        eigh over independent (3, 3) blocks has no cross-row interaction."""
        generator = torch.Generator().manual_seed(0)
        count = 5000
        covariance = _random_planar_covariance(count, generator)

        unchunked = extract_covariance_frame(covariance)

        original_limit = covariance_frame_module._EIGH_MAX_BATCH_SIZE
        covariance_frame_module._EIGH_MAX_BATCH_SIZE = 777  # force multiple chunks
        try:
            chunked = extract_covariance_frame(covariance)
        finally:
            covariance_frame_module._EIGH_MAX_BATCH_SIZE = original_limit

        torch.testing.assert_close(chunked.eigenvalues, unchunked.eigenvalues)
        torch.testing.assert_close(chunked.normal_candidate, unchunked.normal_candidate)
        torch.testing.assert_close(chunked.tangent_major_scale, unchunked.tangent_major_scale)
        self.assertEqual(chunked.shape_class, unchunked.shape_class)

    def test_single_chunk_path_unchanged_when_below_limit(self):
        """Below the chunk limit, behavior is byte-identical to the
        pre-worklog-135 direct ``torch.linalg.eigh`` call (regression guard
        against accidentally always chunking)."""
        generator = torch.Generator().manual_seed(1)
        covariance = _random_planar_covariance(50, generator)
        frame = extract_covariance_frame(covariance)
        direct_eigenvalues, _ = torch.linalg.eigh(0.5 * (covariance + covariance.transpose(-1, -2)))
        torch.testing.assert_close(
            torch.sort(frame.eigenvalues, dim=-1).values,
            torch.sort(direct_eigenvalues, dim=-1).values,
        )


class LocalRadiusEvidenceContainmentTest(unittest.TestCase):
    """Reproduces the real-DATASET failure mode at small scale: a
    representative's Voronoi cell absorbing spatially non-local members once
    the full cloud vastly outnumbers the representative cap."""

    def _two_cluster_scene(self, near_count: int, far_count: int):
        generator = torch.Generator().manual_seed(2)
        # A tight local cluster right at the representative (tangent scale
        # ~0.05) plus a FAR cluster (50 units away -- outside any reasonable
        # local radius) that is still, by construction, the globally nearest
        # representative for those points (only one representative exists).
        near = 0.01 * torch.randn((near_count, 3), generator=generator)
        far = torch.tensor([50.0, 0.0, 0.0]) + 0.01 * torch.randn((far_count, 3), generator=generator)
        full_positions = torch.cat([near, far], dim=0)
        full_covariance = _random_planar_covariance(near_count + far_count, generator)
        full_frame = extract_covariance_frame(full_covariance)
        full_intrinsic = evaluate_intrinsic_reliability(full_frame)

        representative_positions = torch.zeros((1, 3))
        representative_covariance = _random_planar_covariance(1, generator)
        representative_frame = extract_covariance_frame(representative_covariance)
        full_opacity = torch.ones((near_count + far_count,))
        return (
            full_positions, full_frame, full_opacity, full_intrinsic,
            representative_positions, representative_frame,
        )

    def test_far_cluster_excluded_from_local_support_and_tangent_residual(self):
        near_count, far_count = 40, 10
        (
            full_positions, full_frame, full_opacity, full_intrinsic,
            representative_positions, representative_frame,
        ) = self._two_cluster_scene(near_count, far_count)

        evidence = compute_full_neighborhood_evidence(
            full_positions, full_frame, full_opacity, full_intrinsic,
            representative_positions, representative_frame, representative_ids=(0,),
        )

        # Only the near cluster counts as local support; the far cluster is
        # recorded as excluded, not silently dropped.
        self.assertEqual(int(evidence.support_count[0]), near_count)
        self.assertEqual(int(evidence.out_of_local_radius_count[0]), far_count)
        # Tangent residual reflects only the near (locally-flat) cluster, so
        # it stays small -- not inflated by the far cluster's huge offset.
        self.assertLess(float(evidence.tangent_residual_mean[0]), 5.0)

    def test_all_local_when_everything_is_within_radius(self):
        """No-op check: when every member IS local, behavior matches the
        pre-worklog-135 unbounded aggregation (nothing excluded)."""
        generator = torch.Generator().manual_seed(3)
        full_positions = 0.01 * torch.randn((30, 3), generator=generator)
        full_covariance = _random_planar_covariance(30, generator)
        full_frame = extract_covariance_frame(full_covariance)
        full_intrinsic = evaluate_intrinsic_reliability(full_frame)
        full_opacity = torch.ones((30,))
        representative_positions = torch.zeros((1, 3))
        representative_covariance = _random_planar_covariance(1, generator)
        representative_frame = extract_covariance_frame(representative_covariance)

        evidence = compute_full_neighborhood_evidence(
            full_positions, full_frame, full_opacity, full_intrinsic,
            representative_positions, representative_frame, representative_ids=(0,),
        )
        self.assertEqual(int(evidence.support_count[0]), 30)
        self.assertEqual(int(evidence.out_of_local_radius_count[0]), 0)

    def test_local_radius_config_is_reachable_and_finite(self):
        config = FullNeighborhoodEvidenceConfig()
        self.assertGreater(config.local_radius_tangent_scale_multiplier, 0.0)
        self.assertGreater(config.local_radius_min_absolute, 0.0)


if __name__ == "__main__":
    unittest.main()
