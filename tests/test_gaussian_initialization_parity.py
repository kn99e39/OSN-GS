"""Worklog 64: visible Gaussian initialization parity with the Graphdeco
baseline.

``baseline_compatible`` (default) must reproduce
``gaussian-splatting/scene/gaussian_model.py::create_from_pcd`` tensor
semantics exactly: isotropic scale from ``distCUDA2``-equivalent
nearest-neighbor spacing, identity rotation. ``covariance_knn`` is the
pre-worklog-64 local-PCA planar-surfel init, kept only as an explicit
experimental option. Neither mode may change the covariance canonical
visible surface construction/reliability use -- that pathway is fully
separate and untouched by this flag.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig


def _flat_plane_points(n: int = 9) -> torch.Tensor:
    axis = torch.linspace(-0.48, 0.48, n)
    return torch.stack([torch.tensor([x, y, 0.0]) for x in axis for y in axis])


class GaussianInitializationModeValidationTest(unittest.TestCase):
    def test_default_mode_is_baseline_compatible(self):
        self.assertEqual(TorchPipelineConfig().gaussian_initialization_mode, "baseline_compatible")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            TorchOSNGSPipeline(
                TorchPipelineConfig(gaussian_initialization_mode="bogus"), device="cpu"
            )


class BaselineCompatibleScaleRotationTest(unittest.TestCase):
    """Direct tensor-level check against Graphdeco's create_from_pcd formula."""

    def test_isotropic_scale_matches_neighbor_mean_dist2_sqrt(self):
        torch.manual_seed(3)
        points = _flat_plane_points(6)
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
        scales, rotations = pipeline._baseline_compatible_scale_rotation(points)

        # All three axes identical -- isotropic, matching baseline's
        # `torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)`.
        self.assertTrue(torch.allclose(scales[:, 0], scales[:, 1]))
        self.assertTrue(torch.allclose(scales[:, 1], scales[:, 2]))

        expected = torch.sqrt(
            pipeline._graphdeco_neighbor_mean_dist2(points).clamp_min(1e-7)
        )
        self.assertTrue(torch.allclose(scales[:, 0], expected, atol=1e-6))

        # Identity quaternion (w=1, x=y=z=0), matching baseline's
        # `rots[:, 0] = 1`.
        expected_rot = torch.zeros((points.shape[0], 4))
        expected_rot[:, 0] = 1.0
        self.assertTrue(torch.equal(rotations, expected_rot))

    def test_anisotropy_is_unity_at_init(self):
        torch.manual_seed(5)
        points = _flat_plane_points(6)
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
        scales, _ = pipeline._baseline_compatible_scale_rotation(points)
        s_min = scales.min(dim=1).values
        s_max = scales.max(dim=1).values
        self.assertTrue(torch.allclose(s_max / s_min, torch.ones_like(s_min), atol=1e-6))


class GaussianInitializationModePipelineTest(unittest.TestCase):
    """End-to-end: mode flag affects only the trainable model init."""

    def _initialize(self, mode: str):
        torch.manual_seed(11)
        points = _flat_plane_points(9)
        colors = torch.rand(len(points), 3)
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(gaussian_initialization_mode=mode, canonical_covariance_knn=8),
            device="cpu",
        )
        return pipeline, pipeline.initialize(points, colors)

    def test_baseline_compatible_model_init_is_isotropic(self):
        _, state = self._initialize("baseline_compatible")
        scale = state.model.get_scaling.detach()
        s_min = scale.min(dim=1).values
        s_max = scale.max(dim=1).values
        anisotropy = s_max / s_min.clamp_min(1e-12)
        self.assertTrue(torch.allclose(anisotropy, torch.ones_like(anisotropy), atol=1e-4))

        # Identity rotation.
        rotation = state.model.get_rotation.detach()
        self.assertTrue(torch.allclose(rotation[:, 0], torch.ones_like(rotation[:, 0])))
        self.assertTrue(torch.allclose(rotation[:, 1:], torch.zeros_like(rotation[:, 1:])))

    def test_covariance_knn_model_init_stays_anisotropic_planar_surfel(self):
        _, state = self._initialize("covariance_knn")
        scale = state.model.get_scaling.detach()
        s_min = scale.min(dim=1).values
        s_max = scale.max(dim=1).values
        anisotropy = s_max / s_min.clamp_min(1e-12)
        # Planar surfel: normal axis is forced to ~1/25 of the tangent axes
        # by construction (`normal_scale = tangent_scale * 0.04`).
        self.assertTrue(bool((anisotropy > 5.0).all()))

    def test_mode_does_not_change_visible_surface_construction(self):
        """The covariance driving reliability/region-formation/materialization
        is always the local-PCA frame, regardless of gaussian_initialization_mode."""
        _, state_baseline = self._initialize("baseline_compatible")
        _, state_knn = self._initialize("covariance_knn")
        self.assertEqual(
            state_baseline.visible_surface_construction.diagnostic_summary["region_count"],
            state_knn.visible_surface_construction.diagnostic_summary["region_count"],
        )
        self.assertEqual(len(state_baseline.surface_patches), len(state_knn.surface_patches))
        self.assertTrue(
            torch.equal(state_baseline.model.cluster_ids, state_knn.model.cluster_ids)
        )
        self.assertTrue(
            torch.allclose(
                state_baseline.model.surface_uv, state_knn.model.surface_uv, atol=1e-5
            )
        )

    def test_explicit_covariance_override_bypasses_mode_for_both_modes(self):
        torch.manual_seed(13)
        points = _flat_plane_points(9)
        colors = torch.rand(len(points), 3)
        # A real planar-surfel covariance (not a bare isotropic scale) is
        # needed for construction to admit a region; reuse the production
        # local-PCA frame as the caller-supplied override under test.
        reference_pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
        scales, rotations, _ = reference_pipeline._canonical_initial_covariance(
            points, covariance_scales=None, covariance_rotations=None
        )
        for mode in ("baseline_compatible", "covariance_knn"):
            pipeline = TorchOSNGSPipeline(
                TorchPipelineConfig(gaussian_initialization_mode=mode), device="cpu"
            )
            state = pipeline.initialize(
                points, colors, covariance_scales=scales, covariance_rotations=rotations
            )
            self.assertTrue(torch.allclose(state.model.get_scaling.detach(), scales, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
