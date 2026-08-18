"""2DGS surfel primitive: Section 4.1 of arXiv:2403.17888v3.

These tests establish that the branch's primitive really is a 2D Gaussian
surface element -- planar by construction, with an intrinsic normal and no
normal-direction scale that anything could grow -- rather than a 3D Gaussian
whose third scale happens to be small.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surfel_analysis_adapter import (
    EPSILON_REGULARIZED,
    EXACT_RANK2,
    surfel_analysis_covariance,
    surfel_evidence_view,
)
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel, build_rotation


def _surfel_model(count: int = 24, seed: int = 7) -> TorchGaussianSurfelModel:
    torch.manual_seed(seed)
    model = TorchGaussianSurfelModel(sh_degree=1, device="cpu")
    model.initialize(
        positions=torch.rand((count, 3)),
        colors=torch.rand((count, 3)),
        scales=0.01 + torch.rand((count, 2)) * 0.1,
        rotations=torch.rand((count, 4)),
    )
    return model


class SurfelParameterizationTest(unittest.TestCase):
    def test_only_two_trainable_scales_exist(self):
        model = _surfel_model()
        self.assertEqual(model.scale_dim, 2)
        self.assertEqual(model._scaling.shape[1], 2)
        self.assertEqual(model.get_scaling.shape[1], 2)
        # Nothing anywhere in the parameter set has a third scale column.
        for name, tensor in model._optimizer_named_params().items() if model.optimizer else []:
            self.assertNotEqual((name, tensor.shape[-1]), ("scaling", 3))

    def test_volumetric_model_keeps_three_scales(self):
        """The shared base class must be unchanged for the vanilla arm."""

        self.assertEqual(TorchGaussianModel.scale_dim, 3)
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(positions=torch.rand((5, 3)), colors=torch.rand((5, 3)))
        self.assertEqual(model._scaling.shape[1], 3)

    def test_normal_is_intrinsic_cross_product_of_tangents(self):
        """t_w = t_u x t_v, derived rather than stored (paper sec. 4.1)."""

        model = _surfel_model()
        expected = torch.cross(model.get_tangent_u, model.get_tangent_v, dim=1)
        torch.testing.assert_close(model.get_normal, expected, atol=1e-5, rtol=1e-5)

    def test_orientation_is_an_orthonormal_right_handed_frame(self):
        model = _surfel_model()
        rotation = model.get_rotation_matrix
        identity = torch.eye(3).expand_as(rotation)
        torch.testing.assert_close(
            rotation @ rotation.transpose(1, 2), identity, atol=1e-5, rtol=1e-5
        )
        determinant = torch.linalg.det(rotation)
        torch.testing.assert_close(determinant, torch.ones_like(determinant), atol=1e-5, rtol=1e-5)

    def test_splat_to_world_reproduces_the_local_parameterization(self):
        """H (u, v, 1) = p_k + s_u t_u u + s_v t_v v  (paper eqs. 4-5)."""

        model = _surfel_model(count=6)
        transform = model.splat_to_world_uv1()
        self.assertEqual(transform.shape, (len(model), 3, 4))
        scaling = model.get_scaling
        rotation = model.get_rotation_matrix
        for u, v in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-0.7, 2.3)):
            local = torch.tensor([u, v, 1.0]).expand(len(model), 3)
            # Upstream stores the transform row-wise with the translation in
            # the last row, so the local point multiplies from the left.
            mapped = torch.einsum("ni,nij->nj", local, transform)
            expected = (
                model.get_xyz
                + scaling[:, 0:1] * rotation[:, :, 0] * u
                + scaling[:, 1:2] * rotation[:, :, 1] * v
            )
            torch.testing.assert_close(mapped[:, :3], expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                mapped[:, 3], torch.ones(len(model)), atol=1e-6, rtol=1e-6
            )

    def test_no_scale_multiplies_the_normal_direction(self):
        """Upstream's placeholder normal row stays UNIT length at any modifier.

        `scale_to_mat` leaves the third diagonal entry at 1, so the row that
        every consumer discards is a unit normal and never a variance. If a
        third learnable scale ever leaked in, this length would move.
        """

        model = _surfel_model(count=6)
        for modifier in (0.5, 1.0, 2.0):
            normal_row = model.get_splat2world(modifier)[:, 2, :3]
            torch.testing.assert_close(
                normal_row.norm(dim=1), torch.ones(len(model)), atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(normal_row, model.get_normal, atol=1e-5, rtol=1e-5)

    def test_scaling_modifier_scales_only_the_tangent_rows(self):
        model = _surfel_model(count=6)
        base = model.get_splat2world(1.0)
        doubled = model.get_splat2world(2.0)
        torch.testing.assert_close(doubled[:, :2, :3], 2.0 * base[:, :2, :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(doubled[:, 3, :], base[:, 3, :], atol=1e-6, rtol=1e-6)

    def test_build_rotation_matches_the_cuda_column_convention(self):
        """`quat_to_rotmat` in the vendored auxiliary.h is column-major glm."""

        quaternion = torch.tensor([[0.3, -0.2, 0.5, 0.78]])
        rotation = build_rotation(quaternion)[0]
        q = torch.nn.functional.normalize(quaternion, dim=-1)[0]
        w, x, y, z = q.tolist()
        # First glm constructor column of `quat_to_rotmat`.
        expected_first_column = torch.tensor(
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
        )
        torch.testing.assert_close(rotation[:, 0], expected_first_column, atol=1e-6, rtol=1e-6)


class SurfelInitializationTest(unittest.TestCase):
    """Official `create_from_pcd`: two tangent scales, random rotations."""

    def test_pipeline_rejects_unknown_primitive(self):
        with self.assertRaises(ValueError):
            TorchOSNGSPipeline(TorchPipelineConfig(primitive="bogus"), device="cpu")

    def test_default_primitive_is_volumetric(self):
        self.assertEqual(TorchPipelineConfig().primitive, "gaussian_3d")
        self.assertFalse(TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu").is_surfel)

    def test_surfel_init_matches_official_create_from_pcd(self):
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(primitive="surfel_2d"), device="cpu")
        torch.manual_seed(0)
        points = torch.rand((40, 3))
        scales, rotations = pipeline._surfel_compatible_scale_rotation(points)

        self.assertEqual(scales.shape, (40, 2))
        self.assertEqual(rotations.shape, (40, 4))
        # Both tangent scales come from the same isotropic nearest-neighbor
        # spacing, exactly as `log(sqrt(dist2))[..., None].repeat(1, 2)`.
        torch.testing.assert_close(scales[:, 0], scales[:, 1])
        expected = torch.sqrt(pipeline._graphdeco_neighbor_mean_dist2(points).clamp_min(1e-7))
        torch.testing.assert_close(scales[:, 0], expected)

    def test_surfel_init_rotations_are_random_not_identity(self):
        """Official 2DGS uses `torch.rand`, unlike 3DGS's identity quaternion."""

        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(primitive="surfel_2d"), device="cpu")
        torch.manual_seed(3)
        _, rotations = pipeline._surfel_compatible_scale_rotation(torch.rand((64, 3)))
        identity = torch.zeros((64, 4))
        identity[:, 0] = 1.0
        self.assertGreater((rotations - identity).abs().max().item(), 0.1)
        self.assertTrue(bool((rotations >= 0).all() and (rotations < 1).all()))

    def test_surfel_pipeline_builds_a_surfel_model(self):
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(primitive="surfel_2d"), device="cpu")
        model = pipeline._new_model()
        self.assertIsInstance(model, TorchGaussianSurfelModel)
        self.assertEqual(model.scale_dim, 2)


class SurfelAnalysisAdapterTest(unittest.TestCase):
    """The covariance-shaped view is an adapter, never the representation."""

    def test_exact_rank2_covariance_has_a_zero_normal_eigenvalue(self):
        model = _surfel_model()
        covariance = surfel_analysis_covariance(model)
        self.assertEqual(covariance.shape, (len(model), 3, 3))
        eigenvalues = torch.linalg.eigvalsh(covariance)
        self.assertLess(eigenvalues[:, 0].abs().max().item(), 1e-8)
        self.assertGreater(eigenvalues[:, 1].min().item(), 0.0)

    def test_exact_rank2_eigenvalues_are_the_squared_tangent_scales(self):
        model = _surfel_model()
        eigenvalues = torch.linalg.eigvalsh(surfel_analysis_covariance(model))
        expected = torch.sort(model.get_scaling.square(), dim=1).values
        torch.testing.assert_close(eigenvalues[:, 1:], expected, atol=1e-6, rtol=1e-5)

    def test_normal_eigenvector_is_the_intrinsic_normal(self):
        model = _surfel_model()
        _, vectors = torch.linalg.eigh(surfel_analysis_covariance(model))
        alignment = (vectors[:, :, 0] * model.get_normal).sum(dim=1).abs()
        torch.testing.assert_close(alignment, torch.ones(len(model)), atol=1e-4, rtol=1e-4)

    def test_epsilon_mode_must_be_requested_explicitly(self):
        model = _surfel_model()
        default_view = surfel_evidence_view(model)
        self.assertEqual(default_view.mode, EXACT_RANK2)
        self.assertEqual(default_view.normal_sigma.abs().max().item(), 0.0)

        regularized = surfel_evidence_view(model, mode=EPSILON_REGULARIZED, epsilon_ratio=1e-3)
        self.assertEqual(regularized.mode, EPSILON_REGULARIZED)
        self.assertGreater(regularized.normal_sigma.min().item(), 0.0)

    def test_adapter_rejects_a_volumetric_model(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(positions=torch.rand((5, 3)), colors=torch.rand((5, 3)))
        with self.assertRaises(ValueError):
            surfel_analysis_covariance(model)

    def test_adapter_output_is_detached(self):
        model = _surfel_model()
        view = surfel_evidence_view(model)
        for tensor in (view.positions, view.covariance, view.normals, view.opacity):
            self.assertFalse(tensor.requires_grad)


if __name__ == "__main__":
    unittest.main()
