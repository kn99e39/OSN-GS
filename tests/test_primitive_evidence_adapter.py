"""The single evidence interface both arms of the 2DGS comparison go through.

The comparison is only meaningful if a volumetric and a surfel checkpoint
reach the SAME downstream constructor through the SAME tuple. These tests pin
that, and pin the two properties that make the surfel side honest: the
covariance it reports is the true rank-2 geometry, and the epsilon-regularized
surrogate can only be obtained by asking for it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.gaussian.torch_model import GaussianParameterGroups, TorchGaussianModel
from osn_gs.gaussian.torch_primitive_evidence_adapter import (
    PRIMITIVE_GAUSSIAN_3D,
    PRIMITIVE_SURFEL_2D,
    VOLUMETRIC_COVARIANCE_MODE,
    checkpoint_primitive,
    load_primitive_evidence,
    load_primitive_model,
)
from osn_gs.gaussian.torch_surfel_analysis_adapter import EPSILON_REGULARIZED, EXACT_RANK2
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_nurbs import TorchCurveSet
from osn_gs.utils.torch_checkpoint import save_torch_checkpoint


def _write_checkpoint(directory: Path, cls, count: int = 24, seed: int = 4) -> Path:
    torch.manual_seed(seed)
    model = cls(sh_degree=1, device="cpu")
    model.initialize(
        positions=torch.rand((count, 3)),
        colors=torch.rand((count, 3)),
        opacities=torch.full((count, 1), 0.4),
        scales=0.02 + torch.rand((count, cls.scale_dim)) * 0.1,
        rotations=torch.rand((count, 4)),
    )
    model.spatial_lr_scale = 1.0
    model.training_setup(GaussianParameterGroups())
    empty = TorchCurveSet(
        control_points=torch.zeros((0, 4, 3)), observed=torch.zeros((0,), dtype=torch.bool)
    )
    state = TorchPipelineState(
        model=model, base_curves=empty, occlusion_curves=empty,
        surface=None, surface_patches=[], iteration=1234,
    )
    directory.mkdir(parents=True, exist_ok=True)
    save_torch_checkpoint(directory / "checkpoint.pt", state)
    return directory


class PrimitiveDispatchTest(unittest.TestCase):
    def test_dispatch_reads_the_checkpoint_not_the_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volumetric = _write_checkpoint(root / "vol", TorchGaussianModel)
            surfel = _write_checkpoint(root / "surf", TorchGaussianSurfelModel)

            volumetric_model, volumetric_payload = load_primitive_model(volumetric, device="cpu")
            surfel_model, surfel_payload = load_primitive_model(surfel, device="cpu")

        self.assertEqual(checkpoint_primitive(volumetric_payload), PRIMITIVE_GAUSSIAN_3D)
        self.assertEqual(checkpoint_primitive(surfel_payload), PRIMITIVE_SURFEL_2D)
        self.assertIsInstance(surfel_model, TorchGaussianSurfelModel)
        self.assertNotIsInstance(volumetric_model, TorchGaussianSurfelModel)
        self.assertEqual(surfel_model._scaling.shape[1], 2)
        self.assertEqual(volumetric_model._scaling.shape[1], 3)

    def test_legacy_checkpoint_without_the_field_dispatches_to_volumetric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "legacy", TorchGaussianModel)
            payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
            del payload["scale_dim"]
            del payload["primitive_class"]
            torch.save(payload, path / "checkpoint.pt")
            self.assertEqual(checkpoint_primitive(payload), PRIMITIVE_GAUSSIAN_3D)
            evidence = load_primitive_evidence(path, device="cpu")
        self.assertEqual(evidence.primitive, PRIMITIVE_GAUSSIAN_3D)


class CommonEvidenceContractTest(unittest.TestCase):
    def test_both_primitives_produce_the_same_tuple_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volumetric = load_primitive_evidence(
                _write_checkpoint(root / "vol", TorchGaussianModel), device="cpu"
            )
            surfel = load_primitive_evidence(
                _write_checkpoint(root / "surf", TorchGaussianSurfelModel), device="cpu"
            )

        for evidence in (volumetric, surfel):
            self.assertEqual(evidence.positions.shape, (24, 3))
            self.assertEqual(evidence.covariance.shape, (24, 3, 3))
            self.assertEqual(evidence.opacity.shape, (24,))
            self.assertEqual(evidence.normals.shape, (24, 3))
            self.assertEqual(evidence.tangent_scales.shape, (24, 2))
            self.assertEqual(evidence.normal_scale.shape, (24,))
            self.assertEqual(evidence.iteration, 1234)
            torch.testing.assert_close(
                evidence.normals.norm(dim=1), torch.ones(24), atol=1e-5, rtol=1e-5
            )

    def test_volumetric_covariance_is_the_pre_existing_formula(self):
        """The vanilla arm's numbers must stay comparable with the baseline."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "vol", TorchGaussianModel)
            evidence = load_primitive_evidence(path, device="cpu")
            model = evidence.model
            expected = covariance_from_scale_rotation(
                model.get_scaling.detach(), model.get_rotation.detach()
            )
        self.assertEqual(evidence.covariance_mode, VOLUMETRIC_COVARIANCE_MODE)
        torch.testing.assert_close(evidence.covariance, expected)

    def test_volumetric_normal_is_the_minor_principal_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "vol", TorchGaussianModel)
            evidence = load_primitive_evidence(path, device="cpu")
        _, vectors = torch.linalg.eigh(evidence.covariance)
        alignment = (vectors[:, :, 0] * evidence.normals).sum(dim=1).abs()
        torch.testing.assert_close(alignment, torch.ones(24), atol=1e-4, rtol=1e-4)

    def test_surfel_normal_is_the_intrinsic_tangent_plane_normal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "surf", TorchGaussianSurfelModel)
            evidence = load_primitive_evidence(path, device="cpu")
        torch.testing.assert_close(evidence.normals, evidence.model.get_normal)

    def test_surfel_per_primitive_thickness_is_exactly_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "surf", TorchGaussianSurfelModel)
            evidence = load_primitive_evidence(path, device="cpu")
        self.assertEqual(evidence.covariance_mode, EXACT_RANK2)
        self.assertEqual(evidence.normal_scale.abs().max().item(), 0.0)
        eigenvalues = torch.linalg.eigvalsh(evidence.covariance)
        self.assertLess(eigenvalues[:, 0].abs().max().item(), 1e-8)

    def test_epsilon_regularization_must_be_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "surf", TorchGaussianSurfelModel)
            default = load_primitive_evidence(path, device="cpu")
            regularized = load_primitive_evidence(
                path, device="cpu",
                surfel_covariance_mode=EPSILON_REGULARIZED, surfel_epsilon_ratio=1e-2,
            )
        self.assertEqual(default.covariance_mode, EXACT_RANK2)
        self.assertEqual(regularized.covariance_mode, EPSILON_REGULARIZED)
        self.assertGreater(regularized.normal_scale.min().item(), 0.0)
        expected = 1e-2 * regularized.model.get_scaling.detach().min(dim=1).values
        torch.testing.assert_close(regularized.normal_scale, expected)

    def test_evidence_is_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_checkpoint(Path(directory) / "surf", TorchGaussianSurfelModel)
            evidence = load_primitive_evidence(path, device="cpu")
        for tensor in (evidence.positions, evidence.covariance, evidence.opacity, evidence.normals):
            self.assertFalse(tensor.requires_grad)


if __name__ == "__main__":
    unittest.main()
