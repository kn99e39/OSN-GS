"""Checkpoint/model contract for the 2DGS surfel primitive.

A surfel checkpoint stores a two-column `scaling`; a volumetric one stores
three. The branch resolves that incompatibility explicitly -- by recording the
primitive and failing closed on a mismatch -- rather than by reconstructing
fake 3D covariance state.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.gaussian.torch_model import GaussianParameterGroups, TorchGaussianModel
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.surface.torch_nurbs import TorchCurveSet
from osn_gs.utils.torch_checkpoint import load_torch_checkpoint, save_torch_checkpoint


def _state(cls, count: int = 12, seed: int = 5) -> TorchPipelineState:
    torch.manual_seed(seed)
    model = cls(sh_degree=1, device="cpu")
    model.initialize(
        positions=torch.rand((count, 3)),
        colors=torch.rand((count, 3)),
        scales=0.02 + torch.rand((count, cls.scale_dim)) * 0.1,
        rotations=torch.rand((count, 4)),
    )
    model.spatial_lr_scale = 1.0
    model.training_setup(GaussianParameterGroups())
    empty = TorchCurveSet(
        control_points=torch.zeros((0, 4, 3)), observed=torch.zeros((0,), dtype=torch.bool)
    )
    return TorchPipelineState(
        model=model,
        base_curves=empty,
        occlusion_curves=empty,
        surface=None,
        surface_patches=[],
        iteration=42,
    )


class SurfelCheckpointRoundTripTest(unittest.TestCase):
    def test_save_load_reproduces_the_surfel_state(self):
        source = _state(TorchGaussianSurfelModel)
        source.model.active_sh_degree = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_torch_checkpoint(path, source)

            restored = _state(TorchGaussianSurfelModel, seed=99)
            iteration = load_torch_checkpoint(
                path, restored, GaussianParameterGroups(), surface_lr=1e-4
            )

        self.assertEqual(iteration, 42)
        self.assertEqual(restored.model.scale_dim, 2)
        self.assertEqual(restored.model._scaling.shape[1], 2)
        for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"):
            torch.testing.assert_close(
                getattr(restored.model, name).detach(), getattr(source.model, name).detach()
            )
        torch.testing.assert_close(
            restored.model.stable_gaussian_ids, source.model.stable_gaussian_ids
        )
        self.assertEqual(restored.model.active_sh_degree, 1)
        # The derived normal survives because the orientation does.
        torch.testing.assert_close(restored.model.get_normal, source.model.get_normal)

    def test_checkpoint_records_the_primitive(self):
        state = _state(TorchGaussianSurfelModel)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_torch_checkpoint(path, state)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["primitive_class"], "TorchGaussianSurfelModel")
        self.assertEqual(payload["scale_dim"], 2)
        self.assertEqual(tuple(payload["model_raw"]["scaling"].shape)[1], 2)

    def test_volumetric_checkpoint_records_three_scales(self):
        state = _state(TorchGaussianModel)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_torch_checkpoint(path, state)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["primitive_class"], "TorchGaussianModel")
        self.assertEqual(payload["scale_dim"], 3)


class SurfelCheckpointMismatchTest(unittest.TestCase):
    def test_loading_a_surfel_checkpoint_into_a_volumetric_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_torch_checkpoint(path, _state(TorchGaussianSurfelModel))
            with self.assertRaises(ValueError) as context:
                load_torch_checkpoint(
                    path, _state(TorchGaussianModel), GaussianParameterGroups(), surface_lr=1e-4
                )
        self.assertIn("primitive mismatch", str(context.exception))

    def test_loading_a_volumetric_checkpoint_into_a_surfel_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_torch_checkpoint(path, _state(TorchGaussianModel))
            with self.assertRaises(ValueError) as context:
                load_torch_checkpoint(
                    path,
                    _state(TorchGaussianSurfelModel),
                    GaussianParameterGroups(),
                    surface_lr=1e-4,
                )
        self.assertIn("primitive mismatch", str(context.exception))

    def test_legacy_checkpoint_without_the_field_is_treated_as_volumetric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_torch_checkpoint(path, _state(TorchGaussianModel))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["scale_dim"]
            del payload["primitive_class"]
            torch.save(payload, path)
            iteration = load_torch_checkpoint(
                path, _state(TorchGaussianModel), GaussianParameterGroups(), surface_lr=1e-4
            )
        self.assertEqual(iteration, 42)


class SurfelPlyTest(unittest.TestCase):
    def test_ply_declares_exactly_two_scale_properties(self):
        state = _state(TorchGaussianSurfelModel, count=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            state.model.save_ply(path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("property float scale_0", text)
        self.assertIn("property float scale_1", text)
        self.assertNotIn("property float scale_2", text)
        header_end = text.index("end_header")
        properties = text[:header_end].count("property")
        first_row = text[header_end:].splitlines()[1].split()
        self.assertEqual(len(first_row), properties)

    def test_volumetric_ply_still_declares_three_scale_properties(self):
        state = _state(TorchGaussianModel, count=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            state.model.save_ply(path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("property float scale_2", text)


if __name__ == "__main__":
    unittest.main()
