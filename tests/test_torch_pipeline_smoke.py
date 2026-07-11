from __future__ import annotations

"""Smoke test for the active Torch OSN-GS pipeline (Stage 1 visible reconstruction).

Exercises the real training path used by ``train.py`` /
``scripts/train_osn_gs_torch.py``: voxel surface regioning, base curve
fitting, NURBS surface fitting/evaluation, Gaussian model initialization,
and one training iteration through ``TorchOSNGSTrainer``.
"""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required for the torch pipeline smoke test")
class TorchPipelineSmokeTest(unittest.TestCase):
    def _synthetic_points(self, count: int = 400):
        torch.manual_seed(0)
        xy = torch.rand(count, 2) * 2 - 1
        z = 0.2 * torch.sin(xy[:, 0] * 2)
        points = torch.cat([xy, z.unsqueeze(1)], dim=1)
        colors = torch.rand(count, 3)
        return points, colors

    def test_pipeline_initialize_builds_surface_and_gaussians(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig

        points, colors = self._synthetic_points()
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(voxel_grid_resolution=6, base_curve_count=4, visible_surface_resolution_u=6, visible_surface_resolution_v=3),
            device="cpu",
        )
        state = pipeline.initialize(points, colors)

        self.assertEqual(len(state.model), points.shape[0])
        self.assertIsNotNone(state.voxel_regions)
        self.assertGreater(int(state.voxel_regions.region_centers.shape[0]), 0)

        uv = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        surface_points = state.surface.evaluate(uv)
        self.assertEqual(tuple(surface_points.shape), (3, 3))
        self.assertTrue(torch.isfinite(surface_points).all())

    def test_trainer_runs_one_iteration(self):
        from osn_gs.core.torch_pipeline import TorchPipelineConfig
        from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
        from osn_gs.data.torch_scene import TorchScene
        from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
        from osn_gs.render.torch_fallback import TorchCamera

        points, colors = self._synthetic_points(count=200)
        camera = TorchCamera(
            image_height=16,
            image_width=16,
            world_view_transform=torch.eye(4),
            full_proj_transform=torch.eye(4),
            camera_center=torch.zeros(3),
            FoVx=0.8,
            FoVy=0.8,
        )
        images = torch.rand(1, 3, 16, 16)
        scene = TorchScene(
            initial_points=points,
            initial_colors=colors,
            cameras=[camera],
            images=images,
            device="cpu",
        )

        trainer = TorchOSNGSTrainer(
            pipeline_config=TorchPipelineConfig(voxel_grid_resolution=6, base_curve_count=4),
            training_config=TorchTrainingConfig(
                iterations=1,
                progress_log_interval=0,
                timing_log_interval=0,
                prefer_cuda=False,
                write_output_files=False,
            ),
            rasterizer_config=GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True),
            device="cpu",
        )
        result = trainer.train(scene, output_dir="_test_output_unused")

        self.assertEqual(result.state.iteration, 1)
        self.assertTrue(torch.isfinite(torch.tensor(result.state.last_loss)))


if __name__ == "__main__":
    unittest.main()
