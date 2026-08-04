from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.torch_scene import TorchScene
from osn_gs.gaussian.torch_density_control import (
    TorchDensityControlConfig,
    add_densification_stats,
    apply_adaptive_density_control,
    should_run_adc,
)
from osn_gs.gaussian.torch_model import GaussianParameterGroups
from osn_gs.losses.torch_losses import nurbs_surface_loss
from osn_gs.surface.torch_voxel_regions import build_torch_voxel_surface_regions
from osn_gs.utils.torch_checkpoint import load_torch_checkpoint, save_torch_checkpoint


class TrainingRegressionTest(unittest.TestCase):
    def _state(self, count: int = 48):
        torch.manual_seed(7)
        axis = torch.linspace(-0.48, 0.48, 9)
        points = torch.stack([torch.tensor([x, y, 0.0]) for x in axis for y in axis])
        colors = torch.rand(len(points), 3)
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(canonical_covariance_knn=8),
            device="cpu",
        )
        state = pipeline.initialize(points, colors)
        if count < len(points):
            keep = torch.arange(len(points)) < count
            state.model.prune(keep)
        return pipeline, state

    def test_surface_rebuild_uses_canonical_pipeline_and_preserves_model(self):
        pipeline, state = self._state(count=81)
        model_identity = id(state.model)
        patch_identity = id(state.surface_patches[0])
        opacity = state.model._opacity.detach().clone()
        scaling = state.model._scaling.detach().clone()
        pipeline.rebuild_surface_from_certain(state)
        self.assertEqual(id(state.model), model_identity)
        self.assertNotEqual(id(state.surface_patches[0]), patch_identity)
        self.assertEqual(state.visible_surface_construction.construction_state, "constructed")
        self.assertTrue(torch.equal(state.model._opacity, opacity))
        self.assertTrue(torch.equal(state.model._scaling, scaling))
        self.assertEqual(state.surface_topology_version, 1)
    def test_surface_loss_patch_minibatch_is_finite_and_trainable(self):
        _, state = self._state()
        trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
        trainer.torch = torch
        trainer.training_config = TorchTrainingConfig(surface_lr=1e-4)
        trainer._setup_surface_optimizer(state)
        state.surface_optimizer.zero_grad(set_to_none=True)
        state.iteration = 1
        loss = nurbs_surface_loss(state, weight=0.01, max_patches=1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        selected = state.surface_patches[1 % len(state.surface_patches)].control_grid
        self.assertIsNotNone(selected.grad)
        self.assertTrue(torch.isfinite(selected.grad).all())

    def test_surface_maintenance_is_full_canonical_reconstruction(self):
        pipeline, state = self._state(count=81)
        report = pipeline.maintain_surface_from_certain(state)
        self.assertTrue(report["topology_changed"])
        self.assertEqual(report["construction_state"], "constructed")
        self.assertGreater(report["checked"], 0)
        self.assertEqual(state.surface_bad_checks, {})
    def test_legacy_constructor_selector_is_removed(self):
        with self.assertRaises(TypeError):
            TorchPipelineConfig(nurbs_constructor_mode="legacy")
        with self.assertRaises(TypeError):
            TorchPipelineConfig(nurbs_constructor_mode="voxel_patch_stage1")
    def test_bounded_canonical_construction_propagates_membership_and_uv(self):
        axis = torch.linspace(-0.48, 0.48, 21)
        points = torch.stack(
            [torch.tensor([x, y, 0.0]) for x in axis for y in axis]
        )
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(canonical_construction_max_points=81),
            device="cpu",
        )
        state = pipeline.initialize(points, torch.full_like(points, 0.5))
        self.assertEqual(
            state.visible_surface_construction.diagnostic_summary[
                "input_gaussian_count"
            ],
            81,
        )
        self.assertTrue(bool((state.model.cluster_ids >= 0).all()))
        self.assertTrue(bool(torch.isfinite(state.model.surface_uv).all()))
        self.assertEqual(tuple(state.model.surface_uv.shape), (len(points), 2))
    def test_surface_optimizer_sync_preserves_existing_adam_rows(self):
        _, state = self._state(count=24)
        trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
        trainer.torch = torch
        trainer.training_config = TorchTrainingConfig(surface_lr=1e-4)
        trainer._setup_surface_optimizer(state)
        existing = state.surface_patches[0].control_grid
        state.surface_optimizer.zero_grad(set_to_none=True)
        existing.square().sum().backward()
        state.surface_optimizer.step()
        old_step = state.surface_optimizer.state[existing]["step"].clone()

        source = state.surface_patches[0]
        state.surface_patches.append(
            type(source)(
                control_grid=source.control_grid.detach().clone(),
                weights=source.weights.detach().clone(),
                degree_u=source.degree_u,
                degree_v=source.degree_v,
                observed_v_max=source.observed_v_max,
            )
        )
        trainer._sync_surface_optimizer(state)
        self.assertTrue(torch.equal(state.surface_optimizer.state[existing]["step"], old_step))
        registered = {
            id(parameter)
            for group in state.surface_optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIn(id(state.surface_patches[-1].control_grid), registered)
        self.assertIn(id(state.surface_patches[-1].weights), registered)

    def test_position_lr_extent_mode_selects_only_position_scale(self):
        trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
        trainer.training_config = TorchTrainingConfig(position_lr_extent_mode="scene")
        self.assertEqual(trainer._position_lr_extent(12.31, 4.92), 12.31)
        trainer.training_config.position_lr_extent_mode = "calibration"
        self.assertEqual(trainer._position_lr_extent(12.31, 4.92), 4.92)
        trainer.training_config.position_lr_extent_mode = "invalid"
        with self.assertRaises(ValueError):
            trainer._position_lr_extent(12.31, 4.92)

    def test_canonical_spacing_uses_three_neighbor_mean(self):
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        pipeline = TorchOSNGSPipeline(
            TorchPipelineConfig(
                covariance_knn_chunk_size=2,
                covariance_max_scale_ratio=10.0,
            ),
            device="cpu",
        )
        mean_dist2 = pipeline._graphdeco_neighbor_mean_dist2(points)
        self.assertAlmostEqual(float(mean_dist2[0]), 110.0 / 3.0, places=5)
    def test_adc_gradient_source_tracks_screen_space_and_fallback(self):
        _, state = self._state(count=4)
        model = state.model
        viewspace = torch.zeros_like(model._xyz, requires_grad=True)
        viewspace.grad = torch.ones_like(viewspace)
        add_densification_stats(model, viewspace, torch.tensor([0, 1]))
        self.assertEqual(model.density_gradient_sources["screen_space"], 1)
        viewspace.grad.zero_()
        model._xyz.grad = torch.ones_like(model._xyz)
        add_densification_stats(model, viewspace, torch.tensor([0, 1]))
        self.assertEqual(model.density_gradient_sources["xyz_fallback"], 1)

    def test_adc_respects_threshold_without_quantile_fallback(self):
        _, state = self._state()
        state.model.training_setup(GaussianParameterGroups())
        state.model.denom.fill_(1.0)
        state.model.xyz_gradient_accum.fill_(1e-6)
        before = len(state.model)
        report = apply_adaptive_density_control(
            state.model,
            TorchDensityControlConfig(densify_grad_threshold=1.0, max_screen_size=0, max_scale_ratio=0),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertEqual(len(state.model), before)
        self.assertEqual(report.cloned + report.split, 0)

    def test_adc_children_inherit_surface_binding(self):
        _, state = self._state(count=8)
        state.model.training_setup(GaussianParameterGroups())
        state.model.surface_uv[:] = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
        state.model.cluster_ids[:] = torch.arange(8)
        state.model.denom.fill_(1.0)
        state.model.xyz_gradient_accum.zero_()
        state.model.xyz_gradient_accum[0] = 1.0
        parent_uv = state.model.surface_uv[0].clone()
        parent_patch = state.model.cluster_ids[0].clone()
        report = apply_adaptive_density_control(
            state.model,
            TorchDensityControlConfig(densify_grad_threshold=0.1, percent_dense=10.0, max_screen_size=0, max_scale_ratio=0),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertEqual(report.cloned, 1)
        self.assertTrue(torch.equal(state.model.surface_uv[-1], parent_uv))
        self.assertEqual(int(state.model.cluster_ids[-1]), int(parent_patch))

    def test_adc_schedule_matches_original_open_boundaries(self):
        config = TorchDensityControlConfig(
            densify_from_iter=500, densify_until_iter=15000, densification_interval=100
        )
        self.assertFalse(should_run_adc(500, config))
        self.assertTrue(should_run_adc(600, config))
        self.assertFalse(should_run_adc(15000, config))

    def test_world_size_pruning_waits_for_size_threshold(self):
        _, state = self._state(count=8)
        state.model.training_setup(GaussianParameterGroups())
        state.model._opacity.data.fill_(0.0)
        state.model._scaling.data.fill_(3.0)
        before = len(state.model)
        report = apply_adaptive_density_control(
            state.model,
            TorchDensityControlConfig(
                densify_grad_threshold=1.0,
                max_screen_size=20.0,
                max_scale_ratio=0.1,
                screen_size_prune_from_iter=3000,
            ),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertEqual(report.pruned_world, 0)
        self.assertEqual(len(state.model), before)

    def test_adc_shape_change_preserves_existing_gradients(self):
        _, state = self._state(count=8)
        state.model.training_setup(GaussianParameterGroups())
        state.model._xyz.grad = torch.ones_like(state.model._xyz)
        state.model.denom.fill_(1.0)
        state.model.xyz_gradient_accum.zero_()
        state.model.xyz_gradient_accum[0] = 1.0
        apply_adaptive_density_control(
            state.model,
            TorchDensityControlConfig(
                densify_grad_threshold=0.1,
                percent_dense=10.0,
                max_screen_size=0,
                max_scale_ratio=0,
            ),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertTrue(torch.equal(state.model._xyz.grad[:-1], torch.ones_like(state.model._xyz.grad[:-1])))
        self.assertTrue(torch.equal(state.model._xyz.grad[-1], torch.zeros_like(state.model._xyz.grad[-1])))

    def test_adc_parity_mode_drops_survivor_gradients(self):
        _, state = self._state(count=8)
        state.model.training_setup(GaussianParameterGroups())
        state.model._xyz.grad = torch.ones_like(state.model._xyz)
        state.model.denom.fill_(1.0)
        state.model.xyz_gradient_accum.zero_()
        state.model.xyz_gradient_accum[0] = 1.0
        apply_adaptive_density_control(
            state.model,
            TorchDensityControlConfig(
                densify_grad_threshold=0.1,
                percent_dense=10.0,
                max_screen_size=0,
                max_scale_ratio=0,
                preserve_adc_gradients=False,
            ),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertIsNone(state.model._xyz.grad)

    def test_adc_clone_split_and_prune_use_one_shape_transaction(self):
        _, state = self._state(count=6)
        model = state.model
        model.training_setup(GaussianParameterGroups())
        model.denom.fill_(1.0)
        model.xyz_gradient_accum.zero_()
        model.xyz_gradient_accum[:2] = 1.0
        model._scaling.data[0].fill_(torch.log(torch.tensor(1e-3)))
        model._scaling.data[1].fill_(torch.log(torch.tensor(0.1)))
        calls = 0
        original = model.replace_tensors

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        model.replace_tensors = counted
        report = apply_adaptive_density_control(
            model,
            TorchDensityControlConfig(
                densify_grad_threshold=0.1,
                percent_dense=0.01,
                split_samples=2,
                max_screen_size=0,
                max_scale_ratio=0,
            ),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(report.cloned, 1)
        self.assertEqual(report.split, 2)
        self.assertEqual(len(model), 8)

    def test_screen_size_pruning_fires_alongside_simultaneous_growth(self):
        _, state = self._state(count=8)
        model = state.model
        model.training_setup(GaussianParameterGroups())
        model.max_radii2D[0] = 25.0
        model.denom.fill_(1.0)
        model.xyz_gradient_accum.zero_()
        model.xyz_gradient_accum[1] = 1.0
        report = apply_adaptive_density_control(
            model,
            TorchDensityControlConfig(
                densify_grad_threshold=0.1,
                percent_dense=10.0,
                max_screen_size=20.0,
                max_scale_ratio=0,
                screen_size_prune_from_iter=0,
            ),
            scene_extent=1.0,
            iteration=1000,
        )
        self.assertGreater(report.cloned, 0)
        self.assertEqual(report.pruned_screen, 1)

    def test_view_sampling_uses_reproducible_epoch_shuffle(self):
        cameras = list(range(7))
        scene = TorchScene(None, None, cameras, torch.zeros((7, 1)), "cpu", view_sampling_seed=17)
        first = [scene._view_indices(iteration, 1)[0] for iteration in range(1, 8)]
        second = [scene._view_indices(iteration, 1)[0] for iteration in range(8, 15)]
        replay = TorchScene(None, None, cameras, torch.zeros((7, 1)), "cpu", view_sampling_seed=17)
        replay_first = [replay._view_indices(iteration, 1)[0] for iteration in range(1, 8)]
        self.assertEqual(sorted(first), list(range(7)))
        self.assertEqual(sorted(second), list(range(7)))
        self.assertNotEqual(first, second)
        self.assertEqual(first, replay_first)

    def test_retired_maintenance_budget_config_is_rejected(self):
        with self.assertRaises(TypeError):
            TorchTrainingConfig(surface_maintenance_patch_budget=3)
    def test_stream_snapshot_deduplicates_same_iteration(self):
        _, state = self._state(count=4)
        trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
        trainer.torch = torch
        trainer.training_config = TorchTrainingConfig(
            stream_cache_dir="unused",
            stream_every=1,
            stream_queue_size=2,
        )
        trainer._stream_queue = queue.Queue(maxsize=2)
        trainer._stream_thread = object()
        trainer._streamed_iterations = {}
        trainer._stream_last_error_at = 0.0
        trainer._ensure_stream_worker = lambda: None
        trainer._stream_copy_event = lambda device: None
        trainer._stream_payload = lambda current, include_nurbs=False: {"iteration": current.iteration}
        state.iteration = 10
        trainer._stream_snapshot(state, include_nurbs=True)
        trainer._stream_snapshot(state, include_nurbs=True)
        self.assertEqual(trainer._stream_queue.qsize(), 1)

    def test_cpu_snapshot_tensor_is_an_immutable_clone(self):
        trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
        trainer.torch = torch
        source = torch.arange(6, dtype=torch.float32)
        snapshot = trainer._snapshot_tensor(source)
        source.zero_()
        self.assertTrue(torch.equal(snapshot, torch.arange(6, dtype=torch.float32)))
        self.assertNotEqual(snapshot.data_ptr(), source.data_ptr())

    def test_checkpoint_round_trip_restores_raw_state(self):
        _, state = self._state()
        groups = GaussianParameterGroups()
        state.model.training_setup(groups)
        params = []
        for patch in state.surface_patches:
            patch.control_grid = patch.control_grid.detach().requires_grad_(True)
            patch.weights = patch.weights.detach().requires_grad_(True)
            params.extend([patch.control_grid, patch.weights])
        state.surface_optimizer = torch.optim.Adam(params, lr=1e-4)
        state.iteration = 123
        expected = state.model._features_dc.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_torch_checkpoint(path, state)
            state.model._features_dc.data.zero_()
            restored = load_torch_checkpoint(path, state, groups, 1e-4)
        self.assertEqual(restored, 123)
        self.assertTrue(torch.equal(state.model._features_dc, expected))
        self.assertEqual(len(state.surface_patch_confidence), len(state.surface_patches))

    def test_stream_payload_includes_renderer_diagnostic_arrays(self):
        _, state = self._state(count=4)
        state.iteration = 9
        state.model.is_uncertain[1] = True
        state.model.surface_uv[0] = torch.tensor([0.25, 0.75])
        state.model.cluster_ids[0] = 3
        state.model.surface_owner_kind[0] = 1
        state.model.surface_owner_id[0] = 3
        trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
        trainer.torch = torch
        trainer.training_config = TorchTrainingConfig(stream_max_gaussians=0)
        payload = trainer._stream_payload(state)
        self.assertEqual(payload["ids"].shape[0], 4)
        self.assertEqual(payload["uncertain"].tolist(), [0, 1, 0, 0])
        self.assertEqual(payload["uncertainConfidences"].shape[0], 4)
        self.assertEqual(payload["surfaceUvs"].shape, (8,))
        self.assertEqual(payload["surfaceUvs"].tolist()[:2], [0.25, 0.75])
        self.assertEqual(payload["clusterIds"].tolist()[0], 3)
        self.assertEqual(payload["surfaceOwnerKinds"].tolist()[0], 1)
        self.assertEqual(payload["surfaceOwnerIds"].tolist()[0], 3)
        # cluster_ids[0]=3 is out of range for this scene's materialized
        # patch count, so its construction-time surfaceConfidences lookup
        # must fall back to the -1.0 "no assigned patch" sentinel.
        self.assertEqual(payload["surfaceConfidences"].shape[0], 4)
        self.assertEqual(payload["surfaceConfidences"].tolist()[0], -1.0)

    def test_surface_patch_confidence_matches_region_confidence(self):
        _, state = self._state(count=81)
        self.assertEqual(len(state.surface_patch_confidence), len(state.surface_patches))
        for confidence in state.surface_patch_confidence:
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_vectorized_ply_preserves_renderer_header(self):
        _, state = self._state(count=4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "point_cloud.ply"
            state.model.save_ply(path, surface_patch_confidence=state.surface_patch_confidence)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertIn("property float scale_0", lines)
        self.assertIn("property float rot_3", lines)
        self.assertIn("property int uncertain", lines)
        self.assertIn("property float uncertain_confidence", lines)
        self.assertIn("property long surface_owner_id", lines)
        self.assertIn("property long stable_gaussian_id", lines)
        self.assertIn("property float surface_confidence", lines)
        self.assertEqual(sum(line.startswith("element vertex 4") for line in lines), 1)


    def test_density_adaptive_voxels_refine_only_dense_coarse_cells(self):
        dense = torch.tensor(
            [[0.05 + 0.01 * i, 0.05 + 0.005 * (i % 3), 0.05] for i in range(12)],
            dtype=torch.float32,
        )
        sparse = torch.tensor([[0.85, 0.85, 0.85], [0.55, 0.15, 0.15]], dtype=torch.float32)
        points = torch.cat((dense, sparse), dim=0)
        regions = build_torch_voxel_surface_regions(
            points,
            grid_resolution=2,
            normal_knn=3,
            adaptive_density=True,
            max_subdivision_depth=1,
            subdivision_quantile=0.75,
        )
        self.assertEqual(int(regions.region_levels.max()), 1)
        self.assertEqual(int(regions.region_levels.min()), 0)
        self.assertTrue(torch.isfinite(regions.region_centers).all())
        self.assertTrue(torch.isfinite(regions.region_normals).all())
        self.assertEqual(int(regions.point_patch_ids.shape[0]), int(points.shape[0]))

    def test_voxel_density_weights_drive_subdivision(self):
        points = torch.tensor(
            [
                [0.05, 0.05, 0.05],
                [0.10, 0.10, 0.10],
                [0.80, 0.80, 0.80],
                [0.85, 0.85, 0.85],
            ],
            dtype=torch.float32,
        )
        weights = torch.tensor([1.0, 1.0, 20.0, 20.0])
        regions = build_torch_voxel_surface_regions(
            points,
            grid_resolution=2,
            normal_knn=3,
            density_weights=weights,
            adaptive_density=True,
            max_subdivision_depth=1,
            subdivision_quantile=0.75,
        )
        high_weight_levels = regions.region_levels[regions.region_centers.mean(dim=1) > 0.5]
        low_weight_levels = regions.region_levels[regions.region_centers.mean(dim=1) < 0.5]
        self.assertTrue(torch.all(high_weight_levels == 1))
        self.assertTrue(torch.all(low_weight_levels == 0))
        self.assertGreater(float(regions.region_density.max()), float(regions.region_density.min()))


if __name__ == "__main__":
    unittest.main()
