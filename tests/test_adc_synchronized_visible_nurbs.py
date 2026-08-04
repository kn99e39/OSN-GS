from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, nurbs_intermediate_payload
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.torch_scene import TorchScene
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig, apply_adaptive_density_control
from osn_gs.gaussian.torch_model import GaussianParameterGroups
from osn_gs.losses.torch_losses import nurbs_surface_loss
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.utils.torch_checkpoint import load_torch_checkpoint, save_torch_checkpoint


def _sheet():
    axis = torch.linspace(-0.48, 0.48, 9)
    points = torch.stack([
        torch.tensor([x, y, 0.04 * (x * x + y * y)])
        for x in axis for y in axis
    ])
    colors = torch.stack((
        (points[:, 0] + 0.5).clamp(0, 1),
        (points[:, 1] + 0.5).clamp(0, 1),
        torch.full((len(points),), 0.5),
    ), dim=1)
    return points, colors


def _scene():
    points, colors = _sheet()
    camera = TorchCamera(
        image_height=12,
        image_width=12,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
        FoVx=0.8,
        FoVy=0.8,
    )
    return TorchScene(points, colors, [camera], torch.zeros((1, 3, 12, 12)), "cpu")


def test_deferred_state_is_explicit_empty_and_disables_surface_loss():
    points, colors = _sheet()
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    state = pipeline.initialize_deferred(points, colors)
    assert state.visible_nurbs_state == "unavailable_until_adc"
    assert state.visible_nurbs_coverage_semantics == "reliable_core_only"
    assert state.surface is None
    assert state.surface_patches == []
    assert state.surface_optimizer is None
    assert torch.all(state.model.cluster_ids == -1)
    loss = nurbs_surface_loss(state)
    assert float(loss) == 0.0
    payload = nurbs_intermediate_payload(state)
    assert payload["patches"] == []
    assert payload["metadata"]["materialized_surface_count"] == 0
    assert payload["metadata"]["visible_nurbs_state"] == "unavailable_until_adc"


def test_post_adc_transaction_is_detached_rng_neutral_and_observed_only():
    points, colors = _sheet()
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    state = pipeline.initialize_deferred(points, colors)
    state.model.is_uncertain[-1] = True
    trainable_before = {
        name: getattr(state.model, name).detach().clone()
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_uncertain_confidence")
    }
    rng_before = torch.random.get_rng_state().clone()
    event = pipeline.reconstruct_visible_after_adc(
        state, iteration=7, reason="structural_adc_post_commit"
    )
    assert event["success"] is True
    assert event["canonical_input_count"] == len(points) - 1
    assert event["excluded_uncertain_count"] == 1
    assert event["materialized_surface_count"] == 1
    assert event["sample_coverage_ratio"] == 1.0
    assert event["full_coverage_ratio"] == 1.0
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for name, expected in trainable_before.items():
        assert torch.equal(getattr(state.model, name), expected), name
    assert state.model._xyz.grad is None
    assert state.surface is not None
    assert state.surface_optimizer is None


def test_failed_reconstruction_clears_stale_registry_optimizer_and_bindings():
    points, colors = _sheet()
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    state = pipeline.initialize(points, colors)
    trainer = TorchOSNGSTrainer.__new__(TorchOSNGSTrainer)
    trainer.torch = torch
    trainer.training_config = TorchTrainingConfig(surface_lr=1e-4)
    trainer._setup_surface_optimizer(state)
    assert state.surface_optimizer is not None
    state.model.is_uncertain[3:] = True
    event = pipeline.reconstruct_visible_after_adc(
        state, iteration=9, reason="structural_adc_post_commit"
    )
    assert event["success"] is False
    assert state.surface is None
    assert state.surface_patches == []
    assert state.surface_optimizer is None
    certain = ~state.model.is_uncertain
    assert torch.all(state.model.cluster_ids[certain] == -1)
    assert torch.all(state.model.surface_uv[certain] == 0)
    payload = nurbs_intermediate_payload(state)
    assert payload["patches"] == []
    assert payload["metadata"]["last_failure"]["reason"]


def test_adc_allocates_unique_stable_ids_and_preserves_survivors():
    points, colors = _sheet()
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    state = pipeline.initialize_deferred(points, colors)
    model = state.model
    model.training_setup(GaussianParameterGroups())
    old_ids = model.stable_gaussian_ids.clone()
    model.denom.fill_(1.0)
    model.xyz_gradient_accum.fill_(1.0)
    report = apply_adaptive_density_control(
        model,
        TorchDensityControlConfig(
            densify_grad_threshold=0.0,
            percent_dense=10.0,
            max_gaussians=len(model) + 5,
            prune_opacity_threshold=0.0,
            max_screen_size=0.0,
            max_scale_ratio=0.0,
        ),
        scene_extent=1.0,
        iteration=1,
    )
    assert report.cloned == 5
    assert torch.equal(model.stable_gaussian_ids[: len(old_ids)], old_ids)
    assert len(set(model.stable_gaussian_ids.tolist())) == len(model)
    assert int(model.stable_gaussian_ids[-1]) >= len(old_ids)


def test_empty_lifecycle_and_stable_ids_round_trip_checkpoint():
    points, colors = _sheet()
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    state = pipeline.initialize_deferred(points, colors)
    state.model.training_setup(GaussianParameterGroups())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.pt"
        save_torch_checkpoint(path, state)
        restored = pipeline.initialize_deferred(points, colors, state_label="placeholder")
        load_torch_checkpoint(path, restored, GaussianParameterGroups(), 1e-4)
    assert restored.surface is None
    assert restored.surface_optimizer is None
    assert restored.visible_nurbs_state == "unavailable_until_adc"
    assert torch.equal(restored.model.stable_gaussian_ids, state.model.stable_gaussian_ids)
    assert restored.model.next_stable_gaussian_id == state.model.next_stable_gaussian_id


def _train(schedule: str, output_dir: Path, *, max_gaussians: int = 300, opacity_reset_interval: int = 0):
    torch.manual_seed(321)
    trainer = TorchOSNGSTrainer(
        pipeline_config=TorchPipelineConfig(),
        training_config=TorchTrainingConfig(
            iterations=3,
            progress_log_interval=0,
            timing_log_interval=0,
            prefer_cuda=False,
            write_output_files=False,
            stream_nurbs=False,
            surface_rebuild_interval=0,
            visible_nurbs_update_schedule=schedule,
            density_control_interval=0,
            density_control=TorchDensityControlConfig(
                densify_from_iter=0,
                densify_until_iter=3,
                densification_interval=1,
                densify_grad_threshold=0.0,
                percent_dense=10.0,
                max_gaussians=max_gaussians,
                prune_opacity_threshold=0.0,
                max_screen_size=0.0,
                max_scale_ratio=0.0,
                opacity_reset_interval=opacity_reset_interval,
            ),
        ),
        rasterizer_config=GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True),
        device="cpu",
    )
    return trainer.train(_scene(), output_dir)


def test_opacity_reset_without_shape_change_does_not_trigger_reconstruction():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "reset_only"
        _train(
            "adc_post_commit",
            output,
            max_gaussians=81,
            opacity_reset_interval=1,
        )
        events = [
            json.loads(line)
            for line in (output / "visible_nurbs_adc_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [event["reason"] for event in events] == ["training_terminal"]


def test_multi_adc_schedule_records_events_and_keeps_gaussian_control_equal():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline = _train("disabled", root / "baseline")
        experiment = _train("adc_post_commit", root / "experiment")
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_uncertain_confidence"):
            assert torch.equal(
                getattr(baseline.state.model, name),
                getattr(experiment.state.model, name),
            ), name
        lines = (root / "experiment" / "visible_nurbs_adc_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        events = [json.loads(line) for line in lines]
        structural = [event for event in events if event["reason"] == "structural_adc_post_commit"]
        assert len(structural) >= 2
        assert all(event["adc_post_count"] != event["adc_pre_count"] for event in structural)
        assert all("source_fingerprint" in event for event in events)
        assert all("sample_coverage_ratio" in event or not event["success"] for event in events)
        assert events[-1]["reason"] in {"structural_adc_post_commit", "training_terminal"}
        assert experiment.state.visible_nurbs_last_attempt_iteration == 3