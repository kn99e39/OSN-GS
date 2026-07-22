from __future__ import annotations

"""Torch-based OSN-GS training loop."""

from dataclasses import dataclass, field
import json
import queue
import threading
import time
from typing import Any
from pathlib import Path

from osn_gs.core.torch_pipeline import (
    TorchOSNGSPipeline,
    TorchPipelineConfig,
    TorchPipelineState,
    nurbs_intermediate_payload,
)
from osn_gs.data.colmap_scene import estimate_scene_extent
from osn_gs.data.torch_scene import TorchScene
from osn_gs.gaussian.torch_model import GaussianParameterGroups
from osn_gs.gaussian.torch_density_control import (
    TorchDensityControlConfig,
    add_densification_stats,
    apply_adaptive_density_control,
    apply_uncertain_density_control,
    should_run_adc,
    update_max_radii,
)
from osn_gs.losses.torch_losses import (
    image_reconstruction_loss,
    nurbs_surface_loss,
    uncertain_anchor_loss,
    uncertain_confidence_loss,
)
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.utils.torch_checkpoint import load_torch_checkpoint, save_torch_checkpoint
from osn_gs.utils.torch_ops import default_device, psnr_from_mse, require_torch, sh_dc_to_rgb


@dataclass
class TorchTrainingConfig:
    """Controls the training loop and loss weights."""

    iterations: int = 1000
    batch_size: int = 1
    train_resolution_scale: int = 1
    # Image loss weight, matching original 3DGS: (1 - lambda_dssim)*L1 + lambda_dssim*(1 - SSIM).
    lambda_dssim: float = 0.2
    lambda_surface: float = 0.01
    lambda_uncertainty: float = 0.05
    lambda_anchor: float = 0.01
    # 0 evaluates all patches. Positive values rotate a bounded NURBS patch minibatch.
    surface_loss_patch_budget: int = 16
    surface_lr: float = 1.0e-4
    sh_increment_interval: int = 1000
    # Compatibility name: this is now a quality-check interval, not a global rebuild.
    surface_rebuild_interval: int = 1000
    # 0 checks every patch. Positive values rotate a bounded maintenance set.
    surface_maintenance_patch_budget: int = 16
    surface_residual_ratio_threshold: float = 0.03
    surface_residual_patience: int = 3
    surface_local_min_gaussians: int = 64
    surface_local_min_component: int = 16
    enable_local_surface_correction: bool = True
    density_control_interval: int = 500
    save_interval: int = 1000
    save_iterations: tuple[int, ...] = ()
    progress_log_interval: int = 100
    timing_log_interval: int = 100
    stream_url: str = ""
    stream_every: int = 1
    stream_iterations: tuple[int, ...] = ()
    stream_max_gaussians: int = 0
    stream_nurbs: bool = True
    stream_cache_dir: str = ""
    # Bounds full-scene pinned-memory snapshots waiting for serialization/I/O.
    stream_queue_size: int = 2
    write_output_files: bool = True
    resume_checkpoint: str = ""
    prefer_cuda: bool = True
    parameter_groups: GaussianParameterGroups = field(default_factory=GaussianParameterGroups)
    density_control: TorchDensityControlConfig = field(default_factory=TorchDensityControlConfig)


@dataclass
class TorchTrainingResult:
    """Minimal result bundle returned after training."""

    state: TorchPipelineState
    output_dir: Path


class TorchOSNGSTrainer:
    """End-to-end optimization runner for OSN-GS."""

    def __init__(
        self,
        pipeline_config: TorchPipelineConfig | None = None,
        training_config: TorchTrainingConfig | None = None,
        rasterizer_config: GaussianRasterizerConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.torch = require_torch()
        self.training_config = training_config or TorchTrainingConfig()
        self.device = device or default_device(self.training_config.prefer_cuda)
        self.pipeline = TorchOSNGSPipeline(pipeline_config or TorchPipelineConfig(), device=self.device)
        self.rasterizer = OSNGaussianRasterizer(rasterizer_config)
        self._stream_socket: Any | None = None
        self._stream_last_error_at = 0.0
        self._streamed_nurbs_signature: tuple[int, tuple[int, ...]] | None = None
        self._stream_queue: queue.Queue[Any] | None = None
        self._stream_thread: threading.Thread | None = None
        self._streamed_iterations: dict[int, bool] = {}
        print(f"OSN-GS rasterizer backend: {self.rasterizer.backend_source}", flush=True)

    def train(self, scene: TorchScene, output_dir: str | Path) -> TorchTrainingResult:
        """Train the scene and save previews, checkpoints, and point clouds."""

        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state = self.pipeline.initialize(scene.initial_points, scene.initial_colors)
        scene_extent = self._scene_extent(scene.initial_points)
        state.model.spatial_lr_scale = scene_extent
        state.model.training_setup(self.training_config.parameter_groups)
        self._setup_surface_optimizer(state)
        start_iteration = 1
        if str(self.training_config.resume_checkpoint).strip():
            restored = load_torch_checkpoint(
                self.training_config.resume_checkpoint,
                state,
                self.training_config.parameter_groups,
                self.training_config.surface_lr,
            )
            start_iteration = restored + 1
            print(f"OSN-GS resumed: checkpoint={self.training_config.resume_checkpoint} iteration={restored}", flush=True)
        background = torch.zeros((3,), dtype=torch.float32, device=self.device)
        train_wall_start = time.perf_counter()
        for iteration in range(start_iteration, self.training_config.iterations + 1):
            state.model.update_learning_rate(iteration)
            timed = self._should_log_timing(iteration)
            iter_start = self._time_now(timed)
            phase_start = iter_start
            timings: dict[str, float] = {}
            if iteration % self.training_config.sh_increment_interval == 0:
                state.model.oneup_sh_degree()

            batch = scene.sample_views(iteration, self.training_config.batch_size)
            self._record_timing(timings, "sample", phase_start, timed)
            phase_start = self._time_now(timed)
            total = torch.zeros((), dtype=torch.float32, device=self.device)
            # Keep the MSE accumulator on-device; forcing a host scalar per view
            # here would serialize the hot path on GPU→CPU synchronization.
            mse_accum = torch.zeros((), dtype=torch.float32, device=self.device)
            render_packages = []
            for camera, target in zip(batch.cameras, batch.images):
                camera, target = self._prepare_training_view(camera, target)
                render_pkg = self.rasterizer.render(camera, state.model, background)
                render_packages.append(render_pkg)
                image = render_pkg["render"]
                target = target.to(device=self.device, dtype=torch.float32)

                image_loss, mse = image_reconstruction_loss(
                    image,
                    target,
                    self.training_config.lambda_dssim,
                )
                total = total + image_loss
                mse_accum = mse_accum + mse.detach()
            num_cameras = max(len(batch.cameras), 1)
            total = total / num_cameras
            self._record_timing(timings, "render_loss", phase_start, timed)
            phase_start = self._time_now(timed)

            # Mean MSE stays a device tensor: it feeds the uncertainty loss without
            # a GPU→CPU→GPU round trip and is only materialized to a host float
            # when a logging/streaming/saving cadence actually needs it.
            mean_mse = mse_accum / num_cameras
            total = total + self._surface_losses(state, mean_mse)
            self._record_timing(timings, "surface_loss", phase_start, timed)
            phase_start = self._time_now(timed)

            state.model.optimizer.zero_grad(set_to_none=True)
            if state.surface_optimizer is not None:
                state.surface_optimizer.zero_grad(set_to_none=True)
            total.backward()
            self._accumulate_density_stats(state, render_packages)
            self._record_timing(timings, "backward", phase_start, timed)
            phase_start = self._time_now(timed)
            if state.surface_optimizer is not None:
                state.surface_optimizer.step()
                with torch.no_grad():
                    for patch in state.surface_patches:
                        patch.weights.clamp_(min=1e-3, max=1e3)

            self._record_timing(timings, "optim", phase_start, timed)
            phase_start = self._time_now(timed)
            state.iteration = iteration
            # Only pay the GPU→CPU synchronization for metric scalars on
            # iterations that a progress log, stream snapshot, or file save reads.
            if self._needs_metric_scalars(iteration):
                state.last_loss = float(total.detach().cpu())
                state.last_psnr = psnr_from_mse(float(mean_mse.detach().cpu()))

            if self.training_config.surface_rebuild_interval > 0 and iteration % self.training_config.surface_rebuild_interval == 0:
                report = self.pipeline.maintain_surface_from_certain(
                    state,
                    residual_ratio_threshold=self.training_config.surface_residual_ratio_threshold,
                    residual_patience=self.training_config.surface_residual_patience,
                    local_min_gaussians=self.training_config.surface_local_min_gaussians,
                    local_min_component=self.training_config.surface_local_min_component,
                    enable_local_correction=self.training_config.enable_local_surface_correction,
                    patch_ids=self._maintenance_patch_ids(state),
                )
                if report["topology_changed"]:
                    self._sync_surface_optimizer(state)
                print(
                    "OSN-GS surface maintenance: "
                    f"iteration={iteration} checked={report['checked']} "
                    f"max_residual={report['max_residual_ratio']:.6g} "
                    f"candidates={len(report['candidates'])} "
                    f"corrected={len(report['corrected'])} "
                    f"patches={report['patches']} "
                    f"uv_refreshed={report.get('uv_refreshed', 0)}",
                    flush=True,
                )

            if should_run_adc(iteration, self.training_config.density_control):
                adc_before = self._adc_stats(state)
                report = apply_adaptive_density_control(
                    state.model, self.training_config.density_control, scene_extent, iteration=iteration
                )
                print(
                    "OSN-GS ADC: "
                    f"iteration={iteration} cloned={report.cloned} split={report.split} "
                    f"pruned={report.pruned} opacity={report.pruned_opacity} "
                    f"screen={report.pruned_screen} world={report.pruned_world} gaussians={len(state.model)} "
                    f"tracked={adc_before['tracked']} max_grad={adc_before['max_grad']:.6g} "
                    f"mean_grad={adc_before['mean_grad']:.6g} threshold={adc_before['threshold']:.6g}",
                    flush=True,
                )
            reset_interval = int(self.training_config.density_control.opacity_reset_interval)
            densify_until = int(self.training_config.density_control.densify_until_iter)
            if (
                reset_interval > 0
                and iteration < densify_until
                and iteration % reset_interval == 0
            ):
                state.model.reset_opacity()
                print(f"OSN-GS ADC: iteration={iteration} opacity_reset=0.01", flush=True)

            if self.training_config.density_control_interval > 0 and iteration % self.training_config.density_control_interval == 0:
                report = apply_uncertain_density_control(state.model, self.training_config.density_control)
                if report.changed:
                    print(
                        "OSN-GS uncertain cleanup: "
                        f"iteration={iteration} pruned={report.uncertain_pruned} gaussians={len(state.model)}",
                        flush=True,
                    )
            state.model.optimizer.step()
            self._clamp_uncertain_confidence(state)
            self._record_timing(timings, "density", phase_start, timed)
            phase_start = self._time_now(timed)

            self._stream_snapshot(state, include_nurbs=self._should_stream_nurbs(state))
            if self.training_config.write_output_files and self._should_save_iteration(iteration):
                self.save_outputs(state, output_dir / str(iteration), batch.cameras[0])
            self._record_timing(timings, "save", phase_start, timed)
            phase_start = self._time_now(timed)

            if self._should_log_progress(iteration):
                self._log_progress(state)
            self._record_timing(timings, "log", phase_start, timed)
            if timed:
                timings["total"] = self._elapsed(iter_start)
                timings["avg_iter"] = (time.perf_counter() - train_wall_start) / max(iteration, 1)
                self._log_timing(iteration, timings)

        self._stream_snapshot(state, include_nurbs=self._should_stream_nurbs(state, force=True))
        self._finish_stream_worker()
        self._close_stream_socket()
        if self.training_config.write_output_files:
            self.save_outputs(state, output_dir / "final", scene.cameras[0])
        return TorchTrainingResult(state=state, output_dir=output_dir)

    def _maintenance_patch_ids(self, state: TorchPipelineState) -> tuple[int, ...] | None:
        """Rotate a bounded patch set at each maintenance checkpoint."""

        count = len(state.surface_patches)
        budget = max(0, int(self.training_config.surface_maintenance_patch_budget))
        if budget == 0 or budget >= count:
            return None
        interval = max(1, int(self.training_config.surface_rebuild_interval))
        maintenance_pass = max(0, int(state.iteration) // interval - 1)
        start = (maintenance_pass * budget) % count
        return tuple((start + offset) % count for offset in range(budget))

    def _needs_metric_scalars(self, iteration: int) -> bool:
        """True when this iteration's loss/PSNR host scalars will be read.

        Consumers are the progress log, the stream snapshot metadata, and the
        saved metrics file. On every other iteration the scalars stay on-device
        and no GPU→CPU synchronization is forced.
        """

        return (
            self._should_log_progress(iteration)
            or self._should_stream_iteration(iteration)
            or (self.training_config.write_output_files and self._should_save_iteration(iteration))
        )

    def _should_save_iteration(self, iteration: int) -> bool:
        if self.training_config.save_iterations:
            return iteration in set(int(value) for value in self.training_config.save_iterations)
        interval = int(self.training_config.save_interval)
        if interval <= 1:
            return False
        return iteration % interval == 0

    def _should_stream_iteration(self, iteration: int) -> bool:
        if not self.training_config.stream_url and not self.training_config.stream_cache_dir:
            return False
        if iteration in set(int(value) for value in self.training_config.stream_iterations):
            return True
        interval = max(0, int(self.training_config.stream_every))
        return interval > 0 and iteration % interval == 0

    def _should_stream_nurbs(self, state: TorchPipelineState, force: bool = False) -> bool:
        if not self.training_config.stream_nurbs:
            return False
        if force:
            return True
        shape = tuple(int(value) for value in state.surface.control_grid.shape)
        signature = (int(state.iteration), shape)
        if self._streamed_nurbs_signature is None:
            return True
        if self.training_config.surface_rebuild_interval > 0 and state.iteration % self.training_config.surface_rebuild_interval == 0:
            return True
        return False

    def _get_stream_socket(self):
        if self._stream_socket is not None:
            return self._stream_socket
        from websockets.sync.client import connect

        self._stream_socket = connect(self.training_config.stream_url, max_size=None, open_timeout=10, close_timeout=2)
        try:
            self._stream_socket.recv(timeout=1)
        except Exception:
            pass
        print(f"[WS] connected to renderer WebSocket: {self.training_config.stream_url}", flush=True)
        return self._stream_socket

    def _close_stream_socket(self) -> None:
        if self._stream_socket is None:
            return
        try:
            self._stream_socket.close()
        except Exception:
            pass
        self._stream_socket = None

    def _ensure_stream_worker(self) -> None:
        if self._stream_thread is not None:
            return
        self._stream_queue = queue.Queue(maxsize=max(1, int(self.training_config.stream_queue_size)))
        self._stream_thread = threading.Thread(target=self._stream_worker, name="osn-gs-stream", daemon=True)
        self._stream_thread.start()

    def _finish_stream_worker(self) -> None:
        if self._stream_queue is None or self._stream_thread is None:
            return
        self._stream_queue.put(None)
        self._stream_thread.join()
        self._stream_queue = None
        self._stream_thread = None

    def _stream_snapshot(self, state: TorchPipelineState, include_nurbs: bool = False) -> None:
        iteration = int(state.iteration)
        if not self._should_stream_iteration(iteration):
            return
        previous = self._streamed_iterations.get(iteration)
        if previous is not None and (previous or not include_nurbs):
            return
        try:
            self._ensure_stream_worker()
            assert self._stream_queue is not None
            if self._stream_queue.full():
                print(f"[WS] stream queue full; skipped iteration {iteration}", flush=True)
                return
            payload = self._stream_payload(state, include_nurbs=include_nurbs)
            copy_event = self._stream_copy_event(state.model.device)
            self._stream_queue.put_nowait((iteration, include_nurbs, payload, copy_event))
            self._streamed_iterations[iteration] = bool(include_nurbs)
            if len(self._streamed_iterations) > 16:
                self._streamed_iterations.pop(min(self._streamed_iterations), None)
        except queue.Full:
            print(f"[WS] stream queue full; skipped iteration {iteration}", flush=True)
        except Exception as exc:
            now = time.time()
            if now - self._stream_last_error_at > 10:
                print(f"[WS] stream snapshot failed at iteration {iteration}: {exc}", flush=True)
                self._stream_last_error_at = now

    def _stream_worker(self) -> None:
        assert self._stream_queue is not None
        while True:
            item = self._stream_queue.get()
            try:
                if item is None:
                    return
                iteration, include_nurbs, payload, copy_event = item
                if copy_event is not None:
                    copy_event.synchronize()
                payload = self._materialize_stream_payload(payload)
                message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                self._cache_stream_snapshot(iteration, payload, message=message)
                if self.training_config.stream_url:
                    self._get_stream_socket().send(message)
                    capped = " capped" if payload["metadata"]["capped"] else ""
                    nurbs = " + nurbs" if include_nurbs and "nurbs_surface" in payload else ""
                    print(
                        f"[WS] sent iteration {iteration}: "
                        f"{payload['count']}/{payload['metadata']['totalCount']} gaussians{capped}{nurbs}",
                        flush=True,
                    )
            except Exception as exc:
                now = time.time()
                if now - self._stream_last_error_at > 10:
                    print(f"[WS] stream/cache failed at iteration {item[0] if item else '?'}: {exc}", flush=True)
                    self._stream_last_error_at = now
                self._close_stream_socket()
            finally:
                self._stream_queue.task_done()

    def _stream_copy_event(self, device) -> Any | None:
        """Record completion of all non-blocking CUDA-to-pinned-CPU copies."""

        device_type = getattr(device, "type", str(device).split(":", 1)[0])
        if device_type != "cuda" or not self.torch.cuda.is_available():
            return None
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(device=device))
        return event

    def _snapshot_tensor(self, value: Any) -> Any:
        """Clone a snapshot, using pinned CPU memory for asynchronous CUDA copies."""

        source = value.detach()
        if source.device.type == "cuda":
            target = self.torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
            target.copy_(source, non_blocking=True)
            return target
        return source.cpu().clone()

    def _materialize_stream_payload(self, value: Any) -> Any:
        """Convert CPU tensor snapshots to JSON-compatible values in the worker."""

        if self.torch.is_tensor(value):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._materialize_stream_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._materialize_stream_payload(item) for item in value]
        return value

    def _cache_stream_snapshot(
        self, iteration: int, payload: dict[str, Any], message: str | None = None
    ) -> None:
        """Persist stream payloads so a later notebook cell can bulk-send them."""

        cache_dir = str(self.training_config.stream_cache_dir or "").strip()
        if not cache_dir:
            return
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{int(iteration):08d}.json"
        if message is None:
            message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        target.write_text(message, encoding="utf-8")

    def _stream_payload(self, state: TorchPipelineState, include_nurbs: bool = False) -> dict[str, Any]:
        torch = self.torch
        model = state.model
        with torch.no_grad():
            xyz_all = model.get_xyz
            total_count = int(xyz_all.shape[0])
            idx = self._stream_indices(total_count, xyz_all.device)
            xyz = self._snapshot_tensor(xyz_all[idx].float())
            scaling = self._snapshot_tensor(model.get_scaling[idx].float())
            rotation = self._snapshot_tensor(model.get_rotation[idx].float())
            opacity = self._snapshot_tensor(model.get_opacity[idx].float().reshape(-1))
            color = self._snapshot_tensor(torch.clamp(sh_dc_to_rgb(model.get_features_dc[idx, 0, :].detach().float()), 0.0, 1.0))
            sh_degree = int(model.active_sh_degree)
            sh_coefficients = self._snapshot_tensor(model.get_features[idx, : (sh_degree + 1) ** 2, :].float())

        count = int(xyz.shape[0])
        payload: dict[str, Any] = {
            "type": "snapshot",
            "iteration": int(state.iteration),
            "parameterSpace": "render",
            "count": count,
            "positions": xyz.reshape(-1),
            "scales": scaling.reshape(-1),
            "colors": color.reshape(-1),
            "opacities": opacity.reshape(-1),
            "rotations": rotation.reshape(-1),
            "shDegree": sh_degree,
            "shCoefficients": sh_coefficients.reshape(-1),
            "metadata": {
                "source": "osn-gs-training-stream",
                "totalCount": total_count,
                "sentCount": count,
                "capped": count != total_count,
                "loss": float(state.last_loss),
                "psnr": float(state.last_psnr),
            },
        }
        if include_nurbs:
            payload["nurbs_surface"] = self._nurbs_stream_payload(state)
        return payload

    def _stream_indices(self, count: int, device) -> Any:
        limit = max(0, int(self.training_config.stream_max_gaussians))
        if limit > 0 and count > limit:
            return self.torch.linspace(0, count - 1, steps=limit, device=device).long()
        return slice(None)

    def _nurbs_stream_payload(self, state: TorchPipelineState) -> dict[str, Any]:
        surface = state.surface
        grid = self._snapshot_tensor(surface.control_grid)
        self._streamed_nurbs_signature = (
            int(state.iteration),
            tuple(int(value) for value in surface.control_grid.shape),
        )
        weights = self._snapshot_tensor(surface.weights)
        payload = {
            "type": "visible_nurbs_intermediate",
            "iteration": int(state.iteration),
            "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
            "degree_u": int(surface.degree_u),
            "degree_v": int(surface.degree_v),
            "observed_v_max": float(surface.observed_v_max),
            "control_grid_shape": [int(value) for value in grid.shape],
            "control_grid": grid.reshape(-1, 3),
            "weights": weights.reshape(-1),
            "patches": [
                {
                    "patch_id": patch_id,
                    "control_grid_shape": [int(value) for value in patch.control_grid.shape],
                    "control_grid": self._snapshot_tensor(patch.control_grid).reshape(-1, 3),
                    "weights": self._snapshot_tensor(patch.weights).reshape(-1),
                    "degree_u": int(patch.degree_u),
                    "degree_v": int(patch.degree_v),
                }
                for patch_id, patch in enumerate(state.surface_patches)
            ],
            "metadata": {
                "source": "osn_gs_stage1_visible_reconstruction_stream",
                "gaussian_count": len(state.model),
                "uncertain_count": int(state.model.is_uncertain.sum().item()),
                "voxel_role": "initial_bootstrap",
                "surface_topology_version": int(state.surface_topology_version),
                "patch_residual_ratios": dict(state.surface_patch_residuals),
                "flattened": True,
            },
        }
        voxel_payload = self._voxel_regions_payload(state, flatten=True)
        if voxel_payload is not None:
            payload["voxel_regions"] = voxel_payload
        return payload

    def _voxel_regions_payload(self, state: TorchPipelineState, flatten: bool = False) -> dict[str, Any] | None:
        regions = state.voxel_regions
        if regions is None:
            return None

        def snapshot(value, *, flatten_value: bool = False):
            cpu = self._snapshot_tensor(value) if flatten else value.detach().cpu()
            if flatten_value:
                cpu = cpu.reshape(-1)
            return cpu if flatten else cpu.tolist()

        boundary = regions.boundary_mask.detach()
        payload: dict[str, Any] = {
            "type": "voxel_surface_regions",
            "count": int(regions.region_centers.shape[0]),
            "boundary_count": int(boundary.sum().item()),
            "centers": snapshot(regions.region_centers, flatten_value=flatten),
            "normals": snapshot(regions.region_normals, flatten_value=flatten),
            "boundary_mask": snapshot(boundary),
            "voxel_indices": snapshot(regions.voxel_indices, flatten_value=flatten),
            "region_patch_ids": snapshot(regions.region_patch_ids),
            "region_levels": snapshot(regions.region_levels),
            "region_density": snapshot(regions.region_density),
            "region_bounds": snapshot(regions.region_bounds, flatten_value=flatten),
            "flattened": bool(flatten),
        }
        return payload

    def _accumulate_density_stats(self, state: TorchPipelineState, render_packages) -> None:
        """Collect ADC visibility, radius, and screen-space gradient stats."""

        for render_pkg in render_packages:
            visibility = render_pkg.get("visibility_filter")
            viewspace = render_pkg.get("viewspace_points")
            radii = render_pkg.get("radii")
            update_max_radii(state.model, radii, visibility)
            add_densification_stats(state.model, viewspace, visibility)

    def _adc_stats(self, state: TorchPipelineState) -> dict[str, float | int]:
        """Return lightweight ADC diagnostics before a density-control pass."""

        torch = self.torch
        denom = torch.clamp(state.model.denom.detach(), min=1.0)
        grads = torch.nan_to_num((state.model.xyz_gradient_accum.detach() / denom).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        tracked = int((state.model.denom.detach().reshape(-1) > 0).sum().item())
        if grads.numel() == 0:
            max_grad = 0.0
            mean_grad = 0.0
        else:
            max_grad = float(grads.max().detach().cpu())
            mean_grad = float(grads.mean().detach().cpu())
        return {
            "tracked": tracked,
            "max_grad": max_grad,
            "mean_grad": mean_grad,
            "threshold": float(self.training_config.density_control.densify_grad_threshold),
        }

    def _scene_extent(self, points) -> float:
        """Return a conservative, outlier-robust scene extent used by ADC
        size thresholds and (via ``spatial_lr_scale``) the xyz position
        learning-rate magnitude.

        Previously computed as the raw point cloud's bounding-box diagonal
        (``(max - min).norm()``), which is extremely sensitive to a handful
        of far-flung noisy points -- a common artifact of real COLMAP SfM
        reconstructions (not something that shows up on this project's clean
        synthetic oracle-Gaussian benchmarks, which is why this went
        undetected until an actual real-dataset training run). On a real
        185-image garden scene this measured 124.5 while the actual scene
        content (median distance from centroid) was only ~5.0 and baseline
        3DGS's own camera-position-based extent was ~4.9 for the same scene
        -- a ~25x inflated ``spatial_lr_scale``, which oversizes the xyz
        learning rate enough to keep Gaussian positions perpetually
        overshooting instead of converging, visible as persistent blur that
        does not resolve even after many iterations/Gaussians.

        Now reuses ``estimate_scene_extent`` (mean-centered, 90th-percentile
        distance * 1.1) -- the same robust formula already used elsewhere in
        this codebase for exactly this purpose, previously not applied here.
        """

        pts = self.torch.as_tensor(points, dtype=self.torch.float32, device=self.device)
        if pts.numel() == 0:
            return 1.0
        extent = estimate_scene_extent(pts.detach().cpu().numpy())
        return max(float(extent), 1e-6)

    def _should_log_timing(self, iteration: int) -> bool:
        interval = max(0, int(self.training_config.timing_log_interval))
        return iteration == 1 or iteration == self.training_config.iterations or (interval > 0 and iteration % interval == 0)

    def _sync_cuda(self) -> None:
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _time_now(self, enabled: bool) -> float:
        if enabled:
            self._sync_cuda()
            return time.perf_counter()
        return 0.0

    def _elapsed(self, start: float) -> float:
        self._sync_cuda()
        return time.perf_counter() - start

    def _record_timing(self, timings: dict[str, float], name: str, start: float, enabled: bool) -> None:
        if enabled:
            timings[name] = self._elapsed(start)

    def _log_timing(self, iteration: int, timings: dict[str, float]) -> None:
        parts = " ".join(f"{key}={value:.3f}s" for key, value in timings.items())
        print(f"OSN-GS timing: iteration={iteration} {parts}", flush=True)

    def _should_log_progress(self, iteration: int) -> bool:
        interval = max(0, int(self.training_config.progress_log_interval))
        return iteration == 1 or iteration == self.training_config.iterations or (interval > 0 and iteration % interval == 0)

    def _log_progress(self, state: TorchPipelineState) -> None:
        # Stage 1 normally has no uncertain Gaussians. Avoid an extra CUDA
        # reduction in the hot logging path unless the flag tensor is non-empty.
        uncertain_count = 0
        if state.model.is_uncertain.numel() > 0 and bool(state.model.is_uncertain.detach().cpu().any()):
            uncertain_count = int(state.model.is_uncertain.detach().cpu().sum())
        print(
            "OSN-GS progress: "
            f"iteration={state.iteration}/{self.training_config.iterations} "
            f"loss={state.last_loss:.6f} "
            f"psnr={state.last_psnr:.3f} "
            f"gaussians={len(state.model)} "
            f"uncertain={uncertain_count}",
            flush=True,
        )

    def _prepare_training_view(self, camera, target):
        """Optionally downscale the training view to reduce renderer memory pressure."""

        scale = max(1, int(self.training_config.train_resolution_scale))
        if scale <= 1:
            return camera, target
        torch = self.torch
        height = max(1, int(camera.image_height) // scale)
        width = max(1, int(camera.image_width) // scale)
        resized = torch.nn.functional.interpolate(
            target.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return type(camera)(
            image_height=height,
            image_width=width,
            world_view_transform=camera.world_view_transform,
            full_proj_transform=camera.full_proj_transform,
            camera_center=camera.camera_center,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            image_name=camera.image_name,
        ), resized

    def _surface_losses(self, state: TorchPipelineState, residual_mse):
        """Bundle surface and uncertainty regularizers."""

        loss = nurbs_surface_loss(
            state,
            self.training_config.lambda_surface,
            max_patches=self.training_config.surface_loss_patch_budget,
        )
        loss = loss + uncertain_anchor_loss(state, self.training_config.lambda_anchor)
        loss = loss + uncertain_confidence_loss(state, residual_mse, self.training_config.lambda_uncertainty)
        return loss

    def _setup_surface_optimizer(self, state: TorchPipelineState) -> None:
        """Make the initial visible NURBS patches part of the optimization graph."""

        parameters = []
        for patch in state.surface_patches or [state.surface]:
            patch.control_grid = patch.control_grid.detach().requires_grad_(True)
            patch.weights = patch.weights.detach().requires_grad_(True)
            parameters.extend([patch.control_grid, patch.weights])
        state.surface = state.surface_patches[0]
        state.surface_optimizer = self.torch.optim.Adam(
            parameters, lr=float(self.training_config.surface_lr), eps=1e-15
        )

    def _sync_surface_optimizer(self, state: TorchPipelineState) -> None:
        """Register only new local-correction patches without resetting Adam state."""

        if state.surface_optimizer is None:
            self._setup_surface_optimizer(state)
            return
        known = {
            id(parameter)
            for group in state.surface_optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = []
        for patch in state.surface_patches:
            if not patch.control_grid.requires_grad:
                patch.control_grid = patch.control_grid.detach().requires_grad_(True)
            if not patch.weights.requires_grad:
                patch.weights = patch.weights.detach().requires_grad_(True)
            for parameter in (patch.control_grid, patch.weights):
                if id(parameter) not in known:
                    new_parameters.append(parameter)
                    known.add(id(parameter))
        if new_parameters:
            # Keep a single parameter group so checkpoint restore retains the same
            # optimizer group topology while Adam initializes new rows lazily.
            state.surface_optimizer.param_groups[0]["params"].extend(new_parameters)
    def _clamp_uncertain_confidence(self, state: TorchPipelineState) -> None:
        """Keep observed Gaussians at high confidence."""

        if not state.model.is_uncertain.any():
            return
        with self.torch.no_grad():
            certain = ~state.model.is_uncertain
            state.model._confidence[certain] = 12.0

    def save_outputs(self, state: TorchPipelineState, output_dir: Path, camera) -> None:
        """Save human-readable outputs plus a resumable checkpoint."""

        output_dir.mkdir(parents=True, exist_ok=True)
        state.model.save_ply(output_dir / "point_cloud.ply")
        render_pkg = self.rasterizer.render(camera, state.model)
        self._save_ppm(output_dir / "render.ppm", render_pkg["render"])
        self._save_training_state(output_dir / "metrics.txt", state)
        self._save_nurbs_intermediate(output_dir / "nurbs_surface.json", state)
        save_torch_checkpoint(output_dir / "checkpoint.pt", state, {"cuda_rasterizer": self.rasterizer.has_cuda_backend})

    def _save_ppm(self, path: Path, image) -> None:
        """Write a preview image without external image libraries."""

        image = image.detach().cpu().clamp(0.0, 1.0)
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image_u8 = (image * 255.0).to(self.torch.uint8)
        height, width = int(image_u8.shape[0]), int(image_u8.shape[1])
        with path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(image_u8.numpy().tobytes())

    def _save_nurbs_intermediate(self, path: Path, state: TorchPipelineState) -> None:
        """Save the visible NURBS-like intermediate for external visualization."""

        payload = nurbs_intermediate_payload(state)
        voxel_payload = self._voxel_regions_payload(state, flatten=False)
        if voxel_payload is not None:
            payload["voxel_regions"] = voxel_payload
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_training_state(self, path: Path, state: TorchPipelineState) -> None:
        """Persist a minimal text training summary."""

        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"iteration={state.iteration}\n")
            handle.write(f"loss={state.last_loss}\n")
            handle.write(f"psnr={state.last_psnr}\n")
            handle.write(f"gaussians={len(state.model)}\n")
            handle.write(f"uncertain={int(state.model.is_uncertain.sum().item())}\n")
            handle.write(f"cuda_rasterizer={self.rasterizer.has_cuda_backend}\n")
            handle.write("nurbs_intermediate=nurbs_surface.json\n")
