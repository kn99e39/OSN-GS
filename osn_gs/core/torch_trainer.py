from __future__ import annotations

"""Torch-based OSN-GS training loop."""

from dataclasses import dataclass, field
import json
import time
from typing import Any
from pathlib import Path

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, TorchPipelineState
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
from osn_gs.utils.torch_checkpoint import save_torch_checkpoint
from osn_gs.utils.torch_ops import default_device, psnr_from_mse, require_torch, sh_dc_to_rgb


@dataclass
class TorchTrainingConfig:
    """Controls the training loop and loss weights."""

    iterations: int = 1000
    batch_size: int = 1
    train_resolution_scale: int = 1
    lambda_l1: float = 0.8
    lambda_mse: float = 0.2
    lambda_surface: float = 0.01
    lambda_uncertainty: float = 0.05
    lambda_anchor: float = 0.01
    sh_increment_interval: int = 1000
    surface_rebuild_interval: int = 1000
    density_control_interval: int = 500
    save_interval: int = 1000
    save_iterations: tuple[int, ...] = ()
    progress_log_interval: int = 100
    timing_log_interval: int = 100
    stream_url: str = ""
    stream_every: int = 0
    stream_iterations: tuple[int, ...] = ()
    stream_max_gaussians: int = 0
    stream_nurbs: bool = True
    write_output_files: bool = True
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
        print(f"OSN-GS rasterizer backend: {self.rasterizer.backend_source}", flush=True)

    def train(self, scene: TorchScene, output_dir: str | Path) -> TorchTrainingResult:
        """Train the scene and save previews, checkpoints, and point clouds."""

        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state = self.pipeline.initialize(scene.initial_points, scene.initial_colors)
        state.model.training_setup(self.training_config.parameter_groups)
        background = torch.zeros((3,), dtype=torch.float32, device=self.device)
        scene_extent = self._scene_extent(scene.initial_points)
        train_wall_start = time.perf_counter()
        for iteration in range(1, self.training_config.iterations + 1):
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
            image_loss_value = 0.0
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
                    self.training_config.lambda_l1,
                    self.training_config.lambda_mse,
                )
                total = total + image_loss
                image_loss_value += float(mse.detach().cpu())
            total = total / max(len(batch.cameras), 1)
            self._record_timing(timings, "render_loss", phase_start, timed)
            phase_start = self._time_now(timed)

            mean_mse = torch.as_tensor(
                image_loss_value / max(len(batch.cameras), 1),
                dtype=torch.float32,
                device=self.device,
            )
            total = total + self._surface_losses(state, mean_mse)

            state.model.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self._accumulate_density_stats(state, render_packages)
            self._record_timing(timings, "backward", phase_start, timed)
            phase_start = self._time_now(timed)
            state.model.optimizer.step()

            self._clamp_uncertain_confidence(state)
            self._record_timing(timings, "optim", phase_start, timed)
            phase_start = self._time_now(timed)
            state.iteration = iteration
            state.last_loss = float(total.detach().cpu())
            state.last_psnr = psnr_from_mse(image_loss_value / max(len(batch.cameras), 1))

            if self.training_config.surface_rebuild_interval > 0 and iteration % self.training_config.surface_rebuild_interval == 0:
                self.pipeline.rebuild_surface_from_certain(state)
                state.model.training_setup(self.training_config.parameter_groups)

            if should_run_adc(iteration, self.training_config.density_control):
                adc_before = self._adc_stats(state)
                report = apply_adaptive_density_control(state.model, self.training_config.density_control, scene_extent)
                if report.changed:
                    state.model.training_setup(self.training_config.parameter_groups)
                print(
                    "OSN-GS ADC: "
                    f"iteration={iteration} cloned={report.cloned} split={report.split} "
                    f"pruned={report.pruned} gaussians={len(state.model)} "
                    f"tracked={adc_before['tracked']} max_grad={adc_before['max_grad']:.6g} "
                    f"mean_grad={adc_before['mean_grad']:.6g} threshold={adc_before['threshold']:.6g}",
                    flush=True,
                )

            if self.training_config.density_control_interval > 0 and iteration % self.training_config.density_control_interval == 0:
                report = apply_uncertain_density_control(state.model, self.training_config.density_control)
                if report.changed:
                    state.model.training_setup(self.training_config.parameter_groups)
                    print(
                        "OSN-GS uncertain cleanup: "
                        f"iteration={iteration} pruned={report.uncertain_pruned} gaussians={len(state.model)}",
                        flush=True,
                    )
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
        self._close_stream_socket()
        if self.training_config.write_output_files:
            self.save_outputs(state, output_dir / "final", scene.cameras[0])
        return TorchTrainingResult(state=state, output_dir=output_dir)

    def _should_save_iteration(self, iteration: int) -> bool:
        if self.training_config.save_iterations:
            return iteration in set(int(value) for value in self.training_config.save_iterations)
        return self.training_config.save_interval > 0 and iteration % self.training_config.save_interval == 0

    def _should_stream_iteration(self, iteration: int) -> bool:
        if not self.training_config.stream_url:
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
        print(f"[WS] connected to renderer relay: {self.training_config.stream_url}", flush=True)
        return self._stream_socket

    def _close_stream_socket(self) -> None:
        if self._stream_socket is None:
            return
        try:
            self._stream_socket.close()
        except Exception:
            pass
        self._stream_socket = None

    def _stream_snapshot(self, state: TorchPipelineState, include_nurbs: bool = False) -> None:
        if not self._should_stream_iteration(state.iteration):
            return
        try:
            payload = self._stream_payload(state, include_nurbs=include_nurbs)
            self._get_stream_socket().send(json.dumps(payload, separators=(",", ":")))
            capped = " capped" if payload["metadata"]["capped"] else ""
            nurbs = " + nurbs" if include_nurbs and "nurbs_surface" in payload else ""
            print(
                f"[WS] sent iteration {state.iteration}: "
                f"{payload['count']}/{payload['metadata']['totalCount']} gaussians{capped}{nurbs}",
                flush=True,
            )
        except Exception as exc:
            now = time.time()
            if now - self._stream_last_error_at > 10:
                print(f"[WS] stream failed at iteration {state.iteration}: {exc}", flush=True)
                self._stream_last_error_at = now
            self._close_stream_socket()

    def _stream_payload(self, state: TorchPipelineState, include_nurbs: bool = False) -> dict[str, Any]:
        torch = self.torch
        model = state.model
        with torch.no_grad():
            xyz_all = model.get_xyz
            total_count = int(xyz_all.shape[0])
            idx = self._stream_indices(total_count, xyz_all.device)
            xyz = xyz_all[idx].detach().float().cpu()
            scaling = model.get_scaling[idx].detach().float().cpu()
            rotation = model.get_rotation[idx].detach().float().cpu()
            opacity = model.get_opacity[idx].detach().float().reshape(-1).cpu()
            color = torch.clamp(sh_dc_to_rgb(model.get_features_dc[idx, 0, :].detach().float()), 0.0, 1.0).cpu()

        count = int(xyz.shape[0])
        payload: dict[str, Any] = {
            "type": "snapshot",
            "iteration": int(state.iteration),
            "parameterSpace": "render",
            "count": count,
            "positions": xyz.reshape(-1).tolist(),
            "scales": scaling.reshape(-1).tolist(),
            "colors": color.reshape(-1).tolist(),
            "opacities": opacity.reshape(-1).tolist(),
            "rotations": rotation.reshape(-1).tolist(),
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
        grid = surface.control_grid.detach().cpu()
        self._streamed_nurbs_signature = (
            int(state.iteration),
            tuple(int(value) for value in surface.control_grid.shape),
        )
        weights = surface.weights.detach().cpu()
        return {
            "type": "visible_nurbs_intermediate",
            "iteration": int(state.iteration),
            "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
            "degree_u": int(surface.degree_u),
            "degree_v": int(surface.degree_v),
            "observed_v_max": float(surface.observed_v_max),
            "control_grid_shape": [int(value) for value in grid.shape],
            "control_grid": grid.reshape(-1, 3).tolist(),
            "weights": weights.reshape(-1).tolist(),
            "metadata": {
                "source": "osn_gs_stage1_visible_reconstruction_stream",
                "gaussian_count": len(state.model),
                "uncertain_count": int(state.model.is_uncertain.sum().item()),
                "flattened": True,
            },
        }

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
        """Return a conservative scene extent used by ADC size thresholds."""

        pts = self.torch.as_tensor(points, dtype=self.torch.float32, device=self.device)
        if pts.numel() == 0:
            return 1.0
        span = pts.max(dim=0).values - pts.min(dim=0).values
        extent = float(self.torch.linalg.norm(span).detach().cpu())
        return max(extent, 1e-6)

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

        loss = nurbs_surface_loss(state, self.training_config.lambda_surface)
        loss = loss + uncertain_anchor_loss(state, self.training_config.lambda_anchor)
        loss = loss + uncertain_confidence_loss(state, residual_mse, self.training_config.lambda_uncertainty)
        return loss

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

        surface = state.surface
        payload = {
            "type": "visible_nurbs_intermediate",
            "iteration": int(state.iteration),
            "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
            "degree_u": int(surface.degree_u),
            "degree_v": int(surface.degree_v),
            "observed_v_max": float(surface.observed_v_max),
            "control_grid_shape": list(surface.control_grid.shape),
            "control_grid": surface.control_grid.detach().cpu().tolist(),
            "weights": surface.weights.detach().cpu().tolist(),
            "base_curves": state.base_curves.control_points.detach().cpu().tolist(),
            "occlusion_curves": state.occlusion_curves.control_points.detach().cpu().tolist(),
            "metadata": {
                "source": "osn_gs_stage1_visible_reconstruction",
                "gaussian_count": len(state.model),
                "uncertain_count": int(state.model.is_uncertain.sum().item()),
                "final_output_remains_gaussian": True,
            },
        }
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
