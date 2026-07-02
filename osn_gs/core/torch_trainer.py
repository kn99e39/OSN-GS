from __future__ import annotations

"""Torch-based OSN-GS training loop."""

from dataclasses import dataclass, field
import json
from pathlib import Path

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, TorchPipelineState
from osn_gs.data.torch_scene import TorchScene
from osn_gs.gaussian.torch_model import GaussianParameterGroups
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig, apply_uncertain_density_control
from osn_gs.losses.torch_losses import (
    image_reconstruction_loss,
    nurbs_surface_loss,
    uncertain_anchor_loss,
    uncertain_confidence_loss,
)
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.utils.torch_checkpoint import save_torch_checkpoint
from osn_gs.utils.torch_ops import default_device, psnr_from_mse, require_torch


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

    def train(self, scene: TorchScene, output_dir: str | Path) -> TorchTrainingResult:
        """Train the scene and save previews, checkpoints, and point clouds."""

        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state = self.pipeline.initialize(scene.initial_points, scene.initial_colors)
        state.model.training_setup(self.training_config.parameter_groups)
        background = torch.zeros((3,), dtype=torch.float32, device=self.device)
        for iteration in range(1, self.training_config.iterations + 1):
            if iteration % self.training_config.sh_increment_interval == 0:
                state.model.oneup_sh_degree()

            batch = scene.sample_views(iteration, self.training_config.batch_size)
            total = torch.zeros((), dtype=torch.float32, device=self.device)
            image_loss_value = 0.0
            for camera, target in zip(batch.cameras, batch.images):
                camera, target = self._prepare_training_view(camera, target)
                render_pkg = self.rasterizer.render(camera, state.model, background)
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

            mean_mse = torch.as_tensor(
                image_loss_value / max(len(batch.cameras), 1),
                dtype=torch.float32,
                device=self.device,
            )
            total = total + self._surface_losses(state, mean_mse)

            state.model.optimizer.zero_grad(set_to_none=True)
            total.backward()
            state.model.optimizer.step()

            self._clamp_uncertain_confidence(state)
            state.iteration = iteration
            state.last_loss = float(total.detach().cpu())
            state.last_psnr = psnr_from_mse(image_loss_value / max(len(batch.cameras), 1))

            if self.training_config.surface_rebuild_interval > 0 and iteration % self.training_config.surface_rebuild_interval == 0:
                self.pipeline.rebuild_surface_from_certain(state)
                state.model.training_setup(self.training_config.parameter_groups)

            if self.training_config.density_control_interval > 0 and iteration % self.training_config.density_control_interval == 0:
                apply_uncertain_density_control(state.model, self.training_config.density_control)
                state.model.training_setup(self.training_config.parameter_groups)

            if self.training_config.save_interval > 0 and iteration % self.training_config.save_interval == 0:
                self.save_outputs(state, output_dir / f"iteration_{iteration:06d}", batch.cameras[0])

        self.save_outputs(state, output_dir / "final", scene.cameras[0])
        return TorchTrainingResult(state=state, output_dir=output_dir)

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
