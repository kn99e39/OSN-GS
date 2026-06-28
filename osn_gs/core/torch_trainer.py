from __future__ import annotations

"""Torch 기반 OSN-GS 학습 루프.

이 파일은 "결과물을 뽑을 수 있는" 실행 경로의 중심이다.
CUDA rasterizer가 있으면 실제 3DGS-style differentiable rasterization을 사용하고,
없으면 Torch fallback renderer로 pipeline/debug 결과를 저장한다.
"""

from dataclasses import dataclass, field
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
from osn_gs.render.cuda_rasterizer_adapter import RasterizerPipelineOptions, TorchRasterizerAdapter
from osn_gs.utils.torch_checkpoint import save_torch_checkpoint
from osn_gs.utils.torch_ops import default_device, psnr_from_mse, require_torch


@dataclass
class TorchTrainingConfig:
    """학습 loop와 loss weight를 조절하는 설정."""

    # 전체 optimization step 수.
    iterations: int = 1000
    # 현재 scene sampler는 순차 camera batch를 뽑는다.
    batch_size: int = 1
    # image reconstruction loss의 L1/MSE 혼합 계수.
    lambda_l1: float = 0.8
    lambda_mse: float = 0.2
    # NURBS control grid smoothness regularization.
    lambda_surface: float = 0.01
    # uncertain Gaussian confidence가 image residual과 맞도록 유도하는 항.
    lambda_uncertainty: float = 0.05
    # uncertain Gaussian이 NURBS surface anchor에서 지나치게 벗어나지 않도록 하는 항.
    lambda_anchor: float = 0.01
    # 기존 3DGS처럼 일정 iteration마다 SH degree를 한 단계 올린다.
    sh_increment_interval: int = 1000
    # certain Gaussian만으로 surface를 재계산하는 주기.
    surface_rebuild_interval: int = 1000
    # uncertain prune/promotion을 수행하는 주기.
    density_control_interval: int = 500
    # PLY, render preview, checkpoint 저장 주기.
    save_interval: int = 1000
    # device 자동 선택 시 CUDA를 우선할지 여부.
    prefer_cuda: bool = True
    # Gaussian parameter group별 learning rate.
    parameter_groups: GaussianParameterGroups = field(default_factory=GaussianParameterGroups)
    # uncertain Gaussian pruning/promotion threshold.
    density_control: TorchDensityControlConfig = field(default_factory=TorchDensityControlConfig)


@dataclass
class TorchTrainingResult:
    """학습 종료 후 caller에게 반환하는 최소 결과 묶음."""

    state: TorchPipelineState
    output_dir: Path


class TorchOSNGSTrainer:
    """OSN-GS의 end-to-end optimization runner."""

    def __init__(
        self,
        pipeline_config: TorchPipelineConfig | None = None,
        training_config: TorchTrainingConfig | None = None,
        rasterizer_options: RasterizerPipelineOptions | None = None,
        device: str | None = None,
    ) -> None:
        # torch import를 지연시켜 문서/문법 검사 환경에서는 PyTorch가 없어도 파일을 읽을 수 있게 한다.
        self.torch = require_torch()
        self.training_config = training_config or TorchTrainingConfig()
        self.device = device or default_device(self.training_config.prefer_cuda)
        # 구조 생성은 pipeline, 렌더링은 rasterizer adapter, optimization은 trainer가 맡는다.
        self.pipeline = TorchOSNGSPipeline(pipeline_config or TorchPipelineConfig(), device=self.device)
        self.rasterizer = TorchRasterizerAdapter(rasterizer_options)

    def train(self, scene: TorchScene, output_dir: str | Path) -> TorchTrainingResult:
        """scene을 학습하고 PLY/checkpoint/render preview를 저장한다."""

        torch = self.torch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 초기 certain points/colors에서 uncertain Gaussian까지 포함된 모델을 만든다.
        state = self.pipeline.initialize(scene.initial_points, scene.initial_colors)

        # optimizer는 model parameter가 initialize/rebuild 이후 확정된 다음 생성해야 한다.
        state.model.training_setup(self.training_config.parameter_groups)
        background = torch.zeros((3,), dtype=torch.float32, device=self.device)
        for iteration in range(1, self.training_config.iterations + 1):
            # 3DGS의 coarse-to-fine SH schedule을 따르는 hook.
            if iteration % self.training_config.sh_increment_interval == 0:
                state.model.oneup_sh_degree()

            # 현재는 간단한 순차 sampler지만, COLMAP loader 연결 후 random view sampling으로 바꿀 수 있다.
            batch = scene.sample_views(iteration, self.training_config.batch_size)
            total = torch.zeros((), dtype=torch.float32, device=self.device)
            image_loss_value = 0.0
            for camera, target in zip(batch.cameras, batch.images):
                # CUDA rasterizer가 있으면 여기서 diff_gaussian_rasterization이 호출된다.
                render_pkg = self.rasterizer.render(camera, state.model, background)
                image = render_pkg["render"]
                target = target.to(device=self.device, dtype=torch.float32)

                # image loss는 certain/uncertain 모두에 gradient를 흘린다.
                image_loss, mse = image_reconstruction_loss(
                    image,
                    target,
                    self.training_config.lambda_l1,
                    self.training_config.lambda_mse,
                )
                total = total + image_loss
                image_loss_value += float(mse.detach().cpu())
            total = total / max(len(batch.cameras), 1)

            # scalar residual은 uncertain confidence target을 정하는 데 사용한다.
            mean_mse = torch.as_tensor(
                image_loss_value / max(len(batch.cameras), 1),
                dtype=torch.float32,
                device=self.device,
            )

            # surface 관련 loss를 더해 "이미지에 맞지만 구조는 망가지는" 해를 억제한다.
            total = total + self._surface_losses(state, mean_mse)

            # 표준 PyTorch optimization step.
            state.model.optimizer.zero_grad(set_to_none=True)
            total.backward()
            state.model.optimizer.step()

            # certain Gaussian confidence는 항상 high-confidence anchor로 유지한다.
            self._clamp_uncertain_confidence(state)
            state.iteration = iteration
            state.last_loss = float(total.detach().cpu())
            state.last_psnr = psnr_from_mse(image_loss_value / max(len(batch.cameras), 1))

            # 주기적으로 certain만 사용해 NURBS hypothesis를 다시 세운다.
            if self.training_config.surface_rebuild_interval > 0 and iteration % self.training_config.surface_rebuild_interval == 0:
                self.pipeline.rebuild_surface_from_certain(state)
                state.model.training_setup(self.training_config.parameter_groups)

            # uncertain Gaussian을 pruning하거나 confidence가 높아진 점을 certain으로 승격한다.
            if self.training_config.density_control_interval > 0 and iteration % self.training_config.density_control_interval == 0:
                apply_uncertain_density_control(state.model, self.training_config.density_control)
                state.model.training_setup(self.training_config.parameter_groups)

            # 중간 결과 저장. 긴 학습에서 surface/uncertain 변화를 추적하기 위한 hook이다.
            if self.training_config.save_interval > 0 and iteration % self.training_config.save_interval == 0:
                self.save_outputs(state, output_dir / f"iteration_{iteration:06d}", batch.cameras[0])

        # 마지막 결과는 항상 저장한다.
        self.save_outputs(state, output_dir / "final", scene.cameras[0])
        return TorchTrainingResult(state=state, output_dir=output_dir)

    def _surface_losses(self, state: TorchPipelineState, residual_mse):
        """OSN-GS 특유의 surface/uncertainty 관련 regularization 묶음."""

        loss = nurbs_surface_loss(state, self.training_config.lambda_surface)
        loss = loss + uncertain_anchor_loss(state, self.training_config.lambda_anchor)
        loss = loss + uncertain_confidence_loss(state, residual_mse, self.training_config.lambda_uncertainty)
        return loss

    def _clamp_uncertain_confidence(self, state: TorchPipelineState) -> None:
        """certain Gaussian은 관측 기반이므로 confidence를 고정적으로 높게 유지한다."""

        if not state.model.is_uncertain.any():
            return
        with self.torch.no_grad():
            certain = ~state.model.is_uncertain
            state.model._confidence[certain] = 12.0

    def save_outputs(self, state: TorchPipelineState, output_dir: Path, camera) -> None:
        """학습 결과를 사람이 확인 가능한 형태와 재개 가능한 checkpoint로 저장한다."""

        output_dir.mkdir(parents=True, exist_ok=True)
        # PLY에는 uncertain flag를 같이 저장해 후처리/시각화에서 구분할 수 있다.
        state.model.save_ply(output_dir / "point_cloud.ply")
        # Preview render는 빠르게 결과를 훑어보기 위한 PPM이다.
        render_pkg = self.rasterizer.render(camera, state.model)
        self._save_ppm(output_dir / "render.ppm", render_pkg["render"])
        self._save_training_state(output_dir / "metrics.txt", state)
        # checkpoint에는 Gaussian과 surface control grid까지 포함한다.
        save_torch_checkpoint(output_dir / "checkpoint.pt", state, {"cuda_rasterizer": self.rasterizer.has_cuda_backend})

    def _save_ppm(self, path: Path, image) -> None:
        """외부 이미지 라이브러리 없이 preview 이미지를 저장하기 위한 PPM writer."""

        image = image.detach().cpu().clamp(0.0, 1.0)
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image_u8 = (image * 255.0).to(self.torch.uint8)
        height, width = int(image_u8.shape[0]), int(image_u8.shape[1])
        with path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(image_u8.numpy().tobytes())

    def _save_training_state(self, path: Path, state: TorchPipelineState) -> None:
        """학습 로그를 최소 텍스트로 저장한다."""

        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"iteration={state.iteration}\n")
            handle.write(f"loss={state.last_loss}\n")
            handle.write(f"psnr={state.last_psnr}\n")
            handle.write(f"gaussians={len(state.model)}\n")
            handle.write(f"uncertain={int(state.model.is_uncertain.sum().item())}\n")
            handle.write(f"cuda_rasterizer={self.rasterizer.has_cuda_backend}\n")
