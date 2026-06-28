from __future__ import annotations

"""Torch 기반 OSN-GS 파이프라인.

이 파일은 OSN-GS 아이디어가 처음으로 코드 흐름으로 묶이는 곳이다.
입력은 관측된 surface에서 온 certain Gaussian의 초기 point/color이고,
출력은 certain + uncertain Gaussian을 함께 담은 `TorchGaussianModel`이다.

큰 흐름:
1. certain Gaussian center를 관측 surface point cloud로 본다.
2. point cloud에서 base curve를 fitting한다.
3. base curve를 구조 prior로 사용해 occlusion curve를 예측한다.
4. base/occlusion curve를 묶어 NURBS-like surface를 만든다.
5. surface의 비관측 영역에서 uncertain Gaussian 후보를 샘플링한다.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_nurbs import (
    TorchCurveSet,
    TorchNURBSSurface,
    build_torch_surface,
    fit_torch_base_curves,
    predict_torch_occlusion_curves,
    sample_torch_occluded_surface,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchPipelineConfig:
    """Surface reconstruction과 uncertain Gaussian 초기화를 제어하는 설정."""

    # 3DGS와 호환되는 spherical harmonics 최대 차수.
    sh_degree: int = 3
    # 관측 point cloud를 몇 개의 base curve chunk로 나눌지 결정한다.
    base_curve_count: int = 8
    # base curve에서 occlusion curve를 얼마나 멀리 외삽할지 정하는 임시 prior.
    occlusion_offset_scale: float = 0.25
    # NURBS parameter u 방향 샘플 개수. curve 방향 해상도에 대응한다.
    uncertain_samples_u: int = 16
    # NURBS parameter v 방향 샘플 개수. 관측면에서 occluded 면으로 가는 방향이다.
    uncertain_samples_v: int = 3
    # uncertain Gaussian은 관측 근거가 약하므로 초기 opacity를 낮게 둔다.
    uncertain_opacity: float = 0.08
    # NURBS 위에 배치되는 uncertain Gaussian의 초기 scale.
    uncertain_scale: float = 0.025
    # 현재는 nearest certain index를 modulo로 묶는 간단한 cluster id로 사용한다.
    color_cluster_count: int = 6


@dataclass
class TorchPipelineState:
    """학습 중 계속 들고 다니는 OSN-GS의 구조 상태."""

    # 실제 최적화 대상 Gaussian parameter container.
    model: TorchGaussianModel
    # 관측 표면에서 fitting된 curve 묶음.
    base_curves: TorchCurveSet
    # base curve를 구조적으로 외삽해 만든 occluded side curve 묶음.
    occlusion_curves: TorchCurveSet
    # uncertain Gaussian의 anchor이자 구조 regularization 대상인 surface.
    surface: TorchNURBSSurface
    # trainer가 갱신하는 bookkeeping 값들.
    iteration: int = 0
    last_loss: float = 0.0
    last_psnr: float = 0.0


class TorchOSNGSPipeline:
    """OSN-GS의 구조 생성 단계를 담당한다.

    Trainer는 optimization loop만 알고, "uncertain Gaussian을 어떻게 만들지"는
    이 pipeline에 위임한다. 나중에 NURBS fitting 알고리즘을 교체할 때 이
    클래스 내부만 집중적으로 바꾸면 된다.
    """

    def __init__(self, config: TorchPipelineConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device

    def initialize(self, points: Any, colors: Any) -> TorchPipelineState:
        """certain point/color로부터 학습 가능한 전체 Gaussian state를 만든다."""

        torch = require_torch()
        # 입력이 numpy/list여도 CUDA tensor로 통일한다.
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)

        # 1. 관측 Gaussian center에서 base curve fitting.
        base_curves = fit_torch_base_curves(points, self.config.base_curve_count)

        # 2. base curve의 방향/법선 prior를 이용해 occluded curve hypothesis 생성.
        occlusion_curves = predict_torch_occlusion_curves(base_curves, self.config.occlusion_offset_scale)

        # 3. base/occlusion curve를 2D control grid로 묶어 NURBS-like surface 생성.
        surface = build_torch_surface(base_curves, occlusion_curves)

        # 4. surface의 v >= observed_v_max 영역을 비관측 영역으로 보고 샘플링.
        uncertain_points, uv = sample_torch_occluded_surface(
            surface,
            self.config.uncertain_samples_u,
            self.config.uncertain_samples_v,
        )

        # 5. uncertain Gaussian은 직접 color 관측이 없으므로 nearest certain color를 복사한다.
        cluster_ids, uncertain_colors = self._assign_uncertain_colors(points, colors, uncertain_points)

        # 6. certain과 uncertain을 하나의 TorchGaussianModel에 합쳐 optimizer가 한 번에 다루게 한다.
        all_points = torch.cat([points, uncertain_points], dim=0)
        all_colors = torch.cat([colors, uncertain_colors], dim=0)
        certain_count = points.shape[0]
        uncertain_count = uncertain_points.shape[0]

        # certain/uncertain 구분은 loss, density control, promotion policy에서 계속 사용된다.
        uncertain_mask = torch.cat(
            [
                torch.zeros((certain_count,), dtype=torch.bool, device=self.device),
                torch.ones((uncertain_count,), dtype=torch.bool, device=self.device),
            ]
        )

        # certain Gaussian은 surface parameter가 없으므로 uv=(0,0) placeholder를 둔다.
        surface_uv = torch.cat([torch.zeros((certain_count, 2), device=self.device), uv], dim=0)

        # cluster id는 uncertain color prior와 ADC pattern transfer의 연결고리다.
        all_cluster_ids = torch.cat(
            [
                torch.full((certain_count,), -1, dtype=torch.long, device=self.device),
                cluster_ids,
            ]
        )

        # uncertain은 구조 hypothesis이므로 opacity/confidence를 낮게 시작한다.
        opacities = torch.cat(
            [
                torch.full((certain_count, 1), 0.12, dtype=torch.float32, device=self.device),
                torch.full((uncertain_count, 1), self.config.uncertain_opacity, dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )

        # 지금은 isotropic에 가까운 scale 초기화다. 실제 3DGS 연결 시 covariance prior로 확장 가능.
        scales = torch.cat(
            [
                torch.full((certain_count, 3), 0.025, dtype=torch.float32, device=self.device),
                torch.full((uncertain_count, 3), self.config.uncertain_scale, dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )

        # confidence는 "이 Gaussian의 위치/표면 가설을 얼마나 믿는가"에 해당한다.
        confidence = torch.cat(
            [
                torch.ones((certain_count, 1), dtype=torch.float32, device=self.device),
                torch.full((uncertain_count, 1), 0.25, dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )

        # 3DGS-style parameter container 생성.
        model = TorchGaussianModel(sh_degree=self.config.sh_degree, device=self.device)
        model.initialize(
            positions=all_points,
            colors=all_colors,
            opacities=opacities,
            scales=scales,
            uncertain_mask=uncertain_mask,
            surface_uv=surface_uv,
            cluster_ids=all_cluster_ids,
            confidence=confidence,
        )
        return TorchPipelineState(model=model, base_curves=base_curves, occlusion_curves=occlusion_curves, surface=surface)

    def rebuild_surface_from_certain(self, state: TorchPipelineState) -> None:
        """현재 certain Gaussian만 사용해 surface hypothesis를 다시 만든다.

        uncertain Gaussian이 이미지 loss를 크게 만들면, 단순히 위치만 고칠지
        표면 가설 자체를 바꿀지 결정해야 한다. 이 함수는 후자의 hook이다.
        """

        torch = require_torch()
        # 이미 uncertain으로 표시된 점은 surface 재추정의 근거에서 제외한다.
        certain = ~state.model.is_uncertain
        points = state.model.get_xyz.detach()[certain]
        colors = state.model.rgb.detach()[certain]

        # 현 구현은 initialize를 재사용해 surface와 uncertain set을 통째로 다시 만든다.
        rebuilt = self.initialize(points, colors)
        old_uncertain_count = int(state.model.is_uncertain.sum().item())
        state.base_curves = rebuilt.base_curves
        state.occlusion_curves = rebuilt.occlusion_curves
        state.surface = rebuilt.surface
        state.model = rebuilt.model
        if old_uncertain_count == 0:
            state.model.is_uncertain = torch.zeros_like(state.model.is_uncertain)

    def _assign_uncertain_colors(self, certain_points: Any, certain_colors: Any, uncertain_points: Any) -> tuple[Any, Any]:
        """uncertain Gaussian의 초기 색상을 certain Gaussian 색상 prior로 채운다.

        논문 아이디어의 "certain Gaussian 색상 cluster에 uncertain을 할당"하는
        부분의 첫 구현이다. 현재는 nearest certain color를 직접 복사하고,
        cluster id는 간단한 modulo bucket으로 둔다.
        """

        torch = require_torch()
        if uncertain_points.shape[0] == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=torch.float32, device=self.device),
            )
        # surface상 위치와 가장 가까운 관측 Gaussian을 찾아 그 색상을 prior로 사용한다.
        distances = torch.cdist(uncertain_points, certain_points)
        nearest = distances.argmin(dim=1)
        cluster_ids = nearest % max(self.config.color_cluster_count, 1)
        return cluster_ids.long(), certain_colors[nearest]
