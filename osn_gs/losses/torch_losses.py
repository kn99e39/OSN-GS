from __future__ import annotations

"""Torch training losses for OSN-GS.

여기 있는 loss들은 "렌더링 품질"과 "NURBS 기반 구조 가설"을 동시에
최적화하기 위한 최소 구성이다. 추후 ablation을 쉽게 하기 위해 trainer에서
직접 수식을 쓰지 않고 함수로 분리했다.
"""

from typing import Any

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.utils.torch_ops import require_torch


def image_reconstruction_loss(
    image: Any,
    target: Any,
    lambda_l1: float = 0.8,
    lambda_mse: float = 0.2,
) -> tuple[Any, Any]:
    """렌더 이미지와 GT 이미지의 기본 reconstruction loss."""

    # L1은 색상 절대 오차에 robust하고, MSE는 PSNR 계산과 연결된다.
    l1 = (image - target).abs().mean()
    mse = (image - target).square().mean()
    return lambda_l1 * l1 + lambda_mse * mse, mse


def nurbs_surface_loss(state: TorchPipelineState, weight: float = 0.01) -> Any:
    """NURBS-like control grid가 급격히 흔들리지 않게 하는 smoothness loss."""

    return weight * state.surface.smoothness()


def uncertain_anchor_loss(state: TorchPipelineState, weight: float = 0.01) -> Any:
    """uncertain Gaussian이 자신이 샘플링된 surface anchor에서 멀어지는 것을 억제한다."""

    torch = require_torch()
    if not state.model.is_uncertain.any():
        return torch.zeros((), dtype=torch.float32, device=state.model.device)

    # uncertain Gaussian만 surface uv anchor를 갖는다.
    uncertain_xyz = state.model.get_xyz[state.model.is_uncertain]
    uv = state.model.surface_uv[state.model.is_uncertain]
    anchors = state.surface.evaluate(uv)

    # confidence가 낮을수록 surface prior를 더 강하게 적용한다.
    confidence = state.model.get_confidence[state.model.is_uncertain].detach()
    return weight * ((uncertain_xyz - anchors).square() * (1.0 - confidence)).mean()


def uncertain_confidence_loss(state: TorchPipelineState, residual_mse: Any, weight: float = 0.05) -> Any:
    """image residual을 uncertain confidence의 감독 신호로 사용한다.

    residual이 낮으면 surface hypothesis가 이미지와 충돌하지 않는다는 뜻이므로
    confidence target을 높이고, residual이 높으면 confidence를 낮춘다.
    """

    torch = require_torch()
    if not state.model.is_uncertain.any():
        return torch.zeros((), dtype=torch.float32, device=state.model.device)
    confidence = state.model.get_confidence[state.model.is_uncertain]

    # exp(-mse)는 0~1 범위의 부드러운 confidence target으로 쓰기 좋다.
    target_confidence = torch.exp(-residual_mse.detach()).clamp(0.0, 1.0)
    return weight * (confidence - target_confidence).square().mean()
