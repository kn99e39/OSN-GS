from __future__ import annotations

"""Uncertain Gaussian density control.

기존 3DGS의 ADC는 gradient/radius/opacity를 기준으로 split/clone/prune을 수행한다.
OSN-GS에서는 uncertain Gaussian이 NURBS surface hypothesis에서 온 점이라는 차이가
있으므로, 우선 confidence 기반 pruning/promotion policy를 별도로 둔다.
"""

from dataclasses import dataclass

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchDensityControlConfig:
    """uncertain pruning과 promotion threshold."""

    # opacity가 너무 낮으면 렌더링에 의미 있게 기여하지 않는 Gaussian으로 본다.
    prune_opacity_threshold: float = 0.005
    # uncertain confidence가 너무 낮으면 surface hypothesis가 틀렸다고 보고 제거한다.
    prune_uncertain_confidence_threshold: float = 0.05
    # confidence가 충분히 높아진 uncertain은 certain으로 승격한다.
    promote_uncertain_confidence_threshold: float = 0.85
    # scale이 너무 커지는 Gaussian은 surface/detail 표현보다 artifact일 가능성이 크다.
    max_scale: float = 0.35


@dataclass
class TorchDensityControlReport:
    """density control 결과를 로깅/디버깅하기 위한 요약."""

    pruned: int = 0
    promoted: int = 0


def apply_uncertain_density_control(
    model: TorchGaussianModel,
    config: TorchDensityControlConfig,
) -> TorchDensityControlReport:
    """uncertain Gaussian을 pruning하거나 certain으로 승격한다."""

    torch = require_torch()
    if len(model) == 0:
        return TorchDensityControlReport()

    # 정책 판단에는 detach된 현재 값을 사용한다.
    opacity = model.get_opacity[:, 0].detach()
    confidence = model.get_confidence[:, 0].detach()
    max_scale = model.get_scaling.detach().max(dim=1).values

    # uncertain만 confidence pruning 대상이지만, opacity/scale pruning은 전체 Gaussian에 적용한다.
    low_uncertain = model.is_uncertain & (confidence < config.prune_uncertain_confidence_threshold)
    low_opacity = opacity < config.prune_opacity_threshold
    too_large = max_scale > config.max_scale
    keep = ~(low_uncertain | low_opacity | too_large)
    pruned = int((~keep).sum().item())

    # promotion은 아직 uncertain이면서 confidence가 높은 Gaussian만 대상이다.
    promoted_mask = model.is_uncertain & (confidence >= config.promote_uncertain_confidence_threshold) & keep
    promoted = int(promoted_mask.sum().item())
    if pruned > 0:
        # prune 후 mask index가 바뀌므로 promoted_mask도 keep된 좌표계로 줄인다.
        promoted_mask = promoted_mask[keep]
        model.prune(keep)
    if promoted > 0:
        with torch.no_grad():
            # certain으로 승격되면 surface anchor와 cluster prior에서 독립시킨다.
            model.is_uncertain[promoted_mask] = False
            model.surface_uv[promoted_mask] = 0.0
            model.cluster_ids[promoted_mask] = -1
            # confidence logit을 큰 값으로 고정해 high-confidence 상태로 둔다.
            model._confidence[promoted_mask] = 12.0
    return TorchDensityControlReport(pruned=pruned, promoted=promoted)
