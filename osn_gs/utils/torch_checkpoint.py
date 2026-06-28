from __future__ import annotations

"""Torch checkpoint writer for OSN-GS."""

from pathlib import Path
from typing import Any

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.utils.torch_ops import require_torch


def save_torch_checkpoint(path: str | Path, state: TorchPipelineState, extra: dict[str, Any] | None = None) -> None:
    """학습 상태를 후처리/재개 가능한 Torch checkpoint로 저장한다.

    현재는 optimizer state까지 저장하지 않고, Gaussian/surface 상태를 저장한다.
    추후 full resume이 필요하면 trainer에서 optimizer state_dict를 함께 넣으면 된다.
    """

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = state.model

    # 모든 tensor를 CPU로 옮겨 저장하면 CUDA device가 달라도 로드하기 쉽다.
    payload = {
        "iteration": state.iteration,
        "last_loss": state.last_loss,
        "last_psnr": state.last_psnr,
        "xyz": model.get_xyz.detach().cpu(),
        "rgb": model.rgb.detach().cpu(),
        "opacity": model.get_opacity.detach().cpu(),
        "scaling": model.get_scaling.detach().cpu(),
        "rotation": model.get_rotation.detach().cpu(),
        "confidence": model.get_confidence.detach().cpu(),
        "is_uncertain": model.is_uncertain.detach().cpu(),
        "surface_uv": model.surface_uv.detach().cpu(),
        "cluster_ids": model.cluster_ids.detach().cpu(),
        "surface_control_grid": state.surface.control_grid.detach().cpu(),
        "extra": extra or {},
    }
    torch.save(payload, path)
