from __future__ import annotations

"""Torch checkpoint writer for OSN-GS."""

from pathlib import Path
from typing import Any

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.utils.torch_ops import require_torch


def save_torch_checkpoint(path: str | Path, state: TorchPipelineState, extra: dict[str, Any] | None = None) -> None:
    """Save Gaussian state plus the in-memory NURBS intermediate."""

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = state.model

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
        "surface_kind": "visible_nurbs_intermediate",
        "surface_control_grid": state.surface.control_grid.detach().cpu(),
        "surface_weights": state.surface.weights.detach().cpu(),
        "surface_degree_u": int(state.surface.degree_u),
        "surface_degree_v": int(state.surface.degree_v),
        "surface_observed_v_max": float(state.surface.observed_v_max),
        "extra": extra or {},
    }
    torch.save(payload, path)
