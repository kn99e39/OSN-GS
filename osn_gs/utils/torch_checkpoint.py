from __future__ import annotations

"""Versioned OSN-GS checkpoint save/restore."""

from pathlib import Path
from typing import Any

from osn_gs.core.torch_pipeline import TorchPipelineState
from osn_gs.gaussian.torch_model import GaussianParameterGroups
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.utils.torch_ops import require_torch


def save_torch_checkpoint(path: str | Path, state: TorchPipelineState, extra: dict[str, Any] | None = None) -> None:
    """Save raw trainable values, optimizer moments, bindings, and NURBS patches."""

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = state.model
    payload = {
        "format_version": 2,
        "iteration": state.iteration,
        "last_loss": state.last_loss,
        "last_psnr": state.last_psnr,
        "active_sh_degree": model.active_sh_degree,
        "model_raw": {
            "xyz": model._xyz.detach().cpu(),
            "features_dc": model._features_dc.detach().cpu(),
            "features_rest": model._features_rest.detach().cpu(),
            "opacity": model._opacity.detach().cpu(),
            "scaling": model._scaling.detach().cpu(),
            "rotation": model._rotation.detach().cpu(),
            "confidence": model._confidence.detach().cpu(),
            "is_uncertain": model.is_uncertain.detach().cpu(),
            "surface_uv": model.surface_uv.detach().cpu(),
            "cluster_ids": model.cluster_ids.detach().cpu(),
        },
        "density_stats": {
            "xyz_gradient_accum": model.xyz_gradient_accum.detach().cpu(),
            "denom": model.denom.detach().cpu(),
            "max_radii2D": model.max_radii2D.detach().cpu(),
        },
        "model_optimizer": model.optimizer.state_dict() if model.optimizer is not None else None,
        "surface_patches": [
            {
                "control_grid": patch.control_grid.detach().cpu(),
                "weights": patch.weights.detach().cpu(),
                "degree_u": patch.degree_u,
                "degree_v": patch.degree_v,
                "observed_v_max": patch.observed_v_max,
            }
            for patch in state.surface_patches
        ],
        "surface_optimizer": state.surface_optimizer.state_dict() if state.surface_optimizer is not None else None,
        "surface_maintenance": {
            "patch_residuals": dict(state.surface_patch_residuals),
            "bad_checks": dict(state.surface_bad_checks),
            "topology_version": int(state.surface_topology_version),
        },
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_torch_checkpoint(
    path: str | Path,
    state: TorchPipelineState,
    parameter_groups: GaussianParameterGroups,
    surface_lr: float,
) -> int:
    """Restore a v2 checkpoint into an initialized pipeline state."""

    torch = require_torch()
    payload = torch.load(Path(path), map_location=state.model.device, weights_only=False)
    if int(payload.get("format_version", 0)) != 2:
        raise ValueError("Only OSN-GS checkpoint format_version=2 supports resume.")
    raw = payload["model_raw"]
    state.model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        confidence=raw["confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
    )
    state.model.training_setup(parameter_groups)
    if payload.get("model_optimizer") is not None:
        state.model.optimizer.load_state_dict(payload["model_optimizer"])
    state.model.active_sh_degree = int(payload.get("active_sh_degree", 0))
    stats = payload.get("density_stats", {})
    for name in ("xyz_gradient_accum", "denom", "max_radii2D"):
        if name in stats:
            setattr(state.model, name, stats[name].to(state.model.device))

    patches = []
    for saved in payload["surface_patches"]:
        patches.append(TorchNURBSSurface(
            control_grid=saved["control_grid"].to(state.model.device).requires_grad_(True),
            weights=saved["weights"].to(state.model.device).requires_grad_(True),
            degree_u=int(saved["degree_u"]), degree_v=int(saved["degree_v"]),
            observed_v_max=float(saved["observed_v_max"]),
        ))
    state.surface_patches = patches
    state.surface = patches[0]
    parameters = [tensor for patch in patches for tensor in (patch.control_grid, patch.weights)]
    state.surface_optimizer = torch.optim.Adam(parameters, lr=float(surface_lr), eps=1e-15)
    if payload.get("surface_optimizer") is not None:
        state.surface_optimizer.load_state_dict(payload["surface_optimizer"])
    maintenance = payload.get("surface_maintenance", {})
    state.surface_patch_residuals = {
        int(key): float(value)
        for key, value in maintenance.get("patch_residuals", {}).items()
    }
    state.surface_bad_checks = {
        int(key): int(value)
        for key, value in maintenance.get("bad_checks", {}).items()
    }
    state.surface_topology_version = int(maintenance.get("topology_version", 0))
    state.iteration = int(payload["iteration"])
    state.last_loss = float(payload.get("last_loss", 0.0))
    state.last_psnr = float(payload.get("last_psnr", 0.0))
    return state.iteration
