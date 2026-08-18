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
        # Primitive identity. A 2DGS surfel checkpoint stores a two-column
        # `scaling`; a 3D Gaussian checkpoint stores three. Loading one into
        # the other is a hard error rather than something to paper over by
        # fabricating or discarding a normal-direction scale, so the class
        # name is recorded explicitly and checked on restore. Absent on
        # pre-2DGS checkpoints, which are always volumetric.
        "primitive_class": type(state.model).__name__,
        "scale_dim": int(getattr(state.model, "scale_dim", 3)),
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
            "uncertain_confidence": model._uncertain_confidence.detach().cpu(),
            "is_uncertain": model.is_uncertain.detach().cpu(),
            "surface_uv": model.surface_uv.detach().cpu(),
            "cluster_ids": model.cluster_ids.detach().cpu(),
            "surface_owner_kind": model.surface_owner_kind.detach().cpu(),
            "surface_owner_id": model.surface_owner_id.detach().cpu(),
            "stable_gaussian_ids": model.stable_gaussian_ids.detach().cpu(),
            "next_stable_gaussian_id": int(model.next_stable_gaussian_id),
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
                "uv_support_mask": None if patch.uv_support_mask is None else patch.uv_support_mask.detach().cpu(),
            }
            for patch in state.surface_patches
        ],
        "surface_patch_confidence": list(state.surface_patch_confidence),
        "surface_optimizer": state.surface_optimizer.state_dict() if state.surface_optimizer is not None else None,
        "surface_maintenance": {
            "patch_residuals": dict(state.surface_patch_residuals),
            "bad_checks": dict(state.surface_bad_checks),
            "topology_version": int(state.surface_topology_version),
        },
        "visible_nurbs_lifecycle": {
            "state": state.visible_nurbs_state,
            "coverage_semantics": state.visible_nurbs_coverage_semantics,
            "adc_version": int(state.visible_nurbs_adc_version),
            "source_fingerprint": state.visible_nurbs_source_fingerprint,
            "last_attempt_iteration": int(state.visible_nurbs_last_attempt_iteration),
            "last_failure": dict(state.visible_nurbs_last_failure),
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
    # Fail closed on a primitive mismatch. Silently reshaping between a
    # 2-column surfel scaling and a 3-column volumetric one would either
    # invent normal-direction thickness a 2DGS surfel does not have or throw
    # away a real trained axis; neither is an acceptable recovery.
    saved_scale_dim = int(payload.get("scale_dim", 3))
    model_scale_dim = int(getattr(state.model, "scale_dim", 3))
    if saved_scale_dim != model_scale_dim:
        raise ValueError(
            "Checkpoint primitive mismatch: checkpoint was written by "
            f"{payload.get('primitive_class', 'TorchGaussianModel')!r} with "
            f"scale_dim={saved_scale_dim}, but this run's model is "
            f"{type(state.model).__name__!r} with scale_dim={model_scale_dim}. "
            "Set TorchPipelineConfig.primitive to match the checkpoint; a 2DGS "
            "surfel and a volumetric 3D Gaussian are not interconvertible."
        )
    state.model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"),
        surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    state.model.next_stable_gaussian_id = int(raw.get(
        "next_stable_gaussian_id", state.model.next_stable_gaussian_id
    ))
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
        saved_mask = saved.get("uv_support_mask")
        patches.append(TorchNURBSSurface(
            control_grid=saved["control_grid"].to(state.model.device).requires_grad_(True),
            weights=saved["weights"].to(state.model.device).requires_grad_(True),
            degree_u=int(saved["degree_u"]), degree_v=int(saved["degree_v"]),
            observed_v_max=float(saved["observed_v_max"]),
            uv_support_mask=None if saved_mask is None else saved_mask.to(state.model.device),
        ))
    state.surface_patches = patches
    state.surface_patch_confidence = tuple(payload.get("surface_patch_confidence", ()))
    state.surface = patches[0] if patches else None
    parameters = [tensor for patch in patches for tensor in (patch.control_grid, patch.weights)]
    state.surface_optimizer = (
        torch.optim.Adam(parameters, lr=float(surface_lr), eps=1e-15)
        if parameters else None
    )
    if payload.get("surface_optimizer") is not None and state.surface_optimizer is not None:
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
    lifecycle = payload.get("visible_nurbs_lifecycle", {})
    state.visible_nurbs_state = str(lifecycle.get(
        "state", "materialized" if patches else "checkpoint_empty"
    ))
    state.visible_nurbs_coverage_semantics = str(lifecycle.get(
        "coverage_semantics", "reliable_core_only"
    ))
    state.visible_nurbs_adc_version = int(lifecycle.get("adc_version", 0))
    state.visible_nurbs_source_fingerprint = str(lifecycle.get("source_fingerprint", ""))
    state.visible_nurbs_last_attempt_iteration = int(lifecycle.get("last_attempt_iteration", -1))
    state.visible_nurbs_last_failure = dict(lifecycle.get("last_failure", {}))
    state.iteration = int(payload["iteration"])
    state.last_loss = float(payload.get("last_loss", 0.0))
    state.last_psnr = float(payload.get("last_psnr", 0.0))
    return state.iteration
