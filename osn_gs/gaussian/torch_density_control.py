from __future__ import annotations

"""Adaptive density control for the OSN-GS torch Gaussian model.

This module separates two policies:
- 3DGS-style ADC for observed/certain Gaussians: gradient accumulation,
  clone, split, and opacity/size pruning.
- Uncertain Gaussian cleanup: pruning only. Uncertain-to-certain promotion is
  intentionally forbidden until a later policy is specified.
"""

from dataclasses import dataclass

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchDensityControlConfig:
    """Controls 3DGS-style adaptive density control."""

    densify_from_iter: int = 500
    densify_until_iter: int = 0
    densification_interval: int = 0
    densify_grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.005
    percent_dense: float = 0.01
    split_samples: int = 2
    max_screen_size: float = 20.0
    max_scale_ratio: float = 0.1
    max_gaussians: int = 0
    opacity_reset_interval: int = 3000
    screen_size_prune_from_iter: int = 3000
    prune_uncertain_confidence_threshold: float = 0.05


@dataclass
class TorchDensityControlReport:
    """Summary of a density-control pass."""

    cloned: int = 0
    split: int = 0
    pruned: int = 0
    uncertain_pruned: int = 0
    pruned_opacity: int = 0
    pruned_screen: int = 0
    pruned_world: int = 0

    @property
    def changed(self) -> bool:
        return (self.cloned + self.split + self.pruned + self.uncertain_pruned) > 0


def add_densification_stats(model: TorchGaussianModel, viewspace_points, visibility_filter) -> None:
    """Accumulate screen-space gradient norms for visible Gaussians."""

    torch = require_torch()
    if len(model) == 0:
        return
    if visibility_filter is None or len(visibility_filter) == 0:
        return
    visibility_filter = torch.as_tensor(visibility_filter, dtype=torch.long, device=model.device).reshape(-1)
    valid = visibility_filter[(visibility_filter >= 0) & (visibility_filter < len(model))]
    if valid.numel() == 0:
        return

    grad = None
    if viewspace_points is not None and viewspace_points.grad is not None:
        candidate = viewspace_points.grad.detach()
        if candidate.ndim == 2 and candidate.shape[0] >= len(model):
            grad = candidate[:, :2]

    if grad is None or not torch.isfinite(grad[valid]).any() or torch.count_nonzero(grad[valid]).item() == 0:
        xyz_grad = getattr(model._xyz, "grad", None)
        if xyz_grad is None:
            return
        candidate = xyz_grad.detach()
        if candidate.ndim != 2 or candidate.shape[0] < len(model):
            return
        grad = candidate[:, :2]

    grad_xy = torch.nan_to_num(grad[valid], nan=0.0, posinf=0.0, neginf=0.0)
    model.xyz_gradient_accum[valid] += torch.norm(grad_xy, dim=-1, keepdim=True)
    model.denom[valid] += 1.0


def update_max_radii(model: TorchGaussianModel, radii, visibility_filter) -> None:
    """Track the largest observed screen-space radius for pruning."""

    torch = require_torch()
    if len(model) == 0 or radii is None or visibility_filter is None or len(visibility_filter) == 0:
        return
    visibility_filter = torch.as_tensor(visibility_filter, dtype=torch.long, device=model.device).reshape(-1)
    valid = visibility_filter[(visibility_filter >= 0) & (visibility_filter < len(model))]
    if valid.numel() == 0:
        return
    radii = torch.as_tensor(radii, dtype=torch.float32, device=model.device).reshape(-1)
    if radii.numel() < len(model):
        return
    model.max_radii2D[valid] = torch.maximum(model.max_radii2D[valid], radii[valid])


def should_run_adc(iteration: int, config: TorchDensityControlConfig) -> bool:
    """Return True when a 3DGS-style ADC pass should run."""

    if config.densify_until_iter <= 0 or config.densification_interval <= 0:
        return False
    return (
        iteration > max(0, int(config.densify_from_iter))
        and iteration < config.densify_until_iter
        and iteration % config.densification_interval == 0
    )


def apply_adaptive_density_control(
    model: TorchGaussianModel,
    config: TorchDensityControlConfig,
    scene_extent: float,
    iteration: int = 0,
) -> TorchDensityControlReport:
    """Apply basic 3DGS clone/split/prune to certain Gaussians."""

    torch = require_torch()
    if len(model) == 0:
        return TorchDensityControlReport()

    denom = torch.clamp(model.denom, min=1.0)
    grads = model.xyz_gradient_accum / denom
    grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    certain = ~model.is_uncertain
    tracked = model.denom.reshape(-1) > 0
    high_grad = tracked & (grads >= float(config.densify_grad_threshold))
    scale_max = model.get_scaling.detach().max(dim=1).values
    dense_extent = max(float(config.percent_dense) * float(scene_extent), 1e-6)

    clone_mask = certain & high_grad & (scale_max <= dense_extent)
    split_mask = certain & high_grad & (scale_max > dense_extent)

    available = None
    if config.max_gaussians > 0:
        available = max(0, int(config.max_gaussians) - len(model))
        if available == 0:
            clone_mask = torch.zeros_like(clone_mask)
            split_mask = torch.zeros_like(split_mask)

    clone_idx = _limited_indices(clone_mask, available)
    cloned = int(clone_idx.numel())
    if available is not None:
        available = max(0, available - cloned)
    split_samples = max(1, int(config.split_samples))
    parent_limit = None if available is None else max(0, available // split_samples)
    split_idx = _limited_indices(split_mask, parent_limit)
    split = int(split_idx.numel()) * split_samples

    candidate = _shape_transaction_candidates(model, clone_idx, split_idx, split_samples)
    current_certain = ~candidate["is_uncertain"]
    current_scale_max = torch.exp(candidate["scaling"]).max(dim=1).values
    opacity = torch.sigmoid(candidate["opacity"]).reshape(-1)
    opacity_mask = current_certain & (opacity < float(config.prune_opacity_threshold))
    screen_mask = torch.zeros_like(opacity_mask)
    world_mask = torch.zeros_like(opacity_mask)
    split_parent_mask = torch.zeros_like(opacity_mask)
    split_parent_mask[split_idx] = True
    # The old append path reset screen radii after the first growth operation.
    # Preserve that behavior while collapsing all edits into one transaction.
    candidate_radii = torch.zeros_like(opacity_mask, dtype=torch.float32)
    if cloned + split == 0:
        candidate_radii[: len(model)] = model.max_radii2D
    size_pruning_active = iteration > int(config.screen_size_prune_from_iter)
    if size_pruning_active and config.max_screen_size > 0:
        screen_mask = current_certain & (candidate_radii > float(config.max_screen_size))
        if config.max_scale_ratio > 0:
            world_mask = current_certain & (
                current_scale_max > float(config.max_scale_ratio) * float(scene_extent)
            )
    # Split parents had already disappeared before prune-reason accounting.
    eligible_prune = ~split_parent_mask
    opacity_mask &= eligible_prune
    screen_mask &= eligible_prune
    world_mask &= eligible_prune
    prune_mask = split_parent_mask | opacity_mask | screen_mask | world_mask
    pruned_opacity = int(opacity_mask.sum().item())
    pruned_screen = int((screen_mask & ~opacity_mask).sum().item())
    pruned_world = int((world_mask & ~opacity_mask & ~screen_mask).sum().item())

    pruned = int((opacity_mask | screen_mask | world_mask).sum().item())
    if bool(prune_mask.any()) or cloned + split > 0:
        _commit_shape_transaction(model, candidate, ~prune_mask)
    else:
        model._reset_density_stats(len(model))
    return TorchDensityControlReport(
        cloned=cloned,
        split=split,
        pruned=pruned,
        pruned_opacity=pruned_opacity,
        pruned_screen=pruned_screen,
        pruned_world=pruned_world,
    )

def apply_uncertain_density_control(
    model: TorchGaussianModel,
    config: TorchDensityControlConfig,
) -> TorchDensityControlReport:
    """Prune invalid uncertain Gaussians. Promotion is intentionally disabled."""

    if len(model) == 0 or not model.is_uncertain.any():
        return TorchDensityControlReport()
    confidence = model.get_confidence[:, 0].detach()
    prune_mask = model.is_uncertain & (confidence < config.prune_uncertain_confidence_threshold)
    pruned = _prune_mask(model, prune_mask)
    return TorchDensityControlReport(uncertain_pruned=pruned)


def _shape_transaction_candidates(model: TorchGaussianModel, clone_idx, split_idx, samples: int) -> dict:
    """Build clone/split candidates without mutating model tensors."""

    torch = require_torch()
    raw = {
        "xyz": model._xyz.detach(),
        "features_dc": model._features_dc.detach(),
        "features_rest": model._features_rest.detach(),
        "opacity": model._opacity.detach(),
        "scaling": model._scaling.detach(),
        "rotation": model._rotation.detach(),
        "confidence": model._confidence.detach(),
        "is_uncertain": model.is_uncertain,
        "surface_uv": model.surface_uv,
        "cluster_ids": model.cluster_ids,
    }
    additions = {key: [] for key in raw}
    if clone_idx.numel() > 0:
        for key, value in raw.items():
            additions[key].append(value[clone_idx])
    if split_idx.numel() > 0:
        parent_scale = model.get_scaling.detach()[split_idx]
        repeated_scale = parent_scale.repeat_interleave(samples, dim=0)
        offsets = torch.randn_like(repeated_scale) * repeated_scale
        parent_rotation = model.get_rotation.detach()[split_idx].repeat_interleave(samples, dim=0)
        vector = parent_rotation[:, 1:]
        cross = 2.0 * torch.cross(vector, offsets, dim=1)
        offsets = offsets + parent_rotation[:, :1] * cross + torch.cross(vector, cross, dim=1)
        split_values = {
            "xyz": raw["xyz"][split_idx].repeat_interleave(samples, dim=0) + offsets,
            "features_dc": raw["features_dc"][split_idx].repeat_interleave(samples, dim=0),
            "features_rest": raw["features_rest"][split_idx].repeat_interleave(samples, dim=0),
            "opacity": raw["opacity"][split_idx].repeat_interleave(samples, dim=0),
            "scaling": torch.log(torch.clamp(repeated_scale / (0.8 * samples), min=1e-6)),
            "rotation": raw["rotation"][split_idx].repeat_interleave(samples, dim=0),
            "confidence": raw["confidence"][split_idx].repeat_interleave(samples, dim=0),
            "is_uncertain": raw["is_uncertain"][split_idx].repeat_interleave(samples, dim=0),
            "surface_uv": raw["surface_uv"][split_idx].repeat_interleave(samples, dim=0),
            "cluster_ids": raw["cluster_ids"][split_idx].repeat_interleave(samples, dim=0),
        }
        for key, value in split_values.items():
            additions[key].append(value)
    return {
        key: torch.cat([value, *additions[key]], dim=0) if additions[key] else value
        for key, value in raw.items()
    }


def _commit_shape_transaction(model: TorchGaussianModel, candidate: dict, keep_mask) -> None:
    """Apply all ADC growth and pruning with one parameter/Adam-state rebuild."""

    torch = require_torch()
    old_count = len(model)
    old_keep = torch.as_tensor(keep_mask[:old_count], dtype=torch.bool, device=model.device)
    optimizer_keep_indices = torch.nonzero(old_keep, as_tuple=False).reshape(-1)
    selected = {key: value[keep_mask] for key, value in candidate.items()}
    model.replace_tensors(
        xyz=selected["xyz"],
        features_dc=selected["features_dc"],
        features_rest=selected["features_rest"],
        opacity=selected["opacity"],
        scaling=selected["scaling"],
        rotation=selected["rotation"],
        confidence=selected["confidence"],
        uncertain_mask=selected["is_uncertain"],
        surface_uv=selected["surface_uv"],
        cluster_ids=selected["cluster_ids"],
        optimizer_keep_indices=optimizer_keep_indices,
    )

def _prune_mask(model: TorchGaussianModel, prune_mask) -> int:
    torch = require_torch()
    prune_mask = torch.as_tensor(prune_mask, dtype=torch.bool, device=model.device).reshape(-1)
    if prune_mask.numel() != len(model) or not prune_mask.any():
        model._reset_density_stats(len(model))
        return 0
    keep = ~prune_mask
    pruned = int(prune_mask.sum().item())
    model.prune(keep)
    return pruned


def _limited_indices(mask, limit: int | None = None):
    torch = require_torch()
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if limit is not None and idx.numel() > limit:
        idx = idx[: max(0, int(limit))]
    return idx
