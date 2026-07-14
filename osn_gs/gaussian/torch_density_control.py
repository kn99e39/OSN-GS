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

    cloned = _clone_selected(model, clone_mask, available)
    if available is not None:
        available = max(0, available - cloned)
    split = _split_selected(model, split_mask, max(1, int(config.split_samples)), available)

    current_certain = ~model.is_uncertain
    current_scale_max = model.get_scaling.detach().max(dim=1).values
    opacity = model.get_opacity.detach().reshape(-1)
    opacity_mask = current_certain & (opacity < float(config.prune_opacity_threshold))
    screen_mask = torch.zeros_like(opacity_mask)
    world_mask = torch.zeros_like(opacity_mask)
    size_pruning_active = iteration > int(config.screen_size_prune_from_iter)
    if size_pruning_active and config.max_screen_size > 0:
        screen_mask = current_certain & (model.max_radii2D > float(config.max_screen_size))
        if config.max_scale_ratio > 0:
            world_mask = current_certain & (
                current_scale_max > float(config.max_scale_ratio) * float(scene_extent)
            )
    prune_mask = opacity_mask | screen_mask | world_mask
    pruned_opacity = int(opacity_mask.sum().item())
    pruned_screen = int((screen_mask & ~opacity_mask).sum().item())
    pruned_world = int((world_mask & ~opacity_mask & ~screen_mask).sum().item())

    pruned = _prune_mask(model, prune_mask)
    if pruned == 0:
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


def _clone_selected(model: TorchGaussianModel, mask, available: int | None = None) -> int:
    if available is not None and available <= 0:
        return 0
    idx = _limited_indices(mask, available)
    if idx.numel() == 0:
        return 0
    model.append_gaussians_raw(
        xyz=model._xyz.detach()[idx],
        features_dc=model._features_dc.detach()[idx],
        features_rest=model._features_rest.detach()[idx],
        opacity=model._opacity.detach()[idx],
        scaling=model._scaling.detach()[idx],
        rotation=model._rotation.detach()[idx],
        confidence=model._confidence.detach()[idx],
        uncertain_mask=model.is_uncertain[idx],
        surface_uv=model.surface_uv[idx],
        cluster_ids=model.cluster_ids[idx],
    )
    return int(idx.numel())


def _split_selected(model: TorchGaussianModel, mask, samples: int, available: int | None = None) -> int:
    torch = require_torch()
    if available is not None and available <= 0:
        return 0
    parent_limit = None if available is None else max(0, available // samples)
    idx = _limited_indices(mask, parent_limit)
    if idx.numel() == 0:
        return 0

    parent_scale = model.get_scaling.detach()[idx]
    repeated_scale = parent_scale.repeat_interleave(samples, dim=0)
    offsets = torch.randn_like(repeated_scale) * repeated_scale
    parent_rotation = model.get_rotation.detach()[idx].repeat_interleave(samples, dim=0)
    vector = parent_rotation[:, 1:]
    cross = 2.0 * torch.cross(vector, offsets, dim=1)
    offsets = offsets + parent_rotation[:, :1] * cross + torch.cross(vector, cross, dim=1)
    new_xyz = model._xyz.detach()[idx].repeat_interleave(samples, dim=0) + offsets
    new_scaling = torch.log(torch.clamp(repeated_scale / (0.8 * samples), min=1e-6))
    model.append_gaussians_raw(
        xyz=new_xyz,
        features_dc=model._features_dc.detach()[idx].repeat_interleave(samples, dim=0),
        features_rest=model._features_rest.detach()[idx].repeat_interleave(samples, dim=0),
        opacity=model._opacity.detach()[idx].repeat_interleave(samples, dim=0),
        scaling=new_scaling,
        rotation=model._rotation.detach()[idx].repeat_interleave(samples, dim=0),
        confidence=model._confidence.detach()[idx].repeat_interleave(samples, dim=0),
        uncertain_mask=model.is_uncertain[idx].repeat_interleave(samples, dim=0),
        surface_uv=model.surface_uv[idx].repeat_interleave(samples, dim=0),
        cluster_ids=model.cluster_ids[idx].repeat_interleave(samples, dim=0),
    )

    prune_original = torch.zeros((len(model),), dtype=torch.bool, device=model.device)
    prune_original[idx] = True
    _prune_mask(model, prune_original)
    return int(idx.numel() * samples)


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
