from __future__ import annotations

"""Torch-based OSN-GS visible surface reconstruction pipeline."""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_nurbs import (
    TorchCurveSet,
    TorchNURBSSurface,
    fit_torch_base_curves,
    fit_torch_visible_surface,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchPipelineConfig:
    """Controls visible surface reconstruction and Gaussian initialization.

    Stage 1 intentionally reconstructs only the visible surface. Occluded surface
    prediction and uncertain Gaussian sampling are kept for a later stage.
    """

    sh_degree: int = 3
    base_curve_count: int = 8
    visible_surface_resolution_u: int = 8
    visible_surface_resolution_v: int = 4
    visible_surface_resolution_scale: float = 1.0
    # Stage 2 legacy knobs. They are kept in the config for CLI compatibility,
    # but Stage 1 does not use them to create occluded geometry.
    occlusion_offset_scale: float = 0.25
    uncertain_samples_u: int = 16
    uncertain_samples_v: int = 3
    max_uncertain_gaussians: int = 0
    uncertain_opacity: float = 0.08
    uncertain_scale: float = 0.025
    color_cluster_count: int = 6


@dataclass
class TorchPipelineState:
    """Structure state carried throughout training."""

    model: TorchGaussianModel
    base_curves: TorchCurveSet
    occlusion_curves: TorchCurveSet
    surface: TorchNURBSSurface
    iteration: int = 0
    last_loss: float = 0.0
    last_psnr: float = 0.0


class TorchOSNGSPipeline:
    """Builds the Stage 1 visible surface state used by the trainer."""

    def __init__(self, config: TorchPipelineConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device

    def initialize(self, points: Any, colors: Any) -> TorchPipelineState:
        """Build the trainable state from observed points and colors.

        This Stage 1 path fits a visible parametric surface only. It does not
        extrapolate occlusion curves, sample occluded regions, or append
        uncertain Gaussians.
        """

        torch = require_torch()
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)

        base_curves = fit_torch_base_curves(points, self.config.base_curve_count)
        occlusion_curves = self._empty_occlusion_curves(points)
        resolution_u, resolution_v = self._visible_surface_resolution()
        surface = fit_torch_visible_surface(
            points,
            resolution_u=resolution_u,
            resolution_v=resolution_v,
        )

        count = points.shape[0]
        uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        surface_uv = torch.zeros((count, 2), dtype=torch.float32, device=self.device)
        cluster_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        opacities = torch.full((count, 1), 0.12, dtype=torch.float32, device=self.device)
        scales = torch.full((count, 3), 0.025, dtype=torch.float32, device=self.device)
        confidence = torch.ones((count, 1), dtype=torch.float32, device=self.device)

        model = TorchGaussianModel(sh_degree=self.config.sh_degree, device=self.device)
        model.initialize(
            positions=points,
            colors=colors,
            opacities=opacities,
            scales=scales,
            uncertain_mask=uncertain_mask,
            surface_uv=surface_uv,
            cluster_ids=cluster_ids,
            confidence=confidence,
        )
        return TorchPipelineState(model=model, base_curves=base_curves, occlusion_curves=occlusion_curves, surface=surface)

    def rebuild_surface_from_certain(self, state: TorchPipelineState) -> None:
        """Rebuild the visible surface hypothesis from certain Gaussians only."""

        certain = ~state.model.is_uncertain
        points = state.model.get_xyz.detach()[certain]
        colors = state.model.rgb.detach()[certain]

        rebuilt = self.initialize(points, colors)
        state.base_curves = rebuilt.base_curves
        state.occlusion_curves = rebuilt.occlusion_curves
        state.surface = rebuilt.surface
        state.model = rebuilt.model


    def _visible_surface_resolution(self) -> tuple[int, int]:
        """Return the scaled visible NURBS control-grid resolution."""

        scale = max(0.1, float(self.config.visible_surface_resolution_scale))
        resolution_u = max(2, int(round(self.config.visible_surface_resolution_u * scale)))
        resolution_v = max(2, int(round(self.config.visible_surface_resolution_v * scale)))
        return resolution_u, resolution_v

    def _empty_occlusion_curves(self, points: Any) -> TorchCurveSet:
        """Return an explicit empty Stage 2 placeholder."""

        torch = require_torch()
        return TorchCurveSet(
            control_points=torch.empty((0, 3, 3), dtype=torch.float32, device=self.device),
            observed=torch.zeros((0,), dtype=torch.bool, device=self.device),
        )

    def _assign_uncertain_colors(self, certain_points: Any, certain_colors: Any, uncertain_points: Any) -> tuple[Any, Any]:
        """Stage 2 legacy helper for future uncertain Gaussian initialization."""

        torch = require_torch()
        if uncertain_points.shape[0] == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=torch.float32, device=self.device),
            )
        distances = torch.cdist(uncertain_points, certain_points)
        nearest = distances.argmin(dim=1)
        cluster_ids = nearest % max(self.config.color_cluster_count, 1)
        return cluster_ids.long(), certain_colors[nearest]

    def _limit_uncertain_points(self, uncertain_points: Any, uv: Any) -> tuple[Any, Any]:
        """Stage 2 legacy helper for future uncertain Gaussian sampling caps."""

        torch = require_torch()
        max_uncertain = int(self.config.max_uncertain_gaussians)
        if max_uncertain <= 0 or uncertain_points.shape[0] <= max_uncertain:
            return uncertain_points, uv
        indices = torch.linspace(
            0,
            uncertain_points.shape[0] - 1,
            steps=max_uncertain,
            device=uncertain_points.device,
        ).round().long()
        return uncertain_points[indices], uv[indices]
