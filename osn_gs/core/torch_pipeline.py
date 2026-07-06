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
    visible_surface_fit_device: str = "cpu"
    visible_surface_fit_chunk_size: int = 0
    covariance_init: str = "knn"
    covariance_knn_chunk_size: int = 0
    covariance_min_scale: float = 1e-4
    covariance_max_scale_ratio: float = 0.05
    covariance_scale_multiplier: float = 1.0
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
        surface_points = self._surface_fit_points(points)
        fit_chunk_size = self._resolve_visible_surface_fit_chunk_size(surface_points)
        surface = fit_torch_visible_surface(
            surface_points,
            resolution_u=resolution_u,
            resolution_v=resolution_v,
            chunk_size=fit_chunk_size,
        )
        surface = self._move_surface(surface, self.device)

        count = points.shape[0]
        uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        surface_uv = torch.zeros((count, 2), dtype=torch.float32, device=self.device)
        cluster_ids = torch.full((count,), -1, dtype=torch.long, device=self.device)
        opacities = torch.full((count, 1), 0.12, dtype=torch.float32, device=self.device)
        scales = self._initial_covariance_scales(points)
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

    def _initial_covariance_scales(self, points: Any) -> Any:
        """Initialize trainable Gaussian scale from local point spacing.

        Original 3DGS initializes log-scale from sqrt(nearest-neighbor distance
        squared). OSN-GS keeps the same scale+rotation covariance convention but
        uses a chunked torch KNN path instead of the optional simple-knn module.
        """

        torch = require_torch()
        count = int(points.shape[0])
        if count == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        if count == 1 or str(self.config.covariance_init).lower() == "constant":
            base = self._scene_scale(points) * 0.001
            value = max(float(self.config.covariance_min_scale), float(base))
            return torch.full((count, 3), value, dtype=torch.float32, device=self.device)

        nearest_dist2 = self._nearest_neighbor_dist2(points.detach())
        scales = torch.sqrt(torch.clamp(nearest_dist2, min=float(self.config.covariance_min_scale) ** 2))
        scales = scales * float(self.config.covariance_scale_multiplier)
        max_scale = max(float(self.config.covariance_min_scale), self._scene_scale(points) * float(self.config.covariance_max_scale_ratio))
        scales = torch.clamp(scales, min=float(self.config.covariance_min_scale), max=max_scale)
        return scales[:, None].repeat(1, 3)

    def _nearest_neighbor_dist2(self, points: Any) -> Any:
        """Return squared distance to the nearest other point for every point."""

        torch = require_torch()
        count = int(points.shape[0])
        chunk_size = self._resolve_covariance_knn_chunk_size(points)
        nearest = torch.full((count,), float("inf"), dtype=torch.float32, device=points.device)
        all_indices = torch.arange(count, device=points.device)
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            chunk = points[start:end]
            distances = torch.cdist(chunk, points).square()
            local = all_indices[start:end]
            distances[torch.arange(end - start, device=points.device), local] = float("inf")
            nearest[start:end] = distances.min(dim=1).values
        finite = torch.isfinite(nearest)
        if not bool(finite.any()):
            fallback = self._scene_scale(points) * 0.001
            nearest.fill_(max(float(self.config.covariance_min_scale) ** 2, float(fallback) ** 2))
        else:
            fill = nearest[finite].median()
            nearest = torch.where(finite, nearest, fill)
        return nearest

    def _resolve_covariance_knn_chunk_size(self, points: Any) -> int:
        configured = int(self.config.covariance_knn_chunk_size)
        if configured > 0:
            return configured
        torch = require_torch()
        count = max(1, int(points.shape[0]))
        if points.device.type == "cuda" and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(points.device)
            workspace_bytes = max(64 * 1024 * 1024, int(free_bytes * 0.10))
            bytes_per_query = count * 4 * 2
            chunk_size = max(16, min(4096, int(workspace_bytes // max(bytes_per_query, 1))))
            self.config.covariance_knn_chunk_size = chunk_size
            print(
                "OSN-GS covariance KNN chunk: "
                f"auto={chunk_size} free_vram={free_bytes / (1024 ** 3):.2f}GB "
                f"total_vram={total_bytes / (1024 ** 3):.2f}GB points={count}",
                flush=True,
            )
            return chunk_size
        chunk_size = min(1024, count)
        self.config.covariance_knn_chunk_size = chunk_size
        print(f"OSN-GS covariance KNN chunk: auto={chunk_size} device={points.device}", flush=True)
        return chunk_size

    def _scene_scale(self, points: Any) -> float:
        torch = require_torch()
        if points.numel() == 0:
            return 1.0
        span = points.max(dim=0).values - points.min(dim=0).values
        return max(float(torch.linalg.norm(span).detach().cpu()), 1e-6)

    def _surface_fit_points(self, points: Any) -> Any:
        """Move visible-surface fitting inputs to the configured workspace device."""

        fit_device = str(self.config.visible_surface_fit_device or self.device).lower()
        if fit_device == "auto":
            fit_device = "cpu"
        if fit_device not in {"cpu", "cuda"}:
            fit_device = self.device
        return points.detach().to(fit_device)

    def _resolve_visible_surface_fit_chunk_size(self, points: Any) -> int:
        """Choose the visible-surface fit chunk once from runtime memory state."""

        configured = int(self.config.visible_surface_fit_chunk_size)
        if configured > 0:
            return configured

        torch = require_torch()
        point_count = max(1, int(points.shape[0]))
        device = getattr(points, "device", None)
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            # cdist materializes chunk x point_count distances. Keep a modest
            # slice of currently free VRAM for this transient workspace because
            # training tensors, images, and the rasterizer share the same GPU.
            workspace_bytes = max(64 * 1024 * 1024, int(free_bytes * 0.12))
            bytes_per_grid_sample = max(1, point_count) * 4 * 4
            chunk_size = workspace_bytes // bytes_per_grid_sample
            chunk_size = max(64, min(8192, int(chunk_size)))
            self.config.visible_surface_fit_chunk_size = chunk_size
            print(
                "OSN-GS NURBS fit chunk: "
                f"auto={chunk_size} free_vram={free_bytes / (1024 ** 3):.2f}GB "
                f"total_vram={total_bytes / (1024 ** 3):.2f}GB points={point_count}",
                flush=True,
            )
            return chunk_size

        chunk_size = 4096
        self.config.visible_surface_fit_chunk_size = chunk_size
        print(f"OSN-GS NURBS fit chunk: auto={chunk_size} device={device}", flush=True)
        return chunk_size

    def _move_surface(self, surface: TorchNURBSSurface, device: str) -> TorchNURBSSurface:
        """Return a surface whose persistent tensors live on the training device."""

        return TorchNURBSSurface(
            control_grid=surface.control_grid.to(device),
            weights=surface.weights.to(device),
            degree_u=surface.degree_u,
            degree_v=surface.degree_v,
            observed_v_max=surface.observed_v_max,
        )
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
