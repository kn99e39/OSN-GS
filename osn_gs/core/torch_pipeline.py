from __future__ import annotations

"""Torch-based OSN-GS visible surface reconstruction pipeline."""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_nurbs import (
    TorchCurveSet,
    TorchNURBSSurface,
    fit_torch_base_curves,
    fit_torch_visible_surface,
    fit_torch_visible_surface_lsq,
    project_torch_points_to_nurbs,
)
from osn_gs.surface.torch_voxel_regions import TorchVoxelSurfaceRegions, build_torch_voxel_surface_regions
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
    # Parametric fitting controls. "lsq" solves a regularized least-squares
    # system with foot-point parameter correction; "idw" keeps the legacy
    # inverse-distance seed fit only.
    surface_fit_mode: str = "lsq"
    surface_degree_u: int = 2
    surface_degree_v: int = 2
    surface_fit_smoothness: float = 1e-4
    surface_fit_tikhonov: float = 1e-4
    surface_fit_rounds: int = 2
    use_voxel_surface_regions: bool = True
    voxel_grid_resolution: int = 16
    adaptive_voxel_density: bool = True
    voxel_max_subdivision_depth: int = 1
    voxel_density_quantile: float = 0.75
    voxel_density_covariance_weight_cap: float = 10.0
    voxel_normal_knn: int = 16
    voxel_boundary_angle_degrees: float = 35.0
    voxel_min_points_per_region: int = 1
    voxel_normal_chunk_size: int = 4096
    covariance_init: str = "knn"
    covariance_knn_chunk_size: int = 0
    covariance_min_scale: float = 1e-4
    covariance_max_scale_ratio: float = 0.05
    covariance_scale_multiplier: float = 1.0
    surface_projection_chunk_size: int = 65536
    surface_projection_iterations: int = 4
    max_surface_control_points: int = 65536
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
    surface_patches: list[TorchNURBSSurface]
    voxel_regions: TorchVoxelSurfaceRegions | None = None
    surface_optimizer: Any | None = None
    surface_patch_residuals: dict[int, float] = field(default_factory=dict)
    surface_bad_checks: dict[int, int] = field(default_factory=dict)
    surface_topology_version: int = 0
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

        voxel_regions = self._build_voxel_regions(points)
        curve_points = self._curve_placement_points(points, voxel_regions)

        base_curves = fit_torch_base_curves(
            curve_points, self.config.base_curve_count,
            voxel_regions.region_patch_ids if voxel_regions is not None else None,
        )
        occlusion_curves = self._empty_occlusion_curves(points)
        surface_patches = self._fit_surface_patches(curve_points, voxel_regions)
        surface = surface_patches[0]

        count = points.shape[0]
        uncertain_mask = torch.zeros((count,), dtype=torch.bool, device=self.device)
        cluster_ids = self._point_region_ids(voxel_regions, count, points.device)
        surface_uv = self.project_points_to_patches(points, cluster_ids, surface_patches)
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
        return TorchPipelineState(
            model=model,
            base_curves=base_curves,
            occlusion_curves=occlusion_curves,
            surface=surface,
            surface_patches=surface_patches,
            voxel_regions=voxel_regions,
        )

    def maintain_surface_from_certain(
        self,
        state: TorchPipelineState,
        residual_ratio_threshold: float = 0.03,
        residual_patience: int = 3,
        local_min_gaussians: int = 64,
        local_min_component: int = 16,
        enable_local_correction: bool = True,
        refresh_uv: bool = True,
    ) -> dict[str, Any]:
        """Inspect persistent NURBS patches and locally split sustained failures.

        Initialization voxel topology remains frozen. NURBS control points are the
        continuously optimized geometry; voxelization is reused only inside a failing
        patch as an event-triggered topology correction.

        When ``refresh_uv`` is set, certain Gaussian UV bindings are first re-derived
        by foot-point projection onto their patch, so the quality residual measures
        true point-to-surface distance instead of drift against stale parameters.
        """

        torch = require_torch()
        model = state.model
        certain = ~model.is_uncertain
        if not bool(certain.any()):
            return {
                "patches": len(state.surface_patches),
                "checked": 0,
                "max_residual_ratio": 0.0,
                "candidates": [],
                "corrected": [],
                "added_patches": 0,
                "uv_refreshed": 0,
                "topology_changed": False,
            }
        xyz = model.get_xyz.detach()
        certain_xyz = xyz[certain]
        extent = (certain_xyz.amax(dim=0) - certain_xyz.amin(dim=0)).norm().clamp_min(1e-6)
        threshold = max(0.0, float(residual_ratio_threshold))
        patience = max(1, int(residual_patience))
        candidates: list[int] = []
        residuals: dict[int, float] = {}
        uv_refreshed = 0

        for patch_id, patch in enumerate(state.surface_patches):
            indices = torch.nonzero(
                certain & (model.cluster_ids == patch_id), as_tuple=False
            ).reshape(-1)
            if int(indices.numel()) == 0:
                state.surface_bad_checks.pop(patch_id, None)
                continue
            if refresh_uv:
                model.surface_uv[indices] = project_torch_points_to_nurbs(
                    xyz[indices],
                    patch,
                    iterations=int(self.config.surface_projection_iterations),
                    chunk_size=int(self.config.surface_projection_chunk_size),
                )
                uv_refreshed += int(indices.numel())
            quality_indices = indices
            if int(indices.numel()) > 8192:
                sample = torch.linspace(
                    0, indices.numel() - 1, steps=8192, device=indices.device
                ).long()
                quality_indices = indices[sample]
            anchors = patch.evaluate(model.surface_uv[quality_indices].detach()).detach()
            ratio = float(
                ((xyz[quality_indices] - anchors).norm(dim=1).mean() / extent).cpu()
            )
            residuals[patch_id] = ratio
            bad_checks = (
                state.surface_bad_checks.get(patch_id, 0) + 1
                if ratio > threshold
                else 0
            )
            state.surface_bad_checks[patch_id] = bad_checks
            if (
                bad_checks >= patience
                and int(indices.numel()) >= max(4, int(local_min_gaussians))
            ):
                candidates.append(patch_id)

        state.surface_patch_residuals = residuals
        added_patches = 0
        corrected: list[int] = []
        if enable_local_correction:
            for patch_id in candidates:
                added = self._split_failed_patch(
                    state, patch_id, max(4, int(local_min_component))
                )
                if added > 0:
                    added_patches += added
                    corrected.append(patch_id)
                    state.surface_bad_checks[patch_id] = 0
        if added_patches:
            state.surface_topology_version += 1
            state.surface = state.surface_patches[0]

        return {
            "patches": len(state.surface_patches),
            "checked": len(residuals),
            "max_residual_ratio": max(residuals.values(), default=0.0),
            "candidates": candidates,
            "corrected": corrected,
            "added_patches": added_patches,
            "uv_refreshed": uv_refreshed,
            "topology_changed": added_patches > 0,
        }

    def rebuild_surface_from_certain(self, state: TorchPipelineState) -> None:
        """Compatibility wrapper that no longer rebuilds global voxel topology."""

        self.maintain_surface_from_certain(state, enable_local_correction=False)

    def _split_failed_patch(
        self, state: TorchPipelineState, patch_id: int, min_component: int
    ) -> int:
        """Split one persistently bad patch while preserving existing patches."""

        torch = require_torch()
        model = state.model
        mask = (~model.is_uncertain) & (model.cluster_ids == int(patch_id))
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if int(indices.numel()) < max(2 * min_component, 4):
            return 0

        points = model.get_xyz.detach()[indices]
        opacity = model.get_opacity.detach()[indices].reshape(-1)
        scales = model.get_scaling.detach()[indices]
        volume = scales.prod(dim=1).clamp_min(1e-12)
        reference_volume = volume.median().clamp_min(1e-12)
        density_weights = opacity * (reference_volume / volume).clamp(
            0.1, float(self.config.voxel_density_covariance_weight_cap)
        )
        regions = self._build_voxel_regions(points, density_weights, log=False)
        if regions is None:
            return 0

        labels, counts = torch.unique(
            regions.point_patch_ids, sorted=True, return_counts=True
        )
        keep = counts >= int(min_component)
        labels, counts = labels[keep], counts[keep]
        if int(labels.numel()) < 2:
            return 0
        order = torch.argsort(counts, descending=True)
        labels = labels[order]

        current_controls = sum(
            int(patch.control_grid.shape[0] * patch.control_grid.shape[1])
            for patch in state.surface_patches
        )
        remaining_budget = max(
            0, int(self.config.max_surface_control_points) - current_controls
        )
        added = 0
        base_u, base_v = self._visible_surface_resolution()
        for label in labels[1:]:
            component = regions.point_patch_ids == label
            component_indices = indices[component]
            if (
                int(component_indices.numel()) < int(min_component)
                or remaining_budget < 4
            ):
                continue
            target = min(base_u * base_v, remaining_budget)
            aspect = float(base_u) / float(base_v)
            resolution_u = max(
                2, min(base_u, int(round((target * aspect) ** 0.5)))
            )
            resolution_v = max(2, min(base_v, int(target // resolution_u)))
            controls = resolution_u * resolution_v
            if controls > remaining_budget:
                continue

            component_points = model.get_xyz.detach()[component_indices]
            patch = self._fit_visible_patch(component_points, resolution_u, resolution_v)
            new_patch_id = len(state.surface_patches)
            state.surface_patches.append(patch)
            model.cluster_ids[component_indices] = new_patch_id
            model.surface_uv[component_indices] = project_torch_points_to_nurbs(
                component_points,
                patch,
                iterations=int(self.config.surface_projection_iterations),
                chunk_size=int(self.config.surface_projection_chunk_size),
            )
            remaining_budget -= controls
            added += 1
        return added

    def _fit_surface_patches(
        self, curve_points: Any, regions: TorchVoxelSurfaceRegions | None
    ) -> list[TorchNURBSSurface]:
        """Fit density-budgeted visible NURBS charts inside voxel patches."""

        torch = require_torch()
        labels = None if regions is None else regions.region_patch_ids
        labels_aligned = labels is not None and int(labels.numel()) == int(curve_points.shape[0])
        patch_labels = (
            torch.unique(labels, sorted=True)
            if labels_aligned
            else torch.zeros((1,), dtype=torch.long, device=curve_points.device)
        )
        groups = (
            [curve_points[labels == patch_id] for patch_id in patch_labels]
            if labels_aligned
            else [curve_points]
        )
        weight_groups = (
            [regions.region_density[labels == patch_id] for patch_id in patch_labels]
            if labels_aligned and regions is not None
            else [None for _ in groups]
        )
        base_u, base_v = self._visible_surface_resolution()
        max_per_patch = base_u * base_v
        budget = max(4 * len(groups), int(self.config.max_surface_control_points))

        scores = torch.ones((len(groups),), dtype=torch.float32, device=curve_points.device)
        if regions is not None and labels is not None and int(labels.numel()) == int(curve_points.shape[0]):
            density = regions.region_density.to(dtype=torch.float32)
            boundary = regions.boundary_mask.to(dtype=torch.float32)
            for index, patch_id in enumerate(patch_labels):
                mask = labels == patch_id
                patch_density = density[mask].sum().clamp_min(1e-6)
                boundary_fraction = boundary[mask].mean() if bool(mask.any()) else density.new_zeros(())
                scores[index] = torch.sqrt(patch_density) * (1.0 + boundary_fraction)

        target_total = min(budget, max_per_patch * len(groups))
        raw_targets = scores / scores.sum().clamp_min(1e-8) * float(target_total)
        targets = torch.clamp(raw_targets.round().to(torch.long), min=4, max=max_per_patch)
        while int(targets.sum()) > target_total:
            candidates = torch.nonzero(targets > 4, as_tuple=False).reshape(-1)
            if int(candidates.numel()) == 0:
                break
            index = int(candidates[torch.argmax(targets[candidates])])
            targets[index] -= 1

        aspect = float(base_u) / float(base_v)
        resolutions: list[tuple[int, int]] = []
        for target in targets.tolist():
            resolution_u = max(2, min(base_u, int(round((target * aspect) ** 0.5))))
            resolution_v = max(2, min(base_v, int(round(target / resolution_u))))
            while resolution_u * resolution_v > target and resolution_v > 2:
                resolution_v -= 1
            resolutions.append((resolution_u, resolution_v))

        print(
            "OSN-GS NURBS density budget: "
            f"patches={len(groups)} controls={sum(u * v for u, v in resolutions)}/"
            f"{target_total} range={min(u * v for u, v in resolutions)}-"
            f"{max(u * v for u, v in resolutions)}",
            flush=True,
        )
        patches = []
        for points, weights, (resolution_u, resolution_v) in zip(groups, weight_groups, resolutions):
            patches.append(self._fit_visible_patch(points, resolution_u, resolution_v, weights))
        return patches

    def _fit_visible_patch(
        self,
        points: Any,
        resolution_u: int,
        resolution_v: int,
        point_weights: Any | None = None,
    ) -> TorchNURBSSurface:
        """Fit one visible NURBS chart in the configured fitting mode."""

        fit_points = self._surface_fit_points(points)
        chunk_size = self._resolve_visible_surface_fit_chunk_size(fit_points)
        if str(self.config.surface_fit_mode).lower() == "lsq":
            if point_weights is not None:
                point_weights = point_weights.detach().to(fit_points.device)
            patch, _ = fit_torch_visible_surface_lsq(
                fit_points,
                resolution_u=resolution_u,
                resolution_v=resolution_v,
                degree_u=int(self.config.surface_degree_u),
                degree_v=int(self.config.surface_degree_v),
                smoothness_lambda=float(self.config.surface_fit_smoothness),
                tikhonov_lambda=float(self.config.surface_fit_tikhonov),
                correction_rounds=int(self.config.surface_fit_rounds),
                chunk_size=chunk_size,
                point_weights=point_weights,
                projection_iterations=int(self.config.surface_projection_iterations),
            )
        else:
            patch = fit_torch_visible_surface(
                fit_points,
                resolution_u=resolution_u,
                resolution_v=resolution_v,
                chunk_size=chunk_size,
                degree_u=int(self.config.surface_degree_u),
                degree_v=int(self.config.surface_degree_v),
            )
        return self._move_surface(patch, self.device)

    def project_points_to_patches(
        self, points: Any, patch_ids: Any, patches: list[TorchNURBSSurface]
    ) -> Any:
        """Bind points to their patch via foot-point projection onto the NURBS.

        Points with an invalid patch id fall back to the primary patch, matching
        the loss-side anchor fallback.
        """

        torch = require_torch()
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        patch_ids = torch.as_tensor(patch_ids, dtype=torch.long, device=points.device)
        if len(patches) == 1 or int(patch_ids.numel()) != int(points.shape[0]):
            return self.project_points_to_surface(points, patches[0])
        uv = torch.zeros((points.shape[0], 2), dtype=torch.float32, device=points.device)
        for patch_id, patch in enumerate(patches):
            mask = patch_ids == patch_id
            if bool(mask.any()):
                uv[mask] = self.project_points_to_surface(points[mask], patch)
        invalid = (patch_ids < 0) | (patch_ids >= len(patches))
        if bool(invalid.any()):
            uv[invalid] = self.project_points_to_surface(points[invalid], patches[0])
        return uv

    def project_points_to_surface(self, points: Any, surface: TorchNURBSSurface) -> Any:
        """Foot-point projection of points onto one NURBS patch."""

        return project_torch_points_to_nurbs(
            points,
            surface,
            iterations=int(self.config.surface_projection_iterations),
            chunk_size=int(self.config.surface_projection_chunk_size),
        )

    def _point_region_ids(self, regions: TorchVoxelSurfaceRegions | None, count: int, device: Any) -> Any:
        torch = require_torch()
        if regions is None or int(regions.point_patch_ids.numel()) != int(count):
            return torch.full((count,), -1, dtype=torch.long, device=device)
        return regions.point_patch_ids.to(device=device, dtype=torch.long)


    def _build_voxel_regions(
        self, points: Any, density_weights: Any | None = None, log: bool = True
    ) -> TorchVoxelSurfaceRegions | None:
        """Build voxel curve-placement regions before NURBS fitting."""

        if not bool(self.config.use_voxel_surface_regions):
            return None
        regions = build_torch_voxel_surface_regions(
            points.detach(),
            grid_resolution=int(self.config.voxel_grid_resolution),
            normal_knn=int(self.config.voxel_normal_knn),
            boundary_angle_degrees=float(self.config.voxel_boundary_angle_degrees),
            min_points_per_voxel=int(self.config.voxel_min_points_per_region),
            normal_chunk_size=int(self.config.voxel_normal_chunk_size),
            density_weights=density_weights,
            adaptive_density=bool(self.config.adaptive_voxel_density),
            max_subdivision_depth=int(self.config.voxel_max_subdivision_depth),
            subdivision_quantile=float(self.config.voxel_density_quantile),
        )
        region_count = int(regions.region_centers.shape[0])
        boundary_count = int(regions.boundary_mask.sum().detach().cpu()) if region_count else 0
        if log:
            print(
                "OSN-GS voxel bootstrap: "
                f"regions={region_count} boundary={boundary_count} "
                f"grid={int(self.config.voxel_grid_resolution)} "
                f"levels=0-{int(regions.region_levels.max().detach().cpu()) if region_count else 0} "
                f"adaptive={bool(self.config.adaptive_voxel_density)} "
                f"boundary_angle={float(self.config.voxel_boundary_angle_degrees):.1f}",
                flush=True,
            )
        return regions

    def _curve_placement_points(self, points: Any, voxel_regions: TorchVoxelSurfaceRegions | None) -> Any:
        """Use voxel surface areas as the pre-NURBS curve placement domain."""

        if voxel_regions is None or int(voxel_regions.curve_points.shape[0]) < 2:
            return points
        return voxel_regions.curve_points

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

