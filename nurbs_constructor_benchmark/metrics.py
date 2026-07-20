"""Ground-truth geometry, support-domain, and topology metrics."""
from __future__ import annotations
from typing import Any
import torch
from .ground_truth import gt_surface_points, observed_gt_surface_points
from .scenes import SyntheticGaussianScene
from .support_domains import mask_on_grid

_OBSERVED_RADIUS_FACTOR = 3.0
_SUPPORT_TAU_FACTOR = 2.5

def _min_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0: return torch.full((a.shape[0],), float("inf"))
    return torch.cat([torch.cdist(a[start:start + chunk], b).min(dim=1).values for start in range(0, a.shape[0], chunk)])

def sample_generated_surface(state: Any, per_patch: int = 128, respect_trim: bool = True) -> torch.Tensor:
    lin = torch.linspace(0.0, 1.0, per_patch)
    u, v = torch.meshgrid(lin, lin, indexing="ij")
    uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)
    samples = []
    for patch in state.surface_patches:
        patch_uv = uv.to(patch.control_grid.device)
        if respect_trim and getattr(patch, "uv_support_mask", None) is not None:
            patch_uv = patch_uv[patch.support(patch_uv)]
        if patch_uv.numel(): samples.append(patch.evaluate(patch_uv).detach().cpu())
    return torch.cat(samples) if samples else torch.empty((0, 3))

def _rasterize_xy(points: torch.Tensor, resolution: int) -> torch.Tensor:
    mask = torch.zeros((resolution, resolution), dtype=torch.bool)
    if points.numel() == 0: return mask
    cells = ((points[:, :2].cpu() + 1.0) * 0.5 * resolution).long().clamp(0, resolution - 1)
    mask[cells[:, 0], cells[:, 1]] = True
    return mask

def _components(mask: torch.Tensor) -> int:
    seen = torch.zeros_like(mask)
    count, h, w = 0, mask.shape[0], mask.shape[1]
    for i, j in torch.nonzero(mask, as_tuple=False).tolist():
        if seen[i, j]: continue
        count += 1; stack = [(i, j)]; seen[i, j] = True
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True; stack.append((nx, ny))
    return count

def _holes(mask: torch.Tensor) -> int:
    return _components(~mask) - (1 if (~mask).any() else 0)

def _boundary(mask: torch.Tensor) -> torch.Tensor:
    if not mask.any(): return torch.empty((0, 2))
    inner = torch.nn.functional.max_pool2d((~mask).float()[None, None], 3, 1, 1)[0, 0] > 0
    return torch.nonzero(mask & inner, as_tuple=False).float()

def _boundary_distances(gt: torch.Tensor, generated: torch.Tensor, resolution: int) -> tuple[float, float]:
    a, b = _boundary(gt), _boundary(generated)
    if not a.numel() or not b.numel(): return float("inf"), float("inf")
    distance = torch.cdist(a, b) * (2.0 / max(1, resolution - 1))
    forward, reverse = distance.min(1).values, distance.min(0).values
    return float((forward.mean() + reverse.mean()) * 0.5), float(torch.maximum(forward.max(), reverse.max()))

def _adjusted_rand_index(a: torch.Tensor, b: torch.Tensor) -> float:
    n = int(a.shape[0])
    if n == 0: return 1.0
    _, ai = torch.unique(a, return_inverse=True); _, bi = torch.unique(b, return_inverse=True)
    ka, kb = int(ai.max()) + 1, int(bi.max()) + 1
    table = torch.zeros(ka * kb); table.scatter_add_(0, (ai * kb + bi).long(), torch.ones(n)); table = table.reshape(ka, kb)
    comb = lambda x: x * (x - 1.0) / 2.0
    total = comb(torch.tensor(float(n))); expected = comb(table.sum(1)).sum() * comb(table.sum(0)).sum() / total
    maximum = 0.5 * (comb(table.sum(1)).sum() + comb(table.sum(0)).sum()); denom = maximum - expected
    return 1.0 if float(denom) == 0.0 else float((comb(table).sum() - expected) / denom)

def adjusted_rand_index(a: torch.Tensor, b: torch.Tensor) -> float:
    """Public alias of the ARI helper (used by the Phase 1 component-report script)."""
    return _adjusted_rand_index(a, b)

def support_domain_metrics(scene: SyntheticGaussianScene, state: Any, resolution: int = 128) -> tuple[dict[str, Any], dict[str, Any]]:
    gt = mask_on_grid(scene.support_predicate, resolution)
    # Use the same extent-adaptive UV oversampling as patch_union_metrics.
    # Sampling only one UV point per output raster cell fragments a trimmed
    # 64x64 chart when it is scored on this 128x128 world grid, producing
    # artificial enclosed background cells that look like thousands of holes.
    # The union raster is the supported surface representation being measured.
    generated = torch.zeros((resolution, resolution), dtype=torch.bool)
    for patch in state.surface_patches:
        generated |= _patch_xy_mask(patch, resolution)
    intersection = gt & generated
    gt_count, gen_count, common = int(gt.sum()), int(generated.sum()), int(intersection.sum())
    precision = common / gen_count if gen_count else 0.0
    recall = common / gt_count if gt_count else 1.0
    union = gt_count + gen_count - common
    boundary_chamfer, boundary_hausdorff = _boundary_distances(gt, generated, resolution)
    gt_components, gen_components = _components(gt), _components(generated)
    gt_holes, gen_holes = _holes(gt), _holes(generated)
    metrics = {
        "support_grid_resolution": resolution,
        "support_gt_cells": gt_count, "support_generated_cells": gen_count, "support_intersection_cells": common,
        "support_coverage_fraction": recall,
        "support_unsupported_fraction": (gen_count - common) / gen_count if gen_count else 0.0,
        "support_uncovered_fraction": (gt_count - common) / gt_count if gt_count else 0.0,
        "support_precision": precision, "support_recall": recall, "support_iou": common / union if union else 1.0,
        "support_gt_component_count": gt_components, "support_generated_component_count": gen_components,
        "support_gt_hole_count": gt_holes, "support_generated_hole_count": gen_holes,
        "support_gt_euler": gt_components - gt_holes, "support_generated_euler": gen_components - gen_holes,
        "support_topology_mismatch": (gt_components != gen_components) or (gt_holes != gen_holes),
        "support_boundary_chamfer": boundary_chamfer, "support_boundary_hausdorff": boundary_hausdorff,
    }
    return metrics, {"resolution": resolution, "gt_mask": gt.tolist(), "generated_mask": generated.tolist(), "intersection_mask": intersection.tolist()}

def _enclosed_hole_mask(mask: torch.Tensor) -> torch.Tensor:
    """True on complement components fully enclosed by ``mask`` (not touching the border)."""

    h, w = mask.shape
    complement = ~mask
    outside = torch.zeros_like(mask)
    stack = [
        (i, j)
        for i in range(h)
        for j in range(w)
        if (i in (0, h - 1) or j in (0, w - 1)) and complement[i, j] and not outside[i, j]
    ]
    for i, j in stack:
        outside[i, j] = True
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < h and 0 <= ny < w and complement[nx, ny] and not outside[nx, ny]:
                outside[nx, ny] = True
                stack.append((nx, ny))
    return complement & ~outside


def _component_areas(mask: torch.Tensor) -> list[int]:
    """Areas (cell counts) of the 4-connected components of ``mask``."""

    seen = torch.zeros_like(mask)
    areas: list[int] = []
    h, w = mask.shape
    for i, j in torch.nonzero(mask, as_tuple=False).tolist():
        if seen[i, j]:
            continue
        area = 0
        stack = [(i, j)]
        seen[i, j] = True
        while stack:
            x, y = stack.pop()
            area += 1
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
        areas.append(area)
    return sorted(areas, reverse=True)


def _area_histogram(areas: list[int]) -> dict[str, int]:
    bins = {"area_1": 0, "area_2_4": 0, "area_5_16": 0, "area_17_64": 0, "area_65_plus": 0}
    for area in areas:
        if area == 1: bins["area_1"] += 1
        elif area <= 4: bins["area_2_4"] += 1
        elif area <= 16: bins["area_5_16"] += 1
        elif area <= 64: bins["area_17_64"] += 1
        else: bins["area_65_plus"] += 1
    return bins


def _uv_in_mask(uv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
    cell_u = torch.clamp((uv[:, 0] * res_u).long(), 0, res_u - 1)
    cell_v = torch.clamp((uv[:, 1] * res_v).long(), 0, res_v - 1)
    return mask.to(uv.device)[cell_u, cell_v]


def _patch_xy_mask(
    patch: Any, resolution: int, respect_trim: bool = True, support_mask: Any = None
) -> torch.Tensor:
    """One patch's supported surface rasterized on the shared XY grid.

    The UV sampling density adapts to the patch's world extent so small voxel
    patches are still sampled at (at least) ~2 samples per raster cell —
    otherwise the union raster fragments into speckle and fake holes.
    ``support_mask`` overrides the patch's own trim mask (used to score the
    coarse polygon-only support against the density-refined one).
    """

    grid = patch.control_grid.detach().reshape(-1, 3)
    extent = float((grid[:, :2].max(dim=0).values - grid[:, :2].min(dim=0).values).max())
    cell = 2.0 / max(1, resolution)
    per_axis = int(min(384, max(16, -(-extent // cell) * 2 + 2)))
    lin = torch.linspace(0.0, 1.0, per_axis, device=patch.control_grid.device)
    u, v = torch.meshgrid(lin, lin, indexing="ij")
    uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)
    if support_mask is not None:
        uv = uv[_uv_in_mask(uv, support_mask)]
    elif respect_trim and getattr(patch, "uv_support_mask", None) is not None:
        uv = uv[patch.support(uv)]
    if not uv.numel():
        return torch.zeros((resolution, resolution), dtype=torch.bool)
    return _rasterize_xy(patch.evaluate(uv).detach().cpu(), resolution)


def patch_union_metrics(
    scene: SyntheticGaussianScene,
    state: Any,
    resolution: int = 128,
    mask_override: list[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Global patch-union support metrics (Stage 1-D).

    Hole/topology metrics are computed on the world-space union of every
    patch's trimmed support raster — never by summing per-patch UV masks —
    because holes (e.g. planar_hole) only exist in the union. Also reports
    patch overlap/gap ratios and component/hole area histograms so raster
    fragmentation is visible instead of silently smoothed away.
    """

    gt = mask_on_grid(scene.support_predicate, resolution)
    if mask_override is not None:
        patch_masks = [
            _patch_xy_mask(patch, resolution, support_mask=mask)
            for patch, mask in zip(state.surface_patches, mask_override)
        ]
    else:
        patch_masks = [_patch_xy_mask(patch, resolution) for patch in state.surface_patches]
    coverage_count = torch.zeros((resolution, resolution), dtype=torch.long)
    for mask in patch_masks:
        coverage_count += mask.long()
    union = coverage_count > 0
    union_count, gt_count = int(union.sum()), int(gt.sum())
    intersection = int((union & gt).sum())

    gt_holes = _enclosed_hole_mask(gt)
    union_holes = _enclosed_hole_mask(union)
    gt_hole_count_cells = int(gt_holes.sum())
    hole_union = int((gt_holes | union_holes).sum())
    hole_intersection = int((gt_holes & union_holes).sum())

    # Inter-patch gap: uncovered GT cells adjacent to >= 2 distinct patches.
    neighbor_patches = torch.zeros((resolution, resolution), dtype=torch.long)
    for mask in patch_masks:
        dilated = torch.nn.functional.max_pool2d(mask.float()[None, None], 3, 1, 1)[0, 0] > 0.5
        neighbor_patches += dilated.long()
    gap_cells = gt & ~union & (neighbor_patches >= 2)

    component_areas = _component_areas(union)
    hole_areas = _component_areas(union_holes)
    boundary_chamfer, boundary_hausdorff = _boundary_distances(gt, union, resolution)
    overlap_cells = int((coverage_count >= 2).sum())

    # Active-active seams: uncovered cells sitting between >= 2 patches.
    seam_cell_count = int(gap_cells.sum())
    seam_component_count = _components(gap_cells)
    cell_size = 2.0 / max(1, resolution)

    # False holes: enclosed holes of the union that do not overlap a GT hole.
    false_hole_areas: list[int] = []
    seen = torch.zeros_like(union_holes)
    h, w = union_holes.shape
    for i, j in torch.nonzero(union_holes, as_tuple=False).tolist():
        if seen[i, j]:
            continue
        stack = [(i, j)]
        seen[i, j] = True
        component_cells = [(i, j)]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < h and 0 <= ny < w and union_holes[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
                    component_cells.append((nx, ny))
        if not any(bool(gt_holes[x, y]) for x, y in component_cells):
            false_hole_areas.append(len(component_cells))

    metrics = {
        "union_resolution": resolution,
        "union_patch_count": len(patch_masks),
        "union_gt_cells": gt_count,
        "union_generated_cells": union_count,
        "union_coverage_ratio": intersection / gt_count if gt_count else 1.0,
        "union_uncovered_ratio": (gt_count - intersection) / gt_count if gt_count else 0.0,
        "union_unsupported_ratio": (union_count - intersection) / union_count if union_count else 0.0,
        "union_iou": intersection / (gt_count + union_count - intersection) if (gt_count + union_count - intersection) else 1.0,
        "union_component_count": len(component_areas),
        "union_hole_count": len(hole_areas),
        "union_gt_hole_count": len(_component_areas(gt_holes)),
        "union_euler": len(component_areas) - len(hole_areas),
        "union_gt_euler": len(_component_areas(gt)) - len(_component_areas(gt_holes)),
        "union_false_fill_ratio": int((union & gt_holes).sum()) / gt_hole_count_cells if gt_hole_count_cells else 0.0,
        "union_hole_iou": hole_intersection / hole_union if hole_union else 1.0,
        "union_patch_overlap_ratio": overlap_cells / union_count if union_count else 0.0,
        "union_interpatch_gap_ratio": int(gap_cells.sum()) / gt_count if gt_count else 0.0,
        "union_boundary_chamfer": boundary_chamfer,
        "union_boundary_hausdorff": boundary_hausdorff,
        "union_component_areas_top8": component_areas[:8],
        "union_component_area_histogram": _area_histogram(component_areas),
        "union_tiny_component_count": sum(1 for area in component_areas if area <= 4),
        "union_hole_areas_top8": hole_areas[:8],
        "union_hole_area_histogram": _area_histogram(hole_areas),
        "union_tiny_hole_count": sum(1 for area in hole_areas if area <= 4),
        # Stage 1-F seam / false-hole diagnostics.
        "union_seam_cell_count": seam_cell_count,
        "union_seam_length": seam_cell_count * cell_size,
        "union_seam_component_count": seam_component_count,
        "union_false_hole_count": len(false_hole_areas),
        "union_false_hole_area_sum": sum(false_hole_areas),
        "union_tiny_false_hole_count": sum(1 for area in false_hole_areas if area <= 4),
    }
    raster = {
        "resolution": resolution,
        "gt_mask": gt.tolist(),
        "union_mask": union.tolist(),
        "gt_hole_mask": gt_holes.tolist(),
        "union_hole_mask": union_holes.tolist(),
        "overlap_mask": (coverage_count >= 2).tolist(),
        "seam_mask": gap_cells.tolist(),
        "patch_count_grid": coverage_count.tolist(),
    }
    return metrics, raster


def support_boundary_conformality(state: Any) -> dict[str, Any]:
    """How much of the generated support boundary is realized by chart edges.

    The boundary-conformal ideal (GT charts) realizes every support boundary —
    outer contour and holes alike — as a chart *parameter-domain* boundary, not
    as a trim-mask contour. For each patch, boundary cells of its supported UV
    region are split into cells on the UV domain border (conformal) vs interior
    trim-contour cells; an untrimmed patch is fully conformal by construction.
    Read this together with the support-accuracy metrics: it scores *how* the
    boundary is represented, not whether it is in the right place.
    """

    total_boundary = 0
    edge_boundary = 0
    trimmed_patches = 0
    for patch in state.surface_patches:
        mask = getattr(patch, "uv_support_mask", None)
        if mask is None:
            # Untrimmed chart: its whole boundary is the domain edge.
            continue
        mask = mask.detach().cpu().bool()
        trimmed_patches += 1
        res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
        padded = torch.zeros((res_u + 2, res_v + 2), dtype=torch.bool)
        padded[1:-1, 1:-1] = mask
        interior = (
            padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
        boundary = mask & ~interior
        border = torch.zeros_like(mask)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        total_boundary += int(boundary.sum())
        edge_boundary += int((boundary & border).sum())
    return {
        "support_conformality_ratio": edge_boundary / total_boundary if total_boundary else 1.0,
        "support_conformality_boundary_cells": total_boundary,
        "support_conformality_edge_cells": edge_boundary,
        "support_conformality_trimmed_patches": trimmed_patches,
    }


def contour_vs_gt_boundary(
    scene: SyntheticGaussianScene, contour_points_xy: torch.Tensor, resolution: int = 128
) -> tuple[float, float]:
    """(Chamfer, Hausdorff) between refined density contours and the GT support boundary.

    GT boundary cells come from the analytic predicate raster; both point sets
    live in world XY, so the numbers are comparable across scenes.
    """

    if contour_points_xy.numel() == 0:
        return float("inf"), float("inf")
    gt = mask_on_grid(scene.support_predicate, resolution)
    boundary_cells = _boundary(gt)
    if not boundary_cells.numel():
        return float("inf"), float("inf")
    gt_xy = boundary_cells / max(1, resolution - 1) * 2.0 - 1.0
    distance = torch.cdist(contour_points_xy.float(), gt_xy)
    forward, reverse = distance.min(1).values, distance.min(0).values
    chamfer = float((forward.mean() + reverse.mean()) * 0.5)
    hausdorff = float(torch.maximum(forward.max(), reverse.max()))
    return chamfer, hausdorff


def ground_truth_metrics(scene: SyntheticGaussianScene, state: Any, grid_n: int = 128) -> tuple[dict[str, Any], dict[str, Any]]:
    input_pts, input_xy = scene.points.detach().cpu(), scene.points.detach().cpu()[:, :2]
    spacing_matrix = torch.cdist(input_xy, input_xy); spacing_matrix.fill_diagonal_(float("inf"))
    local_spacing = spacing_matrix.min(1).values.clamp_min(torch.finfo(input_xy.dtype).eps)
    spacing = float(local_spacing.median()); observed_radius, support_tau = _OBSERVED_RADIUS_FACTOR * spacing, _SUPPORT_TAU_FACTOR * spacing
    gen = sample_generated_surface(state); gt_full = gt_surface_points(scene, grid_n); obs_gt = observed_gt_surface_points(scene, grid_n, observed_radius)
    acc = _min_dist(gen, gt_full); comp = _min_dist(obs_gt, gen)
    nearest = torch.cdist(gen, input_pts).min(1); local_tau = _SUPPORT_TAU_FACTOR * local_spacing[nearest.indices]
    rms = lambda x: float(x.square().mean().sqrt()) if x.numel() else float("nan")
    result = {
        "accuracy_rms": rms(acc), "accuracy_p95": float(torch.quantile(acc, .95)) if acc.numel() else float("nan"),
        "accuracy_max": float(acc.max()) if acc.numel() else float("nan"), "completeness_rms": rms(comp),
        "chamfer_rms": .5 * (rms(acc) + rms(comp)),
        "support_coverage_uncovered_fraction": float((comp > support_tau).float().mean()) if comp.numel() else float("nan"),
        "support_extrapolation_fraction": float((nearest.values > support_tau).float().mean()) if nearest.values.numel() else float("nan"),
        "support_extrapolation_fraction_local": float((nearest.values > local_tau).float().mean()) if nearest.values.numel() else float("nan"),
        "support_local_threshold_p50": float(local_tau.median()), "support_local_threshold_p95": float(torch.quantile(local_tau, .95)),
        "support_observed_gt_samples": int(obs_gt.shape[0]), "support_threshold": support_tau,
        "topology_gen_patch_count": len(state.surface_patches), "topology_gt_patch_count": int(scene.gt_patch_count),
        "topology_patch_count_delta": len(state.surface_patches) - int(scene.gt_patch_count),
        "topology_patch_count_match": len(state.surface_patches) == int(scene.gt_patch_count),
        # Full (N, 3) points: xy-split labels read column 0 as before, and
        # multi-sheet scenes (close_parallel_sheets) can label by z.
        "topology_label_ari": _adjusted_rand_index(state.model.cluster_ids.detach().cpu().clamp_min(0), scene.gt_patch_label(input_pts)),
    }
    support, raster = support_domain_metrics(scene, state, grid_n); result.update(support)
    return result, raster
