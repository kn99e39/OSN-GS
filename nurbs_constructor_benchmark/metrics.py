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

def support_domain_metrics(scene: SyntheticGaussianScene, state: Any, resolution: int = 128) -> tuple[dict[str, Any], dict[str, Any]]:
    gt = mask_on_grid(scene.support_predicate, resolution)
    generated = _rasterize_xy(sample_generated_surface(state, per_patch=resolution), resolution)
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
        "topology_label_ari": _adjusted_rand_index(state.model.cluster_ids.detach().cpu().clamp_min(0), scene.gt_patch_label(input_xy)),
    }
    support, raster = support_domain_metrics(scene, state, grid_n); result.update(support)
    return result, raster
