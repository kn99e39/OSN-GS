from __future__ import annotations

"""Density-adaptive voxel regions for visible NURBS organization."""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchVoxelSurfaceRegions:
    """Surface-aligned adaptive cells used as NURBS patch candidates."""

    region_centers: Any
    region_normals: Any
    boundary_mask: Any
    voxel_indices: Any
    point_region_ids: Any
    region_patch_ids: Any
    point_patch_ids: Any
    region_levels: Any
    region_density: Any
    region_bounds: Any

    @property
    def curve_points(self) -> Any:
        return self.region_centers


def build_torch_voxel_surface_regions(
    points: Any,
    grid_resolution: int = 16,
    normal_knn: int = 16,
    boundary_angle_degrees: float = 35.0,
    min_points_per_voxel: int = 1,
    normal_chunk_size: int = 4096,
    density_weights: Any | None = None,
    adaptive_density: bool = True,
    max_subdivision_depth: int = 1,
    subdivision_quantile: float = 0.75,
) -> TorchVoxelSurfaceRegions:
    """Build density-adaptive cells and split patches at normal boundaries.

    A coarse cell is subdivided when its weighted Gaussian density is above the
    configured occupied-cell quantile. All cells are represented as AABBs on a
    shared finest grid, so face adjacency remains valid across different levels.
    """

    torch = require_torch()
    points = torch.as_tensor(
        points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None
    )
    count = int(points.shape[0])
    if count == 0:
        return _empty_regions(points.device)

    if density_weights is None:
        density_weights = torch.ones((count,), dtype=points.dtype, device=points.device)
    else:
        density_weights = torch.as_tensor(
            density_weights, dtype=points.dtype, device=points.device
        ).reshape(-1)
        if int(density_weights.numel()) != count:
            raise ValueError("density_weights must contain one value per Gaussian.")
        density_weights = torch.nan_to_num(
            density_weights, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        if not bool((density_weights > 0).any()):
            density_weights = torch.ones_like(density_weights)

    base_resolution = max(2, int(grid_resolution))
    depth = max(0, int(max_subdivision_depth)) if adaptive_density else 0
    fine_resolution = base_resolution * (2 ** depth)
    min_corner = points.min(dim=0).values
    span = (points.max(dim=0).values - min_corner).clamp_min(1e-6)
    normalized = ((points - min_corner) / span).clamp(0.0, 1.0 - 1e-7)

    coarse = torch.floor(normalized * base_resolution).long()
    coarse_linear = (
        coarse[:, 0] * base_resolution * base_resolution
        + coarse[:, 1] * base_resolution
        + coarse[:, 2]
    )
    _, coarse_inverse = torch.unique(coarse_linear, sorted=True, return_inverse=True)
    coarse_count = int(coarse_inverse.max().item()) + 1
    coarse_density = torch.zeros(
        (coarse_count,), dtype=points.dtype, device=points.device
    )
    coarse_density.index_add_(0, coarse_inverse, density_weights)

    levels = torch.zeros((count,), dtype=torch.long, device=points.device)
    if depth > 0 and coarse_density.numel() > 0:
        quantile = min(max(float(subdivision_quantile), 0.0), 1.0)
        threshold = torch.quantile(coarse_density, quantile)
        dense_coarse = coarse_density >= threshold
        levels[dense_coarse[coarse_inverse]] = depth

    point_resolution = base_resolution * torch.pow(
        torch.full_like(levels, 2), levels
    )
    point_cells = torch.floor(normalized * point_resolution[:, None]).long()
    max_cell_count = fine_resolution ** 3
    linear_local = (
        point_cells[:, 0] * point_resolution * point_resolution
        + point_cells[:, 1] * point_resolution
        + point_cells[:, 2]
    )
    encoded = levels * max_cell_count + linear_local
    unique_encoded, inverse, counts = torch.unique(
        encoded, sorted=True, return_inverse=True, return_counts=True
    )

    keep = counts >= max(1, int(min_points_per_voxel))
    if not bool(keep.all()):
        keep_ids = torch.nonzero(keep, as_tuple=False).flatten()
        remap = torch.full(
            (unique_encoded.shape[0],), -1, dtype=torch.long, device=points.device
        )
        remap[keep_ids] = torch.arange(
            keep_ids.numel(), dtype=torch.long, device=points.device
        )
        valid_points = remap[inverse] >= 0
        points_for_normals = points[valid_points]
        weights_for_centers = density_weights[valid_points]
        inverse_valid = remap[inverse[valid_points]]
        point_region_ids = remap[inverse]
        unique_encoded = unique_encoded[keep]
        counts = counts[keep]
    else:
        points_for_normals = points
        weights_for_centers = density_weights
        inverse_valid = inverse
        point_region_ids = inverse

    region_count = int(unique_encoded.shape[0])
    region_levels = unique_encoded // max_cell_count
    local = unique_encoded % max_cell_count
    region_resolution = base_resolution * torch.pow(
        torch.full_like(region_levels, 2), region_levels
    )
    cell_z = local % region_resolution
    cell_y = (local // region_resolution) % region_resolution
    cell_x = local // (region_resolution * region_resolution)
    cell_indices = torch.stack([cell_x, cell_y, cell_z], dim=1).long()
    cell_scale = torch.pow(
        torch.full_like(region_levels, 2), depth - region_levels
    )
    bounds_min = cell_indices * cell_scale[:, None]
    bounds_max = bounds_min + cell_scale[:, None]
    region_bounds = torch.stack([bounds_min, bounds_max], dim=1)

    weighted_points = points_for_normals * weights_for_centers[:, None]
    centers = torch.zeros(
        (region_count, 3), dtype=points.dtype, device=points.device
    )
    centers.index_add_(0, inverse_valid, weighted_points)
    region_density = torch.zeros(
        (region_count,), dtype=points.dtype, device=points.device
    )
    region_density.index_add_(0, inverse_valid, weights_for_centers)
    centers = centers / region_density.clamp_min(1e-8)[:, None]

    normals = _estimate_region_normals(
        points_for_normals, centers, normal_knn, chunk_size=normal_chunk_size
    )
    edges = _adaptive_face_edges(region_bounds)
    boundary_mask = _boundary_mask_from_edges(
        edges, normals, boundary_angle_degrees
    )
    region_patch_ids = _patch_ids_from_edges(
        edges, normals, boundary_angle_degrees
    )
    point_patch_ids = torch.full(
        (count,), -1, dtype=torch.long, device=points.device
    )
    valid_region = point_region_ids >= 0
    point_patch_ids[valid_region] = region_patch_ids[
        point_region_ids[valid_region]
    ]

    return TorchVoxelSurfaceRegions(
        region_centers=centers,
        region_normals=normals,
        boundary_mask=boundary_mask,
        voxel_indices=bounds_min,
        point_region_ids=point_region_ids,
        region_patch_ids=region_patch_ids,
        point_patch_ids=point_patch_ids,
        region_levels=region_levels,
        region_density=region_density,
        region_bounds=region_bounds,
    )


def _empty_regions(device: Any) -> TorchVoxelSurfaceRegions:
    torch = require_torch()
    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    empty_vec = torch.empty((0, 3), dtype=torch.float32, device=device)
    return TorchVoxelSurfaceRegions(
        region_centers=empty_vec,
        region_normals=empty_vec,
        boundary_mask=torch.empty((0,), dtype=torch.bool, device=device),
        voxel_indices=torch.empty((0, 3), dtype=torch.long, device=device),
        point_region_ids=empty_long,
        region_patch_ids=empty_long,
        point_patch_ids=empty_long,
        region_levels=empty_long,
        region_density=torch.empty((0,), dtype=torch.float32, device=device),
        region_bounds=torch.empty((0, 2, 3), dtype=torch.long, device=device),
    )


def _adaptive_face_edges(region_bounds: Any) -> Any:
    """Return face-adjacent region pairs for mixed-resolution AABBs."""

    torch = require_torch()
    count = int(region_bounds.shape[0])
    if count <= 1:
        return torch.empty(
            (0, 2), dtype=torch.long, device=region_bounds.device
        )
    bounds = region_bounds.detach().cpu().tolist()
    negative: dict[tuple[int, int], list[int]] = {}
    positive: dict[tuple[int, int], list[int]] = {}
    for index, (lower, upper) in enumerate(bounds):
        for axis in range(3):
            negative.setdefault((axis, int(lower[axis])), []).append(index)
            positive.setdefault((axis, int(upper[axis])), []).append(index)

    edges: set[tuple[int, int]] = set()
    for key, left_regions in positive.items():
        right_regions = negative.get(key, ())
        axis = key[0]
        other_axes = [value for value in range(3) if value != axis]
        for left in left_regions:
            for right in right_regions:
                if left == right:
                    continue
                overlap = True
                for other in other_axes:
                    low = max(bounds[left][0][other], bounds[right][0][other])
                    high = min(bounds[left][1][other], bounds[right][1][other])
                    if high <= low:
                        overlap = False
                        break
                if overlap:
                    edges.add((min(left, right), max(left, right)))
    if not edges:
        return torch.empty(
            (0, 2), dtype=torch.long, device=region_bounds.device
        )
    return torch.tensor(
        sorted(edges), dtype=torch.long, device=region_bounds.device
    )


def _boundary_mask_from_edges(
    edges: Any, normals: Any, boundary_angle_degrees: float
) -> Any:
    torch = require_torch()
    boundary = torch.zeros(
        (normals.shape[0],), dtype=torch.bool, device=normals.device
    )
    if edges.numel() == 0:
        return boundary
    threshold = torch.cos(
        torch.deg2rad(
            torch.tensor(
                float(boundary_angle_degrees),
                dtype=normals.dtype,
                device=normals.device,
            )
        )
    )
    alignment = torch.abs(
        (normals[edges[:, 0]] * normals[edges[:, 1]]).sum(dim=1)
    )
    cut = edges[alignment < threshold]
    if cut.numel() > 0:
        boundary[cut.reshape(-1)] = True
    return boundary


def _patch_ids_from_edges(
    edges: Any, normals: Any, boundary_angle_degrees: float
) -> Any:
    torch = require_torch()
    count = int(normals.shape[0])
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=normals.device)
    threshold = float(
        torch.cos(
            torch.deg2rad(torch.tensor(float(boundary_angle_degrees)))
        )
    )
    normal_rows = normals.detach().cpu()
    adjacency = [[] for _ in range(count)]
    for left, right in edges.detach().cpu().tolist():
        alignment = abs(float((normal_rows[left] * normal_rows[right]).sum()))
        if alignment >= threshold:
            adjacency[left].append(right)
            adjacency[right].append(left)
    labels = [-1] * count
    patch_id = 0
    for start in range(count):
        if labels[start] >= 0:
            continue
        labels[start] = patch_id
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = patch_id
                    stack.append(neighbor)
        patch_id += 1
    return torch.tensor(labels, dtype=torch.long, device=normals.device)


def _estimate_region_normals(
    points: Any, centers: Any, normal_knn: int, chunk_size: int = 4096
) -> Any:
    torch = require_torch()
    region_count = int(centers.shape[0])
    if region_count == 0:
        return centers
    point_count = int(points.shape[0])
    if point_count < 3:
        fallback = torch.tensor(
            [0.0, 0.0, 1.0], dtype=centers.dtype, device=centers.device
        )
        return fallback.view(1, 3).repeat(region_count, 1)

    k = max(3, min(int(normal_knn), point_count))
    chunk_size = max(1, int(chunk_size))
    normals = torch.empty(
        (region_count, 3), dtype=centers.dtype, device=centers.device
    )
    for start in range(0, region_count, chunk_size):
        end = min(start + chunk_size, region_count)
        distances = torch.cdist(centers[start:end], points)
        nearest = torch.topk(
            distances, k=k, largest=False, dim=1
        ).indices
        neighbors = points[nearest]
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        normals[start:end] = vh[:, -1, :]
    normals = torch.nn.functional.normalize(normals, dim=1)
    mean_normal = torch.nn.functional.normalize(normals.mean(dim=0), dim=0)
    flip = (normals @ mean_normal) < 0
    return torch.where(flip[:, None], -normals, normals)
