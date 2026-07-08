from __future__ import annotations

"""Voxel regions for pre-NURBS visible surface organization."""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchVoxelSurfaceRegions:
    """Surface-aligned voxel regions used as curve placement candidates."""

    region_centers: Any
    region_normals: Any
    boundary_mask: Any
    voxel_indices: Any
    point_region_ids: Any

    @property
    def curve_points(self) -> Any:
        """Return the representative point for each voxel curve area."""

        return self.region_centers


def build_torch_voxel_surface_regions(
    points: Any,
    grid_resolution: int = 32,
    normal_knn: int = 16,
    boundary_angle_degrees: float = 35.0,
    min_points_per_voxel: int = 1,
) -> TorchVoxelSurfaceRegions:
    """Partition observed Gaussians into surface-aligned voxel regions.

    Voxel centers are snapped to the mean Gaussian position inside each occupied
    voxel. Local PCA normals approximate each region's orientation. Neighboring
    voxels whose normals diverge beyond ``boundary_angle_degrees`` are marked as
    boundaries, making them useful anchors for later curve placement.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    count = int(points.shape[0])
    if count == 0:
        empty_long = torch.empty((0,), dtype=torch.long, device=points.device)
        empty_vec = torch.empty((0, 3), dtype=torch.float32, device=points.device)
        return TorchVoxelSurfaceRegions(
            region_centers=empty_vec,
            region_normals=empty_vec,
            boundary_mask=torch.empty((0,), dtype=torch.bool, device=points.device),
            voxel_indices=torch.empty((0, 3), dtype=torch.long, device=points.device),
            point_region_ids=empty_long,
        )

    resolution = max(2, int(grid_resolution))
    min_corner = points.min(dim=0).values
    max_corner = points.max(dim=0).values
    span = torch.clamp(max_corner - min_corner, min=1e-6)
    normalized = torch.clamp((points - min_corner) / span, 0.0, 1.0 - 1e-7)
    point_voxels = torch.floor(normalized * resolution).long()
    point_voxels = torch.clamp(point_voxels, min=0, max=resolution - 1)

    linear = point_voxels[:, 0] * resolution * resolution + point_voxels[:, 1] * resolution + point_voxels[:, 2]
    unique_linear, inverse, counts = torch.unique(linear, sorted=True, return_inverse=True, return_counts=True)
    keep_regions = counts >= max(1, int(min_points_per_voxel))
    if not bool(keep_regions.all()):
        keep_ids = torch.nonzero(keep_regions, as_tuple=False).flatten()
        remap = torch.full((unique_linear.shape[0],), -1, dtype=torch.long, device=points.device)
        remap[keep_ids] = torch.arange(keep_ids.shape[0], dtype=torch.long, device=points.device)
        kept_points = remap[inverse] >= 0
        points = points[kept_points]
        point_voxels = point_voxels[kept_points]
        inverse = remap[inverse[kept_points]]
        unique_linear = unique_linear[keep_regions]
        counts = counts[keep_regions]
        count = int(points.shape[0])

    region_count = int(unique_linear.shape[0])
    centers = torch.zeros((region_count, 3), dtype=points.dtype, device=points.device)
    centers.index_add_(0, inverse, points)
    centers = centers / counts.to(points.dtype).clamp_min(1.0)[:, None]

    voxel_z = unique_linear % resolution
    voxel_y = (unique_linear // resolution) % resolution
    voxel_x = unique_linear // (resolution * resolution)
    voxel_indices = torch.stack([voxel_x, voxel_y, voxel_z], dim=1).long()

    normals = _estimate_region_normals(points, centers, inverse, normal_knn)
    boundary_mask = _normal_boundary_mask(voxel_indices, normals, resolution, boundary_angle_degrees)
    point_region_ids = inverse

    return TorchVoxelSurfaceRegions(
        region_centers=centers,
        region_normals=normals,
        boundary_mask=boundary_mask,
        voxel_indices=voxel_indices,
        point_region_ids=point_region_ids,
    )


def _estimate_region_normals(points: Any, centers: Any, inverse: Any, normal_knn: int) -> Any:
    torch = require_torch()
    region_count = int(centers.shape[0])
    if region_count == 0:
        return centers
    if int(points.shape[0]) < 3:
        fallback = torch.tensor([0.0, 0.0, 1.0], dtype=centers.dtype, device=centers.device)
        return fallback.view(1, 3).repeat(region_count, 1)

    normals = []
    k = max(3, min(int(normal_knn), int(points.shape[0])))
    for region_id in range(region_count):
        members = points[inverse == region_id]
        if int(members.shape[0]) >= 3:
            local = members
        else:
            distances = torch.linalg.norm(points - centers[region_id], dim=1)
            nearest = torch.topk(distances, k=k, largest=False).indices
            local = points[nearest]
        centered = local - local.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except RuntimeError:
            normal = torch.tensor([0.0, 0.0, 1.0], dtype=centers.dtype, device=centers.device)
        normals.append(torch.nn.functional.normalize(normal, dim=0))
    normals = torch.stack(normals, dim=0)
    mean_normal = torch.nn.functional.normalize(normals.mean(dim=0), dim=0)
    flip = (normals @ mean_normal) < 0
    return torch.where(flip[:, None], -normals, normals)


def _normal_boundary_mask(voxel_indices: Any, normals: Any, resolution: int, boundary_angle_degrees: float) -> Any:
    torch = require_torch()
    region_count = int(voxel_indices.shape[0])
    boundary = torch.zeros((region_count,), dtype=torch.bool, device=voxel_indices.device)
    if region_count <= 1:
        return boundary

    linear = voxel_indices[:, 0] * resolution * resolution + voxel_indices[:, 1] * resolution + voxel_indices[:, 2]
    order = torch.argsort(linear)
    sorted_linear = linear[order]
    cos_threshold = float(torch.cos(torch.deg2rad(torch.tensor(float(boundary_angle_degrees), device=normals.device))))
    offsets = (
        resolution * resolution,
        -resolution * resolution,
        resolution,
        -resolution,
        1,
        -1,
    )
    for region_id in range(region_count):
        current = linear[region_id]
        current_xyz = voxel_indices[region_id]
        for offset in offsets:
            neighbor_linear = current + int(offset)
            pos = torch.searchsorted(sorted_linear, neighbor_linear)
            if int(pos) >= region_count or sorted_linear[pos] != neighbor_linear:
                continue
            neighbor_id = order[pos]
            neighbor_xyz = voxel_indices[neighbor_id]
            if torch.abs(current_xyz - neighbor_xyz).sum() != 1:
                continue
            alignment = torch.abs(torch.dot(normals[region_id], normals[neighbor_id]))
            if float(alignment) < cos_threshold:
                boundary[region_id] = True
                boundary[neighbor_id] = True
    return boundary
