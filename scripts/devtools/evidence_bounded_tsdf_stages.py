"""Worklog 127 -- driver stages that are ALLOWED to read historical machinery.

`evidence_bounded_tsdf/` (scale / field / extraction / mesh_ops / synthetic) is
the control-experiment half and imports nothing historical. Everything in THIS
file runs after that mesh exists: the A/B baseline replay, the qualitative
exports, the review case table and the NURBS handoff.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evidence_bounded_tsdf import field as tsdf_field, mesh_ops

REGION_LABELS = ["table_top", "table_side_curved", "table_legs", "patio", "hedge"]

# Historical NURBS configuration, copied from worklog 119's own report so the
# baseline arm is replayed with the SAME capacity it always had. Not tuned here.
BASELINE_RESOLUTION_U = 8
BASELINE_RESOLUTION_V = 4
BASELINE_DEGREE_U = 2
BASELINE_DEGREE_V = 2
BASELINE_CORRECTION_ROUNDS = 2
BASELINE_PROJECTION_ITERATIONS = 4
BASELINE_MIN_PIXEL_SAMPLES = 32
# Patch sampling density for turning fitted NURBS patches into comparable
# geometry. A measurement choice, not a fit parameter.
BASELINE_PATCH_SAMPLES = 24


def replay_historical_visible_nurbs(
    model: Any, cameras: list[Any], representative_maps: list[torch.Tensor],
    depth_maps: list[torch.Tensor], *, device: str, max_charts: int = 0, progress=None,
) -> dict[str, Any]:
    """Replay the historical topology/boundary-first path WITHOUT modification:
    the frozen worklog 107/109 visible topology, then worklog 112-119's
    camera-observed chart blobs, then the existing NURBS fitter at its own
    historical capacity. Returns fitted-patch geometry so the SAME new metrics
    can be measured on it."""

    from osn_gs.surface.torch_camera_induced_visible_adjacency import (
        CameraInducedAdjacencyConfig, accumulate_image_space_pairs, apply_secondary_geometric_gate,
        filter_by_3d_locality,
    )
    from osn_gs.surface.torch_camera_observed_chart_domains import (
        build_view_chart_pixel_samples, valid_pixel_chart_mask,
    )
    from osn_gs.surface.torch_coverage_first_subset_partition import (
        CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph,
    )
    from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq
    from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel
    from dataclasses import replace as _dc_replace

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    with torch.no_grad():
        say("[historical replay, unchanged] surfel orientation + candidate graph")
        # Identical visible-selector restriction to worklogs 107/109/119: the
        # topology is built over non-uncertain surfels, and representative ids
        # (full-model indices) are remapped into that space.
        total_model_count = int(model.get_xyz.shape[0])
        uncertain = model.is_uncertain.reshape(-1).to(torch.bool)
        visible_selector = torch.nonzero(~uncertain, as_tuple=False).reshape(-1)
        visible_count = int(visible_selector.shape[0])
        full_to_visible = torch.full((total_model_count,), -1, dtype=torch.int64, device=device)
        full_to_visible[visible_selector] = torch.arange(visible_count, dtype=torch.int64, device=device)
        full_orientation = derive_surface_orientation_from_surfel(model)
        orientation = _dc_replace(
            full_orientation,
            gaussian_ids=full_orientation.gaussian_ids[visible_selector],
            positions=full_orientation.positions[visible_selector],
            tangent_axis_u=full_orientation.tangent_axis_u[visible_selector],
            tangent_axis_v=full_orientation.tangent_axis_v[visible_selector],
            surface_normal=full_orientation.surface_normal[visible_selector],
            tangent_scale_u=full_orientation.tangent_scale_u[visible_selector],
            tangent_scale_v=full_orientation.tangent_scale_v[visible_selector],
        )
        count = int(orientation.positions.shape[0])
        # `accumulate_image_space_pairs` walks 4-neighbour raster adjacency, so it
        # needs the (H, W) maps, not the flattened ones the field pipeline uses.
        remapped = [
            torch.where(rep >= 0, full_to_visible[rep.clamp(min=0)], torch.full_like(rep, -1)).reshape(
                int(camera.image_height), int(camera.image_width)
            )
            for rep, camera in zip(representative_maps, cameras)
        ]
        local_config = CoverageFirstPartitionConfig()
        config = CameraInducedAdjacencyConfig(local=local_config)
        graph = build_candidate_graph(orientation, config.local, retain_neighbor_index=True, progress=None)

        say("[historical replay, unchanged] camera-induced visible adjacency")
        raw_pairs, _support = accumulate_image_space_pairs(count, remapped, progress=None)
        local_pairs, _mask = filter_by_3d_locality(raw_pairs, count, graph)
        geometry = apply_secondary_geometric_gate(
            local_pairs, orientation, config, neighbor_index=graph.neighbor_index, progress=None
        )
        positive_edges = local_pairs[geometry["kept_mask"]]
        roots = _connected_component_roots(count, positive_edges, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
        subset_ids = subset_id_of_position[inverse]
        topology = {
            "visible_component_count": int(order.shape[0]),
            "largest_component_surfel_fraction": float(counts[order][0]) / count,
            "singleton_surfel_count": int((counts[order] == 1).sum()),
        }
        del graph, raw_pairs, local_pairs, geometry, positive_edges, roots

        say("[historical replay, unchanged] camera-observed chart blobs + NURBS fitting")
        sampled_points: list[np.ndarray] = []
        residuals: list[float] = []
        chart_regions: list[int] = []
        patch_faces: list[np.ndarray] = []
        vertex_base = 0
        chart_count = 0
        stop = False
        for view_index, camera in enumerate(cameras):
            if stop:
                break
            width = int(camera.image_width)
            height = int(camera.image_height)
            rep_map = remapped[view_index]
            valid = rep_map >= 0
            component_map = torch.where(valid, subset_ids[rep_map.clamp(min=0)], torch.full_like(rep_map, -1))
            world = tsdf_field.unproject_pixels(
                camera, torch.arange(height * width, device=device), depth_maps[view_index]
            ).reshape(height, width, 3)
            samples = build_view_chart_pixel_samples(view_index, component_map, rep_map, world)
            if samples.blob_count == 0:
                continue
            mask_valid = valid_pixel_chart_mask(samples, BASELINE_MIN_PIXEL_SAMPLES)
            blob_ids = torch.nonzero(mask_valid, as_tuple=False).reshape(-1).tolist()
            grouped = samples.pixel_order_by_blob
            grouped_uv = samples.pixel_uv[grouped]
            grouped_xyz = samples.pixel_xyz[grouped]
            offsets = samples.blob_pixel_offset.detach().cpu()
            for blob in blob_ids:
                if max_charts and chart_count >= max_charts:
                    stop = True
                    break
                start, end = int(offsets[blob]), int(offsets[blob + 1])
                uv_camera = grouped_uv[start:end]
                points = grouped_xyz[start:end]
                surface, uv_footpoint = fit_torch_visible_surface_lsq(
                    points, resolution_u=BASELINE_RESOLUTION_U, resolution_v=BASELINE_RESOLUTION_V,
                    degree_u=BASELINE_DEGREE_U, degree_v=BASELINE_DEGREE_V, initial_uv=uv_camera,
                    correction_rounds=BASELINE_CORRECTION_ROUNDS,
                    projection_iterations=BASELINE_PROJECTION_ITERATIONS,
                )
                residual = (surface.evaluate(uv_footpoint) - points).norm(dim=-1)
                residuals.append(float(residual.median().item()))
                grid = torch.linspace(0.0, 1.0, BASELINE_PATCH_SAMPLES, device=device)
                uu, vv = torch.meshgrid(grid, grid, indexing="ij")
                patch = surface.evaluate(torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1))
                sampled_points.append(patch.detach().cpu().numpy())
                side = BASELINE_PATCH_SAMPLES
                rows = np.arange(side - 1)
                a, b = np.meshgrid(rows, rows, indexing="ij")
                a = a.reshape(-1)
                b = b.reshape(-1)
                corner = a * side + b
                quad = np.stack([
                    np.stack([corner, corner + 1, corner + side], axis=1),
                    np.stack([corner + 1, corner + side + 1, corner + side], axis=1),
                ], axis=0).reshape(-1, 3)
                patch_faces.append(quad + vertex_base)
                vertex_base += side * side
                chart_regions.append(view_index)
                chart_count += 1
            del samples, world, component_map
            if view_index % 20 == 0:
                say(f"  historical chart fitting: view {view_index}/{len(cameras)}, {chart_count:,} charts")

    vertices = np.concatenate(sampled_points, axis=0) if sampled_points else np.zeros((0, 3), dtype=np.float32)
    faces = np.concatenate(patch_faces, axis=0) if patch_faces else np.zeros((0, 3), dtype=np.int64)
    return {
        "topology": topology,
        "fitted_chart_count": chart_count,
        "per_chart_median_residual": np.asarray(residuals, dtype=np.float64),
        "vertices": vertices.astype(np.float32),
        "faces": faces.astype(np.int64),
        "patch_samples_per_side": BASELINE_PATCH_SAMPLES,
        "capacity": {
            "resolution_u": BASELINE_RESOLUTION_U, "resolution_v": BASELINE_RESOLUTION_V,
            "degree_u": BASELINE_DEGREE_U, "degree_v": BASELINE_DEGREE_V,
            "correction_rounds": BASELINE_CORRECTION_ROUNDS,
            "source": "worklog 119 report's own nurbs_config_FOR_CONTROL_ONLY -- unchanged",
        },
    }


# ------------------------------------------------------------------ field slices
def build_slice(
    field: Any, centre: torch.Tensor, axis: int, half_extent: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(value, authority, crossing) images for a slice centred on `centre`."""

    h = field.h
    device = field.keys.device
    centre_index = torch.floor(centre / h).to(torch.int64)
    axes = [a for a in (0, 1, 2) if a != axis]
    span = torch.arange(-half_extent, half_extent + 1, device=device, dtype=torch.int64)
    first, second = torch.meshgrid(span, span, indexing="ij")
    side = int(span.numel())
    index = torch.empty((side * side, 3), dtype=torch.int64, device=device)
    index[:, axis] = centre_index[axis]
    index[:, axes[0]] = centre_index[axes[0]] + first.reshape(-1)
    index[:, axes[1]] = centre_index[axes[1]] + second.reshape(-1)
    keys = ((index[:, 0] + tsdf_field.KEY_BOUND) * tsdf_field._AXIS_SPAN + (index[:, 1] + tsdf_field.KEY_BOUND)) * tsdf_field._AXIS_SPAN + (index[:, 2] + tsdf_field.KEY_BOUND)
    value, support, found = field.lookup(keys)
    value_image = value.reshape(side, side).detach().cpu().numpy()
    found_image = found.reshape(side, side).detach().cpu().numpy()
    support_image = support.reshape(side, side).detach().cpu().numpy()
    stats = {
        "axis": axis, "half_extent_voxels": half_extent, "side_pixels": side,
        "world_extent": 2.0 * half_extent * h,
        "authoritative_pixels": int(found_image.sum()),
        "unknown_pixels": int((~found_image).sum()),
        "positive_pixels": int((found_image & (value_image > 0)).sum()),
        "negative_pixels": int((found_image & (value_image < 0)).sum()),
        "mean_support_where_authoritative": float(support_image[found_image].mean()) if found_image.any() else 0.0,
    }
    return value_image, found_image, support_image, stats


def slice_to_rgb(value: np.ndarray, authority: np.ndarray) -> np.ndarray:
    """UNKNOWN is a distinct, deliberately non-neutral colour: it must never be
    mistaken for free space."""

    height, width = value.shape
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 0.10
    image[..., 1] = 0.02
    image[..., 2] = 0.16          # UNKNOWN -- dark violet, unlike any field colour
    positive = authority & (value > 0)
    negative = authority & (value < 0)
    magnitude = np.clip(np.abs(np.nan_to_num(value, nan=0.0)), 0.0, 1.0)
    image[positive] = np.stack([
        0.15 + 0.20 * magnitude[positive],
        0.55 + 0.45 * magnitude[positive],
        0.95 * np.ones(int(positive.sum()), dtype=np.float32),
    ], axis=1)
    image[negative] = np.stack([
        0.95 * np.ones(int(negative.sum()), dtype=np.float32),
        0.45 + 0.35 * magnitude[negative],
        0.12 * np.ones(int(negative.sum()), dtype=np.float32),
    ], axis=1)
    # zero crossing: any authoritative pixel with an authoritative sign-opposite
    # 4-neighbour, drawn white so the extracted level set is visible in the slice
    sign = np.where(authority, np.sign(np.nan_to_num(value, nan=0.0)), 0.0)
    crossing = np.zeros_like(authority)
    crossing[:-1, :] |= (sign[:-1, :] * sign[1:, :]) < 0
    crossing[1:, :] |= (sign[:-1, :] * sign[1:, :]) < 0
    crossing[:, :-1] |= (sign[:, :-1] * sign[:, 1:]) < 0
    crossing[:, 1:] |= (sign[:, :-1] * sign[:, 1:]) < 0
    image[crossing] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8)).save(path)


def depth_to_rgb(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    """Turbo-like ramp; +inf (no hit) becomes a distinct dark magenta."""

    finite = np.isfinite(depth)
    normalized = np.zeros_like(depth, dtype=np.float32)
    span = max(high - low, 1e-9)
    normalized[finite] = np.clip((depth[finite] - low) / span, 0.0, 1.0)
    image = np.zeros(depth.shape + (3,), dtype=np.float32)
    image[..., 0] = np.clip(1.6 * normalized - 0.3, 0.0, 1.0)
    image[..., 1] = np.clip(1.2 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0.0, 1.0)
    image[..., 2] = np.clip(1.4 * (1.0 - normalized) - 0.15, 0.0, 1.0)
    image[~finite] = np.array([0.28, 0.02, 0.24], dtype=np.float32)
    return image


def signed_error_to_rgb(error: np.ndarray, valid: np.ndarray, scale: float) -> np.ndarray:
    """Blue = mesh in front of the renderer frontier, red = behind, grey = no
    comparison possible."""

    normalized = np.clip(np.nan_to_num(error, nan=0.0) / max(scale, 1e-9), -1.0, 1.0)
    image = np.full(error.shape + (3,), 0.35, dtype=np.float32)
    positive = valid & (normalized >= 0)
    negative = valid & (normalized < 0)
    image[positive] = np.stack([
        0.20 + 0.75 * normalized[positive], 0.20 * np.ones(int(positive.sum()), dtype=np.float32),
        0.20 * np.ones(int(positive.sum()), dtype=np.float32),
    ], axis=1)
    image[negative] = np.stack([
        0.20 * np.ones(int(negative.sum()), dtype=np.float32),
        0.30 * np.ones(int(negative.sum()), dtype=np.float32),
        0.20 + 0.75 * (-normalized[negative]),
    ], axis=1)
    return image


def support_to_rgb(support: np.ndarray, cap: float) -> np.ndarray:
    normalized = np.clip(np.asarray(support, dtype=np.float32) / max(cap, 1.0), 0.0, 1.0)
    return np.stack([
        0.95 - 0.75 * normalized, 0.25 + 0.6 * normalized, 0.30 + 0.55 * normalized
    ], axis=-1).astype(np.float32)
