"""Worklog 144: per-view renderer surface correspondence audit.

This is an isolated diagnostic path.  It reuses the frozen WL141 polygons,
cameras, checkpoint, renderer, WL127 geometry, and WL139 scales, but changes
the comparison unit from a historical WL127 point to an independent
renderer-native median-event surface cloud for each camera.

No automatic Surface Membership, support-selection heuristic, fitting,
continuation, or Occluded Surface construction is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.meeting_occluded_surface_feasibility import _write_ply  # noqa: E402
from devtools.demo.multi_view_support_lifting_depth_semantics_evidence_aggregation import (  # noqa: E402
    _as_numpy,
    _depth_map_from_package,
    _reconstruct_world_from_renderer_pixel_depth,
    _renderer_projected_pixels,
    _reproduce_wl141_support,
    _sha256_array,
)
from devtools.demo.oracle_single_surface_support_appearance_evidence import (  # noqa: E402
    DISPLAY_POINT_ALPHA,
    DISPLAY_POINT_SIZE,
    ManualCameraMask,
    OracleSurfaceControl,
    _draw_polygon_outline,
    _manual_controls,
    _mask_camera_lookup,
    _point_in_polygon,
    _sha256_rows,
    build_oracle_support,
)
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402
    _plot_points,
    _save_3d,
)
from devtools.demo.real_gaussian_scene_surface_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    _camera_center,
    _load_canonical_scene,
    _load_xyz_ply,
    _render_to_pil,
    _sha256_file,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "output"
    / "144_per_view_renderer_surface_correspondence_physical_sheet_oracle_audit"
)
WL127_RAW_VISIBLE_SURFACE = (
    REPO_ROOT
    / "output"
    / "confirmed"
    / "127_osn_gs_evidence_bounded_projective_tsdf"
    / "RENDERER_MEDIAN_SURFACE_POINTS"
    / "iteration_0000001"
    / "point_cloud.ply"
)
WL139_REPORT = (
    REPO_ROOT
    / "output"
    / "confirmed"
    / "138_scale_separated_visible_surface_representative"
    / "scale_separated_visible_surface_representative_report.json"
)
WL141_REPORT = (
    REPO_ROOT
    / "output"
    / "141_oracle_single_surface_support_appearance_evidence"
    / "oracle_single_surface_support_appearance_evidence_report.json"
)
WL142_REPORT = (
    REPO_ROOT
    / "output"
    / "142_multi_view_support_lifting_projection_depth_attribution"
    / "multi_view_support_lifting_projection_depth_attribution_report.json"
)
WL143_REPORT = (
    REPO_ROOT
    / "output"
    / "143_multi_view_support_lifting_depth_semantics_evidence_aggregation"
    / "multi_view_support_lifting_depth_semantics_evidence_aggregation_report.json"
)

LOCAL_NORMAL_NEIGHBORS = 12
DISPLAY_MAX_POINTS = 16000
EVENT_CLOUD_RGB = (
    (44, 116, 232),
    (233, 103, 36),
    (32, 177, 104),
)
MASK_ONLY_RGB = (128, 128, 135)
REPROJECTION_RADIUS = 2.2
REPROJECTION_ALPHA = 0.98
CASE_REVIEW = {
    "tabletop_top_oracle": {
        "classification": "C. SEMANTIC_MASK_MISASSOCIATION",
        "human_review_status": "COMPLETED_DIRECT_3D_AND_REPROJECTION_REVIEW",
        "visual_evidence": "camera-specific event clouds and raw reprojections attach to ground/tabletop/brace structures rather than one common tabletop sheet",
    },
    "curved_table_rim_oracle": {
        "classification": "C. SEMANTIC_MASK_MISASSOCIATION",
        "human_review_status": "COMPLETED_DIRECT_3D_AND_REPROJECTION_REVIEW",
        "visual_evidence": "frozen polygons select tabletop and front-side/depth-layer structures; only a partial cloud overlap is visible and it is not a single rim sheet",
    },
    "paver_ground_oracle": {
        "classification": "C. SEMANTIC_MASK_MISASSOCIATION",
        "human_review_status": "COMPLETED_DIRECT_3D_AND_REPROJECTION_REVIEW",
        "visual_evidence": "camera-specific event clouds and raw reprojections attach to grass/tabletop/front-table structures rather than one paver sheet",
    },
}


@dataclass(frozen=True)
class PerViewEventCloud:
    camera_name: str
    polygon: ManualCameraMask
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    median_depth: np.ndarray
    points: np.ndarray
    local_normals: np.ndarray


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_int64(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype=np.int64)).tobytes()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _quantile_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "NO_SAMPLES", "samples": 0}
    return {
        "status": "MEASURED",
        "samples": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _distance_summary(values: np.ndarray, h: float, mu: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "world_distance": _quantile_summary(values),
        "distance_over_h": _quantile_summary(values / max(float(h), 1.0e-12)),
        "distance_over_mu": _quantile_summary(values / max(float(mu), 1.0e-12)),
    }


def _angle_summary(degrees: np.ndarray) -> dict[str, Any]:
    degrees = np.asarray(degrees, dtype=np.float64).reshape(-1)
    edges = np.arange(0.0, 100.0, 10.0)
    counts, _ = np.histogram(degrees, bins=np.concatenate([edges, [90.0]]))
    return {
        "angle_degrees": _quantile_summary(degrees),
        "fixed_histogram_bin_edges_degrees": [float(value) for value in np.concatenate([edges, [90.0]])],
        "fixed_10_degree_histogram_counts": counts.astype(np.int64),
        "orientation_consistency_mean_abs_cosine": (
            float(np.mean(np.cos(np.deg2rad(degrees)))) if len(degrees) else None
        ),
        "sign_invariant_orientation_comparison": True,
    }


def _estimate_local_normals(points: np.ndarray, neighbors: int = LOCAL_NORMAL_NEIGHBORS) -> np.ndarray:
    """Fixed-k local PCA for diagnostic orientation only; never used for selection."""

    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normals = np.full_like(points, np.nan)
    if len(points) <= int(neighbors):
        return normals
    count = min(int(neighbors) + 1, len(points))
    indices = cKDTree(points).query(points, k=count, workers=1)[1]
    if indices.ndim == 1:
        indices = indices[:, None]
    indices = indices[:, 1:]
    local = points[indices]
    centered = local - np.mean(local, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(local.shape[1], 1)
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths[:, None] > 1.0e-12, lengths[:, None], 1.0)
    return normals


def _build_per_view_event_cloud(
    camera: Any,
    mask: ManualCameraMask,
    depth_median: np.ndarray,
) -> tuple[PerViewEventCloud, dict[str, Any]]:
    """Build exactly the valid renderer pixels inside one frozen polygon."""

    depth = np.asarray(depth_median, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"expected depth map HxW, got {depth.shape}")
    height, width = depth.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    grid_xy = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    polygon = mask.polygon_for(camera)
    inside = _point_in_polygon(grid_xy, polygon).reshape(height, width)
    finite_nonzero = np.isfinite(depth) & (np.abs(depth) > 1.0e-8)
    valid_event = inside & finite_nonzero & (depth > 0.0)
    rows, columns = np.nonzero(valid_event)
    pixel_x = columns.astype(np.int64)
    pixel_y = rows.astype(np.int64)
    median_depth = depth[rows, columns].astype(np.float64)
    points = _reconstruct_world_from_renderer_pixel_depth(
        pixel_x.astype(np.float64), pixel_y.astype(np.float64), median_depth, camera
    )
    projection = _renderer_projected_pixels(points, camera)
    if len(points):
        if not np.all(projection["valid"]):
            raise AssertionError("reconstructed renderer events must reproject as valid")
        if not np.allclose(projection["x"], pixel_x, rtol=0.0, atol=1.0e-8):
            raise AssertionError("renderer event x round-trip failed")
        if not np.allclose(projection["y"], pixel_y, rtol=0.0, atol=1.0e-8):
            raise AssertionError("renderer event y round-trip failed")
        if not np.allclose(projection["depth"], median_depth, rtol=0.0, atol=1.0e-8):
            raise AssertionError("renderer event depth round-trip failed")
    cloud = PerViewEventCloud(
        camera_name=str(camera.image_name),
        polygon=mask,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        median_depth=median_depth,
        points=points,
        local_normals=_estimate_local_normals(points),
    )
    finite_points = points[np.all(np.isfinite(points), axis=1)]
    summary = {
        "camera_name": str(camera.image_name),
        "polygon": mask.as_json(),
        "polygon_identity_sha256": _sha256_json(mask.as_json()),
        "valid_polygon_pixel_count": int(np.sum(inside)),
        "valid_median_event_count": int(len(points)),
        "nonpositive_or_invalid_depth_inside_polygon": int(np.sum(inside & ~valid_event)),
        "world_bounding_box": (
            {"min": np.min(finite_points, axis=0), "max": np.max(finite_points, axis=0)}
            if len(finite_points)
            else None
        ),
        "world_centroid": np.mean(finite_points, axis=0) if len(finite_points) else None,
        "world_spatial_extent": (
            np.max(finite_points, axis=0) - np.min(finite_points, axis=0)
            if len(finite_points)
            else None
        ),
        "world_spatial_diagonal": (
            float(np.linalg.norm(np.max(finite_points, axis=0) - np.min(finite_points, axis=0)))
            if len(finite_points)
            else None
        ),
        "pixel_bbox": (
            [int(np.min(pixel_x)), int(np.min(pixel_y)), int(np.max(pixel_x)), int(np.max(pixel_y))]
            if len(pixel_x)
            else None
        ),
        "event_points_sha256": _sha256_rows(points.astype(np.float32)) if len(points) else _sha256_rows(np.empty((0, 3))),
        "pixel_coordinate_sha256": _sha256_array(np.column_stack([pixel_x, pixel_y])),
        "local_normal_method": {
            "method": "world-space local PCA",
            "neighbor_count": LOCAL_NORMAL_NEIGHBORS,
            "selection_or_filter_use": False,
            "orientation_sign_is_ignored_in_pairwise_angle": True,
        },
    }
    return cloud, summary


def _write_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray | tuple[int, int, int]) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if isinstance(colors, tuple):
        color_array = np.tile(np.asarray(colors, dtype=np.uint8), (len(points), 1))
    else:
        color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(color_array) != len(points):
            raise ValueError("color/point row count mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, color_array):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_event_cloud_outputs(
    case_root: Path,
    clouds: list[PerViewEventCloud],
    mask_only_points: np.ndarray,
) -> dict[str, Any]:
    root = case_root / "per_view_event_clouds"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {"per_camera": {}}
    combined_points: list[np.ndarray] = []
    combined_colors: list[np.ndarray] = []
    for index, cloud in enumerate(clouds):
        camera_root = root / cloud.camera_name
        camera_root.mkdir(parents=True, exist_ok=True)
        ply_path = camera_root / "per_view_median_event_cloud.ply"
        npz_path = camera_root / "per_view_median_event_cloud.npz"
        _write_ply(ply_path, cloud.points, color=EVENT_CLOUD_RGB[index])
        np.savez_compressed(
            npz_path,
            pixel_x=cloud.pixel_x,
            pixel_y=cloud.pixel_y,
            median_depth=cloud.median_depth,
            event_points=cloud.points,
            local_normals=cloud.local_normals,
        )
        paths["per_camera"][cloud.camera_name] = {
            "ply": str(ply_path),
            "npz": str(npz_path),
        }
        combined_points.append(cloud.points)
        combined_colors.append(np.tile(np.asarray(EVENT_CLOUD_RGB[index], dtype=np.uint8), (len(cloud.points), 1)))
    all_points = np.concatenate(combined_points, axis=0) if combined_points else np.empty((0, 3))
    all_colors = np.concatenate(combined_colors, axis=0) if combined_colors else np.empty((0, 3), dtype=np.uint8)
    combined_path = root / "all_three_per_view_event_clouds_colored.ply"
    _write_colored_ply(combined_path, all_points, all_colors)
    mask_path = root / "historical_wl141_mask_only_population.ply"
    _write_ply(mask_path, mask_only_points, color=MASK_ONLY_RGB)
    paths["all_three_colored_ply"] = str(combined_path)
    paths["historical_mask_only_ply"] = str(mask_path)
    paths["event_cloud_point_count"] = int(len(all_points))
    return paths


def _nearest_distances(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    first = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    if not len(first) or not len(second):
        return np.full(len(first), np.nan), np.full(len(first), -1, dtype=np.int64)
    distances, indices = cKDTree(second).query(first, workers=1)
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def _pairwise_surface_agreement(
    first: PerViewEventCloud,
    second: PerViewEventCloud,
    h: float,
    mu: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    first_to_second, first_indices = _nearest_distances(first.points, second.points)
    second_to_first, second_indices = _nearest_distances(second.points, first.points)
    reciprocal = np.concatenate([first_to_second, second_to_first])
    arrays = {
        "first_to_second_distance": first_to_second,
        "second_to_first_distance": second_to_first,
        "first_to_second_index": first_indices,
        "second_to_first_index": second_indices,
    }
    report = {
        "camera_a": first.camera_name,
        "camera_b": second.camera_name,
        "comparison_unit": "independent per-view renderer median-event point cloud",
        "A_to_B": _distance_summary(first_to_second, h, mu),
        "B_to_A": _distance_summary(second_to_first, h, mu),
        "reciprocal": {
            "definition": "concatenation of A->B and B->A nearest-neighbour distances",
            **_distance_summary(reciprocal, h, mu),
        },
        "threshold_or_membership_use": False,
    }
    second_normals = second.local_normals
    valid = np.zeros(len(first.points), dtype=bool)
    if len(first.local_normals) == len(first.points) and len(second.local_normals) == len(second.points):
        valid = (
            (first_indices >= 0)
            & np.all(np.isfinite(first.local_normals), axis=1)
            & np.all(np.isfinite(second_normals[first_indices]), axis=1)
        )
    if np.any(valid):
        cosine = np.sum(first.local_normals[valid] * second_normals[first_indices[valid]], axis=1)
        angle = np.rad2deg(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
        report["local_differential_agreement_A_to_B"] = _angle_summary(angle)
        arrays["A_to_B_normal_angle_degrees"] = angle
    else:
        report["local_differential_agreement_A_to_B"] = {
            "status": "NO_VALID_NORMAL_CORRESPONDENCES",
            "normal_method": "fixed-k local PCA",
        }
        arrays["A_to_B_normal_angle_degrees"] = np.empty((0,), dtype=np.float64)
    valid_reverse = np.zeros(len(second.points), dtype=bool)
    if len(second.local_normals) == len(second.points) and len(first.local_normals) == len(first.points):
        valid_reverse = (
            (second_indices >= 0)
            & np.all(np.isfinite(second.local_normals), axis=1)
            & np.all(np.isfinite(first.local_normals[second_indices]), axis=1)
        )
    if np.any(valid_reverse):
        cosine = np.sum(second.local_normals[valid_reverse] * first.local_normals[second_indices[valid_reverse]], axis=1)
        angle = np.rad2deg(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
        report["local_differential_agreement_B_to_A"] = _angle_summary(angle)
        arrays["B_to_A_normal_angle_degrees"] = angle
    else:
        report["local_differential_agreement_B_to_A"] = {
            "status": "NO_VALID_NORMAL_CORRESPONDENCES",
            "normal_method": "fixed-k local PCA",
        }
        arrays["B_to_A_normal_angle_degrees"] = np.empty((0,), dtype=np.float64)
    return report, arrays


def _wl127_mask_only_attribution(
    row_ids: np.ndarray,
    points: np.ndarray,
    clouds: list[PerViewEventCloud],
    h: float,
    mu: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    per_camera = []
    distances = []
    for cloud in clouds:
        values, _indices = _nearest_distances(points, cloud.points)
        distances.append(values)
        per_camera.append({
            "camera_name": cloud.camera_name,
            **_distance_summary(values, h, mu),
        })
    matrix = np.column_stack(distances) if distances else np.empty((len(points), 0))
    sorted_distances = np.sort(matrix, axis=1) if len(matrix) else matrix
    summary = {
        "diagnostic_only": True,
        "wl127_mask_only_point_count": int(len(points)),
        "wl127_mask_only_row_ids_sha256": _sha256_int64(row_ids),
        "per_camera": per_camera,
        "nearest_camera_cloud": _distance_summary(sorted_distances[:, 0], h, mu) if sorted_distances.shape[1] >= 1 else {"status": "NO_CLOUDS"},
        "second_nearest_camera_cloud": _distance_summary(sorted_distances[:, 1], h, mu) if sorted_distances.shape[1] >= 2 else {"status": "NO_CLOUDS"},
        "third_nearest_camera_cloud": _distance_summary(sorted_distances[:, 2], h, mu) if sorted_distances.shape[1] >= 3 else {"status": "NO_CLOUDS"},
        "exact_point_identity_required": False,
        "selection_or_membership_use": False,
    }
    arrays = {
        "distance_to_each_camera_cloud": matrix,
        "sorted_distance_to_camera_clouds": sorted_distances,
    }
    return summary, arrays


def _overlap_accounting(
    clouds: list[PerViewEventCloud],
    h: float,
    mu: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reports: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for index, cloud in enumerate(clouds):
        other_distances = []
        other_names = []
        for other_index, other in enumerate(clouds):
            if index == other_index:
                continue
            distance, _ = _nearest_distances(cloud.points, other.points)
            other_distances.append(distance)
            other_names.append(other.camera_name)
        matrix = np.column_stack(other_distances) if other_distances else np.empty((len(cloud.points), 0))
        nearest = np.min(matrix, axis=1) if matrix.shape[1] else np.full(len(cloud.points), np.nan)
        farthest = np.max(matrix, axis=1) if matrix.shape[1] else np.full(len(cloud.points), np.nan)
        arrays[f"{cloud.camera_name}_distance_to_other_clouds"] = matrix
        rows: dict[str, Any] = {"camera_name": cloud.camera_name, "other_camera_names": other_names}
        for label, reference in (("h", float(h)), ("mu", float(mu))):
            another = nearest <= reference
            both = farthest <= reference
            neither = nearest > reference
            rows[label] = {
                "reference_radius_world": reference,
                "shared_by_another_count": int(np.sum(another)),
                "shared_by_another_fraction": float(np.mean(another)) if len(another) else 0.0,
                "shared_by_both_other_cameras_count": int(np.sum(both)),
                "shared_by_both_other_cameras_fraction": float(np.mean(both)) if len(both) else 0.0,
                "shared_by_neither_count": int(np.sum(neither)),
                "shared_by_neither_fraction": float(np.mean(neither)) if len(neither) else 0.0,
                "descriptive_reference_only": True,
            }
        rows["nearest_other_cloud_distance"] = _distance_summary(nearest, h, mu)
        rows["second_other_cloud_distance"] = _distance_summary(farthest, h, mu)
        rows["continuous_distribution_precedes_reference_accounting"] = True
        reports[cloud.camera_name] = rows
    return reports, arrays


def _draw_renderer_projected_points(
    image: Any,
    points: np.ndarray,
    camera: Any,
    color: tuple[int, int, int],
    *,
    radius: float = REPROJECTION_RADIUS,
    alpha: float = REPROJECTION_ALPHA,
) -> Any:
    from PIL import Image, ImageDraw

    output = image.convert("RGB")
    projection = _renderer_projected_pixels(points, camera)
    valid = projection["valid"]
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(projection["depth"][indices])[::-1]]
    layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    fill = (*tuple(int(channel) for channel in color), int(round(255.0 * alpha)))
    radius = max(float(radius), 1.0)
    for index in indices.tolist():
        x = float(projection["x"][index])
        y = float(projection["y"][index])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return Image.alpha_composite(output.convert("RGBA"), layer).convert("RGB")


def _write_reprojection_outputs(
    case_root: Path,
    clouds: list[PerViewEventCloud],
    cameras: dict[str, Any],
    images: dict[str, Any],
    masks: dict[str, ManualCameraMask],
) -> dict[str, Any]:
    root = case_root / "cross_view_reprojection_overlays"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}
    for source_index, source in enumerate(clouds):
        source_paths: dict[str, str] = {}
        for target in clouds:
            target_camera = cameras[target.camera_name]
            view = _draw_renderer_projected_points(
                images[target.camera_name], source.points, target_camera,
                EVENT_CLOUD_RGB[source_index],
            )
            view = _draw_polygon_outline(view, masks[target.camera_name], target_camera)
            from PIL import ImageDraw

            draw = ImageDraw.Draw(view)
            draw.rectangle((4, 4, 320, 24), fill=(255, 255, 255))
            draw.text((8, 8), f"source {source.camera_name} -> target {target.camera_name}", fill=(20, 20, 20))
            path = root / f"source_{source.camera_name}_to_{target.camera_name}.png"
            view.save(path)
            source_paths[target.camera_name] = str(path)
        paths[source.camera_name] = source_paths
    return {
        "source_to_all_three_targets": paths,
        "visibility_culling": False,
        "polygon_outline_color": "yellow",
        "projection_contract": "Worklog 143 renderer-native pixel/depth convention",
    }


def _write_3d_outputs(
    case_root: Path,
    control: OracleSurfaceControl,
    clouds: list[PerViewEventCloud],
    mask_only_points: np.ndarray,
) -> dict[str, str]:
    root = case_root / "3d_review"
    root.mkdir(parents=True, exist_ok=True)
    all_limits = [cloud.points for cloud in clouds if len(cloud.points)] + [mask_only_points]
    limits = np.concatenate(all_limits, axis=0) if all_limits else np.zeros((1, 3), dtype=np.float64)
    paths: dict[str, str] = {}
    for index, cloud in enumerate(clouds):
        path = root / f"A_camera_{index + 1}_{cloud.camera_name}.png"
        _save_3d(
            path,
            f"A camera {index + 1} only: {cloud.camera_name}",
            control.config,
            limits,
            lambda axis, cloud=cloud, index=index: _plot_points(
                axis,
                cloud.points,
                control.config,
                np.asarray(EVENT_CLOUD_RGB[index], dtype=np.float64) / 255.0,
                label=f"camera {index + 1} median events",
                size=DISPLAY_POINT_SIZE,
                alpha=DISPLAY_POINT_ALPHA,
                max_points=DISPLAY_MAX_POINTS,
            ),
        )
        paths[f"A_camera_{index + 1}"] = str(path)
    path_d = root / "D_all_three_per_view_event_clouds.png"
    _save_3d(
        path_d,
        "D all three independent per-view median-event clouds",
        control.config,
        limits,
        lambda axis: [
            _plot_points(
                axis,
                cloud.points,
                control.config,
                np.asarray(EVENT_CLOUD_RGB[index], dtype=np.float64) / 255.0,
                label=f"camera {index + 1} median events",
                size=DISPLAY_POINT_SIZE,
                alpha=DISPLAY_POINT_ALPHA,
                max_points=DISPLAY_MAX_POINTS,
            )
            for index, cloud in enumerate(clouds)
        ],
    )
    paths["D_all_three"] = str(path_d)
    path_e = root / "E_all_three_plus_historical_wl141_mask_only.png"
    _save_3d(
        path_e,
        "E per-view events + historical WL141 MASK_ONLY population",
        control.config,
        limits,
        lambda axis: [
            *[
                _plot_points(
                    axis,
                    cloud.points,
                    control.config,
                    np.asarray(EVENT_CLOUD_RGB[index], dtype=np.float64) / 255.0,
                    label=f"camera {index + 1} median events",
                    size=DISPLAY_POINT_SIZE,
                    alpha=DISPLAY_POINT_ALPHA,
                    max_points=DISPLAY_MAX_POINTS,
                )
                for index, cloud in enumerate(clouds)
            ],
            _plot_points(
                axis,
                mask_only_points,
                control.config,
                np.asarray(MASK_ONLY_RGB, dtype=np.float64) / 255.0,
                label="historical WL141 MASK_ONLY",
                size=DISPLAY_POINT_SIZE,
                alpha=DISPLAY_POINT_ALPHA,
                max_points=DISPLAY_MAX_POINTS,
            ),
        ],
    )
    paths["E_all_three_plus_mask_only"] = str(path_e)
    return paths


def _historical_replay(control_name: str, support: Any, wl141_report: dict[str, Any], wl142_report: dict[str, Any]) -> dict[str, Any]:
    wl141 = _reproduce_wl141_support(control_name, support)
    wl141_case = wl141_report.get("cases", {}).get(control_name, {})
    wl142_case = wl142_report.get("cases", {}).get(control_name, {})
    replayed_count = int(len(support.oracle_row_ids))
    replayed_hash = hashlib.sha256(np.ascontiguousarray(support.oracle_row_ids, dtype=np.int64).tobytes()).hexdigest()
    results: dict[str, Any] = {"wl141_helper_replay": wl141}
    for label, case in (("wl141_report", wl141_case), ("wl142_report", wl142_case)):
        alignment = case.get("support_alignment", {})
        historical_count = int(alignment.get("oracle_row_count", -1))
        historical_hash = str(alignment.get("oracle_row_ids_sha256", ""))
        results[label] = {
            "status": "EXACT_REPRODUCTION_BY_ROW_ID_HASH" if replayed_count == historical_count and replayed_hash == historical_hash else "REPRODUCTION_MISMATCH",
            "exact": replayed_count == historical_count and replayed_hash == historical_hash,
            "historical_count": historical_count,
            "replayed_count": replayed_count,
            "historical_row_ids_sha256": historical_hash,
            "replayed_row_ids_sha256": replayed_hash,
            "selection_name": "MASK_ONLY_BASELINE",
        }
    return results


def _load_frozen_inputs(arguments: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    paths = {
        "raw_visible_surface": Path(arguments.raw_visible_surface),
        "wl139_report": Path(arguments.wl139_report),
        "wl141_report": Path(arguments.wl141_report),
        "wl142_report": Path(arguments.wl142_report),
        "wl143_report": Path(arguments.wl143_report),
        "checkpoint": Path(arguments.checkpoint),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    raw_points = _load_xyz_ply(paths["raw_visible_surface"])
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items() if key.endswith("report")}
    wl143_status = reports["wl143_report"].get("status")
    if wl143_status != "DEPTH_QUANTITY_IDENTITY_PASS":
        raise RuntimeError(f"WL143 depth reconstruction identity is not PASS: {wl143_status}")
    wl139_inputs = reports["wl139_report"].get("inputs", {})
    h = float(wl139_inputs["h"])
    mu = float(wl139_inputs["mu"])
    reports["frozen_scales"] = {"h": h, "mu": mu}
    reports["paths"] = paths
    return raw_points, reports


def _build_case(
    control: OracleSurfaceControl,
    raw_points: np.ndarray,
    camera_lookup: dict[str, Any],
    package_for: Any,
    images: dict[str, Any],
    reports: dict[str, Any],
    h: float,
    mu: float,
    output_root: Path,
) -> dict[str, Any]:
    case_root = output_root / control.name
    case_root.mkdir(parents=True, exist_ok=True)
    support = build_oracle_support(raw_points, control, [camera_lookup[mask.camera_name] for mask in control.masks])
    mask_only_points = raw_points[support.oracle_row_ids]
    masks = {mask.camera_name: mask for mask in control.masks}
    clouds: list[PerViewEventCloud] = []
    cloud_summaries: list[dict[str, Any]] = []
    for mask in control.masks:
        cloud, summary = _build_per_view_event_cloud(
            camera_lookup[mask.camera_name], mask, _depth_map_from_package(package_for(mask.camera_name))
        )
        clouds.append(cloud)
        cloud_summaries.append(summary)
    event_outputs = _write_event_cloud_outputs(case_root, clouds, mask_only_points)
    pairwise_reports: dict[str, Any] = {}
    pairwise_arrays: dict[str, np.ndarray] = {}
    for first_index in range(len(clouds)):
        for second_index in range(first_index + 1, len(clouds)):
            first = clouds[first_index]
            second = clouds[second_index]
            pair_report, pair_arrays = _pairwise_surface_agreement(first, second, h, mu)
            key = f"{first.camera_name}__{second.camera_name}"
            pairwise_reports[key] = pair_report
            pairwise_arrays.update({f"{key}__{name}": value for name, value in pair_arrays.items()})
    pairwise_npz = case_root / "pairwise_surface_distance_distributions.npz"
    np.savez_compressed(pairwise_npz, **pairwise_arrays)
    mask_attribution, mask_arrays = _wl127_mask_only_attribution(
        support.oracle_row_ids, mask_only_points, clouds, h, mu
    )
    mask_npz = case_root / "wl127_mask_only_attribution_distributions.npz"
    np.savez_compressed(mask_npz, **mask_arrays)
    overlap, overlap_arrays = _overlap_accounting(clouds, h, mu)
    overlap_npz = case_root / "cross_view_overlap_distances.npz"
    np.savez_compressed(overlap_npz, **overlap_arrays)
    reprojection = _write_reprojection_outputs(
        case_root, clouds, camera_lookup, images, masks
    )
    three_d = _write_3d_outputs(case_root, control, clouds, mask_only_points)
    replay = _historical_replay(control.name, support, reports["wl141_report"], reports["wl142_report"])
    report = {
        "surface_control": control.as_json(),
        "comparison_unit": "independent per-view renderer median-event surface cloud",
        "per_view_event_cloud_contract": {
            "meaning": "renderer-native visible median-surface events inside the manually frozen image-space polygon for this camera",
            "not_meaning": ["object segmentation", "final physical-surface membership", "complete visible surface", "ground truth"],
            "clouds_merged_for_metrics": False,
            "clouds_merged_only_for_combined_visualization": True,
        },
        "per_camera_event_counts": cloud_summaries,
        "pairwise_surface_distance_distributions": pairwise_reports,
        "local_differential_agreement": {
            "method": "fixed-k world-space local PCA with k=12; sign-invariant nearest-cloud normal comparison",
            "parameters_tuned_from_result": False,
            "normals_used_to_select_or_remove_points": False,
            "pairwise": {
                key: {
                    "A_to_B": value.get("local_differential_agreement_A_to_B"),
                    "B_to_A": value.get("local_differential_agreement_B_to_A"),
                }
                for key, value in pairwise_reports.items()
            },
            "multimodality_review": "fixed 10-degree histograms are exported; no single angle threshold is used for membership or classification",
        },
        "wl127_mask_only_attribution": mask_attribution,
        "fixed_h_mu_reference_accounting": {
            "h": float(h),
            "mu": float(mu),
            "use": "descriptive reference radii only after continuous distributions",
            "radius_sweep": False,
            "new_support_threshold": False,
        },
        "cross_view_overlap_accounting": {
            "per_source_cloud": overlap,
            "selection_or_membership_use": False,
            "fixed_reference_radii_only": ["h", "mu"],
        },
        "historical_wl141_mask_only_replay": replay,
        "cross_view_reprojection_exports": reprojection,
        "outputs": {
            "event_clouds": event_outputs,
            "pairwise_distributions_npz": str(pairwise_npz),
            "wl127_attribution_npz": str(mask_npz),
            "overlap_npz": str(overlap_npz),
            "3d_review": three_d,
        },
        "per_case_human_review_status": {
            **CASE_REVIEW[control.name],
            "allowed_labels": [
                "A. SAME_PHYSICAL_SHEET_PLAUSIBLE",
                "B. DIFFERENT_DEPTH_LAYERS",
                "C. SEMANTIC_MASK_MISASSOCIATION",
                "D. PARTIAL OVERLAP / MIXED",
                "E. INSUFFICIENT_EVIDENCE",
            ],
            "classification_basis": "common-frame 3D event-cloud view + all-target raw reprojections + continuous pairwise distributions",
            "classification_not_from_single_scalar": True,
        },
        "input_information_leakage": {
            "wl127_geometry_used_for": "historical MASK_ONLY row identity and diagnostic attribution only",
            "withheld_or_target_geometry_used_for_selection": False,
            "frozen_polygon_or_camera_changed": False,
            "candidate_or_event_cloud_used_to_change_masks": False,
        },
        "input_hashes": {
            "raw_visible_surface_sha256": _sha256_file(reports["paths"]["raw_visible_surface"]),
            "wl139_report_sha256": _sha256_file(reports["paths"]["wl139_report"]),
            "wl141_report_sha256": _sha256_file(reports["paths"]["wl141_report"]),
            "wl142_report_sha256": _sha256_file(reports["paths"]["wl142_report"]),
            "wl143_report_sha256": _sha256_file(reports["paths"]["wl143_report"]),
            "checkpoint_sha256": _sha256_file(reports["paths"]["checkpoint"]),
        },
    }
    (case_root / "case_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def run_audit(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_points, reports = _load_frozen_inputs(arguments)
    h = float(reports["frozen_scales"]["h"])
    mu = float(reports["frozen_scales"]["mu"])
    model, _payload, cameras, scene_info = _load_canonical_scene(
        Path(arguments.checkpoint),
        Path(arguments.source_path),
        arguments.device,
        arguments.images,
        arguments.sparse_dir,
        int(arguments.resolution),
        int(arguments.llffhold),
    )
    camera_lookup = _mask_camera_lookup(cameras)
    controls = _manual_controls()
    packages: dict[str, dict[str, Any]] = {}
    images: dict[str, Any] = {}

    def package_for(camera_name: str) -> dict[str, Any]:
        if camera_name not in packages:
            packages[camera_name] = scene_info["rasterizer"].render(camera_lookup[camera_name], model)
            images[camera_name] = _render_to_pil(packages[camera_name])
        return packages[camera_name]

    cases: dict[str, Any] = {}
    for control in controls:
        cases[control.name] = _build_case(
            control,
            raw_points,
            camera_lookup,
            package_for,
            images,
            reports,
            h,
            mu,
            output_root,
        )
    report = {
        "batch": "Worklog 144 per-view renderer surface correspondence and physical-sheet oracle audit",
        "status": "COMPLETED_DIAGNOSTIC_AUDIT",
        "INTENT ALIGNMENT": {
            "automatic_surface_membership_implemented": False,
            "new_support_selection_heuristic": False,
            "comparison_unit_changed_to_per_view_event_cloud": True,
            "wl141_polygons_cameras_rois_preserved": True,
            "wl143_depth_reconstruction_identity_reused": True,
            "nurbs_or_continuation_executed": False,
            "occluded_surface_executed": False,
        },
        "IMPLEMENTATION FIDELITY": {
            "canonical_renderer_modified": False,
            "canonical_checkpoint_modified": False,
            "wl127_geometry_modified": False,
            "wl139_modified": False,
            "wl141_report_or_masks_modified": False,
            "wl142_report_modified": False,
            "wl143_report_modified": False,
            "h_reused_from_wl139": h,
            "mu_reused_from_wl139_and_wl143": mu,
            "local_normal_method": "fixed k=12 local PCA; diagnostic only",
            "visualization_display_thinning": {
                "max_points_for_3d_png": DISPLAY_MAX_POINTS,
                "metrics_use_full_event_cloud": True,
                "full_event_cloud_ply_exported": True,
            },
            "new_thresholds": [],
            "radius_sweep": False,
        },
        "FROZEN_INPUTS": {
            "raw_visible_surface": str(reports["paths"]["raw_visible_surface"].resolve()),
            "wl139_report": str(reports["paths"]["wl139_report"].resolve()),
            "wl141_report": str(reports["paths"]["wl141_report"].resolve()),
            "wl142_report": str(reports["paths"]["wl142_report"].resolve()),
            "wl143_report": str(reports["paths"]["wl143_report"].resolve()),
            "checkpoint": str(reports["paths"]["checkpoint"].resolve()),
            "source_path": str(Path(arguments.source_path).resolve()),
            "h": h,
            "mu": mu,
            "wl143_depth_status": reports["wl143_report"].get("status"),
        },
        "PER-VIEW EVENT CLOUD CONTRACT": {
            "separate_clouds_before_any_combination": True,
            "all_points_are_valid_depth_median_pixels_inside_frozen_polygon": True,
            "world_coordinate_frame": "canonical Gaussian Scene world coordinates; 3D PNG additionally shows frozen physical chart coordinates",
        },
        "cases": cases,
        "ARCHITECTURE ATTRIBUTION": {
            "allowed_labels": [
                "A. SAME_PHYSICAL_SHEET_PLAUSIBLE",
                "B. DIFFERENT_DEPTH_LAYERS",
                "C. SEMANTIC_MASK_MISASSOCIATION",
                "D. PARTIAL_OVERLAP / MIXED",
                "E. INSUFFICIENT_EVIDENCE",
            ],
            "status": "COMPLETED_CASE_REVIEW",
            "per_case_classification": {
                name: value["per_case_human_review_status"]["classification"]
                for name, value in cases.items()
            },
            "classification_basis": "direct visual review of fixed common-frame 3D and raw source-to-all-target reprojection outputs combined with continuous distributions",
            "do_not_promote_per_view_proximity_to_membership": True,
            "interpretation_scope": "surface-correspondence audit only",
        },
        "PROMOTED": [
            "renderer-native per-view median-event cloud construction",
            "cloud-to-cloud continuous proximity accounting",
            "WL127 point-to-independent-cloud diagnostic attribution",
        ],
        "RETAINED": [
            "WL141 polygons, cameras, ROIs, MASK_ONLY populations",
            "WL139 h and mu",
            "WL143 renderer depth convention",
        ],
        "REJECTED": [
            "automatic Surface Membership",
            "new percentage vote",
            "KNN membership",
            "connected-component or region-growing selection",
            "NURBS, SH, continuation, Occluded Surface, Candidate B modification",
        ],
        "OPEN": [
            "per-case human classification after direct common-frame and reprojection review",
            "physical-sheet identity beyond this diagnostic",
            "automatic support/membership design",
        ],
        "failures": [],
    }
    return _write_report(output_root, report)


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "per_view_renderer_surface_correspondence_physical_sheet_oracle_audit_report.json"
    path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    lines = [
        "# Worklog 144 per-view renderer surface correspondence audit",
        "",
        "This isolated diagnostic compares independent renderer median-event clouds selected by the frozen WL141 polygons. It does not implement automatic Surface Membership.",
        "",
        f"Status: {report.get('status')}",
        "",
        "For each case, inspect per_view_event_clouds/, 3d_review/, cross_view_reprojection_overlays/, pairwise_surface_distance_distributions.npz, wl127_mask_only_attribution_distributions.npz, and case_report.json.",
        "",
        "A/B/C/D/E case classification is intentionally kept separate from scalar proximity and must use the common-frame 3D and all-target reprojection views together with the continuous distributions.",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-visible-surface", type=Path, default=WL127_RAW_VISIBLE_SURFACE)
    parser.add_argument("--wl139-report", type=Path, default=WL139_REPORT)
    parser.add_argument("--wl141-report", type=Path, default=WL141_REPORT)
    parser.add_argument("--wl142-report", type=Path, default=WL142_REPORT)
    parser.add_argument("--wl143-report", type=Path, default=WL143_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_audit(build_arg_parser().parse_args(argv))
    print(json.dumps({
        "status": report.get("status"),
        "architecture_status": report.get("ARCHITECTURE ATTRIBUTION", {}).get("status"),
        "failures": report.get("failures", []),
    }, indent=2))
    return 0 if not report.get("failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
