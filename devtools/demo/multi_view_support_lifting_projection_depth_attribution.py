"""Worklog 142: projection/depth attribution for WL141 support lifting.

This is an isolated, non-canonical diagnostic.  It replays the frozen WL141
``MASK_ONLY_SUPPORT`` row selection, audits the existing world-to-camera
projection with Gaussian primitive centers, and compares the same selected
rows with the canonical renderer's existing ``depth_median`` image.  The only
new support candidate is ``MASK_PLUS_DEPTH_SUPPORT``: the historical 2-of-3
mask rule plus a fixed, disclosed depth-consistency rule derived from WL139's
frozen ``mu``.

No automatic Surface Membership, appearance/SH signal, trimming, connected
component pruning, representative-driven selection, continuation, or
Occluded Surface is implemented here.  Representative replay is opt-in only
after a human review decision is supplied explicitly on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.meeting_occluded_surface_feasibility import (  # noqa: E402
    _grid_faces,
    _write_ply,
)
from devtools.demo.oracle_single_surface_support_appearance_evidence import (  # noqa: E402
    DISPLAY_POINT_ALPHA,
    DISPLAY_POINT_SIZE,
    ORACLE_GREEN,
    OracleSurfaceControl,
    FrozenOracleSupport,
    _draw_polygon_outline,
    _manual_controls,
    _mask_camera_lookup,
    _point_in_polygon,
    _sha256_rows,
    build_oracle_support,
    support_alignment_report,
    support_domain_diagnostic,
)
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402
    GRAPH_DEGREE_U,
    GRAPH_DEGREE_V,
    GRAPH_RESOLUTION_U,
    GRAPH_RESOLUTION_V,
    GRAPH_SMOOTHNESS_LAMBDA,
    GRAPH_TIKHONOV_LAMBDA,
    SAMPLE_COUNT_U,
    SAMPLE_COUNT_V,
    RepresentativeCaseConfig,
    _configure_axis,
    _normalised_axis,
    _plot_points,
    _plot_surface,
    _save_3d,
    _summary,
    audit_raw_graphness,
    audit_representative_topology,
    fit_physical_chart_surface,
    graphness_report,
    representative_contract,
)
from devtools.demo.real_gaussian_scene_surface_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    _camera_center,
    _draw_projected_points,
    _load_canonical_scene,
    _load_xyz_ply,
    _render_to_pil,
    _sha256_file,
    project_world_points,
)


OUTPUT_ROOT = REPO_ROOT / "output" / "142_multi_view_support_lifting_projection_depth_attribution"
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

# This camera set is used only for the geometry-independent projection control.
# It is not a semantic mask set and is never used to choose support.
PROJECTION_CONTROL_CAMERA_NAMES = ("DSC07960.JPG", "DSC08003.JPG", "DSC08111.JPG")

# The two-of-three cardinality is inherited from WL141.  ``mu`` is the frozen
# WL139 scale quantity and is used as a disclosed, non-evaluation-tuned depth
# tolerance.  No per-case depth tolerance exists.
MASK_VOTE_MIN = 2
DEPTH_VALID_MIN = 2
DEPTH_CONSISTENT_MIN = 2

DEPTH_CONSISTENT_RGB = (20, 198, 82)
BEHIND_FRONTIER_RGB = (235, 54, 42)
IN_FRONT_RGB = (42, 105, 232)
NO_DEPTH_RGB = (144, 145, 151)
MASK_ONLY_RGB = (245, 151, 25)
REMOVED_DEPTH_RGB = (228, 58, 45)
PREDICTED_CYAN = (0.0, 0.70, 0.84)


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


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(_as_numpy(value)).tobytes()).hexdigest()


def _camera_matrix_report(camera: Any) -> dict[str, Any]:
    view = _as_numpy(camera.world_view_transform).astype(np.float64)
    projection = _as_numpy(camera.full_proj_transform).astype(np.float64)
    inverse_view = np.linalg.inv(view)
    center_from_view = (np.asarray([0.0, 0.0, 0.0, 1.0]) @ inverse_view)[:3]
    declared_center = _camera_center(camera)
    return {
        "camera_name": str(camera.image_name),
        "image_size": [int(camera.image_width), int(camera.image_height)],
        "view_matrix_shape": list(view.shape),
        "projection_matrix_shape": list(projection.shape),
        "view_matrix_sha256": _sha256_array(view),
        "projection_matrix_sha256": _sha256_array(projection),
        "view_linear_determinant": float(np.linalg.det(view[:3, :3])),
        "declared_camera_center": declared_center,
        "inverse_view_camera_center": center_from_view,
        "camera_center_l2_gap": float(np.linalg.norm(declared_center - center_from_view)),
        "view_convention": "homogeneous world row-vector @ world_view_transform",
    }


def projection_contract_audit(cameras: Iterable[Any]) -> dict[str, Any]:
    """Audit the exact projection convention used by WL140 overlays."""

    camera_reports = []
    failures: list[str] = []
    for camera in cameras:
        try:
            report = _camera_matrix_report(camera)
        except Exception as error:
            failures.append(f"{camera.image_name}: matrix audit failed: {error!r}")
            continue
        if report["view_matrix_shape"] != [4, 4] or report["projection_matrix_shape"] != [4, 4]:
            failures.append(f"{camera.image_name}: non-4x4 camera matrix")
        if not np.isfinite(report["camera_center_l2_gap"]) or report["camera_center_l2_gap"] > 1.0e-4:
            failures.append(f"{camera.image_name}: camera center mismatch")
        camera_reports.append(report)
    return {
        "status": "PROJECTION_CONTRACT_PASS" if not failures and len(camera_reports) else "PROJECTION_CONTRACT_FAIL",
        "camera_count_audited": int(len(camera_reports)),
        "failures": failures,
        "world_coordinate_convention": "WL127 XYZ and Gaussian checkpoint means are shared world coordinates",
        "extrinsic_convention": "row-vector homogeneous world point multiplied by camera.world_view_transform",
        "handedness_audit": "reported by view linear determinant; no axis reinterpretation performed",
        "camera_center_audit": "inverse(world_view_transform) row [0,0,0,1] compared with camera.camera_center",
        "projection_convention": "world_h @ full_proj_transform; ndc x=(clip_x/w), ndc y=(clip_y/w)",
        "image_scaling": "pixel_x=(ndc_x+1)*0.5*(width-1); pixel_y=(1-ndc_y)*0.5*(height-1)",
        "image_orientation": "x rightward, y downward in image array; no extra flip",
        "pixel_center_convention": "nearest diagnostic sample uses floor(pixel+0.5); projection remains continuous",
        "depth_sign_convention": "camera-space view[:,2] > 0 is in front / renderable",
        "camera_reports": camera_reports,
    }


def known_geometry_projection_control(
    model: Any,
    cameras: Iterable[Any],
    scene_info: dict[str, Any],
    output_root: Path,
    render_cache: dict[str, Any],
) -> dict[str, Any]:
    """Project checkpoint centers and verify they land on renderer support."""

    centers = _as_numpy(model.get_xyz).astype(np.float64).reshape(-1, 3)
    lookup = _mask_camera_lookup(cameras)
    selected_names = [name for name in PROJECTION_CONTROL_CAMERA_NAMES if name in lookup]
    if len(selected_names) < 2:
        selected_names = sorted(lookup)[:3]
    reports: list[dict[str, Any]] = []
    for name in selected_names:
        camera = lookup[name]
        package = scene_info["rasterizer"].render(camera, model)
        render_cache[name] = _render_to_pil(package)
        projection = project_world_points(centers, camera)
        finite = np.isfinite(projection["x"]) & np.isfinite(projection["y"]) & np.isfinite(projection["depth"])
        valid = projection["valid"] & finite
        pixel_x = np.floor(projection["x"] + 0.5).astype(np.int64)
        pixel_y = np.floor(projection["y"] + 0.5).astype(np.int64)
        height = int(camera.image_height)
        width = int(camera.image_width)
        in_map = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
        alpha = _as_numpy(package["rend_alpha"]).astype(np.float64)
        if alpha.ndim == 3:
            alpha = alpha[0]
        alpha_at_center = np.full(len(centers), np.nan, dtype=np.float64)
        sample_valid = valid & in_map
        alpha_at_center[sample_valid] = alpha[pixel_y[sample_valid], pixel_x[sample_valid]]
        visibility_mask = _as_numpy(package["visibility_mask"]).reshape(-1).astype(bool)
        visible_center_valid = valid & visibility_mask
        alpha_support = visible_center_valid & np.isfinite(alpha_at_center) & (alpha_at_center > (1.0 / 255.0))
        overlay_points = centers if len(centers) <= 18000 else centers[np.linspace(0, len(centers) - 1, 18000, dtype=np.int64)]
        overlay = _draw_projected_points(
            render_cache[name],
            overlay_points,
            camera,
            (247, 136, 23),
            radius=1.8,
            alpha=0.90,
        )
        camera_root = output_root / "projection_control" / name
        camera_root.mkdir(parents=True, exist_ok=True)
        image_path = camera_root / "gaussian_centers_projected_over_scene.png"
        overlay.save(image_path)
        reports.append({
            "camera_name": name,
            "gaussian_center_count": int(len(centers)),
            "finite_projection_fraction": float(np.mean(finite)),
            "valid_in_frame_positive_depth_fraction": float(np.mean(valid)),
            "valid_positive_depth_count": int(np.sum(valid)),
            "camera_depth_min": float(np.min(projection["depth"][finite])) if np.any(finite) else None,
            "camera_depth_max": float(np.max(projection["depth"][finite])) if np.any(finite) else None,
            "renderer_visible_gaussian_count": int(np.sum(visibility_mask)),
            "visible_center_projection_count": int(np.sum(visible_center_valid)),
            "visible_center_alpha_support_fraction": float(np.mean(alpha_support[visible_center_valid])) if np.any(visible_center_valid) else 0.0,
            "center_alpha_threshold": 1.0 / 255.0,
            "image_size": [int(camera.image_width), int(camera.image_height)],
            "overlay": str(image_path),
        })
    passed = bool(reports) and all(
        item["finite_projection_fraction"] == 1.0
        and item["valid_positive_depth_count"] > 0
        and item["visible_center_projection_count"] > 0
        and item["visible_center_alpha_support_fraction"] >= 0.50
        for item in reports
    )
    return {
        "status": "PROJECTION_CONTRACT_PASS" if passed else "PROJECTION_CONTRACT_FAIL",
        "control_geometry": "frozen Gaussian primitive centers from checkpoint; no semantic ROI or mask",
        "renderer_alpha_alignment_control": "projected centers sampled against canonical rend_alpha at nearest pixel",
        "renderer_alpha_alignment_pass_threshold": 0.50,
        "camera_names": selected_names,
        "reports": reports,
    }

def _depth_map_from_package(package: dict[str, Any]) -> np.ndarray:
    depth = _as_numpy(package["depth_median"]).astype(np.float64)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"expected 2D depth_median map, got {depth.shape}")
    return depth


def _depth_relation_for_camera(
    points: np.ndarray,
    camera: Any,
    mask: Any,
    depth_median: np.ndarray,
    mu: float,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    projection = project_world_points(points, camera)
    xy = np.column_stack([projection["x"], projection["y"]])
    mask_match = projection["valid"] & _point_in_polygon(xy, mask.polygon_for(camera))
    height, width = depth_median.shape
    pixel_x = np.floor(projection["x"] + 0.5).astype(np.int64)
    pixel_y = np.floor(projection["y"] + 0.5).astype(np.int64)
    in_map = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    sample_valid = projection["valid"] & in_map
    sampled = np.full(len(points), np.nan, dtype=np.float64)
    sampled[sample_valid] = depth_median[pixel_y[sample_valid], pixel_x[sample_valid]]
    renderer_depth_valid = sample_valid & np.isfinite(sampled) & (np.abs(sampled) > 1.0e-8)
    residual = np.full(len(points), np.nan, dtype=np.float64)
    residual[renderer_depth_valid] = projection["depth"][renderer_depth_valid] - sampled[renderer_depth_valid]
    depth_consistent = renderer_depth_valid & (np.abs(residual) <= float(mu))
    behind = renderer_depth_valid & (residual > float(mu))
    in_front = renderer_depth_valid & (residual < -float(mu))
    no_depth = ~renderer_depth_valid
    return {
        "mask_match": mask_match,
        "renderer_depth_valid": renderer_depth_valid,
        "depth_consistent": depth_consistent,
        "behind_visible_frontier": behind,
        "in_front_of_visible_frontier": in_front,
        "no_relevant_renderer_depth": no_depth,
        "camera_depth": projection["depth"],
        "renderer_median_depth": sampled,
        "depth_residual": residual,
        "normalized_depth_residual_over_mu": np.abs(residual) / max(float(mu), 1.0e-12),
    }


def _sum_relation(relations: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.sum(np.stack([relation[key].astype(np.int16) for relation in relations], axis=0), axis=0).astype(np.int16)


def depth_layer_accounting(
    candidate_points: np.ndarray,
    control: OracleSurfaceControl,
    camera_lookup: dict[str, Any],
    depth_maps: dict[str, np.ndarray],
    mu: float,
    h: float,
) -> dict[str, Any]:
    relations: list[dict[str, np.ndarray]] = []
    per_camera: list[dict[str, Any]] = []
    for mask in control.masks:
        relation = _depth_relation_for_camera(candidate_points, camera_lookup[mask.camera_name], mask, depth_maps[mask.camera_name], mu)
        relations.append(relation)
        residual = relation["depth_residual"][relation["renderer_depth_valid"]]
        mask_residual = relation["depth_residual"][relation["mask_match"] & relation["renderer_depth_valid"]]
        per_camera.append({
            "camera_name": mask.camera_name,
            "mask_matches": int(np.sum(relation["mask_match"])),
            "valid_depth_comparisons": int(np.sum(relation["renderer_depth_valid"])),
            "depth_consistent": int(np.sum(relation["depth_consistent"])),
            "behind_visible_frontier": int(np.sum(relation["behind_visible_frontier"])),
            "in_front_of_visible_frontier": int(np.sum(relation["in_front_of_visible_frontier"])),
            "no_relevant_renderer_depth": int(np.sum(relation["no_relevant_renderer_depth"])),
            "depth_residual_over_h": _summary(residual, h=h),
            "mask_matching_depth_residual_over_h": _summary(mask_residual, h=h),
            "normalized_abs_depth_residual_over_mu": _summary(relation["normalized_depth_residual_over_mu"], h=None),
            "mask_matching_normalized_abs_depth_residual_over_mu": _summary(relation["normalized_depth_residual_over_mu"][relation["mask_match"]], h=None),
        })
    mask_matches = _sum_relation(relations, "mask_match")
    valid_depth = _sum_relation(relations, "renderer_depth_valid")
    depth_consistent = _sum_relation(relations, "depth_consistent")
    behind = _sum_relation(relations, "behind_visible_frontier")
    in_front = _sum_relation(relations, "in_front_of_visible_frontier")
    no_depth = _sum_relation(relations, "no_relevant_renderer_depth")
    mask_valid = np.sum(np.stack([(r["mask_match"] & r["renderer_depth_valid"]).astype(np.int16) for r in relations], axis=0), axis=0).astype(np.int16)
    mask_consistent = np.sum(np.stack([(r["mask_match"] & r["depth_consistent"]).astype(np.int16) for r in relations], axis=0), axis=0).astype(np.int16)
    mask_behind = np.sum(np.stack([(r["mask_match"] & r["behind_visible_frontier"]).astype(np.int16) for r in relations], axis=0), axis=0).astype(np.int16)
    mask_in_front = np.sum(np.stack([(r["mask_match"] & r["in_front_of_visible_frontier"]).astype(np.int16) for r in relations], axis=0), axis=0).astype(np.int16)
    mask_only = mask_matches >= MASK_VOTE_MIN
    mask_plus_depth = (
        mask_only
        & (mask_valid >= DEPTH_VALID_MIN)
        & (mask_consistent >= DEPTH_CONSISTENT_MIN)
        & (mask_behind == 0)
        & (mask_in_front == 0)
    )
    all_residuals = np.concatenate([
        relation["depth_residual"][relation["renderer_depth_valid"]]
        for relation in relations
        if np.any(relation["renderer_depth_valid"])
    ]) if any(np.any(r["renderer_depth_valid"]) for r in relations) else np.empty((0,), dtype=np.float64)
    selected_residuals = np.concatenate([
        relation["depth_residual"][mask_only & relation["mask_match"] & relation["renderer_depth_valid"]]
        for relation in relations
        if np.any(mask_only & relation["mask_match"] & relation["renderer_depth_valid"])
    ]) if any(np.any(mask_only & r["mask_match"] & r["renderer_depth_valid"]) for r in relations) else np.empty((0,), dtype=np.float64)
    return {
        "rule": {
            "mask_support": f"at least {MASK_VOTE_MIN} of {len(control.masks)} fixed masks",
            "depth_support": f"at least {DEPTH_VALID_MIN} mask-matching valid renderer depth comparisons and at least {DEPTH_CONSISTENT_MIN} consistent comparisons",
            "contradiction_rule": "reject any mask-matching behind-visible-frontier or in-front contradiction",
            "depth_tolerance_world": float(mu),
            "depth_tolerance_derivation": "frozen WL139 mu; not selected from WL142 labels or output quality",
            "pixel_sample_rule": "nearest image sample floor(projected_pixel + 0.5)",
        },
        "candidate_count": int(len(candidate_points)),
        "mask_only_count": int(np.sum(mask_only)),
        "mask_plus_depth_count": int(np.sum(mask_plus_depth)),
        "removed_by_depth_count": int(np.sum(mask_only & ~mask_plus_depth)),
        "removed_by_depth_fraction_of_mask_only": float(np.mean(mask_only & ~mask_plus_depth) / max(np.mean(mask_only), 1.0e-12)),
        "mask_match_count_summary": _summary(mask_matches),
        "valid_depth_comparison_count_summary": _summary(valid_depth),
        "depth_consistent_view_count_summary": _summary(depth_consistent),
        "behind_visible_frontier_count_summary": _summary(behind),
        "in_front_of_visible_frontier_count_summary": _summary(in_front),
        "no_relevant_renderer_depth_count_summary": _summary(no_depth),
        "mask_matching_valid_depth_count_summary": _summary(mask_valid),
        "mask_matching_depth_consistent_count_summary": _summary(mask_consistent),
        "mask_matching_behind_count_summary": _summary(mask_behind),
        "mask_matching_in_front_count_summary": _summary(mask_in_front),
        "all_point_camera_depth_residual": _summary(all_residuals, h=h),
        "mask_only_point_camera_depth_residual": _summary(selected_residuals, h=h),
        "per_camera": per_camera,
        "arrays": {
            "mask_matches": mask_matches,
            "valid_depth": valid_depth,
            "depth_consistent": depth_consistent,
            "behind": behind,
            "in_front": in_front,
            "no_depth": no_depth,
            "mask_valid": mask_valid,
            "mask_consistent": mask_consistent,
            "mask_behind": mask_behind,
            "mask_in_front": mask_in_front,
            "mask_only": mask_only,
            "mask_plus_depth": mask_plus_depth,
            "relations": relations,
        },
    }


def _component_accounting(points: np.ndarray, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return {"point_count": 0, "connected_component_count": 0, "largest_component_fraction": 0.0, "spatial_extent": None}
    pairs = cKDTree(points).query_pairs(float(h), output_type="ndarray")
    parent = np.arange(len(points), dtype=np.int64)
    size = np.ones(len(points), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
        root_first = find(int(first))
        root_second = find(int(second))
        if root_first == root_second:
            continue
        if size[root_first] < size[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
    roots, counts = np.unique([find(index) for index in range(len(points))], return_counts=True)
    _ = roots
    return {
        "point_count": int(len(points)),
        "connected_component_count_diagnostic_only": int(len(counts)),
        "largest_component_fraction_diagnostic_only": float(np.max(counts) / len(points)),
        "spatial_extent": np.ptp(points, axis=0),
        "component_radius_world": float(h),
        "support_not_modified_by_components": True,
    }


def _chart_occupancy(points: np.ndarray, config: RepresentativeCaseConfig) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return {"occupied_bins": 0, "occupancy_fraction": 0.0, "u_extent": None, "v_extent": None}
    coordinates = points @ np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1)
    bins = np.floor(np.column_stack([
        (coordinates[:, 0] - config.u_bounds[0]) / max(config.u_bounds[1] - config.u_bounds[0], 1.0e-12) * GRAPH_RESOLUTION_U,
        (coordinates[:, 1] - config.v_bounds[0]) / max(config.v_bounds[1] - config.v_bounds[0], 1.0e-12) * GRAPH_RESOLUTION_V,
    ])).astype(np.int64)
    bins = np.clip(bins, 0, np.asarray([GRAPH_RESOLUTION_U - 1, GRAPH_RESOLUTION_V - 1]))
    occupied = len(np.unique(bins, axis=0))
    return {
        "occupied_chart_bins_diagnostic_only": int(occupied),
        "chart_bin_count": int(GRAPH_RESOLUTION_U * GRAPH_RESOLUTION_V),
        "occupancy_fraction_diagnostic_only": float(occupied / (GRAPH_RESOLUTION_U * GRAPH_RESOLUTION_V)),
        "u_extent": [float(np.min(coordinates[:, 0])), float(np.max(coordinates[:, 0]))],
        "v_extent": [float(np.min(coordinates[:, 1])), float(np.max(coordinates[:, 1]))],
    }


def _write_depth_geometry(case_root: Path, raw_points: np.ndarray, support: FrozenOracleSupport, accounting: dict[str, Any], h: float) -> dict[str, str]:
    root = case_root / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    candidate_points = np.asarray(raw_points, dtype=np.float64)[support.candidate_row_ids]
    mask_only = np.asarray(accounting["arrays"]["mask_only"], dtype=bool)
    mask_plus_depth = np.asarray(accounting["arrays"]["mask_plus_depth"], dtype=bool)
    paths = {
        "candidate": root / "candidate_spatial_population.ply",
        "mask_only": root / "mask_only_support.ply",
        "mask_plus_depth": root / "mask_plus_depth_support.ply",
        "removed_by_depth": root / "removed_by_depth_inconsistency.ply",
        "accounting": root / "depth_layer_accounting.npz",
    }
    _write_ply(paths["candidate"], candidate_points, color=(134, 137, 145))
    _write_ply(paths["mask_only"], candidate_points[mask_only], color=(245, 151, 25))
    _write_ply(paths["mask_plus_depth"], candidate_points[mask_plus_depth], color=(20, 198, 82))
    _write_ply(paths["removed_by_depth"], candidate_points[mask_only & ~mask_plus_depth], color=(228, 58, 45))
    arrays = accounting["arrays"]
    relation_arrays = arrays["relations"]
    np.savez_compressed(
        paths["accounting"],
        candidate_row_ids=support.candidate_row_ids,
        mask_only_row_ids=support.candidate_row_ids[mask_only],
        mask_plus_depth_row_ids=support.candidate_row_ids[mask_plus_depth],
        mask_matches=arrays["mask_matches"],
        valid_depth=arrays["valid_depth"],
        depth_consistent=arrays["depth_consistent"],
        behind=arrays["behind"],
        in_front=arrays["in_front"],
        no_depth=arrays["no_depth"],
        mask_valid=arrays["mask_valid"],
        mask_consistent=arrays["mask_consistent"],
        mask_behind=arrays["mask_behind"],
        mask_in_front=arrays["mask_in_front"],
        depth_residuals=np.stack([relation["depth_residual"] for relation in relation_arrays], axis=1),
        camera_depths=np.stack([relation["camera_depth"] for relation in relation_arrays], axis=1),
        renderer_median_depths=np.stack([relation["renderer_median_depth"] for relation in relation_arrays], axis=1),
    )
    _ = h
    return {key: str(path) for key, path in paths.items()}


def _relation_color_groups(accounting: dict[str, Any], support: FrozenOracleSupport, camera_index: int) -> dict[str, np.ndarray]:
    mask_only = np.asarray(accounting["arrays"]["mask_only"], dtype=bool)
    relation = accounting["arrays"]["relations"][camera_index]
    return {
        "depth_consistent": mask_only & relation["depth_consistent"],
        "behind_visible_frontier": mask_only & relation["behind_visible_frontier"],
        "in_front_of_visible_frontier": mask_only & relation["in_front_of_visible_frontier"],
        "no_relevant_renderer_depth": mask_only & relation["no_relevant_renderer_depth"],
    }


def _write_camera_depth_outputs(
    case_root: Path,
    control: OracleSurfaceControl,
    raw_points: np.ndarray,
    support: FrozenOracleSupport,
    accounting: dict[str, Any],
    camera: Any,
    camera_index: int,
    scene_image: Any,
) -> dict[str, str]:
    root = case_root / "camera_overlays" / str(camera.image_name)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "A_gaussian_scene_only": root / "A_gaussian_scene_only.png",
        "B_gaussian_plus_mask_only_support": root / "B_gaussian_plus_mask_only_support.png",
        "C_mask_only_support_depth_relation": root / "C_mask_only_support_depth_relation.png",
        "D_mask_polygon_plus_depth_relation": root / "D_mask_polygon_plus_depth_relation.png",
        "E_removed_by_depth": root / "E_removed_by_depth_inconsistency.png",
    }
    scene_image.convert("RGB").save(paths["A_gaussian_scene_only"])
    candidate_points = raw_points[support.candidate_row_ids]
    mask_only = np.asarray(accounting["arrays"]["mask_only"], dtype=bool)
    mask_only_view = _draw_projected_points(scene_image, candidate_points[mask_only], camera, MASK_ONLY_RGB, radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    mask_only_view.save(paths["B_gaussian_plus_mask_only_support"])
    relation_colors = {
        "depth_consistent": DEPTH_CONSISTENT_RGB,
        "behind_visible_frontier": BEHIND_FRONTIER_RGB,
        "in_front_of_visible_frontier": IN_FRONT_RGB,
        "no_relevant_renderer_depth": NO_DEPTH_RGB,
    }
    groups = _relation_color_groups(accounting, support, camera_index)
    relation_view = scene_image.copy().convert("RGB")
    for name in ("no_relevant_renderer_depth", "in_front_of_visible_frontier", "behind_visible_frontier", "depth_consistent"):
        relation_view = _draw_projected_points(relation_view, candidate_points[groups[name]], camera, relation_colors[name], radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    relation_view.save(paths["C_mask_only_support_depth_relation"])
    mask = next(
        item for item in control.masks if item.camera_name == str(camera.image_name)
    )
    outlined = _draw_polygon_outline(relation_view, mask, camera)
    outlined.save(paths["D_mask_polygon_plus_depth_relation"])
    removed = mask_only & ~np.asarray(accounting["arrays"]["mask_plus_depth"], dtype=bool)
    removed_view = _draw_projected_points(scene_image, candidate_points[removed], camera, REMOVED_DEPTH_RGB, radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    removed_view.save(paths["E_removed_by_depth"])
    return {key: str(path) for key, path in paths.items()}


def _write_3d_review(
    case_root: Path,
    control: OracleSurfaceControl,
    raw_points: np.ndarray,
    support: FrozenOracleSupport,
    accounting: dict[str, Any],
    h: float,
    oracle_representative: Any | None,
) -> dict[str, str]:
    root = case_root / "3d_review"
    root.mkdir(parents=True, exist_ok=True)
    candidate_points = raw_points[support.candidate_row_ids]
    mask_only = np.asarray(accounting["arrays"]["mask_only"], dtype=bool)
    mask_plus_depth = np.asarray(accounting["arrays"]["mask_plus_depth"], dtype=bool)
    removed = mask_only & ~mask_plus_depth
    paths = {
        "all_candidate": root / "all_candidate_points.png",
        "mask_only": root / "mask_only_support.png",
        "mask_plus_depth": root / "mask_plus_depth_support.png",
        "removed_by_depth": root / "removed_by_depth.png",
        "mask_only_vs_depth": root / "mask_only_vs_mask_plus_depth.png",
        "representative": root / "mask_plus_depth_representative_NOT_EXPORTED.txt",
    }
    parts = [candidate_points, candidate_points[mask_only], candidate_points[mask_plus_depth]]
    if oracle_representative is not None:
        parts.append(oracle_representative.sampled_points)
    limits = np.concatenate(parts, axis=0)
    config = control.config
    _save_3d(paths["all_candidate"], "All candidate spatial population", config, limits, lambda axis: _plot_points(axis, candidate_points, config, (0.42, 0.43, 0.47), label="candidate", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE))
    _save_3d(paths["mask_only"], "WL141 MASK_ONLY_SUPPORT", config, limits, lambda axis: _plot_points(axis, candidate_points[mask_only], config, (0.96, 0.59, 0.10), label="MASK_ONLY_SUPPORT", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE))
    _save_3d(paths["mask_plus_depth"], "MASK_PLUS_DEPTH_SUPPORT", config, limits, lambda axis: _plot_points(axis, candidate_points[mask_plus_depth], config, (0.08, 0.78, 0.32), label="MASK_PLUS_DEPTH_SUPPORT", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE))
    _save_3d(paths["removed_by_depth"], "Points removed by depth inconsistency", config, limits, lambda axis: _plot_points(axis, candidate_points[removed], config, (0.90, 0.12, 0.10), label="removed by depth", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE))
    _save_3d(
        paths["mask_only_vs_depth"],
        "MASK_ONLY vs MASK_PLUS_DEPTH local 3D review",
        config,
        limits,
        lambda axis: (
            _plot_points(axis, candidate_points[mask_only], config, (0.96, 0.59, 0.10), label="MASK_ONLY_SUPPORT", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE),
            _plot_points(axis, candidate_points[mask_plus_depth], config, (0.08, 0.78, 0.32), label="MASK_PLUS_DEPTH_SUPPORT", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE),
            _plot_points(axis, candidate_points[removed], config, (0.90, 0.12, 0.10), label="removed by depth", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE),
        ),
    )
    if oracle_representative is not None:
        grid = oracle_representative.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
        _save_3d(paths["representative"], "MASK_PLUS_DEPTH + frozen WL139 representative", config, limits, lambda axis: (_plot_points(axis, candidate_points[mask_plus_depth], config, (0.08, 0.78, 0.32), label="MASK_PLUS_DEPTH_SUPPORT", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE), _plot_surface(axis, grid, config, PREDICTED_CYAN, label="WL139 representative", alpha=0.66)))
    else:
        legacy_marker = root / "mask_plus_depth_representative.png"
        if legacy_marker.exists():
            legacy_marker.unlink()
        paths["representative"].write_text("not exported: explicit human qualitative pass was not supplied\n", encoding="utf-8")
    _ = h
    return {key: str(path) for key, path in paths.items()}


def _reproduce_wl141_support(control_name: str, support: FrozenOracleSupport) -> dict[str, Any]:
    if not WL141_REPORT.exists():
        return {"status": "UNAVAILABLE_WL141_REPORT", "exact": False}
    historical = json.loads(WL141_REPORT.read_text(encoding="utf-8"))
    historical_case = historical.get("cases", {}).get(control_name)
    if historical_case is None:
        return {"status": "UNAVAILABLE_CASE_IN_WL141_REPORT", "exact": False}
    historical_alignment = historical_case.get("support_alignment", {})
    expected_count = int(historical_alignment.get("oracle_row_count", -1))
    expected_hash = str(historical_alignment.get("oracle_row_ids_sha256", ""))
    replayed_hash = hashlib.sha256(support.oracle_row_ids.tobytes()).hexdigest()
    exact = expected_count == len(support.oracle_row_ids) and expected_hash == replayed_hash
    return {
        "status": "EXACT_REPRODUCTION_BY_ROW_ID_HASH" if exact else "REPRODUCTION_MISMATCH",
        "exact": exact,
        "historical_oracle_row_count": expected_count,
        "replayed_oracle_row_count": int(len(support.oracle_row_ids)),
        "historical_oracle_row_ids_sha256": expected_hash,
        "replayed_oracle_row_ids_sha256": replayed_hash,
        "historical_row_id_list_available": False,
        "selection_name": "MASK_ONLY_BASELINE",
    }

def _replay_representative(
    case_root: Path,
    points: np.ndarray,
    config: RepresentativeCaseConfig,
    h: float,
    max_fit_points: int,
    device: str,
) -> dict[str, Any] | None:
    audit = audit_raw_graphness(points, config, h)
    if audit.status != "PASS_GRAPH_LIKE":
        return {"status": "NOT_REPLAYED_GRAPHNESS_FAIL", "graphness": graphness_report(audit, h)}
    representative = fit_physical_chart_surface(points, config, role="full_evaluation_only", max_fit_points=max_fit_points, device_name=device)
    topology = audit_representative_topology(representative.surface, config, domain_u=representative.domain_u, domain_v=representative.domain_v, h=h)
    contract = representative_contract(points, representative.sampled_points, representative.sampled_normals, representative.fitting_residuals, topology, h)
    support_domain = support_domain_diagnostic(points, representative, config)
    geometry_root = case_root / "geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)
    path = geometry_root / "mask_plus_depth_wl139_representative.ply"
    _write_ply(path, representative.sampled_points, faces=_grid_faces(GRAPH_RESOLUTION_U * 12, GRAPH_RESOLUTION_V * 10), color=(0, 188, 224))
    return {
        "status": "REPLAYED_AFTER_EXPLICIT_HUMAN_PASS",
        "graphness": graphness_report(audit, h),
        "topology": topology,
        "contract": contract,
        "support_domain": support_domain,
        "geometry": str(path),
        "representative": representative,
    }


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = Path(arguments.raw_visible_surface)
    checkpoint = Path(arguments.checkpoint)
    wl139_report_path = Path(arguments.wl139_report)
    if not raw_path.exists() or not checkpoint.exists() or not wl139_report_path.exists():
        missing = [str(path) for path in (raw_path, checkpoint, wl139_report_path) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))
    wl139_report = json.loads(wl139_report_path.read_text(encoding="utf-8"))
    h = float(wl139_report["inputs"]["h"])
    mu = float(wl139_report["inputs"]["mu"])
    raw_points = _load_xyz_ply(raw_path)
    controls = _manual_controls()
    control_names = {control.name for control in controls}
    qualitative_pass = set(arguments.qualitative_pass or [])
    unknown_pass = qualitative_pass - control_names
    if unknown_pass:
        raise ValueError(f"unknown --qualitative-pass cases: {sorted(unknown_pass)}")

    model, payload, cameras, scene_info = _load_canonical_scene(
        checkpoint,
        Path(arguments.source_path),
        arguments.device,
        arguments.images,
        arguments.sparse_dir,
        int(arguments.resolution),
        int(arguments.llffhold),
    )
    camera_lookup = _mask_camera_lookup(cameras)
    projection_audit = projection_contract_audit(cameras)
    render_cache: dict[str, Any] = {}
    known_geometry = known_geometry_projection_control(model, cameras, scene_info, output_root, render_cache)
    projection_pass = projection_audit["status"] == "PROJECTION_CONTRACT_PASS" and known_geometry["status"] == "PROJECTION_CONTRACT_PASS"

    supports: dict[str, FrozenOracleSupport] = {}
    for control in controls:
        selected_cameras = [camera_lookup[mask.camera_name] for mask in control.masks]
        supports[control.name] = build_oracle_support(raw_points, control, selected_cameras)

    manifest = {
        "batch": "Worklog 142 multi-view support lifting projection/depth/physical-sheet attribution",
        "historical_support_name": "MASK_ONLY_BASELINE",
        "frozen_wl141_support_replayed_before_depth_filter": True,
        "raw_visible_surface": str(raw_path.resolve()),
        "raw_visible_surface_sha256": _sha256_file(raw_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": payload.get("iteration"),
        "h": h,
        "mu": mu,
        "camera_meta": scene_info["camera_meta"],
        "selection_inputs": ["WL141 candidate crop", "WL141 frozen camera IDs", "WL141 frozen polygons", "unchanged camera projection"],
        "selection_excludes": ["depth", "graphness", "representative", "SH/color", "components", "normals", "continuation", "Candidate B"],
        "depth_filter_is_the_only_new_support_signal": True,
        "depth_rule": {
            "mask_votes": MASK_VOTE_MIN,
            "valid_depth_min": DEPTH_VALID_MIN,
            "consistent_depth_min": DEPTH_CONSISTENT_MIN,
            "tolerance_world": mu,
            "tolerance_derivation": "frozen WL139 mu",
        },
        "controls": [],
        "leakage_disclosure": "WL127 raw XYZ is the observed candidate population; no withheld target exists in this attribution. Depth maps are read-only renderer evidence and do not define semantic ground truth.",
    }
    for control in controls:
        support = supports[control.name]
        manifest["controls"].append({
            **control.as_json(),
            "mask_only_support_alignment": support_alignment_report(support, raw_points),
            "wl141_reproduction": _reproduce_wl141_support(control.name, support),
        })
    manifest_path = output_root / "frozen_wl141_mask_only_replay_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    if not projection_pass:
        report = {
            "batch": manifest["batch"],
            "status": "PROJECTION_CONTRACT_FAIL_STOPPED",
            "INTENT ALIGNMENT": {"automatic_surface_membership_implemented": False, "depth_support_executed": False, "appearance_executed": False, "continuation_executed": False},
            "IMPLEMENTATION FIDELITY": {"canonical_renderer_modified": False, "wl127_modified": False, "wl139_modified": False, "wl141_masks_modified": False},
            "PROJECTION CONTRACT AUDIT": projection_audit,
            "KNOWN-GEOMETRY PROJECTION CONTROL": known_geometry,
            "MASK_ONLY SUPPORT REPRODUCTION": {control.name: _reproduce_wl141_support(control.name, supports[control.name]) for control in controls},
            "STOP_REASON": "Projection contract failed; no depth support or new architecture was constructed.",
            "cases": {},
            "failures": [],
        }
        report_path = output_root / "multi_view_support_lifting_projection_depth_attribution_report.json"
        report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
        (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
        return report

    depth_maps: dict[str, np.ndarray] = {}
    unique_camera_names = sorted({
        mask.camera_name
        for control in controls
        for mask in control.masks
    })
    for camera_name in unique_camera_names:
        camera = camera_lookup[camera_name]
        if camera_name not in render_cache:
            render_cache[camera_name] = _render_to_pil(scene_info["rasterizer"].render(camera, model))
        package = scene_info["rasterizer"].render(camera, model)
        depth_maps[camera_name] = _depth_map_from_package(package)

    case_reports: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for control in controls:
        case_root = output_root / control.name
        case_root.mkdir(parents=True, exist_ok=True)
        support = supports[control.name]
        candidate_points = raw_points[support.candidate_row_ids]
        try:
            accounting = depth_layer_accounting(candidate_points, control, camera_lookup, depth_maps, mu, h)
            geometry = _write_depth_geometry(case_root, raw_points, support, accounting, h)
            representative_result = None
            if control.name in qualitative_pass:
                representative_result = _replay_representative(case_root, raw_points[support.candidate_row_ids][np.asarray(accounting["arrays"]["mask_plus_depth"], dtype=bool)], control.config, h, int(arguments.max_fit_points), arguments.device)
            three_d_outputs = _write_3d_review(
                case_root,
                control,
                raw_points,
                support,
                accounting,
                h,
                representative_result.get("representative") if representative_result is not None and representative_result.get("status") == "REPLAYED_AFTER_EXPLICIT_HUMAN_PASS" else None,
            )
            camera_outputs: dict[str, Any] = {}
            for index, mask in enumerate(control.masks):
                camera = camera_lookup[mask.camera_name]
                camera_outputs[mask.camera_name] = _write_camera_depth_outputs(
                    case_root,
                    control,
                    raw_points,
                    support,
                    accounting,
                    camera,
                    index,
                    render_cache[mask.camera_name],
                )
            arms = {}
            for arm_name, arm_mask in (
                ("CANDIDATE_SPATIAL_POPULATION", np.ones(len(candidate_points), dtype=bool)),
                ("MASK_ONLY_SUPPORT", np.asarray(accounting["arrays"]["mask_only"], dtype=bool)),
                ("MASK_PLUS_DEPTH_SUPPORT", np.asarray(accounting["arrays"]["mask_plus_depth"], dtype=bool)),
                ("REMOVED_BY_DEPTH", np.asarray(accounting["arrays"]["mask_only"], dtype=bool) & ~np.asarray(accounting["arrays"]["mask_plus_depth"], dtype=bool)),
            ):
                arm_points = candidate_points[arm_mask]
                audit = audit_raw_graphness(arm_points, control.config, h)
                arms[arm_name] = {
                    "point_count": int(len(arm_points)),
                    "row_ids_sha256": hashlib.sha256(support.candidate_row_ids[arm_mask].tobytes()).hexdigest(),
                    "xyz_sha256": _sha256_rows(arm_points),
                    "graphness": graphness_report(audit, h),
                    "3d_accounting": {**_component_accounting(arm_points, h), **_chart_occupancy(arm_points, control.config)},
                }
            alignment = support_alignment_report(support, raw_points)
            case_report = {
                "surface_control": control.as_json(),
                "support_alignment": alignment,
                "mask_only_reproduction": _reproduce_wl141_support(control.name, support),
                "depth_accounting": {key: value for key, value in accounting.items() if key not in {"arrays"}},
                "arms": arms,
                "geometry_outputs": geometry,
                "3d_review_outputs": three_d_outputs,
                "camera_outputs": camera_outputs,
                "representative_replay": ({key: value for key, value in representative_result.items() if key != "representative"} if representative_result is not None else {"status": "NOT_EXECUTED_UNTIL_EXPLICIT_HUMAN_PASS"}),
                "human_same_sheet_review_status": "CLEAR_SAME_SHEET_NOT_DECLARED; HUMAN_REVIEW_REQUIRED",
                "provisional_attribution": (
                    "B. DEPTH_LAYER_CONTAMINATION_CANDIDATE"
                    if int(accounting["removed_by_depth_count"]) > 0
                    else "F. MIXED / INCONCLUSIVE"
                ),
                "failure_attribution": {
                    "A_PROJECTION_COORDINATE_CONTRACT": "mechanical projection and renderer-alpha control passed; human overlay review remains required",
                    "B_VISIBILITY_DEPTH_LAYER": {
                        "mask_only_count": int(accounting["mask_only_count"]),
                        "removed_by_depth_count": int(accounting["removed_by_depth_count"]),
                        "interpretation": "mask agreement includes depth-inconsistent candidates; this is contamination evidence, not membership proof",
                    },
                    "C_TRUE_SUPPORT_MEMBERSHIP": "unresolved; renderer depth consistency cannot prove same physical sheet",
                },
            }
            case_report["case_report_path"] = str(case_root / "case_report.json")
            case_reports[control.name] = case_report
            (case_root / "case_report.json").write_text(json.dumps(_jsonable(case_report), indent=2), encoding="utf-8")
        except Exception as error:
            failures.append({"case": control.name, "error": repr(error)})

    report = {
        "batch": manifest["batch"],
        "status": "ISOLATED_NON_CANONICAL_EVALUATION",
        "INTENT ALIGNMENT": {
            "automatic_surface_membership_implemented": False,
            "historical_mask_only_replayed": True,
            "mask_plus_depth_support_candidate": True,
            "appearance_sh_executed": False,
            "continuation_executed": False,
            "occluded_surface_executed": False,
        },
        "IMPLEMENTATION FIDELITY": {
            "canonical_renderer_modified": False,
            "canonical_checkpoint_modified": False,
            "wl127_geometry_modified": False,
            "wl139_module_modified": False,
            "wl141_camera_masks_modified": False,
            "wl141_mask_vote_rule_modified": False,
            "depth_maps_source": "read-only canonical OSNSurfelRasterizer.render()['depth_median']",
            "depth_tolerance_world": mu,
            "depth_tolerance_derivation": "frozen WL139 mu; no tuning",
            "graphness_and_representative_settings_unchanged": True,
            "connected_components_used_to_modify_support": False,
            "appearance_used": False,
            "human_qualitative_pass_cases": sorted(qualitative_pass),
        },
        "PROJECTION CONTRACT AUDIT": projection_audit,
        "KNOWN-GEOMETRY PROJECTION CONTROL": known_geometry,
        "MASK_ONLY SUPPORT REPRODUCTION": {name: case.get("mask_only_reproduction") for name, case in case_reports.items()},
        "PER-VIEW DEPTH-LAYER ACCOUNTING": {name: case.get("depth_accounting") for name, case in case_reports.items()},
        "MASK_ONLY FAILURE ATTRIBUTION": {
            name: {
                "mask_only_count": case["depth_accounting"]["mask_only_count"],
                "removed_by_depth_count": case["depth_accounting"]["removed_by_depth_count"],
                "removed_by_depth_fraction": case["depth_accounting"]["removed_by_depth_fraction_of_mask_only"],
                "interpretation": case["provisional_attribution"],
            }
            for name, case in case_reports.items()
        },
        "MASK_PLUS_DEPTH SUPPORT": {
            name: {
                "count": case["depth_accounting"]["mask_plus_depth_count"],
                "fraction_of_mask_only": float(case["depth_accounting"]["mask_plus_depth_count"] / max(case["depth_accounting"]["mask_only_count"], 1)),
                "fixed_rule": case["depth_accounting"]["rule"],
            }
            for name, case in case_reports.items()
        },
        "PER-CASE 3D ACCOUNTING": {name: case.get("arms") for name, case in case_reports.items()},
        "MULTI-VIEW QUALITATIVE EXPORTS": {name: case.get("camera_outputs") for name, case in case_reports.items()},
        "3D REVIEW EXPORTS": {name: case.get("3d_review_outputs") for name, case in case_reports.items()},
        "HUMAN SAME-SHEET REVIEW STATUS": {name: case.get("human_same_sheet_review_status") for name, case in case_reports.items()},
        "CONDITIONAL WL139 REPRESENTATIVE REPLAY": {name: case.get("representative_replay") for name, case in case_reports.items()},
        "ARCHITECTURE ATTRIBUTION": {
            "verdict": "F. MIXED / INCONCLUSIVE — projection passes mechanically; depth-filtered support still requires human same-sheet review",
            "projection_failure": False,
            "depth_layer_contamination_evidence_present": any(case["depth_accounting"]["removed_by_depth_count"] > 0 for case in case_reports.values()),
            "same_sheet_membership_resolved": False,
            "representative_clean_support_resolved": any(
                case.get("representative_replay", {}).get("status") == "REPLAYED_AFTER_EXPLICIT_HUMAN_PASS"
                for case in case_reports.values()
            ),
        },
        "PROMOTED": ["projection contract, mechanically, pending known-geometry visual review"],
        "RETAINED": ["Single-Surface Support as a distinct layer", "renderer-grounded depth consistency as a necessary support-lifting candidate", "WL139 representative candidate", "graphness applicability veto", "SH appearance as a future separation candidate"],
        "REJECTED": ["image-mask agreement alone as same-sheet proof", "local 3D coherence alone as semantic identity proof", "depth consistency as proof of physical-sheet identity", "appearance/SH used to compensate for projection/depth ambiguity"],
        "OPEN": ["automatic membership", "connectedness/geometric continuity", "appearance evidence", "support-domain trimming", "representative family", "continuation and Occluded Surface"],
        "cases": case_reports,
        "failures": failures,
        "inputs": {
            "raw_visible_surface": str(raw_path.resolve()),
            "raw_visible_surface_sha256": _sha256_file(raw_path),
            "wl139_report": str(wl139_report_path.resolve()),
            "wl141_report": str(WL141_REPORT.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "source_path": str(Path(arguments.source_path).resolve()),
            "frozen_manifest": str(manifest_path.resolve()),
            "output_root": str(output_root.resolve()),
        },
    }
    report_path = output_root / "multi_view_support_lifting_projection_depth_attribution_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Worklog 142 multi-view support lifting attribution",
        "",
        "이 출력은 자동 Surface Membership가 아니다. WL141의 `MASK_ONLY_BASELINE`을 그대로 재현한 뒤, canonical renderer의 기존 `depth_median`과 camera-space point depth를 비교해 `MASK_PLUS_DEPTH_SUPPORT`를 진단한다.",
        "",
        f"Projection/depth verdict: {report.get('ARCHITECTURE ATTRIBUTION', {}).get('verdict', report.get('STOP_REASON', ''))}",
        "",
        "사람이 먼저 확인할 경로: `frozen_wl141_mask_only_replay_manifest.json`, `projection_control/`, 각 case의 `camera_overlays/`, `3d_review/`, `case_report.json`.",
        "",
        "depth tolerance는 WL139 frozen `mu`에서 고정적으로 유도했으며, SH/color, connected-component pruning, trimming, continuation, Occluded Surface는 실행하지 않았다.",
        "",
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-visible-surface", type=Path, default=WL127_RAW_VISIBLE_SURFACE)
    parser.add_argument("--wl139-report", type=Path, default=WL139_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-fit-points", type=int, default=12000)
    parser.add_argument("--qualitative-pass", nargs="*", default=[], help="explicit human-reviewed cases eligible for one frozen WL139 replay")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    report = run_demo(arguments)
    print(json.dumps({"status": report["status"], "verdict": report.get("ARCHITECTURE ATTRIBUTION", {}).get("verdict", report.get("STOP_REASON")), "failures": report["failures"]}, indent=2))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
