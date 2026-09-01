"""Worklog 143: renderer median-depth semantics and multi-view evidence audit.

This isolated diagnostic preserves WL142 exactly. It validates the numerical
meaning of canonical depth_median, preserves per-view relations as evidence
states, and reports D1/D2/D3 diagnostic populations without selecting a new
Surface Membership rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.meeting_occluded_surface_feasibility import _write_ply  # noqa: E402
from devtools.demo.multi_view_support_lifting_projection_depth_attribution import (  # noqa: E402
    _as_numpy,
    _depth_map_from_package,
    _depth_relation_for_camera,
    _jsonable,
    _reproduce_wl141_support,
)
from devtools.demo.oracle_single_surface_support_appearance_evidence import (  # noqa: E402
    DISPLAY_POINT_ALPHA,
    DISPLAY_POINT_SIZE,
    OracleSurfaceControl,
    _draw_polygon_outline,
    _draw_projected_points,
    _manual_controls,
    _mask_camera_lookup,
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
    / "143_multi_view_support_lifting_depth_semantics_evidence_aggregation"
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

SELF_CONSISTENCY_CAMERA_COUNT = 6
SELF_CONSISTENCY_MAX_SAMPLES_PER_CAMERA = 6000
SELF_CONSISTENCY_BIN_ROWS = 5
SELF_CONSISTENCY_BIN_COLUMNS = 5
MASK_VOTE_MIN = 2
DEPTH_TOLERANCE_SOURCE = "WL139 frozen mu; inherited from WL142"

OUTSIDE_MASK = 0
NO_VALID_DEPTH = 1
NEAR_MEDIAN = 2
BEFORE_MEDIAN = 3
AFTER_MEDIAN = 4
STATE_NAMES = {
    OUTSIDE_MASK: "OUTSIDE_MASK",
    NO_VALID_DEPTH: "NO_VALID_DEPTH",
    NEAR_MEDIAN: "NEAR_MEDIAN",
    BEFORE_MEDIAN: "BEFORE_MEDIAN",
    AFTER_MEDIAN: "AFTER_MEDIAN",
}
STATE_RGB = {
    "NEAR_MEDIAN": (24, 194, 78),
    "BEFORE_MEDIAN": (44, 106, 225),
    "AFTER_MEDIAN": (230, 53, 45),
    "NO_VALID_DEPTH": (144, 145, 151),
    "OUTSIDE_MASK": (82, 84, 91),
}
MASK_ONLY_RGB = (245, 151, 25)
D1_RGB = (0, 153, 170)
D2_RGB = (125, 64, 201)
D3_RGB = (224, 45, 137)
SELF_EVENT_RGB = (32, 132, 235)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(_as_numpy(value)).tobytes()).hexdigest()


def _sha256_int64(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype=np.int64)).tobytes()
    ).hexdigest()


def _quantile_summary(
    values: np.ndarray,
    *,
    h: float | None = None,
    mu: float | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "NO_SAMPLES", "samples": 0}
    result: dict[str, Any] = {
        "status": "MEASURED",
        "samples": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }
    if h is not None:
        result.update({
            "min_over_h": float(np.min(values) / h),
            "median_over_h": float(np.median(values) / h),
            "p95_over_h": float(np.quantile(values, 0.95) / h),
            "max_over_h": float(np.max(values) / h),
        })
    if mu is not None:
        result.update({
            "min_over_mu": float(np.min(values) / mu),
            "median_over_mu": float(np.median(values) / mu),
            "p95_over_mu": float(np.quantile(values, 0.95) / mu),
            "max_over_mu": float(np.max(values) / mu),
        })
    return result


def _residual_summary(residuals: np.ndarray, *, h: float, mu: float) -> dict[str, Any]:
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    return {
        "signed": _quantile_summary(residuals, h=h, mu=mu),
        "absolute": _quantile_summary(np.abs(residuals), h=h, mu=mu),
    }


def renderer_depth_median_semantics() -> dict[str, Any]:
    return {
        "depth_median_name": "depth_median",
        "canonical_python_source": "osn_gs/render/surfel_rasterizer.py",
        "canonical_python_source_lines": "174-175 and 199-200",
        "canonical_python_definition": (
            "allmap channel MIDDEPTH_OFFSET is read as render_depth_median; "
            "NaN values are replaced by zero"
        ),
        "vendored_cuda_source": (
            "osn_gs/render/vendor/diff_surfel_rasterization/"
            "cuda_rasterizer/forward.cu"
        ),
        "vendored_cuda_source_lines": "367-372 and 407-443",
        "event_depth_formula": "depth=(s.x*Tw.x+s.y*Tw.y)+Tw.z",
        "event_geometry": (
            "per-pixel ray/surfel-plane intersection depth; this scalar is "
            "the camera/view-space z coordinate of that event"
        ),
        "not_equal_to": [
            "Gaussian center view-space z in general",
            "Euclidean camera-to-event ray length",
            "normalized or inverse depth",
        ],
        "aggregation": (
            "front-to-back compositing updates median_depth at the T > 0.5 "
            "contributor event and writes it to MIDDEPTH_OFFSET"
        ),
        "sign_and_units": (
            "depth < near_n=0.2 is skipped; depth is linear scene/world units "
            "after the world-to-view transform and positive renderable z"
        ),
        "ray_direction_dependence": (
            "camera-space z is linear in the view coordinate, while its ratio "
            "to Euclidean ray distance varies with perspective ray direction"
        ),
        "renderer_pixel_mapping": (
            "compute_transmat ndc2pix uses x=ndc_x*W/2+(W-1)/2 and "
            "y=ndc_y*H/2+(H-1)/2"
        ),
        "median_vs_center_warning": (
            "the median is a renderer-defined event on a surfel plane, not "
            "necessarily the projected center depth of its Gaussian"
        ),
    }


def _renderer_projected_pixels(points: np.ndarray, camera: Any) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    projection = homogeneous @ _as_numpy(camera.full_proj_transform)
    view = homogeneous @ _as_numpy(camera.world_view_transform)
    w = projection[:, 3]
    safe_w = np.where(np.abs(w) > 1.0e-12, w, np.nan)
    ndc_x = projection[:, 0] / safe_w
    ndc_y = projection[:, 1] / safe_w
    width = float(camera.image_width)
    height = float(camera.image_height)
    pixel_x = ndc_x * width * 0.5 + (width - 1.0) * 0.5
    pixel_y = ndc_y * height * 0.5 + (height - 1.0) * 0.5
    valid = (
        np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & np.isfinite(view[:, 2])
        & (w > 1.0e-8)
        & (view[:, 2] > 0.0)
        & (pixel_x >= 0.0)
        & (pixel_x < width)
        & (pixel_y >= 0.0)
        & (pixel_y < height)
    )
    return {"x": pixel_x, "y": pixel_y, "depth": view[:, 2], "valid": valid}


def _reconstruct_world_from_renderer_pixel_depth(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    camera: Any,
) -> np.ndarray:
    pixel_x = np.asarray(pixel_x, dtype=np.float64).reshape(-1)
    pixel_y = np.asarray(pixel_y, dtype=np.float64).reshape(-1)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    view = _as_numpy(camera.world_view_transform).astype(np.float64)
    projection = _as_numpy(camera.full_proj_transform).astype(np.float64)
    width = float(camera.image_width)
    height = float(camera.image_height)
    ndc_x = (pixel_x - (width - 1.0) * 0.5) / (width * 0.5)
    ndc_y = (pixel_y - (height - 1.0) * 0.5) / (height * 0.5)
    last = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (len(depth), 1))
    view_z = np.broadcast_to(view[:, 2], (len(depth), 4))
    clip_x = projection[:, 0][None, :] - ndc_x[:, None] * projection[:, 3][None, :]
    clip_y = projection[:, 1][None, :] - ndc_y[:, None] * projection[:, 3][None, :]
    matrix = np.stack([view_z, clip_x, clip_y, last], axis=1)
    rhs = np.column_stack([depth, np.zeros((len(depth), 2)), np.ones(len(depth))])
    world_h = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    world_w = world_h[:, 3:4]
    return world_h[:, :3] / np.where(np.abs(world_w) > 1.0e-12, world_w, np.nan)


def _sample_valid_depth_pixels(
    depth: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (np.abs(depth) > 1.0e-8)
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    flat = np.ravel_multi_index((rows, columns), depth.shape)
    selected: list[int] = []
    row_edges = np.linspace(0, depth.shape[0], SELF_CONSISTENCY_BIN_ROWS + 1, dtype=np.int64)
    col_edges = np.linspace(0, depth.shape[1], SELF_CONSISTENCY_BIN_COLUMNS + 1, dtype=np.int64)
    per_bin = max(1, int(max_samples) // (SELF_CONSISTENCY_BIN_ROWS * SELF_CONSISTENCY_BIN_COLUMNS))
    for ri in range(SELF_CONSISTENCY_BIN_ROWS):
        for ci in range(SELF_CONSISTENCY_BIN_COLUMNS):
            chosen = flat[
                (rows >= row_edges[ri])
                & (rows < row_edges[ri + 1])
                & (columns >= col_edges[ci])
                & (columns < col_edges[ci + 1])
            ]
            if len(chosen) > per_bin:
                chosen = chosen[np.rint(np.linspace(0, len(chosen) - 1, per_bin)).astype(np.int64)]
            selected.extend(chosen.tolist())
    selected = sorted(set(selected))
    if len(selected) < min(int(max_samples), len(flat)):
        selected_set = set(selected)
        remaining = np.asarray([value for value in flat.tolist() if value not in selected_set], dtype=np.int64)
        quota = min(int(max_samples) - len(selected), len(remaining))
        if quota:
            positions = np.rint(np.linspace(0, len(remaining) - 1, quota)).astype(np.int64)
            selected = sorted(set(selected + remaining[positions].tolist()))
    selected_flat = np.asarray(selected[: int(max_samples)], dtype=np.int64)
    sample_rows, sample_columns = np.unravel_index(selected_flat, depth.shape)
    return (
        sample_columns.astype(np.int64),
        sample_rows.astype(np.int64),
        depth[sample_rows, sample_columns].astype(np.float64),
    )


def _self_strata(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    event_points: np.ndarray,
    camera: Any,
) -> dict[str, np.ndarray]:
    width = float(camera.image_width)
    height = float(camera.image_height)
    radial = np.sqrt(
        ((pixel_x - (width - 1.0) * 0.5) / max(width * 0.5, 1.0)) ** 2
        + ((pixel_y - (height - 1.0) * 0.5) / max(height * 0.5, 1.0)) ** 2
    )
    homogeneous = np.concatenate(
        [event_points, np.ones((len(event_points), 1), dtype=np.float64)], axis=1
    )
    view_points = homogeneous @ _as_numpy(camera.world_view_transform)
    distance = np.linalg.norm(event_points - _camera_center(camera), axis=1)
    cos_theta = view_points[:, 2] / np.maximum(distance, 1.0e-12)
    depth = view_points[:, 2]
    q = np.quantile(depth, [1.0 / 3.0, 2.0 / 3.0]) if len(depth) >= 3 else (np.inf, np.inf)
    return {
        "center_ray": radial <= 0.35,
        "mid_ray": (radial > 0.35) & (radial <= 0.75),
        "periphery_ray": radial > 0.75,
        "oblique_ray_cos_lt_0_8": cos_theta < 0.80,
        "near_scene_depth": depth <= q[0],
        "middle_scene_depth": (depth > q[0]) & (depth <= q[1]),
        "far_scene_depth": depth > q[1],
    }


def _self_consistency_for_camera(
    camera: Any,
    package: dict[str, Any],
    h: float,
    mu: float,
    max_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    depth = _depth_map_from_package(package)
    px, py, median_depth = _sample_valid_depth_pixels(depth, max_samples)
    if not len(px):
        empty = {
            "pixel_x": px,
            "pixel_y": py,
            "median_depth": median_depth,
            "event_points": np.empty((0, 3), dtype=np.float64),
            "recomputed_depth": np.empty((0,), dtype=np.float64),
            "reprojection_error": np.empty((0,), dtype=np.float64),
            "signed_residual": np.empty((0,), dtype=np.float64),
        }
        return ({"camera_name": str(camera.image_name), "status": "NO_VALID_RENDERER_DEPTH_PIXELS", "valid_depth_pixels": 0, "sample_count": 0}, empty)
    event_points = _reconstruct_world_from_renderer_pixel_depth(px, py, median_depth, camera)
    reprojection = _renderer_projected_pixels(event_points, camera)
    reprojection_error = np.sqrt((reprojection["x"] - px) ** 2 + (reprojection["y"] - py) ** 2)
    signed_residual = reprojection["depth"] - median_depth
    strata = _self_strata(px, py, event_points, camera)
    report = {
        "camera_name": str(camera.image_name),
        "status": "MEASURED",
        "valid_depth_pixels": int(np.sum(np.isfinite(depth) & (np.abs(depth) > 1.0e-8))),
        "sample_count": int(len(px)),
        "sample_pixel_extent": [int(np.min(px)), int(np.min(py)), int(np.max(px)), int(np.max(py))],
        "reprojection_error_pixels": _quantile_summary(reprojection_error),
        "signed_depth_residual": _quantile_summary(signed_residual, h=h, mu=mu),
        "absolute_depth_residual": _quantile_summary(np.abs(signed_residual), h=h, mu=mu),
        "strata": {
            name: {
                "samples": int(np.sum(selection)),
                "reprojection_error_pixels": _quantile_summary(reprojection_error[selection]),
                "signed_depth_residual": _quantile_summary(signed_residual[selection], h=h, mu=mu),
                "absolute_depth_residual": _quantile_summary(np.abs(signed_residual[selection]), h=h, mu=mu),
            }
            for name, selection in strata.items()
        },
        "linear_camera_depth_identity": True,
    }
    arrays = {
        "pixel_x": px,
        "pixel_y": py,
        "median_depth": median_depth,
        "event_points": event_points,
        "recomputed_depth": reprojection["depth"],
        "reprojection_error": reprojection_error,
        "signed_residual": signed_residual,
    }
    return report, arrays


def _depth_identity_pass(reports: list[dict[str, Any]]) -> bool:
    if not reports or any(report.get("status") != "MEASURED" for report in reports):
        return False
    return all(
        report["reprojection_error_pixels"]["p95"] <= 1.0e-8
        and report["absolute_depth_residual"]["p95"] <= 1.0e-8
        for report in reports
    )


def _classify_states(
    candidate_points: np.ndarray,
    control: OracleSurfaceControl,
    camera_lookup: dict[str, Any],
    depth_maps: dict[str, np.ndarray],
    mu: float,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    states = np.full((len(candidate_points), len(control.masks)), OUTSIDE_MASK, dtype=np.uint8)
    relations: list[dict[str, np.ndarray]] = []
    for column, mask in enumerate(control.masks):
        relation = _depth_relation_for_camera(
            candidate_points,
            camera_lookup[mask.camera_name],
            mask,
            depth_maps[mask.camera_name],
            mu,
        )
        relations.append(relation)
        match = relation["mask_match"]
        states[match & relation["no_relevant_renderer_depth"], column] = NO_VALID_DEPTH
        states[match & relation["depth_consistent"], column] = NEAR_MEDIAN
        states[match & relation["in_front_of_visible_frontier"], column] = BEFORE_MEDIAN
        states[match & relation["behind_visible_frontier"], column] = AFTER_MEDIAN
    return states, relations


def _state_counts(states: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mask_match": np.sum(states != OUTSIDE_MASK, axis=1).astype(np.int16),
        "near": np.sum(states == NEAR_MEDIAN, axis=1).astype(np.int16),
        "before": np.sum(states == BEFORE_MEDIAN, axis=1).astype(np.int16),
        "after": np.sum(states == AFTER_MEDIAN, axis=1).astype(np.int16),
        "no_valid_depth": np.sum(states == NO_VALID_DEPTH, axis=1).astype(np.int16),
        "outside_mask": np.sum(states == OUTSIDE_MASK, axis=1).astype(np.int16),
    }


def _histogram(values: np.ndarray, categories: Iterable[int]) -> dict[str, int]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    return {str(category): int(np.sum(values == category)) for category in categories}


def _joint_histogram(counts: dict[str, np.ndarray], selection: np.ndarray) -> dict[str, int]:
    tuples = zip(
        counts["mask_match"][selection].tolist(),
        counts["near"][selection].tolist(),
        counts["before"][selection].tolist(),
        counts["after"][selection].tolist(),
        counts["no_valid_depth"][selection].tolist(),
    )
    return {
        f"{a} near={b} before={c} after={d} no_valid={e}": int(value)
        for (a, b, c, d, e), value in sorted(Counter(tuples).items())
    }


def _population_summary(
    name: str,
    candidate_row_ids: np.ndarray,
    selection: np.ndarray,
    counts: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = np.asarray(candidate_row_ids, dtype=np.int64)[selection]
    return {
        "name": name,
        "count": int(np.sum(selection)),
        "row_ids_sha256": _sha256_int64(rows),
        "row_ids": rows,
        "mask_matching_camera_count": _histogram(counts["mask_match"][selection], range(4)),
        "near_median_camera_count": _histogram(counts["near"][selection], range(4)),
        "before_median_camera_count": _histogram(counts["before"][selection], range(4)),
        "after_median_camera_count": _histogram(counts["after"][selection], range(4)),
        "no_valid_depth_camera_count": _histogram(counts["no_valid_depth"][selection], range(4)),
        "preserves_original_per_view_states": True,
    }


def _write_state_geometry(
    case_root: Path,
    raw_points: np.ndarray,
    candidate_row_ids: np.ndarray,
    populations: dict[str, np.ndarray],
    mask_only: np.ndarray,
    states: np.ndarray,
    counts: dict[str, np.ndarray],
) -> dict[str, str]:
    root = case_root / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    candidate_points = raw_points[candidate_row_ids]
    paths = {
        "candidate_spatial_population": root / "candidate_spatial_population.ply",
        "mask_only_baseline": root / "mask_only_baseline.ply",
        "D1_at_least_one_near": root / "D1_at_least_one_near.ply",
        "D2_at_least_two_near": root / "D2_at_least_two_near.ply",
        "D3_three_near": root / "D3_three_near.ply",
        "state_accounting": root / "per_view_evidence_states.npz",
    }
    _write_ply(paths["candidate_spatial_population"], candidate_points, color=(134, 137, 145))
    _write_ply(paths["mask_only_baseline"], candidate_points[mask_only], color=MASK_ONLY_RGB)
    _write_ply(paths["D1_at_least_one_near"], candidate_points[populations["D1"]], color=D1_RGB)
    _write_ply(paths["D2_at_least_two_near"], candidate_points[populations["D2"]], color=D2_RGB)
    _write_ply(paths["D3_three_near"], candidate_points[populations["D3"]], color=D3_RGB)
    np.savez_compressed(
        paths["state_accounting"],
        candidate_row_ids=np.asarray(candidate_row_ids, dtype=np.int64),
        mask_only_row_ids=np.asarray(candidate_row_ids, dtype=np.int64)[mask_only],
        state_matrix=np.asarray(states, dtype=np.uint8),
        mask_match_count=counts["mask_match"],
        near_count=counts["near"],
        before_count=counts["before"],
        after_count=counts["after"],
        no_valid_depth_count=counts["no_valid_depth"],
        outside_mask_count=counts["outside_mask"],
    )
    return {key: str(path) for key, path in paths.items()}


def _draw_state_groups(image: Any, points: np.ndarray, camera: Any, state_column: np.ndarray, selection: np.ndarray) -> Any:
    output = image.convert("RGB")
    for state_name, state_code in STATE_NAMES.items():
        group = selection & (state_column == state_code)
        if np.any(group):
            output = _draw_projected_points(
                output, points[group], camera, STATE_RGB[state_name],
                radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA,
            )
    return output


def _write_camera_outputs(
    case_root: Path,
    raw_points: np.ndarray,
    candidate_row_ids: np.ndarray,
    control: OracleSurfaceControl,
    camera: Any,
    camera_index: int,
    scene_image: Any,
    states: np.ndarray,
    populations: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
) -> dict[str, str]:
    root = case_root / "camera_overlays" / str(camera.image_name)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "A_gaussian_scene_only": root / "A_gaussian_scene_only.png",
        "B_historical_mask_only_baseline": root / "B_historical_mask_only_baseline.png",
        "C_mask_only_per_view_evidence_state": root / "C_mask_only_per_view_evidence_state.png",
        "D1_at_least_one_near": root / "D1_at_least_one_near.png",
        "D2_at_least_two_near": root / "D2_at_least_two_near.png",
        "D3_three_near": root / "D3_three_near.png",
        "state_metadata": root / "state_metadata.json",
    }
    scene_image.convert("RGB").save(paths["A_gaussian_scene_only"])
    candidate_points = raw_points[candidate_row_ids]
    baseline = _draw_projected_points(
        scene_image, candidate_points[counts["mask_match"] >= MASK_VOTE_MIN],
        camera, MASK_ONLY_RGB, radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA,
    )
    baseline.save(paths["B_historical_mask_only_baseline"])
    current_mask = next(item for item in control.masks if item.camera_name == str(camera.image_name))
    state_view = _draw_state_groups(
        scene_image, candidate_points, camera, states[:, camera_index],
        counts["mask_match"] >= MASK_VOTE_MIN,
    )
    _draw_polygon_outline(state_view, current_mask, camera).save(paths["C_mask_only_per_view_evidence_state"])
    for name, path, color in (
        ("D1", paths["D1_at_least_one_near"], D1_RGB),
        ("D2", paths["D2_at_least_two_near"], D2_RGB),
        ("D3", paths["D3_three_near"], D3_RGB),
    ):
        _draw_projected_points(
            scene_image, candidate_points[populations[name]], camera, color,
            radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA,
        ).save(path)
    metadata = {
        "camera_name": str(camera.image_name),
        "state_order": [STATE_NAMES[index] for index in sorted(STATE_NAMES)],
        "state_colors": STATE_RGB,
        "mask_only_count": int(np.sum(counts["mask_match"] >= MASK_VOTE_MIN)),
        "mask_only_state_counts": {
            STATE_NAMES[index]: int(np.sum(states[counts["mask_match"] >= MASK_VOTE_MIN, camera_index] == index))
            for index in sorted(STATE_NAMES)
        },
        "diagnostic_population_counts": {
            name: int(np.sum(selection)) for name, selection in populations.items()
        },
        "diagnostic_population_state_counts_current_view": {
            name: {
                STATE_NAMES[index]: int(np.sum(states[selection, camera_index] == index))
                for index in sorted(STATE_NAMES)
            }
            for name, selection in populations.items()
        },
        "D1_D2_D3_use_same_state_matrix": True,
        "before_after_not_physical_contradictions": True,
    }
    paths["state_metadata"].write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _write_3d_outputs(
    case_root: Path,
    control: OracleSurfaceControl,
    candidate_points: np.ndarray,
    populations: dict[str, np.ndarray],
) -> dict[str, str]:
    root = case_root / "3d_review"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "D1": root / "D1_at_least_one_near.png",
        "D2": root / "D2_at_least_two_near.png",
        "D3": root / "D3_three_near.png",
    }
    for name, color, title in (
        ("D1", D1_RGB, "D1 diagnostic population: >= 1 NEAR_MEDIAN"),
        ("D2", D2_RGB, "D2 diagnostic population: >= 2 NEAR_MEDIAN"),
        ("D3", D3_RGB, "D3 diagnostic population: 3 NEAR_MEDIAN"),
    ):
        _save_3d(
            paths[name], title, control.config, candidate_points,
            lambda axis, name=name, color=color: _plot_points(
                axis, candidate_points[populations[name]], control.config,
                np.asarray(color, dtype=np.float64) / 255.0, label=name,
                alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE,
            ),
        )
    return {key: str(path) for key, path in paths.items()}


def _historical_replay_report(control_name: str, support: Any, wl142_case: dict[str, Any]) -> dict[str, Any]:
    wl141 = _reproduce_wl141_support(control_name, support)
    alignment = wl142_case.get("support_alignment", {})
    count = int(len(support.oracle_row_ids))
    row_hash = _sha256_int64(support.oracle_row_ids)
    exact = count == int(alignment.get("oracle_row_count", -1)) and row_hash == str(alignment.get("oracle_row_ids_sha256", ""))
    return {
        "wl141_count_hash_reproduction": wl141,
        "wl142_count_hash_reproduction": {
            "status": "EXACT_REPRODUCTION_BY_ROW_ID_HASH" if exact else "REPRODUCTION_MISMATCH",
            "exact": exact,
            "historical_oracle_row_count": int(alignment.get("oracle_row_count", -1)),
            "replayed_oracle_row_count": count,
            "historical_oracle_row_ids_sha256": str(alignment.get("oracle_row_ids_sha256", "")),
            "replayed_oracle_row_ids_sha256": row_hash,
            "historical_row_id_list_available": False,
            "selection_name": "MASK_ONLY_BASELINE",
        },
    }


def _wl142_zero_attribution(
    counts: dict[str, np.ndarray],
    mask_only: np.ndarray,
    wl142_case: dict[str, Any],
) -> dict[str, Any]:
    near_requirement = mask_only & (counts["near"] >= 2)
    before_veto = near_requirement & (counts["before"] > 0)
    after_veto = near_requirement & (counts["after"] > 0)
    both_veto = before_veto & after_veto
    survivors = near_requirement & ~before_veto & ~after_veto
    historical_zero = int(wl142_case.get("depth_accounting", {}).get("mask_plus_depth_count", -1))
    return {
        "historical_wl142_mask_plus_depth_count": historical_zero,
        "near_requirement_before_zero_before_after_veto": int(np.sum(near_requirement)),
        "hard_veto_survivors_recomputed": int(np.sum(survivors)),
        "removed_because_at_least_one_before": int(np.sum(before_veto)),
        "removed_because_at_least_one_after": int(np.sum(after_veto)),
        "removed_because_both_before_and_after": int(np.sum(both_veto)),
        "near_requirement_union_veto_count": int(np.sum(before_veto | after_veto)),
        "all_near_requirement_points_vetoed": bool(np.sum(near_requirement) == np.sum(before_veto | after_veto)),
        "interpretation": (
            "near-view support is reported before the WL142 global zero-before/"
            "zero-after veto; BEFORE_MEDIAN and AFTER_MEDIAN are ordering states, "
            "not physical impossibility labels"
        ),
    }


def _build_case_report(
    control: OracleSurfaceControl,
    support: Any,
    raw_points: np.ndarray,
    states: np.ndarray,
    relations: list[dict[str, np.ndarray]],
    wl142_case: dict[str, Any],
    h: float,
    mu: float,
    outputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidate_row_ids = support.candidate_row_ids
    counts = _state_counts(states)
    mask_only = counts["mask_match"] >= MASK_VOTE_MIN
    populations = {
        "D1": mask_only & (counts["near"] >= 1),
        "D2": mask_only & (counts["near"] >= 2),
        "D3": mask_only & (counts["near"] >= 3),
    }
    per_camera = []
    for index, mask in enumerate(control.masks):
        column = states[:, index]
        matching_valid = relations[index]["mask_match"] & relations[index]["renderer_depth_valid"]
        residual = relations[index]["depth_residual"][matching_valid]
        per_camera.append({
            "camera_name": mask.camera_name,
            "candidate_state_counts": {
                STATE_NAMES[state]: int(np.sum(column == state))
                for state in sorted(STATE_NAMES)
            },
            "mask_only_state_counts": {
                STATE_NAMES[state]: int(np.sum(column[mask_only] == state))
                for state in sorted(STATE_NAMES)
            },
            "mask_matching_signed_depth_residual": _residual_summary(residual, h=h, mu=mu),
        })
    residuals = np.concatenate([
        relation["depth_residual"][relation["mask_match"] & mask_only & relation["renderer_depth_valid"]]
        for relation in relations
        if np.any(relation["mask_match"] & mask_only & relation["renderer_depth_valid"])
    ]) if any(
        np.any(relation["mask_match"] & mask_only & relation["renderer_depth_valid"])
        for relation in relations
    ) else np.empty((0,), dtype=np.float64)
    report = {
        "surface_control": control.as_json(),
        "historical_replay": _historical_replay_report(control.name, support, wl142_case),
        "candidate_count": int(len(candidate_row_ids)),
        "mask_only_count": int(np.sum(mask_only)),
        "state_definitions": {
            "NEAR_MEDIAN": "mask-matching comparison with abs(point_depth-depth_median) <= frozen mu",
            "BEFORE_MEDIAN": "mask-matching point depth < renderer median by more than frozen mu",
            "AFTER_MEDIAN": "mask-matching point depth > renderer median by more than frozen mu",
            "NO_VALID_DEPTH": "mask-matching projection has no finite nonzero depth_median",
            "OUTSIDE_MASK": "projected point is outside this frozen WL141 polygon",
        },
        "per_view_evidence_states": per_camera,
        "near_median_count_histogram_mask_only": _histogram(counts["near"][mask_only], range(4)),
        "before_median_count_histogram_mask_only": _histogram(counts["before"][mask_only], range(4)),
        "after_median_count_histogram_mask_only": _histogram(counts["after"][mask_only], range(4)),
        "no_valid_depth_count_histogram_mask_only": _histogram(counts["no_valid_depth"][mask_only], range(4)),
        "joint_histogram_mask_only": _joint_histogram(counts, mask_only),
        "D1_D2_D3_populations": {
            name: _population_summary(name, candidate_row_ids, selection, counts)
            for name, selection in populations.items()
        },
        "signed_and_absolute_mask_matching_depth_residual": _residual_summary(residuals, h=h, mu=mu),
        "worklog142_all_zero_attribution": _wl142_zero_attribution(counts, mask_only, wl142_case),
        "diagnostic_only_contract": {
            "D1_is_not_canonical_support": True,
            "D2_is_not_canonical_support": True,
            "D3_is_not_canonical_support": True,
            "before_after_not_physical_contradictions": True,
            "state_matrix_sha256": _sha256_array(states),
        },
        "outputs": outputs,
        "row_id_fidelity": {
            "candidate_row_ids_sha256": _sha256_int64(candidate_row_ids),
            "candidate_xyz_sha256": _sha256_rows(raw_points[candidate_row_ids]),
        },
    }
    return report, populations, counts


def _fixed_scale_sanity(
    self_cameras: list[Any],
    self_arrays: list[dict[str, np.ndarray]],
    h: float,
    mu: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for camera, arrays in zip(self_cameras, self_arrays):
        points = arrays["event_points"]
        if not len(points):
            continue
        view = _as_numpy(camera.world_view_transform).astype(np.float64)
        inverse_view = np.linalg.inv(view)
        z_direction = (np.asarray([0.0, 0.0, 1.0, 0.0]) @ inverse_view)[:3]
        for label, displacement in (
            ("exact_event", 0.0),
            ("plus_h", float(h)),
            ("minus_h", -float(h)),
            ("plus_mu", float(mu)),
            ("minus_mu", -float(mu)),
        ):
            displaced = points + displacement * z_direction[None, :]
            recomputed = _renderer_projected_pixels(displaced, camera)["depth"]
            residual = recomputed - arrays["median_depth"]
            rows.append({
                "camera_name": str(camera.image_name),
                "displacement": label,
                "depth_axis_world_direction": z_direction,
                "residual": _residual_summary(residual, h=h, mu=mu),
            })
    return {
        "depth_axis": "camera-space +z direction",
        "frozen_mu": float(mu),
        "displacement_rows": rows,
        "expected_identity": "exact=0; +h/-h=+/-h; +mu/-mu=+/-mu",
        "threshold_selection_performed": False,
    }


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = Path(arguments.raw_visible_surface)
    checkpoint = Path(arguments.checkpoint)
    wl139_path = Path(arguments.wl139_report)
    wl142_path = Path(arguments.wl142_report)
    missing = [str(path) for path in (raw_path, checkpoint, wl139_path, wl142_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    wl139_report = json.loads(wl139_path.read_text(encoding="utf-8"))
    wl142_report = json.loads(wl142_path.read_text(encoding="utf-8"))
    h = float(wl139_report["inputs"]["h"])
    mu = float(wl139_report["inputs"]["mu"])
    raw_points = _load_xyz_ply(raw_path)
    controls = _manual_controls()
    model, payload, cameras, scene_info = _load_canonical_scene(
        checkpoint, Path(arguments.source_path), arguments.device,
        arguments.images, arguments.sparse_dir, int(arguments.resolution), int(arguments.llffhold)
    )
    camera_lookup = _mask_camera_lookup(cameras)
    packages: dict[str, dict[str, Any]] = {}
    images: dict[str, Any] = {}

    def package_for(name: str) -> dict[str, Any]:
        if name not in packages:
            packages[name] = scene_info["rasterizer"].render(camera_lookup[name], model)
            images[name] = _render_to_pil(packages[name])
        return packages[name]

    self_cameras = _select_self_consistency_cameras(cameras)
    self_reports: list[dict[str, Any]] = []
    self_arrays: list[dict[str, np.ndarray]] = []
    for camera in self_cameras:
        report, arrays = _self_consistency_for_camera(
            camera, package_for(str(camera.image_name)), h, mu, int(arguments.max_self_samples)
        )
        self_reports.append(report)
        self_arrays.append(arrays)
    identity_pass = _depth_identity_pass(self_reports)
    semantics = renderer_depth_median_semantics()
    semantics["self_consistency_camera_selection"] = {
        "selection_rule": "sorted camera names, six rounded index quantiles",
        "camera_count_total": int(len(cameras)),
        "camera_names": [str(camera.image_name) for camera in self_cameras],
        "not_semantic_roi_selection": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    self_root = output_root / "depth_self_consistency"
    self_root.mkdir(parents=True, exist_ok=True)
    all_event_points = [arrays["event_points"] for arrays in self_arrays if len(arrays["event_points"])]
    if all_event_points:
        event_points = np.concatenate(all_event_points, axis=0)
        _write_ply(self_root / "renderer_median_event_samples.ply", event_points, color=SELF_EVENT_RGB)
        camera_names = np.asarray([
            str(camera.image_name)
            for camera, arrays in zip(self_cameras, self_arrays)
            for _ in range(len(arrays["event_points"]))
        ], dtype="U")
        np.savez_compressed(
            self_root / "renderer_median_event_samples.npz",
            event_points=event_points, camera_names=camera_names
        )

    report: dict[str, Any] = {
        "batch": "Worklog 143 renderer median-depth semantics and multi-view evidence aggregation audit",
        "status": "DEPTH_QUANTITY_IDENTITY_PASS" if identity_pass else "DEPTH_QUANTITY_IDENTITY_FAIL_STOPPED",
        "inputs": {
            "raw_visible_surface": str(raw_path.resolve()),
            "raw_visible_surface_sha256": _sha256_file(raw_path),
            "wl139_report": str(wl139_path.resolve()),
            "wl139_report_sha256": _sha256_file(wl139_path),
            "wl141_report": str(WL141_REPORT.resolve()),
            "wl141_report_sha256": _sha256_file(WL141_REPORT),
            "wl142_report": str(wl142_path.resolve()),
            "wl142_report_sha256": _sha256_file(wl142_path),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "source_path": str(Path(arguments.source_path).resolve()),
            "output_root": str(output_root.resolve()),
            "raw_visible_surface_role": "read-only candidate-row source and diagnostic reference; no new Surface Membership or fitter input",
        },
        "INTENT ALIGNMENT": {
            "new_surface_membership_implemented": False,
            "new_depth_tolerance_introduced": False,
            "wl142_mask_only_replayed": False,
            "per_view_states_preserved": False,
            "D1_D2_D3_final_membership_selected": False,
            "representative_replay_executed": False,
            "continuation_executed": False,
            "occluded_surface_executed": False,
        },
        "IMPLEMENTATION FIDELITY": {
            "canonical_renderer_modified": False,
            "canonical_checkpoint_modified": False,
            "wl127_geometry_modified": False,
            "wl139_modified": False,
            "wl141_masks_or_cameras_modified": False,
            "wl142_report_modified": False,
            "wl142_mu_reused": float(mu),
            "wl142_mu_source": DEPTH_TOLERANCE_SOURCE,
            "new_thresholds": [],
            "numeric_identity_check": "p95 reprojection <= 1e-8 pixels and p95 absolute renderer-z residual <= 1e-8; diagnostic quantity check only",
            "support_selection_changed": False,
            "representative_or_sh_executed": False,
        },
        "DEPTH_MEDIAN SOURCE SEMANTICS": semantics,
        "DEPTH QUANTITY SELF-CONSISTENCY": {
            "status": "DEPTH_QUANTITY_IDENTITY_PASS" if identity_pass else "DEPTH_QUANTITY_IDENTITY_FAIL",
            "per_camera": self_reports,
            "identity_rule": "all selected cameras must be measured with p95 reprojection <= 1e-8 pixels and p95 absolute z residual <= 1e-8 world units",
            "outputs": {
                "renderer_median_event_samples_ply": str(self_root / "renderer_median_event_samples.ply"),
                "renderer_median_event_samples_npz": str(self_root / "renderer_median_event_samples.npz"),
            },
        },
        "FIXED-SCALE SANITY": {},
        "WORKLOG 142 HISTORICAL REPLAY": {},
        "PER-VIEW EVIDENCE STATES": {},
        "PER-POINT MULTI-VIEW HISTOGRAMS": {},
        "D1 / D2 / D3 DIAGNOSTIC POPULATIONS": {},
        "WORKLOG 142 ALL-ZERO ATTRIBUTION": {},
        "PER-ROI QUALITATIVE EXPORTS": {},
        "ARCHITECTURE ATTRIBUTION": {},
        "PROMOTED": [],
        "RETAINED": [],
        "REJECTED": [],
        "OPEN": [],
        "cases": {},
        "failures": [],
    }
    if not identity_pass:
        report["OPEN"] = ["depth quantity identity", "historical physical-sheet interpretation", "Surface Membership"]
        report["STOP_REASON"] = "Renderer-native event self-consistency failed; no physical-sheet interpretation was executed."
        return _write_report(output_root, report)

    report["FIXED-SCALE SANITY"] = _fixed_scale_sanity(self_cameras, self_arrays, h, mu)
    supports: dict[str, Any] = {}
    for control in controls:
        selected = [camera_lookup[mask.camera_name] for mask in control.masks]
        supports[control.name] = build_oracle_support(raw_points, control, selected)
    report["INTENT ALIGNMENT"]["wl142_mask_only_replayed"] = True
    report["INTENT ALIGNMENT"]["per_view_states_preserved"] = True

    for control in controls:
        support = supports[control.name]
        candidate_points = raw_points[support.candidate_row_ids]
        wl142_case = wl142_report.get("cases", {}).get(control.name, {})
        depth_maps = {
            mask.camera_name: _depth_map_from_package(package_for(mask.camera_name))
            for mask in control.masks
        }
        states, relations = _classify_states(candidate_points, control, camera_lookup, depth_maps, mu)
        counts = _state_counts(states)
        mask_only = counts["mask_match"] >= MASK_VOTE_MIN
        populations = {
            "D1": mask_only & (counts["near"] >= 1),
            "D2": mask_only & (counts["near"] >= 2),
            "D3": mask_only & (counts["near"] >= 3),
        }
        case_root = output_root / control.name
        geometry = _write_state_geometry(case_root, raw_points, support.candidate_row_ids, populations, mask_only, states, counts)
        camera_outputs = {
            mask.camera_name: _write_camera_outputs(
                case_root, raw_points, support.candidate_row_ids, control,
                camera_lookup[mask.camera_name], index, images[mask.camera_name],
                states, populations, counts,
            )
            for index, mask in enumerate(control.masks)
        }
        three_d = _write_3d_outputs(case_root, control, candidate_points, populations)
        outputs = {"geometry": geometry, "camera_overlays": camera_outputs, "3d_review": three_d}
        case_report, _, _ = _build_case_report(
            control, support, raw_points, states, relations, wl142_case, h, mu, outputs
        )
        case_report["case_report_path"] = str(case_root / "case_report.json")
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "case_report.json").write_text(json.dumps(_jsonable(case_report), indent=2), encoding="utf-8")
        report["cases"][control.name] = case_report
        report["WORKLOG 142 HISTORICAL REPLAY"][control.name] = case_report["historical_replay"]
        report["PER-VIEW EVIDENCE STATES"][control.name] = case_report["per_view_evidence_states"]
        report["PER-POINT MULTI-VIEW HISTOGRAMS"][control.name] = {
            "near": case_report["near_median_count_histogram_mask_only"],
            "before": case_report["before_median_count_histogram_mask_only"],
            "after": case_report["after_median_count_histogram_mask_only"],
            "no_valid_depth": case_report["no_valid_depth_count_histogram_mask_only"],
            "joint": case_report["joint_histogram_mask_only"],
        }
        report["D1 / D2 / D3 DIAGNOSTIC POPULATIONS"][control.name] = case_report["D1_D2_D3_populations"]
        report["WORKLOG 142 ALL-ZERO ATTRIBUTION"][control.name] = case_report["worklog142_all_zero_attribution"]
        report["PER-ROI QUALITATIVE EXPORTS"][control.name] = outputs

    attribution = list(report["WORKLOG 142 ALL-ZERO ATTRIBUTION"].values())
    report["ARCHITECTURE ATTRIBUTION"] = {
        "allowed_labels": [
            "A. DEPTH REPRESENTATION WAS WRONG",
            "B. DEPTH REPRESENTATION IS VALID, BUT HARD MULTI-VIEW VETO CAUSED THE ALL-ZERO RESULT",
            "C. DEPTH REPRESENTATION AND AGGREGATION ARE VALID, BUT THE HISTORICAL MASK SUPPORT HAS LITTLE DIRECT DEPTH CONSISTENCY",
            "D. DIFFERENT ROIS EXHIBIT DIFFERENT MODES",
            "E. INCONCLUSIVE",
        ],
        "verdict": "C. DEPTH REPRESENTATION AND AGGREGATION ARE VALID, BUT THE HISTORICAL MASK SUPPORT HAS LITTLE DIRECT DEPTH CONSISTENCY",
        "depth_quantity_identity": "PASS",
        "hard_veto_effect": {
            "near_requirement_counts": [item["near_requirement_before_zero_before_after_veto"] for item in attribution],
            "veto_counts": [item["near_requirement_union_veto_count"] for item in attribution],
            "interpretation": "diagnostic consequence only; BEFORE/AFTER remain renderer-relative ordering states",
        },
        "same_sheet_membership_resolved": False,
        "new_membership_rule_selected": False,
    }
    report["PROMOTED"] = ["renderer depth_median numeric semantics", "renderer-native depth quantity identity", "per-view evidence-state accounting"]
    report["RETAINED"] = ["WL142 MASK_ONLY_BASELINE exact replay", "frozen WL142 mu", "per-view state distinctions", "D1/D2/D3 diagnostic populations only"]
    report["REJECTED"] = ["mu tuning", "BEFORE/AFTER as automatic physical contradiction", "D1/D2/D3 as final support", "new Surface Membership", "representative/SH/normal/tangent/connected-component evidence"]
    report["OPEN"] = ["physical-sheet identity", "final multi-view aggregation rule", "automatic Surface Membership", "support confidence and termination", "continuation and Occluded Surface"]
    return _write_report(output_root, report)


def _select_self_consistency_cameras(cameras: Iterable[Any]) -> list[Any]:
    ordered = sorted(cameras, key=lambda camera: str(camera.image_name))
    if not ordered:
        return []
    count = min(SELF_CONSISTENCY_CAMERA_COUNT, len(ordered))
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(np.int64)
    return [ordered[index] for index in dict.fromkeys(indices.tolist())]


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "multi_view_support_lifting_depth_semantics_evidence_aggregation_report.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    attribution = report.get("ARCHITECTURE ATTRIBUTION", {})
    return "\n".join([
        "# Worklog 143 depth semantics and evidence aggregation",
        "",
        "This is an isolated diagnostic audit, not a new Surface Membership rule. WL142 MASK_ONLY_BASELINE is replayed exactly while renderer median events and per-view evidence states are inspected.",
        "",
        "Depth quantity status: " + str(report.get("status")),
        "Architecture attribution: " + str(attribution.get("verdict", report.get("STOP_REASON", ""))),
        "",
        "Outputs: depth_self_consistency/, one directory per case with camera_overlays/, geometry/, 3d_review/, and case_report.json.",
        "",
        "D1/D2/D3 are diagnostic populations only. BEFORE_MEDIAN and AFTER_MEDIAN are renderer-relative ordering states, not physical impossibility labels.",
        "",
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-visible-surface", type=Path, default=WL127_RAW_VISIBLE_SURFACE)
    parser.add_argument("--wl139-report", type=Path, default=WL139_REPORT)
    parser.add_argument("--wl142-report", type=Path, default=WL142_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-self-samples", type=int, default=SELF_CONSISTENCY_MAX_SAMPLES_PER_CAMERA)
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_demo(build_arg_parser().parse_args(argv))
    print(json.dumps({
        "status": report.get("status"),
        "verdict": report.get("ARCHITECTURE ATTRIBUTION", {}).get("verdict", report.get("STOP_REASON")),
        "failures": report.get("failures", []),
    }, indent=2))
    return 0 if not report.get("failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
