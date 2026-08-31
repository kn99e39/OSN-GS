"""Worklog 141: oracle single-surface support and appearance attribution.

This is an isolated, non-canonical evaluation path.  It deliberately stops
short of automatic Surface Membership.  Stage A uses manually frozen
image-space masks to create an evaluation-only oracle support from the WL127
raw Visible Surface rows, then applies the unchanged WL139 graphness audit and
physical-chart representative to baseline and oracle populations.  Stage B
only becomes eligible after qualitative Stage-A review and only when valid
renderer-native Gaussian provenance is available.

The current WL127 PLY contains geometry and appearance fields but no primitive
or contributor identity.  The default run therefore performs the Stage-A
support attribution and provenance audit, and fail-closed skips SH scoring.
No nearest-Gaussian proxy is silently substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.meeting_occluded_surface_feasibility import (  # noqa: E402
    Box,
    _grid_faces,
    _write_ply,
)
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402
    GRAPH_DEGREE_U,
    GRAPH_DEGREE_V,
    GRAPH_RESOLUTION_U,
    GRAPH_RESOLUTION_V,
    GRAPH_SMOOTHNESS_LAMBDA,
    GRAPH_TIKHONOV_LAMBDA,
    RepresentativeCaseConfig,
)

# The conditional expression above cannot be used in an import list.  Keep the
# actual helper imports explicit below so this file remains easy to audit.
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402,E501
    _configure_axis,
    _normal_field_accounting,
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
from devtools.demo.real_gaussian_scene_surface_validation import (  # noqa: E402,E501
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


OUTPUT_ROOT = REPO_ROOT / "output" / "oracle_single_surface_support_appearance_evidence"
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
    / "scale_separated_visible_surface_representative"
    / "scale_separated_visible_surface_representative_report.json"
)

# These are the exact inherited WL139 applicability/fitter constants.  They
# are copied into the report as an audit aid; no WL139 implementation is
# changed by this module.
GRAPH_BIN_SCALE_H = 4.0
GRAPH_MODE_SEPARATION_H = 3.0
GRAPH_MODE_MIN_SIDE_POINTS = 3
GRAPH_MIN_ELIGIBLE_BINS = 8
GRAPH_MAX_MULTIMODE_FRACTION = 0.10

SAMPLE_COUNT_U = 96
SAMPLE_COUNT_V = 40
DISPLAY_POINT_ALPHA = 0.99
DISPLAY_POINT_SIZE = 3.6
DISPLAY_SURFACE_ALPHA = 0.68
REVIEW_CAMERA_COUNT = 3
MIN_ORACLE_SUPPORT_POINTS = 64
MIN_ORACLE_VOTES = 2

BASELINE_GREY = (0.34, 0.37, 0.42)
ORACLE_GREEN = (0.06, 0.72, 0.26)
BASELINE_REPRESENTATIVE_ORANGE = (0.94, 0.38, 0.04)
ORACLE_REPRESENTATIVE_CYAN = (0.00, 0.68, 0.88)
UNSUPPORTED_RED = (0.86, 0.12, 0.10)
SUPPORT_BOUNDARY_YELLOW = (0.98, 0.73, 0.03)
NORMAL_BLUE = (0.10, 0.28, 0.86)


@dataclass(frozen=True)
class ManualCameraMask:
    """A frozen, hand-authored polygon in the 648x420 review image frame."""

    camera_name: str
    polygon_pixels_648x420: tuple[tuple[float, float], ...]
    note: str

    def polygon_for(self, camera: Any) -> np.ndarray:
        width = max(float(getattr(camera, "image_width", 648)), 1.0)
        height = max(float(getattr(camera, "image_height", 420)), 1.0)
        base = np.asarray(self.polygon_pixels_648x420, dtype=np.float64)
        return base * np.asarray([width / 648.0, height / 420.0], dtype=np.float64)

    def as_json(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "coordinate_frame": "review image pixels normalized from 648x420",
            "polygon_pixels_648x420": [list(point) for point in self.polygon_pixels_648x420],
            "note": self.note,
        }


@dataclass(frozen=True)
class OracleSurfaceControl:
    """Candidate crop plus masks; neither field is inferred from a fit."""

    name: str
    surface_label: str
    config: RepresentativeCaseConfig
    masks: tuple[ManualCameraMask, ...]
    selection_basis: str
    min_mask_votes: int = MIN_ORACLE_VOTES

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "surface_label": self.surface_label,
            "candidate_spatial_crop": self.config.as_json(),
            "selection_basis": self.selection_basis,
            "min_mask_votes": int(self.min_mask_votes),
            "masks": [mask.as_json() for mask in self.masks],
            "oracle_truth_warning": (
                "manual evaluation-only support for this architecture audit; "
                "not proposed automatic membership"
            ),
        }


@dataclass
class FrozenOracleSupport:
    control: OracleSurfaceControl
    candidate_row_ids: np.ndarray
    oracle_row_ids: np.ndarray
    mask_vote_counts: np.ndarray
    per_camera: list[dict[str, Any]]


@dataclass
class ArmResult:
    name: str
    points: np.ndarray
    row_ids: np.ndarray
    graphness: Any
    graph_report: dict[str, Any]
    representative: Any | None
    topology: dict[str, Any] | None
    contract: dict[str, Any] | None
    support_domain: dict[str, Any] | None


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


def _sha256_rows(points: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def _point_in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd polygon test with inclusive bounding-box margin."""

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    vertices = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(vertices) < 3:
        return np.zeros(len(points), dtype=bool)
    inside = np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    x_min, y_min = np.min(vertices, axis=0) - 1.0e-9
    x_max, y_max = np.max(vertices, axis=0) + 1.0e-9
    candidate = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    for index in range(len(vertices)):
        first = vertices[index]
        second = vertices[(index + 1) % len(vertices)]
        x1, y1 = first
        x2, y2 = second
        non_horizontal = np.abs(y2 - y1) > 1.0e-12
        if not non_horizontal:
            continue
        crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        inside ^= crosses
    return inside & candidate


def _draw_polygon_outline(image: Any, mask: ManualCameraMask, camera: Any, *, color: tuple[int, int, int] = SUPPORT_BOUNDARY_YELLOW) -> Any:
    from PIL import Image, ImageDraw

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    polygon = mask.polygon_for(camera)
    xy = [tuple(map(float, point)) for point in polygon.tolist()]
    xy.append(xy[0])
    draw.line(xy, fill=tuple(int(round(channel)) for channel in color), width=3, joint="curve")
    return output


def _mask_camera_lookup(cameras: Iterable[Any]) -> dict[str, Any]:
    lookup = {str(camera.image_name): camera for camera in cameras}
    return lookup


def build_oracle_support(
    raw_points: np.ndarray,
    controls: OracleSurfaceControl,
    cameras: Iterable[Any],
) -> FrozenOracleSupport:
    """Select WL127 row IDs using only a candidate crop and frozen masks."""

    raw_points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 3)
    camera_lookup = _mask_camera_lookup(cameras)
    candidate_mask = controls.config.roi_box.contains(raw_points)
    candidate_ids = np.flatnonzero(candidate_mask).astype(np.int64)
    candidate_points = raw_points[candidate_ids]
    votes = np.zeros(len(candidate_ids), dtype=np.int16)
    per_camera: list[dict[str, Any]] = []
    for mask in controls.masks:
        if mask.camera_name not in camera_lookup:
            raise KeyError(f"manual oracle mask camera is unavailable: {mask.camera_name}")
        camera = camera_lookup[mask.camera_name]
        projection = project_world_points(candidate_points, camera)
        projected_xy = np.column_stack([projection["x"], projection["y"]])
        inside = projection["valid"] & _point_in_polygon(projected_xy, mask.polygon_for(camera))
        votes += inside.astype(np.int16)
        per_camera.append({
            "camera_name": mask.camera_name,
            "polygon": mask.as_json(),
            "candidate_points_projected": int(np.sum(projection["valid"])),
            "candidate_points_inside_mask": int(np.sum(inside)),
            "inside_fraction_of_candidate": float(np.mean(inside)) if len(inside) else 0.0,
            "projected_bbox_pixels": (
                [
                    float(np.min(projection["x"][projection["valid"]])),
                    float(np.min(projection["y"][projection["valid"]])),
                    float(np.max(projection["x"][projection["valid"]])),
                    float(np.max(projection["y"][projection["valid"]])),
                ]
                if np.any(projection["valid"])
                else None
            ),
        })
    oracle_ids = candidate_ids[votes >= int(controls.min_mask_votes)]
    return FrozenOracleSupport(
        control=controls,
        candidate_row_ids=candidate_ids,
        oracle_row_ids=oracle_ids.astype(np.int64),
        mask_vote_counts=votes,
        per_camera=per_camera,
    )


def support_alignment_report(
    support: FrozenOracleSupport,
    raw_points: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(raw_points, dtype=np.float64)[support.candidate_row_ids]
    oracle = np.asarray(raw_points, dtype=np.float64)[support.oracle_row_ids]
    vote_counts = support.mask_vote_counts
    vote_histogram = {
        str(vote): int(np.sum(vote_counts == vote))
        for vote in range(len(support.control.masks) + 1)
    }
    # This is only a mechanical pre-fit gate.  It cannot certify semantic
    # surface identity, which remains a human image-alignment decision.
    mechanical_pass = bool(
        len(oracle) >= MIN_ORACLE_SUPPORT_POINTS
        and len(support.control.masks) >= 2
        and support.control.min_mask_votes >= 2
        and all(item["candidate_points_inside_mask"] > 0 for item in support.per_camera)
    )
    return {
        "status": "SUPPORT_ALIGNMENT_PASS" if mechanical_pass else "SUPPORT_ALIGNMENT_FAIL",
        "human_review_status": "PENDING_IMAGE_ALIGNMENT_REVIEW",
        "human_must_verify_same_physical_surface": True,
        "mechanical_gate_does_not_prove_surface_identity": True,
        "candidate_row_count": int(len(candidate)),
        "oracle_row_count": int(len(oracle)),
        "oracle_fraction_of_candidate": float(len(oracle) / max(len(candidate), 1)),
        "candidate_row_ids_sha256": hashlib.sha256(support.candidate_row_ids.tobytes()).hexdigest(),
        "oracle_row_ids_sha256": hashlib.sha256(support.oracle_row_ids.tobytes()).hexdigest(),
        "oracle_xyz_sha256": _sha256_rows(oracle),
        "vote_histogram": vote_histogram,
        "per_camera": support.per_camera,
        "candidate_centroid": np.mean(candidate, axis=0) if len(candidate) else None,
        "oracle_centroid": np.mean(oracle, axis=0) if len(oracle) else None,
    }


def _unsupported_components(supported: np.ndarray) -> tuple[int, int]:
    supported = np.asarray(supported, dtype=bool)
    remaining = {tuple(map(int, index)) for index in np.argwhere(~supported)}
    components = 0
    largest = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        size = 0
        while stack:
            i, j = stack.pop()
            size += 1
            for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        largest = max(largest, size)
    return components, largest


def _quad_areas(grid: np.ndarray) -> np.ndarray:
    first = grid[:-1, :-1]
    second = grid[1:, :-1]
    third = grid[:-1, 1:]
    fourth = grid[1:, 1:]
    return 0.5 * (
        np.linalg.norm(np.cross(second - first, third - first), axis=2)
        + np.linalg.norm(np.cross(fourth - second, third - second), axis=2)
    )


def support_domain_diagnostic(
    oracle_points: np.ndarray,
    representative: Any,
    config: RepresentativeCaseConfig,
) -> dict[str, Any]:
    """Annotate support coverage without trimming or changing the fit."""

    coordinates = np.asarray(oracle_points, dtype=np.float64) @ np.stack(
        [_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1
    )
    u0, u1 = representative.domain_u
    v0, v1 = representative.domain_v
    uv = np.column_stack([
        (coordinates[:, 0] - u0) / max(u1 - u0, 1.0e-12),
        (coordinates[:, 1] - v0) / max(v1 - v0, 1.0e-12),
    ])
    supported = np.zeros((SAMPLE_COUNT_U, SAMPLE_COUNT_V), dtype=bool)
    valid = np.all((uv >= -1.0e-6) & (uv <= 1.0 + 1.0e-6), axis=1)
    indices = np.floor(np.clip(uv[valid], 0.0, 1.0 - 1.0e-12) * np.asarray([SAMPLE_COUNT_U, SAMPLE_COUNT_V])).astype(np.int64)
    supported[indices[:, 0], indices[:, 1]] = True
    grid = np.asarray(representative.sampled_points, dtype=np.float64).reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    areas = _quad_areas(grid)
    cell_supported = (
        supported[:-1, :-1]
        | supported[1:, :-1]
        | supported[:-1, 1:]
        | supported[1:, 1:]
    )
    components, largest = _unsupported_components(supported)
    supported_area = float(np.sum(areas[cell_supported]))
    unsupported_area = float(np.sum(areas[~cell_supported]))
    return {
        "fitted_geometry_unchanged": True,
        "support_annotation_only": True,
        "supported_chart_fraction": float(np.mean(supported)),
        "unsupported_chart_fraction": float(np.mean(~supported)),
        "supported_chart_vertices": int(np.sum(supported)),
        "unsupported_chart_vertices": int(np.sum(~supported)),
        "unsupported_connected_region_count": int(components),
        "largest_unsupported_connected_region_vertices": int(largest),
        "representative_area_over_supported_cells": supported_area,
        "representative_area_over_unsupported_cells": unsupported_area,
        "supported_vertex_mask": supported,
        "cell_supported_mask": cell_supported,
    }


def _fit_arm(
    name: str,
    points: np.ndarray,
    row_ids: np.ndarray,
    config: RepresentativeCaseConfig,
    h: float,
    *,
    max_fit_points: int,
    device_name: str,
) -> ArmResult:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    row_ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    graphness = audit_raw_graphness(points, config, h)
    graph_report = graphness_report(graphness, h)
    representative = None
    topology = None
    contract = None
    support_domain = None
    if graphness.status == "PASS_GRAPH_LIKE":
        representative = fit_physical_chart_surface(
            points,
            config,
            role="full_evaluation_only",
            max_fit_points=int(max_fit_points),
            device_name=device_name,
        )
        topology = audit_representative_topology(
            representative.surface,
            config,
            domain_u=representative.domain_u,
            domain_v=representative.domain_v,
            h=h,
        )
        contract = representative_contract(
            points,
            representative.sampled_points,
            representative.sampled_normals,
            representative.fitting_residuals,
            topology,
            h,
        )
    return ArmResult(name, points, row_ids, graphness, graph_report, representative, topology, contract, support_domain)


def _arm_report(arm: ArmResult) -> dict[str, Any]:
    return {
        "arm": arm.name,
        "point_count": int(len(arm.points)),
        "row_ids_sha256": hashlib.sha256(arm.row_ids.tobytes()).hexdigest(),
        "xyz_sha256": _sha256_rows(arm.points),
        "graphness": arm.graph_report,
        "representative_attempted": arm.representative is not None,
        "representative_role": arm.representative.role if arm.representative is not None else None,
        "representative_contract": arm.contract,
        "topology_contract": arm.topology,
        "support_domain": arm.support_domain,
    }


def _write_support_geometry(case_root: Path, support: FrozenOracleSupport, baseline: ArmResult, oracle: ArmResult) -> dict[str, str]:
    root = case_root / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "baseline_raw": root / "baseline_spatial_population.ply",
        "oracle_raw": root / "oracle_single_surface_support.ply",
        "oracle_support_selection": root / "oracle_support_selection.npz",
    }
    _write_ply(paths["baseline_raw"], baseline.points, color=(104, 111, 124))
    _write_ply(paths["oracle_raw"], oracle.points, color=(18, 190, 68))
    np.savez_compressed(
        paths["oracle_support_selection"],
        candidate_row_ids=support.candidate_row_ids,
        oracle_row_ids=support.oracle_row_ids,
        candidate_vote_counts=support.mask_vote_counts,
    )
    if baseline.representative is not None:
        path = root / "baseline_wl139_representative.ply"
        _write_ply(path, baseline.representative.sampled_points, faces=_grid_faces(SAMPLE_COUNT_U, SAMPLE_COUNT_V), color=(240, 96, 14))
        paths["baseline_representative"] = path
    if oracle.representative is not None:
        path = root / "oracle_wl139_representative.ply"
        _write_ply(path, oracle.representative.sampled_points, faces=_grid_faces(SAMPLE_COUNT_U, SAMPLE_COUNT_V), color=(0, 186, 224))
        paths["oracle_representative"] = path
        if oracle.support_domain is not None:
            mask = np.asarray(oracle.support_domain["supported_vertex_mask"], dtype=bool).reshape(-1)
            supported_path = root / "oracle_representative_supported_chart_vertices.ply"
            unsupported_path = root / "oracle_representative_unsupported_chart_vertices.ply"
            _write_ply(supported_path, oracle.representative.sampled_points[mask], color=(24, 190, 72))
            _write_ply(unsupported_path, oracle.representative.sampled_points[~mask], color=(222, 54, 40))
            paths["oracle_representative_supported_chart_vertices"] = supported_path
            paths["oracle_representative_unsupported_chart_vertices"] = unsupported_path
    return {key: str(path) for key, path in paths.items()}


def _write_3d_review(case_root: Path, baseline: ArmResult, oracle: ArmResult, config: RepresentativeCaseConfig) -> dict[str, str]:
    root = case_root / "3d_review"
    root.mkdir(parents=True, exist_ok=True)
    parts = [baseline.points, oracle.points]
    for arm in (baseline, oracle):
        if arm.representative is not None:
            parts.append(arm.representative.sampled_points)
    limits = np.concatenate(parts, axis=0)
    paths = {
        "baseline_raw": root / "baseline_spatial_population.png",
        "oracle_raw": root / "oracle_single_surface_support.png",
        "baseline_vs_oracle_raw": root / "baseline_vs_oracle_raw.png",
        "oracle_representative": root / "oracle_representative.png",
        "oracle_support_domain": root / "oracle_support_domain_annotation.png",
        "oracle_representative_normals": root / "oracle_representative_normals.png",
    }
    _save_3d(paths["baseline_raw"], "Baseline spatial population", config, limits, lambda axis: _plot_points(axis, baseline.points, config, BASELINE_GREY, label="baseline raw", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE))
    _save_3d(paths["oracle_raw"], "Oracle single-surface support", config, limits, lambda axis: _plot_points(axis, oracle.points, config, ORACLE_GREEN, label="oracle raw support", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE))
    _save_3d(
        paths["baseline_vs_oracle_raw"],
        "Baseline vs oracle raw support",
        config,
        limits,
        lambda axis: (
            _plot_points(axis, baseline.points, config, BASELINE_GREY, label="baseline spatial population", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE),
            _plot_points(axis, oracle.points, config, ORACLE_GREEN, label="oracle single-surface support", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE),
        ),
    )
    if oracle.representative is not None:
        grid = oracle.representative.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
        _save_3d(
            paths["oracle_representative"],
            "Oracle support + frozen WL139 representative",
            config,
            limits,
            lambda axis: (
                _plot_points(axis, oracle.points, config, ORACLE_GREEN, label="oracle raw support", alpha=DISPLAY_POINT_ALPHA, size=DISPLAY_POINT_SIZE),
                _plot_surface(axis, grid, config, ORACLE_REPRESENTATIVE_CYAN, label="WL139 representative", alpha=DISPLAY_SURFACE_ALPHA),
            ),
        )
        mask = np.asarray(oracle.support_domain["supported_vertex_mask"], dtype=bool)
        supported_points = oracle.representative.sampled_points[mask.reshape(-1)]
        unsupported_points = oracle.representative.sampled_points[~mask.reshape(-1)]
        _save_3d(
            paths["oracle_support_domain"],
            "Representative full domain: supported vs unsupported chart vertices",
            config,
            limits,
            lambda axis: (
                _plot_points(axis, supported_points, config, ORACLE_GREEN, label="supported chart vertices", alpha=1.0, size=4.2),
                _plot_points(axis, unsupported_points, config, UNSUPPORTED_RED, label="unsupported chart vertices", alpha=1.0, size=4.2),
            ),
        )
        indices = np.arange(0, SAMPLE_COUNT_U * SAMPLE_COUNT_V, 12, dtype=np.int64)
        points = oracle.representative.sampled_points[indices]
        normals = oracle.representative.sampled_normals[indices]
        basis = np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1)
        local_points = points @ basis
        local_normals = normals @ basis
        _save_3d(
            paths["oracle_representative_normals"],
            "Oracle representative analytic normals",
            config,
            limits,
            lambda axis: (
                _plot_surface(axis, grid, config, ORACLE_REPRESENTATIVE_CYAN, label="WL139 representative", alpha=0.45),
                axis.quiver(local_points[:, 0], local_points[:, 1], local_points[:, 2], local_normals[:, 0], local_normals[:, 1], local_normals[:, 2], length=0.10, normalize=True, color=NORMAL_BLUE, linewidth=0.75, label="analytic normals"),
            ),
        )
    else:
        for key in ("oracle_representative", "oracle_support_domain", "oracle_representative_normals"):
            paths[key].write_text("not exported: oracle graphness did not pass\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}



def _write_prefit_alignment_outputs(
    case_root: Path,
    control: OracleSurfaceControl,
    oracle_points: np.ndarray,
    camera: Any,
    scene_image: Any,
) -> dict[str, str]:
    """Export support-alignment views before any representative fit."""

    root = case_root / "camera_overlays" / str(camera.image_name) / "pre_fit_alignment"
    root.mkdir(parents=True, exist_ok=True)
    scene_path = root / "A_gaussian_scene_only.png"
    oracle_path = root / "B_gaussian_plus_oracle_raw_support.png"
    mask_path = root / "C_oracle_mask_outline.png"
    scene_image.convert("RGB").save(scene_path)
    oracle_view = _draw_projected_points(
        scene_image,
        oracle_points,
        camera,
        (18, 194, 76),
        radius=DISPLAY_POINT_SIZE,
        alpha=DISPLAY_POINT_ALPHA,
    )
    oracle_view.save(oracle_path)
    mask = next(item for item in control.masks if item.camera_name == str(camera.image_name))
    _draw_polygon_outline(oracle_view, mask, camera).save(mask_path)
    return {
        "A_gaussian_scene_only": str(scene_path),
        "B_gaussian_plus_oracle_raw_support": str(oracle_path),
        "C_oracle_mask_outline": str(mask_path),
    }
def _write_camera_outputs(
    case_root: Path,
    control: OracleSurfaceControl,
    baseline: ArmResult,
    oracle: ArmResult,
    camera: Any,
    scene_image: Any,
) -> dict[str, str]:
    from PIL import Image

    root = case_root / "camera_overlays" / str(camera.image_name)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    outputs["A_gaussian_scene_only"] = root / "A_gaussian_scene_only.png"
    outputs["B_gaussian_plus_baseline_raw"] = root / "B_gaussian_plus_baseline_raw.png"
    outputs["D_gaussian_plus_oracle_raw"] = root / "D_gaussian_plus_oracle_raw.png"
    outputs["F_oracle_mask_outline"] = root / "F_oracle_mask_outline.png"
    scene_image.convert("RGB").save(outputs["A_gaussian_scene_only"])
    baseline_raw = _draw_projected_points(scene_image, baseline.points, camera, (184, 112, 36), radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    baseline_raw.save(outputs["B_gaussian_plus_baseline_raw"])
    if baseline.representative is not None:
        baseline_rep = _draw_projected_points(scene_image, baseline.representative.sampled_points, camera, (242, 104, 17), radius=2.2, alpha=DISPLAY_POINT_ALPHA)
        baseline_rep_path = root / "C_gaussian_plus_baseline_representative.png"
        baseline_rep.save(baseline_rep_path)
        outputs["C_gaussian_plus_baseline_representative"] = baseline_rep_path
    else:
        baseline_rep_path = root / "C_baseline_representative_NOT_EXPORTED.txt"
        baseline_rep_path.write_text("not exported: baseline graphness did not pass\n", encoding="utf-8")
        outputs["C_baseline_representative_not_exported"] = baseline_rep_path
    oracle_raw = _draw_projected_points(scene_image, oracle.points, camera, (18, 194, 76), radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    oracle_raw.save(outputs["D_gaussian_plus_oracle_raw"])
    if oracle.representative is not None:
        oracle_rep = _draw_projected_points(scene_image, oracle.representative.sampled_points, camera, (0, 191, 225), radius=2.4, alpha=DISPLAY_POINT_ALPHA)
        oracle_rep_path = root / "E_gaussian_plus_oracle_representative.png"
        oracle_rep.save(oracle_rep_path)
        outputs["E_gaussian_plus_oracle_representative"] = oracle_rep_path
    else:
        oracle_rep_path = root / "E_oracle_representative_NOT_EXPORTED.txt"
        oracle_rep_path.write_text("not exported: oracle graphness did not pass\n", encoding="utf-8")
        outputs["E_oracle_representative_not_exported"] = oracle_rep_path
    mask = next(item for item in control.masks if item.camera_name == str(camera.image_name))
    _draw_polygon_outline(oracle_raw, mask, camera).save(outputs["F_oracle_mask_outline"])
    # Keep a clean neutral background geometry-only view for direct inspection.
    geometry_only = Image.new("RGB", scene_image.size, (248, 248, 248))
    geometry_only = _draw_projected_points(geometry_only, baseline.points, camera, (142, 143, 148), radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    geometry_only = _draw_projected_points(geometry_only, oracle.points, camera, (18, 194, 76), radius=DISPLAY_POINT_SIZE, alpha=DISPLAY_POINT_ALPHA)
    geometry_only.save(root / "G_baseline_vs_oracle_raw_geometry_only.png")
    outputs["G_baseline_vs_oracle_raw_geometry_only"] = root / "G_baseline_vs_oracle_raw_geometry_only.png"
    return {key: str(path) for key, path in outputs.items()}


def _read_ply_property_names(path: Path) -> list[str]:
    properties: list[str] = []
    with Path(path).open("rb") as handle:
        for _ in range(4096):
            line = handle.readline()
            if not line:
                break
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded == "end_header":
                break
            fields = decoded.split()
            if len(fields) >= 3 and fields[0] == "property" and fields[1] != "list":
                properties.append(fields[2])
    return properties


def audit_gaussian_provenance(raw_path: Path, checkpoint: Path) -> dict[str, Any]:
    """Audit, but never invent, raw-row to Gaussian-primitive identity."""

    properties = _read_ply_property_names(raw_path)
    lower = {name.lower() for name in properties}
    direct_names = sorted(name for name in properties if any(token in name.lower() for token in ("primitive_id", "gaussian_id", "surfel_id")))
    contributor_names = sorted(name for name in properties if any(token in name.lower() for token in ("contributor", "event_id", "source_id", "weight")))
    if direct_names:
        status = "DIRECT_RENDERER_PROVENANCE_AVAILABLE"
    elif contributor_names and ("f_dc_0" not in lower or len(contributor_names) > 1):
        status = "CONTRIBUTOR_AGGREGATE_PROVENANCE_AVAILABLE"
    else:
        status = "NO_VALID_PRIMITIVE_PROVENANCE"
    return {
        "status": status,
        "raw_visible_surface_ply": str(Path(raw_path).resolve()),
        "raw_visible_surface_property_names": properties,
        "candidate_direct_identity_fields": direct_names,
        "candidate_contributor_fields": contributor_names,
        "checkpoint": str(Path(checkpoint).resolve()),
        "nearest_primitive_proxy_used": False,
        "valid_for_sh_evaluation": status != "NO_VALID_PRIMITIVE_PROVENANCE",
        "interpretation": (
            "WL127 PLY has geometry and f_dc fields but no renderer event, "
            "primitive, or contributor identity; SH membership evidence is not claimed"
            if status == "NO_VALID_PRIMITIVE_PROVENANCE"
            else "explicit provenance fields require a separate validated reader before SH scoring"
        ),
    }


def _manual_controls() -> tuple[OracleSurfaceControl, ...]:
    """Return the frozen three-surface control set, before any fit is built."""

    tabletop = RepresentativeCaseConfig(
        name="tabletop_top_oracle",
        semantic_label="candidate tabletop upper sheet; image masks define oracle support",
        roi_box=Box((-11.0, 1.0, 2.0), (-7.0, 1.35, 4.0)),
        u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 0.0, 1.0), n_axis=(0.0, 1.0, 0.0),
        u_bounds=(-11.0, -7.0), v_bounds=(2.0, 4.0), n_bounds=(1.0, 1.35), u_cut=-7.0,
        frontier_source="WL141 fixed candidate crop only; semantic support is from frozen camera masks",
    )
    curved_rim = RepresentativeCaseConfig(
        name="curved_table_rim_oracle",
        semantic_label="candidate curved table side / rim; image masks define oracle support",
        roi_box=Box((-0.70, 1.10, 0.35), (1.30, 1.65, 1.70)),
        u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 0.0, 1.0), n_axis=(0.0, 1.0, 0.0),
        u_bounds=(-0.70, 1.30), v_bounds=(0.35, 1.70), n_bounds=(1.10, 1.65), u_cut=1.30,
        frontier_source="WL141 fixed candidate crop only; semantic support is from frozen camera masks",
    )
    ground = RepresentativeCaseConfig(
        name="paver_ground_oracle",
        semantic_label="candidate real ground / paver sheet; image masks define oracle support",
        roi_box=Box((-1.0, 1.5, -0.15), (1.0, 2.5, 0.15)),
        u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0), n_axis=(0.0, 0.0, 1.0),
        u_bounds=(-1.0, 1.0), v_bounds=(1.5, 2.5), n_bounds=(-0.15, 0.15), u_cut=1.0,
        frontier_source="WL141 fixed candidate crop only; semantic support is from frozen camera masks",
    )
    return (
        OracleSurfaceControl(
            "tabletop_top_oracle",
            "TABLETOP_TOP_CANDIDATE",
            tabletop,
            (
                ManualCameraMask("DSC08050.JPG", ((48, 325), (158, 322), (164, 358), (57, 371)), "manual central polygon in fixed review render"),
                ManualCameraMask("DSC08045.JPG", ((553, 292), (634, 288), (637, 340), (565, 356)), "manual central polygon in fixed review render"),
                ManualCameraMask("DSC08017.JPG", ((267, 335), (345, 332), (348, 375), (271, 384)), "manual central polygon in fixed review render"),
            ),
            "candidate crop is a WL140 seed only; support truth is the manually frozen same-surface polygon set",
        ),
        OracleSurfaceControl(
            "curved_table_rim_oracle",
            "CURVED_TABLE_RIM_CANDIDATE",
            curved_rim,
            (
                ManualCameraMask("DSC08043.JPG", ((250, 151), (474, 140), (459, 274), (262, 285)), "manual central rim polygon in fixed review render"),
                ManualCameraMask("DSC07960.JPG", ((205, 216), (387, 207), (391, 259), (215, 271)), "manual central rim polygon in fixed review render"),
                ManualCameraMask("DSC08003.JPG", ((217, 176), (432, 167), (427, 278), (220, 289)), "manual central rim polygon in fixed review render"),
            ),
            "candidate crop is the WL140 primary seed only; support truth is the manually frozen same-surface polygon set",
        ),
        OracleSurfaceControl(
            "paver_ground_oracle",
            "GROUND_PAVER_CANDIDATE",
            ground,
            (
                ManualCameraMask("DSC08081.JPG", ((155, 91), (305, 87), (301, 158), (160, 166)), "manual central paver polygon in fixed review render"),
                ManualCameraMask("DSC07968.JPG", ((369, 199), (468, 198), (466, 220), (371, 222)), "manual central paver polygon in fixed review render"),
                ManualCameraMask("DSC07960.JPG", ((326, 176), (470, 173), (476, 205), (331, 208)), "manual central paver polygon in fixed review render"),
            ),
            "candidate crop is a WL140 ground seed only; support truth is the manually frozen same-surface polygon set",
        ),
    )


def _frozen_manifest(
    controls: tuple[OracleSurfaceControl, ...],
    supports: dict[str, FrozenOracleSupport],
    raw_path: Path,
    checkpoint: Path,
    h: float,
    mu: float,
) -> dict[str, Any]:
    return {
        "batch": "Worklog 141 oracle single-surface support and renderer-native appearance evidence attribution",
        "frozen_before_any_representative_fit": True,
        "raw_visible_surface": str(raw_path.resolve()),
        "raw_visible_surface_sha256": _sha256_file(raw_path),
        "checkpoint": str(checkpoint.resolve()),
        "h": h,
        "mu": mu,
        "selection_rule": "candidate spatial crop AND at least 2 of 3 fixed image-space masks",
        "selection_inputs": [
            "WL127 raw Visible Surface row IDs and XYZ",
            "manually frozen camera IDs",
            "manually frozen image-space polygons",
            "canonical row-convention camera projection",
        ],
        "selection_excludes": [
            "graphness",
            "representative fit or residual",
            "representative quality",
            "SH/color",
            "continuation",
            "Candidate B",
            "VLM/SAM/object segmentation",
        ],
        "manual_choices": [
            "three candidate physical surfaces",
            "three camera IDs and polygons per surface",
            "candidate spatial crops",
            "minimum two mask votes",
        ],
        "full_reference_leakage_disclosure": (
            "WL127 full raw XYZ is used to define the candidate crop and to evaluate/freeze row IDs; "
            "the support masks are fixed image-space choices. No fitted representative, graphness, SH, "
            "or output metric influences support selection. This oracle is not a final membership method."
        ),
        "controls": [],
    }


def _replace_manifest_supports(manifest: dict[str, Any], supports: dict[str, FrozenOracleSupport], raw_points: np.ndarray) -> dict[str, Any]:
    controls = []
    for control in _manual_controls():
        support = supports[control.name]
        controls.append({
            **control.as_json(),
            "support_alignment_pre_fit": support_alignment_report(support, raw_points),
            "candidate_row_ids": support.candidate_row_ids.tolist(),
            "oracle_row_ids": support.oracle_row_ids.tolist(),
        })
    manifest["controls"] = controls
    return manifest


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = Path(arguments.raw_visible_surface)
    checkpoint = Path(arguments.checkpoint)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not Path(arguments.wl139_report).exists():
        raise FileNotFoundError(arguments.wl139_report)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    wl139_report = json.loads(Path(arguments.wl139_report).read_text(encoding="utf-8"))
    h = float(wl139_report["inputs"]["h"])
    mu = float(wl139_report["inputs"]["mu"])
    raw_points = _load_xyz_ply(raw_path)
    controls = _manual_controls()

    model, payload, cameras, scene_info = _load_canonical_scene(
        checkpoint,
        Path(arguments.source_path),
        arguments.device,
        arguments.images,
        arguments.sparse_dir,
        int(arguments.resolution),
        int(arguments.llffhold),
    )
    supports: dict[str, FrozenOracleSupport] = {}
    for control in controls:
        camera_subset = [scene_info_camera for scene_info_camera in cameras if str(scene_info_camera.image_name) in {mask.camera_name for mask in control.masks}]
        supports[control.name] = build_oracle_support(raw_points, control, camera_subset)

    manifest = _frozen_manifest(controls, supports, raw_path, checkpoint, h, mu)
    manifest = _replace_manifest_supports(manifest, supports, raw_points)
    manifest["checkpoint_iteration"] = payload.get("iteration")
    manifest["checkpoint_primitive"] = scene_info["primitive"]
    manifest["camera_meta"] = scene_info["camera_meta"]
    manifest_path = output_root / "frozen_oracle_support_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    # Phase 3 contract: export scene-only and oracle-raw alignment views before
    # any baseline/oracle representative is constructed.
    render_cache: dict[str, Any] = {}
    pre_fit_alignment_outputs: dict[str, Any] = {}
    camera_lookup = _mask_camera_lookup(cameras)
    for control in controls:
        case_root = output_root / control.name
        control_outputs: dict[str, Any] = {}
        for mask in control.masks:
            camera = camera_lookup[mask.camera_name]
            if mask.camera_name not in render_cache:
                render_cache[mask.camera_name] = _render_to_pil(scene_info["rasterizer"].render(camera, model))
            control_outputs[mask.camera_name] = _write_prefit_alignment_outputs(
                case_root,
                control,
                raw_points[supports[control.name].oracle_row_ids],
                camera,
                render_cache[mask.camera_name],
            )
        pre_fit_alignment_outputs[control.name] = control_outputs

    case_reports: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for control in controls:
        case_root = output_root / control.name
        case_root.mkdir(parents=True, exist_ok=True)
        support = supports[control.name]
        baseline_ids = support.candidate_row_ids
        oracle_ids = support.oracle_row_ids
        baseline_points = raw_points[baseline_ids]
        oracle_points = raw_points[oracle_ids]
        alignment = support_alignment_report(support, raw_points)
        try:
            baseline = _fit_arm("BASELINE_SPATIAL_POPULATION", baseline_points, baseline_ids, control.config, h, max_fit_points=int(arguments.max_fit_points), device_name=arguments.device)
            oracle = _fit_arm("ORACLE_SINGLE_SURFACE_SUPPORT", oracle_points, oracle_ids, control.config, h, max_fit_points=int(arguments.max_fit_points), device_name=arguments.device)
            if oracle.representative is not None:
                oracle.support_domain = support_domain_diagnostic(oracle.points, oracle.representative, control.config)
            geometry = _write_support_geometry(case_root, support, baseline, oracle)
            review_3d = _write_3d_review(case_root, baseline, oracle, control.config)
            camera_outputs: dict[str, Any] = {}
            for mask in control.masks:
                camera = camera_lookup[mask.camera_name]
                if mask.camera_name not in render_cache:
                    render_cache[mask.camera_name] = _render_to_pil(scene_info["rasterizer"].render(camera, model))
                camera_outputs[mask.camera_name] = _write_camera_outputs(case_root, control, baseline, oracle, camera, render_cache[mask.camera_name])
            case_report = {
                "surface_control": control.as_json(),
                "support_alignment": alignment,
                "baseline": _arm_report(baseline),
                "oracle": _arm_report(oracle),
                "geometry_outputs": geometry,
                "3d_review_outputs": review_3d,
                "camera_outputs": camera_outputs,
                "pre_fit_alignment_outputs": pre_fit_alignment_outputs.get(control.name, {}),
                "raw_normals_available": False,
                "qualitative_classification": (
                    "INVALID_ORACLE_SUPPORT" if alignment["status"] == "SUPPORT_ALIGNMENT_FAIL"
                    else "PENDING_HUMAN_QUALITATIVE_REVIEW"
                ),
                "qualitative_review_rule": (
                    "human must confirm that A/E and raw support remain attached to one real physical sheet; "
                    "graphness and nearest-distance metrics cannot certify semantics"
                ),
            }
            (case_root / "case_report.json").write_text(json.dumps(_jsonable(case_report), indent=2), encoding="utf-8")
            case_reports[control.name] = case_report
        except Exception as error:
            failures.append({"case": control.name, "error": repr(error)})

    provenance = audit_gaussian_provenance(raw_path, checkpoint)
    human_pending = any(case.get("qualitative_classification") == "PENDING_HUMAN_QUALITATIVE_REVIEW" for case in case_reports.values())
    broad_oracle_representatives = sum(
        bool(case.get("oracle", {}).get("representative_attempted"))
        for case in case_reports.values()
        if case.get("surface_control", {}).get("surface_label") in {"TABLETOP_TOP_CANDIDATE", "CURVED_TABLE_RIM_CANDIDATE", "GROUND_PAVER_CANDIDATE"}
    )
    stage_b = {
        "execution_status": "NOT_EXECUTED_UNTIL_STAGE_A_HUMAN_PASS",
        "provenance_audit": provenance,
        "appearance_signals": [],
        "frozen_membership_edge_dataset": "NOT_CONSTRUCTED",
        "reason": (
            "Stage A qualitative alignment has not been manually promoted in this automated run; "
            "independently, the WL127 raw PLY has no valid primitive/contributor provenance, so no SH "
            "membership evidence is claimed."
        ),
        "nearest_primitive_proxy_used": False,
    }
    if not human_pending and broad_oracle_representatives >= 1 and provenance["valid_for_sh_evaluation"]:
        stage_b["execution_status"] = "IMPLEMENTATION_READY_BUT_EXPLICIT_REVIEW_GATE_REQUIRED"

    report = {
        "batch": "Worklog 141 oracle single-surface support and renderer-native appearance evidence attribution",
        "status": "ISOLATED_NON_CANONICAL_EVALUATION",
        "INTENT ALIGNMENT": {
            "automatic_surface_membership_implemented": False,
            "stage_a_oracle_support": True,
            "stage_b_sh_appearance_scored": False,
            "continuation_executed": False,
            "occluded_surface_executed": False,
        },
        "PHASE 0 WL140 QUALITATIVE CORRECTIONS": [
            "historical_wl139_curved_rim_alignment_control was not table-rim semantic evidence",
            "adjacent_table_side was not accepted as table-side semantic evidence in multi-view overlays",
            "patio_ground_planar was not accepted as reliable ground semantic evidence",
            "the WL140 primary curved_table_rim raw population visibly contained more than one physical trend",
            "graphness PASS is therefore not treated as single-surface membership",
        ],
        "IMPLEMENTATION FIDELITY": {
            "canonical_renderer_modified": False,
            "canonical_checkpoint_modified": False,
            "wl127_geometry_modified": False,
            "wl139_module_modified": False,
            "candidate_b_modified": False,
            "wl139_settings": {
                "resolution_u": GRAPH_RESOLUTION_U,
                "resolution_v": GRAPH_RESOLUTION_V,
                "degree_u": GRAPH_DEGREE_U,
                "degree_v": GRAPH_DEGREE_V,
                "smoothness_lambda": GRAPH_SMOOTHNESS_LAMBDA,
                "tikhonov_lambda": GRAPH_TIKHONOV_LAMBDA,
                "graph_bin_scale_h": GRAPH_BIN_SCALE_H,
                "graph_mode_separation_h": GRAPH_MODE_SEPARATION_H,
                "graph_mode_min_side_points": GRAPH_MODE_MIN_SIDE_POINTS,
                "graph_min_eligible_bins": GRAPH_MIN_ELIGIBLE_BINS,
                "graph_max_multimode_fraction": GRAPH_MAX_MULTIMODE_FRACTION,
            },
            "h": h,
            "mu": mu,
            "oracle_support_frozen_before_fit": True,
            "representative_settings_identical_between_arms": True,
            "raw_display_alpha": DISPLAY_POINT_ALPHA,
            "raw_display_point_size": DISPLAY_POINT_SIZE,
            "display_thinning_changes_metrics": False,
        },
        "STAGE A": {
            "true_physical_surface_control_set": [control.as_json() for control in controls],
            "frozen_support_manifest": str(manifest_path.resolve()),
            "baseline_vs_oracle": case_reports,
            "architecture_gate": (
                "F. MIXED / INCONCLUSIVE — human image alignment and macro-shape review required"
                if human_pending or broad_oracle_representatives == 0
                else "CONDITIONAL_STAGE_A_PASS_PENDING_HUMAN_QUALITATIVE_REVIEW"
            ),
            "support_is_validated_missing_layer": False,
            "qualitative_promotion_not_automatic": True,
        },
        "STAGE B": stage_b,
        "PROMOTED": [],
        "RETAINED": [
            "single-surface support is a distinct candidate architecture layer to evaluate",
            "geometry/geodesic continuity remains a membership backbone candidate",
            "graphness remains a representation-family applicability veto",
            "frozen physical-chart WL139 representative",
        ],
        "REJECTED": [
            "box membership = surface membership",
            "graphness PASS = same physical surface",
            "appearance equality = same surface",
            "nearest Gaussian center used silently as renderer-native provenance",
        ],
        "OPEN": [
            "automatic membership propagation",
            "geometry + appearance fusion rule",
            "SH threshold/confidence",
            "support-domain trimming",
            "non-graph representation",
            "continuation and Occluded Surface",
        ],
        "cases": case_reports,
        "failures": failures,
        "inputs": {
            "raw_visible_surface": str(raw_path.resolve()),
            "raw_visible_surface_sha256": _sha256_file(raw_path),
            "wl139_report": str(Path(arguments.wl139_report).resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "source_path": str(Path(arguments.source_path).resolve()),
            "output_root": str(output_root.resolve()),
        },
    }
    report_path = output_root / "oracle_single_surface_support_appearance_evidence_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    stage_a = report.get("STAGE A", {})
    stage_b = report.get("STAGE B", {})
    return "\n".join([
        "# Worklog 141 oracle single-surface support audit",
        "",
        "이 출력은 자동 Surface Membership가 아니다. WL127 raw row에 대해 고정된 3개 카메라 polygon mask와 2/3 투표로 evaluation-only oracle support를 만들고, 같은 WL139 graphness/fitter를 baseline과 oracle 양쪽에 적용한다.",
        "",
        f"Stage A gate: {stage_a.get('architecture_gate')}",
        f"Stage B: {stage_b.get('execution_status')}",
        "",
        "사람이 먼저 확인할 경로: `frozen_oracle_support_manifest.json`, 각 case의 `camera_overlays/`, `3d_review/`, `case_report.json`.",
        "",
        "WL127 PLY에 primitive/contributor provenance가 없으면 nearest-Gaussian proxy를 사용하지 않고 SH appearance 주장을 하지 않는다.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    report = run_demo(arguments)
    print(json.dumps({"status": report["status"], "stage_a": report["STAGE A"]["architecture_gate"], "stage_b": report["STAGE B"]["execution_status"], "failures": report["failures"]}, indent=2))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
