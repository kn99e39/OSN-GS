"""Worklog 140: qualitative validation on the real trained Gaussian scene.

This is an isolated evaluation path.  It does not create a new representative
or run continuation.  The exact WL139 physical-chart graph representative is
applied, without parameter changes, only after the frozen raw-evidence
graphness gate.  The canonical 2DGS renderer is used read-only for the scene
reference renders; all overlays are written after rendering.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for import_path in (str(REPO_ROOT), str(SCRIPTS_DEVTOOLS)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from devtools.demo.meeting_occluded_surface_feasibility import (  # noqa: E402
    Box,
    _grid_faces,
    _select_box,
    _write_ply,
)
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402
    CURVED_RIM_CASE,
    DISPLAY_POINT_ALPHA,
    DISPLAY_POINT_SIZE,
    GRAPH_DEGREE_U,
    GRAPH_DEGREE_V,
    GRAPH_RESOLUTION_U,
    GRAPH_RESOLUTION_V,
    GRAPH_SMOOTHNESS_LAMBDA,
    GRAPH_TIKHONOV_LAMBDA,
    LEG_CASE,
    RAW_GREY,
    RepresentativeCaseConfig,
    _configure_axis,
    _normal_field_accounting,
    _normalised_axis,
    _plot_points,
    _plot_surface,
    _representative_proximity,
    _save_3d,
    _summary,
    audit_raw_graphness,
    audit_representative_topology,
    fit_physical_chart_surface,
    graphness_report,
    representative_contract,
)
from devtools.demo.scale_separated_visible_surface_representative import FIXED_FITTER_CONFIG  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "output" / "real_gaussian_scene_surface_validation"
WL127_RAW_VISIBLE_SURFACE = (
    REPO_ROOT
    / "output"
    / "confirmed"
    / "127_osn_gs_evidence_bounded_projective_tsdf"
    / "RENDERER_MEDIAN_SURFACE_POINTS"
    / "iteration_0000001"
    / "point_cloud.ply"
)
WL139_REPORT = REPO_ROOT / "output" / "confirmed" / "scale_separated_visible_surface_representative" / "scale_separated_visible_surface_representative_report.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "output" / "arch_2dgs_coverage_first_surface" / "2dgs_run1" / "30000" / "checkpoint.pt"
DEFAULT_SOURCE_PATH = REPO_ROOT / "DATASET"

REVIEW_POINT_ALPHA = 0.97
REVIEW_POINT_SIZE = 3.2
REVIEW_DISPLAY_MAX_POINTS = 18000
REVIEW_CAMERA_COUNT = 3
REVIEW_CAMERA_MIN_SEPARATION = 0.12
REVIEW_CAMERA_MIN_POINTS = 12

# This seed is a demo-only manual choice made from the frozen raw Gaussian
# render/source view. It is recorded in the manifest so that the semantic
# selection is auditable and cannot be mistaken for an error-selected ROI.
CURVED_RIM_PIXEL_SEED = (150, 95, 450, 315)

RAW_OVERLAY_RGB = (255, 211, 0)
REPRESENTATIVE_OVERLAY_RGB = (0, 220, 255)

# This is a post-export human alignment note, not a fitting or selection
# signal.  The historical WL139 coordinates are retained as a control because
# the first raw-scene overlay review placed them on paver/ground rather than
# the intended table rim in DSC08111.JPG.
HISTORICAL_CURVED_RIM_ALIGNMENT_AUDIT = {
    "status": "USER_REVIEW_REQUIRED_ALIGNMENT_MISMATCH",
    "camera_id": "DSC08111.JPG",
    "projected_bbox_pixels": [11.0, 317.0, 134.1, 366.2],
    "manual_observation": (
        "generated raw/representative overlay review placed the historical WL139 "
        "curved-rim ROI on paver/ground; it is not promoted as the WL140 primary"
    ),
    "fit_or_metric_used": False,
}
BOUNDARY_RGB = (255, 75, 40)
NORMAL_RGB = (37, 91, 235)
REPRESENTATIVE_3D_RGB = (0.0, 0.75, 0.90)
BOUNDARY_3D_RGB = (1.0, 0.29, 0.16)
NORMAL_3D_RGB = (0.15, 0.36, 0.92)


@dataclass(frozen=True)
class FrozenReviewRegion:
    config: RepresentativeCaseConfig
    semantic_class: str
    selection_basis: str
    historical_wl139_u_cut: float | None = None

    def as_json(self) -> dict[str, Any]:
        payload = {
            "config": self.config.as_json(),
            "semantic_class": self.semantic_class,
            "selection_basis": self.selection_basis,
            "review_domain_uses_full_roi": True,
        }
        if self.historical_wl139_u_cut is not None:
            payload["historical_wl139_u_cut"] = self.historical_wl139_u_cut
        return payload


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_rows(points: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def _load_xyz_ply(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)


def _full_domain(config: RepresentativeCaseConfig) -> RepresentativeCaseConfig:
    """Return a copy with the WL139 physical domain equal to this full ROI."""

    return replace(config, u_cut=float(config.u_bounds[1]))


def frozen_review_regions() -> tuple[FrozenReviewRegion, ...]:
    """Fixed raw-scene seeds; this function must run before any fit."""

    camera_aligned_curved_rim = RepresentativeCaseConfig(
        name="curved_table_rim",
        semantic_label="camera-aligned curved table side / rim",
        roi_box=Box((-0.70, 1.10, 0.35), (1.30, 1.65, 1.70)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(0.0, 1.0, 0.0),
        u_bounds=(-0.70, 1.30),
        v_bounds=(0.35, 1.70),
        n_bounds=(1.10, 1.65),
        u_cut=1.30,
        frontier_source=(
            "manual fixed seed from DSC08111.JPG raw Gaussian/Visible Surface view; "
            f"pixel window={CURVED_RIM_PIXEL_SEED}; no withheld geometry or fit metric"
        ),
    )

    tabletop = RepresentativeCaseConfig(
        name="tabletop_planar_strip",
        semantic_label="broad planar tabletop strip",
        roi_box=Box((-11.0, 1.0, 2.0), (-7.0, 1.35, 4.0)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(0.0, 1.0, 0.0),
        u_bounds=(-11.0, -7.0),
        v_bounds=(2.0, 4.0),
        n_bounds=(1.0, 1.35),
        u_cut=-7.0,
        frontier_source="manual raw-scene semantic seed; no holdout used in WL140",
    )
    adjacent_side = RepresentativeCaseConfig(
        name="adjacent_table_side",
        semantic_label="additional coherent table-side / curved-edge region",
        roi_box=Box((-4.0, 0.9, 3.5), (-2.0, 1.8, 4.5)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(0.0, 1.0, 0.0),
        u_bounds=(-4.0, -2.0),
        v_bounds=(3.5, 4.5),
        n_bounds=(0.9, 1.8),
        u_cut=-2.0,
        frontier_source="manual raw-scene semantic seed; no holdout used in WL140",
    )
    patio = RepresentativeCaseConfig(
        name="patio_ground_planar",
        semantic_label="broad planar patio / ground control",
        roi_box=Box((-1.0, 1.5, -0.15), (1.0, 2.5, 0.15)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        n_axis=(0.0, 0.0, 1.0),
        u_bounds=(-1.0, 1.0),
        v_bounds=(1.5, 2.5),
        n_bounds=(-0.15, 0.15),
        u_cut=1.0,
        frontier_source="manual raw-scene semantic seed; no holdout used in WL140",
    )
    hedge = RepresentativeCaseConfig(
        name="hedge_background_complex",
        semantic_label="complex hedge / background foliage-like control",
        roi_box=Box((-11.0, 2.0, 0.0), (-9.5, 3.5, 2.5)),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(1.0, 0.0, 0.0),
        u_bounds=(2.0, 3.5),
        v_bounds=(0.0, 2.5),
        n_bounds=(-11.0, -9.5),
        u_cut=3.5,
        frontier_source="manual raw-scene semantic seed; no holdout used in WL140",
    )
    return (
        FrozenReviewRegion(
            config=_full_domain(tabletop),
            semantic_class="BROAD_PLANAR_SURFACE",
            selection_basis="existing semantic tabletop crop from raw WL127 geometry",
        ),
        FrozenReviewRegion(
            config=_full_domain(camera_aligned_curved_rim),
            semantic_class="BROAD_CURVED_SURFACE_POSITIVE_CONTROL",
            selection_basis=(
                "manual camera-aligned table-side/rim seed from raw Gaussian/Visible Surface "
                f"inspection; DSC08111.JPG pixel window={CURVED_RIM_PIXEL_SEED}"
            ),
        ),
        FrozenReviewRegion(
            config=replace(
                _full_domain(CURVED_RIM_CASE),
                name="historical_wl139_curved_rim_alignment_control",
                semantic_label="historical WL139 curved-rim coordinate alignment control",
                frontier_source="historical WL139 ROI copied read-only; semantic alignment is not accepted without user review",
            ),
            semantic_class="HISTORICAL_COORDINATE_ALIGNMENT_CONTROL",
            selection_basis="exact historical WL139 curved-rim ROI, copied read-only for alignment audit only",
            historical_wl139_u_cut=float(CURVED_RIM_CASE.u_cut),
        ),
        FrozenReviewRegion(
            config=_full_domain(adjacent_side),
            semantic_class="BROAD_CURVED_SURFACE_SECOND_CASE",
            selection_basis="fixed adjacent table-side window from raw evidence inspection",
        ),
        FrozenReviewRegion(
            config=_full_domain(patio),
            semantic_class="BROAD_PLANAR_SURFACE_GROUND_CONTROL",
            selection_basis="fixed patio-ground plane window from raw evidence inspection",
        ),
        FrozenReviewRegion(
            config=_full_domain(LEG_CASE),
            semantic_class="THIN_MULTI_SHEET_STRUCTURE",
            selection_basis="exact historical WL139 leg/brace ROI, copied read-only",
        ),
        FrozenReviewRegion(
            config=_full_domain(hedge),
            semantic_class="COMPLEX_BACKGROUND_FOLIAGE_LIKE_CONTROL",
            selection_basis="fixed background/foliage-like window from raw evidence inspection",
        ),
    )


def _camera_tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def project_world_points(points: np.ndarray, camera: Any) -> dict[str, np.ndarray]:
    """Project world points with the same row-vector convention as the repo."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    view = homogeneous @ _camera_tensor_to_numpy(camera.world_view_transform)
    clip = homogeneous @ _camera_tensor_to_numpy(camera.full_proj_transform)
    w = clip[:, 3]
    safe_w = np.maximum(w, 1.0e-12)
    ndc_x = clip[:, 0] / safe_w
    ndc_y = clip[:, 1] / safe_w
    width = int(camera.image_width)
    height = int(camera.image_height)
    pixel_x = (ndc_x + 1.0) * 0.5 * max(width - 1, 1)
    pixel_y = (1.0 - ndc_y) * 0.5 * max(height - 1, 1)
    valid = (w > 1.0e-8) & (view[:, 2] > 0.0) & (pixel_x >= 0.0) & (pixel_x < width) & (pixel_y >= 0.0) & (pixel_y < height)
    return {"x": pixel_x, "y": pixel_y, "depth": view[:, 2], "valid": valid}


def _camera_center(camera: Any) -> np.ndarray:
    return _camera_tensor_to_numpy(camera.camera_center).reshape(3)


def select_review_cameras(points: np.ndarray, cameras: Iterable[Any], count: int = REVIEW_CAMERA_COUNT) -> list[dict[str, Any]]:
    """Select cameras from raw ROI projection only, before any representative fit."""

    points = np.asarray(points, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    candidates: list[dict[str, Any]] = []
    for camera in cameras:
        projection = project_world_points(points, camera)
        valid = projection["valid"]
        if int(np.sum(valid)) < REVIEW_CAMERA_MIN_POINTS:
            continue
        x = projection["x"][valid]
        y = projection["y"][valid]
        width = max(int(camera.image_width), 1)
        height = max(int(camera.image_height), 1)
        bbox_fraction = float((np.ptp(x) * np.ptp(y)) / (width * height))
        view_direction = _camera_center(camera) - centroid
        view_direction /= max(float(np.linalg.norm(view_direction)), 1.0e-12)
        candidates.append({
            "camera_name": str(camera.image_name),
            "valid_projected_points": int(np.sum(valid)),
            "projected_fraction": float(np.mean(valid)),
            "projected_bbox_fraction": bbox_fraction,
            "projected_bbox_pixels": [
                float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))
            ],
            "camera_center": _camera_center(camera),
            "view_direction": view_direction,
            "camera": camera,
        })
    candidates.sort(key=lambda item: (-item["projected_fraction"], -item["projected_bbox_fraction"], item["camera_name"]))
    selected: list[dict[str, Any]] = []
    while candidates and len(selected) < int(count):
        if not selected:
            chosen_index = 0
        else:
            def score(item: dict[str, Any]) -> tuple[float, float, str]:
                similarity = max(float(np.dot(item["view_direction"], prior["view_direction"])) for prior in selected)
                diversity = 1.0 - similarity
                base = 0.70 * item["projected_fraction"] + 0.30 * item["projected_bbox_fraction"]
                return (base + 0.25 * diversity, item["projected_fraction"], item["camera_name"])
            chosen_index = max(range(len(candidates)), key=lambda index: score(candidates[index]))
        selected.append(candidates.pop(chosen_index))
    if not selected:
        ordered = sorted(cameras, key=lambda camera: str(camera.image_name))
        selected = [{"camera_name": str(camera.image_name), "valid_projected_points": 0, "projected_fraction": 0.0, "projected_bbox_fraction": 0.0, "projected_bbox_pixels": None, "camera_center": _camera_center(camera), "view_direction": np.zeros(3), "camera": camera} for camera in ordered[: int(count)]]
    return selected


def _load_canonical_scene(checkpoint: Path, source_path: Path, device: str, images: str, sparse_dir: str, resolution: int, llffhold: int) -> tuple[Any, Any, list[Any], dict[str, Any]]:
    from coverage_first_surfel_partition_export import PRIMITIVE_SURFEL_2D, checkpoint_primitive, load_primitive_model
    from maximal_visible_connectivity_export import load_all_train_cameras
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    model, payload = load_primitive_model(checkpoint, device=device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"WL140 requires the frozen 2DGS surfel checkpoint, got primitive={primitive!r}")
    cameras, camera_meta = load_all_train_cameras(source_path, images, sparse_dir, resolution, llffhold, device)
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    return model, payload, cameras, {"camera_meta": camera_meta, "rasterizer": rasterizer, "primitive": primitive}


def _render_to_pil(package: dict[str, Any]) -> Any:
    from PIL import Image

    tensor = package["render"].detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    array = (tensor.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _display_points(points: np.ndarray, max_points: int = REVIEW_DISPLAY_MAX_POINTS) -> np.ndarray:
    from devtools.demo.parametric_surface_continuation import deterministic_subsample

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= int(max_points):
        return points
    return deterministic_subsample(points, int(max_points))


def _draw_projected_points(image: Any, points: np.ndarray, camera: Any, color: tuple[int, int, int], *, radius: float = REVIEW_POINT_SIZE, alpha: float = REVIEW_POINT_ALPHA) -> Any:
    from PIL import Image, ImageDraw

    projection = project_world_points(points, camera)
    valid = projection["valid"]
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(projection["depth"][indices])[::-1]]
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    fill = (int(color[0]), int(color[1]), int(color[2]), int(round(255.0 * alpha)))
    r = max(float(radius), 1.0)
    for index in indices.tolist():
        x = float(projection["x"][index])
        y = float(projection["y"][index])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def _crop_to_projected_roi(image: Any, points: np.ndarray, camera: Any, margin: int = 24) -> Any:
    projection = project_world_points(points, camera)
    valid = projection["valid"]
    if not np.any(valid):
        return image
    width, height = image.size
    left = max(0, int(np.floor(np.min(projection["x"][valid]))) - margin)
    right = min(width, int(np.ceil(np.max(projection["x"][valid]))) + margin + 1)
    top = max(0, int(np.floor(np.min(projection["y"][valid]))) - margin)
    bottom = min(height, int(np.ceil(np.max(projection["y"][valid]))) + margin + 1)
    if right - left < 24 or bottom - top < 24:
        return image
    return image.crop((left, top, right, bottom))


def write_camera_overlays(case_root: Path, roi_points: np.ndarray, representative_points: np.ndarray, camera: Any, scene_image: Any) -> dict[str, str]:
    camera_root = case_root / "camera_overlays" / str(camera.image_name)
    camera_root.mkdir(parents=True, exist_ok=True)
    raw = _display_points(roi_points)
    representative = _display_points(representative_points)
    scene_only = scene_image.copy()
    scene_raw = _draw_projected_points(scene_only, raw, camera, RAW_OVERLAY_RGB)
    scene_rep = _draw_projected_points(scene_only, representative, camera, REPRESENTATIVE_OVERLAY_RGB)
    from PIL import Image

    geometry_only = _draw_projected_points(
        _draw_projected_points(Image.new("RGB", scene_image.size, (245, 245, 245)), raw, camera, RAW_OVERLAY_RGB),
        representative,
        camera,
        REPRESENTATIVE_OVERLAY_RGB,
    )
    paths = {
        "A_gaussian_scene_only": camera_root / "A_gaussian_scene_only.png",
        "B_gaussian_plus_raw_visible_surface": camera_root / "B_gaussian_plus_raw_visible_surface.png",
        "C_gaussian_plus_surface_representative": camera_root / "C_gaussian_plus_surface_representative.png",
        "D_raw_evidence_plus_representative": camera_root / "D_raw_evidence_plus_representative.png",
        "B_roi_crop": camera_root / "B_roi_crop.png",
        "C_roi_crop": camera_root / "C_roi_crop.png",
        "D_roi_crop": camera_root / "D_roi_crop.png",
    }
    scene_only.save(paths["A_gaussian_scene_only"])
    scene_raw.save(paths["B_gaussian_plus_raw_visible_surface"])
    scene_rep.save(paths["C_gaussian_plus_surface_representative"])
    geometry_only.save(paths["D_raw_evidence_plus_representative"])
    _crop_to_projected_roi(scene_raw, roi_points, camera).save(paths["B_roi_crop"])
    _crop_to_projected_roi(scene_rep, roi_points, camera).save(paths["C_roi_crop"])
    _crop_to_projected_roi(geometry_only, roi_points, camera).save(paths["D_roi_crop"])
    return {key: str(value) for key, value in paths.items()}


def _plot_boundary(axis: Any, grid: np.ndarray, config: RepresentativeCaseConfig, *, label: str = "chart boundary") -> None:
    local = np.asarray(grid, dtype=np.float64)
    local = local.reshape(local.shape[0], local.shape[1], 3)
    local = np.stack([local[..., :] @ np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1) for _ in [0]], axis=0)[0]
    edges = (local[0], local[-1], local[:, 0], local[:, -1])
    for edge in edges:
        axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=BOUNDARY_3D_RGB, linewidth=1.6, label=label if edge is edges[0] else None)


def write_3d_review(case_root: Path, points: np.ndarray, representative: Any, config: RepresentativeCaseConfig) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    grid = representative.sampled_points.reshape(96, 40, 3)
    limits = np.concatenate([points, representative.sampled_points], axis=0)
    geometry_root = case_root / "3d_review"
    paths = {
        "raw_visible_surface": geometry_root / "raw_visible_surface.png",
        "representative_shaded_wireframe": geometry_root / "representative_shaded_wireframe.png",
        "raw_plus_representative": geometry_root / "raw_plus_representative.png",
        "representative_normals": geometry_root / "representative_normals.png",
        "support_chart_boundary": geometry_root / "support_chart_boundary.png",
    }
    _save_3d(paths["raw_visible_surface"], "WL127 raw Visible Surface Evidence", config, limits, lambda axis: _plot_points(axis, points, config, RAW_GREY, label="raw Visible Surface Evidence", alpha=DISPLAY_POINT_ALPHA))
    def representative_wire(axis: Any) -> None:
        _plot_surface(axis, grid, config, REPRESENTATIVE_3D_RGB, label="WL139 physical-chart representative", alpha=0.62)
        local = np.asarray(grid, dtype=np.float64)
        basis = np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1)
        local = local @ basis
        axis.plot_wireframe(local[..., 0], local[..., 1], local[..., 2], rstride=4, cstride=4, color="black", linewidth=0.38, alpha=0.70)
    _save_3d(paths["representative_shaded_wireframe"], "WL139 representative: shaded + wireframe", config, limits, representative_wire)
    _save_3d(paths["raw_plus_representative"], "Raw Visible Surface + WL139 representative", config, limits, lambda axis: (_plot_points(axis, points, config, RAW_GREY, label="raw Visible Surface Evidence", alpha=DISPLAY_POINT_ALPHA), _plot_surface(axis, grid, config, REPRESENTATIVE_3D_RGB, label="WL139 physical-chart representative", alpha=0.58)))
    indices = np.linspace(0, len(representative.sampled_points) - 1, 96, dtype=np.int64)
    basis = np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1)
    normal_points = representative.sampled_points[indices] @ basis
    normal_vectors = representative.sampled_normals[indices] @ basis
    def normal_draw(axis: Any) -> None:
        _plot_points(axis, points, config, RAW_GREY, label="raw Visible Surface Evidence", alpha=DISPLAY_POINT_ALPHA)
        _plot_surface(axis, grid, config, REPRESENTATIVE_3D_RGB, label="WL139 representative", alpha=0.44)
        axis.quiver(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2], normal_vectors[:, 0], normal_vectors[:, 1], normal_vectors[:, 2], length=0.10, normalize=True, color=NORMAL_3D_RGB, linewidth=0.70, label="analytic normals")
    _save_3d(paths["representative_normals"], "WL139 representative analytic normals", config, limits, normal_draw)
    _save_3d(paths["support_chart_boundary"], "WL139 chart support and boundary", config, limits, lambda axis: (_plot_points(axis, points, config, RAW_GREY, label="raw Visible Surface Evidence", alpha=DISPLAY_POINT_ALPHA), _plot_surface(axis, grid, config, REPRESENTATIVE_3D_RGB, label="WL139 representative", alpha=0.48), _plot_boundary(axis, grid, config)))
    return {key: str(value) for key, value in paths.items()}


def write_geometry(case_root: Path, raw_points: np.ndarray, representative: Any, config: RepresentativeCaseConfig) -> dict[str, str]:
    root = case_root / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_visible_surface": root / "raw_visible_surface.ply",
        "wl139_physical_chart_representative": root / "wl139_physical_chart_representative.ply",
        "representative_npz": root / "wl139_physical_chart_representative.npz",
    }
    _write_ply(paths["raw_visible_surface"], raw_points, color=(112, 116, 126))
    _write_ply(paths["wl139_physical_chart_representative"], representative.sampled_points, faces=_grid_faces(96, 40), color=(0, 220, 255))
    np.savez_compressed(paths["representative_npz"], sampled_points=representative.sampled_points, sampled_normals=representative.sampled_normals, sampled_uv=representative.sampled_uv, control_grid=representative.control_grid, fit_input_sha256=np.asarray(representative.fit_input_sha256))
    return {key: str(value) for key, value in paths.items()}


def _write_graphness(path: Path, points: np.ndarray, config: RepresentativeCaseConfig, audit: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1)
    local = np.asarray(points, dtype=np.float64) @ axes
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    scatter = axis.scatter(local[:, 0], local[:, 1], c=local[:, 2], cmap="viridis", s=4.0, alpha=0.97, linewidths=0)
    axis.set_xlabel("physical u")
    axis.set_ylabel("physical v")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"WL140 raw graphness audit: {audit.status}")
    figure.colorbar(scatter, ax=axis, label="physical n")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _raw_case_report(region: FrozenReviewRegion, points: np.ndarray, audit: Any, h: float, cameras: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "region": region.as_json(),
        "raw_point_count": int(len(points)),
        "raw_input_sha256": _sha256_rows(points),
        "graphness": graphness_report(audit, h),
        "camera_ids": [item["camera_name"] for item in cameras],
        "camera_selection": [{key: value for key, value in item.items() if key != "camera"} for item in cameras],
        "representative_attempted": False,
        "withheld_reference_used": False,
    }


def run_validation(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    if not WL127_RAW_VISIBLE_SURFACE.exists():
        raise FileNotFoundError(WL127_RAW_VISIBLE_SURFACE)
    if not WL139_REPORT.exists():
        raise FileNotFoundError(WL139_REPORT)
    wl139_report = json.loads(WL139_REPORT.read_text(encoding="utf-8"))
    h = float(wl139_report["inputs"]["h"])
    mu = float(wl139_report["inputs"]["mu"])
    raw_scene_points = _load_xyz_ply(WL127_RAW_VISIBLE_SURFACE)

    # This is intentionally completed before a fit is constructed or any fit
    # image is written.  The manifest is the frozen review-set contract.
    regions = frozen_review_regions()
    review_cases: list[dict[str, Any]] = []
    for region in regions:
        points = _select_box(raw_scene_points, region.config.roi_box, len(raw_scene_points))
        cameras: list[dict[str, Any]] = []
        review_cases.append({"region": region, "points": points, "cameras": cameras})

    model, payload, all_cameras, scene_info = _load_canonical_scene(
        Path(arguments.checkpoint),
        Path(arguments.source_path),
        arguments.device,
        arguments.images,
        arguments.sparse_dir,
        int(arguments.resolution),
        int(arguments.llffhold),
    )
    for item in review_cases:
        item["cameras"] = select_review_cameras(item["points"], all_cameras, REVIEW_CAMERA_COUNT)
    manifest = {
        "batch": "Worklog 140 real Gaussian-scene qualitative Surface Construction validation",
        "frozen_before_representative": True,
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "checkpoint_iteration": payload.get("iteration"),
        "checkpoint_primitive": scene_info["primitive"],
        "source_path": str(Path(arguments.source_path).resolve()),
        "camera_meta": scene_info["camera_meta"],
        "camera_selection_uses_representative": False,
        "selection_inputs": ["raw WL127 Visible Surface ROI XYZ", "camera projection and raw ROI projected coverage"],
        "selection_excludes": ["fitted representative", "representative metrics", "continuation", "withheld reference"],
        "raw_visible_surface": str(WL127_RAW_VISIBLE_SURFACE.resolve()),
        "regions": [{**item["region"].as_json(), "raw_point_count": int(len(item["points"])), "camera_ids": [cam["camera_name"] for cam in item["cameras"]]} for item in review_cases],
        "manual_choices": [
            "seven semantic ROI/chart seeds",
            "camera count=3",
            "camera selection score from raw ROI projection only",
            f"camera-aligned curved-rim pixel seed={CURVED_RIM_PIXEL_SEED}",
        ],
        "coordinate_alignment_audit": {
            "primary_curved_rim_seed": {
                "camera_id": "DSC08111.JPG",
                "pixel_window": CURVED_RIM_PIXEL_SEED,
                "status": "MANUAL_RAW_SCENE_SEED; USER_REVIEW_REQUIRED",
                "representative_quality_not_used": True,
            },
            "historical_wl139_curved_rim_control": HISTORICAL_CURVED_RIM_ALIGNMENT_AUDIT,
        },
        "leakage_disclosure": "full WL127 geometry defines ROI membership and target visual evidence; no withheld subset exists in WL140, and no representative output is used to choose ROI or camera",
    }
    manifest_path = output_root / "frozen_review_set.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    render_cache: dict[str, Any] = {}
    case_reports: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for item in review_cases:
        region: FrozenReviewRegion = item["region"]
        config = region.config
        points = item["points"]
        case_root = output_root / config.name
        case_root.mkdir(parents=True, exist_ok=True)
        try:
            audit = audit_raw_graphness(points, config, h)
            report = _raw_case_report(region, points, audit, h, item["cameras"])
            report["graphness_visual"] = str(case_root / "graphness_audit.png")
            _write_graphness(case_root / "graphness_audit.png", points, config, audit)
            _write_ply(case_root / "raw_visible_surface_roi.ply", points, color=(112, 116, 126))
            representative = None
            topology = None
            contract = None
            if audit.status == "PASS_GRAPH_LIKE":
                representative = fit_physical_chart_surface(
                    points,
                    config,
                    role="full_evaluation_only",
                    max_fit_points=int(arguments.max_fit_points),
                    device_name=arguments.device,
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
                report["representative_attempted"] = True
                report["representative_mechanism"] = "exact WL139 physical-chart-constrained B-spline graph representative; full ROI evaluation fit, no parameter changes"
                report["representative_role"] = representative.role
                report["representative_contract"] = contract
                report["topology_contract"] = topology
                report["normal_field"] = _normal_field_accounting(representative.sampled_normals)
                report["geometry"] = write_geometry(case_root, points, representative, config)
                report["3d_review"] = write_3d_review(case_root, points, representative, config)
                representative_points = representative.sampled_points
                for camera_item in item["cameras"]:
                    camera = camera_item["camera"]
                    if camera.image_name not in render_cache:
                        render_cache[camera.image_name] = _render_to_pil(scene_info["rasterizer"].render(camera, model))
                    camera_item["outputs"] = write_camera_overlays(case_root, points, representative_points, camera, render_cache[camera.image_name])
                report["camera_overlay_outputs"] = {cam["camera_name"]: cam.get("outputs", {}) for cam in item["cameras"]}
                report["qualitative_classification"] = "USER_REVIEW_REQUIRED"
                report["qualitative_review_rule"] = "automated geometry contracts do not determine macro-shape or Gaussian-scene alignment"
            else:
                for camera_item in item["cameras"]:
                    camera = camera_item["camera"]
                    if camera.image_name not in render_cache:
                        render_cache[camera.image_name] = _render_to_pil(scene_info["rasterizer"].render(camera, model))
                    reference_root = case_root / "camera_reference" / str(camera.image_name)
                    reference_root.mkdir(parents=True, exist_ok=True)
                    reference_path = reference_root / "gaussian_scene_only.png"
                    render_cache[camera.image_name].save(reference_path)
                    camera_item["outputs"] = {"A_gaussian_scene_only": str(reference_path)}
                report["qualitative_classification"] = "OUT_OF_DOMAIN_GRAPHNESS_FAIL"
                report["qualitative_review_rule"] = "representative was not forced after the WL139 graphness gate failed"
            report["withheld_reference_used"] = False
            report["continuation_executed"] = False
            report["occluded_surface_executed"] = False
            report["camera_overlay_outputs"] = {cam["camera_name"]: cam.get("outputs", {}) for cam in item["cameras"]}
            report_path = case_root / "case_report.json"
            report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
            case_reports[config.name] = report
        except Exception as error:
            failures.append({"case": config.name, "error": repr(error)})

    graph_pass = sum(case["graphness"]["status"] == "PASS_GRAPH_LIKE" for case in case_reports.values())
    graph_fail = sum(case["graphness"]["status"] != "PASS_GRAPH_LIKE" for case in case_reports.values())
    report = {
        "batch": "Worklog 140 real Gaussian-scene qualitative Surface Construction validation",
        "status": "ISOLATED_NON_CANONICAL_EVALUATION",
        "INTENT ALIGNMENT": {
            "evaluated_architecture": "trained Gaussian Scene -> renderer-grounded WL127 Visible Surface Evidence -> WL139 graphness -> WL139 physical-chart representative",
            "continuation_evaluated": False,
            "occluded_surface_evaluated": False,
            "new_representative_implemented": False,
        },
        "IMPLEMENTATION FIDELITY": {
            "canonical_renderer_modified": False,
            "canonical_checkpoint_modified": False,
            "wl127_geometry_modified": False,
            "wl139_module_modified": False,
            "wl139_fixed_settings_reused": {
                "resolution_u": GRAPH_RESOLUTION_U,
                "resolution_v": GRAPH_RESOLUTION_V,
                "degree_u": GRAPH_DEGREE_U,
                "degree_v": GRAPH_DEGREE_V,
                "smoothness_lambda": GRAPH_SMOOTHNESS_LAMBDA,
                "tikhonov_lambda": GRAPH_TIKHONOV_LAMBDA,
                "graphness_bin_scale_h": 4.0,
                "graphness_multimode_threshold_h": 3.0,
            },
            "h": h,
            "mu": mu,
            "review_point_alpha": REVIEW_POINT_ALPHA,
            "review_point_size": REVIEW_POINT_SIZE,
            "display_only_thinning_does_not_change_metrics": True,
            "withheld_reference_used": False,
            "continuation_code_path_executed": False,
            "candidate_b_used": False,
        },
        "FROZEN REAL-SCENE REVIEW SET": manifest,
        "COORDINATE ALIGNMENT AUDIT": {
            "primary_curved_rim_seed": manifest["coordinate_alignment_audit"]["primary_curved_rim_seed"],
            "historical_wl139_curved_rim_control": HISTORICAL_CURVED_RIM_ALIGNMENT_AUDIT,
        },
        "GRAPHNESS / APPLICABILITY MAP": {name: case["graphness"] for name, case in case_reports.items()},
        "PER-ROI REPRESENTATIVE CONTRACT": {name: case.get("representative_contract") for name, case in case_reports.items()},
        "PER-ROI GEOMETRIC SANITY": {name: case.get("topology_contract") for name, case in case_reports.items()},
        "GAUSSIAN-SCENE CAMERA IDS": {name: case.get("camera_ids", []) for name, case in case_reports.items()},
        "RAW REVIEW EXPORT PATHS": {name: {"case_root": str(output_root / name), "camera_overlays": case.get("camera_overlay_outputs", {}), "3d_review": case.get("3d_review"), "graphness": case.get("graphness_visual")} for name, case in case_reports.items()},
        "PER-ROI QUALITATIVE CLASSIFICATION": {name: case.get("qualitative_classification", "USER_REVIEW_REQUIRED") for name, case in case_reports.items()},
        "QUALITATIVE ARCHITECTURE VERDICT": "USER REVIEW REQUIRED",
        "SCENE-LEVEL APPLICABILITY SUMMARY": {
            "intended_domain_graph_like_cases": 3,
            "graphness_pass": int(graph_pass),
            "graphness_fail": int(graph_fail),
            "qualitative_clear_pass": "USER_REVIEW_REQUIRED",
            "mixed": "USER_REVIEW_REQUIRED",
            "clear_fail": "USER_REVIEW_REQUIRED",
            "not_a_generalization_percentage": True,
        },
        "PROMOTED": ["WL139 physical-chart graph representative remains the exact evaluated mechanism for graph-like ROIs; human visual confirmation is still required"],
        "RETAINED": ["separation of Gaussian Scene render, raw Visible Surface Evidence, and Surface Representative", "WL139 graphness applicability gate"],
        "REJECTED": ["forcing a representative on materially multivalued thin/background regions", "using proximity metrics as a substitute for macro-shape review"],
        "OPEN": ["support-aware trimming", "non-graph and multi-sheet representatives", "automatic chart discovery", "human review of Gaussian-scene alignment", "all Occluded Surface and continuation questions"],
        "cases": case_reports,
        "failures": failures,
        "inputs": {
            "checkpoint": str(Path(arguments.checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(arguments.checkpoint)),
            "source_path": str(Path(arguments.source_path).resolve()),
            "wl127_raw_visible_surface": str(WL127_RAW_VISIBLE_SURFACE.resolve()),
            "wl127_raw_visible_surface_sha256": _sha256_file(WL127_RAW_VISIBLE_SURFACE),
            "wl139_report": str(WL139_REPORT.resolve()),
            "frozen_review_manifest": str(manifest_path.resolve()),
            "output_root": str(output_root.resolve()),
        },
    }
    report_path = output_root / "real_gaussian_scene_surface_validation_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Worklog 140 real Gaussian-scene Surface Construction validation",
        "",
        "이 출력은 실제 frozen Gaussian Scene render, WL127 raw Visible Surface Evidence, WL139 physical-chart representative를 같은 카메라에서 비교하기 위한 격리 평가 결과다.",
        "",
        "대표면은 WL139 구현을 그대로 재사용했고, graphness 실패 ROI에는 대표면을 강제하지 않았다. continuation, pseudo-occlusion, Candidate B, Occluded Surface는 실행하지 않았다.",
        "",
        f"질적 판정: {report.get('QUALITATIVE ARCHITECTURE VERDICT', 'USER REVIEW REQUIRED')}",
        "",
        "사람이 확인할 핵심 경로는 `frozen_review_set.json`, 각 case의 `camera_overlays/`, `3d_review/`, `case_report.json`이다.",
        "",
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    report = run_validation(build_arg_parser().parse_args(argv))
    print(json.dumps({
        "status": report["status"],
        "qualitative_verdict": report["QUALITATIVE ARCHITECTURE VERDICT"],
        "cases": list(report["cases"]),
        "failures": report["failures"],
    }, indent=2))
    return 0 if report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
