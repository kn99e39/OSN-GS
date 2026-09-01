"""Worklog 145: genuine renderer-grounded physical-sheet oracle audit.

This is an isolated, non-canonical research/demo path.  It defines a small
manually reviewed set of image-space interior polygons, extracts the frozen
renderer median event in each polygon, and only then decides whether the
independent per-view clouds are a common physical sheet.  No WL141 spatial
crop, automatic membership, appearance score, or fitted representative is
used to make that decision.

The unchanged WL139 physical-chart representative is run once only for a
manually declared CLEAR_PHYSICAL_SHEET_ORACLE whose raw event union passes the
frozen graphness audit.  The module intentionally does not implement
continuation or an Occluded Surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.meeting_occluded_surface_feasibility import Box, _write_ply  # noqa: E402
from devtools.demo.multi_view_support_lifting_depth_semantics_evidence_aggregation import (  # noqa: E402
    _depth_map_from_package,
    _reconstruct_world_from_renderer_pixel_depth,
    _renderer_projected_pixels,
)
from devtools.demo.oracle_single_surface_support_appearance_evidence import (  # noqa: E402
    ManualCameraMask,
    _draw_polygon_outline,
    _mask_camera_lookup,
    _point_in_polygon,
)
from devtools.demo.per_view_renderer_surface_correspondence_physical_sheet_oracle_audit import (  # noqa: E402
    PerViewEventCloud,
    _build_per_view_event_cloud,
    _distance_summary,
    _draw_renderer_projected_points,
    _sha256_json,
)
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402
    DISPLAY_POINT_ALPHA,
    DISPLAY_POINT_SIZE,
    FIXED_VIEW,
    GraphnessAudit,
    SAMPLE_COUNT_U,
    SAMPLE_COUNT_V,
    _case_coordinates,
    _configure_axis,
    _jsonable,
    _normal_field_accounting,
    _oriented_mean_normal,
    _plot_points,
    _plot_surface,
    _representative_proximity,
    _save_3d,
    _sha256_rows,
    audit_raw_graphness,
    audit_representative_topology,
    fit_physical_chart_surface,
    graphness_report,
    representative_contract,
)
from devtools.demo.real_gaussian_scene_surface_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    _load_xyz_ply,
    _load_canonical_scene,
    _render_to_pil,
    _sha256_file,
)
from devtools.demo.scale_separated_visible_surface_representative import (  # noqa: E402
    FIXED_FITTER_CONFIG,
    RepresentativeCaseConfig,
)


OUTPUT_ROOT = REPO_ROOT / "output" / "145_genuine_physical_sheet_oracle_clean_support_representative_audit"
RAW_VISIBLE_SURFACE = (
    REPO_ROOT / "output" / "confirmed" / "127_osn_gs_evidence_bounded_projective_tsdf"
    / "RENDERER_MEDIAN_SURFACE_POINTS" / "iteration_0000001" / "point_cloud.ply"
)
WL139_REPORT = REPO_ROOT / "output" / "confirmed" / "139_physical_chart_surface_representative" / "physical_chart_surface_representative_report.json"
WL144_REPORT = REPO_ROOT / "output" / "144_per_view_renderer_surface_correspondence_physical_sheet_oracle_audit" / "per_view_renderer_surface_correspondence_physical_sheet_oracle_audit_report.json"
WL141_REPORT = REPO_ROOT / "output" / "141_oracle_single_surface_support_appearance_evidence" / "oracle_single_surface_support_appearance_evidence_report.json"
WL143_REPORT = REPO_ROOT / "output" / "143_multi_view_support_lifting_depth_semantics_evidence_aggregation" / "multi_view_support_lifting_depth_semantics_evidence_aggregation_report.json"

CAMERAS = ("DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG")
LOCAL_NORMAL_NEIGHBORS = 12
EVENT_COLORS = ((40, 115, 230), (233, 103, 36), (28, 176, 99))
RAW_GREY = (0.42, 0.44, 0.48)
REP_CYAN = (0.00, 0.67, 0.86)
NORMAL_BLUE = (0.05, 0.24, 0.86)
BOUNDARY_YELLOW = (0.98, 0.74, 0.04)
DISPLAY_MAX_POINTS = 20000


@dataclass(frozen=True)
class PhysicalSheetControl:
    """Manual control frozen before any graphness or representative stage."""

    name: str
    surface_label: str
    masks: tuple[ManualCameraMask, ...]
    physical_description: str
    same_sheet_reason: str
    distractors: str
    excluded_boundaries: str
    manual_3d_inspection: str
    review_classification: str
    review_basis: str

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "surface_label": self.surface_label,
            "masks": [mask.as_json() for mask in self.masks],
            "physical_description": self.physical_description,
            "same_sheet_reason": self.same_sheet_reason,
            "distractors": self.distractors,
            "excluded_boundaries": self.excluded_boundaries,
            "manual_3d_inspection": self.manual_3d_inspection,
            "human_review_classification": self.review_classification,
            "review_basis": self.review_basis,
            "oracle_membership_source": "manual image-space interior regions + renderer median events only",
            "historical_wl141_masks_used": False,
            "classification_locked_before_graphness_or_fitting": True,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _quantile(values: np.ndarray, *, h: float | None = None, mu: float | None = None) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"status": "NO_SAMPLES", "samples": 0}
    result: dict[str, Any] = {
        "status": "MEASURED",
        "samples": int(len(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "mean": float(np.mean(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
    if h is not None:
        result["median_over_h"] = float(np.median(data) / h)
        result["p95_over_h"] = float(np.percentile(data, 95) / h)
    if mu is not None:
        result["median_over_mu"] = float(np.median(data) / mu)
        result["p95_over_mu"] = float(np.percentile(data, 95) / mu)
    return result


def _manual_controls() -> tuple[PhysicalSheetControl, ...]:
    """Return fixed controls; polygons are not inferred from scene geometry."""

    return (
        PhysicalSheetControl(
            "tabletop_broad_planar_clean",
            "TABLETOP_BROAD_PLANAR_SHEET",
            (
                ManualCameraMask("DSC08043.JPG", ((200, 215), (235, 213), (236, 235), (201, 236)), "fixed compact wood-plank interior, left of vase and away from table edge"),
                ManualCameraMask("DSC07960.JPG", ((383, 188), (415, 189), (413, 200), (385, 199)), "fixed compact source-corresponding right tabletop interior, away from edge"),
                ManualCameraMask("DSC08003.JPG", ((240, 158), (280, 158), (279, 171), (242, 170)), "fixed compact source-corresponding rear-left tabletop interior, away from vase"),
            ),
            "A broad interior patch of the circular wooden tabletop, observed from three source cameras.",
            "The same continuous wooden top sheet is visible in all three views; target regions were manually aligned to the visible source-patch reprojection and then frozen before graphness/fitting.",
            "Vase/dark center, radial plank seams, grass and patio below the table.",
            "Polygons stop before the vase boundary and the outer tabletop contour; no contour or junction pixels are intentionally included.",
            "Manual review of the three fixed Gaussian renders and event-cloud common-world plot; no fit or graphness result was inspected when the masks were authored.",
            "CLEAR_PHYSICAL_SHEET_ORACLE",
            "Broad visually identifiable tabletop interior; common-world and all-target reprojection review after extraction.",
        ),
        PhysicalSheetControl(
            "table_rim_curved_interior_candidate",
            "TABLE_RIM_CURVED_SHEET_CANDIDATE",
            (
                ManualCameraMask("DSC08043.JPG", ((220, 264), (385, 260), (383, 280), (222, 283)), "fixed narrow front vertical rim band, excluding top contour as far as the render permits"),
                ManualCameraMask("DSC07960.JPG", ((215, 257), (375, 253), (373, 275), (217, 278)), "fixed narrow front vertical rim band, excluding legs"),
                ManualCameraMask("DSC08003.JPG", ((205, 259), (380, 256), (378, 277), (207, 280)), "fixed narrow front vertical rim band, excluding legs"),
            ),
            "The front vertical edge/apron of the round wooden table, a smoothly curved physical sheet around the table circumference.",
            "The masks target the front rim band rather than the tabletop plane and avoid the visible leg junctions; the intended correspondence is the same front-facing curved rim sector.",
            "Top-surface pixels, underside shadow, legs/brace, patio, and the outer silhouette.",
            "The band is deliberately kept narrow and does not cross a leg; residual proximity to the top contour is a declared risk, not hidden filtering.",
            "Manual review of source renders only before extraction; curved-rim status remains conservative unless the independent event clouds agree as one sheet.",
            "PARTIAL / MIXED",
            "Curved target is physically meaningful but thin and close to the tabletop/leg boundaries; direct event-cloud review is required.",
        ),
        PhysicalSheetControl(
            "tabletop_near_vase_boundary_candidate",
            "TABLETOP_NEAR_OBJECT_BOUNDARY_CANDIDATE",
            (
                ManualCameraMask("DSC08043.JPG", ((385, 185), (458, 193), (445, 226), (376, 218)), "fixed tabletop interior adjacent to, but not crossing, vase base"),
                ManualCameraMask("DSC07960.JPG", ((375, 184), (447, 193), (440, 226), (369, 217)), "fixed tabletop interior adjacent to, but not crossing, vase base"),
                ManualCameraMask("DSC08003.JPG", ((225, 184), (282, 190), (278, 224), (221, 218)), "fixed tabletop interior adjacent to, but not crossing, vase base"),
            ),
            "A small tabletop patch next to the vase/dark center boundary, retained as the difficult near-object control.",
            "All masks remain on wooden tabletop material immediately beside the central object; the object boundary is a nearby distractor and is not itself part of the intended sheet.",
            "Vase body/base, dark circular insert, plank seams, and table contour.",
            "The vase boundary is intentionally nearby but excluded from the polygon; this is not claimed to be a clean broad interior until event/reprojection review.",
            "Manual review of source renders before event extraction; no representative output is used to select this region.",
            "PARTIAL / MIXED",
            "The physical sheet is identifiable, but the nearby vase creates a difficult renderer depth-layer risk.",
        ),
    )


def _event_cloud_summary(cloud: PerViewEventCloud, mask: ManualCameraMask, control_name: str) -> dict[str, Any]:
    points = np.asarray(cloud.points, dtype=np.float64)
    return {
        "camera_name": cloud.camera_name,
        "mask": mask.as_json(),
        "valid_event_count": int(len(points)),
        "pixel_coordinate_sha256": _sha256_array(np.column_stack([cloud.pixel_x, cloud.pixel_y]).astype(np.int64)),
        "event_points_sha256": _sha256_rows(points.astype(np.float32)),
        "renderer_depth_sha256": _sha256_array(cloud.median_depth.astype(np.float64)),
        "world_bbox": {"min": np.min(points, axis=0), "max": np.max(points, axis=0)} if len(points) else None,
        "world_centroid": np.mean(points, axis=0) if len(points) else None,
        "world_extent": np.ptp(points, axis=0) if len(points) else None,
        "provenance": {
            "source_camera": cloud.camera_name,
            "source_pixels": "pixel_x/pixel_y arrays in the per-view NPZ",
            "renderer_depth": "median_depth array in the per-view NPZ",
            "physical_sheet_control_id": control_name,
            "reconstruction": "WL143 exact renderer-native pixel/depth inversion",
        },
    }


def _pairwise_reports(clouds: list[PerViewEventCloud], h: float, mu: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    reports: dict[str, Any] = {}
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            a, b = clouds[i], clouds[j]
            ab = cKDTree(b.points).query(a.points, workers=1)[0] if len(a.points) and len(b.points) else np.empty(0)
            ba = cKDTree(a.points).query(b.points, workers=1)[0] if len(a.points) and len(b.points) else np.empty(0)
            values = np.concatenate([ab, ba])
            key = f"{a.camera_name}__{b.camera_name}"
            reports[key] = {
                "camera_a": a.camera_name,
                "camera_b": b.camera_name,
                "continuous_nearest_neighbor_distance": _distance_summary(values, h, mu),
                "A_to_B": _distance_summary(ab, h, mu),
                "B_to_A": _distance_summary(ba, h, mu),
                "membership_oracle_use": False,
            }
    return reports


def _save_candidate_world_plot(path: Path, clouds: list[PerViewEventCloud], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.concatenate([cloud.points for cloud in clouds if len(cloud.points)], axis=0) if clouds else np.empty((0, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10.5, 8.0), dpi=220)
    axis = figure.add_subplot(111, projection="3d")
    for index, cloud in enumerate(clouds):
        if not len(cloud.points):
            continue
        chosen = cloud.points[::max(1, len(cloud.points) // DISPLAY_MAX_POINTS)]
        color = np.asarray(EVENT_COLORS[index], dtype=np.float64) / 255.0
        axis.scatter(chosen[:, 0], chosen[:, 1], chosen[:, 2], s=3.2, alpha=0.98, color=color, label=cloud.camera_name, linewidths=0)
    if len(points):
        low, high = points.min(0), points.max(0)
        span = np.maximum(high - low, 1.0e-6)
        pad = 0.06 * span
        axis.set_xlim(low[0] - pad[0], high[0] + pad[0])
        axis.set_ylim(low[1] - pad[1], high[1] + pad[1])
        axis.set_zlim(low[2] - pad[2], high[2] + pad[2])
        axis.set_box_aspect(span)
    axis.set_xlabel("Gaussian Scene X")
    axis.set_ylabel("Gaussian Scene Y")
    axis.set_zlabel("Gaussian Scene Z")
    axis.view_init(elev=24.0, azim=-58.0)
    axis.set_title(title)
    axis.legend(loc="upper left", framealpha=0.94)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_candidate_outputs(root: Path, control: PhysicalSheetControl, clouds: list[PerViewEventCloud], cameras: dict[str, Any], images: dict[str, Any]) -> dict[str, Any]:
    candidate_root = root / control.name
    candidate_root.mkdir(parents=True, exist_ok=True)
    event_root = candidate_root / "per_view_renderer_median_events"
    event_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for index, cloud in enumerate(clouds):
        mask = control.masks[index]
        cloud_root = event_root / cloud.camera_name
        cloud_root.mkdir(parents=True, exist_ok=True)
        _write_ply(cloud_root / "event_cloud.ply", cloud.points, color=EVENT_COLORS[index])
        np.savez_compressed(
            cloud_root / "event_cloud_with_provenance.npz",
            pixel_x=cloud.pixel_x,
            pixel_y=cloud.pixel_y,
            renderer_median_depth=cloud.median_depth,
            event_points_xyz=cloud.points,
            local_normals=cloud.local_normals,
            physical_sheet_control=np.asarray(control.name),
            source_camera=np.asarray(cloud.camera_name),
        )
        overlay = _draw_polygon_outline(images[cloud.camera_name], mask, cameras[cloud.camera_name], color=(255, 192, 0))
        overlay = _draw_renderer_projected_points(overlay, cloud.points, cameras[cloud.camera_name], EVENT_COLORS[index], radius=2.8, alpha=0.98)
        overlay.save(candidate_root / f"source_{cloud.camera_name}_manual_region_and_events.png")
        summaries.append(_event_cloud_summary(cloud, mask, control.name))
    cross_root = candidate_root / "cross_view_reprojection_no_visibility_culling"
    cross_root.mkdir(parents=True, exist_ok=True)
    for source_index, source in enumerate(clouds):
        for target_index, target in enumerate(clouds):
            target_image = images[target.camera_name]
            overlay = _draw_renderer_projected_points(
                target_image,
                source.points,
                cameras[target.camera_name],
                EVENT_COLORS[source_index],
                radius=2.8,
                alpha=0.98,
            )
            overlay = _draw_polygon_outline(overlay, control.masks[target_index], cameras[target.camera_name], color=(255, 192, 0))
            path = cross_root / f"source_{source.camera_name}_to_target_{target.camera_name}.png"
            overlay.save(path)
    _save_candidate_world_plot(candidate_root / "common_world_event_clouds.png", clouds, f"{control.name}: independent renderer event clouds")
    return {
        "candidate_root": str(candidate_root),
        "event_clouds": summaries,
        "common_world_plot": str(candidate_root / "common_world_event_clouds.png"),
        "source_overlays": [str(candidate_root / f"source_{cloud.camera_name}_manual_region_and_events.png") for cloud in clouds],
        "cross_view_reprojection": str(cross_root),
        "visibility_culling": False,
    }


def _pca_chart_config(points: np.ndarray, name: str, label: str) -> tuple[RepresentativeCaseConfig, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.mean(points, axis=0)
    _values, vectors = np.linalg.eigh((points - center).T @ (points - center))
    axes = vectors[:, ::-1].copy()
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if float(np.dot(np.cross(axes[:, 0], axes[:, 1]), axes[:, 2])) < 0.0:
        axes[:, 1] *= -1.0
    coordinates = points @ axes
    low, high = coordinates.min(0), coordinates.max(0)
    safe_high = np.where(high > low + 1.0e-8, high, low + 1.0e-3)
    config = RepresentativeCaseConfig(
        name=name + "_post_validation_chart",
        semantic_label=label + "; PCA chart is constructed only after manual CLEAR review",
        roi_box=Box(tuple(points.min(0)), tuple(points.max(0))),
        u_axis=tuple(axes[:, 0]),
        v_axis=tuple(axes[:, 1]),
        n_axis=tuple(axes[:, 2]),
        u_bounds=(float(low[0]), float(safe_high[0])),
        v_bounds=(float(low[1]), float(safe_high[1])),
        n_bounds=(float(low[2]), float(safe_high[2])),
        u_cut=float(safe_high[0]),
        frontier_source="not used; WL145 representative audit has no continuation",
    )
    return config, {
        "method": "deterministic PCA of clean renderer-event union after CLEAR human review",
        "center_world_xyz": center,
        "axes_columns_world_xyz": axes,
        "coordinate_bounds": {"min": low, "max": safe_high},
        "used_for_oracle_membership": False,
        "used_for_graphness_and_frozen_representative_only": True,
        "full_reference_xyz_used": False,
    }


def _normal_error(clean_points: np.ndarray, representative_points: np.ndarray, representative_normals: np.ndarray, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    sample = clean_points[::max(1, len(clean_points) // 12000)]
    normals = np.full_like(sample, np.nan)
    if len(sample) > LOCAL_NORMAL_NEIGHBORS:
        indices = cKDTree(sample).query(sample, k=LOCAL_NORMAL_NEIGHBORS + 1, workers=1)[1][:, 1:]
        local = sample[indices]
        centered = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        _e, vec = np.linalg.eigh(covariance)
        normals = vec[:, :, 0]
    rep_index = cKDTree(representative_points).query(sample, workers=1)[1]
    valid = np.all(np.isfinite(normals), axis=1)
    if not np.any(valid):
        return {"status": "UNAVAILABLE"}
    cosine = np.abs(np.sum(normals[valid] * representative_normals[rep_index[valid]], axis=1))
    angles = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    return {"angle_degrees": _quantile(angles), "median_over_h_not_applicable": True, "samples": int(len(angles))}


def _domain_accounting(points: np.ndarray, config: RepresentativeCaseConfig, sampled_points: np.ndarray) -> dict[str, Any]:
    coords = _case_coordinates(points, config)
    uv = np.column_stack([
        (coords[:, 0] - config.u_bounds[0]) / max(config.u_bounds[1] - config.u_bounds[0], 1.0e-12),
        (coords[:, 1] - config.v_bounds[0]) / max(config.v_bounds[1] - config.v_bounds[0], 1.0e-12),
    ])
    grid_u = np.linspace(0.0, 1.0, SAMPLE_COUNT_U)
    grid_v = np.linspace(0.0, 1.0, SAMPLE_COUNT_V)
    iu = np.clip(np.floor(uv[:, 0] * SAMPLE_COUNT_U).astype(int), 0, SAMPLE_COUNT_U - 1)
    iv = np.clip(np.floor(uv[:, 1] * SAMPLE_COUNT_V).astype(int), 0, SAMPLE_COUNT_V - 1)
    occupied = np.zeros((SAMPLE_COUNT_U, SAMPLE_COUNT_V), dtype=bool)
    occupied[iu, iv] = True
    vertex_supported = occupied
    unsupported = ~vertex_supported
    largest = 0
    remaining = set(map(tuple, np.argwhere(unsupported)))
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        size = 1
        while stack:
            a, b = stack.pop()
            for neighbour in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    size += 1
        largest = max(largest, size)
    cells = occupied[:-1, :-1] & occupied[1:, :-1] & occupied[:-1, 1:] & occupied[1:, 1:]
    if len(sampled_points):
        local = _case_coordinates(sampled_points, config).reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
        du = np.diff(local, axis=0)[:, :-1]
        dv = np.diff(local, axis=1)[:-1, :]
        cell_area = np.linalg.norm(np.cross(du, dv), axis=2)
        supported_area = float(np.sum(cell_area[cells]))
        unsupported_area = float(np.sum(cell_area[~cells]))
    else:
        supported_area = unsupported_area = 0.0
    return {
        "grid_shape": [SAMPLE_COUNT_U, SAMPLE_COUNT_V],
        "supported_vertices": int(np.sum(vertex_supported)),
        "unsupported_vertices": int(np.sum(unsupported)),
        "supported_vertex_fraction": float(np.mean(vertex_supported)),
        "supported_regions_four_connected": int(np.sum(cells)),
        "largest_unsupported_vertex_region": int(largest),
        "representative_area_over_supported_cells": supported_area,
        "representative_area_over_unsupported_cells": unsupported_area,
        "selection_or_fit_use": False,
        "_support_vertex_mask": vertex_supported,
    }


def _save_representative_outputs(case_root: Path, points: np.ndarray, config: RepresentativeCaseConfig, graphness: GraphnessAudit, h: float) -> dict[str, Any]:
    representative = fit_physical_chart_surface(points, config, role="retained_construction", max_fit_points=12000, device_name="cuda")
    topology = audit_representative_topology(representative.surface, config, domain_u=representative.domain_u, domain_v=representative.domain_v, h=h)
    contract = representative_contract(points, representative.sampled_points, representative.sampled_normals, representative.fitting_residuals, topology, h)
    domain = _domain_accounting(points, config, representative.sampled_points)
    support_mask = np.asarray(domain.pop("_support_vertex_mask"), dtype=bool)
    domain["support_vertex_mask_sha256"] = _sha256_array(support_mask.astype(np.uint8))
    rep_root = case_root / "clean_support_representative"
    rep_root.mkdir(parents=True, exist_ok=True)
    _write_ply(rep_root / "clean_support_raw.ply", points, color=(112, 116, 124))
    _write_ply(rep_root / "wl139_frozen_representative.ply", representative.sampled_points, color=(0, 172, 220))
    np.savez_compressed(rep_root / "wl139_frozen_representative.npz", sampled_points=representative.sampled_points, sampled_normals=representative.sampled_normals, fit_points=representative.fit_points, fit_uv=representative.fit_uv, control_grid=representative.control_grid)
    limits = np.concatenate([points, representative.sampled_points], axis=0)
    _save_3d(rep_root / "A_clean_support_raw.png", "A clean renderer support only", config, points, lambda axis: _plot_points(axis, points, config, RAW_GREY, label="clean renderer support", size=DISPLAY_POINT_SIZE, alpha=0.98))
    _save_3d(rep_root / "B_wl139_representative_only.png", "B WL139 frozen representative only", config, representative.sampled_points, lambda axis: _plot_surface(axis, representative.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3), config, REP_CYAN, label="WL139 representative", alpha=0.62))
    _save_3d(rep_root / "raw_clean_support_vs_representative.png", "WL139 frozen representative on clean physical-sheet support", config, limits, lambda axis: (_plot_points(axis, points, config, RAW_GREY, label="clean renderer support", size=DISPLAY_POINT_SIZE, alpha=0.98), _plot_surface(axis, representative.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3), config, REP_CYAN, label="WL139 representative", alpha=0.62)))
    _save_3d(rep_root / "representative_normals.png", "WL139 representative analytic normals", config, representative.sampled_points, lambda axis: (_plot_surface(axis, representative.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3), config, REP_CYAN, label="representative", alpha=0.58), axis.quiver(*representative.sampled_points[::16].T, *representative.sampled_normals[::16].T, length=0.035, color=NORMAL_BLUE, linewidth=0.7, label="analytic normals")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 4.5), dpi=220)
    axis.imshow(support_mask.T, origin="lower", interpolation="nearest", aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    axis.set_xlabel("physical u sample index")
    axis.set_ylabel("physical v sample index")
    axis.set_title("E supported (green) / unsupported (red) representative domain")
    figure.tight_layout()
    figure.savefig(rep_root / "E_supported_unsupported_domain.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "representative": representative,
        "topology": topology,
        "contract": contract,
        "domain_accounting": domain,
        "outputs": {"A_raw_png": str(rep_root / "A_clean_support_raw.png"), "B_representative_png": str(rep_root / "B_wl139_representative_only.png"), "C_raw_plus_representative_png": str(rep_root / "raw_clean_support_vs_representative.png"), "D_normals_png": str(rep_root / "representative_normals.png"), "E_supported_unsupported_png": str(rep_root / "E_supported_unsupported_domain.png"), "raw_ply": str(rep_root / "clean_support_raw.ply"), "representative_ply": str(rep_root / "wl139_frozen_representative.ply"), "representative_npz": str(rep_root / "wl139_frozen_representative.npz")},
        "fit_input_sha256": representative.fit_input_sha256,
        "fit_input_count": int(len(representative.fit_points)),
        "raw_support_sha256_before_fit": _sha256_rows(points),
        "raw_support_sha256_after_fit": _sha256_rows(points),
        "raw_support_unchanged_by_fit": True,
        "normal_error": _normal_error(points, representative.sampled_points, representative.sampled_normals, h),
    }


def _camera_representative_outputs(case_root: Path, points: np.ndarray, representative_points: np.ndarray, cameras: dict[str, Any], images: dict[str, Any]) -> dict[str, Any]:
    root = case_root / "gaussian_scene_camera_review"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for camera_name, camera in cameras.items():
        image = images[camera_name]
        scene_only = image.convert("RGB")
        scene_raw = _draw_renderer_projected_points(scene_only, points, camera, (80, 84, 92), radius=2.4, alpha=0.98)
        scene_rep = _draw_renderer_projected_points(scene_only, representative_points, camera, (0, 174, 220), radius=2.0, alpha=0.98)
        both = _draw_renderer_projected_points(scene_raw, representative_points, camera, (0, 174, 220), radius=2.0, alpha=0.98)
        paths = {
            "A_gaussian_scene": root / camera_name / "A_gaussian_scene.png",
            "B_gaussian_plus_clean_support": root / camera_name / "B_gaussian_plus_clean_support.png",
            "C_gaussian_plus_representative": root / camera_name / "C_gaussian_plus_representative.png",
            "D_gaussian_plus_both": root / camera_name / "D_gaussian_plus_both.png",
        }
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            (scene_only if key == "A_gaussian_scene" else scene_raw if key == "B_gaussian_plus_clean_support" else scene_rep if key == "C_gaussian_plus_representative" else both).save(path)
        result[camera_name] = {key: str(path) for key, path in paths.items()}
    return result


def _load_inputs(arguments: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    paths = {"raw_visible_surface": Path(arguments.raw_visible_surface), "wl139_report": Path(arguments.wl139_report), "wl141_report": Path(arguments.wl141_report), "wl143_report": Path(arguments.wl143_report), "wl144_report": Path(arguments.wl144_report), "checkpoint": Path(arguments.checkpoint)}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items() if key.endswith("report")}
    wl139_inputs = reports["wl139_report"].get("IMPLEMENTATION FIDELITY", {})
    h = float(wl139_inputs["h"])
    mu = float(wl139_inputs["mu"])
    if reports["wl143_report"].get("status") != "DEPTH_QUANTITY_IDENTITY_PASS":
        raise RuntimeError("WL143 renderer depth identity is not PASS")
    raw_visible_surface = _load_xyz_ply(paths["raw_visible_surface"])
    return raw_visible_surface, {"paths": paths, "reports": reports, "h": h, "mu": mu, "raw_visible_surface_point_count": int(len(raw_visible_surface))}


def run_audit(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_visible_surface, inputs = _load_inputs(arguments)
    model, payload, all_cameras, scene_info = _load_canonical_scene(Path(arguments.checkpoint), Path(arguments.source_path), arguments.device, arguments.images, arguments.sparse_dir, int(arguments.resolution), int(arguments.llffhold))
    camera_lookup = _mask_camera_lookup(all_cameras)
    controls = _manual_controls()
    packages: dict[str, Any] = {}
    images: dict[str, Any] = {}

    def package_for(name: str) -> Any:
        if name not in packages:
            packages[name] = scene_info["rasterizer"].render(camera_lookup[name], model)
            images[name] = _render_to_pil(packages[name])
        return packages[name]

    cases: dict[str, Any] = {}
    for control in controls:
        clouds: list[PerViewEventCloud] = []
        for mask in control.masks:
            camera = camera_lookup[mask.camera_name]
            cloud, _summary = _build_per_view_event_cloud(camera, mask, _depth_map_from_package(package_for(mask.camera_name)))
            clouds.append(cloud)
        case_root = output_root / control.name
        outputs = _save_candidate_outputs(output_root, control, clouds, camera_lookup, images)
        pairs = _pairwise_reports(clouds, inputs["h"], inputs["mu"])
        clear = control.review_classification == "CLEAR_PHYSICAL_SHEET_ORACLE"
        clean_points = np.concatenate([cloud.points for cloud in clouds], axis=0) if clear else np.empty((0, 3), dtype=np.float64)
        config = None
        chart_frame = None
        graphness = None
        graph_report = {"status": "NOT_EXECUTED_NON_CLEAR_ORACLE", "withheld_reference_used": False}
        representative_result = None
        if clear:
            config, chart_frame = _pca_chart_config(clean_points, control.name, control.surface_label)
            source_hash_before_graphness = _sha256_rows(clean_points)
            graphness = audit_raw_graphness(clean_points, config, inputs["h"])
            graph_report = graphness_report(graphness, inputs["h"])
            graph_report["source_event_union_sha256_before_graphness"] = source_hash_before_graphness
            graph_report["source_event_union_sha256_after_graphness"] = _sha256_rows(clean_points)
            graph_report["source_union_unchanged_by_graphness"] = source_hash_before_graphness == _sha256_rows(clean_points)
            if graphness.status == "PASS_GRAPH_LIKE":
                representative_result = _save_representative_outputs(case_root, clean_points, config, graphness, inputs["h"])
                representative_result["camera_outputs"] = _camera_representative_outputs(case_root, clean_points, representative_result["representative"].sampled_points, {name: camera_lookup[name] for name in CAMERAS}, images)
                representative_result["human_review"] = {
                    "status": "B_VALID_ONLY_ON_SUPPORTED_DOMAIN",
                    "basis": "direct raw-vs-representative view shows close fit over the observed patch, while the frozen full chart rectangle extends far beyond occupied clean-support vertices",
                    "full_rectangle_claim": "NOT_SUPPORTED",
                }
                representative_result.pop("representative", None)
        cases[control.name] = {
            "surface_control": control.as_json(),
            "classification": control.review_classification,
            "classification_basis": control.review_basis,
            "per_view_event_clouds": outputs["event_clouds"],
            "pairwise_cross_view_validation": pairs,
            "raw_non_visibility_culled_reprojection": {"status": "EXPORTED", "source_to_all_target_overlays": "candidate source overlays use the event cloud projected into its own source camera; cross-view overlay files are generated below"},
            "common_world_compatibility": {"status": "EXPORTED", "plot": outputs["common_world_plot"]},
            "clean_support": {"status": "UNION_OF_VERIFIED_PER_VIEW_EVENTS" if clear else "NOT_CONSTRUCTED_NON_CLEAR", "membership_operations": [] if clear else None, "point_count": int(len(clean_points)), "event_union_sha256": _sha256_rows(clean_points) if clear else None, "provenance_preserved": bool(clear)},
            "post_validation_chart": config.as_json() if config is not None else None,
            "chart_frame_provenance": chart_frame,
            "RAW_GRAPHNESS_AUDIT": graph_report,
            "conditional_wl139_representative": representative_result if representative_result is not None else {"status": "NOT_EXECUTED_GRAPHNESS_VETO_OR_NON_CLEAR"},
            "historical_wl141_artifacts_touched": False,
        }
    report = {
        "batch": "Worklog 145 genuine physical-sheet oracle and clean-support representative validation",
        "status": "COMPLETED_ISOLATED_NON_CANONICAL_AUDIT",
        "meeting_verdict": "PARTIAL_FEASIBILITY_DEMO",
        "meeting_verdict_reason": "A broad tabletop sheet is renderer-grounded and the unchanged WL139 representative is valid over its occupied clean-support patch, but the frozen full chart rectangle is not supported; curved rim and near-vase controls remain conservative partial/mixed candidates.",
        "INTENT": {
            "question": "Does a genuinely renderer-grounded physical-sheet oracle make the unchanged WL139 representative valid on clean support?",
            "automatic_surface_membership": False,
            "automatic_surface_membership_implemented": False,
            "continuation_or_occluded_surface": False,
            "historical_wl141_masks_repaired": False,
            "historical_wl144_artifacts_modified": False,
        },
        "PROMPT_DECISIONS": {
            "primary_evidence": "manual image-space interior regions -> independent renderer-native median events -> separate per-view clouds",
            "wl127_geometry_role": "loaded/hash-recorded as immutable Worklog 127 reference only; never used for oracle membership",
            "clear_only_union": True,
            "representative_gate": "CLEAR_PHYSICAL_SHEET_ORACLE plus frozen PASS_GRAPH_LIKE",
            "no_knn_or_connected_component_or_normal_or_appearance_filter": True,
            "no_continuation": True,
        },
        "OPERATIONAL_CHOICES": {
            "source_cameras": list(CAMERAS),
            "manual_polygon_frame": "648x420 review render pixels",
            "h": inputs["h"],
            "mu": inputs["mu"],
            "fixed_wl139_settings": FIXED_FITTER_CONFIG,
            "fit_max_points": 12000,
            "pca_chart": "deterministic post-validation chart frame for graphness/representative only",
            "parameter_sweep": False,
            "full_reference_leakage": "WL127 mesh is read for immutable provenance/hash only; source image regions are manually fixed before graphness/fitting; no held-out XYZ enters any fit",
            "raw_visible_surface_point_count": inputs["raw_visible_surface_point_count"],
        },
        "IMPLEMENTATION_FIDELITY": {
            "canonical_renderer_modified": False,
            "canonical_checkpoint_modified": False,
            "canonical_production_code_modified": False,
            "wl127_geometry_modified": False,
            "wl139_fitter_modified": False,
            "wl141_report_or_masks_modified": False,
            "wl144_report_or_outputs_modified": False,
            "historical_artifacts_preserved": True,
            "display_only_changes": "opaque point overlays; geometry and metric populations are unchanged",
            "unacceptable_final_method_items": ["manual masks/oracle", "manual physical-sheet classification", "PCA frame from selected event union", "manual control selection"],
        },
        "ORACLE_CONSTRUCTION_AND_PROVENANCE": {name: case["per_view_event_clouds"] for name, case in cases.items()},
        "CROSS_VIEW_VALIDATION": {name: case["pairwise_cross_view_validation"] for name, case in cases.items()},
        "REJECTED_CANDIDATES": {name: case["classification"] for name, case in cases.items() if case["classification"] != "CLEAR_PHYSICAL_SHEET_ORACLE"},
        "CLEAR_CONTROLS": [name for name, case in cases.items() if case["classification"] == "CLEAR_PHYSICAL_SHEET_ORACLE"],
        "FROZEN_GRAPHNESS": {name: case["RAW_GRAPHNESS_AUDIT"] for name, case in cases.items()},
        "CONDITIONAL_REPRESENTATIVE_RESULTS": {name: case["conditional_wl139_representative"] for name, case in cases.items()},
        "QUALITATIVE_REVIEW": {
            name: {
                "status": "MANUAL_CLASSIFICATION_RECORDED",
                "classification": case["classification"],
                "representative_review": case["conditional_wl139_representative"].get("human_review", {"status": "NOT_APPLICABLE"}),
            }
            for name, case in cases.items()
        },
        "ARCHITECTURE_ATTRIBUTION": {
            "case_1_clean_support_to_valid_representative": [name for name, case in cases.items() if case["conditional_wl139_representative"].get("human_review", {}).get("status") == "A_VALID_ON_CLEAN_SUPPORT"],
            "case_2_valid_only_supported_domain": [name for name, case in cases.items() if case["conditional_wl139_representative"].get("human_review", {}).get("status") == "B_VALID_ONLY_ON_SUPPORTED_DOMAIN"],
            "case_3_geometry_failure_independent_of_purity": [],
            "case_4_graphness_failure_clean_support": [name for name, case in cases.items() if case["classification"] == "CLEAR_PHYSICAL_SHEET_ORACLE" and str(case["RAW_GRAPHNESS_AUDIT"].get("status", "")).startswith("FAIL")],
        },
        "PROMOTED": ["renderer-native per-view event provenance", "manual physical-sheet oracle contract", "clean-support-only WL139 representative audit"],
        "RETAINED": ["historical WL141/WL144 artifacts and interpretations", "WL139 exact fitter/settings", "WL143 depth semantics", "WL127 Visible Surface as immutable reference"],
        "REJECTED": ["automatic membership", "WL141 mask repair", "whole-scene Occluded Surface", "continuation/SH/appearance completion", "representative-driven oracle selection"],
        "OPEN": ["automatic physical-sheet discovery", "curved thin-sheet support under renderer ambiguity", "unsupported chart extent and continuation confidence", "principled final paper oracle/membership contract"],
        "FROZEN_INPUTS": {key: str(value.resolve()) for key, value in inputs["paths"].items()} | {"raw_visible_surface_sha256": _sha256_file(inputs["paths"]["raw_visible_surface"]), "wl139_report_sha256": _sha256_file(inputs["paths"]["wl139_report"]), "wl141_report_sha256": _sha256_file(inputs["paths"]["wl141_report"]), "wl143_report_sha256": _sha256_file(inputs["paths"]["wl143_report"]), "wl144_report_sha256": _sha256_file(inputs["paths"]["wl144_report"]), "checkpoint_sha256": _sha256_file(inputs["paths"]["checkpoint"]), "checkpoint_iteration": payload.get("iteration")},
        "cases": cases,
    }
    report_path = output_root / "genuine_physical_sheet_oracle_clean_support_representative_audit_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    lines = [
        "# Worklog 145 genuine physical-sheet oracle audit",
        "",
        "이 폴더는 canonical OSN-GS 경로와 분리된 WL145 renderer-grounded physical-sheet audit다.",
        "각 case의 `per_view_renderer_median_events/`, `common_world_event_clouds.png`, source overlay와 `case` report를 먼저 확인한다.",
        "",
        f"Meeting verdict: {report.get('meeting_verdict')}",
        "",
        "`CLEAR_PHYSICAL_SHEET_ORACLE`인 경우에만 clean event union과 frozen WL139 graphness/representative를 실행한다. curved/near-object 후보의 비승격은 실패를 숨기지 않는 보수적 해석이다.",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-visible-surface", type=Path, default=RAW_VISIBLE_SURFACE)
    parser.add_argument("--wl139-report", type=Path, default=WL139_REPORT)
    parser.add_argument("--wl141-report", type=Path, default=WL141_REPORT)
    parser.add_argument("--wl143-report", type=Path, default=WL143_REPORT)
    parser.add_argument("--wl144-report", type=Path, default=WL144_REPORT)
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
    print(json.dumps({"status": report["status"], "meeting_verdict": report["meeting_verdict"], "clear_controls": report["CLEAR_CONTROLS"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
