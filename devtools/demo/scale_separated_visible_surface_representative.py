"""Worklog 138: scale-separated Visible Surface representative audit.

This is an isolated, non-canonical diagnostic track.  The frozen Worklog 127
mesh remains the raw Visible Surface evidence carrier.  The existing NURBS
fitter is applied twice per fixed ROI: once to the full visible ROI for
evaluation-only macro-reference geometry, and once to retained visible rows
for all structural reasoning and continuation.

The withheld rows are never passed to the retained fit, its UV binding, or its
continuation.  They are used only for the declared raw-reference evaluation
and final comparison images.  The physical frontier is the frozen WL136
holdout frontier, not a NURBS domain edge.
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
    Holdout,
    _grid_faces,
    _select_box,
    _write_ply,
    build_fixed_holdout,
    build_self_continuation,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    PRIMARY_ROI as CURVED_RIM_ROI,
    deterministic_indices,
    deterministic_subsample,
    estimate_point_normals,
    _plot_coords,
)
from devtools.demo.semantically_aligned_occluded_surface_demo import (  # noqa: E402
    SEMANTIC_CONFIG as WL136_SEMANTIC_CONFIG,
)
from osn_gs.surface.torch_nurbs import (  # noqa: E402
    TorchNURBSSurface,
    fit_torch_visible_surface_lsq,
    project_torch_points_to_nurbs,
)


WORKLOG_127_MESH = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz"
WORKLOG_127_FIELD = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz"

# Existing fitter configuration.  This is frozen before either case is run;
# it is deliberately shared by full and retained fits and is not swept.
FIXED_FITTER_CONFIG: dict[str, Any] = {
    "resolution_u": 8,
    "resolution_v": 4,
    "degree_u": 2,
    "degree_v": 2,
    "smoothness_lambda": 1.0e-4,
    "tikhonov_lambda": 1.0e-4,
    "correction_rounds": 2,
    "chunk_size": 8192,
    "projection_iterations": 2,
}

DISPLAY_VOXEL_WORLD = 0.02
DISPLAY_POINT_SIZE = 2.35
DISPLAY_POINT_ALPHA = 0.98
DISPLAY_REFERENCE_ALPHA = 0.99
DISPLAY_SURFACE_ALPHA = 0.34
DISPLAY_PREDICTION_ALPHA = 0.50

# Frozen post-evaluation interpretation thresholds.  These do not select a
# fit, a continuation extent, or a display; they only distinguish a useful
# representative from a continuation that remains insufficient.
CONTINUATION_MEDIAN_LIMIT_OVER_H = 5.0
CONTINUATION_MIN_COVERAGE_LE_2H = 0.20

OBSERVED_GREY = (0.48, 0.50, 0.54)
RAW_REFERENCE_GREEN = (0.08, 0.70, 0.28)
FULL_REPRESENTATIVE_BLUE = (0.08, 0.30, 0.82)
RETAINED_REPRESENTATIVE_ORANGE = (0.96, 0.42, 0.06)
PREDICTED_CYAN = (0.00, 0.70, 0.82)
FRONTIER_YELLOW = (0.98, 0.78, 0.05)
RAW_NORMAL_RED = (0.86, 0.08, 0.10)
REP_NORMAL_BLUE = (0.05, 0.25, 0.88)

FIXED_VIEW = {"elev": 24.0, "azim": -58.0}
SECOND_VIEW = {"elev": 15.0, "azim": 28.0}


@dataclass(frozen=True)
class RepresentativeCaseConfig:
    name: str
    semantic_label: str
    roi_box: Box
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    n_axis: tuple[float, float, float]
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    n_bounds: tuple[float, float]
    u_cut: float
    frontier_source: str

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_label": self.semantic_label,
            "roi_box": self.roi_box.as_json(),
            "u_axis": list(self.u_axis),
            "v_axis": list(self.v_axis),
            "n_axis": list(self.n_axis),
            "u_bounds": list(self.u_bounds),
            "v_bounds": list(self.v_bounds),
            "n_bounds": list(self.n_bounds),
            "u_cut": self.u_cut,
            "frontier_source": self.frontier_source,
        }


@dataclass
class FittedRepresentative:
    surface: TorchNURBSSurface
    fit_points: np.ndarray
    fit_uv: np.ndarray
    sampled_points: np.ndarray
    sampled_normals: np.ndarray
    control_grid: np.ndarray
    fitting_residuals: np.ndarray
    fit_input_sha256: str


@dataclass
class RepresentativeContinuation:
    frontier_raw: np.ndarray
    frontier_representative: np.ndarray
    frontier_uv: np.ndarray
    frontier_normals: np.ndarray
    frontier_tangent_v: np.ndarray
    direction: np.ndarray
    points_grid: np.ndarray
    normals_grid: np.ndarray
    l_values: np.ndarray
    projection_gaps: np.ndarray
    boundary_position_gaps: np.ndarray
    boundary_normal_angles: np.ndarray


@dataclass
class CaseAudit:
    config: RepresentativeCaseConfig
    full_points: np.ndarray
    retained_points: np.ndarray
    withheld_points: np.ndarray
    holdout: Holdout
    raw_scale_audit: dict[str, Any]
    full_representative: FittedRepresentative
    retained_representative: FittedRepresentative
    retained_shape_audit: dict[str, Any]
    continuation: RepresentativeContinuation
    raw_metrics: dict[str, Any]
    macro_metrics: dict[str, Any]
    raw_frontier_normals: np.ndarray
    visuals: dict[str, str]
    geometry: dict[str, str]


# The leg ROI is copied from the frozen WL136 H1 contract.  The curved rim is
# the earlier fixed curved-rim ROI, retained here as a second scale-audit case;
# no bounds are inferred from withheld geometry.
LEG_CASE = RepresentativeCaseConfig(
    name="wl136_leg_brace",
    semantic_label="WL136 actual leg / brace H1 ROI",
    roi_box=WL136_SEMANTIC_CONFIG.leg_box,
    u_axis=(0.0, 1.0, 0.0),
    v_axis=(0.0, 0.0, 1.0),
    n_axis=(1.0, 0.0, 0.0),
    u_bounds=WL136_SEMANTIC_CONFIG.leg_u_bounds,
    v_bounds=WL136_SEMANTIC_CONFIG.leg_v_bounds,
    n_bounds=WL136_SEMANTIC_CONFIG.leg_n_bounds,
    u_cut=WL136_SEMANTIC_CONFIG.leg_u_cut,
    frontier_source="frozen WL136 H1 physical frontier from retained-only holdout frame",
)

CURVED_RIM_CASE = RepresentativeCaseConfig(
    name="curved_table_rim",
    semantic_label="curved table side / rim scale audit",
    roi_box=Box(
        (CURVED_RIM_ROI.u_bounds[0], CURVED_RIM_ROI.n_bounds[0], CURVED_RIM_ROI.v_bounds[0]),
        (CURVED_RIM_ROI.u_bounds[1], CURVED_RIM_ROI.n_bounds[1], CURVED_RIM_ROI.v_bounds[1]),
    ),
    u_axis=CURVED_RIM_ROI.axis_u,
    v_axis=CURVED_RIM_ROI.axis_v,
    n_axis=CURVED_RIM_ROI.axis_n,
    u_bounds=CURVED_RIM_ROI.u_bounds,
    v_bounds=CURVED_RIM_ROI.v_bounds,
    n_bounds=CURVED_RIM_ROI.n_bounds,
    u_cut=CURVED_RIM_ROI.u_bounds[0] + CURVED_RIM_ROI.holdout_u_cut * (CURVED_RIM_ROI.u_bounds[1] - CURVED_RIM_ROI.u_bounds[0]),
    frontier_source="frozen curved-rim physical holdout frontier from the prior bounded demo ROI",
)


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


def _normalised_axis(axis: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(axis), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("case axis must be non-zero")
    return value / norm


def _case_coordinates(points: np.ndarray, config: RepresentativeCaseConfig) -> np.ndarray:
    axes = np.stack(
        [_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)],
        axis=1,
    )
    return np.asarray(points, dtype=np.float64) @ axes


def _build_case_holdout(points: np.ndarray, config: RepresentativeCaseConfig) -> Holdout:
    return build_fixed_holdout(
        points,
        name=config.name + "_physical_holdout",
        u_axis=config.u_axis,
        v_axis=config.v_axis,
        n_axis=config.n_axis,
        u_bounds=config.u_bounds,
        v_bounds=config.v_bounds,
        n_bounds=config.n_bounds,
        u_cut=config.u_cut,
        permitted_volume=config.roi_box,
    )


def _oriented_mean_normal(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    if len(normals) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    reference = normals[0] / max(float(np.linalg.norm(normals[0])), 1.0e-12)
    aligned = normals.copy()
    signs = np.where(aligned @ reference < 0.0, -1.0, 1.0)
    aligned *= signs[:, None]
    result = aligned.mean(axis=0)
    return result / max(float(np.linalg.norm(result)), 1.0e-12)


def _angles_to_reference(normals: np.ndarray, reference: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12, None)
    reference = reference / max(float(np.linalg.norm(reference)), 1.0e-12)
    return np.degrees(np.arccos(np.clip(np.abs(normals @ reference), 0.0, 1.0)))


def _summary(values: np.ndarray, *, over_h: float | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "UNAVAILABLE", "samples": 0}
    result: dict[str, Any] = {
        "status": "MEASURED",
        "samples": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }
    if over_h is not None:
        result["median_over_h"] = float(np.median(values) / over_h)
        result["p95_over_h"] = float(np.percentile(values, 95) / over_h)
    return result


def raw_surface_scale_audit(points: np.ndarray, h: float, *, max_normal_points: int = 5000) -> dict[str, Any]:
    """Quantify raw local roughness and scale using retained visible rows only."""

    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 8:
        raise ValueError("scale audit needs at least eight retained points")
    sample = deterministic_subsample(points, max_normal_points)
    normals = estimate_point_normals(sample, k=min(20, len(sample)))
    centered = sample - sample.mean(axis=0, keepdims=True)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    global_normal = vectors[:, 0]
    if normals is not None:
        global_normal = _oriented_mean_normal(normals)
        normal_angles = _angles_to_reference(normals, global_normal)
    else:
        normal_angles = np.empty((0,), dtype=np.float64)
    global_residual = np.abs((points - points.mean(axis=0, keepdims=True)) @ global_normal)
    tree = cKDTree(sample)
    neighbour_count = min(8, len(sample))
    distances, _ = tree.query(sample, k=neighbour_count, workers=1)
    nearest = distances[:, 1] if neighbour_count > 1 else distances[:, 0]
    local_scale = distances[:, -1] if neighbour_count > 1 else nearest
    extent = np.ptp(points, axis=0)
    return {
        "population": "retained visible geometry only",
        "point_count": int(len(points)),
        "normal_sample_count": int(len(sample)),
        "raw_pca_normal_dispersion_degrees": _summary(normal_angles),
        "raw_pca_reference_normal": global_normal,
        "nearest_neighbor_world": _summary(nearest, over_h=h),
        "local_surface_scale_k8_world": _summary(local_scale, over_h=h),
        "local_plane_residual_world": _summary(global_residual, over_h=h),
        "coarse_spatial_extent_world_xyz": extent,
        "coarse_spatial_extent_over_h_xyz": extent / h,
        "coarse_diagonal_world": float(np.linalg.norm(extent)),
        "coarse_diagonal_over_h": float(np.linalg.norm(extent) / h),
        "h_world": float(h),
    }


def _fit_uv(points: np.ndarray, config: RepresentativeCaseConfig, *, retained_domain: bool) -> np.ndarray:
    coordinates = _case_coordinates(points, config)
    u0, u1 = config.u_bounds
    v0, v1 = config.v_bounds
    if retained_domain:
        u1 = config.u_cut
    u = (coordinates[:, 0] - u0) / max(float(u1 - u0), 1.0e-12)
    v = (coordinates[:, 1] - v0) / max(float(v1 - v0), 1.0e-12)
    uv = np.column_stack([u, v]).astype(np.float32)
    if np.any(uv < -1.0e-5) or np.any(uv > 1.0 + 1.0e-5):
        raise AssertionError("fixed case UV binding escaped the declared domain")
    return np.clip(uv, 0.0, 1.0)


def fit_existing_nurbs(points: np.ndarray, config: RepresentativeCaseConfig, h: float, *, retained_domain: bool, max_fit_points: int, device_name: str) -> FittedRepresentative:
    """Fit the existing NURBS surface with the frozen configuration."""

    import torch

    points = np.asarray(points, dtype=np.float64)
    fit_indices = deterministic_indices(len(points), max_fit_points)
    fit_points = points[fit_indices].copy()
    initial_uv = _fit_uv(fit_points, config, retained_domain=retained_domain)
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name)
    torch_points = torch.as_tensor(fit_points, dtype=torch.float32, device=device)
    torch_uv = torch.as_tensor(initial_uv, dtype=torch.float32, device=device)
    kwargs = dict(FIXED_FITTER_CONFIG)
    kwargs["initial_uv"] = torch_uv
    with torch.no_grad():
        surface, final_uv = fit_torch_visible_surface_lsq(torch_points, **kwargs)
        u = torch.linspace(0.0, 1.0, 96, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
        v = torch.linspace(0.0, 1.0, 40, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        sample_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        sampled_points_torch, sampled_normals_torch = surface.evaluate_with_normals(sample_uv)
        fit_points_torch = torch.as_tensor(fit_points, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
        residuals = torch.linalg.norm(surface.evaluate(final_uv) - fit_points_torch, dim=1)
        sampled_points = sampled_points_torch.detach().cpu().numpy().astype(np.float64)
        sampled_normals = sampled_normals_torch.detach().cpu().numpy().astype(np.float64)
        final_uv_np = final_uv.detach().cpu().numpy().astype(np.float64)
        control_grid = surface.control_grid.detach().cpu().numpy().astype(np.float64)
        residual_values = residuals.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(sampled_points).all() or not np.isfinite(sampled_normals).all():
        raise ValueError("NURBS representative evaluation is non-finite")
    return FittedRepresentative(
        surface=surface,
        fit_points=fit_points,
        fit_uv=final_uv_np,
        sampled_points=sampled_points,
        sampled_normals=sampled_normals,
        control_grid=control_grid,
        fitting_residuals=residual_values,
        fit_input_sha256=_sha256_rows(fit_points),
    )


def _representative_stats(rep: FittedRepresentative, raw_points: np.ndarray, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    distances = cKDTree(rep.sampled_points).query(deterministic_subsample(raw_points, 12000), workers=1)[0]
    representative_angles = _angles_to_reference(rep.sampled_normals, _oriented_mean_normal(rep.sampled_normals))
    residual = rep.fitting_residuals
    return {
        "raw_to_representative_distance": _summary(distances, over_h=h),
        "representative_analytic_normal_variation_degrees": _summary(representative_angles),
        "representative_fitting_residual": _summary(residual, over_h=h),
        "fit_point_count": int(len(rep.fit_points)),
        "fit_input_sha256": rep.fit_input_sha256,
        "representative_sample_count": int(len(rep.sampled_points)),
        "h_world": float(h),
    }


def _visible_macro_shape_audit(rep: FittedRepresentative, retained_points: np.ndarray, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    raw = np.asarray(retained_points, dtype=np.float64)
    distances = cKDTree(rep.sampled_points).query(deterministic_subsample(raw, 12000), workers=1)[0]
    finite = bool(np.isfinite(rep.sampled_points).all() and np.isfinite(rep.sampled_normals).all())
    # This is a fixed shape-audit criterion, independent of continuation and
    # withheld evaluation.  It is disclosed rather than tuned per case.
    median_over_h = float(np.median(distances) / h)
    p95_over_h = float(np.percentile(distances, 95) / h)
    passed = bool(finite and median_over_h <= 3.0 and p95_over_h <= 12.0)
    return {
        "status": "PASS" if passed else "STOP",
        "criterion": "finite representative and retained raw-to-representative median <= 3h, p95 <= 12h",
        "uses_withheld_reference": False,
        "raw_to_representative_median_over_h": median_over_h,
        "raw_to_representative_p95_over_h": p95_over_h,
        "representative_finite": finite,
    }


def _representative_frontier_continuation(
    holdout: Holdout,
    retained_rep: FittedRepresentative,
    *,
    h: float,
) -> RepresentativeContinuation:
    """Map the frozen physical frontier and continue in its analytic frame."""

    import torch

    # build_self_continuation is called only for the frozen WL136 physical
    # frontier.  Its continuation direction is discarded; no raw tangent is
    # used for the candidate geometry.
    frozen = build_self_continuation(holdout, frontier_band_fraction=0.10, frontier_bins=24, continuation_samples=32)
    frontier_raw = np.asarray(frozen.frontier_points, dtype=np.float64)
    device = retained_rep.surface.control_grid.device
    frontier_uv_torch = project_torch_points_to_nurbs(
        torch.as_tensor(frontier_raw, dtype=torch.float32, device=device),
        retained_rep.surface,
        iterations=2,
        chunk_size=8192,
    )
    with torch.no_grad():
        frontier_rep_torch, du_torch, dv_torch = retained_rep.surface.evaluate_with_derivatives(frontier_uv_torch)
        frontier_normals_torch = retained_rep.surface.evaluate_with_normals(frontier_uv_torch)[1]
    frontier_uv = frontier_uv_torch.detach().cpu().numpy().astype(np.float64)
    frontier_representative = frontier_rep_torch.detach().cpu().numpy().astype(np.float64)
    frontier_normals = frontier_normals_torch.detach().cpu().numpy().astype(np.float64)
    tangent_v = dv_torch.detach().cpu().numpy().astype(np.float64)
    tangent_v /= np.clip(np.linalg.norm(tangent_v, axis=1, keepdims=True), 1.0e-12, None)
    projection_gaps = np.linalg.norm(frontier_representative - frontier_raw, axis=1)

    u_axis = _normalised_axis(holdout.u_axis)
    directions = u_axis[None, :] - np.sum(frontier_normals * u_axis[None, :], axis=1, keepdims=True) * frontier_normals
    direction_lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    du_np = du_torch.detach().cpu().numpy().astype(np.float64)
    directions = np.where(direction_lengths > 1.0e-10, directions, du_np)
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12, None)
    directions = np.where(np.sum(directions * u_axis[None, :], axis=1, keepdims=True) < 0.0, -directions, directions)
    direction = directions.mean(axis=0)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    if float(direction @ u_axis) < 0.0:
        direction = -direction

    extent = float(holdout.u_bounds[1] - holdout.u_cut)
    l_values = np.linspace(0.0, extent, 32, dtype=np.float64)
    points_grid = frontier_representative[None, :, :] + l_values[:, None, None] * directions[None, :, :]
    normals_grid = np.cross(directions[None, :, :], tangent_v[None, :, :])
    normals_grid /= np.clip(np.linalg.norm(normals_grid, axis=2, keepdims=True), 1.0e-12, None)
    signs = np.sum(normals_grid * frontier_normals[None, :, :], axis=2, keepdims=True) < 0.0
    normals_grid = np.where(signs, -normals_grid, normals_grid)
    normals_grid = np.tile(normals_grid, (len(l_values), 1, 1))
    boundary_position_gaps = np.linalg.norm(points_grid[0] - frontier_representative, axis=1)
    boundary_normal_angles = np.degrees(
        np.arccos(np.clip(np.abs(np.sum(normals_grid[0] * frontier_normals, axis=1)), 0.0, 1.0))
    )
    _ = h  # h is part of the explicit call contract; extent remains physical.
    return RepresentativeContinuation(
        frontier_raw=frontier_raw,
        frontier_representative=frontier_representative,
        frontier_uv=frontier_uv,
        frontier_normals=frontier_normals,
        frontier_tangent_v=tangent_v,
        direction=direction,
        points_grid=points_grid,
        normals_grid=normals_grid,
        l_values=l_values,
        projection_gaps=projection_gaps,
        boundary_position_gaps=boundary_position_gaps,
        boundary_normal_angles=boundary_normal_angles,
    )


def _representative_reference(rep: FittedRepresentative, config: RepresentativeCaseConfig, samples_u: int = 48, samples_v: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    u = np.linspace(config.u_cut, 1.0, int(samples_u), dtype=np.float32)
    v = np.linspace(0.0, 1.0, int(samples_v), dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv = torch.as_tensor(np.column_stack([uu.reshape(-1), vv.reshape(-1)]), dtype=torch.float32, device=rep.surface.control_grid.device)
    with torch.no_grad():
        points, normals = rep.surface.evaluate_with_normals(uv)
    return (
        points.detach().cpu().numpy().astype(np.float64),
        normals.detach().cpu().numpy().astype(np.float64),
        uv.detach().cpu().numpy().astype(np.float64),
    )


def _reference_metrics(reference_points: np.ndarray, reference_normals: np.ndarray | None, continuation: RepresentativeContinuation, h: float, config: RepresentativeCaseConfig, *, label: str) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    reference_points = np.asarray(reference_points, dtype=np.float64)
    predicted_points = continuation.points_grid.reshape(-1, 3)
    predicted_normals = continuation.normals_grid.reshape(-1, 3)
    distances, nearest = cKDTree(predicted_points).query(reference_points, workers=1)
    result: dict[str, Any] = {
        "reference": label,
        "evaluation_population": label,
        "sample_count": int(len(reference_points)),
        "point_to_surface_distance": _summary(distances, over_h=h),
        "coverage": {
            "fraction_le_h": float(np.mean(distances <= h)),
            "fraction_le_2h": float(np.mean(distances <= 2.0 * h)),
        },
        "normal_angular_error": {"status": "UNAVAILABLE"},
        "distance_from_frontier_bins": [],
        "withheld_xyz_entered_construction": False,
    }
    if reference_normals is not None:
        ref = reference_normals / np.clip(np.linalg.norm(reference_normals, axis=1, keepdims=True), 1.0e-12, None)
        pred = predicted_normals[nearest]
        pred = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1.0e-12, None)
        angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(ref * pred, axis=1)), 0.0, 1.0)))
        result["normal_angular_error"] = _summary(angles)
    coordinates = _case_coordinates(reference_points, config)
    distances_from_frontier = np.maximum(coordinates[:, 0] - float(config.u_cut), 0.0)
    max_distance = max(float(config.u_bounds[1] - config.u_cut), 1.0e-12)
    edges = np.linspace(0.0, max_distance, 5)
    for index in range(4):
        mask = (distances_from_frontier >= edges[index]) & (distances_from_frontier <= edges[index + 1] if index == 3 else distances_from_frontier < edges[index + 1])
        if np.any(mask):
            result["distance_from_frontier_bins"].append({
                "bin": index,
                "distance_world": [float(edges[index]), float(edges[index + 1])],
                "samples": int(np.sum(mask)),
                "distance_median_over_h": float(np.median(distances[mask]) / h),
                "distance_p95_over_h": float(np.percentile(distances[mask], 95) / h),
            })
    return result


def _frontier_mapping_report(continuation: RepresentativeContinuation, retained_points: np.ndarray, h: float) -> dict[str, Any]:
    raw_normals = estimate_point_normals(deterministic_subsample(retained_points, 5000), k=20)
    if raw_normals is None:
        raw_rep_angles = np.empty((0,), dtype=np.float64)
    else:
        from scipy.spatial import cKDTree

        sampled_points = deterministic_subsample(retained_points, 5000)
        nearest = cKDTree(sampled_points).query(continuation.frontier_raw, workers=1)[1]
        raw_frontier = raw_normals[nearest]
        raw_frontier = raw_frontier / np.clip(np.linalg.norm(raw_frontier, axis=1, keepdims=True), 1.0e-12, None)
        rep_frontier = continuation.frontier_normals / np.clip(np.linalg.norm(continuation.frontier_normals, axis=1, keepdims=True), 1.0e-12, None)
        raw_rep_angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(raw_frontier * rep_frontier, axis=1)), 0.0, 1.0)))
    return {
        "frontier_source": "frozen physical world-space frontier; not a NURBS UV edge",
        "sample_count": int(len(continuation.frontier_raw)),
        "frontier_to_retained_representative_position_gap": _summary(continuation.projection_gaps, over_h=h),
        "representative_analytic_normals": continuation.frontier_normals,
        "representative_tangent_v": continuation.frontier_tangent_v,
        "representative_u_direction": continuation.direction,
        "raw_pca_vs_representative_normal_angle": _summary(raw_rep_angles),
        "boundary_position_gap_representative_to_prediction": _summary(continuation.boundary_position_gaps, over_h=h),
        "boundary_normal_angle_discontinuity_degrees": _summary(continuation.boundary_normal_angles),
        "physical_frontier_used_as_termination": True,
        "nurbs_uv_edge_used_as_termination": False,
    }


def _display_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points
    cells = np.floor(points / DISPLAY_VOXEL_WORLD).astype(np.int64)
    _cells, first = np.unique(cells, axis=0, return_index=True)
    return deterministic_subsample(points[np.sort(first)], max_points)


def _plot_points(axis: Any, points: np.ndarray, color: Any, *, size: float = DISPLAY_POINT_SIZE, alpha: float = DISPLAY_POINT_ALPHA, max_points: int = 14000, label: str | None = None) -> None:
    points = _display_subsample(points, max_points)
    if len(points):
        display = _plot_coords(points)
        axis.scatter(display[:, 0], display[:, 1], display[:, 2], s=size, alpha=alpha, color=color, linewidths=0, label=label)


def _plot_surface(axis: Any, points_grid: np.ndarray, color: Any, *, alpha: float = DISPLAY_SURFACE_ALPHA) -> None:
    if points_grid.ndim != 3 or points_grid.shape[0] < 2 or points_grid.shape[1] < 2:
        return
    display = _plot_coords(points_grid.reshape(-1, 3)).reshape(points_grid.shape[0], points_grid.shape[1], 3)
    axis.plot_surface(display[:, :, 0], display[:, :, 1], display[:, :, 2], color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=False)


def _configure_axis(axis: Any, limits_points: np.ndarray, *, view: dict[str, float] = FIXED_VIEW) -> None:
    limits_points = np.asarray(limits_points, dtype=np.float64)
    low = np.min(limits_points, axis=0)
    high = np.max(limits_points, axis=0)
    span = np.maximum(high - low, 1.0e-6)
    padding = 0.06 * span
    limits = np.stack([low - padding, high + padding], axis=1)[[0, 2, 1]]
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    try:
        axis.set_box_aspect(span[[0, 2, 1]])
    except AttributeError:
        pass
    axis.view_init(elev=view["elev"], azim=view["azim"])
    axis.set_xlabel("world x")
    axis.set_ylabel("world z")
    axis.set_zlabel("world y")


def _save_raw_figure(path: Path, title: str, limits: np.ndarray, draw: Any, *, view: dict[str, float] = FIXED_VIEW) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8.8, 7.3), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    _configure_axis(axis, limits, view=view)
    draw(axis, figure)
    axis.set_title(title)
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _raw_normals(points: np.ndarray, *, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray] | None:
    sampled = deterministic_subsample(points, max_points)
    normals = estimate_point_normals(sampled, k=min(20, len(sampled)))
    if normals is None:
        return None
    return sampled, normals


def write_case_visuals(case: CaseAudit, output_root: Path) -> dict[str, str]:
    case_root = output_root / case.config.name
    case_root.mkdir(parents=True, exist_ok=True)
    raw_normals = _raw_normals(case.retained_points)
    full_surface_grid = case.full_representative.sampled_points.reshape(96, 40, 3)
    retained_surface_grid = case.retained_representative.sampled_points.reshape(96, 40, 3)
    continuation_grid = case.continuation.points_grid
    macro_points, macro_normals, _macro_uv = _representative_reference(case.full_representative, case.config)
    macro_grid = macro_points.reshape(48, 32, 3)
    all_base = np.concatenate([case.full_points, case.retained_points, case.withheld_points, full_surface_grid.reshape(-1, 3), retained_surface_grid.reshape(-1, 3), continuation_grid.reshape(-1, 3), macro_points], axis=0)
    paths = {
        "raw_visible_surface": case_root / "raw_visible_surface.png",
        "raw_vs_full_representative": case_root / "raw_vs_full_representative.png",
        "retained_raw_vs_retained_representative": case_root / "retained_raw_vs_retained_representative.png",
        "raw_normals_vs_representative_normals": case_root / "raw_normals_vs_representative_normals.png",
        "representative_continuation": case_root / "representative_continuation.png",
        "continuation_vs_raw_reference": case_root / "continuation_vs_raw_reference.png",
        "continuation_vs_macro_reference": case_root / "continuation_vs_macro_reference.png",
    }

    def draw_raw(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.full_points, OBSERVED_GREY, alpha=DISPLAY_REFERENCE_ALPHA, label="raw full WL127 visible ROI")
        _plot_points(axis, case.retained_points, OBSERVED_GREY, size=2.7, alpha=DISPLAY_POINT_ALPHA, label="retained visible")
        _plot_points(axis, case.withheld_points, RAW_REFERENCE_GREEN, size=2.7, alpha=DISPLAY_REFERENCE_ALPHA, label="held-out reference")

    def draw_full(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.full_points, OBSERVED_GREY, alpha=DISPLAY_REFERENCE_ALPHA, label="raw WL127 visible evidence")
        _plot_surface(axis, full_surface_grid, FULL_REPRESENTATIVE_BLUE)
        _plot_points(axis, full_surface_grid.reshape(-1, 3), FULL_REPRESENTATIVE_BLUE, size=1.2, alpha=0.92, max_points=5000, label="full evaluation-only NURBS representative")

    def draw_retained(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.retained_points, OBSERVED_GREY, size=2.7, alpha=DISPLAY_POINT_ALPHA, label="retained raw evidence")
        _plot_surface(axis, retained_surface_grid, RETAINED_REPRESENTATIVE_ORANGE)
        _plot_points(axis, retained_surface_grid.reshape(-1, 3), RETAINED_REPRESENTATIVE_ORANGE, size=1.2, alpha=0.95, max_points=5000, label="retained-only NURBS representative")

    def draw_normals(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.retained_points, OBSERVED_GREY, size=2.7, alpha=DISPLAY_POINT_ALPHA, label="raw retained evidence")
        _plot_surface(axis, retained_surface_grid, FULL_REPRESENTATIVE_BLUE, alpha=0.25)
        if raw_normals is not None:
            raw_points, normals = raw_normals
            display = _plot_coords(raw_points)
            normal_display = normals[:, [0, 2, 1]]
            axis.quiver(display[:, 0], display[:, 1], display[:, 2], normal_display[:, 0], normal_display[:, 1], normal_display[:, 2], length=0.06, normalize=True, color=RAW_NORMAL_RED, linewidth=0.65, alpha=0.95, label="raw PCA normals")
        rep_points = deterministic_subsample(case.retained_representative.sampled_points, 900)
        rep_normals = case.retained_representative.sampled_normals[np.linspace(0, len(case.retained_representative.sampled_normals) - 1, len(rep_points), dtype=np.int64)]
        display = _plot_coords(rep_points)
        normal_display = rep_normals[:, [0, 2, 1]]
        axis.quiver(display[:, 0], display[:, 1], display[:, 2], normal_display[:, 0], normal_display[:, 1], normal_display[:, 2], length=0.08, normalize=True, color=REP_NORMAL_BLUE, linewidth=0.8, alpha=0.95, label="analytic representative normals")

    def draw_continuation(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.retained_points, OBSERVED_GREY, size=2.7, alpha=DISPLAY_POINT_ALPHA, label="raw retained visible evidence")
        _plot_surface(axis, retained_surface_grid, RETAINED_REPRESENTATIVE_ORANGE, alpha=0.25)
        _plot_surface(axis, continuation_grid, PREDICTED_CYAN, alpha=DISPLAY_PREDICTION_ALPHA)
        _plot_points(axis, continuation_grid.reshape(-1, 3), PREDICTED_CYAN, size=1.5, alpha=DISPLAY_POINT_ALPHA, max_points=6000, label="representative-frame continuation")
        _plot_points(axis, case.continuation.frontier_raw, FRONTIER_YELLOW, size=12.0, alpha=1.0, max_points=256, label="frozen physical frontier")

    def draw_raw_compare(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.retained_points, OBSERVED_GREY, size=2.7, alpha=DISPLAY_POINT_ALPHA, label="retained raw evidence")
        _plot_points(axis, case.withheld_points, RAW_REFERENCE_GREEN, size=2.7, alpha=DISPLAY_REFERENCE_ALPHA, label="raw held-out reference")
        _plot_surface(axis, continuation_grid, PREDICTED_CYAN, alpha=DISPLAY_PREDICTION_ALPHA)
        _plot_points(axis, continuation_grid.reshape(-1, 3), PREDICTED_CYAN, size=1.5, alpha=DISPLAY_POINT_ALPHA, max_points=6000, label="predicted continuation")
        _plot_points(axis, case.continuation.frontier_raw, FRONTIER_YELLOW, size=12.0, alpha=1.0, max_points=256, label="physical frontier")

    def draw_macro_compare(axis: Any, _figure: Any) -> None:
        _plot_points(axis, case.retained_points, OBSERVED_GREY, size=2.7, alpha=DISPLAY_POINT_ALPHA, label="retained raw evidence")
        _plot_surface(axis, macro_grid, FULL_REPRESENTATIVE_BLUE, alpha=0.38)
        _plot_points(axis, macro_points, FULL_REPRESENTATIVE_BLUE, size=1.4, alpha=0.94, max_points=6000, label="held-out full NURBS representative")
        _plot_surface(axis, continuation_grid, PREDICTED_CYAN, alpha=DISPLAY_PREDICTION_ALPHA)
        _plot_points(axis, continuation_grid.reshape(-1, 3), PREDICTED_CYAN, size=1.5, alpha=DISPLAY_POINT_ALPHA, max_points=6000, label="predicted continuation")
        _plot_points(axis, case.continuation.frontier_raw, FRONTIER_YELLOW, size=12.0, alpha=1.0, max_points=256, label="physical frontier")

    _save_raw_figure(paths["raw_visible_surface"], f"{case.config.name}: raw WL127 Visible Surface ROI", all_base, draw_raw)
    _save_raw_figure(paths["raw_vs_full_representative"], f"{case.config.name}: raw evidence vs full evaluation-only representative", all_base, draw_full)
    _save_raw_figure(paths["retained_raw_vs_retained_representative"], f"{case.config.name}: retained raw vs retained-only representative", all_base, draw_retained)
    _save_raw_figure(paths["raw_normals_vs_representative_normals"], f"{case.config.name}: raw PCA normals vs analytic representative normals", all_base, draw_normals)
    _save_raw_figure(paths["representative_continuation"], f"{case.config.name}: representative-frame short continuation", all_base, draw_continuation)
    _save_raw_figure(paths["continuation_vs_raw_reference"], f"{case.config.name}: continuation vs raw held-out reference", all_base, draw_raw_compare)
    _save_raw_figure(paths["continuation_vs_macro_reference"], f"{case.config.name}: continuation vs macro held-out representative", all_base, draw_macro_compare, view=SECOND_VIEW)
    return {key: str(value) for key, value in paths.items()}


def write_case_geometry(case: CaseAudit, output_root: Path) -> dict[str, str]:
    case_root = output_root / case.config.name
    case_root.mkdir(parents=True, exist_ok=True)
    npz_path = case_root / "scale_separated_geometry.npz"
    np.savez_compressed(
        npz_path,
        raw_full_reference_points=case.full_points.astype(np.float32),
        retained_raw_points=case.retained_points.astype(np.float32),
        withheld_raw_reference_points_evaluation_only=case.withheld_points.astype(np.float32),
        full_representative_points=case.full_representative.sampled_points.astype(np.float32),
        retained_representative_points=case.retained_representative.sampled_points.astype(np.float32),
        retained_representative_normals=case.retained_representative.sampled_normals.astype(np.float32),
        full_control_grid=case.full_representative.control_grid.astype(np.float32),
        retained_control_grid=case.retained_representative.control_grid.astype(np.float32),
        physical_frontier_raw=case.continuation.frontier_raw.astype(np.float32),
        physical_frontier_representative=case.continuation.frontier_representative.astype(np.float32),
        predicted_continuation_points=case.continuation.points_grid.astype(np.float32),
        predicted_continuation_normals=case.continuation.normals_grid.astype(np.float32),
    )
    raw_path = case_root / "raw_visible_surface.ply"
    retained_path = case_root / "retained_visible_surface.ply"
    withheld_path = case_root / "withheld_reference_evaluation_only.ply"
    full_rep_path = case_root / "full_evaluation_only_representative.ply"
    retained_rep_path = case_root / "retained_representative.ply"
    predicted_path = case_root / "representative_continuation.ply"
    frontier_path = case_root / "physical_frontier.ply"
    _write_ply(raw_path, case.full_points, color=(120, 128, 140))
    _write_ply(retained_path, case.retained_points, color=(120, 128, 140))
    _write_ply(withheld_path, case.withheld_points, color=(24, 185, 74))
    _write_ply(full_rep_path, case.full_representative.sampled_points, color=(20, 76, 210))
    _write_ply(retained_rep_path, case.retained_representative.sampled_points, color=(235, 96, 20))
    _write_ply(predicted_path, case.continuation.points_grid, faces=_grid_faces(*case.continuation.points_grid.shape[:2]), color=(0, 184, 209))
    _write_ply(frontier_path, case.continuation.frontier_raw, color=(250, 198, 12))
    return {
        "npz": str(npz_path),
        "raw_visible_surface_ply": str(raw_path),
        "retained_visible_surface_ply": str(retained_path),
        "withheld_reference_evaluation_only_ply": str(withheld_path),
        "full_evaluation_only_representative_ply": str(full_rep_path),
        "retained_representative_ply": str(retained_rep_path),
        "representative_continuation_ply": str(predicted_path),
        "physical_frontier_ply": str(frontier_path),
    }


def _case_report(case: CaseAudit, h: float) -> dict[str, Any]:
    return {
        "roi": case.config.as_json(),
        "visible_fraction": float(len(case.retained_points) / max(len(case.full_points), 1)),
        "withheld_fraction": float(len(case.withheld_points) / max(len(case.full_points), 1)),
        "raw_surface_scale_audit": _jsonable(case.raw_scale_audit),
        "full_evaluation_only_representative": {
            "fit_contract": "full visible ROI, including later held-out region; never used for prediction",
            "settings": FIXED_FITTER_CONFIG,
            "fitting_residual": _summary(case.full_representative.fitting_residuals, over_h=h),
            "fit_input_sha256": case.full_representative.fit_input_sha256,
        },
        "retained_representative": {
            "fit_contract": "retained visible points only",
            "settings": FIXED_FITTER_CONFIG,
            "stats": _jsonable(_representative_stats(case.retained_representative, case.retained_points, h)),
        },
        "visible_macro_shape_audit": case.retained_shape_audit,
        "physical_frontier_mapping": _frontier_mapping_report(case.continuation, case.retained_points, h),
        "representative_frame_continuation": {
            "mechanism": "short first-order tangent-plane continuation from the projected frozen physical frontier",
            "uses_retained_representative_only": True,
            "uses_full_representative_for_construction": False,
            "adds_curvature": False,
            "continuation_extent_world": float(case.continuation.l_values[-1]),
            "continuation_extent_rule": "exact physical holdout length from the fixed ROI and u_cut",
            "control_grid_shape": list(case.retained_representative.control_grid.shape),
        },
        "raw_held_out_metric": _jsonable(case.raw_metrics),
        "macro_held_out_representative_metric": _jsonable(case.macro_metrics),
        "continuation_assessment": _continuation_assessment(case),
        "visuals": case.visuals,
        "geometry": case.geometry,
    }


def _continuation_assessment(case: CaseAudit) -> dict[str, Any]:
    """Classify the frozen result after evaluation; never feeds construction."""

    raw_distance = case.raw_metrics["point_to_surface_distance"]
    raw_coverage = case.raw_metrics["coverage"]
    macro_distance = case.macro_metrics["point_to_surface_distance"]
    macro_coverage = case.macro_metrics["coverage"]
    raw_pass = bool(
        raw_distance["median_over_h"] <= CONTINUATION_MEDIAN_LIMIT_OVER_H
        and raw_coverage["fraction_le_2h"] >= CONTINUATION_MIN_COVERAGE_LE_2H
    )
    macro_pass = bool(
        macro_distance["median_over_h"] <= CONTINUATION_MEDIAN_LIMIT_OVER_H
        and macro_coverage["fraction_le_2h"] >= CONTINUATION_MIN_COVERAGE_LE_2H
    )
    return {
        "status": "PASS" if raw_pass and macro_pass else "INSUFFICIENT",
        "criterion": {
            "median_over_h_at_most": CONTINUATION_MEDIAN_LIMIT_OVER_H,
            "coverage_fraction_le_2h_at_least": CONTINUATION_MIN_COVERAGE_LE_2H,
        },
        "raw_reference_pass": raw_pass,
        "macro_reference_pass": macro_pass,
        "used_to_construct_or_tune_prediction": False,
    }


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    field = np.load(arguments.field_cache, allow_pickle=True)
    mesh = np.load(arguments.mesh_cache, allow_pickle=True)
    h = float(field["h"])
    mu = float(field["mu"])
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    configs = [LEG_CASE, CURVED_RIM_CASE]
    cases: list[CaseAudit] = []
    failures: list[dict[str, str]] = []
    for config in configs:
        try:
            roi_points = _select_box(vertices, config.roi_box, int(arguments.max_patch_points))
            holdout = _build_case_holdout(roi_points, config)
            if len(holdout.retained_points) < 32 or len(holdout.withheld_points) < 32:
                raise ValueError("fixed ROI does not contain enough retained and withheld rows")
            retained = deterministic_subsample(holdout.retained_points, int(arguments.max_patch_points))
            withheld = deterministic_subsample(holdout.withheld_points, int(arguments.max_reference_points))
            full = deterministic_subsample(holdout.full_points, int(arguments.max_patch_points))
            scale_audit = raw_surface_scale_audit(retained, h)
            full_rep = fit_existing_nurbs(full, config, h, retained_domain=False, max_fit_points=int(arguments.max_fit_points), device_name=arguments.device)
            retained_rep = fit_existing_nurbs(retained, config, h, retained_domain=True, max_fit_points=int(arguments.max_fit_points), device_name=arguments.device)
            shape_audit = _visible_macro_shape_audit(retained_rep, retained, h)
            if shape_audit["status"] != "PASS":
                # Honor Worklog 138 STOP CONDITION 1.  No continuation or
                # withheld error is produced for a failed macro-shape audit.
                raise RuntimeError("retained NURBS representative failed the fixed visible macro-shape audit")
            continuation = _representative_frontier_continuation(holdout, retained_rep, h=h)
            reference_normals = estimate_point_normals(withheld, k=min(20, len(withheld)))
            raw_metrics = _reference_metrics(withheld, reference_normals, continuation, h, config, label="raw held-out reconstructed visible evidence only")
            macro_points, macro_normals, _macro_uv = _representative_reference(full_rep, config)
            macro_metrics = _reference_metrics(macro_points, macro_normals, continuation, h, config, label="held-out reconstructed surface representative only")
            raw_frontier_normals = reference_normals if reference_normals is not None else np.empty((0, 3), dtype=np.float64)
            case = CaseAudit(
                config=config,
                full_points=full,
                retained_points=retained,
                withheld_points=withheld,
                holdout=holdout,
                raw_scale_audit=scale_audit,
                full_representative=full_rep,
                retained_representative=retained_rep,
                retained_shape_audit=shape_audit,
                continuation=continuation,
                raw_metrics=raw_metrics,
                macro_metrics=macro_metrics,
                raw_frontier_normals=raw_frontier_normals,
                visuals={},
                geometry={},
            )
            case.visuals = write_case_visuals(case, output_root)
            case.geometry = write_case_geometry(case, output_root)
            (output_root / config.name / "case_report.json").write_text(json.dumps(_jsonable(_case_report(case, h)), indent=2), encoding="utf-8")
            cases.append(case)
        except Exception as error:
            failures.append({"case": config.name, "error": repr(error)})

    report: dict[str, Any] = {
        "batch": "Worklog 138 scale-separated Visible Surface representative closure",
        "status": "NON_CANONICAL_DIAGNOSTIC_DEMO",
        "INTENT ALIGNMENT": {
            "question": "Is raw WL127 evidence too fine-scale/noisy for structural continuation, and does fixed NURBS recover macro geometry?",
            "raw_visible_surface_role": "faithful renderer-grounded evidence carrier; never smoothed in place",
            "surface_representative_role": "smooth geometric reasoning aid only",
            "occluded_surface_solved": False,
        },
        "IMPLEMENTATION FIDELITY": {
            "canonical_code_modified": False,
            "historical_worklogs_modified": False,
            "wl127_mesh_modified": False,
            "wl136_holdout_and_frontier_modified": False,
            "h_or_mu_tuned": False,
            "nurbs_redesigned": False,
            "manual_choices": ["two fixed ROIs, local axes, bounds, and u_cut", "fixed raw display opacity/point size"],
            "heuristics": ["local PCA normal audit", "geometric projection to retained NURBS", "first-order representative tangent-plane continuation"],
            "full_reference_information_used": ["fixed ROI selection by direct inspection", "full mesh only supplies declared raw evaluation target and evaluation-only full representative"],
            "withheld_reference_roles": ["raw held-out evaluation", "macro held-out representative evaluation", "final comparison visualization"],
            "withheld_geometry_entered_retained_fit": False,
            "withheld_geometry_entered_continuation": False,
            "nurbs_uv_edge_used_as_physical_termination": False,
            "display_only_thinning": {"voxel_world": DISPLAY_VOXEL_WORLD, "metrics_unchanged": True},
            "isolated_module": str(Path(__file__).resolve()),
        },
        "RAW SURFACE SCALE AUDIT": {case.config.name: _jsonable(case.raw_scale_audit) for case in cases},
        "FULL EVALUATION-ONLY REPRESENTATIVE": {case.config.name: {
            "settings": FIXED_FITTER_CONFIG,
            "fit_input_sha256": case.full_representative.fit_input_sha256,
            "fitting_residual": _summary(case.full_representative.fitting_residuals, over_h=h),
            "used_for_retained_prediction": False,
        } for case in cases},
        "RETAINED REPRESENTATIVE": {case.config.name: _jsonable(_representative_stats(case.retained_representative, case.retained_points, h)) for case in cases},
        "VISIBLE MACRO-SHAPE AUDIT": {case.config.name: case.retained_shape_audit for case in cases},
        "PHYSICAL FRONTIER -> REPRESENTATIVE MAPPING": {case.config.name: _frontier_mapping_report(case.continuation, case.retained_points, h) for case in cases},
        "RAW NORMAL vs REPRESENTATIVE NORMAL": {case.config.name: _frontier_mapping_report(case.continuation, case.retained_points, h)["raw_pca_vs_representative_normal_angle"] for case in cases},
        "REPRESENTATIVE-FRAME CONTINUATION": {case.config.name: {
            "extent_world": float(case.continuation.l_values[-1]),
            "mechanism": "retained representative analytic tangent plane at frozen physical frontier",
            "withheld_reference_used": False,
        } for case in cases},
        "RAW HELD-OUT METRIC": {case.config.name: _jsonable(case.raw_metrics) for case in cases},
        "MACRO HELD-OUT REPRESENTATIVE METRIC": {case.config.name: _jsonable(case.macro_metrics) for case in cases},
        "RAW VISUALIZATION PATHS": {case.config.name: case.visuals for case in cases},
        "cases": [_case_report(case, h) for case in cases],
        "failures": failures,
        "PROMOTED": {
            "status": "SCALE_SEPARATED_REPRESENTATIVE_DIAGNOSTIC_ONLY",
            "meaning": "raw Visible Surface Evidence and smooth Surface Representative are reported separately",
        },
        "RETAINED": {
            "status": "RAW_EVIDENCE_CARRIER_AND_FIXED_NURBS_AUDIT",
            "meaning": "raw mesh remains unchanged; NURBS is only a non-canonical representative hypothesis",
        },
        "REJECTED": {
            "status": "RAW_MESH_PCA_AS_DIRECT_STRUCTURAL_FRAME",
            "meaning": "raw local PCA normals are diagnostics, not the continuation frame",
        },
        "OPEN": {
            "status": "REPRESENTATIVE_SCALE_LONG_RANGE_PRIOR_TERMINATION_OCCLUSION",
            "meaning": "best scale, hidden-shape prior, junction transfer, and persistent occlusion remain open",
        },
        "inputs": {"mesh_cache": str(arguments.mesh_cache), "field_cache": str(arguments.field_cache), "h": h, "mu": mu, "h_source": "WL127 field.npz read-only"},
        "manual_demo_configuration": {config.name: config.as_json() for config in configs},
        "post_evaluation_continuation_criteria": {
            "median_over_h_at_most": CONTINUATION_MEDIAN_LIMIT_OVER_H,
            "coverage_fraction_le_2h_at_least": CONTINUATION_MIN_COVERAGE_LE_2H,
            "construction_or_tuning_use": False,
        },
    }
    all_shape_pass = bool(len(cases) == 2 and all(case.retained_shape_audit["status"] == "PASS" for case in cases))
    all_continuation_pass = bool(len(cases) == 2 and all(_continuation_assessment(case)["status"] == "PASS" for case in cases))
    if all_shape_pass and all_continuation_pass:
        report["meeting_verdict"] = "A"
        report["meeting_verdict_description"] = "A. Scale-separated representative and fixed short continuation are useful non-canonical diagnostic evidence."
    elif all_shape_pass:
        report["meeting_verdict"] = "B"
        report["meeting_verdict_description"] = "B. The fixed NURBS representative is useful for visible macro-shape reasoning, but continuation remains insufficient on the held-out references."
    elif not cases:
        report["meeting_verdict"] = "C"
        report["meeting_verdict_description"] = "C. The current NURBS representative is insufficient under the fixed visible macro-shape audit."
    else:
        report["meeting_verdict"] = "D"
        report["meeting_verdict_description"] = "D. Inconclusive because at least one mandatory case did not complete the fixed audit."
    report_path = output_root / "scale_separated_visible_surface_representative_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    lines = [
        "# Worklog 138 scale-separated Visible Surface representative audit",
        "",
        "이 폴더는 canonical raw WL127 Visible Surface를 수정하지 않고, 기존 NURBS fitter를",
        "full evaluation-only representative와 retained-only representative로 분리해 점검한",
        "비정규 diagnostic track이다. raw point는 display-only voxel thinning 뒤에도 near-opaque로",
        "그려지며, 저장된 geometry와 metric population은 바뀌지 않는다.",
        "",
        "## 출력",
        "",
        "각 case 폴더에 `raw_visible_surface.png`, `raw_vs_full_representative.png`,",
        "`retained_raw_vs_retained_representative.png`, `raw_normals_vs_representative_normals.png`,",
        "`representative_continuation.png`, `continuation_vs_raw_reference.png`,",
        "`continuation_vs_macro_reference.png`와 PLY/NPZ가 있다.",
        "",
        f"## 판정: {report['meeting_verdict']}",
        "",
        report["meeting_verdict_description"],
        "",
        "full representative는 withheld region을 포함해 만들었지만 evaluation-only이며 retained fit와 continuation에 들어가지 않는다.",
        "물리 frontier는 frozen world-space holdout frontier이고 NURBS UV edge를 termination으로 사용하지 않는다.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-cache", type=Path, default=WORKLOG_127_MESH)
    parser.add_argument("--field-cache", type=Path, default=WORKLOG_127_FIELD)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/138_scale_separated_visible_surface_representative")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-fit-points", type=int, default=12000)
    parser.add_argument("--max-reference-points", type=int, default=12000)
    parser.add_argument("--max-patch-points", type=int, default=24000)
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_demo(build_arg_parser().parse_args(argv))
    print(json.dumps({"meeting_verdict": report["meeting_verdict"], "cases": len(report["cases"]), "failures": report["failures"]}, indent=2))
    return 0 if report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
