"""Worklog 139: physical-chart-constrained surface representative closure.

This is an isolated, non-canonical research/demo path.  It replays the exact
Worklog 138 unconstrained NURBS as a frozen baseline, then fits a tensor-product
B-spline graph whose physical ``u`` and ``v`` coordinates are fixed by the
existing ROI chart.  Only the normal-coordinate scalar field is solved.

Retained and full fits are explicit, separate roles.  The full fit is
evaluation-only and cannot be passed to continuation construction.  Raw
withheld XYZ is never consumed by the retained graph fit or prediction.
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
    Holdout,
    Prediction,
    _grid_faces,
    _surface_area,
    _write_ply,
    build_self_continuation,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    deterministic_indices,
    deterministic_subsample,
    estimate_point_normals,
)
from devtools.demo.scale_separated_visible_surface_representative import (  # noqa: E402
    CURVED_RIM_CASE,
    FIXED_FITTER_CONFIG,
    LEG_CASE,
    FittedRepresentative,
    RepresentativeCaseConfig,
    RepresentativeContinuation,
    _build_case_holdout,
    _case_coordinates,
    _normalised_axis,
    _representative_frontier_continuation,
    _sha256_rows,
    fit_existing_nurbs,
)
from osn_gs.surface.torch_nurbs import (  # noqa: E402
    TorchNURBSSurface,
    _second_difference_penalty,
)


OUTPUT_ROOT = REPO_ROOT / "output/physical_chart_surface_representative"
WL138_MODULE = REPO_ROOT / "devtools/demo/scale_separated_visible_surface_representative.py"
WL138_CONFIRMED_ROOT = REPO_ROOT / "output/confirmed/scale_separated_visible_surface_representative"
CANDIDATE_B_ARCHIVE = REPO_ROOT / "output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"

# Frozen historical representation settings.  The scalar graph solve uses the
# same resolution, degree, and regularization values.  UV correction rounds are
# intentionally inapplicable because physical UV is the contract being fixed.
GRAPH_RESOLUTION_U = int(FIXED_FITTER_CONFIG["resolution_u"])
GRAPH_RESOLUTION_V = int(FIXED_FITTER_CONFIG["resolution_v"])
GRAPH_DEGREE_U = int(FIXED_FITTER_CONFIG["degree_u"])
GRAPH_DEGREE_V = int(FIXED_FITTER_CONFIG["degree_v"])
GRAPH_SMOOTHNESS_LAMBDA = float(FIXED_FITTER_CONFIG["smoothness_lambda"])
GRAPH_TIKHONOV_LAMBDA = float(FIXED_FITTER_CONFIG["tikhonov_lambda"])

GRAPH_BIN_SCALE_H = 4.0
GRAPH_MODE_SEPARATION_H = 3.0
GRAPH_MODE_MIN_SIDE_POINTS = 3
GRAPH_MIN_ELIGIBLE_BINS = 8
GRAPH_MAX_MULTIMODE_FRACTION = 0.10

SAMPLE_COUNT_U = 96
SAMPLE_COUNT_V = 40
DISPLAY_VOXEL_WORLD = 0.02
DISPLAY_POINT_ALPHA = 0.98
DISPLAY_REFERENCE_ALPHA = 0.99
DISPLAY_POINT_SIZE = 3.4
DISPLAY_SURFACE_ALPHA = 0.58
FIXED_VIEW = {"elev": 25.0, "azim": -58.0}

# Frozen post-evaluation interpretation only.  These values are inherited from
# WL138 and never feed fitting, chart construction, or continuation geometry.
USEFUL_MEDIAN_LIMIT_OVER_H = 5.0
USEFUL_COVERAGE_LE_2H_MIN = 0.20

RAW_GREY = (0.43, 0.45, 0.49)
WITHHELD_GREEN = (0.05, 0.68, 0.24)
UNCONSTRAINED_BLUE = (0.10, 0.28, 0.82)
CONSTRAINED_ORANGE = (0.96, 0.37, 0.04)
FRONTIER_YELLOW = (0.98, 0.76, 0.02)
RAW_CONTINUATION_MAGENTA = (0.78, 0.08, 0.62)
WL138_CONTINUATION_PURPLE = (0.42, 0.12, 0.78)
CHART_CONTINUATION_CYAN = (0.00, 0.70, 0.82)
NORMAL_BLUE = (0.04, 0.24, 0.86)


@dataclass
class GraphnessAudit:
    status: str
    bin_size_world: float
    total_chart_bins: int
    occupied_bins: int
    eligible_mode_bins: int
    multimode_bins: int
    multimode_fraction: float
    within_bin_spreads: np.ndarray
    chart_coverage: float
    gross_hole_count: int
    occupied_bin_indices: np.ndarray
    multimode_bin_indices: np.ndarray


@dataclass
class PhysicalChartRepresentative:
    role: str
    surface: TorchNURBSSurface
    domain_u: tuple[float, float]
    domain_v: tuple[float, float]
    fit_points: np.ndarray
    fit_uv: np.ndarray
    fit_input_sha256: str
    scalar_control_grid: np.ndarray
    control_grid: np.ndarray
    sampled_uv: np.ndarray
    sampled_points: np.ndarray
    sampled_normals: np.ndarray
    fitting_residuals: np.ndarray
    physical_u_precision_error: np.ndarray
    physical_v_precision_error: np.ndarray


@dataclass
class ChartContinuation:
    source_role: str
    frontier_raw: np.ndarray
    frontier_representative: np.ndarray
    frontier_uv: np.ndarray
    frontier_normals: np.ndarray
    frontier_tangent_v: np.ndarray
    directions: np.ndarray
    points_grid: np.ndarray
    normals_grid: np.ndarray
    l_values: np.ndarray
    projection_gaps: np.ndarray
    boundary_position_gaps: np.ndarray
    boundary_normal_angles: np.ndarray


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact_manifest(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        str(path.relative_to(root)).replace("\\", "/"): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_frozen_wl138_case(
    root: Path,
    config: RepresentativeCaseConfig,
    *,
    max_fit_points: int,
    device_name: str,
) -> dict[str, Any]:
    """Load exact confirmed WL138 populations/control grid without rewriting it."""

    import torch

    case_root = Path(root) / config.name
    geometry_path = case_root / "scale_separated_geometry.npz"
    report_path = case_root / "case_report.json"
    geometry = np.load(geometry_path, allow_pickle=True)
    historical_report = json.loads(report_path.read_text(encoding="utf-8"))
    full = np.asarray(geometry["raw_full_reference_points"], dtype=np.float64)
    retained = np.asarray(geometry["retained_raw_points"], dtype=np.float64)
    withheld = np.asarray(geometry["withheld_raw_reference_points_evaluation_only"], dtype=np.float64)
    physical_u = _case_coordinates(full, config)[:, 0]
    retained_mask = physical_u <= float(config.u_cut) + 1.0e-12
    withheld_mask = physical_u > float(config.u_cut) + 1.0e-12
    if int(np.sum(retained_mask)) != len(retained) or int(np.sum(withheld_mask)) != len(withheld):
        raise AssertionError("confirmed WL138 full population does not reproduce retained/withheld counts")
    if not np.allclose(full[retained_mask], retained, atol=1.0e-7, rtol=0.0):
        raise AssertionError("confirmed WL138 retained population/order changed")
    if not np.allclose(full[withheld_mask], withheld, atol=1.0e-7, rtol=0.0):
        raise AssertionError("confirmed WL138 withheld population/order changed")
    holdout = Holdout(
        name=config.name + "_confirmed_wl138_physical_holdout",
        full_points=full,
        retained_points=retained,
        withheld_points=withheld,
        retained_mask=retained_mask,
        withheld_mask=withheld_mask,
        u_axis=_normalised_axis(config.u_axis),
        v_axis=_normalised_axis(config.v_axis),
        n_axis=_normalised_axis(config.n_axis),
        u_bounds=tuple(map(float, config.u_bounds)),
        v_bounds=tuple(map(float, config.v_bounds)),
        n_bounds=tuple(map(float, config.n_bounds)),
        u_cut=float(config.u_cut),
        permitted_volume=config.roi_box,
    )
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name)
    control = torch.as_tensor(geometry["retained_control_grid"], dtype=torch.float32, device=device)
    surface = TorchNURBSSurface(
        control_grid=control,
        weights=torch.ones(control.shape[:2], dtype=torch.float32, device=device),
        degree_u=GRAPH_DEGREE_U,
        degree_v=GRAPH_DEGREE_V,
        observed_v_max=1.0,
    )
    fit_points = retained[deterministic_indices(len(retained), int(max_fit_points))].copy()
    baseline = FittedRepresentative(
        surface=surface,
        fit_points=fit_points,
        fit_uv=np.empty((0, 2), dtype=np.float64),
        sampled_points=np.asarray(geometry["retained_representative_points"], dtype=np.float64),
        sampled_normals=np.asarray(geometry["retained_representative_normals"], dtype=np.float64),
        control_grid=np.asarray(geometry["retained_control_grid"], dtype=np.float64),
        fitting_residuals=np.empty((0,), dtype=np.float64),
        fit_input_sha256=_sha256_rows(fit_points),
    )
    historical_stats = historical_report["retained_representative"]["stats"]
    if baseline.fit_input_sha256 != historical_stats["fit_input_sha256"]:
        raise AssertionError("confirmed WL138 fitter population SHA changed")

    replay = _representative_frontier_continuation(holdout, baseline, h=float(historical_stats["h_world"]))
    saved_frontier_raw = np.asarray(geometry["physical_frontier_raw"], dtype=np.float64)
    saved_frontier_rep = np.asarray(geometry["physical_frontier_representative"], dtype=np.float64)
    saved_prediction = np.asarray(geometry["predicted_continuation_points"], dtype=np.float64)
    saved_normals = np.asarray(geometry["predicted_continuation_normals"], dtype=np.float64)
    replay_validation = {
        "frontier_raw_max_gap": float(np.max(np.linalg.norm(replay.frontier_raw - saved_frontier_raw, axis=1))),
        "frontier_representative_max_gap": float(np.max(np.linalg.norm(replay.frontier_representative - saved_frontier_rep, axis=1))),
        "prediction_max_gap": float(np.max(np.linalg.norm(replay.points_grid - saved_prediction, axis=2))),
        "normal_max_gap": float(np.max(np.linalg.norm(replay.normals_grid - saved_normals, axis=2))),
    }
    # Geometry used as the historical A/B arm is the exact confirmed artifact.
    replay.frontier_raw = saved_frontier_raw
    replay.frontier_representative = saved_frontier_rep
    replay.points_grid = saved_prediction
    replay.normals_grid = saved_normals
    replay.projection_gaps = np.linalg.norm(saved_frontier_rep - saved_frontier_raw, axis=1)
    replay.boundary_position_gaps = np.linalg.norm(saved_prediction[0] - saved_frontier_rep, axis=1)
    replay.boundary_normal_angles = _normal_angles(saved_normals[0], replay.frontier_normals)
    return {
        "holdout": holdout,
        "full": full,
        "retained": retained,
        "withheld": withheld,
        "baseline": baseline,
        "baseline_continuation": replay,
        "historical_stats": historical_stats,
        "replay_validation": replay_validation,
        "geometry_path": str(geometry_path),
        "report_path": str(report_path),
    }

def _summary(values: np.ndarray, *, h: float | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "UNAVAILABLE", "samples": 0}
    report: dict[str, Any] = {
        "status": "MEASURED",
        "samples": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
    }
    if h is not None:
        report.update({
            "median_over_h": float(np.median(values) / h),
            "p95_over_h": float(np.percentile(values, 95) / h),
            "mean_over_h": float(np.mean(values) / h),
        })
    return report


def _normal_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.clip(np.linalg.norm(first, axis=1, keepdims=True), 1.0e-12, None)
    second = second / np.clip(np.linalg.norm(second, axis=1, keepdims=True), 1.0e-12, None)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(first * second, axis=1)), 0.0, 1.0)))


def _oriented_mean_normal(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    if not len(normals):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    reference = normals[0] / max(float(np.linalg.norm(normals[0])), 1.0e-12)
    aligned = np.where((normals @ reference)[:, None] < 0.0, -normals, normals)
    mean = aligned.mean(axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1.0e-12)


def _grid_components(indices: np.ndarray) -> int:
    remaining = {tuple(map(int, row)) for row in np.asarray(indices, dtype=np.int64)}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            i, j = stack.pop()
            for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    return components


def audit_raw_graphness(points: np.ndarray, config: RepresentativeCaseConfig, h: float) -> GraphnessAudit:
    """Audit retained raw rows only for compatibility with ``n=f(u,v)``."""

    points = np.asarray(points, dtype=np.float64)
    coordinates = _case_coordinates(points, config)
    bin_size = float(GRAPH_BIN_SCALE_H * h)
    u0, u1 = config.u_bounds[0], config.u_cut
    v0, v1 = config.v_bounds
    bins_u = max(1, int(math.ceil((u1 - u0) / bin_size)))
    bins_v = max(1, int(math.ceil((v1 - v0) / bin_size)))
    iu = np.clip(np.floor((coordinates[:, 0] - u0) / bin_size).astype(np.int64), 0, bins_u - 1)
    iv = np.clip(np.floor((coordinates[:, 1] - v0) / bin_size).astype(np.int64), 0, bins_v - 1)
    keys = np.column_stack([iu, iv])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    spreads: list[float] = []
    eligible = 0
    multimode: list[np.ndarray] = []
    for index, key in enumerate(unique):
        values = np.sort(coordinates[inverse == index, 2])
        if len(values) >= 2:
            spreads.append(float(np.percentile(values, 95) - np.percentile(values, 5)))
        if len(values) < 2 * GRAPH_MODE_MIN_SIDE_POINTS:
            continue
        eligible += 1
        gaps = np.diff(values)
        split_indices = np.arange(1, len(values))
        valid = (
            (split_indices >= GRAPH_MODE_MIN_SIDE_POINTS)
            & ((len(values) - split_indices) >= GRAPH_MODE_MIN_SIDE_POINTS)
            & (gaps >= GRAPH_MODE_SEPARATION_H * h)
        )
        if bool(np.any(valid)):
            multimode.append(key.copy())
    occupied_set = {tuple(map(int, row)) for row in unique}
    holes = 0
    for i in range(1, bins_u - 1):
        for j in range(1, bins_v - 1):
            if (i, j) in occupied_set:
                continue
            if all(neighbour in occupied_set for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))):
                holes += 1
    multimode_fraction = float(len(multimode) / max(eligible, 1))
    if eligible < GRAPH_MIN_ELIGIBLE_BINS:
        status = "INCONCLUSIVE_INSUFFICIENT_OCCUPIED_BINS"
    elif multimode_fraction > GRAPH_MAX_MULTIMODE_FRACTION:
        status = "FAIL_MATERIALLY_MULTIVALUED"
    else:
        status = "PASS_GRAPH_LIKE"
    return GraphnessAudit(
        status=status,
        bin_size_world=bin_size,
        total_chart_bins=int(bins_u * bins_v),
        occupied_bins=int(len(unique)),
        eligible_mode_bins=int(eligible),
        multimode_bins=int(len(multimode)),
        multimode_fraction=multimode_fraction,
        within_bin_spreads=np.asarray(spreads, dtype=np.float64),
        chart_coverage=float(len(unique) / max(bins_u * bins_v, 1)),
        gross_hole_count=int(holes),
        occupied_bin_indices=unique.astype(np.int64),
        multimode_bin_indices=np.asarray(multimode, dtype=np.int64).reshape(-1, 2),
    )


def graphness_report(audit: GraphnessAudit, h: float) -> dict[str, Any]:
    return {
        "status": audit.status,
        "input_population": "retained raw geometry only",
        "bin_rule": f"fixed physical chart bin width = {GRAPH_BIN_SCALE_H}h",
        "bin_size_world": audit.bin_size_world,
        "mode_rule": f"adjacent n gap >= {GRAPH_MODE_SEPARATION_H}h with >= {GRAPH_MODE_MIN_SIDE_POINTS} samples on each side",
        "total_chart_bins": audit.total_chart_bins,
        "occupied_chart_bins": audit.occupied_bins,
        "eligible_mode_bins": audit.eligible_mode_bins,
        "multimode_bins": audit.multimode_bins,
        "multimode_fraction": audit.multimode_fraction,
        "within_bin_n_spread": _summary(audit.within_bin_spreads, h=h),
        "chart_coverage": audit.chart_coverage,
        "gross_four_neighbour_holes": audit.gross_hole_count,
        "withheld_reference_used": False,
    }


def _physical_domain(config: RepresentativeCaseConfig, role: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if role == "retained_construction":
        return (float(config.u_bounds[0]), float(config.u_cut)), tuple(map(float, config.v_bounds))
    if role == "full_evaluation_only":
        return tuple(map(float, config.u_bounds)), tuple(map(float, config.v_bounds))
    raise ValueError(f"unknown graph representative role: {role}")


def _fixed_physical_uv(
    points: np.ndarray,
    config: RepresentativeCaseConfig,
    domain_u: tuple[float, float],
    domain_v: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _case_coordinates(points, config)
    uv = np.column_stack([
        (coordinates[:, 0] - domain_u[0]) / max(domain_u[1] - domain_u[0], 1.0e-12),
        (coordinates[:, 1] - domain_v[0]) / max(domain_v[1] - domain_v[0], 1.0e-12),
    ])
    if np.any(uv < -1.0e-5) or np.any(uv > 1.0 + 1.0e-5):
        raise AssertionError("physical chart UV escaped its declared world-space domain")
    return np.clip(uv, 0.0, 1.0).astype(np.float32), coordinates[:, 2].astype(np.float64)


def _greville_abscissae(knots: np.ndarray, degree: int, control_count: int) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64)
    if degree <= 0:
        return np.zeros((control_count,), dtype=np.float64)
    values = np.asarray([np.mean(knots[index + 1 : index + degree + 1]) for index in range(control_count)])
    values[0] = 0.0
    values[-1] = 1.0
    return values


def fit_physical_chart_surface(
    points: np.ndarray,
    config: RepresentativeCaseConfig,
    *,
    role: str,
    max_fit_points: int,
    device_name: str,
) -> PhysicalChartRepresentative:
    """Fit only normal-coordinate controls at immutable physical chart UVs."""

    import torch

    points = np.asarray(points, dtype=np.float64)
    fit_indices = deterministic_indices(len(points), int(max_fit_points))
    fit_points = points[fit_indices].copy()
    domain_u, domain_v = _physical_domain(config, role)
    fit_uv, fit_n = _fixed_physical_uv(fit_points, config, domain_u, domain_v)
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name)
    dtype = torch.float32
    zero_control = torch.zeros((GRAPH_RESOLUTION_U, GRAPH_RESOLUTION_V, 3), dtype=dtype, device=device)
    surface = TorchNURBSSurface(
        control_grid=zero_control,
        weights=torch.ones((GRAPH_RESOLUTION_U, GRAPH_RESOLUTION_V), dtype=dtype, device=device),
        degree_u=GRAPH_DEGREE_U,
        degree_v=GRAPH_DEGREE_V,
        observed_v_max=1.0,
    )
    knots_u, knots_v = surface.knot_vectors()
    greville_u = _greville_abscissae(knots_u.detach().cpu().numpy(), GRAPH_DEGREE_U, GRAPH_RESOLUTION_U)
    greville_v = _greville_abscissae(knots_v.detach().cpu().numpy(), GRAPH_DEGREE_V, GRAPH_RESOLUTION_V)
    control_u = domain_u[0] + greville_u * (domain_u[1] - domain_u[0])
    control_v = domain_v[0] + greville_v * (domain_v[1] - domain_v[0])

    # Deterministic affine graph seed.  This only anchors the same 1e-4
    # Tikhonov term used by WL138; no seed/tuning choice observes the holdout.
    seed_design = np.column_stack([np.ones(len(fit_uv)), fit_uv])
    seed_coefficients = np.linalg.lstsq(seed_design, fit_n, rcond=None)[0]
    seed_n = (
        seed_coefficients[0]
        + seed_coefficients[1] * greville_u[:, None]
        + seed_coefficients[2] * greville_v[None, :]
    )

    torch_uv = torch.as_tensor(fit_uv, dtype=dtype, device=device)
    torch_n = torch.as_tensor(fit_n, dtype=dtype, device=device)
    with torch.no_grad():
        basis_u, basis_v = surface._basis_values(torch_uv)
        rows = torch.einsum("qi,qj->qij", basis_u, basis_v).reshape(len(fit_points), -1)
        scale = max(float(len(fit_points)), 1.0)
        penalty = _second_difference_penalty(GRAPH_RESOLUTION_U, GRAPH_RESOLUTION_V, dtype, device)
        identity = torch.eye(GRAPH_RESOLUTION_U * GRAPH_RESOLUTION_V, dtype=dtype, device=device)
        system = (
            rows.T @ rows / scale
            + GRAPH_SMOOTHNESS_LAMBDA * penalty
            + GRAPH_TIKHONOV_LAMBDA * identity
        )
        rhs = rows.T @ torch_n / scale + GRAPH_TIKHONOV_LAMBDA * torch.as_tensor(seed_n.reshape(-1), dtype=dtype, device=device)
        try:
            scalar_solution = torch.linalg.solve(system, rhs)
        except Exception:
            scalar_solution = torch.linalg.lstsq(system, rhs[:, None]).solution[:, 0]
        scalar_control = scalar_solution.reshape(GRAPH_RESOLUTION_U, GRAPH_RESOLUTION_V)
        axis_u = torch.as_tensor(_normalised_axis(config.u_axis), dtype=dtype, device=device)
        axis_v = torch.as_tensor(_normalised_axis(config.v_axis), dtype=dtype, device=device)
        axis_n = torch.as_tensor(_normalised_axis(config.n_axis), dtype=dtype, device=device)
        torch_control_u = torch.as_tensor(control_u, dtype=dtype, device=device)[:, None, None]
        torch_control_v = torch.as_tensor(control_v, dtype=dtype, device=device)[None, :, None]
        control_grid = (
            torch_control_u * axis_u[None, None, :]
            + torch_control_v * axis_v[None, None, :]
            + scalar_control[:, :, None] * axis_n[None, None, :]
        )
        surface.control_grid = control_grid
        sample_u = torch.linspace(0.0, 1.0, SAMPLE_COUNT_U, dtype=dtype, device=device)
        sample_v = torch.linspace(0.0, 1.0, SAMPLE_COUNT_V, dtype=dtype, device=device)
        uu, vv = torch.meshgrid(sample_u, sample_v, indexing="ij")
        sample_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        sampled_points_t, sampled_normals_t = surface.evaluate_with_normals(sample_uv)
        fitted_points_t = surface.evaluate(torch_uv)
        residuals_t = torch.linalg.norm(fitted_points_t - torch.as_tensor(fit_points, dtype=dtype, device=device), dim=1)
    sampled_points = sampled_points_t.detach().cpu().numpy().astype(np.float64)
    sampled_normals = sampled_normals_t.detach().cpu().numpy().astype(np.float64)
    sampled_uv = sample_uv.detach().cpu().numpy().astype(np.float64)
    sampled_coordinates = _case_coordinates(sampled_points, config)
    expected_u = domain_u[0] + sampled_uv[:, 0] * (domain_u[1] - domain_u[0])
    expected_v = domain_v[0] + sampled_uv[:, 1] * (domain_v[1] - domain_v[0])
    error_u = np.abs(sampled_coordinates[:, 0] - expected_u)
    error_v = np.abs(sampled_coordinates[:, 1] - expected_v)
    tolerance = 2.0e-5 * max(domain_u[1] - domain_u[0], domain_v[1] - domain_v[0], 1.0)
    if float(max(np.max(error_u), np.max(error_v))) > tolerance:
        raise AssertionError("B-spline graph failed fixed physical-chart linear precision")
    return PhysicalChartRepresentative(
        role=role,
        surface=surface,
        domain_u=domain_u,
        domain_v=domain_v,
        fit_points=fit_points,
        fit_uv=fit_uv.astype(np.float64),
        fit_input_sha256=_sha256_rows(fit_points),
        scalar_control_grid=scalar_control.detach().cpu().numpy().astype(np.float64),
        control_grid=control_grid.detach().cpu().numpy().astype(np.float64),
        sampled_uv=sampled_uv,
        sampled_points=sampled_points,
        sampled_normals=sampled_normals,
        fitting_residuals=residuals_t.detach().cpu().numpy().astype(np.float64),
        physical_u_precision_error=error_u,
        physical_v_precision_error=error_v,
    )


def select_physical_heldout_samples(points: np.ndarray, config: RepresentativeCaseConfig) -> np.ndarray:
    """Select held-out side from world-space physical coordinates, never UV."""

    points = np.asarray(points, dtype=np.float64)
    physical_u = _case_coordinates(points, config)[:, 0]
    return np.asarray(physical_u > float(config.u_cut) + 1.0e-8, dtype=bool)


def _evaluate_surface_derivatives(surface: TorchNURBSSurface, sampled_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    uv = torch.as_tensor(sampled_uv, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    with torch.no_grad():
        points, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    return (
        points.detach().cpu().numpy().astype(np.float64),
        deriv_u.detach().cpu().numpy().astype(np.float64),
        deriv_v.detach().cpu().numpy().astype(np.float64),
    )


def audit_representative_topology(
    surface: TorchNURBSSurface,
    config: RepresentativeCaseConfig,
    *,
    domain_u: tuple[float, float],
    domain_v: tuple[float, float],
    h: float,
) -> dict[str, Any]:
    """Attribute physical-chart reversals, orientation flips, and overlap."""

    from scipy.spatial import cKDTree

    sample_u = np.linspace(0.0, 1.0, SAMPLE_COUNT_U, dtype=np.float32)
    sample_v = np.linspace(0.0, 1.0, SAMPLE_COUNT_V, dtype=np.float32)
    uu, vv = np.meshgrid(sample_u, sample_v, indexing="ij")
    uv = np.column_stack([uu.reshape(-1), vv.reshape(-1)])
    points, deriv_u, deriv_v = _evaluate_surface_derivatives(surface, uv)
    coordinates = _case_coordinates(points, config)
    axis_u = _normalised_axis(config.u_axis)
    axis_v = _normalised_axis(config.v_axis)
    du_u = deriv_u @ axis_u
    du_v = deriv_u @ axis_v
    dv_u = deriv_v @ axis_u
    dv_v = deriv_v @ axis_v
    jacobian = du_u * dv_v - du_v * dv_u
    tolerance = 1.0e-8
    u_reversals = int(np.sum(du_u <= tolerance))
    v_reversals = int(np.sum(dv_v <= tolerance))
    orientation_flips = int(np.sum(jacobian <= tolerance))

    # Multi-valued occupancy requires disconnected parameter islands to enter
    # one physical chart bin with separated normal coordinate values.
    bin_size = GRAPH_BIN_SCALE_H * h
    iu = np.floor((coordinates[:, 0] - domain_u[0]) / bin_size).astype(np.int64)
    iv = np.floor((coordinates[:, 1] - domain_v[0]) / bin_size).astype(np.int64)
    keys = np.column_stack([iu, iv])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    duplicate_bins = 0
    max_components = 1
    for index in range(len(unique)):
        members = np.flatnonzero(inverse == index)
        if len(members) < 2:
            continue
        grid_indices = np.column_stack([members // SAMPLE_COUNT_V, members % SAMPLE_COUNT_V])
        components = _grid_components(grid_indices)
        max_components = max(max_components, components)
        n_values = coordinates[members, 2]
        if components > 1 and float(np.ptp(n_values)) >= GRAPH_MODE_SEPARATION_H * h:
            duplicate_bins += 1

    # Near crossings exclude a local five-cell parameter neighbourhood.
    pairs = cKDTree(points).query_pairs(r=float(h), output_type="ndarray")
    if len(pairs):
        first_i, first_j = pairs[:, 0] // SAMPLE_COUNT_V, pairs[:, 0] % SAMPLE_COUNT_V
        second_i, second_j = pairs[:, 1] // SAMPLE_COUNT_V, pairs[:, 1] % SAMPLE_COUNT_V
        nonlocal_pair = (np.abs(first_i - second_i) > 2) | (np.abs(first_j - second_j) > 2)
        near_crossings = int(np.sum(nonlocal_pair))
    else:
        near_crossings = 0
    grid = points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    area = float(_surface_area(grid))
    footprint = float((domain_u[1] - domain_u[0]) * (domain_v[1] - domain_v[0]))
    expected_u = domain_u[0] + uv[:, 0] * (domain_u[1] - domain_u[0])
    expected_v = domain_v[0] + uv[:, 1] * (domain_v[1] - domain_v[0])
    return {
        "sample_count": int(len(points)),
        "physical_u_reversal_count": u_reversals,
        "physical_v_reversal_count": v_reversals,
        "jacobian_orientation_flip_count": orientation_flips,
        "jacobian": _summary(jacobian),
        "duplicate_multivalued_chart_bins": int(duplicate_bins),
        "maximum_parameter_islands_per_chart_bin": int(max_components),
        "near_self_crossing_pair_count_within_h": near_crossings,
        "representative_area": area,
        "physical_chart_footprint_area": footprint,
        "area_inflation_ratio": float(area / max(footprint, 1.0e-12)),
        "physical_u_affine_precision_error": _summary(np.abs(coordinates[:, 0] - expected_u)),
        "physical_v_affine_precision_error": _summary(np.abs(coordinates[:, 1] - expected_v)),
        "topology_contract_valid": bool(u_reversals == 0 and v_reversals == 0 and orientation_flips == 0 and duplicate_bins == 0),
    }


def _representative_proximity(
    raw_points: np.ndarray,
    representative_points: np.ndarray,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    raw = deterministic_subsample(np.asarray(raw_points, dtype=np.float64), 12000)
    representative = np.asarray(representative_points, dtype=np.float64)
    raw_to_rep = cKDTree(representative).query(raw, workers=1)[0]
    rep_to_raw = cKDTree(raw).query(representative, workers=1)[0]
    return {
        "raw_to_representative": _summary(raw_to_rep, h=h),
        "representative_to_raw": _summary(rep_to_raw, h=h),
        "symmetric_chamfer_like_mean_over_h": float(0.5 * (np.mean(raw_to_rep) + np.mean(rep_to_raw)) / h),
        "symmetric_median_over_h": float(0.5 * (np.median(raw_to_rep) + np.median(rep_to_raw)) / h),
    }


def _normal_field_accounting(normals: np.ndarray) -> dict[str, Any]:
    normals = np.asarray(normals, dtype=np.float64).reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    flat = normals.reshape(-1, 3)
    mean_normal = _oriented_mean_normal(flat)
    mean_angles = _normal_angles(flat, np.tile(mean_normal[None, :], (len(flat), 1)))
    adjacent = np.concatenate([
        _normal_angles(normals[1:].reshape(-1, 3), normals[:-1].reshape(-1, 3)),
        _normal_angles(normals[:, 1:].reshape(-1, 3), normals[:, :-1].reshape(-1, 3)),
    ])
    return {
        "analytic_normal_variation_to_oriented_mean_degrees": _summary(mean_angles),
        "local_adjacent_normal_angle_degrees": _summary(adjacent),
    }


def representative_contract(
    raw_points: np.ndarray,
    representative_points: np.ndarray,
    representative_normals: np.ndarray,
    fitting_residuals: np.ndarray,
    topology: dict[str, Any],
    h: float,
) -> dict[str, Any]:
    return {
        "PROXIMITY": _representative_proximity(raw_points, representative_points, h),
        "fixed_uv_fitting_residual": _summary(fitting_residuals, h=h),
        "TOPOLOGY / CHART": topology,
        "GEOMETRY": {
            **_normal_field_accounting(representative_normals),
            "representative_area": topology["representative_area"],
            "physical_chart_footprint_area": topology["physical_chart_footprint_area"],
            "area_inflation_ratio": topology["area_inflation_ratio"],
        },
    }


def build_chart_continuation(
    holdout: Holdout,
    retained_representative: PhysicalChartRepresentative,
) -> ChartContinuation:
    """Apply the frozen first-order rule in the retained graph frame only."""

    import torch

    if retained_representative.role != "retained_construction":
        raise AssertionError("continuation accepts retained_construction representative only")
    frozen = build_self_continuation(
        holdout,
        frontier_band_fraction=0.10,
        frontier_bins=24,
        continuation_samples=32,
    )
    frontier_raw = np.asarray(frozen.frontier_points, dtype=np.float64)
    frontier_uv, _frontier_n = _fixed_physical_uv(
        frontier_raw,
        RepresentativeCaseConfig(
            name="frontier_mapping",
            semantic_label="frontier_mapping",
            roi_box=holdout.permitted_volume,
            u_axis=tuple(map(float, holdout.u_axis)),
            v_axis=tuple(map(float, holdout.v_axis)),
            n_axis=tuple(map(float, holdout.n_axis)),
            u_bounds=tuple(map(float, holdout.u_bounds)),
            v_bounds=tuple(map(float, holdout.v_bounds)),
            n_bounds=tuple(map(float, holdout.n_bounds)),
            u_cut=float(holdout.u_cut),
            frontier_source="frozen physical frontier",
        ),
        retained_representative.domain_u,
        retained_representative.domain_v,
    )
    uv_t = torch.as_tensor(frontier_uv, dtype=retained_representative.surface.control_grid.dtype, device=retained_representative.surface.control_grid.device)
    with torch.no_grad():
        frontier_t, du_t, dv_t = retained_representative.surface.evaluate_with_derivatives(uv_t)
        normals_t = retained_representative.surface.evaluate_with_normals(uv_t)[1]
    frontier_representative = frontier_t.detach().cpu().numpy().astype(np.float64)
    du = du_t.detach().cpu().numpy().astype(np.float64)
    tangent_v = dv_t.detach().cpu().numpy().astype(np.float64)
    frontier_normals = normals_t.detach().cpu().numpy().astype(np.float64)
    tangent_v /= np.clip(np.linalg.norm(tangent_v, axis=1, keepdims=True), 1.0e-12, None)
    projection_gaps = np.linalg.norm(frontier_representative - frontier_raw, axis=1)
    u_axis = _normalised_axis(holdout.u_axis)
    directions = u_axis[None, :] - np.sum(frontier_normals * u_axis[None, :], axis=1, keepdims=True) * frontier_normals
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.where(lengths > 1.0e-10, directions, du)
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12, None)
    directions = np.where(np.sum(directions * u_axis[None, :], axis=1, keepdims=True) < 0.0, -directions, directions)
    extent = float(holdout.u_bounds[1] - holdout.u_cut)
    l_values = np.linspace(0.0, extent, 32, dtype=np.float64)
    points_grid = frontier_representative[None, :, :] + l_values[:, None, None] * directions[None, :, :]
    one_row_normals = np.cross(directions, tangent_v)
    one_row_normals /= np.clip(np.linalg.norm(one_row_normals, axis=1, keepdims=True), 1.0e-12, None)
    one_row_normals = np.where(np.sum(one_row_normals * frontier_normals, axis=1, keepdims=True) < 0.0, -one_row_normals, one_row_normals)
    normals_grid = np.tile(one_row_normals[None, :, :], (len(l_values), 1, 1))
    boundary_gaps = np.linalg.norm(points_grid[0] - frontier_representative, axis=1)
    boundary_angles = _normal_angles(normals_grid[0], frontier_normals)
    return ChartContinuation(
        source_role=retained_representative.role,
        frontier_raw=frontier_raw,
        frontier_representative=frontier_representative,
        frontier_uv=frontier_uv.astype(np.float64),
        frontier_normals=frontier_normals,
        frontier_tangent_v=tangent_v,
        directions=directions,
        points_grid=points_grid,
        normals_grid=normals_grid,
        l_values=l_values,
        projection_gaps=projection_gaps,
        boundary_position_gaps=boundary_gaps,
        boundary_normal_angles=boundary_angles,
    )


def _raw_prediction_view(prediction: Prediction, retained_points: np.ndarray) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    raw_normals = estimate_point_normals(retained_points, k=min(20, len(retained_points)))
    if raw_normals is None:
        source_normals = np.asarray(prediction.normals_grid[0], dtype=np.float64)
    else:
        nearest = cKDTree(retained_points).query(prediction.frontier_points, workers=1)[1]
        source_normals = raw_normals[nearest]
    return {
        "name": "WL134 raw-frame continuation",
        "points_grid": np.asarray(prediction.points_grid, dtype=np.float64),
        "normals_grid": np.asarray(prediction.normals_grid, dtype=np.float64),
        "boundary_source_points": np.asarray(prediction.frontier_points, dtype=np.float64),
        "boundary_source_normals": np.asarray(source_normals, dtype=np.float64),
    }


def _wl138_prediction_view(continuation: RepresentativeContinuation) -> dict[str, Any]:
    return {
        "name": "WL138 unconstrained-representative continuation",
        "points_grid": continuation.points_grid,
        "normals_grid": continuation.normals_grid,
        "boundary_source_points": continuation.frontier_representative,
        "boundary_source_normals": continuation.frontier_normals,
    }


def _chart_prediction_view(continuation: ChartContinuation) -> dict[str, Any]:
    return {
        "name": "WL139 physical-chart-representative continuation",
        "points_grid": continuation.points_grid,
        "normals_grid": continuation.normals_grid,
        "boundary_source_points": continuation.frontier_representative,
        "boundary_source_normals": continuation.frontier_normals,
    }


def evaluate_prediction(
    reference_points: np.ndarray,
    reference_normals: np.ndarray | None,
    prediction: dict[str, Any],
    config: RepresentativeCaseConfig,
    h: float,
    *,
    reference_label: str,
) -> dict[str, Any]:
    """Evaluate a frozen prediction with fixed physical distance bins."""

    from scipy.spatial import cKDTree

    reference_points = np.asarray(reference_points, dtype=np.float64)
    points_grid = np.asarray(prediction["points_grid"], dtype=np.float64)
    normals_grid = np.asarray(prediction["normals_grid"], dtype=np.float64)
    predicted_points = points_grid.reshape(-1, 3)
    predicted_normals = normals_grid.reshape(-1, 3)
    distances, nearest = cKDTree(predicted_points).query(reference_points, workers=1)
    result: dict[str, Any] = {
        "prediction": prediction["name"],
        "reference": reference_label,
        "reference_count": int(len(reference_points)),
        "prediction_point_count": int(len(predicted_points)),
        "point_to_surface_distance": _summary(distances, h=h),
        "coverage": {
            "fraction_le_h": float(np.mean(distances <= h)),
            "fraction_le_2h": float(np.mean(distances <= 2.0 * h)),
        },
        "normal_angular_error_degrees": {"status": "UNAVAILABLE"},
        "distance_from_physical_frontier_bins": [],
        "boundary_position_continuity": _summary(
            np.linalg.norm(points_grid[0] - np.asarray(prediction["boundary_source_points"]), axis=1),
            h=h,
        ),
        "boundary_normal_continuity_degrees": _summary(
            _normal_angles(normals_grid[0], np.asarray(prediction["boundary_source_normals"])),
        ),
        "metric_fed_back_into_geometry": False,
    }
    if reference_normals is not None:
        result["normal_angular_error_degrees"] = _summary(
            _normal_angles(np.asarray(reference_normals, dtype=np.float64), predicted_normals[nearest])
        )
    physical_distance = np.maximum(_case_coordinates(reference_points, config)[:, 0] - float(config.u_cut), 0.0)
    edges = np.asarray([0.0, 2.0 * h, 4.0 * h, 8.0 * h, 16.0 * h, np.inf], dtype=np.float64)
    labels = ["0-2h", "2-4h", "4-8h", "8-16h", ">16h"]
    for index, label in enumerate(labels):
        mask = (physical_distance >= edges[index]) & (physical_distance < edges[index + 1])
        values = distances[mask]
        result["distance_from_physical_frontier_bins"].append({
            "bin": label,
            "samples": int(np.sum(mask)),
            "distance": _summary(values, h=h),
        })
    return result


def _full_macro_reference_contract(
    withheld_points: np.ndarray,
    withheld_normals: np.ndarray | None,
    full_representative: PhysicalChartRepresentative,
    config: RepresentativeCaseConfig,
    h: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    heldout_mask = select_physical_heldout_samples(full_representative.sampled_points, config)
    macro_points = full_representative.sampled_points[heldout_mask]
    macro_normals = full_representative.sampled_normals[heldout_mask]
    proximity = _representative_proximity(withheld_points, macro_points, h)
    normal_report: dict[str, Any] = {"status": "UNAVAILABLE"}
    if withheld_normals is not None:
        nearest = cKDTree(macro_points).query(withheld_points, workers=1)[1]
        normal_report = _summary(_normal_angles(withheld_normals, macro_normals[nearest]))
    raw_to_macro = proximity["raw_to_representative"]
    macro_to_raw = proximity["representative_to_raw"]
    valid = bool(
        raw_to_macro["median_over_h"] <= 3.0
        and raw_to_macro["p95_over_h"] <= 12.0
        and macro_to_raw["median_over_h"] <= 3.0
        and macro_to_raw["p95_over_h"] <= 12.0
    )
    return ({
        "status": "VALID_MACRO_REFERENCE" if valid else "INVALID_MACRO_REFERENCE",
        "selection_semantics": "sample XYZ -> physical ROI coordinates -> physical u > fixed u_cut",
        "nurbs_parameter_u_used_for_selection": False,
        "full_representative_role": full_representative.role,
        "used_for_prediction_construction": False,
        "heldout_macro_sample_count": int(len(macro_points)),
        "raw_withheld_vs_macro_proximity": proximity,
        "raw_withheld_vs_macro_normal_error_degrees": normal_report,
        "validity_criterion": "both raw->macro and macro->raw median <=3h and p95 <=12h, plus fixed chart topology by construction",
    }, macro_points, macro_normals)


def _frontier_report(
    continuation: ChartContinuation,
    retained_points: np.ndarray,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    sample = deterministic_subsample(retained_points, 5000)
    raw_normals = estimate_point_normals(sample, k=min(20, len(sample)))
    if raw_normals is None:
        raw_angles = np.empty((0,), dtype=np.float64)
    else:
        nearest = cKDTree(sample).query(continuation.frontier_raw, workers=1)[1]
        raw_angles = _normal_angles(raw_normals[nearest], continuation.frontier_normals)
    frontier_mean = _oriented_mean_normal(continuation.frontier_normals)
    frontier_variation = _normal_angles(
        continuation.frontier_normals,
        np.tile(frontier_mean[None, :], (len(continuation.frontier_normals), 1)),
    )
    return {
        "source": "same frozen physical world-space frontier as WL138",
        "sample_count": int(len(continuation.frontier_raw)),
        "physical_frontier_to_representative_gap": _summary(continuation.projection_gaps, h=h),
        "frontier_representative_normal_variation_degrees": _summary(frontier_variation),
        "raw_pca_vs_representative_normal_angle_degrees": _summary(raw_angles),
        "boundary_position_continuity": _summary(continuation.boundary_position_gaps, h=h),
        "boundary_normal_continuity_degrees": _summary(continuation.boundary_normal_angles),
        "nurbs_uv_edge_used_as_termination": False,
    }


def _display_subsample(points: np.ndarray, max_points: int = 16000) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points
    cells = np.floor(points / DISPLAY_VOXEL_WORLD).astype(np.int64)
    _unique, first = np.unique(cells, axis=0, return_index=True)
    return deterministic_subsample(points[np.sort(first)], max_points)


def _local_points(points: np.ndarray, config: RepresentativeCaseConfig) -> np.ndarray:
    return _case_coordinates(np.asarray(points, dtype=np.float64).reshape(-1, 3), config)


def _plot_points(axis: Any, points: np.ndarray, config: RepresentativeCaseConfig, color: Any, *, label: str, size: float = DISPLAY_POINT_SIZE, alpha: float = DISPLAY_POINT_ALPHA, max_points: int = 16000) -> None:
    selected = _display_subsample(points, max_points)
    if not len(selected):
        return
    local = _local_points(selected, config)
    axis.scatter(local[:, 0], local[:, 1], local[:, 2], s=size, alpha=alpha, color=color, linewidths=0, label=label)


def _plot_surface(axis: Any, points_grid: np.ndarray, config: RepresentativeCaseConfig, color: Any, *, label: str, alpha: float = DISPLAY_SURFACE_ALPHA) -> None:
    grid = np.asarray(points_grid, dtype=np.float64)
    local = _local_points(grid.reshape(-1, 3), config).reshape(grid.shape)
    axis.plot_surface(local[:, :, 0], local[:, :, 1], local[:, :, 2], color=color, alpha=alpha, linewidth=0.15, edgecolor=(*color, min(alpha + 0.12, 1.0)), antialiased=True, shade=False, label=label)


def _plot_frontier(axis: Any, points: np.ndarray, config: RepresentativeCaseConfig, *, label: str = "physical frontier") -> None:
    local = _local_points(points, config)
    axis.scatter(local[:, 0], local[:, 1], local[:, 2], s=22.0, alpha=1.0, color=FRONTIER_YELLOW, linewidths=0, label=label)


def _configure_axis(axis: Any, limits_points: np.ndarray, config: RepresentativeCaseConfig, *, view: dict[str, float] = FIXED_VIEW) -> None:
    local = _local_points(limits_points, config)
    low, high = np.min(local, axis=0), np.max(local, axis=0)
    span = np.maximum(high - low, 1.0e-6)
    padding = np.maximum(0.06 * span, 0.25 * np.min(span[span > 1.0e-6]) if np.any(span > 1.0e-6) else 1.0e-3)
    axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
    axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
    axis.set_zlim(low[2] - padding[2], high[2] + padding[2])
    axis.set_box_aspect(np.maximum(span, 0.18 * np.max(span)))
    axis.view_init(elev=float(view["elev"]), azim=float(view["azim"]))
    axis.set_xlabel("physical u")
    axis.set_ylabel("physical v")
    axis.set_zlabel("physical n")
    axis.grid(True, alpha=0.25)


def _save_3d(path: Path, title: str, config: RepresentativeCaseConfig, limits: np.ndarray, draw: Any, *, view: dict[str, float] = FIXED_VIEW) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10.5, 8.0), dpi=220)
    axis = figure.add_subplot(111, projection="3d")
    draw(axis)
    _configure_axis(axis, limits, config, view=view)
    axis.set_title(title)
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(handles, labels, loc="upper left", framealpha=0.94)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_graphness_figure(path: Path, points: np.ndarray, config: RepresentativeCaseConfig, audit: GraphnessAudit) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local = _local_points(points, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    plot = axis.scatter(local[:, 0], local[:, 1], c=local[:, 2], cmap="viridis", s=5.0, alpha=0.97, linewidths=0)
    if len(audit.multimode_bin_indices):
        centers = np.column_stack([
            config.u_bounds[0] + (audit.multimode_bin_indices[:, 0] + 0.5) * audit.bin_size_world,
            config.v_bounds[0] + (audit.multimode_bin_indices[:, 1] + 0.5) * audit.bin_size_world,
        ])
        axis.scatter(centers[:, 0], centers[:, 1], marker="x", s=42, linewidths=1.4, color="red", label="clearly separated n modes")
        axis.legend(loc="upper left")
    axis.set_xlabel("physical u")
    axis.set_ylabel("physical v")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Retained raw graphness audit: {audit.status}")
    figure.colorbar(plot, ax=axis, label="physical n")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_candidate_visuals(
    case_root: Path,
    config: RepresentativeCaseConfig,
    retained_points: np.ndarray,
    graphness: GraphnessAudit,
    baseline: FittedRepresentative,
    baseline_continuation: RepresentativeContinuation,
    constrained: PhysicalChartRepresentative | None,
) -> dict[str, str]:
    case_root.mkdir(parents=True, exist_ok=True)
    baseline_grid = baseline.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    frontier = baseline_continuation.frontier_raw
    limits_parts = [retained_points, baseline.sampled_points, frontier]
    if constrained is not None:
        limits_parts.append(constrained.sampled_points)
    limits = np.concatenate(limits_parts, axis=0)
    paths = {
        "baseline_folded_representative": case_root / "baseline_folded_representative.png",
        "phase5_unconstrained": case_root / ("curved_raw_vs_unconstrained_nurbs.png" if config.name == CURVED_RIM_CASE.name else "raw_vs_unconstrained_nurbs.png"),
        "graphness_audit": case_root / "graphness_audit.png",
    }
    baseline_draw = lambda axis: (
        _plot_points(axis, retained_points, config, RAW_GREY, label="retained raw Visible Surface"),
        _plot_surface(axis, baseline_grid, config, UNCONSTRAINED_BLUE, label="WL138 unconstrained NURBS"),
        _plot_frontier(axis, frontier, config),
    )
    _save_3d(paths["baseline_folded_representative"], "WL138 frozen unconstrained representative", config, limits, baseline_draw)
    _save_3d(paths["phase5_unconstrained"], "Raw retained vs unconstrained NURBS", config, limits, baseline_draw)
    _write_graphness_figure(paths["graphness_audit"], retained_points, config, graphness)
    if constrained is not None:
        constrained_grid = constrained.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
        paths.update({
            "raw_vs_chart_constrained_representative": case_root / "raw_vs_chart_constrained_representative.png",
            "phase5_chart_constrained": case_root / ("curved_raw_vs_chart_constrained_nurbs.png" if config.name == CURVED_RIM_CASE.name else "raw_vs_chart_constrained_nurbs.png"),
            "unconstrained_vs_constrained_representative": case_root / "unconstrained_vs_constrained_representative.png",
            "representative_normals": case_root / "representative_normals.png",
        })
        constrained_draw = lambda axis: (
            _plot_points(axis, retained_points, config, RAW_GREY, label="retained raw Visible Surface"),
            _plot_surface(axis, constrained_grid, config, CONSTRAINED_ORANGE, label="physical-chart-constrained NURBS"),
            _plot_frontier(axis, frontier, config),
        )
        _save_3d(paths["raw_vs_chart_constrained_representative"], "Raw retained vs physical-chart-constrained representative", config, limits, constrained_draw)
        _save_3d(paths["phase5_chart_constrained"], "Raw retained vs chart-constrained NURBS", config, limits, constrained_draw)
        _save_3d(
            paths["unconstrained_vs_constrained_representative"],
            "Unconstrained vs physical-chart-constrained representative",
            config,
            limits,
            lambda axis: (
                _plot_points(axis, retained_points, config, RAW_GREY, label="retained raw Visible Surface"),
                _plot_surface(axis, baseline_grid, config, UNCONSTRAINED_BLUE, label="WL138 unconstrained NURBS", alpha=0.40),
                _plot_surface(axis, constrained_grid, config, CONSTRAINED_ORANGE, label="chart-constrained NURBS", alpha=0.62),
                _plot_frontier(axis, frontier, config),
            ),
        )
        normal_indices = np.linspace(0, len(constrained.sampled_points) - 1, 72, dtype=np.int64)
        normal_points = _local_points(constrained.sampled_points[normal_indices], config)
        axes = np.stack([_normalised_axis(config.u_axis), _normalised_axis(config.v_axis), _normalised_axis(config.n_axis)], axis=1)
        normal_vectors = constrained.sampled_normals[normal_indices] @ axes
        def normals_draw(axis: Any) -> None:
            _plot_points(axis, retained_points, config, RAW_GREY, label="retained raw Visible Surface")
            _plot_surface(axis, constrained_grid, config, CONSTRAINED_ORANGE, label="chart-constrained NURBS", alpha=0.48)
            axis.quiver(normal_points[:, 0], normal_points[:, 1], normal_points[:, 2], normal_vectors[:, 0], normal_vectors[:, 1], normal_vectors[:, 2], length=0.10, normalize=True, color=NORMAL_BLUE, linewidth=0.75, label="analytic normals")
        _save_3d(paths["representative_normals"], "Physical-chart representative analytic normals", config, limits, normals_draw)
    return {key: str(value) for key, value in paths.items()}


def write_continuation_visuals(
    case_root: Path,
    config: RepresentativeCaseConfig,
    retained_points: np.ndarray,
    withheld_points: np.ndarray,
    constrained: PhysicalChartRepresentative,
    raw_prediction: Prediction,
    baseline_continuation: RepresentativeContinuation,
    chart_continuation: ChartContinuation,
) -> dict[str, str]:
    constrained_grid = constrained.sampled_points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    limits = np.concatenate([
        retained_points,
        withheld_points,
        constrained.sampled_points,
        raw_prediction.points,
        baseline_continuation.points_grid.reshape(-1, 3),
        chart_continuation.points_grid.reshape(-1, 3),
    ], axis=0)
    paths = {
        "chart_constrained_continuation": case_root / "chart_constrained_continuation.png",
        "continuation_vs_raw_reference": case_root / "continuation_vs_raw_reference.png",
        "controlled_continuation_ab": case_root / "controlled_continuation_ab.png",
    }
    _save_3d(
        paths["chart_constrained_continuation"],
        "Chart-constrained first-order continuation",
        config,
        limits,
        lambda axis: (
            _plot_points(axis, retained_points, config, RAW_GREY, label="retained raw Visible Surface"),
            _plot_surface(axis, constrained_grid, config, CONSTRAINED_ORANGE, label="retained chart representative"),
            _plot_surface(axis, chart_continuation.points_grid, config, CHART_CONTINUATION_CYAN, label="chart-frame continuation", alpha=0.68),
            _plot_frontier(axis, chart_continuation.frontier_raw, config),
        ),
    )
    _save_3d(
        paths["continuation_vs_raw_reference"],
        "Chart continuation vs raw held-out reference",
        config,
        limits,
        lambda axis: (
            _plot_points(axis, retained_points, config, RAW_GREY, label="retained raw Visible Surface"),
            _plot_points(axis, withheld_points, config, WITHHELD_GREEN, label="raw held-out reference", size=3.7, alpha=DISPLAY_REFERENCE_ALPHA),
            _plot_surface(axis, chart_continuation.points_grid, config, CHART_CONTINUATION_CYAN, label="chart-frame continuation", alpha=0.72),
            _plot_frontier(axis, chart_continuation.frontier_raw, config),
        ),
    )
    _save_3d(
        paths["controlled_continuation_ab"],
        "Frozen controlled continuation A/B",
        config,
        limits,
        lambda axis: (
            _plot_points(axis, withheld_points, config, WITHHELD_GREEN, label="raw held-out reference", size=3.7, alpha=DISPLAY_REFERENCE_ALPHA),
            _plot_surface(axis, raw_prediction.points_grid, config, RAW_CONTINUATION_MAGENTA, label="WL134 raw-frame", alpha=0.38),
            _plot_surface(axis, baseline_continuation.points_grid, config, WL138_CONTINUATION_PURPLE, label="WL138 unconstrained frame", alpha=0.40),
            _plot_surface(axis, chart_continuation.points_grid, config, CHART_CONTINUATION_CYAN, label="WL139 chart frame", alpha=0.68),
            _plot_frontier(axis, chart_continuation.frontier_raw, config),
        ),
    )
    return {key: str(value) for key, value in paths.items()}


def write_geometry(
    case_root: Path,
    retained_points: np.ndarray,
    withheld_points: np.ndarray,
    baseline: FittedRepresentative,
    constrained: PhysicalChartRepresentative | None,
    chart_continuation: ChartContinuation | None,
) -> dict[str, str]:
    geometry_root = case_root / "geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {
        "retained_raw": geometry_root / "retained_raw.ply",
        "withheld_raw_reference": geometry_root / "withheld_raw_reference.ply",
        "wl138_unconstrained_representative": geometry_root / "wl138_unconstrained_representative.ply",
    }
    _write_ply(outputs["retained_raw"], retained_points, color=(112, 116, 126))
    _write_ply(outputs["withheld_raw_reference"], withheld_points, color=(13, 173, 61))
    _write_ply(outputs["wl138_unconstrained_representative"], baseline.sampled_points, faces=_grid_faces(SAMPLE_COUNT_U, SAMPLE_COUNT_V), color=(26, 71, 209))
    if constrained is not None:
        outputs["chart_constrained_representative"] = geometry_root / "chart_constrained_representative.ply"
        outputs["chart_constrained_npz"] = geometry_root / "chart_constrained_representative.npz"
        _write_ply(outputs["chart_constrained_representative"], constrained.sampled_points, faces=_grid_faces(SAMPLE_COUNT_U, SAMPLE_COUNT_V), color=(245, 94, 10))
        np.savez_compressed(
            outputs["chart_constrained_npz"],
            sampled_points=constrained.sampled_points,
            sampled_normals=constrained.sampled_normals,
            sampled_uv=constrained.sampled_uv,
            control_grid=constrained.control_grid,
            scalar_control_grid=constrained.scalar_control_grid,
            fit_input_sha256=np.asarray(constrained.fit_input_sha256),
        )
    if chart_continuation is not None:
        outputs["chart_constrained_continuation"] = geometry_root / "chart_constrained_continuation.ply"
        outputs["chart_constrained_continuation_npz"] = geometry_root / "chart_constrained_continuation.npz"
        _write_ply(
            outputs["chart_constrained_continuation"],
            chart_continuation.points_grid.reshape(-1, 3),
            faces=_grid_faces(*chart_continuation.points_grid.shape[:2]),
            color=(0, 179, 209),
        )
        np.savez_compressed(
            outputs["chart_constrained_continuation_npz"],
            points_grid=chart_continuation.points_grid,
            normals_grid=chart_continuation.normals_grid,
            frontier_raw=chart_continuation.frontier_raw,
            frontier_representative=chart_continuation.frontier_representative,
            frontier_uv=chart_continuation.frontier_uv,
            l_values=chart_continuation.l_values,
        )
    return {key: str(value) for key, value in outputs.items()}


def _case_decision(config: RepresentativeCaseConfig, passed: set[str], failed: set[str]) -> str:
    if config.name in passed and config.name in failed:
        raise ValueError(f"case appears in both qualitative pass/fail lists: {config.name}")
    if config.name in passed:
        return "PASS"
    if config.name in failed:
        return "FAIL"
    return "PENDING_MANUAL_INSPECTION"


def _useful_signal(metrics: dict[str, Any]) -> bool:
    distance = metrics["point_to_surface_distance"]
    coverage = metrics["coverage"]
    return bool(
        distance["median_over_h"] <= USEFUL_MEDIAN_LIMIT_OVER_H
        and coverage["fraction_le_2h"] >= USEFUL_COVERAGE_LE_2H_MIN
    )


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    wl138_sha_before = _file_sha256(WL138_MODULE)
    wl138_root = Path(arguments.wl138_root)
    wl138_manifest_before = _artifact_manifest(wl138_root)
    historical_root_report = json.loads(
        (wl138_root / "scale_separated_visible_surface_representative_report.json").read_text(encoding="utf-8")
    )
    h = float(historical_root_report["inputs"]["h"])
    mu = float(historical_root_report["inputs"]["mu"])
    passed = set(arguments.qualitative_pass or [])
    failed = set(arguments.qualitative_fail or [])
    valid_names = {CURVED_RIM_CASE.name, LEG_CASE.name}
    if not passed.issubset(valid_names) or not failed.issubset(valid_names):
        raise ValueError("qualitative decisions contain an unknown case")

    case_reports: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for config in (CURVED_RIM_CASE, LEG_CASE):
        try:
            frozen = _load_frozen_wl138_case(
                wl138_root,
                config,
                max_fit_points=int(arguments.max_fit_points),
                device_name=arguments.device,
            )
            holdout = frozen["holdout"]
            retained = frozen["retained"]
            withheld = frozen["withheld"]
            full = frozen["full"]
            baseline = frozen["baseline"]
            baseline_continuation = frozen["baseline_continuation"]
            baseline_topology = audit_representative_topology(
                baseline.surface,
                config,
                domain_u=(float(config.u_bounds[0]), float(config.u_cut)),
                domain_v=tuple(map(float, config.v_bounds)),
                h=h,
            )
            baseline_contract = representative_contract(
                retained,
                baseline.sampled_points,
                baseline.sampled_normals,
                baseline.fitting_residuals,
                baseline_topology,
                h,
            )
            baseline_contract["frontier_projection_gap"] = _summary(baseline_continuation.projection_gaps, h=h)
            baseline_contract.pop("fixed_uv_fitting_residual", None)
            baseline_contract["historical_wl138_fitting_residual"] = frozen["historical_stats"]["representative_fitting_residual"]
            baseline_contract["exact_artifact_replay_validation"] = frozen["replay_validation"]

            graphness = audit_raw_graphness(retained, config, h)
            graph_report = graphness_report(graphness, h)
            constrained: PhysicalChartRepresentative | None = None
            constrained_contract: dict[str, Any] | None = None
            constrained_topology: dict[str, Any] | None = None
            if graphness.status == "PASS_GRAPH_LIKE":
                constrained = fit_physical_chart_surface(
                    retained,
                    config,
                    role="retained_construction",
                    max_fit_points=int(arguments.max_fit_points),
                    device_name=arguments.device,
                )
                constrained_topology = audit_representative_topology(
                    constrained.surface,
                    config,
                    domain_u=constrained.domain_u,
                    domain_v=constrained.domain_v,
                    h=h,
                )
                constrained_contract = representative_contract(
                    retained,
                    constrained.sampled_points,
                    constrained.sampled_normals,
                    constrained.fitting_residuals,
                    constrained_topology,
                    h,
                )

            decision = _case_decision(config, passed, failed)
            if graphness.status != "PASS_GRAPH_LIKE":
                qualitative_status = "NOT_APPLICABLE_RAW_NOT_GRAPH_LIKE"
            elif constrained_topology is None or not constrained_topology["topology_contract_valid"]:
                qualitative_status = "FAIL_TOPOLOGY_CONTRACT"
            else:
                qualitative_status = decision

            case_root = output_root / config.name
            visuals = write_candidate_visuals(
                case_root,
                config,
                retained,
                graphness,
                baseline,
                baseline_continuation,
                constrained,
            )

            continuation_report: dict[str, Any] = {
                "status": "NOT_EXECUTED_QUALITATIVE_GATE_NOT_PASS",
                "qualitative_gate": qualitative_status,
            }
            full_reference_report: dict[str, Any] = {
                "status": "NOT_FIT_BEFORE_RETAINED_QUALITATIVE_AND_TOPOLOGY_GATE",
                "used_for_prediction_construction": False,
            }
            chart_continuation: ChartContinuation | None = None
            chart_metrics: dict[str, Any] | None = None
            if qualitative_status == "PASS" and constrained is not None:
                chart_continuation = build_chart_continuation(holdout, constrained)
                raw_prediction = build_self_continuation(
                    holdout,
                    frontier_band_fraction=0.10,
                    frontier_bins=24,
                    continuation_samples=32,
                )
                reference_normals = estimate_point_normals(withheld, k=min(20, len(withheld)))
                raw_metrics = evaluate_prediction(
                    withheld,
                    reference_normals,
                    _raw_prediction_view(raw_prediction, retained),
                    config,
                    h,
                    reference_label="raw held-out reconstructed Visible Surface evidence",
                )
                wl138_metrics = evaluate_prediction(
                    withheld,
                    reference_normals,
                    _wl138_prediction_view(baseline_continuation),
                    config,
                    h,
                    reference_label="raw held-out reconstructed Visible Surface evidence",
                )
                chart_metrics = evaluate_prediction(
                    withheld,
                    reference_normals,
                    _chart_prediction_view(chart_continuation),
                    config,
                    h,
                    reference_label="raw held-out reconstructed Visible Surface evidence",
                )

                full_representative = fit_physical_chart_surface(
                    deterministic_subsample(holdout.full_points, int(arguments.max_patch_points)),
                    config,
                    role="full_evaluation_only",
                    max_fit_points=int(arguments.max_fit_points),
                    device_name=arguments.device,
                )
                full_topology = audit_representative_topology(
                    full_representative.surface,
                    config,
                    domain_u=full_representative.domain_u,
                    domain_v=full_representative.domain_v,
                    h=h,
                )
                full_reference_report, macro_points, macro_normals = _full_macro_reference_contract(
                    withheld,
                    reference_normals,
                    full_representative,
                    config,
                    h,
                )
                full_reference_report["topology"] = full_topology
                if full_reference_report["status"] == "VALID_MACRO_REFERENCE":
                    full_reference_report["frozen_prediction_vs_macro_reference"] = evaluate_prediction(
                        macro_points,
                        macro_normals,
                        _chart_prediction_view(chart_continuation),
                        config,
                        h,
                        reference_label="held-out reconstructed surface representative",
                    )
                continuation_report = {
                    "status": "EXECUTED_AFTER_QUALITATIVE_AND_TOPOLOGY_PASS",
                    "mechanism": "same first-order representative-frame continuation at frozen physical frontier",
                    "extent_world": float(chart_continuation.l_values[-1]),
                    "withheld_reference_used_for_construction": False,
                    "physical_frontier": _frontier_report(chart_continuation, retained, h),
                    "WL134_raw_frame": raw_metrics,
                    "WL138_unconstrained_representative_frame": wl138_metrics,
                    "WL139_physical_chart_representative_frame": chart_metrics,
                    "useful_signal": _useful_signal(chart_metrics),
                    "useful_signal_criterion": {
                        "median_over_h_at_most": USEFUL_MEDIAN_LIMIT_OVER_H,
                        "coverage_fraction_le_2h_at_least": USEFUL_COVERAGE_LE_2H_MIN,
                        "used_for_geometry": False,
                    },
                }
                visuals.update(write_continuation_visuals(
                    case_root,
                    config,
                    retained,
                    withheld,
                    constrained,
                    raw_prediction,
                    baseline_continuation,
                    chart_continuation,
                ))

            geometry = write_geometry(
                case_root,
                retained,
                withheld,
                baseline,
                constrained,
                chart_continuation,
            )
            case_report = {
                "case": config.as_json(),
                "retained_point_count": int(len(retained)),
                "withheld_point_count": int(len(withheld)),
                "retained_input_sha256": _sha256_rows(retained),
                "withheld_reference_sha256": _sha256_rows(withheld),
                "WL138 FOLDED BASELINE": baseline_contract,
                "FOLDING ATTRIBUTION": baseline_topology,
                "RAW GRAPHNESS AUDIT": graph_report,
                "PHYSICAL-CHART-CONSTRAINED REPRESENTATIVE": constrained_contract,
                "QUALITATIVE MACRO-SHAPE GATE": {
                    "status": qualitative_status,
                    "decision_source": "explicit post-render human inspection" if decision != "PENDING_MANUAL_INSPECTION" else "awaiting explicit post-render human inspection",
                    "low_proximity_can_override_visual_failure": False,
                    "topology_contract_valid": bool(constrained_topology and constrained_topology["topology_contract_valid"]),
                },
                "UNCONSTRAINED vs CHART-CONSTRAINED A/B": {
                    "unconstrained": baseline_contract,
                    "chart_constrained": constrained_contract,
                },
                "TOPOLOGY / JACOBIAN / SELF-OVERLAP ACCOUNTING": {
                    "unconstrained": baseline_topology,
                    "chart_constrained": constrained_topology,
                },
                "NORMAL-FIELD ACCOUNTING": {
                    "unconstrained": _normal_field_accounting(baseline.sampled_normals),
                    "chart_constrained": _normal_field_accounting(constrained.sampled_normals) if constrained is not None else None,
                },
                "PHYSICAL FRONTIER MAPPING": continuation_report.get("physical_frontier", {"status": "NOT_EXECUTED_BEFORE_GATE"}),
                "CONTROLLED CONTINUATION A/B": continuation_report,
                "DISTANCE-FROM-FRONTIER RESULT": (
                    chart_metrics["distance_from_physical_frontier_bins"] if chart_metrics is not None else {"status": "NOT_EVALUATED_BEFORE_GATE"}
                ),
                "FULL MACRO REFERENCE VALIDITY": full_reference_report,
                "visuals": visuals,
                "geometry": geometry,
            }
            (case_root / "case_report.json").write_text(json.dumps(_jsonable(case_report), indent=2), encoding="utf-8")
            case_reports[config.name] = case_report
        except Exception as error:
            failures.append({"case": config.name, "error": repr(error)})

    primary = case_reports.get(CURVED_RIM_CASE.name)
    if primary is None:
        verdict = "E"
        verdict_description = "E. IMPLEMENTATION / ATTRIBUTION INCONCLUSIVE"
    elif primary["RAW GRAPHNESS AUDIT"]["status"] != "PASS_GRAPH_LIKE":
        verdict = "D"
        verdict_description = "D. RAW ROI IS NOT GRAPH-LIKE"
    elif primary["QUALITATIVE MACRO-SHAPE GATE"]["status"] in {"FAIL", "FAIL_TOPOLOGY_CONTRACT"}:
        verdict = "C"
        verdict_description = "C. GRAPH-CONSTRAINED REPRESENTATIVE INSUFFICIENT"
    elif primary["QUALITATIVE MACRO-SHAPE GATE"]["status"] != "PASS":
        verdict = "E"
        verdict_description = "E. IMPLEMENTATION / ATTRIBUTION INCONCLUSIVE — awaiting explicit qualitative inspection"
    elif primary["CONTROLLED CONTINUATION A/B"].get("useful_signal", False):
        verdict = "A"
        verdict_description = "A. PHYSICAL-CHART REPRESENTATIVE PROMOTED"
    else:
        verdict = "B"
        verdict_description = "B. REPRESENTATIVE PROMOTED, CONTINUATION STILL LIMITED"

    baseline_has_chart_pathology = bool(primary and (
        primary["FOLDING ATTRIBUTION"]["physical_u_reversal_count"] > 0
        or primary["FOLDING ATTRIBUTION"]["physical_v_reversal_count"] > 0
        or primary["FOLDING ATTRIBUTION"]["jacobian_orientation_flip_count"] > 0
        or primary["FOLDING ATTRIBUTION"]["duplicate_multivalued_chart_bins"] > 0
    ))
    any_baseline_has_chart_pathology = any(
        case["FOLDING ATTRIBUTION"]["physical_u_reversal_count"] > 0
        or case["FOLDING ATTRIBUTION"]["physical_v_reversal_count"] > 0
        or case["FOLDING ATTRIBUTION"]["jacobian_orientation_flip_count"] > 0
        or case["FOLDING ATTRIBUTION"]["duplicate_multivalued_chart_bins"] > 0
        for case in case_reports.values()
    )
    primary_chart_valid = bool(primary and primary["QUALITATIVE MACRO-SHAPE GATE"]["topology_contract_valid"])
    primary_macro_pass = bool(primary and primary["QUALITATIVE MACRO-SHAPE GATE"]["status"] == "PASS")
    true_result = {
        "status": "NOT_EXECUTED_CONTROLLED_SIGNAL_INSUFFICIENT",
        "reason": "requires a genuinely useful controlled curved-rim signal under the frozen WL138 criterion",
    }
    if primary and primary["CONTROLLED CONTINUATION A/B"].get("useful_signal", False):
        if CANDIDATE_B_ARCHIVE.exists():
            true_result = {
                "status": "NOT_EXECUTED_PENDING_CANDIDATE_B_SUPPORTED_MICRO_REGION_VALIDATION",
                "reason": "controlled gate passed; Candidate B exists but a broad adjacent persistent-occlusion region still requires read-only validation",
                "candidate_b_archive": str(CANDIDATE_B_ARCHIVE),
            }
        else:
            true_result = {
                "status": "NOT_EXECUTED_CANDIDATE_B_ARCHIVE_UNAVAILABLE",
                "reason": "bounded 0-16h controlled signal exists, but the required read-only Candidate B archive is absent from this workspace",
                "candidate_b_archive": str(CANDIDATE_B_ARCHIVE),
                "frozen_supported_local_range_world": float(16.0 * h),
            }

    report = {
        "batch": "Worklog 139 physical-chart-constrained Surface Representative and controlled continuation closure",
        "status": "ISOLATED_NON_CANONICAL_RESEARCH_DEMO",
        "meeting_verdict": verdict,
        "meeting_verdict_description": verdict_description,
        "INTENT ALIGNMENT": {
            "question_folding_attribution": "Was WL138 pathological geometry caused by allowing the parameterization to distort the physical chart?",
            "question_continuation_basis": "Does fixed physical-chart semantics yield one coherent macro surface usable for controlled continuation?",
            "raw_evidence_is_representative": False,
            "occluded_surface_solved": False,
        },
        "IMPLEMENTATION FIDELITY": {
            "canonical_code_modified": False,
            "historical_fitter_modified": False,
            "wl127_mesh_modified": False,
            "wl138_module_sha256_before": wl138_sha_before,
            "wl138_module_sha256_after": _file_sha256(WL138_MODULE),
            "wl138_historical_output_path_written": False,
            "wl138_confirmed_manifest_before": wl138_manifest_before,
            "wl138_confirmed_manifest_after": _artifact_manifest(wl138_root),
            "wl138_confirmed_artifacts_unchanged": wl138_manifest_before == _artifact_manifest(wl138_root),
            "h": h,
            "mu": mu,
            "fixed_settings": {
                "resolution_u": GRAPH_RESOLUTION_U,
                "resolution_v": GRAPH_RESOLUTION_V,
                "degree_u": GRAPH_DEGREE_U,
                "degree_v": GRAPH_DEGREE_V,
                "smoothness_lambda": GRAPH_SMOOTHNESS_LAMBDA,
                "tikhonov_lambda": GRAPH_TIKHONOV_LAMBDA,
                "operational_difference": "physical UV is immutable, so WL138 3D foot-point correction rounds are mathematically prohibited; solve only scalar n controls once",
            },
            "display_only": {
                "point_alpha": DISPLAY_POINT_ALPHA,
                "reference_alpha": DISPLAY_REFERENCE_ALPHA,
                "point_size": DISPLAY_POINT_SIZE,
                "voxel_world": DISPLAY_VOXEL_WORLD,
                "geometry_and_metrics_unchanged": True,
            },
            "withheld_xyz_entered_retained_fit": False,
            "withheld_xyz_entered_continuation": False,
            "full_representative_entered_continuation": False,
        },
        "WL138 FOLDED BASELINE": {name: case["WL138 FOLDED BASELINE"] for name, case in case_reports.items()},
        "FOLDING ATTRIBUTION": {
            "cases": {name: case["FOLDING ATTRIBUTION"] for name, case in case_reports.items()},
            "primary_chart_pathology_measured": baseline_has_chart_pathology,
            "any_case_chart_pathology_measured": any_baseline_has_chart_pathology,
            "primary_attribution": "curved rim has no measured reversal/flip; chart constraint guarantees topology but does not attribute a pre-existing fold there" if not baseline_has_chart_pathology else "physical-chart reversal/multi-occupancy explains primary visible folding",
            "cross_case_attribution": "leg/brace WL138 folding has measured chart reversal/flip, but retained raw leg is materially multi-valued and cannot validate the graph family",
            "closure_answer": "partially attributed: free-3D parameterization permits real pathology on leg/brace, while curved-rim baseline pathology is not a discrete chart reversal",
        },
        "RAW GRAPHNESS AUDIT": {name: case["RAW GRAPHNESS AUDIT"] for name, case in case_reports.items()},
        "PHYSICAL-CHART-CONSTRAINED REPRESENTATIVE": {name: case["PHYSICAL-CHART-CONSTRAINED REPRESENTATIVE"] for name, case in case_reports.items()},
        "QUALITATIVE MACRO-SHAPE GATE": {name: case["QUALITATIVE MACRO-SHAPE GATE"] for name, case in case_reports.items()},
        "UNCONSTRAINED vs CHART-CONSTRAINED A/B": {name: case["UNCONSTRAINED vs CHART-CONSTRAINED A/B"] for name, case in case_reports.items()},
        "TOPOLOGY / JACOBIAN / SELF-OVERLAP ACCOUNTING": {name: case["TOPOLOGY / JACOBIAN / SELF-OVERLAP ACCOUNTING"] for name, case in case_reports.items()},
        "NORMAL-FIELD ACCOUNTING": {name: case["NORMAL-FIELD ACCOUNTING"] for name, case in case_reports.items()},
        "PHYSICAL FRONTIER MAPPING": {name: case["PHYSICAL FRONTIER MAPPING"] for name, case in case_reports.items()},
        "CONTROLLED CONTINUATION A/B": {name: case["CONTROLLED CONTINUATION A/B"] for name, case in case_reports.items()},
        "DISTANCE-FROM-FRONTIER RESULT": {name: case["DISTANCE-FROM-FRONTIER RESULT"] for name, case in case_reports.items()},
        "FULL MACRO REFERENCE VALIDITY": {name: case["FULL MACRO REFERENCE VALIDITY"] for name, case in case_reports.items()},
        "CONDITIONAL TRUE-OCCLUDED MICRO DEMO": true_result,
        "PROMOTED": {
            "Visible Surface Evidence and Surface Representative are separate layers": True,
            "physical chart preservation for graph-like representative": bool(primary_chart_valid and primary_macro_pass),
            "chart-constrained NURBS as broad macro representative": bool(primary_chart_valid and primary_macro_pass),
            "representative analytic frame for bounded continuation": bool(verdict == "A"),
        },
        "RETAINED": [
            "NURBS/B-spline representation family",
            "WL127 raw TSDF mesh as immutable evidence carrier",
            "first-order continuation as frozen local diagnostic propagation",
            "raw normals as diagnostic/support quantities only",
        ],
        "REJECTED": [
            "unconstrained free-3D NURBS as sufficient macro representative when physical chart reversals are measured",
            "raw-to-surface nearest distance as sufficient macro-shape validation",
            "NURBS parameter-space semantics replacing physical world-space semantics",
        ] if any_baseline_has_chart_pathology else [
            "raw-to-surface nearest distance as sufficient macro-shape validation",
            "NURBS parameter-space semantics replacing physical world-space semantics",
        ],
        "OPEN": [
            "non-graph and multi-sheet representatives",
            "thin structures when graphness fails",
            "long-range completion and junction/closure prior",
            "confidence and principled continuation extent",
        ],
        "cases": case_reports,
        "failures": failures,
        "inputs": {
            "confirmed_wl138_root": str(wl138_root),
            "historical_wl127_mesh_cache_provenance": historical_root_report["inputs"]["mesh_cache"],
            "historical_wl127_field_cache_provenance": historical_root_report["inputs"]["field_cache"],
            "candidate_b_archive_checked": str(CANDIDATE_B_ARCHIVE),
            "output_root": str(output_root),
        },
    }
    report_path = output_root / "physical_chart_surface_representative_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    lines = [
        "# Worklog 139 physical-chart-constrained Surface Representative",
        "",
        "이 출력은 WL127 raw evidence와 WL138 frozen unconstrained baseline을 수정하지 않고,",
        "physical u/v chart를 고정한 8x4 degree-2 scalar graph B-spline을 별도 평가한다.",
        "raw/reference point는 near-opaque이며 모든 display thinning은 metric/geometry와 분리된다.",
        "",
        f"판정: {report['meeting_verdict_description']}",
        "",
        "각 case 폴더의 PNG는 동일 local physical-chart view를 사용하며 PLY/NPZ는 geometry/ 폴더에 있다.",
        "full chart representative는 evaluation-only이고 physical u > fixed u_cut으로만 held-out side를 선택한다.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl138-root", type=Path, default=WL138_CONFIRMED_ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-fit-points", type=int, default=12000)
    parser.add_argument("--max-reference-points", type=int, default=12000)
    parser.add_argument("--max-patch-points", type=int, default=24000)
    parser.add_argument("--qualitative-pass", action="append", choices=(CURVED_RIM_CASE.name, LEG_CASE.name), default=[])
    parser.add_argument("--qualitative-fail", action="append", choices=(CURVED_RIM_CASE.name, LEG_CASE.name), default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_demo(build_arg_parser().parse_args(argv))
    print(json.dumps({
        "meeting_verdict": report["meeting_verdict"],
        "description": report["meeting_verdict_description"],
        "cases": list(report["cases"]),
        "failures": report["failures"],
        "true_occluded": report["CONDITIONAL TRUE-OCCLUDED MICRO DEMO"]["status"],
    }, indent=2))
    return 0 if report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
