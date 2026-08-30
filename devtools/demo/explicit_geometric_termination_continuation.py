"""Explicit geometric termination mapping for the frozen Worklog 129 arm.

This is a separate, non-canonical attribution experiment.  It preserves the
Worklog 128/129 fit, ROIs, holdout population, and first-order continuation
model, but replaces the arbitrary rectangular NURBS ``u=1`` start edge with
the intersection of the frozen observed-side NURBS and the fixed physical
holdout plane.

No withheld XYZ is used to construct roots, directions, or the continuation
horizon.  The withheld reference is used only for evaluation and display.
No second derivative or stronger completion rule is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.corrected_first_order_parametric_continuation import (  # noqa: E402
    FrozenCase,
    _load_frozen_case,
    _surface_from_grid,
)
from devtools.demo.parametric_continuation_attribution import (  # noqa: E402
    _angle_degrees,
    _evaluate_boundary,
    _prediction_metrics,
    _roi_vertex_contract,
    _write_distance_figure,
    boundary_support_report,
    distance_to_termination_report,
    geometric_interface_report,
    trace_mesh_interface,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    PRIMARY_ROI,
    SECONDARY_ROI,
    ROIConfig,
    _jsonable,
    _plot_coords,
    _scatter3d,
    _set_equal_3d_limits,
    deterministic_indices,
    estimate_point_normals,
    roi_coordinates,
)


WORKLOG_128_COMMIT = "2d87366b910873562b9dfc223408d85257c5af9f"
WORKLOG_129_COMMIT = "1ca0da5"
WORKLOG_130_COMMIT = "8b2b4e7"
HOLDOUT_CUT = 0.58
ROOT_SAMPLES = 257
ROOT_BISECTION_STEPS = 48
ROOT_DEDUP_TOLERANCE = 1e-7
ROOT_RESIDUAL_TOLERANCE = 1e-6
TERMINATION_SAMPLES = 32
PREDICTION_L_SAMPLES = 96
TARGET_SUPPORT_HORIZON = 2.0
MATERIAL_MEDIAN_IMPROVEMENT = 0.25
MATERIAL_COVERAGE_GAIN = 0.10


@dataclass
class RootRow:
    v: float
    roots_u: list[float]
    selected_u: float | None
    root_residuals: list[float]
    selected_root_index: int | None


@dataclass
class TerminationCurve:
    gamma_uv: np.ndarray
    gamma_points: np.ndarray
    derivatives_uv: np.ndarray
    physical_direction: np.ndarray
    local_u_derivative: np.ndarray
    support_fit_distance: np.ndarray
    support_mesh_distance: np.ndarray
    root_rows: list[RootRow]
    valid_mask: np.ndarray


@dataclass
class ExplicitPrediction:
    points: np.ndarray
    normals: np.ndarray
    l_values: np.ndarray
    valid_v: np.ndarray
    support_mask: np.ndarray


def _axis(config: ROIConfig, name: str) -> np.ndarray:
    value = np.asarray(getattr(config, name), dtype=np.float64)
    return value / np.clip(np.linalg.norm(value), 1e-12, None)


def termination_plane(config: ROIConfig) -> dict[str, Any]:
    """Return the fixed physical plane from ROI metadata only."""

    axis_u = _axis(config, "axis_u")
    axis_v = _axis(config, "axis_v")
    axis_n = _axis(config, "axis_n")
    origin = np.asarray(config.origin, dtype=np.float64)
    local_u = float(config.u_bounds[0]) + float(config.holdout_u_cut) * float(config.u_bounds[1] - config.u_bounds[0])
    world_u = float(origin @ axis_u) + local_u
    return {
        "origin": origin,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "axis_n": axis_n,
        "local_u_world_coordinate": local_u,
        "world_plane_equation": f"dot(x, axis_u) = {world_u:.12g}",
        "world_u_coordinate": world_u,
        "source": "frozen ROI origin/axis/bounds and holdout_u_cut only",
        "withheld_xyz_used": False,
    }


def _local_u_value(points: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ plane["axis_u"]


def _evaluate_surface_points(surface: Any, uv: np.ndarray) -> np.ndarray:
    import torch

    tensor_uv = torch.as_tensor(uv, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    with torch.no_grad():
        points = surface.evaluate(tensor_uv)
    return points.detach().cpu().numpy().astype(np.float64)


def _evaluate_surface_derivatives(surface: Any, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    tensor_uv = torch.as_tensor(uv, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    with torch.no_grad():
        points, derivative_u, derivative_v = surface.evaluate_with_derivatives(tensor_uv)
    return (
        points.detach().cpu().numpy().astype(np.float64),
        derivative_u.detach().cpu().numpy().astype(np.float64),
        derivative_v.detach().cpu().numpy().astype(np.float64),
    )


def _bisect_root(surface: Any, v: float, left: float, right: float, plane: dict[str, Any]) -> tuple[float, float]:
    uv = np.asarray([[left, v], [right, v]], dtype=np.float64)
    values = _local_u_value(_evaluate_surface_points(surface, uv), plane) - float(plane["world_u_coordinate"])
    f_left, f_right = map(float, values)
    if abs(f_left) <= ROOT_RESIDUAL_TOLERANCE:
        return float(left), abs(f_left)
    if abs(f_right) <= ROOT_RESIDUAL_TOLERANCE:
        return float(right), abs(f_right)
    if f_left * f_right > 0.0:
        raise ValueError("bisection interval does not bracket a root")
    for _ in range(ROOT_BISECTION_STEPS):
        middle = 0.5 * (left + right)
        f_middle = float(_local_u_value(_evaluate_surface_points(surface, np.asarray([[middle, v]], dtype=np.float64)), plane)[0] - plane["world_u_coordinate"])
        if abs(f_middle) <= ROOT_RESIDUAL_TOLERANCE:
            left = right = middle
            break
        if f_left * f_middle <= 0.0:
            right, f_right = middle, f_middle
        else:
            left, f_left = middle, f_middle
    root = 0.5 * (left + right)
    residual = abs(float(_local_u_value(_evaluate_surface_points(surface, np.asarray([[root, v]], dtype=np.float64)), plane)[0] - plane["world_u_coordinate"]))
    return root, residual


def solve_plane_intersections(surface: Any, plane: dict[str, Any], samples_v: int = TERMINATION_SAMPLES) -> tuple[list[RootRow], np.ndarray]:
    """Find every deterministic u-root for each fixed v sample."""

    v_values = np.linspace(0.0, 1.0, int(samples_v), dtype=np.float64)
    u_values = np.linspace(0.0, 1.0, ROOT_SAMPLES, dtype=np.float64)
    rows: list[RootRow] = []
    for value_v in v_values.tolist():
        uv = np.column_stack([u_values, np.full_like(u_values, value_v)])
        surface_points = _evaluate_surface_points(surface, uv)
        function_values = _local_u_value(surface_points, plane) - float(plane["world_u_coordinate"])
        roots: list[float] = []
        residuals: list[float] = []
        for index in range(len(u_values) - 1):
            left_value = float(function_values[index])
            right_value = float(function_values[index + 1])
            if abs(left_value) <= ROOT_RESIDUAL_TOLERANCE:
                roots.append(float(u_values[index]))
                residuals.append(abs(left_value))
            if left_value * right_value < 0.0:
                root, residual = _bisect_root(surface, value_v, float(u_values[index]), float(u_values[index + 1]), plane)
                roots.append(root)
                residuals.append(residual)
        if abs(float(function_values[-1])) <= ROOT_RESIDUAL_TOLERANCE:
            roots.append(1.0)
            residuals.append(abs(float(function_values[-1])))
        order = np.argsort(np.asarray(roots), kind="mergesort") if roots else np.zeros((0,), dtype=np.int64)
        unique_roots: list[float] = []
        unique_residuals: list[float] = []
        for index in order.tolist():
            root = float(roots[index])
            if not unique_roots or abs(root - unique_roots[-1]) > ROOT_DEDUP_TOLERANCE:
                unique_roots.append(root)
                unique_residuals.append(float(residuals[index]))
            else:
                unique_residuals[-1] = min(unique_residuals[-1], float(residuals[index]))
        # The selected branch is the largest observed-side root.  This is a
        # fixed observed-only convention and is never chosen by target error.
        selected = unique_roots[-1] if unique_roots else None
        selected_index = len(unique_roots) - 1 if unique_roots else None
        rows.append(RootRow(value_v, unique_roots, selected, unique_residuals, selected_index))
    return rows, v_values


def physical_first_order_direction(surface: Any, gamma_uv: np.ndarray, plane: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the physical +u axis into the NURBS tangent plane and normalise d(local_u)/dl=1."""

    points, derivative_u, derivative_v = _evaluate_surface_derivatives(surface, gamma_uv)
    axis_u = np.asarray(plane["axis_u"], dtype=np.float64)
    directions = np.full_like(points, np.nan, dtype=np.float64)
    duv = np.full((len(points), 2), np.nan, dtype=np.float64)
    local_derivative = np.full((len(points),), np.nan, dtype=np.float64)
    for index, (su, sv) in enumerate(zip(derivative_u, derivative_v)):
        jacobian = np.column_stack([su, sv])
        gram = jacobian.T @ jacobian
        try:
            coefficients = np.linalg.solve(gram + 1e-10 * np.eye(2), jacobian.T @ axis_u)
        except np.linalg.LinAlgError:
            continue
        tangent_projection = jacobian @ coefficients
        denominator = float(axis_u @ tangent_projection)
        if not np.isfinite(denominator) or denominator <= 1e-8:
            continue
        coefficients = coefficients / denominator
        direction = jacobian @ coefficients
        duv[index] = coefficients
        directions[index] = direction
        local_derivative[index] = float(axis_u @ direction)
    return points, duv, directions, local_derivative


def _termination_curve(
    surface: Any,
    plane: dict[str, Any],
    vertex_roi: Any,
    replay_fit_points: np.ndarray,
    h: float,
) -> TerminationCurve:
    from scipy.spatial import cKDTree

    root_rows, v_values = solve_plane_intersections(surface, plane)
    valid = np.asarray([row.selected_u is not None for row in root_rows], dtype=bool)
    gamma_uv = np.column_stack([
        np.asarray([row.selected_u if row.selected_u is not None else np.nan for row in root_rows], dtype=np.float64),
        v_values,
    ])
    gamma_points = np.full((len(root_rows), 3), np.nan, dtype=np.float64)
    derivatives_uv = np.full((len(root_rows), 2), np.nan, dtype=np.float64)
    directions = np.full((len(root_rows), 3), np.nan, dtype=np.float64)
    local_derivative = np.full((len(root_rows),), np.nan, dtype=np.float64)
    if bool(valid.any()):
        selected_uv = gamma_uv[valid]
        points, duv, direction, derivative = physical_first_order_direction(surface, selected_uv, plane)
        gamma_points[valid] = points
        derivatives_uv[valid] = duv
        directions[valid] = direction
        local_derivative[valid] = derivative
    fit_tree = cKDTree(replay_fit_points)
    mesh_tree = cKDTree(vertex_roi.observed_points)
    fit_distance = np.full((len(root_rows),), np.nan, dtype=np.float64)
    mesh_distance = np.full((len(root_rows),), np.nan, dtype=np.float64)
    if bool(valid.any()):
        fit_distance[valid] = fit_tree.query(gamma_points[valid], workers=1)[0] / h
        mesh_distance[valid] = mesh_tree.query(gamma_points[valid], workers=1)[0] / h
    return TerminationCurve(gamma_uv, gamma_points, derivatives_uv, directions, local_derivative, fit_distance, mesh_distance, root_rows, valid)


def build_explicit_prediction(curve: TerminationCurve, config: ROIConfig, h: float) -> ExplicitPrediction:
    valid = curve.valid_mask & np.isfinite(curve.support_fit_distance) & np.isfinite(curve.physical_direction).all(axis=1)
    l_max = float(config.u_bounds[1] - config.u_bounds[0]) * (1.0 - float(config.holdout_u_cut))
    l_values = np.linspace(0.0, l_max, PREDICTION_L_SAMPLES, dtype=np.float64)
    valid_v = curve.gamma_uv[valid, 1]
    gamma = curve.gamma_points[valid]
    direction = curve.physical_direction[valid]
    support = curve.support_fit_distance[valid] <= TARGET_SUPPORT_HORIZON
    if len(gamma) == 0:
        return ExplicitPrediction(np.zeros((0, 3)), np.zeros((0, 3)), l_values, valid_v, support)
    points_grid = gamma[None, :, :] + l_values[:, None, None] * direction[None, :, :]
    if len(valid_v) >= 2:
        derivative_s = np.gradient(points_grid, valid_v, axis=1, edge_order=1)
    else:
        derivative_s = np.zeros_like(points_grid)
    derivative_l = np.broadcast_to(direction[None, :, :], points_grid.shape)
    normals = np.cross(derivative_l, derivative_s, axis=-1)
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12, None)
    return ExplicitPrediction(points_grid.reshape(-1, 3), normals.reshape(-1, 3), l_values, valid_v, support)


def _fixed_plane_patch(config: ROIConfig, plane: dict[str, Any]) -> np.ndarray:
    origin = np.asarray(config.origin, dtype=np.float64)
    return np.asarray([
        origin + plane["local_u_world_coordinate"] * plane["axis_u"] + config.v_bounds[0] * plane["axis_v"] + config.n_bounds[0] * plane["axis_n"],
        origin + plane["local_u_world_coordinate"] * plane["axis_u"] + config.v_bounds[1] * plane["axis_v"] + config.n_bounds[0] * plane["axis_n"],
        origin + plane["local_u_world_coordinate"] * plane["axis_u"] + config.v_bounds[1] * plane["axis_v"] + config.n_bounds[1] * plane["axis_n"],
        origin + plane["local_u_world_coordinate"] * plane["axis_u"] + config.v_bounds[0] * plane["axis_v"] + config.n_bounds[1] * plane["axis_n"],
    ], dtype=np.float64)


def _interface_metrics(points: np.ndarray, trace: Any, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return {"status": "no_valid_termination_points"}
    distances = cKDTree(trace.interface_points).query(points, workers=1)[0]
    return {
        "position_error_over_h": {
            "median": float(np.median(distances) / h),
            "p95": float(np.percentile(distances, 95) / h),
        },
        "interface_support_fraction": {
            "le_h": float(np.mean(distances <= h)),
            "le_2h": float(np.mean(distances <= 2.0 * h)),
        },
        "samples": int(len(points)),
        "definition": "candidate termination samples to fixed mesh-face interface",
    }


def _surface_grid(surface: Any, samples_u: int = 24, samples_v: int = 16) -> np.ndarray:
    uv = np.asarray([[u, v] for u in np.linspace(0.0, 1.0, samples_u) for v in np.linspace(0.0, 1.0, samples_v)], dtype=np.float64)
    return _evaluate_surface_points(surface, uv)


def _load_wl130_parameterization(report_path: Path, case_name: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for item in report.get("cases", []):
        if item.get("roi", {}).get("name") == case_name:
            return item["parameterization"]
    raise KeyError(f"missing Worklog 130 parameterization for {case_name}")


def _report_parameterization(item: dict[str, Any]) -> dict[str, Any]:
    u = item["u"]
    materially_different = bool(
        u["pearson"] < 0.99
        or u["spearman"] < 0.99
        or u["median_absolute_shift"] > 0.02
        or u["p95_absolute_shift"] > 0.10
        or abs(u["affine_best_fit"]["slope"] - 1.0) > 0.05
    )
    return {
        "measured_worklog_130_values_preserved": item,
        "materially_different_from_identity": materially_different,
        "identity_interpretation_rule": "Pearson/Spearman >=.99, median shift <=.02, p95 shift <=.10, affine slope within .05 of 1",
        "old_terminal_u_ge_0.95_gate_used": False,
    }


def _normal_metrics(reference: np.ndarray, predicted: np.ndarray, normals: np.ndarray, h: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    return _prediction_metrics(reference, predicted, normals, h)


def _arm_metrics(reference: np.ndarray, points: np.ndarray, normals: np.ndarray, h: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    return _normal_metrics(reference, points, normals, h)


def _distance_bins(
    reference: np.ndarray,
    predicted: np.ndarray,
    predicted_normals: np.ndarray,
    interface_points: np.ndarray,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    target_interface_distance = cKDTree(interface_points).query(reference, workers=1)[0] / h
    if len(predicted):
        metrics, distances, nearest = _arm_metrics(reference, predicted, predicted_normals, h)
        reference_normals = estimate_point_normals(reference, k=20)
        angles = None
        if reference_normals is not None:
            angles = _angle_degrees(reference_normals, predicted_normals[nearest])
    else:
        metrics = {"status": "no explicit termination roots; no prediction surface"}
        distances = np.zeros((0,), dtype=np.float64)
        angles = None
    edges = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)
    labels = ("0–1h", "1–2h", "2–4h", "4–8h", "8–16h", ">16h")
    rows = []
    for index, label in enumerate(labels):
        selected = (target_interface_distance >= edges[index]) & (target_interface_distance < edges[index + 1])
        row: dict[str, Any] = {"bin": label, "reference_count": int(selected.sum())}
        if len(predicted) and bool(selected.any()):
            row.update({
                "median_error_over_h": float(np.median(distances[selected]) / h),
                "p95_error_over_h": float(np.percentile(distances[selected], 95) / h),
                "coverage_le_h": float(np.mean(distances[selected] <= h)),
                "coverage_le_2h": float(np.mean(distances[selected] <= 2.0 * h)),
                "normal_median_degrees": float(np.median(angles[selected])) if angles is not None else None,
                "normal_p95_degrees": float(np.percentile(angles[selected], 95)) if angles is not None else None,
            })
        else:
            row.update({key: None for key in ("median_error_over_h", "p95_error_over_h", "coverage_le_h", "coverage_le_2h", "normal_median_degrees", "normal_p95_degrees")})
        rows.append(row)
    return {
        "distance_definition": "Euclidean distance from frozen mesh-face interface, divided by h",
        "fixed_bins": rows,
        "all_reference_metrics": metrics,
    }


def _supported_attribution(
    case: FrozenCase,
    curve: TerminationCurve,
    prediction: ExplicitPrediction,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    supported_valid = curve.valid_mask.copy()
    supported_valid[curve.valid_mask] = curve.support_fit_distance[curve.valid_mask] <= TARGET_SUPPORT_HORIZON
    supported_indices = np.flatnonzero(supported_valid)
    if len(supported_indices) == 0:
        return {
            "status": "empty",
            "support_rule": "nearest observed fitting point <= 2h",
            "termination_curve_fraction": 0.0,
            "target_population": 0,
        }
    supported_v = curve.gamma_uv[supported_indices, 1]
    target_uv = roi_coordinates(case.reference_eval_points, case.config)
    target_v = (target_uv[:, 1] - case.config.v_bounds[0]) / float(case.config.v_bounds[1] - case.config.v_bounds[0])
    # The attribution target is selected from the fixed v correspondence, not
    # from prediction error.  It is never used to alter the full population.
    target_mask = np.min(np.abs(target_v[:, None] - supported_v[None, :]), axis=1) <= (1.0 / max(len(curve.gamma_uv) - 1, 1))
    target = case.reference_eval_points[target_mask]
    prediction_points = prediction.points.reshape(PREDICTION_L_SAMPLES, len(prediction.valid_v), 3)
    supported_columns = prediction.support_mask
    attribution_points = prediction_points[:, supported_columns, :].reshape(-1, 3)
    attribution_normals = prediction.normals.reshape(PREDICTION_L_SAMPLES, len(prediction.valid_v), 3)[:, supported_columns, :].reshape(-1, 3)
    if len(target) == 0 or len(attribution_points) == 0:
        return {
            "status": "empty",
            "support_rule": "nearest observed fitting point <= 2h",
            "termination_curve_fraction": float(len(supported_indices) / max(int(curve.valid_mask.sum()), 1)),
            "target_population": int(len(target)),
        }
    metrics, _distances, _nearest = _prediction_metrics(target, attribution_points, attribution_normals, h)
    return {
        "status": "available",
        "support_rule": "nearest observed fitting point <= 2h",
        "termination_curve_fraction": float(len(supported_indices) / max(int(curve.valid_mask.sum()), 1)),
        "termination_curve_supported_samples": int(len(supported_indices)),
        "target_population": int(len(target)),
        "target_fraction_of_original_worklog_129_population": float(len(target) / max(len(case.reference_eval_points), 1)),
        "target_selection": "fixed manual-v correspondence to supported Gamma samples; no prediction error selection",
        "metrics": metrics,
    }


def _case_report(
    case: FrozenCase,
    parameterization: dict[str, Any],
    plane: dict[str, Any],
    curve: TerminationCurve,
    prediction: ExplicitPrediction,
    vertex_roi: Any,
    trace: Any,
    historical_points: np.ndarray,
    historical_normals: np.ndarray,
    corrected_metrics_frozen: dict[str, Any],
    h: float,
    corrected_source: str,
) -> dict[str, Any]:
    full_metrics_a, distances_a, nearest_a = _arm_metrics(case.reference_eval_points, historical_points, historical_normals, h)
    if len(prediction.points):
        full_metrics_b, distances_b, nearest_b = _arm_metrics(case.reference_eval_points, prediction.points, prediction.normals, h)
    else:
        full_metrics_b = {"status": "no explicit termination roots; no Arm B prediction surface"}
        distances_b = np.zeros((0,), dtype=np.float64)
        nearest_b = np.zeros((0,), dtype=np.int64)
    valid_gamma = curve.gamma_points[curve.valid_mask]
    finite_fit_support = curve.support_fit_distance[np.isfinite(curve.support_fit_distance)]
    finite_mesh_support = curve.support_mesh_distance[np.isfinite(curve.support_mesh_distance)]
    finite_local_derivative = curve.local_u_derivative[np.isfinite(curve.local_u_derivative)]

    def summary(values: np.ndarray) -> dict[str, Any]:
        if len(values) == 0:
            return {"status": "no valid termination roots"}
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "fraction_le_h": float(np.mean(values <= 1.0)),
            "fraction_le_2h": float(np.mean(values <= 2.0)),
        }

    def derivative_summary(values: np.ndarray) -> dict[str, Any]:
        if len(values) == 0:
            return {"status": "no valid termination roots"}
        return {
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    # Historical edge is materialised by the caller and stored on the report.
    return {
        "roi": case.config.as_json(),
        "parameterization_reinterpretation": parameterization,
        "termination_plane": {key: value for key, value in plane.items() if key not in ("origin", "axis_u", "axis_v", "axis_n")},
        "root_contract": {
            "sample_count_v": int(len(curve.root_rows)),
            "root_coverage_over_v_domain": float(np.mean(curve.valid_mask)),
            "multiple_root_v_count": int(sum(len(row.roots_u) > 1 for row in curve.root_rows)),
            "maximum_root_count": int(max((len(row.roots_u) for row in curve.root_rows), default=0)),
            "selection_rule": "largest u root in [0,1], observed-only and target-independent",
            "root_residual_tolerance_physical_local_u": ROOT_RESIDUAL_TOLERANCE,
            "root_rows": [row.__dict__ for row in curve.root_rows],
        },
        "termination_support": {
            "support_rule_for_attribution": "nearest observed fitting point <= 2h",
            "root_count": int(len(valid_gamma)),
            "fit_distance_over_h": summary(finite_fit_support),
            "observed_mesh_distance_over_h": summary(finite_mesh_support),
            "observed_supported_termination_fraction": float(np.mean(finite_fit_support <= TARGET_SUPPORT_HORIZON)) if len(finite_fit_support) else 0.0,
        },
        "physical_direction_contract": {
            "direction_source": "projection of fixed ROI axis_u into frozen NURBS tangent plane",
            "normalization": "d local_u_world / dl = +1",
            "local_u_derivative": derivative_summary(finite_local_derivative),
            "withheld_xyz_used": False,
        },
        "explicit_prediction": {
            "predicted_point_count": int(len(prediction.points)),
            "continuation_extent_physical_local_u": float(prediction.l_values[-1]) if len(prediction.l_values) else 0.0,
            "l_definition": "fixed ROI withheld extent = (u1-u0)*(1-holdout_u_cut)",
            "actual_local_u_span": {
                "min": float(np.min(roi_coordinates(prediction.points, case.config)[:, 0])) if len(prediction.points) else None,
                "max": float(np.max(roi_coordinates(prediction.points, case.config)[:, 0])) if len(prediction.points) else None,
                "span": float(np.ptp(roi_coordinates(prediction.points, case.config)[:, 0])) if len(prediction.points) else None,
            },
            "first_order_only": True,
        },
        "arm_A_historical_u1_first_order": {
            "source": corrected_source,
            "full_population_metrics_recomputed": full_metrics_a,
            "frozen_worklog_129_metrics": corrected_metrics_frozen,
            "full_population_metric_unchanged": bool(
                math.isclose(full_metrics_a["median_over_h"], corrected_metrics_frozen["point_to_predicted_surface_distance"]["median_over_h"], rel_tol=2e-5, abs_tol=2e-5)
                and math.isclose(full_metrics_a["p95_over_h"], corrected_metrics_frozen["point_to_predicted_surface_distance"]["p95_over_h"], rel_tol=2e-5, abs_tol=2e-5)
                and math.isclose(full_metrics_a["coverage_le_h"], corrected_metrics_frozen["withheld_reference_coverage"]["fraction_le_h"], rel_tol=2e-5, abs_tol=2e-5)
                and math.isclose(full_metrics_a["coverage_le_2h"], corrected_metrics_frozen["withheld_reference_coverage"]["fraction_le_2h"], rel_tol=2e-5, abs_tol=2e-5)
            ),
            "actual_local_u_span": {
                "min": float(np.min(roi_coordinates(historical_points, case.config)[:, 0])),
                "max": float(np.max(roi_coordinates(historical_points, case.config)[:, 0])),
                "span": float(np.ptp(roi_coordinates(historical_points, case.config)[:, 0])),
            },
        },
        "arm_B_explicit_termination_first_order": {
            "full_population_metrics": full_metrics_b,
            "withheld_evaluation_population": "exact frozen Worklog 129 reference_eval_points",
            "first_order_only": True,
            "no_withheld_xyz_in_root_direction_or_L": True,
        },
        "full_population_ab": {
            "historical_arm_A": full_metrics_a,
            "explicit_arm_B": full_metrics_b,
            "evaluation_population_unchanged": int(len(case.reference_eval_points)),
        },
        "supported_termination_attribution": _supported_attribution(case, curve, prediction, h),
        "distance_to_termination_ab": {
            "arm_A": _distance_bins(case.reference_eval_points, historical_points, historical_normals, trace.interface_points, h),
            "arm_B": _distance_bins(case.reference_eval_points, prediction.points, prediction.normals, trace.interface_points, h),
        },
        "target_connectivity_caution": {
            "mesh_connectivity_fragmentation_components": int(len(trace.component_sizes)),
            "interface_seeded_component_count": int(sum(item["interface_connected"] for item in trace.component_sizes)),
            "physical_sheet_inference": "not made from disconnected TSDF components",
        },
        "raw_corrected_distances_a": distances_a,
        "raw_explicit_distances_b": distances_b,
        "raw_nearest_a": nearest_a,
        "raw_nearest_b": nearest_b,
    }


def _write_primary_figure(
    output_path: Path,
    case: FrozenCase,
    vertex_roi: Any,
    trace: Any,
    plane_patch: np.ndarray,
    fitted_surface_points: np.ndarray,
    historical_edge: np.ndarray,
    gamma_points: np.ndarray,
    historical_points: np.ndarray,
    explicit_points: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(22, 16), facecolor="white")
    axes = [figure.add_subplot(3, 3, index + 1, projection="3d") for index in range(8)]
    all_points = np.concatenate([
        vertex_roi.full_points,
        trace.interface_points,
        plane_patch,
        fitted_surface_points,
        historical_edge,
        gamma_points,
        historical_points,
        explicit_points,
    ], axis=0)
    panels = [
        ("A  Observed WL127 surface", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.65, 0.55)]),
        ("B  Fixed physical holdout plane", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.45, 0.22), (trace.interface_points, (0.98, 0.78, 0.08), 4.0, 0.92)]),
        ("C  Frozen fitted NURBS surface", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.35, 0.16), (fitted_surface_points, (0.38, 0.50, 0.82), 0.8, 0.36)]),
        ("D  Historical rectangular u=1 edge", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.40, 0.18), (historical_edge, (0.12, 0.35, 0.85), 7.0, 0.95)]),
        ("E  Explicit GEOMETRIC_TERMINATION_CURVE Γ", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.36, 0.16), (trace.interface_points, (0.98, 0.78, 0.08), 2.5, 0.60), (gamma_points, (0.10, 0.72, 0.28), 8.0, 0.98)]),
        ("F  Arm A: historical u=1 first order", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.30, 0.14), (historical_points, (0.05, 0.70, 0.88), 1.2, 0.30)]),
        ("G  Arm B: explicit-termination first order", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.30, 0.14), (gamma_points, (0.10, 0.72, 0.28), 4.0, 0.80), (explicit_points, (0.00, 0.62, 0.42), 1.6, 0.90)]),
        ("H  Arm B versus withheld reference", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.25, 0.10), (explicit_points, (0.00, 0.62, 0.42), 1.6, 0.82), (vertex_roi.withheld_points, (0.82, 0.14, 0.12), 0.65, 0.70)]),
    ]
    for axis, (title, items) in zip(axes, panels):
        _set_equal_3d_limits(axis, all_points)
        axis.set_title(title, fontsize=12, pad=9)
        for points, colour, size, alpha in items:
            if len(points):
                _scatter3d(axis, points, colour, size=size, alpha=alpha)
        if title.startswith("B"):
            axis.add_collection3d(Poly3DCollection([plane_patch], facecolors=(0.98, 0.78, 0.08, 0.12), edgecolors=(0.98, 0.78, 0.08, 0.65), linewidths=1.0))
    figure.suptitle("Explicit geometric termination mapping — first-order attribution only", fontsize=20)
    figure.text(0.02, 0.015, "gray observed | blue historical u=1 edge | yellow fixed plane/interface | green Γ | cyan Arm A | teal Arm B | red withheld", fontsize=11)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_meeting_figure(output_path: Path, case: FrozenCase, vertex_roi: Any, prediction: ExplicitPrediction, h: float, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(20, 10), facecolor="white")
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    all_points = np.concatenate([vertex_roi.full_points, prediction.points], axis=0)
    titles = ["FULL REFERENCE", "VISIBLE-ONLY", "EXPLICIT TERMINATION + FIRST-ORDER", "PREDICTION vs WITHHELD REFERENCE"]
    for axis, title in zip(axes, titles):
        _set_equal_3d_limits(axis, all_points)
        axis.set_title(title, fontsize=14)
    _scatter3d(axes[0], vertex_roi.full_points, (0.58, 0.60, 0.63), size=0.75, alpha=0.58)
    _scatter3d(axes[1], vertex_roi.observed_points, (0.58, 0.60, 0.63), size=0.85, alpha=0.76)
    _scatter3d(axes[2], vertex_roi.observed_points, (0.58, 0.60, 0.63), size=0.55, alpha=0.23)
    _scatter3d(axes[2], prediction.points, (0.00, 0.62, 0.42), size=1.5, alpha=0.90)
    _scatter3d(axes[3], prediction.points, (0.00, 0.62, 0.42), size=1.6, alpha=0.85)
    _scatter3d(axes[3], vertex_roi.withheld_points, (0.82, 0.14, 0.12), size=0.8, alpha=0.70)
    figure.suptitle("Explicit geometric termination: controlled meeting holdout", fontsize=19)
    figure.text(0.02, 0.02, f"median error / h = {metrics['median_over_h']:.3f}    coverage ≤ h = {metrics['coverage_le_h']:.1%}    coverage ≤ 2h = {metrics['coverage_le_2h']:.1%}", fontsize=13)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def _write_readme(output_root: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Explicit geometric termination continuation",
        "",
        "Worklog 128, 129, and 130 are preserved. This analysis replaces only",
        "the continuation start boundary: frozen NURBS u=1 is compared with a",
        "fixed-plane/NURBS intersection curve. The continuation remains first-order",
        "and uses the physical ROI +u direction. No second-order or occluded",
        "surface implementation is included.",
        "",
        f"## Verdict: `{report['meeting_verdict']}`",
        "",
        "The full Worklog 129 withheld population remains the canonical A/B",
        "evaluation population. The observed-supported termination population is",
        "reported separately for attribution only.",
        "",
        "## Outputs",
        "",
        "- `explicit_geometric_termination_report.json`",
        "- `curved_rim_explicit_termination_figure.png`",
        "- `distance_to_termination_ab.png`",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ab_distance_figure(output_path: Path, cases: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(cases), figsize=(8 * len(cases), 6), facecolor="white", squeeze=False)
    labels = ("0–1h", "1–2h", "2–4h", "4–8h", "8–16h", ">16h")
    x = np.arange(len(labels))
    for axis, case_report in zip(axes[0], cases):
        for arm_key, colour, marker in (("arm_A", "#1677b8", "o"), ("arm_B", "#00a878", "s")):
            rows = case_report["distance_to_termination_ab"][arm_key]["fixed_bins"]
            median = [row["median_error_over_h"] for row in rows]
            coverage = [row["coverage_le_h"] for row in rows]
            axis.plot(x, median, color=colour, marker=marker, label=f"{arm_key} median / h")
            axis.plot(x, coverage, color=colour, marker=marker, linestyle=":", alpha=0.75, label=f"{arm_key} coverage ≤ h")
        axis.set_xticks(x, labels, rotation=25)
        axis.set_ylabel("error / h (solid), coverage ≤ h (dotted)")
        axis.set_title(case_report["roi"]["semantic_label"])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Arm A versus explicit geometric termination by fixed distance bin", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_analysis(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    old_root = Path(arguments.worklog128_out)
    wl129_root = Path(arguments.worklog129_out)
    wl130_report_path = Path(arguments.worklog130_report)
    field = np.load(arguments.field_cache, allow_pickle=True)
    h = float(field["h"])
    mu = float(field["mu"])
    mesh_bundle = np.load(arguments.mesh_cache, allow_pickle=True)
    vertices = np.asarray(mesh_bundle["vertices"], dtype=np.float64)
    report_cases: list[dict[str, Any]] = []
    case_runtime: list[tuple[FrozenCase, Any, Any, Any, ExplicitPrediction, np.ndarray, np.ndarray, np.ndarray]] = []
    for config in (PRIMARY_ROI, SECONDARY_ROI):
        case = _load_frozen_case(old_root / config.name, config)
        observed_surface = _surface_from_grid(case.control_grid, device=str(device))
        plane = termination_plane(config)
        vertex_roi = _roi_vertex_contract(vertices, config)
        trace = trace_mesh_interface(Path(arguments.mesh_cache), vertices, vertex_roi)
        curve = _termination_curve(observed_surface, plane, vertex_roi, case.observed_points, h)
        prediction = build_explicit_prediction(curve, config, h)
        historical_npz = np.load(wl129_root / f"{config.name}_corrected_arm.npz", allow_pickle=True)
        historical_points = np.asarray(historical_npz["corrected_points"], dtype=np.float64)
        historical_normals = np.asarray(historical_npz["corrected_normals"], dtype=np.float64)
        historical_source = "frozen Worklog 129 corrected arm NPZ"
        wl129_report = json.loads(Path(arguments.worklog129_report).read_text(encoding="utf-8"))
        corrected_metrics_frozen = next(item["corrected_arm"] for item in wl129_report["arms"] if item["roi"]["name"] == config.name)
        parameterization = _report_parameterization(_load_wl130_parameterization(wl130_report_path, config.name))
        case_report = _case_report(
            case, parameterization, plane, curve, prediction, vertex_roi, trace,
            historical_points, historical_normals, corrected_metrics_frozen, h, historical_source,
        )
        edge_points, edge_tangent, edge_normals = _evaluate_boundary(observed_surface, str(device), TERMINATION_SAMPLES)
        case_report["arm_A_historical_u1_first_order"]["interface"] = _interface_metrics(edge_points, trace, h)
        case_report["arm_B_explicit_termination_first_order"]["interface"] = _interface_metrics(curve.gamma_points[curve.valid_mask], trace, h)
        case_report["arm_A_historical_u1_first_order"]["termination_object"] = "rectangular NURBS parameter-domain edge u=1; not called geometric termination"
        case_report["arm_B_explicit_termination_first_order"]["termination_object"] = "GEOMETRIC_TERMINATION_CURVE Gamma(s) = S(u_gamma(s), v_gamma(s))"
        case_report["arm_A_historical_u1_first_order"]["interface_support_fraction"] = case_report["arm_A_historical_u1_first_order"]["interface"]["interface_support_fraction"]
        case_report["arm_B_explicit_termination_first_order"]["interface_support_fraction"] = case_report["arm_B_explicit_termination_first_order"]["interface"].get("interface_support_fraction", {"status": "no valid termination roots"})
        report_cases.append(case_report)
        case_runtime.append((case, vertex_roi, trace, observed_surface, prediction, historical_points, historical_normals, edge_points))
        if config.name == PRIMARY_ROI.name:
            fitted_surface_points = _surface_grid(observed_surface)
            plane_patch = _fixed_plane_patch(config, plane)
            _write_primary_figure(
                output_root / "curved_rim_explicit_termination_figure.png",
                case, vertex_roi, trace, plane_patch, fitted_surface_points, edge_points,
                curve.gamma_points[curve.valid_mask], historical_points, prediction.points,
            )

    primary_report = next(item for item in report_cases if item["roi"]["name"] == PRIMARY_ROI.name)
    a_metrics = primary_report["arm_A_historical_u1_first_order"]["full_population_metrics_recomputed"]
    b_metrics = primary_report["arm_B_explicit_termination_first_order"]["full_population_metrics"]
    median_improvement = 1.0 - b_metrics["median_over_h"] / max(a_metrics["median_over_h"], 1e-12)
    coverage_gain = b_metrics["coverage_le_h"] - a_metrics["coverage_le_h"]
    positive_quantitative = bool(median_improvement >= MATERIAL_MEDIAN_IMPROVEMENT and coverage_gain >= MATERIAL_COVERAGE_GAIN)
    positive_visual_proxy = bool(b_metrics["median_over_h"] <= 2.0 and b_metrics["p95_over_h"] <= 12.0)
    meeting_figure_created = bool(positive_quantitative and positive_visual_proxy)
    if meeting_figure_created:
        case, vertex_roi, trace, _surface, prediction, _a_points, _a_normals, _edge = next(item for item in case_runtime if item[0].config.name == PRIMARY_ROI.name)
        _write_meeting_figure(
            output_root / "curved_rim_meeting_figure.png", case, vertex_roi, prediction, h, b_metrics,
        )
    _write_ab_distance_figure(output_root / "distance_to_termination_ab.png", report_cases)
    if not positive_quantitative:
        verdict = "C_EXPLICIT_TERMINATION_DOES_NOT_MATERIALLY_HELP"
    elif not positive_visual_proxy:
        verdict = "B_EXPLICIT_TERMINATION_HELPS_BUT_FIRST_ORDER_INSUFFICIENT"
    else:
        verdict = "A_WRONG_TERMINATION_EDGE_DOMINANT_FAILURE"
    report = {
        "batch": "Worklog 131 explicit geometric termination mapping for first-order parametric continuation",
        "status": "NON_CANONICAL_ATTRIBUTION_ONLY",
        "intent_alignment": {
            "worklog_128_preserved": True,
            "worklog_129_preserved": True,
            "worklog_130_preserved": True,
            "same_first_order_model": True,
            "only_continuation_start_contract_changed": True,
            "second_order_or_curvature_added": False,
            "true_occluded_prototype": "NOT_EXECUTED",
        },
        "implementation_fidelity": {
            "manual_choices": "the two historical ROI definitions and holdout_u_cut=0.58",
            "heuristics": "deterministic largest-u root selection, bisection sampling, PCA normals, Euclidean interface bins",
            "full_reference_roles": ["fixed ROI mesh vertices/faces for interface support and display", "evaluation target", "distance bins"],
            "withheld_xyz_in_root_direction_or_L": False,
            "unacceptable_final_method_shortcuts": ["manual plane movement", "target-selected root/branch", "target-fitted horizon", "mesh-disconnection-as-physical-sheet inference"],
            "canonical_code_modified": False,
        },
        "inputs": {
            "worklog_128_commit": WORKLOG_128_COMMIT,
            "worklog_129_commit": WORKLOG_129_COMMIT,
            "worklog_130_commit": WORKLOG_130_COMMIT,
            "worklog_128_output": str(old_root),
            "worklog_129_output": str(wl129_root),
            "worklog_130_report": str(wl130_report_path),
            "reference_mesh": str(arguments.mesh_cache),
            "h": h,
            "mu": mu,
            "holdout_u_cut": HOLDOUT_CUT,
            "frozen_fit_and_evaluation_population": True,
        },
        "meeting_figure": {
            "created": meeting_figure_created,
            "quantitative_gate": {
                "median_improvement_fraction": median_improvement,
                "required_at_least": MATERIAL_MEDIAN_IMPROVEMENT,
                "coverage_gain": coverage_gain,
                "required_at_least_gain": MATERIAL_COVERAGE_GAIN,
            },
            "visual_proxy": {
                "primary_median_over_h": b_metrics["median_over_h"],
                "primary_p95_over_h": b_metrics["p95_over_h"],
                "fixed_proxy": "median <= 2h and p95 <= 12h",
            },
        },
        "cases": report_cases,
        "true_occluded_prototype": {
            "status": "NOT_EXECUTED",
            "reason": "bounded attribution stopped after controlled holdout decision; no canonical occluded-surface implementation",
        },
        "meeting_verdict": verdict,
    }
    # Arrays are intentionally omitted from the JSON report; the figures and
    # the deterministic frozen NPZ inputs remain the geometry artefacts.
    for item in report["cases"]:
        for key in ("raw_corrected_distances_a", "raw_explicit_distances_b", "raw_nearest_a", "raw_nearest_b"):
            item.pop(key, None)
    (output_root / "explicit_geometric_termination_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _write_readme(output_root, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklog128-out", type=Path, default=REPO_ROOT / "output/demo_parametric_surface_continuation")
    parser.add_argument("--worklog129-out", type=Path, default=REPO_ROOT / "output/demo_corrected_first_order_parametric_continuation")
    parser.add_argument("--worklog129-report", type=Path, default=REPO_ROOT / "output/demo_corrected_first_order_parametric_continuation/corrected_first_order_parametric_continuation_report.json")
    parser.add_argument("--worklog130-report", type=Path, default=REPO_ROOT / "output/demo_parametric_continuation_attribution/parametric_continuation_attribution_report.json")
    parser.add_argument("--mesh-cache", type=Path, default=REPO_ROOT / "output/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz")
    parser.add_argument("--field-cache", type=Path, default=REPO_ROOT / "output/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/demo_explicit_geometric_termination_continuation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_analysis(build_arg_parser().parse_args(argv))
    print(json.dumps({"verdict": report["meeting_verdict"], "cases": len(report["cases"]), "meeting_figure": report["meeting_figure"]["created"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
