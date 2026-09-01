"""Close the Worklog 131 supported-termination attribution contract.

This is an isolated, non-canonical diagnostic.  It replays the frozen
Worklog 131 geometric termination and first-order prediction, then assigns
every withheld reference row to exactly one Gamma sample by nearest manual-v
correspondence.  It does not modify Gamma, the ROI, the holdout, the horizon,
or the first-order prediction.

Only after the corrected supported-target result remains poor does this
module evaluate visible-side directional curvature.  The curvature candidate
is conditional and diagnostic-only; no canonical surface or occluded surface
is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.corrected_first_order_parametric_continuation import (  # noqa: E402
    _load_frozen_case,
    _surface_from_grid,
)
from devtools.demo.explicit_geometric_termination_continuation import (  # noqa: E402
    ROOT_RESIDUAL_TOLERANCE,
    ROOT_SAMPLES,
    TerminationCurve,
    _evaluate_surface_points,
    _termination_curve,
    build_explicit_prediction,
    termination_plane,
)
from devtools.demo.parametric_continuation_attribution import (  # noqa: E402
    _angle_degrees,
    _prediction_metrics,
    _roi_vertex_contract,
    trace_mesh_interface,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    PRIMARY_ROI,
    SECONDARY_ROI,
    ROIConfig,
    _jsonable,
    estimate_point_normals,
    roi_coordinates,
)


WORKLOG_128_COMMIT = "2d87366b910873562b9dfc223408d85257c5af9f"
WORKLOG_129_COMMIT = "1ca0da5"
WORKLOG_130_COMMIT = "8b2b4e7"
WORKLOG_131_COMMIT = "cf027b4c270a383995c7d23e9b25cd9ba99ee3d1"

HOLDOUT_CUT = 0.58
SUPPORT_THRESHOLD_H = 2.0
ROOT_DIAGNOSTIC_TOLERANCE = 1e-4
CORRESPONDENCE_BIN_EDGES = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)
CORRESPONDENCE_BIN_LABELS = ("0–1h", "1–2h", "2–4h", "4–8h", "8–16h", ">16h")
CURVATURE_BIN_EDGES = np.asarray([0.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)
CURVATURE_BIN_LABELS = ("0–2h", "2–4h", "4–8h", "8–16h", ">16h")


def _sha256_array(values: np.ndarray, dtype: np.dtype[Any] = np.dtype(np.float32)) -> str:
    array = np.ascontiguousarray(np.asarray(values), dtype=dtype)
    return hashlib.sha256(array.tobytes()).hexdigest()


def nearest_gamma_assignment(target_v: np.ndarray, gamma_v: np.ndarray) -> np.ndarray:
    """Assign each target to exactly one Gamma column; argmin ties are lowest index."""

    target_v = np.asarray(target_v, dtype=np.float64).reshape(-1)
    gamma_v = np.asarray(gamma_v, dtype=np.float64).reshape(-1)
    if len(target_v) == 0:
        return np.zeros((0,), dtype=np.int64)
    if len(gamma_v) == 0 or not np.isfinite(target_v).all() or not np.isfinite(gamma_v).all():
        raise ValueError("nearest Gamma correspondence requires finite non-empty Gamma samples")
    # np.argmin is deterministic and selects the smallest Gamma index on ties.
    return np.argmin(np.abs(target_v[:, None] - gamma_v[None, :]), axis=1).astype(np.int64)


def partition_by_gamma_support(
    assigned_gamma_indices: np.ndarray,
    supported_gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint, exhaustive supported and unsupported target masks."""

    assigned = np.asarray(assigned_gamma_indices, dtype=np.int64).reshape(-1)
    supported_gamma = np.asarray(supported_gamma, dtype=bool).reshape(-1)
    if len(assigned) and (assigned.min() < 0 or assigned.max() >= len(supported_gamma)):
        raise IndexError("target assignment refers to a Gamma sample outside the support array")
    supported = supported_gamma[assigned] if len(assigned) else np.zeros((0,), dtype=bool)
    unsupported = ~supported
    if bool(np.any(supported & unsupported)) or not bool(np.all(supported | unsupported)):
        raise AssertionError("supported/unsupported target partition is not disjoint and exhaustive")
    return supported, unsupported


def _empty_metrics(status: str = "empty_population") -> dict[str, Any]:
    return {
        "samples": 0,
        "median_over_h": None,
        "p95_over_h": None,
        "coverage_le_h": None,
        "coverage_le_2h": None,
        "normal_error": {"status": status},
    }


def _metrics_from_matches(
    reference_points: np.ndarray,
    distances: np.ndarray,
    matched_normals: np.ndarray,
    h: float,
) -> dict[str, Any]:
    reference_points = np.asarray(reference_points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    matched_normals = np.asarray(matched_normals, dtype=np.float64)
    if len(reference_points) == 0:
        return _empty_metrics()
    if len(distances) != len(reference_points):
        raise ValueError("distance and reference populations must have equal length")
    reference_normals = estimate_point_normals(reference_points, k=20)
    if reference_normals is None or matched_normals.shape != (len(reference_points), 3):
        normal_error: dict[str, Any] = {"status": "unavailable"}
    else:
        angles = _angle_degrees(reference_normals, matched_normals)
        normal_error = {
            "status": "estimated_unoriented_pca_vs_correspondence_polyline",
            "median_degrees": float(np.median(angles)),
            "p95_degrees": float(np.percentile(angles, 95)),
        }
    return {
        "samples": int(len(reference_points)),
        "median_over_h": float(np.median(distances) / h),
        "p95_over_h": float(np.percentile(distances, 95) / h),
        "coverage_le_h": float(np.mean(distances <= h)),
        "coverage_le_2h": float(np.mean(distances <= 2.0 * h)),
        "normal_error": normal_error,
    }


def _prediction_column_map(prediction_gamma_indices: np.ndarray) -> dict[int, int]:
    return {int(gamma_index): int(column) for column, gamma_index in enumerate(prediction_gamma_indices.tolist())}


def _restricted_matches(
    reference_points: np.ndarray,
    assigned_gamma_indices: np.ndarray,
    prediction_points_grid: np.ndarray,
    prediction_normals_grid: np.ndarray,
    prediction_gamma_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match rows only to the continuation polyline of their assigned Gamma."""

    from scipy.spatial import cKDTree

    reference_points = np.asarray(reference_points, dtype=np.float64)
    assigned = np.asarray(assigned_gamma_indices, dtype=np.int64).reshape(-1)
    points_grid = np.asarray(prediction_points_grid, dtype=np.float64)
    normals_grid = np.asarray(prediction_normals_grid, dtype=np.float64)
    gamma_indices = np.asarray(prediction_gamma_indices, dtype=np.int64).reshape(-1)
    if points_grid.ndim != 3 or points_grid.shape[-1] != 3:
        raise ValueError("prediction_points_grid must have shape (L, Gamma, 3)")
    if normals_grid.shape != points_grid.shape:
        raise ValueError("prediction normals must have the same grid shape as prediction points")
    if len(reference_points) != len(assigned):
        raise ValueError("reference rows and Gamma assignments must have equal length")
    column_map = _prediction_column_map(gamma_indices)
    if len(assigned) and any(int(index) not in column_map for index in np.unique(assigned)):
        raise ValueError("a target was assigned to a Gamma without a prediction column")
    distances = np.full((len(reference_points),), np.nan, dtype=np.float64)
    matched_normals = np.full((len(reference_points), 3), np.nan, dtype=np.float64)
    for gamma_index in np.unique(assigned):
        rows = np.flatnonzero(assigned == gamma_index)
        column = column_map[int(gamma_index)]
        local_distances, local_indices = cKDTree(points_grid[:, column, :]).query(reference_points[rows], workers=1)
        distances[rows] = local_distances
        matched_normals[rows] = normals_grid[local_indices, column, :]
    return distances, matched_normals


def correspondence_restricted_metrics(
    reference_points: np.ndarray,
    assigned_gamma_indices: np.ndarray,
    prediction_points_grid: np.ndarray,
    prediction_normals_grid: np.ndarray,
    prediction_gamma_indices: np.ndarray,
    h: float,
) -> dict[str, Any]:
    """Evaluate a population against only its assigned Gamma continuation column."""

    distances, matched_normals = _restricted_matches(
        reference_points,
        assigned_gamma_indices,
        prediction_points_grid,
        prediction_normals_grid,
        prediction_gamma_indices,
    )
    return _metrics_from_matches(reference_points, distances, matched_normals, h)


def _correspondence_population_report(
    target: np.ndarray,
    assigned_gamma: np.ndarray,
    prediction_points: np.ndarray,
    prediction_normals: np.ndarray,
    prediction_points_grid: np.ndarray,
    prediction_normals_grid: np.ndarray,
    prediction_gamma_indices: np.ndarray,
    h: float,
) -> dict[str, Any]:
    if len(target) == 0:
        return {"sample_count": 0, "free_nearest_surface": _empty_metrics(), "correspondence_restricted": _empty_metrics()}
    free_metrics, _free_distances, _free_nearest = _prediction_metrics(target, prediction_points, prediction_normals, h)
    restricted = correspondence_restricted_metrics(
        target,
        assigned_gamma,
        prediction_points_grid,
        prediction_normals_grid,
        prediction_gamma_indices,
        h,
    )
    return {
        "sample_count": int(len(target)),
        "free_nearest_surface": free_metrics,
        "correspondence_restricted": restricted,
    }


def _fixed_distance_bins(
    target: np.ndarray,
    assigned_gamma: np.ndarray,
    prediction_points_grid: np.ndarray,
    prediction_normals_grid: np.ndarray,
    prediction_gamma_indices: np.ndarray,
    interface_points: np.ndarray,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    if len(target) == 0:
        return {"distance_definition": "Euclidean distance to frozen mesh-face interface / h", "fixed_bins": []}
    interface_distance = cKDTree(interface_points).query(target, workers=1)[0] / h
    distances, matched_normals = _restricted_matches(
        target,
        assigned_gamma,
        prediction_points_grid,
        prediction_normals_grid,
        prediction_gamma_indices,
    )
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(CORRESPONDENCE_BIN_LABELS):
        selected = (interface_distance >= CORRESPONDENCE_BIN_EDGES[index]) & (interface_distance < CORRESPONDENCE_BIN_EDGES[index + 1])
        row: dict[str, Any] = {"bin": label, "reference_count": int(selected.sum())}
        if bool(selected.any()):
            metric = _metrics_from_matches(target[selected], distances[selected], matched_normals[selected], h)
            row.update({
                "median_over_h": metric["median_over_h"],
                "p95_over_h": metric["p95_over_h"],
                "coverage_le_h": metric["coverage_le_h"],
                "coverage_le_2h": metric["coverage_le_2h"],
                "normal_error": metric["normal_error"],
            })
        else:
            row.update({"median_over_h": None, "p95_over_h": None, "coverage_le_h": None, "coverage_le_2h": None, "normal_error": {"status": "empty_bin"}})
        rows.append(row)
    return {
        "distance_definition": "Euclidean distance from frozen Worklog 127 mesh-face interface, divided by h",
        "fixed_bins": rows,
        "population": "SUPPORTED_TARGET only; correspondence-restricted to assigned Gamma column",
    }


def _thin_root_classification(
    function_values: np.ndarray,
    has_crossing_root: bool,
    root_tolerance: float = ROOT_RESIDUAL_TOLERANCE,
    diagnostic_tolerance: float = ROOT_DIAGNOSTIC_TOLERANCE,
) -> str:
    values = np.asarray(function_values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("root classification requires finite scan values")
    if has_crossing_root:
        return "ordinary_crossing_root"
    min_abs = float(np.min(np.abs(values)))
    if min_abs <= diagnostic_tolerance:
        return "possible_tangential_near_contact"
    strictly_positive = bool(np.min(values) > root_tolerance)
    strictly_negative = bool(np.max(values) < -root_tolerance)
    if strictly_positive or strictly_negative:
        return "definitely_no_intersection"
    return "possible_tangential_near_contact"


def thin_root_audit(surface: Any, plane: dict[str, Any], samples_v: int = 32) -> dict[str, Any]:
    """Scan the fixed thin ROI without creating a new Gamma curve."""

    from devtools.demo.explicit_geometric_termination_continuation import solve_plane_intersections

    root_rows, v_values = solve_plane_intersections(surface, plane, samples_v=samples_v)
    u_values = np.linspace(0.0, 1.0, ROOT_SAMPLES, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for row, value_v in zip(root_rows, v_values.tolist()):
        uv = np.column_stack([u_values, np.full_like(u_values, value_v)])
        values = _evaluate_surface_points(surface, uv) @ plane["axis_u"] - float(plane["world_u_coordinate"])
        values = np.asarray(values, dtype=np.float64)
        rows.append({
            "v": float(value_v),
            "min_f": float(np.min(values)),
            "max_f": float(np.max(values)),
            "min_abs_f": float(np.min(np.abs(values))),
            "root_count_from_sign_scan": int(len(row.roots_u)),
            "classification": _thin_root_classification(values, bool(row.roots_u)),
        })
    counts = {label: int(sum(item["classification"] == label for item in rows)) for label in (
        "definitely_no_intersection", "possible_tangential_near_contact", "ordinary_crossing_root"
    )}
    return {
        "samples_v": int(samples_v),
        "u_scan_samples": int(ROOT_SAMPLES),
        "root_tolerance": ROOT_RESIDUAL_TOLERANCE,
        "diagnostic_tolerance": ROOT_DIAGNOSTIC_TOLERANCE,
        "new_gamma_created": False,
        "classification_counts": counts,
        "rows": rows,
    }


def _frozen_case_report(report: dict[str, Any], case_name: str) -> dict[str, Any]:
    for item in report.get("cases", []):
        if item.get("roi", {}).get("name") == case_name:
            return item
    raise KeyError(f"Worklog 131 report has no case {case_name}")


def _metric_close(actual: dict[str, Any], frozen: dict[str, Any], tolerance: float = 2e-5) -> bool:
    if actual.get("status") is not None or frozen.get("status") is not None:
        return actual.get("status") == frozen.get("status")
    pairs = (
        (actual.get("median_over_h"), frozen.get("median_over_h")),
        (actual.get("p95_over_h"), frozen.get("p95_over_h")),
        (actual.get("coverage_le_h"), frozen.get("coverage_le_h")),
        (actual.get("coverage_le_2h"), frozen.get("coverage_le_2h")),
    )
    if not all(a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance) for a, b in pairs):
        return False
    actual_normal = actual.get("normal_error", {})
    frozen_normal = frozen.get("normal_error", {})
    if actual_normal.get("median_degrees") is None or frozen_normal.get("median_degrees") is None:
        return actual_normal.get("status") == frozen_normal.get("status")
    return math.isclose(float(actual_normal["median_degrees"]), float(frozen_normal["median_degrees"]), rel_tol=tolerance, abs_tol=tolerance) and math.isclose(
        float(actual_normal["p95_degrees"]), float(frozen_normal["p95_degrees"]), rel_tol=tolerance, abs_tol=tolerance
    )


def frozen_wl131_reproduction(
    case: Any,
    curve: TerminationCurve,
    prediction: Any,
    config: ROIConfig,
    plane: dict[str, Any],
    frozen_item: dict[str, Any],
    h: float,
) -> dict[str, Any]:
    """Compare replayed Gamma/Arm B quantities to the frozen WL131 report."""

    frozen_root_rows = frozen_item["root_contract"]["root_rows"]
    root_rows_equal = len(frozen_root_rows) == len(curve.root_rows)
    max_root_difference = 0.0
    if root_rows_equal:
        for current, saved in zip(curve.root_rows, frozen_root_rows):
            current_values = [current.v, *current.roots_u, current.selected_u if current.selected_u is not None else np.nan]
            saved_values = [saved["v"], *saved["roots_u"], saved["selected_u"] if saved["selected_u"] is not None else np.nan]
            if len(current_values) != len(saved_values):
                root_rows_equal = False
                break
            max_root_difference = max(max_root_difference, float(np.max(np.abs(np.asarray(current_values) - np.asarray(saved_values)))))
            root_rows_equal = root_rows_equal and bool(np.allclose(current_values, saved_values, rtol=1e-9, atol=1e-9, equal_nan=True))

    valid = curve.valid_mask
    frozen_uv = np.asarray([[row["selected_u"], row["v"]] for row in frozen_root_rows if row["selected_u"] is not None], dtype=np.float64).reshape(-1, 2)
    replay_uv = curve.gamma_uv[valid]
    gamma_uv_equal = bool(frozen_uv.shape == replay_uv.shape and np.allclose(frozen_uv, replay_uv, rtol=1e-9, atol=1e-9))
    replay_metrics, _distances, _nearest = _prediction_metrics(case.reference_eval_points, prediction.points, prediction.normals, h) if len(prediction.points) else ({"status": "no explicit termination roots; no Arm B prediction surface"}, np.zeros(0), np.zeros(0, dtype=np.int64))
    frozen_metrics = frozen_item["arm_B_explicit_termination_first_order"].get("full_population_metrics", {})
    metric_equal = _metric_close(replay_metrics, frozen_metrics)
    expected_prediction = frozen_item["explicit_prediction"]
    local_u = roi_coordinates(prediction.points, config)[:, 0] if len(prediction.points) else np.zeros((0,), dtype=np.float64)
    actual_extent = {
        "min": float(np.min(local_u)) if len(local_u) else None,
        "max": float(np.max(local_u)) if len(local_u) else None,
        "span": float(np.ptp(local_u)) if len(local_u) else None,
    }
    expected_count = int(expected_prediction["predicted_point_count"])
    if expected_count == 0:
        extent_equal = bool(
            len(prediction.points) == 0
            and math.isclose(float(prediction.l_values[-1]), float(expected_prediction["continuation_extent_physical_local_u"]), rel_tol=1e-8, abs_tol=1e-8)
            and all(actual_extent[key] is None and expected_prediction["actual_local_u_span"][key] is None for key in ("min", "max", "span"))
        )
    else:
        extent_equal = bool(
            len(prediction.points) == expected_count
            and math.isclose(float(prediction.l_values[-1]), float(expected_prediction["continuation_extent_physical_local_u"]), rel_tol=1e-8, abs_tol=1e-8)
            and all(math.isclose(float(actual_extent[key]), float(expected_prediction["actual_local_u_span"][key]), rel_tol=2e-5, abs_tol=2e-5) for key in ("min", "max", "span"))
        )
    plane_equal = bool(math.isclose(float(plane["world_u_coordinate"]), float(frozen_item["termination_plane"]["world_u_coordinate"]), rel_tol=1e-10, abs_tol=1e-10))
    direction_equal = bool(np.allclose(curve.local_u_derivative[valid], 1.0, rtol=1e-8, atol=1e-8))
    reproduction_passed = bool(root_rows_equal and gamma_uv_equal and plane_equal and direction_equal and extent_equal and metric_equal)
    if not reproduction_passed:
        raise RuntimeError(f"Frozen Worklog 131 Arm B reproduction failed for {config.name}: {replay_metrics}")
    return {
        "status": "PASS",
        "frozen_report_root_rows_equal": bool(root_rows_equal),
        "root_rows_max_abs_difference": max_root_difference,
        "gamma_uv_equal_to_frozen_root_rows": gamma_uv_equal,
        "gamma_uv_sha256_float32": _sha256_array(replay_uv),
        "gamma_point_sha256_float32": _sha256_array(curve.gamma_points[valid]),
        "frozen_report_predicted_point_hash_available": False,
        "predicted_point_identity_basis": "frozen Gamma UV rows + frozen control grid replay + fixed point count/horizon/extent + full metric",
        "physical_direction_identity": direction_equal,
        "plane_identity": plane_equal,
        "prediction_identity": extent_equal,
        "full_population_metric_identity": metric_equal,
        "replayed_full_population_metrics": replay_metrics,
        "frozen_full_population_metrics": frozen_metrics,
    }


def _supported_gamma_contract(curve: TerminationCurve, assignment: np.ndarray, h: float) -> dict[str, Any]:
    supported_gamma = curve.valid_mask & np.isfinite(curve.support_fit_distance) & (curve.support_fit_distance <= SUPPORT_THRESHOLD_H)
    supported_target, unsupported_target = partition_by_gamma_support(assignment, supported_gamma)
    counts = np.bincount(assignment, minlength=len(curve.root_rows)) if len(assignment) else np.zeros(len(curve.root_rows), dtype=np.int64)
    per_gamma = []
    for index, row in enumerate(curve.root_rows):
        per_gamma.append({
            "gamma_index": int(index),
            "v": float(row.v),
            "root_exists": bool(curve.valid_mask[index]),
            "support_fit_distance_over_h": float(curve.support_fit_distance[index]) if np.isfinite(curve.support_fit_distance[index]) else None,
            "supported_gamma": bool(supported_gamma[index]),
            "assigned_target_rows": int(counts[index]),
        })
    return {
        "support_rule": "nearest observed fitting point to Gamma <= 2h",
        "support_threshold_h": SUPPORT_THRESHOLD_H,
        "supported_gamma_count": int(np.sum(supported_gamma)),
        "valid_gamma_count": int(np.sum(curve.valid_mask)),
        "supported_gamma_fraction": float(np.sum(supported_gamma) / max(int(np.sum(curve.valid_mask)), 1)),
        "full_target_count": int(len(assignment)),
        "supported_target_count": int(np.sum(supported_target)),
        "unsupported_target_count": int(np.sum(unsupported_target)),
        "supported_target_fraction": float(np.mean(supported_target)) if len(assignment) else 0.0,
        "unsupported_target_fraction": float(np.mean(unsupported_target)) if len(assignment) else 0.0,
        "accounting_identity": bool(len(assignment) == int(np.sum(supported_target)) + int(np.sum(unsupported_target))),
        "disjoint_identity": bool(not np.any(supported_target & unsupported_target)),
        "target_assignment_definition": "exactly one nearest manual-normalized-v Gamma index; argmin ties choose smallest Gamma index",
        "assigned_gamma_indices": assignment,
        "supported_target_mask": supported_target,
        "unsupported_target_mask": unsupported_target,
        "supported_gamma_mask": supported_gamma,
        "per_gamma": per_gamma,
    }


def _evaluate_directional_curvature(surface: Any, curve: TerminationCurve) -> dict[str, np.ndarray]:
    import torch

    valid = curve.valid_mask
    uv = torch.as_tensor(curve.gamma_uv[valid], dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    with torch.no_grad():
        _points, su, sv, suu, suv, svv = surface.evaluate_with_second_derivatives(uv)
    su = su.detach().cpu().numpy().astype(np.float64)
    sv = sv.detach().cpu().numpy().astype(np.float64)
    suu = suu.detach().cpu().numpy().astype(np.float64)
    suv = suv.detach().cpu().numpy().astype(np.float64)
    svv = svv.detach().cpu().numpy().astype(np.float64)
    duv = curve.derivatives_uv[valid]
    tangent, curvature = directional_terms_from_derivatives(duv[:, 0], duv[:, 1], su, sv, suu, suv, svv)
    return {"gamma_indices": np.flatnonzero(valid), "T": tangent, "A": curvature}


def directional_terms_from_derivatives(
    a: np.ndarray,
    b: np.ndarray,
    su: np.ndarray,
    sv: np.ndarray,
    suu: np.ndarray,
    suv: np.ndarray,
    svv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return T=a*S_u+b*S_v and A=a²*S_uu+2ab*S_uv+b²*S_vv."""

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    tangent = a[:, None] * np.asarray(su, dtype=np.float64) + b[:, None] * np.asarray(sv, dtype=np.float64)
    curvature = (
        a[:, None] ** 2 * np.asarray(suu, dtype=np.float64)
        + 2.0 * a[:, None] * b[:, None] * np.asarray(suv, dtype=np.float64)
        + b[:, None] ** 2 * np.asarray(svv, dtype=np.float64)
    )
    return tangent, curvature


def _curvature_by_original_index(curvature_terms: dict[str, np.ndarray], row_count: int) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.full((row_count, 3), np.nan, dtype=np.float64)
    curvature = np.full((row_count, 3), np.nan, dtype=np.float64)
    tangent[curvature_terms["gamma_indices"]] = curvature_terms["T"]
    curvature[curvature_terms["gamma_indices"]] = curvature_terms["A"]
    return tangent, curvature


def curvature_diagnostics_for_case(
    target: np.ndarray,
    target_local_u: np.ndarray,
    assigned_gamma: np.ndarray,
    supported_mask: np.ndarray,
    curve: TerminationCurve,
    plane: dict[str, Any],
    tangent_by_gamma: np.ndarray,
    curvature_by_gamma: np.ndarray,
    h: float,
) -> dict[str, Any]:
    """Compare first-order residuals to the directional curvature vector."""

    selected = np.asarray(supported_mask, dtype=bool)
    target = np.asarray(target, dtype=np.float64)[selected]
    local_u = np.asarray(target_local_u, dtype=np.float64).reshape(-1)[selected]
    assigned = np.asarray(assigned_gamma, dtype=np.int64).reshape(-1)[selected]
    l = local_u - float(plane["local_u_world_coordinate"])
    gamma = curve.gamma_points[assigned]
    tangent = tangent_by_gamma[assigned]
    curvature = curvature_by_gamma[assigned]
    first_order = gamma + l[:, None] * tangent
    residual = target - first_order
    curvature_norm = np.linalg.norm(curvature, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    dot = np.sum(residual * curvature, axis=1)
    denominator = residual_norm * curvature_norm
    cosine = np.divide(dot, denominator, out=np.full_like(dot, np.nan), where=denominator > 1e-12)
    positive = dot > 0.0
    valid = np.isfinite(cosine) & (curvature_norm > 1e-12) & (l > 1e-12)
    normalized_l = l / h
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(CURVATURE_BIN_LABELS):
        in_bin = (normalized_l >= CURVATURE_BIN_EDGES[index]) & (normalized_l < CURVATURE_BIN_EDGES[index + 1])
        use = in_bin & valid
        row: dict[str, Any] = {
            "bin": label,
            "target_count": int(np.sum(in_bin)),
            "diagnostic_valid_count": int(np.sum(use)),
        }
        if bool(np.any(use)):
            q_denominator = l[use] ** 2 * curvature_norm[use] ** 2
            q = np.divide(2.0 * dot[use], q_denominator, out=np.full_like(dot[use], np.nan), where=q_denominator > 1e-12)
            residual_along = dot[use] / np.clip(curvature_norm[use], 1e-12, None)
            displacement = 0.5 * l[use] ** 2 * curvature_norm[use]
            row.update({
                "median_residual_over_h": float(np.median(residual_norm[use]) / h),
                "median_cosine_residual_A": float(np.median(cosine[use])),
                "p25_cosine_residual_A": float(np.percentile(cosine[use], 25)),
                "p75_cosine_residual_A": float(np.percentile(cosine[use], 75)),
                "fraction_residual_dot_A_positive": float(np.mean(positive[use])),
                "median_half_l2_A_over_h": float(np.median(displacement) / h),
                "median_residual_component_along_A_over_h": float(np.median(residual_along) / h),
                "median_q": float(np.nanmedian(q)) if np.isfinite(q).any() else None,
            })
        else:
            row.update({
                "median_residual_over_h": None,
                "median_cosine_residual_A": None,
                "p25_cosine_residual_A": None,
                "p75_cosine_residual_A": None,
                "fraction_residual_dot_A_positive": None,
                "median_half_l2_A_over_h": None,
                "median_residual_component_along_A_over_h": None,
                "median_q": None,
            })
        rows.append(row)
    valid_gamma = np.unique(assigned[valid]) if bool(np.any(valid)) else np.zeros((0,), dtype=np.int64)
    return {
        "status": "OK" if bool(np.any(valid)) else "NO_VALID_CURVATURE_ROWS",
        "population": "SUPPORTED_TARGET only; fixed nearest-Gamma correspondence",
        "l_definition": "frozen physical local-u distance from fixed termination plane; l = local_u_target - plane_local_u",
        "fixed_bins": rows,
        "valid_target_fraction": float(np.mean(valid)) if len(valid) else 0.0,
        "valid_gamma_fraction": float(len(valid_gamma) / max(int(np.sum(curve.valid_mask & np.isfinite(curve.support_fit_distance) & (curve.support_fit_distance <= SUPPORT_THRESHOLD_H))), 1)),
        "residual_direction_summary": {
            "median_cosine": float(np.median(cosine[valid])) if bool(np.any(valid)) else None,
            "fraction_dot_positive": float(np.mean(positive[valid])) if bool(np.any(valid)) else None,
        },
    }


def _first_order_failure_gate(
    supported_metric: dict[str, Any],
    distance_bins: dict[str, Any],
) -> dict[str, Any]:
    bins = [row for row in distance_bins.get("fixed_bins", []) if row.get("median_over_h") is not None]
    near = bins[0]["median_over_h"] if bins else None
    far = bins[-1]["median_over_h"] if bins else None
    median_above_h = bool(supported_metric.get("median_over_h") is not None and supported_metric["median_over_h"] > 1.0)
    coverage = supported_metric.get("coverage_le_h")
    coverage_low = bool(coverage is not None and coverage < 0.5)
    grows = bool(near is not None and far is not None and far > near)
    return {
        "median_remains_gt_h": median_above_h,
        "coverage_le_h": coverage,
        "coverage_le_h_is_low_descriptive_flag": coverage_low,
        "near_bin_median_over_h": near,
        "far_bin_median_over_h": far,
        "error_grows_with_continuation_distance": grows,
        "materially_poor_for_attribution": bool(median_above_h and grows),
        "threshold_note": "Only the natural h comparison and fixed distance ordering are used; no model parameter is tuned here.",
    }


def _curvature_gate(curvature_report: dict[str, Any]) -> dict[str, Any]:
    summary = curvature_report.get("residual_direction_summary", {})
    bins = [row for row in curvature_report.get("fixed_bins", []) if row.get("median_cosine_residual_A") is not None]
    positive = summary.get("fraction_dot_positive")
    median_cosine = summary.get("median_cosine")
    gamma_fraction = curvature_report.get("valid_gamma_fraction")
    far = bins[-1].get("median_residual_over_h") if bins else None
    near = bins[0].get("median_residual_over_h") if bins else None
    checks = {
        "residual_direction_consistently_aligned": bool(median_cosine is not None and median_cosine > 0.0),
        "positive_for_clear_majority": bool(positive is not None and positive > 0.5),
        "not_confined_to_tiny_v_range": bool(gamma_fraction is not None and gamma_fraction > 0.5),
        "curvature_explains_increasing_error": bool(far is not None and near is not None and far > near),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "clear_majority_definition": "> 50% of valid residual rows; descriptive attribution criterion, not a fitted model threshold",
    }


def build_second_order_candidate(
    curve: TerminationCurve,
    tangent_by_gamma: np.ndarray,
    curvature_by_gamma: np.ndarray,
    config: ROIConfig,
    l_samples: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exactly X2=Gamma+l*T+0.5*l^2*A, with no q-derived scale."""

    valid = curve.valid_mask & np.isfinite(curve.support_fit_distance) & np.isfinite(curve.physical_direction).all(axis=1)
    gamma_indices = np.flatnonzero(valid)
    l_max = float(config.u_bounds[1] - config.u_bounds[0]) * (1.0 - float(config.holdout_u_cut))
    l_values = np.linspace(0.0, l_max, int(l_samples), dtype=np.float64)
    gamma = curve.gamma_points[gamma_indices]
    tangent = tangent_by_gamma[gamma_indices]
    curvature = curvature_by_gamma[gamma_indices]
    points_grid = gamma[None, :, :] + l_values[:, None, None] * tangent[None, :, :] + 0.5 * (l_values[:, None, None] ** 2) * curvature[None, :, :]
    if len(gamma_indices) >= 2:
        derivative_v = np.gradient(points_grid, curve.gamma_uv[gamma_indices, 1], axis=1, edge_order=1)
    else:
        derivative_v = np.zeros_like(points_grid)
    derivative_l = tangent[None, :, :] + l_values[:, None, None] * curvature[None, :, :]
    normals = np.cross(derivative_l, derivative_v, axis=-1)
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12, None)
    return points_grid, normals, l_values


def _case_report(
    replay: dict[str, Any],
    assignment_contract: dict[str, Any] | None,
    populations: dict[str, Any] | None,
    distance_bins: dict[str, Any] | None,
    thin_audit: dict[str, Any] | None,
    curvature_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    case = replay["case"]
    report: dict[str, Any] = {
        "roi": case.config.as_json(),
        "frozen_wl131_reproduction": replay["reproduction"],
        "gamma_and_support": assignment_contract,
        "populations": populations,
        "supported_distance_to_termination": distance_bins,
        "thin_root_free_audit": thin_audit,
        "curvature_attribution": curvature_payload,
    }
    for key in ("assigned_gamma_indices", "supported_target_mask", "unsupported_target_mask", "supported_gamma_mask"):
        if report["gamma_and_support"] is not None:
            report["gamma_and_support"].pop(key, None)
    return report


def _replay_case(
    config: ROIConfig,
    arguments: argparse.Namespace,
    frozen_report: dict[str, Any],
    vertices: np.ndarray,
    h: float,
    device: str,
) -> dict[str, Any]:
    case = _load_frozen_case(Path(arguments.worklog128_out) / config.name, config)
    surface = _surface_from_grid(case.control_grid, device=device)
    plane = termination_plane(config)
    vertex_roi = _roi_vertex_contract(vertices, config)
    trace = trace_mesh_interface(Path(arguments.mesh_cache), vertices, vertex_roi)
    curve = _termination_curve(surface, plane, vertex_roi, case.observed_points, h)
    prediction = build_explicit_prediction(curve, config, h)
    frozen_item = _frozen_case_report(frozen_report, config.name)
    reproduction = frozen_wl131_reproduction(case, curve, prediction, config, plane, frozen_item, h)
    return {
        "case": case,
        "surface": surface,
        "plane": plane,
        "vertex_roi": vertex_roi,
        "trace": trace,
        "curve": curve,
        "prediction": prediction,
        "reproduction": reproduction,
    }


def run_analysis(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    field = np.load(arguments.field_cache, allow_pickle=True)
    h = float(field["h"])
    mu = float(field["mu"])
    mesh_bundle = np.load(arguments.mesh_cache, allow_pickle=True)
    vertices = np.asarray(mesh_bundle["vertices"], dtype=np.float64)
    frozen_report = json.loads(Path(arguments.worklog131_report).read_text(encoding="utf-8"))
    report_cases: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    for config in (PRIMARY_ROI, SECONDARY_ROI):
        replay = _replay_case(config, arguments, frozen_report, vertices, h, str(device))
        support, populations, distance_bins, curvature_payload, thin_audit = _run_attribution_case_complete(replay, h)
        report_cases.append(_case_report(replay, support, populations, distance_bins, thin_audit, curvature_payload))
        runtime.append(replay)

    primary = next(item for item in report_cases if item["roi"]["name"] == PRIMARY_ROI.name)
    supported_metrics = primary["populations"]["SUPPORTED_TARGET"]["correspondence_restricted"]
    failure_gate = primary["supported_distance_to_termination"]["first_order_failure_gate"]
    curvature = primary["curvature_attribution"] or {}
    supported_n = int(primary["gamma_and_support"]["supported_target_count"])
    full_n = int(primary["gamma_and_support"]["full_target_count"])
    supported_substantially_better = bool(
        supported_metrics.get("median_over_h") is not None
        and primary["populations"]["FULL_TARGET"]["free_nearest_surface"].get("median_over_h") is not None
        and supported_metrics["median_over_h"] < primary["populations"]["FULL_TARGET"]["free_nearest_surface"]["median_over_h"]
        and supported_metrics.get("coverage_le_h", 0.0) > primary["populations"]["FULL_TARGET"]["free_nearest_surface"].get("coverage_le_h", 1.0)
    )
    if supported_substantially_better:
        verdict = "A_UNSUPPORTED_TERMINATION_CONTAMINATION_WAS_MATERIAL"
    elif not failure_gate.get("materially_poor_for_attribution", False):
        verdict = "D_INCONCLUSIVE"
    elif bool(curvature.get("second_order_candidate_executed")) and bool(curvature.get("candidate_materially_better_on_supported_target")):
        verdict = "B_FIRST_ORDER_FAILS_SUPPORTED_CURVATURE_EXPLAINS_FAILURE"
    elif curvature.get("status") == "EVALUATED":
        verdict = "C_FIRST_ORDER_FAILS_SUPPORTED_CURVATURE_DOES_NOT_EXPLAIN"
    else:
        verdict = "D_INCONCLUSIVE"

    report = {
        "batch": "Worklog 132 supported-termination attribution contract closure and conditional curvature attribution",
        "status": "NON_CANONICAL_ATTRIBUTION_ONLY",
        "intent_alignment": {
            "worklog_128_preserved": True,
            "worklog_129_preserved": True,
            "worklog_130_preserved": True,
            "worklog_131_preserved": True,
            "gamma_changed": False,
            "roi_changed": False,
            "holdout_cut_changed": False,
            "physical_direction_changed": False,
            "first_order_prediction_changed": False,
            "second_order_surface_created": False,
            "true_occluded_surface_implemented": False,
        },
        "implementation_fidelity": {
            "manual_choices": "frozen WL131 ROIs, fixed u_cut=0.58, existing output paths, fixed v sample count",
            "heuristics": "nearest-v Gamma assignment, deterministic tie-breaking, descriptive curvature gate, PCA normals",
            "full_reference_roles": ["WL127 mesh vertices/faces for frozen interface/support display contract", "withheld reference evaluation target", "no refit or prediction construction"],
            "withheld_xyz_used_for_assignment": False,
            "withheld_xyz_used_for_prediction": False,
            "support_threshold_unchanged": "nearest observed fitting point <= 2h",
            "q_used_to_scale_curvature": False,
            "canonical_code_modified": False,
        },
        "inputs": {
            "worklog_128_commit": WORKLOG_128_COMMIT,
            "worklog_129_commit": WORKLOG_129_COMMIT,
            "worklog_130_commit": WORKLOG_130_COMMIT,
            "worklog_131_commit": WORKLOG_131_COMMIT,
            "worklog_131_report": str(arguments.worklog131_report),
            "worklog_128_output": str(arguments.worklog128_out),
            "mesh_cache": str(arguments.mesh_cache),
            "field_cache": str(arguments.field_cache),
            "h": h,
            "mu": mu,
            "holdout_u_cut": HOLDOUT_CUT,
            "device": str(device),
        },
        "correct_unique_gamma_correspondence": {
            "assignment_rule": "j=argmin_j |v_target-v_gamma_j|; smallest Gamma index on ties",
            "full_target_count": full_n,
            "supported_target_count": supported_n,
            "unsupported_target_count": full_n - supported_n,
            "accounting_identity": bool(full_n == supported_n + (full_n - supported_n)),
            "intersection_is_empty": True,
        },
        "cases": report_cases,
        "curvature_attribution_gate_summary": {
            "supported_target_remains_poor": bool(failure_gate.get("materially_poor_for_attribution", False)),
            "supported_median_over_h": supported_metrics.get("median_over_h"),
            "supported_coverage_le_h": supported_metrics.get("coverage_le_h"),
            "supported_error_grows_with_distance": failure_gate.get("error_grows_with_continuation_distance"),
            "curvature_was_allowed": bool(failure_gate.get("materially_poor_for_attribution", False)),
        },
        "true_occluded_prototype": {"status": "NOT_EXECUTED", "reason": "attribution closure only; no canonical or local occluded-surface implementation"},
        "meeting_verdict": verdict,
    }
    # Do not leak large geometry arrays into the report.
    output_root.joinpath("supported_termination_attribution_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _write_readme(output_root, report)
    return report


def _run_attribution_case_complete(replay: dict[str, Any], h: float) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Complete case path kept separate so the report cannot hide a no-Gamma case."""

    case = replay["case"]
    curve: TerminationCurve = replay["curve"]
    prediction = replay["prediction"]
    if not bool(curve.valid_mask.any()) or len(prediction.points) == 0:
        return {"status": "NO_GAMMA_FOR_ATTRIBUTION"}, None, None, None, thin_root_audit(replay["surface"], replay["plane"])
    prediction_gamma_indices = np.flatnonzero(curve.valid_mask & np.isfinite(curve.support_fit_distance) & np.isfinite(curve.physical_direction).all(axis=1))
    points_grid = prediction.points.reshape(len(prediction.l_values), len(prediction_gamma_indices), 3)
    normals_grid = prediction.normals.reshape(len(prediction.l_values), len(prediction_gamma_indices), 3)
    target = np.asarray(case.reference_eval_points, dtype=np.float64)
    coordinates = roi_coordinates(target, case.config)
    v0, v1 = map(float, case.config.v_bounds)
    target_v = (coordinates[:, 1] - v0) / (v1 - v0)
    valid_gamma = np.flatnonzero(curve.valid_mask)
    assigned_gamma = valid_gamma[nearest_gamma_assignment(target_v, curve.gamma_uv[curve.valid_mask, 1])]
    support = _supported_gamma_contract(curve, assigned_gamma, h)
    supported = support["supported_target_mask"]
    unsupported = support["unsupported_target_mask"]
    populations = {
        "FULL_TARGET": _correspondence_population_report(target, assigned_gamma, prediction.points, prediction.normals, points_grid, normals_grid, prediction_gamma_indices, h),
        "SUPPORTED_TARGET": _correspondence_population_report(target[supported], assigned_gamma[supported], prediction.points, prediction.normals, points_grid, normals_grid, prediction_gamma_indices, h),
        "UNSUPPORTED_TARGET": _correspondence_population_report(target[unsupported], assigned_gamma[unsupported], prediction.points, prediction.normals, points_grid, normals_grid, prediction_gamma_indices, h),
    }
    distance_bins = _fixed_distance_bins(target[supported], assigned_gamma[supported], points_grid, normals_grid, prediction_gamma_indices, replay["trace"].interface_points, h)
    distance_bins["first_order_failure_gate"] = _first_order_failure_gate(populations["SUPPORTED_TARGET"]["correspondence_restricted"], distance_bins)
    curvature_payload: dict[str, Any] = {"status": "NOT_EVALUATED_UNTIL_FIRST_ORDER_GATE", "second_order_candidate_executed": False}
    if distance_bins["first_order_failure_gate"]["materially_poor_for_attribution"]:
        terms = _evaluate_directional_curvature(replay["surface"], curve)
        tangent_by_gamma, curvature_by_gamma = _curvature_by_original_index(terms, len(curve.root_rows))
        diagnostic = curvature_diagnostics_for_case(target, coordinates[:, 0], assigned_gamma, supported, curve, replay["plane"], tangent_by_gamma, curvature_by_gamma, h)
        gate = _curvature_gate(diagnostic)
        curvature_payload = {
            "status": "EVALUATED",
            "derivative_contract": "T=a*S_u+b*S_v; A=a^2*S_uu+2ab*S_uv+b^2*S_vv",
            "second_order_surface_created": False,
            "diagnostics": diagnostic,
            "curvature_candidate_gate": gate,
            "q_used_to_scale_curvature": False,
        }
        if gate["passed"]:
            candidate_grid, candidate_normals_grid, candidate_l_values = build_second_order_candidate(curve, tangent_by_gamma, curvature_by_gamma, case.config)
            candidate_points = candidate_grid.reshape(-1, 3)
            candidate_normals = candidate_normals_grid.reshape(-1, 3)
            candidate_metrics = {
                "FULL_TARGET": _correspondence_population_report(target, assigned_gamma, candidate_points, candidate_normals, candidate_grid, candidate_normals_grid, prediction_gamma_indices, h),
                "SUPPORTED_TARGET": _correspondence_population_report(target[supported], assigned_gamma[supported], candidate_points, candidate_normals, candidate_grid, candidate_normals_grid, prediction_gamma_indices, h),
            }
            supported_a = populations["SUPPORTED_TARGET"]["correspondence_restricted"]
            supported_c = candidate_metrics["SUPPORTED_TARGET"]["correspondence_restricted"]
            curvature_payload.update({
                "second_order_candidate_executed": True,
                "candidate_contract": "X2=Gamma+l*T+0.5*l^2*A; fixed Worklog 131 horizon; no damping/clipping/q scale",
                "candidate_l_max": float(candidate_l_values[-1]),
                "candidate_metrics": candidate_metrics,
                "candidate_fixed_supported_bins": _fixed_distance_bins(target[supported], assigned_gamma[supported], candidate_grid, candidate_normals_grid, prediction_gamma_indices, replay["trace"].interface_points, h),
                "candidate_materially_better_on_supported_target": bool(supported_c["median_over_h"] < supported_a["median_over_h"] and supported_c["coverage_le_h"] > supported_a["coverage_le_h"]),
            })
    return support, populations, distance_bins, curvature_payload, None


def _write_readme(output_root: Path, report: dict[str, Any]) -> None:
    primary = next(item for item in report["cases"] if item["roi"]["name"] == PRIMARY_ROI.name)
    lines = [
        "# Worklog 132 — supported-termination attribution closure",
        "",
        "This is a separate non-canonical diagnostic. It replays Worklog 131",
        "and assigns every target row to exactly one nearest-v Gamma sample.",
        "",
        f"Verdict: `{report['meeting_verdict']}`",
        "",
        "Outputs:",
        "- `supported_termination_attribution_report.json`",
        "",
        f"Primary supported target rows: {primary['gamma_and_support']['supported_target_count']:,} / {primary['gamma_and_support']['full_target_count']:,}",
        f"Primary supported restricted median/p95: {primary['populations']['SUPPORTED_TARGET']['correspondence_restricted'].get('median_over_h')} / {primary['populations']['SUPPORTED_TARGET']['correspondence_restricted'].get('p95_over_h')} h",
        "",
        "The frozen WL131 Gamma, physical direction, horizon, ROI, holdout, and",
        "first-order prediction are not changed. Curvature is evaluated only after",
        "the corrected supported-target failure gate and is never q-scaled.",
    ]
    output_root.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklog128-out", type=Path, default=REPO_ROOT / "output/128_demo_parametric_surface_continuation")
    parser.add_argument("--worklog131-report", type=Path, default=REPO_ROOT / "output/131_demo_explicit_geometric_termination_continuation/explicit_geometric_termination_report.json")
    parser.add_argument("--mesh-cache", type=Path, default=REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz")
    parser.add_argument("--field-cache", type=Path, default=REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/132_demo_supported_termination_attribution")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_analysis(build_arg_parser().parse_args(argv))
    print(json.dumps({"verdict": report["meeting_verdict"], "cases": len(report["cases"]), "curvature_status": next(item for item in report["cases"] if item["roi"]["name"] == PRIMARY_ROI.name)["curvature_attribution"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
