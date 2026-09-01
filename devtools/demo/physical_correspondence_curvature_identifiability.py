"""Diagnostic closure for physical correspondence and curvature identifiability.

This module is a separate, non-canonical Worklog 133 track.  It replays the
frozen Worklog 132 geometry and first-order continuation, compares its
parametric-v assignment with a physical-v assignment, and decomposes the
residual using a fixed mesh-interface representation offset.  It never
constructs a second-order candidate or changes the meeting-demo geometry.
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

from devtools.demo.explicit_geometric_termination_continuation import TerminationCurve  # noqa: E402
from devtools.demo.parametric_continuation_attribution import (  # noqa: E402
    _angle_degrees,
    trace_mesh_interface,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    PRIMARY_ROI,
    SECONDARY_ROI,
    _jsonable,
    estimate_point_normals,
    roi_coordinates,
)
from devtools.demo.supported_termination_attribution import (  # noqa: E402
    SUPPORT_THRESHOLD_H,
    _correspondence_population_report,
    _curvature_by_original_index,
    _evaluate_directional_curvature,
    _fixed_distance_bins,
    _metric_close,
    _replay_case,
    _sha256_array,
    _supported_gamma_contract,
    nearest_gamma_assignment,
    thin_root_audit,
)


WORKLOG_128_COMMIT = "2d87366b910873562b9dfc223408d85257c5af9f"
WORKLOG_129_COMMIT = "1ca0da5"
WORKLOG_130_COMMIT = "8b2b4e7"
WORKLOG_131_COMMIT = "cf027b4c270a383995c7d23e9b25cd9ba99ee3d1"
WORKLOG_132_COMMIT = "c66c554dd9ddfa1295fd223cfaeed558b0e287aa"

CURVATURE_BIN_EDGES = np.asarray([0.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)
CURVATURE_BIN_LABELS = ("0–2h", "2–4h", "4–8h", "8–16h", ">16h")


def _resolve_existing(path: Path) -> Path:
    """Use the confirmed artifact mirror when the ordinary output is absent."""

    path = Path(path)
    if path.exists():
        return path
    output_root = REPO_ROOT / "output"
    try:
        relative = path.relative_to(output_root)
    except ValueError:
        return path
    confirmed = output_root / "confirmed" / relative
    return confirmed if confirmed.exists() else path


def physical_normalized_v(points: np.ndarray, config: Any) -> np.ndarray:
    """Use the same frozen ROI physical-v definition for targets and Gamma."""

    coordinates = roi_coordinates(points, config)
    v0, v1 = map(float, config.v_bounds)
    return (coordinates[:, 1] - v0) / (v1 - v0)


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    result = spearmanr(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))
    return float(result.statistic) if np.isfinite(result.statistic) else None


def _inversion_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        return 0.0
    row, column = np.triu_indices(len(values), k=1)
    return float(np.mean(values[row] > values[column]))


def gamma_v_audit(gamma_points: np.ndarray, gamma_uv: np.ndarray, config: Any) -> dict[str, Any]:
    """Compare fitted parametric-v and physical Gamma-v without target XYZ."""

    gamma_points = np.asarray(gamma_points, dtype=np.float64)
    gamma_uv = np.asarray(gamma_uv, dtype=np.float64)
    gamma_parametric = gamma_uv[:, 1]
    gamma_physical = physical_normalized_v(gamma_points, config)
    difference = gamma_physical - gamma_parametric
    return {
        "coordinate_definition": "both use ROI origin/axis_v/v_bounds; parametric uses frozen Gamma UV[:,1], physical uses frozen Gamma XYZ",
        "gamma_v_parametric": gamma_parametric,
        "gamma_v_physical": gamma_physical,
        "pearson": _pearson(gamma_parametric, gamma_physical),
        "spearman": _spearman(gamma_parametric, gamma_physical),
        "median_absolute_difference": float(np.median(np.abs(difference))),
        "p95_absolute_difference": float(np.percentile(np.abs(difference), 95)),
        "max_absolute_difference": float(np.max(np.abs(difference))) if len(difference) else None,
        "monotonic_inversion_fraction_physical": _inversion_fraction(gamma_physical),
        "monotonic_inversion_fraction_parametric": _inversion_fraction(gamma_parametric),
        "range_difference": {
            "min_physical_minus_parametric": float(np.min(gamma_physical) - np.min(gamma_parametric)),
            "max_physical_minus_parametric": float(np.max(gamma_physical) - np.max(gamma_parametric)),
        },
        "endpoint_values": {
            "parametric_first": float(gamma_parametric[0]),
            "parametric_last": float(gamma_parametric[-1]),
            "physical_first": float(gamma_physical[0]),
            "physical_last": float(gamma_physical[-1]),
        },
        "gamma_v_parametric_sha256_float32": _sha256_array(gamma_parametric),
        "gamma_v_physical_sha256_float32": _sha256_array(gamma_physical),
    }


def _write_gamma_v_plot(output_path: Path, audit: dict[str, Any], case_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parametric = np.asarray(audit["gamma_v_parametric"], dtype=np.float64)
    physical = np.asarray(audit["gamma_v_physical"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.2, 6.0), facecolor="white")
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#999999", linestyle="--", linewidth=1.2, label="identity")
    axis.plot(parametric, physical, color="#087f8c", linewidth=1.1, alpha=0.72)
    axis.scatter(parametric, physical, color="#d95f02", s=32, zorder=3)
    for index, (x_value, y_value) in enumerate(zip(parametric, physical)):
        axis.text(x_value, y_value, str(index), fontsize=7, ha="left", va="bottom")
    axis.set_xlabel("Gamma v — fitted NURBS parameter")
    axis.set_ylabel("Gamma v — physical/manual ROI")
    axis.set_title(f"Physical correspondence audit — {case_label}")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.20)
    axis.legend(loc="upper left")
    axis.text(0.03, 0.04, f"Pearson {audit['pearson']:.5f}  |  Spearman {audit['spearman']:.5f}\nmedian |Δv| {audit['median_absolute_difference']:.5f}", transform=axis.transAxes, fontsize=10, bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9})
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _assignment_contract(
    target_v: np.ndarray,
    gamma_v_parametric: np.ndarray,
    gamma_v_physical: np.ndarray,
    valid_gamma_indices: np.ndarray,
    supported_gamma: np.ndarray,
) -> dict[str, Any]:
    old_local = nearest_gamma_assignment(target_v, gamma_v_parametric)
    physical_local = nearest_gamma_assignment(target_v, gamma_v_physical)
    old_assignment = valid_gamma_indices[old_local]
    physical_assignment = valid_gamma_indices[physical_local]
    old_supported = supported_gamma[old_assignment]
    physical_supported = supported_gamma[physical_assignment]
    disagreement = old_assignment != physical_assignment
    old_counts = np.bincount(old_assignment, minlength=len(supported_gamma))
    physical_counts = np.bincount(physical_assignment, minlength=len(supported_gamma))
    confusion = np.zeros((len(supported_gamma), len(supported_gamma)), dtype=np.int64)
    np.add.at(confusion, (old_assignment, physical_assignment), 1)
    per_gamma = []
    for index in range(len(supported_gamma)):
        per_gamma.append({
            "gamma_index": int(index),
            "old_parametric_v_count": int(old_counts[index]),
            "physical_v_count": int(physical_counts[index]),
            "change": int(physical_counts[index] - old_counts[index]),
            "supported_gamma": bool(supported_gamma[index]),
        })
    return {
        "old_assignment": old_assignment,
        "physical_assignment": physical_assignment,
        "old_supported_mask": old_supported,
        "physical_supported_mask": physical_supported,
        "assignment_disagreement_count": int(np.sum(disagreement)),
        "assignment_disagreement_fraction": float(np.mean(disagreement)) if len(disagreement) else 0.0,
        "supported_target_count_old": int(np.sum(old_supported)),
        "supported_target_count_physical": int(np.sum(physical_supported)),
        "unsupported_target_count_old": int(np.sum(~old_supported)),
        "unsupported_target_count_physical": int(np.sum(~physical_supported)),
        "confusion_old_rows_physical_columns": confusion,
        "per_gamma": per_gamma,
        "unique_old_assignment": bool(len(old_assignment) == len(target_v)),
        "unique_physical_assignment": bool(len(physical_assignment) == len(target_v)),
    }


def _evaluate_surface_boundary_tangents(surface: Any, gamma_uv: np.ndarray) -> np.ndarray:
    import torch

    uv = torch.as_tensor(gamma_uv, dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    with torch.no_grad():
        _points, _su, sv = surface.evaluate_with_derivatives(uv)
    return sv.detach().cpu().numpy().astype(np.float64)


def boundary_bias_contract(
    replay: dict[str, Any],
    gamma_physical_v: np.ndarray,
    valid_gamma_indices: np.ndarray,
    h: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Use only fixed face-derived interface samples to define B[j]."""

    curve: TerminationCurve = replay["curve"]
    trace = replay["trace"]
    config = replay["case"].config
    interface_points = np.asarray(trace.interface_points, dtype=np.float64)
    if len(interface_points) == 0:
        return {"status": "NO_INTERFACE_POINTS", "selection_uses_target_error": False}, np.full_like(curve.gamma_points, np.nan)
    interface_v = physical_normalized_v(interface_points, config)
    gamma_points = curve.gamma_points[valid_gamma_indices]
    gamma_uv = curve.gamma_uv[valid_gamma_indices]
    gamma_sv = _evaluate_surface_boundary_tangents(replay["surface"], gamma_uv)
    chosen_indices = np.argmin(np.abs(gamma_physical_v[:, None] - interface_v[None, :]), axis=1)
    order = np.argsort(interface_v, kind="mergesort")
    position_by_index = {int(point_index): position for position, point_index in enumerate(order.tolist())}
    boundary = np.full_like(curve.gamma_points, np.nan, dtype=np.float64)
    per_gamma = []
    tangent_angles = []
    for local_index, gamma_index in enumerate(valid_gamma_indices.tolist()):
        interface_index = int(chosen_indices[local_index])
        offset = interface_points[interface_index] - gamma_points[local_index]
        boundary[gamma_index] = offset
        position = position_by_index[interface_index]
        if len(order) >= 2:
            neighbour_position = 1 if position == 0 else (len(order) - 2 if position == len(order) - 1 else position - 1)
            if position not in (0, len(order) - 1):
                neighbour_position = position + 1
            neighbour_index = int(order[neighbour_position])
            interface_tangent = interface_points[neighbour_index] - interface_points[interface_index]
            if np.linalg.norm(interface_tangent) > 1e-12 and np.linalg.norm(gamma_sv[local_index]) > 1e-12:
                tangent_angles.append(float(_angle_degrees(gamma_sv[local_index][None, :], interface_tangent[None, :])[0]))
        per_gamma.append({
            "gamma_index": int(gamma_index),
            "gamma_v_physical": float(gamma_physical_v[local_index]),
            "interface_sample_index": interface_index,
            "interface_v_physical": float(interface_v[interface_index]),
            "B_xyz": offset,
            "B_norm_over_h": float(np.linalg.norm(offset) / h),
        })
    norms = np.asarray([item["B_norm_over_h"] for item in per_gamma], dtype=np.float64)
    return {
        "status": "OK",
        "selection_rule": "one nearest fixed face-derived interface sample in physical-v; argmin ties choose smallest interface row index",
        "selection_uses_target_error": False,
        "normal_angle_discontinuity_degrees": {"status": "unavailable; interface vertex normals are not stored in the frozen trace"},
        "tangent_disagreement_degrees": {
            "status": "estimated from deterministic adjacent interface-v sample",
            "median": float(np.median(tangent_angles)) if tangent_angles else None,
            "p95": float(np.percentile(tangent_angles, 95)) if tangent_angles else None,
            "samples": int(len(tangent_angles)),
        },
        "B_norm_over_h": {
            "median": float(np.median(norms)) if len(norms) else None,
            "p95": float(np.percentile(norms, 95)) if len(norms) else None,
            "max": float(np.max(norms)) if len(norms) else None,
        },
        "per_gamma": per_gamma,
    }, boundary


def _orientation_summary(residual: np.ndarray, curvature: np.ndarray) -> dict[str, Any]:
    residual_norm = np.linalg.norm(residual, axis=1)
    curvature_norm = np.linalg.norm(curvature, axis=1)
    dot = np.sum(residual * curvature, axis=1)
    valid = (residual_norm > 1e-12) & (curvature_norm > 1e-12)
    cosine = dot[valid] / (residual_norm[valid] * curvature_norm[valid])
    positive = dot[valid] > 0.0
    return {
        "median_cosine": float(np.median(cosine)) if len(cosine) else None,
        "p25_cosine": float(np.percentile(cosine, 25)) if len(cosine) else None,
        "p75_cosine": float(np.percentile(cosine, 75)) if len(cosine) else None,
        "fraction_dot_positive": float(np.mean(positive)) if len(positive) else None,
        "valid_count": int(len(cosine)),
    }


def residual_bin_report(
    target: np.ndarray,
    l: np.ndarray,
    raw_residual: np.ndarray,
    delta_residual: np.ndarray,
    curvature: np.ndarray,
    h: float,
) -> dict[str, Any]:
    normalized_l = np.asarray(l, dtype=np.float64) / h
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(CURVATURE_BIN_LABELS):
        selected = (normalized_l >= CURVATURE_BIN_EDGES[index]) & (normalized_l < CURVATURE_BIN_EDGES[index + 1])
        row: dict[str, Any] = {"bin": label, "target_count": int(np.sum(selected))}
        if bool(np.any(selected)):
            raw_norm = np.linalg.norm(raw_residual[selected], axis=1)
            delta_norm = np.linalg.norm(delta_residual[selected], axis=1)
            row.update({
                "raw": {
                    "median_norm_over_h": float(np.median(raw_norm) / h),
                    "p95_norm_over_h": float(np.percentile(raw_norm, 95) / h),
                    "orientation": _orientation_summary(raw_residual[selected], curvature[selected]),
                },
                "bias_corrected": {
                    "median_norm_over_h": float(np.median(delta_norm) / h),
                    "p95_norm_over_h": float(np.percentile(delta_norm, 95) / h),
                    "orientation": _orientation_summary(delta_residual[selected], curvature[selected]),
                },
            })
        else:
            row.update({"raw": None, "bias_corrected": None})
        rows.append(row)
    return {
        "population": "physical-correspondence SUPPORTED_TARGET only",
        "l_definition": "frozen physical local-u distance from fixed termination plane",
        "fixed_bins": rows,
    }


def curvature_identifiability_report(
    target: np.ndarray,
    l: np.ndarray,
    raw_residual: np.ndarray,
    delta_residual: np.ndarray,
    curvature: np.ndarray,
    boundary_floor_median_over_h: float,
    h: float,
) -> dict[str, Any]:
    normalized_l = np.asarray(l, dtype=np.float64) / h
    curvature_signal = 0.5 * np.square(l) * np.linalg.norm(curvature, axis=1) / h
    raw_norm = np.linalg.norm(raw_residual, axis=1) / h
    delta_norm = np.linalg.norm(delta_residual, axis=1) / h
    floor = max(float(boundary_floor_median_over_h), 1e-12)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(CURVATURE_BIN_LABELS):
        selected = (normalized_l >= CURVATURE_BIN_EDGES[index]) & (normalized_l < CURVATURE_BIN_EDGES[index + 1])
        row: dict[str, Any] = {"bin": label, "target_count": int(np.sum(selected))}
        if bool(np.any(selected)):
            signal = curvature_signal[selected]
            raw_ratio = signal / np.maximum(raw_norm[selected], 1e-12)
            delta_ratio = signal / np.maximum(delta_norm[selected], 1e-12)
            signal_floor_ratio = signal / floor
            row.update({
                "median_raw_residual_over_h": float(np.median(raw_norm[selected])),
                "median_delta_residual_over_h": float(np.median(delta_norm[selected])),
                "median_curvature_signal_over_h": float(np.median(signal)),
                "median_curvature_signal_over_boundary_floor": float(np.median(signal_floor_ratio)),
                "p95_curvature_signal_over_boundary_floor": float(np.percentile(signal_floor_ratio, 95)),
                "median_curvature_signal_over_raw_residual": float(np.median(raw_ratio)),
                "median_curvature_signal_over_delta_residual": float(np.median(delta_ratio)),
                "raw_orientation": _orientation_summary(raw_residual[selected], curvature[selected]),
                "bias_corrected_orientation": _orientation_summary(delta_residual[selected], curvature[selected]),
            })
        else:
            row.update({key: None for key in ("median_raw_residual_over_h", "median_delta_residual_over_h", "median_curvature_signal_over_h", "median_curvature_signal_over_boundary_floor", "p95_curvature_signal_over_boundary_floor", "median_curvature_signal_over_raw_residual", "median_curvature_signal_over_delta_residual", "raw_orientation", "bias_corrected_orientation")})
        rows.append(row)
    valid_signal = np.isfinite(curvature_signal)
    signal_floor = curvature_signal[valid_signal] / floor if bool(np.any(valid_signal)) else np.zeros((0,), dtype=np.float64)
    delta_orientation = _orientation_summary(delta_residual, curvature)
    if len(signal_floor) and float(np.mean(signal_floor < 1.0)) >= 0.5:
        classification = "CURVATURE_NOT_IDENTIFIABLE"
    elif delta_orientation["median_cosine"] is not None and delta_orientation["median_cosine"] <= 0.0 and (delta_orientation["fraction_dot_positive"] or 0.0) < 0.5:
        classification = "CURVATURE_IDENTIFIABLE_BUT_ANTI_ALIGNED"
    elif delta_orientation["median_cosine"] is not None:
        classification = "CURVATURE_PREDICTIVE"
    else:
        classification = "INCONCLUSIVE"
    return {
        "boundary_error_floor_median_over_h": float(boundary_floor_median_over_h),
        "fixed_bins": rows,
        "classification": classification,
        "classification_basis": "natural signal/floor comparison and signed residual-curvature orientation; no fitted threshold or q scale",
        "overall_bias_corrected_orientation": delta_orientation,
    }


def _frozen_curvature_matches(current: dict[str, Any], frozen: dict[str, Any]) -> bool:
    if current.get("status") != frozen.get("status"):
        return False
    if current.get("status") != "OK":
        return True
    current_summary = current["residual_direction_summary"]
    frozen_summary = frozen["residual_direction_summary"]
    if not math.isclose(float(current_summary["median_cosine"]), float(frozen_summary["median_cosine"]), rel_tol=2e-5, abs_tol=2e-5) or not math.isclose(float(current_summary["fraction_dot_positive"]), float(frozen_summary["fraction_dot_positive"]), rel_tol=2e-5, abs_tol=2e-5):
        return False
    for current_row, frozen_row in zip(current["fixed_bins"], frozen["fixed_bins"]):
        for key in ("median_residual_over_h", "median_cosine_residual_A", "fraction_residual_dot_A_positive"):
            if current_row.get(key) is None or frozen_row.get(key) is None or not math.isclose(float(current_row[key]), float(frozen_row[key]), rel_tol=2e-5, abs_tol=2e-5):
                return False
    return True


def _wl132_identity(
    replay: dict[str, Any],
    wl132_item: dict[str, Any] | None,
    old_assignment: np.ndarray,
    old_populations: dict[str, Any],
    old_curvature: dict[str, Any],
    frozen_target_hash: str | None,
) -> dict[str, Any]:
    curve: TerminationCurve = replay["curve"]
    valid = np.flatnonzero(curve.valid_mask)
    current_support = curve.valid_mask & np.isfinite(curve.support_fit_distance) & (curve.support_fit_distance <= SUPPORT_THRESHOLD_H)
    current_counts = np.bincount(old_assignment, minlength=len(curve.root_rows))
    if wl132_item is None:
        return {
            "status": "PASS_SOURCE_REPLAY_ONLY",
            "frozen_report_available": False,
            "target_population_count": int(len(old_assignment)),
            "gamma_uv_sha256_float32": _sha256_array(curve.gamma_uv[curve.valid_mask, :]),
            "gamma_xyz_sha256_float32": _sha256_array(curve.gamma_points[curve.valid_mask, :]),
            "physical_direction_sha256_float32": _sha256_array(curve.physical_direction[curve.valid_mask, :]),
        }
    frozen_repro = wl132_item["frozen_wl131_reproduction"]
    saved_support = np.asarray([item["supported_gamma"] for item in wl132_item["gamma_and_support"]["per_gamma"]], dtype=bool)
    saved_counts = np.asarray([item["assigned_target_rows"] for item in wl132_item["gamma_and_support"]["per_gamma"]], dtype=np.int64)
    saved_metrics = wl132_item["populations"]["SUPPORTED_TARGET"]["correspondence_restricted"]
    current_metrics = old_populations["SUPPORTED_TARGET"]["correspondence_restricted"]
    saved_curvature = wl132_item["curvature_attribution"].get("diagnostics", {})
    saved_assignment_rule = wl132_item["gamma_and_support"].get("target_assignment_definition")
    checks = {
        "gamma_uv_hash_equal": _sha256_array(curve.gamma_uv[curve.valid_mask, :]) == frozen_repro["gamma_uv_sha256_float32"],
        "gamma_xyz_hash_equal": _sha256_array(curve.gamma_points[curve.valid_mask, :]) == frozen_repro["gamma_point_sha256_float32"],
        "physical_direction_equal": bool(replay["reproduction"]["physical_direction_identity"]),
        "support_mask_equal": bool(np.array_equal(current_support, saved_support)),
        "target_population_equal": frozen_target_hash is not None and _sha256_array(replay["case"].reference_eval_points) == frozen_target_hash and len(old_assignment) == int(wl132_item["gamma_and_support"]["full_target_count"]),
        "wl132_assignment_rule_equal": saved_assignment_rule == "exactly one nearest manual-normalized-v Gamma index; argmin ties choose smallest Gamma index",
        "old_assignment_counts_equal": bool(np.array_equal(current_counts, saved_counts)),
        "supported_count_equal": int(np.sum(current_support[old_assignment])) == int(wl132_item["gamma_and_support"]["supported_target_count"]),
        "unsupported_count_equal": int(np.sum(~current_support[old_assignment])) == int(wl132_item["gamma_and_support"]["unsupported_target_count"]),
        "correspondence_metric_equal": _metric_close(current_metrics, saved_metrics),
        "curvature_diagnostics_equal": _frozen_curvature_matches(old_curvature, saved_curvature),
    }
    passed = bool(all(checks.values()))
    if not passed:
        raise RuntimeError(f"Worklog 132 frozen identity failed: {checks}")
    return {
        "status": "PASS",
        "frozen_report_available": True,
        "checks": checks,
        "gamma_uv_sha256_float32": _sha256_array(curve.gamma_uv[curve.valid_mask, :]),
        "gamma_xyz_sha256_float32": _sha256_array(curve.gamma_points[curve.valid_mask, :]),
        "physical_direction_sha256_float32": _sha256_array(curve.physical_direction[curve.valid_mask, :]),
        "target_population_sha256_float32": _sha256_array(replay["case"].reference_eval_points),
        "wl132_assignment_population": "replayed exact physical-target-v to parametric-Gamma-v nearest assignment",
        "wl132_physical_direction_identity": bool(replay["reproduction"]["physical_direction_identity"]),
    }


def _case_analysis(replay: dict[str, Any], wl132_item: dict[str, Any] | None, h: float, frozen_target_hash: str | None, output_root: Path | None = None) -> dict[str, Any]:
    case = replay["case"]
    config = case.config
    curve: TerminationCurve = replay["curve"]
    prediction = replay["prediction"]
    vertex_total = int(len(replay["vertex_roi"].full_points))
    vertex_observed = int(len(replay["vertex_roi"].observed_points))
    vertex_withheld = int(len(replay["vertex_roi"].withheld_points))
    partition = {
        "full_roi_vertices": vertex_total,
        "observed_roi_vertices": vertex_observed,
        "withheld_roi_vertices": vertex_withheld,
        "observed_fraction": float(vertex_observed / max(vertex_total, 1)),
        "withheld_fraction": float(vertex_withheld / max(vertex_total, 1)),
        "boundary_attached_rule": "observed u <= frozen holdout_u_cut and withheld u > frozen holdout_u_cut",
    }
    if not bool(curve.valid_mask.any()) or len(prediction.points) == 0:
        return {
            "roi": config.as_json(),
            "wl132_frozen_reproduction": replay["reproduction"],
            "roi_vertex_partition": partition,
            "target_population_contract": {"samples": int(len(case.reference_eval_points))},
            "status": "NO_GAMMA",
            "thin_negative_control": thin_root_audit(replay["surface"], replay["plane"]),
        }
    valid_gamma = np.flatnonzero(curve.valid_mask)
    prediction_gamma_indices = np.flatnonzero(curve.valid_mask & np.isfinite(curve.support_fit_distance) & np.isfinite(curve.physical_direction).all(axis=1))
    prediction_grid = prediction.points.reshape(len(prediction.l_values), len(prediction_gamma_indices), 3)
    normals_grid = prediction.normals.reshape(len(prediction.l_values), len(prediction_gamma_indices), 3)
    target = np.asarray(case.reference_eval_points, dtype=np.float64)
    coordinates = roi_coordinates(target, config)
    target_v = physical_normalized_v(target, config)
    gamma_v_parametric = curve.gamma_uv[valid_gamma, 1]
    gamma_v_physical = physical_normalized_v(curve.gamma_points[valid_gamma], config)
    support_state = curve.valid_mask & np.isfinite(curve.support_fit_distance) & (curve.support_fit_distance <= SUPPORT_THRESHOLD_H)
    assignment = _assignment_contract(target_v, gamma_v_parametric, gamma_v_physical, valid_gamma, support_state)
    old_assignment = assignment["old_assignment"]
    physical_assignment = assignment["physical_assignment"]
    old_supported = assignment["old_supported_mask"]
    physical_supported = assignment["physical_supported_mask"]
    old_populations = {
        "FULL_TARGET": _correspondence_population_report(target, old_assignment, prediction.points, prediction.normals, prediction_grid, normals_grid, prediction_gamma_indices, h),
        "SUPPORTED_TARGET": _correspondence_population_report(target[old_supported], old_assignment[old_supported], prediction.points, prediction.normals, prediction_grid, normals_grid, prediction_gamma_indices, h),
        "UNSUPPORTED_TARGET": _correspondence_population_report(target[~old_supported], old_assignment[~old_supported], prediction.points, prediction.normals, prediction_grid, normals_grid, prediction_gamma_indices, h),
    }
    physical_populations = {
        "FULL_TARGET": _correspondence_population_report(target, physical_assignment, prediction.points, prediction.normals, prediction_grid, normals_grid, prediction_gamma_indices, h),
        "SUPPORTED_TARGET": _correspondence_population_report(target[physical_supported], physical_assignment[physical_supported], prediction.points, prediction.normals, prediction_grid, normals_grid, prediction_gamma_indices, h),
        "UNSUPPORTED_TARGET": _correspondence_population_report(target[~physical_supported], physical_assignment[~physical_supported], prediction.points, prediction.normals, prediction_grid, normals_grid, prediction_gamma_indices, h),
    }
    old_bins = _fixed_distance_bins(target[old_supported], old_assignment[old_supported], prediction_grid, normals_grid, prediction_gamma_indices, replay["trace"].interface_points, h)
    physical_bins = _fixed_distance_bins(target[physical_supported], physical_assignment[physical_supported], prediction_grid, normals_grid, prediction_gamma_indices, replay["trace"].interface_points, h)
    curvature_terms = _evaluate_directional_curvature(replay["surface"], curve)
    tangent_by_gamma, curvature_by_gamma = _curvature_by_original_index(curvature_terms, len(curve.root_rows))
    old_l = coordinates[:, 0] - float(replay["plane"]["local_u_world_coordinate"])
    old_curvature_target = target[old_supported]
    old_curvature_l = old_l[old_supported]
    old_curvature_gamma = old_assignment[old_supported]
    old_raw_residual = old_curvature_target - (curve.gamma_points[old_curvature_gamma] + old_curvature_l[:, None] * tangent_by_gamma[old_curvature_gamma])
    old_curvature_diag = _curvature_summary(old_curvature_target, old_curvature_l, old_curvature_gamma, old_raw_residual, curvature_by_gamma, h)
    wl132_identity = _wl132_identity(replay, wl132_item, old_assignment, old_populations, old_curvature_diag, frozen_target_hash)
    if output_root is not None and config.name == PRIMARY_ROI.name:
        _write_gamma_v_plot(output_root / "gamma_v_parametric_vs_physical.png", {"gamma_v_parametric": gamma_v_parametric, "gamma_v_physical": gamma_v_physical, "pearson": _pearson(gamma_v_parametric, gamma_v_physical), "spearman": _spearman(gamma_v_parametric, gamma_v_physical), "median_absolute_difference": float(np.median(np.abs(gamma_v_physical - gamma_v_parametric)))}, config.semantic_label)
    bias_report, boundary_by_gamma = boundary_bias_contract(replay, gamma_v_physical, valid_gamma, h)
    physical_target = target[physical_supported]
    physical_l = old_l[physical_supported]
    physical_gamma = physical_assignment[physical_supported]
    raw_residual = physical_target - (curve.gamma_points[physical_gamma] + physical_l[:, None] * tangent_by_gamma[physical_gamma])
    delta_residual = physical_target - (curve.gamma_points[physical_gamma] + boundary_by_gamma[physical_gamma] + physical_l[:, None] * tangent_by_gamma[physical_gamma])
    curvature_for_target = curvature_by_gamma[physical_gamma]
    residuals = residual_bin_report(physical_target, physical_l, raw_residual, delta_residual, curvature_for_target, h)
    floor_median = float(bias_report["B_norm_over_h"]["median"])
    identifiability = curvature_identifiability_report(physical_target, physical_l, raw_residual, delta_residual, curvature_for_target, floor_median, h)
    physical_gamma_audit = gamma_v_audit(curve.gamma_points[valid_gamma], curve.gamma_uv[valid_gamma], config)
    return {
        "roi": config.as_json(),
        "wl132_frozen_reproduction": replay["reproduction"],
        "roi_vertex_partition": partition,
        "wl132_identity": wl132_identity,
        "status": "OK",
        "continuation_extent": {
            "physical_local_u": float(prediction.l_values[-1]) if len(prediction.l_values) else 0.0,
            "fixed_rule": "replayed frozen WL132 first-order horizon; no target-selected length",
        },
        "gamma_v_audit": physical_gamma_audit,
        "assignment_contract": assignment,
        "worklog_132_old_parametric_correspondence": {
            "populations": old_populations,
            "supported_distance_bins": old_bins,
            "curvature_diagnostics_replayed": old_curvature_diag,
        },
        "physical_correspondence": {
            "populations": physical_populations,
            "supported_distance_bins": physical_bins,
        },
        "boundary_representation_error_floor": bias_report,
        "raw_vs_bias_corrected_residual": residuals,
        "curvature_signal_identifiability": identifiability,
        "target_population_contract": {
            "samples": int(len(target)),
            "target_xyz_sha256_float32": _sha256_array(target),
            "withheld_geometry_used_for_assignment": True,
            "assignment_use_scope": "evaluation-only physical-v derivation and nearest-Gamma correspondence; not fitter/prediction input",
            "withheld_geometry_used_for_bias_selection": False,
        },
    }


def _curvature_summary(target: np.ndarray, l: np.ndarray, assigned_gamma: np.ndarray, residual: np.ndarray, curvature_by_gamma: np.ndarray, h: float) -> dict[str, Any]:
    normalized_l = np.asarray(l, dtype=np.float64) / h
    curvature = curvature_by_gamma[np.asarray(assigned_gamma, dtype=np.int64)]
    orientation = _orientation_summary(residual, curvature)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(CURVATURE_BIN_LABELS):
        selected = (normalized_l >= CURVATURE_BIN_EDGES[index]) & (normalized_l < CURVATURE_BIN_EDGES[index + 1])
        if bool(np.any(selected)):
            residual_norm = np.linalg.norm(residual[selected], axis=1)
            curv_norm = np.linalg.norm(curvature[selected], axis=1)
            dot = np.sum(residual[selected] * curvature[selected], axis=1)
            cosine = dot / np.maximum(residual_norm * curv_norm, 1e-12)
            rows.append({
                "bin": label,
                "target_count": int(np.sum(selected)),
                "median_residual_over_h": float(np.median(residual_norm) / h),
                "median_cosine_residual_A": float(np.median(cosine)),
                "fraction_residual_dot_A_positive": float(np.mean(dot > 0.0)),
            })
        else:
            rows.append({"bin": label, "target_count": 0, "median_residual_over_h": None, "median_cosine_residual_A": None, "fraction_residual_dot_A_positive": None})
    return {"status": "OK", "fixed_bins": rows, "residual_direction_summary": {"median_cosine": orientation["median_cosine"], "fraction_dot_positive": orientation["fraction_dot_positive"]}}


def run_analysis(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch

    arguments.worklog128_out = _resolve_existing(arguments.worklog128_out)
    arguments.worklog131_report = _resolve_existing(arguments.worklog131_report)
    arguments.worklog130_report = _resolve_existing(arguments.worklog130_report)
    arguments.worklog132_report = _resolve_existing(arguments.worklog132_report)
    arguments.mesh_cache = _resolve_existing(arguments.mesh_cache)
    arguments.field_cache = _resolve_existing(arguments.field_cache)
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
    wl132_report = None
    if Path(arguments.worklog132_report).exists():
        wl132_report = json.loads(Path(arguments.worklog132_report).read_text(encoding="utf-8"))
    wl130_report = None
    if Path(arguments.worklog130_report).exists():
        wl130_report = json.loads(Path(arguments.worklog130_report).read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for config in (PRIMARY_ROI, SECONDARY_ROI):
        replay = _replay_case(config, arguments, json.loads(Path(arguments.worklog131_report).read_text(encoding="utf-8")), vertices, h, str(device))
        wl132_item = None if wl132_report is None else next((item for item in wl132_report.get("cases", []) if item.get("roi", {}).get("name") == config.name), None)
        frozen_geometry_path = _resolve_existing(Path(arguments.worklog128_out) / config.name / "demo_geometry.npz")
        frozen_target_hash = None
        if frozen_geometry_path.exists():
            with np.load(frozen_geometry_path, allow_pickle=True) as frozen_geometry:
                frozen_target_hash = _sha256_array(frozen_geometry["reference_eval_points"])
        cases.append(_case_analysis(replay, wl132_item, h, frozen_target_hash, output_root))
    primary = next(item for item in cases if item["roi"]["name"] == PRIMARY_ROI.name)
    if primary["status"] != "OK":
        verdict = "D"
    else:
        assignment = primary["assignment_contract"]
        physical_metric = primary["physical_correspondence"]["populations"]["SUPPORTED_TARGET"]["correspondence_restricted"]
        old_metric = primary["worklog_132_old_parametric_correspondence"]["populations"]["SUPPORTED_TARGET"]["correspondence_restricted"]
        ident = primary["curvature_signal_identifiability"]
        assignment_material = bool(
            assignment["assignment_disagreement_count"] > 0
            and (
                assignment["supported_target_count_physical"] != assignment["supported_target_count_old"]
                or not _metric_close(physical_metric, old_metric)
            )
        )
        boundary_corrected_still_poor = bool(physical_metric["median_over_h"] > 1.0 and physical_metric["coverage_le_h"] < 0.5)
        if assignment_material:
            verdict = "B"
        elif ident["classification"] == "CURVATURE_NOT_IDENTIFIABLE":
            verdict = "C"
        elif boundary_corrected_still_poor:
            verdict = "A"
        else:
            verdict = "D"
    report = {
        "batch": "Worklog 133 physical correspondence and curvature identifiability closure",
        "status": "NON_CANONICAL_DIAGNOSTIC_ONLY",
        "intent_alignment": {
            "worklog_128_to_132_preserved": True,
            "gamma_changed": False,
            "wl132_prediction_changed": False,
            "second_order_candidate_constructed": False,
            "third_order_added": False,
            "q_fitted_or_used": False,
            "canonical_code_modified": False,
        },
        "implementation_fidelity": {
            "manual_choices": "frozen WL132 ROIs and output artifact paths only",
            "heuristics": "nearest physical-v correspondence, nearest fixed interface-v sample for B, deterministic adjacent tangent estimate",
            "full_reference_roles": ["frozen mesh interface evidence", "withheld target evaluation and evaluation-only physical-v correspondence", "no refit/prediction construction"],
            "target_xyz_used_for_evaluation_correspondence": True,
            "target_xyz_used_for_fit_or_prediction": False,
            "target_error_in_B_selection": False,
            "boundary_bias_correction_is_prediction_method": False,
            "support_threshold": "unchanged <=2h",
        },
        "verdict_basis": {
            "primary_assignment_confounded": bool(assignment_material) if primary["status"] == "OK" else False,
            "primary_physical_supported_metric": primary.get("physical_correspondence", {}).get("populations", {}).get("SUPPORTED_TARGET", {}).get("correspondence_restricted"),
            "primary_curvature_classification": primary.get("curvature_signal_identifiability", {}).get("classification"),
            "decision_rule": "B if physical-v correspondence changes assignments and the supported population or restricted metrics; otherwise C if curvature is below the fixed representation floor, A if the WL132 negative remains poor with identifiable/non-predictive curvature, D otherwise",
        },
        "inputs": {
            "worklog_128_commit": WORKLOG_128_COMMIT,
            "worklog_129_commit": WORKLOG_129_COMMIT,
            "worklog_130_commit": WORKLOG_130_COMMIT,
            "worklog_131_commit": WORKLOG_131_COMMIT,
            "worklog_132_commit": WORKLOG_132_COMMIT,
            "worklog_130_report": str(arguments.worklog130_report),
            "worklog_132_report": str(arguments.worklog132_report),
            "worklog_131_report": str(arguments.worklog131_report),
            "h": h,
            "mu": mu,
            "holdout_u_cut": 0.58,
            "device": str(device),
        },
        "worklog_130_v_parameterization": {
            "status": "FROZEN_REPORT_AVAILABLE" if wl130_report is not None else "UNAVAILABLE",
            "curved_rim": next((item["parameterization"]["v"] for item in wl130_report.get("cases", []) if item.get("roi", {}).get("name") == PRIMARY_ROI.name), None) if wl130_report is not None else None,
        },
        "cases": cases,
        "architecture_verdict": verdict,
        "architecture_verdict_description": {
            "A": "STRONG FEASIBILITY DEMO",
            "B": "PARTIAL FEASIBILITY DEMO — physical-v correspondence confounds the WL132 attribution",
            "C": "NEGATIVE FEASIBILITY RESULT — curvature is not identifiable above the representation floor",
            "D": "INCONCLUSIVE",
        }[verdict],
        "true_occluded_prototype": {"status": "NOT_EXECUTED", "reason": "diagnostic-only correspondence/floor closure; no occluded surface"},
    }
    output_root.joinpath("physical_correspondence_curvature_identifiability_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _write_readme(output_root, report)
    return report


def _write_readme(output_root: Path, report: dict[str, Any]) -> None:
    primary = next(item for item in report["cases"] if item["roi"]["name"] == PRIMARY_ROI.name)
    lines = [
        "# Worklog 133 — physical correspondence and curvature identifiability closure",
        "",
        "Separate non-canonical diagnostic. WL132 Gamma, physical first-order",
        "prediction, ROI, holdout, horizon, and support state are replayed read-only.",
        "",
        f"Architecture verdict: `{report['architecture_verdict']}`",
        "",
        "Outputs:",
        "- `physical_correspondence_curvature_identifiability_report.json`",
        "- `gamma_v_parametric_vs_physical.png`",
        "",
        f"Physical Gamma-v assignment disagreement: {primary['assignment_contract']['assignment_disagreement_count']} rows ({primary['assignment_contract']['assignment_disagreement_fraction']:.3%})",
        f"Physical supported target: {primary['assignment_contract']['supported_target_count_physical']:,} / {primary['target_population_contract']['samples']:,}",
        f"Curvature classification: `{primary['curvature_signal_identifiability']['classification']}`",
        "",
        "No second-order candidate, q scaling, third-order continuation, or canonical",
        "production change was made.",
    ]
    output_root.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklog128-out", type=Path, default=REPO_ROOT / "output/128_demo_parametric_surface_continuation")
    parser.add_argument("--worklog130-report", type=Path, default=REPO_ROOT / "output/130_demo_parametric_continuation_attribution/parametric_continuation_attribution_report.json")
    parser.add_argument("--worklog131-report", type=Path, default=REPO_ROOT / "output/131_demo_explicit_geometric_termination_continuation/explicit_geometric_termination_report.json")
    parser.add_argument("--worklog132-report", type=Path, default=REPO_ROOT / "output/132_demo_supported_termination_attribution/supported_termination_attribution_report.json")
    parser.add_argument("--mesh-cache", type=Path, default=REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz")
    parser.add_argument("--field-cache", type=Path, default=REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/133_demo_physical_correspondence_curvature_identifiability")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_analysis(build_arg_parser().parse_args(argv))
    primary = next(item for item in report["cases"] if item["roi"]["name"] == PRIMARY_ROI.name)
    print(json.dumps({"verdict": report["architecture_verdict"], "cases": len(report["cases"]), "primary_identity": primary.get("wl132_identity", {}).get("status"), "curvature": primary.get("curvature_signal_identifiability", {}).get("classification")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
