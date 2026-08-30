"""Worklog 134: isolated occlusion-bounded surface feasibility demo.

This file is deliberately a meeting-demo namespace.  It measures observed
table surface relationships, builds two fixed pseudo-occlusion holdouts from
the frozen WL127 Visible Surface, and optionally applies the same simple
geometric primitives to one Candidate-B-supported table region.

The construction functions accept retained visible geometry and fixed
constraints only.  Held-out reference points are passed to evaluation after
construction and are never used for branch, extent, or endpoint selection.
No canonical module or renderer is changed by this diagnostic.
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

from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    PRIMARY_ROI,
    SECONDARY_ROI,
    _jsonable,
    _plot_coords,
    _scatter3d,
    _set_equal_3d_limits,
    deterministic_subsample,
    estimate_point_normals,
    roi_coordinates,
)


OBSERVED_GREY = (0.58, 0.60, 0.63)
TOP_GREEN = (0.22, 0.62, 0.25)
SIDE_BLUE = (0.12, 0.40, 0.85)
LEG_MAGENTA = (0.82, 0.18, 0.70)
PSEUDO_ORANGE = (0.95, 0.48, 0.08)
TRUE_OCCLUDED_ORANGE = (0.98, 0.45, 0.08)
FREE_RED = (0.86, 0.13, 0.12)
FRONTIER_YELLOW = (0.98, 0.80, 0.08)
NORMAL_BLUE = (0.05, 0.28, 0.92)
JUNCTION_MAGENTA = (0.85, 0.05, 0.70)
PREDICTED_CYAN = (0.00, 0.72, 0.82)
HELDOUT_GREEN = (0.10, 0.75, 0.32)

WORKLOG_127_MESH = REPO_ROOT / "output/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz"
WORKLOG_127_FIELD = REPO_ROOT / "output/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz"
CANDIDATE_B_ARCHIVE = REPO_ROOT / "output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"


@dataclass(frozen=True)
class Box:
    """Axis-aligned demo-only world-space box."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def contains(self, points: np.ndarray, *, tolerance: float = 0.0) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64) - float(tolerance)
        upper = np.asarray(self.upper, dtype=np.float64) + float(tolerance)
        return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)

    def as_json(self) -> dict[str, list[float]]:
        return {"lower": list(self.lower), "upper": list(self.upper)}


@dataclass(frozen=True)
class TableDemoConfig:
    """One manually fixed table-focused meeting configuration."""

    table_scene_box: Box
    top_patch_box: Box
    side_patch_box: Box
    leg_patch_box: Box
    h1_side_u_cut: float
    h1_side_u_extent: tuple[float, float]
    h1_side_v_extent: tuple[float, float]
    h1_side_n_extent: tuple[float, float]
    h1_pseudo_volume: Box
    h2_pseudo_volume: Box
    true_rim_volume: Box
    true_leg_volume: Box
    h1_frontier_band_fraction: float = 0.10
    frontier_bins: int = 32
    continuation_samples: int = 32

    def as_json(self) -> dict[str, Any]:
        return {
            "table_scene_box": self.table_scene_box.as_json(),
            "top_patch_box": self.top_patch_box.as_json(),
            "side_patch_box": self.side_patch_box.as_json(),
            "leg_patch_box": self.leg_patch_box.as_json(),
            "h1_side_u_cut": self.h1_side_u_cut,
            "h1_side_u_extent": list(self.h1_side_u_extent),
            "h1_side_v_extent": list(self.h1_side_v_extent),
            "h1_side_n_extent": list(self.h1_side_n_extent),
            "h1_pseudo_volume": self.h1_pseudo_volume.as_json(),
            "h2_pseudo_volume": self.h2_pseudo_volume.as_json(),
            "true_rim_volume": self.true_rim_volume.as_json(),
            "true_leg_volume": self.true_leg_volume.as_json(),
            "h1_frontier_band_fraction": self.h1_frontier_band_fraction,
            "frontier_bins": self.frontier_bins,
            "continuation_samples": self.continuation_samples,
        }


TABLE_DEMO = TableDemoConfig(
    # Fixed presentation crop from the WL127 table view.
    table_scene_box=Box((-12.0, -1.0, -10.0), (20.0, 2.0, 6.0)),
    # Adjacent upper/lower strips around the fixed curved side ROI.  These
    # are semantic demo seeds, not an automatic segmentation result.
    top_patch_box=Box((-6.6, 1.75, 3.0), (-4.7, 2.10, 4.2)),
    side_patch_box=Box((-6.6, 1.10, 3.0), (-4.7, 1.70, 4.2)),
    leg_patch_box=Box((-0.30, 0.50, 0.70), (0.40, 1.10, 1.20)),
    h1_side_u_cut=-5.498,
    h1_side_u_extent=(-6.6, -4.7),
    h1_side_v_extent=(3.0, 4.2),
    h1_side_n_extent=(1.10, 1.70),
    h1_pseudo_volume=Box((-5.55, 1.05, 2.95), (-4.65, 1.75, 4.25)),
    h2_pseudo_volume=Box((-5.55, 1.05, 2.95), (-4.65, 1.75, 4.25)),
    # Candidate-B-supported under-table local boxes.  These are fixed before
    # evaluating the controlled holdout and are not fitted to a hidden target.
    true_rim_volume=Box((-7.0, 0.0, 2.0), (-4.0, 3.0, 5.0)),
    true_leg_volume=Box((-1.0, -1.0, 0.30), (1.0, 3.0, 4.0)),
)


@dataclass
class PatchStats:
    label: str
    points: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    tangent_1: np.ndarray
    tangent_2: np.ndarray
    normal_dispersion_median_degrees: float
    normal_dispersion_p95_degrees: float
    plane_residual_median: float
    plane_residual_p95: float
    spatial_extent: np.ndarray


@dataclass
class Holdout:
    name: str
    full_points: np.ndarray
    retained_points: np.ndarray
    withheld_points: np.ndarray
    retained_mask: np.ndarray
    withheld_mask: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    n_axis: np.ndarray
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    n_bounds: tuple[float, float]
    u_cut: float
    permitted_volume: Box

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "full_point_count": int(len(self.full_points)),
            "retained_point_count": int(len(self.retained_points)),
            "withheld_point_count": int(len(self.withheld_points)),
            "retained_fraction": float(np.mean(self.retained_mask)) if len(self.retained_mask) else 0.0,
            "withheld_fraction": float(np.mean(self.withheld_mask)) if len(self.withheld_mask) else 0.0,
            "u_cut": self.u_cut,
            "u_bounds": list(self.u_bounds),
            "v_bounds": list(self.v_bounds),
            "n_bounds": list(self.n_bounds),
            "boundary_attached": bool(
                len(self.retained_points)
                and len(self.withheld_points)
                and float(np.max(self._u_normalized(self.retained_points))) <= self._cut_normalized() + 1e-12
                and float(np.min(self._u_normalized(self.withheld_points))) > self._cut_normalized()
            ),
            "interior_hole_only": False,
            "permitted_volume": self.permitted_volume.as_json(),
        }

    def _u_normalized(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points) @ self.u_axis
        return (values - self.u_bounds[0]) / (self.u_bounds[1] - self.u_bounds[0])

    def _cut_normalized(self) -> float:
        return (self.u_cut - self.u_bounds[0]) / (self.u_bounds[1] - self.u_bounds[0])


@dataclass
class Prediction:
    name: str
    status: str
    points_grid: np.ndarray
    normals_grid: np.ndarray
    frontier_points: np.ndarray
    frontier_tangent: np.ndarray
    continuation_direction: np.ndarray
    l_values: np.ndarray
    branch_diagnostics: dict[str, Any]

    @property
    def points(self) -> np.ndarray:
        return self.points_grid.reshape(-1, 3)

    @property
    def normals(self) -> np.ndarray:
        return self.normals_grid.reshape(-1, 3)

    @property
    def triangle_count(self) -> int:
        if self.points_grid.ndim != 3 or self.points_grid.shape[0] < 2 or self.points_grid.shape[1] < 2:
            return 0
        return int(2 * (self.points_grid.shape[0] - 1) * (self.points_grid.shape[1] - 1))


@dataclass
class FreeSpaceProxy:
    name: str
    points: np.ndarray
    radius: float

    def violation_mask(self, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.float64)
        if len(candidates) == 0 or len(self.points) == 0:
            return np.zeros((len(candidates),), dtype=bool)
        from scipy.spatial import cKDTree

        distances = cKDTree(self.points).query(candidates, workers=1)[0]
        return np.asarray(distances <= float(self.radius), dtype=bool)


def _normalised_axis(axis: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(axis), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    return value / norm


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values), dtype=np.float32).tobytes()).hexdigest()


def _select_box(vertices: np.ndarray, box: Box, max_points: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    mask = box.contains(vertices)
    return deterministic_subsample(vertices[mask], max_points)


def _median_oriented_normal(normals: np.ndarray) -> np.ndarray:
    reference = np.asarray(normals[0], dtype=np.float64)
    aligned = np.asarray(normals, dtype=np.float64).copy()
    signs = np.where(aligned @ reference < 0.0, -1.0, 1.0)
    aligned *= signs[:, None]
    normal = np.mean(aligned, axis=0)
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        normal = reference
        length = max(float(np.linalg.norm(normal)), 1e-12)
    return normal / length


def measure_patch(label: str, points: np.ndarray, *, max_normal_points: int = 5000) -> PatchStats:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 8:
        raise ValueError(f"patch {label} needs at least eight points")
    normal_points = deterministic_subsample(points, max_normal_points)
    normals = estimate_point_normals(normal_points, k=20)
    if normals is None:
        centered = normal_points - normal_points.mean(axis=0, keepdims=True)
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        normals = np.tile(vectors[:, 0][None, :], (len(normal_points), 1))
    normal = _median_oriented_normal(normals)
    angles = np.degrees(np.arccos(np.clip(np.abs(normals @ normal), 0.0, 1.0)))
    centered_all = points - points.mean(axis=0, keepdims=True)
    covariance = centered_all.T @ centered_all
    _values, vectors = np.linalg.eigh(covariance)
    tangent_1 = vectors[:, 2]
    tangent_2 = vectors[:, 1]
    if abs(float(tangent_1 @ np.array([1.0, 0.0, 0.0]))) < abs(float(tangent_2 @ np.array([1.0, 0.0, 0.0]))):
        tangent_1, tangent_2 = tangent_2, tangent_1
    tangent_1 /= max(float(np.linalg.norm(tangent_1)), 1e-12)
    tangent_2 /= max(float(np.linalg.norm(tangent_2)), 1e-12)
    residual = np.abs(centered_all @ normal)
    return PatchStats(
        label=label,
        points=points,
        centroid=points.mean(axis=0),
        normal=normal,
        tangent_1=tangent_1,
        tangent_2=tangent_2,
        normal_dispersion_median_degrees=float(np.median(angles)),
        normal_dispersion_p95_degrees=float(np.percentile(angles, 95)),
        plane_residual_median=float(np.median(residual)),
        plane_residual_p95=float(np.percentile(residual, 95)),
        spatial_extent=np.ptp(points, axis=0),
    )


def _patch_stats_report(stats: PatchStats) -> dict[str, Any]:
    return {
        "label": stats.label,
        "point_count": int(len(stats.points)),
        "robust_median_normal": stats.normal,
        "normal_angular_dispersion_degrees": {
            "median": stats.normal_dispersion_median_degrees,
            "p95": stats.normal_dispersion_p95_degrees,
        },
        "dominant_tangent_directions": np.stack([stats.tangent_1, stats.tangent_2]),
        "local_plane_residual": {
            "median": stats.plane_residual_median,
            "p95": stats.plane_residual_p95,
        },
        "spatial_extent_xyz": stats.spatial_extent,
    }


def measure_junction_relation(top: PatchStats, side: PatchStats) -> dict[str, Any]:
    """Measure an observed top-side relation from retained visible patches."""

    theta = float(np.degrees(np.arccos(np.clip(abs(float(top.normal @ side.normal)), 0.0, 1.0))))
    count = min(len(top.points), len(side.points), 512)
    top_normals = estimate_point_normals(deterministic_subsample(top.points, count), k=min(20, count))
    side_normals = estimate_point_normals(deterministic_subsample(side.points, count), k=min(20, count))
    if top_normals is None or side_normals is None:
        pair_angles = np.zeros((0,), dtype=np.float64)
    else:
        pair_angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(top_normals * side_normals, axis=1)), 0.0, 1.0)))
    return {
        "status": "MEASURED" if count else "UNAVAILABLE",
        "theta_visible_degrees": theta,
        "theta_pairwise_dispersion_degrees": {
            "median": float(np.median(pair_angles)) if len(pair_angles) else None,
            "p95": float(np.percentile(pair_angles, 95)) if len(pair_angles) else None,
            "samples": int(len(pair_angles)),
        },
        "source": "retained observed top-like and side-like visible patches only",
        "hard_coded_right_angle": False,
    }


def build_fixed_holdout(
    points: np.ndarray,
    *,
    name: str,
    u_axis: Iterable[float],
    v_axis: Iterable[float],
    n_axis: Iterable[float],
    u_bounds: tuple[float, float],
    v_bounds: tuple[float, float],
    n_bounds: tuple[float, float],
    u_cut: float,
    permitted_volume: Box,
) -> Holdout:
    """Apply a fixed one-sided boundary mask before any evaluation."""

    points = np.asarray(points, dtype=np.float64)
    u_axis = _normalised_axis(u_axis)
    v_axis = _normalised_axis(v_axis)
    n_axis = _normalised_axis(n_axis)
    coordinates = np.column_stack([points @ u_axis, points @ v_axis, points @ n_axis])
    u0, u1 = map(float, u_bounds)
    v0, v1 = map(float, v_bounds)
    n0, n1 = map(float, n_bounds)
    full = (
        (coordinates[:, 0] >= u0)
        & (coordinates[:, 0] <= u1)
        & (coordinates[:, 1] >= v0)
        & (coordinates[:, 1] <= v1)
        & (coordinates[:, 2] >= n0)
        & (coordinates[:, 2] <= n1)
    )
    full_points = points[full]
    full_coordinates = coordinates[full]
    retained = full_coordinates[:, 0] <= float(u_cut)
    withheld = full_coordinates[:, 0] > float(u_cut)
    return Holdout(
        name=name,
        full_points=full_points,
        retained_points=full_points[retained],
        withheld_points=full_points[withheld],
        retained_mask=retained,
        withheld_mask=withheld,
        u_axis=u_axis,
        v_axis=v_axis,
        n_axis=n_axis,
        u_bounds=(u0, u1),
        v_bounds=(v0, v1),
        n_bounds=(n0, n1),
        u_cut=float(u_cut),
        permitted_volume=permitted_volume,
    )


def _frontier_frame(
    retained_points: np.ndarray,
    *,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    n_axis: np.ndarray,
    u_cut: float,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    band_fraction: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate frontier and first-order continuation directions from retained points."""

    retained_points = np.asarray(retained_points, dtype=np.float64)
    coordinates = np.column_stack([retained_points @ u_axis, retained_points @ v_axis, retained_points @ n_axis])
    band = coordinates[:, 0] >= float(u_cut) - float(band_fraction) * (float(u1) - float(u0))
    candidates = retained_points[band] if int(np.sum(band)) >= max(8, bins // 2) else retained_points
    candidate_coordinates = coordinates[band] if int(np.sum(band)) >= max(8, bins // 2) else coordinates
    order = np.argsort(candidate_coordinates[:, 1], kind="mergesort")
    candidates = candidates[order]
    candidate_coordinates = candidate_coordinates[order]
    edges = np.linspace(float(v0), float(v1), int(bins) + 1)
    frontier_rows: list[np.ndarray] = []
    for index in range(int(bins)):
        selected = (candidate_coordinates[:, 1] >= edges[index]) & (candidate_coordinates[:, 1] < edges[index + 1] if index + 1 < bins else candidate_coordinates[:, 1] <= edges[index + 1])
        if bool(np.any(selected)):
            local = np.flatnonzero(selected)
            frontier_rows.append(candidates[local[np.argmax(candidate_coordinates[local, 0])]])
    if len(frontier_rows) < 4:
        local = np.argsort(candidate_coordinates[:, 0], kind="mergesort")[-max(4, min(len(candidates), bins)):]
        frontier = candidates[local]
    else:
        frontier = np.stack(frontier_rows, axis=0)
    frontier_coordinates = np.column_stack([frontier @ u_axis, frontier @ v_axis, frontier @ n_axis]
    )
    frontier = frontier[np.argsort(frontier_coordinates[:, 1], kind="mergesort")]
    frontier_centered = frontier - frontier.mean(axis=0, keepdims=True)
    _values, vectors = np.linalg.eigh(frontier_centered.T @ frontier_centered)
    tangent = vectors[:, 2]
    if float(tangent @ v_axis) < 0.0:
        tangent = -tangent
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    normals = estimate_point_normals(retained_points, k=20)
    if normals is None:
        centered = retained_points - retained_points.mean(axis=0, keepdims=True)
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        normal = vectors[:, 0]
    else:
        normal = _median_oriented_normal(normals)
    continuation = u_axis - float(u_axis @ normal) * normal
    if float(np.linalg.norm(continuation)) <= 1e-10:
        continuation = u_axis.copy()
    continuation /= max(float(np.linalg.norm(continuation)), 1e-12)
    if float(continuation @ u_axis) < 0.0:
        continuation = -continuation
    return frontier, tangent, continuation, normal


def _surface_normals_from_directions(direction: np.ndarray, tangent: np.ndarray, count: int, width: int) -> np.ndarray:
    normal = np.cross(direction, tangent)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return np.tile(normal[None, None, :], (int(count), int(width), 1))


def build_self_continuation(holdout: Holdout, *, frontier_band_fraction: float = 0.10, frontier_bins: int = 32, continuation_samples: int = 32) -> Prediction:
    """Build a ruled first-order self-continuation from retained geometry only."""

    frontier, tangent, direction, normal = _frontier_frame(
        holdout.retained_points,
        u_axis=holdout.u_axis,
        v_axis=holdout.v_axis,
        n_axis=holdout.n_axis,
        u_cut=holdout.u_cut,
        u0=holdout.u_bounds[0],
        u1=holdout.u_bounds[1],
        v0=holdout.v_bounds[0],
        v1=holdout.v_bounds[1],
        band_fraction=frontier_band_fraction,
        bins=frontier_bins,
    )
    extent = float(holdout.u_bounds[1] - holdout.u_cut)
    l_values = np.linspace(0.0, extent, int(continuation_samples), dtype=np.float64)
    points_grid = frontier[None, :, :] + l_values[:, None, None] * direction[None, None, :]
    normals_grid = _surface_normals_from_directions(direction, tangent, len(l_values), len(frontier))
    return Prediction(
        name=holdout.name,
        status="VALID",
        points_grid=points_grid,
        normals_grid=normals_grid,
        frontier_points=frontier,
        frontier_tangent=tangent,
        continuation_direction=direction,
        l_values=l_values,
        branch_diagnostics={
            "mechanism": "retained-visible frontier tangent sweep",
            "source_reference_used": False,
            "second_order_taylor_used": False,
            "extent_world": extent,
            "normal_estimate": normal,
        },
    )


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    vector = np.asarray(vector, dtype=np.float64)
    return vector * math.cos(angle_radians) + np.cross(axis, vector) * math.sin(angle_radians) + axis * float(axis @ vector) * (1.0 - math.cos(angle_radians))


def _branch_prediction(frontier: np.ndarray, tangent: np.ndarray, normal: np.ndarray, angle_degrees: float, sign: int, extent: float, samples: int, u_axis: np.ndarray) -> Prediction:
    branch_normal = _rotate_about_axis(normal, tangent, math.radians(float(sign) * float(angle_degrees)))
    direction = np.cross(tangent, branch_normal)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    if float(direction @ u_axis) < 0.0:
        direction = -direction
    l_values = np.linspace(0.0, float(extent), int(samples), dtype=np.float64)
    points_grid = frontier[None, :, :] + l_values[:, None, None] * direction[None, None, :]
    normals_grid = np.tile(branch_normal[None, None, :], (len(l_values), len(frontier), 1))
    return Prediction(
        name=f"junction_branch_{'plus' if sign > 0 else 'minus'}",
        status="UNVALIDATED",
        points_grid=points_grid,
        normals_grid=normals_grid,
        frontier_points=frontier,
        frontier_tangent=tangent,
        continuation_direction=direction,
        l_values=l_values,
        branch_diagnostics={
            "sign": int(sign),
            "transferred_angle_degrees": float(angle_degrees),
            "branch_normal": branch_normal,
            "source_reference_used": False,
            "second_order_taylor_used": False,
        },
    )


def validate_branch(prediction: Prediction, *, permitted_volume: Box, free_space: FreeSpaceProxy | None, source_normal: np.ndarray, source_frontier: np.ndarray, u_axis: np.ndarray) -> dict[str, Any]:
    points = prediction.points
    inside = permitted_volume.contains(points, tolerance=1e-8)
    free_violation = free_space.violation_mask(points) if free_space is not None else np.zeros((len(points),), dtype=bool)
    back_through = bool(float(prediction.continuation_direction @ u_axis) < 0.0)
    source_plane = np.sum((points - source_frontier.mean(axis=0)) * source_normal[None, :], axis=1)
    back_through = back_through or bool(np.all(source_plane < -1e-8))
    return {
        "valid": bool(np.all(inside) and not bool(np.any(free_violation)) and not back_through),
        "inside_permitted_volume": bool(np.all(inside)),
        "outside_permitted_volume_point_count": int(np.sum(~inside)),
        "known_free_space_violation_point_count": int(np.sum(free_violation)),
        "points_back_through_visible_source": bool(back_through),
    }


def select_junction_branch(branches: dict[str, tuple[Prediction, dict[str, Any]]]) -> tuple[str, Prediction | None, dict[str, Any]]:
    """Select only an unambiguous valid branch; no reference/error access."""

    valid = [(name, prediction, diagnostics) for name, (prediction, diagnostics) in branches.items() if diagnostics.get("valid", False)]
    if len(valid) == 1:
        name, prediction, diagnostics = valid[0]
        prediction.status = "VALID"
        return "SELECTED", prediction, {"selected_branch": name, "valid_branch_count": 1, "branches": branches}
    if len(valid) > 1:
        return "AMBIGUOUS", None, {"selected_branch": None, "valid_branch_count": len(valid), "branches": branches}
    return "NO_VALID_TRANSFER", None, {"selected_branch": None, "valid_branch_count": 0, "branches": branches}


def build_junction_transfer(
    holdout: Holdout,
    *,
    source_top: PatchStats,
    source_side: PatchStats,
    source_frontier: np.ndarray,
    source_tangent: np.ndarray,
    source_normal: np.ndarray,
    free_space: FreeSpaceProxy | None = None,
    extent_fraction: float = 0.42,
    continuation_samples: int = 32,
) -> tuple[Prediction | None, dict[str, Any]]:
    """Transfer the observed top-side angle to a retained target frontier."""

    relation = measure_junction_relation(source_top, source_side)
    extent = float(holdout.u_bounds[1] - holdout.u_cut) * float(extent_fraction)
    candidates: dict[str, tuple[Prediction, dict[str, Any]]] = {}
    for sign, label in ((1, "plus_theta"), (-1, "minus_theta")):
        branch = _branch_prediction(source_frontier, source_tangent, source_normal, relation["theta_visible_degrees"], sign, extent, continuation_samples, holdout.u_axis)
        diagnostics = validate_branch(
            branch,
            permitted_volume=holdout.permitted_volume,
            free_space=free_space,
            source_normal=source_normal,
            source_frontier=source_frontier,
            u_axis=holdout.u_axis,
        )
        candidates[label] = (branch, diagnostics)
    status, prediction, selection = select_junction_branch(candidates)
    return prediction, {
        "status": status,
        "theta_visible_degrees": relation["theta_visible_degrees"],
        "theta_measurement": relation,
        "direct_continuation_is_not_used_as_template": True,
        "branch_selection_uses_withheld_reference": False,
        "extent_world": extent,
        "branches": {name: {**prediction.branch_diagnostics, **diagnostics} for name, (prediction, diagnostics) in candidates.items()},
        "selection": selection,
    }


def _serializable_transfer_report(transfer: dict[str, Any]) -> dict[str, Any]:
    """Drop in-memory Prediction objects before writing the JSON report."""

    report = dict(transfer)
    selection = dict(report.get("selection", {}))
    report["selection"] = {
        "selected_branch": selection.get("selected_branch"),
        "valid_branch_count": int(selection.get("valid_branch_count", 0)),
    }
    return report


def _surface_area(points_grid: np.ndarray) -> float:
    points_grid = np.asarray(points_grid, dtype=np.float64)
    if points_grid.ndim != 3 or points_grid.shape[0] < 2 or points_grid.shape[1] < 2:
        return 0.0
    a = points_grid[:-1, :-1]
    b = points_grid[1:, :-1]
    c = points_grid[:-1, 1:]
    d = points_grid[1:, 1:]
    return float(0.5 * (np.linalg.norm(np.cross(b - a, c - a), axis=-1) + np.linalg.norm(np.cross(d - b, c - b), axis=-1)).sum())


def _masked_grid_area(points_grid: np.ndarray, point_mask: np.ndarray) -> float:
    """Approximate area touched by a boolean point mask on a ruled grid."""

    points_grid = np.asarray(points_grid, dtype=np.float64)
    point_mask = np.asarray(point_mask, dtype=bool).reshape(points_grid.shape[:2])
    if points_grid.ndim != 3 or points_grid.shape[0] < 2 or points_grid.shape[1] < 2:
        return 0.0
    a = points_grid[:-1, :-1]
    b = points_grid[1:, :-1]
    c = points_grid[:-1, 1:]
    d = points_grid[1:, 1:]
    triangle_areas = np.stack([
        0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1),
        0.5 * np.linalg.norm(np.cross(d - b, c - b), axis=-1),
    ], axis=0)
    touched = np.stack([
        point_mask[:-1, :-1] | point_mask[1:, :-1] | point_mask[:-1, 1:],
        point_mask[1:, 1:] | point_mask[1:, :-1] | point_mask[:-1, 1:],
    ], axis=0)
    return float(triangle_areas[touched].sum())


def _grid_faces(rows: int, columns: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for row in range(max(0, int(rows) - 1)):
        for column in range(max(0, int(columns) - 1)):
            first = row * int(columns) + column
            second = first + 1
            third = first + int(columns)
            fourth = third + 1
            faces.extend(((first, third, second), (second, third, fourth)))
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _write_ply(path: Path, points: np.ndarray, *, faces: np.ndarray | None = None, color: tuple[int, int, int] = (160, 160, 160)) -> None:
    """Write a small ASCII PLY for direct inspection in a mesh viewer."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces if faces is not None else np.empty((0, 3), dtype=np.int64), dtype=np.int64).reshape(-1, 3)
    red, green, blue = (int(channel) for channel in color)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.8f} {y:.8f} {z:.8f} {red} {green} {blue}" for x, y, z in points)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_raw_controlled_overlay(
    holdout: Holdout,
    prediction: Prediction | None,
    output_path: Path,
    *,
    title: str,
    branch_predictions: dict[str, Prediction] | None = None,
) -> None:
    """Export one plain fixed-view overlay for quick advisor inspection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branch_predictions = branch_predictions or {}
    generated = [branch.points for branch in branch_predictions.values()]
    all_points = np.concatenate([
        holdout.full_points,
        holdout.withheld_points,
        prediction.points if prediction is not None else np.empty((0, 3)),
        *generated,
    ], axis=0)
    figure = plt.figure(figsize=(8, 7), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    _configure_axis(axis, all_points)
    _scatter3d(axis, holdout.retained_points, OBSERVED_GREY, size=1.1, alpha=0.48)
    _scatter3d(axis, holdout.withheld_points, HELDOUT_GREEN, size=1.2, alpha=0.72)
    if prediction is not None and prediction.status == "VALID":
        _scatter3d(axis, prediction.points, PREDICTED_CYAN, size=1.4, alpha=0.92)
    for branch in branch_predictions.values():
        _scatter3d(axis, branch.points, JUNCTION_MAGENTA, size=1.0, alpha=0.62)
    _scatter3d(axis, holdout.retained_points[np.argsort(holdout.retained_points @ holdout.u_axis)[-min(128, len(holdout.retained_points)):]], FRONTIER_YELLOW, size=6.0, alpha=0.95)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.set_zlabel("y")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_controlled_geometry(
    holdout: Holdout,
    prediction: Prediction | None,
    output_root: Path,
    *,
    branch_predictions: dict[str, Prediction] | None = None,
) -> dict[str, str]:
    """Write raw point/mesh artifacts, including evaluation-only reference rows."""

    branch_predictions = branch_predictions or {}
    output_root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "full_reference_points": holdout.full_points.astype(np.float32),
        "retained_visible_points": holdout.retained_points.astype(np.float32),
        "withheld_reference_points_evaluation_only": holdout.withheld_points.astype(np.float32),
        "frontier_points_retained_only": prediction.frontier_points.astype(np.float32) if prediction is not None else np.empty((0, 3), dtype=np.float32),
        "predicted_points": prediction.points.astype(np.float32) if prediction is not None and prediction.status == "VALID" else np.empty((0, 3), dtype=np.float32),
    }
    for name, branch in branch_predictions.items():
        arrays[f"{name}_points"] = branch.points.astype(np.float32)
    npz_path = output_root / "controlled_geometry.npz"
    np.savez_compressed(npz_path, **arrays)
    _write_ply(output_root / "retained_visible_points.ply", holdout.retained_points, color=(148, 153, 160))
    _write_ply(output_root / "withheld_reference_points_evaluation_only.ply", holdout.withheld_points, color=(26, 191, 82))
    if prediction is not None and prediction.status == "VALID":
        _write_ply(output_root / "predicted_continuation.ply", prediction.points_grid, faces=_grid_faces(*prediction.points_grid.shape[:2]), color=(0, 184, 209))
    for name, branch in branch_predictions.items():
        _write_ply(output_root / f"{name}.ply", branch.points_grid, faces=_grid_faces(*branch.points_grid.shape[:2]), color=(217, 13, 178))
    return {
        "npz": str(npz_path),
        "retained_points_ply": str(output_root / "retained_visible_points.ply"),
        "withheld_reference_points_ply": str(output_root / "withheld_reference_points_evaluation_only.ply"),
        "predicted_mesh_ply": str(output_root / "predicted_continuation.ply") if prediction is not None and prediction.status == "VALID" else None,
    }


def evaluate_controlled_case(holdout: Holdout, prediction: Prediction | None, *, h: float, free_space: FreeSpaceProxy | None = None) -> dict[str, Any]:
    """Evaluate only the held-out reconstructed visible-surface reference."""

    report: dict[str, Any] = {
        "evaluation_population": "held-out reconstructed visible-surface reference only",
        "withheld_reference_point_count": int(len(holdout.withheld_points)),
        "generated_surface_point_count": int(len(prediction.points)) if prediction is not None else 0,
        "generated_surface_triangle_count": int(prediction.triangle_count) if prediction is not None else 0,
        "generated_surface_area": _surface_area(prediction.points_grid) if prediction is not None else 0.0,
        "source_reference_used_for_prediction": False,
        "metric_fed_back_into_construction": False,
    }
    if prediction is None or not len(prediction.points):
        report.update({
            "status": "NO_GENERATED_SURFACE",
            "point_to_surface": {"median_over_h": None, "p95_over_h": None},
            "coverage": {"fraction_le_h": None, "fraction_le_2h": None},
            "normal_error": {"status": "unavailable"},
        })
        return report
    from scipy.spatial import cKDTree

    distances, nearest = cKDTree(prediction.points).query(holdout.withheld_points, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    reference_normals = estimate_point_normals(holdout.withheld_points, k=20)
    if reference_normals is None:
        normal_error = {"status": "unavailable"}
    else:
        matched = prediction.normals[nearest]
        cosine = np.clip(np.abs(np.sum(reference_normals * matched, axis=1)), 0.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        normal_error = {
            "status": "estimated_unoriented_pca_vs_ruled_patch",
            "median_degrees": float(np.median(angles)),
            "p95_degrees": float(np.percentile(angles, 95)),
        }
    free_violation = free_space.violation_mask(prediction.points) if free_space is not None else np.zeros((len(prediction.points),), dtype=bool)
    inside = holdout.permitted_volume.contains(prediction.points, tolerance=1e-8)
    frontier_reference_distances = cKDTree(holdout.retained_points).query(prediction.frontier_points, workers=1)[0]
    source_normals = estimate_point_normals(holdout.retained_points, k=20)
    if source_normals is None:
        boundary_normal = {"status": "unavailable"}
    else:
        source_nearest = cKDTree(holdout.retained_points).query(prediction.frontier_points, workers=1)[1]
        cosine = np.clip(np.abs(np.sum(source_normals[source_nearest] * prediction.normals_grid[0], axis=1)), 0.0, 1.0)
        boundary_angles = np.degrees(np.arccos(cosine))
        boundary_normal = {
            "status": "estimated_unoriented_pca_vs_generated",
            "median_degrees": float(np.median(boundary_angles)),
            "p95_degrees": float(np.percentile(boundary_angles, 95)),
        }
    outside = ~inside
    known_violation = free_violation | outside
    report.update({
        "status": "EVALUATED",
        "point_to_surface": {
            "median_over_h": float(np.median(distances) / h),
            "p95_over_h": float(np.percentile(distances, 95) / h),
        },
        "coverage": {
            "fraction_le_h": float(np.mean(distances <= h)),
            "fraction_le_2h": float(np.mean(distances <= 2.0 * h)),
        },
        "normal_error": normal_error,
        "known_free_space_violation_count": int(np.sum(known_violation)),
        "known_free_space_violation_area": _masked_grid_area(prediction.points_grid, known_violation),
        "explicit_free_proxy_violation_count": int(np.sum(free_violation)),
        "outside_supplied_pseudo_occluded_region_point_count": int(np.sum(~inside)),
        "outside_supplied_pseudo_occluded_region_fraction": float(np.mean(~inside)),
        "frontier_attachment_gap_over_h": float(np.median(frontier_reference_distances) / h),
        "frontier_attachment_gap_p95_over_h": float(np.percentile(frontier_reference_distances, 95) / h),
        "boundary_continuity": {
            "position_gap_over_h_median": float(np.median(frontier_reference_distances) / h),
            "position_gap_over_h_p95": float(np.percentile(frontier_reference_distances, 95) / h),
            "normal_angle": boundary_normal,
        },
    })
    return report


def _table_scene_mask(points: np.ndarray, box: Box) -> np.ndarray:
    return box.contains(points)


def _draw_box(axis: Any, box: Box, color: Any, *, linewidth: float = 1.0, alpha: float = 0.65) -> None:
    lower = np.asarray(box.lower, dtype=np.float64)
    upper = np.asarray(box.upper, dtype=np.float64)
    corners = np.asarray([
        [lower[0], lower[1], lower[2]], [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]], [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]], [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]], [lower[0], upper[1], upper[2]],
    ])
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    display = _plot_coords(corners)
    for first, second in edges:
        axis.plot(display[[first, second], 0], display[[first, second], 1], display[[first, second], 2], color=color, linewidth=linewidth, alpha=alpha)


def _draw_arrows(axis: Any, points: np.ndarray, vectors: np.ndarray, color: Any, *, count: int = 10, length: float = 0.25) -> None:
    if len(points) == 0:
        return
    indices = np.linspace(0, len(points) - 1, min(count, len(points)), dtype=np.int64)
    display_points = _plot_coords(points[indices])
    display_vectors = _plot_coords(points[indices] + length * vectors[indices]) - display_points
    axis.quiver(display_points[:, 0], display_points[:, 1], display_points[:, 2], display_vectors[:, 0], display_vectors[:, 1], display_vectors[:, 2], color=color, linewidth=0.8, arrow_length_ratio=0.22)


def _configure_axis(axis: Any, limits_points: np.ndarray) -> None:
    _set_equal_3d_limits(axis, limits_points)
    axis.view_init(elev=24, azim=-60)
    axis.set_title("")


def _local_coordinates(points: np.ndarray, holdout: Holdout) -> np.ndarray:
    """Express points in the holdout's readable local continuation frame."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    coordinates = np.column_stack([
        points @ holdout.u_axis - holdout.u_cut,
        points @ holdout.v_axis - 0.5 * (holdout.v_bounds[0] + holdout.v_bounds[1]),
        points @ holdout.n_axis - float(np.median(holdout.retained_points @ holdout.n_axis)),
    ])
    return coordinates


def _configure_local_axis(axis: Any, all_points: np.ndarray, holdout: Holdout, *, azim: float = -58.0, elev: float = 32.0) -> None:
    local = _local_coordinates(all_points, holdout)
    minimum = local.min(axis=0)
    maximum = local.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    padding = 0.06 * span
    axis.set_xlim(float(minimum[0] - padding[0]), float(maximum[0] + padding[0]))
    axis.set_ylim(float(minimum[1] - padding[1]), float(maximum[1] + padding[1]))
    axis.set_zlim(float(minimum[2] - padding[2]), float(maximum[2] + padding[2]))
    axis.set_box_aspect(tuple(np.maximum(span, 1e-6)))
    axis.view_init(elev=elev, azim=azim)
    axis.set_xlabel("u: continuation direction")
    axis.set_ylabel("v: along patch")
    axis.set_zlabel("n: surface height")


def _plot_local_cloud_surface(axis: Any, points: np.ndarray, holdout: Holdout, color: Any, *, alpha: float = 0.55, max_points: int = 1400) -> None:
    """Draw a readable low-density surface skin over a point cloud."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return
    local = _local_coordinates(deterministic_subsample(points, max_points), holdout)
    axis.scatter(local[:, 0], local[:, 1], local[:, 2], s=2.0, color=color, alpha=min(0.95, alpha + 0.15), depthshade=False)
    if len(local) < 8:
        return
    try:
        import matplotlib.tri as mtri

        triangulation = mtri.Triangulation(local[:, 0], local[:, 1])
        axis.plot_trisurf(
            local[:, 0],
            local[:, 1],
            local[:, 2],
            triangles=triangulation.triangles,
            color=color,
            alpha=alpha,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
    except (ValueError, RuntimeError):
        # The point cloud remains useful even when a degenerate local
        # triangulation cannot be formed.
        return


def _plot_local_grid_surface(axis: Any, grid: np.ndarray, holdout: Holdout, color: Any, *, alpha: float = 0.80) -> None:
    if grid is None or np.asarray(grid).ndim != 3 or np.asarray(grid).shape[0] < 2 or np.asarray(grid).shape[1] < 2:
        return
    local = _local_coordinates(np.asarray(grid).reshape(-1, 3), holdout).reshape(np.asarray(grid).shape)
    axis.plot_surface(
        local[:, :, 0],
        local[:, :, 1],
        local[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0.15,
        edgecolor=(0.10, 0.35, 0.40, 0.30),
        antialiased=True,
        shade=True,
    )


def _plot_local_frontier(axis: Any, frontier: np.ndarray, holdout: Holdout) -> None:
    if len(frontier) == 0:
        return
    local = _local_coordinates(frontier, holdout)
    order = np.argsort(local[:, 1], kind="mergesort")
    local = local[order]
    axis.plot(local[:, 0], local[:, 1], local[:, 2], color=FRONTIER_YELLOW, linewidth=2.4, marker="o", markersize=2.5)


def _plot_local_boundary(axis: Any, holdout: Holdout) -> None:
    n_center = float(np.median(holdout.retained_points @ holdout.n_axis))
    corners = np.asarray([
        [holdout.u_cut, holdout.v_bounds[0], holdout.n_bounds[0]],
        [holdout.u_cut, holdout.v_bounds[1], holdout.n_bounds[0]],
        [holdout.u_cut, holdout.v_bounds[1], holdout.n_bounds[1]],
        [holdout.u_cut, holdout.v_bounds[0], holdout.n_bounds[1]],
    ])
    world = corners[:, 0, None] * holdout.u_axis[None, :] + corners[:, 1, None] * holdout.v_axis[None, :] + corners[:, 2, None] * holdout.n_axis[None, :]
    local = _local_coordinates(world, holdout)
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        axis.plot(local[[first, second], 0], local[[first, second], 1], local[[first, second], 2], color=FRONTIER_YELLOW, linewidth=1.2, alpha=0.55)
    axis.plot([0.0, 0.0], [holdout.v_bounds[0] - 0.5 * sum(holdout.v_bounds), holdout.v_bounds[1] - 0.5 * sum(holdout.v_bounds)], [n_center - float(np.median(holdout.retained_points @ holdout.n_axis)), n_center - float(np.median(holdout.retained_points @ holdout.n_axis))], color=FRONTIER_YELLOW, linewidth=1.5)


def _plot_local_footprint(axis: Any, holdout: Holdout, prediction: Prediction | None, branches: dict[str, Prediction]) -> None:
    retained = _local_coordinates(holdout.retained_points, holdout)
    target = _local_coordinates(holdout.withheld_points, holdout)
    axis.scatter(retained[:, 0], retained[:, 1], s=1.5, color=OBSERVED_GREY, alpha=0.36, label="retained")
    axis.scatter(target[:, 0], target[:, 1], s=1.5, color=HELDOUT_GREEN, alpha=0.48, label="withheld reference")
    if prediction is not None and prediction.status == "VALID":
        local = _local_coordinates(prediction.points, holdout)
        axis.scatter(local[:, 0], local[:, 1], s=2.0, color=PREDICTED_CYAN, alpha=0.72, label="prediction")
    for branch in branches.values():
        local = _local_coordinates(branch.points, holdout)
        axis.scatter(local[:, 0], local[:, 1], s=1.6, color=JUNCTION_MAGENTA, alpha=0.48)
    axis.axvline(0.0, color=FRONTIER_YELLOW, linewidth=1.5)
    axis.set_xlabel("u from visible termination")
    axis.set_ylabel("v along patch")
    axis.set_title("Footprint: retained / withheld / predicted")
    axis.grid(alpha=0.22)


def _plot_local_profile(axis: Any, holdout: Holdout, prediction: Prediction | None, branches: dict[str, Prediction]) -> None:
    retained = _local_coordinates(holdout.retained_points, holdout)
    target = _local_coordinates(holdout.withheld_points, holdout)
    axis.scatter(retained[:, 0], retained[:, 2], s=1.5, color=OBSERVED_GREY, alpha=0.36)
    axis.scatter(target[:, 0], target[:, 2], s=1.5, color=HELDOUT_GREEN, alpha=0.48)
    if prediction is not None and prediction.status == "VALID":
        local = _local_coordinates(prediction.points, holdout)
        axis.scatter(local[:, 0], local[:, 2], s=2.0, color=PREDICTED_CYAN, alpha=0.72)
    for branch in branches.values():
        local = _local_coordinates(branch.points, holdout)
        axis.scatter(local[:, 0], local[:, 2], s=1.6, color=JUNCTION_MAGENTA, alpha=0.48)
    axis.axvline(0.0, color=FRONTIER_YELLOW, linewidth=1.5)
    axis.set_xlabel("u from visible termination")
    axis.set_ylabel("n: surface height")
    axis.set_title("Side profile: continuation shape")
    axis.grid(alpha=0.22)


def write_intuitive_controlled_figure(
    holdout: Holdout,
    prediction: Prediction | None,
    *,
    output_path: Path,
    title: str,
    selected_branches: dict[str, Prediction] | None = None,
) -> None:
    """Export a local surface view that is readable as geometry, not a line cloud."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = selected_branches or {}
    target = holdout.withheld_points
    generated = [branch.points for branch in branches.values()]
    all_points = np.concatenate([
        holdout.full_points,
        target,
        prediction.points if prediction is not None else np.empty((0, 3)),
        *generated,
    ], axis=0)
    figure = plt.figure(figsize=(22, 9), facecolor="white")
    axes = [figure.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]
    for axis in axes:
        _configure_local_axis(axis, all_points, holdout)

    _plot_local_cloud_surface(axes[0], holdout.retained_points, holdout, OBSERVED_GREY, alpha=0.32)
    _plot_local_cloud_surface(axes[0], target, holdout, HELDOUT_GREEN, alpha=0.58)
    _plot_local_frontier(axes[0], prediction.frontier_points if prediction is not None else np.empty((0, 3)), holdout)
    axes[0].set_title("A  Full reference\ngreen = withheld reference")

    _plot_local_cloud_surface(axes[1], holdout.retained_points, holdout, OBSERVED_GREY, alpha=0.42)
    _plot_local_frontier(axes[1], prediction.frontier_points if prediction is not None else np.empty((0, 3)), holdout)
    _plot_local_boundary(axes[1], holdout)
    axes[1].set_title("B  Visible-only input\nyellow = boundary / frontier")

    _plot_local_cloud_surface(axes[2], holdout.retained_points, holdout, OBSERVED_GREY, alpha=0.20)
    if prediction is not None and prediction.status == "VALID":
        _plot_local_grid_surface(axes[2], prediction.points_grid, holdout, PREDICTED_CYAN)
    for branch in branches.values():
        _plot_local_grid_surface(axes[2], branch.points_grid, holdout, JUNCTION_MAGENTA, alpha=0.54)
    axes[2].set_title("C  Continuation surface\ncyan = prediction, magenta = H2 branches")

    _plot_local_cloud_surface(axes[3], holdout.retained_points, holdout, OBSERVED_GREY, alpha=0.22)
    _plot_local_cloud_surface(axes[3], target, holdout, HELDOUT_GREEN, alpha=0.48)
    if prediction is not None and prediction.status == "VALID":
        _plot_local_grid_surface(axes[3], prediction.points_grid, holdout, PREDICTED_CYAN)
    for branch in branches.values():
        _plot_local_grid_surface(axes[3], branch.points_grid, holdout, JUNCTION_MAGENTA, alpha=0.46)
    _plot_local_frontier(axes[3], prediction.frontier_points if prediction is not None else np.empty((0, 3)), holdout)
    axes[3].set_title("D  Overlay\ngreen = withheld, cyan = generated")

    figure.suptitle(title + " — local geometry view", fontsize=16, y=0.98)
    figure.text(0.5, 0.01, "Local coordinates are a display transform; withheld reference is evaluation/overlay only.", ha="center", fontsize=10)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_intuitive_raw_overlay(
    holdout: Holdout,
    prediction: Prediction | None,
    output_path: Path,
    *,
    title: str,
    branch_predictions: dict[str, Prediction] | None = None,
) -> None:
    """Export compact raw 3D + footprint/profile inspection views."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = branch_predictions or {}
    target = holdout.withheld_points
    generated = [branch.points for branch in branches.values()]
    all_points = np.concatenate([
        holdout.full_points,
        target,
        prediction.points if prediction is not None else np.empty((0, 3)),
        *generated,
    ], axis=0)
    figure = plt.figure(figsize=(16, 7), facecolor="white")
    axis = figure.add_subplot(1, 2, 1, projection="3d")
    _configure_local_axis(axis, all_points, holdout, azim=-58.0, elev=32.0)
    _plot_local_cloud_surface(axis, holdout.retained_points, holdout, OBSERVED_GREY, alpha=0.28)
    _plot_local_cloud_surface(axis, target, holdout, HELDOUT_GREEN, alpha=0.52)
    if prediction is not None and prediction.status == "VALID":
        _plot_local_grid_surface(axis, prediction.points_grid, holdout, PREDICTED_CYAN)
    for branch in branches.values():
        _plot_local_grid_surface(axis, branch.points_grid, holdout, JUNCTION_MAGENTA, alpha=0.48)
    _plot_local_frontier(axis, prediction.frontier_points if prediction is not None else np.empty((0, 3)), holdout)
    axis.set_title("3D local surface")
    profile_axis = figure.add_subplot(2, 2, 2)
    _plot_local_footprint(profile_axis, holdout, prediction, branches)
    side_axis = figure.add_subplot(2, 2, 4)
    _plot_local_profile(side_axis, holdout, prediction, branches)
    figure.suptitle(title + " — raw inspectable geometry", fontsize=15, y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_controlled_figure(
    holdout: Holdout,
    prediction: Prediction | None,
    *,
    output_path: Path,
    title: str,
    template: dict[str, Any] | None = None,
    selected_branches: dict[str, Prediction] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = holdout.withheld_points
    frontier = prediction.frontier_points if prediction is not None else np.empty((0, 3))
    all_points = np.concatenate([holdout.full_points, target, frontier, prediction.points if prediction is not None else np.empty((0, 3))], axis=0)
    figure = plt.figure(figsize=(22, 9), facecolor="white")
    axes = [figure.add_subplot(1, 5, index + 1, projection="3d") for index in range(5)]
    for axis in axes:
        _configure_axis(axis, all_points)
    _scatter3d(axes[0], holdout.full_points, OBSERVED_GREY, size=1.0, alpha=0.55)
    _scatter3d(axes[0], target, HELDOUT_GREEN, size=1.2, alpha=0.85)
    axes[0].set_title("Original reconstructed reference\nheld-out reference in green")
    _scatter3d(axes[1], holdout.retained_points, OBSERVED_GREY, size=1.0, alpha=0.60)
    _draw_box(axes[1], holdout.permitted_volume, PSEUDO_ORANGE, linewidth=1.4)
    axes[1].set_title("Pseudo-occluded input\nretained surface + permitted volume")
    _scatter3d(axes[2], holdout.retained_points, OBSERVED_GREY, size=1.0, alpha=0.48)
    _scatter3d(axes[2], frontier, FRONTIER_YELLOW, size=10.0, alpha=0.95)
    if len(frontier):
        _draw_arrows(axes[2], frontier, np.tile(prediction.frontier_tangent[None, :], (len(frontier), 1)) if prediction is not None else np.zeros_like(frontier), NORMAL_BLUE, length=0.18)
    if prediction is not None:
        _draw_arrows(axes[2], frontier, np.tile(prediction.continuation_direction[None, :], (len(frontier), 1)), JUNCTION_MAGENTA if template else NORMAL_BLUE, length=0.25)
    if template is not None:
        axes[2].text2D(0.03, 0.04, f"observed theta = {template.get('theta_visible_degrees', float('nan')):.1f} deg", transform=axes[2].transAxes, fontsize=9)
    axes[2].set_title("Observed frontier / pattern\nfrontier yellow, directions blue/magenta")
    if prediction is not None and prediction.status == "VALID":
        _scatter3d(axes[3], prediction.points, PREDICTED_CYAN, size=1.2, alpha=0.90)
    if selected_branches:
        for index, branch in enumerate(selected_branches.values()):
            _scatter3d(axes[3], branch.points, PREDICTED_CYAN if index == 0 else JUNCTION_MAGENTA, size=1.0, alpha=0.64)
    axes[3].set_title("Completion prediction only\ncyan = generated geometry")
    _scatter3d(axes[4], holdout.retained_points, OBSERVED_GREY, size=0.8, alpha=0.35)
    if prediction is not None and prediction.status == "VALID":
        _scatter3d(axes[4], prediction.points, PREDICTED_CYAN, size=1.3, alpha=0.85)
    _scatter3d(axes[4], target, HELDOUT_GREEN, size=1.0, alpha=0.80)
    axes[4].set_title("Prediction vs held-out reference\ncyan vs green")
    figure.suptitle(title, fontsize=18, y=0.99)
    figure.text(0.5, 0.01, "Held-out reconstructed visible-surface reference — not physical ground truth", ha="center", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_scene_overview(table_points: np.ndarray, top: PatchStats, side: PatchStats, leg: PatchStats, candidate_b_points: np.ndarray, free_points: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = np.concatenate([table_points, top.points, side.points, leg.points, candidate_b_points, free_points], axis=0)
    figure = plt.figure(figsize=(12, 9), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    _configure_axis(axis, all_points)
    _scatter3d(axis, table_points, OBSERVED_GREY, size=0.35, alpha=0.20)
    _scatter3d(axis, top.points, TOP_GREEN, size=1.3, alpha=0.75)
    _scatter3d(axis, side.points, SIDE_BLUE, size=1.3, alpha=0.75)
    _scatter3d(axis, leg.points, LEG_MAGENTA, size=1.5, alpha=0.80)
    if len(candidate_b_points):
        _scatter3d(axis, candidate_b_points, TRUE_OCCLUDED_ORANGE, size=8.0, alpha=0.95)
    if len(free_points):
        _scatter3d(axis, free_points, FREE_RED, size=5.0, alpha=0.45)
    _draw_box(axis, TABLE_DEMO.true_rim_volume, TRUE_OCCLUDED_ORANGE, linewidth=1.2, alpha=0.7)
    _draw_box(axis, TABLE_DEMO.true_leg_volume, TRUE_OCCLUDED_ORANGE, linewidth=1.2, alpha=0.7)
    axis.set_title("Worklog 134 table demo configuration\ngray visible mesh | green top-like | blue side/rim | magenta leg | orange Candidate-B occluded queries | red front-of-surface free proxy")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _candidate_b_evidence(path: Path, *, rim_volume: Box, leg_volume: Box) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), {"status": "UNAVAILABLE"}
    archive = np.load(path, allow_pickle=True)
    positions = np.asarray(archive["positions"], dtype=np.float64)
    global_b = np.asarray(archive["global_B"], dtype=np.int8)
    kinds = np.asarray(archive["kind"])
    occluded_mask = global_b == 2
    true_volume_mask = rim_volume.contains(positions) | leg_volume.contains(positions)
    occluded = positions[occluded_mask & true_volume_mask]
    # Only R4 is used as an explicit known front-of-surface free-space proxy.
    free_mask = (kinds == "R4_FRONT_OF_SURFACE_PROBE") & (global_b == 1) & true_volume_mask
    free = positions[free_mask]
    return occluded, free, {
        "status": "AVAILABLE",
        "candidate_b_occluded_query_count": int(len(occluded)),
        "explicit_front_of_surface_free_proxy_query_count": int(len(free)),
        "occluded_state_definition": "global_B == 2 only; not every OBSERVED Candidate-B query is treated as free space",
        "free_state_definition": "R4_FRONT_OF_SURFACE_PROBE with global_B == 1 only",
        "used_as_hidden_reference": False,
    }


def _true_diagnostics(predictions: list[tuple[str, Prediction | None]], *, volumes: list[Box], candidate_b_points: np.ndarray, free_proxy: FreeSpaceProxy, h: float) -> dict[str, Any]:
    all_points = [prediction.points for _name, prediction in predictions if prediction is not None and len(prediction.points)]
    if not all_points:
        return {"status": "NO_GENERATED_SURFACE", "generated_area": 0.0}
    combined = np.concatenate(all_points, axis=0)
    inside = np.zeros((len(combined),), dtype=bool)
    for volume in volumes:
        inside |= volume.contains(combined, tolerance=1e-8)
    free_violation = free_proxy.violation_mask(combined)
    if len(candidate_b_points):
        from scipy.spatial import cKDTree

        occupancy = cKDTree(candidate_b_points).query(combined, workers=1)[0] <= 4.0 * h
    else:
        occupancy = np.zeros((len(combined),), dtype=bool)
    area = sum(_surface_area(prediction.points_grid) for _name, prediction in predictions if prediction is not None)
    attachment = []
    for _name, prediction in predictions:
        if prediction is not None and len(prediction.frontier_points):
            attachment.append(float(np.linalg.norm(prediction.points_grid[0] - prediction.frontier_points, axis=1).mean() / h))
    return {
        "status": "EVALUATED_GEOMETRICALLY",
        "generated_point_count": int(len(combined)),
        "generated_area": float(area),
        "known_free_space_violation_count": int(np.sum(free_violation)),
        "known_free_space_violation_fraction": float(np.mean(free_violation)),
        "outside_allowed_occluded_volume_point_count": int(np.sum(~inside)),
        "outside_allowed_occluded_volume_fraction": float(np.mean(~inside)),
        "candidate_b_occluded_occupancy_fraction_within_4h": float(np.mean(occupancy)) if len(occupancy) else 0.0,
        "frontier_attachment_gap_over_h_mean": float(np.mean(attachment)) if attachment else None,
        "closure_tolerance_over_h": 2.0,
        "closure": "OPEN_COMPLETION",
        "closure_endpoint_gap_over_h": None,
    }


def write_true_figure(visible: np.ndarray, candidate_b: np.ndarray, free: np.ndarray, rim_prediction: Prediction | None, leg_prediction: Prediction | None, output_path: Path, *, novel: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated = [p.points for p in (rim_prediction, leg_prediction) if p is not None]
    all_points = np.concatenate([visible, candidate_b, free] + generated, axis=0)
    figure = plt.figure(figsize=(22, 9), facecolor="white")
    axes = [figure.add_subplot(1, 5, index + 1, projection="3d") for index in range(5)]
    for axis in axes:
        _configure_axis(axis, all_points)
    _scatter3d(axes[0], visible, OBSERVED_GREY, size=0.55, alpha=0.40)
    axes[0].set_title("Visible table surface")
    _scatter3d(axes[1], visible, OBSERVED_GREY, size=0.45, alpha=0.25)
    _scatter3d(axes[1], candidate_b, TRUE_OCCLUDED_ORANGE, size=9.0, alpha=0.95)
    _scatter3d(axes[1], free, FREE_RED, size=6.0, alpha=0.70)
    axes[1].set_title("Candidate-B occluded queries\nred = explicit R4 free proxy")
    if rim_prediction is not None:
        _scatter3d(axes[2], rim_prediction.frontier_points, FRONTIER_YELLOW, size=10.0, alpha=0.95)
        _draw_arrows(axes[2], rim_prediction.frontier_points, np.tile(rim_prediction.continuation_direction[None, :], (len(rim_prediction.frontier_points), 1)), JUNCTION_MAGENTA, length=0.28)
    _scatter3d(axes[2], visible, OBSERVED_GREY, size=0.40, alpha=0.25)
    axes[2].set_title("Observed top-side template\nfrontier yellow, transfer direction magenta")
    for prediction in (rim_prediction, leg_prediction):
        if prediction is not None:
            _scatter3d(axes[3], prediction.points, PREDICTED_CYAN, size=1.5, alpha=0.90)
    axes[3].set_title("Predicted occluded surface\ncyan = heuristic candidate")
    _scatter3d(axes[4], visible, OBSERVED_GREY, size=0.40, alpha=0.25)
    for prediction in (rim_prediction, leg_prediction):
        if prediction is not None:
            _scatter3d(axes[4], prediction.points, PREDICTED_CYAN, size=1.5, alpha=0.90)
    axes[4].set_title("Visible + predicted combined")
    figure.suptitle("Feasibility prototype — heuristic continuation, not the final method" + (" — second view" if novel else ""), fontsize=18, y=0.99)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_report_case(holdout: Holdout, prediction: Prediction | None, evaluation: dict[str, Any], *, mechanism: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value = {
        "holdout": holdout.as_json(),
        "mechanism": mechanism,
        "prediction": {
            "status": prediction.status if prediction is not None else "NONE",
            "point_count": int(len(prediction.points)) if prediction is not None else 0,
            "triangle_count": int(prediction.triangle_count) if prediction is not None else 0,
            "surface_area": _surface_area(prediction.points_grid) if prediction is not None else 0.0,
            "extent_world": float(prediction.l_values[-1]) if prediction is not None and len(prediction.l_values) else 0.0,
            "source_reference_used": False,
            "second_order_taylor_used": False,
        },
        "evaluation": evaluation,
    }
    if extra:
        value.update(extra)
    return value


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    field = np.load(arguments.field_cache, allow_pickle=True)
    h = float(field["h"])
    mesh = np.load(arguments.mesh_cache, allow_pickle=True)
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    config = TABLE_DEMO
    table_points = _select_box(vertices, config.table_scene_box, int(arguments.max_scene_points))
    top_points = _select_box(vertices, config.top_patch_box, int(arguments.max_patch_points))
    side_points = _select_box(vertices, config.side_patch_box, int(arguments.max_patch_points))
    leg_points = _select_box(vertices, config.leg_patch_box, int(arguments.max_patch_points))
    top_stats = measure_patch("observed_top_like", top_points)
    side_stats = measure_patch("observed_side_like", side_points)
    leg_stats = measure_patch("observed_leg_brace", leg_points)
    junction = measure_junction_relation(top_stats, side_stats)
    candidate_b_points, free_points, candidate_b_report = _candidate_b_evidence(Path(arguments.candidate_b_archive), rim_volume=config.true_rim_volume, leg_volume=config.true_leg_volume)
    write_scene_overview(table_points, top_stats, side_stats, leg_stats, candidate_b_points, free_points, output_root / "table_demo_scene_overview.png")
    scene_geometry_path = output_root / "table_demo_geometry.npz"
    np.savez_compressed(
        scene_geometry_path,
        table_visible_points=table_points.astype(np.float32),
        top_like_patch_points=top_points.astype(np.float32),
        side_like_patch_points=side_points.astype(np.float32),
        leg_brace_patch_points=leg_points.astype(np.float32),
        candidate_b_occluded_query_points=candidate_b_points.astype(np.float32),
        candidate_b_explicit_free_proxy_points=free_points.astype(np.float32),
    )
    _write_ply(output_root / "table_visible_points.ply", table_points, color=(148, 153, 160))
    _write_ply(output_root / "top_like_patch_points.ply", top_points, color=(56, 158, 64))
    _write_ply(output_root / "side_like_patch_points.ply", side_points, color=(31, 102, 217))
    _write_ply(output_root / "leg_brace_patch_points.ply", leg_points, color=(209, 46, 178))

    side_holdout = build_fixed_holdout(
        side_points,
        name="H1_side_self_continuation",
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(0.0, 1.0, 0.0),
        u_bounds=config.h1_side_u_extent,
        v_bounds=config.h1_side_v_extent,
        n_bounds=config.h1_side_n_extent,
        u_cut=config.h1_side_u_cut,
        permitted_volume=config.h1_pseudo_volume,
    )
    h1_prediction = build_self_continuation(side_holdout, frontier_band_fraction=config.h1_frontier_band_fraction, frontier_bins=config.frontier_bins, continuation_samples=config.continuation_samples)
    h1_evaluation = evaluate_controlled_case(side_holdout, h1_prediction, h=h)
    h1_raw_overlay = output_root / "H1_self_continuation" / "raw_fixed_view_overlay.png"
    write_intuitive_raw_overlay(side_holdout, h1_prediction, h1_raw_overlay, title="H1 fixed-view holdout overlay")
    h1_geometry = write_controlled_geometry(side_holdout, h1_prediction, output_root / "H1_self_continuation")
    write_intuitive_controlled_figure(side_holdout, h1_prediction, output_path=output_root / "H1_self_continuation" / "controlled_pseudo_occlusion.png", title="Controlled pseudo-occlusion — H1 self-continuation")

    h2_prediction, h2_selection = build_junction_transfer(
        side_holdout,
        source_top=top_stats,
        source_side=side_stats,
        source_frontier=h1_prediction.frontier_points,
        source_tangent=h1_prediction.frontier_tangent,
        source_normal=side_stats.normal,
        free_space=None,
        extent_fraction=0.42,
        continuation_samples=config.continuation_samples,
    )
    h2_evaluation = evaluate_controlled_case(side_holdout, h2_prediction, h=h)
    h2_branches = {}
    if h2_selection.get("selection", {}).get("branches"):
        for name, item in h2_selection["selection"]["branches"].items():
            h2_branches[name] = item[0]
    h2_raw_overlay = output_root / "H2_junction_transfer" / "raw_fixed_view_overlay.png"
    write_intuitive_raw_overlay(side_holdout, h2_prediction, h2_raw_overlay, title="H2 fixed-view holdout overlay", branch_predictions=h2_branches)
    h2_geometry = write_controlled_geometry(side_holdout, h2_prediction, output_root / "H2_junction_transfer", branch_predictions=h2_branches)
    write_intuitive_controlled_figure(side_holdout, h2_prediction, output_path=output_root / "H2_junction_transfer" / "controlled_pseudo_occlusion.png", title="Controlled pseudo-occlusion — H2 observed junction-pattern transfer", selected_branches=h2_branches)

    # This is a qualitative gate for the human meeting review.  It records
    # fixed h-scaled diagnostics and never changes a construction parameter.
    h1_gate = {
        "prediction_nonzero": bool(h1_prediction is not None and len(h1_prediction.points) > 0),
        "zero_free_space_violations": int(h1_evaluation.get("known_free_space_violation_count", 0)) == 0,
        "inside_pseudo_volume": int(h1_evaluation.get("outside_supplied_pseudo_occluded_region_point_count", 1)) == 0,
        "nonzero_extent": bool(h1_prediction is not None and h1_prediction.l_values[-1] > 0.0),
        "manual_qualitative_review_required": True,
    }
    h2_gate = {
        "branch_status": h2_selection["status"],
        "prediction_nonzero": bool(h2_prediction is not None and len(h2_prediction.points) > 0),
        "zero_free_space_violations": int(h2_evaluation.get("known_free_space_violation_count", 0)) == 0,
        "inside_pseudo_volume": int(h2_evaluation.get("outside_supplied_pseudo_occluded_region_point_count", 1)) == 0,
        "nonzero_extent": bool(h2_prediction is not None and h2_prediction.l_values[-1] > 0.0),
        "manual_qualitative_review_required": True,
    }
    if arguments.execute_true_occluded and arguments.controlled_gate != "positive":
        raise ValueError("--execute-true-occluded requires an explicit positive controlled gate")
    gate_status = {
        "review": "MANUAL_REVIEW_REQUIRED",
        "positive": "CONTROLLED_HOLDOUT_PASSED_MANUAL_REVIEW",
        "negative": "CONTROLLED_HOLDOUT_FAILS",
    }[arguments.controlled_gate]
    controlled_gate = {
        "status": gate_status,
        "rule": "At least one controlled case must be qualitatively useful, non-catastrophic, attached, nonzero-length, and free-space safe; no fitted numeric success threshold is invented.",
        "H1": h1_gate,
        "H2": h2_gate,
        "manual_selection": arguments.controlled_gate,
        "manual_selection_reason": "selected after direct inspection of the fixed raw overlay and fixed withheld-only metrics; no construction parameter was changed",
        "true_occluded_execution_requested": bool(arguments.execute_true_occluded),
    }

    true_result: dict[str, Any] = {"status": "NOT_EXECUTED", "reason": "conditional prototype is run only after controlled qualitative gate review"}
    if arguments.execute_true_occluded:
        true_rim_holdout = build_fixed_holdout(
            side_points,
            name="TRUE_RIM_VISIBLE_FRONTIER",
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 0.0, 1.0),
            n_axis=(0.0, 1.0, 0.0),
            u_bounds=config.h1_side_u_extent,
            v_bounds=config.h1_side_v_extent,
            n_bounds=config.h1_side_n_extent,
            u_cut=config.h1_side_u_cut,
            permitted_volume=config.true_rim_volume,
        )
        true_leg_holdout = build_fixed_holdout(
            leg_points,
            name="TRUE_LEG_VISIBLE_FRONTIER",
            u_axis=(0.0, 1.0, 0.0),
            v_axis=(0.0, 0.0, 1.0),
            n_axis=(1.0, 0.0, 0.0),
            u_bounds=(0.50, 1.10),
            v_bounds=(0.70, 1.20),
            n_bounds=(-0.30, 0.40),
            u_cut=0.848,
            permitted_volume=config.true_leg_volume,
        )
        free_proxy = FreeSpaceProxy("Candidate-B R4 front-of-surface proxy", free_points, radius=2.0 * h)
        true_leg_prediction = build_self_continuation(true_leg_holdout, frontier_band_fraction=0.10, frontier_bins=24, continuation_samples=24)
        true_rim_prediction, true_rim_selection = build_junction_transfer(
            true_rim_holdout,
            source_top=top_stats,
            source_side=side_stats,
            source_frontier=h1_prediction.frontier_points,
            source_tangent=h1_prediction.frontier_tangent,
            source_normal=side_stats.normal,
            free_space=free_proxy,
            extent_fraction=0.42,
            continuation_samples=24,
        )
        diagnostics = _true_diagnostics(
            [("rim_junction_transfer", true_rim_prediction), ("leg_self_continuation", true_leg_prediction)],
            volumes=[config.true_rim_volume, config.true_leg_volume],
            candidate_b_points=candidate_b_points,
            free_proxy=free_proxy,
            h=h,
        )
        write_true_figure(table_points, candidate_b_points, free_points, true_rim_prediction, true_leg_prediction, output_root / "figure_B_true_occluded_surface_feasibility.png")
        write_true_figure(table_points, candidate_b_points, free_points, true_rim_prediction, true_leg_prediction, output_root / "true_occluded_novel_view.png", novel=True)
        true_result = {
            "status": "EXECUTED",
            "region": "fixed under-table rim + leg/brace local boxes",
            "why_occluded": "Candidate-B global_B == 2 query evidence in the fixed local boxes; no hidden reference asserted",
            "candidate_b_evidence": candidate_b_report,
            "junction_transfer": _serializable_transfer_report(true_rim_selection),
            "predicted_surfaces": {
                "rim_junction_transfer": {
                    "point_count": int(len(true_rim_prediction.points)) if true_rim_prediction is not None else 0,
                    "area": _surface_area(true_rim_prediction.points_grid) if true_rim_prediction is not None else 0.0,
                },
                "leg_self_continuation": {"point_count": int(len(true_leg_prediction.points)), "area": _surface_area(true_leg_prediction.points_grid)},
            },
            "geometric_diagnostics": diagnostics,
            "no_hidden_ground_truth_metrics": True,
        }

    report = {
        "batch": "Worklog 134 meeting feasibility demo: occlusion-bounded surface continuation with controlled pseudo-occlusion",
        "status": "NON_CANONICAL_MEETING_DEMO",
        "intent_alignment": {
            "visible_surface_source": "frozen WL127 reconstructed Visible Surface mesh",
            "completion_prior": ["self-continuation", "observed junction-pattern transfer", "geometric closure only if surfaces meet"],
            "symmetry_used": False,
            "canonical_production_modified": False,
            "historical_worklogs_127_to_133_modified": False,
            "second_order_or_third_order_continuation_used": False,
            "true_occluded_scene_wide_completion": False,
        },
        "implementation_fidelity": {
            "manual_choices": "one fixed table crop, top-like/side-like/leg boxes, H1/H2 cuts, pseudo volumes, true Candidate-B local boxes",
            "heuristics": "PCA local normal/tangent, ruled linear strip, Rodrigues +/- observed-angle branches, deterministic nearest Candidate-B proxy checks",
            "withheld_reference_roles": ["evaluation target", "final overlay visualization", "quantitative error only"],
            "withheld_xyz_entered_prediction": False,
            "withheld_normals_entered_prediction": False,
            "withheld_endpoint_or_distance_entered_prediction": False,
            "branch_selection_uses_withheld_reference": False,
            "bridge_endpoint_uses_withheld_reference": False,
            "observed_free_space_definition": "controlled: outside supplied pseudo volume; true: R4_FRONT_OF_SURFACE_PROBE Candidate-B evidence only",
            "held_out_reference_name": "held-out reconstructed visible-surface reference",
        },
        "manual_demo_configuration": {
            "config": config.as_json(),
            "candidate_b_archive": str(arguments.candidate_b_archive),
            "candidate_b_evidence": candidate_b_report,
            "h": h,
            "mu": float(field["mu"]),
        },
        "observed_table_surface_patterns": {
            "top_like": _patch_stats_report(top_stats),
            "side_like": _patch_stats_report(side_stats),
            "leg_brace": _patch_stats_report(leg_stats),
        },
        "measured_top_side_junction_angle": junction,
        "H1_self_continuation_controlled_holdout": _build_report_case(side_holdout, h1_prediction, h1_evaluation, mechanism="retained visible frontier + local tangent ruled self-continuation", extra={"gate_inputs": h1_gate}),
        "H2_junction_transfer_controlled_holdout": _build_report_case(side_holdout, h2_prediction, h2_evaluation, mechanism="retained observed top-side theta transferred around target frontier tangent", extra={"junction_transfer": _serializable_transfer_report(h2_selection), "gate_inputs": h2_gate}),
        "known_free_space_violation_accounting": {
            "controlled_definition": "outside manually supplied pseudo-occluded volume is vetoed",
            "true_definition": candidate_b_report.get("free_state_definition"),
            "every_candidate_b_observed_query_treated_as_free": False,
        },
        "controlled_holdout_quantitative_result": {
            "H1": h1_evaluation,
            "H2": h2_evaluation,
        },
        "controlled_holdout_qualitative_result": {
            "H1": "review controlled_pseudo_occlusion.png; self-continuation is fixed and nonzero",
            "H2": h2_selection["status"],
        },
        "controlled_feasibility_gate": controlled_gate,
        "true_occluded_table_prototype": true_result,
        "meeting_figure_exports": {
            "scene_overview": str(output_root / "table_demo_scene_overview.png"),
            "scene_geometry_npz": str(scene_geometry_path),
            "scene_geometry_ply": [
                str(output_root / "table_visible_points.ply"),
                str(output_root / "top_like_patch_points.ply"),
                str(output_root / "side_like_patch_points.ply"),
                str(output_root / "leg_brace_patch_points.ply"),
            ],
            "H1_raw_fixed_view_overlay": str(h1_raw_overlay),
            "H1_geometry": h1_geometry,
            "H2_raw_fixed_view_overlay": str(h2_raw_overlay),
            "H2_geometry": h2_geometry,
            "figure_A_controlled": str(output_root / "H1_self_continuation" / "controlled_pseudo_occlusion.png"),
            "figure_B_true_occluded": str(output_root / "figure_B_true_occluded_surface_feasibility.png") if arguments.execute_true_occluded else None,
            "true_novel_view": str(output_root / "true_occluded_novel_view.png") if arguments.execute_true_occluded else None,
        },
        "meeting_verdict": {"review": "D", "positive": "D", "negative": "C"}[arguments.controlled_gate],
        "meeting_verdict_description": "D. DEMO INCONCLUSIVE — controlled results require qualitative review before conditional execution.",
    }
    report["meeting_verdict_description"] = {
        "review": "D. INCONCLUSIVE: controlled results require qualitative review before conditional execution.",
        "positive": "D. INCONCLUSIVE: controlled review passed, but this run did not complete the conditional true-occluded prototype.",
        "negative": "C. CONTROLLED HOLDOUT FAILS: the fixed non-planar continuation did not recover the withheld reconstructed visible-surface reference, so the conditional true-occluded prototype was not executed.",
    }[arguments.controlled_gate]
    output_root.joinpath("meeting_occluded_surface_feasibility_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    output_root.joinpath("README.md").write_text(
        "# Worklog 134 — meeting occluded-surface feasibility demo\n\n"
        f"Meeting verdict: `{report['meeting_verdict']}`\n\n"
        "The controlled H1/H2 results are separate from canonical OSN-GS.\n"
        "Held-out reconstructed visible-surface reference is evaluation-only.\n"
        "controlled_pseudo_occlusion.png is a local u/v/n surface view; raw_fixed_view_overlay.png includes 3D, footprint, and side-profile views.\n"
        "NPZ/PLY geometry files are emitted beside each case for direct inspection.\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-cache", type=Path, default=WORKLOG_127_MESH)
    parser.add_argument("--field-cache", type=Path, default=WORKLOG_127_FIELD)
    parser.add_argument("--candidate-b-archive", type=Path, default=CANDIDATE_B_ARCHIVE)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/meeting_occluded_surface_feasibility")
    parser.add_argument("--max-scene-points", type=int, default=18000)
    parser.add_argument("--max-patch-points", type=int, default=24000)
    parser.add_argument("--controlled-gate", choices=("review", "positive", "negative"), default="review")
    parser.add_argument("--execute-true-occluded", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_demo(build_arg_parser().parse_args(argv))
    print(json.dumps({"meeting_verdict": report["meeting_verdict"], "H1": report["H1_self_continuation_controlled_holdout"]["evaluation"]["status"], "H2": report["H2_junction_transfer_controlled_holdout"]["junction_transfer"]["status"], "true_occluded": report["true_occluded_table_prototype"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
