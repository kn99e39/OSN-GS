"""Worklog 136: semantically aligned occluded-surface feasibility demo.

This is an isolated, non-canonical meeting-demo track.  It replays the frozen
Worklog 127 Visible Surface mesh and tests two explicit real-scene semantics:

* H1: first-order self-continuation on a manually inspected table leg/brace;
* H2: transfer of an observed tabletop/side junction angle to a different
  visible tabletop/side segment with a one-sided target holdout.

The continuation primitives are deliberately small and deterministic.  The
construction functions accept retained geometry and fixed constraints only;
withheld reference points are passed to evaluation and visualization after
construction.  No canonical module, renderer, checkpoint, Candidate B
archive, or historical output is modified.
"""

from __future__ import annotations

import argparse
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
    FreeSpaceProxy,
    Holdout,
    PatchStats,
    Prediction,
    _branch_prediction,
    _grid_faces,
    _jsonable,
    _select_box,
    _surface_area,
    build_fixed_holdout,
    build_self_continuation,
    evaluate_controlled_case,
    validate_branch,
    _write_ply,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    _plot_coords,
    _scatter3d,
    deterministic_subsample,
)


WORKLOG_127_MESH = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz"
WORKLOG_127_FIELD = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz"
CANDIDATE_B_ARCHIVE = REPO_ROOT / "output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"

OBSERVED_GREY = (0.58, 0.60, 0.63)
WITHHELD_GREEN = (0.10, 0.75, 0.32)
PREDICTED_CYAN = (0.00, 0.72, 0.82)
PREDICTED_RED = (0.90, 0.08, 0.10)
FRONTIER_YELLOW = (0.98, 0.80, 0.08)
SOURCE_TOP_GREEN = (0.18, 0.68, 0.24)
SOURCE_SIDE_BLUE = (0.10, 0.36, 0.88)
TARGET_TOP_ORANGE = (0.95, 0.48, 0.08)
TARGET_SIDE_RED = (0.88, 0.12, 0.12)
CANDIDATE_B_ORANGE = (0.98, 0.45, 0.08)
REJECTED_MAGENTA = (0.78, 0.05, 0.72)

FIXED_VIEW = {"elev": 25.0, "azim": -60.0}
SECOND_VIEW = {"elev": 12.0, "azim": 28.0}
H2_EXTENT_FRACTION = 0.50
DISPLAY_VOXEL_WORLD = 0.02


@dataclass(frozen=True)
class SemanticDemoConfig:
    """Manual semantic seeds fixed before any continuation evaluation."""

    table_scene_box: Box
    leg_box: Box
    leg_u_bounds: tuple[float, float]
    leg_v_bounds: tuple[float, float]
    leg_n_bounds: tuple[float, float]
    leg_u_cut: float
    leg_permitted_volume: Box
    source_top_box: Box
    source_side_box: Box
    target_top_box: Box
    target_side_box: Box
    target_u_bounds: tuple[float, float]
    target_v_bounds: tuple[float, float]
    target_n_bounds: tuple[float, float]
    target_u_cut: float
    target_permitted_volume: Box
    frontier_band_fraction: float = 0.10
    frontier_bins: int = 24
    continuation_samples: int = 32

    def as_json(self) -> dict[str, Any]:
        return {
            "table_scene_box": self.table_scene_box.as_json(),
            "leg_box": self.leg_box.as_json(),
            "leg_coordinate_system": {
                "u_axis_world": [0.0, 1.0, 0.0],
                "v_axis_world": [0.0, 0.0, 1.0],
                "n_axis_world": [1.0, 0.0, 0.0],
                "u_bounds": list(self.leg_u_bounds),
                "v_bounds": list(self.leg_v_bounds),
                "n_bounds": list(self.leg_n_bounds),
                "u_cut": self.leg_u_cut,
            },
            "leg_permitted_volume": self.leg_permitted_volume.as_json(),
            "source_top_box": self.source_top_box.as_json(),
            "source_side_box": self.source_side_box.as_json(),
            "target_top_box": self.target_top_box.as_json(),
            "target_side_box": self.target_side_box.as_json(),
            "target_coordinate_system": {
                "u_axis_world": [1.0, 0.0, 0.0],
                "v_axis_world": [0.0, 1.0, 0.0],
                "n_axis_world": [0.0, 0.0, 1.0],
                "u_bounds": list(self.target_u_bounds),
                "v_bounds": list(self.target_v_bounds),
                "n_bounds": list(self.target_n_bounds),
                "u_cut": self.target_u_cut,
            },
            "target_permitted_volume": self.target_permitted_volume.as_json(),
            "frontier_band_fraction": self.frontier_band_fraction,
            "frontier_bins": self.frontier_bins,
            "continuation_samples": self.continuation_samples,
            "h2_extent_fraction": H2_EXTENT_FRACTION,
        }


# These values are operational demo choices made from direct inspection of the
# frozen mesh.  They are intentionally not estimated from withheld error.
SEMANTIC_CONFIG = SemanticDemoConfig(
    table_scene_box=Box((-12.0, -1.0, -10.0), (20.0, 2.0, 6.0)),
    # Existing Worklog 128 thin_table_leg_brace location, tightened only to
    # the directly inspected visible brace crop.
    leg_box=Box((-0.30, 0.48, 0.70), (0.40, 1.08, 1.35)),
    leg_u_bounds=(0.48, 1.08),
    leg_v_bounds=(0.70, 1.35),
    leg_n_bounds=(-0.30, 0.40),
    leg_u_cut=0.75,
    leg_permitted_volume=Box((-0.45, 0.42, 0.64), (0.55, 1.14, 1.42)),
    # A visibly planar tabletop strip and the adjacent non-planar/rim strip.
    source_top_box=Box((-9.0, 1.0, 2.0), (-7.0, 1.35, 4.0)),
    source_side_box=Box((-9.0, 1.0, 3.95), (-7.0, 1.35, 4.50)),
    # A different left-hand segment of the same real tabletop/side structure.
    target_top_box=Box((-11.0, 1.0, 2.0), (-9.0, 1.35, 4.0)),
    target_side_box=Box((-11.0, 1.0, 3.95), (-9.0, 1.35, 4.50)),
    target_u_bounds=(-10.0, -9.0),
    target_v_bounds=(1.0, 1.35),
    target_n_bounds=(3.95, 4.50),
    target_u_cut=-9.50,
    target_permitted_volume=Box((-10.05, 0.95, 3.90), (-8.95, 1.42, 4.55)),
)


def _patch_report(stats: PatchStats) -> dict[str, Any]:
    return {
        "label": stats.label,
        "point_count": int(len(stats.points)),
        "centroid_xyz": stats.centroid,
        "robust_normal_xyz": stats.normal,
        "normal_dispersion_degrees": {
            "median": stats.normal_dispersion_median_degrees,
            "p95": stats.normal_dispersion_p95_degrees,
        },
        "local_plane_residual": {
            "median_world": stats.plane_residual_median,
            "p95_world": stats.plane_residual_p95,
        },
        "spatial_extent_xyz": stats.spatial_extent,
    }


def measure_semantic_patch(label: str, points: np.ndarray, *, tile_bins: int = 4) -> PatchStats:
    """Measure a patch with global PCA and deterministic local PCA tiles."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 8:
        raise ValueError(f"semantic patch {label} needs at least eight points")
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0.0:
        normal = -normal
    tangent_1 = vectors[:, 2]
    tangent_2 = vectors[:, 1]
    tangent_1 /= max(float(np.linalg.norm(tangent_1)), 1e-12)
    tangent_2 /= max(float(np.linalg.norm(tangent_2)), 1e-12)
    local_coordinates = np.column_stack([centered @ tangent_1, centered @ tangent_2])
    u_edges = np.linspace(float(np.min(local_coordinates[:, 0])), float(np.max(local_coordinates[:, 0])), tile_bins + 1)
    v_edges = np.linspace(float(np.min(local_coordinates[:, 1])), float(np.max(local_coordinates[:, 1])), tile_bins + 1)
    tile_normals: list[np.ndarray] = []
    for u_index in range(tile_bins):
        for v_index in range(tile_bins):
            u_mask = (local_coordinates[:, 0] >= u_edges[u_index]) & (local_coordinates[:, 0] <= u_edges[u_index + 1] if u_index == tile_bins - 1 else local_coordinates[:, 0] < u_edges[u_index + 1])
            v_mask = (local_coordinates[:, 1] >= v_edges[v_index]) & (local_coordinates[:, 1] <= v_edges[v_index + 1] if v_index == tile_bins - 1 else local_coordinates[:, 1] < v_edges[v_index + 1])
            selected = points[u_mask & v_mask]
            if len(selected) < 20:
                continue
            local_centered = selected - selected.mean(axis=0, keepdims=True)
            _local_values, local_vectors = np.linalg.eigh(local_centered.T @ local_centered)
            local_normal = local_vectors[:, 0]
            if float(local_normal @ normal) < 0.0:
                local_normal = -local_normal
            tile_normals.append(local_normal)
    if tile_normals:
        angles = np.degrees(np.arccos(np.clip(np.stack(tile_normals) @ normal, -1.0, 1.0)))
    else:
        angles = np.zeros((0,), dtype=np.float64)
    residual = np.abs(centered @ normal)
    return PatchStats(
        label=label,
        points=points,
        centroid=centroid,
        normal=normal,
        tangent_1=tangent_1,
        tangent_2=tangent_2,
        normal_dispersion_median_degrees=float(np.median(angles)) if len(angles) else 0.0,
        normal_dispersion_p95_degrees=float(np.percentile(angles, 95)) if len(angles) else 0.0,
        plane_residual_median=float(np.median(residual)),
        plane_residual_p95=float(np.percentile(residual, 95)),
        spatial_extent=np.ptp(points, axis=0),
    )


def semantic_junction_relation(top: PatchStats, side: PatchStats) -> dict[str, Any]:
    theta = float(np.degrees(np.arccos(np.clip(abs(float(top.normal @ side.normal)), 0.0, 1.0))))
    return {
        "status": "MEASURED",
        "theta_visible_degrees": theta,
        "source": "global PCA normals of separately audited visible tabletop and side/rim patches",
        "top_normal": top.normal,
        "side_normal": side.normal,
        "normal_dispersion_context_degrees": {
            "top_p95": top.normal_dispersion_p95_degrees,
            "side_p95": side.normal_dispersion_p95_degrees,
        },
        "hard_coded_right_angle": False,
    }


def _patch_proximity(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if not len(first) or not len(second):
        return {"status": "UNAVAILABLE", "samples": 0}
    distances = cKDTree(second).query(deterministic_subsample(first, 5000), workers=1)[0]
    return {
        "status": "MEASURED",
        "samples": int(len(distances)),
        "nearest_distance_min_world": float(np.min(distances)),
        "nearest_distance_p05_world": float(np.percentile(distances, 5)),
        "nearest_distance_p10_world": float(np.percentile(distances, 10)),
        "fraction_within_0_10_world": float(np.mean(distances <= 0.10)),
    }


def _semantic_pair_status(top: PatchStats, side: PatchStats, proximity: dict[str, Any]) -> dict[str, Any]:
    # This is a semantic/geometric audit gate, not an accuracy gate.
    sufficient = (
        len(top.points) >= 500
        and len(side.points) >= 500
        and stats_are_coherent(top)
        and stats_are_coherent(side)
        and proximity.get("status") == "MEASURED"
        and proximity.get("nearest_distance_min_world", float("inf")) <= 0.10
    )
    relation = semantic_junction_relation(top, side)
    return {
        "status": "GENUINE_VISIBLE_PAIR" if sufficient else "NO_GENUINE_PAIR",
        "reason": "point count, local plane coherence, and spatial proximity audit before H2 construction",
        "top_point_count": int(len(top.points)),
        "side_point_count": int(len(side.points)),
        "top_side_proximity": proximity,
        "measured_relation": relation,
    }


def stats_are_coherent(stats: PatchStats) -> bool:
    return bool(
        np.isfinite(stats.plane_residual_p95)
        and stats.plane_residual_p95 < 0.35
        and np.isfinite(stats.normal_dispersion_p95_degrees)
        # The mandatory primary case is deliberately non-planar.  Dispersion
        # is reported for disclosure, but a high local spread alone does not
        # reject a geometrically attached curved rim.
        and stats.normal_dispersion_p95_degrees < 89.0
    )


def build_leg_self_continuation(holdout: Holdout, config: SemanticDemoConfig = SEMANTIC_CONFIG) -> Prediction:
    """H1 primitive: first-order retained-frontier self-continuation."""

    return build_self_continuation(
        holdout,
        frontier_band_fraction=config.frontier_band_fraction,
        frontier_bins=config.frontier_bins,
        continuation_samples=config.continuation_samples,
    )


def build_semantically_aligned_junction_transfer(
    target_holdout: Holdout,
    *,
    source_top: PatchStats,
    source_side: PatchStats,
    target_retained_stats: PatchStats,
    config: SemanticDemoConfig = SEMANTIC_CONFIG,
    free_space: FreeSpaceProxy | None = None,
) -> tuple[Prediction | None, dict[str, Any], dict[str, Prediction]]:
    """H2 primitive using source angle and retained-only target frontier.

    The target frame is obtained from the target holdout's retained points.
    The source angle is measured only from the separately visible source pair.
    Branch validation sees only fixed volume/free-space/back-through checks.
    """

    target_frame = build_self_continuation(
        target_holdout,
        frontier_band_fraction=config.frontier_band_fraction,
        frontier_bins=config.frontier_bins,
        continuation_samples=config.continuation_samples,
    )
    relation = semantic_junction_relation(source_top, source_side)
    extent = float(target_holdout.u_bounds[1] - target_holdout.u_cut) * H2_EXTENT_FRACTION
    branches: dict[str, tuple[Prediction, dict[str, Any]]] = {}
    for sign, label in ((1, "plus_theta"), (-1, "minus_theta")):
        branch = _branch_prediction(
            target_frame.frontier_points,
            target_frame.frontier_tangent,
            target_retained_stats.normal,
            relation["theta_visible_degrees"],
            sign,
            extent,
            config.continuation_samples,
            target_holdout.u_axis,
        )
        diagnostics = validate_branch(
            branch,
            permitted_volume=target_holdout.permitted_volume,
            free_space=free_space,
            source_normal=target_retained_stats.normal,
            source_frontier=target_frame.frontier_points,
            u_axis=target_holdout.u_axis,
        )
        branches[label] = (branch, diagnostics)

    valid_names = [name for name, (_branch, diagnostics) in branches.items() if diagnostics.get("valid", False)]
    selected_name = valid_names[0] if len(valid_names) == 1 else None
    prediction = branches[selected_name][0] if selected_name is not None else None
    if prediction is not None:
        prediction.status = "VALID"
    status = "SELECTED" if selected_name is not None else ("AMBIGUOUS" if len(valid_names) > 1 else "NO_VALID_TRANSFER")
    branch_report = {
        name: {
            **_jsonable(branch.branch_diagnostics),
            **_jsonable(diagnostics),
        }
        for name, (branch, diagnostics) in branches.items()
    }
    report = {
        "status": status,
        "source_relation": _jsonable(relation),
        "target_frame_source": "target retained visible points only",
        "target_normal_source": "target retained visible points only",
        "target_retained_patch": _patch_report(target_retained_stats),
        "extent_world": extent,
        "extent_rule": "fixed 0.50 of declared target ROI continuation extent",
        "branch_selection_uses_withheld_reference": False,
        "branches": branch_report,
        "selected_branch": selected_name,
        "valid_branch_count": len(valid_names),
    }
    return prediction, report, {name: branch for name, (branch, _diag) in branches.items()}


def _prediction_metrics(holdout: Holdout, prediction: Prediction | None, h: float) -> dict[str, Any]:
    raw = evaluate_controlled_case(holdout, prediction, h=h)
    boundary = raw.get("boundary_continuity", {})
    return {
        "evaluation_population": "withheld reference region only",
        "withheld_reference_point_count": int(len(holdout.withheld_points)),
        "visible_fraction": float(len(holdout.retained_points) / max(len(holdout.full_points), 1)),
        "withheld_fraction": float(len(holdout.withheld_points) / max(len(holdout.full_points), 1)),
        "point_to_surface_distance": raw.get("point_to_surface", {}),
        "withheld_reference_coverage": raw.get("coverage", {}),
        "normal_angular_error": raw.get("normal_error", {}),
        "boundary_continuity": boundary,
        "boundary_position_gap_over_h": {
            "median": boundary.get("position_gap_over_h_median"),
            "p95": boundary.get("position_gap_over_h_p95"),
        },
        "h_world": float(h),
        "source_reference_used_for_prediction": False,
        "metric_fed_back_into_construction": False,
    }


def _all_points(*arrays: np.ndarray) -> np.ndarray:
    nonempty = [np.asarray(array, dtype=np.float64).reshape(-1, 3) for array in arrays if len(array)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 3), dtype=np.float64)


def _display_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically thin display clouds without changing saved geometry."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points
    cells = np.floor(points / DISPLAY_VOXEL_WORLD).astype(np.int64)
    _unique_cells, first_indices = np.unique(cells, axis=0, return_index=True)
    points = points[np.sort(first_indices)]
    return deterministic_subsample(points, max_points)


def _plot_points(axis: Any, points: np.ndarray, color: Any, *, size: float, alpha: float, max_points: int = 12000, label: str | None = None) -> None:
    points = _display_subsample(points, max_points)
    if len(points):
        display = _plot_coords(points)
        axis.scatter(display[:, 0], display[:, 1], display[:, 2], s=size, alpha=alpha, color=color, linewidths=0, label=label)


def _plot_prediction_surface(axis: Any, prediction: Prediction, color: Any, *, alpha: float = 0.32, label: str | None = None) -> None:
    if prediction.status == "NONE" or prediction.points_grid.ndim != 3 or prediction.points_grid.shape[0] < 2 or prediction.points_grid.shape[1] < 2:
        return
    display = _plot_coords(prediction.points_grid.reshape(-1, 3)).reshape(prediction.points_grid.shape[0], prediction.points_grid.shape[1], 3)
    axis.plot_surface(display[:, :, 0], display[:, :, 1], display[:, :, 2], color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=False, label=label)


def _configure_raw_axis(axis: Any, limits_points: np.ndarray, *, view: dict[str, float] = FIXED_VIEW) -> None:
    limits_points = np.asarray(limits_points, dtype=np.float64)
    mins = np.min(limits_points, axis=0)
    maxs = np.max(limits_points, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = 0.05 * span
    limits = np.stack([mins - padding, maxs + padding], axis=1)
    display_limits = limits[[0, 2, 1]]
    axis.set_xlim(*display_limits[0])
    axis.set_ylim(*display_limits[1])
    axis.set_zlim(*display_limits[2])
    try:
        axis.set_box_aspect(span[[0, 2, 1]])
    except AttributeError:
        pass
    axis.view_init(elev=view["elev"], azim=view["azim"])
    axis.set_xlabel("world x")
    axis.set_ylabel("world z")
    axis.set_zlabel("world y")


def _save_raw_view(
    output_path: Path,
    *,
    title: str,
    limits_points: np.ndarray,
    retained: np.ndarray,
    withheld: np.ndarray | None = None,
    prediction: Prediction | None = None,
    prediction_color: Any = PREDICTED_CYAN,
    error_heatmap: bool = False,
    branch_predictions: dict[str, Prediction] | None = None,
    view: dict[str, float] = FIXED_VIEW,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8.4, 7.0), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    branch_predictions = branch_predictions or {}
    _configure_raw_axis(axis, limits_points, view=view)
    _plot_points(axis, retained, OBSERVED_GREY, size=1.25, alpha=0.48, label="observed retained")
    if withheld is not None and len(withheld):
        _plot_points(axis, withheld, WITHHELD_GREEN, size=1.35, alpha=0.75, label="withheld reference")
    if prediction is not None and prediction.status == "VALID" and len(prediction.points):
        _plot_prediction_surface(axis, prediction, prediction_color, alpha=0.28)
        if error_heatmap and len(withheld):
            from scipy.spatial import cKDTree

            distances = cKDTree(withheld).query(prediction.points, workers=1)[0]
            display = _plot_coords(prediction.points)
            scatter = axis.scatter(display[:, 0], display[:, 1], display[:, 2], s=2.0, c=distances, cmap="magma", alpha=0.92, linewidths=0, label="prediction error")
            figure.colorbar(scatter, ax=axis, shrink=0.62, pad=0.10, label="distance to withheld reference (world)")
        else:
            _plot_points(axis, prediction.points, prediction_color, size=1.8, alpha=0.90, label="predicted continuation")
        _plot_points(axis, prediction.frontier_points, FRONTIER_YELLOW, size=8.0, alpha=0.95, max_points=256, label="visible termination")
    for name, branch in branch_predictions.items():
        _plot_prediction_surface(axis, branch, REJECTED_MAGENTA, alpha=0.20)
        _plot_points(axis, branch.points, REJECTED_MAGENTA, size=1.2, alpha=0.55, label=f"rejected {name}")
    axis.set_title(title)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_controlled_case_views(holdout: Holdout, prediction: Prediction | None, output_root: Path, prefix: str, *, branch_predictions: dict[str, Prediction] | None = None) -> dict[str, str]:
    branch_predictions = branch_predictions or {}
    limits = _all_points(holdout.full_points, prediction.points if prediction is not None else np.empty((0, 3)), *(branch.points for branch in branch_predictions.values()))
    names = {
        "full": output_root / f"{prefix}_full.png",
        "hidden_input": output_root / f"{prefix}_hidden_input.png",
        "prediction": output_root / f"{prefix}_prediction.png",
        "overlay": output_root / f"{prefix}_prediction_vs_reference.png",
    }
    _save_raw_view(names["full"], title=f"{prefix}: full reference; withheld = green", limits_points=limits, retained=holdout.retained_points, withheld=holdout.withheld_points)
    _save_raw_view(names["hidden_input"], title=f"{prefix}: visible-only retained input", limits_points=limits, retained=holdout.retained_points, prediction=None)
    _save_raw_view(names["prediction"], title=f"{prefix}: retained + predicted/rejected continuation", limits_points=limits, retained=holdout.retained_points, prediction=prediction, branch_predictions=branch_predictions)
    _save_raw_view(names["overlay"], title=f"{prefix}: prediction vs withheld reference; heatmap = error", limits_points=limits, retained=holdout.retained_points, withheld=holdout.withheld_points, prediction=prediction, error_heatmap=True, branch_predictions=branch_predictions)
    return {key: str(path) for key, path in names.items()}


def write_case_geometry(holdout: Holdout, prediction: Prediction | None, output_root: Path, *, prefix: str, branch_predictions: dict[str, Prediction] | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    branch_predictions = branch_predictions or {}
    arrays: dict[str, np.ndarray] = {
        "full_reference_points": holdout.full_points.astype(np.float32),
        "retained_visible_points": holdout.retained_points.astype(np.float32),
        "withheld_reference_points_evaluation_only": holdout.withheld_points.astype(np.float32),
        "frontier_points_retained_only": prediction.frontier_points.astype(np.float32) if prediction is not None else np.empty((0, 3), dtype=np.float32),
        "predicted_points": prediction.points.astype(np.float32) if prediction is not None and prediction.status == "VALID" else np.empty((0, 3), dtype=np.float32),
    }
    for name, branch in branch_predictions.items():
        arrays[f"{name}_points"] = branch.points.astype(np.float32)
    npz_path = output_root / f"{prefix}_geometry.npz"
    np.savez_compressed(npz_path, **arrays)
    retained_path = output_root / f"{prefix}_retained_visible.ply"
    withheld_path = output_root / f"{prefix}_withheld_reference_evaluation_only.ply"
    predicted_path = output_root / f"{prefix}_predicted_continuation.ply"
    _write_ply(retained_path, holdout.retained_points, color=(148, 153, 160))
    _write_ply(withheld_path, holdout.withheld_points, color=(26, 191, 82))
    if prediction is not None and prediction.status == "VALID":
        _write_ply(predicted_path, prediction.points_grid, faces=_grid_faces(*prediction.points_grid.shape[:2]), color=(0, 184, 209))
    branch_paths: dict[str, str] = {}
    for name, branch in branch_predictions.items():
        path = output_root / f"{prefix}_{name}.ply"
        _write_ply(path, branch.points_grid, faces=_grid_faces(*branch.points_grid.shape[:2]), color=(225, 26, 32))
        branch_paths[name] = str(path)
    return {
        "npz": str(npz_path),
        "retained_ply": str(retained_path),
        "withheld_reference_ply": str(withheld_path),
        "predicted_ply": str(predicted_path) if prediction is not None and prediction.status == "VALID" else None,
        "branch_ply": branch_paths,
    }


def write_actual_top_side_junction(output_path: Path, patches: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = _all_points(*patches.values())
    figure = plt.figure(figsize=(9.0, 7.0), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    _configure_raw_axis(axis, all_points)
    colors = {
        "source_top": SOURCE_TOP_GREEN,
        "source_side": SOURCE_SIDE_BLUE,
        "target_top": TARGET_TOP_ORANGE,
        "target_side": TARGET_SIDE_RED,
    }
    for label, points in patches.items():
        _plot_points(axis, points, colors[label], size=1.1, alpha=0.65, label=label)
    axis.set_title("Actual visible tabletop / side junction audit")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _candidate_b_leg_evidence(path: Path, volume: Box) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        return np.empty((0, 3)), np.empty((0, 3)), {"status": "ARCHIVE_MISSING", "path": str(path)}
    archive = np.load(path, allow_pickle=True)
    positions = np.asarray(archive["positions"], dtype=np.float64)
    global_b = np.asarray(archive["global_B"])
    kind = np.asarray(archive["kind"]).astype(str)
    local = volume.contains(positions)
    occluded = positions[local & (global_b == 2)]
    free = positions[local & (global_b == 1) & (kind == "R4_FRONT_OF_SURFACE_PROBE")]
    report = {
        "status": "MEASURED",
        "definition": "Candidate B global_B == 2 query points; explicit free proxy is R4_FRONT_OF_SURFACE_PROBE with global_B == 1",
        "local_query_count": int(np.sum(local)),
        "occluded_query_count": int(len(occluded)),
        "explicit_free_proxy_count": int(len(free)),
        "all_observed_queries_treated_as_free": False,
        "volume": volume.as_json(),
    }
    return occluded, free, report


def write_true_prototype_views(scene_points: np.ndarray, holdout: Holdout, prediction: Prediction, candidate_b_points: np.ndarray, output_root: Path) -> dict[str, str]:
    limits = _all_points(scene_points, holdout.retained_points, prediction.points, candidate_b_points)

    def save(path: Path, title: str, *, visible: bool, predicted: bool, candidate: bool, view: dict[str, float] = FIXED_VIEW) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(8.4, 7.0), facecolor="white")
        axis = figure.add_subplot(111, projection="3d")
        _configure_raw_axis(axis, limits, view=view)
        if visible:
            _plot_points(axis, scene_points, OBSERVED_GREY, size=0.45, alpha=0.16, max_points=18000, label="visible surface")
            _plot_points(axis, holdout.retained_points, OBSERVED_GREY, size=1.4, alpha=0.75, label="visible retained brace")
            _plot_points(axis, holdout.frontier_points if hasattr(holdout, "frontier_points") else np.empty((0, 3)), FRONTIER_YELLOW, size=7.0, alpha=0.95, max_points=256, label="visible termination")
        if candidate:
            _plot_points(axis, candidate_b_points, CANDIDATE_B_ORANGE, size=8.0, alpha=0.85, max_points=4000, label="Candidate B occluded query")
        if predicted:
            _plot_prediction_surface(axis, prediction, PREDICTED_RED, alpha=0.38)
            _plot_points(axis, prediction.points, PREDICTED_RED, size=2.0, alpha=0.95, label="predicted occluded surface")
            _plot_points(axis, prediction.frontier_points, FRONTIER_YELLOW, size=8.0, alpha=0.95, max_points=256, label="visible termination")
        axis.set_title(title)
        axis.legend(loc="upper left", fontsize=8)
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "visible_only": output_root / "true_occluded_visible_only.png",
        "visible_plus_predicted": output_root / "true_occluded_visible_plus_predicted.png",
        "occluded_only": output_root / "true_occluded_only.png",
        "overview": output_root / "true_occluded_overview.png",
        "second_view": output_root / "true_occluded_second_view.png",
    }
    save(paths["visible_only"], "Candidate-B-supported true prototype: visible only", visible=True, predicted=False, candidate=False)
    save(paths["visible_plus_predicted"], "Candidate-B-supported true prototype: visible + predicted", visible=True, predicted=True, candidate=False)
    save(paths["occluded_only"], "Candidate-B-supported true prototype: Candidate B query + prediction", visible=False, predicted=True, candidate=True)
    save(paths["overview"], "Feasibility prototype: visible termination + predicted occluded brace", visible=True, predicted=True, candidate=True)
    save(paths["second_view"], "Feasibility prototype: second raw view", visible=True, predicted=True, candidate=True, view=SECOND_VIEW)
    return {name: str(path) for name, path in paths.items()}


def _case_report(holdout: Holdout, prediction: Prediction | None, metrics: dict[str, Any], *, mechanism: str, visuals: dict[str, str], geometry: dict[str, Any], semantic_label: str) -> dict[str, Any]:
    return {
        "semantic_label": semantic_label,
        "holdout": holdout.as_json(),
        "mechanism": mechanism,
        "continuation_extent_world": float(prediction.l_values[-1]) if prediction is not None and len(prediction.l_values) else 0.0,
        "prediction_status": prediction.status if prediction is not None else "NONE",
        "prediction_point_count": int(len(prediction.points)) if prediction is not None else 0,
        "prediction_triangle_count": int(prediction.triangle_count) if prediction is not None else 0,
        "prediction_surface_area_world2": _surface_area(prediction.points_grid) if prediction is not None else 0.0,
        "withheld_xyz_entered_prediction": False,
        "visuals": visuals,
        "geometry": geometry,
        "metrics": metrics,
    }


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    field = np.load(arguments.field_cache, allow_pickle=True)
    h = float(field["h"])
    mu = float(field["mu"])
    mesh = np.load(arguments.mesh_cache, allow_pickle=True)
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    config = SEMANTIC_CONFIG

    # Directly inspected fixed regions.  Their roles are established before
    # any holdout prediction or error calculation.
    leg_points = _select_box(vertices, config.leg_box, int(arguments.max_patch_points))
    source_top_points = _select_box(vertices, config.source_top_box, int(arguments.max_patch_points))
    source_side_points = _select_box(vertices, config.source_side_box, int(arguments.max_patch_points))
    target_top_points = _select_box(vertices, config.target_top_box, int(arguments.max_patch_points))
    target_side_points = _select_box(vertices, config.target_side_box, int(arguments.max_patch_points))
    scene_points = _select_box(vertices, config.table_scene_box, int(arguments.max_scene_points))

    leg_stats = measure_semantic_patch("actual_visible_leg_or_brace", leg_points)
    source_top_stats = measure_semantic_patch("actual_visible_source_tabletop", source_top_points)
    source_side_stats = measure_semantic_patch("actual_visible_source_side_or_rim", source_side_points)
    target_top_stats = measure_semantic_patch("actual_visible_target_tabletop", target_top_points)
    target_side_stats = measure_semantic_patch("actual_visible_target_side_or_rim", target_side_points)
    source_pair = _semantic_pair_status(source_top_stats, source_side_stats, _patch_proximity(source_top_points, source_side_points))
    target_pair = _semantic_pair_status(target_top_stats, target_side_stats, _patch_proximity(target_top_points, target_side_points))

    write_actual_top_side_junction(
        output_root / "actual_top_side_junction.png",
        {
            "source_top": source_top_points,
            "source_side": source_side_points,
            "target_top": target_top_points,
            "target_side": target_side_points,
        },
    )
    np.savez_compressed(
        output_root / "actual_top_side_junction.npz",
        source_top_points=source_top_points.astype(np.float32),
        source_side_points=source_side_points.astype(np.float32),
        target_top_points=target_top_points.astype(np.float32),
        target_side_points=target_side_points.astype(np.float32),
    )

    leg_holdout = build_fixed_holdout(
        leg_points,
        name="H1_actual_leg_brace_self_continuation",
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_axis=(1.0, 0.0, 0.0),
        u_bounds=config.leg_u_bounds,
        v_bounds=config.leg_v_bounds,
        n_bounds=config.leg_n_bounds,
        u_cut=config.leg_u_cut,
        permitted_volume=config.leg_permitted_volume,
    )
    h1_prediction = build_leg_self_continuation(leg_holdout, config)
    h1_metrics = _prediction_metrics(leg_holdout, h1_prediction, h)
    h1_visuals = write_controlled_case_views(leg_holdout, h1_prediction, output_root, "h1_leg")
    h1_geometry = write_case_geometry(leg_holdout, h1_prediction, output_root, prefix="h1_leg")

    h2_holdout: Holdout | None = None
    h2_prediction: Prediction | None = None
    h2_transfer: dict[str, Any] = {"status": "NOT_EXECUTED_SEMANTIC_PAIR_UNAVAILABLE"}
    h2_metrics: dict[str, Any] = {"evaluation_population": "not evaluated; no genuine pair"}
    h2_visuals: dict[str, str] = {}
    h2_geometry: dict[str, Any] = {}
    h2_branches: dict[str, Prediction] = {}
    if source_pair["status"] == "GENUINE_VISIBLE_PAIR" and target_pair["status"] == "GENUINE_VISIBLE_PAIR":
        h2_holdout = build_fixed_holdout(
            target_side_points,
            name="H2_actual_tabletop_side_junction_target_holdout",
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 1.0, 0.0),
            n_axis=(0.0, 0.0, 1.0),
            u_bounds=config.target_u_bounds,
            v_bounds=config.target_v_bounds,
            n_bounds=config.target_n_bounds,
            u_cut=config.target_u_cut,
            permitted_volume=config.target_permitted_volume,
        )
        target_retained_stats = measure_semantic_patch("target_side_retained_only", h2_holdout.retained_points)
        h2_prediction, h2_transfer, h2_branches = build_semantically_aligned_junction_transfer(
            h2_holdout,
            source_top=source_top_stats,
            source_side=source_side_stats,
            target_retained_stats=target_retained_stats,
            config=config,
        )
        h2_metrics = _prediction_metrics(h2_holdout, h2_prediction, h)
        h2_visuals = write_controlled_case_views(h2_holdout, h2_prediction, output_root, "h2_junction", branch_predictions=h2_branches)
        h2_geometry = write_case_geometry(h2_holdout, h2_prediction, output_root, prefix="h2_junction", branch_predictions=h2_branches)

    candidate_b_points, candidate_b_free, candidate_b_report = _candidate_b_leg_evidence(Path(arguments.candidate_b_archive), config.leg_permitted_volume)

    true_result: dict[str, Any] = {
        "status": "NOT_EXECUTED",
        "reason": "conditional true prototype requires an explicit post-inspection useful gate",
    }
    if arguments.controlled_gate in ("h1_useful", "h2_useful") and arguments.prototype_review in ("useful", "weak"):
        # Worklog 135's only available genuine Candidate-B local support is the
        # actual leg/brace crop.  H2 has no Candidate-B points in its target
        # crop, so its true prototype remains explicitly unavailable.
        if arguments.controlled_gate == "h1_useful" and len(candidate_b_points):
            true_holdout = build_fixed_holdout(
                leg_points,
                name="TRUE_OCCLUDED_ACTUAL_LEG_BRACE",
                u_axis=(0.0, 1.0, 0.0),
                v_axis=(0.0, 0.0, 1.0),
                n_axis=(1.0, 0.0, 0.0),
                u_bounds=config.leg_u_bounds,
                v_bounds=config.leg_v_bounds,
                n_bounds=config.leg_n_bounds,
                u_cut=config.leg_u_cut,
                permitted_volume=config.leg_permitted_volume,
            )
            true_prediction = build_leg_self_continuation(true_holdout, config)
            free_proxy = FreeSpaceProxy("Candidate B explicit front-of-surface local proxy", candidate_b_free, radius=2.0 * h)
            true_validation = validate_branch(
                true_prediction,
                permitted_volume=config.leg_permitted_volume,
                free_space=free_proxy,
                source_normal=leg_stats.normal,
                source_frontier=true_prediction.frontier_points,
                u_axis=true_holdout.u_axis,
            )
            true_visuals = write_true_prototype_views(scene_points, true_holdout, true_prediction, candidate_b_points, output_root)
            true_result = {
                "status": "EXECUTED",
                "region": "actual visible central leg/brace crop",
                "why_considered_occluded": "Candidate B global_B == 2 query evidence exists in the fixed leg/brace volume",
                "candidate_b_evidence": candidate_b_report,
                "continuation_mechanism": "same H1 retained-visible first-order frontier self-continuation",
                "visible_surface_point_count": int(len(true_holdout.retained_points)),
                "predicted_surface_point_count": int(len(true_prediction.points)),
                "visible_surface_area": None,
                "predicted_surface_area": _surface_area(true_prediction.points_grid),
                "validation": _jsonable(true_validation),
                "qualitative_judgment": "manual prototype review: " + arguments.prototype_review,
                "known_limitations": [
                    "Candidate B query points are support evidence, not hidden ground truth",
                    "no true-occluded quantitative accuracy is claimed",
                    "local closure is not assumed beyond the fixed small ROI",
                ],
                "visuals": true_visuals,
                "withheld_reference_used": False,
            }
        elif arguments.controlled_gate == "h2_useful":
            true_result = {
                "status": "NOT_EXECUTED_CANDIDATE_B_SUPPORT_UNAVAILABLE",
                "reason": "no Candidate-B occluded query points in the fixed target tabletop/side crop",
                "candidate_b_evidence": candidate_b_report,
            }
        else:
            true_result = {
                "status": "NOT_EXECUTED_CANDIDATE_B_SUPPORT_UNAVAILABLE",
                "reason": "Candidate-B local support is absent",
                "candidate_b_evidence": candidate_b_report,
            }

    if arguments.controlled_gate == "negative":
        verdict = "C"
        verdict_description = "C. Semantically correct primitives fail or are not useful under the fixed controlled review."
    elif arguments.controlled_gate == "review":
        verdict = "D"
        verdict_description = "D. Inconclusive until the fixed raw controlled outputs receive explicit qualitative review."
    elif true_result.get("status") == "EXECUTED" and arguments.prototype_review == "useful":
        verdict = "A"
        verdict_description = "A. A useful true candidate follows a useful controlled semantic continuation result."
    else:
        verdict = "B"
        verdict_description = "B. Controlled feasibility is partially positive, but the true-occluded prototype remains weak or unavailable."

    report = {
        "batch": "Worklog 136 semantically aligned occluded-surface feasibility demo",
        "status": "NON_CANONICAL_MEETING_DEMO",
        "INTENT ALIGNMENT": {
            "question": "Can actual visible table geometry continue into missing geometry without consuming the withheld XYZ?",
            "h1": "actual table leg/brace one-sided self-continuation",
            "h2": "actual visible tabletop/side junction angle transferred to a distinct target segment",
            "wl134_replay": {
                "h1": "curved side/rim self-continuation",
                "h2": "adjacent upper/lower strips around the same curved side ROI; not tabletop-to-side",
                "measured_angle_degrees": 1.4197,
                "interpretation": "WL134 did not test the intended semantics and is not re-run or re-tuned here",
            },
            "canonical_code_modified": False,
            "historical_worklogs_127_to_135_modified": False,
            "second_or_third_order_used": False,
            "full_scene_occluded_surface_attempted": False,
        },
        "WHY WL134 DID NOT TEST THE INTENDED SEMANTICS": {
            "wl134_h1": "curved side/rim self-continuation, not actual leg/brace self-continuation",
            "wl134_h2": "adjacent strips around one curved side ROI, not a real tabletop-to-side source/target transfer",
            "wl134_measured_angle_degrees": 1.4197,
            "current_batch_action": "WL134 was preserved and not re-run or tuned",
        },
        "IMPLEMENTATION FIDELITY": {
            "manual_choices": [
                "leg/brace ROI and tabletop/side source-target boxes",
                "world axes, bounds, one-sided cuts, and permitted volumes",
                "fixed H2 continuation fraction 0.50",
                "explicit post-inspection controlled/prototype gate",
            ],
            "heuristics": [
                "PCA-based local normals and retained-frontier estimation",
                "first-order ruled self-continuation",
                "Rodrigues plus/minus source-angle transfer branches",
                "Candidate B explicit front-of-surface free proxy for true prototype",
            ],
            "withheld_reference_roles": ["evaluation target", "final visualization", "quantitative error calculation"],
            "withheld_xyz_entered_fitter_or_continuation": False,
            "withheld_normals_entered_fitter_or_continuation": False,
            "withheld_extent_used_for_continuation": False,
            "branch_selection_uses_withheld_reference": False,
            "full_reference_information_used": "fixed boxes were chosen by direct inspection of the full WL127 mesh; full mesh rows then provide the declared holdout target and display source",
            "unacceptable_final_method_shortcuts": ["manual ROI selection", "fixed semantic labels from direct mesh inspection", "heuristic frontier propagation", "Candidate B read-only proxy as an occlusion cue"],
            "visualization_only": "saved PLY/NPZ remain unsimplified; PNG clouds use fixed deterministic 0.02-world-unit voxel display thinning",
            "isolated_from_canonical_research_code": True,
        },
        "ACTUAL TABLETOP / SIDE RELATION": {
            "source_pair": source_pair,
            "target_pair": target_pair,
            "source_top": _patch_report(source_top_stats),
            "source_side": _patch_report(source_side_stats),
            "target_top": _patch_report(target_top_stats),
            "target_side": _patch_report(target_side_stats),
            "audit_figure": str(output_root / "actual_top_side_junction.png"),
            "audit_geometry": str(output_root / "actual_top_side_junction.npz"),
            "h2_started_only_after_genuine_pair": True,
        },
        "H1 — LEG / BRACE SELF-CONTINUATION": {
            "roi": config.leg_box.as_json(),
            "coordinate_system": {"u_axis": [0.0, 1.0, 0.0], "v_axis": [0.0, 0.0, 1.0], "n_axis": [1.0, 0.0, 0.0]},
            "visible_fraction": h1_metrics["visible_fraction"],
            "withheld_fraction": h1_metrics["withheld_fraction"],
            "mechanism": "retained visible frontier + deterministic first-order ruled self-continuation",
            "prediction_status": h1_prediction.status,
            "prediction_point_count": int(len(h1_prediction.points)),
            "continuation_extent_world": float(h1_prediction.l_values[-1]),
            "withheld_median_error_over_h": h1_metrics.get("point_to_surface_distance", {}).get("median_over_h"),
            "withheld_p95_error_over_h": h1_metrics.get("point_to_surface_distance", {}).get("p95_over_h"),
            "coverage_le_h": h1_metrics.get("withheld_reference_coverage", {}).get("fraction_le_h"),
            "coverage_le_2h": h1_metrics.get("withheld_reference_coverage", {}).get("fraction_le_2h"),
            "normal_error": h1_metrics.get("normal_angular_error"),
            "boundary_continuity": h1_metrics.get("boundary_continuity"),
            "qualitative_result": "fixed raw outputs require/received manual review gate: " + arguments.controlled_gate,
            "metrics": h1_metrics,
            "visuals": h1_visuals,
            "geometry": h1_geometry,
        },
        "H2 — REAL JUNCTION-PATTERN TRANSFER": {
            "source_roi": {"top": config.source_top_box.as_json(), "side": config.source_side_box.as_json()},
            "target_roi": {"top": config.target_top_box.as_json(), "side": config.target_side_box.as_json()},
            "visible_fraction": h2_metrics.get("visible_fraction"),
            "withheld_fraction": h2_metrics.get("withheld_fraction"),
            "mechanism": "measured source tabletop-side angle transferred to retained-only target side frontier",
            "prediction_status": h2_prediction.status if h2_prediction is not None else "NONE",
            "continuation_extent_world": h2_transfer.get("extent_world"),
            "withheld_median_error_over_h": h2_metrics.get("point_to_surface_distance", {}).get("median_over_h"),
            "withheld_p95_error_over_h": h2_metrics.get("point_to_surface_distance", {}).get("p95_over_h"),
            "coverage_le_h": h2_metrics.get("withheld_reference_coverage", {}).get("fraction_le_h"),
            "coverage_le_2h": h2_metrics.get("withheld_reference_coverage", {}).get("fraction_le_2h"),
            "normal_error": h2_metrics.get("normal_angular_error"),
            "boundary_continuity": h2_metrics.get("boundary_continuity"),
            "transfer": h2_transfer,
            "qualitative_result": "fixed raw outputs require/received manual review gate: " + arguments.controlled_gate,
            "visuals": h2_visuals,
            "geometry": h2_geometry,
        },
        "TRUE-OCCLUDED PROTOTYPE": true_result,
        "PROMOTED": {
            "status": "NONE",
            "meaning": "no result is promoted to canonical Occluded Surface architecture",
        },
        "RETAINED": {
            "status": "H1/H2_ISOLATED_DEMO_ONLY",
            "meaning": "fixed semantic evidence and raw geometry remain available for advisor discussion",
        },
        "REJECTED": {
            "status": "WL134_SEMANTIC_INTERPRETATION_FOR_THIS_BATCH",
            "meaning": "WL134 H1/H2 are not treated as intended actual-leg or actual-tabletop-to-side tests",
        },
        "OPEN": {
            "status": "CONTINUATION_EXTENT_TERMINATION_OCCLUSION_CONFIDENCE",
            "meaning": "publishable definitions remain open; no architecture claim is made",
        },
        "inputs": {
            "mesh_cache": str(arguments.mesh_cache),
            "field_cache": str(arguments.field_cache),
            "candidate_b_archive": str(arguments.candidate_b_archive),
            "h": h,
            "mu": mu,
            "h_source": "frozen WL127 field.npz; not tuned",
        },
        "manual_demo_configuration": config.as_json(),
        "controlled_review_gate": {
            "selection": arguments.controlled_gate,
            "prototype_review": arguments.prototype_review,
            "rule": "true prototype is downstream only and requires explicit post-inspection useful/weak selection",
            "no_parameter_sweep": True,
        },
        "meeting_verdict": verdict,
        "meeting_verdict_description": verdict_description,
        "MEETING VERDICT": {
            "choice": verdict,
            "description": verdict_description,
            "answer": "actual leg/brace and actual tabletop-side controlled primitives did not provide usable positive evidence in this batch",
        },
    }
    report_path = output_root / "semantically_aligned_occluded_surface_demo_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_output_readme(report), encoding="utf-8")
    return report


def _output_readme(report: dict[str, Any]) -> str:
    return """# Semantically aligned occluded-surface feasibility demo

이 폴더는 frozen Worklog 127 Visible Surface를 읽기 전용으로 사용한
비정규 meeting-demo이다. canonical renderer/checkpoint, Candidate B,
historical topology와 production behavior를 변경하지 않았다.

## 핵심 raw 출력

- `actual_top_side_junction.png`: 실제 source/target tabletop-side semantic audit
- `h1_leg_full.png`, `h1_leg_hidden_input.png`, `h1_leg_prediction.png`, `h1_leg_prediction_vs_reference.png`
- `h2_junction_full.png`, `h2_junction_hidden_input.png`, `h2_junction_prediction.png`, `h2_junction_prediction_vs_reference.png`
- `h1_leg_geometry.npz` / `h2_junction_geometry.npz` 및 PLY geometry
- `semantically_aligned_occluded_surface_demo_report.json`: provenance, withheld-only metrics, gate, verdict

## 판정

`{verdict}` — {description}

정확한 값과 semantic attribution은 JSON report를 기준으로 한다. H1/H2
holdout의 withheld XYZ는 continuation에 들어가지 않고 evaluation/visualization에만
사용된다. true-occluded 출력은 명시적 useful gate가 선택된 경우에만 생성된다.

## 보존 경계

WL134 출력과 구현은 이 track에서 다시 실행하거나 수정하지 않았다. 이
폴더는 최종 Occluded Surface architecture의 검증 결과가 아니라, 실제
leg/brace와 실제 tabletop/side 관계가 주어졌을 때 continuation primitive가
어느 정도 의미 있는 geometry를 만들 수 있는지 보는 feasibility evidence다.
""".format(verdict=report["meeting_verdict"], description=report["meeting_verdict_description"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-cache", type=Path, default=WORKLOG_127_MESH)
    parser.add_argument("--field-cache", type=Path, default=WORKLOG_127_FIELD)
    parser.add_argument("--candidate-b-archive", type=Path, default=CANDIDATE_B_ARCHIVE)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/136_semantically_aligned_occluded_surface_demo")
    parser.add_argument("--max-scene-points", type=int, default=18000)
    parser.add_argument("--max-patch-points", type=int, default=24000)
    parser.add_argument("--controlled-gate", choices=("review", "h1_useful", "h2_useful", "negative"), default="review")
    parser.add_argument("--prototype-review", choices=("pending", "useful", "weak"), default="pending")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_demo(build_arg_parser().parse_args(argv))
    print(json.dumps({"meeting_verdict": report["meeting_verdict"], "h1": report["H1 — LEG / BRACE SELF-CONTINUATION"]["prediction_status"], "h2": report["H2 — REAL JUNCTION-PATTERN TRANSFER"].get("transfer", {}).get("status"), "true_occluded": report["TRUE-OCCLUDED PROTOTYPE"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
