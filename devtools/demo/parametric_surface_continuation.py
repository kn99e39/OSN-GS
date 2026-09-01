"""Bounded meeting demo: real-scene parametric surface continuation.

This module is deliberately outside ``scripts/devtools`` and the canonical
OSN-GS runtime.  It consumes the Worklog 127 extracted mesh read-only, removes
a manually fixed boundary-attached holdout, fits the existing NURBS fitter on
the retained points, and extends the fitted boundary with one deterministic
finite-difference rule.

The withheld reference points never enter ``fit_torch_visible_surface_lsq``.
They are loaded again only for evaluation, visualization, and error reporting.
The implementation is a feasibility demonstration, not an Occluded Surface
implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osn_gs.surface.torch_nurbs import (  # noqa: E402
    TorchNURBSSurface,
    fit_torch_visible_surface_lsq,
)


OBSERVED_GREY = (0.58, 0.60, 0.63)
PREDICTED_ORANGE = (0.95, 0.34, 0.08)
REFERENCE_BLUE = (0.20, 0.42, 0.72)
TERMINATION_YELLOW = (0.98, 0.78, 0.08)
HOLDOUT_RED = (0.82, 0.14, 0.12)

# Worklog 127's packed voxel-key convention, used only for the Figure 1
# sparse-field proxy.  It is not used by fitting or continuation.
_KEY_BOUND = 1 << 19
_AXIS_SPAN = _KEY_BOUND << 1
_STRIDE_Z = 1
_STRIDE_Y = _AXIS_SPAN
_STRIDE_X = _AXIS_SPAN * _AXIS_SPAN


@dataclass(frozen=True)
class ROIConfig:
    """A manually fixed local affine ROI and boundary holdout contract."""

    name: str
    semantic_label: str
    origin: tuple[float, float, float]
    axis_u: tuple[float, float, float]
    axis_v: tuple[float, float, float]
    axis_n: tuple[float, float, float]
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    n_bounds: tuple[float, float]
    holdout_u_cut: float
    continuation_rule: str = "boundary finite-difference tangent sweep"

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HoldoutPartition:
    full_mask: np.ndarray
    observed_mask: np.ndarray
    withheld_mask: np.ndarray
    u_norm: np.ndarray
    v_norm: np.ndarray
    coordinates: np.ndarray


@dataclass
class CaseResult:
    config: ROIConfig
    full_points: np.ndarray
    observed_points: np.ndarray
    withheld_points: np.ndarray
    predicted_points: np.ndarray
    predicted_normals: np.ndarray | None
    reference_eval_points: np.ndarray
    reference_distances: np.ndarray
    metrics: dict[str, Any]
    control_grid: np.ndarray
    continuation_control_grid: np.ndarray


PRIMARY_ROI = ROIConfig(
    name="curved_table_rim",
    semantic_label="curved table side / rim",
    origin=(0.0, 0.0, 0.0),
    # The selected right/front rim is a surface graph z = f(x, y).  The
    # axes/ranges are fixed demo-only choices, not estimated from the mesh.
    axis_u=(1.0, 0.0, 0.0),
    axis_v=(0.0, 0.0, 1.0),
    axis_n=(0.0, 1.0, 0.0),
    u_bounds=(-6.6, -4.7),
    v_bounds=(3.0, 4.2),
    n_bounds=(1.10, 1.70),
    holdout_u_cut=0.58,
)

SECONDARY_ROI = ROIConfig(
    name="thin_table_leg_brace",
    semantic_label="thin table leg / brace",
    origin=(0.0, 0.0, 0.0),
    # A vertical local strip on the central thin structure.  u is vertical;
    # the withheld portion is the lower boundary-attached segment.
    axis_u=(0.0, 1.0, 0.0),
    axis_v=(0.0, 0.0, 1.0),
    axis_n=(1.0, 0.0, 0.0),
    u_bounds=(0.48, 1.08),
    v_bounds=(0.75, 1.35),
    n_bounds=(-0.45, 0.20),
    holdout_u_cut=0.58,
)


def default_configs() -> tuple[ROIConfig, ROIConfig]:
    """Return the two fixed mandatory demo ROIs."""

    return PRIMARY_ROI, SECONDARY_ROI


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Return evenly spread, order-stable rows without random state."""

    points = np.asarray(points)
    limit = max(1, int(max_points))
    if len(points) <= limit:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    indices = np.unique(indices)
    return points[indices].copy()


def deterministic_indices(count: int, max_points: int) -> np.ndarray:
    """Order-stable subsample indices, exposed for contract tests."""

    count = int(count)
    limit = max(1, int(max_points))
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, limit, dtype=np.int64))


def _normalised_axis(axis: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(axis), dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise ValueError("ROI axis must be non-zero")
    return value / norm


def roi_coordinates(points: np.ndarray, config: ROIConfig) -> np.ndarray:
    """Project points into the manually fixed ``(u, v, n)`` frame."""

    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(config.origin, dtype=np.float64)
    axes = np.stack(
        [_normalised_axis(config.axis_u), _normalised_axis(config.axis_v), _normalised_axis(config.axis_n)],
        axis=1,
    )
    return (points - origin) @ axes


def build_holdout_partition(points: np.ndarray, config: ROIConfig) -> HoldoutPartition:
    """Build a deterministic boundary-attached holdout from fixed ranges.

    ``u_norm <= holdout_u_cut`` is fitting input and ``u_norm > cut`` is the
    withheld reference.  Because the cut is one side of a rectangular local
    domain, this is continuation across a visible termination, not a central
    interior hole.
    """

    coordinates = roi_coordinates(points, config)
    u0, u1 = map(float, config.u_bounds)
    v0, v1 = map(float, config.v_bounds)
    n0, n1 = map(float, config.n_bounds)
    if not (u1 > u0 and v1 > v0 and n1 > n0):
        raise ValueError(f"invalid ROI bounds for {config.name}")
    cut = float(config.holdout_u_cut)
    if not (0.0 < cut < 1.0):
        raise ValueError("holdout_u_cut must be strictly between 0 and 1")
    u_norm = (coordinates[:, 0] - u0) / (u1 - u0)
    v_norm = (coordinates[:, 1] - v0) / (v1 - v0)
    full = (
        (u_norm >= 0.0)
        & (u_norm <= 1.0)
        & (v_norm >= 0.0)
        & (v_norm <= 1.0)
        & (coordinates[:, 2] >= n0)
        & (coordinates[:, 2] <= n1)
    )
    observed = full & (u_norm <= cut)
    withheld = full & (u_norm > cut)
    return HoldoutPartition(full, observed, withheld, u_norm, v_norm, coordinates)


def boundary_attached_summary(partition: HoldoutPartition, cut: float) -> dict[str, Any]:
    """Summarise the one-sided holdout contract for the machine report."""

    observed_u = partition.u_norm[partition.observed_mask]
    withheld_u = partition.u_norm[partition.withheld_mask]
    if observed_u.size == 0 or withheld_u.size == 0:
        raise ValueError("ROI must contain both observed and withheld geometry")
    return {
        "observed_u_max": float(observed_u.max()),
        "withheld_u_min": float(withheld_u.min()),
        "boundary_attached": bool(observed_u.max() <= cut and withheld_u.min() > cut),
        "interior_hole_only": False,
        "cut": float(cut),
    }


def build_continuation_control_grid(control_grid: Any, holdout_cut: float) -> Any:
    """Extend a fitted NURBS boundary by one fixed tangent rule.

    The first two control columns define the boundary finite difference.  The
    missing/visible physical-domain ratio is fixed by the manually declared
    holdout cut; no withheld XYZ coordinate is consulted.
    """

    import torch

    grid = torch.as_tensor(control_grid, dtype=torch.float32)
    if grid.ndim != 3 or grid.shape[0] < 2:
        raise ValueError("continuation needs a (U,V,3) grid with at least two U columns")
    cut = float(holdout_cut)
    if not (0.0 < cut < 1.0):
        raise ValueError("holdout_cut must be strictly between 0 and 1")
    boundary = grid[-1]
    tangent_step = boundary - grid[-2]
    missing_over_visible = (1.0 - cut) / cut
    steps = torch.linspace(0.0, 1.0, int(grid.shape[0]), dtype=grid.dtype, device=grid.device)
    return boundary[None, :, :] + steps[:, None, None] * missing_over_visible * tangent_step[None, :, :]


def _surface_grid(surface: TorchNURBSSurface, samples_u: int, samples_v: int) -> tuple[Any, Any]:
    import torch

    u = torch.linspace(0.0, 1.0, int(samples_u), dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    v = torch.linspace(0.0, 1.0, int(samples_v), dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    return surface.evaluate(uv), uv


def _surface_grid_with_normals(surface: TorchNURBSSurface, samples_u: int, samples_v: int) -> tuple[Any, Any, Any]:
    import torch

    u = torch.linspace(0.0, 1.0, int(samples_u), dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    v = torch.linspace(0.0, 1.0, int(samples_v), dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    points, normals = surface.evaluate_with_normals(uv)
    return points, normals, uv


def estimate_point_normals(points: np.ndarray, k: int = 20) -> np.ndarray | None:
    """Estimate unoriented PCA normals for a local reference point cloud."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    count = min(max(4, int(k)), len(points))
    _, neighbours = cKDTree(points).query(points, k=count, workers=1)
    local = points[neighbours]
    centered = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(count - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(lengths, 1e-12, None)


def evaluate_withheld_geometry(
    withheld_reference: np.ndarray,
    predicted_surface_points: np.ndarray,
    h: float,
    *,
    reference_normals: np.ndarray | None = None,
    predicted_normals: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate only withheld reference rows against the prediction."""

    withheld_reference = np.asarray(withheld_reference, dtype=np.float64)
    predicted_surface_points = np.asarray(predicted_surface_points, dtype=np.float64)
    if len(withheld_reference) == 0 or len(predicted_surface_points) == 0:
        raise ValueError("withheld and predicted point sets must be non-empty")
    if float(h) <= 0:
        raise ValueError("h must be positive")
    from scipy.spatial import cKDTree

    tree = cKDTree(predicted_surface_points)
    distances, nearest = tree.query(withheld_reference, workers=1)
    normal_metrics: dict[str, Any] = {"status": "unavailable"}
    if reference_normals is not None and predicted_normals is not None:
        ref = np.asarray(reference_normals, dtype=np.float64)
        pred = np.asarray(predicted_normals, dtype=np.float64)[nearest]
        ref = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
        pred = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12, None)
        cosine = np.clip(np.abs(np.sum(ref * pred, axis=1)), 0.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        finite = np.isfinite(angles)
        if finite.any():
            normal_metrics = {
                "status": "estimated_unoriented_pca_vs_nurbs",
                "median_degrees": float(np.median(angles[finite])),
                "p95_degrees": float(np.percentile(angles[finite], 95)),
                "samples": int(finite.sum()),
            }
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "evaluation_population": "withheld_reference_only",
        "samples": int(len(distances)),
        "point_to_predicted_surface_distance": {
            "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "median_over_h": float(np.median(distances) / h),
            "p95_over_h": float(np.percentile(distances, 95) / h),
        },
        "withheld_reference_coverage": {
            "fraction_le_h": float(np.mean(distances <= h)),
            "fraction_le_2h": float(np.mean(distances <= 2.0 * h)),
        },
        "normal_angular_error": normal_metrics,
        "distances": distances,
        "nearest_predicted_indices": nearest.astype(np.int64),
    }


def boundary_continuity_metrics(observed_surface: TorchNURBSSurface, continuation_surface: TorchNURBSSurface) -> dict[str, Any]:
    """Measure the fitted/predicted interface without reference geometry."""

    import torch

    v = torch.linspace(0.0, 1.0, 32, dtype=observed_surface.control_grid.dtype, device=observed_surface.control_grid.device)
    uv_observed = torch.stack([torch.ones_like(v), v], dim=1)
    uv_predicted = torch.stack([torch.zeros_like(v), v], dim=1)
    observed, observed_normal = observed_surface.evaluate_with_normals(uv_observed)
    predicted, predicted_normal = continuation_surface.evaluate_with_normals(uv_predicted)
    position_gap = torch.linalg.norm(observed - predicted, dim=1)
    cosine = torch.clamp(torch.abs(torch.sum(observed_normal * predicted_normal, dim=1)), 0.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cosine))
    return {
        "position_gap": {
            "median": float(torch.median(position_gap).item()),
            "max": float(torch.max(position_gap).item()),
        },
        "normal_angle_discontinuity_degrees": {
            "median": float(torch.median(angle).item()),
            "p95": float(torch.quantile(angle, 0.95).item()),
        },
        "interface_definition": "fitted observed NURBS u=1 versus predicted continuation u=0",
    }


def _sha256_rows(points: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def _fit_one_case(
    reference_vertices: np.ndarray,
    config: ROIConfig,
    h: float,
    *,
    device_name: str,
    max_fit_points: int,
    max_reference_points: int,
) -> CaseResult:
    import torch

    partition = build_holdout_partition(reference_vertices, config)
    if not partition.observed_mask.any() or not partition.withheld_mask.any():
        raise ValueError(f"ROI {config.name} has no observed or withheld geometry")
    full_points = deterministic_subsample(reference_vertices[partition.full_mask], 12000)
    observed_all = reference_vertices[partition.observed_mask]
    withheld_all = reference_vertices[partition.withheld_mask]
    observed_points = deterministic_subsample(observed_all, 12000)
    reference_eval_points = deterministic_subsample(withheld_all, max_reference_points)
    fit_indices = deterministic_indices(len(observed_all), max_fit_points)
    fit_points = observed_all[fit_indices].copy()

    # This is the only point array passed to the existing NURBS fitter.  The
    # withheld array is deliberately not concatenated or used for UV setup.
    u_norm = partition.u_norm[partition.observed_mask][fit_indices]
    v_norm = partition.v_norm[partition.observed_mask][fit_indices]
    visible_cut = float(config.holdout_u_cut)
    initial_uv = np.stack([u_norm / visible_cut, v_norm], axis=1).astype(np.float32)
    if np.any(initial_uv[:, 0] > 1.0 + 1e-6) or np.any(initial_uv[:, 0] < -1e-6):
        raise AssertionError("observed-only UV binding crossed the holdout boundary")

    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name
    torch_points = torch.as_tensor(fit_points, dtype=torch.float32, device=device)
    torch_uv = torch.as_tensor(initial_uv, dtype=torch.float32, device=device)
    with torch.no_grad():
        surface, _ = fit_torch_visible_surface_lsq(
            torch_points,
            resolution_u=8,
            resolution_v=4,
            degree_u=2,
            degree_v=2,
            smoothness_lambda=1e-4,
            tikhonov_lambda=1e-4,
            correction_rounds=2,
            chunk_size=8192,
            projection_iterations=2,
            initial_uv=torch_uv,
        )
        continuation_grid = build_continuation_control_grid(surface.control_grid, visible_cut)
        continuation_surface = TorchNURBSSurface(
            control_grid=continuation_grid,
            weights=torch.ones(continuation_grid.shape[:2], dtype=continuation_grid.dtype, device=continuation_grid.device),
            degree_u=2,
            degree_v=2,
            observed_v_max=1.0,
        )
        predicted_torch, predicted_normals_torch, _ = _surface_grid_with_normals(continuation_surface, 96, 32)
        _, _ = _surface_grid(surface, 64, 24)

        predicted_points = predicted_torch.detach().cpu().numpy().astype(np.float64)
        predicted_normals = predicted_normals_torch.detach().cpu().numpy().astype(np.float64)
        continuity = boundary_continuity_metrics(surface, continuation_surface)
        control_grid = surface.control_grid.detach().cpu().numpy().astype(np.float64)
        continuation_control_grid = continuation_grid.detach().cpu().numpy().astype(np.float64)

    # Normals are estimated only on the withheld reference sample.  They are
    # not used to choose the ROI, fit parameters, or continuation length.
    reference_normals = estimate_point_normals(reference_eval_points, k=20)
    metric_payload = evaluate_withheld_geometry(
        reference_eval_points,
        predicted_points,
        h,
        reference_normals=reference_normals,
        predicted_normals=predicted_normals if reference_normals is not None else None,
    )
    distances = metric_payload.pop("distances")
    nearest = metric_payload.pop("nearest_predicted_indices")
    metric_payload["boundary_continuity"] = continuity
    total_roi_points = len(observed_all) + len(withheld_all)
    metric_payload["visible_fraction"] = float(len(observed_all) / max(total_roi_points, 1))
    metric_payload["withheld_fraction"] = float(len(withheld_all) / max(total_roi_points, 1))
    metric_payload["counts"] = {
        "full_roi_vertices": int(partition.full_mask.sum()),
        "observed_vertices": int(len(observed_all)),
        "withheld_vertices": int(len(withheld_all)),
        "fit_vertices": int(len(fit_points)),
    }
    metric_payload["fit_input_sha256"] = _sha256_rows(fit_points)
    metric_payload["withheld_reference_sha256"] = _sha256_rows(reference_eval_points)
    metric_payload["withheld_rows_in_fit_input"] = 0
    metric_payload["boundary_holdout"] = boundary_attached_summary(partition, visible_cut)
    metric_payload["continuation_extent"] = {
        "local_u_length": float((config.u_bounds[1] - config.u_bounds[0]) * (1.0 - visible_cut)),
        "visible_local_u_length": float((config.u_bounds[1] - config.u_bounds[0]) * visible_cut),
        "missing_over_visible_ratio": float((1.0 - visible_cut) / visible_cut),
    }
    metric_payload["continuation_mechanism"] = config.continuation_rule
    metric_payload["qualitative_result"] = _qualitative_label(metric_payload)
    metric_payload["device"] = str(device)
    metric_payload["fit_contract"] = {
        "fitter": "osn_gs.surface.torch_nurbs.fit_torch_visible_surface_lsq",
        "input": "retained observed points only",
        "withheld_geometry_used_for_fit": False,
        "continuation_parameter_sweep": False,
    }
    return CaseResult(
        config=config,
        full_points=full_points,
        observed_points=observed_points,
        withheld_points=deterministic_subsample(withheld_all, 12000),
        predicted_points=predicted_points,
        predicted_normals=predicted_normals,
        reference_eval_points=reference_eval_points,
        reference_distances=distances,
        metrics=metric_payload,
        control_grid=control_grid,
        continuation_control_grid=continuation_control_grid,
    )


def _qualitative_label(metrics: dict[str, Any]) -> str:
    median_h = metrics["point_to_predicted_surface_distance"]["median_over_h"]
    coverage = metrics["withheld_reference_coverage"]["fraction_le_h"]
    if median_h <= 2.0 and coverage >= 0.50:
        return "visually plausible / low withheld error"
    if median_h <= 5.0 and coverage >= 0.20:
        return "mixed / usable only as a weak feasibility signal"
    return "weak / continuation error is substantial"


def _plot_coords(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    # Keep the same semantic orientation in every panel: horizontal x, depth z,
    # vertical y.  This is a display transform only.
    return points[:, [0, 2, 1]]


def _set_equal_3d_limits(axes: Any, all_points: np.ndarray) -> None:
    coords = _plot_coords(all_points)
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * 0.55, 1e-3)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))
    axes.view_init(elev=24, azim=-60)
    axes.set_xlabel("x")
    axes.set_ylabel("z")
    axes.set_zlabel("y")


def _scatter3d(axes: Any, points: np.ndarray, colour: Any, *, size: float = 1.2, alpha: float = 0.72) -> None:
    coords = _plot_coords(points)
    axes.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=size, c=[colour], alpha=alpha, linewidths=0)


def _table_scene_mask(points: np.ndarray) -> np.ndarray:
    """Fixed presentation crop for the representative table view only."""

    points = np.asarray(points)
    return (
        (points[:, 0] >= -12.0)
        & (points[:, 0] <= 20.0)
        & (points[:, 1] >= -1.0)
        & (points[:, 1] <= 2.0)
        & (points[:, 2] >= -10.0)
        & (points[:, 2] <= 6.0)
    )


def write_case_figure(case: CaseResult, output_path: Path, *, primary: bool = False) -> None:
    """Write the clean four-panel holdout figure with a shared camera."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    all_points = np.concatenate(
        [case.full_points, case.observed_points, case.withheld_points, case.predicted_points], axis=0
    )
    titles = [
        "A  Full-evidence reference",
        "B  Visible-only holdout input",
        "C  NURBS continuation",
        "D  Prediction vs withheld reference / error",
    ]
    for axes_item, title in zip(axes, titles):
        axes_item.set_title(title, fontsize=14, pad=10)
        _set_equal_3d_limits(axes_item, all_points)
    _scatter3d(axes[0], case.full_points, REFERENCE_BLUE, size=1.0, alpha=0.50)
    axes[0].text2D(0.03, 0.92, "real WL127 mesh before holdout", transform=axes[0].transAxes, fontsize=10)

    _scatter3d(axes[1], case.observed_points, OBSERVED_GREY, size=1.1, alpha=0.78)
    cut_boundary = case.control_grid[-1]
    _scatter3d(axes[1], cut_boundary, TERMINATION_YELLOW, size=8.0, alpha=1.0)
    axes[1].text2D(0.03, 0.92, "withheld side absent; yellow = visible termination", transform=axes[1].transAxes, fontsize=10)

    _scatter3d(axes[2], case.observed_points, OBSERVED_GREY, size=0.9, alpha=0.48)
    _scatter3d(axes[2], case.predicted_points, PREDICTED_ORANGE, size=1.7, alpha=0.92)
    axes[2].text2D(0.03, 0.92, "gray observed / orange predicted", transform=axes[2].transAxes, fontsize=10)

    distances = case.reference_distances
    norm = Normalize(vmin=0.0, vmax=max(float(np.percentile(distances, 95)), 2.0e-6))
    coords_ref = _plot_coords(case.reference_eval_points)
    heat = axes[3].scatter(
        coords_ref[:, 0], coords_ref[:, 1], coords_ref[:, 2],
        s=4.0, c=distances, cmap="inferno", norm=norm, alpha=0.80, linewidths=0,
    )
    _scatter3d(axes[3], case.predicted_points, PREDICTED_ORANGE, size=1.3, alpha=0.48)
    colorbar = fig.colorbar(heat, ax=axes[3], shrink=0.62, pad=0.02)
    colorbar.set_label("withheld reference distance")
    axes[3].text2D(0.03, 0.92, "orange prediction / heatmap = withheld reference", transform=axes[3].transAxes, fontsize=10)

    metric = case.metrics["point_to_predicted_surface_distance"]
    coverage = case.metrics["withheld_reference_coverage"]["fraction_le_h"]
    fig.suptitle(
        f"Controlled boundary holdout — {case.config.semantic_label}  |  fixed continuation, non-canonical demo",
        fontsize=18,
        y=0.985,
    )
    fig.text(
        0.50,
        0.015,
        f"withheld median error / h = {metric['median_over_h']:.2f}    ·    coverage ≤ h = {coverage:.1%}    ·    "+
        f"observed {case.metrics['counts']['observed_vertices']:,} / withheld {case.metrics['counts']['withheld_vertices']:,}",
        ha="center",
        fontsize=13,
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.45"},
    )
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.06, top=0.93, wspace=0.02, hspace=0.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_no_completion_figure(case: CaseResult, output_path: Path) -> None:
    """Export a small sanity baseline: withheld side remains absent."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    all_points = np.concatenate([case.full_points, case.observed_points, case.withheld_points], axis=0)
    _set_equal_3d_limits(ax, all_points)
    _scatter3d(ax, case.observed_points, OBSERVED_GREY, size=1.2, alpha=0.8)
    _scatter3d(ax, case.control_grid[-1], TERMINATION_YELLOW, size=10, alpha=1.0)
    ax.set_title("Sanity baseline — NO COMPLETION\nwithheld region intentionally absent")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _decode_field_keys(keys: np.ndarray, h: float) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    iz = keys % _AXIS_SPAN - _KEY_BOUND
    rest = keys // _AXIS_SPAN
    iy = rest % _AXIS_SPAN - _KEY_BOUND
    ix = rest // _AXIS_SPAN - _KEY_BOUND
    return (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * float(h)


def _sample_field_centers(field_path: Path, h: float, max_points: int = 12000) -> np.ndarray:
    if not field_path.exists():
        return np.empty((0, 3), dtype=np.float32)
    bundle = np.load(field_path, allow_pickle=True)
    keys = bundle["keys"]
    # The packed field is stored in key order, not scene order.  Take a larger
    # deterministic candidate sample, then apply the fixed table crop so the
    # presentation proxy is not dominated by far-field single-view bands.
    indices = deterministic_indices(len(keys), max_points * 20)
    points = _decode_field_keys(keys[indices], h)
    points = points[_table_scene_mask(points)]
    return deterministic_subsample(points, max_points)


def _evidence_coverages(evidence_path: Path, raycast_path: Path, h: float) -> dict[str, Any]:
    evidence = np.load(evidence_path) if evidence_path.exists() else None
    raycast = np.load(raycast_path) if raycast_path.exists() else None
    if evidence is None or raycast is None:
        return {"status": "unavailable"}
    distance = np.asarray(evidence["distance"])
    finite = np.isfinite(distance)
    counted = np.asarray(raycast["counted"])
    return {
        "renderer_evidence_coverage_le_h": float(np.mean(distance[finite] <= h)),
        "renderer_evidence_coverage_le_2h": float(np.mean(distance[finite] <= 2.0 * h)),
        "renderer_evidence_finite_events": int(finite.sum()),
        "ray_hit_coverage": float(counted[:, 1].sum() / max(counted[:, 0].sum(), 1)),
        "source": "WL127 cached evidence.npz and raycast.npz; read-only",
    }


def write_figure_one(
    mesh_path: Path,
    field_path: Path,
    evidence_path: Path,
    raycast_path: Path,
    h: float,
    output_path: Path,
    *,
    original_render_path: Path,
) -> dict[str, Any]:
    """Create the first meeting figure from read-only WL127 artifacts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh = np.load(mesh_path, allow_pickle=True)
    mesh_vertices = np.asarray(mesh["vertices"])
    mesh_points = deterministic_subsample(mesh_vertices[_table_scene_mask(mesh_vertices)], 16000)
    field_points = _sample_field_centers(field_path, h, 16000)
    if len(field_points):
        field_points = field_points[_table_scene_mask(field_points)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 7.3), facecolor="white")
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    all_for_limits = mesh_points if len(field_points) == 0 else np.concatenate([mesh_points, field_points], axis=0)
    for ax in axes[1:]:
        _set_equal_3d_limits(ax, all_for_limits)
    _scatter3d(axes[1], field_points if len(field_points) else mesh_points, REFERENCE_BLUE, size=0.8, alpha=0.55)
    axes[1].set_title("Renderer median evidence\n→ sparse authoritative field", fontsize=14)
    _scatter3d(axes[2], mesh_points, OBSERVED_GREY, size=0.9, alpha=0.62)
    axes[2].set_title("Worklog 127 reconstructed\nVisible Surface", fontsize=14)

    image_loaded = False
    if original_render_path.exists():
        from PIL import Image

        image = Image.open(original_render_path)
        axes[0].remove()
        image_axis = fig.add_subplot(1, 3, 1)
        image_axis.imshow(image)
        image_axis.axis("off")
        image_axis.set_title("Original 2DGS\ncheckpoint render", fontsize=14)
        image_loaded = True
    else:
        _scatter3d(axes[0], mesh_points, REFERENCE_BLUE, size=0.9, alpha=0.62)
        axes[0].set_title("Original 2DGS\nrender unavailable", fontsize=14)

    coverages = _evidence_coverages(evidence_path, raycast_path, h)
    if coverages.get("status") != "unavailable":
        fig.text(
            0.50,
            0.02,
            f"renderer evidence coverage ≤ h: {coverages['renderer_evidence_coverage_le_h']:.2%}    →    "+
            f"ray-hit coverage: {coverages['ray_hit_coverage']:.2%}",
            ha="center",
            fontsize=18,
            bbox={"facecolor": "#f5f5f5", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.5"},
        )
    fig.text(0.335, 0.52, "→", fontsize=34, ha="center", va="center")
    fig.text(0.665, 0.52, "→", fontsize=34, ha="center", va="center")
    fig.suptitle("Visible Surface is now real — renderer evidence to scene-scale reconstruction", fontsize=20, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.09, top=0.86, wspace=0.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"original_render_loaded": image_loaded, "coverages": coverages}


def write_error_map_figure(case: CaseResult, output_path: Path) -> None:
    """Backup 3D geometric error map on the withheld reference."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(9, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    all_points = np.concatenate([case.reference_eval_points, case.predicted_points], axis=0)
    _set_equal_3d_limits(ax, all_points)
    coords = _plot_coords(case.reference_eval_points)
    vmax = max(float(np.percentile(case.reference_distances, 95)), 1e-6)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2], s=5.0, c=case.reference_distances,
        cmap="inferno", norm=Normalize(0.0, vmax), linewidths=0,
    )
    _scatter3d(ax, case.predicted_points, PREDICTED_ORANGE, size=1.0, alpha=0.42)
    fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.03, label="distance to predicted continuation")
    ax.set_title(f"Backup withheld geometric error map — {case.config.semantic_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_normal_error_figure(case: CaseResult, output_path: Path) -> None:
    """Backup normal-error map using the same local PCA estimate as the report."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.spatial import cKDTree

    normals = estimate_point_normals(case.reference_eval_points, k=20)
    if normals is None or case.predicted_normals is None:
        return
    nearest = cKDTree(case.predicted_points).query(case.reference_eval_points, workers=1)[1]
    pred = case.predicted_normals[nearest]
    cosine = np.clip(np.abs(np.sum(normals * pred, axis=1)), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    fig = plt.figure(figsize=(9, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _set_equal_3d_limits(ax, np.concatenate([case.reference_eval_points, case.predicted_points], axis=0))
    coords = _plot_coords(case.reference_eval_points)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=5.0, c=angles, cmap="magma", norm=Normalize(0, 90), linewidths=0)
    fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.03, label="unoriented normal angle (degrees)")
    ax.set_title(f"Backup normal angular error — {case.config.semantic_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_tsdf_context_figures(
    vertices: np.ndarray,
    support_count: np.ndarray,
    field_path: Path,
    h: float,
    output_root: Path,
) -> dict[str, Any]:
    """Export read-only WL127 context backups without changing the TSDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = _table_scene_mask(vertices)
    ids = np.flatnonzero(mask)
    ids = ids[deterministic_indices(len(ids), 18000)] if len(ids) else ids
    points = vertices[ids]
    supports = np.asarray(support_count)[ids] if len(ids) else np.empty((0,), dtype=np.int32)
    fig = plt.figure(figsize=(9, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        colour = np.where(supports <= 1, "#d6279f", "#3b3b46")
        coords = _plot_coords(points)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1.0, c=colour, linewidths=0, alpha=0.65)
        _set_equal_3d_limits(ax, points)
    ax.set_title("Backup WL127 TSDF low-support context\nmagenta = support_count ≤ 1")
    fig.tight_layout()
    low_support_path = output_root / "backup_tsdf_low_support.png"
    fig.savefig(low_support_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    field_points = _sample_field_centers(field_path, h, 20000)
    field_points = field_points[_table_scene_mask(field_points)] if len(field_points) else field_points
    fig = plt.figure(figsize=(9, 7), facecolor="white")
    ax = fig.add_subplot(111)
    if len(field_points):
        scatter = ax.scatter(field_points[:, 0], field_points[:, 2], s=1.0, c=field_points[:, 1], cmap="viridis", linewidths=0)
        fig.colorbar(scatter, ax=ax, label="world y")
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_title("Backup WL127 authoritative field slice\nfixed table presentation crop")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    field_slice_path = output_root / "backup_tsdf_field_slice.png"
    fig.savefig(field_slice_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "low_support_figure": str(low_support_path),
        "field_slice_figure": str(field_slice_path),
        "candidate_b_vs_geometric_occlusion": {
            "status": "NOT_RENDERED_IN_ISOLATED_DEMO",
            "reason": "no Candidate B render/query export is consumed by this controlled holdout track",
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_case_artifacts(case: CaseResult, output_root: Path) -> None:
    case_root = output_root / case.config.name
    case_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_root / "demo_geometry.npz",
        full_points=case.full_points.astype(np.float32),
        observed_points=case.observed_points.astype(np.float32),
        withheld_points=case.withheld_points.astype(np.float32),
        predicted_points=case.predicted_points.astype(np.float32),
        reference_eval_points=case.reference_eval_points.astype(np.float32),
        reference_distances=case.reference_distances.astype(np.float32),
        control_grid=case.control_grid.astype(np.float32),
        continuation_control_grid=case.continuation_control_grid.astype(np.float32),
    )
    write_case_figure(case, case_root / "four_panel_holdout.png", primary=case.config.name == PRIMARY_ROI.name)
    write_no_completion_figure(case, case_root / "no_completion_baseline.png")
    write_error_map_figure(case, case_root / "backup_error_map.png")
    write_normal_error_figure(case, case_root / "backup_normal_error.png")
    (case_root / "case_report.json").write_text(
        json.dumps(_jsonable({"roi": case.config.as_json(), "metrics": case.metrics}), indent=2),
        encoding="utf-8",
    )


def _write_quantitative_table(cases: list[CaseResult], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = ["case", "visible %", "withheld %", "median / h", "p95 / h", "≤ h", "≤ 2h", "normal median"]
    rows = []
    for case in cases:
        m = case.metrics
        normal = m["normal_angular_error"].get("median_degrees", float("nan"))
        rows.append([
            case.config.name,
            f"{m['visible_fraction']:.1%}",
            f"{m['withheld_fraction']:.1%}",
            f"{m['point_to_predicted_surface_distance']['median_over_h']:.2f}",
            f"{m['point_to_predicted_surface_distance']['p95_over_h']:.2f}",
            f"{m['withheld_reference_coverage']['fraction_le_h']:.1%}",
            f"{m['withheld_reference_coverage']['fraction_le_2h']:.1%}",
            "n/a" if not math.isfinite(normal) else f"{normal:.1f}°",
        ])
    fig, ax = plt.subplots(figsize=(15, 2.6 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.75)
    ax.set_title("Backup quantitative table — withheld reference only", fontsize=15, pad=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_demo(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    field_bundle = np.load(arguments.field_cache, allow_pickle=True)
    h = float(field_bundle["h"])
    mu = float(field_bundle["mu"])
    reference_bundle = np.load(arguments.mesh_cache, allow_pickle=True)
    reference_vertices = np.asarray(reference_bundle["vertices"])
    configs = list(default_configs())
    if arguments.only_primary:
        configs = configs[:1]

    cases: list[CaseResult] = []
    failures: list[dict[str, str]] = []
    for config in configs:
        try:
            case = _fit_one_case(
                reference_vertices,
                config,
                h,
                device_name=arguments.device,
                max_fit_points=arguments.max_fit_points,
                max_reference_points=arguments.max_reference_points,
            )
            cases.append(case)
            _write_case_artifacts(case, output_root)
        except Exception as error:  # keep a mandatory case visible in the report
            failures.append({"case": config.name, "error": repr(error)})

    figure_one = write_figure_one(
        arguments.mesh_cache,
        arguments.field_cache,
        arguments.evidence_cache,
        arguments.raycast_cache,
        h,
        output_root / "figure_1_visible_surface_is_real.png",
        original_render_path=arguments.original_render,
    )
    context_figures = write_tsdf_context_figures(
        reference_vertices,
        np.asarray(reference_bundle["vertex_support_count"]),
        arguments.field_cache,
        h,
        output_root,
    )
    if cases:
        primary = next((case for case in cases if case.config.name == PRIMARY_ROI.name), None)
        if primary is not None:
            # Figure 2 is the primary curved-rim case.  The per-case copy is a
            # backup artifact; this exact filename is the hero slide figure.
            write_case_figure(primary, output_root / "figure_2_parametric_continuation.png", primary=True)

    if cases:
        _write_quantitative_table(cases, output_root / "backup_quantitative_table.png")

    report = {
        "batch": "real-scene parametric surface continuation feasibility demonstration",
        "status": "NON_CANONICAL_MEETING_DEMO",
        "canonical_preservation": {
            "production_behavior_modified": False,
            "wl127_mesh_read_only": True,
            "candidate_b_modified": False,
            "historical_topology_modified": False,
            "canonical_h_or_mu_tuned": False,
            "nurbs_fitter_redesigned": False,
            "full_scene_occluded_surface_attempted": False,
        },
        "inputs": {
            "reference_mesh": str(arguments.mesh_cache),
            "reference_mesh_role": "full-evidence WL127 Visible Surface; evaluation target and display source",
            "field_cache": str(arguments.field_cache),
            "h": h,
            "mu": mu,
            "h_source": "WL127 field.npz, read-only",
            "evidence_cache": str(arguments.evidence_cache),
            "raycast_cache": str(arguments.raycast_cache),
            "original_render": str(arguments.original_render),
        },
        "leakage_and_disclosure": {
            "roi_selection": "manual fixed boxes and axes, chosen before fitting from direct mesh inspection",
            "coordinate_system": "manual fixed affine axes and fixed numeric bounds; no withheld XYZ statistics",
            "continuation_extent": "fixed by declared ROI extent and holdout_u_cut, not selected from withheld error",
            "full_reference_masking": "the full mesh is read to apply the declared fixed mask and to separate observed rows from the evaluation target; this is disclosed and is not fitter input",
            "withheld_geometry_allowed_uses": ["evaluation target", "final visualization", "quantitative error"],
            "withheld_xyz_entered_fitter": False,
            "unacceptable_final_method_shortcuts": [
                "manual ROI/extent selection",
                "boundary tangent heuristic",
                "using the full reference mesh as an evaluation oracle",
            ],
        },
        "figure_1": figure_one,
        "backup_figures": context_figures,
        "cases": [
            {"roi": case.config.as_json(), "metrics": _jsonable(case.metrics)} for case in cases
        ],
        "failures": failures,
        "true_occluded_prototype": {
            "status": "NOT_EXECUTED",
            "reason": "optional conditional track intentionally left isolated; no canonical Occluded Surface claim",
        },
        "meeting_verdict": "PENDING_UNTIL_CASES_REVIEWED",
    }
    primary_case = next((case for case in cases if case.config.name == PRIMARY_ROI.name), None)
    if failures or primary_case is None:
        report["meeting_verdict"] = "NEGATIVE_FEASIBILITY_RESULT"
    elif primary_case.metrics["qualitative_result"].startswith("visually plausible"):
        report["meeting_verdict"] = "STRONG_FEASIBILITY_DEMO"
    elif primary_case.metrics["qualitative_result"].startswith("mixed"):
        report["meeting_verdict"] = "PARTIAL_FEASIBILITY_DEMO"
    else:
        report["meeting_verdict"] = "NEGATIVE_FEASIBILITY_RESULT"
    (output_root / "parametric_surface_continuation_report.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )
    (output_root / "README.md").write_text(_demo_readme(report), encoding="utf-8")
    return report


def _demo_readme(report: dict[str, Any]) -> str:
    verdict = report["meeting_verdict"]
    case_lines = []
    for case in report.get("cases", []):
        metrics = case["metrics"]
        dist = metrics["point_to_predicted_surface_distance"]
        cov = metrics["withheld_reference_coverage"]
        case_lines.append(
            f"- `{case['roi']['name']}`: visible {metrics['visible_fraction']:.1%}, withheld {metrics['withheld_fraction']:.1%}, "
            f"median/p95 = {dist['median_over_h']:.2f}/{dist['p95_over_h']:.2f} h, "
            f"coverage ≤h/≤2h = {cov['fraction_le_h']:.1%}/{cov['fraction_le_2h']:.1%}; {metrics['qualitative_result']}"
        )
    return """# Real-scene parametric surface continuation — meeting demo

이 폴더는 Worklog 127 Visible Surface를 읽기 전용으로 사용한 **비정규
feasibility demonstration**이다. 최종 Occluded Surface architecture나
production behavior를 수정하지 않았다.

## 핵심 출력

- `figure_1_visible_surface_is_real.png`: Original 2DGS → renderer median evidence → WL127 Visible Surface
- `figure_2_parametric_continuation.png`: curved table rim 4-panel holdout hero figure
- `<case>/four_panel_holdout.png`: 각 ROI의 backup 4-panel figure
- `backup_quantitative_table.png`: withheld-only 정량표
- `parametric_surface_continuation_report.json`: 전체 provenance/metric 계약

## Controlled feasibility result

""" + "\n".join(case_lines) + f"""

## 판정

`{verdict}`

True-occluded prototype: **NOT EXECUTED**. 이 batch는 controlled holdout
figures와 verdict에서 멈추며 canonical Occluded Surface로 진행하지 않는다.

## Fidelity

ROI/축/범위/holdout cut은 demo-only 수동 선택이다. fitter는
`fit_torch_visible_surface_lsq`를 그대로 호출하고, continuation은 fit된
마지막 control-column의 boundary finite difference를 고정 연장한다. withheld
XYZ는 evaluation/visualization/error에만 사용한다.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_mesh = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz"
    default_field = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz"
    default_evidence = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/evidence.npz"
    default_raycast = REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/raycast.npz"
    default_original = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/render.ppm"
    parser.add_argument("--mesh-cache", type=Path, default=default_mesh)
    parser.add_argument("--field-cache", type=Path, default=default_field)
    parser.add_argument("--evidence-cache", type=Path, default=default_evidence)
    parser.add_argument("--raycast-cache", type=Path, default=default_raycast)
    parser.add_argument("--original-render", type=Path, default=default_original)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/128_demo_parametric_surface_continuation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-fit-points", type=int, default=12000)
    parser.add_argument("--max-reference-points", type=int, default=12000)
    parser.add_argument("--only-primary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    report = run_demo(arguments)
    print(json.dumps({"verdict": report["meeting_verdict"], "cases": len(report["cases"]), "failures": report["failures"]}, indent=2))
    return 0 if report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
