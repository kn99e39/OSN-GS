"""Attribution-only analysis for the Worklog 129 continuation failure.

This module is deliberately separate from both Worklog 128 and Worklog 129.
It replays the frozen observed-side fit with diagnostics, then audits the
relationship between the manual ROI coordinates, the final NURBS footpoint
coordinates, the actual mesh interface, and the interface-connected withheld
mesh sheet.  It does not change either historical experiment and it does not
implement a stronger completion rule.

The large WL127 mesh is read with a small-row streaming reader for ``faces``.
Only the vertices are materialised; face connectivity is accumulated only for
the two fixed ROIs.  Face-derived geometry is used for attribution,
visualisation, and evaluation, never as fitter input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.corrected_first_order_parametric_continuation import (  # noqa: E402
    FrozenCase,
    _load_frozen_case,
    _metrics_for_prediction,
    _surface_from_grid,
    evaluate_corrected_surface,
)
from devtools.demo.parametric_surface_continuation import (  # noqa: E402
    PRIMARY_ROI,
    SECONDARY_ROI,
    ROIConfig,
    _jsonable,
    _plot_coords,
    _scatter3d,
    _set_equal_3d_limits,
    build_holdout_partition,
    deterministic_indices,
    deterministic_subsample,
    estimate_point_normals,
    evaluate_withheld_geometry,
    roi_coordinates,
)
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq  # noqa: E402


WORKLOG_128_COMMIT = "2d87366b910873562b9dfc223408d85257c5af9f"
WORKLOG_129_COMMIT = "1ca0da5"
HOLDOUT_CUT = 0.58
FIT_KWARGS = {
    "resolution_u": 8,
    "resolution_v": 4,
    "degree_u": 2,
    "degree_v": 2,
    "smoothness_lambda": 1e-4,
    "tikhonov_lambda": 1e-4,
    "correction_rounds": 2,
    "chunk_size": 8192,
    "projection_iterations": 2,
}
FROZEN_FIT_ATOL = 2e-5
FROZEN_FIT_RTOL = 2e-5
FACE_CHUNK_ROWS = 1_000_000
INTERFACE_MAX_POINTS = 50_000
BOUNDARY_SAMPLES = 128
UV_TERMINATION_THRESHOLD = 0.95
TARGET_SINGLE_SHEET_THRESHOLD = 0.90
DISTANCE_BIN_LABELS = ("0–1h", "1–2h", "2–4h", "4–8h", "8–16h", ">16h")
DISTANCE_BIN_EDGES = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf], dtype=np.float64)


@dataclass
class ReplayedFit:
    case: FrozenCase
    surface: Any
    initial_uv: np.ndarray
    final_uv: np.ndarray
    control_grid: np.ndarray
    fit_points: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class VertexROI:
    config: ROIConfig
    u_norm: np.ndarray
    v_norm: np.ndarray
    roi_mask: np.ndarray
    observed_mask: np.ndarray
    withheld_mask: np.ndarray
    full_points: np.ndarray
    observed_points: np.ndarray
    withheld_points: np.ndarray
    observed_ids: np.ndarray
    withheld_ids: np.ndarray


@dataclass
class MeshTrace:
    interface_points: np.ndarray
    interface_v: np.ndarray
    interface_seed_ids: np.ndarray
    withheld_mesh_active: np.ndarray
    withheld_component: np.ndarray
    withheld_face_component: np.ndarray
    withheld_face_count: int
    component_sizes: list[dict[str, Any]]


@dataclass
class CaseAnalysis:
    replay: ReplayedFit
    vertex_roi: VertexROI
    trace: MeshTrace
    boundary_points: np.ndarray
    boundary_tangent_v: np.ndarray
    boundary_normals: np.ndarray
    corrected_points: np.ndarray
    corrected_normals: np.ndarray
    corrected_distances: np.ndarray
    corrected_nearest: np.ndarray
    report: dict[str, Any]


def _normalised_axis(axis: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(axis), dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        raise ValueError("ROI axis must be non-zero")
    return value / length


def _roi_vertex_contract(vertices: np.ndarray, config: ROIConfig) -> VertexROI:
    """Build fixed ROI masks without retaining a full 3-column coordinate array."""

    vertices = np.asarray(vertices)
    origin = np.asarray(config.origin, dtype=np.float64)
    axis_u = _normalised_axis(config.axis_u)
    axis_v = _normalised_axis(config.axis_v)
    axis_n = _normalised_axis(config.axis_n)
    u_offset = float(origin @ axis_u)
    v_offset = float(origin @ axis_v)
    n_offset = float(origin @ axis_n)
    u_norm = ((vertices @ axis_u - u_offset - float(config.u_bounds[0])) / (float(config.u_bounds[1]) - float(config.u_bounds[0]))).astype(np.float32)
    v_norm = ((vertices @ axis_v - v_offset - float(config.v_bounds[0])) / (float(config.v_bounds[1]) - float(config.v_bounds[0]))).astype(np.float32)
    n = vertices @ axis_n - n_offset
    full = (
        (u_norm >= 0.0)
        & (u_norm <= 1.0)
        & (v_norm >= 0.0)
        & (v_norm <= 1.0)
        & (n >= float(config.n_bounds[0]))
        & (n <= float(config.n_bounds[1]))
    )
    del n
    observed = full & (u_norm <= float(config.holdout_u_cut))
    withheld = full & (u_norm > float(config.holdout_u_cut))
    observed_ids = np.flatnonzero(observed).astype(np.int64)
    withheld_ids = np.flatnonzero(withheld).astype(np.int64)
    return VertexROI(
        config=config,
        u_norm=u_norm,
        v_norm=v_norm,
        roi_mask=np.asarray(full, dtype=bool),
        observed_mask=np.asarray(observed, dtype=bool),
        withheld_mask=np.asarray(withheld, dtype=bool),
        full_points=np.asarray(vertices[full], dtype=np.float64),
        observed_points=np.asarray(vertices[observed], dtype=np.float64),
        withheld_points=np.asarray(vertices[withheld], dtype=np.float64),
        observed_ids=observed_ids,
        withheld_ids=withheld_ids,
    )


def _iter_npy_rows_from_npz(path: Path, member: str, rows_per_chunk: int = FACE_CHUNK_ROWS):
    """Yield rows from an uncompressed 2-D NPY member without loading it all."""

    with zipfile.ZipFile(path) as archive:
        with archive.open(member, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            elif version == (3, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_3_0(stream)
            else:
                raise ValueError(f"unsupported NPY header version {version}")
            if len(shape) != 2 or fortran_order:
                raise ValueError(f"expected a C-order 2-D member: {member} {shape} {fortran_order}")
            row_count, column_count = map(int, shape)
            itemsize = int(dtype.itemsize) * column_count
            for start in range(0, row_count, max(1, int(rows_per_chunk))):
                count = min(max(1, int(rows_per_chunk)), row_count - start)
                expected = count * itemsize
                payload = stream.read(expected)
                if len(payload) != expected:
                    raise ValueError(f"truncated NPY member {member} at row {start}")
                rows = np.frombuffer(payload, dtype=dtype, count=count * column_count)
                yield start, rows.reshape(count, column_count)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(int(size), dtype=np.int32)
        self.rank = np.zeros(int(size), dtype=np.uint8)

    def find(self, value: int) -> int:
        value = int(value)
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            next_value = int(self.parent[value])
            self.parent[value] = root
            value = next_value
        return root

    def union(self, first: int, second: int) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _edge_interface_points(
    face_points: np.ndarray,
    face_u: np.ndarray,
    face_v: np.ndarray,
    cut: float,
) -> tuple[list[np.ndarray], list[float]]:
    points: list[np.ndarray] = []
    values: list[float] = []
    for left, right in ((0, 1), (1, 2), (2, 0)):
        u_left = face_u[:, left]
        u_right = face_u[:, right]
        crosses = ((u_left <= cut) & (u_right > cut)) | ((u_right <= cut) & (u_left > cut))
        if not bool(crosses.any()):
            continue
        denominator = u_right[crosses] - u_left[crosses]
        alpha = (float(cut) - u_left[crosses]) / np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
        left_points = face_points[crosses, left]
        right_points = face_points[crosses, right]
        points.append(left_points + alpha[:, None] * (right_points - left_points))
        values.append(face_v[crosses, left] + alpha * (face_v[crosses, right] - face_v[crosses, left]))
    if not points:
        return [], []
    return [np.concatenate(points, axis=0)], [np.concatenate(values, axis=0)]


def trace_mesh_interface(mesh_cache: Path, vertices: np.ndarray, vertex_roi: VertexROI) -> MeshTrace:
    """Extract the fixed-u interface and withheld-side connectivity from faces."""

    withheld_ids = vertex_roi.withheld_ids
    if len(withheld_ids) == 0:
        raise ValueError(f"ROI {vertex_roi.config.name} has no withheld vertices")
    union = _UnionFind(len(withheld_ids))
    mesh_active = np.zeros(len(withheld_ids), dtype=bool)
    interface_chunks: list[np.ndarray] = []
    interface_v_chunks: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    face_components: list[np.ndarray] = []
    withheld_face_count = 0
    cut = float(vertex_roi.config.holdout_u_cut)

    for _start, faces in _iter_npy_rows_from_npz(mesh_cache, "faces.npy"):
        faces = np.asarray(faces, dtype=np.int64)
        in_roi = vertex_roi.roi_mask[faces].all(axis=1)
        if not bool(in_roi.any()):
            continue
        roi_faces = faces[in_roi]
        roi_observed = vertex_roi.observed_mask[roi_faces]
        roi_withheld = vertex_roi.withheld_mask[roi_faces]
        crossing = roi_observed.any(axis=1) & roi_withheld.any(axis=1)
        if bool(crossing.any()):
            cross_faces = roi_faces[crossing]
            cross_points = vertices[cross_faces]
            cross_u = vertex_roi.u_norm[cross_faces]
            cross_v = vertex_roi.v_norm[cross_faces]
            point_part, v_part = _edge_interface_points(cross_points, cross_u, cross_v, cut)
            interface_chunks.extend(point_part)
            interface_v_chunks.extend(np.asarray(item, dtype=np.float64) for item in v_part)
            withheld_cross = cross_faces[vertex_roi.withheld_mask[cross_faces]]
            if len(withheld_cross):
                seeds.append(withheld_cross.astype(np.int64, copy=False))
                mesh_active[np.searchsorted(withheld_ids, np.unique(withheld_cross))] = True
                for edge_a, edge_b in ((0, 1), (1, 2), (2, 0)):
                    both = vertex_roi.withheld_mask[cross_faces[:, edge_a]] & vertex_roi.withheld_mask[cross_faces[:, edge_b]]
                    if bool(both.any()):
                        a = np.searchsorted(withheld_ids, cross_faces[both, edge_a])
                        b = np.searchsorted(withheld_ids, cross_faces[both, edge_b])
                        for ia, ib in zip(a.tolist(), b.tolist()):
                            union.union(ia, ib)

        all_withheld = roi_withheld.all(axis=1)
        if bool(all_withheld.any()):
            withheld_faces = roi_faces[all_withheld]
            withheld_face_count += int(len(withheld_faces))
            local_faces = np.searchsorted(withheld_ids, withheld_faces)
            mesh_active[np.unique(local_faces)] = True
            face_components.append(local_faces[:, 0].copy())
            for edge_a, edge_b in ((0, 1), (1, 2), (2, 0)):
                for ia, ib in zip(local_faces[:, edge_a].tolist(), local_faces[:, edge_b].tolist()):
                    union.union(ia, ib)

    if not interface_chunks:
        raise ValueError(f"no mesh faces cross fixed holdout cut for {vertex_roi.config.name}")
    interface_points = np.concatenate(interface_chunks, axis=0).astype(np.float64, copy=False)
    interface_v = np.concatenate(interface_v_chunks, axis=0).astype(np.float64, copy=False)
    interface_points = deterministic_subsample(interface_points, INTERFACE_MAX_POINTS)
    interface_v = deterministic_subsample(interface_v, INTERFACE_MAX_POINTS)
    seed_ids = np.unique(np.concatenate(seeds, axis=0)) if seeds else np.zeros((0,), dtype=np.int64)
    seed_local = np.searchsorted(withheld_ids, seed_ids)
    seed_roots = {union.find(value) for value in seed_local.tolist()}
    roots = np.asarray([union.find(index) for index in range(len(withheld_ids))], dtype=np.int32)
    connected = mesh_active & np.asarray([root in seed_roots for root in roots], dtype=bool)

    counts: dict[int, int] = {}
    for root, active in zip(roots.tolist(), mesh_active.tolist()):
        if active:
            counts[int(root)] = counts.get(int(root), 0) + 1
    face_roots = np.concatenate(face_components, axis=0) if face_components else np.zeros((0,), dtype=np.int32)
    face_root_values = np.asarray([union.find(index) for index in face_roots.tolist()], dtype=np.int32)
    face_count_by_root: dict[int, int] = {}
    for root in face_root_values.tolist():
        face_count_by_root[int(root)] = face_count_by_root.get(int(root), 0) + 1
    component_sizes = [
        {
            "root": int(root),
            "withheld_vertex_count": int(count),
            "withheld_face_count": int(face_count_by_root.get(root, 0)),
            "interface_connected": bool(root in seed_roots),
        }
        for root, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return MeshTrace(
        interface_points=interface_points,
        interface_v=interface_v,
        interface_seed_ids=seed_ids,
        withheld_mesh_active=mesh_active,
        withheld_component=connected,
        withheld_face_component=face_root_values,
        withheld_face_count=withheld_face_count,
        component_sizes=component_sizes,
    )


def _fit_input_and_initial_uv(case: FrozenCase) -> tuple[np.ndarray, np.ndarray]:
    fit_points = np.asarray(case.observed_points, dtype=np.float64)
    coords = roi_coordinates(fit_points, case.config)
    cut = float(case.config.holdout_u_cut)
    initial_uv = np.stack([coords[:, 0] - case.config.u_bounds[0], coords[:, 1] - case.config.v_bounds[0]], axis=1)
    initial_uv[:, 0] /= float(case.config.u_bounds[1] - case.config.u_bounds[0]) * cut
    initial_uv[:, 1] /= float(case.config.v_bounds[1] - case.config.v_bounds[0])
    return fit_points, initial_uv.astype(np.float32)


def replay_frozen_fit(case: FrozenCase, device: str) -> ReplayedFit:
    """Replay WL128 with identical input and settings, collecting final UV."""

    import torch

    fit_points, initial_uv = _fit_input_and_initial_uv(case)
    torch_points = torch.as_tensor(fit_points, dtype=torch.float32, device=device)
    torch_initial_uv = torch.as_tensor(initial_uv, dtype=torch.float32, device=device)
    with torch.no_grad():
        surface, final_uv, diagnostics = fit_torch_visible_surface_lsq(
            torch_points,
            initial_uv=torch_initial_uv,
            collect_diagnostics=True,
            **FIT_KWARGS,
        )
    actual_grid = surface.control_grid.detach().cpu().numpy().astype(np.float64)
    difference = actual_grid - case.control_grid
    max_abs = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(np.square(difference))))
    sufficiently_identical = bool(np.allclose(actual_grid, case.control_grid, atol=FROZEN_FIT_ATOL, rtol=FROZEN_FIT_RTOL))
    if diagnostics is None:
        raise AssertionError("fit diagnostics were not returned")
    final_uv_np = final_uv.detach().cpu().numpy().astype(np.float64)
    initial_uv_np = initial_uv.astype(np.float64)
    return ReplayedFit(
        case=case,
        surface=surface,
        initial_uv=initial_uv_np,
        final_uv=final_uv_np,
        control_grid=actual_grid,
        fit_points=fit_points,
        diagnostics={
            "max_absolute_control_grid_difference": max_abs,
            "rms_control_grid_difference": rms,
            "absolute_tolerance": FROZEN_FIT_ATOL,
            "relative_tolerance": FROZEN_FIT_RTOL,
            "sufficiently_identical": sufficiently_identical,
            "fit_input_row_count": int(len(fit_points)),
            "diagnostic_round_count": int(len(diagnostics.rounds)),
        },
    )


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(first) < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    result = spearmanr(first, second)
    return float(result.statistic) if np.isfinite(result.statistic) else None


def _inversion_count(values: np.ndarray) -> tuple[int, int]:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    bit = np.zeros(len(values) + 2, dtype=np.int64)
    inversions = 0
    seen = 0
    for rank in ranks.tolist():
        index = int(rank) + 1
        prefix = 0
        cursor = index
        while cursor:
            prefix += int(bit[cursor])
            cursor -= cursor & -cursor
        inversions += seen - prefix
        cursor = index
        while cursor < len(bit):
            bit[cursor] += 1
            cursor += cursor & -cursor
        seen += 1
    pairs = len(values) * (len(values) - 1) // 2
    return int(inversions), int(pairs)


def _affine_fit(initial: np.ndarray, final: np.ndarray) -> dict[str, Any]:
    if len(initial) < 2 or np.ptp(initial) <= 1e-12:
        return {"slope": None, "intercept": None, "samples": int(len(initial))}
    matrix = np.column_stack([initial, np.ones(len(initial), dtype=np.float64)])
    slope, intercept = np.linalg.lstsq(matrix, final, rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept), "samples": int(len(initial))}


def _uv_axis_report(initial: np.ndarray, final: np.ndarray, axis: int) -> dict[str, Any]:
    initial_axis = np.asarray(initial[:, axis], dtype=np.float64)
    final_axis = np.asarray(final[:, axis], dtype=np.float64)
    inversion_count, pair_count = _inversion_count(final_axis[np.argsort(initial_axis, kind="mergesort")])
    terminal = final_axis[initial_axis >= 0.90]
    local = _affine_fit(initial_axis[initial_axis >= 0.90], final_axis[initial_axis >= 0.90])
    return {
        "axis": "u" if axis == 0 else "v",
        "pearson": _pearson(initial_axis, final_axis),
        "spearman": _spearman(initial_axis, final_axis),
        "monotonic_inversion_count": inversion_count,
        "monotonic_pair_count": pair_count,
        "monotonic_inversion_fraction": float(inversion_count / max(pair_count, 1)),
        "median_absolute_shift": float(np.median(np.abs(final_axis - initial_axis))),
        "p95_absolute_shift": float(np.percentile(np.abs(final_axis - initial_axis), 95)),
        "affine_best_fit": _affine_fit(initial_axis, final_axis),
        "local_affine_near_termination": local,
        "initial_min": float(np.min(initial_axis)),
        "initial_max": float(np.max(initial_axis)),
        "final_min": float(np.min(final_axis)),
        "final_max": float(np.max(final_axis)),
        "manual_terminal_initial_definition": "initial u >= 0.90" if axis == 0 else "not applicable",
        "manual_terminal_final_distribution": (
            {
                "samples": int(len(terminal)),
                "min": float(np.min(terminal)),
                "p05": float(np.percentile(terminal, 5)),
                "median": float(np.median(terminal)),
                "p95": float(np.percentile(terminal, 95)),
                "max": float(np.max(terminal)),
                "fraction_final_u_ge_0.95": float(np.mean(terminal >= UV_TERMINATION_THRESHOLD)),
            }
            if len(terminal)
            else {"samples": 0}
        ),
    }


def parameterization_report(replay: ReplayedFit) -> dict[str, Any]:
    axes = [_uv_axis_report(replay.initial_uv, replay.final_uv, axis) for axis in (0, 1)]
    terminal = axes[0]["manual_terminal_final_distribution"]
    stable = bool(
        terminal.get("samples", 0) > 0
        and terminal.get("median", -np.inf) >= 0.95
        and terminal.get("fraction_final_u_ge_0.95", 0.0) >= 0.90
        and axes[0]["monotonic_inversion_fraction"] <= 0.01
    )
    return {
        "u": axes[0],
        "v": axes[1],
        "final_u_still_corresponds_to_manual_observed_termination": stable,
        "interpretation": "YES" if stable else "NO_OR_UNSUPPORTED",
        "decision_rule": "terminal final-u median >= 0.95, >=90% terminal rows >=0.95, and inversion fraction <=1%",
    }


def _evaluate_boundary(surface: Any, device: str, samples: int = BOUNDARY_SAMPLES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    v = torch.linspace(0.0, 1.0, int(samples), dtype=surface.control_grid.dtype, device=device)
    uv = torch.stack([torch.ones_like(v), v], dim=1)
    points, tangent_u, tangent_v = surface.evaluate_with_derivatives(uv)
    points, normals = surface.evaluate_with_normals(uv)
    del tangent_u
    return (
        points.detach().cpu().numpy().astype(np.float64),
        tangent_v.detach().cpu().numpy().astype(np.float64),
        normals.detach().cpu().numpy().astype(np.float64),
    )


def boundary_support_report(replay: ReplayedFit, vertex_roi: VertexROI, boundary_points: np.ndarray, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    fit_tree = cKDTree(replay.fit_points)
    mesh_tree = cKDTree(vertex_roi.observed_points)
    fit_dist, nearest_fit = fit_tree.query(boundary_points, workers=1)
    mesh_dist, _ = mesh_tree.query(boundary_points, workers=1)
    nearest_final_u = replay.final_uv[nearest_fit, 0]
    v_values = replay.final_uv[:, 1]
    bin_edges = np.linspace(0.0, 1.0, 33)
    bin_index = np.clip(np.digitize(v_values, bin_edges, right=False) - 1, 0, 31)
    rows = []
    for index in range(32):
        selected = bin_index == index
        rows.append({
            "v_bin": int(index),
            "fit_point_count": int(selected.sum()),
            "footpoints_u_ge_0.95": int(np.sum(selected & (replay.final_uv[:, 0] >= UV_TERMINATION_THRESHOLD))),
            "final_u_median": float(np.median(replay.final_uv[selected, 0])) if selected.any() else None,
        })
    fit_support_h = fit_dist <= h
    fit_support_2h = fit_dist <= 2.0 * h
    mesh_support_h = mesh_dist <= h
    mesh_support_2h = mesh_dist <= 2.0 * h
    return {
        "boundary_parameter": "fitted NURBS u=1, v in 32 fixed bins",
        "bin_basis": "final fitted footpoint v for observed fitting samples",
        "nearest_observed_fitting_point_distance_over_h": {
            "median": float(np.median(fit_dist) / h),
            "p95": float(np.percentile(fit_dist, 95) / h),
        },
        "nearest_observed_tsdf_mesh_distance_over_h": {
            "median": float(np.median(mesh_dist) / h),
            "p95": float(np.percentile(mesh_dist, 95) / h),
        },
        "nearest_observed_final_footpoint_u": {
            "median": float(np.median(nearest_final_u)),
            "p05": float(np.percentile(nearest_final_u, 5)),
            "p95": float(np.percentile(nearest_final_u, 95)),
        },
        "direct_observed_support_fraction": {
            "fitting_points_le_h": float(np.mean(fit_support_h)),
            "fitting_points_le_2h": float(np.mean(fit_support_2h)),
            "tsdf_mesh_le_h": float(np.mean(mesh_support_h)),
            "tsdf_mesh_le_2h": float(np.mean(mesh_support_2h)),
        },
        "unsupported_v_bin_fraction": {
            "no_footpoint_u_ge_0.95": float(np.mean([row["footpoints_u_ge_0.95"] == 0 for row in rows])),
            "boundary_fit_distance_gt_2h": float(np.mean(~fit_support_2h)),
            "boundary_mesh_distance_gt_2h": float(np.mean(~mesh_support_2h)),
        },
        "v_bins": rows,
        "classification": "PARAMETRIC DOMAIN EDGE" if float(np.mean(mesh_support_h)) < 0.50 else "potentially observed-supported boundary",
        "observed_geometry_only": True,
    }


def _interface_tangents(interface_points: np.ndarray, interface_v: np.ndarray, query_v: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    values = np.asarray(interface_v, dtype=np.float64).reshape(-1, 1)
    tree = cKDTree(values)
    count = min(64, len(interface_points))
    output = np.zeros((len(query_v), 3), dtype=np.float64)
    for index, value in enumerate(np.asarray(query_v, dtype=np.float64)):
        _, neighbours = tree.query([[value]], k=count, workers=1)
        neighbours = np.asarray(neighbours).reshape(-1)
        local_v = interface_v[neighbours]
        local_p = interface_points[neighbours]
        centered = local_v - float(np.mean(local_v))
        denominator = float(np.dot(centered, centered))
        if denominator <= 1e-12:
            continue
        output[index] = ((centered[:, None] * (local_p - local_p.mean(axis=0))).sum(axis=0) / denominator)
    lengths = np.linalg.norm(output, axis=1, keepdims=True)
    return output / np.clip(lengths, 1e-12, None)


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.clip(np.linalg.norm(first, axis=1, keepdims=True), 1e-12, None)
    second = second / np.clip(np.linalg.norm(second, axis=1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(first * second, axis=1)), 0.0, 1.0)))


def geometric_interface_report(
    replay: ReplayedFit,
    vertex_roi: VertexROI,
    trace: MeshTrace,
    boundary_points: np.ndarray,
    boundary_tangent_v: np.ndarray,
    boundary_normals: np.ndarray,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    interface_tree = cKDTree(trace.interface_points)
    interface_dist, nearest_interface = interface_tree.query(boundary_points, workers=1)
    interface_tangent = _interface_tangents(trace.interface_points, trace.interface_v, np.linspace(0.0, 1.0, len(boundary_points)))
    tangent_angles = _angle_degrees(boundary_tangent_v, interface_tangent)
    reference_normals = estimate_point_normals(vertex_roi.full_points, k=20)
    normal_angles: np.ndarray | None = None
    if reference_normals is not None:
        full_tree = cKDTree(vertex_roi.full_points)
        _, nearest_full = full_tree.query(trace.interface_points[nearest_interface], workers=1)
        normal_angles = _angle_degrees(boundary_normals, reference_normals[nearest_full])
    report = {
        "interface_definition": "mesh face edges crossing the fixed manual u_cut=0.58 inside the fixed ROI",
        "interface_point_count": int(len(trace.interface_points)),
        "fitted_boundary_to_interface_distance_over_h": {
            "median": float(np.median(interface_dist) / h),
            "p95": float(np.percentile(interface_dist, 95) / h),
        },
        "fitted_boundary_coverage_of_interface": {
            "boundary_points_le_h": float(np.mean(interface_dist <= h)),
            "boundary_points_le_2h": float(np.mean(interface_dist <= 2.0 * h)),
        },
        "boundary_tangent_agreement_degrees": {
            "median": float(np.median(tangent_angles)),
            "p95": float(np.percentile(tangent_angles, 95)),
        },
        "boundary_normal_agreement_degrees": (
            {
                "status": "estimated_unoriented_PCA_normal_vs_NURBS",
                "median": float(np.median(normal_angles)),
                "p95": float(np.percentile(normal_angles, 95)),
            }
            if normal_angles is not None
            else {"status": "unavailable"}
        ),
        "boundary_points": boundary_points,
        "interface_nearest_indices": nearest_interface.astype(np.int64),
        "observed_side_fitted_surface": "Worklog 128 frozen control grid replay",
    }
    return report


def target_coherence_report(case: FrozenCase, vertex_roi: VertexROI, trace: MeshTrace, h: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    component_sizes = trace.component_sizes
    interface_components = [item for item in component_sizes if item["interface_connected"]]
    competing_components = [item for item in component_sizes if not item["interface_connected"]]
    interface_vertex_count = int(np.sum(trace.withheld_component))
    interface_face_count = int(sum(item["withheld_face_count"] for item in interface_components))
    total_vertices = int(len(vertex_roi.withheld_points))
    total_faces = int(trace.withheld_face_count)
    interface_points = vertex_roi.withheld_points[trace.withheld_component]
    competing_points = vertex_roi.withheld_points[~trace.withheld_component]
    separation: dict[str, Any] = {"status": "no competing sheet vertices"}
    if len(interface_points) and len(competing_points):
        distances, _ = cKDTree(interface_points).query(competing_points, workers=1)
        separation = {
            "status": "available",
            "median_over_h": float(np.median(distances) / h),
            "p05_over_h": float(np.percentile(distances, 5) / h),
            "p95_over_h": float(np.percentile(distances, 95) / h),
            "samples": int(len(distances)),
        }
    eval_indices = deterministic_indices(len(vertex_roi.withheld_ids), len(case.reference_eval_points))
    eval_component = trace.withheld_component[eval_indices]
    original_eval_target = np.asarray(case.reference_eval_points, dtype=np.float64)
    connected_eval_target = original_eval_target[eval_component]
    if not len(connected_eval_target):
        connected_eval_target = interface_points
    vertex_fraction = interface_vertex_count / max(total_vertices, 1)
    face_fraction = interface_face_count / max(total_faces, 1)
    component_count = len(component_sizes)
    return {
        "mesh_connectivity_definition": "withheld vertices unioned by mesh-face edges, seeded by faces crossing fixed u_cut",
        "total_withheld_mesh_components": int(component_count),
        "interface_connected_component_count": int(len(interface_components)),
        "total_withheld_vertices": total_vertices,
        "mesh_active_withheld_vertices": int(np.sum(trace.withheld_mesh_active)),
        "interface_connected_withheld_vertices": interface_vertex_count,
        "interface_connected_withheld_vertex_fraction": float(vertex_fraction),
        "interface_connected_fraction_of_mesh_active_vertices": float(
            interface_vertex_count / max(int(np.sum(trace.withheld_mesh_active)), 1)
        ),
        "total_withheld_faces": total_faces,
        "interface_connected_withheld_faces": interface_face_count,
        "interface_connected_withheld_face_fraction": float(face_fraction),
        "component_sizes": component_sizes,
        "competing_sheet_nearest_separation": separation,
        "unrelated_top_side_leg_brace_sheets_coexist": bool(len(competing_components) > 0),
        "target_predominantly_one_interface_connected_sheet": bool(vertex_fraction >= TARGET_SINGLE_SHEET_THRESHOLD and face_fraction >= TARGET_SINGLE_SHEET_THRESHOLD),
        "target_single_sheet_threshold": TARGET_SINGLE_SHEET_THRESHOLD,
        "original_worklog_129_evaluation_population": int(len(original_eval_target)),
        "interface_connected_evaluation_population": int(len(connected_eval_target)),
        "interface_connected_eval_mask_fraction": float(np.mean(eval_component)) if len(eval_component) else 0.0,
        "interface_connected_target_points": connected_eval_target,
    }


def _normal_error_for_prediction(reference_points: np.ndarray, distances: np.ndarray, predicted_points: np.ndarray, predicted_normals: np.ndarray) -> dict[str, Any]:
    reference_normals = estimate_point_normals(reference_points, k=20)
    if reference_normals is None:
        return {"status": "unavailable"}
    from scipy.spatial import cKDTree

    nearest = cKDTree(predicted_points).query(reference_points, workers=1)[1]
    angles = _angle_degrees(reference_normals, predicted_normals[nearest])
    return {
        "status": "estimated_unoriented_pca_vs_nurbs",
        "median_degrees": float(np.median(angles)),
        "p95_degrees": float(np.percentile(angles, 95)),
    }


def _prediction_metrics(reference_points: np.ndarray, predicted_points: np.ndarray, predicted_normals: np.ndarray, h: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    distances, nearest = cKDTree(predicted_points).query(reference_points, workers=1)
    normals = _normal_error_for_prediction(reference_points, distances, predicted_points, predicted_normals)
    return (
        {
            "samples": int(len(reference_points)),
            "median_over_h": float(np.median(distances) / h),
            "p95_over_h": float(np.percentile(distances, 95) / h),
            "coverage_le_h": float(np.mean(distances <= h)),
            "coverage_le_2h": float(np.mean(distances <= 2.0 * h)),
            "normal_error": normals,
        },
        np.asarray(distances, dtype=np.float64),
        np.asarray(nearest, dtype=np.int64),
    )


def target_population_report(
    case: FrozenCase,
    vertex_roi: VertexROI,
    trace: MeshTrace,
    corrected_points: np.ndarray,
    corrected_normals: np.ndarray,
    h: float,
    frozen_metrics: dict[str, Any],
) -> dict[str, Any]:
    original_metrics, original_distances, original_nearest = _prediction_metrics(case.reference_eval_points, corrected_points, corrected_normals, h)
    eval_indices = deterministic_indices(len(vertex_roi.withheld_ids), len(case.reference_eval_points))
    connected_mask = trace.withheld_component[eval_indices]
    connected_target = case.reference_eval_points[connected_mask]
    if len(connected_target) == 0:
        connected_target = vertex_roi.withheld_points[trace.withheld_component]
    connected_metrics, _connected_distances, _connected_nearest = _prediction_metrics(connected_target, corrected_points, corrected_normals, h)
    saved_dist = frozen_metrics["point_to_predicted_surface_distance"]
    saved_cov = frozen_metrics["withheld_reference_coverage"]
    unchanged = bool(
        math.isclose(original_metrics["median_over_h"], float(saved_dist["median_over_h"]), rel_tol=2e-5, abs_tol=2e-5)
        and math.isclose(original_metrics["p95_over_h"], float(saved_dist["p95_over_h"]), rel_tol=2e-5, abs_tol=2e-5)
        and math.isclose(original_metrics["coverage_le_h"], float(saved_cov["fraction_le_h"]), rel_tol=2e-5, abs_tol=2e-5)
        and math.isclose(original_metrics["coverage_le_2h"], float(saved_cov["fraction_le_2h"]), rel_tol=2e-5, abs_tol=2e-5)
    )
    return {
        "frozen_worklog_129_metrics": {
            "median_over_h": float(saved_dist["median_over_h"]),
            "p95_over_h": float(saved_dist["p95_over_h"]),
            "coverage_le_h": float(saved_cov["fraction_le_h"]),
            "coverage_le_2h": float(saved_cov["fraction_le_2h"]),
        },
        "recomputed_original_population": original_metrics,
        "interface_connected_population": connected_metrics,
        "original_population_metric_unchanged": unchanged,
        "original_population_definition": "exact frozen Worklog 129 reference_eval_points",
        "interface_connected_population_definition": "same fixed ROI withheld mesh order, restricted to face-connected component(s) seeded by fixed u_cut interface",
        "withheld_geometry_role": "evaluation only",
        "raw_corrected_distances": original_distances,
        "raw_corrected_nearest": original_nearest,
    }


def distance_to_termination_report(
    case: FrozenCase,
    trace: MeshTrace,
    corrected_points: np.ndarray,
    corrected_normals: np.ndarray,
    h: float,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    target = case.reference_eval_points
    distance_to_interface, _ = cKDTree(trace.interface_points).query(target, workers=1)
    metrics, prediction_distances, nearest = _prediction_metrics(target, corrected_points, corrected_normals, h)
    reference_normals = estimate_point_normals(target, k=20)
    predicted_angles = None
    if reference_normals is not None:
        predicted_angles = _angle_degrees(reference_normals, corrected_normals[nearest])
    normalized = distance_to_interface / h
    bins = []
    for index, label in enumerate(DISTANCE_BIN_LABELS):
        selected = (normalized >= DISTANCE_BIN_EDGES[index]) & (normalized < DISTANCE_BIN_EDGES[index + 1])
        row: dict[str, Any] = {
            "bin": label,
            "distance_definition": "Euclidean distance from frozen Worklog 127 mesh face interface, divided by h",
            "reference_count": int(selected.sum()),
        }
        if bool(selected.any()):
            row.update({
                "distance_median_over_h": float(np.median(normalized[selected])),
                "geometry_median_error_over_h": float(np.median(prediction_distances[selected] / h)),
                "geometry_p95_error_over_h": float(np.percentile(prediction_distances[selected] / h, 95)),
                "coverage_le_h": float(np.mean(prediction_distances[selected] <= h)),
                "coverage_le_2h": float(np.mean(prediction_distances[selected] <= 2.0 * h)),
                "normal_median_degrees": float(np.median(predicted_angles[selected])) if predicted_angles is not None else None,
                "normal_p95_degrees": float(np.percentile(predicted_angles[selected], 95)) if predicted_angles is not None else None,
            })
        else:
            row.update({
                "distance_median_over_h": None,
                "geometry_median_error_over_h": None,
                "geometry_p95_error_over_h": None,
                "coverage_le_h": None,
                "coverage_le_2h": None,
                "normal_median_degrees": None,
                "normal_p95_degrees": None,
            })
        bins.append(row)
    return {
        "distance_definition": "Euclidean point-to-interface distance; geodesic tracing was not used",
        "fixed_bins": bins,
        "all_reference_metrics": metrics,
        "raw_distance_to_interface_over_h": normalized,
    }


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def _load_corrected_prediction(output_root: Path, case: FrozenCase, surface: Any, device: str) -> tuple[np.ndarray, np.ndarray, str]:
    path = output_root / f"{case.config.name}_corrected_arm.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return (
            np.asarray(data["corrected_points"], dtype=np.float64),
            np.asarray(data["corrected_normals"], dtype=np.float64),
            "frozen Worklog 129 corrected arm NPZ",
        )
    points, normals = evaluate_corrected_surface(surface, case.config.holdout_u_cut)
    return points, normals, "re-evaluated Worklog 129 corrected analytic arm from frozen control grid"


def _write_boundary_support_figure(
    output_path: Path,
    vertex_roi: VertexROI,
    trace: MeshTrace,
    replay: ReplayedFit,
    boundary_points: np.ndarray,
    h: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(20, 11), facecolor="white")
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(5)]
    all_points = np.concatenate([vertex_roi.full_points, trace.interface_points, boundary_points], axis=0)
    for axis in axes:
        _set_equal_3d_limits(axis, all_points)
    _scatter3d(axes[0], vertex_roi.observed_points, (0.58, 0.60, 0.63), size=0.7, alpha=0.45)
    axes[0].set_title("Observed WL127 surface")
    _scatter3d(axes[1], vertex_roi.observed_points, (0.58, 0.60, 0.63), size=0.55, alpha=0.28)
    _scatter3d(axes[1], trace.interface_points, (0.98, 0.78, 0.08), size=4.0, alpha=0.95)
    axes[1].set_title("True fixed-u mesh interface")
    _scatter3d(axes[2], vertex_roi.observed_points, (0.58, 0.60, 0.63), size=0.55, alpha=0.24)
    _scatter3d(axes[2], boundary_points, (0.12, 0.70, 0.88), size=8.0, alpha=0.95)
    axes[2].set_title("Fitted NURBS u=1 boundary")
    fit_dist, _ = cKDTree(replay.fit_points).query(boundary_points, workers=1)
    scatter = axes[3].scatter(*_plot_coords(boundary_points).T, c=np.clip(fit_dist / h, 0.0, 16.0), cmap="magma", s=8.0)
    figure.colorbar(scatter, ax=axes[3], shrink=0.6, pad=0.08, label="nearest observed fit point / h")
    axes[3].set_title("Boundary support heatmap")
    axes[4].set_visible(False)
    uv_axis = figure.add_subplot(2, 3, 6)
    uv_axis.scatter(replay.initial_uv[:, 0], replay.initial_uv[:, 1], s=2, alpha=0.18, color="#9aa0a6", label="initial UV")
    uv_axis.scatter(replay.final_uv[:, 0], replay.final_uv[:, 1], s=2, alpha=0.22, color="#16a6c7", label="final footpoint UV")
    uv_axis.axvline(1.0, color="#e6a700", linewidth=2, label="u=1 boundary")
    uv_axis.set_xlim(-0.05, 1.25)
    uv_axis.set_ylim(-0.05, 1.05)
    uv_axis.set_xlabel("u")
    uv_axis.set_ylabel("v")
    uv_axis.set_title("Final footpoint UV support")
    uv_axis.legend(fontsize=8, loc="best")
    figure.suptitle("Attribution: parameter boundary versus geometric termination", fontsize=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_qualitative_figure(
    output_path: Path,
    vertex_roi: VertexROI,
    trace: MeshTrace,
    boundary_points: np.ndarray,
    corrected_points: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(20, 11), facecolor="white")
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(6)]
    all_points = np.concatenate([vertex_roi.full_points, trace.interface_points, boundary_points, corrected_points], axis=0)
    for axis in axes:
        _set_equal_3d_limits(axis, all_points)
    panels = [
        ("1  Observed WL127 surface", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.7, 0.42)]),
        ("2  True fixed holdout interface", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.55, 0.22), (trace.interface_points, (0.98, 0.78, 0.08), 5.0, 0.95)]),
        ("3  Fitted NURBS u=1 boundary", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.55, 0.22), (boundary_points, (0.12, 0.70, 0.88), 8.0, 0.95)]),
        ("4  Frozen first-order prediction", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.45, 0.18), (corrected_points, (0.10, 0.75, 0.85), 1.5, 0.82)]),
        ("5  Withheld reference", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.45, 0.18), (vertex_roi.withheld_points, (0.82, 0.14, 0.12), 0.75, 0.75)]),
        ("6  Boundary / reference overview", [(vertex_roi.observed_points, (0.58, 0.60, 0.63), 0.35, 0.14), (trace.interface_points, (0.98, 0.78, 0.08), 4.0, 0.92), (corrected_points, (0.10, 0.75, 0.85), 1.3, 0.70), (vertex_roi.withheld_points, (0.82, 0.14, 0.12), 0.55, 0.42)]),
    ]
    for axis, (title, items) in zip(axes, panels):
        axis.set_title(title, fontsize=12)
        for points, color, size, alpha in items:
            _scatter3d(axis, points, color, size=size, alpha=alpha)
    figure.suptitle("Curved-rim continuation attribution — not a completion claim", fontsize=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_distance_figure(output_path: Path, case_reports: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
    x = np.arange(len(DISTANCE_BIN_LABELS))
    for axis, case_report in zip(axes, case_reports):
        bins = case_report["distance_to_termination"]["fixed_bins"]
        med = [row["geometry_median_error_over_h"] for row in bins]
        p95 = [row["geometry_p95_error_over_h"] for row in bins]
        coverage = [row["coverage_le_h"] for row in bins]
        axis.plot(x, med, marker="o", color="#1677b8", label="median error / h")
        axis.plot(x, p95, marker="s", color="#d45500", label="p95 error / h")
        twin = axis.twinx()
        twin.plot(x, coverage, marker="^", color="#2c9b54", label="coverage ≤ h")
        twin.set_ylim(0.0, 1.0)
        twin.set_ylabel("coverage ≤ h")
        axis.set_xticks(x, DISTANCE_BIN_LABELS, rotation=25)
        axis.set_ylabel("geometry error / h")
        axis.set_title(case_report["roi"]["semantic_label"])
        axis.grid(alpha=0.25)
        handles_a, labels_a = axis.get_legend_handles_labels()
        handles_b, labels_b = twin.get_legend_handles_labels()
        axis.legend(handles_a + handles_b, labels_a + labels_b, fontsize=8, loc="upper left")
    figure.suptitle("Fixed continuation error by Euclidean distance from geometric termination", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_report_readme(output_root: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Parameterization / termination / target-coherence attribution",
        "",
        "This is a diagnostic-only follow-up to Worklog 129. Worklog 128 and",
        "Worklog 129 remain frozen; no second-order continuation or canonical",
        "Occluded Surface implementation is included.",
        "",
        f"## Verdict: `{report['meeting_verdict']}`",
        "",
        "The analysis first reproduced the frozen fit, then audited final",
        "footpoint UV, face-derived termination, withheld-sheet connectivity,",
        "and fixed distance bins. All reference geometry is evaluation/display",
        "only and is never passed to the fitter.",
        "",
        "## Outputs",
        "",
        "- `parametric_continuation_attribution_report.json`: full audit report",
        "- `curved_rim_attribution.png`: meeting-safe six-view attribution figure",
        "- `curved_rim_boundary_support.png`: boundary support heatmap and UV support",
        "- `distance_to_termination.png`: fixed-bin distance/error plot",
        "",
    ]
    for item in report.get("cases", []):
        phase = item["parameterization"]
        terminal = phase["u"]["manual_terminal_final_distribution"]
        lines.append(
            f"- `{item['roi']['name']}`: frozen fit identical={item['frozen_fit_reproduction']['sufficiently_identical']}; "
            f"terminal final-u median={terminal.get('median')}; target interface-connected vertex fraction="
            f"{item['target_coherence']['interface_connected_withheld_vertex_fraction']:.3f}"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_frozen_wl129_metrics(path: Path, case_name: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    for arm in report.get("arms", []):
        if arm.get("roi", {}).get("name") == case_name:
            return arm["corrected_arm"]
    raise KeyError(f"missing Worklog 129 corrected metrics for {case_name}")


def run_analysis(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    old_root = Path(arguments.worklog128_out)
    wl129_report_path = Path(arguments.worklog129_report)
    field = np.load(arguments.field_cache, allow_pickle=True)
    h = float(field["h"])
    mu = float(field["mu"])
    cases: list[CaseAnalysis] = []
    report_cases: list[dict[str, Any]] = []

    mesh_bundle = np.load(arguments.mesh_cache, allow_pickle=True)
    vertices = np.asarray(mesh_bundle["vertices"], dtype=np.float64)
    configs = (PRIMARY_ROI, SECONDARY_ROI)
    for config in configs:
        case = _load_frozen_case(old_root / config.name, config)
        replay = replay_frozen_fit(case, str(device))
        if not replay.diagnostics["sufficiently_identical"]:
            stopped = {
                "batch": "parameterization / termination / target-coherence attribution",
                "status": "STOPPED_FROZEN_FIT_REPRODUCTION_FAILED",
                "preservation": {
                    "worklog_128_commit": WORKLOG_128_COMMIT,
                    "worklog_129_commit": WORKLOG_129_COMMIT,
                    "existing_modules_modified": False,
                    "second_order_candidate_executed": False,
                },
                "h": h,
                "mu": mu,
                "cases": [{"roi": config.as_json(), "frozen_fit_reproduction": replay.diagnostics}],
                "meeting_verdict": "INCONCLUSIVE",
            }
            (output_root / "parametric_continuation_attribution_report.json").write_text(json.dumps(_jsonable(stopped), indent=2), encoding="utf-8")
            _write_report_readme(output_root, stopped)
            return stopped

        vertex_roi = _roi_vertex_contract(vertices, config)
        trace = trace_mesh_interface(Path(arguments.mesh_cache), vertices, vertex_roi)
        boundary_points, boundary_tangent_v, boundary_normals = _evaluate_boundary(replay.surface, str(device))
        support = boundary_support_report(replay, vertex_roi, boundary_points, h)
        interface = geometric_interface_report(replay, vertex_roi, trace, boundary_points, boundary_tangent_v, boundary_normals, h)
        corrected_points, corrected_normals, corrected_source = _load_corrected_prediction(Path(arguments.worklog129_out), case, replay.surface, str(device))
        from scipy.spatial import cKDTree

        corrected_distances, corrected_nearest = cKDTree(corrected_points).query(case.reference_eval_points, workers=1)
        frozen_metrics = _load_frozen_wl129_metrics(wl129_report_path, config.name)
        target_coherence = target_coherence_report(case, vertex_roi, trace, h)
        target_metrics = target_population_report(case, vertex_roi, trace, corrected_points, corrected_normals, h, frozen_metrics)
        distance_report = distance_to_termination_report(case, trace, corrected_points, corrected_normals, h)
        parameterization = parameterization_report(replay)
        case_report = {
            "roi": config.as_json(),
            "frozen_fit_reproduction": replay.diagnostics,
            "frozen_fit_reproduction_source": "same Worklog 128 observed_points and reconstructed initial UV; no withheld rows",
            "fit_configuration": FIT_KWARGS,
            "parameterization": parameterization,
            "boundary_support": support,
            "geometric_termination_agreement": interface,
            "target_coherence": target_coherence,
            "worklog_129_prediction_source": corrected_source,
            "target_population_evaluation": target_metrics,
            "distance_to_termination": distance_report,
            "reference_and_fitter_contract": {
                "manual_roi_and_cut_unchanged": True,
                "holdout_u_cut": HOLDOUT_CUT,
                "fitter_input_rows": int(len(replay.fit_points)),
                "withheld_xyz_in_fitter": False,
                "full_reference_roles": ["mesh-face interface extraction", "target coherence", "evaluation", "visualisation"],
            },
        }
        cases.append(CaseAnalysis(replay, vertex_roi, trace, boundary_points, boundary_tangent_v, boundary_normals, corrected_points, corrected_normals, corrected_distances, corrected_nearest, case_report))
        report_cases.append(case_report)
        if config.name == PRIMARY_ROI.name:
            _write_boundary_support_figure(output_root / "curved_rim_boundary_support.png", vertex_roi, trace, replay, boundary_points, h)
            _write_qualitative_figure(output_root / "curved_rim_attribution.png", vertex_roi, trace, boundary_points, corrected_points)

    primary = next(item for item in report_cases if item["roi"]["name"] == PRIMARY_ROI.name)
    gate = {
        "frozen_fit_reproduced": all(item["frozen_fit_reproduction"]["sufficiently_identical"] for item in report_cases),
        "final_footpoint_parameterization_stable_near_termination": all(item["parameterization"]["final_u_still_corresponds_to_manual_observed_termination"] for item in report_cases),
        "u1_boundary_observed_supported": all(item["boundary_support"]["direct_observed_support_fraction"]["tsdf_mesh_le_h"] >= 0.50 for item in report_cases),
        "fitted_boundary_agrees_with_fixed_mesh_interface": all(item["geometric_termination_agreement"]["fitted_boundary_coverage_of_interface"]["boundary_points_le_2h"] >= 0.50 for item in report_cases),
        "primary_target_predominantly_one_interface_connected_sheet": bool(primary["target_coherence"]["target_predominantly_one_interface_connected_sheet"]),
    }
    gate["passed"] = bool(all(gate.values()))
    if not gate["passed"]:
        if not gate["final_footpoint_parameterization_stable_near_termination"] or not gate["u1_boundary_observed_supported"] or not gate["fitted_boundary_agrees_with_fixed_mesh_interface"]:
            verdict = "A_PARAMETERIZATION_CONTRACT_FAILED"
        elif not gate["primary_target_predominantly_one_interface_connected_sheet"]:
            verdict = "B_TARGET_COHERENCE_FAILED"
        else:
            verdict = "E_INCONCLUSIVE"
    else:
        verdict = "C_BOUNDARY_TARGET_VALID_FIRST_ORDER_CURVATURE_ATTRIBUTION_PENDING"
    report = {
        "batch": "parameterization / termination / target-coherence attribution before occluded-surface continuation",
        "status": "NON_CANONICAL_ATTRIBUTION_ONLY",
        "intent_alignment": {
            "worklog_128_preserved": True,
            "worklog_129_preserved": True,
            "canonical_production_modified": False,
            "second_order_continuation_added": False,
            "true_occluded_surface_attempted": False,
        },
        "inputs": {
            "worklog_128_output": str(old_root),
            "worklog_129_output": str(arguments.worklog129_out),
            "worklog_129_report": str(wl129_report_path),
            "reference_mesh": str(arguments.mesh_cache),
            "h": h,
            "mu": mu,
            "manual_roi_and_holdout_contract": "fixed Worklog 128/129 ROI axes, bounds, and u_cut=0.58",
            "full_reference_leakage_disclosure": "full vertices/faces define mesh interface and target coherence; withheld XYZ never enters refit or prediction",
        },
        "attribution_gate": gate,
        "cases": report_cases,
        "curvature_attribution": {
            "status": "NOT_EXECUTED",
            "reason": "attribution gate did not pass; second-order continuation was not added",
        },
        "second_order_result": {"status": "NOT_EXECUTED"},
        "meeting_verdict": verdict,
    }
    _write_distance_figure(output_root / "distance_to_termination.png", report_cases)
    (output_root / "parametric_continuation_attribution_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _write_report_readme(output_root, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklog128-out", type=Path, default=REPO_ROOT / "output/128_demo_parametric_surface_continuation")
    parser.add_argument("--worklog129-out", type=Path, default=REPO_ROOT / "output/129_demo_corrected_first_order_parametric_continuation")
    parser.add_argument("--worklog129-report", type=Path, default=REPO_ROOT / "output/129_demo_corrected_first_order_parametric_continuation/corrected_first_order_parametric_continuation_report.json")
    parser.add_argument("--mesh-cache", type=Path, default=REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz")
    parser.add_argument("--field-cache", type=Path, default=REPO_ROOT / "output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/field.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/130_demo_parametric_continuation_attribution")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_analysis(build_arg_parser().parse_args(argv))
    print(json.dumps({"verdict": report["meeting_verdict"], "cases": len(report.get("cases", [])), "gate_passed": report.get("attribution_gate", {}).get("passed", False)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
