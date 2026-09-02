"""Worklog 148: replay WL145 and compare two frozen materializations.

This is an isolated, non-canonical audit.  It loads the exact WL145 frozen
oracle and representative artifact, verifies the committed 314/3840 support
baseline, and then materializes either the full 96 x 40 sample rectangle or
only cells whose four existing support vertices are occupied.  No fitting,
chart recomputation, mask repair, trimming, continuation, membership, or
Candidate B operation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.genuine_physical_sheet_oracle_clean_support_representative_audit import (  # noqa: E402
    _domain_accounting,
)
from devtools.demo.meeting_occluded_surface_feasibility import Box  # noqa: E402
from devtools.demo.per_view_renderer_surface_correspondence_physical_sheet_oracle_audit import (  # noqa: E402
    _draw_renderer_projected_points,
)
from devtools.demo.physical_chart_surface_representative import (  # noqa: E402
    FIXED_VIEW,
    SAMPLE_COUNT_U,
    SAMPLE_COUNT_V,
    _case_coordinates,
    _sha256_rows,
)
from devtools.demo.real_gaussian_scene_surface_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    _load_canonical_scene,
    _render_to_pil,
)
from devtools.demo.scale_separated_visible_surface_representative import (  # noqa: E402
    RepresentativeCaseConfig,
)


OUTPUT_ROOT = REPO_ROOT / "output" / "148_wl145_baseline_reconciliation_support_constrained_materialization_audit"
WL145_OUTPUT_ROOT = REPO_ROOT / "output" / "145_genuine_physical_sheet_oracle_clean_support_representative_audit"
WL145_REPORT = WL145_OUTPUT_ROOT / "genuine_physical_sheet_oracle_clean_support_representative_audit_report.json"
FROZEN_REPRESENTATIVE = (
    WL145_OUTPUT_ROOT
    / "tabletop_broad_planar_clean"
    / "clean_support_representative"
    / "wl139_frozen_representative.npz"
)
WL139_REPORT = REPO_ROOT / "output" / "confirmed" / "139_physical_chart_surface_representative" / "physical_chart_surface_representative_report.json"
WL127_STATE_ARCHIVE = REPO_ROOT / "output" / "confirmed" / "127_wl123_novel_view_observed_occluded_visualization" / "gaussian_center_global_states.npz"
WL127_STATE_REPORT = REPO_ROOT / "output" / "confirmed" / "127_wl123_novel_view_observed_occluded_visualization" / "wl123_fixed_observed_occluded_visualization_report.json"

EVENT_UNION_COUNT = 1586
EVENT_UNION_SHA256 = "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78"
REPRESENTATIVE_COUNT = SAMPLE_COUNT_U * SAMPLE_COUNT_V
SUPPORT_COUNT = 314
UNSUPPORTED_COUNT = 3526
SUPPORT_MASK_SHA256 = "23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9"
FROZEN_REPRESENTATIVE_XYZ_SHA256 = "5fe79de62c6842cb02a99fa940f2e3ffa7fcd4c51165db606c693029aa59c941"
FROZEN_REPRESENTATIVE_NORMALS_SHA256 = "e01b51ddc2bc43586fdb16f1772c0f3a4e99ec46a81bfbb8202f8c78377ad05b"
FROZEN_FIT_POINTS_SHA256 = EVENT_UNION_SHA256
EXPECTED_WL145_FULLY_SUPPORTED_CELLS = 211

CAMERAS = ("DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG")
EVENT_ROOT = (
    WL145_OUTPUT_ROOT
    / "tabletop_broad_planar_clean"
    / "per_view_renderer_median_events"
)
ORACLE_RGB = (255, 211, 0)
FULL_DOMAIN_RGB = (0, 174, 220)
SUPPORT_DOMAIN_RGB = (38, 190, 96)
NORMALIZED_GRAY = (0.60, 0.60, 0.62)
OBSERVED_RGB = (0.10, 0.85, 0.35)
OCCLUDED_RGB = (0.92, 0.18, 0.18)
UNRESOLVED_RGB = (0.60, 0.60, 0.62)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_array(value: np.ndarray, *, dtype: Any | None = None) -> str:
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _config_from_json(payload: dict[str, Any]) -> RepresentativeCaseConfig:
    box = payload["roi_box"]
    return RepresentativeCaseConfig(
        name=str(payload["name"]),
        semantic_label=str(payload["semantic_label"]),
        roi_box=Box(tuple(box["lower"]), tuple(box["upper"])),
        u_axis=tuple(payload["u_axis"]),
        v_axis=tuple(payload["v_axis"]),
        n_axis=tuple(payload["n_axis"]),
        u_bounds=tuple(payload["u_bounds"]),
        v_bounds=tuple(payload["v_bounds"]),
        n_bounds=tuple(payload["n_bounds"]),
        u_cut=float(payload["u_cut"]),
        frontier_source=str(payload["frontier_source"]),
    )


def _four_connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return deterministic 4-connected components of a boolean grid."""

    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for seed in map(tuple, np.argwhere(mask)):
        if visited[seed]:
            continue
        queue = deque([seed])
        visited[seed] = True
        members: list[tuple[int, int]] = []
        while queue:
            i, j = queue.popleft()
            members.append((int(i), int(j)))
            for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                ni, nj = neighbour
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
        components.append(np.asarray(members, dtype=np.int64))
    return components


def _cell_mask_from_support(support_vertices: np.ndarray) -> np.ndarray:
    """Use the exact committed WL145 all-four occupancy relation."""

    occupied = np.asarray(support_vertices, dtype=bool)
    if occupied.shape != (SAMPLE_COUNT_U, SAMPLE_COUNT_V):
        raise ValueError(f"expected {(SAMPLE_COUNT_U, SAMPLE_COUNT_V)} support vertices, got {occupied.shape}")
    return (
        occupied[:-1, :-1]
        & occupied[1:, :-1]
        & occupied[:-1, 1:]
        & occupied[1:, 1:]
    )


def _materialized_vertex_indices(cell_mask: np.ndarray) -> np.ndarray:
    vertices = np.zeros((SAMPLE_COUNT_U, SAMPLE_COUNT_V), dtype=bool)
    vertices[:-1, :-1] |= cell_mask
    vertices[1:, :-1] |= cell_mask
    vertices[:-1, 1:] |= cell_mask
    vertices[1:, 1:] |= cell_mask
    return np.flatnonzero(vertices.reshape(-1)).astype(np.int64)


def _cell_areas(sampled_points: np.ndarray) -> np.ndarray:
    """Reproduce WL145's existing parallelogram cell-area convention."""

    grid = np.asarray(sampled_points, dtype=np.float64).reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    du = np.diff(grid, axis=0)[:, :-1]
    dv = np.diff(grid, axis=1)[:-1, :]
    return np.linalg.norm(np.cross(du, dv), axis=2)


def _topology_accounting(cell_mask: np.ndarray) -> dict[str, Any]:
    components = _four_connected_components(cell_mask)
    rejected = ~np.asarray(cell_mask, dtype=bool)
    rejected_components = _four_connected_components(rejected)
    holes = [component for component in rejected_components if not np.any(
        (component[:, 0] == 0)
        | (component[:, 0] == rejected.shape[0] - 1)
        | (component[:, 1] == 0)
        | (component[:, 1] == rejected.shape[1] - 1)
    )]
    isolated = 0
    for i, j in map(tuple, np.argwhere(cell_mask)):
        neighbours = [
            (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
        ]
        if not any(0 <= ni < cell_mask.shape[0] and 0 <= nj < cell_mask.shape[1] and cell_mask[ni, nj] for ni, nj in neighbours):
            isolated += 1
    return {
        "cell_grid_shape": [int(cell_mask.shape[0]), int(cell_mask.shape[1])],
        "fully_supported_cells": int(np.sum(cell_mask)),
        "rejected_cells": int(np.sum(rejected)),
        "materialized_connected_regions": int(len(components)),
        "materialized_region_sizes": sorted((int(len(component)) for component in components), reverse=True),
        "isolated_cells": int(isolated),
        "rejected_connected_regions": int(len(rejected_components)),
        "holes": int(len(holes)),
        "hole_sizes": sorted((int(len(component)) for component in holes), reverse=True),
        "connectivity": "4-connected",
        "hole_definition": "rejected-cell component not touching the cell-grid border",
    }


def _distance_summary(values: np.ndarray, h: float, mu: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "NO_SAMPLES", "samples": 0}
    median = float(np.median(values))
    p95 = float(np.percentile(values, 95))
    mean = float(np.mean(values))
    return {
        "status": "MEASURED",
        "samples": int(len(values)),
        "median": median,
        "p95": p95,
        "mean": mean,
        "median_over_h": median / h,
        "p95_over_h": p95 / h,
        "mean_over_h": mean / h,
        "median_over_mu": median / mu,
        "p95_over_mu": p95 / mu,
        "mean_over_mu": mean / mu,
    }


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(source) or not len(target):
        return np.full((len(source),), np.inf, dtype=np.float64)
    return cKDTree(target).query(source, k=1, workers=1)[0]


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(source) or not len(target):
        return np.full((len(source),), -1, dtype=np.int64)
    return cKDTree(target).query(source, k=1, workers=1)[1].astype(np.int64)


def _normal_error_summary(
    source_points: np.ndarray,
    source_normals: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    h: float,
    mu: float,
) -> dict[str, Any]:
    """Measure normal error at the same nearest-vertex correspondence as XYZ."""

    source_points = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    source_normals = np.asarray(source_normals, dtype=np.float64).reshape(-1, 3)
    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    target_normals = np.asarray(target_normals, dtype=np.float64).reshape(-1, 3)
    if not len(source_points) or not len(target_points) or not len(target_normals):
        return {"status": "UNAVAILABLE", "samples": 0, "reason": "empty normal population"}
    indices = _nearest_indices(source_points, target_points)
    valid = (
        (indices >= 0)
        & (np.linalg.norm(source_normals, axis=1) > 1.0e-12)
        & (np.linalg.norm(target_normals[indices], axis=1) > 1.0e-12)
    )
    if not np.any(valid):
        return {"status": "UNAVAILABLE", "samples": 0, "reason": "no finite source/target normals"}
    source_unit = source_normals[valid] / np.linalg.norm(source_normals[valid], axis=1, keepdims=True)
    target_unit = target_normals[indices[valid]] / np.linalg.norm(target_normals[indices[valid]], axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(source_unit * target_unit, axis=1)), 0.0, 1.0)))
    return {
        "status": "MEASURED",
        "samples": int(len(angles)),
        "median_degrees": float(np.median(angles)),
        "p95_degrees": float(np.percentile(angles, 95)),
        "mean_degrees": float(np.mean(angles)),
        "distance_correspondence": "nearest materialized representative vertex",
        "h": float(h),
        "mu": float(mu),
    }


def _arm_metrics(
    name: str,
    cell_mask: np.ndarray,
    oracle_points: np.ndarray,
    representative_points: np.ndarray,
    oracle_normals: np.ndarray,
    representative_normals: np.ndarray,
    cell_areas: np.ndarray,
    h: float,
    mu: float,
) -> dict[str, Any]:
    vertex_indices = _materialized_vertex_indices(cell_mask)
    materialized_points = representative_points[vertex_indices]
    oracle_to_surface = _nearest_distances(oracle_points, materialized_points)
    surface_to_oracle = _nearest_distances(materialized_points, oracle_points)
    materialized_area = float(np.sum(cell_areas[cell_mask]))
    return {
        "name": name,
        "materialized_vertex_count": int(len(materialized_points)),
        "materialized_vertex_indices_sha256": _sha256_array(vertex_indices, dtype=np.int64),
        "materialized_area": materialized_area,
        "oracle_to_materialized_surface": _distance_summary(oracle_to_surface, h, mu),
        "materialized_surface_to_oracle": _distance_summary(surface_to_oracle, h, mu),
        "normal_angular_error": _normal_error_summary(
            oracle_points,
            oracle_normals,
            materialized_points,
            representative_normals[vertex_indices],
            h,
            mu,
        ),
        "fixed_reference_coverage": {
            "oracle_points": int(len(oracle_points)),
            "within_h_count": int(np.sum(oracle_to_surface <= h)),
            "within_h_fraction": float(np.mean(oracle_to_surface <= h)),
            "within_mu_count": int(np.sum(oracle_to_surface <= mu)),
            "within_mu_fraction": float(np.mean(oracle_to_surface <= mu)),
        },
        "domain_accounting": _topology_accounting(cell_mask),
        "materialized_geometry_source": "existing frozen representative vertices incident to materializable cells",
        "distance_contract": "nearest vertex of the exported materialized mesh; no new/interpolated geometry",
        "new_geometry_fitted": False,
    }


def _load_frozen_baseline(
    report_path: Path = WL145_REPORT,
    representative_path: Path = FROZEN_REPRESENTATIVE,
    wl139_report_path: Path = WL139_REPORT,
    event_root: Path = EVENT_ROOT,
) -> dict[str, Any]:
    """Load and gate the exact committed WL145 baseline before any A/B work."""

    for path in (report_path, representative_path, wl139_report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    wl139_report = json.loads(wl139_report_path.read_text(encoding="utf-8"))
    case = report["cases"]["tabletop_broad_planar_clean"]
    representative_result = case["conditional_wl139_representative"]
    chart_payload = case["post_validation_chart"]
    config = _config_from_json(chart_payload)
    frozen = np.load(representative_path)
    required = {"fit_points", "sampled_points", "sampled_normals"}
    if not required.issubset(set(frozen.files)):
        raise AssertionError(f"frozen representative is missing {required - set(frozen.files)}")
    fit_points = np.asarray(frozen["fit_points"], dtype=np.float64)
    representative_points = np.asarray(frozen["sampled_points"], dtype=np.float64)
    representative_normals = np.asarray(frozen["sampled_normals"], dtype=np.float64)

    event_paths = [event_root / camera / "event_cloud_with_provenance.npz" for camera in CAMERAS]
    missing_events = [path for path in event_paths if not path.exists()]
    if missing_events:
        raise FileNotFoundError(", ".join(str(path) for path in missing_events))
    event_arrays = [np.load(path, allow_pickle=True) for path in event_paths]
    event_union = np.concatenate(
        [np.asarray(array["event_points_xyz"], dtype=np.float64) for array in event_arrays],
        axis=0,
    )
    event_normals = np.concatenate(
        [np.asarray(array["local_normals"], dtype=np.float64) for array in event_arrays],
        axis=0,
    )
    if not np.array_equal(event_union, fit_points):
        raise AssertionError("frozen fit_points are not the exact ordered WL145 per-view event union")
    if event_normals.shape != event_union.shape:
        raise AssertionError("WL145 per-view event normals do not match event union shape")

    event_report = case["clean_support"]
    if (
        int(event_report["point_count"]) != EVENT_UNION_COUNT
        or str(event_report.get("event_union_sha256")) != EVENT_UNION_SHA256
        or _sha256_rows(event_union) != EVENT_UNION_SHA256
    ):
        raise AssertionError("WL145 exact event union replay failed")
    if len(event_union) != EVENT_UNION_COUNT:
        raise AssertionError("WL145 event union count failed")
    if _sha256_rows(fit_points) != FROZEN_FIT_POINTS_SHA256:
        raise AssertionError("frozen fit_points fingerprint failed")
    if len(representative_points) != REPRESENTATIVE_COUNT:
        raise AssertionError("frozen representative sample count failed")
    if representative_points.shape != representative_normals.shape or representative_points.shape != (REPRESENTATIVE_COUNT, 3):
        raise AssertionError("frozen representative point/normal shape failed")
    if _sha256_array(representative_points) != FROZEN_REPRESENTATIVE_XYZ_SHA256:
        raise AssertionError("frozen representative XYZ fingerprint failed")
    if _sha256_array(representative_normals) != FROZEN_REPRESENTATIVE_NORMALS_SHA256:
        raise AssertionError("frozen representative normal fingerprint failed")

    h = float(wl139_report["IMPLEMENTATION FIDELITY"]["h"])
    mu = float(wl139_report["IMPLEMENTATION FIDELITY"]["mu"])
    if float(report["OPERATIONAL_CHOICES"]["h"]) != h or float(report["OPERATIONAL_CHOICES"]["mu"]) != mu:
        raise AssertionError("WL145 and WL139 h/mu operational values differ")
    # This call is the committed WL145 accounting implementation. It is a
    # read-only replay; fitting is intentionally absent from this module.
    domain = _domain_accounting(event_union, config, representative_points)
    support_vertices = np.asarray(domain.pop("_support_vertex_mask"), dtype=bool)
    support_hash = _sha256_array(support_vertices.astype(np.uint8))
    if int(domain["supported_vertices"]) != SUPPORT_COUNT or int(domain["unsupported_vertices"]) != UNSUPPORTED_COUNT:
        raise AssertionError("WL145 exact support vertex count replay failed")
    if support_hash != SUPPORT_MASK_SHA256:
        raise AssertionError("WL145 exact support-mask hash replay failed")
    cell_mask = _cell_mask_from_support(support_vertices)
    fully_supported_cells = int(np.sum(cell_mask))
    if fully_supported_cells != EXPECTED_WL145_FULLY_SUPPORTED_CELLS:
        raise AssertionError("WL145 all-four cell replay failed")
    if int(domain["supported_regions_four_connected"]) != fully_supported_cells:
        raise AssertionError("WL145 report cell accounting mismatch")

    return {
        "report": report,
        "wl139_report": wl139_report,
        "case": case,
        "representative_result": representative_result,
        "config": config,
        "oracle_points": event_union,
        "oracle_normals": event_normals,
        "fit_points": fit_points,
        "event_paths": [str(path) for path in event_paths],
        "representative_points": representative_points,
        "representative_normals": representative_normals,
        "support_vertices": support_vertices,
        "cell_mask": cell_mask,
        "h": h,
        "mu": mu,
        "domain": domain,
        "baseline": {
            "event_union_count": EVENT_UNION_COUNT,
            "event_union_sha256": EVENT_UNION_SHA256,
            "event_camera_counts": {
                camera: int(len(array["event_points_xyz"]))
                for camera, array in zip(CAMERAS, event_arrays)
            },
            "representative_sample_count": REPRESENTATIVE_COUNT,
            "supported_vertices": SUPPORT_COUNT,
            "unsupported_vertices": UNSUPPORTED_COUNT,
            "support_mask_sha256": SUPPORT_MASK_SHA256,
            "fully_supported_cells": fully_supported_cells,
            "report_supported_regions_four_connected": int(domain["supported_regions_four_connected"]),
            "event_normals_available": True,
        },
    }


def _write_mesh_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz"):
            handle.write(f"property float {name}\n")
        for name in ("red", "green", "blue"):
            handle.write(f"property uchar {name}\n")
        handle.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for point, normal in zip(points, normals):
            handle.write(" ".join(f"{float(value):.9g}" for value in (*point, *normal)))
            handle.write(f" {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _mesh_faces(cell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.argwhere(np.asarray(cell_mask, dtype=bool))
    used = _materialized_vertex_indices(cell_mask)
    remap = {int(index): position for position, index in enumerate(used.tolist())}
    faces: list[tuple[int, int, int]] = []
    for i, j in cells.tolist():
        p00 = int(np.ravel_multi_index((i, j), (SAMPLE_COUNT_U, SAMPLE_COUNT_V)))
        p10 = int(np.ravel_multi_index((i + 1, j), (SAMPLE_COUNT_U, SAMPLE_COUNT_V)))
        p11 = int(np.ravel_multi_index((i + 1, j + 1), (SAMPLE_COUNT_U, SAMPLE_COUNT_V)))
        p01 = int(np.ravel_multi_index((i, j + 1), (SAMPLE_COUNT_U, SAMPLE_COUNT_V)))
        faces.extend(((remap[p00], remap[p10], remap[p11]), (remap[p00], remap[p11], remap[p01])))
    return used, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _save_mesh_outputs(root: Path, baseline: dict[str, Any], arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    points = baseline["representative_points"]
    normals = baseline["representative_normals"]
    oracle = baseline["oracle_points"]
    output: dict[str, Any] = {}
    for key, color in (("A_full_domain", FULL_DOMAIN_RGB), ("B_support_constrained", SUPPORT_DOMAIN_RGB)):
        mask = np.ones((SAMPLE_COUNT_U - 1, SAMPLE_COUNT_V - 1), dtype=bool) if key.startswith("A") else baseline["cell_mask"]
        used, faces = _mesh_faces(mask)
        path = root / "materialized_meshes" / f"{key}.ply"
        _write_mesh_ply(path, points[used], normals[used], faces, color)
        output[key] = {
            "ply": str(path),
            "vertex_count": int(len(used)),
            "face_count": int(len(faces)),
            "vertex_indices_sha256": _sha256_array(used, dtype=np.int64),
            "exact_representative_xyz": True,
            "exact_representative_normals": True,
        }
    oracle_path = root / "materialized_meshes" / "clean_oracle.ply"
    _write_mesh_ply(oracle_path, oracle, baseline["oracle_normals"], np.empty((0, 3), dtype=np.int64), ORACLE_RGB)
    output["clean_oracle"] = {"ply": str(oracle_path), "vertex_count": int(len(oracle)), "event_union_sha256": _sha256_rows(oracle)}

    replay_path = root / "frozen_inputs" / "wl145_frozen_representative_replay.npz"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        replay_path,
        fit_points=baseline["fit_points"],
        event_union_points=baseline["oracle_points"],
        event_union_normals=baseline["oracle_normals"],
        sampled_points=baseline["representative_points"],
        sampled_normals=baseline["representative_normals"],
        support_vertex_mask=baseline["support_vertices"],
        materializable_cells=baseline["cell_mask"],
    )
    output["frozen_input_replay"] = {
        "npz": str(replay_path),
        "event_union_exact_fit_points": True,
        "representative_xyz_exact": True,
        "representative_normals_exact": True,
    }
    return output


def _save_chart_diagnostic(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = baseline["config"]
    oracle = baseline["oracle_points"]
    coords = _case_coordinates(oracle, config)
    uv = np.column_stack([
        (coords[:, 0] - config.u_bounds[0]) / (config.u_bounds[1] - config.u_bounds[0]),
        (coords[:, 1] - config.v_bounds[0]) / (config.v_bounds[1] - config.v_bounds[0]),
    ])
    uv_index = np.clip(np.floor(uv * np.array([SAMPLE_COUNT_U, SAMPLE_COUNT_V])).astype(np.int64), 0, np.array([SAMPLE_COUNT_U - 1, SAMPLE_COUNT_V - 1]))
    support = baseline["support_vertices"]
    cells = baseline["cell_mask"]
    rejected = ~cells
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), dpi=180)
    ax = axes[0, 0]
    ax.scatter(uv_index[:, 0], uv_index[:, 1], s=5, c="#e6a700", alpha=0.8, linewidths=0)
    ax.set_title("raw oracle UV samples (binned index)")
    ax.set_xlim(-0.5, SAMPLE_COUNT_U - 0.5)
    ax.set_ylim(-0.5, SAMPLE_COUNT_V - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("u vertex index")
    ax.set_ylabel("v vertex index")
    for ax, image, title, cmap in (
        (axes[0, 1], support.T, "occupied/support vertices", "Greens"),
        (axes[1, 0], cells.T, "fully-supported cells (all four True)", "Greens"),
        (axes[1, 1], rejected.T, "rejected cells", "Reds"),
    ):
        ax.imshow(image, origin="lower", interpolation="nearest", aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("u index")
        ax.set_ylabel("v index")
    figure.suptitle("WL145 frozen support semantics: 96 x 40; no smoothing/fill")
    figure.tight_layout()
    path = root / "chart_space_96x40_diagnostic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    np.savez_compressed(
        root / "chart_space_96x40_diagnostic.npz",
        oracle_uv=uv,
        oracle_uv_binned_indices=uv_index,
        occupied_support_vertices=support,
        fully_supported_cells=cells,
        rejected_cells=rejected,
    )
    return {
        "png": str(path),
        "npz": str(root / "chart_space_96x40_diagnostic.npz"),
        "grid_shape": [SAMPLE_COUNT_U, SAMPLE_COUNT_V],
        "raw_oracle_uv_samples": int(len(uv)),
        "smoothing_or_fill": False,
    }


def _plot_common_world(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    oracle = baseline["oracle_points"]
    points = baseline["representative_points"]
    cell_mask = baseline["cell_mask"]
    grid = points.reshape(SAMPLE_COUNT_U, SAMPLE_COUNT_V, 3)
    support_faces: list[np.ndarray] = []
    for i, j in np.argwhere(cell_mask):
        support_faces.append(grid[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]])
    all_points = np.concatenate([oracle, points], axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    span = np.maximum(upper - lower, 1.0e-9)
    lower = lower - 0.04 * span
    upper = upper + 0.04 * span

    def save(name: str, title: str, *, show_full: bool, show_support: bool) -> Path:
        figure = plt.figure(figsize=(9.0, 7.0), dpi=180)
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(oracle[:, 0], oracle[:, 1], oracle[:, 2], s=7, c="#e6a700", alpha=0.98, linewidths=0, label="clean oracle")
        if show_full:
            surface = Poly3DCollection([grid[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]] for i in range(SAMPLE_COUNT_U - 1) for j in range(SAMPLE_COUNT_V - 1)], alpha=0.42, facecolor="#00aede", edgecolor="none", label="full-domain representative")
            axis.add_collection3d(surface)
        if show_support:
            surface = Poly3DCollection(support_faces, alpha=0.86, facecolor="#26be60", edgecolor="#137738", linewidth=0.15, label="support-constrained materialization")
            axis.add_collection3d(surface)
        axis.set_xlim(float(lower[0]), float(upper[0]))
        axis.set_ylim(float(lower[1]), float(upper[1]))
        axis.set_zlim(float(lower[2]), float(upper[2]))
        axis.set_box_aspect(span)
        axis.view_init(**FIXED_VIEW)
        axis.set_title(title)
        axis.set_xlabel("world X")
        axis.set_ylabel("world Y")
        axis.set_zlabel("world Z")
        axis.legend(loc="upper right")
        path = root / "common_world_matched" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return path

    paths = {
        "A_full_domain": save("A_full_domain.png", "A: clean oracle + full-domain representative", show_full=True, show_support=False),
        "B_support_constrained": save("B_support_constrained.png", "B: clean oracle + frozen support-constrained materialization", show_full=False, show_support=True),
        "A_B_matched": save("A_B_matched.png", "A/B matched common-world view", show_full=True, show_support=True),
    }
    return {key: str(value) for key, value in paths.items()}


def _tensor_hash(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _save_real_scene_exports(root: Path, baseline: dict[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    model, payload, all_cameras, scene_info = _load_canonical_scene(
        Path(arguments.checkpoint), Path(arguments.source_path), arguments.device,
        arguments.images, arguments.sparse_dir, int(arguments.resolution), int(arguments.llffhold),
    )
    camera_lookup = {str(camera.image_name): camera for camera in all_cameras}
    missing = [name for name in CAMERAS if name not in camera_lookup]
    if missing:
        raise AssertionError(f"required WL145 cameras missing: {missing}")
    state_archive = Path(arguments.state_archive)
    state_report = Path(arguments.state_report)
    states = np.asarray(np.load(state_archive)["global_state"], dtype=np.int8)
    if not state_report.exists():
        raise FileNotFoundError(state_report)
    state_report_payload = json.loads(state_report.read_text(encoding="utf-8"))
    expected_checkpoint = str(Path(state_report_payload["checkpoint"]).resolve())
    actual_checkpoint = str(Path(arguments.checkpoint).resolve())
    if expected_checkpoint != actual_checkpoint:
        raise AssertionError("Candidate B state archive checkpoint differs from requested frozen checkpoint")
    gaussian_count = int(model.get_xyz.shape[0])
    if len(states) != gaussian_count:
        raise AssertionError(f"Candidate B state row count {len(states)} != checkpoint Gaussian count {gaussian_count}")
    geometry_before = {name: _tensor_hash(getattr(model, name)) for name in ("_xyz", "_scaling", "_rotation", "_opacity") if hasattr(model, name)}
    # The public model field is get_xyz; protect it explicitly as well.
    geometry_before["get_xyz"] = _tensor_hash(model.get_xyz)
    images: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for name in CAMERAS:
        camera = camera_lookup[name]
        image = _render_to_pil(scene_info["rasterizer"].render(camera, model))
        images[name] = image
        oracle = baseline["oracle_points"]
        representative = baseline["representative_points"]
        support_indices = _materialized_vertex_indices(baseline["cell_mask"])
        support = representative[support_indices]
        view_root = root / "real_scene_camera_review" / name
        view_root.mkdir(parents=True, exist_ok=True)
        a = image.convert("RGB")
        b = _draw_renderer_projected_points(a, oracle, camera, ORACLE_RGB, radius=2.6, alpha=0.99)
        c = _draw_renderer_projected_points(a, representative, camera, FULL_DOMAIN_RGB, radius=2.1, alpha=0.98)
        d = _draw_renderer_projected_points(a, support, camera, SUPPORT_DOMAIN_RGB, radius=2.6, alpha=0.99)
        e = _draw_renderer_projected_points(b, representative, camera, FULL_DOMAIN_RGB, radius=2.1, alpha=0.98)
        f = _draw_renderer_projected_points(b, support, camera, SUPPORT_DOMAIN_RGB, radius=2.6, alpha=0.99)
        named = {
            "A_gaussian_scene": a,
            "B_gaussian_plus_clean_oracle": b,
            "C_gaussian_plus_full_domain_representative": c,
            "D_gaussian_plus_support_constrained_materialization": d,
            "E_gaussian_plus_clean_oracle_plus_full_domain": e,
            "F_gaussian_plus_clean_oracle_plus_support_constrained": f,
        }
        paths = {}
        for key, output_image in named.items():
            path = view_root / f"{key}.png"
            output_image.save(path)
            paths[key] = str(path)
        (view_root / "README.md").write_text(
            "# Real-scene matched camera review\n\n"
            "이 camera 폴더의 A-F는 동일한 frozen Gaussian render를 배경으로 한다. "
            "A는 Gaussian Scene, B는 clean oracle, C는 full-domain representative, "
            "D는 frozen support-constrained materialization, E/F는 oracle과 각 representative의 결합이다.\n\n"
            "overlay는 review용이며 Gaussian row·checkpoint·renderer를 변경하지 않는다.\n",
            encoding="utf-8",
        )
        outputs[name] = paths

    # Mandatory fixed Gaussian visualization pair.  The state archive is
    # frozen Candidate B; only in-memory display colour tensors are replaced.
    from coverage_first_surfel_partition_export import _rgb_to_f_dc  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if np.any(~np.isin(states, np.array([0, 1, 2], dtype=np.int8))):
        raise AssertionError("unexpected Candidate B state code")
    colours = np.full((gaussian_count, 3), UNRESOLVED_RGB, dtype=np.float32)
    colours[states == 1] = OBSERVED_RGB
    colours[states == 2] = OCCLUDED_RGB
    original_dc = model._features_dc.detach().clone()
    original_rest = model._features_rest.detach().clone()
    original_degree = model.active_sh_degree
    pair_outputs: dict[str, Any] = {}
    try:
        with torch.no_grad():
            model._features_dc.copy_(_rgb_to_f_dc(torch.as_tensor(colours, dtype=torch.float32, device=model.device)).unsqueeze(1))
            model._features_rest.zero_()
            model.active_sh_degree = 0
            for name in CAMERAS:
                camera = camera_lookup[name]
                pair = _render_to_pil(scene_info["rasterizer"].render(camera, model))
                path = root / "mandatory_gaussian_visualization_pair" / name / "H_observed_occluded.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                pair.save(path)
                pair_outputs[name] = str(path)
    finally:
        with torch.no_grad():
            model._features_dc.copy_(original_dc)
            model._features_rest.copy_(original_rest)
            model.active_sh_degree = original_degree
    geometry_after = {name: _tensor_hash(getattr(model, name)) for name in ("_xyz", "_scaling", "_rotation", "_opacity") if hasattr(model, name)}
    geometry_after["get_xyz"] = _tensor_hash(model.get_xyz)
    if geometry_before != geometry_after:
        raise AssertionError("Gaussian geometry changed during visualization export")
    original_paths = {}
    for name in CAMERAS:
        path = root / "mandatory_gaussian_visualization_pair" / name / "G_original_scene.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        images[name].save(path)
        original_paths[name] = str(path)
        (path.parent / "README.md").write_text(
            "# Mandatory Gaussian visualization pair\n\n"
            "G_original_scene.png와 H_observed_occluded.png는 동일한 checkpoint, "
            "iteration, camera, resolution, background, renderer 및 Gaussian row를 사용한다. "
            "H는 색상만 Candidate B global state로 바꾼 결과다.\n\n"
            "초록=OBSERVED, 빨강=OCCLUDED, 회색=UNRESOLVED. marker Gaussian과 geometry 변경은 없다.\n",
            encoding="utf-8",
        )
    return {
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "checkpoint_iteration": payload.get("iteration"),
        "camera_ids": list(CAMERAS),
        "camera_resolution": {name: [int(camera_lookup[name].image_width), int(camera_lookup[name].image_height)] for name in CAMERAS},
        "A_to_F": outputs,
        "mandatory_original_scene": original_paths,
        "mandatory_observed_occluded": pair_outputs,
        "mandatory_pair_contract": {
            "same_checkpoint": True,
            "same_iteration": True,
            "same_camera_framing": True,
            "same_resolution": True,
            "same_renderer_and_background": True,
            "gaussian_row_count": gaussian_count,
            "observed_count": int(np.sum(states == 1)),
            "occluded_count": int(np.sum(states == 2)),
            "unresolved_count": int(np.sum(states == 0)),
            "colour_only_override": True,
            "marker_gaussians_added": 0,
            "protected_geometry_unchanged": True,
            "palette": {"OBSERVED": OBSERVED_RGB, "OCCLUDED": OCCLUDED_RGB, "UNRESOLVED": UNRESOLVED_RGB},
        },
    }



def _save_preview_pngs(root: Path, scene_output: dict[str, Any]) -> dict[str, str]:
    """Create the shared top-level preview_png folder used by output review."""

    from PIL import Image, ImageDraw

    preview = root / "preview_png"
    preview.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    chart = root / "chart_space_96x40_diagnostic.png"
    common = root / "common_world_matched" / "A_B_matched.png"
    for source, name in ((chart, "chart_space_96x40_diagnostic.png"), (common, "common_world_A_B_matched.png")):
        if source.exists():
            target = preview / name
            Image.open(source).convert("RGB").save(target)
            outputs[name] = str(target)
    if scene_output.get("status") == "SKIPPED_BY_ARGUMENT":
        return outputs
    keys = [
        "A_gaussian_scene",
        "B_gaussian_plus_clean_oracle",
        "C_gaussian_plus_full_domain_representative",
        "D_gaussian_plus_support_constrained_materialization",
        "E_gaussian_plus_clean_oracle_plus_full_domain",
        "F_gaussian_plus_clean_oracle_plus_support_constrained",
    ]
    for camera_name, paths in scene_output["A_to_F"].items():
        images = [Image.open(paths[key]).convert("RGB") for key in keys]
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        sheet = Image.new("RGB", (2 * width, 3 * height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (key, image) in enumerate(zip(keys, images)):
            x = (index % 2) * width
            y = (index // 2) * height
            sheet.paste(image, (x, y))
            draw.rectangle((x + 4, y + 4, x + 360, y + 28), fill="white")
            draw.text((x + 8, y + 7), key, fill="black")
        target = preview / f"real_scene_{camera_name}_A_to_F.png"
        sheet.save(target)
        outputs[target.name] = str(target)
        pair_images = [
            Image.open(scene_output["mandatory_original_scene"][camera_name]).convert("RGB"),
            Image.open(scene_output["mandatory_observed_occluded"][camera_name]).convert("RGB"),
        ]
        pair = Image.new("RGB", (pair_images[0].width * 2, pair_images[0].height), "white")
        pair.paste(pair_images[0], (0, 0))
        pair.paste(pair_images[1], (pair_images[0].width, 0))
        target = preview / f"mandatory_{camera_name}_G_H.png"
        pair.save(target)
        outputs[target.name] = str(target)
    return outputs


def _git_baseline() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = "UNAVAILABLE"
    return {
        "history": [
            "41bcc661 worklog144",
            "6f7482e worklog145 introduced",
            "6197563 output-path normalization only",
            "1f10a96 worklog147 mismatch documentation only",
        ],
        "wl145_implementation_commit": "6f7482e",
        "current_head_at_export": head,
        "output_is_gitignored": True,
        "historical_wl145_preserved": True,
    }


def run_audit(arguments: argparse.Namespace) -> dict[str, Any]:
    # Baseline gate happens before output creation and before scene loading.
    baseline = _load_frozen_baseline(Path(arguments.wl145_report), Path(arguments.representative), Path(arguments.wl139_report), Path(arguments.event_root))
    output_root = Path(arguments.out)
    output_root.mkdir(parents=True, exist_ok=True)
    cell_areas = _cell_areas(baseline["representative_points"])
    full_cells = np.ones_like(baseline["cell_mask"], dtype=bool)
    arms = {
        "A_full_domain": _arm_metrics("A_full_domain", full_cells, baseline["oracle_points"], baseline["representative_points"], baseline["oracle_normals"], baseline["representative_normals"], cell_areas, baseline["h"], baseline["mu"]),
        "B_support_constrained": _arm_metrics("B_support_constrained", baseline["cell_mask"], baseline["oracle_points"], baseline["representative_points"], baseline["oracle_normals"], baseline["representative_normals"], cell_areas, baseline["h"], baseline["mu"]),
    }
    area_a = arms["A_full_domain"]["materialized_area"]
    area_b = arms["B_support_constrained"]["materialized_area"]
    arms["B_support_constrained"]["area_accounting"] = {
        "area_retained_by_arm_b": area_b,
        "area_removed_from_arm_a": area_a - area_b,
        "removed_fraction_from_arm_a": (area_a - area_b) / area_a if area_a else None,
        "retained_fraction_of_arm_a": area_b / area_a if area_a else None,
        "area_convention": "WL145 existing per-cell ||du x dv|| convention",
    }
    arms["A_full_domain"]["area_accounting"] = {
        "area_retained_by_arm_b": None,
        "area_removed_from_arm_a": 0.0,
        "removed_fraction_from_arm_a": 0.0,
        "retained_fraction_of_arm_a": 1.0,
        "area_convention": "WL145 existing per-cell ||du x dv|| convention",
    }
    mesh_outputs = _save_mesh_outputs(output_root, baseline, arms)
    (output_root / "materialized_meshes" / "README.md").write_text(
        "# Materialized mesh exports\n\n"
        "A는 frozen representative의 전체 96x40 cell rectangle, B는 WL145 exact all-four "
        "support cell만 사용한다. vertex는 기존 representative row이며 새 vertex/fit은 없다.\n",
        encoding="utf-8",
    )
    chart_output = _save_chart_diagnostic(output_root, baseline)
    world_output = _plot_common_world(output_root, baseline)
    (output_root / "common_world_matched" / "README.md").write_text(
        "# Common-world matched views\n\n"
        "모든 그림은 동일한 world XYZ limits와 fixed camera view를 사용한다. "
        "A/B의 차이는 materialized cell set뿐이다.\n",
        encoding="utf-8",
    )
    if arguments.skip_scene:
        scene_output: dict[str, Any] = {"status": "SKIPPED_BY_ARGUMENT"}
    else:
        scene_output = _save_real_scene_exports(output_root, baseline, arguments)
    preview_output = _save_preview_pngs(output_root, scene_output)

    b_cov = arms["B_support_constrained"]["fixed_reference_coverage"]
    b_surface_error = arms["B_support_constrained"]["materialized_surface_to_oracle"]
    case_a = bool(
        b_cov["within_h_fraction"] >= arms["A_full_domain"]["fixed_reference_coverage"]["within_h_fraction"]
        and b_surface_error["median_over_h"] < arms["A_full_domain"]["materialized_surface_to_oracle"]["median_over_h"]
        and arms["B_support_constrained"]["materialized_area"] < arms["A_full_domain"]["materialized_area"]
    )
    report: dict[str, Any] = {
        "batch": "Worklog 148 committed WL145 baseline reconciliation and support-constrained materialization audit",
        "status": "COMPLETED_ISOLATED_NON_CANONICAL_AUDIT",
        "AGENT INTERPRETATION OF INTENT": "Separate the mathematical full parametric rectangle from an evidence-supported visible-surface materialization while holding the WL145/WL139 representative geometry exactly fixed.",
        "PROMPT-REQUIRED DECISIONS": {
            "historical_prose_support": "248 / 3840",
            "committed_exact_replay_support": "314 / 3840",
            "subsequent_quantitative_experiments_use": "314 / 3840 and exact support-mask hash",
            "reason": "314 is reproduced by the committed 6f7482e _domain_accounting implementation and the preserved WL145 artifact; 248 has no committed executable provenance.",
            "arm_b_name": "FROZEN SUPPORT-CONSTRAINED MATERIALIZATION",
            "occupancy_relation": "all four existing support vertices must be True",
        },
        "AGENT-INTRODUCED OPERATIONAL CHOICES": {
            "distance_population": "exact existing representative sample vertices incident to the materialized cells; cKDTree nearest-neighbour distances",
            "topology_connectivity": "4-connected cell components",
            "holes": "rejected cell components not touching the cell-grid border",
            "area": "the existing WL145 per-cell ||du x dv|| convention",
            "real_scene_cameras": list(CAMERAS),
            "no_threshold_tuning": True,
        },
        "IMPLEMENTATION ASSUMPTIONS": {
            "oracle": "frozen NPZ fit_points, exactly the 1586-point verified renderer-event union",
            "representative": "frozen NPZ sampled_points and sampled_normals, 96 x 40 row-major grid",
            "materialization_surface": "cell-incident existing sample vertices plus diagnostic mesh faces; no interpolated vertices",
            "qualitative_review": "exported matched camera and common-world views remain review evidence, not automatic membership",
        },
        "IMPLEMENTATION FIDELITY STATEMENT": {
            "fitting_called_after_frozen_load": False,
            "representative_xyz_shared_by_a_and_b": True,
            "representative_normals_shared_by_a_and_b": True,
            "support_occupancy_changed": False,
            "pca_recomputed": False,
            "nurbs_refit": False,
            "automatic_surface_membership": False,
            "sh_or_appearance_selection": False,
            "continuation": False,
            "occluded_surface": False,
            "candidate_b_modified": False,
        },
        "GIT / BASELINE RECONCILIATION": _git_baseline(),
        "248 PROSE DISCREPANCY CHECK": {
            "historical_prose_value": 248,
            "current_preserved_wl145_json_field": "supported_regions_four_connected",
            "current_json_field_value": int(baseline["domain"]["supported_regions_four_connected"]),
            "fully_supported_cell_count_from_committed_relation": int(np.sum(baseline["cell_mask"])),
            "historical_248_equals_fully_supported_cell_count": False,
            "conclusion": "248 remains prose-only / provenance-unverified.",
            "implementation_changed": False,
        },
        "COMMITTED WL145 EXACT REPLAY": baseline["baseline"],
        "SUPPORT ANNOTATION SEMANTICS": {
            "definition": "UV occupancy on a 96 x 40 raster",
            "not_final_canonical_surface_membership": True,
            "not_geometric_visibility_truth": True,
            "not_distance_derived_support_field": True,
            "not_canonical_trimmed_nurbs_domain": True,
            "purpose": "diagnostic test of Parametric Domain != Evidence-Supported Visible Surface Domain",
        },
        "FULL-DOMAIN BASELINE": arms["A_full_domain"],
        "FROZEN SUPPORT-CONSTRAINED MATERIALIZATION": arms["B_support_constrained"],
        "QUANTITATIVE A/B": arms,
        "COVERAGE ACCOUNTING": {name: arm["fixed_reference_coverage"] for name, arm in arms.items()},
        "AREA ACCOUNTING": {name: arm["area_accounting"] for name, arm in arms.items()},
        "DOMAIN TOPOLOGY": {name: arm["domain_accounting"] for name, arm in arms.items()},
        "REAL-SCENE QUALITATIVE A/B": {
            "status": scene_output.get("status", "EXPORTED"),
            "identical_camera_framing": True,
            "exports": scene_output,
            "review_note": "A-F overlays show the same rendered Gaussian background; only diagnostic geometry overlays differ. Mandatory Original Scene/Observed-Occluded pair is exported separately with fixed palette and identical Gaussian rows.",
        },
        "CHART-SPACE DIAGNOSTIC": chart_output,
        "COMMON-WORLD MATCHED VIEWS": world_output,
        "PREVIEW_PNG": preview_output,
        "ARCHITECTURE ATTRIBUTION": {
            "case": "A_DOMAIN_SEPARATION_HYPOTHESIS_SUPPORTED" if case_a else "D_MIXED_OR_INCONCLUSIVE",
            "basis": "B is evaluated with unchanged frozen geometry; attribution is based on measured coverage, surface-to-oracle error, and removed unsupported rectangle area.",
            "representative_geometry_changed": False,
            "occupancy_rule_promoted": False,
        },
        "PROMOTED": ["Parametric Domain and Evidence-Supported Visible Surface Domain as distinct audit objects"] if case_a else [],
        "RETAINED": ["UV occupancy annotation as diagnostic baseline only", "WL145 oracle/provenance", "WL139 frozen representative XYZ/normals", "h and mu", "renderer/checkpoint/cameras", "Candidate B"],
        "REJECTED": ["248-based replacement mask", "support-mask tuning", "canonical NURBS trimming", "automatic Surface Membership", "refit", "continuation", "Occluded Surface"],
        "OPEN": ["final canonical Surface Membership contract", "whether UV occupancy is sufficient beyond this tabletop control", "independent qualitative review of real-scene overlays"],
        "FROZEN_INPUTS": {
            "wl145_report": str(Path(arguments.wl145_report).resolve()),
            "frozen_representative": str(Path(arguments.representative).resolve()),
            "wl139_report": str(Path(arguments.wl139_report).resolve()),
            "representative_xyz_sha256": FROZEN_REPRESENTATIVE_XYZ_SHA256,
            "representative_normals_sha256": FROZEN_REPRESENTATIVE_NORMALS_SHA256,
            "fit_points_sha256": FROZEN_FIT_POINTS_SHA256,
            "support_mask_sha256": SUPPORT_MASK_SHA256,
            "h": baseline["h"],
            "mu": baseline["mu"],
        },
        "outputs": {"mesh": mesh_outputs, "chart": chart_output, "common_world": world_output, "real_scene": scene_output, "preview_png": preview_output},
    }
    report_path = output_root / "wl145_baseline_reconciliation_support_constrained_materialization_audit_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(
        "# Worklog 148 materialization audit\n\n"
        "이 output은 WL145 exact baseline을 먼저 검증한 뒤 동일 frozen WL139 representative에 대해 full-domain(A)와 frozen support-constrained(B)를 비교한 격리 진단이다.\n\n"
        "- baseline: 1586 event union, 314/3840 support vertices, exact support-mask hash\n"
        "- B cell rule: existing support vertices 네 개가 모두 True인 cell만 materialize\n"
        "- A/B geometry: 동일한 frozen representative XYZ/normals\n"
        "- `chart_space_96x40_diagnostic.png`는 smoothing/fill 없는 occupancy semantics를 직접 표시한다.\n"
        "- 실제 장면의 A-F overlay와 mandatory Original Scene/Observed-Occluded pair는 `real_scene_camera_review/` 및 `mandatory_gaussian_visualization_pair/`에 있다.\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl145-report", type=Path, default=WL145_REPORT)
    parser.add_argument("--representative", type=Path, default=FROZEN_REPRESENTATIVE)
    parser.add_argument("--event-root", type=Path, default=EVENT_ROOT)
    parser.add_argument("--wl139-report", type=Path, default=WL139_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--state-archive", type=Path, default=WL127_STATE_ARCHIVE)
    parser.add_argument("--state-report", type=Path, default=WL127_STATE_REPORT)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-scene", action="store_true", help="skip CUDA real-scene exports; baseline and A/B metrics still run")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_audit(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_case": report["ARCHITECTURE ATTRIBUTION"]["case"], "baseline": report["COMMITTED WL145 EXACT REPLAY"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
