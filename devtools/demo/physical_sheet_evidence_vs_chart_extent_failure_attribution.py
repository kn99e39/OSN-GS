"""Worklog 149: physical-sheet evidence versus chart-extent failure attribution.

Isolated diagnostic only. This module replays the frozen WL148/WL145 baseline,
measures fixed-axis extrema leverage and full-PCA orientation leverage for every
renderer event, and exports provenance review imagery. The full reference is
used only for this audit and is never used to filter, refit, or alter a result.
It never filters events, changes support, refits NURBS, or implements
membership/continuation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.wl145_baseline_reconciliation_support_constrained_materialization import (  # noqa: E402
    CAMERAS as WL145_CAMERAS,
    EVENT_ROOT as WL145_EVENT_ROOT,
    FROZEN_REPRESENTATIVE,
    WL139_REPORT,
    WL145_REPORT,
    _load_frozen_baseline,
    _materialized_vertex_indices,
    _sha256_array,
    _topology_accounting,
)
from devtools.demo.physical_chart_surface_representative import FIXED_VIEW, _sha256_rows  # noqa: E402
from devtools.demo.per_view_renderer_surface_correspondence_physical_sheet_oracle_audit import (  # noqa: E402
    _draw_renderer_projected_points,
)
from devtools.demo.real_gaussian_scene_surface_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    _load_canonical_scene,
    _render_to_pil,
    project_world_points,
)

OUTPUT_ROOT = REPO_ROOT / "output" / "149_physical_sheet_evidence_vs_chart_extent_failure_attribution"
WL148_ROOT = REPO_ROOT / "temp" / "148_wl145_baseline_reconciliation_support_constrained_materialization_audit"
WL148_REPORT = WL148_ROOT / "wl145_baseline_reconciliation_support_constrained_materialization_audit_report.json"
WL148_REPLAY = WL148_ROOT / "frozen_inputs" / "wl145_frozen_representative_replay.npz"
WL145_OUTPUT_ROOT = REPO_ROOT / "output" / "confirmed" / "145_genuine_physical_sheet_oracle_clean_support_representative_audit"
WL145_REPORT_PATH = WL145_OUTPUT_ROOT / "genuine_physical_sheet_oracle_clean_support_representative_audit_report.json"
WL145_REPRESENTATIVE = WL145_OUTPUT_ROOT / "tabletop_broad_planar_clean" / "clean_support_representative" / "wl139_frozen_representative.npz"
WL145_EVENT_ROOT = WL145_OUTPUT_ROOT / "tabletop_broad_planar_clean" / "per_view_renderer_median_events"
WL139_REPORT_PATH = REPO_ROOT / "output" / "confirmed" / "139_physical_chart_surface_representative" / "physical_chart_surface_representative_report.json"
WL127_STATE_ARCHIVE = REPO_ROOT / "output" / "confirmed" / "127_wl123_novel_view_observed_occluded_visualization" / "gaussian_center_global_states.npz"
WL127_STATE_REPORT = REPO_ROOT / "output" / "confirmed" / "127_wl123_novel_view_observed_occluded_visualization" / "wl123_fixed_observed_occluded_visualization_report.json"
CAMERAS = tuple(WL145_CAMERAS)
CAMERA_COLORS = {CAMERAS[0]: "#2468e8", CAMERAS[1]: "#e76f24", CAMERAS[2]: "#20a968"}
CAMERA_RGB = {CAMERAS[0]: (36, 104, 232), CAMERAS[1]: (231, 111, 36), CAMERAS[2]: (32, 169, 104)}
OWNER_RGB = (236, 30, 150)
ORACLE_RGB = (255, 190, 0)
NEUTRAL_RGB = (88, 94, 104)
REVIEW_RANK_COUNT = 20


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pca_frame(points: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = points.mean(axis=0)
    _values, vectors = np.linalg.eigh((points - center).T @ (points - center))
    axes = vectors[:, ::-1].copy()
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0:
            axes[:, column] *= -1
    if float(np.dot(np.cross(axes[:, 0], axes[:, 1]), axes[:, 2])) < 0:
        axes[:, 1] *= -1
    return {"centroid": center, "axes": axes, "projected": points @ axes}


def _extent(projected: np.ndarray) -> dict[str, float]:
    low = np.asarray(projected).min(axis=0)
    high = np.asarray(projected).max(axis=0)
    span = high - low
    return {
        "u_min": float(low[0]), "u_max": float(high[0]),
        "v_min": float(low[1]), "v_max": float(high[1]),
        "n_min": float(low[2]), "n_max": float(high[2]),
        "u_span": float(span[0]), "v_span": float(span[1]),
        "rectangular_chart_area": float(span[0] * span[1]),
    }


def _summary(values: Any) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"status": "UNAVAILABLE", "samples": 0}
    return {
        "status": "MEASURED", "samples": int(len(values)),
        "min": float(values.min()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "mean": float(values.mean()),
        "max": float(values.max()),
        "zero_or_near_zero_count": int(np.sum(np.abs(values) <= 1.0e-12)),
    }


def _owners(values: np.ndarray, index: int) -> list[int]:
    target = float(values[index])
    return np.flatnonzero(np.isclose(values, target, atol=1.0e-12, rtol=0.0)).astype(np.int64).tolist()


def _fixed_axis_loo(projected: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original = _extent(projected)
    rows = []
    umin = set(_owners(projected[:, 0], int(np.argmin(projected[:, 0]))))
    umax = set(_owners(projected[:, 0], int(np.argmax(projected[:, 0]))))
    vmin = set(_owners(projected[:, 1], int(np.argmin(projected[:, 1]))))
    vmax = set(_owners(projected[:, 1], int(np.argmax(projected[:, 1]))))
    for event_id in range(len(projected)):
        loo = _extent(np.delete(projected, event_id, axis=0))
        row = {
            "event_id": int(event_id),
            "projected_u": float(projected[event_id, 0]),
            "projected_v": float(projected[event_id, 1]),
            "projected_n": float(projected[event_id, 2]),
            "is_u_min_owner": event_id in umin, "is_u_max_owner": event_id in umax,
            "is_v_min_owner": event_id in vmin, "is_v_max_owner": event_id in vmax,
        }
        for field in ("u_span", "v_span", "rectangular_chart_area"):
            delta = float(loo[field] - original[field])
            row["leave_one_out_delta_" + field] = delta
            row["extent_reduction_" + field] = -delta
        row["relative_area_change"] = float(
            (loo["rectangular_chart_area"] - original["rectangular_chart_area"])
            / original["rectangular_chart_area"]
        )
        rows.append(row)
    ranking = sorted(rows, key=lambda row: (
        -abs(row["extent_reduction_rectangular_chart_area"]),
        -abs(row["extent_reduction_u_span"]),
        -abs(row["extent_reduction_v_span"]),
        row["event_id"],
    ))
    return {
        "original": original,
        "u_span_influence": _summary([r["extent_reduction_u_span"] for r in rows]),
        "v_span_influence": _summary([r["extent_reduction_v_span"] for r in rows]),
        "rectangular_area_influence": _summary([r["extent_reduction_rectangular_chart_area"] for r in rows]),
        "absolute_rectangular_area_influence": _summary([abs(r["extent_reduction_rectangular_chart_area"]) for r in rows]),
        "ranking_is_diagnostic_only": True, "no_keep_reject_threshold": True,
    }, ranking


def _align_axes(base_axes: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    candidates = []
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        aligned = axes * np.asarray(signs, dtype=np.float64)[None, :]
        rotation = base_axes.T @ aligned
        candidates.append((float(np.trace(rotation)), signs, aligned))
    _, signs, aligned = max(candidates, key=lambda item: (item[0], item[1]))
    rotation = base_axes.T @ aligned
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))
    return aligned, np.asarray(signs, dtype=np.int8), angle


def _full_pca_loo(points: np.ndarray, base: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original = _extent(base["projected"])
    rows = []
    for event_id in range(len(points)):
        rest = np.delete(points, event_id, axis=0)
        loo_frame = _pca_frame(rest)
        aligned, signs, joint_angle = _align_axes(base["axes"], loo_frame["axes"])
        loo = _extent(rest @ aligned)
        axis_angles = [
            math.degrees(math.acos(float(np.clip(abs(np.dot(base["axes"][:, i], loo_frame["axes"][:, i])), 0.0, 1.0))))
            for i in range(3)
        ]
        row = {
            "event_id": int(event_id),
            "pca_origin_shift_world": float(np.linalg.norm(loo_frame["centroid"] - base["centroid"])),
            "axis_u_angle_degrees": float(axis_angles[0]),
            "axis_v_angle_degrees": float(axis_angles[1]),
            "axis_n_angle_degrees": float(axis_angles[2]),
            "right_handed_sign_alignment": signs.tolist(),
            "joint_axis_rotation_angle_degrees": float(joint_angle),
        }
        for field in ("u_span", "v_span", "rectangular_chart_area"):
            delta = float(loo[field] - original[field])
            row["leave_one_out_delta_" + field] = delta
            row["extent_reduction_" + field] = -delta
        rows.append(row)
    ranking = sorted(rows, key=lambda row: (
        -abs(row["leave_one_out_delta_rectangular_chart_area"]),
        -row["joint_axis_rotation_angle_degrees"],
        row["event_id"],
    ))
    summary = {
        "original": original,
        "axis_u_angle_distribution_degrees": _summary([r["axis_u_angle_degrees"] for r in rows]),
        "axis_v_angle_distribution_degrees": _summary([r["axis_v_angle_degrees"] for r in rows]),
        "axis_n_angle_distribution_degrees": _summary([r["axis_n_angle_degrees"] for r in rows]),
        "joint_axis_rotation_distribution_degrees": _summary([r["joint_axis_rotation_angle_degrees"] for r in rows]),
        "origin_shift_distribution_world": _summary([r["pca_origin_shift_world"] for r in rows]),
        "u_span_influence": _summary([r["extent_reduction_u_span"] for r in rows]),
        "v_span_influence": _summary([r["extent_reduction_v_span"] for r in rows]),
        "rectangular_area_influence": _summary([r["extent_reduction_rectangular_chart_area"] for r in rows]),
        "absolute_rectangular_area_influence": _summary([abs(r["extent_reduction_rectangular_chart_area"]) for r in rows]),
        "ranking_is_diagnostic_only": True, "no_keep_reject_threshold": True,
    }
    return summary, ranking


def _load_provenance(baseline: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = []
    offset = 0
    for camera, path in zip(CAMERAS, baseline["event_paths"]):
        data = np.load(path, allow_pickle=True)
        count = len(data["event_points_xyz"])
        arrays.append({
            "event_id": np.arange(offset, offset + count, dtype=np.int64),
            "source_camera": np.full(count, camera, dtype=object),
            "source_pixel_x": np.asarray(data["pixel_x"], dtype=np.float64),
            "source_pixel_y": np.asarray(data["pixel_y"], dtype=np.float64),
            "depth": np.asarray(data["renderer_median_depth"], dtype=np.float64),
            "world_xyz": np.asarray(data["event_points_xyz"], dtype=np.float64),
            "normal": np.asarray(data["local_normals"], dtype=np.float64),
        })
        offset += count
    result = {key: np.concatenate([item[key] for item in arrays], axis=0) for key in arrays[0]}
    if not np.array_equal(result["world_xyz"], baseline["oracle_points"]):
        raise AssertionError("WL145 provenance order differs from frozen WL148 event union")
    return result


def _provenance_rows(provenance: dict[str, np.ndarray], projected: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(projected)):
        row = {
            "event_id": int(i), "source_camera": str(provenance["source_camera"][i]),
            "source_pixel_x": float(provenance["source_pixel_x"][i]),
            "source_pixel_y": float(provenance["source_pixel_y"][i]),
            "renderer_median_event_depth": float(provenance["depth"][i]),
            "world_x": float(provenance["world_xyz"][i, 0]),
            "world_y": float(provenance["world_xyz"][i, 1]),
            "world_z": float(provenance["world_xyz"][i, 2]),
            "event_normal_x": float(provenance["normal"][i, 0]),
            "event_normal_y": float(provenance["normal"][i, 1]),
            "event_normal_z": float(provenance["normal"][i, 2]),
            "chart_u": float(projected[i, 0]), "chart_v": float(projected[i, 1]), "chart_n": float(projected[i, 2]),
        }
        rows.append(row)
    return rows


def _merge_rows(provenance_rows: list[dict[str, Any]], fixed: list[dict[str, Any]], pca: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed_map = {r["event_id"]: r for r in fixed}
    pca_map = {r["event_id"]: r for r in pca}
    rows = []
    for row in provenance_rows:
        event_id = row["event_id"]
        merged = dict(row)
        merged.update({"fixed_" + key: value for key, value in fixed_map[event_id].items() if key != "event_id"})
        merged.update({"pca_" + key: value for key, value in pca_map[event_id].items() if key != "event_id"})
        rows.append(merged)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_contracts() -> dict[str, Any]:
    compact = np.array([[x, y, 0.0] for x in (-1, -.5, 0, .5, 1) for y in (-1, -.5, 0, .5, 1)], dtype=np.float64)
    far = np.vstack([compact, [[8, 0, 0]]])
    frame_a = _pca_frame(far)
    _, rows_a = _fixed_axis_loo(frame_a["projected"])
    far_id = len(far) - 1
    row_a = {row["event_id"]: row for row in rows_a}[far_id]
    pass_a = row_a["extent_reduction_u_span"] > 1.0

    base_b = np.array([[x, y, z] for x in (-2, 0, 2) for y in (-2, 0, 2) for z in (-.1, .1)], dtype=np.float64)
    rotating = np.vstack([base_b, [[.5, 0, 4]]])
    frame_b = _pca_frame(rotating)
    fixed_b, rows_b = _fixed_axis_loo(frame_b["projected"])
    pca_b, rows_pca_b = _full_pca_loo(rotating, frame_b)
    rotating_id = len(rotating) - 1
    row_b_fixed = {row["event_id"]: row for row in rows_b}[rotating_id]
    row_b_pca = {row["event_id"]: row for row in rows_pca_b}[rotating_id]
    pass_b = (
        abs(row_b_fixed["extent_reduction_rectangular_chart_area"]) <= 1.0e-12
        and row_b_pca["joint_axis_rotation_angle_degrees"] > 1.0e-3
    )

    duplicated = np.vstack([compact, compact[0], compact[4], compact[-1], compact[20]])
    frame_c = _pca_frame(duplicated)
    _, rows_c = _fixed_axis_loo(frame_c["projected"])
    pass_c = max(abs(r["extent_reduction_rectangular_chart_area"]) for r in rows_c) <= 1.0e-12
    return {
        "fixture_A_compact_planar_plus_far_same_plane": {
            "fixed_axis_extent_leverage_detected": bool(pass_a),
            "point_not_automatically_rejected": True, "far_event_id": far_id,
        },
        "fixture_B_compact_population_plus_rotating_point": {
            "orientation_leverage_distinguished": bool(pass_b),
            "fixed_axis_area_reduction": float(row_b_fixed["extent_reduction_rectangular_chart_area"]),
            "joint_axis_rotation_degrees": float(row_b_pca["joint_axis_rotation_angle_degrees"]),
        },
        "fixture_C_no_dominant_individual_extrema": {
            "no_dominant_individual_extent_leverage": bool(pass_c),
            "automatic_rejection_decision": False,
        },
        "all_synthetic_contracts_pass": bool(pass_a and pass_b and pass_c),
    }


def _save_chart_plot(root: Path, provenance: dict[str, np.ndarray], projected: np.ndarray, fixed_ranking: list[dict[str, Any]], pca_ranking: list[dict[str, Any]], owner_ids: set[int]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = root / "chart_space_attribution.png"
    fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
    for camera in CAMERAS:
        mask = provenance["source_camera"] == camera
        ax.scatter(projected[mask, 0], projected[mask, 1], s=12, alpha=.95, c=CAMERA_COLORS[camera], label=camera, linewidths=0)
    review_ids = sorted(set(int(r["event_id"]) for r in fixed_ranking[:REVIEW_RANK_COUNT]) | set(int(r["event_id"]) for r in pca_ranking[:REVIEW_RANK_COUNT]))
    ax.scatter(projected[list(owner_ids), 0], projected[list(owner_ids), 1], s=90, marker="*", c="#ec1e96", edgecolors="black", label="exact extrema owner")
    ax.scatter(projected[review_ids, 0], projected[review_ids, 1], s=45, marker="o", facecolors="none", edgecolors="black", label="top diagnostic ranking")
    for i in sorted(owner_ids):
        ax.annotate("ID " + str(i), (projected[i, 0], projected[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("WL139 chart u (world projection)")
    ax.set_ylabel("WL139 chart v (world projection)")
    ax.set_title("WL149 chart-space provenance and extrema ownership; no filtering")
    ax.grid(alpha=.2)
    ax.legend(loc="best", framealpha=.94)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return str(path)


def _world_plot(root: Path, baseline: dict[str, Any], provenance: dict[str, np.ndarray], owner_ids: set[int], owner_id: int | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    oracle = baseline["oracle_points"]
    representative = baseline["representative_points"]
    grid = representative.reshape(96, 40, 3)
    full_faces = [grid[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]] for i in range(95) for j in range(39)]
    support_faces = [grid[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]] for i, j in np.argwhere(baseline["cell_mask"])]
    all_points = np.concatenate([oracle, representative])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(high - low, 1.0e-9)
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    for camera in CAMERAS:
        mask = provenance["source_camera"] == camera
        ax.scatter(oracle[mask, 0], oracle[mask, 1], oracle[mask, 2], s=9, alpha=.96, c=CAMERA_COLORS[camera], label="events " + camera, linewidths=0)
    ax.add_collection3d(Poly3DCollection(full_faces, alpha=.12, facecolor="#00aede", edgecolor="none"))
    ax.add_collection3d(Poly3DCollection(support_faces, alpha=.72, facecolor="#26be60", edgecolor="#137738", linewidth=.15))
    selected = [owner_id] if owner_id is not None else sorted(owner_ids)
    points = oracle[selected]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=115, marker="*", c="#ec1e96", edgecolors="black", label="exact extrema owner")
    for i in selected:
        ax.text(float(oracle[i, 0]), float(oracle[i, 1]), float(oracle[i, 2]), " ID " + str(i), color="black")
    pad = .05 * span
    ax.set_xlim(low[0] - pad[0], high[0] + pad[0]); ax.set_ylim(low[1] - pad[1], high[1] + pad[1]); ax.set_zlim(low[2] - pad[2], high[2] + pad[2])
    ax.set_box_aspect(span); ax.view_init(**FIXED_VIEW)
    ax.set_xlabel("world X"); ax.set_ylabel("world Y"); ax.set_zlabel("world Z")
    ax.set_title("WL149 world context" if owner_id is None else "WL149 exact extrema owner ID " + str(owner_id))
    ax.legend(loc="upper left", framealpha=.94)
    path = root / ("common_world_provenance/all_events_representative_and_B.png" if owner_id is None else "owner_world_context/event_" + f"{owner_id:04d}" + "_world_context.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)
    return str(path)


def _mark(image: Any, x: float, y: float, label: str) -> Any:
    from PIL import ImageDraw
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    r = 8
    draw.ellipse((x-r, y-r, x+r, y+r), outline=OWNER_RGB, width=3)
    draw.line((x-12, y, x+12, y), fill=OWNER_RGB, width=2)
    draw.line((x, y-12, x, y+12), fill=OWNER_RGB, width=2)
    draw.rectangle((x+10, y-15, x+15+8*len(label), y+4), fill=(255, 255, 255))
    draw.text((x+13, y-13), label, fill=OWNER_RGB)
    return output


def _crop(image: Any, x: float, y: float, margin: int = 100) -> Any:
    w, h = image.size
    cx, cy = int(round(x)), int(round(y))
    box = (max(0, cx-margin), max(0, cy-margin), min(w, cx+margin+1), min(h, cy+margin+1))
    return image.crop(box).resize(((box[2]-box[0])*2, (box[3]-box[1])*2))


def _tensor_hash(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _save_mandatory_pair(root: Path, model: Any, scene_info: dict[str, Any], camera_lookup: dict[str, Any], images: dict[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    import torch
    from coverage_first_surfel_partition_export import _rgb_to_f_dc
    states = np.asarray(np.load(Path(arguments.state_archive))["global_state"], dtype=np.int8)
    state_report = json.loads(Path(arguments.state_report).read_text(encoding="utf-8"))
    if str(Path(state_report["checkpoint"]).resolve()) != str(Path(arguments.checkpoint).resolve()):
        raise AssertionError("Candidate B state archive checkpoint mismatch")
    if len(states) != int(model.get_xyz.shape[0]):
        raise AssertionError("Candidate B state row count mismatch")
    before = {"get_xyz": _tensor_hash(model.get_xyz)}
    before.update({n: _tensor_hash(getattr(model, n)) for n in ("_xyz", "_scaling", "_rotation", "_opacity") if hasattr(model, n)})
    colours = np.full((len(states), 3), .60, dtype=np.float32)
    colours[states == 1] = (.10, .85, .35); colours[states == 2] = (.92, .18, .18)
    original_dc = model._features_dc.detach().clone()
    original_rest = model._features_rest.detach().clone()
    original_degree = model.active_sh_degree
    observed_paths = {}
    try:
        with torch.no_grad():
            model._features_dc.copy_(_rgb_to_f_dc(torch.as_tensor(colours, dtype=torch.float32, device=model.device)).unsqueeze(1))
            model._features_rest.zero_(); model.active_sh_degree = 0
            for name in CAMERAS:
                path = root / "mandatory_gaussian_visualization_pair" / name / "H_observed_occluded.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                _render_to_pil(scene_info["rasterizer"].render(camera_lookup[name], model)).save(path)
                observed_paths[name] = str(path)
    finally:
        with torch.no_grad():
            model._features_dc.copy_(original_dc); model._features_rest.copy_(original_rest); model.active_sh_degree = original_degree
    after = {"get_xyz": _tensor_hash(model.get_xyz)}
    after.update({n: _tensor_hash(getattr(model, n)) for n in ("_xyz", "_scaling", "_rotation", "_opacity") if hasattr(model, n)})
    if before != after:
        raise AssertionError("Gaussian geometry changed during colour-only visualization")
    original_paths = {}
    for name, image in images.items():
        path = root / "mandatory_gaussian_visualization_pair" / name / "G_original_scene.png"
        path.parent.mkdir(parents=True, exist_ok=True); image.save(path); original_paths[name] = str(path)
    return {"camera_ids": list(CAMERAS), "gaussian_row_count": int(len(states)), "original_scene": original_paths, "observed_occluded": observed_paths, "marker_gaussians_added": 0, "geometry_unchanged": True, "colour_only_override": True, "palette": {"OBSERVED": (.10, .85, .35), "OCCLUDED": (.92, .18, .18), "UNRESOLVED": (.60, .60, .62)}}


def _save_owner_reviews(root: Path, baseline: dict[str, Any], provenance: dict[str, np.ndarray], owner_ids: set[int], images: dict[str, Any], cameras: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for event_id in sorted(owner_ids):
        camera_name = str(provenance["source_camera"][event_id])
        x, y = float(provenance["source_pixel_x"][event_id]), float(provenance["source_pixel_y"][event_id])
        label = "ID " + str(event_id)
        owner_root = root / "extrema_owner_reviews" / f"event_{event_id:04d}_{camera_name}"
        owner_root.mkdir(parents=True, exist_ok=True)
        scene = _mark(images[camera_name], x, y, label)
        clean = _draw_renderer_projected_points(images[camera_name], baseline["oracle_points"], cameras[camera_name], NEUTRAL_RGB, radius=2.8, alpha=.98)
        clean = _mark(clean, x, y, label)
        scene_path = owner_root / "source_camera_gaussian_event_marked.png"
        clean_path = owner_root / "source_camera_clean_oracle_event_id.png"
        crop_path = owner_root / "source_camera_local_crop.png"
        scene.save(scene_path); clean.save(clean_path); _crop(clean, x, y).save(crop_path)
        cross = {}
        for target_name in CAMERAS:
            projection = project_world_points(baseline["oracle_points"][event_id:event_id+1], cameras[target_name])
            if not bool(projection["valid"][0]):
                continue
            target = _mark(images[target_name], float(projection["x"][0]), float(projection["y"][0]), label)
            target_path = owner_root / "cross_view_projection" / f"projected_to_{target_name}.png"
            target_path.parent.mkdir(parents=True, exist_ok=True); target.save(target_path); cross[target_name] = str(target_path)
        output[str(event_id)] = {
            "event_id": event_id, "source_camera": camera_name, "source_pixel": [x, y],
            "source_camera_gaussian_render": str(scene_path), "source_camera_clean_oracle_overlay": str(clean_path),
            "readable_local_crop": str(crop_path), "cross_view_projection_localization": cross,
            "review_status": "AMBIGUOUS",
            "review_basis": "Frozen provenance has camera/pixel/depth/XYZ/normal but no primitive/contributor identity or ground-truth physical-sheet label.",
        }
    return output


def _load_wl148_artifact(baseline: dict[str, Any]) -> dict[str, Any]:
    if not WL148_REPORT.exists() or not WL148_REPLAY.exists():
        raise FileNotFoundError("committed WL148 temp artifact is missing")
    report = json.loads(WL148_REPORT.read_text(encoding="utf-8"))
    replay = np.load(WL148_REPLAY, allow_pickle=True)
    expected = report["COMMITTED WL145 EXACT REPLAY"]
    if int(expected["event_union_count"]) != len(baseline["oracle_points"]) or str(expected["event_union_sha256"]) != _sha256_rows(baseline["oracle_points"]):
        raise AssertionError("WL148 report baseline mismatch")
    for key, baseline_key in (("event_union_points", "oracle_points"), ("sampled_points", "representative_points"), ("sampled_normals", "representative_normals"), ("support_vertex_mask", "support_vertices"), ("materializable_cells", "cell_mask")):
        np.testing.assert_array_equal(replay[key], baseline[baseline_key])
    return {"report_path": str(WL148_REPORT.resolve()), "report_sha256": _sha256_file(WL148_REPORT), "replay_path": str(WL148_REPLAY.resolve()), "replay_sha256": _sha256_file(WL148_REPLAY), "exact_replay": True, "report_baseline": _jsonable(expected)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--wl145-report", type=Path, default=WL145_REPORT_PATH)
    parser.add_argument("--representative", type=Path, default=WL145_REPRESENTATIVE)
    parser.add_argument("--wl139-report", type=Path, default=WL139_REPORT_PATH)
    parser.add_argument("--event-root", type=Path, default=WL145_EVENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--state-archive", type=Path, default=WL127_STATE_ARCHIVE)
    parser.add_argument("--state-report", type=Path, default=WL127_STATE_REPORT)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-scene", action="store_true")
    return parser


def run_audit(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.out); root.mkdir(parents=True, exist_ok=True)
    baseline = _load_frozen_baseline(Path(arguments.wl145_report), Path(arguments.representative), Path(arguments.wl139_report), Path(arguments.event_root))
    wl148 = _load_wl148_artifact(baseline)
    provenance = _load_provenance(baseline)
    frame = _pca_frame(baseline["oracle_points"])
    cfg_axes = np.column_stack([baseline["config"].u_axis, baseline["config"].v_axis, baseline["config"].n_axis])
    if not np.allclose(frame["axes"], cfg_axes, atol=1.0e-12, rtol=0):
        raise AssertionError("WL145 PCA frame replay differs from frozen chart axes")
    projected = frame["projected"]; extent = _extent(projected)
    extrema = {
        "u_min_owner_ids": _owners(projected[:, 0], int(np.argmin(projected[:, 0]))),
        "u_max_owner_ids": _owners(projected[:, 0], int(np.argmax(projected[:, 0]))),
        "v_min_owner_ids": _owners(projected[:, 1], int(np.argmin(projected[:, 1]))),
        "v_max_owner_ids": _owners(projected[:, 1], int(np.argmax(projected[:, 1]))),
    }
    owner_ids = set(sum(extrema.values(), []))
    fixed_summary, fixed_ranking = _fixed_axis_loo(projected)
    pca_summary, pca_ranking = _full_pca_loo(baseline["oracle_points"], frame)
    p_rows = _provenance_rows(provenance, projected)
    rows = _merge_rows(p_rows, fixed_ranking, pca_ranking)
    for row in rows:
        row["review_extrema_owner"] = int(row["event_id"]) in owner_ids
        row["fixed_area_rank"] = next(i for i, r in enumerate(fixed_ranking, 1) if r["event_id"] == row["event_id"])
        row["pca_area_rank"] = next(i for i, r in enumerate(pca_ranking, 1) if r["event_id"] == row["event_id"])
        row["fixed_area_leverage_is_not_a_decision"] = True
        row["pca_orientation_is_not_a_decision"] = True
    baseline_report = {
        "event_count": len(baseline["oracle_points"]), "event_union_sha256": _sha256_rows(baseline["oracle_points"]),
        "camera_counts": baseline["baseline"]["event_camera_counts"], "representative_shape": list(baseline["representative_points"].shape),
        "representative_xyz_sha256": _sha256_array(baseline["representative_points"]), "representative_normals_sha256": _sha256_array(baseline["representative_normals"]),
        "support_vertices": int(np.sum(baseline["support_vertices"])), "unsupported_vertices": int(np.sum(~baseline["support_vertices"])),
        "support_mask_sha256": _sha256_array(baseline["support_vertices"].astype(np.uint8)), "fully_supported_cells": int(np.sum(baseline["cell_mask"])),
        "wl148_temp_artifact": wl148,
    }
    (root / "baseline_reconciliation.json").write_text(json.dumps(_jsonable(baseline_report), indent=2), encoding="utf-8")
    extent_report = {
        "pca_centroid_origin_world_xyz": frame["centroid"],
        "coordinate_origin_convention": "WL145 projects world XYZ by axis columns without subtracting centroid; centroid is reported for PCA covariance only",
        "pca_basis_axes_columns_world_xyz": frame["axes"],
        "projected_coordinate_convention": "world_xyz @ basis_axes; columns u,v,n",
        "original_projected_coordinate_bounds": extent,
        **extrema,
        "original_u_span": extent["u_span"], "original_v_span": extent["v_span"],
        "original_rectangular_chart_coordinate_area": extent["rectangular_chart_area"],
        "owner_status_term": "extrema owner / high-leverage sample; not called outlier",
        "owner_provenance": [r for r in p_rows if r["event_id"] in owner_ids],
    }
    chart_full = {**extent_report, "fixed_axis_leave_one_out": fixed_summary, "full_pca_leave_one_out": pca_summary, "fixed_axis_top_diagnostic_ranking": fixed_ranking[:REVIEW_RANK_COUNT], "full_pca_top_diagnostic_ranking": pca_ranking[:REVIEW_RANK_COUNT]}
    (root / "chart_attribution.json").write_text(json.dumps(_jsonable(chart_full), indent=2), encoding="utf-8")
    (root / "full_per_point_influence.json").write_text(json.dumps(_jsonable(rows), indent=2), encoding="utf-8")
    _write_csv(root / "full_per_point_influence.csv", rows)
    np.savez_compressed(root / "full_per_point_influence.npz", event_id=np.arange(len(rows), dtype=np.int64), projected_uvn=projected, fixed_area_reduction=np.asarray([r["fixed_extent_reduction_rectangular_chart_area"] for r in rows]), pca_area_reduction=np.asarray([r["pca_extent_reduction_rectangular_chart_area"] for r in rows]), pca_axis_angles_degrees=np.asarray([[r["pca_axis_u_angle_degrees"], r["pca_axis_v_angle_degrees"], r["pca_axis_n_angle_degrees"]] for r in rows]))
    chart_plot = _save_chart_plot(root, provenance, projected, fixed_ranking, pca_ranking, owner_ids)
    common_plot = _world_plot(root, baseline, provenance, owner_ids)
    for event_id in sorted(owner_ids):
        _world_plot(root, baseline, provenance, owner_ids, owner_id=event_id)
    synthetic = _synthetic_contracts()
    (root / "synthetic_contract_results.json").write_text(json.dumps(_jsonable(synthetic), indent=2), encoding="utf-8")
    if arguments.skip_scene:
        scene = {"status": "SKIPPED_BY_ARGUMENT"}
    else:
        model, payload, all_cameras, scene_info = _load_canonical_scene(Path(arguments.checkpoint), Path(arguments.source_path), arguments.device, arguments.images, arguments.sparse_dir, int(arguments.resolution), int(arguments.llffhold))
        lookup = {str(c.image_name): c for c in all_cameras}
        if any(name not in lookup for name in CAMERAS):
            raise AssertionError("required frozen camera missing")
        images = {name: _render_to_pil(scene_info["rasterizer"].render(lookup[name], model)) for name in CAMERAS}
        pair = _save_mandatory_pair(root, model, scene_info, lookup, images, arguments)
        owner_reviews = _save_owner_reviews(root, baseline, provenance, owner_ids, images, lookup)
        scene = {"status": "EXPORTED", "checkpoint": str(Path(arguments.checkpoint).resolve()), "checkpoint_iteration": payload.get("iteration"), "camera_ids": list(CAMERAS), "camera_resolution": {name: [int(lookup[name].image_width), int(lookup[name].image_height)] for name in CAMERAS}, "mandatory_pair": pair, "extrema_owner_reviews": owner_reviews, "same_renderer_background_and_framing": True}
    owner_status = {str(i): {"event_id": i, "status": "AMBIGUOUS", "basis": "No stored primitive/contributor identity or ground-truth physical-sheet label; camera/pixel/depth/XYZ/normal alone is insufficient."} for i in sorted(owner_ids)}
    if scene.get("extrema_owner_reviews"):
        for key, value in scene["extrema_owner_reviews"].items():
            owner_status[key]["exported_review_bundle"] = value
    camera_accounting = {}
    for camera in CAMERAS:
        camera_rows = [r for r in rows if r["source_camera"] == camera]
        camera_accounting[camera] = {"event_count": len(camera_rows), "exact_extrema_owner_ids": sorted(r["event_id"] for r in camera_rows if r["event_id"] in owner_ids), "chart_u_summary": _summary([r["chart_u"] for r in camera_rows]), "chart_v_summary": _summary([r["chart_v"] for r in camera_rows])}
    provenance_report = {"event_count": len(rows), "event_id_definition": "row order in frozen WL145 per-view union, camera order DSC08043, DSC07960, DSC08003", "fields_available": ["event_id", "source_camera", "source_pixel_x", "source_pixel_y", "renderer_median_event_depth", "world_xyz", "event_normal", "chart_u", "chart_v", "chart_n"], "missing_required_fields": [], "exact_extrema_owner_ids": sorted(owner_ids), "exact_extrema_owner_rows": [r for r in rows if r["event_id"] in owner_ids], "camera_accounting": camera_accounting, "complete_influence_ranking_available": True}
    verdict = {"architecture_verdict": "UNRESOLVED", "next_frontier": "obtain defensible physical-sheet/contributor identity before ordering membership-first versus chart-first correction", "mechanically_established": ["global PCA plus coordinate extrema produces the frozen rectangle", "fixed-axis and full-PCA LOO distributions are available for all 1586 events", "exact extrema owners and provenance review bundle are exported"], "not_established": ["owner events are not automatically proven wrong-sheet", "owner events are not automatically proven legitimate same-sheet evidence", "no filtering or chart heuristic is justified"], "owner_review_status": owner_status, "evidence_limitation": "frozen provenance has camera/pixel/depth/XYZ/normal but no primitive/contributor identity or ground-truth physical-sheet label", "synthetic_pass_does_not_establish_real_scene_result": True}
    (root / "renderer_event_provenance.json").write_text(json.dumps(_jsonable(provenance_report), indent=2), encoding="utf-8")
    (root / "architecture_verdict.json").write_text(json.dumps(_jsonable(verdict), indent=2), encoding="utf-8")
    report = {
        "batch": "Worklog 149 physical-sheet evidence versus chart-extent failure attribution",
        "status": "COMPLETED_ISOLATED_NON_CANONICAL_DIAGNOSTIC",
        "INTENT ALIGNMENT": {"diagnostic_only": True, "baseline_preserved": True, "canonical_production_changed": False, "wl148_temp_artifact_modified": False, "event_population_filtered": False, "chart_optimized_or_reduced": False},
        "AGENT INTERPRETATION OF INTENT": "Attribute giant chart extent to evidence contamination, extrema-chart sensitivity, both, or unresolved without introducing a filter or chart heuristic.",
        "BASELINE RECONCILIATION": baseline_report,
        "IMPLEMENTATION FIDELITY": {"fixed_pca_axes_are_original_axes": True, "fixed_axis_loo_only_changes_extrema": True, "full_pca_loo_recomputes_pca_per_omission": True, "pca_sign_ambiguity_handled": True, "all_1586_points_reported": True, "provenance_loaded_from_frozen_npz": True, "withheld_reference_geometry_evaluation_only": True, "membership_threshold_added": False, "outlier_filter_added": False, "robust_pca_added": False, "percentile_clipping_added": False, "support_mask_changed": False, "representative_refit": False, "nurbs_changed": False, "graphness_used_as_membership": False, "appearance_or_sh_used_as_membership": False, "continuation": False, "occluded_surface": False, "candidate_b_modified": False, "gaussian_visualization_geometry_unchanged": scene.get("mandatory_pair", {}).get("geometry_unchanged")},
        "EXTREMA OWNERSHIP": extent_report,
        "FIXED-PCA EXTENT LEVERAGE": fixed_summary,
        "PCA ORIENTATION LEVERAGE": pca_summary,
        "RENDERER-EVENT PROVENANCE": provenance_report,
        "REAL-SCENE PHYSICAL-SHEET REVIEW": {"status": scene.get("status"), "exact_extrema_owner_reviews": owner_status, "review_rule": "CLEAR_INTENDED_TABLETOP_EVIDENCE / CLEAR_WRONG_SHEET_OR_STRUCTURE / AMBIGUOUS; all exact owners remain AMBIGUOUS from stored evidence", "chart_plot": chart_plot, "common_world_plot": common_plot, "scene_exports": scene, "two_dimensional_footprint_not_used_as_proof": True},
        "FAILURE DECOMPOSITION": {"A_EVIDENCE_POPULATION": {"total_events": len(rows), "per_camera": camera_accounting, "world_xyz_bbox": np.ptp(provenance["world_xyz"], axis=0), "world_xyz_centroid": provenance["world_xyz"].mean(axis=0), "membership_threshold_used": False}, "B_PCA_ORIENTATION": pca_summary, "C_EXTREMA_DEFINED_CHART": {"original": extent, "exact_owner_ids": sorted(owner_ids), "fixed_axis": fixed_summary, "full_pca": pca_summary}, "D_SUPPORTED_MATERIALIZATION": {"support_vertices": int(np.sum(baseline["support_vertices"])), "support_mask_sha256": _sha256_array(baseline["support_vertices"].astype(np.uint8)), "fully_supported_cells": int(np.sum(baseline["cell_mask"])), "component_sizes": _topology_accounting(baseline["cell_mask"])["materialized_region_sizes"], "relationship_only": True}},
        "SYNTHETIC CONTRACT RESULTS": synthetic,
        "ARCHITECTURE RESULT": {"verdict": "UNRESOLVED", "reason": "Chart mechanical sensitivity is established, but frozen provenance does not establish whether influential events belong to the intended tabletop sheet or another structure.", "next_batch_requirement": verdict["next_frontier"]},
        "RETAINED": ["WL145/WL148 event union, representative, support occupancy, h, mu", "global PCA/extrema chart baseline", "full per-point fixed-axis/full-PCA influence", "camera/pixel/depth/XYZ/normal provenance"],
        "REJECTED": ["event filtering", "outlier label as input", "robust PCA", "percentile clipping", "chart shrinking", "support threshold", "refit", "automatic Surface Membership", "continuation", "Occluded Surface"],
        "OPEN": ["physical primitive/contributor identity for influential events", "human review of exact owner source-camera crops and world context", "whether next correction is upstream, downstream, or both"],
        "OUTPUTS": {"baseline_reconciliation": str((root / "baseline_reconciliation.json").resolve()), "chart_attribution": str((root / "chart_attribution.json").resolve()), "full_per_point_json": str((root / "full_per_point_influence.json").resolve()), "full_per_point_csv": str((root / "full_per_point_influence.csv").resolve()), "full_per_point_npz": str((root / "full_per_point_influence.npz").resolve()), "chart_space": chart_plot, "common_world": common_plot, "renderer_event_provenance": str((root / "renderer_event_provenance.json").resolve()), "architecture_verdict": str((root / "architecture_verdict.json").resolve()), "scene": scene},
    }
    report_path = root / "physical_sheet_evidence_vs_chart_extent_failure_attribution_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (root / "README.md").write_text("# Worklog 149 attribution bundle\n\nFrozen WL148 event/representative를 변경하지 않고 fixed-axis extrema leverage와 full-PCA orientation leverage를 분리했다. Exact extrema owner의 camera crop, provenance, world context를 제공한다.\n\n자동 filter, robust PCA, chart trimming, membership threshold, refit, continuation, Occluded Surface는 실행하지 않았다. Verdict는 UNRESOLVED이며 owner review가 필요하다.\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run_audit(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_verdict": report["ARCHITECTURE RESULT"]["verdict"], "event_count": report["BASELINE RECONCILIATION"]["event_count"], "exact_extrema_owner_ids": sorted(set(report["EXTREMA OWNERSHIP"]["u_min_owner_ids"] + report["EXTREMA OWNERSHIP"]["u_max_owner_ids"] + report["EXTREMA OWNERSHIP"]["v_min_owner_ids"] + report["EXTREMA OWNERSHIP"]["v_max_owner_ids"])), "synthetic_pass": report["SYNTHETIC CONTRACT RESULTS"]["all_synthetic_contracts_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
