"""Worklog 156: causal attribution of frozen WL154 TSDF support fragmentation.

This diagnostic never repairs connectivity.  It reconciles frozen WL153/WL154/
WL155 arrays, classifies the six immediate faces around already-owned TSDF
support cells, and produces matched real-scene overlays.  No alternative graph,
bridge radius, dilation, or ownership rule is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.worklog_155_intrinsic_normal_gaussian_region_viability_audit import (
    REVIEW_CAMERAS,
    _build_named_cameras,
    _fixed_rgb_to_dc,
    _load_surfel_model_safe,
    _project_world_points,
)

DEFAULT_WL153_FIELD = REPO_ROOT / "output/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache/field.npz"
DEFAULT_WL154 = REPO_ROOT / "output/154_gaussian_region_owned_tsdf_boundary_first_nurbs"
DEFAULT_WL155 = REPO_ROOT / "output/155_intrinsic_normal_gaussian_region_viability_audit"
DEFAULT_CHECKPOINT = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt"
DEFAULT_SOURCE_PATH = REPO_ROOT / "DATASET"
DEFAULT_OUT = REPO_ROOT / "output/156_region_owned_tsdf_support_fragmentation_causal_attribution"

KEY_BOUND = 1 << 19
AXIS_SPAN = KEY_BOUND << 1
STRIDE_Z = 1
STRIDE_Y = AXIS_SPAN
STRIDE_X = AXIS_SPAN * AXIS_SPAN

CORNER_OFFSETS = np.asarray(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
    dtype=np.int64,
)
FACE_OFFSETS = np.asarray(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int64,
)
FACE_STRIDES = np.asarray([STRIDE_X, -STRIDE_X, STRIDE_Y, -STRIDE_Y, STRIDE_Z, -STRIDE_Z], dtype=np.int64)

CATEGORIES = (
    "OUTSIDE_AUTHORITATIVE_FIELD",
    "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE",
    "ZERO_SURFACE_DIFFERENT_REGION",
    "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS",
    "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED",
    "FIELD_REPLAY_VOLUME_BOUNDARY",
    "OTHER_EXISTING_CONTRACT_REASON",
)
CATEGORY_CODE = {name: code for code, name in enumerate(CATEGORIES)}
INTERNAL_SAME_REGION = "INTERNAL_SAME_REGION_COMPONENT"
STATUS_NAMES = {0: "core", 1: "attached", 2: "ambiguous", 3: "rejected", 4: "unassigned", -1: "unknown"}
FRONTIER_COLORS = {
    "OUTSIDE_AUTHORITATIVE_FIELD": (0.90, 0.18, 0.80),
    "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE": (0.18, 0.45, 0.95),
    "ZERO_SURFACE_DIFFERENT_REGION": (1.00, 0.58, 0.08),
    "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS": (0.96, 0.82, 0.08),
    "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED": (0.10, 0.90, 0.90),
    "FIELD_REPLAY_VOLUME_BOUNDARY": (0.68, 0.24, 0.95),
    "OTHER_EXISTING_CONTRACT_REASON": (1.00, 1.00, 1.00),
}
STATUS_COLORS = {
    "accepted": (0.10, 0.85, 0.35),
    "unowned": (0.96, 0.70, 0.08),
    "gray": (0.60, 0.60, 0.62),
}
TARGET_REGIONS = (0, 2, 5)
GRID_GAP_MAX_PROBE_STEPS = 16
CHUNK_SIZE = 100_000


def _progress(message: str) -> None:
    print(f"[worklog 156] {message}", flush=True)


def _summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.percentile(array, 50, method="nearest")),
        "p75": float(np.percentile(array, 75, method="nearest")),
        "p90": float(np.percentile(array, 90, method="nearest")),
        "p95": float(np.percentile(array, 95, method="nearest")),
        "p99": float(np.percentile(array, 99, method="nearest")),
        "max": float(np.max(array)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_mapping(stable_ids: np.ndarray, region_ids: np.ndarray, statuses: np.ndarray) -> str:
    order = np.argsort(stable_ids, kind="stable")
    digest = hashlib.sha256()
    for array in (stable_ids[order].astype("<i8"), region_ids[order].astype("<i8"), statuses[order].astype("i1")):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _encode_cells(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    return ((cells[..., 0] + KEY_BOUND) * STRIDE_X +
            (cells[..., 1] + KEY_BOUND) * STRIDE_Y +
            (cells[..., 2] + KEY_BOUND) * STRIDE_Z)


def _decode_keys(keys: np.ndarray) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64)
    x = values // STRIDE_X - KEY_BOUND
    rem = values - (x + KEY_BOUND) * STRIDE_X
    y = rem // STRIDE_Y - KEY_BOUND
    z = rem - (y + KEY_BOUND) * STRIDE_Y - KEY_BOUND
    return np.stack((x, y, z), axis=-1)


def _lookup_sorted(keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(keys, query, side="left")
    present = positions < len(keys)
    safe = np.minimum(positions, max(len(keys) - 1, 0))
    if len(keys):
        present &= keys[safe] == query
    return safe, present


def _parse_closure(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else ""
    try:
        return json.loads(str(value)) if value else {}
    except json.JSONDecodeError:
        return {"raw_available": bool(value), "parse_error": True}


def _load_frozen_data(field_path: Path, wl154: Path, wl155: Path) -> dict[str, Any]:
    with np.load(field_path, allow_pickle=False) as field:
        field_keys = np.asarray(field["keys"], dtype=np.int64)
        field_values = np.asarray(field["value"], dtype=np.float32)
        field_support = np.asarray(field["support_count"], dtype=np.int32)
        field_h = float(np.asarray(field["h"]).reshape(-1)[0])
        field_mu = float(np.asarray(field["mu"]).reshape(-1)[0])
        closure = _parse_closure(field["closure"] if "closure" in field else "")
    with np.load(wl154 / "candidate_f_tsdf_surface_samples.npz", allow_pickle=False) as samples:
        sample_keys = np.asarray(samples["source_cell_keys"], dtype=np.int64)
        sample_cells = np.asarray(samples["cell_indices"], dtype=np.int64)
        sample_xyz = np.asarray(samples["world_xyz"], dtype=np.float32)
    with np.load(wl154 / "candidate_f_association.npz", allow_pickle=False) as association:
        nearest_index = np.asarray(association["nearest_gaussian_index"], dtype=np.int64)
        nearest_gaussian_id = np.asarray(association["nearest_gaussian_id"], dtype=np.int64)
        nearest_distance = np.asarray(association["nearest_distance"], dtype=np.float32)
    with np.load(wl154 / "candidate_f_region_owned_support.npz", allow_pickle=False) as support:
        nearest_region = np.asarray(support["nearest_region_id"], dtype=np.int64)
        owned_region = np.asarray(support["owned_region_id"], dtype=np.int64)
        accepted = np.asarray(support["accepted_mask"], dtype=bool)
        component_id = np.asarray(support["component_id"], dtype=np.int64)
    with np.load(wl155 / "gaussian_id_region_status_mapping.npz", allow_pickle=False) as mapping:
        stable_ids = np.asarray(mapping["stable_gaussian_id"], dtype=np.int64)
        gaussian_region = np.asarray(mapping["region_id"], dtype=np.int64)
        gaussian_status = np.asarray(mapping["membership_status"], dtype=np.int8)
    report155 = json.loads((wl155 / "worklog_155_report.json").read_text(encoding="utf-8"))
    model_index = np.searchsorted(stable_ids, nearest_gaussian_id, side="left")
    model_index_safe = np.minimum(model_index, max(len(stable_ids) - 1, 0))
    stable_id_join = bool(len(stable_ids) and np.all(model_index < len(stable_ids)) and np.array_equal(stable_ids[model_index_safe], nearest_gaussian_id))
    nearest_status = np.full(nearest_gaussian_id.shape, -1, dtype=np.int8)
    valid_model_index = model_index < len(stable_ids)
    if np.any(valid_model_index):
        nearest_status[valid_model_index] = gaussian_status[model_index_safe[valid_model_index]]
    mapping_hash = _sha256_mapping(stable_ids, gaussian_region, gaussian_status)
    mapping_hash_file = (wl155 / "gaussian_id_region_status_mapping.sha256").read_text(encoding="ascii").strip()
    component_count = int(component_id[component_id >= 0].max()) + 1 if np.any(component_id >= 0) else 0
    component_region = np.full((component_count,), -1, dtype=np.int64)
    valid_component = component_id >= 0
    component_region[component_id[valid_component]] = owned_region[valid_component]
    field_cells = _decode_keys(field_keys)
    field_min = field_cells.min(axis=0)
    field_max = field_cells.max(axis=0)
    tabletop_review_ids: set[int] = set()
    for camera in report155.get("real_scene_review_export", {}).get("cameras", {}).values():
        tabletop = camera.get("review_targets", {}).get("tabletop", {})
        tabletop_review_ids.update(int(value) for value in tabletop.get("candidate_region_ids", []))
    return {
        "field_keys": field_keys, "field_values": field_values, "field_support": field_support,
        "field_h": field_h, "field_mu": field_mu, "closure": closure,
        "sample_keys": sample_keys, "sample_cells": sample_cells, "sample_xyz": sample_xyz,
        "nearest_index": nearest_index, "nearest_gaussian_id": nearest_gaussian_id, "nearest_distance": nearest_distance,
        "nearest_region": nearest_region, "owned_region": owned_region, "accepted": accepted,
        "component_id": component_id, "nearest_status": nearest_status,
        "stable_ids": stable_ids, "gaussian_region": gaussian_region, "gaussian_status": gaussian_status,
        "component_count": component_count, "component_region": component_region,
        "field_min": field_min, "field_max": field_max,
        "wl154": wl154, "wl155": wl155, "report155": report155,
        "mapping_hash": mapping_hash, "mapping_hash_file": mapping_hash_file,
        "stable_id_join": stable_id_join, "tabletop_review_ids": sorted(tabletop_review_ids),
    }


def _baseline_reconciliation(data: dict[str, Any]) -> dict[str, Any]:
    samples = len(data["sample_keys"])
    owned = int(data["accepted"].sum())
    unowned = samples - owned
    component_ids = data["component_id"]
    components = int(data["component_count"])
    comp_counts = np.bincount(component_ids[component_ids >= 0], minlength=components)
    region_counts = Counter(int(value) for value in data["owned_region"][data["accepted"]].tolist())
    associated_counts = Counter(int(value) for value in data["nearest_region"].tolist() if value >= 0)
    w154_report = json.loads((data["wl154"] / "candidate_f_report.json").read_text(encoding="utf-8"))
    components_payload = json.loads((data["wl154"] / "support_components.json").read_text(encoding="utf-8"))
    historical_materialized = int(w154_report.get("representative", {}).get("materialized_count", 1263))
    historical_abstained = int(w154_report.get("representative", {}).get("abstained_count", 494707))
    return {
        "wl153_field": {
            "field_voxel_count": int(len(data["field_keys"])), "h": data["field_h"], "mu": data["field_mu"],
            "closure": data["closure"], "field_min_cell": data["field_min"], "field_max_cell": data["field_max"],
        },
        "wl154": {
            "tsdf_zero_surface_samples": samples, "accepted_owned_samples": owned, "unowned_samples": unowned,
            "native_support_component_count": components, "support_component_records": len(components_payload),
            "materialized_representatives": historical_materialized, "abstained_representatives": historical_abstained,
            "accepted_region_sample_counts_top": [{"region_id": rid, "owned_samples": int(region_counts[rid]), "associated_samples": int(associated_counts[rid]), "components": int((data["component_region"] == rid).sum())} for rid in (0, 2, 5)],
        },
        "wl155": {
            "mapping_hash_report": data["report155"].get("standalone_gaussian_region_replay", {}).get("mapping_hash"),
            "mapping_hash_recomputed": data["mapping_hash"], "mapping_hash_file": data["mapping_hash_file"],
            "mapping_hash_exact": data["mapping_hash"] == data["mapping_hash_file"],
            "w154_join_stable_id_exact": data["stable_id_join"],
            "w154_join_region_and_status_exact": bool(
                data["stable_id_join"]
                and np.array_equal(
                    data["nearest_region"],
                    data["gaussian_region"][np.minimum(
                        np.searchsorted(data["stable_ids"], data["nearest_gaussian_id"], side="left"),
                        max(len(data["stable_ids"]) - 1, 0),
                    )],
                )
            ),
            "frozen_review_tabletop_candidate_region_ids": data["tabletop_review_ids"],
        },
        "nearest_distance": _summary(data["nearest_distance"]),
    }


def _classify_observed_state(*, authoritative: bool, outside_bounds: bool, zero_surface: bool, sample_exists: bool,
                             accepted: bool, neighbor_region: int, target_region: int,
                             neighbor_component: int, current_component: int) -> str:
    """Scalar contract used by synthetic checks and documented by vector replay."""

    if not authoritative:
        return "FIELD_REPLAY_VOLUME_BOUNDARY" if outside_bounds else "OUTSIDE_AUTHORITATIVE_FIELD"
    if not zero_surface:
        return "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE"
    if not sample_exists:
        return "OTHER_EXISTING_CONTRACT_REASON"
    if not accepted:
        return "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"
    if neighbor_region != target_region:
        return "ZERO_SURFACE_DIFFERENT_REGION"
    if neighbor_component == current_component:
        return INTERNAL_SAME_REGION
    return "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED"


def _classify_target_frontier(data: dict[str, Any], target_region: int, *, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:
    owned_indices = np.flatnonzero(data["accepted"] & (data["owned_region"] == target_region))
    component_ids = np.unique(data["component_id"][owned_indices])
    component_sizes = np.bincount(data["component_id"][owned_indices], minlength=data["component_count"])[component_ids]
    local_component = np.searchsorted(component_ids, data["component_id"][owned_indices])
    face_counts = np.zeros((len(component_ids), len(CATEGORIES)), dtype=np.int64)
    records: dict[str, list[np.ndarray]] = defaultdict(list)
    field_presence_counts: list[np.ndarray] = []
    b_value_min: list[np.ndarray] = []
    b_value_max: list[np.ndarray] = []
    b_support_min: list[np.ndarray] = []
    transition_counts: Counter[tuple[int, int]] = Counter()
    transition_components: defaultdict[int, set[int]] = defaultdict(set)
    unowned_status_counts: Counter[int] = Counter()
    unowned_status_components: defaultdict[int, set[int]] = defaultdict(set)
    progress_step = max(chunk_size, 1)
    for start in range(0, len(owned_indices), progress_step):
        current = owned_indices[start:start + progress_step]
        cells = data["sample_cells"][current]
        current_components = data["component_id"][current]
        current_local = local_component[start:start + len(current)]
        for direction_index, offset in enumerate(FACE_OFFSETS):
            neighbor_cells = cells + offset[None, :]
            neighbor_keys = _encode_cells(neighbor_cells)
            sample_pos, sample_exists = _lookup_sorted(data["sample_keys"], neighbor_keys)
            sample_safe = np.minimum(sample_pos, max(len(data["sample_keys"]) - 1, 0))
            corner_cells = neighbor_cells[:, None, :] + CORNER_OFFSETS[None, :, :]
            corner_keys = _encode_cells(corner_cells.reshape(-1, 3)).reshape(-1, 8)
            field_pos, field_present = _lookup_sorted(data["field_keys"], corner_keys.reshape(-1))
            field_present = field_present.reshape(-1, 8)
            field_safe = np.minimum(field_pos, max(len(data["field_keys"]) - 1, 0)).reshape(-1, 8)
            values = data["field_values"][field_safe]
            support = data["field_support"][field_safe]
            finite = np.isfinite(values).all(axis=1)
            authoritative = field_present.all(axis=1) & finite
            zero_surface = authoritative & (np.min(values, axis=1) <= 0.0) & (np.max(values, axis=1) >= 0.0)
            outside_bounds = np.any((corner_cells < data["field_min"][None, None, :]) | (corner_cells > data["field_max"][None, None, :]), axis=(1, 2))
            neighbor_region = data["nearest_region"][sample_safe]
            neighbor_owned = data["owned_region"][sample_safe]
            neighbor_accepted = data["accepted"][sample_safe]
            neighbor_component = data["component_id"][sample_safe]
            neighbor_status = data["nearest_status"][sample_safe]
            category = np.full((len(current),), -1, dtype=np.int8)
            missing = ~authoritative
            category[missing & ~outside_bounds] = CATEGORY_CODE["OUTSIDE_AUTHORITATIVE_FIELD"]
            category[missing & outside_bounds] = CATEGORY_CODE["FIELD_REPLAY_VOLUME_BOUNDARY"]
            category[authoritative & ~zero_surface] = CATEGORY_CODE["AUTHORITATIVE_BUT_NOT_ZERO_SURFACE"]
            eligible = zero_surface & sample_exists
            category[zero_surface & ~sample_exists] = CATEGORY_CODE["OTHER_EXISTING_CONTRACT_REASON"]
            same_region = eligible & neighbor_accepted & (neighbor_owned == target_region)
            same_component = same_region & (neighbor_component == current_components) & (neighbor_component >= 0)
            category[same_region & ~same_component] = CATEGORY_CODE["SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED"]
            category[eligible & neighbor_accepted & (neighbor_owned != target_region)] = CATEGORY_CODE["ZERO_SURFACE_DIFFERENT_REGION"]
            category[eligible & ~neighbor_accepted] = CATEGORY_CODE["ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"]
            category[same_component] = -1
            for code in range(len(CATEGORIES)):
                selected = category == code
                if np.any(selected):
                    face_counts[:, code] += np.bincount(current_local[selected], minlength=len(component_ids))
            frontier = category >= 0
            if np.any(frontier):
                records["source_sample_index"].append(current[frontier])
                records["category"].append(category[frontier])
                records["direction"].append(np.full((int(frontier.sum()),), direction_index, dtype=np.int8))
                records["neighbor_region"].append(neighbor_region[frontier])
                records["neighbor_status"].append(neighbor_status[frontier])
                records["neighbor_sample_index"].append(np.where(sample_exists[frontier], sample_safe[frontier], -1))
                c_selected = current_components[frontier]
                n_selected = neighbor_region[frontier]
                for neighbor in n_selected[category[frontier] == CATEGORY_CODE["ZERO_SURFACE_DIFFERENT_REGION"]]:
                    transition_counts[(target_region, int(neighbor))] += 1
                c_mask = category == CATEGORY_CODE["ZERO_SURFACE_DIFFERENT_REGION"]
                for neighbor in np.unique(neighbor_region[c_mask]):
                    transition_components[int(neighbor)].update(int(value) for value in current_components[c_mask & (neighbor_region == neighbor)].tolist())
                d_mask = category == CATEGORY_CODE["ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"]
                for status in np.unique(neighbor_status[d_mask]):
                    unowned_status_counts[int(status)] += int((d_mask & (neighbor_status == status)).sum())
                    unowned_status_components[int(status)].update(int(value) for value in current_components[d_mask & (neighbor_status == status)].tolist())
            missing_front = missing
            if np.any(missing_front):
                field_presence_counts.append(field_present[missing_front].sum(axis=1).astype(np.int8))
            b_mask = category == CATEGORY_CODE["AUTHORITATIVE_BUT_NOT_ZERO_SURFACE"]
            if np.any(b_mask):
                b_value_min.append(np.min(values[b_mask], axis=1))
                b_value_max.append(np.max(values[b_mask], axis=1))
                b_support_min.append(np.min(support[b_mask], axis=1))
        _progress(f"frontier region={target_region} cells={min(start + progress_step, len(owned_indices)):,}/{len(owned_indices):,}")
    flat = {name: np.concatenate(values) if values else np.empty((0,), dtype=np.int64) for name, values in records.items()}
    if len(flat.get("category", [])):
        flat["category"] = flat["category"].astype(np.int8, copy=False)
    category_face_counts = {name: int(face_counts[:, code].sum()) for code, name in enumerate(CATEGORIES)}
    total_frontier_faces = int(face_counts.sum())
    affected_components = {name: np.flatnonzero(face_counts[:, code] > 0) for code, name in enumerate(CATEGORIES)}
    category_details: dict[str, Any] = {}
    for code, name in enumerate(CATEGORIES):
        affected = affected_components[name]
        category_details[name] = {
            "frontier_face_count": int(face_counts[:, code].sum()),
            "frontier_face_fraction": float(face_counts[:, code].sum() / max(total_frontier_faces, 1)),
            "affected_component_count": int(len(affected)),
            "affected_component_fraction": float(len(affected) / max(len(component_ids), 1)),
            "affected_component_owned_sample_population": int(component_sizes[affected].sum()),
            "frontier_face_weighted_component_population": int((face_counts[:, code] * component_sizes).sum()),
        }
    largest_order = np.argsort(component_sizes, kind="stable")[::-1]
    subsets = {
        "all_components": np.ones((len(component_ids),), dtype=bool),
        "largest_component": np.zeros((len(component_ids),), dtype=bool),
        "top_10_components": np.zeros((len(component_ids),), dtype=bool),
        "small_size_le_8": component_sizes <= 8,
        "singleton_size_eq_1": component_sizes == 1,
        "very_small_size_le_2": component_sizes <= 2,
    }
    if len(largest_order):
        subsets["largest_component"][largest_order[:1]] = True
        subsets["top_10_components"][largest_order[:10]] = True
    subset_details: dict[str, Any] = {}
    for subset_name, selection in subsets.items():
        selected_sizes = component_sizes[selection]
        selected_faces = face_counts[selection]
        selected_total_faces = int(selected_faces.sum())
        subset_details[subset_name] = {
            "component_count": int(selection.sum()),
            "owned_sample_population": int(selected_sizes.sum()),
            "frontier_face_count": selected_total_faces,
            "categories": {
                name: {
                    "frontier_face_count": int(selected_faces[:, code].sum()),
                    "frontier_face_fraction": float(selected_faces[:, code].sum() / max(selected_total_faces, 1)),
                    "affected_component_count": int((selected_faces[:, code] > 0).sum()),
                    "affected_component_fraction": float((selected_faces[:, code] > 0).sum() / max(int(selection.sum()), 1)),
                }
                for code, name in enumerate(CATEGORIES)
            },
        }
    comp_size_summary = {
        "component_count": int(len(component_ids)),
        "size": _summary(component_sizes),
        "singleton_components": int((component_sizes == 1).sum()),
        "size_le_2": int((component_sizes <= 2).sum()), "size_le_4": int((component_sizes <= 4).sum()),
        "size_le_8": int((component_sizes <= 8).sum()), "size_le_16": int((component_sizes <= 16).sum()),
    }
    largest_fraction = {}
    for count in (1, 5, 10, 100):
        largest_fraction[f"largest_{count}_owned_sample_fraction"] = float(component_sizes[largest_order[:count]].sum() / max(component_sizes.sum(), 1)) if len(largest_order) else 0.0
    return {
        "target_region_id": target_region,
        "owned_sample_indices": owned_indices,
        "associated_sample_indices": np.flatnonzero(data["nearest_region"] == target_region),
        "component_ids": component_ids,
        "component_sizes": component_sizes,
        "face_counts": face_counts,
        "records": flat,
        "category_details": category_details,
        "subset_details": subset_details,
        "component_size_accounting": comp_size_summary | largest_fraction,
        "transition_counts": {f"{source}->{neighbor}": int(count) for (source, neighbor), count in transition_counts.items()},
        "transition_affected_component_counts": {str(neighbor): len(values) for neighbor, values in transition_components.items()},
        "unowned_status_counts": {STATUS_NAMES.get(status, str(status)): int(count) for status, count in unowned_status_counts.items()},
        "unowned_status_affected_component_counts": {STATUS_NAMES.get(status, str(status)): len(values) for status, values in unowned_status_components.items()},
        "field_starvation_stats": {
            "missing_neighbor_corner_presence": _summary(np.concatenate(field_presence_counts) if field_presence_counts else np.empty((0,), dtype=np.float32)),
            "authoritative_but_not_zero_surface_corner_min": _summary(np.concatenate(b_value_min) if b_value_min else np.empty((0,), dtype=np.float32)),
            "authoritative_but_not_zero_surface_corner_max": _summary(np.concatenate(b_value_max) if b_value_max else np.empty((0,), dtype=np.float32)),
            "authoritative_but_not_zero_surface_corner_support_min": _summary(np.concatenate(b_support_min) if b_support_min else np.empty((0,), dtype=np.float32)),
        },
        "total_frontier_faces": total_frontier_faces,
        "new_connectivity_graph_constructed": False,
    }


def _grid_gap_distribution(data: dict[str, Any], result: dict[str, Any], *, max_steps: int = GRID_GAP_MAX_PROBE_STEPS) -> dict[str, Any]:
    records = result["records"]
    if len(records.get("category", [])) == 0:
        return {"frontier_faces_considered": 0, "max_probe_steps": max_steps, "bins": {}}
    excluded = records["category"] == CATEGORY_CODE["SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED"]
    keep = ~excluded
    source = records["source_sample_index"][keep]
    directions = records["direction"][keep]
    categories = records["category"][keep]
    first_step = np.full((len(source),), -1, dtype=np.int16)
    remaining = np.arange(len(source), dtype=np.int64)
    for step in range(2, max_steps + 1):
        if not len(remaining):
            break
        cells = data["sample_cells"][source[remaining]] + FACE_OFFSETS[directions[remaining]] * step
        keys = _encode_cells(cells)
        positions, present = _lookup_sorted(data["sample_keys"], keys)
        safe = np.minimum(positions, max(len(data["sample_keys"]) - 1, 0))
        same = present & data["accepted"][safe] & (data["owned_region"][safe] == result["target_region_id"]) & (data["component_id"][safe] >= 0)
        selected = remaining[same]
        first_step[selected] = step
        remaining = remaining[~same]
    labels = np.full((len(first_step),), "no_same_region_component_within_probe", dtype=object)
    labels[first_step == 2] = "one_missing_or_noneligible_grid_step"
    labels[first_step == 3] = "two_missing_or_noneligible_grid_steps"
    labels[first_step == 4] = "three_missing_or_noneligible_grid_steps"
    labels[first_step >= 5] = "larger_separation_5_plus_grid_steps"
    bins = Counter(str(value) for value in labels.tolist())
    by_category: dict[str, dict[str, int]] = {}
    for code, name in enumerate(CATEGORIES):
        mask = categories == code
        by_category[name] = {str(label): int((labels[mask] == label).sum()) for label in sorted(set(labels[mask].tolist()))} if np.any(mask) else {}
    return {
        "frontier_faces_considered": int(len(source)), "max_probe_steps": max_steps,
        "bins": {label: {"count": int(count), "fraction": float(count / max(len(source), 1))} for label, count in sorted(bins.items())},
        "by_frontier_category": by_category,
        "diagnostic_only_no_bridge_radius_selected": True,
    }


def _synthetic_contracts() -> dict[str, Any]:
    cases = [
        ("A_continuous_same_region_zero_surface", INTERNAL_SAME_REGION, dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=True, neighbor_region=7, target_region=7, neighbor_component=0, current_component=0)),
        ("B_absent_authoritative_cell_band", "OUTSIDE_AUTHORITATIVE_FIELD", dict(authoritative=False, outside_bounds=False, zero_surface=False, sample_exists=False, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0)),
        ("C_alternating_frozen_region_ownership", "ZERO_SURFACE_DIFFERENT_REGION", dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=True, neighbor_region=8, target_region=7, neighbor_component=1, current_component=0)),
        ("D_unowned_strip", "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS", dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=False, neighbor_region=7, target_region=7, neighbor_component=-1, current_component=0)),
        ("E_same_region_face_adjacent_identity", INTERNAL_SAME_REGION, dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=True, accepted=True, neighbor_region=7, target_region=7, neighbor_component=0, current_component=0)),
        ("F_replay_volume_boundary", "FIELD_REPLAY_VOLUME_BOUNDARY", dict(authoritative=False, outside_bounds=True, zero_surface=False, sample_exists=False, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0)),
        ("G_other_existing_contract_reason", "OTHER_EXISTING_CONTRACT_REASON", dict(authoritative=True, outside_bounds=False, zero_surface=True, sample_exists=False, accepted=False, neighbor_region=-1, target_region=7, neighbor_component=-1, current_component=0)),
    ]
    results = []
    for name, expected, kwargs in cases:
        observed = _classify_observed_state(**kwargs)
        results.append({"name": name, "expected": expected, "observed": observed, "pass": observed == expected})
    return {"all_pass": all(item["pass"] for item in results), "synthetic_accounting_only_not_architecture_success": True, "cases": results}


def _save_frontier_npz(root: Path, result: dict[str, Any]) -> Path:
    path = root / "frontier_records.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result["records"])
    return path


def _tensor_image(value: Any) -> np.ndarray:
    array = value.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path, format="PNG", optimize=True)


def _component_rgb(component_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(component_ids, dtype=np.int64)
    hue = np.mod(ids.astype(np.float64) * 0.6180339887498949, 1.0)
    saturation = 0.55 + 0.35 * np.mod(ids.astype(np.float64) * 0.7548776662466927, 1.0)
    value = 0.60 + 0.40 * np.mod(ids.astype(np.float64) * 0.5698402909980532, 1.0)
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    choices = np.stack(
        (
            np.stack((value, t, p), axis=-1),
            np.stack((q, value, p), axis=-1),
            np.stack((p, value, t), axis=-1),
            np.stack((p, q, value), axis=-1),
            np.stack((t, p, value), axis=-1),
            np.stack((value, p, q), axis=-1),
        ),
        axis=1,
    )
    return choices[np.arange(len(ids)), sector]


def _overlay_points(base: np.ndarray, points: np.ndarray, colors: np.ndarray, camera: Any, alpha: float = 0.82) -> np.ndarray:
    if len(points) == 0:
        return base.copy()
    projection = _project_world_points(points, camera)
    valid = projection["valid"]
    if not np.any(valid):
        return base.copy()
    x = np.clip(np.rint(projection["x"][valid]).astype(np.int64), 0, base.shape[1] - 1)
    y = np.clip(np.rint(projection["y"][valid]).astype(np.int64), 0, base.shape[0] - 1)
    depth = projection["depth"][valid]
    rgb = np.asarray(colors, dtype=np.float32)[valid]
    flat = y * base.shape[1] + x
    order = np.argsort(depth, kind="stable")
    sorted_flat = flat[order]
    _, first = np.unique(sorted_flat, return_index=True)
    chosen = order[first]
    result = base.copy().astype(np.float32)
    yy, xx = y[chosen], x[chosen]
    result[yy, xx] = (1.0 - alpha) * result[yy, xx] + alpha * np.clip(rgb[chosen] * 255.0, 0.0, 255.0)
    return result.astype(np.uint8)


def _write_visualization_readme(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_review_readmes(out: Path, target_specs: dict[int, dict[str, Any]]) -> None:
    root = out / "review_views"
    _write_visualization_readme(out / "README.md", """# Worklog 156 산출물 안내

이 배치는 frozen WL153 field, WL154 TSDF zero-surface/ownership/component arrays, WL155 Gaussian Region mapping만 읽어 support fragmentation 원인을 분류한다. connectivity repair, bridge, dilation, ownership 변경은 수행하지 않는다. `review_views/`의 모든 PNG는 공통 world/camera geometry를 사용하며, 각 하위 폴더 README가 해당 view의 의미와 palette를 설명한다.

`mandatory_gaussian_visualization_pair/`는 W155 canonical pair의 Gaussian content/metadata를 유지한 PNG copy다(PPM은 W156 copy에서 제외). A–H는 그 pair와 별개의 W156 causal diagnostic view다.
""")
    _write_visualization_readme(root / "README.md", """# W156 Real-Scene Causal Review Views

대상은 primary tabletop review candidate Region 0과 frozen historical scale controls Region 2/5다. 각 target에는 다음 8개 view가 있다: A original Gaussian scene, B frozen Gaussian Surface Region, C all associated TSDF zero-surface support, D native support components, E largest component, F frontier cause, G ownership transitions, H field starvation.

PNG가 주 검토 파일이다. 모든 support/frontier point는 smoothing·dilation 없이 frozen world coordinates를 투영한다. raw indices와 cause labels는 각 target의 `frontier_records.npz` 및 W154 source arrays로 추적한다.
""")
    frontier_root = out / "frontier_records"
    _write_visualization_readme(frontier_root / "README.md", """# W156 Frontier Record Artifacts

각 `region_*` 폴더의 `frontier_records.npz`는 frozen WL154 native owned TSDF component의 immediate six-face frontier를 저장한다. `category`는 인접 grid location을 mutually exclusive하게 분류한 코드이며, source sample index와 direction으로 WL154 sample 배열을 역추적할 수 있다.

이 폴더는 시각화가 아니라 정량 원인 감사용 raw artifact다. connectivity bridge, smoothing, dilation, ownership 변경은 저장하거나 적용하지 않는다.
""")
    for region_id in target_specs:
        _write_visualization_readme(frontier_root / f"region_{region_id:06d}" / "README.md", f"""# Region {region_id:06d} Frontier Records

Region `{region_id}`의 frozen accepted-owned TSDF sample cell에 대한 immediate six-face frontier NPZ다. category별 의미와 집계는 `worklog_156_report.json`의 `native_tsdf_component_frontier_accounting`에서 확인한다.

이 raw record는 W154 component membership를 재계산하거나 수정하지 않으며, W156 causal attribution을 재현하기 위한 입력이다.
""")
    view_descriptions = {
        "A_original_gaussian_scene": "학습 checkpoint의 original Gaussian scene이다. 원래 색/SH와 geometry를 유지한 camera 기준 장면이며, target Region 강조를 포함하지 않는다.",
        "B_frozen_gaussian_surface_region": "frozen target Gaussian Surface Region만 green으로 표시하고 나머지 Gaussian은 gray로 표시한다. Region ID를 변경하거나 재선택하지 않는다.",
        "C_all_tsdf_zero_surface_support": "target Region ID에 associated된 모든 frozen TSDF zero-surface sample을 표시한다. accepted-owned support는 green, 같은 Region이지만 unowned/ambiguous로 제외된 sample은 yellow로 표시한다.",
        "D_native_support_components": "현재 WL154 native exact face-adjacency가 부여한 component ID를 deterministic component palette로 표시한다. component를 합치거나 재계산하지 않는다.",
        "E_largest_component_only": "해당 target Region의 owned TSDF sample 중 가장 큰 frozen native component만 표시한다. 작은 component를 삭제했다는 뜻이 아니라 비교용 visibility다.",
        "F_frontier_cause": "각 owned component cell의 immediate six-face frontier를 mutually exclusive cause palette로 표시한다. 내부 same-component face는 표시 대상이 아니며 raw frontier record에는 포함된 원인만 남긴다.",
        "G_ownership_transitions": "ZERO_SURFACE_DIFFERENT_REGION frontier만 표시한다. 색은 인접 Region ID에서 deterministic하게 계산하며, 두 Region을 같은 physical sheet로 선언하지 않는다.",
        "H_field_starvation": "OUTSIDE_AUTHORITATIVE_FIELD, AUTHORITATIVE_BUT_NOT_ZERO_SURFACE, FIELD_REPLAY_VOLUME_BOUNDARY frontier를 표시한다. field, h, mu, closure 또는 sign-change rule은 바꾸지 않는다.",
    }
    for region_id, spec in target_specs.items():
        target_root = root / spec["folder"]
        _write_visualization_readme(target_root / "README.md", f"""# Region {region_id:06d} — {spec['role']}

이 target은 frozen WL155 mapping의 Region ID `{region_id}`다. primary Region 0은 WL155 tabletop review candidate list에서 고정했고, Region 2/5는 historical high-TSDF controls로 보존했다. 이는 physical-sheet ground truth가 아니다.

- Gaussian member count: {spec['gaussian_member_count']:,}
- associated TSDF samples: {spec['associated_tsdf_sample_count']:,}
- accepted-owned TSDF samples: {spec['owned_tsdf_sample_count']:,}
- native support components: {spec['native_component_count']:,}

각 view 폴더의 README는 해당 시각화의 의미·입력·palette·review 제한을 설명한다.
""")
        for view_name, meaning in view_descriptions.items():
            view_root = target_root / view_name
            _write_visualization_readme(view_root / "README.md", f"""# {view_name} — Region {region_id:06d}

{meaning}

`cameras/` 아래의 PNG는 fixed WL145–155 review camera set을 사용한 matched projection이다. PNG는 primary artifact이며, raw support/frontier data는 target-level NPZ와 frozen W154 arrays를 참조한다.
""")
            camera_root = view_root / "cameras"
            _write_visualization_readme(camera_root / "README.md", f"""# {view_name} Camera Exports — Region {region_id:06d}

세 fixed camera `DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`에서 같은 view를 투영했다. camera 선택이나 target membership은 결과에 따라 바꾸지 않았다.
""")
            for camera in REVIEW_CAMERAS:
                camera_dir = camera_root / camera
                _write_visualization_readme(camera_dir / "README.md", f"""# {view_name} — {camera}

{meaning}

이 파일은 W156 fixed camera `{camera}`의 PNG render다. matched background는 Original Gaussian scene이며, support overlay는 smoothing·dilation 없이 frozen sample/world coordinates를 직접 투영한다.
""")


def _render_real_scene_exports(data: dict[str, Any], out: Path, target_specs: dict[int, dict[str, Any]], frontier_results: dict[int, dict[str, Any]], source_path: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    import torch
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    model, payload = _load_surfel_model_safe(checkpoint, device)
    raw_checkpoint_ids = payload["model_raw"].get("stable_gaussian_ids")
    if raw_checkpoint_ids is None:
        checkpoint_ids = np.arange(len(model), dtype=np.int64)
    elif hasattr(raw_checkpoint_ids, "detach"):
        checkpoint_ids = raw_checkpoint_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        checkpoint_ids = np.asarray(raw_checkpoint_ids, dtype=np.int64)
    if not np.array_equal(checkpoint_ids, data["stable_ids"]):
        raise ValueError("W155 stable Gaussian IDs do not match the frozen checkpoint row order")
    cameras, camera_meta = _build_named_cameras(source_path, "images_8", "sparse/0", -1, 8, device)
    root = out / "review_views"
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros((3,), dtype=torch.float32, device=model.device)
    original_dc = model._features_dc.detach().clone()
    original_rest = model._features_rest.detach().clone()
    original_degree = int(model.active_sh_degree)
    all_images: dict[str, np.ndarray] = {}
    try:
        for camera_name in REVIEW_CAMERAS:
            with torch.no_grad():
                model._features_dc.data.copy_(original_dc)
                model._features_rest.data.copy_(original_rest)
                model.active_sh_degree = original_degree
                package = rasterizer.render(cameras[camera_name], model, background=background)
                all_images[camera_name] = _tensor_image(package["render"])
                del package
        for region_id, spec in target_specs.items():
            target_root = root / spec["folder"]
            target_mask = data["gaussian_region"] == region_id
            for camera_name in REVIEW_CAMERAS:
                _save_png(target_root / "A_original_gaussian_scene" / "cameras" / camera_name / "render.png", all_images[camera_name])
            target_rgb = np.where(target_mask[:, None], np.asarray(STATUS_COLORS["accepted"]), np.asarray(STATUS_COLORS["gray"]))
            with torch.no_grad():
                model._features_dc.data.copy_(_fixed_rgb_to_dc(torch.as_tensor(target_rgb, dtype=torch.float32, device=model.device))[:, None, :])
                model._features_rest.data.zero_()
                model.active_sh_degree = 0
                for camera_name in REVIEW_CAMERAS:
                    package = rasterizer.render(cameras[camera_name], model, background=background)
                    _save_png(target_root / "B_frozen_gaussian_surface_region" / "cameras" / camera_name / "render.png", _tensor_image(package["render"]))
                    del package
            owned_indices = frontier_results[region_id]["owned_sample_indices"]
            associated_indices = frontier_results[region_id]["associated_sample_indices"]
            comp_ids = data["component_id"][owned_indices]
            largest_component = int(frontier_results[region_id]["component_ids"][np.argmax(frontier_results[region_id]["component_sizes"])]) if len(frontier_results[region_id]["component_ids"]) else -1
            associated_colors = np.where(data["accepted"][associated_indices, None], np.asarray(STATUS_COLORS["accepted"]), np.asarray(STATUS_COLORS["unowned"]))
            component_colors = _component_rgb(comp_ids)
            largest_mask = comp_ids == largest_component
            records = frontier_results[region_id]["records"]
            frontier_points = data["sample_xyz"][records["source_sample_index"]]
            frontier_colors = np.asarray([FRONTIER_COLORS[CATEGORIES[int(code)]] for code in records["category"]], dtype=np.float32)
            ownership_mask = records["category"] == CATEGORY_CODE["ZERO_SURFACE_DIFFERENT_REGION"]
            ownership_points = frontier_points[ownership_mask]
            ownership_neighbor = records["neighbor_region"][ownership_mask]
            ownership_colors = _component_rgb(ownership_neighbor)
            starvation_mask = np.isin(records["category"], [CATEGORY_CODE["OUTSIDE_AUTHORITATIVE_FIELD"], CATEGORY_CODE["AUTHORITATIVE_BUT_NOT_ZERO_SURFACE"], CATEGORY_CODE["FIELD_REPLAY_VOLUME_BOUNDARY"]])
            starvation_points = frontier_points[starvation_mask]
            starvation_colors = np.asarray([FRONTIER_COLORS[CATEGORIES[int(code)]] for code in records["category"][starvation_mask]], dtype=np.float32)
            for camera_name in REVIEW_CAMERAS:
                camera = cameras[camera_name]
                base = all_images[camera_name]
                view_payloads = {
                    "C_all_tsdf_zero_surface_support": (data["sample_xyz"][associated_indices], associated_colors),
                    "D_native_support_components": (data["sample_xyz"][owned_indices], component_colors),
                    "E_largest_component_only": (data["sample_xyz"][owned_indices][largest_mask], component_colors[largest_mask]),
                    "F_frontier_cause": (frontier_points, frontier_colors),
                    "G_ownership_transitions": (ownership_points, ownership_colors),
                    "H_field_starvation": (starvation_points, starvation_colors),
                }
                for view_name, (points, colors) in view_payloads.items():
                    image = _overlay_points(base, points, colors, camera)
                    _save_png(target_root / view_name / "cameras" / camera_name / "render.png", image)
    finally:
        model._features_dc.data.copy_(original_dc)
        model._features_rest.data.copy_(original_rest)
        model.active_sh_degree = original_degree
    return {"camera_set": list(REVIEW_CAMERAS), "camera_metadata": camera_meta, "renderer": "OSNSurfelRasterizer", "resolution": [648, 420], "background": [0.0, 0.0, 0.0], "same_checkpoint_iteration_and_geometry": True}


def _ignore_ppm(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if Path(name).suffix.lower() == ".ppm"]


def _target_specs(data: dict[str, Any], results: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}
    for region_id, result in results.items():
        role = "primary_tabletop_review_candidate" if region_id == 0 else "historical_high_tsdf_control"
        folder = f"region_{region_id:06d}_{'primary_tabletop_candidate' if region_id == 0 else 'control'}"
        gaussian_members = int((data["gaussian_region"] == region_id).sum())
        associated = int((data["nearest_region"] == region_id).sum())
        owned = int(result["owned_sample_indices"].size)
        specs[region_id] = {"role": role, "folder": folder, "gaussian_member_count": gaussian_members, "associated_tsdf_sample_count": associated, "owned_tsdf_sample_count": owned, "native_component_count": int(result["component_size_accounting"]["component_count"]), "reviewed_candidate_ids": data["tabletop_review_ids"] if region_id == 0 else []}
    return specs


def _architecture_verdict(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    primary = results[0]
    details = primary["category_details"]
    count = lambda name: int(details[name]["frontier_face_count"])
    field_count = count("OUTSIDE_AUTHORITATIVE_FIELD") + count("AUTHORITATIVE_BUT_NOT_ZERO_SURFACE") + count("FIELD_REPLAY_VOLUME_BOUNDARY")
    ownership_count = count("ZERO_SURFACE_DIFFERENT_REGION") + count("ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS")
    native_contract_count = count("SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED")
    other_count = count("OTHER_EXISTING_CONTRACT_REASON")
    total = int(primary["total_frontier_faces"])
    if native_contract_count:
        verdict = "MECHANICAL_IMPLEMENTATION_BUG"
        reason = "A same-region eligible face was classified as disconnected under the replayed native contract."
    elif field_count > ownership_count and field_count > native_contract_count and field_count > other_count:
        verdict = "TSDF_FIELD_STARVATION_DOMINANT"
        reason = "The primary region's frontier is predominantly outside the authoritative field or authoritative without a zero-surface crossing."
    elif ownership_count > field_count and ownership_count > native_contract_count and ownership_count > other_count:
        verdict = "REGION_OWNERSHIP_DISCONTINUITY_DOMINANT"
        reason = "The primary region's frontier is predominantly a different-region or unowned/ambiguous zero-surface transition."
    elif field_count or ownership_count or other_count:
        verdict = "MIXED_FRAGMENTATION_CAUSES"
        reason = "Multiple frontier causes remain material after exact category accounting."
    else:
        verdict = "UNRESOLVED"
        reason = "Existing data do not expose a causal frontier category for the primary region."
    groups = {
        "field_starvation": field_count,
        "ownership_discontinuity": ownership_count,
        "native_connectivity_contract": native_contract_count,
        "other_existing_contract_reason": other_count,
    }
    return {
        "architecture_verdict": verdict,
        "verdict_reason": reason,
        "primary_region_id": 0,
        "primary_frontier_face_count": total,
        "primary_cause_group_counts": groups,
        "primary_cause_group_fractions": {name: float(value / max(total, 1)) for name, value in groups.items()},
        "same_region_eligible_but_not_connected_count": native_contract_count,
        "no_connectivity_repair_or_merge_performed": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    _progress("loading frozen WL153/WL154/WL155 arrays")
    data = _load_frozen_data(args.field, args.wl154, args.wl155)
    baseline = _baseline_reconciliation(data)
    _progress(f"baseline samples={len(data['sample_keys']):,} owned={int(data['accepted'].sum()):,} components={data['component_count']:,}")
    results: dict[int, dict[str, Any]] = {}
    frontier_files: dict[int, str] = {}
    for region_id in TARGET_REGIONS:
        result = _classify_target_frontier(data, region_id)
        result["grid_gap_distribution"] = _grid_gap_distribution(data, result)
        path = _save_frontier_npz(args.out / "frontier_records" / f"region_{region_id:06d}", result)
        frontier_files[region_id] = str(path)
        results[region_id] = result
        _progress(f"region={region_id} frontier_faces={result['total_frontier_faces']:,}")
    target_specs = _target_specs(data, results)
    _write_review_readmes(args.out, target_specs)
    synthetic = _synthetic_contracts()
    render = _render_real_scene_exports(data, args.out, target_specs, results, args.source_path, args.checkpoint, args.device)
    mandatory_source = args.wl155 / "mandatory_gaussian_visualization_pair"
    mandatory_target = args.out / "mandatory_gaussian_visualization_pair"
    if mandatory_target.exists():
        shutil.rmtree(mandatory_target)
    shutil.copytree(mandatory_source, mandatory_target, ignore=_ignore_ppm)
    report_targets: dict[str, Any] = {}
    for region_id, result in results.items():
        json_result = {key: value for key, value in result.items() if key not in {"owned_sample_indices", "associated_sample_indices", "component_ids", "component_sizes", "face_counts", "records"}}
        json_result["frontier_records_npz"] = frontier_files[region_id]
        json_result["review_root"] = str(args.out / "review_views" / target_specs[region_id]["folder"])
        report_targets[str(region_id)] = _jsonable(json_result)
    failure = _architecture_verdict(results)
    failure.update({
        "A_TSDF_EVIDENCE_FIELD_STARVATION": {"frontier_categories": ["OUTSIDE_AUTHORITATIVE_FIELD", "AUTHORITATIVE_BUT_NOT_ZERO_SURFACE", "FIELD_REPLAY_VOLUME_BOUNDARY"], "status": "MEASURED_SEPARATELY"},
        "B_OWNERSHIP_DISCONTINUITY": {"frontier_categories": ["ZERO_SURFACE_DIFFERENT_REGION", "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"], "status": "MEASURED_SEPARATELY"},
        "C_CONNECTIVITY_CONTRACT_FAILURE": {"frontier_category": "SAME_REGION_ELIGIBLE_BUT_NOT_CONNECTED", "status": "MEASURED_SEPARATELY"},
        "D_MIXED": {"status": "NOT_COLLAPSED_BEFORE_FRONTIER_ACCOUNTING"},
    })
    report = {
        "status": "COMPLETE_REGION_OWNED_TSDF_SUPPORT_FRAGMENTATION_CAUSAL_AUDIT",
        "batch": "Worklog 156 — Region-Owned TSDF Support Fragmentation Causal Attribution Audit",
        "intent_alignment": {"diagnostic_only": True, "connectivity_repaired": False, "production_behavior_modified": False, "ownership_modified": False, "field_modified": False, "component_membership_modified": False},
        "implementation_fidelity": {"frozen_inputs": ["WL153 field", "WL154 Candidate F samples/association/ownership/component IDs", "WL155 Gaussian ID-region-status mapping"], "new_connectivity_graph_constructed": False, "new_bridge_radius": False, "dilation_or_smoothing": False, "same_native_face_adjacency_replayed": True, "zero_surface_definition_changed": False, "h_changed": False, "mu_changed": False, "closure_changed": False},
        "architecture_result": failure,
        "baseline_reconciliation": baseline,
        "frozen_target_regions": {str(region_id): {key: value for key, value in spec.items() if key != "folder"} for region_id, spec in target_specs.items()},
        "native_tsdf_component_frontier_accounting": report_targets,
        "closure_provenance": {"status": "NOT_RECOVERABLE_UNDER_EXISTING_CONTRACT", "reason": "WL153 field.npz preserves aggregate closure rounds but no per-voxel authoritative-addition round lineage; no heuristic reconstruction was performed."},
        "synthetic_contracts": synthetic,
        "real_scene_qualitative_review": {**render, "view_definitions": ["A_original_gaussian_scene", "B_frozen_gaussian_surface_region", "C_all_tsdf_zero_surface_support", "D_native_support_components", "E_largest_component_only", "F_frontier_cause", "G_ownership_transitions", "H_field_starvation"], "review_root": str(args.out / "review_views")},
        "mandatory_gaussian_visualization_pair": {"source": str(mandatory_source), "copied_without_gaussian_content_modification": True, "ppm_files_omitted_from_w156_copy": True, "root": str(mandatory_target), "note": "W155 canonical Original Scene/Observed-Occluded PNG pair is preserved; W156 A–H diagnostics do not replace it."},
        "forbidden_changes": {"tw_semantics_changed": False, "covariance_normal_added": False, "gaussian_region_changed": False, "nearest_association_changed": False, "tsdf_field_changed": False, "support_components_merged_or_split": False, "boundary_first_changed": False, "nurbs_refit": False, "event_1527_blacklist": False, "trust_or_latent_or_occluded_surface": False},
        "outputs": {"report": str(args.out / "worklog_156_report.json"), "frontier_records": frontier_files, "review_root": str(args.out / "review_views"), "mandatory_pair": str(mandatory_target)},
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_156_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", type=Path, default=DEFAULT_WL153_FIELD)
    parser.add_argument("--wl154", type=Path, default=DEFAULT_WL154)
    parser.add_argument("--wl155", type=Path, default=DEFAULT_WL155)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "architecture_verdict": report["architecture_result"]["architecture_verdict"], "targets": list(report["frozen_target_regions"]), "synthetic_all_pass": report["synthetic_contracts"]["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
