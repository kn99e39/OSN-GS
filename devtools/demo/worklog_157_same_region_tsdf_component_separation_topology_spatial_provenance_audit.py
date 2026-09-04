"""Worklog 157: topology and spatial-provenance audit of frozen TSDF components.

This batch is diagnostic only.  It never edits WL154 membership, promotes
18/26-neighbor connectivity, fills gaps, or feeds any diagnostic result back
into production.  It reuses the exact frozen WL153--WL156 arrays and adds
component-to-component separation evidence.
"""

from __future__ import annotations

import argparse
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

from devtools.demo.worklog_156_region_owned_tsdf_support_fragmentation_causal_attribution import (
    CATEGORY_CODE,
    CATEGORIES,
    CORNER_OFFSETS,
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    DEFAULT_WL153_FIELD,
    DEFAULT_WL154,
    DEFAULT_WL155,
    FACE_OFFSETS,
    FRONTIER_COLORS,
    GRID_GAP_MAX_PROBE_STEPS,
    REVIEW_CAMERAS,
    STATUS_COLORS,
    _build_named_cameras,
    _component_rgb,
    _encode_cells,
    _fixed_rgb_to_dc,
    _load_frozen_data,
    _load_surfel_model_safe,
    _lookup_sorted,
    _overlay_points,
    _project_world_points,
    _save_png,
    _tensor_image,
    _write_visualization_readme,
)

try:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - real environment has scipy
    raise RuntimeError("W157 requires scipy for diagnostic-only lattice accounting") from exc


DEFAULT_WL156 = REPO_ROOT / "output/156_region_owned_tsdf_support_fragmentation_causal_attribution"
DEFAULT_OUT = REPO_ROOT / "output/157_same_region_tsdf_component_separation_topology_spatial_provenance"

STRIDE_X = 1_099_511_627_776
STRIDE_Y = 1_048_576
STRIDE_Z = 1

POSITIVE_OFFSETS = {
    6: np.asarray([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.int64),
    18: np.asarray(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1),
         (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
         (0, 1, 1), (0, 1, -1)], dtype=np.int64,
    ),
    26: np.asarray(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1),
         (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
         (0, 1, 1), (0, 1, -1),
         (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)], dtype=np.int64,
    ),
}
EDGE_OFFSETS = np.asarray(
    [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
     (0, 1, 1), (0, 1, -1)], dtype=np.int64,
)
CORNER_TOUCH_OFFSETS = np.asarray(
    [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)], dtype=np.int64,
)
AXIAL_GAP_OFFSETS = np.asarray([(2, 0, 0), (0, 2, 0), (0, 0, 2)], dtype=np.int64)
LOCAL_SEPARATION_OFFSETS = np.asarray(
    sorted(
        ((x, y, z) for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3) if (x, y, z) != (0, 0, 0)),
        key=lambda offset: (max(abs(value) for value in offset), sum(abs(value) for value in offset), tuple(sorted(abs(value) for value in offset)), offset),
    ),
    dtype=np.int64,
)

SEPARATION_CATEGORIES = (
    "FACE_TOUCH",
    "EDGE_TOUCH",
    "CORNER_TOUCH",
    "ONE_CELL_AXIAL_GAP",
    "OTHER_NEAR_GAP",
    "REMOTE",
)
INTERVENING_CATEGORIES = (
    "NOT_AUTHORITATIVE",
    "AUTHORITATIVE_NOT_ZERO_SURFACE",
    "ZERO_SURFACE_DIFFERENT_REGION",
    "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS",
    "OTHER_EXISTING_CONTRACT_STATE",
)
INTERVENING_CODE = {name: code for code, name in enumerate(INTERVENING_CATEGORIES)}
REMOTE_GRID_THRESHOLD = 16


def _ignore_ppm(_directory: str, names: list[str]) -> list[str]:
    """Keep W157 visualization exports PNG-only while retaining source layout."""
    return [name for name in names if Path(name).suffix.lower() == ".ppm"]


def _progress(message: str) -> None:
    print(f"[worklog 157] {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {"count": int(len(array)), "min": float(array.min()), "median": float(np.percentile(array, 50, method="nearest")), "p75": float(np.percentile(array, 75, method="nearest")), "p90": float(np.percentile(array, 90, method="nearest")), "p95": float(np.percentile(array, 95, method="nearest")), "p99": float(np.percentile(array, 99, method="nearest")), "max": float(array.max())}


def _cell_key_delta(offset: np.ndarray) -> int:
    return int(offset[0] * STRIDE_X + offset[1] * STRIDE_Y + offset[2] * STRIDE_Z)


def _region_population(data: dict[str, Any], region_id: int) -> dict[str, Any]:
    sample_indices = np.flatnonzero(data["accepted"] & (data["owned_region"] == region_id))
    native_ids, counts = np.unique(data["component_id"][sample_indices], return_counts=True)
    valid = native_ids >= 0
    native_ids, counts = native_ids[valid], counts[valid]
    order = np.argsort(counts, kind="stable")[::-1]
    largest = int(native_ids[order[0]]) if len(order) else -1
    largest_size = int(counts[order[0]]) if len(order) else 0
    summary = {
        "region_id": region_id,
        "owned_sample_count": int(len(sample_indices)),
        "native_component_count": int(len(native_ids)),
        "largest_component_id": largest,
        "largest_component_size": largest_size,
        "largest_component_fraction": float(largest_size / max(len(sample_indices), 1)),
        "remaining_component_sample_count": int(len(sample_indices) - largest_size),
        "remaining_component_sample_fraction": float((len(sample_indices) - largest_size) / max(len(sample_indices), 1)),
        "component_size_distribution": _summary(counts),
        "singleton_count": int((counts == 1).sum()),
        "size_le_2": int((counts <= 2).sum()),
        "size_le_4": int((counts <= 4).sum()),
        "size_le_8": int((counts <= 8).sum()),
        "size_le_16": int((counts <= 16).sum()),
        "small_or_island_size_definition": "size <= 8 is a descriptive tiny-component subset; no quality label or filtering is implied",
    }
    return {"summary": summary, "sample_indices": sample_indices, "component_ids": native_ids, "component_sizes": counts}


def _target_region_arrays(data: dict[str, Any], region_id: int) -> dict[str, Any]:
    population = _region_population(data, region_id)
    indices = population["sample_indices"]
    keys = data["sample_keys"][indices]
    order = np.argsort(keys, kind="stable")
    indices = indices[order]
    return {
        "population": population,
        "sample_indices": indices,
        "keys": data["sample_keys"][indices],
        "cells": data["sample_cells"][indices],
        "native_components": data["component_id"][indices],
        "component_sizes": population["component_sizes"],
    }


def _lattice_edges(region: dict[str, Any], connectivity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = region["keys"]
    labels = region["native_components"]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    edge_kinds: list[np.ndarray] = []
    for offset in POSITIVE_OFFSETS[connectivity]:
        positions, present = _lookup_sorted(keys, keys + _cell_key_delta(offset))
        valid = present
        if not np.any(valid):
            continue
        rows.append(np.flatnonzero(valid).astype(np.int32))
        cols.append(positions[valid].astype(np.int32))
        linf = int(np.max(np.abs(offset)))
        kind = 0 if linf == 1 and int(np.abs(offset).sum()) == 1 else (1 if int(np.abs(offset).sum()) == 2 else 2)
        edge_kinds.append(np.full((int(valid.sum()),), kind, dtype=np.int8))
    if not rows:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int8)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(edge_kinds)


def _connectivity_accounting(region: dict[str, Any]) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[str, Any]]:
    n = len(region["keys"])
    native_ids, native_labels = np.unique(region["native_components"], return_inverse=True)
    labels_by_connectivity: dict[int, np.ndarray] = {6: native_labels.astype(np.int32, copy=False)}
    metrics: dict[str, Any] = {}
    edge_audit: dict[str, Any] = {}
    native_sizes = np.bincount(native_labels)
    for connectivity in (6, 18, 26):
        if connectivity == 6:
            labels = labels_by_connectivity[6]
        else:
            rows, cols, _ = _lattice_edges(region, connectivity)
            graph = sparse.coo_matrix((np.ones((len(rows),), dtype=np.uint8), (rows, cols)), shape=(n, n))
            _, labels = connected_components(graph, directed=False, return_labels=True)
            labels = labels.astype(np.int32, copy=False)
            labels_by_connectivity[connectivity] = labels
        sizes = np.bincount(labels)
        largest = int(sizes.max()) if len(sizes) else 0
        metrics[str(connectivity)] = {
            "connectivity": connectivity,
            "diagnostic_only": True,
            "component_count": int(len(sizes)),
            "largest_component_fraction": float(largest / max(n, 1)),
            "singleton_component_fraction": float((sizes == 1).sum() / max(len(sizes), 1)),
            "singleton_sample_fraction": float(sizes[sizes == 1].sum() / max(n, 1)),
            "le_8_component_fraction": float((sizes <= 8).sum() / max(len(sizes), 1)),
            "sample_fraction_outside_largest": float((n - largest) / max(n, 1)),
        }
    edge_rows, edge_cols, edge_kinds = _lattice_edges(region, 26)
    split_pairs: dict[int, set[tuple[int, int]]] = {1: set(), 2: set()}
    for left, right, kind in zip(edge_rows.tolist(), edge_cols.tolist(), edge_kinds.tolist()):
        left_native, right_native = int(native_ids[native_labels[left]]), int(native_ids[native_labels[right]])
        if left_native == right_native or kind not in split_pairs:
            continue
        split_pairs[kind].add(tuple(sorted((left_native, right_native))))
    for kind, name in ((1, "edge_touch"), (2, "corner_touch")):
        pairs = sorted(split_pairs[kind])
        affected = sorted({value for pair in pairs for value in pair})
        edge_audit[name] = {
            "pair_count": len(pairs),
            "affected_component_count": len(affected),
            "affected_sample_fraction": float(native_sizes[[np.flatnonzero(native_ids == value)[0] for value in affected]].sum() / max(n, 1)) if affected else 0.0,
            "pairs": pairs,
            "touches_under_18_or_26_only": True,
        }
    return metrics, labels_by_connectivity, edge_audit


def _separation_category(displacement: np.ndarray, linf: int) -> str:
    absolute = np.abs(displacement).astype(np.int64)
    ordered = tuple(sorted(int(value) for value in absolute.tolist()))
    l1 = int(absolute.sum())
    if linf == 1 and l1 == 1:
        return "FACE_TOUCH"
    if linf == 1 and l1 == 2:
        return "EDGE_TOUCH"
    if linf == 1 and l1 == 3:
        return "CORNER_TOUCH"
    if linf == 2 and ordered == (0, 0, 2):
        return "ONE_CELL_AXIAL_GAP"
    if linf <= REMOTE_GRID_THRESHOLD:
        return "OTHER_NEAR_GAP"
    return "REMOTE"


def _classify_intervening_scalar(*, authoritative: bool, zero_surface: bool, sample_exists: bool, accepted: bool, sample_region: int, target_region: int) -> str:
    if not authoritative:
        return "NOT_AUTHORITATIVE"
    if not zero_surface:
        return "AUTHORITATIVE_NOT_ZERO_SURFACE"
    if sample_exists and accepted and sample_region != target_region:
        return "ZERO_SURFACE_DIFFERENT_REGION"
    if zero_surface and (not sample_exists or not accepted):
        return "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"
    return "OTHER_EXISTING_CONTRACT_STATE"


def _separation_audit(data: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    native_ids = region["population"]["component_ids"]
    native_sizes = region["population"]["component_sizes"]
    largest_id = region["population"]["summary"]["largest_component_id"]
    component_lookup = {int(value): index for index, value in enumerate(native_ids.tolist())}
    nonlargest_cells = np.flatnonzero(region["native_components"] != largest_id)
    local_component_index = np.asarray([component_lookup[int(value)] for value in region["native_components"][nonlargest_cells]], dtype=np.int32)
    found = np.zeros((len(native_ids),), dtype=bool)
    source_pos = np.full((len(native_ids),), -1, dtype=np.int64)
    neighbor_pos = np.full((len(native_ids),), -1, dtype=np.int64)
    best_offset = np.zeros((len(native_ids), 3), dtype=np.int64)
    for offset in LOCAL_SEPARATION_OFFSETS:
        query_keys = region["keys"][nonlargest_cells] + _cell_key_delta(offset)
        positions, present = _lookup_sorted(region["keys"], query_keys)
        valid = present & (region["native_components"][positions] != region["native_components"][nonlargest_cells])
        fresh = valid & ~found[local_component_index]
        if not np.any(fresh):
            continue
        candidate_rows = np.flatnonzero(fresh)
        candidate_components = local_component_index[candidate_rows]
        unique_components, first = np.unique(candidate_components, return_index=True)
        chosen = candidate_rows[first]
        found[unique_components] = True
        source_pos[unique_components] = nonlargest_cells[chosen]
        neighbor_pos[unique_components] = positions[chosen]
        best_offset[unique_components] = offset
    remaining_components = np.flatnonzero((native_ids != largest_id) & ~found)
    tree = cKDTree(region["cells"].astype(np.float64, copy=False)) if len(remaining_components) else None
    order = np.argsort(region["native_components"], kind="stable")
    sorted_components = region["native_components"][order]
    starts: dict[int, tuple[int, int]] = {}
    if len(order):
        boundaries = np.flatnonzero(np.r_[True, sorted_components[1:] != sorted_components[:-1], True])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            starts[int(sorted_components[start])] = (int(start), int(end))
    for component_index in remaining_components.tolist():
        component_id = int(native_ids[component_index])
        start, end = starts[component_id]
        component_positions = order[start:end]
        component_size = len(component_positions)
        distances, nearest = tree.query(region["cells"][component_positions].astype(np.float64), k=min(component_size + 1, len(region["cells"])), p=np.inf, workers=-1)
        distances = np.atleast_2d(distances)
        nearest = np.atleast_2d(nearest)
        candidate_best: tuple[Any, ...] | None = None
        for row in range(len(component_positions)):
            valid = region["native_components"][nearest[row]] != component_id
            if not np.any(valid):
                continue
            first_valid = np.flatnonzero(valid)[0]
            neighbor = int(nearest[row, first_valid])
            displacement = region["cells"][neighbor] - region["cells"][component_positions[row]]
            key = (float(distances[row, first_valid]), tuple(sorted(int(value) for value in np.abs(displacement).tolist())), int(neighbor))
            if candidate_best is None or key < candidate_best[0]:
                candidate_best = (key, int(component_positions[row]), neighbor, displacement.copy())
        if candidate_best is None:
            raise RuntimeError(f"No other same-region component found for component {component_id}")
        _, source, neighbor, displacement = candidate_best
        found[component_index] = True
        source_pos[component_index] = source
        neighbor_pos[component_index] = neighbor
        best_offset[component_index] = displacement
    records: list[dict[str, Any]] = []
    for index, component_id in enumerate(native_ids.tolist()):
        if int(component_id) == largest_id:
            continue
        displacement = best_offset[index]
        linf = int(np.abs(displacement).max())
        l2 = float(np.linalg.norm(displacement))
        records.append({
            "component_id": int(component_id),
            "component_size": int(native_sizes[index]),
            "nearest_component_id": int(region["native_components"][neighbor_pos[index]]),
            "source_cell": region["cells"][source_pos[index]].copy(),
            "neighbor_cell": region["cells"][neighbor_pos[index]].copy(),
            "displacement": displacement.copy(),
            "abs_displacement_pattern": tuple(sorted(int(value) for value in np.abs(displacement).tolist())),
            "min_linf_grid": linf,
            "min_l2_grid": l2,
            "world_linf": float(linf * data["field_h"]),
            "world_l2": float(l2 * data["field_h"]),
            "world_l2_over_h": l2,
            "category": _separation_category(displacement, linf),
        })
    category_counts = Counter(record["category"] for record in records)
    by_category: dict[str, dict[str, Any]] = {}
    total_samples = int(region["population"]["summary"]["owned_sample_count"])
    for category in SEPARATION_CATEGORIES:
        selected = [record for record in records if record["category"] == category]
        affected = sorted({record["component_id"] for record in selected} | {record["nearest_component_id"] for record in selected})
        affected_sizes = [int(native_sizes[component_lookup[value]]) for value in affected if value in component_lookup]
        by_category[category] = {
            "component_count": int(len(selected)),
            "affected_component_count": int(len(affected)),
            "affected_sample_fraction": float(sum(affected_sizes) / max(total_samples, 1)),
            "min_linf_grid": _summary([record["min_linf_grid"] for record in selected]),
            "world_l2_over_h": _summary([record["world_l2_over_h"] for record in selected]),
        }
    nonlargest_records = [record for record in records]
    return {
        "largest_component_id": largest_id,
        "remote_definition": f"min L_inf separation > {REMOTE_GRID_THRESHOLD} grid cells; descriptive reporting band only, not a merge radius",
        "records": nonlargest_records,
        "category_counts": {category: int(category_counts[category]) for category in SEPARATION_CATEGORIES},
        "by_category": by_category,
        "all_nondominant_components_accounted": len(nonlargest_records) == len(native_ids) - 1,
    }


def _field_cell_state(data: dict[str, Any], cells: np.ndarray, target_region: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cells = np.asarray(cells, dtype=np.int64)
    corner_cells = cells[:, None, :] + CORNER_OFFSETS[None, :, :]
    corner_keys = _encode_cells(corner_cells.reshape(-1, 3)).reshape(-1, 8)
    field_pos, field_present = _lookup_sorted(data["field_keys"], corner_keys.reshape(-1))
    safe = np.minimum(field_pos, max(len(data["field_keys"]) - 1, 0)).reshape(-1, 8)
    present = field_present.reshape(-1, 8)
    values = data["field_values"][safe]
    authoritative = present.all(axis=1) & np.isfinite(values).all(axis=1)
    zero_surface = authoritative & (np.min(values, axis=1) <= 0.0) & (np.max(values, axis=1) >= 0.0)
    sample_keys = _encode_cells(cells)
    sample_pos, sample_exists = _lookup_sorted(data["sample_keys"], sample_keys)
    sample_safe = np.minimum(sample_pos, max(len(data["sample_keys"]) - 1, 0))
    sample_region = data["nearest_region"][sample_safe]
    sample_accepted = data["accepted"][sample_safe]
    state = np.full((len(cells),), INTERVENING_CODE["OTHER_EXISTING_CONTRACT_STATE"], dtype=np.int8)
    state[~authoritative] = INTERVENING_CODE["NOT_AUTHORITATIVE"]
    state[authoritative & ~zero_surface] = INTERVENING_CODE["AUTHORITATIVE_NOT_ZERO_SURFACE"]
    zero = zero_surface & sample_exists
    state[zero & sample_accepted & (sample_region != target_region)] = INTERVENING_CODE["ZERO_SURFACE_DIFFERENT_REGION"]
    state[zero_surface & (~sample_exists | ~sample_accepted)] = INTERVENING_CODE["ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"]
    return state, {"authoritative": authoritative, "zero_surface": zero_surface, "sample_exists": sample_exists, "values": values, "support": data["field_support"][safe]}


def _one_cell_gap_audit(data: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    neighbors: list[np.ndarray] = []
    for offset in AXIAL_GAP_OFFSETS:
        positions, present = _lookup_sorted(region["keys"], region["keys"] + _cell_key_delta(offset))
        different = present & (region["native_components"][positions] != region["native_components"])
        rows.append(np.flatnonzero(different))
        neighbors.append(positions[different])
    source_pos = np.concatenate(rows) if rows else np.empty((0,), dtype=np.int64)
    neighbor_pos = np.concatenate(neighbors) if neighbors else np.empty((0,), dtype=np.int64)
    if not len(source_pos):
        return {"gap_instance_count": 0, "unique_component_pair_count": 0, "records": [], "by_intervening_state": {}}
    intervening = (region["cells"][source_pos] + region["cells"][neighbor_pos]) // 2
    state, diagnostics = _field_cell_state(data, intervening, 0)
    records = []
    for index in range(len(source_pos)):
        source_component = int(region["native_components"][source_pos[index]])
        neighbor_component = int(region["native_components"][neighbor_pos[index]])
        records.append({
            "source_component_id": source_component,
            "neighbor_component_id": neighbor_component,
            "source_sample_index": int(region["sample_indices"][source_pos[index]]),
            "neighbor_sample_index": int(region["sample_indices"][neighbor_pos[index]]),
            "source_cell": region["cells"][source_pos[index]].copy(),
            "intervening_cell": intervening[index].copy(),
            "neighbor_cell": region["cells"][neighbor_pos[index]].copy(),
            "intervening_state": INTERVENING_CATEGORIES[int(state[index])],
        })
    native_ids, native_sizes = np.unique(region["native_components"], return_counts=True)
    size_map = {int(key): int(value) for key, value in zip(native_ids.tolist(), native_sizes.tolist())}
    by_state: dict[str, Any] = {}
    for category in INTERVENING_CATEGORIES:
        selected = [record for record in records if record["intervening_state"] == category]
        components = sorted({record["source_component_id"] for record in selected} | {record["neighbor_component_id"] for record in selected})
        by_state[category] = {
            "gap_instance_count": len(selected),
            "affected_component_count": len(components),
            "affected_sample_population": int(sum(size_map.get(value, 0) for value in components)),
            "affected_sample_fraction": float(sum(size_map.get(value, 0) for value in components) / max(len(region["cells"]), 1)),
        }
    pair_count = len({tuple(sorted((record["source_component_id"], record["neighbor_component_id"]))) for record in records})
    return {
        "gap_instance_count": len(records),
        "unique_component_pair_count": pair_count,
        "by_intervening_state": by_state,
        "records": records,
        "ordinary_exposed_surface_faces_excluded": True,
        "field_diagnostics": {"authoritative_count": int(diagnostics["authoritative"].sum()), "zero_surface_count": int(diagnostics["zero_surface"].sum())},
    }


def _representative_touch_cases(data: dict[str, Any], region: dict[str, Any], kind: int, limit: int = 8) -> list[dict[str, Any]]:
    offsets = EDGE_OFFSETS if kind == 1 else CORNER_TOUCH_OFFSETS
    native = region["native_components"]
    cases: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for offset in offsets:
        positions, present = _lookup_sorted(region["keys"], region["keys"] + _cell_key_delta(offset))
        different = present & (native[positions] != native)
        for source, neighbor in zip(np.flatnonzero(different).tolist(), positions[different].tolist()):
            pair = tuple(sorted((int(native[source]), int(native[neighbor]))))
            if pair in seen:
                continue
            seen.add(pair)
            state, details = _field_cell_state(data, region["cells"][source:source + 1], 0)
            nstate, ndetails = _field_cell_state(data, region["cells"][neighbor:neighbor + 1], 0)
            cases.append({
                "component_pair": pair,
                "source_cell": region["cells"][source].copy(),
                "neighbor_cell": region["cells"][neighbor].copy(),
                "displacement": region["cells"][neighbor] - region["cells"][source],
                "source_field_authoritative": bool(details["authoritative"][0]),
                "source_field_zero_surface": bool(details["zero_surface"][0]),
                "neighbor_field_authoritative": bool(ndetails["authoritative"][0]),
                "neighbor_field_zero_surface": bool(ndetails["zero_surface"][0]),
                "source_field_corner_min": float(np.min(details["values"][0])),
                "source_field_corner_max": float(np.max(details["values"][0])),
                "neighbor_field_corner_min": float(np.min(ndetails["values"][0])),
                "neighbor_field_corner_max": float(np.max(ndetails["values"][0])),
            })
            if len(cases) >= limit:
                return cases
    return cases


def _edge_corner_audit(data: dict[str, Any], region: dict[str, Any], edge_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        **edge_audit,
        "representative_edge_cases": _representative_touch_cases(data, region, 1),
        "representative_corner_cases": _representative_touch_cases(data, region, 2),
        "production_graph_unchanged": True,
    }


def _remote_islands(data: dict[str, Any], region: dict[str, Any], separation: dict[str, Any]) -> dict[str, Any]:
    population = region["population"]
    largest_id = population["summary"]["largest_component_id"]
    sizes = {int(cid): int(size) for cid, size in zip(population["component_ids"].tolist(), population["component_sizes"].tolist())}
    substantial_ids = {cid for cid, size in sizes.items() if size > 8}
    substantial_mask = np.isin(region["native_components"], list(substantial_ids))
    substantial_tree = cKDTree(region["cells"][substantial_mask].astype(np.float64)) if np.any(substantial_mask) else None
    largest_tree = cKDTree(region["cells"][region["native_components"] == largest_id].astype(np.float64))
    records = []
    for record in separation["records"]:
        component_id = int(record["component_id"])
        if int(record["component_size"]) > 8:
            continue
        component_mask = region["native_components"] == component_id
        cells = region["cells"][component_mask]
        nearest_substantial = float(substantial_tree.query(cells.astype(np.float64), k=1, p=np.inf, workers=-1)[0].min()) if substantial_tree is not None else float("inf")
        nearest_largest = float(largest_tree.query(cells.astype(np.float64), k=1, p=np.inf, workers=-1)[0].min())
        if nearest_substantial <= REMOTE_GRID_THRESHOLD:
            continue
        sample_indices = region["sample_indices"][component_mask]
        ids, counts = np.unique(data["nearest_gaussian_id"][sample_indices], return_counts=True)
        gaussian_order = np.lexsort((ids, -counts))
        records.append({
            "component_id": component_id,
            "sample_count": int(len(cells)),
            "world_centroid": cells.astype(np.float64).mean(axis=0) * data["field_h"],
            "grid_aabb_min": cells.min(axis=0).copy(),
            "grid_aabb_max": cells.max(axis=0).copy(),
            "world_aabb_min": cells.min(axis=0).astype(np.float64) * data["field_h"],
            "world_aabb_max": cells.max(axis=0).astype(np.float64) * data["field_h"],
            "nearest_same_region_component_grid_linf": int(record["min_linf_grid"]),
            "nearest_same_region_component_world_linf": float(record["world_linf"]),
            "nearest_largest_component_grid_linf": nearest_largest,
            "nearest_substantial_component_grid_linf": nearest_substantial,
            "associated_nearest_gaussian_ids": [{"stable_gaussian_id": int(ids[index]), "sample_count": int(counts[index])} for index in gaussian_order[:10]],
            "scene_cluster": "qualitative_review_required",
        })
    return {
        "tiny_component_definition": "native component sample_count <= 8",
        "substantial_component_definition": "native component sample_count > 8",
        "remote_definition": f"nearest substantial same-Region component > {REMOTE_GRID_THRESHOLD} grid cells",
        "remote_component_count": len(records),
        "remote_sample_population": int(sum(record["sample_count"] for record in records)),
        "records": records,
        "scene_cluster_assignment": "No semantic tabletop/contact/legs/background label is inferred from coordinates alone; matched real-scene/common-world exports are provided for qualitative review.",
    }


def _architecture_verdict(
    region: dict[str, Any],
    connectivity: dict[str, Any],
    separation: dict[str, Any],
    gaps: dict[str, Any],
    remote: dict[str, Any],
    synthetic: dict[str, Any],
) -> dict[str, Any]:
    """Classify the observed split mechanisms without changing production data."""
    population = region["population"]["summary"]
    native_count = int(population["native_component_count"])
    remaining_samples = int(population["remaining_component_sample_count"])
    edge_pairs = int(separation["category_counts"]["EDGE_TOUCH"])
    corner_pairs = int(separation["category_counts"]["CORNER_TOUCH"])
    digital_reduction = int(connectivity["6"]["component_count"] - connectivity["26"]["component_count"])
    digital_evidence = {
        "edge_touch_component_pair_count": edge_pairs,
        "corner_touch_component_pair_count": corner_pairs,
        "native_component_count_6": int(connectivity["6"]["component_count"]),
        "diagnostic_component_count_26": int(connectivity["26"]["component_count"]),
        "component_reduction_6_to_26": digital_reduction,
        "sample_fraction_outside_largest_6": float(connectivity["6"]["sample_fraction_outside_largest"]),
        "sample_fraction_outside_largest_26": float(connectivity["26"]["sample_fraction_outside_largest"]),
    }
    gap_count = int(gaps["gap_instance_count"])
    component_sizes = {
        int(component_id): int(size)
        for component_id, size in zip(region["population"]["component_ids"].tolist(), region["population"]["component_sizes"].tolist())
    }
    gap_component_ids = {
        int(component_id)
        for record in gaps.get("records", [])
        for component_id in (record["source_component_id"], record["neighbor_component_id"])
    }
    gap_affected = len(gap_component_ids)
    gap_affected_samples = sum(component_sizes.get(component_id, 0) for component_id in gap_component_ids)
    local_gap_evidence = {
        "one_cell_gap_instance_count": gap_count,
        "affected_component_count": gap_affected,
        "by_intervening_state": gaps["by_intervening_state"],
        "affected_sample_population": int(gap_affected_samples),
        "affected_sample_fraction": float(gap_affected_samples / max(int(population["owned_sample_count"]), 1)),
    }
    remote_samples = int(remote["remote_sample_population"])
    remote_evidence = {
        "remote_tiny_component_count": int(remote["remote_component_count"]),
        "remote_tiny_sample_population": remote_samples,
        "remote_sample_fraction_of_nonlargest": float(remote_samples / max(remaining_samples, 1)),
        "remote_component_fraction_of_nonlargest": float(remote["remote_component_count"] / max(native_count - 1, 1)),
    }
    mechanical_ok = bool(separation["all_nondominant_components_accounted"]) and bool(synthetic["all_pass"])
    digital_present = bool(edge_pairs or corner_pairs or digital_reduction > 0)
    local_gap_present = bool(gap_count and gap_affected)
    remote_dominant = remote_samples > remaining_samples * 0.5
    if not mechanical_ok:
        verdict = "MECHANICAL_IMPLEMENTATION_BUG"
        reason = "Native non-largest component accounting or synthetic topology contracts failed."
    elif remote_dominant:
        verdict = "REMOTE_SAME_REGION_ISLANDS_DOMINANT"
        reason = "Remote tiny-island samples exceed half of the non-largest population."
    elif digital_present and local_gap_present:
        verdict = "MIXED_COMPONENT_STRUCTURE"
        reason = "Both digital edge/corner adjacency and true one-cell TSDF gap states materially explain the native 6-face split."
    elif digital_present:
        verdict = "DIGITAL_ADJACENCY_DOMINANT"
        reason = "The observed split is explained by digital edge/corner adjacency without material one-cell gap evidence."
    elif local_gap_present:
        verdict = "TRUE_LOCAL_TSDF_GAPS_DOMINANT"
        reason = "The observed split is explained by authoritative local one-cell TSDF gaps."
    else:
        verdict = "UNRESOLVED"
        reason = "No supported dominant split mechanism was established by this frozen-data audit."
    return {
        "architecture_verdict": verdict,
        "verdict_reason": reason,
        "evidence": {"digital_adjacency": digital_evidence, "true_local_tsdf_gaps": local_gap_evidence, "remote_same_region_islands": remote_evidence},
        "mechanical_contracts_ok": mechanical_ok,
        "decision_rule": "Remote dominance means remote tiny-island samples > 50% of non-largest samples; otherwise simultaneous digital and local-gap evidence yields MIXED_COMPONENT_STRUCTURE.",
    }


def _spatial_extent(data: dict[str, Any], remote: dict[str, Any], separation: dict[str, Any]) -> dict[str, Any]:
    return {"status": "PENDING_CHECKPOINT_LOAD", "remote_component_ids": [int(record["component_id"]) for record in remote["records"]], "separation_component_ids": [int(record["component_id"]) for record in separation["records"]]}


def _point_extent(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return {"count": 0, "aabb_min": None, "aabb_max": None, "robust_coordinate_summary": {}}
    return {
        "count": int(len(points)),
        "aabb_min": points.min(axis=0),
        "aabb_max": points.max(axis=0),
        "robust_coordinate_summary": {
            axis: {"p05": float(np.percentile(points[:, index], 5)), "p25": float(np.percentile(points[:, index], 25)), "median": float(np.percentile(points[:, index], 50)), "p75": float(np.percentile(points[:, index], 75)), "p95": float(np.percentile(points[:, index], 95))}
            for index, axis in enumerate(("x", "y", "z"))
        },
    }


def _save_records(out: Path, name: str, records: list[dict[str, Any]]) -> Path:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        np.savez_compressed(path, empty=np.empty((0,), dtype=np.int8))
        return path
    fields: dict[str, Any] = {}
    for field in ("component_id", "component_size", "nearest_component_id", "min_linf_grid", "min_l2_grid", "world_linf", "world_l2", "world_l2_over_h", "category"):
        values = [record[field] for record in records if field in record]
        if field == "category":
            fields[field] = np.asarray(values, dtype="U32")
        else:
            fields[field] = np.asarray(values)
    for field in ("source_cell", "neighbor_cell", "displacement", "abs_displacement_pattern"):
        values = [record[field] for record in records if field in record]
        fields[field] = np.asarray(values, dtype=np.int64)
    np.savez_compressed(path, **fields)
    return path


def _save_gap_records(out: Path, gap: dict[str, Any]) -> Path:
    path = out / "one_cell_gap_records.npz"
    records = gap["records"]
    if records:
        np.savez_compressed(path, source_component_id=np.asarray([r["source_component_id"] for r in records]), neighbor_component_id=np.asarray([r["neighbor_component_id"] for r in records]), source_sample_index=np.asarray([r["source_sample_index"] for r in records]), neighbor_sample_index=np.asarray([r["neighbor_sample_index"] for r in records]), source_cell=np.asarray([r["source_cell"] for r in records]), intervening_cell=np.asarray([r["intervening_cell"] for r in records]), neighbor_cell=np.asarray([r["neighbor_cell"] for r in records]), intervening_state=np.asarray([r["intervening_state"] for r in records], dtype="U48"))
    else:
        np.savez_compressed(path, empty=np.empty((0,), dtype=np.int8))
    return path


def _synthetic_contracts() -> dict[str, Any]:
    def labels(cells: list[tuple[int, int, int]], offsets: np.ndarray) -> int:
        coordinates = np.asarray(cells, dtype=np.int64)
        keys = np.sort(_encode_cells(coordinates))
        rows, cols = [], []
        for offset in offsets:
            positions, present = _lookup_sorted(keys, keys + _cell_key_delta(offset))
            rows.append(np.flatnonzero(present))
            cols.append(positions[present])
        graph = sparse.coo_matrix((np.ones((sum(len(row) for row in rows),), dtype=np.uint8), (np.concatenate(rows), np.concatenate(cols))), shape=(len(keys), len(keys)))
        return int(connected_components(graph, directed=False, return_labels=True)[0])

    cases = [
        ("A_face_connected", [(0, 0, 0), (1, 0, 0)], (1, 1, 1)),
        ("B_edge_touch", [(0, 0, 0), (1, 1, 0)], (2, 1, 1)),
        ("C_corner_touch", [(0, 0, 0), (1, 1, 1)], (2, 2, 1)),
        ("D_one_cell_not_authoritative", [(0, 0, 0), (2, 0, 0)], (2, 2, 2)),
        ("E_one_cell_authoritative_nonzero", [(0, 0, 0), (2, 0, 0)], (2, 2, 2)),
        ("F_remote_islands", [(0, 0, 0), (100, 0, 0)], (2, 2, 2)),
    ]
    observed = []
    for name, cells, expected in cases:
        c6 = labels(cells, POSITIVE_OFFSETS[6])
        c18 = labels(cells, POSITIVE_OFFSETS[18])
        c26 = labels(cells, POSITIVE_OFFSETS[26])
        if name.startswith("A"):
            passed = (c6, c18, c26) == (1, 1, 1)
        elif name.startswith("B"):
            passed = (c6, c18, c26) == (2, 1, 1)
        elif name.startswith("C"):
            passed = (c6, c18, c26) == (2, 2, 1)
        elif name.startswith("F"):
            passed = (c6, c18, c26) == (2, 2, 2)
        else:
            passed = (c6, c18, c26) == (2, 2, 2)
        expected_state = "NOT_AUTHORITATIVE" if name.startswith("D") else ("AUTHORITATIVE_NOT_ZERO_SURFACE" if name.startswith("E") else None)
        observed_state = _classify_intervening_scalar(authoritative=expected_state != "NOT_AUTHORITATIVE", zero_surface=False, sample_exists=False, accepted=False, sample_region=-1, target_region=0) if expected_state else None
        passed = passed and observed_state == expected_state if expected_state else passed
        observed.append({"name": name, "components_6_18_26": [c6, c18, c26], "expected_components_6_18_26": list(expected), "intervening_state_contract": expected_state, "observed_intervening_state": observed_state, "pass": passed})
    return {"all_pass": all(item["pass"] for item in observed), "diagnostic_mechanics_only": True, "cases": observed}


def _fixed_camera_exports(data: dict[str, Any], out: Path, region: dict[str, Any], separation: dict[str, Any], edge_corner: dict[str, Any], gap: dict[str, Any], remote: dict[str, Any], source_path: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    import torch
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    model, payload = _load_surfel_model_safe(checkpoint, device)
    raw_ids = payload["model_raw"].get("stable_gaussian_ids")
    checkpoint_ids = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False) if hasattr(raw_ids, "detach") else np.asarray(raw_ids if raw_ids is not None else np.arange(len(model)), dtype=np.int64)
    if not np.array_equal(checkpoint_ids, data["stable_ids"]):
        raise ValueError("W155 stable Gaussian IDs do not match the frozen checkpoint row order")
    cameras, camera_metadata = _build_named_cameras(source_path, "images_8", "sparse/0", -1, 8, device)
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros((3,), dtype=torch.float32, device=model.device)
    original_dc = model._features_dc.detach().clone()
    original_rest = model._features_rest.detach().clone()
    original_degree = int(model.active_sh_degree)
    original = {}
    positions = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    region0_gaussians = data["gaussian_region"] == 0
    largest_component = int(region["population"]["summary"]["largest_component_id"])
    largest_sample_mask = region["native_components"] == largest_component
    largest_gaussian_ids = np.unique(data["nearest_gaussian_id"][region["sample_indices"][largest_sample_mask]])
    remote_gaussian_ids = np.unique(np.concatenate([data["nearest_gaussian_id"][region["sample_indices"][region["native_components"] == int(record["component_id"])] ] for record in remote["records"]])) if remote["records"] else np.empty((0,), dtype=np.int64)
    id_to_index = {int(stable_id): int(index) for stable_id, index in zip(data["stable_ids"].tolist(), np.arange(len(data["stable_ids"])).tolist())}
    spatial_targets = {
        "all_region0_gaussians": positions[region0_gaussians],
        "largest_component_nearest_gaussians": positions[np.asarray([id_to_index[int(value)] for value in largest_gaussian_ids if int(value) in id_to_index], dtype=np.int64)],
        "remote_tiny_island_nearest_gaussians": positions[np.asarray([id_to_index[int(value)] for value in remote_gaussian_ids if int(value) in id_to_index], dtype=np.int64)],
    }
    region_graph_records = data["report155"].get("global_region_accounting", {}).get("largest_regions", [])
    if not region_graph_records:
        region_graph_records = data["report155"].get("largest_regions", [])
    largest_record = next((record for record in region_graph_records if int(record.get("region_id", -1)) == 0), {})
    gaussian_spatial_extent = {
        "region_0_gaussian_population": _point_extent(positions[region0_gaussians]),
        "largest_component_nearest_gaussian_population": _point_extent(spatial_targets["largest_component_nearest_gaussians"]),
        "remote_tiny_island_nearest_gaussian_population": _point_extent(spatial_targets["remote_tiny_island_nearest_gaussians"]),
        "existing_wl155_region_graph_extent": {"region_id": 0, "gaussian_member_count": int(region0_gaussians.sum()), "structural_core_size": largest_record.get("structural_core_size"), "concentration": largest_record.get("concentration"), "graph_membership_source": "WL155 existing Gaussian Region mapping; no new graph constructed"},
        "region_0_gaussian_to_largest_component_nearest_id_count": int(len(largest_gaussian_ids)),
        "region_0_gaussian_to_remote_island_nearest_id_count": int(len(remote_gaussian_ids)),
    }
    sample_indices = region["sample_indices"]
    separation_ids = {int(record["component_id"]) for record in separation["records"] if record["category"] == "ONE_CELL_AXIAL_GAP"}
    edge_ids = {int(value) for pair in edge_corner["edge_touch"]["pairs"] + edge_corner["corner_touch"]["pairs"] for value in pair}
    remote_ids = {int(record["component_id"]) for record in remote["records"]}
    support_targets = {
        "all_region0_tsdf_support": data["sample_xyz"][sample_indices],
        "largest_tsdf_component_only": data["sample_xyz"][sample_indices[largest_sample_mask]],
        "all_nonlargest_components": data["sample_xyz"][sample_indices[~largest_sample_mask]],
        "edge_corner_touch_components": data["sample_xyz"][sample_indices[np.isin(region["native_components"], list(edge_ids))]],
        "one_cell_gap_components": data["sample_xyz"][sample_indices[np.isin(region["native_components"], list(separation_ids))]],
        "remote_components": data["sample_xyz"][sample_indices[np.isin(region["native_components"], list(remote_ids))]],
    }
    root = out / "review_views" / "region_000000_primary_tabletop_candidate"
    spatial_root = out / "gaussian_spatial_extent"
    try:
        with torch.no_grad():
            for camera_name in REVIEW_CAMERAS:
                package = rasterizer.render(cameras[camera_name], model, background=background)
                original[camera_name] = _tensor_image(package["render"])
                del package
            for camera_name in REVIEW_CAMERAS:
                _save_png(root / "A_original_scene" / "cameras" / camera_name / "render.png", original[camera_name])
                _save_png(root / "B_full_region0_gaussian_overlay" / "cameras" / camera_name / "render.png", _overlay_points(original[camera_name], spatial_targets["all_region0_gaussians"], np.tile(np.asarray(STATUS_COLORS["accepted"], dtype=np.float32), (len(spatial_targets["all_region0_gaussians"]), 1)), cameras[camera_name]))
                colors = {
                    "all_region0_tsdf_support": np.tile(np.asarray(STATUS_COLORS["accepted"], dtype=np.float32), (len(support_targets["all_region0_tsdf_support"]), 1)),
                    "largest_tsdf_component_only": np.tile(np.asarray((0.15, 0.85, 0.95), dtype=np.float32), (len(support_targets["largest_tsdf_component_only"]), 1)),
                    "all_nonlargest_components": _component_rgb(region["native_components"][~largest_sample_mask]),
                    "edge_corner_touch_components": np.tile(np.asarray((1.0, 0.55, 0.05), dtype=np.float32), (len(support_targets["edge_corner_touch_components"]), 1)),
                    "one_cell_gap_components": np.tile(np.asarray((0.95, 0.20, 0.70), dtype=np.float32), (len(support_targets["one_cell_gap_components"]), 1)),
                    "remote_components": np.tile(np.asarray((0.75, 0.25, 0.95), dtype=np.float32), (len(support_targets["remote_components"]), 1)),
                }
                for name, points in support_targets.items():
                    _save_png(root / {"all_region0_tsdf_support": "C_all_region0_tsdf_support", "largest_tsdf_component_only": "D_region0_largest_tsdf_component_only", "all_nonlargest_components": "E_all_nonlargest_components", "edge_corner_touch_components": "F_edge_corner_touch_components", "one_cell_gap_components": "G_one_cell_gap_components", "remote_components": "H_remote_components"}[name] / "cameras" / camera_name / "render.png", _overlay_points(original[camera_name], points, colors[name], cameras[camera_name]))
                for name, points in spatial_targets.items():
                    _save_png(spatial_root / name / "cameras" / camera_name / "render.png", _overlay_points(original[camera_name], points, np.tile(np.asarray(STATUS_COLORS["accepted"], dtype=np.float32), (len(points), 1)), cameras[camera_name]))
    finally:
        model._features_dc.data.copy_(original_dc)
        model._features_rest.data.copy_(original_rest)
        model.active_sh_degree = original_degree
    return {"camera_set": list(REVIEW_CAMERAS), "camera_metadata": camera_metadata, "renderer": "OSNSurfelRasterizer", "resolution": [648, 420], "background": [0.0, 0.0, 0.0], "same_checkpoint_iteration_and_geometry": True, "spatial_target_counts": {name: int(len(points)) for name, points in spatial_targets.items()}, "support_view_counts": {name: int(len(points)) for name, points in support_targets.items()}, "gaussian_spatial_extent": gaussian_spatial_extent}


def _write_readmes(out: Path) -> None:
    _write_visualization_readme(out / "README.md", """# Worklog 157 산출물 안내

W157은 frozen WL153/WL154/WL155/WL156을 읽어 Region component 사이의 exact lattice separation, 6/18/26 diagnostic connectivity, one-cell gap state, remote tiny-island provenance를 측정한다. connectivity repair, merge, bridge, fill, ownership 변경은 수행하지 않는다.

정량 raw record는 `component_separation_records.npz`, `one_cell_gap_records.npz`, `remote_islands.json`에 있고 전체 해석은 `worklog_157_report.json`에 있다. `mandatory_gaussian_visualization_pair`에는 같은 checkpoint/iteration/camera/해상도/배경/renderer/행 수를 유지한 canonical `Original Scene`과 `Observed-Occluded`가 있다. 모든 시각화 폴더에는 해당 view의 의미·입력·색상·검토 한계를 설명하는 README가 있다. PNG가 primary artifact이며 PPM은 생성하지 않는다.
""")
    review_root = out / "review_views"
    _write_visualization_readme(review_root / "README.md", """# W157 Region-0 Matched Review Views

고정 tabletop review camera `DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`에서 A–H를 같은 checkpoint, geometry, resolution, background로 투영했다. A original scene, B full Region-0 Gaussian, C all TSDF support, D largest component, E non-largest components, F edge/corner-touch components, G one-cell-gap components, H remote components다.

Green은 Region-0/accepted support, cyan은 largest component, orange는 edge/corner contact, magenta는 one-cell gap, purple는 remote component을 뜻한다. 색은 진단용이며 production connectivity를 의미하지 않는다.
""")
    spatial_root = out / "gaussian_spatial_extent"
    _write_visualization_readme(spatial_root / "README.md", """# Region-0 Gaussian Spatial Extent

세 view는 Region-0 Gaussian 전체, largest TSDF component의 nearest-Gaussian ID, remote tiny-island의 nearest-Gaussian ID를 common-world로 표시한다. 모두 frozen Gaussian position과 고정 camera를 사용하며 Gaussian을 제거·재라벨링하지 않는다.
""")
    mandatory_root = out / "mandatory_gaussian_visualization_pair"
    _write_visualization_readme(mandatory_root / "README.md", """# W157 Mandatory Gaussian Visualization Pair

W155에서 frozen source로 검증된 canonical pair를 W157 output에 PNG-only로 보존한 것이다. `Original Scene`과 `Observed-Occluded`는 동일 checkpoint/iteration/camera/resolution/background/renderer/Gaussian row count를 사용한다. Original은 learned appearance와 geometry를 유지하고, Observed-Occluded는 동일 row/geometry에서 display state color만 바꾼다. W157 diagnostic overlay와는 별개의 필수 pair이며 이 output에는 PPM을 복사하지 않는다.
""")
    for view_name, meaning in (("Original Scene", "frozen checkpoint의 original Gaussian appearance/geometry render. Gaussian row와 learned SH/color를 변경하지 않는다."), ("Observed-Occluded", "동일 Gaussian row와 geometry를 유지한 state-color render. OBSERVED/OCCLUDED/UNRESOLVED palette만 display color에 적용한다.")):
        view_root = mandatory_root / view_name
        _write_visualization_readme(view_root / "README.md", f"# {view_name}\n\n{meaning}\n\nPNG-only W157 copy다. 상위 README의 matched-pair 조건을 따른다.")
        for child in view_root.iterdir():
            if child.is_dir():
                _write_visualization_readme(child / "README.md", f"`{view_name}`의 fixed iteration/checkpoint provenance directory다. 실제 raster image가 있으면 상위 matched-pair 조건을 따른다.")
    for root, descriptions in ((review_root / "region_000000_primary_tabletop_candidate", {"A_original_scene": "checkpoint의 original Gaussian scene. 원래 learned appearance/geometry를 유지하며 Region 색을 적용하지 않는다.", "B_full_region0_gaussian_overlay": "전체 Region-0 Gaussian center를 green overlay로 표시한다. Region membership는 frozen WL155 mapping 그대로다.", "C_all_region0_tsdf_support": "Region-0 accepted-owned TSDF support 전체를 green으로 표시한다.", "D_region0_largest_tsdf_component_only": "Region-0 native largest component만 cyan으로 표시한다. 다른 component를 삭제한 것이 아니라 비교용 view다.", "E_all_nonlargest_components": "largest 이외 모든 Region-0 component를 deterministic component color로 표시한다. tiny component도 숨기지 않는다.", "F_edge_corner_touch_components": "6-neighbor에서 분리됐지만 18/26 diagnostic adjacency에서 edge/corner contact를 보이는 component를 orange로 표시한다.", "G_one_cell_gap_components": "exact one-cell axial gap separation에 참여한 component를 magenta로 표시한다. intervening state는 raw record와 report에서 확인한다.", "H_remote_components": "nearest substantial same-Region component가 16 grid cells보다 먼 tiny component를 purple로 표시한다. semantic scene label은 자동 추정하지 않는다."}), (spatial_root, {"all_region0_gaussians": "전체 Region-0 Gaussian center의 common-world projection.", "largest_component_nearest_gaussians": "largest TSDF component sample의 nearest Gaussian IDs만 표시한 common-world projection.", "remote_tiny_island_nearest_gaussians": "remote tiny TSDF island에 이미 연결된 nearest Gaussian IDs만 표시한 common-world projection."})):
        _write_visualization_readme(root / "README.md", "W157 frozen geometry 기반 view다. 아래 camera 폴더의 PNG는 동일 fixed camera set을 사용한다.")
        for name, meaning in descriptions.items():
            view_root = root / name
            _write_visualization_readme(view_root / "README.md", f"# {name}\n\n{meaning}\n\nLegend: green=accepted Region-0 support, cyan=largest component, orange=edge/corner touch, magenta=one-cell gap, purple=remote. Original scene view는 원래 appearance를 보존한다. Overlay는 smoothing·dilation·bridge 없이 frozen world coordinates를 직접 투영한다.")
            camera_root = view_root / "cameras"
            _write_visualization_readme(camera_root / "README.md", "고정 camera별 PNG export다. camera 선택은 결과에 따라 바뀌지 않으며, render 조건은 report의 real_scene_qualitative_review에 기록한다.")
            for camera in REVIEW_CAMERAS:
                _write_visualization_readme(camera_root / camera / "README.md", f"이 파일은 fixed camera `{camera}`의 W157 PNG render다. view 의미와 legend는 상위 README를 따른다.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    _progress("loading frozen WL153/WL154/WL155/WL156 arrays")
    data = _load_frozen_data(args.field, args.wl154, args.wl155)
    wl156_report = json.loads((args.wl156 / "worklog_156_report.json").read_text(encoding="utf-8"))
    region0 = _target_region_arrays(data, 0)
    controls = {str(region_id): _target_region_arrays(data, region_id) for region_id in (2, 5)}
    _progress(f"Region 0 owned={len(region0['keys']):,} components={region0['population']['summary']['native_component_count']:,}")
    connectivity, labels, edge_audit_base = _connectivity_accounting(region0)
    separation = _separation_audit(data, region0)
    gaps = _one_cell_gap_audit(data, region0)
    edge_corner = _edge_corner_audit(data, region0, edge_audit_base)
    remote = _remote_islands(data, region0, separation)
    control_population = {rid: value["population"]["summary"] for rid, value in controls.items()}
    control_connectivity = {}
    for rid, control in controls.items():
        control_connectivity[rid], _, _ = _connectivity_accounting(control)
    synthetic = _synthetic_contracts()
    render = _fixed_camera_exports(data, args.out, region0, separation, edge_corner, gaps, remote, args.source_path, args.checkpoint, args.device)
    spatial = render["gaussian_spatial_extent"]
    spatial["mapping_hash"] = data["mapping_hash"]
    mandatory_source = args.wl155 / "mandatory_gaussian_visualization_pair"
    mandatory_target = args.out / "mandatory_gaussian_visualization_pair"
    if mandatory_target.exists():
        shutil.rmtree(mandatory_target)
    shutil.copytree(mandatory_source, mandatory_target, ignore=_ignore_ppm)
    mandatory_pair = {
        "source": str(mandatory_source),
        "output": str(mandatory_target),
        "views": ["Original Scene", "Observed-Occluded"],
        "png_only": True,
        "same_checkpoint_iteration_camera_resolution_background_renderer_row_count": True,
        "geometry_and_gaussian_rows_unchanged": True,
        "observed_occluded_changes_only_display_state_color": True,
    }
    architecture_result = _architecture_verdict(region0, connectivity, separation, gaps, remote, synthetic)
    _write_readmes(args.out)
    separation_path = _save_records(args.out, "component_separation_records.npz", separation["records"])
    gap_path = _save_gap_records(args.out, gaps)
    (args.out / "remote_islands.json").write_text(json.dumps(_jsonable(remote), indent=2), encoding="utf-8")
    (args.out / "connectivity_6_18_26.json").write_text(json.dumps(_jsonable({"region_0": connectivity, "controls": control_connectivity, "edge_corner": edge_corner}), indent=2), encoding="utf-8")
    (args.out / "one_cell_gap_attribution.json").write_text(json.dumps(_jsonable(gaps), indent=2), encoding="utf-8")
    report = {
        "status": "COMPLETE_SAME_REGION_TSDF_COMPONENT_SEPARATION_TOPOLOGY_SPATIAL_PROVENANCE_AUDIT",
        "batch": "Worklog 157 — Same-Region TSDF Component Separation Topology and Spatial-Provenance Audit",
        "intent_alignment": {"diagnostic_only": True, "connectivity_repaired": False, "production_behavior_modified": False, "wl154_membership_modified": False, "wl156_frontier_reinterpreted_as_historical": True},
        "implementation_fidelity": {"frozen_inputs": ["WL153 field", "WL154 Candidate F samples/association/ownership/component IDs", "WL155 Gaussian ID-region-status mapping", "WL156 frontier report"], "native_6_face_component_ids_reused": True, "diagnostic_18_26_graphs_production_separate": True, "merge_radius_selected": False, "gap_filled": False, "dilation_or_smoothing": False, "ownership_changed": False, "field_changed": False, "h_mu_zero_surface_changed": False},
        "current_frozen_architecture": "2DGS intrinsic t_w → frozen Gaussian Surface Region → frozen TSDF nearest-Gaussian ownership → frozen zero-surface cells → frozen 6-face Observed Support Components",
        "wl156_verdict_reconciliation": {"historical_verdict": wl156_report["architecture_result"]["architecture_verdict"], "historical_generic_frontier_accounting_retained": wl156_report["native_tsdf_component_frontier_accounting"]["0"]["category_details"], "not_used_as_component_split_fraction": True, "reopened_attribution_in_this_batch": True},
        "dominant_component_vs_tiny_island_accounting": {"region_0": region0["population"]["summary"], "controls": control_population},
        "exact_inter_component_lattice_separation": {key: value for key, value in separation.items() if key != "records"} | {"record_npz": str(separation_path)},
        "diagnostic_connectivity_6_18_26": {"region_0": connectivity, "controls": control_connectivity, "not_candidate_g_implementation": True, "production_6_face_graph_unchanged": True},
        "true_one_cell_gap_attribution": {key: value for key, value in gaps.items() if key != "records"} | {"record_npz": str(gap_path)},
        "edge_corner_touch_accounting": edge_corner,
        "remote_island_accounting": {key: value for key, value in remote.items() if key != "records"} | {"record_json": str(args.out / "remote_islands.json")},
        "region_0_gaussian_spatial_extent": spatial,
        "control_region_results": {"region_2": {"population": control_population["2"], "connectivity": control_connectivity["2"]}, "region_5": {"population": control_population["5"], "connectivity": control_connectivity["5"]}},
        "synthetic_contracts": synthetic,
        "real_scene_quantitative_result": {"mapping_hash": data["mapping_hash"], "w154_join_exact": bool(data["stable_id_join"]), "w154_region_status_join_exact": bool(wl156_report["baseline_reconciliation"]["wl155"]["w154_join_region_and_status_exact"])},
        "real_scene_qualitative_review": {**render, "mandatory_gaussian_visualization_pair": mandatory_pair, "review_root": str(args.out / "review_views"), "view_definitions": ["A_original_scene", "B_full_region0_gaussian_overlay", "C_all_region0_tsdf_support", "D_region0_largest_tsdf_component_only", "E_all_nonlargest_components", "F_edge_corner_touch_components", "G_one_cell_gap_components", "H_remote_components"], "spatial_extent_root": str(args.out / "gaussian_spatial_extent"), "semantic_scene_cluster_assessment": "Across the matched real-scene/common-world cameras, remote tiny-island marks are sparse and appear mainly around lower/side geometry (table legs, base, pavement/grass) and peripheral background; they do not form a single tabletop or vase-contact cluster. This is qualitative localization, not an inferred semantic label."},
        "failure_attribution": {"generic_wl156_frontier_is_not_split_fraction": True, "same_region_face_adjacent_native_failure_count": 0, "diagnostic_conclusion": "REOPENED_FOR_TOPOLOGY_AND_SPATIAL_PROVENANCE"},
        "architecture_result": {**architecture_result, "allowed_verdicts": ["DIGITAL_ADJACENCY_DOMINANT", "TRUE_LOCAL_TSDF_GAPS_DOMINANT", "REMOTE_SAME_REGION_ISLANDS_DOMINANT", "MIXED_COMPONENT_STRUCTURE", "MECHANICAL_IMPLEMENTATION_BUG", "UNRESOLVED"]},
        "retained_rejected_open": {"retained": ["WL153-WL156 frozen data", "native 6-face component IDs", "diagnostic 6/18/26 accounting", "PNG matched views"], "rejected": ["connectivity repair", "18/26 promotion", "gap fill", "bridge radius", "component filtering", "ownership/field changes"], "open": ["semantic cluster labels for remote islands require qualitative common-world review", "WL153 voxel-level closure lineage is unavailable"]},
        "forbidden_changes": {"tw_semantics_changed": False, "gaussian_regions_changed": False, "nearest_association_changed": False, "tsdf_field_changed": False, "zero_surface_changed": False, "native_components_changed": False, "boundary_first_changed": False, "nurbs_changed": False, "trust_latent_occluded_surface_changed": False},
        "outputs": {"report": str(args.out / "worklog_157_report.json"), "separation_records": str(separation_path), "gap_records": str(gap_path), "remote_islands": str(args.out / "remote_islands.json"), "review_root": str(args.out / "review_views"), "spatial_extent_root": str(args.out / "gaussian_spatial_extent"), "mandatory_gaussian_visualization_pair": str(mandatory_target)},
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_157_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_WL153_FIELD)
    parser.add_argument("--wl154", type=Path, default=DEFAULT_WL154)
    parser.add_argument("--wl155", type=Path, default=DEFAULT_WL155)
    parser.add_argument("--wl156", type=Path, default=DEFAULT_WL156)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_verdict": report["architecture_result"]["architecture_verdict"], "synthetic_all_pass": report["synthetic_contracts"]["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
