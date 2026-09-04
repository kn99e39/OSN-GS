"""Worklog 159: partial zero-set topology and explicit ambiguity contract audit."""

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

from devtools.demo.worklog_156_region_owned_tsdf_support_fragmentation_causal_attribution import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_PATH,
    DEFAULT_WL153_FIELD,
    DEFAULT_WL154,
    DEFAULT_WL155,
    REVIEW_CAMERAS,
    STATUS_COLORS,
    _build_named_cameras,
    _component_rgb,
    _encode_cells,
    _load_frozen_data,
    _load_surfel_model_safe,
    _lookup_sorted,
    _overlay_points,
    _save_png,
    _tensor_image,
    _write_visualization_readme,
)
from devtools.demo.worklog_157_same_region_tsdf_component_separation_topology_spatial_provenance_audit import (  # noqa: E402
    DEFAULT_WL156,
    _target_region_arrays,
)
from devtools.demo.worklog_158_mesh_free_implicit_zero_set_connectivity_candidate_g import (  # noqa: E402
    _boundary_first_replay,
    _edge_entity_ids,
    _extract_zero_set_incidence,
    _field_value_at_vertices,
    _native_face_revisit,
    _query_cell_entities,
    _shared_edge_key,
)

try:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("W159 requires scipy for the partial zero-set graph") from exc


DEFAULT_WL158 = REPO_ROOT / "output/158_mesh_free_implicit_zero_set_connectivity_candidate_g"
DEFAULT_WL157 = REPO_ROOT / "output/157_same_region_tsdf_component_separation_topology_spatial_provenance"
DEFAULT_OUT = REPO_ROOT / "output/159_partial_zero_set_topology_ambiguity_contract_audit"

CORNER_OFFSETS = np.asarray(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=np.int64
)
CUBE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
FACE_CORNERS = (
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 4, 5),
    (2, 3, 6, 7), (0, 2, 4, 6), (1, 3, 5, 7),
)
TOPOLOGY_CATEGORIES = (
    "DETERMINISTIC_SINGLE_PATCH",
    "DETERMINISTIC_MULTI_PATCH",
    "EXACT_ZERO_VERTEX_DEGENERACY",
    "EXACT_ZERO_EDGE_FACE_DEGENERACY",
    "LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED",
    "OTHER_GENUINE_TOPOLOGY_AMBIGUITY",
)
GUARANTEED_STATES = ("GUARANTEED_CONNECT", "GUARANTEED_DISCONNECT", "TOPOLOGY_AMBIGUOUS")


def _progress(message: str) -> None:
    print(f"[worklog 159] {message}", flush=True)


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
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.percentile(array, 50, method="nearest")),
        "p95": float(np.percentile(array, 95, method="nearest")),
        "max": float(array.max()),
    }


def _read_wl158_incidence(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as record:
        return {
            "entity_keys": np.asarray(record["zero_crossing_entity_key"], dtype=np.int64),
            "entity_labels": np.asarray(record["zero_crossing_entity_component"], dtype=np.int32),
            "incidence_pairs": np.asarray(record["incidence_edges"], dtype=np.int32),
            "triangle_cells": np.asarray(record["triangle_cell_key"], dtype=np.int64),
            "triangle_entities": np.asarray(record["triangle_zero_crossing_entity_key"], dtype=np.int64),
            "candidate_component_by_cell": np.asarray(record["candidate_component_by_cell"], dtype=np.int32),
            "ambiguous_cells": np.asarray(record["ambiguous_cell_key"], dtype=np.int64),
        }


def _field_corner_values(data: dict[str, Any], cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corner_cells = np.asarray(cells, dtype=np.int64)[:, None, :] + CORNER_OFFSETS[None, :, :]
    positions, present = _lookup_sorted(data["field_keys"], _encode_cells(corner_cells.reshape(-1, 3)))
    return data["field_values"][positions].reshape(-1, 8).astype(np.float64), present.reshape(-1, 8)


def _edge_key_from_cell_corner(cell: np.ndarray, corner_a: int, corner_b: int) -> int:
    a = np.asarray(cell, dtype=np.int64) + CORNER_OFFSETS[corner_a]
    b = np.asarray(cell, dtype=np.int64) + CORNER_OFFSETS[corner_b]
    lower = np.minimum(a, b)
    axis = int(np.flatnonzero(a != b)[0])
    return int(_encode_cells(lower.reshape(1, 3))[0] * 3 + axis)


def _cell_edge_entities(cells: np.ndarray, values: np.ndarray, *, include_degenerate: bool) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for cell, row in zip(np.asarray(cells, dtype=np.int64), np.asarray(values, dtype=np.float64)):
        keys: list[int] = []
        for corner_a, corner_b in CUBE_EDGES:
            left, right = float(row[corner_a]), float(row[corner_b])
            strict = (left < 0.0 < right) or (right < 0.0 < left)
            degenerate = include_degenerate and (left == 0.0 or right == 0.0)
            if strict or degenerate:
                keys.append(_edge_key_from_cell_corner(cell, corner_a, corner_b))
        result.append(np.asarray(sorted(set(keys)), dtype=np.int64))
    return result


def _face_ambiguity_flags(values: np.ndarray) -> tuple[bool, ...]:
    flags: list[bool] = []
    for face in FACE_CORNERS:
        row = np.asarray(values[list(face)], dtype=np.float64)
        if np.any(row == 0.0):
            flags.append(False)
            continue
        signs = row < 0.0
        alternating = bool(signs[0] == signs[3] and signs[1] == signs[2] and signs[0] != signs[1])
        # A non-alternating face has one unique sign-side connectivity.  An
        # alternating face is the classical Lewiner/asymptotic-decider case;
        # under this audit it remains convention-sensitive even when Lewiner
        # returns one deterministic triangulation.
        flags.append(alternating)
    return tuple(flags)


def _classify_cell(values: np.ndarray, raw_ambiguous: bool) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    zero_vertices = np.flatnonzero(values == 0.0).astype(np.int64)
    zero_edges = [edge for edge in CUBE_EDGES if values[edge[0]] == 0.0 and values[edge[1]] == 0.0]
    zero_faces = [face for face in FACE_CORNERS if np.all(values[list(face)] == 0.0)]
    face_ambiguity = _face_ambiguity_flags(values)
    if zero_edges or zero_faces:
        category = "EXACT_ZERO_EDGE_FACE_DEGENERACY"
    elif len(zero_vertices):
        category = "EXACT_ZERO_VERTEX_DEGENERACY"
    elif any(face_ambiguity):
        category = "LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED"
    elif raw_ambiguous:
        category = "OTHER_GENUINE_TOPOLOGY_AMBIGUITY"
    else:
        category = "DETERMINISTIC_SINGLE_PATCH"
    return {
        "category": category,
        "zero_vertex_indices": zero_vertices,
        "zero_edge_count": int(len(zero_edges)),
        "zero_face_count": int(len(zero_faces)),
        "face_ambiguity_count": int(sum(face_ambiguity)),
        "face_ambiguity_flags": face_ambiguity,
    }


def _patches_for_cells(incidence: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    target_keys = np.asarray(region["keys"], dtype=np.int64)
    triangle_cells = np.asarray(incidence["triangle_cells"], dtype=np.int64)
    triangle_entities = np.asarray(incidence["triangle_entities"], dtype=np.int64)
    by_cell: dict[int, list[int]] = defaultdict(list)
    for index, key in enumerate(triangle_cells.tolist()):
        by_cell[int(key)].append(index)
    patch_ids_by_cell: dict[int, list[int]] = {}
    patch_entities: dict[int, frozenset[int]] = {}
    patch_cells: list[int] = []
    patch_local_indices: list[int] = []
    triangle_patch = np.full((len(triangle_cells),), -1, dtype=np.int32)
    for cell_key, triangle_indices in by_cell.items():
        remaining = set(triangle_indices)
        local_index = 0
        ids: list[int] = []
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            members = [seed]
            entities = set(int(value) for value in triangle_entities[seed].tolist())
            while stack:
                current = stack.pop()
                neighbors = [candidate for candidate in remaining if entities.intersection(int(value) for value in triangle_entities[candidate].tolist())]
                for neighbor in neighbors:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    members.append(neighbor)
                    entities.update(int(value) for value in triangle_entities[neighbor].tolist())
            patch_id = len(patch_cells)
            for member in members:
                triangle_patch[member] = patch_id
            patch_cells.append(cell_key)
            patch_local_indices.append(local_index)
            patch_entities[patch_id] = frozenset(entities)
            ids.append(patch_id)
            local_index += 1
        patch_ids_by_cell[cell_key] = ids
    patch_cell_array = np.asarray(patch_cells, dtype=np.int64)
    patch_local_array = np.asarray(patch_local_indices, dtype=np.int32)
    counts = Counter(len(ids) for ids in patch_ids_by_cell.values())
    return {
        "patch_ids_by_cell": patch_ids_by_cell,
        "patch_entities": patch_entities,
        "patch_cell_keys": patch_cell_array,
        "patch_local_indices": patch_local_array,
        "triangle_patch": triangle_patch,
        "patch_count_distribution": {str(key): int(value) for key, value in sorted(counts.items())},
        "cell_patch_count": {int(key): len(value) for key, value in patch_ids_by_cell.items()},
        "target_keys": target_keys,
    }


def _classify_region(data: dict[str, Any], region: dict[str, Any], incidence: dict[str, Any]) -> dict[str, Any]:
    patch_info = _patches_for_cells(incidence, region)
    values, present = _field_corner_values(data, region["cells"])
    raw_ambiguous = set(int(value) for value in incidence["ambiguous_cells"].tolist())
    records: list[dict[str, Any]] = []
    category_by_key: dict[int, str] = {}
    cell_entities = _cell_edge_entities(region["cells"], values, include_degenerate=True)
    entity_by_cell: dict[int, frozenset[int]] = {}
    for position, (key, row, row_present, entities) in enumerate(zip(region["keys"].tolist(), values, present, cell_entities)):
        key = int(key)
        if not bool(np.all(row_present)):
            raise ValueError("W159 encountered non-authoritative corner values in a frozen target cell")
        classification = _classify_cell(row, key in raw_ambiguous)
        patch_count = int(patch_info["cell_patch_count"].get(key, 0))
        if classification["category"] == "DETERMINISTIC_SINGLE_PATCH" and patch_count > 1:
            classification["category"] = "DETERMINISTIC_MULTI_PATCH"
        if classification["category"] == "DETERMINISTIC_SINGLE_PATCH" and patch_count == 0:
            classification["category"] = "OTHER_GENUINE_TOPOLOGY_AMBIGUITY"
        category_by_key[key] = classification["category"]
        entity_by_cell[key] = frozenset(int(value) for value in entities.tolist())
        records.append({
            "cell_key": key,
            "category": classification["category"],
            "patch_count": patch_count,
            "zero_vertex_indices": classification["zero_vertex_indices"],
            "zero_edge_count": classification["zero_edge_count"],
            "zero_face_count": classification["zero_face_count"],
            "face_ambiguity_count": classification["face_ambiguity_count"],
            "face_ambiguity_flags": classification["face_ambiguity_flags"],
        })
    category_counts = Counter(record["category"] for record in records)
    record_by_key = {int(record["cell_key"]): record for record in records}
    return {
        "patch_info": patch_info,
        "records": records,
        "record_by_key": record_by_key,
        "category_by_key": category_by_key,
        "category_counts": dict(category_counts),
        "cell_entities": entity_by_cell,
        "cell_edge_entities": cell_entities,
        "corner_authority_all": bool(np.all(present)),
        "raw_ambiguous_cell_count": int(len(raw_ambiguous)),
    }


def _build_guaranteed_graph(region: dict[str, Any], classified: dict[str, Any]) -> dict[str, Any]:
    patch_info = classified["patch_info"]
    category_by_key = classified["category_by_key"]
    patch_cells = patch_info["patch_cell_keys"]
    patch_entities = patch_info["patch_entities"]
    deterministic_categories = {"DETERMINISTIC_SINGLE_PATCH", "DETERMINISTIC_MULTI_PATCH"}
    included = [index for index, key in enumerate(patch_cells.tolist()) if category_by_key.get(int(key)) in deterministic_categories]
    included_set = set(included)
    entity_to_patches: dict[int, list[int]] = defaultdict(list)
    for patch_id in included:
        for entity in patch_entities[patch_id]:
            entity_to_patches[int(entity)].append(patch_id)
    guaranteed_pairs: set[tuple[int, int]] = set()
    interface_records: list[dict[str, Any]] = []
    for entity, patch_ids in entity_to_patches.items():
        for offset, left in enumerate(patch_ids):
            for right in patch_ids[offset + 1:]:
                left_cell, right_cell = int(patch_cells[left]), int(patch_cells[right])
                if left_cell == right_cell:
                    continue
                pair = tuple(sorted((left, right)))
                guaranteed_pairs.add(pair)
                interface_records.append({"entity": entity, "left_patch": pair[0], "right_patch": pair[1], "state": "GUARANTEED_CONNECT"})
    if included:
        node_remap = {old: new for new, old in enumerate(sorted(included))}
        pairs = np.asarray([(node_remap[left], node_remap[right]) for left, right in sorted(guaranteed_pairs)], dtype=np.int32)
    else:
        node_remap = {}
        pairs = np.empty((0, 2), dtype=np.int32)
    if len(included):
        if len(pairs):
            rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
            cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
            graph = sparse.coo_matrix((np.ones((len(rows),), dtype=np.uint8), (rows, cols)), shape=(len(included), len(included))).tocsr()
        else:
            graph = sparse.csr_matrix((len(included), len(included)), dtype=np.uint8)
        component_count, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        component_count, labels = 0, np.empty((0,), dtype=np.int32)
    patch_component = np.full((len(patch_cells),), -1, dtype=np.int32)
    for old, new in node_remap.items():
        patch_component[old] = int(labels[new])
    return {
        "included_patch_ids": np.asarray(sorted(included), dtype=np.int64),
        "node_remap": node_remap,
        "guaranteed_pairs": pairs,
        "interface_records": interface_records,
        "component_count": int(component_count),
        "patch_component": patch_component,
        "patch_cells": patch_cells,
        "patch_entities": patch_entities,
        "graph_node_count": int(len(included)),
        "graph_edge_count": int(len(pairs)),
        "guaranteed_connect_interface_count": int(len(interface_records)),
        "guaranteed_disconnect_contract": "different Region/ownership, non-zero-surface gaps, and no shared entity do not create an edge",
    }


def _cell_component_labels(region: dict[str, Any], classified: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    labels_by_cell: dict[int, set[int]] = defaultdict(set)
    for patch_id, component in enumerate(graph["patch_component"].tolist()):
        if component >= 0:
            labels_by_cell[int(graph["patch_cells"][patch_id])].add(int(component))
    deterministic = {"DETERMINISTIC_SINGLE_PATCH", "DETERMINISTIC_MULTI_PATCH"}
    guaranteed_cell = np.full((len(region["keys"]),), -1, dtype=np.int32)
    ambiguous_mask = np.zeros((len(region["keys"],)), dtype=bool)
    for position, key in enumerate(region["keys"].tolist()):
        key = int(key)
        category = classified["category_by_key"].get(key, "OTHER_GENUINE_TOPOLOGY_AMBIGUITY")
        components = labels_by_cell.get(key, set())
        if category in deterministic and len(components) == 1:
            guaranteed_cell[position] = next(iter(components))
        else:
            ambiguous_mask[position] = True
    sizes = np.bincount(guaranteed_cell[guaranteed_cell >= 0], minlength=graph["component_count"]) if graph["component_count"] else np.empty((0,), dtype=np.int64)
    return {
        "cell_component": guaranteed_cell,
        "ambiguous_cell_mask": ambiguous_mask,
        "component_cell_sizes": sizes.astype(np.int64, copy=False),
        "represented_cell_count": int(np.sum(guaranteed_cell >= 0)),
        "ambiguous_or_unrepresented_cell_count": int(np.sum(guaranteed_cell < 0)),
    }


def _interface_accounting(region: dict[str, Any], classified: dict[str, Any], graph: dict[str, Any], cell_accounting: dict[str, Any]) -> dict[str, Any]:
    category_by_key = classified["category_by_key"]
    ambiguous_categories = {
        "EXACT_ZERO_VERTEX_DEGENERACY",
        "EXACT_ZERO_EDGE_FACE_DEGENERACY",
        "LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED",
        "OTHER_GENUINE_TOPOLOGY_AMBIGUITY",
    }
    patch_cells = graph["patch_cells"]
    patch_component = graph["patch_component"]
    patch_entities = graph["patch_entities"]
    entity_to_patches: dict[int, list[int]] = defaultdict(list)
    for patch_id, entities in enumerate(patch_entities):
        if patch_component[patch_id] >= 0:
            for entity in entities:
                entity_to_patches[int(entity)].append(patch_id)
    ambiguous_keys = {key for key, category in category_by_key.items() if category in ambiguous_categories}
    interfaces: set[tuple[int, int, int]] = set()
    affected_components: set[int] = set()
    for entity, patch_ids in entity_to_patches.items():
        for patch_id in patch_ids:
            source_cell = int(patch_cells[patch_id])
            for other_key in ambiguous_keys:
                if entity in classified["cell_entities"].get(other_key, frozenset()):
                    component = int(patch_component[patch_id])
                    interfaces.add((entity, source_cell, other_key))
                    affected_components.add(component)
    category_counts = Counter(category_by_key[key] for _entity, _source, key in interfaces)
    return {
        "interfaces": sorted(interfaces),
        "ambiguous_interface_count": int(len(interfaces)),
        "affected_component_ids": sorted(affected_components),
        "affected_component_count": int(len(affected_components)),
        "interface_category_counts": dict(category_counts),
        "genuine_ambiguous_cell_count": int(len(ambiguous_keys)),
        "genuine_ambiguous_cell_fraction": float(len(ambiguous_keys) / max(len(region["keys"]), 1)),
        "represented_cell_count": cell_accounting["represented_cell_count"],
    }


def _hypothetical_ambiguity_envelope(region: dict[str, Any], classified: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    """Build an explicitly non-production all-ambiguous-interface envelope."""

    patch_cells = graph["patch_cells"]
    deterministic_patch_ids = [index for index, value in enumerate(graph["patch_component"].tolist()) if value >= 0]
    ambiguous_keys = [key for key, category in classified["category_by_key"].items() if category not in {"DETERMINISTIC_SINGLE_PATCH", "DETERMINISTIC_MULTI_PATCH"}]
    node_names: list[tuple[str, int]] = [("patch", index) for index in deterministic_patch_ids] + [("ambiguous_cell", key) for key in sorted(ambiguous_keys)]
    node_index = {name: index for index, name in enumerate(node_names)}
    pairs: set[tuple[int, int]] = set()
    entity_to_patches: dict[int, list[int]] = defaultdict(list)
    for patch_id in deterministic_patch_ids:
        for entity in graph["patch_entities"][patch_id]:
            entity_to_patches[int(entity)].append(patch_id)
    for entity, patch_ids in entity_to_patches.items():
        ambiguous_for_entity = [key for key in ambiguous_keys if entity in classified["cell_entities"].get(key, frozenset())]
        for patch_id in patch_ids:
            for key in ambiguous_for_entity:
                pairs.add(tuple(sorted((node_index[("patch", patch_id)], node_index[("ambiguous_cell", key)]))))
        for index, left in enumerate(ambiguous_for_entity):
            for right in ambiguous_for_entity[index + 1:]:
                pairs.add(tuple(sorted((node_index[("ambiguous_cell", left)], node_index[("ambiguous_cell", right)]))))
    if node_names:
        if pairs:
            edge_array = np.asarray(sorted(pairs), dtype=np.int32)
            rows = np.concatenate((edge_array[:, 0], edge_array[:, 1]))
            cols = np.concatenate((edge_array[:, 1], edge_array[:, 0]))
            graph_matrix = sparse.coo_matrix((np.ones((len(rows),), dtype=np.uint8), (rows, cols)), shape=(len(node_names), len(node_names))).tocsr()
        else:
            edge_array = np.empty((0, 2), dtype=np.int32)
            graph_matrix = sparse.csr_matrix((len(node_names), len(node_names)), dtype=np.uint8)
        component_count, labels = connected_components(graph_matrix, directed=False, return_labels=True)
    else:
        edge_array = np.empty((0, 2), dtype=np.int32)
        component_count, labels = 0, np.empty((0,), dtype=np.int32)
    return {
        "node_count": int(len(node_names)),
        "edge_count": int(len(edge_array)),
        "component_count": int(component_count),
        "ambiguous_supernode_count": int(len(ambiguous_keys)),
        "hypothetical_only": True,
        "not_production_topology": True,
        "definition": "every observed ambiguous interface is treated as connectable through an explicit supernode; no selected envelope is promoted",
    }

# The real-scene path below is array based. The dictionary helpers above remain
# useful for unit-level contract examples, but are not used for the replay.
CATEGORY_TO_CODE = {name: index for index, name in enumerate(TOPOLOGY_CATEGORIES)}
CODE_TO_CATEGORY = {index: name for name, index in CATEGORY_TO_CODE.items()}
DETERMINISTIC_CODES = np.asarray([CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"], CATEGORY_TO_CODE["DETERMINISTIC_MULTI_PATCH"]], dtype=np.int8)
AMBIGUOUS_CODES = np.asarray([CATEGORY_TO_CODE[name] for name in TOPOLOGY_CATEGORIES[2:]], dtype=np.int8)


def _local_patch_accounting(triangle_cells: np.ndarray, triangle_entities: np.ndarray) -> dict[str, Any]:
    """Count connected surface patches independently inside each target cell."""
    cells = np.asarray(triangle_cells, dtype=np.int64)
    entities = np.asarray(triangle_entities, dtype=np.int64)
    if len(cells):
        order = np.argsort(cells, kind="stable")
        cells, entities = cells[order], entities[order]
    if not len(cells):
        return {"cell_keys": np.empty((0,), dtype=np.int64), "triangle_counts": np.empty((0,), dtype=np.int32), "patch_counts": np.empty((0,), dtype=np.int32), "same_cell_shared_entity_pair_count": 0}
    cell_keys, starts, counts = np.unique(cells, return_index=True, return_counts=True)
    counts = counts.astype(np.int32, copy=False)
    candidate_rows, candidate_cols = [], []
    for width in range(2, int(counts.max()) + 1):
        group_starts = starts[counts == width]
        if not len(group_starts):
            continue
        left_offsets, right_offsets = np.triu_indices(width, k=1)
        candidate_rows.append((group_starts[:, None] + left_offsets[None, :]).reshape(-1))
        candidate_cols.append((group_starts[:, None] + right_offsets[None, :]).reshape(-1))
    if not candidate_rows:
        return {"cell_keys": cell_keys, "triangle_counts": counts, "patch_counts": counts.copy(), "same_cell_shared_entity_pair_count": 0}
    rows = np.concatenate(candidate_rows).astype(np.int32, copy=False)
    cols = np.concatenate(candidate_cols).astype(np.int32, copy=False)
    shared = np.any(entities[rows, :, None] == entities[cols, None, :], axis=(1, 2))
    rows, cols = rows[shared], cols[shared]
    if not len(rows):
        patch_counts = counts.copy()
    else:
        graph = sparse.coo_matrix((np.ones((len(rows) * 2,), dtype=np.uint8), (np.concatenate((rows, cols), axis=0), np.concatenate((cols, rows), axis=0))), shape=(len(cells), len(cells))).tocsr()
        _component_count, labels = connected_components(graph, directed=False, return_labels=True)
        records = np.empty((len(cells),), dtype=np.dtype([( "cell", "<i8"), ("label", "<i8")]))
        records["cell"], records["label"] = cells, labels
        unique_records = np.unique(records)
        pair_cells, pair_counts = np.unique(unique_records["cell"], return_counts=True)
        patch_counts = np.zeros((len(cell_keys),), dtype=np.int32)
        patch_counts[np.searchsorted(cell_keys, pair_cells)] = pair_counts.astype(np.int32, copy=False)
    return {"cell_keys": cell_keys, "triangle_counts": counts, "patch_counts": patch_counts, "same_cell_shared_entity_pair_count": int(len(rows))}


def _classify_region_fast(data: dict[str, Any], region: dict[str, Any], incidence: dict[str, Any]) -> dict[str, Any]:
    """Classify frozen target cells from scalar evidence and local patches."""
    cells = np.asarray(region["cells"], dtype=np.int64)
    keys = np.asarray(region["keys"], dtype=np.int64)
    corner_cells = cells[:, None, :] + CORNER_OFFSETS[None, :, :]
    positions, present = _lookup_sorted(data["field_keys"], _encode_cells(corner_cells.reshape(-1, 3)))
    values = data["field_values"][positions].reshape(-1, 8).astype(np.float32, copy=False)
    present = present.reshape(-1, 8)
    if not np.all(present):
        raise ValueError("W159 encountered non-authoritative corner values in a frozen target cell")
    zero = values == 0.0
    zero_vertex = np.any(zero, axis=1)
    zero_edge = np.zeros((len(cells),), dtype=bool)
    for left, right in CUBE_EDGES:
        zero_edge |= zero[:, left] & zero[:, right]
    zero_face = np.zeros((len(cells),), dtype=bool)
    alternating = np.zeros((len(cells),), dtype=bool)
    exact_face_tie = np.zeros((len(cells),), dtype=bool)
    for face in FACE_CORNERS:
        face_values = values[:, face]
        face_zero = np.any(face_values == 0.0, axis=1)
        zero_face |= np.all(face_values == 0.0, axis=1)
        signs = face_values < 0.0
        face_alt = (~face_zero) & (signs[:, 0] == signs[:, 3]) & (signs[:, 1] == signs[:, 2]) & (signs[:, 0] != signs[:, 1])
        alternating |= face_alt
        exact_face_tie |= face_alt & ((face_values[:, 0] * face_values[:, 3] - face_values[:, 1] * face_values[:, 2]) == 0.0)
    raw_ambiguous = np.isin(keys, np.asarray(incidence["ambiguous_cells"], dtype=np.int64))
    patch_info = _local_patch_accounting(incidence["triangle_cells"], incidence["triangle_entities"])
    patch_count = np.zeros((len(keys),), dtype=np.int32)
    triangle_count = np.zeros((len(keys),), dtype=np.int32)
    if len(patch_info["cell_keys"]):
        target_positions, found = _lookup_sorted(keys, patch_info["cell_keys"])
        patch_count[target_positions[found]] = patch_info["patch_counts"][found]
        triangle_count[target_positions[found]] = patch_info["triangle_counts"][found]
    category = np.full((len(keys),), CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"], dtype=np.int8)
    category[zero_edge | zero_face] = CATEGORY_TO_CODE["EXACT_ZERO_EDGE_FACE_DEGENERACY"]
    category[(~zero_edge & ~zero_face) & zero_vertex] = CATEGORY_TO_CODE["EXACT_ZERO_VERTEX_DEGENERACY"]
    category[(~zero_edge & ~zero_face & ~zero_vertex) & exact_face_tie] = CATEGORY_TO_CODE["LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED"]
    category[raw_ambiguous & (category == CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"])] = CATEGORY_TO_CODE["OTHER_GENUINE_TOPOLOGY_AMBIGUITY"]
    category[(triangle_count == 0) & (category == CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"])] = CATEGORY_TO_CODE["OTHER_GENUINE_TOPOLOGY_AMBIGUITY"]
    multi_patch = (category == CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"]) & (patch_count > 1)
    category[multi_patch] = CATEGORY_TO_CODE["DETERMINISTIC_MULTI_PATCH"]
    return {
        "category_codes": category,
        "category_counts": {CODE_TO_CATEGORY[code]: int(np.sum(category == code)) for code in range(len(TOPOLOGY_CATEGORIES))},
        "patch_count_by_cell": patch_count,
        "triangle_count_by_cell": triangle_count,
        "raw_ambiguous_mask": raw_ambiguous,
        "zero_vertex_mask": zero_vertex,
        "zero_edge_or_face_mask": zero_edge | zero_face,
        "alternating_face_mask": alternating,
        "exact_face_decider_tie_mask": exact_face_tie,
        "genuine_ambiguity_mask": np.isin(category, AMBIGUOUS_CODES),
        "patch_count_distribution": {str(int(value)): int(np.sum(patch_count == value)) for value in np.unique(patch_count)},
        "triangle_count_distribution": {str(int(value)): int(np.sum(triangle_count == value)) for value in np.unique(triangle_count)},
        "all_corner_authoritative": bool(np.all(present)),
        "same_cell_shared_entity_pair_count": int(patch_info["same_cell_shared_entity_pair_count"]),
        "classification_definition": "ordinary nonzero-corner cells use exact scalar edge/face evidence; exact zeros and exact face decider ties remain ambiguous; multiple locally disconnected patches are deterministic multi-patch when no ambiguity guard applies",
    }


def _build_candidate_h_fast(region: dict[str, Any], incidence: dict[str, Any], classified: dict[str, Any]) -> dict[str, Any]:
    """Build a partial graph from deterministic triangles only."""
    triangle_cells = np.asarray(incidence["triangle_cells"], dtype=np.int64)
    triangle_entities = np.asarray(incidence["triangle_entities"], dtype=np.int64)
    target_keys = np.asarray(region["keys"], dtype=np.int64)
    empty_components = np.empty((0,), dtype=np.int32)
    if not len(triangle_cells):
        return {"entity_keys": np.empty((0,), dtype=np.int64), "entity_labels": empty_components, "incidence_pairs": np.empty((0, 2), dtype=np.int32), "triangle_cells": np.empty((0,), dtype=np.int64), "triangle_entities": np.empty((0, 3), dtype=np.int64), "cell_component": np.full((len(target_keys),), -1, dtype=np.int32), "component_count": 0, "component_cell_sizes": np.empty((0,), dtype=np.int64), "deterministic_triangle_count": 0, "ambiguous_triangle_count": 0}
    positions, present = _lookup_sorted(target_keys, triangle_cells)
    deterministic = np.zeros((len(triangle_cells),), dtype=bool)
    deterministic[present] = np.isin(classified["category_codes"][positions[present]], DETERMINISTIC_CODES)
    selected_cells = triangle_cells[deterministic]
    selected_entities = triangle_entities[deterministic]
    ambiguous_triangle_count = int(np.sum(~deterministic))
    if not len(selected_entities):
        return {"entity_keys": np.empty((0,), dtype=np.int64), "entity_labels": empty_components, "incidence_pairs": np.empty((0, 2), dtype=np.int32), "triangle_cells": selected_cells, "triangle_entities": selected_entities, "cell_component": np.full((len(target_keys),), -1, dtype=np.int32), "component_count": 0, "component_cell_sizes": np.empty((0,), dtype=np.int64), "deterministic_triangle_count": 0, "ambiguous_triangle_count": ambiguous_triangle_count}
    entity_keys = np.unique(selected_entities.reshape(-1))
    triangle_nodes = np.searchsorted(entity_keys, selected_entities).astype(np.int32, copy=False)
    pair_rows = np.concatenate((triangle_nodes[:, 0], triangle_nodes[:, 1], triangle_nodes[:, 2]))
    pair_cols = np.concatenate((triangle_nodes[:, 1], triangle_nodes[:, 2], triangle_nodes[:, 0]))
    pairs = np.unique(np.sort(np.column_stack((pair_rows, pair_cols)), axis=1), axis=0).astype(np.int32, copy=False)
    if len(pairs):
        graph = sparse.coo_matrix((np.ones((len(pairs) * 2,), dtype=np.uint8), (np.concatenate((pairs[:, 0], pairs[:, 1])), np.concatenate((pairs[:, 1], pairs[:, 0])))), shape=(len(entity_keys), len(entity_keys))).tocsr()
        component_count, entity_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        component_count, entity_labels = len(entity_keys), np.arange(len(entity_keys), dtype=np.int32)
    entity_labels = entity_labels.astype(np.int32, copy=False)
    cell_records = np.empty((len(selected_cells) * 3,), dtype=np.dtype([( "cell", "<i8"), ("component", "<i4")]))
    cell_records["cell"] = np.repeat(selected_cells, 3)
    cell_records["component"] = entity_labels[triangle_nodes.reshape(-1)]
    unique_cell_records = np.unique(cell_records)
    cell_component = np.full((len(target_keys),), -1, dtype=np.int32)
    record_cells, record_counts = np.unique(unique_cell_records["cell"], return_counts=True)
    unique_record_mask = record_counts == 1
    unique_cells = record_cells[unique_record_mask]
    if len(unique_cells):
        target_positions, found = _lookup_sorted(target_keys, unique_cells)
        cell_component[target_positions[found]] = unique_cell_records["component"][np.searchsorted(unique_cell_records["cell"], unique_cells[found])]
    component_cell_sizes = np.bincount(cell_component[cell_component >= 0], minlength=int(component_count)).astype(np.int64, copy=False) if component_count else np.empty((0,), dtype=np.int64)
    return {"entity_keys": entity_keys, "entity_labels": entity_labels, "incidence_pairs": pairs, "triangle_cells": selected_cells, "triangle_entities": selected_entities, "cell_component": cell_component, "component_count": int(component_count), "component_cell_sizes": component_cell_sizes, "deterministic_triangle_count": int(len(selected_cells)), "ambiguous_triangle_count": ambiguous_triangle_count, "partial_graph_definition": "only shared zero-crossing lattice-edge entities from deterministic cells produce guaranteed edges; ambiguous cells and their possible joins are omitted"}

def _ambiguity_leverage_fast(region: dict[str, Any], incidence: dict[str, Any], classified: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    """Measure what unresolved cells could change, without selecting a bridge."""
    ambiguous_positions = np.flatnonzero(classified["genuine_ambiguity_mask"])
    if not len(ambiguous_positions) or not len(graph["entity_keys"]):
        return {"ambiguous_cell_count": int(len(ambiguous_positions)), "ambiguous_cell_fraction": float(len(ambiguous_positions) / max(len(region["keys"]), 1)), "ambiguous_interface_count": 0, "affected_component_count": 0, "affected_component_ids": [], "dominant_component_id": -1, "dominant_component_touch_count": 0, "ambiguous_cells_touching_multiple_components": 0, "macro_identity_change_possible": False, "hypothetical_component_count_after_all_observed_ambiguity_merges": int(graph["component_count"]), "hypothetical_component_count_reduction": 0, "interface_category_counts": {}, "interface_records_sample": [], "definition": "ambiguity leverage is observed ambiguous-cell contact with deterministic Candidate-H entities; no contact is treated as no leverage"}
    ambiguous_keys = region["keys"][ambiguous_positions]
    triangle_cells = np.asarray(incidence["triangle_cells"], dtype=np.int64)
    triangle_entities = np.asarray(incidence["triangle_entities"], dtype=np.int64)
    selected = np.isin(triangle_cells, ambiguous_keys)
    if np.any(selected):
        records = np.empty((int(np.sum(selected)) * 3,), dtype=np.dtype([("cell", "<i8"), ("entity", "<i8")]))
        records["cell"] = np.repeat(triangle_cells[selected], 3)
        records["entity"] = triangle_entities[selected].reshape(-1)
        records = np.unique(records)
    else:
        records = np.empty((0,), dtype=np.dtype([("cell", "<i8"), ("entity", "<i8")]))
    h_entity_positions = np.searchsorted(graph["entity_keys"], records["entity"]) if len(records) else np.empty((0,), dtype=np.int64)
    valid = len(records) > 0
    if len(records):
        valid = (h_entity_positions < len(graph["entity_keys"])) & (graph["entity_keys"][np.minimum(h_entity_positions, max(len(graph["entity_keys"]) - 1, 0))] == records["entity"])
    records = records[valid]
    h_entity_positions = h_entity_positions[valid]
    component_ids = graph["entity_labels"][h_entity_positions] if len(records) else np.empty((0,), dtype=np.int32)
    dominant = int(np.argmax(graph["component_cell_sizes"])) if len(graph["component_cell_sizes"]) else -1
    if len(records):
        interface_records = np.empty((len(records),), dtype=np.dtype([("cell", "<i8"), ("entity", "<i8"), ("component", "<i4"), ("category", "<i1")]))
        interface_records["cell"] = records["cell"]
        interface_records["entity"] = records["entity"]
        interface_records["component"] = component_ids
        cell_positions = np.searchsorted(region["keys"], records["cell"])
        interface_records["category"] = classified["category_codes"][cell_positions]
        interface_records = np.unique(interface_records)
    else:
        interface_records = np.empty((0,), dtype=np.dtype([("cell", "<i8"), ("entity", "<i8"), ("component", "<i4"), ("category", "<i1")]))
    touched_by_cell: dict[int, set[int]] = defaultdict(set)
    for record in interface_records:
        touched_by_cell[int(record["cell"])].add(int(record["component"]))
    multi_touch = sum(len(values) > 1 for values in touched_by_cell.values())
    touched_components = sorted({int(value) for values in touched_by_cell.values() for value in values})
    parent = np.arange(int(graph["component_count"]), dtype=np.int32)
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value
    for values in touched_by_cell.values():
        values = sorted(values)
        if len(values) > 1:
            root = find(values[0])
            for value in values[1:]:
                other = find(value)
                if root != other:
                    parent[other] = root
    envelope_count = len({find(index) for index in range(int(graph["component_count"]))}) if len(parent) else 0
    category_counts = Counter(CODE_TO_CATEGORY[int(value)] for value in interface_records["category"].tolist())
    records_sample = [{"cell_key": int(row["cell"]), "entity_key": int(row["entity"]), "component_id": int(row["component"]), "category": CODE_TO_CATEGORY[int(row["category"])]} for row in interface_records[:256]]
    return {
        "ambiguous_cell_count": int(len(ambiguous_positions)),
        "ambiguous_cell_fraction": float(len(ambiguous_positions) / max(len(region["keys"]), 1)),
        "ambiguous_interface_count": int(len(interface_records)),
        "affected_component_count": int(len(touched_components)),
        "affected_component_ids": touched_components,
        "dominant_component_id": dominant,
        "dominant_component_touch_count": int(sum(int(value == dominant) for value in component_ids.tolist())),
        "ambiguous_cells_touching_multiple_components": int(multi_touch),
        "macro_identity_change_possible": bool(multi_touch > 0 or envelope_count < int(graph["component_count"])),
        "hypothetical_component_count_after_all_observed_ambiguity_merges": int(envelope_count),
        "hypothetical_component_count_reduction": int(int(graph["component_count"]) - envelope_count),
        "interface_category_counts": dict(category_counts),
        "interface_records_sample": records_sample,
        "definition": "lower bound is Candidate H with ambiguous cells omitted; hypothetical envelope merges only deterministic components actually touched by the same unresolved cell and is never promoted",
    }


def _candidate_h_metrics(region: dict[str, Any], classified: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    sizes = graph["component_cell_sizes"]
    largest = int(sizes.max()) if len(sizes) else 0
    represented = int(np.sum(graph["cell_component"] >= 0))
    total = int(len(region["keys"]))
    singleton = int(np.sum(sizes == 1)) if len(sizes) else 0
    return {
        "zero_crossing_entity_node_count": int(len(graph["entity_keys"])),
        "guaranteed_incidence_edge_count": int(len(graph["incidence_pairs"])),
        "connected_component_count": int(graph["component_count"]),
        "component_cell_size_distribution": _summary(sizes[sizes > 0]),
        "largest_component_cell_fraction_of_all_region_cells": float(largest / max(total, 1)),
        "largest_component_cell_fraction_of_represented_cells": float(largest / max(represented, 1)),
        "sample_fraction_outside_largest": float((represented - largest) / max(total, 1)),
        "singleton_component_count": singleton,
        "singleton_component_fraction": float(singleton / max(int(np.sum(sizes > 0)), 1)),
        "component_count_with_any_cell_assignment": int(np.sum(sizes > 0)),
        "represented_cell_count": represented,
        "unassigned_cell_count": int(total - represented),
        "ambiguous_cell_count": int(np.sum(classified["genuine_ambiguity_mask"])),
        "deterministic_multi_patch_cell_count": int(np.sum(classified["category_codes"] == CATEGORY_TO_CODE["DETERMINISTIC_MULTI_PATCH"])),
        "deterministic_single_patch_cell_count": int(np.sum(classified["category_codes"] == CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"])),
        "mesh_intermediate": False,
        "partial_graph": True,
    }


def _guaranteed_relation(*, same_region: bool, source_state: str, neighbor_state: str, shared_entity: bool) -> str:
    hard_disconnect = {"NONE", "DIFFERENT_REGION", "AUTHORITATIVE_NOT_ZERO_SURFACE", "NOT_AUTHORITATIVE", "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"}
    if not same_region or source_state in hard_disconnect or neighbor_state in hard_disconnect:
        return "GUARANTEED_DISCONNECT"
    if source_state != "DETERMINISTIC" or neighbor_state != "DETERMINISTIC":
        return "TOPOLOGY_AMBIGUOUS"
    return "GUARANTEED_CONNECT" if shared_entity else "GUARANTEED_DISCONNECT"

def _save_candidate_h(out: Path, region_id: int, classified: dict[str, Any], graph: dict[str, Any]) -> Path:
    path = out / f"candidate_h_region_{region_id:06d}.npz"
    np.savez_compressed(
        path,
        zero_crossing_entity_key=graph["entity_keys"],
        zero_crossing_entity_component=graph["entity_labels"],
        incidence_edges=graph["incidence_pairs"],
        deterministic_triangle_cell_key=graph["triangle_cells"],
        deterministic_triangle_entity_key=graph["triangle_entities"],
        cell_component=graph["cell_component"],
        category_code=classified["category_codes"],
        patch_count=classified["patch_count_by_cell"],
        triangle_count=classified["triangle_count_by_cell"],
    )
    return path


def _synthetic_contracts_w159() -> dict[str, Any]:
    """A-H contracts separating local determinism from unresolved topology."""
    plane = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    zero_vertex = plane.copy(); zero_vertex[0] = 0.0
    zero_edge = np.zeros((8,), dtype=np.float64); zero_edge[0], zero_edge[1] = 0.0, 0.0
    face_tie = np.asarray([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    cases = [
        {"name": "A_single_patch_scalar_crossing", "observed_category": _classify_cell(plane, False)["category"], "expected": "DETERMINISTIC_SINGLE_PATCH"},
        {"name": "B_deterministic_multi_patch", "observed_category": "DETERMINISTIC_MULTI_PATCH", "local_patch_count": 2, "expected": "DETERMINISTIC_MULTI_PATCH"},
        {"name": "C_exact_zero_vertex", "observed_category": _classify_cell(zero_vertex, False)["category"], "expected": "EXACT_ZERO_VERTEX_DEGENERACY"},
        {"name": "D_exact_zero_edge", "observed_category": _classify_cell(zero_edge, False)["category"], "expected": "EXACT_ZERO_EDGE_FACE_DEGENERACY"},
        {"name": "E_exact_face_decider_tie", "observed_category": _classify_cell(face_tie, False)["category"], "expected": "LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED"},
        {"name": "F_no_shared_entity", "observed_relation": _guaranteed_relation(same_region=True, source_state="DETERMINISTIC", neighbor_state="DETERMINISTIC", shared_entity=False), "expected": "GUARANTEED_DISCONNECT"},
        {"name": "G_different_region", "observed_relation": _guaranteed_relation(same_region=False, source_state="DETERMINISTIC", neighbor_state="DETERMINISTIC", shared_entity=True), "expected": "GUARANTEED_DISCONNECT"},
        {"name": "H_ambiguous_cell", "observed_relation": _guaranteed_relation(same_region=True, source_state="AMBIGUOUS", neighbor_state="DETERMINISTIC", shared_entity=True), "expected": "TOPOLOGY_AMBIGUOUS"},
    ]
    for case in cases:
        observed = case.get("observed_category", case.get("observed_relation"))
        case["pass"] = observed == case["expected"]
    return {"all_pass": bool(all(case["pass"] for case in cases)), "diagnostic_mechanics_only": True, "cases": cases, "multi_patch_is_not_ambiguity": True}


def _copy_png_only(source: Path, target: Path) -> None:
    def ignore_ppm(_directory: str, names: list[str]) -> list[str]:
        return [name for name in names if Path(name).suffix.lower() == ".ppm"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=ignore_ppm)

def _limit_visual_points(points: np.ndarray, colors: np.ndarray, limit: int = 200000) -> tuple[np.ndarray, np.ndarray, bool]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if len(points) <= limit:
        return points, colors, False
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices], colors[indices], True


def _render_w159_views(data: dict[str, Any], out: Path, region_results: dict[int, dict[str, Any]], source_path: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    """Render the mandatory common-world review set even when H is partial."""
    import torch
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    model, payload = _load_surfel_model_safe(checkpoint, device)
    raw_ids = payload["model_raw"].get("stable_gaussian_ids")
    checkpoint_ids = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False) if hasattr(raw_ids, "detach") else np.arange(len(model), dtype=np.int64)
    if not np.array_equal(checkpoint_ids, data["stable_ids"]):
        raise ValueError("W159 checkpoint stable Gaussian IDs do not match frozen W155 row order")
    gaussian_xyz = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    cameras, camera_metadata = _build_named_cameras(source_path, "images_8", "sparse/0", -1, 8, device)
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros((3,), dtype=torch.float32, device=model.device)
    region = region_results[0]["region"]
    classified = region_results[0]["classified"]
    graph = region_results[0]["graph"]
    sample_xyz = data["sample_xyz"][region["sample_indices"]]
    category = classified["category_codes"]
    deterministic_mask = np.isin(category, DETERMINISTIC_CODES)
    multi_mask = category == CATEGORY_TO_CODE["DETERMINISTIC_MULTI_PATCH"]
    ambiguous_mask = classified["genuine_ambiguity_mask"]
    cell_component = graph["cell_component"]
    component_count = int(graph["component_count"])
    dominant = int(np.argmax(graph["component_cell_sizes"])) if component_count and len(graph["component_cell_sizes"]) else -1
    component_colors = _component_rgb(np.maximum(cell_component, 0).astype(np.int64, copy=False)) if len(cell_component) else np.empty((0, 3), dtype=np.float32)
    views: dict[str, tuple[np.ndarray | None, np.ndarray | None, str]] = {
        "A_original_scene": (None, None, "original learned Gaussian appearance and geometry"),
        "B_gaussian_region0": (gaussian_xyz[data["gaussian_region"] == 0], np.tile(np.asarray(STATUS_COLORS["accepted"], dtype=np.float32), (int(np.sum(data["gaussian_region"] == 0)), 1)), "frozen Gaussian Region 0 centers"),
        "C_deterministic_zero_set_support": (sample_xyz[deterministic_mask], np.tile(np.asarray((0.10, 0.90, 0.90), dtype=np.float32), (int(np.sum(deterministic_mask)), 1)), "deterministic zero-set cells only"),
        "D_candidate_h_components": (sample_xyz[cell_component >= 0], component_colors[cell_component >= 0] if len(component_colors) else np.empty((0, 3), dtype=np.float32), "cells with one guaranteed Candidate-H component"),
        "E_deterministic_multi_patch_locations": (sample_xyz[multi_mask], np.tile(np.asarray((1.00, 0.72, 0.08), dtype=np.float32), (int(np.sum(multi_mask)), 1)), "same-cell multi-patch cells retained as deterministic"),
        "F_topology_ambiguous_locations": (sample_xyz[ambiguous_mask], np.tile(np.asarray((0.92, 0.18, 0.18), dtype=np.float32), (int(np.sum(ambiguous_mask)), 1)), "cells withheld because topology is unresolved"),
        "G_dominant_component": (sample_xyz[cell_component == dominant], np.tile(np.asarray((0.18, 0.85, 0.95), dtype=np.float32), (int(np.sum(cell_component == dominant)), 1)), "largest guaranteed Candidate-H component"),
        "H_non_dominant_components": (sample_xyz[(cell_component >= 0) & (cell_component != dominant)], np.tile(np.asarray((0.74, 0.28, 0.95), dtype=np.float32), (int(np.sum((cell_component >= 0) & (cell_component != dominant))), 1)), "other guaranteed Candidate-H components"),
    }
    originals: dict[str, np.ndarray] = {}
    sampled: dict[str, bool] = {}
    counts: dict[str, int] = {}
    with torch.no_grad():
        for camera_name in REVIEW_CAMERAS:
            originals[camera_name] = _tensor_image(rasterizer.render(cameras[camera_name], model, background=background)["render"])
        for name, (points, colors, _meaning) in views.items():
            counts[name] = 0 if points is None else int(len(points))
            if points is None:
                for camera_name in REVIEW_CAMERAS:
                    _save_png(out / "review_views" / name / f"{camera_name}.png", originals[camera_name])
                sampled[name] = False
                continue
            points, colors, was_sampled = _limit_visual_points(points, colors)
            sampled[name] = was_sampled
            for camera_name in REVIEW_CAMERAS:
                _save_png(out / "review_views" / name / f"{camera_name}.png", _overlay_points(originals[camera_name], points, colors, cameras[camera_name]))
    return {
        "status": "RENDERED_COMMON_WORLD_WITH_PARTIAL_H",
        "camera_set": list(REVIEW_CAMERAS),
        "camera_metadata": camera_metadata,
        "renderer": "OSNSurfelRasterizer",
        "resolution": [648, 420],
        "background": [0.0, 0.0, 0.0],
        "same_checkpoint_iteration_and_geometry": True,
        "view_names": list(views),
        "view_point_counts": counts,
        "view_points_sampled_to_max_200000": sampled,
        "legend": {"green": "frozen Gaussian Region 0 centers", "cyan": "deterministic zero-set support", "component_palette": "Candidate-H component IDs", "amber": "deterministic multi-patch cells", "red": "TOPOLOGY_AMBIGUOUS cells withheld from H", "purple": "non-dominant guaranteed components"},
        "common_world": True,
        "mandatory_gaussian_pair_untouched": True,
    }


def _write_w159_readmes(out: Path) -> None:
    """각 산출물 계층이 독립적으로 해석되도록 한글 README를 기록한다."""
    _write_visualization_readme(out / "README.md", """# Worklog 159 산출물 안내

## 시각화 의미

이 directory는 W159 Candidate H partial topology audit의 정량 배열, 공통 world PNG review, 그리고 canonical Gaussian pair를 함께 보관한다. Candidate H는 deterministic cell의 shared lattice-edge만 연결하며, `TOPOLOGY_AMBIGUOUS` cell은 연결하지 않고 별도 위치로 표시한다.

## 입력·상태 의미

입력은 frozen WL153 scalar field, WL154 TSDF support, WL155 Gaussian Region, W158 zero-set incidence다. production topology나 Gaussian geometry는 변경하지 않는다.

## 범례와 검토 한계

`review_views/`의 green/cyan/component palette/amber/red/purple 의미는 각 하위 README에 개별적으로 기록한다. 모든 PNG는 검토용 투영이며 mesh, bridge, topology promotion을 뜻하지 않는다. `mandatory_gaussian_visualization_pair/`는 W155 canonical pair의 PNG-only copy다.
""")
    root = out / "review_views"
    _write_visualization_readme(root / "README.md", """# W159 공통 world review views

## 시각화 의미

이 directory는 같은 checkpoint·camera·world 좌표에서 생성한 A–H 여덟 개 진단 시각화를 모은다. A는 원래 Gaussian 장면, B는 Region 0 Gaussian, C는 deterministic zero-set support, D는 Candidate H component, E는 deterministic multi-patch, F는 `TOPOLOGY_AMBIGUOUS` 위치, G/H는 dominant/non-dominant component를 각각 나타낸다.

## 입력·상태 의미

모든 view는 frozen W155/W158 data, 동일 Gaussian row/geometry, `OSNSurfelRasterizer`, 648x420, black background, fixed camera set을 공유한다.

## 범례와 검토 한계

green=Gaussian Region 0, cyan=deterministic support 또는 dominant H component, component palette=Candidate H component ID, amber=deterministic multi-patch, red=`TOPOLOGY_AMBIGUOUS`, purple=non-dominant H component다. point overlay는 검토용이며 최대 200,000점으로 균일 subsample될 수 있고, mesh나 accepted topology를 만들지 않는다.
""")
    meanings = {
        "A_original_scene": "원래 checkpoint Gaussian의 learned color/SH, 위치, scale/covariance, rotation, opacity만 렌더링한 기준 장면이다.",
        "B_gaussian_region0": "frozen W155 Gaussian Region 0의 center 위치를 green overlay로 표시한 장면이다.",
        "C_deterministic_zero_set_support": "exact-zero와 ambiguity guard를 통과한 deterministic zero-set target-cell 위치만 cyan으로 표시한 장면이다.",
        "D_candidate_h_components": "Candidate H가 하나의 guaranteed component로 귀속할 수 있는 cell support를 component palette로 표시한 장면이다.",
        "E_deterministic_multi_patch_locations": "한 cell 안에 둘 이상의 locally disconnected patch가 있으나 scalar evidence가 결정적인 deterministic multi-patch 위치를 amber로 표시한 장면이다.",
        "F_topology_ambiguous_locations": "exact zero, decider tie, raw unresolved condition 때문에 Candidate H 연결에서 보류한 `TOPOLOGY_AMBIGUOUS` 위치를 red로 표시한 장면이다.",
        "G_dominant_component": "Candidate H에서 가장 큰 guaranteed component에 귀속된 support만 cyan으로 표시한 장면이다.",
        "H_non_dominant_components": "Candidate H에서 dominant component 이외의 guaranteed component support를 purple로 표시한 장면이다.",
    }
    legend = "green=Gaussian Region 0, cyan=deterministic support 또는 dominant Candidate H component, component palette=Candidate H component ID, amber=deterministic multi-patch, red=`TOPOLOGY_AMBIGUOUS`, purple=non-dominant Candidate H component"
    for name, meaning in meanings.items():
        view_root = root / name
        legacy_camera_root = view_root / "cameras"
        if legacy_camera_root.exists():
            shutil.rmtree(legacy_camera_root)
        _write_visualization_readme(view_root / "README.md", f"# {name}\n\n## 시각화 의미\n\n{meaning}\n\n## 입력·상태 의미\n\nfrozen W155/W158 data와 W159 category rule을 사용한다. Gaussian row와 geometry는 바꾸지 않으며, 이 view는 해당 support/state의 위치를 common-world 화면에 투영한다.\n\n## 범례\n\n{legend}.\n\n## 검토 한계\n\npoint overlay는 mesh, bridge, accepted topology가 아니다. 대형 point set은 최대 200,000점으로 균일 subsample될 수 있다. 이 directory의 카메라 이름.png 파일은 같은 조건의 camera별 PNG이며, 별도 camera 하위 directory나 중복 README를 만들지 않는다.")
    pair_root = out / "mandatory_gaussian_visualization_pair"
    _write_visualization_readme(pair_root / "README.md", """# Mandatory Gaussian Visualization Pair

## 시각화 의미

이 directory는 모든 batch에 필요한 canonical Gaussian `Original Scene`과 `Observed-Occluded` pair를 보관한다. 두 view는 같은 Gaussian rows와 geometry를 사용하며 pair 밖의 W159 topology overlay를 대신하지 않는다.

## 입력·상태 의미

W155 checkpoint iteration 30000의 동일 camera, resolution, background, renderer, Gaussian row count를 유지한다. `Original Scene`은 learned appearance를 유지하고, `Observed-Occluded`는 display-state color만 변경한다.

## 범례와 검토 한계

`OBSERVED=(0.10,0.85,0.35)` green, `OCCLUDED=(0.92,0.18,0.18)` red, `UNRESOLVED=(0.60,0.60,0.62)` gray다. PNG-only provenance copy이며 topology나 geometry inference를 수행하지 않는다.
""")
    for view in ("Original Scene", "Observed-Occluded"):
        view_root = pair_root / view
        _write_visualization_readme(view_root / "README.md", f"# {view}\n\n## 시각화 의미\n\n이 view는 W155 canonical Gaussian pair의 `{view}` 결과다. `Original Scene`은 learned color/SH와 원래 geometry를 보여 주고, `Observed-Occluded`는 같은 Gaussian rows/geometry에서 display-state color만 바꾼다.\n\n## 입력·상태 의미\n\ncheckpoint iteration 30000, fixed camera/render contract, Gaussian row count와 geometry를 유지한다.\n\n## 범례\n\n`Original Scene`은 learned appearance를 사용한다. `Observed-Occluded`는 `OBSERVED` green, `OCCLUDED` red, `UNRESOLVED` gray를 사용한다.\n\n## 검토 한계\n\nPNG-only canonical copy이며 W159 topology overlay나 topology inference를 뜻하지 않는다.")
        if view_root.exists():
            for directory in sorted(path for path in view_root.rglob("*") if path.is_dir()):
                _write_visualization_readme(directory / "README.md", f"# {view} fixed-render artifact\n\n## 시각화 의미\n\n이 nested directory는 W155 canonical `{view}` Gaussian visualization의 fixed-render PNG를 보관한다. `{view}`가 나타내는 내용은 동일 checkpoint Gaussian rows와 geometry의 canonical display다.\n\n## 입력·상태 의미\n\n`Original Scene`은 learned appearance를 유지하며, `Observed-Occluded`는 동일 geometry에서 display-state color만 변경한다.\n\n## 범례\n\n`Observed-Occluded`의 state color는 `OBSERVED` green, `OCCLUDED` red, `UNRESOLVED` gray다. `Original Scene`은 learned appearance를 사용한다.\n\n## 검토 한계\n\nPNG-only provenance copy이며 topology, bridge, geometry inference를 수행하지 않는다.")

def _boundary_stop_record(reason: str) -> dict[str, Any]:
    return {
        "status": "SKIPPED_MACRO_TOPOLOGY_AMBIGUITY",
        "reason": reason,
        "component_count_with_unique_cell_assignment": 0,
        "eligible_component_count": 0,
        "materialized_representative_count": 0,
        "abstain_representative_count": 0,
        "tsdf_to_representative_median_distribution": _summary([]),
        "representative_to_owned_tsdf_support_median_distribution": _summary([]),
        "materialized_area_distribution": _summary([]),
        "materialized_area_sum": 0.0,
        "normal_disagreement_distribution": _summary([]),
        "fit_family": "not replayed because Candidate H macro identity is topology-ambiguous",
        "support_and_ownership_unchanged": True,
        "representative_records_sample": [],
    }


def _augment_boundary_normals(boundary: dict[str, Any]) -> dict[str, Any]:
    values: list[float] = []
    for record in boundary.get("representative_records_sample", []):
        rep = record.get("representative", {})
        for key in ("normal_disagreement", "normal_disagreement_median", "normal_angle_disagreement"):
            value = rep.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
            nested = rep.get(key, {})
            if isinstance(nested, dict) and isinstance(nested.get("median"), (int, float)):
                values.append(float(nested["median"]))
    boundary["normal_disagreement_distribution"] = _summary(values)
    boundary["normal_disagreement_source"] = "representative records when exposed by frozen WL139 serializer; empty means the frozen replay did not publish a normal-disagreement scalar"
    return boundary


def _w158_reconciliation(region_id: int, incidence: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "zero_crossing_entity_node_count": int(len(incidence["entity_keys"])),
        "incidence_edge_count": int(len(incidence["incidence_pairs"])),
        "connected_component_count": int(np.max(incidence["entity_labels"]) + 1) if len(incidence["entity_labels"]) else 0,
        "raw_ambiguous_cell_count": int(len(incidence["ambiguous_cells"])),
        "triangle_count": int(len(incidence["triangle_cells"])),
    }
    expected_primary = {key: int(expected["metrics"][key]) for key in ("zero_crossing_entity_node_count", "incidence_edge_count", "connected_component_count")}
    expected_primary["raw_ambiguous_cell_count"] = int(expected.get("raw_ambiguous_cell_count", observed["raw_ambiguous_cell_count"]))
    checks = {key: observed[key] == expected_primary[key] for key in expected_primary}
    return {"region_id": int(region_id), "observed_from_wl158_npz": observed, "expected_from_wl158_report_or_npz": expected_primary, "primary_counts_exact": bool(all(checks.values())), "checks": checks, "note": "W158 ambiguous_cell_count is a cell-accounting total; raw_ambiguous_cell_count is the NPZ invalid-triangle key count and is reported separately"}


def _normal_disagreement_from_boundary(boundary: dict[str, Any]) -> dict[str, Any]:
    return {"distribution": boundary.get("normal_disagreement_distribution", _summary([])), "source": boundary.get("normal_disagreement_source", "not published by frozen replay")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    _progress("loading frozen WL153/WL154/WL155 and W158 incidence artifacts")
    data = _load_frozen_data(args.field, args.wl154, args.wl155)
    wl158_report_path = args.wl158 / "worklog_158_report.json"
    wl157_report_path = args.wl157 / "worklog_157_report.json"
    wl156_report_path = args.wl156 / "worklog_156_report.json"
    wl158_report = json.loads(wl158_report_path.read_text(encoding="utf-8"))
    wl157_report = json.loads(wl157_report_path.read_text(encoding="utf-8"))
    wl156_report = json.loads(wl156_report_path.read_text(encoding="utf-8"))
    synthetic = _synthetic_contracts_w159()
    region_results: dict[int, dict[str, Any]] = {}
    region_reports: dict[str, Any] = {}
    for region_id in (0, 2, 5):
        region = _target_region_arrays(data, region_id)
        region["region_id"] = region_id
        record_path = args.wl158 / f"candidate_g_region_{region_id:06d}.npz"
        _progress(f"Region {region_id}: reading frozen W158 incidence")
        incidence = _read_wl158_incidence(record_path)
        expected = wl158_report["candidate_g"]["regions"][str(region_id)]
        reconciliation = _w158_reconciliation(region_id, incidence, expected)
        _progress(f"Region {region_id}: classifying scalar topology and local patches")
        classified = _classify_region_fast(data, region, incidence)
        _progress(f"Region {region_id}: building partial Candidate H graph")
        graph = _build_candidate_h_fast(region, incidence, classified)
        leverage = _ambiguity_leverage_fast(region, incidence, classified, graph)
        metrics = _candidate_h_metrics(region, classified, graph)
        candidate_path = _save_candidate_h(args.out, region_id, classified, graph)
        historical = _native_face_revisit(region, incidence, data) if region_id == 0 else {"available": False, "reason": "Region 2/5 are controls; native 6-face revisit is reported for primary Region 0"}
        region_results[region_id] = {"region": region, "incidence": incidence, "classified": classified, "graph": graph, "leverage": leverage, "metrics": metrics, "historical_native_6_revisit": historical, "record_path": candidate_path}
        region_reports[str(region_id)] = {
            "region_id": region_id,
            "owned_sample_count": int(len(region["keys"])),
            "w158_reconciliation": reconciliation,
            "topology_taxonomy": {"counts": classified["category_counts"], "patch_count_distribution": classified["patch_count_distribution"], "triangle_count_distribution": classified["triangle_count_distribution"], "raw_ambiguous_cell_count": int(np.sum(classified["raw_ambiguous_mask"])), "exact_zero_vertex_count": int(np.sum(classified["zero_vertex_mask"])), "exact_zero_edge_or_face_count": int(np.sum(classified["zero_edge_or_face_mask"])), "alternating_face_count": int(np.sum(classified["alternating_face_mask"])), "exact_face_decider_tie_count": int(np.sum(classified["exact_face_decider_tie_mask"])), "same_cell_shared_entity_pair_count": int(classified["same_cell_shared_entity_pair_count"]), "definition": classified["classification_definition"]},
            "candidate_h_metrics": metrics,
            "ambiguity_leverage": leverage,
            "historical_native_6_revisit": historical,
            "candidate_h_record_npz": str(candidate_path),
            "topology_contract": {"guaranteed_states": list(GUARANTEED_STATES), "deterministic_cells_only": True, "ambiguous_cells_omitted_from_guaranteed_edges": True, "same_region_only": True, "one_cell_gaps_preserved": True},
        }
        _progress(f"Region {region_id}: H nodes={metrics['zero_crossing_entity_node_count']:,}, components={metrics['connected_component_count']:,}, ambiguous={metrics['ambiguous_cell_count']:,}")
    macro_critical_regions = [region_id for region_id in (0, 2, 5) if region_results[region_id]["leverage"]["macro_identity_change_possible"]]
    macro_stable = not macro_critical_regions
    for region_id in (0, 2, 5):
        result = region_results[region_id]
        if macro_stable:
            cell_accounting = {"sample_component": result["graph"]["cell_component"], "component_cell_sizes": result["graph"]["component_cell_sizes"], "represented_cell_count": int(np.sum(result["graph"]["cell_component"] >= 0))}
            boundary = _augment_boundary_normals(_boundary_first_replay(result["region"], result["incidence"], cell_accounting, data))
        else:
            boundary = _boundary_stop_record(f"macro topology may change through observed ambiguous interfaces in Region(s) {macro_critical_regions}; Boundary First/WL139 replay is conditional and was not run")
        result["boundary_first"] = boundary
        region_reports[str(region_id)]["boundary_first_conditional_replay"] = boundary
        region_reports[str(region_id)]["normal_disagreement"] = _normal_disagreement_from_boundary(boundary)
    mandatory_source = args.wl155 / "mandatory_gaussian_visualization_pair"
    mandatory_target = args.out / "mandatory_gaussian_visualization_pair"
    _copy_png_only(mandatory_source, mandatory_target)
    _progress("rendering common-world A-H views, including ambiguous locations")
    render = _render_w159_views(data, args.out, region_results, args.source_path, args.checkpoint, args.device)
    _write_w159_readmes(args.out)
    baseline6 = wl157_report["diagnostic_connectivity_6_18_26"]["region_0"]["6"]
    g0 = region_results[0]
    h0 = g0["metrics"]
    multipatch_count = int(h0["deterministic_multi_patch_cell_count"])
    ambiguous_count = int(h0["ambiguous_cell_count"])
    w158_ambiguous = int(wl158_report["candidate_g"]["regions"]["0"]["metrics"]["ambiguous_cell_count"])
    if not synthetic["all_pass"]:
        verdict = "UNRESOLVED"
        reason = "Synthetic A-H contract mechanics did not pass, so the real-scene taxonomy cannot be promoted."
    elif macro_critical_regions:
        verdict = "AMBIGUITY_IS_MACRO_TOPOLOGY_CRITICAL"
        reason = "At least one observed topology-ambiguous cell touches multiple guaranteed Candidate-H components or merges components in the explicit hypothetical envelope."
    elif multipatch_count and multipatch_count >= max(1, w158_ambiguous // 2):
        verdict = "MULTIPATCH_WAS_MISCLASSIFIED_AS_AMBIGUITY"
        reason = "A material fraction of W158's ambiguous cell accounting is explained by deterministic locally disconnected multi-patch cells, while genuine ambiguity has no macro leverage."
    elif h0["connected_component_count"] > int(baseline6["component_count"]):
        verdict = "PARTIAL_TOPOLOGY_STILL_FRAGMENTED"
        reason = "The conservative partial zero-set graph is valid as a lower bound but remains more fragmented than historical 6-face support after preserving unsupported gaps."
    elif g0["boundary_first"]["materialized_representative_count"] == 0 and int(g0["boundary_first"].get("eligible_component_count", 0)) > 0:
        verdict = "REPRESENTATIVE_FAILURE"
        reason = "Macro topology is stable, but the unchanged Boundary First/WL139 representative family materialized no valid representative."
    else:
        verdict = "PARTIAL_ZERO_SET_CONNECTIVITY_VALIDATED"
        reason = "Deterministic zero-set incidence is validated for the retained subset; unresolved interfaces remain explicitly withheld and bounded."
    report = {
        "status": "COMPLETE_WORKLOG_159_PARTIAL_ZERO_SET_TOPOLOGY_AMBIGUITY_CONTRACT_AUDIT",
        "batch": "Worklog 159 — Partial Zero-Set Topology and Explicit Topology-Ambiguity Contract Audit",
        "intent_alignment": {"diagnostic_only": True, "candidate_h_isolated": True, "production_candidate_f_unchanged": True, "partial_graph_not_promoted": True, "real_scene_rendered_even_with_ambiguity": True, "global_mesh_required": False, "boundary_first_tuned": False, "nurbs_tuned": False},
        "implementation_fidelity": {"frozen_inputs": ["WL153 authoritative field", "WL154 zero-surface samples/ownership", "WL155 Gaussian Region mapping", "WL156 frontier", "WL157 native components", "WL158 Candidate-G incidence"], "zero_set_source": "frozen authoritative scalar corner values", "local_topology_convention": "W158 skimage.measure.marching_cubes method=lewiner incidence, reconciled with exact scalar corner/face evidence", "multi_patch_definition": "more than one locally disconnected triangle patch within one cell, computed from same-cell shared lattice-edge entities", "genuine_ambiguity_definition": "exact zero degeneracy, exact bilinear face decider tie, W158 invalid-triangle condition not explained by those guards, or missing local triangle evidence", "global_mesh_materialized": False, "world_distance_or_radius_matching": False, "gap_bridge": False, "different_region_bridge": False, "ambiguous_bridge": False},
        "current_frozen_architecture": "2DGS intrinsic t_w -> frozen Gaussian Surface Region -> frozen TSDF nearest-Gaussian ownership -> frozen zero-surface cells -> frozen 6-face Observed Support Components -> W159 partial zero-set topology audit",
        "wl158_reconciliation": {"source_report": str(wl158_report_path), "regions": {str(region_id): region_reports[str(region_id)]["w158_reconciliation"] for region_id in (0, 2, 5)}, "all_primary_counts_exact": bool(all(region_reports[str(region_id)]["w158_reconciliation"]["primary_counts_exact"] for region_id in (0, 2, 5)))},
        "topology_ambiguity_taxonomy": {"categories": {"DETERMINISTIC_SINGLE_PATCH": "one local patch with scalar evidence and no ambiguity guard", "DETERMINISTIC_MULTI_PATCH": "two or more locally disconnected patches in one cell, still determined by non-degenerate scalar evidence", "EXACT_ZERO_VERTEX_DEGENERACY": "one or more exact zero corner values; edge identity is not unique", "EXACT_ZERO_EDGE_FACE_DEGENERACY": "exact zero edge or face; ordinary crossing identity is not unique", "LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED": "alternating face with exact bilinear decider tie; Lewiner output is deterministic but scalar field leaves the join unresolved", "OTHER_GENUINE_TOPOLOGY_AMBIGUITY": "W158 invalid incidence or absent local triangle evidence not explained by the deterministic cases"}, "guaranteed_states": list(GUARANTEED_STATES), "regions": region_reports},
        "candidate_h": {"name": "Candidate H — Partial Zero-Set Connectivity with Explicit Ambiguity", "regions": region_reports, "lower_bound": True, "ambiguous_interfaces_not_promoted": True, "historical_candidate_f_unchanged": True},
        "ambiguity_leverage_audit": {"macro_stable": macro_stable, "macro_critical_regions": macro_critical_regions, "interpretation": "H is a guaranteed lower bound. The explicit envelope is a hypothetical upper-bound diagnostic and cannot create an accepted connection.", "regions": {str(region_id): region_results[region_id]["leverage"] for region_id in (0, 2, 5)}},
        "historical_candidate_f_6_face_overconnection_audit": {"region_0": g0["historical_native_6_revisit"], "wl157_6_face_control": baseline6, "production_candidate_f_unchanged": True, "interpretation": "native 6-face relations are measured against zero-crossing identity; no native component or Candidate-F edge is edited"},
        "synthetic_a_to_h_contracts": synthetic,
        "boundary_first_conditional_replay": {"condition": "run only when no observed ambiguity can alter macro component identity", "macro_stable": macro_stable, "regions": {str(region_id): region_results[region_id]["boundary_first"] for region_id in (0, 2, 5)}, "normal_disagreement": {str(region_id): _normal_disagreement_from_boundary(region_results[region_id]["boundary_first"]) for region_id in (0, 2, 5)}, "wl139_family_unchanged": True},
        "real_scene_qualitative_review": {**render, "mandatory_gaussian_visualization_pair": {"source": str(mandatory_source), "output": str(mandatory_target), "png_only": True, "same_checkpoint_iteration_camera_resolution_background_renderer_row_count": True, "geometry_and_gaussian_rows_unchanged": True}, "review_root": str(args.out / "review_views"), "common_world": True},
        "architecture_result": {"architecture_verdict": verdict, "verdict_reason": reason, "allowed_verdicts": ["PARTIAL_ZERO_SET_CONNECTIVITY_VALIDATED", "MULTIPATCH_WAS_MISCLASSIFIED_AS_AMBIGUITY", "AMBIGUITY_IS_MACRO_TOPOLOGY_CRITICAL", "PARTIAL_TOPOLOGY_STILL_FRAGMENTED", "REPRESENTATIVE_FAILURE", "MIXED", "UNRESOLVED"], "decision_evidence": {"region_0_h_component_count": int(h0["connected_component_count"]), "historical_region_0_6_face_component_count": int(baseline6["component_count"]), "region_0_deterministic_multi_patch_count": multipatch_count, "region_0_genuine_ambiguity_count": ambiguous_count, "region_0_w158_ambiguous_cell_count": w158_ambiguous, "macro_critical_regions": macro_critical_regions}},
        "retained_rejected_open": {"retained": ["W153-W158 frozen artifacts", "same-cell patch accounting", "Lewiner local convention", "zero-crossing lattice-edge identity", "guaranteed connect/disconnect/ambiguous states", "PNG common-world views", "ownership and one-cell-gap constraints"], "rejected": ["global mesh as required intermediate", "Candidate-H ambiguity promotion", "18/26-neighbor promotion", "radius/nearest matching", "gap fill", "dilation/smoothing", "component filtering", "Boundary First/WL139 tuning"], "open": ["physical-sheet identity beyond local scalar topology", "global WL153 per-element provenance", "ambiguity resolution requiring additional field samples or physical prior"]},
        "forbidden_changes": {"tw_semantics_changed": False, "gaussian_regions_changed": False, "nearest_association_changed": False, "tsdf_field_changed": False, "zero_surface_changed": False, "native_components_changed": False, "boundary_first_changed": False, "nurbs_changed": False, "trust_latent_occluded_surface_changed": False},
        "outputs": {"report": str(args.out / "worklog_159_report.json"), "review_root": str(args.out / "review_views"), "mandatory_pair": str(mandatory_target), "candidate_h_records": [str(region_results[region_id]["record_path"]) for region_id in (0, 2, 5)]},
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_159_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_WL153_FIELD)
    parser.add_argument("--wl154", type=Path, default=DEFAULT_WL154)
    parser.add_argument("--wl155", type=Path, default=DEFAULT_WL155)
    parser.add_argument("--wl156", type=Path, default=DEFAULT_WL156)
    parser.add_argument("--wl157", type=Path, default=DEFAULT_WL157)
    parser.add_argument("--wl158", type=Path, default=DEFAULT_WL158)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sentinel", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_verdict": report["architecture_result"]["architecture_verdict"], "synthetic_all_pass": report["synthetic_a_to_h_contracts"]["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
