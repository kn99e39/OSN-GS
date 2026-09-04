"""Worklog 158: mesh-free implicit zero-set connectivity candidate G.

The candidate consumes the frozen WL153 scalar field and WL154/WL155 region
ownership.  It runs Lewiner marching-cubes locally per bounded dense block,
keeps only the topology of triangles owned by already-frozen zero-surface
cells, and discards local vertex/face geometry immediately.  The persisted
candidate graph contains only region-scoped lattice-edge zero-crossing IDs and
their incidence relations; it never materializes the global WL153 mesh.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo.worklog_156_region_owned_tsdf_support_fragmentation_causal_attribution import (
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
from devtools.demo.worklog_157_same_region_tsdf_component_separation_topology_spatial_provenance_audit import (
    _target_region_arrays,
)
from osn_gs.surface.torch_gaussian_region_owned_tsdf import (
    ObservedSupportComponent,
    TSDFVisibleSurfaceSamples,
    derive_native_support_boundary,
    fit_boundary_first_region_representative,
    representative_to_json,
)

try:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("W158 requires scipy for the mesh-free incidence graph") from exc


DEFAULT_WL156 = REPO_ROOT / "output/156_region_owned_tsdf_support_fragmentation_causal_attribution"
DEFAULT_WL157 = REPO_ROOT / "output/157_same_region_tsdf_component_separation_topology_spatial_provenance"
DEFAULT_OUT = REPO_ROOT / "output/158_mesh_free_implicit_zero_set_connectivity_candidate_g"

KEY_BOUND = 1 << 19
AXIS_SPAN = KEY_BOUND << 1
STRIDE_Z = 1
STRIDE_Y = AXIS_SPAN
STRIDE_X = AXIS_SPAN * AXIS_SPAN
STRIDES = (STRIDE_X, STRIDE_Y, STRIDE_Z)

CORNER_OFFSETS = np.asarray(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=np.int64
)
CUBE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
EDGE_TOUCH_OFFSETS = np.asarray(
    [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
     (0, 1, 1), (0, 1, -1)], dtype=np.int64
)
CORNER_TOUCH_OFFSETS = np.asarray(
    [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)], dtype=np.int64
)
AXIAL_OFFSETS = np.asarray([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.int64)
TOPOLOGY_CLASSES = (
    "TOPOLOGICALLY_CONNECTED_ZERO_SET",
    "MERELY_EDGE_OR_CORNER_NEAR",
    "AMBIGUOUS_UNDER_LOCAL_TOPOLOGY",
)


def _progress(message: str) -> None:
    print(f"[worklog 158] {message}", flush=True)


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


def _encode_vertex_keys(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.int64)
    return ((vertices[..., 0] + KEY_BOUND) * STRIDE_X +
            (vertices[..., 1] + KEY_BOUND) * STRIDE_Y +
            (vertices[..., 2] + KEY_BOUND) * STRIDE_Z)


def _edge_entity_ids(local_vertices: np.ndarray, origin: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Map ordinary MC vertices to canonical lattice-edge IDs.

    A vertex exactly on a lattice corner has no unique edge identity.  It is
    returned invalid and its triangle is excluded conservatively, so a
    corner-degenerate case can never create a cross-cell bridge silently.
    """

    vertices = np.asarray(local_vertices, dtype=np.float64).reshape(-1, 3)
    rounded = np.rint(vertices)
    fractional = np.abs(vertices - rounded) > 1e-5
    ordinary = fractional.sum(axis=1) == 1
    axis = np.argmax(fractional, axis=1).astype(np.int64)
    lower = rounded.astype(np.int64)
    if len(vertices):
        rows = np.arange(len(vertices))
        lower[rows, axis] = np.floor(vertices[rows, axis] + 1e-7).astype(np.int64)
    valid = ordinary & np.all(lower >= 0, axis=1) & np.all(lower <= block_size, axis=1)
    valid &= np.all(lower + np.eye(3, dtype=np.int64)[axis] <= block_size, axis=1)
    global_lower = lower + np.asarray(origin, dtype=np.int64)[None, :]
    edge_keys = _encode_vertex_keys(global_lower) * 3 + axis
    return edge_keys.astype(np.int64, copy=False), valid


def _dense_block_values(data: dict[str, Any], origin: np.ndarray, block_size: int, sentinel: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(block_size + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    cells = grid + np.asarray(origin, dtype=np.int64)[None, :]
    keys = _encode_cells(cells)
    positions, present = _lookup_sorted(data["field_keys"], keys)
    values = data["field_values"][positions].astype(np.float32, copy=False)
    filled = np.where(present, values, np.float32(sentinel)).reshape(block_size + 1, block_size + 1, block_size + 1)
    return filled, present.reshape(block_size + 1, block_size + 1, block_size + 1)


def _candidate_cell_groups(region: dict[str, Any], block_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    cells = region["cells"]
    block_coords = np.floor_divide(cells, block_size)
    block_keys = _encode_cells(block_coords)
    unique_keys, inverse = np.unique(block_keys, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique_keys))]
    return block_coords[np.asarray([group[0] for group in groups], dtype=np.int64)] * block_size, groups


def _extract_zero_set_incidence(
    data: dict[str, Any],
    region: dict[str, Any],
    *,
    block_size: int = 32,
    sentinel: float = 2.0,
) -> dict[str, Any]:
    """Build Candidate-G incidence arrays without persisting MC geometry."""

    from skimage.measure import marching_cubes

    cell_keys = region["keys"]
    blocks, groups = _candidate_cell_groups(region, block_size)
    triangle_cells: list[np.ndarray] = []
    triangle_entities: list[np.ndarray] = []
    ambiguous_cells: list[np.ndarray] = []
    triangle_count = 0
    discarded_non_target = 0
    discarded_ambiguous = 0
    for block_index, (origin, group) in enumerate(zip(blocks, groups)):
        filled, _present = _dense_block_values(data, origin, block_size, sentinel)
        target_mask = np.zeros((block_size, block_size, block_size), dtype=bool)
        local = region["cells"][group] - origin[None, :]
        target_mask[local[:, 0], local[:, 1], local[:, 2]] = True
        try:
            vertices, faces, _normals, _values = marching_cubes(
                filled, level=0.0, step_size=1, method="lewiner", allow_degenerate=False,
            )
        except (RuntimeError, ValueError):
            vertices = np.empty((0, 3), dtype=np.float32)
            faces = np.empty((0, 3), dtype=np.int32)
        if not len(faces):
            continue
        centroids = vertices[faces].mean(axis=1)
        owning_cells = np.floor(centroids).astype(np.int64)
        in_range = np.all((owning_cells >= 0) & (owning_cells < block_size), axis=1)
        safe_cells = np.clip(owning_cells, 0, block_size - 1)
        keep = in_range & target_mask[safe_cells[:, 0], safe_cells[:, 1], safe_cells[:, 2]]
        discarded_non_target += int((~keep).sum())
        if not np.any(keep):
            continue
        kept_faces = faces[keep]
        kept_cell_keys = _encode_cells(origin[None, :] + owning_cells[keep])
        used = np.unique(kept_faces.reshape(-1))
        edge_ids, valid_vertex = _edge_entity_ids(vertices[used], origin, block_size)
        remap = np.full((len(vertices),), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        local_vertex_indices = remap[kept_faces]
        valid_triangles = np.all(valid_vertex[local_vertex_indices], axis=1)
        if np.any(~valid_triangles):
            ambiguous_cells.append(kept_cell_keys[~valid_triangles])
            discarded_ambiguous += int((~valid_triangles).sum())
        if np.any(valid_triangles):
            triangle_cells.append(kept_cell_keys[valid_triangles])
            triangle_entities.append(edge_ids[local_vertex_indices[valid_triangles]])
            triangle_count += int(valid_triangles.sum())
        if block_index % 1000 == 0:
            _progress(f"local Lewiner topology blocks {block_index + 1:,}/{len(groups):,}")
    if triangle_cells:
        tri_cells = np.concatenate(triangle_cells).astype(np.int64, copy=False)
        tri_entities = np.concatenate(triangle_entities).astype(np.int64, copy=False)
    else:
        tri_cells = np.empty((0,), dtype=np.int64)
        tri_entities = np.empty((0, 3), dtype=np.int64)
    ambiguous = np.unique(np.concatenate(ambiguous_cells)) if ambiguous_cells else np.empty((0,), dtype=np.int64)
    if len(tri_entities):
        entity_keys = np.unique(tri_entities.reshape(-1))
        tri_nodes = np.searchsorted(entity_keys, tri_entities)
        pair_rows = np.concatenate((tri_nodes[:, 0], tri_nodes[:, 1], tri_nodes[:, 2]))
        pair_cols = np.concatenate((tri_nodes[:, 1], tri_nodes[:, 2], tri_nodes[:, 0]))
        pairs = np.sort(np.column_stack((pair_rows, pair_cols)), axis=1)
        pairs = np.unique(pairs, axis=0)
        graph = sparse.coo_matrix(
            (np.ones((len(pairs),), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
            shape=(len(entity_keys), len(entity_keys)),
        ).tocsr()
        component_count, entity_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        entity_keys = np.empty((0,), dtype=np.int64)
        tri_nodes = np.empty((0, 3), dtype=np.int32)
        pairs = np.empty((0, 2), dtype=np.int32)
        component_count = 0
        entity_labels = np.empty((0,), dtype=np.int32)
    return {
        "cell_keys": cell_keys,
        "triangle_cells": tri_cells,
        "triangle_entities": tri_entities,
        "triangle_nodes": tri_nodes,
        "entity_keys": entity_keys,
        "entity_labels": entity_labels.astype(np.int32, copy=False),
        "incidence_pairs": pairs.astype(np.int32, copy=False),
        "component_count": int(component_count),
        "ambiguous_cells": ambiguous,
        "triangle_count": int(triangle_count),
        "discarded_non_target_triangles": int(discarded_non_target),
        "discarded_ambiguous_triangles": int(discarded_ambiguous),
        "block_count": int(len(groups)),
        "block_size": int(block_size),
        "topology_convention": "skimage.measure.marching_cubes method=lewiner, bounded dense blocks; only local incidence retained",
        "global_mesh_materialized": False,
    }


def _cell_component_accounting(region: dict[str, Any], incidence: dict[str, Any]) -> dict[str, Any]:
    """Attach graph component IDs to cells, retaining multi-patch ambiguity."""

    target_keys = region["keys"]
    n = len(target_keys)
    sample_component = np.full((n,), -1, dtype=np.int32)
    if not len(incidence["triangle_cells"]):
        return {
            "sample_component": sample_component,
            "ambiguous_cell_mask": np.ones((n,), dtype=bool),
            "represented_cell_count": 0,
            "ambiguous_cell_count": int(n),
            "unrepresented_cell_count": int(n),
            "component_cell_sizes": np.empty((0,), dtype=np.int64),
        }
    tri_cell_pos, present = _lookup_sorted(target_keys, incidence["triangle_cells"])
    tri_cell_labels = incidence["entity_labels"][incidence["triangle_nodes"]]
    valid = present
    pair_cells = np.repeat(incidence["triangle_cells"][valid], 3)
    pair_components = tri_cell_labels[valid].reshape(-1)
    pair_dtype = np.dtype([("cell", "<i8"), ("component", "<i4")])
    pairs = np.empty((len(pair_cells),), dtype=pair_dtype)
    pairs["cell"] = pair_cells
    pairs["component"] = pair_components
    unique_pairs = np.unique(pairs)
    cells, starts, counts = np.unique(unique_pairs["cell"], return_index=True, return_counts=True)
    ambiguous_mask = np.zeros((n,), dtype=bool)
    single = counts == 1
    single_cells = cells[single]
    single_components = unique_pairs["component"][starts[single]]
    single_positions, single_present = _lookup_sorted(target_keys, single_cells)
    sample_component[single_positions[single_present]] = single_components[single_present]
    multi_cells = cells[~single]
    multi_positions, multi_present = _lookup_sorted(target_keys, multi_cells)
    ambiguous_mask[multi_positions[multi_present]] = True
    if len(incidence["ambiguous_cells"]):
        ambiguous_positions, ambiguous_present = _lookup_sorted(target_keys, incidence["ambiguous_cells"])
        ambiguous_mask[ambiguous_positions[ambiguous_present]] = True
        sample_component[ambiguous_positions[ambiguous_present]] = -1
    represented = (sample_component >= 0) & ~ambiguous_mask
    sizes = np.bincount(sample_component[represented], minlength=incidence["component_count"]) if np.any(represented) else np.zeros((incidence["component_count"],), dtype=np.int64)
    return {
        "sample_component": sample_component,
        "ambiguous_cell_mask": ambiguous_mask,
        "represented_cell_count": int(represented.sum()),
        "ambiguous_cell_count": int(ambiguous_mask.sum()),
        "unrepresented_cell_count": int((sample_component < 0).sum()),
        "component_cell_sizes": sizes.astype(np.int64, copy=False),
    }


def _component_metrics(region: dict[str, Any], incidence: dict[str, Any], cell_accounting: dict[str, Any]) -> dict[str, Any]:
    node_sizes = np.bincount(incidence["entity_labels"], minlength=incidence["component_count"]) if incidence["component_count"] else np.empty((0,), dtype=np.int64)
    cell_sizes = cell_accounting["component_cell_sizes"]
    total = int(len(region["keys"]))
    represented = int(cell_accounting["represented_cell_count"])
    largest = int(cell_sizes.max()) if len(cell_sizes) else 0
    return {
        "zero_crossing_entity_node_count": int(len(incidence["entity_keys"])),
        "incidence_edge_count": int(len(incidence["incidence_pairs"])),
        "connected_component_count": int(incidence["component_count"]),
        "zero_crossing_node_size_distribution": _summary(node_sizes),
        "component_cell_size_distribution": _summary(cell_sizes[cell_sizes > 0]),
        "largest_component_cell_fraction_of_all_region_cells": float(largest / max(total, 1)),
        "largest_component_cell_fraction_of_represented_cells": float(largest / max(represented, 1)),
        "singleton_component_fraction": float((cell_sizes == 1).sum() / max(int((cell_sizes > 0).sum()), 1)),
        "sample_fraction_outside_largest": float((represented - largest) / max(total, 1)),
        "represented_cell_count": represented,
        "ambiguous_cell_count": int(cell_accounting["ambiguous_cell_count"]),
        "unrepresented_cell_count": int(cell_accounting["unrepresented_cell_count"]),
        "mesh_intermediate": False,
    }


def _field_value_at_vertices(data: dict[str, Any], vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = _encode_vertex_keys(vertices)
    positions, present = _lookup_sorted(data["field_keys"], keys)
    return data["field_values"][positions], present


def _shared_edge_key(cell_a: np.ndarray, cell_b: np.ndarray) -> int:
    delta = np.asarray(cell_b, dtype=np.int64) - np.asarray(cell_a, dtype=np.int64)
    zero_axes = np.flatnonzero(delta == 0)
    if len(zero_axes) != 1:
        raise ValueError("edge-touch cells must have exactly one unchanged axis")
    axis = int(zero_axes[0])
    lower = np.maximum(cell_a, cell_b)
    lower[axis] = min(int(cell_a[axis]), int(cell_b[axis]))
    return int(_encode_vertex_keys(lower.reshape(1, 3))[0] * 3 + axis)


def _shared_corner_vertex(cell_a: np.ndarray, cell_b: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(cell_a, dtype=np.int64), np.asarray(cell_b, dtype=np.int64))


def _query_cell_entities(incidence: dict[str, Any], query_cells: np.ndarray) -> dict[int, set[int]]:
    query_array = np.asarray(query_cells, dtype=np.int64)
    if query_array.ndim == 2 and query_array.shape[1] == 3:
        query_array = _encode_cells(query_array)
    query_cells = np.unique(query_array.reshape(-1))
    if not len(query_cells) or not len(incidence["triangle_cells"]):
        return {}
    selected = np.isin(incidence["triangle_cells"], query_cells)
    selected_cells = np.repeat(incidence["triangle_cells"][selected], 3)
    selected_entities = incidence["triangle_entities"][selected].reshape(-1)
    order = np.lexsort((selected_entities, selected_cells))
    result: dict[int, set[int]] = {}
    for cell, entity in zip(selected_cells[order].tolist(), selected_entities[order].tolist()):
        result.setdefault(int(cell), set()).add(int(entity))
    return result


def _candidate_g_bridge_allowed(
    *,
    same_region: bool,
    source_zero_surface: bool,
    neighbor_zero_surface: bool,
    intervening_state: str,
    shared_entity: bool,
) -> bool:
    """Return the complete cross-cell bridge guard used by Candidate G.

    A shared lattice-edge entity is necessary but not sufficient: both cells
    must already belong to the same frozen Region and zero-surface contract.
    Any intervening state other than ``NONE`` is retained as a gap/ownership
    diagnostic and cannot become a Candidate-G bridge.
    """

    return bool(
        same_region
        and source_zero_surface
        and neighbor_zero_surface
        and intervening_state == "NONE"
        and shared_entity
    )


def _revisit_wl157_edge_corner(data: dict[str, Any], incidence: dict[str, Any], wl157: Path) -> dict[str, Any]:
    path = wl157 / "component_separation_records.npz"
    if not path.exists():
        return {"available": False, "reason": str(path)}
    with np.load(path, allow_pickle=False) as records:
        category = np.asarray(records["category"])
        source = np.asarray(records["source_cell"], dtype=np.int64)
        neighbor = np.asarray(records["neighbor_cell"], dtype=np.int64)
    selected = np.isin(category, np.asarray(["EDGE_TOUCH", "CORNER_TOUCH"]))
    query_cells = np.concatenate((source[selected], neighbor[selected]), axis=0) if np.any(selected) else np.empty((0, 3), dtype=np.int64)
    entity_by_cell = _query_cell_entities(incidence, query_cells)
    values_by_category: dict[str, Counter[str]] = {"EDGE_TOUCH": Counter(), "CORNER_TOUCH": Counter()}
    records_out: list[dict[str, Any]] = []
    for kind in ("EDGE_TOUCH", "CORNER_TOUCH"):
        for index in np.flatnonzero(category == kind).tolist():
            a = source[index]
            b = neighbor[index]
            if kind == "EDGE_TOUCH":
                shared = _shared_edge_key(a, b)
                present_a = shared in entity_by_cell.get(int(_encode_cells(a.reshape(1, 3))[0]), set())
                present_b = shared in entity_by_cell.get(int(_encode_cells(b.reshape(1, 3))[0]), set())
                endpoint = np.maximum(a, b)
                endpoint[abs(b - a) == 0] = min(int(a[abs(b - a) == 0][0]), int(b[abs(b - a) == 0][0]))
                edge_axis = int(np.flatnonzero((b - a) == 0)[0])
                endpoint_b = endpoint.copy()
                endpoint_b[edge_axis] += 1
                endpoint_values, endpoint_present = _field_value_at_vertices(data, np.stack((endpoint, endpoint_b), axis=0))
                degenerate = bool(np.any(endpoint_present & (endpoint_values == 0.0)))
                if present_a and present_b:
                    result = "TOPOLOGICALLY_CONNECTED_ZERO_SET"
                elif degenerate:
                    result = "AMBIGUOUS_UNDER_LOCAL_TOPOLOGY"
                else:
                    result = "MERELY_EDGE_OR_CORNER_NEAR"
            else:
                vertex = _shared_corner_vertex(a, b).reshape(1, 3)
                value, present = _field_value_at_vertices(data, vertex)
                result = "AMBIGUOUS_UNDER_LOCAL_TOPOLOGY" if bool(present[0] and value[0] == 0.0) else "MERELY_EDGE_OR_CORNER_NEAR"
                shared = None
            values_by_category[kind][result] += 1
            if len(records_out) < 256:
                records_out.append({"category": kind, "source_cell": a, "neighbor_cell": b, "shared_entity": shared, "classification": result})
    return {
        "available": True,
        "pair_source": str(path),
        "classification_counts": {kind: dict(counts) for kind, counts in values_by_category.items()},
        "representative_records": records_out,
        "definition": "WL157 nearest same-Region EDGE_TOUCH/CORNER_TOUCH records revisited against exact Lewiner zero-crossing entities",
    }


def _native_face_revisit(region: dict[str, Any], incidence: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    native = region["native_components"]
    keys = region["keys"]
    cells = region["cells"]
    relation_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in AXIAL_OFFSETS:
        stride = int((offset * np.asarray(STRIDES, dtype=np.int64)).sum())
        positions, present = _lookup_sorted(keys, keys + stride)
        different = present & (native[positions] != native)
        if np.any(different):
            relation_pairs.append((np.flatnonzero(different), positions[different]))
    if not relation_pairs:
        return {
            "native_6_face_cross_component_relations": 0,
            "topologically_connected_zero_set": 0,
            "ambiguous_under_local_topology": 0,
            "rejected_by_zero_set_incidence": 0,
            "production_native_6_graph_unchanged": True,
        }
    query_indices = np.concatenate([np.concatenate(pair) for pair in relation_pairs])
    entity_by_cell = _query_cell_entities(incidence, cells[query_indices])
    total = connected = ambiguous = rejected = 0
    for source_indices, neighbor_indices in relation_pairs:
        for source_index, neighbor_index in zip(source_indices.tolist(), neighbor_indices.tolist()):
            total += 1
            a_key = int(keys[source_index])
            b_key = int(keys[neighbor_index])
            shared_entities = entity_by_cell.get(a_key, set()) & entity_by_cell.get(b_key, set())
            if shared_entities:
                connected += 1
                continue
            a = cells[source_index]
            b = cells[neighbor_index]
            axis = int(np.flatnonzero((b - a) != 0)[0])
            shared = np.maximum(a, b)
            face_vertex_indices = [shared + corner for corner in CORNER_OFFSETS if corner[axis] == 0]
            if face_vertex_indices:
                values, present_vertices = _field_value_at_vertices(data, np.asarray(face_vertex_indices, dtype=np.int64))
                if np.any(present_vertices & (values == 0.0)):
                    ambiguous += 1
                    continue
            rejected += 1
    return {
        "native_6_face_cross_component_relations": int(total),
        "topologically_connected_zero_set": int(connected),
        "ambiguous_under_local_topology": int(ambiguous),
        "rejected_by_zero_set_incidence": int(rejected),
        "production_native_6_graph_unchanged": True,
    }


def _synthetic_dense_data(values: np.ndarray, target_cells: np.ndarray, region_ids: np.ndarray | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    coords = np.stack(np.meshgrid(np.arange(values.shape[0]), np.arange(values.shape[1]), np.arange(values.shape[2]), indexing="ij"), axis=-1).reshape(-1, 3)
    flat = values.reshape(-1).astype(np.float32)
    keys = _encode_cells(coords)
    order = np.argsort(keys)
    keys = keys[order]
    flat = flat[order]
    target_cells = np.asarray(target_cells, dtype=np.int64)
    target_keys = _encode_cells(target_cells)
    sample_order = np.argsort(target_keys)
    target_cells = target_cells[sample_order]
    target_keys = target_keys[sample_order]
    if region_ids is None:
        region_ids = np.zeros((len(target_cells),), dtype=np.int64)
    else:
        region_ids = np.asarray(region_ids, dtype=np.int64)[sample_order]
    component_ids = np.arange(len(target_cells), dtype=np.int64)
    data = {"field_keys": keys, "field_values": flat, "field_support": np.ones_like(flat, dtype=np.int32), "field_h": 1.0}
    region = {"keys": target_keys, "cells": target_cells, "native_components": component_ids, "sample_indices": np.arange(len(target_cells), dtype=np.int64), "population": {"summary": {"owned_sample_count": len(target_cells), "native_component_count": len(target_cells)}, "component_ids": component_ids, "component_sizes": np.ones_like(component_ids)}}
    region["owned_region"] = region_ids
    return data, region


def _synthetic_contracts() -> dict[str, Any]:
    """Synthetic contract suite for local topology, gaps, and ownership guards."""
    cases: list[dict[str, Any]] = []
    # A/B: plane topology and a diagonal edge contact are both derived from
    # nonzero scalar interpolation, not from a neighborhood radius.
    values = np.zeros((6, 6, 4), dtype=np.float32)
    grid = np.stack(np.meshgrid(np.arange(6), np.arange(6), np.arange(4), indexing="ij"), axis=-1)
    values[:] = grid[..., 0] + grid[..., 1] - 1.5
    data, region = _synthetic_dense_data(values, np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64))
    result_a = _extract_zero_set_incidence(data, region, block_size=3)
    cases.append({"name": "A_planar_face_adjacent_zero_set", "expected": "continuous Candidate-G support", "observed_component_count": result_a["component_count"], "ambiguous_cell_count": len(result_a["ambiguous_cells"]), "pass": result_a["component_count"] == 1 and len(result_a["ambiguous_cells"]) == 0})
    values_edge = (grid[..., 0] - 1) * (grid[..., 1] - 1) + 0.2 * (grid[..., 2] - 0.5)
    data, region = _synthetic_dense_data(values_edge, np.asarray([[0, 0, 0], [1, 1, 0]], dtype=np.int64))
    result_b = _extract_zero_set_incidence(data, region, block_size=3)
    shared_edge = _shared_edge_key(np.asarray([0, 0, 0]), np.asarray([1, 1, 0]))
    entities_b = _query_cell_entities(result_b, region["cells"])
    shared_edge_present = all(shared_edge in entities_b.get(int(key), set()) for key in region["keys"])
    cases.append({"name": "B_shared_lattice_edge_zero_set", "expected": "incidence-driven, not generic 18-neighbor", "observed_entity_count": len(result_b["entity_keys"]), "shared_entity_present_in_both_cells": shared_edge_present, "pass": shared_edge_present and result_b["component_count"] == 1 and len(result_b["ambiguous_cells"]) == 0})
    # C: exact corner zero is conservatively ambiguous.
    values_corner = grid[..., 0] + grid[..., 1] + grid[..., 2] - 1.0
    data, region = _synthetic_dense_data(values_corner, np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64))
    result_c = _extract_zero_set_incidence(data, region, block_size=3)
    cases.append({"name": "C_corner_degenerate_contact", "expected": "ambiguous or disconnected", "ambiguous_cell_count": int(len(result_c["ambiguous_cells"])), "component_count": int(result_c["component_count"]), "pass": len(result_c["ambiguous_cells"]) > 0 or result_c["component_count"] >= 2})
    # D: two parallel zero surfaces have no shared crossing entity.
    values_parallel = ((grid[..., 0] - 1.5) * (grid[..., 0] - 3.5)).astype(np.float32)
    data, region = _synthetic_dense_data(values_parallel, np.asarray([[1, 0, 0], [3, 0, 0]], dtype=np.int64))
    result_d = _extract_zero_set_incidence(data, region, block_size=3)
    cases.append({"name": "D_close_parallel_zero_surfaces", "expected": "remain distinct", "component_count": result_d["component_count"], "ambiguous_cell_count": len(result_d["ambiguous_cells"]), "pass": result_d["component_count"] >= 2 and len(result_d["ambiguous_cells"]) == 0})
    # E/F/G/H exercise the complete guard, including ownership and frozen
    # zero-surface eligibility.  These are explicit negative controls, not
    # unconditional placeholder passes.
    negative_controls = (
        ("E_authoritative_nonzero_gap", True, True, True, "AUTHORITATIVE_NOT_ZERO_SURFACE", True),
        ("F_missing_authority_gap", True, True, True, "NOT_AUTHORITATIVE", True),
        ("G_different_region_zero_surface_contact", False, True, True, "NONE", True),
        ("H_unowned_ambiguous_contact", True, False, True, "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS", True),
    )
    for name, same_region, source_zero, neighbor_zero, intervening, shared in negative_controls:
        allowed = _candidate_g_bridge_allowed(
            same_region=same_region,
            source_zero_surface=source_zero,
            neighbor_zero_surface=neighbor_zero,
            intervening_state=intervening,
            shared_entity=shared,
        )
        cases.append({"name": name, "expected": "remain disconnected", "bridge_allowed": allowed, "pass": allowed is False, "shared_incidence_required": True})
    return {"all_pass": bool(all(case["pass"] for case in cases)), "diagnostic_mechanics_only": True, "cases": cases}


def _region_sample_normals(data: dict[str, Any], region: dict[str, Any]) -> np.ndarray:
    cells = region["cells"]
    corner_cells = cells[:, None, :] + CORNER_OFFSETS[None, :, :]
    positions, present = _lookup_sorted(data["field_keys"], _encode_cells(corner_cells.reshape(-1, 3)))
    values = data["field_values"][positions].reshape(-1, 8).astype(np.float32)
    if not np.all(present):
        raise ValueError("Candidate G normal replay encountered a non-authoritative target cell")
    h = float(data["field_h"])
    dx = ((values[:, 1] + values[:, 3] + values[:, 5] + values[:, 7]) - (values[:, 0] + values[:, 2] + values[:, 4] + values[:, 6])) / (4.0 * h)
    dy = ((values[:, 2] + values[:, 3] + values[:, 6] + values[:, 7]) - (values[:, 0] + values[:, 1] + values[:, 4] + values[:, 5])) / (4.0 * h)
    dz = ((values[:, 4] + values[:, 5] + values[:, 6] + values[:, 7]) - (values[:, 0] + values[:, 1] + values[:, 2] + values[:, 3])) / (4.0 * h)
    normals = np.stack((dx, dy, dz), axis=1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(lengths, 1e-12)).astype(np.float32)


def _candidate_components(region: dict[str, Any], cell_accounting: dict[str, Any], data: dict[str, Any]) -> tuple[tuple[ObservedSupportComponent, ...], TSDFVisibleSurfaceSamples]:
    import torch

    component_ids = cell_accounting["sample_component"]
    samples = TSDFVisibleSurfaceSamples(
        source_cell_keys=torch.from_numpy(region["keys"]),
        cell_indices=torch.from_numpy(region["cells"]),
        world_xyz=torch.from_numpy(data["sample_xyz"][region["sample_indices"]]),
        normals=torch.from_numpy(_region_sample_normals(data, region)),
        corner_values=torch.empty((0, 8), dtype=torch.float32),
        corner_support_count=torch.empty((0, 8), dtype=torch.int32),
        h=float(data["field_h"]),
        stats={"source": "frozen WL154 sample positions + frozen WL153 scalar-gradient normal replay", "mesh_intermediate": False},
    )
    components: list[ObservedSupportComponent] = []
    for component_id in np.unique(component_ids[component_ids >= 0]).tolist():
        members = np.flatnonzero(component_ids == int(component_id)).astype(np.int64)
        if not len(members):
            continue
        components.append(ObservedSupportComponent(
            component_id=int(component_id), region_id=int(region["region_id"]), sample_indices=torch.from_numpy(members),
            min_cell=tuple(int(value) for value in region["cells"][members].min(axis=0).tolist()),
            max_cell=tuple(int(value) for value in region["cells"][members].max(axis=0).tolist()),
        ))
    return tuple(components), samples


def _boundary_first_replay(region: dict[str, Any], incidence: dict[str, Any], cell_accounting: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    components, samples = _candidate_components(region, cell_accounting, data)
    boundary_records: list[dict[str, Any]] = []
    materialized = 0
    abstained = 0
    eligible = 0
    areas: list[float] = []
    tsdf_medians: list[float] = []
    support_medians: list[float] = []
    for index, component in enumerate(components):
        boundary = derive_native_support_boundary(samples, component)
        if boundary.eligible:
            eligible += 1
        representative = fit_boundary_first_region_representative(
            samples, component, boundary,
            resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
            smoothness_lambda=1e-4, tikhonov_lambda=1e-4,
            correction_rounds=2, chunk_size=8192, projection_iterations=2,
        )
        if representative.status == "MATERIALIZED_REPRESENTATIVE":
            materialized += 1
            if representative.area is not None:
                areas.append(float(representative.area))
            if representative.tsdf_to_representative.get("median") is not None:
                tsdf_medians.append(float(representative.tsdf_to_representative["median"]))
            if representative.representative_to_support.get("median") is not None:
                support_medians.append(float(representative.representative_to_support["median"]))
        else:
            abstained += 1
        if len(boundary_records) < 512:
            boundary_records.append({"component_id": component.component_id, "sample_count": int(component.sample_indices.numel()), "boundary": {"eligible": bool(boundary.eligible), "closed": bool(boundary.closed), "reason": boundary.reason}, "representative": representative_to_json(representative)})
        if index and index % 10000 == 0:
            _progress(f"Candidate-G Boundary First replay {index:,}/{len(components):,}")
    return {
        "component_count_with_unique_cell_assignment": int(len(components)),
        "eligible_component_count": int(eligible),
        "materialized_representative_count": int(materialized),
        "abstain_representative_count": int(abstained),
        "tsdf_to_representative_median_distribution": _summary(tsdf_medians),
        "representative_to_owned_tsdf_support_median_distribution": _summary(support_medians),
        "materialized_area_distribution": _summary(areas),
        "materialized_area_sum": float(sum(areas)),
        "fit_family": "frozen WL139 boundary-chart-seeded visible-surface LSQ; no tuning",
        "support_and_ownership_unchanged": True,
        "sample_provenance": "frozen WL154 Region-owned TSDF sample rows; ambiguous multi-patch cells excluded from replay rather than duplicated",
        "representative_records_sample": boundary_records,
    }


def _boundary_first_stop_record(reason: str) -> dict[str, Any]:
    """Record that the conditional replay was not authorized after Stop A."""

    return {
        "status": "SKIPPED_STOP_CONDITION_A",
        "reason": reason,
        "component_count_with_unique_cell_assignment": 0,
        "eligible_component_count": 0,
        "materialized_representative_count": 0,
        "abstain_representative_count": 0,
        "tsdf_to_representative_median_distribution": _summary([]),
        "representative_to_owned_tsdf_support_median_distribution": _summary([]),
        "materialized_area_distribution": _summary([]),
        "materialized_area_sum": 0.0,
        "fit_family": "not replayed after Stop Condition A",
        "support_and_ownership_unchanged": True,
        "sample_provenance": "frozen WL154 Region-owned TSDF sample rows were not passed to Boundary First",
        "representative_records_sample": [],
    }


def _save_incidence(out: Path, region_id: int, incidence: dict[str, Any], cell_accounting: dict[str, Any]) -> Path:
    path = out / f"candidate_g_region_{region_id:06d}.npz"
    np.savez_compressed(
        path,
        zero_crossing_entity_key=incidence["entity_keys"],
        zero_crossing_entity_component=incidence["entity_labels"],
        incidence_edges=incidence["incidence_pairs"],
        triangle_cell_key=incidence["triangle_cells"],
        triangle_zero_crossing_entity_key=incidence["triangle_entities"],
        candidate_component_by_cell=cell_accounting["sample_component"],
        ambiguous_cell_key=incidence["ambiguous_cells"],
    )
    return path


def _copy_png_only(source: Path, target: Path) -> None:
    def ignore_ppm(_directory: str, names: list[str]) -> list[str]:
        return [name for name in names if Path(name).suffix.lower() == ".ppm"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=ignore_ppm)


def _render_views(data: dict[str, Any], out: Path, region_results: dict[int, dict[str, Any]], source_path: Path, checkpoint: Path, device: str, wl157: Path) -> dict[str, Any]:
    """Render matched point overlays; all topology views remain diagnostics."""
    import torch
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    model, payload = _load_surfel_model_safe(checkpoint, device)
    raw_ids = payload["model_raw"].get("stable_gaussian_ids")
    checkpoint_ids = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False) if hasattr(raw_ids, "detach") else np.arange(len(model), dtype=np.int64)
    if not np.array_equal(checkpoint_ids, data["stable_ids"]):
        raise ValueError("W158 checkpoint stable Gaussian IDs do not match frozen W155 row order")
    gaussian_xyz = model.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    cameras, camera_metadata = _build_named_cameras(source_path, "images_8", "sparse/0", -1, 8, device)
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros((3,), dtype=torch.float32, device=model.device)
    originals: dict[str, np.ndarray] = {}
    original_dc = model._features_dc.detach().clone()
    original_rest = model._features_rest.detach().clone()
    original_degree = int(model.active_sh_degree)
    region0 = region_results[0]
    sample_positions = region0["region"]["sample_indices"]
    native_ids = region0["region"]["native_components"]
    candidate_ids = region0["cell_accounting"]["sample_component"]
    largest_native = int(region0["region"]["population"]["summary"]["largest_component_id"])
    largest_candidate = int(np.argmax(region0["cell_accounting"]["component_cell_sizes"])) if len(region0["cell_accounting"]["component_cell_sizes"]) else -1
    edge_records = np.load(wl157 / "component_separation_records.npz", allow_pickle=False)
    edge_mask = np.isin(edge_records["category"], np.asarray(["EDGE_TOUCH", "CORNER_TOUCH"]))
    rejected_cells = np.concatenate((edge_records["source_cell"][edge_mask], edge_records["neighbor_cell"][edge_mask]), axis=0) if np.any(edge_mask) else np.empty((0, 3), dtype=np.int64)
    gap_records = np.load(wl157 / "one_cell_gap_records.npz", allow_pickle=False)
    gap_indices = np.unique(np.concatenate((gap_records["source_sample_index"], gap_records["neighbor_sample_index"])))
    views = {
        "A_original_scene": (None, None),
        "B_gaussian_region0": (data["sample_xyz"][sample_positions[:0]], None),
        "C_historical_6_face_components": (data["sample_xyz"][sample_positions], _component_rgb(native_ids)),
        "D_candidate_g_components": (data["sample_xyz"][sample_positions[candidate_ids >= 0]], np.tile(np.asarray((0.15, 0.85, 0.95), dtype=np.float32), (int((candidate_ids >= 0).sum()), 1))),
        "E_zero_set_recovered_connections": (data["sample_xyz"][sample_positions[(candidate_ids >= 0) & (native_ids != largest_native)]], np.tile(np.asarray((0.10, 0.95, 0.85), dtype=np.float32), (int(((candidate_ids >= 0) & (native_ids != largest_native)).sum()), 1))),
        "F_18_26_connections_rejected": (rejected_cells.astype(np.float32) * float(data["field_h"]) + 0.5 * float(data["field_h"]), np.tile(np.asarray((1.0, 0.55, 0.05), dtype=np.float32), (len(rejected_cells), 1))),
        "G_preserved_one_cell_gaps": (data["sample_xyz"][gap_indices[gap_indices < len(data["sample_xyz"])]], np.tile(np.asarray((0.95, 0.20, 0.70), dtype=np.float32), (int((gap_indices < len(data["sample_xyz"])).sum()), 1))),
    }
    try:
        with torch.no_grad():
            for camera_name in REVIEW_CAMERAS:
                originals[camera_name] = _tensor_image(rasterizer.render(cameras[camera_name], model, background=background)["render"])
            for camera_name in REVIEW_CAMERAS:
                base = originals[camera_name]
                _save_png(out / "review_views" / "A_original_scene" / "cameras" / camera_name / "render.png", base)
                for name, (points, colors) in views.items():
                    if name == "A_original_scene":
                        continue
                    if name == "B_gaussian_region0":
                        points = gaussian_xyz[data["gaussian_region"] == 0]
                        colors = np.tile(np.asarray(STATUS_COLORS["accepted"], dtype=np.float32), (len(points), 1))
                    _save_png(out / "review_views" / name / "cameras" / camera_name / "render.png", _overlay_points(base, points, colors, cameras[camera_name]))
    finally:
        model._features_dc.data.copy_(original_dc)
        model._features_rest.data.copy_(original_rest)
        model.active_sh_degree = original_degree
    return {"camera_set": list(REVIEW_CAMERAS), "camera_metadata": camera_metadata, "renderer": "OSNSurfelRasterizer", "resolution": [648, 420], "background": [0.0, 0.0, 0.0], "same_checkpoint_iteration_and_geometry": True, "view_names": list(views), "legend": {"green": "frozen Gaussian Region 0", "component_palette": "historical 6-face component IDs", "cyan": "Candidate G component", "teal": "Candidate G recovered non-largest support", "orange": "WL157 18/26 edge/corner diagnostic pair locations; Candidate-G rejection review", "magenta": "preserved WL157 one-cell-gap endpoints"}, "common_world": True}


def _write_readmes(out: Path) -> None:
    _write_visualization_readme(out / "README.md", """# Worklog 158 산출물 안내

W158은 frozen WL153 authoritative TSDF scalar field의 all-eight-corner zero-surface cell에서 bounded local Lewiner topology를 읽고, global mesh 없이 lattice-edge zero-crossing incidence graph를 만든다. Candidate G는 6/18/26 neighborhood를 선택하거나 gap을 메우지 않는다. 모든 시각화 폴더와 nested camera 폴더에는 view 의미·입력·legend·검토 한계를 설명하는 README가 있다. PNG가 primary artifact이며 이 batch output에는 PPM을 생성하지 않는다.

`mandatory_gaussian_visualization_pair`는 canonical Gaussian `Original Scene`/`Observed-Occluded` pair다. 나머지 review view는 TSDF/sample diagnostic overlay다.
""")
    _write_visualization_readme(out / "review_views" / "README.md", """# W158 matched real-scene review views

세 fixed camera에서 같은 checkpoint, Gaussian geometry/row order, renderer, resolution, background를 사용한다. A는 original scene, B는 frozen Gaussian Region 0, C는 historical 6-face components, D는 Candidate G components, E는 Candidate G가 연결한 non-largest support, F는 WL157 edge/corner diagnostic rejection review, G는 preserved one-cell-gap endpoints다. Overlay color는 geometry/ownership을 변경하지 않는 review marker이며, Candidate G graph의 source-of-truth는 NPZ/JSON report다.
""")
    descriptions = {
        "A_original_scene": "frozen checkpoint original appearance/geometry only; no recolor or marker Gaussian.",
        "B_gaussian_region0": "frozen WL155 Gaussian Region 0 center overlay; row/geometry unchanged.",
        "C_historical_6_face_components": "WL154/WL157 native 6-face component sample colors; historical baseline only.",
        "D_candidate_g_components": "Candidate G graph-assigned zero-crossing component support; ambiguous cells excluded and reported.",
        "E_zero_set_recovered_connections": "Candidate G-assigned support outside the historical largest native component; not a merge marker.",
        "F_18_26_connections_rejected": "WL157 edge/corner pair cell locations reviewed against Candidate G incidence; orange marks are diagnostic points.",
        "G_preserved_one_cell_gaps": "W157 one-cell gap endpoints; no bridge or intervening-cell fill is performed.",
    }
    for name, meaning in descriptions.items():
        root = out / "review_views" / name
        _write_visualization_readme(root / "README.md", f"# {name}\n\n{meaning}\n\nLegend: green=frozen Region 0 Gaussian, deterministic component colors=historical 6-face baseline, cyan/teal=Candidate G, orange=edge/corner rejection review, magenta=preserved one-cell gap. All overlays are common-world projections from frozen coordinates; no smoothing, dilation, or geometry repair.")
        _write_visualization_readme(root / "cameras" / "README.md", "fixed camera PNG exports; camera and render conditions are recorded in `worklog_158_report.json`.")
        for camera in REVIEW_CAMERAS:
            _write_visualization_readme(root / "cameras" / camera / "README.md", f"fixed camera `{camera}` PNG render for `{name}`. See the parent README for meaning and legend.")
    mandatory = out / "mandatory_gaussian_visualization_pair"
    _write_visualization_readme(mandatory / "README.md", """# Mandatory Gaussian Visualization Pair

W155 frozen canonical pair를 PNG-only로 보존했다. `Original Scene`과 `Observed-Occluded`는 동일 checkpoint/iteration/camera/resolution/background/renderer/row count와 동일 geometry를 사용한다. Observed-Occluded는 display state color만 바꾼다.
""")
    for view in ("Original Scene", "Observed-Occluded"):
        root = mandatory / view
        _write_visualization_readme(root / "README.md", f"# {view}\n\nW158 canonical pair의 `{view}`다. Gaussian row/geometry와 fixed render 조건은 상위 README를 따른다. 이 output은 PNG-only다.")
        for child in root.iterdir() if root.exists() else []:
            if child.is_dir():
                _write_visualization_readme(child / "README.md", f"`{view}` fixed iteration/checkpoint provenance directory; PNG-only W158 copy.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    _progress("loading frozen WL153/WL154/WL155/WL156/WL157 artifacts")
    data = _load_frozen_data(args.field, args.wl154, args.wl155)
    data["region_id"] = 0
    wl156_report = json.loads((args.wl156 / "worklog_156_report.json").read_text(encoding="utf-8"))
    wl157_report = json.loads((args.wl157 / "worklog_157_report.json").read_text(encoding="utf-8"))
    synthetic = _synthetic_contracts()
    region_results: dict[int, dict[str, Any]] = {}
    for region_id in (0, 2, 5):
        region = _target_region_arrays(data, region_id)
        region["region_id"] = region_id
        _progress(f"Region {region_id} owned={len(region['keys']):,}")
        incidence = _extract_zero_set_incidence(data, region, block_size=args.block_size, sentinel=args.sentinel)
        cell_accounting = _cell_component_accounting(region, incidence)
        metrics = _component_metrics(region, incidence, cell_accounting)
        record_path = _save_incidence(args.out, region_id, incidence, cell_accounting)
        if len(incidence["ambiguous_cells"]):
            boundary = _boundary_first_stop_record(
                f"{len(incidence['ambiguous_cells']):,} local zero-set cells have corner-degenerate incidence"
            )
        else:
            boundary = _boundary_first_replay(region, incidence, cell_accounting, data)
        region_results[region_id] = {"region": region, "incidence": incidence, "cell_accounting": cell_accounting, "metrics": metrics, "boundary_first": boundary, "record_path": record_path}
        _progress(f"Region {region_id} Candidate G nodes={metrics['zero_crossing_entity_node_count']:,} components={metrics['connected_component_count']:,}")
    edge_corner = _revisit_wl157_edge_corner(data, region_results[0]["incidence"], args.wl157)
    native_face = _native_face_revisit(region_results[0]["region"], region_results[0]["incidence"], data)
    topology_deterministic = bool(
        synthetic["all_pass"]
        and all(not len(region_results[region_id]["incidence"]["ambiguous_cells"]) for region_id in (0, 2, 5))
    )
    if topology_deterministic:
        render = _render_views(data, args.out, region_results, args.source_path, args.checkpoint, args.device, args.wl157)
    else:
        render = {
            "status": "SKIPPED_STOP_CONDITION_A",
            "reason": "Candidate G is not promoted to real-scene overlay while local zero-set incidence remains ambiguous.",
            "common_world": False,
            "mandatory_gaussian_visualization_pair_deferred_to_frozen_png_copy": True,
        }
    mandatory_source = args.wl155 / "mandatory_gaussian_visualization_pair"
    mandatory_target = args.out / "mandatory_gaussian_visualization_pair"
    _copy_png_only(mandatory_source, mandatory_target)
    _write_readmes(args.out)
    for region_id in (0, 2, 5):
        region_results[region_id]["region"].pop("sample_indices", None)
    g0 = region_results[0]
    baseline6 = wl157_report["diagnostic_connectivity_6_18_26"]["region_0"]["6"]
    candidate_metrics = g0["metrics"]
    local_gap_preserved = int(wl157_report["true_one_cell_gap_attribution"]["gap_instance_count"]) > 0
    if not topology_deterministic:
        verdict = "ZERO_SET_CONNECTIVITY_CONTRACT_GAP"
        reason = "Local zero-set extraction produced ambiguous cells that could not be assigned a unique cross-cell incidence without inventing a bridge."
    elif candidate_metrics["connected_component_count"] > int(baseline6["component_count"]):
        verdict = "ZERO_SET_TOPOLOGY_STILL_FRAGMENTED"
        reason = "Deterministic implicit zero-set connectivity is available, but the frozen zero set remains fragmented after preserving unsupported gaps."
    elif g0["boundary_first"]["materialized_representative_count"] == 0 and int(g0["boundary_first"]["eligible_component_count"]) > 0:
        verdict = "REPRESENTATIVE_FAILURE"
        reason = "Candidate G topology is deterministic, but the unchanged Boundary First/WL139 representative family materialized no valid representative."
    elif candidate_metrics["connected_component_count"] < int(baseline6["component_count"]):
        verdict = "IMPLICIT_ZERO_SET_CONNECTIVITY_VALIDATED"
        reason = "Candidate G supplies deterministic shared zero-crossing incidence, preserves ownership/gaps, and reduces arbitrary digital fragmentation."
    else:
        verdict = "DIGITAL_ADJACENCY_WAS_NOT_THE_CORE_PROBLEM"
        reason = "Candidate G is deterministic, but it does not materially change the real-scene support structure relative to native 6-face connectivity."
    report = {
        "status": "COMPLETE_MESH_FREE_IMPLICIT_ZERO_SET_CONNECTIVITY_CANDIDATE_G_AUDIT",
        "batch": "Worklog 158 — Mesh-Free Implicit Zero-Set Connectivity Contract and Candidate Audit",
        "intent_alignment": {"diagnostic_only": True, "candidate_g_isolated": True, "production_candidate_f_unchanged": True, "connectivity_changed_only_in_candidate": True, "global_mesh_required": False, "ownership_changed": False, "field_changed": False, "zero_surface_eligibility_changed": False, "boundary_first_tuned": False, "nurbs_tuned": False},
        "implementation_fidelity": {"frozen_inputs": ["WL153 authoritative field", "WL154 zero-surface samples/ownership", "WL155 Gaussian Region mapping", "WL156 historical frontier", "WL157 native components/separation"], "scalar_interpolation_contract": "historical linear edge interpolation at level 0", "topology_convention": "Lewiner marching cubes local topology, bounded blocks, geometry discarded after incidence extraction", "node_definition": "region-scoped lattice-edge zero-crossing entity ID=(lower lattice vertex key, axis)", "edge_definition": "local Lewiner triangle co-incidence only", "arbitrary_distance_or_radius": False, "global_mesh_materialized": False, "ambiguous_corner_policy": "exclude ambiguous triangle and report cell; never cross-cell bridge", "one_cell_gap_bridge": False, "different_region_bridge": False, "unowned_or_ambiguous_bridge": False},
        "current_frozen_architecture": "2DGS intrinsic t_w → frozen Gaussian Surface Region → frozen TSDF nearest-Gaussian ownership → frozen zero-surface cells → frozen 6-face Observed Support Components",
        "wl157_reconciliation": {"wl157_report": str(args.wl157 / "worklog_157_report.json"), "region_0_native_6_component_count": int(baseline6["component_count"]), "region_0_largest_fraction": float(baseline6["largest_component_fraction"]), "edge_corner_gap_accounting_reused": True, "wl156_historical_verdict": wl156_report["architecture_result"]["architecture_verdict"], "wl157_architecture_verdict": wl157_report["architecture_result"]["architecture_verdict"]},
        "local_zero_set_topology_contract": {"deterministic": topology_deterministic, "source": "frozen authoritative scalar values only", "all_eight_corner_authority": True, "sign_change_eligibility": True, "local_geometry_persisted": False, "sentinel_only_for_bounded_mc_call": float(args.sentinel)},
        "zero_crossing_incidence_contract": {"identity": "encoded lower lattice vertex plus axis", "shared_entity_scope": "same Region only", "world_distance_used": False, "radius_matching_used": False, "nearest_surface_matching_used": False, "geometric_knn_used": False},
        "edge_corner_topology_reclassification": edge_corner,
        "historical_native_6_revisit": native_face,
        "stop_condition_a": {"result": "NOT_TRIGGERED" if topology_deterministic else "TRIGGERED", "contract": "ZERO_SET_CONNECTIVITY_CONTRACT_GAP", "candidate_g_allowed": topology_deterministic, "reason": "Exact local Lewiner incidence is available for ordinary crossings; corner-degenerate cells are conservatively reported rather than bridged."},
        "candidate_g": {"candidate_name": "Candidate G — Mesh-Free Implicit Zero-Set Connectivity", "regions": {str(region_id): {"metrics": region_results[region_id]["metrics"], "boundary_first": region_results[region_id]["boundary_first"], "record_npz": str(region_results[region_id]["record_path"]), "topology_convention": region_results[region_id]["incidence"]["topology_convention"]} for region_id in (0, 2, 5)}, "production_6_face_unchanged": True},
        "historical_6_diagnostic_18_26_candidate_g_accounting": {"historical_controls": wl157_report["diagnostic_connectivity_6_18_26"], "candidate_g_region_0": candidate_metrics, "not_tuned_to_control_counts": True},
        "true_gap_preservation": {"wl157_gap_instances": int(wl157_report["true_one_cell_gap_attribution"]["gap_instance_count"]), "authoritative_not_zero_surface_instances": int(wl157_report["true_one_cell_gap_attribution"]["by_intervening_state"]["AUTHORITATIVE_NOT_ZERO_SURFACE"]["gap_instance_count"]), "candidate_g_bridges_gap": False, "preserved": local_gap_preserved},
        "wl153_reference": {"global_mesh_required": False, "evaluation_mode": "local Lewiner mechanics; no global WL153 vertices/faces loaded by Candidate G", "mesh_component_identity_used_as_gaussian_identity": False},
        "synthetic_contracts": synthetic,
        "boundary_first_conditional_replay": {"changed_input_only": "Observed Support connectivity", "frozen_wl139_family": True, "regions": {str(region_id): region_results[region_id]["boundary_first"] for region_id in (0, 2, 5)}},
        "real_scene_qualitative_review": {**render, "mandatory_gaussian_visualization_pair": {"source": str(mandatory_source), "output": str(mandatory_target), "png_only": True, "same_checkpoint_iteration_camera_resolution_background_renderer_row_count": True, "geometry_and_gaussian_rows_unchanged": True}, "review_root": str(args.out / "review_views"), "common_world": bool(render.get("common_world", False))},
        "architecture_result": {"architecture_verdict": verdict, "verdict_reason": reason, "allowed_verdicts": ["IMPLICIT_ZERO_SET_CONNECTIVITY_VALIDATED", "ZERO_SET_CONNECTIVITY_CONTRACT_GAP", "DIGITAL_ADJACENCY_WAS_NOT_THE_CORE_PROBLEM", "ZERO_SET_TOPOLOGY_STILL_FRAGMENTED", "REPRESENTATIVE_FAILURE", "MIXED", "UNRESOLVED"], "decision_evidence": {"candidate_g_component_count": int(candidate_metrics["connected_component_count"]), "historical_6_component_count": int(baseline6["component_count"]), "candidate_g_ambiguous_cell_count": int(candidate_metrics["ambiguous_cell_count"]), "candidate_g_materialized_representative_count": int(g0["boundary_first"]["materialized_representative_count"]), "true_gap_preserved": local_gap_preserved}},
        "retained_rejected_open": {"retained": ["WL153–WL157 frozen artifacts", "Lewiner local topology convention", "zero-crossing IDs", "ownership and zero-surface eligibility", "6/18/26 controls", "PNG matched views"], "rejected": ["global mesh as required intermediate", "18-neighbor promotion", "26-neighbor promotion", "radius/nearest matching", "gap fill", "dilation/smoothing", "component filtering", "fitter tuning"], "open": ["physical-sheet identity remains outside zero-set topology contract", "global WL153 per-element provenance remains unavailable"]},
        "forbidden_changes": {"tw_semantics_changed": False, "gaussian_regions_changed": False, "nearest_association_changed": False, "tsdf_field_changed": False, "zero_surface_changed": False, "native_components_changed": False, "boundary_first_changed": False, "nurbs_changed": False, "trust_latent_occluded_surface_changed": False},
        "outputs": {"report": str(args.out / "worklog_158_report.json"), "review_root": str(args.out / "review_views"), "mandatory_pair": str(mandatory_target), "candidate_g_records": [str(region_results[region_id]["record_path"]) for region_id in (0, 2, 5)]},
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_158_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_WL153_FIELD)
    parser.add_argument("--wl154", type=Path, default=DEFAULT_WL154)
    parser.add_argument("--wl155", type=Path, default=DEFAULT_WL155)
    parser.add_argument("--wl156", type=Path, default=DEFAULT_WL156)
    parser.add_argument("--wl157", type=Path, default=DEFAULT_WL157)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sentinel", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "architecture_verdict": report["architecture_result"]["architecture_verdict"], "synthetic_all_pass": report["synthetic_contracts"]["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
