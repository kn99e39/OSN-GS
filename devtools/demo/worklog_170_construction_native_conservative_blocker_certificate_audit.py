from __future__ import annotations

"""Worklog 170: audit construction-native conservative blocker evidence.

This diagnostic keeps W167-W169 frozen.  It recovers the owning cell and all
eight authoritative corner samples for every W168 zero-set first hit, records
the marching-cubes interpolation and ray/cell extent, and audits whether any
stored quantity has a construction-justified one-sided relationship to a
physical blocker.  It does not define a margin or modify blocker behavior.
"""

import argparse
import dataclasses
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for _path in (str(DEVTOOLS), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from devtools.demo import worklog_167_raw_zero_set_ray_blocker_audit as w167  # noqa: E402
from devtools.demo import worklog_168_raw_zero_set_first_hit_positive_occlusion_evidence_audit as w168  # noqa: E402
from devtools.demo import worklog_169_w168_premature_zero_set_blocker_counterexample_attribution as w169  # noqa: E402
from evidence_bounded_tsdf.extraction import ExtractedSurface, cell_corner_offsets  # noqa: E402
from evidence_bounded_tsdf.field import SparseProjectiveTSDF  # noqa: E402


DEFAULT_W168_ROOT = REPO_ROOT / "output/168_raw_zero_set_first_hit_positive_per_view_occlusion_evidence_audit"
DEFAULT_W169_ROOT = REPO_ROOT / "output/169_w168_strict_premature_zero_set_blocker_counterexample_attribution"
DEFAULT_OUT = REPO_ROOT / "output/170_construction_native_conservative_blocker_certificate_audit"

VERDICT_NO_CERTIFICATE = "NO_CONSTRUCTION_NATIVE_BLOCKER_CERTIFICATE"
EDGE_INTERPOLATION = 1
EXACT_LATTICE_CORNER = 0
NOT_A_SINGLE_CELL_EDGE = 2


def _write_json(path: Path, value: Any) -> None:
    w167._write_json(path, value)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    return w167._distribution(np.asarray(values, dtype=np.float64))


def _reconstruct_field(name: str, fixture: dict[str, Any]) -> SparseProjectiveTSDF:
    """Recreate the exact W168 field carrier without changing its extraction."""

    if name != "layered_two_sheet":
        field, _metadata = w167._synthetic_field(fixture["analytic"]["surfaces"][0])
        return field

    import torch

    front, rear = w168._layered_surfaces()
    values = np.arange(-40, 41, dtype=np.int64)
    gx, gy, gz = np.meshgrid(values, values, values, indexing="ij")
    indices = np.column_stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)))
    points = (indices.astype(np.float64) + 0.5) * w167.SYNTHETIC_H
    support = front.support_mask(points) & rear.support_mask(points)
    indices = indices[support]
    points = points[support]
    signed = np.minimum(points[:, 2] - w168.LAYER_FRONT_Z, w168.LAYER_REAR_Z - points[:, 2])
    phi = np.clip(signed / w167.SYNTHETIC_MU, -1.0, 1.0).astype(np.float32)
    order = np.argsort(w167._encode_keys(indices), kind="stable")
    keys = w167._encode_keys(indices[order])
    return SparseProjectiveTSDF(
        keys=torch.as_tensor(keys, dtype=torch.int64),
        value=torch.as_tensor(phi[order], dtype=torch.float32),
        support_count=torch.ones((len(keys),), dtype=torch.int32),
        h=w167.SYNTHETIC_H,
        mu=w167.SYNTHETIC_MU,
    )


def _extraction_reproduction(field: SparseProjectiveTSDF, frozen: ExtractedSurface) -> dict[str, Any]:
    replay = w167._historical_extract(field)
    checks = {
        "vertices_bitwise_equal": np.array_equal(replay.vertices, frozen.vertices),
        "faces_exact": np.array_equal(replay.faces, frozen.faces),
        "vertex_support_count_exact": np.array_equal(replay.vertex_support_count, frozen.vertex_support_count),
        "vertex_field_value_bitwise_equal": np.array_equal(replay.vertex_field_value, frozen.vertex_field_value),
        "h_exact": replay.h == frozen.h,
    }
    checks["all_exact"] = all(checks.values())
    return checks


def _owning_cells(vertices: np.ndarray, faces: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Recover the extractor's owning-cell rule from frozen triangle geometry."""

    lattice_triangles = vertices[faces] / h - 0.5
    cells = np.floor(lattice_triangles.mean(axis=1)).astype(np.int64)
    return cells, lattice_triangles


def _lookup_corners(field: SparseProjectiveTSDF, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    offsets = cell_corner_offsets()
    indices = cells[:, None, :] + offsets[None, :, :]
    keys_np = w167._encode_keys(indices.reshape(-1, 3)).reshape(len(cells), 8)
    keys = torch.as_tensor(keys_np.reshape(-1), dtype=torch.int64, device=field.keys.device)
    value, count, found = field.lookup(keys)
    return (
        indices,
        keys_np,
        value.reshape(len(cells), 8).detach().cpu().numpy(),
        np.column_stack((
            count.reshape(-1).detach().cpu().numpy(),
            found.reshape(-1).detach().cpu().numpy().astype(np.int32),
        )).reshape(len(cells), 8, 2),
    )


def _trilinear(corner_values: np.ndarray, local_points: np.ndarray) -> np.ndarray:
    offsets = cell_corner_offsets()
    result = np.zeros((len(local_points),), dtype=np.float64)
    for corner, (dx, dy, dz) in enumerate(offsets):
        weight = (
            (local_points[:, 0] if dx else 1.0 - local_points[:, 0])
            * (local_points[:, 1] if dy else 1.0 - local_points[:, 1])
            * (local_points[:, 2] if dz else 1.0 - local_points[:, 2])
        )
        result += weight * corner_values[:, corner]
    return result


def _edge_interpolation(
    triangle_lattice: np.ndarray,
    cells: np.ndarray,
    corner_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recover each selected triangle vertex's zero-crossing edge exactly."""

    offsets = cell_corner_offsets()
    offset_to_slot = {tuple(int(v) for v in row): i for i, row in enumerate(offsets)}
    local = triangle_lattice - cells[:, None, :]
    shape = local.shape[:2]
    status = np.full(shape, NOT_A_SINGLE_CELL_EDGE, dtype=np.int8)
    axis = np.full(shape, -1, dtype=np.int8)
    endpoint_phi = np.full((*shape, 2), np.nan, dtype=np.float64)
    geometric_fraction = np.full(shape, np.nan, dtype=np.float64)
    field_fraction = np.full(shape, np.nan, dtype=np.float64)
    coordinate_residual = np.full(shape, np.nan, dtype=np.float64)
    for ray in range(shape[0]):
        for vertex in range(shape[1]):
            xyz = local[ray, vertex]
            lattice_residual = np.abs(xyz - np.rint(xyz))
            if np.all(lattice_residual == 0.0):
                status[ray, vertex] = EXACT_LATTICE_CORNER
                continue
            # The frozen world transform multiplies lattice coordinates by h
            # and later division by h need not restore the two integer edge
            # coordinates bitwise.  Marching cubes guarantees every vertex is
            # on one grid edge, so recover that edge topologically: the axis
            # farthest from its nearest lattice plane is the varying axis.
            # This is an identity reconstruction, not a semantic tolerance.
            varying = int(np.argmax(lattice_residual))
            fixed_axes = [item for item in range(3) if item != varying]
            fixed = np.rint(xyz).astype(np.int64)
            if not all(fixed[item] in (0, 1) for item in fixed_axes):
                continue
            low = fixed.copy()
            high = low.copy()
            low[varying] = 0
            high[varying] = 1
            p0 = float(corner_values[ray, offset_to_slot[tuple(low.tolist())]])
            p1 = float(corner_values[ray, offset_to_slot[tuple(high.tolist())]])
            status[ray, vertex] = EDGE_INTERPOLATION
            axis[ray, vertex] = varying
            endpoint_phi[ray, vertex] = (p0, p1)
            geometric_fraction[ray, vertex] = xyz[varying]
            if p0 != p1:
                alpha = p0 / (p0 - p1)
                field_fraction[ray, vertex] = alpha
                coordinate_residual[ray, vertex] = xyz[varying] - alpha
    return {
        "triangle_vertex_local_cell_coordinate": local,
        "status": status,
        "varying_axis": axis,
        "endpoint_phi": endpoint_phi,
        "geometric_fraction": geometric_fraction,
        "field_zero_fraction": field_fraction,
        "coordinate_residual": coordinate_residual,
    }


def _ray_cell_extent(
    origins: np.ndarray,
    directions: np.ndarray,
    camera_depth_scale: np.ndarray,
    cells: np.ndarray,
    h: float,
) -> dict[str, np.ndarray]:
    lower = (cells.astype(np.float64) + 0.5) * h
    upper = lower + h
    entry = np.full((len(cells),), -np.inf, dtype=np.float64)
    exit_ = np.full((len(cells),), np.inf, dtype=np.float64)
    valid = np.ones((len(cells),), dtype=bool)
    for axis in range(3):
        direction = directions[:, axis]
        stationary = direction == 0.0
        valid &= ~(stationary & ((origins[:, axis] < lower[:, axis]) | (origins[:, axis] > upper[:, axis])))
        safe = np.where(stationary, 1.0, direction)
        a = (lower[:, axis] - origins[:, axis]) / safe
        b = (upper[:, axis] - origins[:, axis]) / safe
        near = np.where(stationary, -np.inf, np.minimum(a, b))
        far = np.where(stationary, np.inf, np.maximum(a, b))
        entry = np.maximum(entry, near)
        exit_ = np.minimum(exit_, far)
    valid &= entry <= exit_
    parameter_span = exit_ - entry
    return {
        "cell_world_min": lower,
        "cell_world_max": upper,
        "ray_intersects_owning_cell": valid,
        "ray_entry_parameter": entry,
        "ray_exit_parameter": exit_,
        "ray_camera_depth_span": parameter_span * camera_depth_scale,
        "ray_world_length_span": parameter_span * np.linalg.norm(directions, axis=1),
        "cell_diagonal_world": np.full((len(cells),), math.sqrt(3.0) * h, dtype=np.float64),
    }


def audit_first_hit_evidence(name: str, fixture: dict[str, Any], field: SparseProjectiveTSDF, out: Path) -> dict[str, Any]:
    hit_mask = fixture["zero_hit"].status == w167.STATUS_HIT
    ray_ids = np.flatnonzero(hit_mask)
    triangle_ids = fixture["zero_hit"].triangle_id[hit_mask]
    face_cells, all_triangle_lattice = _owning_cells(fixture["vertices"], fixture["faces"], field.h)
    cells = face_cells[triangle_ids]
    triangles = fixture["vertices"][fixture["faces"][triangle_ids]]
    triangle_lattice = all_triangle_lattice[triangle_ids]
    corner_indices, corner_keys, corner_values, count_found = _lookup_corners(field, cells)
    corner_counts = count_found[:, :, 0].astype(np.int32)
    corner_found = count_found[:, :, 1].astype(bool)
    cell_min = np.min(corner_values, axis=1)
    cell_max = np.max(corner_values, axis=1)
    sign_changing = (cell_min <= 0.0) & (cell_max >= 0.0)
    hit_xyz = fixture["zero_hit"].world_xyz[hit_mask]
    local_hit = hit_xyz / field.h - 0.5 - cells
    trilinear_at_hit = _trilinear(corner_values.astype(np.float64), local_hit)
    interpolation = _edge_interpolation(triangle_lattice, cells, corner_values)
    extent = _ray_cell_extent(
        fixture["rays"].origins[hit_mask],
        fixture["rays"].directions[hit_mask],
        fixture["rays"].camera_depth_scale[hit_mask],
        cells,
        field.h,
    )
    cell_key = w167._encode_keys(cells)
    interior = fixture["interior"][hit_mask]
    boundary = fixture["boundary"][hit_mask]
    premature = w168._interval_masks(
        fixture["analytic"]["first_depth"], fixture["analytic"]["first_hit"],
        fixture["zero_hit"].depth, fixture["zero_hit"].status,
    )["premature"][hit_mask]
    edge_status_counts = Counter(interpolation["status"].reshape(-1).tolist())
    finite_edge_residual = interpolation["coordinate_residual"][np.isfinite(interpolation["coordinate_residual"])]
    artifact = out / "first_hit_cell_evidence.npz"
    np.savez_compressed(
        artifact,
        ray_index=ray_ids,
        triangle_id=triangle_ids,
        component_id=fixture["zero_hit"].component_id[hit_mask],
        owning_cell_index=cells,
        owning_cell_key=cell_key,
        corner_index=corner_indices,
        corner_key=corner_keys,
        corner_tsdf_value=corner_values,
        corner_support_count=corner_counts,
        corner_has_authority=corner_found,
        cell_min_tsdf=cell_min,
        cell_max_tsdf=cell_max,
        cell_sign_changing=sign_changing,
        first_hit_xyz=hit_xyz,
        first_hit_barycentric=fixture["zero_hit"].barycentric[hit_mask],
        first_hit_local_cell_coordinate=local_hit,
        trilinear_tsdf_at_first_hit=trilinear_at_hit,
        triangle_vertex_world=triangles,
        triangle_vertex_local_cell_coordinate=interpolation["triangle_vertex_local_cell_coordinate"],
        triangle_vertex_interpolation_status=interpolation["status"],
        triangle_vertex_edge_axis=interpolation["varying_axis"],
        triangle_vertex_edge_endpoint_phi=interpolation["endpoint_phi"],
        triangle_vertex_geometric_fraction=interpolation["geometric_fraction"],
        triangle_vertex_field_zero_fraction=interpolation["field_zero_fraction"],
        triangle_vertex_coordinate_residual=interpolation["coordinate_residual"],
        interior=interior,
        silhouette_or_support_boundary=boundary,
        w168_premature=premature,
        **extent,
    )
    report = {
        "fixture": name,
        "zero_set_first_hit_rays": int(len(ray_ids)),
        "unique_first_hit_triangles": int(len(np.unique(triangle_ids))),
        "unique_owning_cells": int(len(np.unique(cell_key))),
        "all_hit_cells_eight_corner_authoritative": bool(np.all(corner_found)),
        "all_hit_cells_sign_changing": bool(np.all(sign_changing)),
        "corner_value": _distribution(corner_values.reshape(-1)),
        "corner_support_count_histogram": {str(k): int(v) for k, v in sorted(Counter(corner_counts.reshape(-1).tolist()).items())},
        "interpolation": {
            "triangle_vertex_count": int(interpolation["status"].size),
            "edge_interpolation_count": int(edge_status_counts[EDGE_INTERPOLATION]),
            "exact_lattice_corner_count": int(edge_status_counts[EXACT_LATTICE_CORNER]),
            "not_single_cell_edge_count": int(edge_status_counts[NOT_A_SINGLE_CELL_EDGE]),
            "edge_coordinate_residual": _distribution(finite_edge_residual),
            "trilinear_tsdf_at_triangle_hit": _distribution(trilinear_at_hit),
            "meaning": "marching-cubes edge interpolation and triangle barycentric hit are representation geometry only",
        },
        "cell_ray_extent": {
            "all_rays_intersect_recovered_owning_cell": bool(np.all(extent["ray_intersects_owning_cell"])),
            "cell_diagonal_world": math.sqrt(3.0) * field.h,
            "cell_diagonal_over_h": math.sqrt(3.0),
            "ray_world_length_inside_cell": _distribution(extent["ray_world_length_span"]),
            "ray_camera_depth_span_inside_cell": _distribution(extent["ray_camera_depth_span"]),
            "one_sided_uncertainty_bound": False,
        },
        "interior_boundary": {
            "interior_hit_count": int(np.count_nonzero(interior)),
            "boundary_hit_count": int(np.count_nonzero(boundary)),
            "other_hit_count": int(np.count_nonzero(~(interior | boundary))),
            "premature_interior": int(np.count_nonzero(premature & interior)),
            "premature_boundary": int(np.count_nonzero(premature & boundary)),
            "premature_other": int(np.count_nonzero(premature & ~(interior | boundary))),
        },
        "projective_depth_observation_provenance": "UNAVAILABLE_W168_FIXTURE_USES_DIRECT_ANALYTIC_SIGNED_DISTANCE_CARRIER",
        "source_camera_support": "UNAVAILABLE_SUPPORT_COUNT_IS_UNIFORM_SYNTHETIC_CARRIER_NOT_CAMERA_IDENTITY",
        "artifact": str(artifact.resolve()),
    }
    _write_json(out / "fixture_report.json", report)
    out.joinpath("README.md").write_text(
        f"# W170 `{name}` Construction Evidence\n\n"
        "이 directory는 W168 raw zero-set first hit마다 historical extractor의 owning-cell rule로 cell identity를 복구하고, 여덟 TSDF corner value/`support_count`, selected triangle의 marching-cubes edge interpolation, hit barycentric coordinate 및 ray가 cell 내부를 지나는 geometric extent를 기록한다. `first_hit_cell_evidence.npz`가 ray-level numeric artifact다.\n\n"
        "W168 fixture는 analytic signed distance를 `SparseProjectiveTSDF` carrier에 직접 넣었으므로 corner `support_count=1`은 source-camera provenance가 아니다. cell size, diagonal, ray chord, `mu`, interpolation residual은 physical surface에 대한 one-sided uncertainty bound로 해석하지 않는다. 이 output은 visualization이나 blocker classifier가 아니다.\n",
        encoding="utf-8",
    )
    return report


def available_evidence_contract() -> dict[str, Any]:
    field_schema = [item.name for item in dataclasses.fields(SparseProjectiveTSDF)]
    extracted_schema = [item.name for item in dataclasses.fields(ExtractedSurface)]
    noninjective_a = np.asarray((-0.5, 0.5), dtype=np.float32)
    noninjective_b = np.asarray((0.0, 0.0), dtype=np.float32)
    return {
        "sparse_field_stored_fields": field_schema,
        "extracted_surface_stored_fields": extracted_schema,
        "per_view_phi_stored": False,
        "source_view_or_camera_id_stored": False,
        "projective_depth_sample_stored": False,
        "per_view_weight_stored": False,
        "fused_value_semantics": "uniform arithmetic mean of authoritative clipped phi_v; support_count stores only the number of contributors",
        "extraction_semantics": "all eight corners authoritative and min(phi)<=0<=max(phi), followed by level-0 marching-cubes interpolation",
        "owning_cell_stored_on_surface": False,
        "owning_cell_recoverable_from_frozen_triangle": True,
        "fusion_noninjectivity_control": {
            "history_a": noninjective_a.tolist(),
            "history_b": noninjective_b.tolist(),
            "same_count": len(noninjective_a) == len(noninjective_b),
            "same_fused_mean": float(noninjective_a.mean()) == float(noninjective_b.mean()),
            "different_per_view_extrema_and_signs": True,
            "consequence": "stored mean and count cannot recover per-view extrema, identity, agreement, or a one-sided envelope",
        },
    }


def one_sided_semantic_audit() -> dict[str, Any]:
    return {
        "hit_cell_identity": {"available": "RECOVERABLE_NOT_STORED", "one_sided_physical_relation": False, "reason": "spatial ownership locates representation geometry only"},
        "eight_corner_tsdf_values": {"available": True, "one_sided_physical_relation": False, "reason": "fused clipped means are not lower/upper envelopes and lose per-view extrema"},
        "eight_corner_support_counts": {"available": True, "one_sided_physical_relation": False, "reason": "counts contain neither view identity nor signed agreement"},
        "marching_cubes_interpolation": {"available": True, "one_sided_physical_relation": False, "reason": "level-0 interpolation is exact for the stored scalar representation, not an enclosure of a physical surface"},
        "projective_depth_observations": {"available_in_fusion_call_only": True, "stored_per_voxel": False, "one_sided_physical_relation": False, "reason": "the renderer median event is camera-relative ordering evidence, not independently certified physical first-hit truth"},
        "source_camera_support": {"available_in_fusion_loop_only": True, "stored_per_voxel": False, "one_sided_physical_relation": False, "reason": "source identity and contributor values are discarded after summation"},
        "voxel_size_h": {"available": True, "one_sided_physical_relation": False, "reason": "sampling scale derived from median footprint is not a worst-case reconstruction error bound"},
        "truncation_mu": {"available": True, "one_sided_physical_relation": False, "reason": "authority-band width and clipping scale do not bound zero-set displacement toward or away from a target camera"},
        "cell_diagonal_and_ray_chord": {"available": True, "one_sided_physical_relation": False, "reason": "geometric extent bounds the cell, not the unknown physical surface or target-ray intersection"},
        "iso_value_zero": {"available": True, "one_sided_physical_relation": False, "reason": "a fused representation level has no physical enclosure semantics"},
        "w169_observed_max_error": {"available": True, "one_sided_physical_relation": False, "reason": "empirical fixture maximum is not a construction theorem and cannot become a margin"},
    }


def derive_certificate_decision(evidence: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    usable = [name for name, record in semantic.items() if bool(record.get("one_sided_physical_relation", False))]
    assert not usable
    assert evidence["fusion_noninjectivity_control"]["same_fused_mean"]
    return {
        "verdict": VERDICT_NO_CERTIFICATE,
        "certificate_defined": False,
        "usable_one_sided_quantities": usable,
        "reason": "the existing store and extracted surface contain no construction-justified one-sided enclosure of the underlying physical target-ray blocker",
        "indistinguishability": "different per-view signed-distance histories collapse to the same stored mean/support_count, while source identity/depth extrema are absent",
        "no_fallback_margin": True,
    }


def _prepare_output(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fixture_root = out / "fixtures"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    for name in ("README.md", "worklog_170_report.json"):
        path = out / name
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _prepare_output(args.out)
    immutable_paths = (
        [args.w168_root / "worklog_168_report.json"]
        + [args.w168_root / "synthetic" / name / "fixture_audit.npz" for name in w168.FIXTURE_NAMES]
        + [args.w169_root / "worklog_169_report.json"]
        + [args.w169_root / "fixtures" / name / "premature_attribution.npz" for name in w168.FIXTURE_NAMES]
    )
    hashes_before = {str(path.resolve()): w167._sha256_file(path) for path in immutable_paths}
    w169_report = json.loads((args.w169_root / "worklog_169_report.json").read_text(encoding="utf-8"))
    if w169_report["7_geometry_vs_numerical_attribution"]["verdict"] != w169.ATTR_MIXED:
        raise RuntimeError("W169 MIXED attribution was not reproduced")

    fixtures_root = args.out / "fixtures"
    fixtures_root.mkdir(parents=True, exist_ok=True)
    fixtures_root.joinpath("README.md").write_text(
        "# W170 Frozen Fixture Construction Evidence\n\n"
        "각 하위 directory는 W168 fixture의 모든 valid raw zero-set first hit을 exact triangle ID에 join한 뒤 owning cell, eight-corner field values/counts, interpolation, ray/cell extent를 저장한다. 이 값은 representation geometry audit이며 physical blocker certificate가 아니다.\n\n"
        "W168/W169 artifact는 read-only이며 실행 전후 SHA-256를 비교한다. source camera provenance는 fixture carrier와 stored `SparseProjectiveTSDF` schema 모두에서 이용할 수 없다. margin, threshold, component filter 또는 geometry correction을 적용하지 않는다.\n",
        encoding="utf-8",
    )
    reproduction: dict[str, Any] = {}
    fixture_evidence: dict[str, Any] = {}
    for name in w168.FIXTURE_NAMES:
        print(f"[worklog 170] auditing construction evidence for {name}", flush=True)
        fixture = w168._build_fixture(name)
        artifact = args.w168_root / "synthetic" / name / "fixture_audit.npz"
        w168_check = w169.reproduce_w168_fixture(fixture, artifact)
        if not w168_check["all_exact"]:
            raise RuntimeError(f"W168 reproduction failed for {name}")
        field = _reconstruct_field(name, fixture)
        extraction_check = _extraction_reproduction(field, fixture["extracted"])
        if not extraction_check["all_exact"]:
            raise RuntimeError(f"historical extraction reproduction failed for {name}")
        reproduction[name] = {"w168_fixture": w168_check, "field_to_extraction": extraction_check}
        fixture_out = fixtures_root / name
        fixture_out.mkdir(parents=True, exist_ok=True)
        fixture_evidence[name] = audit_first_hit_evidence(name, fixture, field, fixture_out)

    hashes_after = {str(path.resolve()): w167._sha256_file(path) for path in immutable_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("W168/W169 artifact changed during read-only W170 audit")
    evidence = available_evidence_contract()
    semantic = one_sided_semantic_audit()
    decision = derive_certificate_decision(evidence, semantic)
    total_hits = sum(item["zero_set_first_hit_rays"] for item in fixture_evidence.values())
    total_interior = sum(item["interior_boundary"]["interior_hit_count"] for item in fixture_evidence.values())
    total_boundary = sum(item["interior_boundary"]["boundary_hit_count"] for item in fixture_evidence.values())
    report = {
        "status": "COMPLETE_WL170_CONSTRUCTION_NATIVE_CONSERVATIVE_BLOCKER_CERTIFICATE_AUDIT",
        "worklog": 170,
        "1_intent_alignment": {
            "status": "PASS",
            "question": "Can the existing historical SDF/TSDF construction provide a non-tuned conservative per-view blocker certificate stronger than BEHIND_ZEROSET?",
            "diagnostic_only": True,
            "epsilon_or_tolerance": None,
            "construction_modified": False,
            "blocker_semantics_modified": False,
        },
        "2_available_construction_native_evidence": {
            "schema": evidence,
            "fixture_first_hit_evidence": fixture_evidence,
            "reproduction": reproduction,
            "immutable_sha256_before": hashes_before,
            "immutable_sha256_after": hashes_after,
            "hashes_unchanged": hashes_before == hashes_after,
        },
        "3_one_sided_semantic_justification": semantic,
        "4_certificate_definition": {
            **decision,
            "mathematical_predicate": None,
            "note": "no predicate is defined because no available quantity has a construction-native one-sided physical meaning",
        },
        "5_synthetic_soundness": {
            "status": "NOT_APPLICABLE_NO_CERTIFICATE",
            "certificate_evaluated": False,
            "implication": "CERTIFIED_BLOCKED(v,x) => analytic GT_BLOCKED(v,x)",
            "false_positives": None,
            "reason": "testing an invented cell-size, mu, empirical-error, or source-median margin would violate the audit contract",
        },
        "6_coverage_abstention": {
            "status": "NOT_APPLICABLE_NO_CERTIFICATE",
            "certification_coverage": None,
            "false_negatives": None,
            "abstentions": None,
            "audited_representation_first_hits": total_hits,
            "audited_interior_first_hits": total_interior,
            "audited_boundary_first_hits": total_boundary,
            "per_fixture_representation_accounting": {name: value["interior_boundary"] for name, value in fixture_evidence.items()},
            "reason": "BEHIND_ZEROSET remains a geometry fact; it is not relabeled as a physical-certificate abstention classifier",
        },
        "7_architecture_result": {
            **decision,
            "behind_zeroset_status": "PRESERVED_REPRESENTATION_LEVEL_GEOMETRIC_FACT_ONLY",
            "direct_observation_unavailable_equivalence": False,
            "answer": "No. The existing historical SDF/TSDF construction does not retain or justify a non-tuned one-sided physical blocker bound stronger than BEHIND_ZEROSET.",
            "w161_pause_preserved": True,
        },
        "8_retained_rejected_open": {
            "retained": ["W167 raw zero-set blocker", "W168 strict failure", "W169 MIXED attribution", "historical h/mu/iso-value/TSDF construction", "all components", "strict BEHIND_ZEROSET relation", "W161 pause"],
            "rejected": ["c*h margin", "W169 maximum-error margin", "threshold sweep", "zero-set offset/erosion", "component filtering or trusted subset", "TSDF modification", "NURBS", "global OCCLUDED/UNRESOLVED", "Eligibility or continuation"],
            "open": ["a future blocker certificate would require new semantically justified evidence or construction output and therefore a separately authorized architecture change"],
        },
        "outputs": {
            "root": str(args.out.resolve()),
            "fixture_artifacts": {name: value["artifact"] for name, value in fixture_evidence.items()},
            "visualizations_generated": False,
            "png_count": 0,
            "ppm_count": 0,
            "w153_replay_cache_copied_to_temp": False,
        },
        "runtime_seconds": time.time() - started,
    }
    _write_json(args.out / "worklog_170_report.json", report)
    args.out.joinpath("README.md").write_text(
        "# Worklog 170 — Construction-Native Conservative Blocker Certificate Audit\n\n"
        "이 output은 W168 zero-set first hit의 owning cell, eight-corner TSDF values/weights, marching-cubes interpolation 및 cell/ray extent를 전수 기록하고, historical construction이 physical blocker에 대한 non-tuned one-sided bound를 보존하는지 감사한다. fixture별 `first_hit_cell_evidence.npz`가 numeric artifact다.\n\n"
        f"Architecture verdict: `{decision['verdict']}`. `BEHIND_ZEROSET`은 representation-level geometry fact로만 유지한다. field mean/count와 mesh interpolation에는 physical one-sided enclosure semantics가 없고 source-view identity/depth extrema도 저장되지 않으므로 `CERTIFIED_BLOCKED` predicate를 정의하지 않았다.\n\n"
        "새 visualization, epsilon, margin, threshold, geometry correction, component filtering, global state 또는 continuation은 없다. W168/W169 artifact hash는 실행 전후 동일하게 검증한다.\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w168-root", type=Path, default=DEFAULT_W168_ROOT)
    parser.add_argument("--w169-root", type=Path, default=DEFAULT_W169_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "verdict": report["7_architecture_result"]["verdict"], "answer": report["7_architecture_result"]["answer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
