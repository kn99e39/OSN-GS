from __future__ import annotations

"""Worklog 167: frozen raw zero-set mesh as a camera-ray blocker.

This file is diagnostic-only.  It deliberately does not import Candidate-B,
NURBS, Boundary First, continuation, Eligibility, or any production
visibility/occlusion classifier.  The real-scene input is the byte-preserved
W166 NPZ export of W153's historical typed ``ExtractedSurface``.

The central primitive is a deterministic, two-sided Moller--Trumbore
ray/triangle test.  A small screen-tile broad phase is used for the real
scene; it changes no triangles or coordinates and every broad-phase candidate
is checked by the same exact primitive.  Synthetic controls use the same
primitive directly and construct their zero-set through the historical sparse
TSDF + all-eight-corner Lewiner extraction path.
"""

import argparse
import colorsys
import hashlib
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS = REPO_ROOT / "scripts" / "devtools"
for _path in (str(DEVTOOLS), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_SOURCE = REPO_ROOT / "output/166_historical_sdf_zero_surface_mesh_export/historical_sdf_zero_surface_raw.npz"
DEFAULT_W153_SOURCE = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache/typed_extracted_surface.npz"
DEFAULT_W153_ACCOUNTING = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/native_topology_accounting.json"
DEFAULT_W153_RUNTIME = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache/replay_input_runtime.json"
DEFAULT_DEPTH_CACHE = REPO_ROOT / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit/replay_cache/renderer_median_depth_maps.npz"
DEFAULT_OUT = REPO_ROOT / "output/167_raw_zero_set_ray_blocker_audit"
DEFAULT_DATASET = REPO_ROOT / "DATASET"

HISTORICAL_H = 0.012105485424399376
HISTORICAL_MU = 0.03631645627319813
SYNTHETIC_H = 0.05
SYNTHETIC_MU = 3.0 * SYNTHETIC_H
SYNTHETIC_FOV = 0.7
SYNTHETIC_IMAGE = 64
SYNTHETIC_CAMERA_ORIGIN = np.asarray((0.0, 0.0, -4.0), dtype=np.float64)
SYNTHETIC_PLANE_HALF = 1.15
SYNTHETIC_SPHERE_CENTER = np.asarray((0.0, 0.0, 0.35), dtype=np.float64)
SYNTHETIC_SPHERE_RADIUS = 1.0
SYNTHETIC_QUERY_OFFSET = 0.50
REAL_QUERY_OFFSET = 8.0 * HISTORICAL_H
REAL_SAMPLE_STRIDE = 4
SCREEN_TILE_SIZE = 16

# This is a numerical convention of the primitive, not a geometry or
# acceptance threshold.  Parallelism is decided by exact zero.  No sweep is
# allowed by the W167 contract.
RAY_DETERMINANT_EPS = 0.0

STATUS_HIT = "HIT"
STATUS_NO_HIT = "NO_HIT"
STATUS_AMBIGUOUS = "AMBIGUOUS"

REL_ZEROSET_FIRST_SURFACE = "ZEROSET_FIRST_SURFACE"
REL_BEHIND_ZEROSET_SURFACE = "BEHIND_ZEROSET_SURFACE"
REL_IN_FRONT_OF_ZEROSET_SURFACE = "IN_FRONT_OF_ZEROSET_SURFACE"
REL_NO_DECISION = "NO_DECISION"

FRAGMENT_RGB = (1.0, 0.48, 0.05)
HIT_RGB = (0.10, 0.85, 0.35)
RELATION_RGB = {
    "FIRST_HIT_SURFACE": HIT_RGB,
    REL_ZEROSET_FIRST_SURFACE: (0.98, 0.78, 0.12),
    REL_BEHIND_ZEROSET_SURFACE: (0.92, 0.18, 0.18),
    REL_IN_FRONT_OF_ZEROSET_SURFACE: (0.16, 0.75, 0.95),
    REL_NO_DECISION: (0.60, 0.60, 0.62),
    "FRAGMENT_FIRST_HIT": FRAGMENT_RGB,
    "CAMERA_POSITION": (0.95, 0.95, 0.95),
}
CAMERA_RGB = (0.95, 0.95, 0.95)
MESH_RGB = (0.30, 0.52, 0.75)

REVIEW_CAMERAS = ("DSC07960.JPG", "DSC08003.JPG", "DSC08043.JPG")

# These are the frozen W160--W165 image-space polygons.  The contact control
# is the W162/W164 union of the two existing regions, not a new ROI.
REVIEW_POLYGONS: dict[str, dict[str, tuple[tuple[float, float], ...]]] = {
    "tabletop": {
        "DSC08043.JPG": ((200, 215), (235, 213), (236, 235), (201, 236)),
        "DSC07960.JPG": ((383, 188), (415, 189), (413, 200), (385, 199)),
        "DSC08003.JPG": ((240, 158), (280, 158), (279, 171), (242, 170)),
    },
    "table_side_lower_geometry": {
        "DSC08043.JPG": ((220, 264), (385, 260), (383, 280), (222, 283)),
        "DSC07960.JPG": ((215, 257), (375, 253), (373, 275), (217, 278)),
        "DSC08003.JPG": ((205, 259), (380, 256), (378, 277), (207, 280)),
    },
    "vase_foreground_structure": {
        "DSC08043.JPG": ((385, 185), (458, 193), (445, 226), (376, 218)),
        "DSC07960.JPG": ((375, 184), (447, 193), (440, 226), (369, 217)),
        "DSC08003.JPG": ((225, 184), (282, 190), (278, 224), (221, 218)),
    },
}
CONTACT_REGIONS = ("tabletop", "vase_foreground_structure")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _npz_descriptors(path: Path) -> dict[str, dict[str, Any]]:
    """Read only NPY headers so synthetic-only validation does not load W166."""

    from numpy.lib import format as npy_format

    result: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"):
                continue
            with archive.open(member, "r") as handle:
                version = npy_format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = npy_format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = npy_format.read_array_header_2_0(handle)
                elif version == (3, 0):
                    shape, fortran, dtype = npy_format.read_array_header_3_0(handle)
                else:
                    raise ValueError(f"unsupported NPY header version {version}")
            if fortran:
                raise ValueError(f"Fortran-order source array is not supported: {member.filename}")
            name = Path(member.filename).stem
            result[name] = {"shape": [int(value) for value in shape], "dtype": str(dtype), "nbytes": int(np.prod(shape, dtype=np.int64) * dtype.itemsize)}
    return result


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = np.sort(values)
    return {
        "count": int(ordered.size),
        "min": float(ordered[0]),
        "median": float(np.percentile(ordered, 50.0)),
        "mean": float(ordered.mean()),
        "p95": float(np.percentile(ordered, 95.0)),
        "max": float(ordered[-1]),
    }


@dataclass(frozen=True)
class RayBundle:
    origins: np.ndarray
    directions: np.ndarray
    camera_depth_scale: np.ndarray
    pixel_row: np.ndarray | None = None
    pixel_col: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(self.origins.shape[0])
        if self.origins.shape != (n, 3) or self.directions.shape != (n, 3):
            raise ValueError("rays must be (N, 3)")
        if self.camera_depth_scale.shape != (n,):
            raise ValueError("camera_depth_scale must be (N,)")


@dataclass(frozen=True)
class FirstHitResult:
    status: np.ndarray
    depth: np.ndarray
    world_xyz: np.ndarray
    triangle_id: np.ndarray
    component_id: np.ndarray
    barycentric: np.ndarray
    valid_positive_depth_intersections: np.ndarray
    first_hit_tie_count: np.ndarray
    coplanar_ambiguity_count: np.ndarray


@dataclass(frozen=True)
class AnalyticSurface:
    name: str
    kind: str
    center: np.ndarray
    tangent_u: np.ndarray | None = None
    tangent_v: np.ndarray | None = None
    normal: np.ndarray | None = None
    half_u: float | None = None
    half_v: float | None = None
    radius: float | None = None

    def intersect(self, rays: RayBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        direction = rays.directions
        origin = rays.origins
        if self.kind == "plane_rectangle":
            assert self.normal is not None and self.tangent_u is not None and self.tangent_v is not None
            assert self.half_u is not None and self.half_v is not None
            denom = direction @ self.normal
            numerator = float((self.center - origin[0]) @ self.normal)
            safe = np.where(denom != 0.0, denom, 1.0)
            depth = numerator / safe
            point = origin + depth[:, None] * direction
            relative = point - self.center
            local_u = relative @ self.tangent_u
            local_v = relative @ self.tangent_v
            hit = (
                (denom != 0.0)
                & (depth > 0.0)
                & (np.abs(local_u) <= self.half_u)
                & (np.abs(local_v) <= self.half_v)
            )
            return np.where(hit, depth, np.nan), hit, point
        if self.kind == "sphere":
            assert self.radius is not None
            offset = origin - self.center
            a = np.sum(direction * direction, axis=1)
            b = 2.0 * np.sum(direction * offset, axis=1)
            c = float(np.sum(offset[0] * offset[0])) - self.radius * self.radius
            discriminant = b * b - 4.0 * a * c
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            safe_a = np.where(a != 0.0, a, 1.0)
            near = (-b - root) / (2.0 * safe_a)
            far = (-b + root) / (2.0 * safe_a)
            depth = np.where((near > 0.0) & valid, near, np.where((far > 0.0) & valid, far, np.nan))
            point = origin + np.where(np.isfinite(depth), depth, 0.0)[:, None] * direction
            return depth, np.isfinite(depth), point
        raise ValueError(self.kind)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        if self.kind == "plane_rectangle":
            assert self.normal is not None
            return (np.asarray(points) - self.center) @ self.normal
        if self.kind == "sphere":
            assert self.radius is not None
            return np.linalg.norm(np.asarray(points) - self.center, axis=1) - self.radius
        raise ValueError(self.kind)

    def support_mask(self, points: np.ndarray) -> np.ndarray:
        if self.kind == "plane_rectangle":
            assert self.tangent_u is not None and self.tangent_v is not None
            assert self.half_u is not None and self.half_v is not None
            relative = np.asarray(points) - self.center
            return (np.abs(relative @ self.tangent_u) <= self.half_u) & (np.abs(relative @ self.tangent_v) <= self.half_v)
        return np.ones((len(points),), dtype=bool)


def _synthetic_rays(image: int = SYNTHETIC_IMAGE) -> RayBundle:
    rows, cols = np.indices((image, image), dtype=np.float64)
    ndc_x = (2.0 * cols.reshape(-1) + 1.0) / image - 1.0
    ndc_y = (2.0 * rows.reshape(-1) + 1.0) / image - 1.0
    directions = np.column_stack((
        ndc_x * math.tan(SYNTHETIC_FOV * 0.5),
        ndc_y * math.tan(SYNTHETIC_FOV * 0.5),
        np.ones_like(ndc_x),
    ))
    origins = np.repeat(SYNTHETIC_CAMERA_ORIGIN[None, :], image * image, axis=0)
    return RayBundle(origins, directions, np.ones((image * image,), dtype=np.float64), rows.reshape(-1).astype(np.int64), cols.reshape(-1).astype(np.int64))


def _plane_surface(name: str, oblique: bool = False) -> AnalyticSurface:
    if oblique:
        angle = math.radians(28.0)
        tangent_u = np.asarray((math.cos(angle), 0.0, -math.sin(angle)), dtype=np.float64)
    else:
        tangent_u = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    tangent_v = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    normal = np.cross(tangent_u, tangent_v)
    return AnalyticSurface(name, "plane_rectangle", np.zeros(3, dtype=np.float64), tangent_u, tangent_v, normal, SYNTHETIC_PLANE_HALF, SYNTHETIC_PLANE_HALF)


def _sphere_surface() -> AnalyticSurface:
    return AnalyticSurface("curved_sphere", "sphere", SYNTHETIC_SPHERE_CENTER.copy(), radius=SYNTHETIC_SPHERE_RADIUS)


def _encode_keys(index: np.ndarray) -> np.ndarray:
    bound = 1 << 19
    span = bound << 1
    return ((index[:, 0] + bound) * span + (index[:, 1] + bound)) * span + (index[:, 2] + bound)


def _synthetic_field(surface: AnalyticSurface) -> tuple[Any, dict[str, Any]]:
    """Build a fixed small sparse field using the historical TSDF type."""

    import torch
    from evidence_bounded_tsdf.field import SparseProjectiveTSDF

    values = np.arange(-40, 41, dtype=np.int64)
    gx, gy, gz = np.meshgrid(values, values, values, indexing="ij")
    indices = np.column_stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)))
    points = (indices.astype(np.float64) + 0.5) * SYNTHETIC_H
    if surface.kind == "plane_rectangle":
        support = surface.support_mask(points)
        indices = indices[support]
        points = points[support]
        signed = surface.signed_distance(points)
    else:
        signed = surface.signed_distance(points)
    phi = np.clip(signed / SYNTHETIC_MU, -1.0, 1.0).astype(np.float32)
    order = np.argsort(_encode_keys(indices), kind="stable")
    keys = _encode_keys(indices[order])
    field = SparseProjectiveTSDF(
        keys=torch.as_tensor(keys, dtype=torch.int64),
        value=torch.as_tensor(phi[order], dtype=torch.float32),
        support_count=torch.ones((len(keys),), dtype=torch.int32),
        h=SYNTHETIC_H,
        mu=SYNTHETIC_MU,
    )
    return field, {
        "h": SYNTHETIC_H,
        "mu": SYNTHETIC_MU,
        "grid_index_min": values.min().item(),
        "grid_index_max": values.max().item(),
        "authoritative_voxels": int(len(keys)),
        "unknown_outside_plane_support": surface.kind == "plane_rectangle",
        "construction": "SparseProjectiveTSDF with uniform support_count=1 and normalized clipped analytic signed distance; historical extraction below",
    }


def _historical_extract(field: Any) -> Any:
    from evidence_bounded_tsdf.extraction import extract_zero_level_set

    return extract_zero_level_set(field, block=16, batch_blocks=64, sentinel=2.0)


def _component_labels(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic scipy vertex-component labels and component sizes."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        labels = np.arange(vertex_count, dtype=np.int64)
        return labels, labels.copy(), np.zeros((vertex_count,), dtype=np.int64)
    source = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    target = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    graph = coo_matrix((np.ones((len(source),), dtype=np.int8), (source, target)), shape=(vertex_count, vertex_count)).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    face_labels = labels[faces[:, 0]]
    vertex_sizes = np.bincount(labels, minlength=component_count)
    face_sizes = np.bincount(face_labels, minlength=component_count)
    return labels, face_labels, np.column_stack((vertex_sizes, face_sizes))


def _ray_triangle_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    triangles: np.ndarray,
    triangle_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized two-sided Moller--Trumbore candidates.

    Outputs are per ray: valid-hit count, best t, best face, barycentric, and
    exact coplanar ambiguity count.  ``t > 0`` is strict and no barycentric
    tolerance is applied.  Degenerate triangles are ignored.
    """

    n_rays = len(origins)
    count = np.zeros((n_rays,), dtype=np.int64)
    best_t = np.full((n_rays,), np.inf, dtype=np.float64)
    best_face = np.full((n_rays,), -1, dtype=np.int64)
    best_bary = np.full((n_rays, 3), np.nan, dtype=np.float64)
    coplanar = np.zeros((n_rays,), dtype=np.int64)
    if not len(triangles) or not n_rays:
        return count, best_t, best_face, best_bary, coplanar

    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    cross_edges = np.cross(edge1, edge2)
    nondegenerate = np.any(cross_edges != 0.0, axis=1)
    if not np.any(nondegenerate):
        return count, best_t, best_face, best_bary, coplanar
    v0 = v0[nondegenerate]
    edge1 = edge1[nondegenerate]
    edge2 = edge2[nondegenerate]
    cross_edges = cross_edges[nondegenerate]
    triangle_ids = np.asarray(triangle_ids, dtype=np.int64)[nondegenerate]

    # [R, T, 3] is bounded by the caller's chunks, not by the full mesh.
    pvec = np.cross(directions[:, None, :], edge2[None, :, :])
    det = np.sum(edge1[None, :, :] * pvec, axis=2)
    inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=det != RAY_DETERMINANT_EPS)
    tvec = origins[:, None, :] - v0[None, :, :]
    u = np.sum(tvec * pvec, axis=2) * inv_det
    qvec = np.cross(tvec, edge1[None, :, :])
    v = np.sum(directions[:, None, :] * qvec, axis=2) * inv_det
    t = np.sum(edge2[None, :, :] * qvec, axis=2) * inv_det
    valid = (
        (det != RAY_DETERMINANT_EPS)
        & (u >= 0.0)
        & (v >= 0.0)
        & ((u + v) <= 1.0)
        & (t > 0.0)
    )
    count += valid.sum(axis=1, dtype=np.int64)

    # A ray lying in a non-degenerate triangle plane has no defensible unique
    # first-hit interpretation. Exact equality is intentional; no epsilon is
    # used to turn near-coplanar cases into ambiguity.
    normal = cross_edges
    plane_offset = np.sum((origins[:, None, :] - v0[None, :, :]) * normal[None, :, :], axis=2)
    coplanar += np.count_nonzero((det == RAY_DETERMINANT_EPS) & (plane_offset == 0.0), axis=1)

    for ray_index in range(n_rays):
        valid_ids = np.flatnonzero(valid[ray_index])
        if not len(valid_ids):
            continue
        values = t[ray_index, valid_ids]
        faces_here = triangle_ids[valid_ids]
        order = np.lexsort((faces_here, values))
        chosen = int(valid_ids[order[0]])
        best_t[ray_index] = float(t[ray_index, chosen])
        best_face[ray_index] = int(triangle_ids[chosen])
        best_bary[ray_index] = (1.0 - u[ray_index, chosen] - v[ray_index, chosen], u[ray_index, chosen], v[ray_index, chosen])
    return count, best_t, best_face, best_bary, coplanar


def intersect_first_hit_bruteforce(
    rays: RayBundle,
    vertices: np.ndarray,
    faces: np.ndarray,
    component_by_face: np.ndarray | None = None,
    triangle_chunk: int = 20_000,
    ray_chunk: int = 32,
) -> FirstHitResult:
    """Exact deterministic first-hit query over a small/controlled mesh."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(rays.origins)
    count = np.zeros((n,), dtype=np.int64)
    best_t = np.full((n,), np.inf, dtype=np.float64)
    best_face = np.full((n,), -1, dtype=np.int64)
    best_bary = np.full((n, 3), np.nan, dtype=np.float64)
    coplanar = np.zeros((n,), dtype=np.int64)
    for ray_start in range(0, n, ray_chunk):
        ray_stop = min(ray_start + ray_chunk, n)
        for face_start in range(0, len(faces), triangle_chunk):
            face_stop = min(face_start + triangle_chunk, len(faces))
            tri_ids = np.arange(face_start, face_stop, dtype=np.int64)
            local = faces[face_start:face_stop]
            batch = _ray_triangle_batch(rays.origins[ray_start:ray_stop], rays.directions[ray_start:ray_stop], vertices[local], tri_ids)
            count[ray_start:ray_stop] += batch[0]
            coplanar[ray_start:ray_stop] += batch[4]
            for local_ray in range(ray_stop - ray_start):
                candidate_face = int(batch[2][local_ray])
                candidate_t = float(batch[1][local_ray])
                global_ray = ray_start + local_ray
                if candidate_face < 0:
                    continue
                if candidate_t < best_t[global_ray] or (candidate_t == best_t[global_ray] and candidate_face < best_face[global_ray]):
                    best_t[global_ray] = candidate_t
                    best_face[global_ray] = candidate_face
                    best_bary[global_ray] = batch[3][local_ray]
    status = np.full((n,), STATUS_NO_HIT, dtype=object)
    status[coplanar > 0] = STATUS_AMBIGUOUS
    status[(best_face >= 0) & (coplanar == 0)] = STATUS_HIT
    depth = best_t * rays.camera_depth_scale
    depth[best_face < 0] = np.nan
    safe_t = np.where(best_face >= 0, best_t, 0.0)
    xyz = rays.origins + safe_t[:, None] * rays.directions
    xyz[best_face < 0] = np.nan
    component = np.full((n,), -1, dtype=np.int64)
    if component_by_face is not None:
        mask = best_face >= 0
        component[mask] = np.asarray(component_by_face, dtype=np.int64)[best_face[mask]]
    tie_count = np.zeros((n,), dtype=np.int64)
    # Count exact first-depth ties by a second deterministic scan only for the
    # small controls. The real screen path tracks this in its tile batches.
    for ray_index in range(n):
        if best_face[ray_index] < 0:
            continue
        point_count = 0
        for face_start in range(0, len(faces), triangle_chunk):
            face_stop = min(face_start + triangle_chunk, len(faces))
            tri_ids = np.arange(face_start, face_stop, dtype=np.int64)
            local = faces[face_start:face_stop]
            batch = _ray_triangle_batch(rays.origins[ray_index:ray_index + 1], rays.directions[ray_index:ray_index + 1], vertices[local], tri_ids)
            if batch[0][0] and np.isfinite(batch[1][0]) and batch[1][0] == best_t[ray_index]:
                point_count += 1
        tie_count[ray_index] = point_count
    return FirstHitResult(status, depth, xyz, best_face, component, best_bary, count, tie_count, coplanar)


class ScreenTileIndex:
    """Deterministic projected AABB broad phase for a fixed camera/ROI."""

    def __init__(self, tile_to_faces: dict[int, np.ndarray], tiles_x: int, tiles_y: int, stats: dict[str, Any]):
        self.tile_to_faces = tile_to_faces
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.stats = stats

    @classmethod
    def build(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        camera: Any,
        roi_bounds: tuple[int, int, int, int],
        tile_size: int = SCREEN_TILE_SIZE,
        face_chunk: int = 500_000,
    ) -> "ScreenTileIndex":
        width, height = int(camera.image_width), int(camera.image_height)
        x_min, y_min, x_max, y_max = roi_bounds
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        tile_parts: dict[int, list[np.ndarray]] = {}
        candidate_faces = 0
        assignment_count = 0
        view = camera.world_view_transform.detach().cpu().numpy().astype(np.float64)
        projection = camera.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        linear = view[:3, :3]
        translation = view[3, :3]
        for start in range(0, len(faces), face_chunk):
            stop = min(start + face_chunk, len(faces))
            local_faces = np.asarray(faces[start:stop], dtype=np.int64)
            triangles = np.asarray(vertices[local_faces], dtype=np.float64)
            camera_xyz = triangles @ linear + translation
            # ``full_proj_transform`` already contains the world-view matrix
            # under the repository's row-vector convention.  Applying
            # ``world_view`` here as well would move every projected AABB a
            # second time and silently remove the true candidate triangles.
            homogeneous = np.concatenate((triangles, np.ones((len(triangles), 3, 1), dtype=np.float64)), axis=2)
            clip = homogeneous @ projection
            w = clip[..., 3]
            safe_w = np.where(w != 0.0, w, 1.0)
            px = ((clip[..., 0] / safe_w + 1.0) * width - 1.0) * 0.5
            py = ((clip[..., 1] / safe_w + 1.0) * height - 1.0) * 0.5
            finite = np.all(np.isfinite(px) & np.isfinite(py), axis=1)
            in_front = np.any(camera_xyz[..., 2] > 0.0, axis=1)
            tri_x_min = np.min(px, axis=1)
            tri_x_max = np.max(px, axis=1)
            tri_y_min = np.min(py, axis=1)
            tri_y_max = np.max(py, axis=1)
            candidate = finite & in_front & (tri_x_max >= x_min) & (tri_x_min <= x_max) & (tri_y_max >= y_min) & (tri_y_min <= y_max)
            if not np.any(candidate):
                continue
            ids = np.arange(start, stop, dtype=np.int64)[candidate]
            tx0 = np.floor(np.maximum(tri_x_min[candidate], x_min) / tile_size).astype(np.int64).clip(0, tiles_x - 1)
            tx1 = np.floor(np.minimum(tri_x_max[candidate], x_max) / tile_size).astype(np.int64).clip(0, tiles_x - 1)
            ty0 = np.floor(np.maximum(tri_y_min[candidate], y_min) / tile_size).astype(np.int64).clip(0, tiles_y - 1)
            ty1 = np.floor(np.minimum(tri_y_max[candidate], y_max) / tile_size).astype(np.int64).clip(0, tiles_y - 1)
            widths = tx1 - tx0 + 1
            heights = ty1 - ty0 + 1
            repeat_count = widths * heights
            total = int(repeat_count.sum())
            if total == 0:
                continue
            base = np.repeat(np.cumsum(repeat_count) - repeat_count, repeat_count)
            offset = np.arange(total, dtype=np.int64) - base
            width_rep = np.repeat(widths, repeat_count)
            ty_rep = np.repeat(ty0, repeat_count) + offset // width_rep
            tx_rep = np.repeat(tx0, repeat_count) + offset % width_rep
            face_rep = np.repeat(ids, repeat_count)
            tile_ids = ty_rep * tiles_x + tx_rep
            order = np.argsort(tile_ids, kind="stable")
            tile_ids = tile_ids[order]
            face_rep = face_rep[order]
            unique, starts = np.unique(tile_ids, return_index=True)
            ends = np.concatenate((starts[1:], np.asarray([len(tile_ids)])))
            for tile, begin, end in zip(unique.tolist(), starts.tolist(), ends.tolist()):
                tile_parts.setdefault(int(tile), []).append(face_rep[begin:end])
            candidate_faces += len(ids)
            assignment_count += total
        tile_to_faces = {tile: np.concatenate(parts) for tile, parts in tile_parts.items()}
        return cls(tile_to_faces, tiles_x, tiles_y, {
            "roi_bounds_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "tile_size_pixels": tile_size,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "candidate_face_rows": int(candidate_faces),
            "tile_assignment_rows": int(assignment_count),
            "tiles_with_candidates": len(tile_to_faces),
            "broad_phase": "projected triangle AABB only; every candidate receives exact ray-triangle test",
        })


def intersect_first_hit_screen_index(
    rays: RayBundle,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Any,
    index: ScreenTileIndex,
    component_by_face: np.ndarray | None = None,
    ray_chunk: int = 16,
    triangle_chunk: int = 20_000,
) -> FirstHitResult:
    n = len(rays.origins)
    count = np.zeros((n,), dtype=np.int64)
    best_t = np.full((n,), np.inf, dtype=np.float64)
    best_face = np.full((n,), -1, dtype=np.int64)
    best_bary = np.full((n, 3), np.nan, dtype=np.float64)
    coplanar = np.zeros((n,), dtype=np.int64)
    tie_count = np.zeros((n,), dtype=np.int64)
    if rays.pixel_row is None or rays.pixel_col is None:
        raise ValueError("screen index queries require pixel rows and columns")
    width = int(camera.image_width)
    for ray_start in range(0, n, ray_chunk):
        ray_stop = min(ray_start + ray_chunk, n)
        local_rows = rays.pixel_row[ray_start:ray_stop]
        local_cols = rays.pixel_col[ray_start:ray_stop]
        tile_ids = (local_rows // SCREEN_TILE_SIZE) * index.tiles_x + (local_cols // SCREEN_TILE_SIZE)
        for tile in np.unique(tile_ids).tolist():
            selected = np.flatnonzero(tile_ids == tile)
            face_ids = index.tile_to_faces.get(int(tile), np.zeros((0,), dtype=np.int64))
            if not len(face_ids):
                continue
            for face_start in range(0, len(face_ids), triangle_chunk):
                selected_faces = face_ids[face_start:face_start + triangle_chunk]
                batch = _ray_triangle_batch(
                    rays.origins[ray_start:ray_stop][selected],
                    rays.directions[ray_start:ray_stop][selected],
                    vertices[faces[selected_faces]],
                    selected_faces,
                )
                global_indices = ray_start + selected
                count[global_indices] += batch[0]
                coplanar[global_indices] += batch[4]
                for position, global_ray in enumerate(global_indices.tolist()):
                    candidate_face = int(batch[2][position])
                    candidate_t = float(batch[1][position])
                    if candidate_face < 0:
                        continue
                    if candidate_t < best_t[global_ray] or (candidate_t == best_t[global_ray] and candidate_face < best_face[global_ray]):
                        best_t[global_ray] = candidate_t
                        best_face[global_ray] = candidate_face
                        best_bary[global_ray] = batch[3][position]
    # Exact first-depth tie accounting is a small second pass over only the
    # chosen tile candidate list. This preserves tie determinism and avoids
    # making an ambiguity threshold.
    for ray_index in range(n):
        if best_face[ray_index] < 0:
            continue
        tile = int((rays.pixel_row[ray_index] // SCREEN_TILE_SIZE) * index.tiles_x + (rays.pixel_col[ray_index] // SCREEN_TILE_SIZE))
        face_ids = index.tile_to_faces.get(tile, np.zeros((0,), dtype=np.int64))
        ties = 0
        for face_start in range(0, len(face_ids), triangle_chunk):
            selected_faces = face_ids[face_start:face_start + triangle_chunk]
            batch = _ray_triangle_batch(rays.origins[ray_index:ray_index + 1], rays.directions[ray_index:ray_index + 1], vertices[faces[selected_faces]], selected_faces)
            if batch[0][0] and batch[1][0] == best_t[ray_index]:
                ties += 1
        tie_count[ray_index] = ties
    status = np.full((n,), STATUS_NO_HIT, dtype=object)
    status[coplanar > 0] = STATUS_AMBIGUOUS
    status[(best_face >= 0) & (coplanar == 0)] = STATUS_HIT
    depth = best_t * rays.camera_depth_scale
    depth[best_face < 0] = np.nan
    safe_t = np.where(best_face >= 0, best_t, 0.0)
    xyz = rays.origins + safe_t[:, None] * rays.directions
    xyz[best_face < 0] = np.nan
    component = np.full((n,), -1, dtype=np.int64)
    if component_by_face is not None:
        mask = best_face >= 0
        component[mask] = np.asarray(component_by_face, dtype=np.int64)[best_face[mask]]
    return FirstHitResult(status, depth, xyz, best_face, component, best_bary, count, tie_count, coplanar)


def _query_relation(query_depth: float, hit: FirstHitResult, index: int) -> str:
    status = str(hit.status[index])
    if status != STATUS_HIT:
        return REL_NO_DECISION
    first_depth = float(hit.depth[index])
    if query_depth < first_depth:
        return REL_IN_FRONT_OF_ZEROSET_SURFACE
    if query_depth > first_depth:
        return REL_BEHIND_ZEROSET_SURFACE
    return REL_ZEROSET_FIRST_SURFACE


def _query_ladder(rays: RayBundle, hit: FirstHitResult, offset: float, component_by_face: np.ndarray | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in np.flatnonzero(hit.status == STATUS_HIT).tolist():
        first_depth = float(hit.depth[index])
        for label, delta in (("Q_before", -offset), ("Q_surface", 0.0), ("Q_behind", offset)):
            query_depth = first_depth + delta
            relation = _query_relation(query_depth, hit, index)
            query_xyz = rays.origins[index] + query_depth * rays.directions[index]
            records.append({
                "ray_index": int(index),
                "label": label,
                "query_depth": float(query_depth),
                "query_world_xyz": query_xyz.tolist(),
                "first_hit_depth": first_depth,
                "first_hit_world_xyz": hit.world_xyz[index].tolist(),
                "triangle_id": int(hit.triangle_id[index]),
                "component_id": int(hit.component_id[index]),
                "barycentric": hit.barycentric[index].tolist(),
                "relation": relation,
                "offset": float(delta),
                "surface_membership_threshold_used": False,
                "candidate_b_median_used": False,
            })
    return records


def _synthetic_leakage(surface: AnalyticSurface, vertices: np.ndarray, faces: np.ndarray, h: float) -> dict[str, Any]:
    signed_vertices = surface.signed_distance(vertices)
    centers = vertices[faces].mean(axis=1) if len(faces) else np.zeros((0, 3), dtype=np.float64)
    signed_centers = surface.signed_distance(centers)
    rays = _synthetic_rays()
    analytic_depth, analytic_hit, _ = surface.intersect(rays)
    projected_depth = np.full((len(vertices),), np.nan, dtype=np.float64)
    # Project every synthetic mesh vertex along its own pixel ray when its
    # projection is inside the raster. This is an attribution diagnostic, not
    # a membership rule.
    relative = vertices - SYNTHETIC_CAMERA_ORIGIN
    z = relative[:, 2]
    px = relative[:, 0] / np.maximum(z, 1e-12)
    py = relative[:, 1] / np.maximum(z, 1e-12)
    col = np.rint((px / math.tan(SYNTHETIC_FOV * 0.5) + 1.0) * SYNTHETIC_IMAGE * 0.5 - 0.5).astype(np.int64)
    row = np.rint((py / math.tan(SYNTHETIC_FOV * 0.5) + 1.0) * SYNTHETIC_IMAGE * 0.5 - 0.5).astype(np.int64)
    valid = (z > 0.0) & (row >= 0) & (row < SYNTHETIC_IMAGE) & (col >= 0) & (col < SYNTHETIC_IMAGE)
    analytic_flat = np.where(valid, analytic_depth[np.clip(row, 0, SYNTHETIC_IMAGE - 1) * SYNTHETIC_IMAGE + np.clip(col, 0, SYNTHETIC_IMAGE - 1)], np.nan)
    projected_depth[valid] = z[valid]
    behind_vertices = valid & analytic_hit[np.clip(row, 0, SYNTHETIC_IMAGE - 1) * SYNTHETIC_IMAGE + np.clip(col, 0, SYNTHETIC_IMAGE - 1)] & (projected_depth > analytic_flat)
    return {
        "vertex_signed_distance_to_analytic_surface": _distribution(signed_vertices),
        "triangle_center_signed_distance_to_analytic_surface": _distribution(signed_centers),
        "vertices_exactly_on_analytic_surface": int(np.count_nonzero(signed_vertices == 0.0)),
        "triangles_exactly_on_analytic_surface_by_center": int(np.count_nonzero(signed_centers == 0.0)),
        "vertices_behind_analytic_first_surface_raw": int(np.count_nonzero(behind_vertices)),
        "vertices_behind_analytic_first_surface_over_one_grid_cell": int(np.count_nonzero(behind_vertices & ((projected_depth - analytic_flat) > h))),
        "triangle_centers_with_abs_distance_over_one_grid_cell": int(np.count_nonzero(np.abs(signed_centers) > h)),
        "false_internal_or_back_surface_components": "reported as raw signed-distance distributions; no component is removed or repaired",
        "analytic_no_hit_vertices_not_interpreted_as_leakage": True,
    }


def _synthetic_fixture(name: str) -> tuple[AnalyticSurface, Any, dict[str, Any]]:
    surface = {"fronto_parallel_plane": _plane_surface("fronto_parallel_plane"), "oblique_plane": _plane_surface("oblique_plane", True), "curved_sphere": _sphere_surface()}[name]
    field, field_meta = _synthetic_field(surface)
    extracted = _historical_extract(field)
    return surface, extracted, {"field": field_meta, "extraction_stats": extracted.stats}


def _synthetic_metrics(surface: AnalyticSurface, extracted: Any, hit: FirstHitResult) -> dict[str, Any]:
    rays = _synthetic_rays()
    analytic_depth, analytic_hit, _ = surface.intersect(rays)
    zero_hit = hit.status == STATUS_HIT
    analytic_valid = analytic_hit
    error = hit.depth[analytic_valid & zero_hit] - analytic_depth[analytic_valid & zero_hit]
    errors = np.abs(error)
    interior = analytic_valid & ~zero_hit
    silhouette = np.zeros_like(analytic_valid)
    if surface.kind == "plane_rectangle":
        assert surface.tangent_u is not None and surface.tangent_v is not None
        _, _, points = surface.intersect(rays)
        local_u = (points - surface.center) @ surface.tangent_u
        local_v = (points - surface.center) @ surface.tangent_v
        pixel_world = np.maximum(analytic_depth, 1e-6) * (2.0 * math.tan(SYNTHETIC_FOV * 0.5) / SYNTHETIC_IMAGE)
        silhouette = analytic_valid & (np.minimum(surface.half_u - np.abs(local_u), surface.half_v - np.abs(local_v)) <= 2.0 * pixel_world)
    hit_missed = analytic_valid & ~zero_hit
    false_hits = ~analytic_valid & zero_hit
    premature = analytic_valid & zero_hit & (hit.depth < analytic_depth)
    delayed = analytic_valid & zero_hit & (hit.depth > analytic_depth)
    return {
        "total_camera_rays": int(len(rays.origins)),
        "analytic_hit_rays": int(analytic_valid.sum()),
        "analytic_no_hit_rays": int((~analytic_valid).sum()),
        "zero_set_hit_rays": int(zero_hit.sum()),
        "zero_set_no_hit_rays": int((~zero_hit).sum()),
        "analytic_hit_zero_set_hit_rate": float(np.count_nonzero(analytic_valid & zero_hit) / max(int(analytic_valid.sum()), 1)),
        "first_hit_abs_depth_error": _distribution(errors),
        "first_hit_signed_depth_error": _distribution(error),
        "first_hit_abs_depth_error_over_h": _distribution(errors / SYNTHETIC_H),
        "premature_blocker_count_raw": int(premature.sum()),
        "delayed_blocker_count_raw": int(delayed.sum()),
        "missed_blocker_count": int(hit_missed.sum()),
        "false_zero_set_blocker_count_among_analytic_no_hit": int(false_hits.sum()),
        "analytic_hit_split": {
            "interior_rays": int((analytic_valid & ~silhouette).sum()),
            "silhouette_support_boundary_rays": int((analytic_valid & silhouette).sum()),
            "zero_set_missed_interior": int((hit_missed & ~silhouette).sum()),
            "zero_set_missed_silhouette_support_boundary": int((hit_missed & silhouette).sum()),
            "premature_interior": int((premature & ~silhouette).sum()),
            "premature_silhouette_support_boundary": int((premature & silhouette).sum()),
            "delayed_interior": int((delayed & ~silhouette).sum()),
            "delayed_silhouette_support_boundary": int((delayed & silhouette).sum()),
        },
        "zero_set_ambiguous_rays": int(np.count_nonzero(hit.status == STATUS_AMBIGUOUS)),
        "first_depth_tie_rays": int(np.count_nonzero(hit.first_hit_tie_count > 1)),
    }


def _fixture_exports(root: Path, name: str, surface: AnalyticSurface, extracted: Any, hit: FirstHitResult, metrics: dict[str, Any], leakage: dict[str, Any], ladder: list[dict[str, Any]]) -> None:
    fixture_root = root / name
    fixture_root.mkdir(parents=True, exist_ok=True)
    _write_visual_readme(fixture_root, f"W167 synthetic `{name}`", f"historical sparse TSDF + Lewiner zero-set control `{name}`", "the fixture is analytic ground truth; this directory groups the six required diagnostic views")
    vertices = np.asarray(extracted.vertices, dtype=np.float64)
    faces = np.asarray(extracted.faces, dtype=np.int64)
    _, face_components, component_sizes = _component_labels(faces, len(vertices))
    hit_component = hit.component_id.copy()
    if len(face_components):
        hit_component[hit.triangle_id >= 0] = face_components[hit.triangle_id[hit.triangle_id >= 0]]
    hit = FirstHitResult(hit.status, hit.depth, hit.world_xyz, hit.triangle_id, hit_component, hit.barycentric, hit.valid_positive_depth_intersections, hit.first_hit_tie_count, hit.coplanar_ambiguity_count)
    np.savez_compressed(fixture_root / "ray_results.npz", status=hit.status.astype(str), depth=hit.depth, world_xyz=hit.world_xyz, triangle_id=hit.triangle_id, component_id=hit.component_id, barycentric=hit.barycentric, valid_positive_depth_intersections=hit.valid_positive_depth_intersections, first_hit_tie_count=hit.first_hit_tie_count)
    _write_json(fixture_root / "report.json", {"surface": surface.__dict__, "metrics": metrics, "leakage": leakage, "component_accounting": {"component_count": int(len(component_sizes)), "component_vertex_sizes": component_sizes[:, 0].tolist(), "component_face_sizes": component_sizes[:, 1].tolist()}, "query_ladder": ladder})
    _write_json(fixture_root / "query_ladders.json", ladder)
    for kind in ("raw_zero_set_mesh", "first_hit_surface", "query_ladders", "blocker_relation", "component_provenance", "common_world"):
        directory = fixture_root / kind
        directory.mkdir(exist_ok=True)
        _write_visual_readme(directory, f"W167 synthetic `{name}` / `{kind}`", f"synthetic fixture `{name}`", "Analytic surface is ground truth only for synthetic controls; no production visibility state is assigned.")
    _save_world_views(fixture_root / "raw_zero_set_mesh", vertices, [], [], "raw zero-set mesh", seed=None)
    _save_world_views(fixture_root / "first_hit_surface", hit.world_xyz[hit.status == STATUS_HIT], [], [], "first-hit surface", seed=None)
    ladder_xyz = [np.asarray(row["query_world_xyz"], dtype=np.float64) for row in ladder]
    ladder_labels = [row["relation"] for row in ladder]
    _save_world_views(fixture_root / "query_ladders", np.asarray(ladder_xyz), [], ladder_labels, "query ladders", seed=None)
    relation_points = [np.asarray(row["query_world_xyz"], dtype=np.float64) for row in ladder]
    _save_world_views(fixture_root / "blocker_relation", np.asarray(relation_points), [], ladder_labels, "blocker relation", seed=None)
    component_points = hit.world_xyz[hit.status == STATUS_HIT]
    component_ids = hit.component_id[hit.status == STATUS_HIT]
    _save_world_views(fixture_root / "component_provenance", np.zeros((0, 3)), component_points, component_ids.tolist(), "component provenance", seed=None)
    _save_world_views(fixture_root / "common_world", vertices, component_points, component_ids.tolist(), "common world", seed=None)


def _write_visual_readme(path: Path, title: str, input_state: str, limitation: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("README.md").write_text(
        f"# {title}\n\n"
        f"- Input/state: {input_state}.\n"
        "- Palette: raw mesh blue; first-hit green; Q_before cyan; Q_surface yellow; Q_behind red; no decision gray; fragment first-hit orange; cameras white.\n"
        "- Rendering: deterministic world-coordinate orthographic projections (`perspective`, `top`, `side`), no smoothing, remeshing, filtering, or geometry mutation.\n"
        f"- Review limitation: {limitation}\n",
        encoding="utf-8",
    )


def _world_projection(points: np.ndarray, mode: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    points = np.asarray(points, dtype=np.float64)
    if mode == "top":
        uv = points[:, [0, 1]]
    elif mode == "side":
        uv = points[:, [0, 2]]
    elif mode == "perspective":
        # Fixed isometric camera for common-world review; it is not a
        # replacement camera calibration or a geometric query.
        u = 0.8660254 * points[:, 0] - 0.5 * points[:, 1]
        v = 0.2886751 * points[:, 0] + 0.2886751 * points[:, 1] + 0.8164966 * points[:, 2]
        uv = np.column_stack((u, v))
    else:
        raise ValueError(mode)
    if len(uv):
        bounds = (float(uv[:, 0].min()), float(uv[:, 0].max()), float(uv[:, 1].min()), float(uv[:, 1].max()))
    else:
        bounds = (-1.0, 1.0, -1.0, 1.0)
    return uv, bounds


def _render_world_png(path: Path, base_points: np.ndarray, overlays: list[tuple[np.ndarray, tuple[int, int, int]]], mode: str, title: str) -> None:
    from PIL import Image, ImageDraw

    size = (1100, 800)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:] = np.asarray((24, 28, 34), dtype=np.uint8)
    all_points = [np.asarray(base_points, dtype=np.float64)] + [np.asarray(p, dtype=np.float64) for p, _ in overlays if len(p)]
    all_points = [p.reshape(-1, 3) for p in all_points if len(p)]
    if not all_points:
        Image.fromarray(image, mode="RGB").save(path, format="PNG", optimize=True)
        return
    all_uv = [_world_projection(p, mode)[0] for p in all_points]
    combined = np.concatenate(all_uv, axis=0)
    xmin, xmax = float(combined[:, 0].min()), float(combined[:, 0].max())
    ymin, ymax = float(combined[:, 1].min()), float(combined[:, 1].max())
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)

    def pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uv, _ = _world_projection(points, mode)
        x = np.rint(50.0 + 1000.0 * (uv[:, 0] - xmin) / dx).astype(np.int64).clip(0, size[0] - 1)
        y = np.rint(750.0 - 700.0 * (uv[:, 1] - ymin) / dy).astype(np.int64).clip(0, size[1] - 1)
        return x, y

    if len(base_points):
        x, y = pixels(np.asarray(base_points, dtype=np.float64).reshape(-1, 3))
        image[y, x] = np.asarray(MESH_RGB, dtype=np.float64).reshape(1, 3) * 255.0
    for points, color in overlays:
        if len(points):
            x, y = pixels(np.asarray(points, dtype=np.float64).reshape(-1, 3))
            image[y, x] = np.asarray(color, dtype=np.float64).reshape(1, 3) * 255.0
    output = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(output)
    draw.text((24, 20), title, fill=(240, 240, 240))
    draw.text((24, 45), "blue mesh | cyan before | yellow surface | red behind | orange fragment", fill=(190, 200, 210))
    output.save(path, format="PNG", optimize=True)


def _save_world_views(directory: Path, base_points: np.ndarray, overlay_points: Any, labels: list[str] | list[int], title: str, seed: int | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    base_points = np.asarray(base_points, dtype=np.float64).reshape(-1, 3) if len(np.asarray(base_points)) else np.zeros((0, 3), dtype=np.float64)
    overlay_points = np.asarray(overlay_points, dtype=np.float64).reshape(-1, 3) if len(np.asarray(overlay_points)) else np.zeros((0, 3), dtype=np.float64)
    overlays: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if len(overlay_points):
        if labels and isinstance(labels[0], str):
            colors = {name: tuple(int(v * 255) for v in rgb) for name, rgb in RELATION_RGB.items()}
            for relation in RELATION_RGB:
                mask = np.asarray([value == relation for value in labels], dtype=bool)
                overlays.append((overlay_points[mask], colors[relation]))
        else:
            ids = np.asarray(labels, dtype=np.int64)
            unique = np.unique(ids)
            for component_id in unique.tolist():
                hue = (component_id * 0.6180339887498949) % 1.0
                rgb = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(hue, 0.72, 0.95))
                overlays.append((overlay_points[ids == component_id], rgb))
    for mode in ("perspective", "top", "side"):
        _render_world_png(directory / f"{mode}.png", base_points, overlays, mode, f"W167 {title} / {mode}")


def _real_camera_rays(camera: Any, pixels: np.ndarray) -> RayBundle:
    width, height = int(camera.image_width), int(camera.image_height)
    rows = pixels[:, 0].astype(np.float64)
    cols = pixels[:, 1].astype(np.float64)
    ndc_x = (2.0 * cols + 1.0) / width - 1.0
    ndc_y = (2.0 * rows + 1.0) / height - 1.0
    camera_dirs = np.column_stack((ndc_x * math.tan(float(camera.FoVx) * 0.5), ndc_y * math.tan(float(camera.FoVy) * 0.5), np.ones_like(ndc_x)))
    view = camera.world_view_transform.detach().cpu().numpy().astype(np.float64)
    directions = camera_dirs @ np.linalg.inv(view[:3, :3])
    origin = camera.camera_center.detach().cpu().numpy().astype(np.float64)
    return RayBundle(np.repeat(origin[None, :], len(pixels), axis=0), directions, np.ones((len(pixels),), dtype=np.float64), pixels[:, 0].astype(np.int64), pixels[:, 1].astype(np.int64))


def _polygon_mask(pixels: np.ndarray, polygon: tuple[tuple[float, float], ...]) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float64)
    x = pixels[:, 1].astype(np.float64)
    y = pixels[:, 0].astype(np.float64)
    inside = np.zeros((len(pixels),), dtype=bool)
    x0, y0 = points[-1]
    for x1, y1 in points:
        crossing = ((y1 > y) != (y0 > y)) & (x < (x0 - x1) * (y - y1) / np.where(y1 != y0, y1 - y0, 1.0) + x1)
        inside ^= crossing
        x0, y0 = x1, y1
    return inside


def _roi_pixels(camera_name: str, width: int, height: int, stride: int) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int, int]]:
    all_pixels = np.stack(np.indices((height, width), dtype=np.int64), axis=-1).reshape(-1, 2)
    sampled = all_pixels[(all_pixels[:, 0] % stride == 0) & (all_pixels[:, 1] % stride == 0)]
    masks: dict[str, np.ndarray] = {}
    union = np.zeros((len(sampled),), dtype=bool)
    for name, camera_polygons in REVIEW_POLYGONS.items():
        mask = _polygon_mask(sampled, camera_polygons[camera_name])
        masks[name] = mask
        union |= mask
    contact = masks[CONTACT_REGIONS[0]] | masks[CONTACT_REGIONS[1]]
    masks["tabletop_vase_contact"] = contact
    selected_mask = union | masks["table_side_lower_geometry"]
    selected = sampled[selected_mask]
    selected_masks = {name: mask[selected_mask] for name, mask in masks.items()}
    if not len(selected):
        return selected, selected_masks, (0, 0, 0, 0)
    return selected, selected_masks, (int(selected[:, 1].min()), int(selected[:, 0].min()), int(selected[:, 1].max()), int(selected[:, 0].max()))


def _major_component_ids(component_sizes: np.ndarray, count: int = 20) -> set[int]:
    if not len(component_sizes):
        return set()
    order = np.lexsort((np.arange(len(component_sizes)), -component_sizes[:, 0]))
    return set(order[:count].tolist())


def _real_scene_replay(args: argparse.Namespace, source: Path, out: Path, component_by_face: np.ndarray, component_sizes: np.ndarray) -> dict[str, Any]:
    from maximal_visible_connectivity_export import load_all_train_cameras

    cameras, camera_meta = load_all_train_cameras(args.dataset, args.images, args.sparse_dir, args.resolution, args.llffhold, "cpu")
    by_name = {str(camera.image_name): camera for camera in cameras}
    depth_cache = np.load(args.depth_cache, allow_pickle=False)["depth"]
    with np.load(source, allow_pickle=False) as bundle:
        vertices = np.asarray(bundle["vertices"], dtype=np.float64)
        faces = np.asarray(bundle["faces"], dtype=np.int64)
    major_ids = _major_component_ids(component_sizes)
    scene_root = out / "real_scene"
    scene_root.mkdir(parents=True, exist_ok=True)
    _write_visual_readme(scene_root, "W167 real-scene replay", "W166 immutable raw mesh plus frozen W160–W165 camera/ROI samples", "real-scene replay has no physical hidden-space ground truth")
    _write_visual_readme(scene_root / "projection", "W167 real per-camera projection", "fixed camera ROI samples and exact first-hit pixel overlay", "projection PNGs are qualitative diagnostics")
    per_camera: dict[str, Any] = {}
    all_ladders: list[dict[str, Any]] = []
    all_hit_points: list[np.ndarray] = []
    all_behind_points: list[np.ndarray] = []
    all_fragment_points: list[np.ndarray] = []
    for camera_name in args.real_camera_names:
        camera = by_name[camera_name]
        camera_index = next(i for i, candidate in enumerate(cameras) if str(candidate.image_name) == camera_name)
        height, width = int(camera.image_height), int(camera.image_width)
        pixels, roi_masks, roi_bounds = _roi_pixels(camera_name, width, height, args.sample_stride)
        rays = _real_camera_rays(camera, pixels)
        index = ScreenTileIndex.build(vertices, faces, camera, roi_bounds, tile_size=SCREEN_TILE_SIZE, face_chunk=args.face_chunk)
        hit = intersect_first_hit_screen_index(rays, vertices, faces, camera, index, component_by_face=component_by_face, ray_chunk=args.ray_chunk, triangle_chunk=args.triangle_chunk)
        ladder = _query_ladder(rays, hit, REAL_QUERY_OFFSET, component_by_face)
        for record in ladder:
            record["camera"] = camera_name
            record["roi_names"] = [name for name, mask in roi_masks.items() if int(record["ray_index"]) < len(mask) and bool(mask[int(record["ray_index"])] )]
        all_ladders.extend(ladder)
        hit_mask = hit.status == STATUS_HIT
        fragment_mask = hit_mask & ~np.isin(hit.component_id, np.asarray(sorted(major_ids), dtype=np.int64))
        behind_mask = np.asarray([row["relation"] == REL_BEHIND_ZEROSET_SURFACE for row in ladder[2::3]], dtype=bool) if ladder else np.zeros((0,), dtype=bool)
        all_hit_points.append(hit.world_xyz[hit_mask])
        all_fragment_points.append(hit.world_xyz[fragment_mask])
        if ladder:
            all_behind_points.append(np.asarray([row["query_world_xyz"] for row in ladder if row["relation"] == REL_BEHIND_ZEROSET_SURFACE], dtype=np.float64))
        projection_dir = scene_root / "projection" / camera_name.replace(".JPG", "")
        projection_dir.mkdir(parents=True, exist_ok=True)
        _write_visual_readme(projection_dir, f"W167 real `{camera_name}` projection", "fixed camera and frozen W160–W165 ROI pixel samples", "real-scene replay is not physical ground truth; images show raw first-hit diagnostics only")
        # Camera projection review: no mesh recoloring is used; points are
        # plotted at their exact projected pixel and remain separately typed.
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (width, height), (24, 28, 34))
        draw = ImageDraw.Draw(image)
        for row_index, col_index in zip(pixels[:, 0].tolist(), pixels[:, 1].tolist()):
            draw.point((col_index, row_index), fill=(80, 90, 105))
        for idx in np.flatnonzero(hit_mask).tolist():
            color = FRAGMENT_RGB if fragment_mask[idx] else HIT_RGB
            rgb = tuple(int(v * 255) for v in color)
            draw.ellipse((int(pixels[idx, 1]) - 2, int(pixels[idx, 0]) - 2, int(pixels[idx, 1]) + 2, int(pixels[idx, 0]) + 2), fill=rgb)
        image.save(projection_dir / "first_hits.png", format="PNG", optimize=True)
        _write_visual_readme(scene_root / "raw_zero_set_mesh", "W167 real raw_zero_set_mesh", "W166 immutable historical raw NPZ", "PNG is a review projection; raw NPZ/OBJ remains authoritative")
        _write_visual_readme(scene_root / "first_hit_surface", "W167 real first_hit_surface", "exact first-hit points from fixed ROI rays", "real-scene replay is not physical ground truth")
        _write_visual_readme(scene_root / "query_ladders", "W167 real query_ladders", "Q_before/Q_surface/Q_behind on identical fixed rays", "surface relation is geometric diagnostic only")
        _write_visual_readme(scene_root / "blocker_relation", "W167 real blocker_relation", "diagnostic relation categories", "NO_HIT and AMBIGUOUS remain NO_DECISION")
        _write_visual_readme(scene_root / "component_provenance", "W167 real component_provenance", "first-hit points colored by exact faces-adjacency component ID", "component labels are attribution only; no component is filtered")
        _write_visual_readme(scene_root / "common_world", "W167 real common_world", "same raw mesh and fixed-camera first-hit/query populations in shared world coordinates", "qualitative review only; no physical hidden-surface claim")
        np.savez_compressed(scene_root / f"{camera_name.replace('.JPG', '')}_ray_results.npz", pixel=pixels, status=hit.status.astype(str), depth=hit.depth, world_xyz=hit.world_xyz, triangle_id=hit.triangle_id, component_id=hit.component_id, barycentric=hit.barycentric, valid_positive_depth_intersections=hit.valid_positive_depth_intersections, first_hit_tie_count=hit.first_hit_tie_count)
        states = {STATUS_HIT: int(np.count_nonzero(hit.status == STATUS_HIT)), STATUS_NO_HIT: int(np.count_nonzero(hit.status == STATUS_NO_HIT)), STATUS_AMBIGUOUS: int(np.count_nonzero(hit.status == STATUS_AMBIGUOUS))}
        region_counts = {}
        for region, mask in roi_masks.items():
            region_hit = mask & hit_mask
            region_fragment = region_hit & fragment_mask
            region_counts[region] = {
                "ray_count": int(mask.sum()),
                "first_hit_count": int(region_hit.sum()),
                "no_decision_ray_count": int(np.count_nonzero(mask & ~hit_mask)),
                "query_count": int(3 * region_hit.sum()),
                "query_before_count": int(region_hit.sum()),
                "query_surface_count": int(region_hit.sum()),
                "query_behind_count": int(region_hit.sum()),
                "query_no_decision_count": int(3 * np.count_nonzero(mask & ~hit_mask)),
                "major_component_first_hit_count": int(np.count_nonzero(region_hit & ~region_fragment)),
                "fragment_first_hit_count": int(region_fragment.sum()),
                "first_hit_depth_distribution": _distribution(hit.depth[region_hit]),
            }
        hit_components = hit.component_id[hit_mask]
        unique, counts = np.unique(hit_components, return_counts=True)
        component_rows = [{"component_id": int(component), "first_hit_count": int(count), "major_component": int(component) in major_ids, "component_vertex_count": int(component_sizes[component, 0]), "component_triangle_count": int(component_sizes[component, 1]), "component_vertex_fraction": float(component_sizes[component, 0] / max(len(vertices), 1)), "component_triangle_fraction": float(component_sizes[component, 1] / max(len(faces), 1))} for component, count in zip(unique.tolist(), counts.tolist())]
        query_counts = {relation: int(sum(row["relation"] == relation for row in ladder)) for relation in (REL_ZEROSET_FIRST_SURFACE, REL_IN_FRONT_OF_ZEROSET_SURFACE, REL_BEHIND_ZEROSET_SURFACE, REL_NO_DECISION)}
        per_camera[camera_name] = {"camera_index": camera_index, "image_shape": [height, width], "roi_bounds_xyxy": list(roi_bounds), "query_ray_count": int(len(rays.origins)), "status_counts": states, "major_component_first_hit_count": int(np.count_nonzero(hit_mask & np.isin(hit.component_id, np.asarray(sorted(major_ids), dtype=np.int64)))), "fragment_first_hit_count": int(fragment_mask.sum()), "first_hit_depth_distribution": _distribution(hit.depth[hit_mask]), "query_relation_counts": query_counts, "component_provenance": component_rows, "roi_counts": region_counts, "index": index.stats, "anomalous_cases": [{"ray_index": int(i), "pixel": pixels[i].tolist(), "status": str(hit.status[i]), "depth": None if not np.isfinite(hit.depth[i]) else float(hit.depth[i]), "triangle_id": int(hit.triangle_id[i]), "component_id": int(hit.component_id[i])} for i in np.flatnonzero((hit.status != STATUS_HIT) | fragment_mask).tolist()]}
    raw_vertices = vertices
    hit_points = np.concatenate(all_hit_points) if all_hit_points else np.zeros((0, 3), dtype=np.float64)
    behind_points = np.concatenate(all_behind_points) if all_behind_points else np.zeros((0, 3), dtype=np.float64)
    fragment_points = np.concatenate(all_fragment_points) if all_fragment_points else np.zeros((0, 3), dtype=np.float64)
    _save_world_views(scene_root / "raw_zero_set_mesh", raw_vertices, [], [], "real raw zero-set mesh", seed=None)
    camera_positions = np.asarray([by_name[name].camera_center.detach().cpu().numpy() for name in args.real_camera_names], dtype=np.float64)
    _save_world_views(scene_root / "first_hit_surface", np.zeros((0, 3)), hit_points, ["FIRST_HIT_SURFACE"] * len(hit_points), "real first-hit surface", seed=None)
    ladder_points = np.asarray([row["query_world_xyz"] for row in all_ladders], dtype=np.float64) if all_ladders else np.zeros((0, 3))
    ladder_labels = [row["relation"] for row in all_ladders]
    _save_world_views(scene_root / "query_ladders", np.zeros((0, 3)), ladder_points, ladder_labels, "real query ladders", seed=None)
    blocker_labels = [
        "FRAGMENT_FIRST_HIT" if row["label"] == "Q_surface" and int(row["component_id"]) not in major_ids else row["relation"]
        for row in all_ladders
    ]
    _save_world_views(scene_root / "blocker_relation", np.zeros((0, 3)), ladder_points, blocker_labels, "real blocker relation", seed=None)
    # Component provenance uses a compact direct render to avoid losing the
    # per-hit IDs in the common-world helper's generic label contract.
    component_ids_all = []
    for camera_name in args.real_camera_names:
        data = np.load(scene_root / f"{camera_name.replace('.JPG', '')}_ray_results.npz", allow_pickle=False)
        component_ids_all.extend(data["component_id"][data["status"] == STATUS_HIT].tolist())
    _save_world_views(scene_root / "component_provenance", np.zeros((0, 3)), hit_points, component_ids_all, "real component provenance", seed=None)
    common_points = np.concatenate((hit_points, behind_points, camera_positions, fragment_points)) if len(hit_points) or len(behind_points) or len(camera_positions) or len(fragment_points) else np.zeros((0, 3))
    common_labels = ["FIRST_HIT_SURFACE"] * len(hit_points) + [REL_BEHIND_ZEROSET_SURFACE] * len(behind_points) + ["CAMERA_POSITION"] * len(camera_positions) + ["FRAGMENT_FIRST_HIT"] * len(fragment_points)
    _save_world_views(scene_root / "common_world", raw_vertices, common_points, common_labels, "real common world", seed=None)
    report = {"status": "COMPLETE_REAL_SCENE_REPLAY", "camera_meta": camera_meta, "camera_names": list(args.real_camera_names), "roi_names": ["tabletop", "tabletop_vase_contact", "table_side_lower_geometry", "vase_foreground_structure"], "sample_stride_pixels": args.sample_stride, "query_offset_world": REAL_QUERY_OFFSET, "per_camera": per_camera, "fragment_attribution_only": True, "physical_ground_truth": "NONE_NON_ORACLE_REVIEW_REFERENCE", "outputs": {"root": str(scene_root.resolve())}}
    _write_json(scene_root / "real_scene_replay_report.json", report)
    return report


def _load_component_data(source: Path, accounting_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(source, allow_pickle=False) as bundle:
        vertices = np.asarray(bundle["vertices"], dtype=np.float64)
        faces = np.asarray(bundle["faces"], dtype=np.int64)
    labels, face_labels, sizes = _component_labels(faces, len(vertices))
    historical = json.loads(accounting_path.read_text(encoding="utf-8"))
    if int(len(sizes)) != int(historical["connected_component_count"]):
        raise RuntimeError(f"component count mismatch: computed={len(sizes)} historical={historical['connected_component_count']}")
    return face_labels, sizes, {"vertex_labels_available": True, "face_component_count": int(len(sizes)), "historical_component_count": int(historical["connected_component_count"]), "historical_accounting_path": str(accounting_path)}


def _run_synthetic(out: Path) -> dict[str, Any]:
    synthetic_root = out / "synthetic"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    _write_visual_readme(synthetic_root, "W167 synthetic controls", "three fixed analytic fixtures extracted through the historical sparse TSDF path", "the real-scene blocker question is not answered by synthetic controls alone")
    results: dict[str, Any] = {}
    for name in ("fronto_parallel_plane", "oblique_plane", "curved_sphere"):
        surface, extracted, meta = _synthetic_fixture(name)
        vertices = np.asarray(extracted.vertices, dtype=np.float64)
        faces = np.asarray(extracted.faces, dtype=np.int64)
        _, face_components, component_sizes = _component_labels(faces, len(vertices))
        rays = _synthetic_rays()
        hit = intersect_first_hit_bruteforce(rays, vertices, faces, face_components, triangle_chunk=20_000, ray_chunk=16)
        metrics = _synthetic_metrics(surface, extracted, hit)
        leakage = _synthetic_leakage(surface, vertices, faces, SYNTHETIC_H)
        ladder = _query_ladder(rays, hit, SYNTHETIC_QUERY_OFFSET, face_components)
        _fixture_exports(synthetic_root, name, surface, extracted, hit, metrics, leakage, ladder)
        results[name] = {"construction": meta, "metrics": metrics, "leakage": leakage, "query_ladder_counts": {relation: int(sum(row["relation"] == relation for row in ladder)) for relation in (REL_ZEROSET_FIRST_SURFACE, REL_IN_FRONT_OF_ZEROSET_SURFACE, REL_BEHIND_ZEROSET_SURFACE, REL_NO_DECISION)}, "component_count": int(len(component_sizes)), "vertex_count": int(len(vertices)), "face_count": int(len(faces))}
    _write_json(synthetic_root / "synthetic_report.json", results)
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    source_shapes = _npz_descriptors(source)
    synthetic = _run_synthetic(args.out)
    if args.real:
        component_labels, component_sizes, component_meta = _load_component_data(source, args.accounting)
    else:
        historical = json.loads(args.accounting.read_text(encoding="utf-8"))
        component_labels = np.zeros((0,), dtype=np.int64)
        component_sizes = np.zeros((0, 2), dtype=np.int64)
        component_meta = {"vertex_labels_available": False, "real_scene_component_labeling": "NOT_RUN", "historical_component_count": int(historical["connected_component_count"]), "historical_accounting_path": str(args.accounting)}
    real = None
    if args.real:
        real = _real_scene_replay(args, source, args.out, component_labels, component_sizes)
    historical_report = json.loads(args.w166_report.read_text(encoding="utf-8")) if args.w166_report.exists() else {}
    source_hash = _sha256_file(source)
    verdict = "REAL_SCENE_REVIEW_REQUIRED"
    # A one-cell support boundary is an explicitly reported limitation of a
    # masked grid, not a reason to hide the interior result.  The fixed gate
    # therefore requires no interior miss and a p95 first-hit error below one
    # frozen synthetic grid cell; silhouette misses remain visible in the
    # accounting and never become invisible "success".
    synthetic_supported = all(
        value["metrics"]["analytic_hit_split"]["zero_set_missed_interior"] == 0
        and float(value["metrics"]["first_hit_abs_depth_error_over_h"]["p95"] or 0.0) < 1.0
        for value in synthetic.values()
    )
    if not synthetic_supported:
        verdict = "ZEROSET_BLOCKER_FAILS_GEOMETRICALLY"
    report = {
        "status": "COMPLETE_WL167_RAW_ZERO_SET_RAY_BLOCKER_AUDIT",
        "worklog": 167,
        "intent_alignment": {"diagnostic_only": True, "question": "Can the historical raw zero-set serve as an explicit camera-ray blocker?", "production_behavior_modified": False, "candidate_b_modified": False, "nurbs_fit": False, "mesh_repaired": False, "new_rois": False},
        "implementation_fidelity": {"ray_triangle": "two-sided Moller-Trumbore; strict t>0; exact barycentric bounds; deterministic (depth, triangle_id) tie key", "determinant_epsilon": RAY_DETERMINANT_EPS, "determinant_epsilon_is_swept": False, "broad_phase": "real-scene projected triangle AABB screen tiles; all candidates checked by same primitive", "query_ladder": {"synthetic_offset": SYNTHETIC_QUERY_OFFSET, "real_offset": REAL_QUERY_OFFSET, "surface_query_is_exact_first_hit": True, "candidate_b_median_used": False}},
        "historical_zero_set_preservation": {"primary_source": str(source), "primary_source_sha256": source_hash, "historical_w166_source_sha256": historical_report.get("source_npz_sha256"), "source_shapes": source_shapes, "geometry_modification": "NONE", "historical_w153_h": HISTORICAL_H, "historical_w153_mu": HISTORICAL_MU, "historical_extraction": "W153 typed ExtractedSurface from 943a764 renderer-median -> projective TSDF -> all-eight-corner Lewiner path; W166 raw NPZ byte-preserved export", "component_filtering": False, "repair_or_smoothing": False, "w153_accounting": str(args.accounting)},
        "component_labeling": component_meta,
        "ray_mesh_contract": {"statuses": [STATUS_HIT, STATUS_NO_HIT, STATUS_AMBIGUOUS], "returned_per_ray": ["status", "first_hit_camera_depth", "first_hit_world_xyz", "triangle_id", "component_id", "barycentric", "valid_positive_depth_intersections"], "ambiguous_definition": "exact coplanar ray/nondegenerate triangle relation; no learned or tuned threshold", "tie_definition": "minimum positive t, then minimum triangle ID; equal-depth adjacent triangles are retained in tie_count and not silently discarded"},
        "synthetic_analytic_blocker_result": synthetic,
        "zero_set_hidden_space_leakage_diagnostic": {"synthetic": "raw signed-distance, projected behind-surface, and component accounting retained per fixture", "real": "no hidden-space ground truth claimed; only geometry and component location reported"},
        "fragment_attribution": {"major_components": "top 20 components by exact vertex count, attribution only", "small_fragments": "all other disconnected components, no semantic size threshold", "fragment_confound_triggered": False, "reason": "real-scene ground truth is unavailable; no false blocker claim is made from fragment count alone"},
        "real_scene_replay": real if real is not None else {"status": "NOT_RUN", "reason": "pass --real to execute the frozen 3-camera replay"},
        "qualitative_review_exports": {"required_types": ["raw_zero_set_mesh", "first_hit_surface", "query_ladders", "blocker_relation", "component_provenance", "common_world"], "png_primary": True, "readme_per_visualization_directory": True, "common_world_views": ["perspective", "top", "side"], "synthetic_root": str((args.out / "synthetic").resolve()), "real_root": str((args.out / "real_scene").resolve()) if real is not None else None},
        "architecture_result": {"verdict": verdict, "synthetic_controls_supported": synthetic_supported, "synthetic_gate": "zero_set_missed_interior == 0 and p95(|z_s-z*|/h) < 1 fixed synthetic grid cell; silhouette/support misses are retained", "reason": "Synthetic raw zero-set control is supported in interior regions; real scene remains non-oracle qualitative review." if synthetic_supported else "At least one synthetic fixture misses a known analytic blocker in a controlled interior ray set or exceeds one fixed grid-cell p95 error."},
        "retained_rejected_open": {"retained": ["historical SDF/TSDF semantics", "historical h/mu/grid/extraction", "W160 Candidate-B", "W161 paused spatial-domain status", "W162-W165", "W166 raw mesh", "Gaussian Regions", "Boundary First", "NURBS", "continuation", "Eligibility"], "rejected": ["median-depth blocker", "expected-depth blocker", "TSDF sign global oracle", "Surface Owner", "latent continuation eligibility", "mesh repair/filtering", "NURBS fitting", "inferred hidden geometry", "NO_HIT forced to visible or occluded"], "open": ["human review of real-scene first-hit coherence", "whether real fragment hits materially contradict intended surfaces", "future architecture promotion only after review"]},
        "inputs": {"source_npz": str(source), "w166_report": str(args.w166_report), "depth_cache": str(args.depth_cache), "dataset": str(args.dataset), "w153_replay_cache_excluded_from_temp_mirror": True},
        "runtime_seconds": {"total": time.time() - started},
    }
    _write_json(args.out / "worklog_167_report.json", report)
    args.out.joinpath("README.md").write_text(
        "# Worklog 167 — Raw SDF/TSDF Zero-Set Surface as Explicit Camera-Ray Blocker\n\n"
        "이 output은 W166 immutable raw zero-set mesh에 대한 diagnostic-only ray/triangle first-hit audit이다. `synthetic/`는 historical sparse TSDF + all-eight-corner Lewiner 추출을 거친 analytic plane/sphere control이며, `real_scene/`는 요청 시 frozen W160–W165 camera/ROI의 non-oracle replay이다.\n\n"
        f"Architecture verdict: `{verdict}`. Candidate-B, canonical renderer, SDF/TSDF construction, mesh geometry, NURBS, continuation, Eligibility는 변경하지 않았다.\n\n"
        "PNG는 review용이며, raw NPZ/OBJ가 geometry의 authoritative artifact다. NO_HIT와 AMBIGUOUS는 `NO_DECISION`으로 fail-closed 유지한다.\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--w166-report", type=Path, default=REPO_ROOT / "output/166_historical_sdf_zero_surface_mesh_export/mesh_export_report.json")
    parser.add_argument("--accounting", type=Path, default=DEFAULT_W153_ACCOUNTING)
    parser.add_argument("--depth-cache", type=Path, default=DEFAULT_DEPTH_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--real-camera-names", nargs=3, default=list(REVIEW_CAMERAS))
    parser.add_argument("--sample-stride", type=int, default=REAL_SAMPLE_STRIDE)
    parser.add_argument("--face-chunk", type=int, default=500_000)
    parser.add_argument("--triangle-chunk", type=int, default=20_000)
    parser.add_argument("--ray-chunk", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    report = run(args)
    print(json.dumps({"status": report["status"], "architecture_result": report["architecture_result"]["verdict"], "real_status": report["real_scene_replay"]["status"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
