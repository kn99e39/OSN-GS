"""Candidate F: Gaussian-region-owned, evidence-bounded TSDF surfaces.

This module is intentionally isolated from the production visible-surface
construction path.  Gaussian rows own identity and region membership; the
renderer-derived TSDF owns the observed geometry.  A representative is fit
only after native TSDF cell topology and a support-derived boundary chart are
available.

The implementation does not materialize a mesh.  Zero-level samples are
extracted directly from authoritative TSDF cells and retain their source cell
keys, corner values, and support counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from osn_gs.utils.torch_ops import require_torch


torch = require_torch()


# These values are the WL127 sparse-key contract.  Keep the packing local so
# this isolated candidate remains usable with replay caches without importing
# the devtools package into the runtime surface package.
KEY_BOUND = 1 << 19
_AXIS_SPAN = KEY_BOUND << 1
STRIDE_Z = 1
STRIDE_Y = _AXIS_SPAN
STRIDE_X = _AXIS_SPAN * _AXIS_SPAN

MEMBERSHIP_CORE = "core"
MEMBERSHIP_ATTACHED = "attached"
MEMBERSHIP_AMBIGUOUS = "ambiguous"
MEMBERSHIP_REJECTED = "rejected"
MEMBERSHIP_UNASSIGNED = "unassigned"

OBSERVED = "OBSERVED"
OCCLUDED = "OCCLUDED"
UNRESOLVED = "UNRESOLVED"
UNOWNED_TSDF_SUPPORT = "UNOWNED_TSDF_SUPPORT"
ABSTAIN_REPRESENTATIVE = "ABSTAIN_REPRESENTATIVE"
MATERIALIZED_REPRESENTATIVE = "MATERIALIZED_REPRESENTATIVE"

_CORNER_OFFSETS = np.asarray(
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ],
    dtype=np.int64,
)
_CUBE_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _as_cpu_tensor(value: Any, *, dtype: Any | None = None) -> Any:
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    result = result.detach().cpu()
    return result.to(dtype=dtype) if dtype is not None else result


def _normalize_rows(values: Any, eps: float = 1e-12) -> Any:
    lengths = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / torch.clamp(lengths, min=eps)


def encode_cell_keys(cell_indices: Any) -> Any:
    """Encode integer cell/voxel indices with the WL127 sparse key packing."""

    indices = _as_cpu_tensor(cell_indices, dtype=torch.int64)
    if indices.ndim == 1:
        indices = indices.reshape(1, 3)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("cell_indices must have shape (N, 3)")
    x, y, z = indices.unbind(dim=1)
    return (x + KEY_BOUND) * STRIDE_X + (y + KEY_BOUND) * STRIDE_Y + (z + KEY_BOUND)


def decode_cell_keys(keys: Any) -> Any:
    """Decode WL127 sparse keys to integer cell/voxel indices."""

    values = _as_cpu_tensor(keys, dtype=torch.int64).reshape(-1)
    x = torch.div(values, STRIDE_X, rounding_mode="floor") - KEY_BOUND
    rem = values - (x + KEY_BOUND) * STRIDE_X
    y = torch.div(rem, STRIDE_Y, rounding_mode="floor") - KEY_BOUND
    z = rem - (y + KEY_BOUND) * STRIDE_Y - KEY_BOUND
    return torch.stack((x, y, z), dim=1)


@dataclass(frozen=True)
class EvidenceBoundedTSDFField:
    keys: Any
    value: Any
    support_count: Any
    h: float
    mu: float
    closure: str = ""

    def __post_init__(self) -> None:
        keys = _as_cpu_tensor(self.keys, dtype=torch.int64).reshape(-1)
        value = _as_cpu_tensor(self.value, dtype=torch.float32).reshape(-1)
        support = _as_cpu_tensor(self.support_count, dtype=torch.int32).reshape(-1)
        if not (keys.numel() == value.numel() == support.numel()):
            raise ValueError("TSDF keys, value, and support_count must have equal length")
        if keys.numel() > 1 and not bool(torch.all(keys[1:] > keys[:-1])):
            raise ValueError("TSDF keys must be sorted and unique")
        if not np.isfinite(float(self.h)) or float(self.h) <= 0.0:
            raise ValueError("TSDF voxel size h must be positive")
        if not np.isfinite(float(self.mu)) or float(self.mu) <= 0.0:
            raise ValueError("TSDF truncation mu must be positive")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "support_count", support)
        object.__setattr__(self, "h", float(self.h))
        object.__setattr__(self, "mu", float(self.mu))

    @classmethod
    def from_npz(cls, path: str) -> "EvidenceBoundedTSDFField":
        with np.load(path, allow_pickle=False) as data:
            closure = data["closure"] if "closure" in data else ""
            if isinstance(closure, np.ndarray):
                closure = str(closure.reshape(-1)[0]) if closure.size else ""
            return cls(
                keys=data["keys"],
                value=data["value"],
                support_count=data["support_count"],
                h=float(np.asarray(data["h"]).reshape(-1)[0]),
                mu=float(np.asarray(data["mu"]).reshape(-1)[0]),
                closure=str(closure),
            )


@dataclass(frozen=True)
class TSDFVisibleSurfaceSamples:
    source_cell_keys: Any
    cell_indices: Any
    world_xyz: Any
    normals: Any
    corner_values: Any
    corner_support_count: Any
    h: float
    stats: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = _as_cpu_tensor(self.source_cell_keys, dtype=torch.int64).reshape(-1)
        cells = _as_cpu_tensor(self.cell_indices, dtype=torch.int64).reshape(-1, 3)
        xyz = _as_cpu_tensor(self.world_xyz, dtype=torch.float32).reshape(-1, 3)
        normals = _as_cpu_tensor(self.normals, dtype=torch.float32).reshape(-1, 3)
        values = _as_cpu_tensor(self.corner_values, dtype=torch.float32).reshape(-1, 8)
        support = _as_cpu_tensor(self.corner_support_count, dtype=torch.int32).reshape(-1, 8)
        count = keys.shape[0]
        if cells.shape[0] != count or xyz.shape[0] != count or normals.shape[0] != count:
            raise ValueError("TSDF surface cell, xyz, and normal arrays must have equal first dimension")
        # The replay runner may persist corner provenance to disk before the
        # memory-heavy topology stage and carry explicit empty placeholders in
        # RAM.  A non-empty corner table must still align row-for-row.
        if values.shape[0] not in (0, count) or support.shape[0] not in (0, count):
            raise ValueError("TSDF corner provenance arrays must be empty or aligned with samples")
        if count > 1 and not bool(torch.all(keys[1:] > keys[:-1])):
            raise ValueError("TSDF surface source cell keys must be sorted and unique")
        object.__setattr__(self, "source_cell_keys", keys)
        object.__setattr__(self, "cell_indices", cells)
        object.__setattr__(self, "world_xyz", xyz)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "corner_values", values)
        object.__setattr__(self, "corner_support_count", support)
        object.__setattr__(self, "h", float(self.h))


@dataclass(frozen=True)
class NearestGaussianAssociation:
    nearest_gaussian_index: Any
    nearest_gaussian_id: Any
    nearest_distance: Any
    backend: str
    stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GaussianRegionMembership:
    region_ids: Any
    status: tuple[str, ...]
    accepted_mask: Any
    accounting: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionOwnedTSDFSupport:
    nearest_region_id: Any
    nearest_membership_status: tuple[str, ...]
    owned_region_id: Any
    membership_status: tuple[str, ...]
    accepted_mask: Any
    accounting: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedSupportComponent:
    component_id: int
    region_id: int
    sample_indices: Any
    min_cell: tuple[int, int, int]
    max_cell: tuple[int, int, int]


@dataclass(frozen=True)
class ObservedSupportBoundary:
    component_id: int
    region_id: int
    closed: bool
    eligible: bool
    reason: str
    boundary_world: Any
    loops: tuple[Any, ...]
    tangent_u: Any
    tangent_v: Any
    normal: Any
    chart_origin: Any
    chart_occupancy: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateFRepresentative:
    component_id: int
    region_id: int
    status: str
    reason: str
    surface: Any | None
    uv: Any | None
    tsdf_to_representative: Mapping[str, Any]
    representative_to_support: Mapping[str, Any]
    area: float | None = None


@dataclass(frozen=True)
class CandidateFResult:
    samples: TSDFVisibleSurfaceSamples
    association: NearestGaussianAssociation
    gaussian_membership: GaussianRegionMembership
    support: RegionOwnedTSDFSupport
    components: tuple[ObservedSupportComponent, ...]
    boundaries: tuple[ObservedSupportBoundary, ...]
    representatives: tuple[CandidateFRepresentative, ...]
    component_ids: Any
    accounting: Mapping[str, Any]


def _lookup_sorted(keys: Any, query: Any) -> tuple[Any, Any]:
    """Return query positions and exact-presence mask for sorted int64 keys."""

    positions = torch.searchsorted(keys, query)
    present = positions < keys.numel()
    safe = torch.clamp(positions, max=max(int(keys.numel()) - 1, 0))
    if keys.numel():
        present = present & (keys[safe] == query)
    else:
        present = torch.zeros_like(present, dtype=torch.bool)
    return safe, present


def _finite_normal_from_corners(corner_values: Any, h: float) -> Any:
    values = corner_values.reshape(-1, 8)
    # The trilinear gradient at the cell centre is the average of opposite
    # face differences.  It is an observation-derived normal, not a Gaussian
    # normal and not a PCA estimate.
    dx = ((values[:, 1] + values[:, 3] + values[:, 5] + values[:, 7]) -
          (values[:, 0] + values[:, 2] + values[:, 4] + values[:, 6])) / (4.0 * h)
    dy = ((values[:, 2] + values[:, 3] + values[:, 6] + values[:, 7]) -
          (values[:, 0] + values[:, 1] + values[:, 4] + values[:, 5])) / (4.0 * h)
    dz = ((values[:, 4] + values[:, 5] + values[:, 6] + values[:, 7]) -
          (values[:, 0] + values[:, 1] + values[:, 2] + values[:, 3])) / (4.0 * h)
    return _normalize_rows(torch.stack((dx, dy, dz), dim=1))


def _edge_intersections(cells: Any, values: Any, h: float) -> tuple[Any, Any]:
    device = values.device
    offsets = torch.as_tensor(_CORNER_OFFSETS, dtype=torch.float32, device=device)
    edge_points: list[Any] = []
    edge_valid: list[Any] = []
    for start, end in _CUBE_EDGES:
        a = values[:, start]
        b = values[:, end]
        valid = ((a <= 0.0) & (b >= 0.0)) | ((a >= 0.0) & (b <= 0.0))
        denominator = b - a
        t = torch.where(torch.abs(denominator) > 1e-12, -a / denominator, torch.zeros_like(a))
        t = torch.clamp(t, 0.0, 1.0)
        local = offsets[start].unsqueeze(0) + t.unsqueeze(1) * (offsets[end] - offsets[start]).unsqueeze(0)
        edge_points.append((cells.to(torch.float32) + local + 0.5) * h)
        edge_valid.append(valid)
    points = torch.stack(edge_points, dim=1)
    valid = torch.stack(edge_valid, dim=1)
    count = valid.sum(dim=1)
    safe_count = torch.clamp(count, min=1).to(points.dtype).unsqueeze(1)
    return (points * valid.unsqueeze(-1)).sum(dim=1) / safe_count, count


def extract_tsdf_zero_surface_samples(
    field: EvidenceBoundedTSDFField,
    *,
    chunk_size: int = 1_000_000,
    device: str | Any = "cpu",
) -> TSDFVisibleSurfaceSamples:
    """Extract one deterministic zero-surface sample per authoritative cell.

    A cell is eligible only when all eight corners are present in the sparse
    TSDF and the scalar values straddle zero.  Missing/unknown corners never
    become geometry.  Source cell keys are retained as stable sample IDs.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    keys = field.keys
    values = field.value
    support = field.support_count
    offsets = torch.as_tensor(_CORNER_OFFSETS, dtype=torch.int64)
    out_keys: list[Any] = []
    out_cells: list[Any] = []
    out_xyz: list[Any] = []
    out_normals: list[Any] = []
    out_values: list[Any] = []
    out_support: list[Any] = []
    candidate_count = 0
    authoritative_count = 0
    straddle_count = 0
    target_device = torch.device(device)
    for start in range(0, keys.numel(), chunk_size):
        source_keys = keys[start:start + chunk_size]
        cells = decode_cell_keys(source_keys)
        candidate_count += int(source_keys.numel())
        corner_cells = cells[:, None, :] + offsets[None, :, :]
        corner_keys = encode_cell_keys(corner_cells.reshape(-1, 3)).reshape(-1, 8)
        positions, present = _lookup_sorted(keys, corner_keys.reshape(-1))
        present = present.reshape(-1, 8)
        authoritative = torch.all(present, dim=1)
        authoritative_count += int(authoritative.sum())
        if not bool(torch.any(authoritative)):
            continue
        selected_positions = positions.reshape(-1, 8)[authoritative]
        selected_cells = cells[authoritative]
        selected_keys = source_keys[authoritative]
        selected_values = values[selected_positions]
        selected_support = support[selected_positions]
        straddles = (torch.amin(selected_values, dim=1) <= 0.0) & (torch.amax(selected_values, dim=1) >= 0.0)
        straddle_count += int(straddles.sum())
        if not bool(torch.any(straddles)):
            continue
        selected_cells = selected_cells[straddles].to(target_device)
        selected_keys = selected_keys[straddles]
        selected_values = selected_values[straddles].to(target_device)
        selected_support = selected_support[straddles]
        xyz, edge_count = _edge_intersections(selected_cells, selected_values, field.h)
        # Every straddling cell has at least one valid edge under the inclusive
        # endpoint rule.  Keep this guard explicit for NaN-corrupted replays.
        finite = torch.isfinite(xyz).all(dim=1) & (edge_count.to(target_device) > 0)
        xyz = xyz[finite]
        selected_keys = selected_keys[finite]
        selected_cells = selected_cells[finite]
        selected_values = selected_values[finite]
        selected_support = selected_support[finite]
        if xyz.numel() == 0:
            continue
        normals = _finite_normal_from_corners(selected_values, field.h)
        out_keys.append(selected_keys.cpu())
        out_cells.append(selected_cells.cpu())
        out_xyz.append(xyz.cpu())
        out_normals.append(normals.cpu())
        out_values.append(selected_values.cpu())
        out_support.append(selected_support.cpu())
    empty = torch.empty
    if out_keys:
        result = TSDFVisibleSurfaceSamples(
            source_cell_keys=torch.cat(out_keys),
            cell_indices=torch.cat(out_cells),
            world_xyz=torch.cat(out_xyz),
            normals=torch.cat(out_normals),
            corner_values=torch.cat(out_values),
            corner_support_count=torch.cat(out_support),
            h=field.h,
            stats={
                "candidate_cells": candidate_count,
                "authoritative_cells": authoritative_count,
                "zero_straddling_cells": straddle_count,
                "surface_sample_count": len(torch.cat(out_keys)),
                "unknown_corners_excluded": candidate_count - authoritative_count,
                "mesh_intermediate": False,
            },
        )
    else:
        result = TSDFVisibleSurfaceSamples(
            source_cell_keys=empty((0,), dtype=torch.int64),
            cell_indices=empty((0, 3), dtype=torch.int64),
            world_xyz=empty((0, 3), dtype=torch.float32),
            normals=empty((0, 3), dtype=torch.float32),
            corner_values=empty((0, 8), dtype=torch.float32),
            corner_support_count=empty((0, 8), dtype=torch.int32),
            h=field.h,
            stats={
                "candidate_cells": candidate_count,
                "authoritative_cells": authoritative_count,
                "zero_straddling_cells": straddle_count,
                "surface_sample_count": 0,
                "unknown_corners_excluded": candidate_count - authoritative_count,
                "mesh_intermediate": False,
            },
        )
    return result


def associate_tsdf_samples_to_gaussians(
    sample_xyz: Any,
    gaussian_xyz: Any,
    gaussian_ids: Any,
    *,
    chunk_size: int = 131_072,
    torch_pair_limit: int = 8_000_000,
    progress: Any | None = None,
) -> NearestGaussianAssociation:
    """Associate every TSDF sample to its exact nearest Gaussian in Euclidean space.

    There is deliberately no radius or confidence rejection.  For larger
    replays a single-threaded KD-tree is used for memory bounded exact nearest
    neighbour search; ties are resolved by stable Gaussian ID.
    """

    points = _as_cpu_tensor(sample_xyz, dtype=torch.float32).reshape(-1, 3)
    gaussians = _as_cpu_tensor(gaussian_xyz, dtype=torch.float32).reshape(-1, 3)
    ids = _as_cpu_tensor(gaussian_ids, dtype=torch.int64).reshape(-1)
    if gaussians.shape[0] != ids.shape[0] or gaussians.shape[0] == 0:
        raise ValueError("gaussian_xyz and gaussian_ids must be non-empty and aligned")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    order = torch.argsort(ids, stable=True)
    sorted_gaussians = gaussians[order]
    sorted_ids = ids[order]
    sample_count = int(points.shape[0])
    gaussian_count = int(gaussians.shape[0])
    if sample_count * gaussian_count <= torch_pair_limit:
        nearest_sorted: list[Any] = []
        nearest_dist: list[Any] = []
        for start in range(0, sample_count, chunk_size):
            distances = torch.cdist(points[start:start + chunk_size], sorted_gaussians)
            distance, index = torch.min(distances, dim=1)
            nearest_sorted.append(index)
            nearest_dist.append(distance)
            if progress is not None:
                progress(f"association {min(start + chunk_size, sample_count):,}/{sample_count:,}")
        sorted_index = torch.cat(nearest_sorted) if nearest_sorted else torch.empty((0,), dtype=torch.int64)
        distances = torch.cat(nearest_dist) if nearest_dist else torch.empty((0,), dtype=torch.float32)
        backend = "torch_cdist"
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(sorted_gaussians.numpy())
        neighbour_count = min(8, gaussian_count)
        distance_chunks: list[np.ndarray] = []
        index_chunks: list[np.ndarray] = []
        for start in range(0, sample_count, chunk_size):
            chunk_distances, chunk_indices = tree.query(
                points[start:start + chunk_size].numpy(), k=neighbour_count, workers=1
            )
            chunk_distances = np.asarray(chunk_distances, dtype=np.float64)
            chunk_indices = np.asarray(chunk_indices, dtype=np.int64)
            if neighbour_count == 1:
                chunk_distances = chunk_distances.reshape(-1, 1)
                chunk_indices = chunk_indices.reshape(-1, 1)
            minimum = chunk_distances[:, 0]
            tied = chunk_distances == minimum[:, None]
            # Gaussian rows are sorted by stable ID before building the tree,
            # so argmin over sorted-tree indices is the stable-ID tie-break.
            tie_rank = np.where(tied, chunk_indices, np.iinfo(np.int64).max)
            chosen = tie_rank.argmin(axis=1)
            index_chunks.append(chunk_indices[np.arange(chunk_indices.shape[0]), chosen])
            distance_chunks.append(minimum)
            if progress is not None:
                progress(f"association {min(start + chunk_size, sample_count):,}/{sample_count:,}")
        sorted_index = torch.from_numpy(np.concatenate(index_chunks).astype(np.int64, copy=False))
        distances = torch.from_numpy(np.concatenate(distance_chunks).astype(np.float32, copy=False))
        backend = "scipy_cKDTree"
    original_index = order[sorted_index] if sorted_index.numel() else torch.empty((0,), dtype=torch.int64)
    nearest_ids = ids[original_index] if original_index.numel() else torch.empty((0,), dtype=torch.int64)
    distance_np = distances.numpy()
    stats = {
        "sample_count": sample_count,
        "gaussian_count": gaussian_count,
        "backend": backend,
        "distance_min": float(distance_np.min()) if distance_np.size else None,
        "distance_median": float(np.median(distance_np)) if distance_np.size else None,
        "distance_p95": float(np.percentile(distance_np, 95)) if distance_np.size else None,
        "distance_p99": float(np.percentile(distance_np, 99)) if distance_np.size else None,
        "distance_max": float(distance_np.max()) if distance_np.size else None,
        "rejection_radius": None,
        "tie_break": "smallest_stable_gaussian_id_among_returned_exact_minima",
    }
    return NearestGaussianAssociation(original_index, nearest_ids, distances, backend, stats)


def gaussian_region_membership_from_partition(partition: Any) -> GaussianRegionMembership:
    """Map region-coherent 2DGS partition roles to Candidate F statuses."""

    from osn_gs.surface.torch_region_coherent_surfel_partition import (
        PARTITION_ROLES,
        ROLE_ISOLATED_FALLBACK,
        ROLE_OWNERSHIP_PROPAGATED,
        ROLE_STRUCTURAL_CORE,
    )

    region_ids = _as_cpu_tensor(partition.subset_ids, dtype=torch.int64).reshape(-1)
    roles = _as_cpu_tensor(partition.partition_role, dtype=torch.int64).reshape(-1)
    ambiguous = _as_cpu_tensor(partition.ambiguous_multi_region, dtype=torch.bool).reshape(-1)
    role_structural = PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)
    role_propagated = PARTITION_ROLES.index(ROLE_OWNERSHIP_PROPAGATED)
    role_isolated = PARTITION_ROLES.index(ROLE_ISOLATED_FALLBACK)
    status: list[str] = []
    for role, is_ambiguous in zip(roles.tolist(), ambiguous.tolist()):
        if role == role_structural:
            status.append(MEMBERSHIP_CORE)
        elif role == role_propagated:
            status.append(MEMBERSHIP_AMBIGUOUS if is_ambiguous else MEMBERSHIP_ATTACHED)
        elif role == role_isolated:
            status.append(MEMBERSHIP_UNASSIGNED)
        else:
            status.append(MEMBERSHIP_REJECTED)
    accepted = torch.as_tensor([item in (MEMBERSHIP_CORE, MEMBERSHIP_ATTACHED) for item in status], dtype=torch.bool)
    rejected_merge_mask = getattr(partition, "rejected_merge_mask", None)
    rejected_merges = int(_as_cpu_tensor(rejected_merge_mask, dtype=torch.bool).sum()) if rejected_merge_mask is not None else 0
    role_counts = {name: int(sum(item == name for item in status)) for name in (
        MEMBERSHIP_CORE, MEMBERSHIP_ATTACHED, MEMBERSHIP_AMBIGUOUS, MEMBERSHIP_REJECTED, MEMBERSHIP_UNASSIGNED
    )}
    accounting = {
        "gaussian_count": int(region_ids.numel()),
        "region_count": int(torch.unique(region_ids).numel()) if region_ids.numel() else 0,
        "accepted_gaussian_count": int(accepted.sum()),
        "role_counts": role_counts,
        "rejected_merge_count": rejected_merges,
        "identity_source": "2DGS_region_coherent_partition",
        "normal_source": "2DGS_surfel_intrinsic_normal",
        "rejection_threshold": None,
    }
    return GaussianRegionMembership(region_ids, tuple(status), accepted, accounting)


def assign_region_owned_tsdf_support(
    association: NearestGaussianAssociation,
    gaussian_membership: GaussianRegionMembership,
) -> RegionOwnedTSDFSupport:
    nearest_region = gaussian_membership.region_ids[association.nearest_gaussian_index]
    nearest_status = tuple(gaussian_membership.status[int(index)] for index in association.nearest_gaussian_index.tolist())
    accepted = gaussian_membership.accepted_mask[association.nearest_gaussian_index]
    owned_region = torch.where(nearest_region >= 0, nearest_region, torch.full_like(nearest_region, -1))
    owned_region = torch.where(accepted, owned_region, torch.full_like(owned_region, -1))
    statuses = tuple(status if ok else UNOWNED_TSDF_SUPPORT for status, ok in zip(nearest_status, accepted.tolist()))
    counts = {name: int(sum(status == name for status in statuses)) for name in (
        MEMBERSHIP_CORE, MEMBERSHIP_ATTACHED, MEMBERSHIP_AMBIGUOUS, MEMBERSHIP_REJECTED,
        MEMBERSHIP_UNASSIGNED, UNOWNED_TSDF_SUPPORT
    )}
    return RegionOwnedTSDFSupport(
        nearest_region_id=nearest_region,
        nearest_membership_status=nearest_status,
        owned_region_id=owned_region,
        membership_status=statuses,
        accepted_mask=accepted,
        accounting={
            "surface_sample_count": int(accepted.numel()),
            "owned_sample_count": int(accepted.sum()),
            "unowned_sample_count": int((~accepted).sum()),
            "membership_status_counts": counts,
            "ownership_rule": "nearest_gaussian_euclidean_then_existing_gaussian_region_id",
            "synthetic_bridging": False,
        },
    )


def build_native_tsdf_support_components(
    samples: TSDFVisibleSurfaceSamples,
    support: RegionOwnedTSDFSupport,
) -> tuple[tuple[ObservedSupportComponent, ...], Any]:
    """Build connected components from adjacent native TSDF cells only."""

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = int(samples.source_cell_keys.numel())
    accepted = support.accepted_mask.numpy().astype(bool, copy=False)
    region = support.owned_region_id.numpy()
    cells = samples.cell_indices.numpy()
    keys = samples.source_cell_keys.numpy()
    component_ids = np.full((n,), -1, dtype=np.int64)
    if n == 0 or not accepted.any():
        return tuple(), torch.from_numpy(component_ids)
    edge_a_chunks: list[np.ndarray] = []
    edge_b_chunks: list[np.ndarray] = []
    accepted_indices = np.flatnonzero(accepted)
    # The field keys are sorted.  Search each of the three positive face
    # neighbours in bulk, avoiding a Python dictionary whose memory would be
    # needlessly large for the real replay cache.
    for stride in (STRIDE_X, STRIDE_Y, STRIDE_Z):
        target_keys = keys[accepted_indices] + stride
        positions = np.searchsorted(keys, target_keys, side="left")
        found = positions < n
        safe = np.minimum(positions, max(n - 1, 0))
        found &= keys[safe] == target_keys
        found &= accepted[safe]
        found &= region[safe] == region[accepted_indices]
        edge_a_chunks.append(accepted_indices[found])
        edge_b_chunks.append(safe[found])
    if edge_a_chunks:
        edge_a = np.concatenate(edge_a_chunks)
        edge_b = np.concatenate(edge_b_chunks)
        local_a = np.searchsorted(accepted_indices, edge_a).astype(np.int32, copy=False)
        local_b = np.searchsorted(accepted_indices, edge_b).astype(np.int32, copy=False)
        # Only positive face neighbours are stored.  csgraph's undirected
        # traversal treats this as an undirected relation, so explicitly
        # duplicating every edge would double the peak CSR memory for the real
        # 21M-sample replay without changing the component labels.
        graph = csr_matrix(
            (np.ones(local_a.shape[0], dtype=np.uint8), (local_a, local_b)),
            shape=(len(accepted_indices), len(accepted_indices)),
        )
    else:
        graph = csr_matrix((len(accepted_indices), len(accepted_indices)))
    count, labels = connected_components(graph, directed=False, return_labels=True)
    # Group labels by one stable sort instead of rescanning the full 21M-row
    # label array once per component.  The latter is quadratic when the
    # evidence contains many disconnected native supports.
    label_order = np.argsort(labels, kind="stable")
    sorted_labels = labels[label_order]
    split_points = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(label_order, split_points)
    records: list[tuple[int, int, np.ndarray]] = []
    for members_local in groups:
        members = accepted_indices[members_local]
        parent_region = int(region[members[0]])
        records.append((parent_region, int(keys[members].min()), members))
    records.sort(key=lambda item: (item[0], item[1]))
    components: list[ObservedSupportComponent] = []
    for canonical_id, (parent_region, _, members) in enumerate(records):
        component_ids[members] = canonical_id
        min_cell = tuple(int(value) for value in cells[members].min(axis=0).tolist())
        max_cell = tuple(int(value) for value in cells[members].max(axis=0).tolist())
        components.append(ObservedSupportComponent(
            component_id=canonical_id,
            region_id=parent_region,
            sample_indices=torch.from_numpy(members.astype(np.int64, copy=False)),
            min_cell=min_cell,
            max_cell=max_cell,
        ))
    return tuple(components), torch.from_numpy(component_ids)


def _trace_grid_boundary(mask: np.ndarray) -> list[np.ndarray]:
    """Trace oriented boundaries of a union of occupied unit chart cells."""

    height, width = mask.shape
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    outgoing: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            # Vertices use (u, v), with v increasing downward in this local
            # integer chart.  Each edge keeps the occupied cell on its left.
            candidates = (
                ((col, row), (col + 1, row)),
                ((col + 1, row), (col + 1, row + 1)),
                ((col + 1, row + 1), (col, row + 1)),
                ((col, row + 1), (col, row)),
            )
            neighbours = ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))
            for edge, neighbour in zip(candidates, neighbours):
                nr, nc = neighbour
                if nr < 0 or nr >= height or nc < 0 or nc >= width or not mask[nr, nc]:
                    edges.add(edge)
                    outgoing.setdefault(edge[0], set()).add(edge[1])
    loops: list[np.ndarray] = []
    while edges:
        start, end = min(edges)
        edges.remove((start, end))
        outgoing[start].remove(end)
        chain = [start, end]
        current = end
        while current != start:
            options = outgoing.get(current)
            if not options:
                break
            edge = (current, min(options))
            edges.remove(edge)
            options.remove(edge[1])
            current = edge[1]
            chain.append(current)
        if current == start and len(chain) >= 4:
            loops.append(np.asarray(chain[:-1], dtype=np.int64))
    loops.sort(key=lambda loop: (-abs(float(_signed_area(loop))), tuple(loop[0].tolist())))
    return loops


def _signed_area(loop: np.ndarray) -> float:
    if len(loop) < 3:
        return 0.0
    x = loop[:, 0].astype(np.float64)
    y = loop[:, 1].astype(np.float64)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _canonical_chart_frame(points: Any, normals: Any) -> tuple[Any, Any, Any, Any]:
    center = points.mean(dim=0)
    normal_sum = normals.sum(dim=0)
    if float(torch.linalg.vector_norm(normal_sum)) <= 1e-12:
        normal = _normalize_rows(normals[:1])[0]
    else:
        normal = _normalize_rows(normal_sum.reshape(1, 3))[0]
    if float(torch.linalg.vector_norm(normal)) <= 1e-12:
        normal = torch.tensor((0.0, 0.0, 1.0), dtype=points.dtype)
    axis_order = sorted(range(3), key=lambda axis: (abs(float(normal[axis])), axis))
    reference = torch.zeros((3,), dtype=points.dtype)
    reference[axis_order[0]] = 1.0
    tangent_u = _normalize_rows(torch.cross(reference, normal, dim=0).reshape(1, 3))[0]
    tangent_v = _normalize_rows(torch.cross(normal, tangent_u, dim=0).reshape(1, 3))[0]
    # Canonical signs make equivalent runs independent of row order.
    for vector in (tangent_u, tangent_v):
        dominant = int(torch.argmax(torch.abs(vector)))
        if float(vector[dominant]) < 0.0:
            vector.mul_(-1.0)
    return center, tangent_u, tangent_v, normal


def derive_native_support_boundary(
    samples: TSDFVisibleSurfaceSamples,
    component: ObservedSupportComponent,
    *,
    max_chart_extent: int = 8192,
) -> ObservedSupportBoundary:
    """Create a boundary-first chart from a native TSDF support component."""

    indices = component.sample_indices.to(torch.int64)
    if int(indices.numel()) < 3:
        # A one- or two-cell native component cannot produce a closed
        # three-vertex boundary and can never satisfy the frozen 8x4 control
        # grid.  Record the same explicit abstain reason without allocating a
        # per-component chart frame for potentially hundreds of thousands of
        # tiny components.
        empty = torch.empty((0, 3), dtype=torch.float32)
        return ObservedSupportBoundary(
            component.component_id, component.region_id, False, False,
            "insufficient_native_tsdf_support_for_boundary", empty, tuple(),
            torch.zeros((3,), dtype=torch.float32), torch.zeros((3,), dtype=torch.float32),
            torch.zeros((3,), dtype=torch.float32), torch.zeros((3,), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.bool),
            {"native_cell_count": int(indices.numel()), "source": "native_tsdf_cell_face_adjacency"},
        )
    points = samples.world_xyz[indices]
    normals = samples.normals[indices]
    center, tangent_u, tangent_v, normal = _canonical_chart_frame(points, normals)
    projection = torch.stack(((points - center) @ tangent_u, (points - center) @ tangent_v), dim=1)
    # The frozen WL139 family has an 8-by-4 control grid.  Orient the local
    # boundary chart so its longer native span is the first parameter axis;
    # this is a local chart-frame convention, not a scene-level PCA/extrema
    # search and it leaves the support topology unchanged.
    span_u = float(projection[:, 0].amax() - projection[:, 0].amin())
    span_v = float(projection[:, 1].amax() - projection[:, 1].amin())
    if span_u < span_v:
        tangent_u, tangent_v = tangent_v, tangent_u
        projection = torch.stack(((points - center) @ tangent_u, (points - center) @ tangent_v), dim=1)
    # Cell centres can land exactly on half-integers relative to the support
    # centroid.  ``torch.round`` uses ties-to-even and would collapse every
    # other row/column in that case; half-up quantization keeps the native
    # chart lattice deterministic and one-to-one for regular supports.
    grid = torch.floor(projection / samples.h + 0.5).to(torch.int64)
    low = grid.amin(dim=0)
    high = grid.amax(dim=0)
    extent = high - low + 1
    if int(extent.max()) > max_chart_extent:
        empty = torch.empty((0, 3), dtype=torch.float32)
        return ObservedSupportBoundary(component.component_id, component.region_id, False, False,
                                       "boundary_chart_extent_exceeds_memory_guard", empty, tuple(),
                                       tangent_u, tangent_v, normal, center, torch.empty((0, 0), dtype=torch.bool),
                                       {"native_cell_count": int(indices.numel())})
    occupancy = np.zeros((int(extent[1]), int(extent[0])), dtype=bool)
    shifted = (grid - low).numpy()
    occupancy[shifted[:, 1], shifted[:, 0]] = True
    loops_grid = _trace_grid_boundary(occupancy)
    base = center + low[0].to(center.dtype) * samples.h * tangent_u + low[1].to(center.dtype) * samples.h * tangent_v
    loops_world: list[Any] = []
    for loop in loops_grid:
        uv = torch.from_numpy(loop).to(center.dtype)
        loops_world.append(base[None, :] + samples.h * (uv[:, 0:1] * tangent_u[None, :] + uv[:, 1:2] * tangent_v[None, :]))
    if not loops_world:
        reason = "native_support_boundary_not_closed"
        boundary_world = torch.empty((0, 3), dtype=torch.float32)
        eligible = False
    elif len(loops_world) != 1:
        reason = "native_support_boundary_has_multiple_loops"
        boundary_world = loops_world[0]
        eligible = False
    elif loops_world[0].shape[0] < 3:
        reason = "native_support_boundary_has_fewer_than_three_vertices"
        boundary_world = loops_world[0]
        eligible = False
    else:
        reason = "ok"
        boundary_world = loops_world[0]
        eligible = True
    return ObservedSupportBoundary(
        component_id=component.component_id,
        region_id=component.region_id,
        closed=bool(loops_world) and len(loops_world) == 1,
        eligible=eligible,
        reason=reason,
        boundary_world=boundary_world,
        loops=tuple(loops_world),
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normal=normal,
        chart_origin=base,
        chart_occupancy=torch.from_numpy(occupancy),
        provenance={
            "source": "native_tsdf_cell_adjacency",
            "native_cell_count": int(indices.numel()),
            "chart_unique_u": int(torch.unique(grid[:, 0]).numel()),
            "chart_unique_v": int(torch.unique(grid[:, 1]).numel()),
            "chart_grid_min": [int(value) for value in low.tolist()],
            "chart_grid_max": [int(value) for value in high.tolist()],
            "synthetic_bridge": False,
            "global_pca": False,
        },
    )


def _surface_area(surface: Any, *, resolution: int = 32) -> float:
    u = torch.linspace(0.0, 1.0, resolution, dtype=torch.float32)
    v = torch.linspace(0.0, 1.0, resolution, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=1)
    values, du, dv = surface.evaluate_with_derivatives(uv)
    values = values.reshape(resolution, resolution, 3)
    du = du.reshape(resolution, resolution, 3)
    dv = dv.reshape(resolution, resolution, 3)
    step_u = 1.0 / max(resolution - 1, 1)
    step_v = 1.0 / max(resolution - 1, 1)
    return float((torch.linalg.vector_norm(torch.cross(du, dv, dim=-1), dim=-1) * step_u * step_v).sum())


def _metric_summary(values: Any) -> dict[str, Any]:
    tensor = _as_cpu_tensor(values, dtype=torch.float32).reshape(-1)
    finite = torch.isfinite(tensor)
    array = tensor[finite].numpy()
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "p95": float(np.percentile(array, 95)) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def fit_boundary_first_region_representative(
    samples: TSDFVisibleSurfaceSamples,
    component: ObservedSupportComponent,
    boundary: ObservedSupportBoundary,
    *,
    resolution_u: int = 8,
    resolution_v: int = 4,
    degree_u: int = 2,
    degree_v: int = 2,
    smoothness_lambda: float = 1e-4,
    tikhonov_lambda: float = 1e-4,
    correction_rounds: int = 2,
    chunk_size: int = 8192,
    projection_iterations: int = 2,
) -> CandidateFRepresentative:
    """Fit the approved WL139 LSQ family after boundary/chart validation."""

    point_indices = component.sample_indices.to(torch.int64)
    points = samples.world_xyz[point_indices]
    if not boundary.eligible:
        return CandidateFRepresentative(component.component_id, component.region_id, ABSTAIN_REPRESENTATIVE,
                                        boundary.reason, None, None, {}, {})
    control_count = resolution_u * resolution_v
    if points.shape[0] < control_count:
        return CandidateFRepresentative(component.component_id, component.region_id, ABSTAIN_REPRESENTATIVE,
                                        "insufficient_native_tsdf_support_for_control_grid", None, None, {}, {})
    projected = torch.stack(((points - boundary.chart_origin) @ boundary.tangent_u,
                             (points - boundary.chart_origin) @ boundary.tangent_v), dim=1)
    boundary_projected = torch.stack(((boundary.boundary_world - boundary.chart_origin) @ boundary.tangent_u,
                                      (boundary.boundary_world - boundary.chart_origin) @ boundary.tangent_v), dim=1)
    mins = boundary_projected.amin(dim=0)
    maxs = boundary_projected.amax(dim=0)
    span = maxs - mins
    if bool(torch.any(span <= 1e-12)):
        return CandidateFRepresentative(component.component_id, component.region_id, ABSTAIN_REPRESENTATIVE,
                                        "degenerate_boundary_chart_span", None, None, {}, {})
    initial_uv = (projected - mins) / span
    from osn_gs.surface.torch_nurbs import TorchNURBSSurface, fit_torch_visible_surface_lsq

    # Boundary-derived UVs must be able to identify the requested control
    # grid.  A rank-deficient chart is a principled abstain, not a reason to
    # silently lower the model capacity or invent continuation.
    rank_probe = TorchNURBSSurface(
        control_grid=torch.zeros((resolution_u, resolution_v, 3), dtype=points.dtype),
        weights=torch.ones((resolution_u, resolution_v), dtype=points.dtype),
        degree_u=degree_u,
        degree_v=degree_v,
    )
    basis_u, basis_v = rank_probe._basis_values(initial_uv)
    design = torch.einsum("qi,qj->qij", basis_u, basis_v).reshape(points.shape[0], -1)
    rank = int(torch.linalg.matrix_rank(design).item()) if design.numel() else 0
    required_rank = min(control_count, int(points.shape[0]))
    if rank < required_rank:
        return CandidateFRepresentative(component.component_id, component.region_id, ABSTAIN_REPRESENTATIVE,
                                        "boundary_chart_design_matrix_rank_deficient", None, None,
                                        {"design_matrix_rank": rank, "required_rank": required_rank}, {})

    # The frozen WL139 family is used only after the TSDF-derived boundary
    # chart has established the parameter domain.
    fit_result = fit_torch_visible_surface_lsq(
        points,
        resolution_u=resolution_u,
        resolution_v=resolution_v,
        degree_u=degree_u,
        degree_v=degree_v,
        smoothness_lambda=smoothness_lambda,
        tikhonov_lambda=tikhonov_lambda,
        correction_rounds=correction_rounds,
        chunk_size=chunk_size,
        projection_iterations=projection_iterations,
        initial_uv=initial_uv,
        collect_diagnostics=True,
    )
    surface, uv, diagnostics = fit_result
    fitted = surface.evaluate(uv)
    tsdf_error = torch.linalg.vector_norm(fitted - points, dim=1)
    _, du, dv = surface.evaluate_with_derivatives(uv)
    fitted_normals = _normalize_rows(torch.cross(du, dv, dim=1))
    normal_cos = torch.abs(torch.sum(fitted_normals * samples.normals[point_indices], dim=1))
    normal_degrees = torch.rad2deg(torch.arccos(torch.clamp(normal_cos, -1.0, 1.0)))
    try:
        area = _surface_area(surface)
    except Exception:
        area = None
    from scipy.spatial import cKDTree

    support_tree = cKDTree(points.numpy())
    grid_u = torch.linspace(0.0, 1.0, 32)
    grid_v = torch.linspace(0.0, 1.0, 16)
    mesh_u, mesh_v = torch.meshgrid(grid_u, grid_v, indexing="ij")
    representative_points = surface.evaluate(torch.stack((mesh_u.reshape(-1), mesh_v.reshape(-1)), dim=1)).detach().cpu().numpy()
    support_distances, _ = support_tree.query(representative_points, k=1, workers=1)
    return CandidateFRepresentative(
        component_id=component.component_id,
        region_id=component.region_id,
        status=MATERIALIZED_REPRESENTATIVE,
        reason="ok",
        surface=surface,
        uv=uv,
        tsdf_to_representative=_metric_summary(tsdf_error),
        representative_to_support=_metric_summary(torch.from_numpy(np.asarray(support_distances, dtype=np.float32))),
        area=area,
    )


def run_candidate_f(
    samples: TSDFVisibleSurfaceSamples,
    gaussian_xyz: Any,
    gaussian_ids: Any,
    partition: Any,
    *,
    association_chunk_size: int = 131_072,
    torch_pair_limit: int = 8_000_000,
    fit_kwargs: Mapping[str, Any] | None = None,
    progress: Any | None = None,
) -> CandidateFResult:
    """Run Candidate F from already-derived Gaussian partition and TSDF samples."""

    membership = gaussian_region_membership_from_partition(partition)
    association = associate_tsdf_samples_to_gaussians(
        samples.world_xyz, gaussian_xyz, gaussian_ids,
        chunk_size=association_chunk_size,
        torch_pair_limit=torch_pair_limit,
        progress=progress,
    )
    support = assign_region_owned_tsdf_support(association, membership)
    components, component_ids = build_native_tsdf_support_components(samples, support)
    boundaries = tuple(derive_native_support_boundary(samples, component) for component in components)
    kwargs = dict(fit_kwargs or {})
    representatives = tuple(
        fit_boundary_first_region_representative(samples, component, boundary, **kwargs)
        for component, boundary in zip(components, boundaries)
    )
    region_component_counts: dict[str, int] = {}
    for component in components:
        region_component_counts[str(component.region_id)] = region_component_counts.get(str(component.region_id), 0) + 1
    accounting = {
        "candidate": "F",
        "schema_version": 1,
        "gaussian_branch": dict(membership.accounting),
        "tsdf_branch": dict(samples.stats),
        "association": dict(association.stats),
        "region_owned_support": dict(support.accounting),
        "topology": {
            "component_count": len(components),
            "region_component_counts": region_component_counts,
            "adjacency": "native_tsdf_cell_face_adjacency",
            "synthetic_bridging": False,
            "mesh_intermediate": False,
        },
        "boundary": {
            "eligible_count": int(sum(boundary.eligible for boundary in boundaries)),
            "ineligible_count": int(sum(not boundary.eligible for boundary in boundaries)),
            "reasons": {reason: sum(boundary.reason == reason for boundary in boundaries)
                        for reason in sorted({boundary.reason for boundary in boundaries})},
        },
        "representative": {
            "materialized_count": int(sum(item.status == MATERIALIZED_REPRESENTATIVE for item in representatives)),
            "abstained_count": int(sum(item.status == ABSTAIN_REPRESENTATIVE for item in representatives)),
            "statuses": {status: sum(item.status == status for item in representatives)
                         for status in sorted({item.status for item in representatives})},
            "family": "WL139_boundary_chart_seeded_visible_surface_lsq",
        },
        "forbidden_paths": {
            "tsdf_identity_rediscovery": False,
            "gaussian_center_as_observed_surface": False,
            "rejection_radius": False,
            "normal_matching_or_vote": False,
            "synthetic_bridge": False,
            "global_pca_or_extrema": False,
            "latent_continuation": False,
            "event_1527_blacklist": False,
            "new_classifier_or_trust": False,
        },
    }
    return CandidateFResult(samples, association, membership, support, components, boundaries,
                            representatives, component_ids, accounting)


def representative_to_json(representative: CandidateFRepresentative) -> dict[str, Any]:
    return {
        "component_id": representative.component_id,
        "region_id": representative.region_id,
        "status": representative.status,
        "reason": representative.reason,
        "tsdf_to_representative": dict(representative.tsdf_to_representative),
        "representative_to_support": dict(representative.representative_to_support),
        "area": representative.area,
        "surface_materialized": representative.surface is not None,
    }


__all__ = [
    "ABSTAIN_REPRESENTATIVE",
    "EvidenceBoundedTSDFField",
    "GaussianRegionMembership",
    "MATERIALIZED_REPRESENTATIVE",
    "NearestGaussianAssociation",
    "ObservedSupportBoundary",
    "ObservedSupportComponent",
    "RegionOwnedTSDFSupport",
    "TSDFVisibleSurfaceSamples",
    "UNOWNED_TSDF_SUPPORT",
    "associate_tsdf_samples_to_gaussians",
    "assign_region_owned_tsdf_support",
    "build_native_tsdf_support_components",
    "decode_cell_keys",
    "derive_native_support_boundary",
    "encode_cell_keys",
    "extract_tsdf_zero_surface_samples",
    "fit_boundary_first_region_representative",
    "gaussian_region_membership_from_partition",
    "representative_to_json",
    "run_candidate_f",
]
