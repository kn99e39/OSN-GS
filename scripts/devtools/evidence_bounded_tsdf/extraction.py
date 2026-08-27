from __future__ import annotations

"""Worklog 127 -- MASKED ZERO LEVEL-SET EXTRACTION (directive section 7).

A grid cell is eligible for surface extraction ONLY IF

    all 8 cell corners have field authority
    AND min(phi_corners) <= 0 <= max(phi_corners).

Missing corner values are never synthesized, UNKNOWN cells are never
interpolated through, and no mesh hole filling / repair / watertightness forcing
runs anywhere. The output is allowed to be open, disconnected and fragmented.

HOW THE MASK IS ENFORCED. `skimage.measure.marching_cubes` accepts a `mask`,
but its mask is tested at a SINGLE cell corner, not all eight (measured
directly, see `tests/test_evidence_bounded_projective_tsdf.py`), so it does not
implement the contract above. Instead this module

  1. fills UNKNOWN corners with a sentinel far outside the phi range,
  2. runs the standard Lewiner marching cubes per dense block, and
  3. DISCARDS every triangle whose owning cell is not eligible.

Marching cubes is per-cell -- the triangles of a cell are a function of that
cell's own 8 corner values alone -- so the kept triangles are provably identical
to a true 8-corner-masked extraction, whatever sentinel is used. A regression
test asserts exactly that by running the extraction twice with opposite
sentinels and comparing the kept triangles bitwise.

The sentinel exists ONLY so a general-purpose implementation can be called; no
sentinel value ever reaches an extracted triangle, is exported, or is treated as
a field value. UNKNOWN stays UNKNOWN.
"""

from dataclasses import dataclass, field as dataclass_field
from typing import Any

import numpy as np
import torch

from .field import SparseProjectiveTSDF, decode_keys

# Outside the phi range [-1, +1] on the camera-facing side. Any value works; see
# the module docstring and `test_sentinel_choice_cannot_change_kept_triangles`.
UNKNOWN_SENTINEL = 2.0


@dataclass
class ExtractedSurface:
    vertices: np.ndarray            # (V, 3) float64 world space
    faces: np.ndarray               # (F, 3) int64
    vertex_support_count: np.ndarray  # (V,) int32 -- nearest-lattice-corner support
    vertex_field_value: np.ndarray    # (V,) float32 -- nearest-lattice-corner phi
    h: float
    stats: dict[str, Any] = dataclass_field(default_factory=dict)


def cell_corner_offsets() -> np.ndarray:
    return np.array(
        [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=np.int64
    )


def _block_keys(block_index: torch.Tensor, block: int, span: int) -> torch.Tensor:
    """(B, span, span, span) voxel index grid for each block, as decoded ints."""

    device = block_index.device
    offsets = torch.arange(span, dtype=torch.int64, device=device)
    grid = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing="ij"), dim=-1)
    return block_index.reshape(-1, 1, 1, 1, 3) * block + grid.reshape(1, span, span, span, 3)


def _encode(index: torch.Tensor) -> torch.Tensor:
    from .field import KEY_BOUND, _AXIS_SPAN

    return ((index[..., 0] + KEY_BOUND) * _AXIS_SPAN + (index[..., 1] + KEY_BOUND)) * _AXIS_SPAN + (
        index[..., 2] + KEY_BOUND
    )


def candidate_blocks(field: SparseProjectiveTSDF, block: int) -> torch.Tensor:
    """Every block that could OWN an eligible cell. Cells owned by block b span
    voxel indices [b*block, b*block + block], so a block is a candidate if it or
    any of its seven `-1` neighbours contains an authoritative voxel. Empty
    blocks simply produce nothing."""

    index = decode_keys(field.keys)
    base = torch.div(index, block, rounding_mode="floor")
    packed = torch.unique(_encode(base))
    unpacked = decode_keys(packed)
    grown = [unpacked]
    for dx in (0, -1):
        for dy in (0, -1):
            for dz in (0, -1):
                if dx == dy == dz == 0:
                    continue
                shift = torch.tensor([dx, dy, dz], dtype=torch.int64, device=unpacked.device)
                grown.append(unpacked + shift)
    return decode_keys(torch.unique(_encode(torch.cat(grown, dim=0))))


def extract_zero_level_set(
    field: SparseProjectiveTSDF, *, block: int = 64, batch_blocks: int = 8,
    sentinel: float = UNKNOWN_SENTINEL, progress=None,
) -> ExtractedSurface:
    from skimage.measure import marching_cubes

    span = block + 1
    blocks = candidate_blocks(field, block)
    total_blocks = int(blocks.shape[0])
    device = field.keys.device

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    values: list[np.ndarray] = []
    eligible_cells = 0
    authoritative_cells = 0
    discarded_triangles = 0
    mislocated_triangles = 0
    processed_blocks = 0
    vertex_base = 0

    for start in range(0, total_blocks, batch_blocks):
        batch = blocks[start : start + batch_blocks]
        index = _block_keys(batch, block, span)
        keys = _encode(index)
        value, count, found = field.lookup(keys.reshape(-1))
        shape = (int(batch.shape[0]), span, span, span)
        value = value.reshape(shape)
        count = count.reshape(shape)
        found = found.reshape(shape)

        # 8-corner authority and the sign-change requirement, computed here and
        # nowhere else. Cell (i, j, k) uses corners [i:i+2, j:j+2, k:k+2].
        corner_ok = torch.ones((shape[0], block, block, block), dtype=torch.bool, device=device)
        filled = torch.where(found, value, torch.full_like(value, float(sentinel)))
        cell_min = torch.full((shape[0], block, block, block), float("inf"), device=device)
        cell_max = torch.full((shape[0], block, block, block), float("-inf"), device=device)
        for dx, dy, dz in cell_corner_offsets():
            sl = (slice(None), slice(dx, dx + block), slice(dy, dy + block), slice(dz, dz + block))
            corner_ok &= found[sl]
            cell_min = torch.minimum(cell_min, filled[sl])
            cell_max = torch.maximum(cell_max, filled[sl])
        crossing = (cell_min <= 0.0) & (cell_max >= 0.0)
        eligible = corner_ok & crossing
        authoritative_cells += int(corner_ok.sum().item())
        eligible_cells += int(eligible.sum().item())

        eligible_cpu = eligible.detach().cpu().numpy()
        filled_cpu = filled.detach().cpu().numpy().astype(np.float64)
        count_cpu = count.detach().cpu().numpy()
        value_cpu = value.detach().cpu().numpy()
        origin_cpu = (batch * block).detach().cpu().numpy()

        for b in range(shape[0]):
            if not eligible_cpu[b].any():
                continue
            processed_blocks += 1
            verts, tris, _normals, _vals = marching_cubes(
                filled_cpu[b], level=0.0, step_size=1, method="lewiner", allow_degenerate=False
            )
            if tris.shape[0] == 0:
                continue
            centroid = verts[tris].mean(axis=1)
            cell = np.floor(centroid).astype(np.int64)
            in_range = np.all((cell >= 0) & (cell < block), axis=1)
            cell = np.clip(cell, 0, block - 1)
            keep = eligible_cpu[b][cell[:, 0], cell[:, 1], cell[:, 2]] & in_range
            discarded_triangles += int((~keep).sum())
            corners = verts[tris]
            inside = np.all(
                (corners >= cell[:, None, :] - 1e-9) & (corners <= cell[:, None, :] + 1.0 + 1e-9), axis=(1, 2)
            )
            mislocated_triangles += int((keep & ~inside).sum())
            tris = tris[keep]
            if tris.shape[0] == 0:
                continue
            used, remapped = np.unique(tris.reshape(-1), return_inverse=True)
            local = verts[used]
            lattice = np.clip(np.rint(local).astype(np.int64), 0, span - 1)
            supports.append(count_cpu[b][lattice[:, 0], lattice[:, 1], lattice[:, 2]])
            values.append(value_cpu[b][lattice[:, 0], lattice[:, 1], lattice[:, 2]])
            vertices.append((local + origin_cpu[b][None, :]) * field.h + 0.5 * field.h)
            faces.append(remapped.reshape(-1, 3).astype(np.int64) + vertex_base)
            vertex_base += int(local.shape[0])
        if progress is not None:
            progress(f"extraction {min(start + batch_blocks, total_blocks):,}/{total_blocks:,} blocks")

    if vertices:
        all_vertices = np.concatenate(vertices, axis=0)
        all_faces = np.concatenate(faces, axis=0)
        all_supports = np.concatenate(supports, axis=0)
        all_values = np.concatenate(values, axis=0)
    else:
        all_vertices = np.zeros((0, 3), dtype=np.float64)
        all_faces = np.zeros((0, 3), dtype=np.int64)
        all_supports = np.zeros((0,), dtype=np.int32)
        all_values = np.zeros((0,), dtype=np.float32)

    merged_vertices, merged_faces, merged_supports, merged_values = weld_block_seams(
        all_vertices, all_faces, all_supports, all_values, field.h
    )
    return ExtractedSurface(
        vertices=merged_vertices, faces=merged_faces,
        vertex_support_count=merged_supports.astype(np.int32),
        vertex_field_value=merged_values.astype(np.float32), h=field.h,
        stats={
            "candidate_blocks": total_blocks,
            "blocks_with_eligible_cells": processed_blocks,
            "block_side_voxels": block,
            "cells_with_all_eight_authoritative_corners": authoritative_cells,
            "eligible_cells_authoritative_and_sign_changing": eligible_cells,
            "triangles_discarded_because_cell_not_eligible": discarded_triangles,
            "kept_triangles_outside_their_claimed_cell": mislocated_triangles,
            "vertices_before_seam_weld": int(all_vertices.shape[0]),
            "sentinel": float(sentinel),
        },
    )


def weld_block_seams(
    vertices: np.ndarray, faces: np.ndarray, supports: np.ndarray, values: np.ndarray, h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blocks own DISJOINT cells, so no triangle is duplicated; only the shared
    lattice edges on a block face are computed twice, and marching cubes gives
    them bitwise identical positions from identical corner values. Welding by
    exact quantized position therefore removes seam duplicates only -- it never
    merges distinct geometry and never closes a hole."""

    if vertices.shape[0] == 0:
        return vertices, faces, supports, values
    quantum = h * 1e-6
    keyed = np.rint(vertices / quantum).astype(np.int64)
    order = np.lexsort((keyed[:, 2], keyed[:, 1], keyed[:, 0]))
    ordered = keyed[order]
    starts = np.ones(ordered.shape[0], dtype=bool)
    if ordered.shape[0] > 1:
        starts[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    group = np.cumsum(starts) - 1
    inverse = np.empty(ordered.shape[0], dtype=np.int64)
    inverse[order] = group
    keep = order[starts]
    return vertices[keep], inverse[faces], supports[keep], values[keep]
