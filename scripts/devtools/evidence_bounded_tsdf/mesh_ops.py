from __future__ import annotations

"""Worklog 127 -- MESH MEASUREMENT, RAYCAST AND DISTANCE (directive 11-13, 16).

Geometry only. Nothing here modifies a mesh: there is no hole filling, no
repair, no smoothing, no decimation and no watertightness forcing anywhere in
this module. It measures what extraction produced and casts rays at it.

The "raycast" is a pixel-centre ray/triangle intersection implemented as a
z-buffered rasterization. That is the SAME ray candidate B's frontier is
compared against (the rounded pixel's ray), so the two depths are measured on
identical rays and the comparison is apples to apples. Depth is interpolated in
1/z, which is exact for this projection because the canonical projection's
homogeneous w IS the camera-space z.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .field import CANONICAL_NEAR_N, KEY_BOUND, _AXIS_SPAN


# --------------------------------------------------------------------- basics
def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def connected_components(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Vertex-connectivity components of the extracted mesh. Reported, never
    used to change the mesh."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as scipy_components

    if vertex_count == 0:
        return np.zeros((0,), dtype=np.int64), 0
    if faces.shape[0] == 0:
        return np.arange(vertex_count, dtype=np.int64), vertex_count
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count)).tocsr()
    count, labels = scipy_components(graph, directed=False)
    return labels.astype(np.int64), int(count)


# ------------------------------------------------------------------- PLY output
def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_count, face_count = int(vertices.shape[0]), int(faces.shape[0])
    header = [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {vertex_count}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {face_count}", "property list uchar int vertex_indices", "end_header", ""]
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        positions = np.ascontiguousarray(vertices, dtype="<f4")
        if colors is None:
            handle.write(positions.tobytes())
        else:
            record = np.empty(vertex_count, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            record["p"] = positions
            record["c"] = np.ascontiguousarray(colors, dtype=np.uint8)
            handle.write(record.tobytes())
        if face_count:
            record = np.empty(face_count, dtype=[("n", "u1"), ("i", "<i4", 3)])
            record["n"] = 3
            record["i"] = np.ascontiguousarray(faces, dtype="<i4")
            handle.write(record.tobytes())
    return {"vertices": vertex_count, "faces": face_count}


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(points.shape[0])
    header = [
        "ply", "format binary_little_endian 1.0", f"element vertex {count}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header", "",
    ]
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        record = np.empty(count, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        record["p"] = np.ascontiguousarray(points, dtype="<f4")
        record["c"] = np.ascontiguousarray(colors, dtype=np.uint8)
        handle.write(record.tobytes())
    return count


# ------------------------------------------------- point / triangle distance
def point_triangle_distance(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Exact unsigned distance from (N,3) points to (N,3,3) triangles."""

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = points - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = points - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = torch.where(denom.abs() > 1e-30, denom, torch.ones_like(denom))

    v_int = vb / safe
    w_int = vc / safe
    closest = a + ab * v_int.unsqueeze(-1) + ac * w_int.unsqueeze(-1)

    # vertex / edge regions, applied in the standard priority order
    region_a = (d1 <= 0) & (d2 <= 0)
    region_b = (d3 >= 0) & (d4 <= d3)
    region_c = (d6 >= 0) & (d5 <= d6)
    region_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    region_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    region_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    def _line(p0, direction, numerator, denominator):
        t = numerator / torch.where(denominator.abs() > 1e-30, denominator, torch.ones_like(denominator))
        return p0 + direction * t.clamp(0.0, 1.0).unsqueeze(-1)

    closest = torch.where(region_bc.unsqueeze(-1), _line(b, c - b, d4 - d3, (d4 - d3) + (d5 - d6)), closest)
    closest = torch.where(region_ac.unsqueeze(-1), _line(a, ac, d2, d2 - d6), closest)
    closest = torch.where(region_ab.unsqueeze(-1), _line(a, ab, d1, d1 - d3), closest)
    closest = torch.where(region_c.unsqueeze(-1), c.expand_as(closest), closest)
    closest = torch.where(region_b.unsqueeze(-1), b.expand_as(closest), closest)
    closest = torch.where(region_a.unsqueeze(-1), a.expand_as(closest), closest)
    degenerate = denom.abs() <= 1e-30
    closest = torch.where(degenerate.unsqueeze(-1), a.expand_as(closest), closest)
    return torch.linalg.norm(points - closest, dim=-1)


@dataclass
class TriangleCellIndex:
    """Triangles bucketed by the grid cell that owns them, so a nearest-surface
    query only ever touches a bounded neighbourhood."""

    cell_keys: torch.Tensor       # (C,) ascending unique cell keys
    cell_start: torch.Tensor      # (C,) offset into `order`
    cell_count: torch.Tensor      # (C,)
    order: torch.Tensor           # (F,) triangle ids grouped by cell
    triangles: torch.Tensor       # (F,3,3) float32 vertex positions
    h: float


def build_triangle_cell_index(vertices: torch.Tensor, faces: torch.Tensor, h: float) -> TriangleCellIndex:
    triangles = vertices[faces]
    centroid = triangles.mean(dim=1)
    index = torch.floor(centroid / h).to(torch.int64)
    keys = ((index[:, 0] + KEY_BOUND) * _AXIS_SPAN + (index[:, 1] + KEY_BOUND)) * _AXIS_SPAN + (index[:, 2] + KEY_BOUND)
    order = torch.argsort(keys)
    ordered = keys[order]
    unique, counts = torch.unique_consecutive(ordered, return_counts=True)
    start = torch.cumsum(counts, dim=0) - counts
    return TriangleCellIndex(
        cell_keys=unique, cell_start=start, cell_count=counts, order=order, triangles=triangles, h=h
    )


def _ring_offsets(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    shell = np.max(np.abs(grid), axis=1) == radius
    return grid[shell]


def nearest_surface_distance(
    points: torch.Tensor, index: TriangleCellIndex, *, max_radius: int = 3, chunk: int = 1_000_000,
    progress=None,
) -> torch.Tensor:
    """Distance from each point to the nearest extracted triangle, exact for
    every distance the expanding-ring bound can certify. Points still unresolved
    after `max_radius` rings come back as +inf, which the report calls
    NO LOCAL EXTRACTED SURFACE -- never a filled-in number."""

    h = index.h
    total = int(points.shape[0])
    best = torch.full((total,), float("inf"), dtype=torch.float32, device=points.device)
    if index.cell_keys.numel() == 0:
        return best
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = points[start:stop]
        block_best = best[start:stop]
        base = torch.floor(block / h).to(torch.int64)
        active = torch.arange(block.shape[0], device=block.device)
        for radius in range(0, max_radius + 1):
            if active.numel() == 0:
                break
            for offset in _ring_offsets(radius):
                shift = torch.tensor(offset, dtype=torch.int64, device=block.device)
                cell = base[active] + shift
                key = ((cell[:, 0] + KEY_BOUND) * _AXIS_SPAN + (cell[:, 1] + KEY_BOUND)) * _AXIS_SPAN + (
                    cell[:, 2] + KEY_BOUND
                )
                position = torch.searchsorted(index.cell_keys, key)
                clamped = position.clamp(max=index.cell_keys.numel() - 1)
                found = (position < index.cell_keys.numel()) & (index.cell_keys[clamped] == key)
                counts = torch.where(found, index.cell_count[clamped], torch.zeros_like(clamped))
                if int(counts.sum().item()) == 0:
                    continue
                starts = torch.where(found, index.cell_start[clamped], torch.zeros_like(clamped))
                query_row = torch.repeat_interleave(active, counts)
                run_base = torch.repeat_interleave(starts, counts)
                prefix = torch.cumsum(counts, dim=0) - counts
                within = torch.arange(int(counts.sum().item()), device=block.device) - torch.repeat_interleave(prefix, counts)
                triangle_ids = index.order[run_base + within]
                distance = point_triangle_distance(block[query_row], index.triangles[triangle_ids])
                block_best = block_best.scatter_reduce(0, query_row, distance, reduce="amin")
            # any cell not yet visited is at L-inf cell distance >= radius + 1,
            # so nothing it holds can be closer than radius * h.
            active = active[block_best[active] > radius * h]
        best[start:stop] = block_best
        if progress is not None:
            progress(f"nearest-surface distance {stop:,}/{total:,}")
    return best


# ------------------------------------------------------------------- raycast
def rasterize_mesh_depth(
    camera: Any, vertices: torch.Tensor, faces: torch.Tensor, *,
    tiers: tuple[int, ...] = (2, 4, 16, 64), face_chunk: int = 2_000_000,
) -> tuple[torch.Tensor, dict[str, int]]:
    """First mesh-hit camera-space depth at every pixel centre, or +inf.

    Exact pixel-centre ray/triangle intersection: a pixel is written iff its
    centre is inside the projected triangle, and the written depth is the exact
    intersection depth (linear in 1/z, which is exact for this projection)."""

    width, height = int(camera.image_width), int(camera.image_height)
    device = vertices.device
    depth_buffer = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)

    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    homogeneous = torch.cat([vertices.to(torch.float32), ones], dim=1)
    vertex_depth = (homogeneous @ camera.world_view_transform)[:, 2].contiguous()
    clip = homogeneous @ camera.full_proj_transform
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 0, w, torch.ones_like(w))
    px = ((clip[:, 0] / safe_w + 1.0) * width - 1.0) * 0.5
    py = ((clip[:, 1] / safe_w + 1.0) * height - 1.0) * 0.5
    vertex_ok = (w > 0) & (vertex_depth >= CANONICAL_NEAR_N)
    del clip, homogeneous

    stats = {"triangles_total": int(faces.shape[0]), "triangles_clipped_out": 0,
             "triangles_rasterized": 0, "triangles_beyond_largest_tier": 0}

    for start in range(0, int(faces.shape[0]), face_chunk):
        block = faces[start : start + face_chunk]
        ok = vertex_ok[block].all(dim=1)
        stats["triangles_clipped_out"] += int((~ok).sum().item())
        block = block[ok]
        if block.numel() == 0:
            continue
        x = px[block]
        y = py[block]
        z = vertex_depth[block]
        inv_z = 1.0 / z
        lo_c = torch.ceil(x.min(dim=1).values).to(torch.int64).clamp(min=0)
        hi_c = torch.floor(x.max(dim=1).values).to(torch.int64).clamp(max=width - 1)
        lo_r = torch.ceil(y.min(dim=1).values).to(torch.int64).clamp(min=0)
        hi_r = torch.floor(y.max(dim=1).values).to(torch.int64).clamp(max=height - 1)
        extent = torch.maximum(hi_c - lo_c + 1, hi_r - lo_r + 1)
        alive = (hi_c >= lo_c) & (hi_r >= lo_r)
        remaining = alive.clone()
        for tier in tiers:
            selected = remaining & (extent <= tier)
            remaining &= ~selected
            count = int(selected.sum().item())
            if count == 0:
                continue
            stats["triangles_rasterized"] += count
            grid = torch.arange(tier, device=device)
            dc, dr = torch.meshgrid(grid, grid, indexing="ij")
            dc = dc.reshape(1, -1)
            dr = dr.reshape(1, -1)
            cols = lo_c[selected].unsqueeze(1) + dc
            rows = lo_r[selected].unsqueeze(1) + dr
            valid = (cols <= hi_c[selected].unsqueeze(1)) & (rows <= hi_r[selected].unsqueeze(1))
            xs = x[selected]
            ys = y[selected]
            fc = cols.to(torch.float32)
            fr = rows.to(torch.float32)
            x0, x1, x2 = xs[:, 0:1], xs[:, 1:2], xs[:, 2:3]
            y0, y1, y2 = ys[:, 0:1], ys[:, 1:2], ys[:, 2:3]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            w0 = (x1 - fc) * (y2 - fr) - (x2 - fc) * (y1 - fr)
            w1 = (x2 - fc) * (y0 - fr) - (x0 - fc) * (y2 - fr)
            w2 = (x0 - fc) * (y1 - fr) - (x1 - fc) * (y0 - fr)
            positive = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            negative = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            inside = valid & (positive | negative) & (area.abs() > 1e-20)
            if not bool(inside.any()):
                continue
            safe_area = torch.where(area.abs() > 1e-20, area, torch.ones_like(area))
            b0, b1, b2 = w0 / safe_area, w1 / safe_area, w2 / safe_area
            iz = inv_z[selected]
            inv_depth = b0 * iz[:, 0:1] + b1 * iz[:, 1:2] + b2 * iz[:, 2:3]
            hit = inside & (inv_depth > 0)
            pixel = (rows * width + cols)[hit]
            value = (1.0 / inv_depth)[hit]
            depth_buffer = depth_buffer.scatter_reduce(0, pixel, value, reduce="amin")
        stats["triangles_beyond_largest_tier"] += int(remaining.sum().item())
    return depth_buffer.reshape(height, width), stats


def rasterize_mesh_shaded(
    camera: Any, vertices: torch.Tensor, faces: torch.Tensor, vertex_colors: torch.Tensor, *,
    background: tuple[float, float, float] = (0.07, 0.08, 0.10),
    light_direction: tuple[float, float, float] = (0.35, 0.55, -0.77),
    ambient: float = 0.35, diffuse: float = 0.65, shaded: bool = True,
    tiers: tuple[int, ...] = (2, 4, 16, 64), face_chunk: int = 2_000_000,
) -> torch.Tensor:
    """Z-buffered triangle rasterization -- an actual mesh render, not a
    marker point cloud.

    Uses the SAME pixel-centre ray/triangle test as `rasterize_mesh_depth`
    (identical camera convention), so the silhouette matches. When
    `shaded=True`, per-face brightness is plain Lambertian off the face's own
    geometric normal (`light_direction` is a fixed world-space direction, not
    learned data) times the barycentric interpolation of `vertex_colors` -- no
    Gaussian covariance normal, no invented lighting model, geometry-only.
    Set `shaded=False` when `vertex_colors` itself ENCODES data (a support-count
    ramp, a state colour) so lighting brightness cannot be confused with the
    encoded value; the mesh is then flat-colour-interpolated with no shading.
    Background pixels (no triangle hit) keep `background` untouched.
    """

    width, height = int(camera.image_width), int(camera.image_height)
    device = vertices.device
    depth_buffer = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    color_buffer = torch.tensor(background, dtype=torch.float32, device=device).reshape(1, 3).expand(
        height * width, 3
    ).contiguous()

    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    homogeneous = torch.cat([vertices.to(torch.float32), ones], dim=1)
    vertex_depth = (homogeneous @ camera.world_view_transform)[:, 2].contiguous()
    clip = homogeneous @ camera.full_proj_transform
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 0, w, torch.ones_like(w))
    px = ((clip[:, 0] / safe_w + 1.0) * width - 1.0) * 0.5
    py = ((clip[:, 1] / safe_w + 1.0) * height - 1.0) * 0.5
    vertex_ok = (w > 0) & (vertex_depth >= CANONICAL_NEAR_N)
    del clip, homogeneous

    light = torch.tensor(light_direction, dtype=torch.float32, device=device)
    light = light / light.norm().clamp_min(1e-12)

    for start in range(0, int(faces.shape[0]), face_chunk):
        block = faces[start : start + face_chunk]
        ok = vertex_ok[block].all(dim=1)
        block = block[ok]
        if block.numel() == 0:
            continue
        x, y, z = px[block], py[block], vertex_depth[block]
        inv_z = 1.0 / z
        if shaded:
            corners = vertices[block]
            normals = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)
            shade = (ambient + diffuse * (normals @ light).abs()).clamp(0.0, 1.0)
        else:
            shade = torch.ones((block.shape[0],), dtype=torch.float32, device=device)
        colors = vertex_colors[block]

        lo_c = torch.ceil(x.min(dim=1).values).to(torch.int64).clamp(min=0)
        hi_c = torch.floor(x.max(dim=1).values).to(torch.int64).clamp(max=width - 1)
        lo_r = torch.ceil(y.min(dim=1).values).to(torch.int64).clamp(min=0)
        hi_r = torch.floor(y.max(dim=1).values).to(torch.int64).clamp(max=height - 1)
        extent = torch.maximum(hi_c - lo_c + 1, hi_r - lo_r + 1)
        alive = (hi_c >= lo_c) & (hi_r >= lo_r)
        remaining = alive.clone()
        for tier in tiers:
            selected = remaining & (extent <= tier)
            remaining &= ~selected
            count = int(selected.sum().item())
            if count == 0:
                continue
            grid = torch.arange(tier, device=device)
            dc, dr = torch.meshgrid(grid, grid, indexing="ij")
            dc, dr = dc.reshape(1, -1), dr.reshape(1, -1)
            cols = lo_c[selected].unsqueeze(1) + dc
            rows = lo_r[selected].unsqueeze(1) + dr
            valid = (cols <= hi_c[selected].unsqueeze(1)) & (rows <= hi_r[selected].unsqueeze(1))
            xs, ys = x[selected], y[selected]
            fc, fr = cols.to(torch.float32), rows.to(torch.float32)
            x0, x1, x2 = xs[:, 0:1], xs[:, 1:2], xs[:, 2:3]
            y0, y1, y2 = ys[:, 0:1], ys[:, 1:2], ys[:, 2:3]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            w0 = (x1 - fc) * (y2 - fr) - (x2 - fc) * (y1 - fr)
            w1 = (x2 - fc) * (y0 - fr) - (x0 - fc) * (y2 - fr)
            w2 = (x0 - fc) * (y1 - fr) - (x1 - fc) * (y0 - fr)
            positive = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            negative = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            inside = valid & (positive | negative) & (area.abs() > 1e-20)
            if not bool(inside.any()):
                continue
            safe_area = torch.where(area.abs() > 1e-20, area, torch.ones_like(area))
            b0, b1, b2 = w0 / safe_area, w1 / safe_area, w2 / safe_area
            iz = inv_z[selected]
            inv_depth = b0 * iz[:, 0:1] + b1 * iz[:, 1:2] + b2 * iz[:, 2:3]
            hit = inside & (inv_depth > 0)
            if not bool(hit.any()):
                continue
            pixel = (rows * width + cols)[hit]
            depth_here = (1.0 / inv_depth)[hit]

            # per-pixel triangle index (row within `selected`) for gathering color
            triangle_row = torch.arange(int(count), device=device).unsqueeze(1).expand_as(hit)[hit]
            face_color = colors[selected][triangle_row]
            b0h, b1h, b2h = b0[hit], b1[hit], b2[hit]
            interpolated = (
                b0h.unsqueeze(1) * face_color[:, 0] + b1h.unsqueeze(1) * face_color[:, 1]
                + b2h.unsqueeze(1) * face_color[:, 2]
            )
            final_color = interpolated * shade[selected][triangle_row].unsqueeze(1)

            # z-test against the running depth buffer: a fragment survives only
            # if it beats what is already there for its pixel AND is the
            # nearest fragment among this tier's own batch for that pixel
            # (scatter_reduce gives the true per-pixel minimum in one pass, no
            # chunk-order dependence).
            depth_buffer = depth_buffer.scatter_reduce(0, pixel, depth_here, reduce="amin")
            winner = depth_here <= depth_buffer[pixel] + 1e-9
            if bool(winner.any()):
                color_buffer[pixel[winner]] = final_color[winner]
    return color_buffer.reshape(height, width, 3)
