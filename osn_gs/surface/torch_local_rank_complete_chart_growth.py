from __future__ import annotations

"""Worklog 114 -- Local Rank-Complete NURBS Chart Growth.

Worklog 113 traced the B/C failure signatures (rectangular-domain mismatch,
fixed-capacity failure) to a single root cause: the chart UNIT itself
("one camera-connected blob == one fixed 8x4 NURBS chart") operates at the
wrong scale for large blobs. This module replaces that unit -- WITHOUT
touching canonical topology, blob CONNECTIVITY, or the fixed NURBS
configuration -- with a deterministic decomposition of each existing
per-view same-component blob into multiple LOCAL, mathematically
rank-complete chart domains.

A local chart's boundary is a REPRESENTATION SEAM, not a physical surface
boundary, a visible-component boundary, or an occlusion boundary (directive
central intent). The closure condition is derived purely from the fixed
NURBS model's own design-matrix column rank (`resolution_u * resolution_v`
basis functions) -- never from a scene-tuned residual/pixel-count/area/
occupancy threshold (directive sections 4 and 9/17).

Reuses `label_same_component_blobs`
(`osn_gs/surface/torch_camera_observed_chart_domains.py`, frozen/unmodified)
for blob connectivity so a local chart can never span two different
canonical `visible_component_id`s or cross an image-space hole/occluded gap
-- that guarantee is structural (inherited from the frozen blob labeling),
not re-checked here.

Deterministic seeding rule (directive section 5): within each connected
remaining region, the seed is the pixel with MAXIMUM Euclidean distance from
the region's own raster boundary (a "pole of inaccessibility", computed via
`scipy.ndimage.distance_transform_edt`), with ties broken by the smallest
(row, col). Growth then proceeds by breadth-first search from that seed over
the existing image-space 4-neighbor graph, with each BFS frontier level
sorted by (row, col) for determinism. Candidate local-chart sizes are
checked starting at `resolution_u * resolution_v` (the mathematical minimum
possible for full column rank) and then in fixed steps of
`_RANK_CHECK_STEP` pixels (a fixed algorithmic constant of this growth
procedure, not a scene-tuned geometry threshold) until the design matrix
first reaches full column rank; if the ENTIRE connected remaining region
(every one of its pixels) still cannot reach full rank, it is left
unresolved (directive section 4's explicit "leave it unresolved" instruction
-- the NURBS model is never changed to force a fit).
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_camera_observed_chart_domains import label_same_component_blobs
from osn_gs.surface.torch_nurbs import TorchNURBSSurface

_EPS = 1e-9
_RANK_CHECK_STEP = 4  # fixed algorithmic step size for candidate growth beyond the minimum, not a tuned geometry threshold

REASON_TOO_FEW_PIXELS = "TOO_FEW_PIXELS"
REASON_INSUFFICIENT_RANK_CLOSURE = "INSUFFICIENT_RANK_CLOSURE"
REASON_RUNTIME_CAP_SKIPPED = "RUNTIME_CAP_SKIPPED"


@dataclass
class LocalChart:
    """One local, rank-complete NURBS chart domain within one camera-observed
    same-component blob. `pixel_rows`/`pixel_cols` (numpy int arrays) are the
    exact member pixels grown from actual observed support -- no invented
    samples. `xyz`/`uv` (torch tensors) are ready to pass directly to
    `fit_torch_visible_surface_lsq`."""

    view_index: int
    component_id: int
    pixel_rows: Any
    pixel_cols: Any
    representative_ids: Any
    xyz: Any
    uv: Any
    rank: int
    full_capacity: int


@dataclass
class UnresolvedRegion:
    """A connected, same-component pixel region that could not be turned
    into a local NURBS chart this batch -- still VISIBLE TOPOLOGY EVIDENCE
    (directive section 7), never fabricated into a surface, never to be
    later read as occluded/unknown/free-space/visible-termination."""

    view_index: int
    component_id: int
    pixel_rows: Any
    pixel_cols: Any
    reason: str


def _farthest_point_seed(mask: Any) -> tuple[int, int]:
    """Deterministic "pole of inaccessibility" seed: the mask pixel with
    maximum distance from the region's own boundary, ties broken by the
    smallest (row, col).

    `distance_transform_edt` treats out-of-array space as unbounded (not as
    background), so a region touching the raster's own edge would otherwise
    get a bogus monotonic gradient toward that edge instead of a genuine
    interior pole. Pad with one pixel of False on every side first so the
    array's own frame edge always counts as real boundary -- a pixel at the
    edge of the observed frame genuinely IS at the boundary of what was
    observed."""

    import numpy as np
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    max_val = dist.max()
    candidates = np.argwhere(dist == max_val)
    order = np.lexsort((candidates[:, 1], candidates[:, 0]))
    row, col = candidates[order[0]]
    return int(row), int(col)


def _bfs_order(mask: Any, seed: tuple[int, int]) -> list[tuple[int, int]]:
    """Deterministic BFS pixel visitation order from `seed`, restricted to
    `mask`, 4-neighbor connectivity, each frontier level sorted by (row, col).
    Eager (materializes the ENTIRE order) -- kept for tests/callers that
    genuinely need the full order; the production growth path below uses
    `_bfs_levels` instead so it never has to traverse more of a huge blob
    than one chart's own closure actually requires."""

    order: list[tuple[int, int]] = []
    for level in _bfs_levels(mask, seed):
        order.extend(level)
    return order


def _bfs_levels(mask: Any, seed: tuple[int, int]):
    """Yield successive BFS frontier levels (each a sorted list of (row,
    col)) from `seed`, restricted to `mask`, 4-neighbor connectivity.

    A generator, not an eagerly-built list: a caller that stops requesting
    levels early (because it already found what it needed) never pays for
    visiting the rest of a large connected region -- this is what keeps
    growth cost proportional to the SIZE OF THE EXTRACTED CHART, not the
    size of the blob it was carved from (the earlier eager `_bfs_order`-based
    implementation was O(blob_size) per chart, i.e. O(blob_size^2 / chart_size)
    per blob -- intractable for real 100k+ pixel blobs, fixed this batch
    after the first real-scene smoke test stalled)."""

    import numpy as np

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    frontier = [seed]
    visited[seed] = True
    while frontier:
        yield frontier
        next_frontier: list[tuple[int, int]] = []
        for row, col in frontier:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    next_frontier.append((nr, nc))
        next_frontier.sort()
        frontier = next_frontier


def _bbox_uv(rows: Any, cols: Any, device: Any) -> Any:
    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()
    row_t = torch.as_tensor(rows, dtype=torch.float32, device=device)
    col_t = torch.as_tensor(cols, dtype=torch.float32, device=device)
    row_min, row_max = row_t.min(), row_t.max()
    col_min, col_max = col_t.min(), col_t.max()
    row_span = (row_max - row_min).clamp_min(_EPS)
    col_span = (col_max - col_min).clamp_min(_EPS)
    u = (row_t - row_min) / row_span
    v = (col_t - col_min) / col_span
    return torch.stack([u, v], dim=1).clamp(0.0, 1.0)


def _design_matrix_rank(uv: Any, dummy_surface: TorchNURBSSurface) -> int:
    """Column rank of the fixed model's tensor-product basis design matrix
    at `uv` -- depends only on the fixed `(resolution_u, resolution_v,
    degree_u, degree_v)` structure via clamped knot vectors, never on
    `dummy_surface`'s (unused) control-point values."""

    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()
    with torch.no_grad():
        basis_u, basis_v, _du, _dv = dummy_surface._basis_tables(uv)
        design = (basis_u[:, :, None] * basis_v[:, None, :]).reshape(basis_u.shape[0], -1)
        try:
            return int(torch.linalg.matrix_rank(design).item())
        except Exception:
            return -1


def _grow_one_local_domain(
    mask: Any, full_capacity: int, cpu_dummy_surface: TorchNURBSSurface
) -> list[tuple[int, int]] | None:
    """Grow ONE rank-complete local domain from `mask`'s own pole-of-
    inaccessibility seed. Returns the member pixel list at first full column
    rank, or `None` if even the entire connected `mask` cannot reach full
    rank (directive section 4: leave unresolved, never force a fit).

    Every candidate-size rank check runs on CPU tensors regardless of the
    pipeline's overall device: these are tiny (<=few hundred rows, 32
    columns) matrices checked potentially many times per patch, and a CUDA
    device round-trip per check dominates runtime by orders of magnitude
    over the tiny matmul/SVD itself (measured during this batch's own smoke
    test). Only the final accepted chart's data is later moved to the real
    device for NURBS fitting."""

    import numpy as np

    if int(mask.sum()) < full_capacity:
        return None
    seed = _farthest_point_seed(mask)

    order: list[tuple[int, int]] = []
    next_check = full_capacity
    for level in _bfs_levels(mask, seed):
        order.extend(level)
        while next_check <= len(order):
            subset = order[:next_check]
            rows = np.array([p[0] for p in subset])
            cols = np.array([p[1] for p in subset])
            uv = _bbox_uv(rows, cols, "cpu")
            if _design_matrix_rank(uv, cpu_dummy_surface) == full_capacity:
                return subset
            next_check += _RANK_CHECK_STEP

    # BFS exhausted (`order` now holds the ENTIRE connected mask) without a
    # step landing exactly on `len(order)` -- check the full region exactly
    # once before giving up, per directive section 4 ("if a connected
    # observed region can never reach full rank: leave it unresolved").
    if order:
        rows = np.array([p[0] for p in order])
        cols = np.array([p[1] for p in order])
        uv = _bbox_uv(rows, cols, "cpu")
        if _design_matrix_rank(uv, cpu_dummy_surface) == full_capacity:
            return order
    return None


def grow_local_rank_complete_charts(
    view_index: int,
    component_id_map: Any,
    representative_id_map: Any,
    world_points: Any,
    resolution_u: int = 8,
    resolution_v: int = 4,
    degree_u: int = 2,
    degree_v: int = 2,
    max_patches_per_blob: int | None = 2000,
) -> tuple[list[LocalChart], list[UnresolvedRegion]]:
    """Decompose every camera-observed same-component blob in this view into
    deterministic local rank-complete NURBS chart domains.

    `max_patches_per_blob` is a RUNTIME SAFETY VALVE only (bounds worst-case
    compute on a pathologically large single blob) -- it is not part of the
    architecture's closure semantics and is reported separately
    (`REASON_RUNTIME_CAP_SKIPPED`) whenever it is actually hit, so its effect
    on the measured results stays auditable. Pass `None` to disable it.
    """

    import numpy as np
    from scipy.ndimage import label as ndi_label

    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()

    device = component_id_map.device
    full_capacity = int(resolution_u) * int(resolution_v)
    # Growth-phase rank checks always run on CPU (see `_grow_one_local_domain`
    # docstring) -- a separate CPU-resident dummy surface avoids any
    # per-candidate CUDA round-trip. The real `device` is used only once per
    # ACCEPTED chart below (final rank/uv record + xyz gather for fitting).
    cpu_dummy_surface = TorchNURBSSurface(
        control_grid=torch.zeros((resolution_u, resolution_v, 3), dtype=torch.float32, device="cpu"),
        weights=torch.ones((resolution_u, resolution_v), dtype=torch.float32, device="cpu"),
        degree_u=degree_u,
        degree_v=degree_v,
    )
    device_dummy_surface = TorchNURBSSurface(
        control_grid=torch.zeros((resolution_u, resolution_v, 3), dtype=torch.float32, device=device),
        weights=torch.ones((resolution_u, resolution_v), dtype=torch.float32, device=device),
        degree_u=degree_u,
        degree_v=degree_v,
    )

    blob_labels = label_same_component_blobs(component_id_map)
    blob_labels_np = blob_labels.detach().cpu().numpy()
    comp_np = component_id_map.detach().cpu().numpy()
    rep_np = representative_id_map.detach().cpu().numpy()

    charts: list[LocalChart] = []
    unresolved: list[UnresolvedRegion] = []

    if not (blob_labels_np >= 0).any():
        return charts, unresolved

    blob_count = int(blob_labels_np.max()) + 1
    four_conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    for blob_id in range(blob_count):
        base_mask = blob_labels_np == blob_id
        if not base_mask.any():
            continue
        component_id = int(comp_np[base_mask][0])
        queue: list[Any] = [base_mask]
        patches_this_blob = 0
        while queue:
            region = queue.pop(0)
            if not region.any():
                continue
            labeled, num = ndi_label(region, structure=four_conn)
            if num > 1:
                fragments = []
                for comp_idx in range(1, num + 1):
                    comp_mask = labeled == comp_idx
                    rows_f, cols_f = np.nonzero(comp_mask)
                    key = (int(rows_f.min()), int(cols_f[rows_f == rows_f.min()].min()))
                    fragments.append((key, comp_mask))
                fragments.sort(key=lambda item: item[0])
                queue = [f[1] for f in fragments] + queue
                continue
            if max_patches_per_blob is not None and patches_this_blob >= max_patches_per_blob:
                rows, cols = np.nonzero(region)
                unresolved.append(UnresolvedRegion(view_index, component_id, rows, cols, REASON_RUNTIME_CAP_SKIPPED))
                continue
            if int(region.sum()) < full_capacity:
                rows, cols = np.nonzero(region)
                unresolved.append(UnresolvedRegion(view_index, component_id, rows, cols, REASON_TOO_FEW_PIXELS))
                continue
            result = _grow_one_local_domain(region, full_capacity, cpu_dummy_surface)
            if result is None:
                rows, cols = np.nonzero(region)
                unresolved.append(UnresolvedRegion(view_index, component_id, rows, cols, REASON_INSUFFICIENT_RANK_CLOSURE))
                continue
            rows = np.array([p[0] for p in result])
            cols = np.array([p[1] for p in result])
            uv = _bbox_uv(rows, cols, device)
            row_t = torch.as_tensor(rows, dtype=torch.int64, device=device)
            col_t = torch.as_tensor(cols, dtype=torch.int64, device=device)
            xyz = world_points[row_t, col_t]
            rep_ids = rep_np[rows, cols]
            rank = _design_matrix_rank(uv, device_dummy_surface)
            charts.append(LocalChart(view_index, component_id, rows, cols, rep_ids, xyz, uv, rank, full_capacity))
            patches_this_blob += 1

            remainder = region.copy()
            remainder[rows, cols] = False
            if remainder.any():
                queue.append(remainder)

    return charts, unresolved
