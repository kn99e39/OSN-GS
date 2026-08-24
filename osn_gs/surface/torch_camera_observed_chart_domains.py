from __future__ import annotations

"""Worklog 111 -- Camera-Observed Chart Domains for Representative-Only Visible NURBS.

Worklog 110 established AMBIGUOUS/LAYERED SUPPORT for non-representative
renderer evidence -- it stays out of NURBS fitting entirely in this batch.
This module builds NURBS CHART CANDIDATES from ONLY the frozen Worklog
107/109 canonical topology's MEDIAN_SURFACE_REPRESENTATIVE population,
using the per-view representative maps' own image-space pixel coordinates
as the chart UV parameterization (directive section 4) -- not a new 3D PCA
parameterization, not a raw-3D kNN topology (both previously failed, see
docs/agent_memory/project_intrinsic_integrability_local_chart_atlas.md and
project_patch_identifiability_capacity_gate.md).

A "camera-observed chart candidate" is one CONNECTED (4-neighbor) image-space
region within ONE training view whose every pixel's representative surfel
maps to the SAME canonical `visible_component_id` (Worklog 107/109's
component id, read-only input, never modified here). Connectivity is
restricted to same-component neighbors ONLY, so two different canonical
components adjacent in image space can never land in the same chart
candidate (directive section 5's explicit requirement) -- this is a
structural guarantee of the labeling, not a post-hoc filter.

Labeling uses `scipy.sparse.csgraph.connected_components` over an edge list
built from right/down pixel-neighbor pairs whose representative-remapped
component id matches (scipy is an existing project dependency, see
`torch_single_chart_uv_validity.py`). This is exact connected-component
labeling, not an approximate iterative flood -- and does not require looping
per distinct component id.

Each chart candidate collects ONE (u, v) sample per distinct member
representative surfel: the mean pixel row/column of every pixel this view
assigned to that representative WITHIN this blob, normalized by the blob's
own pixel bounding box into [0, 1]^2. This is a genuine observed-chart
parameterization (directive section 4), not claimed to be metric-preserving.
"""

from dataclasses import dataclass
from typing import Any

_EPS = 1e-9


@dataclass
class ViewChartCandidates:
    """One training view's camera-observed chart candidates.

    All fields are ragged per-blob data flattened with a `blob_of_member`
    index tensor, so this stays a small, torch-native, GPU-movable
    structure instead of a Python list of per-blob dicts.
    """

    view_index: int
    blob_component_id: Any  # (B,) int64 -- canonical visible_component_id per blob
    blob_of_member: Any  # (M,) int64 -- which blob each member row belongs to
    member_representative_id: Any  # (M,) int64 -- visible-index-space representative id
    member_uv: Any  # (M, 2) float32 in [0, 1]^2, normalized per-blob
    member_pixel_count: Any  # (M,) int64 -- how many pixels of this view supported this member in this blob
    blob_pixel_total: Any  # (B,) int64 -- total pixels in each blob

    @property
    def blob_count(self) -> int:
        return int(self.blob_component_id.shape[0])


def label_same_component_blobs(component_id_map: Any) -> Any:
    """Connected-component label a ``(H, W)`` int64 map, 4-connectivity,
    restricted to same-value neighbor edges (``-1`` = invalid, never
    connects to anything, including other ``-1`` pixels). Returns a
    ``(H, W)`` int64 label map with ``-1`` preserved at invalid pixels and
    a fresh 0-based label for every other connected blob.

    Exact algorithm (not an approximate iterative flood): builds the sparse
    right/down neighbor-edge list restricted to matching, valid component
    ids, then runs `scipy.sparse.csgraph.connected_components` over the
    full pixel graph (isolated valid pixels become singleton blobs, exactly
    as a real 1-pixel chart candidate should).
    """

    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    arr = component_id_map.detach().cpu().numpy() if hasattr(component_id_map, "detach") else np.asarray(component_id_map)
    h, w = arr.shape
    n = h * w
    valid = arr >= 0

    right_match = valid[:, :-1] & valid[:, 1:] & (arr[:, :-1] == arr[:, 1:])
    down_match = valid[:-1, :] & valid[1:, :] & (arr[:-1, :] == arr[1:, :])

    row_idx = np.arange(n).reshape(h, w)
    src_right = row_idx[:, :-1][right_match]
    dst_right = row_idx[:, 1:][right_match]
    src_down = row_idx[:-1, :][down_match]
    dst_down = row_idx[1:, :][down_match]

    src = np.concatenate([src_right, src_down])
    dst = np.concatenate([dst_right, dst_down])
    data = np.ones((src.shape[0],), dtype=np.int8)
    graph = coo_matrix((data, (src, dst)), shape=(n, n))

    _, raw_labels = connected_components(graph, directed=False)
    labels = raw_labels.reshape(h, w).astype(np.int64)
    labels[~valid] = -1

    # `connected_components` assigns every pixel (including invalid ones) a
    # label from a shared global namespace; renumber the valid blobs to a
    # dense 0-based range so downstream code can size tensors by blob_count.
    valid_labels = labels[valid]
    if valid_labels.size == 0:
        out = np.full((h, w), -1, dtype=np.int64)
        return _to_like(component_id_map, out) if hasattr(component_id_map, "device") else out
    unique_labels, _dense = np.unique(valid_labels, return_inverse=True)
    dense_map = np.full((int(unique_labels.max()) + 1,), -1, dtype=np.int64)
    dense_map[unique_labels] = np.arange(unique_labels.shape[0], dtype=np.int64)
    out = np.full((h, w), -1, dtype=np.int64)
    out[valid] = dense_map[labels[valid]]

    return _to_like(component_id_map, out) if hasattr(component_id_map, "device") else out


def _to_like(reference: Any, array: Any) -> Any:
    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()
    return torch.as_tensor(array, dtype=torch.int64, device=reference.device)


def build_view_chart_candidates(view_index: int, component_id_map: Any, representative_id_map: Any) -> ViewChartCandidates:
    """Build this view's chart candidates from its ``(H, W)`` canonical
    component-id map (representative id remapped through Worklog 107/109
    ``subset_ids``, ``-1`` = no representative) and its raw ``(H, W)``
    representative-id map (visible-index space, ``-1`` = none).

    Structural guarantee (directive section 5): a blob's every pixel shares
    exactly one component id by construction of
    :func:`label_same_component_blobs`, so two canonical components adjacent
    in image space can never share a blob -- this is never checked
    post-hoc, it cannot happen.
    """

    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()

    device = component_id_map.device
    h, w = component_id_map.shape
    blob_labels = label_same_component_blobs(component_id_map)
    valid = blob_labels >= 0
    if not bool(valid.any()):
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_f = torch.zeros((0, 2), dtype=torch.float32, device=device)
        return ViewChartCandidates(view_index, empty_i, empty_i, empty_i, empty_f, empty_i, empty_i)

    row_coords, col_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )

    blob_flat = blob_labels[valid]
    rep_flat = representative_id_map[valid]
    row_flat = row_coords[valid]
    col_flat = col_coords[valid]
    blob_count = int(blob_flat.max().item()) + 1

    # Per-blob canonical component id (uniform within a blob by construction).
    comp_flat = component_id_map[valid]
    blob_component_id = torch.zeros((blob_count,), dtype=torch.int64, device=device)
    blob_component_id.scatter_(0, blob_flat, comp_flat)

    blob_pixel_total = torch.zeros((blob_count,), dtype=torch.int64, device=device)
    blob_pixel_total.index_add_(0, blob_flat, torch.ones_like(blob_flat))

    # One (blob, representative) member row per distinct pair, with the
    # mean pixel row/col over that pair's pixels as its raw chart coordinate.
    rep_max = int(rep_flat.max().item()) + 1 if int(rep_flat.numel()) > 0 else 1
    pair_key = blob_flat * rep_max + rep_flat
    unique_keys, inverse = torch.unique(pair_key, return_inverse=True)
    member_count = int(unique_keys.shape[0])

    sum_row = torch.zeros((member_count,), dtype=torch.float32, device=device)
    sum_col = torch.zeros((member_count,), dtype=torch.float32, device=device)
    counts = torch.zeros((member_count,), dtype=torch.int64, device=device)
    sum_row.index_add_(0, inverse, row_flat)
    sum_col.index_add_(0, inverse, col_flat)
    counts.index_add_(0, inverse, torch.ones_like(inverse))

    member_blob = unique_keys // rep_max
    member_representative_id = unique_keys % rep_max
    mean_row = sum_row / counts.to(torch.float32)
    mean_col = sum_col / counts.to(torch.float32)

    # Normalize each member's raw pixel coordinate by ITS OWN blob's pixel
    # bounding box -- a genuine per-chart [0, 1]^2 domain (directive section 4).
    min_row = torch.full((blob_count,), float("inf"), device=device)
    max_row = torch.full((blob_count,), float("-inf"), device=device)
    min_col = torch.full((blob_count,), float("inf"), device=device)
    max_col = torch.full((blob_count,), float("-inf"), device=device)
    min_row.scatter_reduce_(0, blob_flat, row_flat, reduce="amin", include_self=True)
    max_row.scatter_reduce_(0, blob_flat, row_flat, reduce="amax", include_self=True)
    min_col.scatter_reduce_(0, blob_flat, col_flat, reduce="amin", include_self=True)
    max_col.scatter_reduce_(0, blob_flat, col_flat, reduce="amax", include_self=True)
    row_span = torch.clamp(max_row - min_row, min=_EPS)
    col_span = torch.clamp(max_col - min_col, min=_EPS)

    u = (mean_row - min_row[member_blob]) / row_span[member_blob]
    v = (mean_col - min_col[member_blob]) / col_span[member_blob]
    member_uv = torch.stack([u, v], dim=1).clamp(0.0, 1.0)

    return ViewChartCandidates(
        view_index=view_index,
        blob_component_id=blob_component_id,
        blob_of_member=member_blob,
        member_representative_id=member_representative_id,
        member_uv=member_uv,
        member_pixel_count=counts,
        blob_pixel_total=blob_pixel_total,
    )


@dataclass
class ViewPixelChartSamples:
    """Worklog 112 addition -- one training view's chart candidates using
    DENSE PER-PIXEL renderer-native surface samples, not one mean sample per
    representative surfel (that collapse is exactly what Worklog 111 did and
    what this batch's directive forbids repeating, section 6). Every valid
    renderer pixel keeps its OWN (u, v) and its OWN unprojected
    renderer-native 3D surface point -- `representative_id` is carried along
    for coverage ACCOUNTING only (section 4), never used as fitting geometry.

    Uses the exact same :func:`label_same_component_blobs` as
    :class:`ViewChartCandidates` on the identical `component_id_map` input,
    so blob membership is byte-identical to what Worklog 111's
    `build_view_chart_candidates` would produce for the same view (directive
    section 1/5: same chart connected-component labeling, changed only the
    3D fitting target).
    """

    view_index: int
    blob_component_id: Any  # (B,) int64
    pixel_blob_id: Any  # (P,) int64 -- which blob each valid pixel belongs to
    pixel_uv: Any  # (P, 2) float32 in [0, 1]^2, normalized per-blob (same convention as WL111)
    pixel_xyz: Any  # (P, 3) float32 -- renderer-native unprojected surface point, NOT a surfel center
    pixel_representative_id: Any  # (P,) int64 -- accounting only, never fitting geometry
    blob_pixel_total: Any  # (B,) int64 -- count of valid pixels per blob (the fitting-support count, section 7)

    @property
    def blob_count(self) -> int:
        return int(self.blob_component_id.shape[0])


def build_view_chart_pixel_samples(
    view_index: int, component_id_map: Any, representative_id_map: Any, world_points: Any
) -> ViewPixelChartSamples:
    """Dense per-pixel counterpart of :func:`build_view_chart_candidates`.
    ``world_points`` is the renderer-native unprojected surface point map,
    ``(H, W, 3)``, produced by unprojecting the SAME forward kernel's own
    median-crossing depth channel (`out_others[MIDDEPTH_OFFSET]`, see
    `osn_gs.render.surfel_geometry.depths_to_points`) -- not a surfel
    center. A pixel is valid iff its ``representative_id_map`` entry is
    ``>= 0`` (identical validity condition to Worklog 111 and to the
    forward kernel's own median-crossing test)."""

    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()

    device = component_id_map.device
    h, w = component_id_map.shape
    blob_labels = label_same_component_blobs(component_id_map)
    valid = blob_labels >= 0
    if not bool(valid.any()):
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_f = torch.zeros((0, 2), dtype=torch.float32, device=device)
        empty_f3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
        return ViewPixelChartSamples(view_index, empty_i, empty_i, empty_f, empty_f3, empty_i, empty_i)

    row_coords, col_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    blob_flat = blob_labels[valid]
    rep_flat = representative_id_map[valid]
    row_flat = row_coords[valid]
    col_flat = col_coords[valid]
    xyz_flat = world_points[valid]
    blob_count = int(blob_flat.max().item()) + 1

    comp_flat = component_id_map[valid]
    blob_component_id = torch.zeros((blob_count,), dtype=torch.int64, device=device)
    blob_component_id.scatter_(0, blob_flat, comp_flat)

    blob_pixel_total = torch.zeros((blob_count,), dtype=torch.int64, device=device)
    blob_pixel_total.index_add_(0, blob_flat, torch.ones_like(blob_flat))

    min_row = torch.full((blob_count,), float("inf"), device=device)
    max_row = torch.full((blob_count,), float("-inf"), device=device)
    min_col = torch.full((blob_count,), float("inf"), device=device)
    max_col = torch.full((blob_count,), float("-inf"), device=device)
    min_row.scatter_reduce_(0, blob_flat, row_flat, reduce="amin", include_self=True)
    max_row.scatter_reduce_(0, blob_flat, row_flat, reduce="amax", include_self=True)
    min_col.scatter_reduce_(0, blob_flat, col_flat, reduce="amin", include_self=True)
    max_col.scatter_reduce_(0, blob_flat, col_flat, reduce="amax", include_self=True)
    row_span = torch.clamp(max_row - min_row, min=_EPS)
    col_span = torch.clamp(max_col - min_col, min=_EPS)

    u = (row_flat - min_row[blob_flat]) / row_span[blob_flat]
    v = (col_flat - min_col[blob_flat]) / col_span[blob_flat]
    pixel_uv = torch.stack([u, v], dim=1).clamp(0.0, 1.0)

    return ViewPixelChartSamples(
        view_index=view_index,
        blob_component_id=blob_component_id,
        pixel_blob_id=blob_flat,
        pixel_uv=pixel_uv,
        pixel_xyz=xyz_flat,
        pixel_representative_id=rep_flat,
        blob_pixel_total=blob_pixel_total,
    )


def valid_pixel_chart_mask(view_samples: ViewPixelChartSamples, min_pixel_samples: int) -> Any:
    """Boolean ``(blob_count,)`` mask of blobs with ``>= min_pixel_samples``
    valid renderer-native PIXEL samples (directive section 7: fitting
    eligibility is measured by independent pixel-surface samples, never by
    representative-component size, section 8)."""

    return view_samples.blob_pixel_total >= int(min_pixel_samples)


def valid_chart_mask(view_charts: ViewChartCandidates, min_members: int) -> Any:
    """Boolean ``(blob_count,)`` mask of blobs with ``>= min_members``
    DISTINCT member representatives -- the mathematically-derived minimum
    (directive section 7: ``resolution_u * resolution_v`` control points
    need at least that many independent samples for the fit to be
    data-determined, not just regularizer-determined). No scene tuning."""

    from osn_gs.utils.torch_ops import require_torch
    torch = require_torch()
    member_count_per_blob = torch.zeros((view_charts.blob_count,), dtype=torch.int64, device=view_charts.blob_component_id.device)
    member_count_per_blob.index_add_(0, view_charts.blob_of_member, torch.ones_like(view_charts.blob_of_member))
    return member_count_per_blob >= int(min_members)
