from __future__ import annotations

"""Worklog 69: single-chart PCA-UV parameterization validity diagnostics.

A region's existing accepted topology (region formation, chart boundary,
ownership gating) is never touched here -- this module only asks whether
the SAME topology's existing PCA-UV parameterization (`pca_parameterize_points`,
`osn_gs/surface/torch_nurbs.py`, reused unmodified) is a valid single-chart
layout for the region's evidence: no UV duplicate collisions, 3D-neighbor
structure preserved in UV, accepted edges don't cross when projected to UV,
UV triangles don't fold, interior evidence actually lands inside the
boundary polygon, and the evidence isn't secretly two near-parallel sheets
merged into one region.

Every function here takes raw tensors -- same isolation convention as the
other `torch_*` diagnostic modules in this package (`torch_local_orientation_
folding.py`, `torch_parametric_diagnostics.py`).
"""

from typing import Any, Sequence

from osn_gs.surface.torch_boundary_self_intersection import _segments_intersect
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-8


def uv_duplicate_diagnostics(uv: Any, *, near_collision_ratio: float = 0.05) -> dict[str, Any]:
    """Exact/near-duplicate UV positions -- two distinct 3D evidence points
    mapping to (near-)identical UV, a direct single-chart-invalidity signal
    (the parameterization is not injective there). ``near_collision_ratio``
    is relative to the UV layout's own median nearest-neighbor spacing."""

    torch = require_torch()
    n = int(uv.shape[0])
    if n < 2:
        return {"uv_duplicate_count": 0, "uv_near_collision_count": 0, "uv_median_spacing": None}
    d = torch.cdist(uv, uv)
    d.fill_diagonal_(float("inf"))
    nearest = d.min(dim=1).values
    median_spacing = float(nearest.median().clamp_min(_EPS))
    duplicate_count = int((nearest < 1e-6).sum())
    near_collision_count = int((nearest < median_spacing * near_collision_ratio).sum())
    return {
        "uv_duplicate_count": duplicate_count,
        "uv_near_collision_count": near_collision_count,
        "uv_median_spacing": median_spacing,
    }


def neighborhood_preservation(positions_3d: Any, uv: Any, *, k: int = 8) -> dict[str, Any]:
    """Jaccard overlap between each point's k-nearest-neighbor set in 3D vs
    in UV. Low overlap means the parameterization does not preserve local
    structure (points close in 3D become UV-distant, or vice versa) --
    exactly what a folded/multi-sheet region does to a naive PCA-UV layout."""

    torch = require_torch()
    n = int(positions_3d.shape[0])
    neighbors = min(k, n - 1)
    if neighbors < 1:
        return {"neighborhood_preservation_mean": None, "neighborhood_preservation_min": None}
    d3 = torch.cdist(positions_3d, positions_3d)
    d3.fill_diagonal_(float("inf"))
    duv = torch.cdist(uv, uv)
    duv.fill_diagonal_(float("inf"))
    knn3 = d3.topk(neighbors, dim=1, largest=False).indices
    knnuv = duv.topk(neighbors, dim=1, largest=False).indices
    overlaps = []
    for i in range(n):
        set3 = set(knn3[i].tolist())
        setuv = set(knnuv[i].tolist())
        union = set3 | setuv
        overlaps.append(len(set3 & setuv) / len(union) if union else 1.0)
    return {
        "neighborhood_preservation_mean": float(sum(overlaps) / len(overlaps)),
        "neighborhood_preservation_min": float(min(overlaps)),
    }


def accepted_edge_uv_crossings(uv_by_id: dict[Any, tuple[float, float]], accepted_edges: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    """Reuses `_segments_intersect` (unmodified, same 2D math already used
    for boundary-loop validation) to count how many pairs of the region's
    OWN already-accepted topology edges cross when projected into its own
    UV layout -- edges that never cross in a valid planar-ish parameterization."""

    segments = [
        (uv_by_id[a], uv_by_id[b]) for a, b in accepted_edges
        if a in uv_by_id and b in uv_by_id and a != b
    ]
    crossings = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a1, a2 = segments[i]
            b1, b2 = segments[j]
            if {a1, a2} & {b1, b2}:
                continue  # sharing an endpoint is normal graph adjacency, not a crossing
            kind, _ = _segments_intersect(a1, a2, b1, b2)
            if kind == "proper":
                crossings += 1
    return {"accepted_edge_uv_crossing_count": crossings, "accepted_edge_pair_count": len(segments)}


def uv_triangulation_diagnostics(positions_3d: Any, uv: Any) -> dict[str, Any]:
    """Delaunay-triangulates the UV layout (scipy, an already-available
    dependency, unmodified) purely to get a reasonable LOCAL mesh
    connectivity over the evidence -- a 2D Delaunay triangulation of any
    point set is, by construction, always winding-consistent in UV space
    itself, so a "fold" can never show up as a UV-orientation-sign
    disagreement there. The real signal is in 3D: two UV-adjacent triangles
    (sharing a UV edge) whose 3D NORMALS disagree indicate the underlying
    surface folds back on itself at a place the flat UV layout smooths over
    -- exactly the local (adjacency-only, never a single global reference)
    convention `torch_local_orientation_folding.py` already established for
    the fitted-surface sample grid, reused here for the raw evidence mesh.
    Also reports UV-area / 3D-area distortion per triangle (independent of
    fold detection)."""

    torch = require_torch()
    n = int(uv.shape[0])
    if n < 4:
        return {
            "triangle_fold_count": 0, "triangle_total_count": 0, "triangle_adjacent_pair_count": 0,
            "area_distortion_median": None, "area_distortion_p95": None, "area_distortion_max": None,
        }
    import numpy as np
    from scipy.spatial import Delaunay, QhullError

    uv_np = uv.detach().cpu().numpy()
    pos_np = positions_3d.detach().cpu().numpy()
    try:
        tri = Delaunay(uv_np)
    except QhullError:
        return {
            "triangle_fold_count": 0, "triangle_total_count": 0, "triangle_adjacent_pair_count": 0,
            "area_distortion_median": None, "area_distortion_p95": None, "area_distortion_max": None,
        }
    simplices = tri.simplices
    distortions = []
    triangle_normals = []
    for a, b, c in simplices:
        uv_a, uv_b, uv_c = uv_np[a], uv_np[b], uv_np[c]
        uv_area = abs((uv_b[0] - uv_a[0]) * (uv_c[1] - uv_a[1]) - (uv_b[1] - uv_a[1]) * (uv_c[0] - uv_a[0])) * 0.5
        pos_a, pos_b, pos_c = pos_np[a], pos_np[b], pos_np[c]
        cross = np.cross(pos_b - pos_a, pos_c - pos_a)
        pos_area = 0.5 * float(np.linalg.norm(cross))
        triangle_normals.append(cross / (np.linalg.norm(cross) + 1e-12))
        if uv_area > _EPS and pos_area > _EPS:
            distortions.append(max(uv_area / pos_area, pos_area / uv_area))

    # `neighbors[i, e]` is the simplex sharing simplex i's edge opposite
    # vertex e (scipy convention), or -1 on the outer boundary.
    fold_count = 0
    adjacent_pair_count = 0
    seen_pairs = set()
    for i, row in enumerate(tri.neighbors):
        for neighbor in row:
            if neighbor < 0:
                continue
            pair = (min(i, neighbor), max(i, neighbor))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            adjacent_pair_count += 1
            dot = float(np.dot(triangle_normals[i], triangle_normals[neighbor]))
            if dot < 0.0:
                fold_count += 1

    def _percentiles(values):
        if not values:
            return None, None, None
        arr = np.asarray(values)
        return float(np.median(arr)), float(np.percentile(arr, 95)), float(arr.max())

    median_d, p95_d, max_d = _percentiles(distortions)
    return {
        "triangle_fold_count": fold_count, "triangle_total_count": len(simplices),
        "triangle_adjacent_pair_count": adjacent_pair_count,
        "area_distortion_median": median_d, "area_distortion_p95": p95_d, "area_distortion_max": max_d,
    }


def interior_within_boundary(interior_uv: Any, boundary_uv_ordered: Any) -> dict[str, Any]:
    """Ray-casting point-in-polygon test: how many interior evidence points
    (in UV) actually fall INSIDE the boundary loop's own UV polygon. A
    single-chart parameterization where interior evidence lands outside its
    own boundary is a direct validity failure, independent of fitting
    error."""

    n_interior = int(interior_uv.shape[0]) if interior_uv is not None else 0
    if n_interior == 0:
        return {"interior_outside_boundary_count": 0, "interior_total_count": 0}
    polygon = boundary_uv_ordered.detach().cpu().tolist()
    points = interior_uv.detach().cpu().tolist()
    outside = 0
    m = len(polygon)
    for px, py in points:
        inside = False
        j = m - 1
        for i in range(m):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        if not inside:
            outside += 1
    return {"interior_outside_boundary_count": outside, "interior_total_count": n_interior}


def parallel_sheet_suspicion(
    positions_3d: Any, normal_axis: Any, *, gap_ratio_threshold: float = 3.0, min_cluster_fraction: float = 0.1,
) -> dict[str, Any]:
    """Projects evidence onto the region's own dominant normal axis and
    looks for the largest gap in the sorted 1D projection, AMONG gaps whose
    split leaves at least ``min_cluster_fraction`` of all points on EACH
    side. Without that minimum-side-size requirement, a single routine
    outlier Gaussian (common in real ADC-trained data -- this whole
    project's own worklog history is full of extreme per-Gaussian scale
    outliers) trivially creates an enormous apparent "gap" against the rest
    of the cloud, which is not evidence of two merged sheets at all. Two
    near-parallel sheets merged into one region show up as two
    well-populated, well-separated clusters along this axis; a single
    coherent (if noisy) surface does not. ``gap_ratio_threshold`` compares
    the largest QUALIFYING gap to the median ALONG-AXIS spacing elsewhere in
    the same projection -- scale-free, not an absolute distance."""

    torch = require_torch()
    n = int(positions_3d.shape[0])
    if n < 6:
        return {"parallel_sheet_gap_ratio": None, "parallel_sheet_suspected": False}
    axis = normal_axis / normal_axis.norm().clamp_min(_EPS)
    projection = (positions_3d @ axis)
    sorted_projection, _ = torch.sort(projection)
    gaps = sorted_projection[1:] - sorted_projection[:-1]

    min_side = max(3, int(round(n * min_cluster_fraction)))
    valid_indices = [i for i in range(int(gaps.numel())) if (i + 1) >= min_side and (n - (i + 1)) >= min_side]
    if not valid_indices:
        return {"parallel_sheet_gap_ratio": None, "parallel_sheet_suspected": False}
    valid_index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=gaps.device)
    best_local = int(gaps[valid_index_tensor].argmax())
    best_idx = valid_indices[best_local]

    largest_gap = float(gaps[best_idx])
    other_gaps = torch.cat((gaps[:best_idx], gaps[best_idx + 1 :]))
    median_other_gap = float(other_gaps.median()) if int(other_gaps.numel()) else 0.0
    ratio = largest_gap / max(median_other_gap, _EPS)
    return {
        "parallel_sheet_gap_ratio": ratio,
        "parallel_sheet_suspected": bool(median_other_gap > _EPS and ratio >= gap_ratio_threshold),
    }
