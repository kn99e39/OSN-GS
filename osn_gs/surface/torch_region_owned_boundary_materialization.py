from __future__ import annotations

"""Worklog 70: dense boundary materialization from region-owned full evidence.

worklog 69 found the dominant single-chart-invalidity signal is not
multi-sheet folding but a scale mismatch: a chart boundary built from only
3-4 REPRESENTATIVE points cannot geometrically contain the region's much
larger region-owned FULL evidence footprint (worklog 67). This module
builds a DENSE boundary that actually reflects that footprint, without
touching region formation or representative topology -- the existing
representative boundary is the topology/provenance SEED every new vertex is
anchored to, never replaced wholesale.

All geometry here is done directly in WORLD 3D space, not a PCA-UV
reprojection: an earlier revision of this module projected the boundary and
evidence into a single shared `pca_parameterize_points` frame first, but
that frame is recomputed from whatever points are in the combined set --
adding outlying evidence changes the principal axes and can silently rotate
which original edge a given point is nearest to (caught during this
worklog's own test-fixture debugging). A chart boundary loop is already
(approximately) planar in 3D by construction, so 3D point-to-segment
distance for edge ownership, and a 3D tangent projection for ordering, are
both well-defined and avoid that instability entirely.

Algorithm (per boundary EDGE, never a global hull/rectangle):
  1. Assign each evidence point to its nearest ORIGINAL boundary edge (3D
     point-to-segment distance) -- this is the edge's "owned" evidence.
  2. Among an edge's owned evidence, keep only points genuinely farther from
     the loop's world-space centroid than both of the edge's own endpoints
     (by more than one local-evidence-scale unit -- the same already-
     established per-region full-cloud spacing estimate used elsewhere in
     this pipeline, never an arbitrary constant).
  3. A raw dump of every qualifying point, ordered only by its 1D tangent
     projection, is NOT safe on real noisy full-cloud evidence: dense real
     data has enough perpendicular scatter that consecutive projected points
     can still zig-zag past each other and self-intersect (measured
     empirically on the real 22-patch dataset before this revision -- most
     patches failed `validate_simple_closed_loop` once every qualifying
     point was inserted raw). Instead, the edge's tangent is divided into
     bins of width `local_evidence_scale` (so bin resolution follows the
     same non-arbitrary local spacing estimate, not a tuned constant), and
     only the single FARTHEST-from-centroid qualifying point in each bin is
     kept -- a coarser but still genuinely dense polyline that is
     monotonically ordered along the edge by construction (bin index can
     only increase), which is what actually keeps it locally non-self-
     crossing.
  4. The selected points for one edge are spliced in, in bin order, as an
     ordered sub-polyline replacing that edge; every new segment inherits
     the ORIGINAL edge's provenance type (physical_termination/crease/
     observation_frontier/partition_seam -- never invented, never mixed).
     An edge with no qualifying evidence is left exactly as the original
     representative boundary -- no gap is ever interpolated.
  5. The resulting loop is validated with the existing
     `validate_simple_closed_loop` (unmodified) -- ordered, simple,
     non-self-intersecting, non-branching by construction (each vertex has
     exactly two neighbors). If validation fails, materialization fails
     closed (`state="boundary_materialization_failed"`) rather than
     silently falling back to a convex hull or forcing a partial fix -- the
     per-edge binning is only a LOCAL coherence heuristic, not a global
     non-self-intersection guarantee (two different edges' polylines can
     still cross each other), so this final check is the real safety net.

Deliberately isolated (same convention as the other worklog-69/68 modules):
every function takes raw tensors, never a region/pipeline object.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-8


@dataclass(frozen=True)
class DenseBoundarySegment:
    node_a: Any
    node_b: Any
    segment_kind: str
    is_extension: bool  # True if either endpoint is a newly-inserted evidence vertex


@dataclass(frozen=True)
class DenseBoundaryResult:
    ordered_ids: tuple[Any, ...]
    ordered_positions: Any  # (K, 3)
    segments: tuple[DenseBoundarySegment, ...]
    extension_count: int  # total number of newly-inserted evidence vertices (not edges)
    original_vertex_count: int
    state: str  # "materialized" | "boundary_materialization_failed"
    reasons: tuple[str, ...]


def _point_segment_distance_3d(points: Any, a: Any, b: Any) -> Any:
    """Vectorized 3D point-to-segment distance, ``points`` is (M, 3)."""

    ab = b - a
    ab_len2 = (ab * ab).sum().clamp_min(_EPS)
    t = ((points - a) @ ab) / ab_len2
    t = t.clamp(0.0, 1.0)
    projection = a[None, :] + t[:, None] * ab[None, :]
    return (points - projection).norm(dim=-1)


def materialize_dense_boundary(
    boundary_positions_3d: Any,
    boundary_ids: Sequence[Any],
    segment_kinds: Sequence[str],
    evidence_positions_3d: Any,
    evidence_ids: Sequence[Any],
    *,
    local_evidence_scale: float,
) -> DenseBoundaryResult:
    """``segment_kinds[i]`` is the provenance type of edge
    ``(boundary_ids[i], boundary_ids[(i+1) % N])``. Returns a
    `DenseBoundaryResult` with ``state="boundary_materialization_failed"``
    (never a partial/best-effort loop) if the extended loop is not a valid
    simple closed polygon."""

    torch = require_torch()
    n = int(boundary_positions_3d.shape[0])
    m = int(evidence_positions_3d.shape[0])
    if n < 3:
        return DenseBoundaryResult((), boundary_positions_3d[:0], (), 0, n, "boundary_materialization_failed", ("boundary_too_small",))
    if m == 0:
        # No evidence to extend with -- the original boundary is already the answer.
        segments = tuple(
            DenseBoundarySegment(boundary_ids[i], boundary_ids[(i + 1) % n], segment_kinds[i], False)
            for i in range(n)
        )
        return DenseBoundaryResult(tuple(boundary_ids), boundary_positions_3d, segments, 0, n, "materialized", ())

    scale = max(float(local_evidence_scale), _EPS)
    centroid = boundary_positions_3d.mean(dim=0)

    # Assign each evidence point to its nearest ORIGINAL boundary edge, in
    # world space.
    edge_distances = torch.stack([
        _point_segment_distance_3d(evidence_positions_3d, boundary_positions_3d[i], boundary_positions_3d[(i + 1) % n])
        for i in range(n)
    ], dim=1)  # (M, N)
    nearest_edge = edge_distances.argmin(dim=1)

    boundary_dist_from_centroid = (boundary_positions_3d - centroid[None, :]).norm(dim=-1)
    evidence_dist_from_centroid = (evidence_positions_3d - centroid[None, :]).norm(dim=-1)

    new_ids: list[Any] = []
    new_positions: list[Any] = []
    new_segments: list[DenseBoundarySegment] = []
    extension_count = 0
    for i in range(n):
        new_ids.append(boundary_ids[i])
        new_positions.append(boundary_positions_3d[i])
        a_pos, b_pos = boundary_positions_3d[i], boundary_positions_3d[(i + 1) % n]
        endpoint_max = max(float(boundary_dist_from_centroid[i]), float(boundary_dist_from_centroid[(i + 1) % n]))
        owned_mask = nearest_edge == i
        owned_indices = torch.nonzero(owned_mask, as_tuple=False).reshape(-1)
        kind = segment_kinds[i]

        selected_local: list[int] = []
        if int(owned_indices.numel()) > 0:
            owned_dist = evidence_dist_from_centroid[owned_indices]
            qualifies = owned_dist > (endpoint_max + scale)
            selected_local = owned_indices[qualifies].tolist()

        if selected_local:
            # Bin the qualifying points along the edge's own world-space
            # tangent (bin width == local_evidence_scale) and keep only the
            # farthest-from-centroid point per bin -- a dense but locally
            # monotonic (never-backtracking) sub-polyline, safe against raw
            # per-point noise unlike inserting every qualifying point.
            tangent = b_pos - a_pos
            tangent_len = float(tangent.norm().clamp_min(_EPS))
            bin_count = max(1, int(math.ceil(tangent_len / scale)))
            projections = [
                float(((evidence_positions_3d[idx] - a_pos) @ tangent) / (tangent_len * tangent_len))
                for idx in selected_local
            ]
            best_per_bin: dict[int, tuple[float, int]] = {}
            for local_index, proj in zip(selected_local, projections):
                bin_index = min(bin_count - 1, max(0, int(proj * bin_count)))
                dist = float(evidence_dist_from_centroid[local_index])
                current = best_per_bin.get(bin_index)
                if current is None or dist > current[0]:
                    best_per_bin[bin_index] = (dist, local_index)
            ordered_local = [best_per_bin[b][1] for b in sorted(best_per_bin)]

            prev_id = boundary_ids[i]
            for local_index in ordered_local:
                ext_id = evidence_ids[local_index]
                new_ids.append(ext_id)
                new_positions.append(evidence_positions_3d[local_index])
                new_segments.append(DenseBoundarySegment(prev_id, ext_id, kind, True))
                prev_id = ext_id
            new_segments.append(DenseBoundarySegment(prev_id, boundary_ids[(i + 1) % n], kind, True))
            extension_count += len(ordered_local)
        else:
            new_segments.append(DenseBoundarySegment(boundary_ids[i], boundary_ids[(i + 1) % n], kind, False))

    new_positions_tensor = torch.stack(new_positions, dim=0)
    world_points = [tuple(float(v) for v in row) for row in new_positions_tensor.detach().cpu().tolist()]
    report = validate_simple_closed_loop(world_points)
    if not report.is_simple_polygon:
        return DenseBoundaryResult(
            tuple(new_ids), new_positions_tensor, tuple(new_segments), extension_count, n,
            "boundary_materialization_failed", ("self_intersection_check_failed",) + report.reasons,
        )
    return DenseBoundaryResult(
        tuple(new_ids), new_positions_tensor, tuple(new_segments), extension_count, n, "materialized", (),
    )
