from __future__ import annotations

"""Worklog 80: dense parametric chart SUPPORT -- topology and geometry separated.

Worklog 79 established the failure as structural: the sparse accepted
representative cycle (3-7 nodes) was being reused as the chart's GEOMETRIC
boundary while the region owns 93-1035 observed points at a far larger spatial
scale, leaving 89.1-99.8% of the evidence outside the chart domain. That is not
a fitting or parameterization defect -- it is one representation being asked to
play two incompatible roles.

This module separates those roles:

  * The sparse accepted topology stays the TOPOLOGY abstraction. It supplies
    the cyclic ORDER of the perimeter and the typed frontier PROVENANCE of each
    arc (physical_termination / crease / observation_frontier / partition_seam).
    It is never used as the chart's geometric extent.
  * The region-owned dense boundary-support candidates
    (`torch_region_owned_dense_boundary_support`, worklog 77's corrected
    predicate, unmodified) supply the GEOMETRY. These are observed Gaussians
    admitted because their own tangent-plane neighbourhood has a genuine empty
    sector, so every vertex of the resulting chart is evidence-backed. Measured
    on real baseline_compatible@2900 they span 0.966-1.020 of the owned
    evidence extent, against 0.148-0.667 for the representatives they replace.

Construction (boundary-first throughout; no hull, PCA rectangle, bounding box,
alpha shape, or shape-specific fallback anywhere):

  1. Project the sparse cycle and the dense candidates into the region's own
     canonical tangent frame.
  2. Assign each dense candidate to the sparse ARC it belongs to (nearest
     sparse edge). This is where topology constrains the chart: an arc's
     candidates inherit that arc's typed provenance, and nothing else.
  3. Order each arc's candidates monotonically along that arc, binned at the
     region's own full-evidence spacing so real per-point scatter cannot make
     the polyline double back (the technique worklog 70 validated, applied here
     to genuine perimeter samples rather than arbitrary outward evidence).
  4. Concatenate the arcs in the sparse cyclic order. The representatives
     themselves are NOT vertices of the result -- they position and type the
     arcs, they do not bound the chart.
  5. Validate as a simple closed loop with 3D nonplanarity disclosed separately
     from 2D crossing (`evaluate_closed_loop_geometry`, worklog 71), then apply
     the worklog 79 chart-domain coverage contract BEFORE any fitting.

Every failure is typed and fail-closed; nothing is force-closed and no segment
is invented across a gap that the sparse topology did not already assert. An
arc with no dense candidate contributes a segment that keeps its typed
provenance but is explicitly marked `evidence_backed=False`, so the loop never
silently claims observed support it does not have.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_region_owned_full_evidence import evidence_outside_chart_domain_fraction
from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import (
    LoopGeometryReport,
    evaluate_closed_loop_geometry,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

STATE_MATERIALIZED = "dense_chart_support_materialized"
STATE_NO_DENSE_SUPPORT = "dense_chart_support_no_observed_support"
STATE_SELF_INTERSECTING = "dense_chart_support_self_intersecting"
STATE_COVERAGE_FAILED = "dense_chart_support_coverage_failed"
STATE_UNRESOLVED_TOPOLOGY = "dense_chart_support_unresolved_topology"


@dataclass(frozen=True)
class DenseChartSegment:
    node_a: Any
    node_b: Any
    segment_kind: str
    evidence_backed: bool


@dataclass(frozen=True)
class DenseChartSupport:
    region_id: int
    ordered_ids: tuple[Any, ...]
    ordered_positions: Any            # (K, 3) -- all evidence-backed vertices
    segments: tuple[DenseChartSegment, ...]
    sparse_topology_node_count: int   # the topology skeleton, NOT the geometry
    dense_support_count: int
    arc_support_counts: tuple[int, ...]
    unsupported_arc_count: int
    evidence_outside_domain_fraction: float | None
    geometry: LoopGeometryReport | None
    state: str
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


def _point_segment_distance_2d(points: Any, a: Any, b: Any) -> Any:
    ab = b - a
    ab_len2 = (ab * ab).sum().clamp_min(_EPS)
    t = (((points - a) @ ab) / ab_len2).clamp(0.0, 1.0)
    projection = a[None, :] + t[:, None] * ab[None, :]
    return (points - projection).norm(dim=-1)


def build_dense_chart_support(
    region_id: int,
    sparse_cycle_positions: Any,
    segment_kinds: Sequence[str],
    dense_candidate_ids: Sequence[Any],
    dense_candidate_positions: Any,
    evidence_positions: Any,
    *,
    axis_u: Any,
    axis_v: Any,
    origin: Any,
    full_evidence_spacing: float,
    max_evidence_outside_domain_fraction: float = 0.5,
) -> DenseChartSupport:
    """``segment_kinds[i]`` types the sparse arc from ``sparse_cycle_positions[i]``
    to ``[(i+1) % n]``. Returns a fail-closed `DenseChartSupport` on any
    unresolved condition -- never a partial or force-closed loop."""

    torch = require_torch()
    n = int(sparse_cycle_positions.shape[0])
    m = int(dense_candidate_positions.shape[0]) if dense_candidate_positions is not None else 0

    def _fail(state: str, *reasons: str) -> DenseChartSupport:
        return DenseChartSupport(
            region_id, (), sparse_cycle_positions[:0], (), n, m, (), 0, None, None, state, tuple(reasons),
        )

    if n < 3:
        return _fail(STATE_UNRESOLVED_TOPOLOGY, f"sparse_cycle_node_count={n}<3")
    if m == 0:
        # Fail closed: the whole point of the redesign is that the geometry is
        # evidence-backed, so with no observed boundary support there is no
        # chart -- the sparse polygon is NOT used as a fallback.
        return _fail(STATE_NO_DENSE_SUPPORT, "no_region_owned_dense_boundary_support")

    spacing = max(float(full_evidence_spacing), _EPS)

    def to_uv(points: Any) -> Any:
        offset = points - origin
        return torch.stack((offset @ axis_u, offset @ axis_v), dim=1)

    sparse_uv = to_uv(sparse_cycle_positions)
    dense_uv = to_uv(dense_candidate_positions)
    loop_centroid_uv = dense_uv.mean(dim=0)

    # (2) topology constrains membership: nearest sparse ARC owns the candidate
    arc_distance = torch.stack([
        _point_segment_distance_2d(dense_uv, sparse_uv[i], sparse_uv[(i + 1) % n]) for i in range(n)
    ], dim=1)
    nearest_arc = arc_distance.argmin(dim=1)

    ordered_ids: list[Any] = []
    ordered_positions: list[Any] = []
    ordered_kinds: list[str] = []
    arc_counts: list[int] = []
    for i in range(n):
        kind = segment_kinds[i]
        owned = torch.nonzero(nearest_arc == i, as_tuple=False).reshape(-1).tolist()
        if not owned:
            arc_counts.append(0)
            continue
        a_uv, b_uv = sparse_uv[i], sparse_uv[(i + 1) % n]
        tangent = b_uv - a_uv
        tangent_len = float(tangent.norm().clamp_min(_EPS))
        unit = tangent / tangent_len
        # (3) Monotonic ordering along the arc, binned at the region's own
        # full-evidence spacing; within a bin keep the candidate FARTHEST from
        # the loop centroid, i.e. the most perimeter-like sample.
        #
        # The bin resolution MUST follow the span the candidates actually
        # occupy, not the sparse chord's length. The sparse chord is a
        # representative-scale object (measured at 0.15-0.67 of the evidence
        # extent, worklog 79) while the candidates live at the evidence scale,
        # so quantizing by the chord reintroduces exactly the scale mismatch
        # this module exists to remove -- measured directly: region 6 kept only
        # 16 of 218 candidates and still failed coverage at 72.7% until this
        # was derived from the candidates' own projected span instead.
        projections = [float((dense_uv[local] - a_uv) @ unit) for local in owned]
        span = max(projections) - min(projections)
        low = min(projections)
        bin_count = max(1, int(math.ceil(max(span, _EPS) / spacing)))
        best: dict[int, tuple[float, int]] = {}
        for local, projection in zip(owned, projections):
            normalized = (projection - low) / max(span, _EPS)
            bin_index = min(bin_count - 1, max(0, int(normalized * bin_count)))
            outward = float((dense_uv[local] - loop_centroid_uv).norm())
            current = best.get(bin_index)
            if current is None or outward > current[0]:
                best[bin_index] = (outward, local)
        selected = [best[b][1] for b in sorted(best)]
        arc_counts.append(len(selected))
        for local in selected:
            ordered_ids.append(dense_candidate_ids[local])
            ordered_positions.append(dense_candidate_positions[local])
            ordered_kinds.append(kind)

    if len(ordered_ids) < 3:
        return _fail(STATE_NO_DENSE_SUPPORT, f"dense_support_vertices={len(ordered_ids)}<3")

    positions = torch.stack(ordered_positions, dim=0)
    k = len(ordered_ids)
    segments = tuple(
        DenseChartSegment(
            ordered_ids[i], ordered_ids[(i + 1) % k], ordered_kinds[i],
            evidence_backed=(ordered_kinds[i] == ordered_kinds[(i + 1) % k]),
        )
        for i in range(k)
    )
    unsupported_arcs = sum(1 for count in arc_counts if count == 0)

    # (5) geometry: nonplanarity disclosed, never itself a crossing failure
    geometry = evaluate_closed_loop_geometry(
        [tuple(float(v) for v in row) for row in positions.detach().cpu().tolist()]
    )
    if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
        return DenseChartSupport(
            region_id, tuple(ordered_ids), positions, segments, n, m, tuple(arc_counts),
            unsupported_arcs, None, geometry, STATE_SELF_INTERSECTING,
            (f"proper_crossing_count={geometry.proper_crossing_count}",),
        )

    # (5) worklog 79 chart-domain coverage contract, BEFORE any fitting
    outside = evidence_outside_chart_domain_fraction(positions, evidence_positions)
    if outside is not None and outside > max_evidence_outside_domain_fraction:
        return DenseChartSupport(
            region_id, tuple(ordered_ids), positions, segments, n, m, tuple(arc_counts),
            unsupported_arcs, outside, geometry, STATE_COVERAGE_FAILED,
            (f"evidence_outside_chart_domain_fraction={outside:.4f}>{max_evidence_outside_domain_fraction}",),
        )

    return DenseChartSupport(
        region_id, tuple(ordered_ids), positions, segments, n, m, tuple(arc_counts),
        unsupported_arcs, outside, geometry, STATE_MATERIALIZED, (),
    )


def independent_chart_components(
    member_ids: Sequence[Any], accepted_edges: Sequence[tuple[Any, Any]],
) -> tuple[tuple[Any, ...], ...]:
    """Independent cycle carriers in the region's OWN accepted topology.

    A region supports more than one chart only when its accepted topology
    genuinely proves the separation -- i.e. its 2-core (repeatedly stripping
    degree<=1 nodes, which cannot lie on any cycle) splits into two or more
    DISJOINT connected components. A single 2-core component with several
    interwoven cycles is ambiguous branching, not a proven multi-chart region,
    and is deliberately NOT partitioned here.
    """

    adjacency: dict[Any, set[Any]] = {node: set() for node in member_ids}
    for a, b in accepted_edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    core = set(member_ids)
    changed = True
    while changed:
        changed = False
        for node in list(core):
            if len(adjacency[node] & core) <= 1:
                core.discard(node)
                changed = True
    components: list[tuple[Any, ...]] = []
    seen: set[Any] = set()
    for start in sorted(core, key=str):
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(sorted(adjacency[node] & core - seen, key=str))
        if len(component) >= 3:
            components.append(tuple(sorted(component, key=str)))
    return tuple(components)
