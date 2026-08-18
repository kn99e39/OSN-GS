from __future__ import annotations

"""Worklog 96 -- typed seed-curve construction, with the legacy
``eligible_parametric_chart_boundary`` gate removed as a hard entrance
requirement.

Four seed semantics, always disclosed explicitly and never conflated:

- ``SEED_PHYSICAL_BOUNDARY`` / ``SEED_CREASE_FEATURE`` /
  ``SEED_OBSERVATION_FRONTIER``: the existing Worklog 79/80 sparse
  parametric chart boundary (``construct_region_parametric_chart_boundaries``
  output, already produced by the fixed canonical-construction pipeline),
  typed by its own existing ``segment_kind``. This is preserved and
  preferred whenever it exists -- it is NOT reconstructed here.
- ``SEED_INTERIOR_CONSTRUCTION``: only used when no boundary/typed seed
  curve survives. A small, coverage-preserving, farthest-point-sampled set
  of region-evidence anchors, each individually verified supported by the
  latent surface before being used. Anchors are STARTING LOCATIONS ONLY --
  no adjacency/connectivity is built between them from raw Gaussian
  centers; each anchor independently seeds its own traced curve. This
  never fabricates physical-boundary semantics for an interior seed.

No convex hull, PCA rectangle, bounding box, alpha shape, forced closure,
or arbitrary gap bridging anywhere in this module.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_tracer import sample_segment_continuous_support
from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

SEED_PHYSICAL_BOUNDARY = "physical_boundary"
SEED_CREASE_FEATURE = "crease_feature"
SEED_OBSERVATION_FRONTIER = "observation_frontier"
SEED_INTERIOR_CONSTRUCTION = "interior_construction"

_CHART_SEGMENT_KIND_TO_SEED_TYPE = {
    "physical_termination": SEED_PHYSICAL_BOUNDARY,
    "crease": SEED_CREASE_FEATURE,
    "observation_frontier": SEED_OBSERVATION_FRONTIER,
    # partition_seam has no direct physical/crease/frontier provenance of
    # its own; it is disclosed as an observation-frontier-class seed
    # (weakest, non-physical claim) rather than invented as boundary.
    "partition_seam": SEED_OBSERVATION_FRONTIER,
}
_ELIGIBLE_CHART_STATUS = "eligible_parametric_chart_boundary"

DEFAULT_STEPS_PER_BOUNDARY_EDGE = 6
DEFAULT_INTERIOR_ANCHOR_COUNT = 6
_MIN_ANCHOR_SUPPORT_CHECK = True  # always verify each interior anchor individually; never skip


@dataclass(frozen=True)
class SeedCurve:
    seed_id: str
    seed_type: str
    points: Any  # (N, 3), continuously supported (verified segment-by-segment at construction time)
    provenance: str


def _densify_boundary_edge(
    position_a: Any, position_b: Any, support: LatentSurfaceSupport, steps: int,
) -> tuple[Any, bool]:
    return sample_segment_continuous_support(support, position_a, position_b, steps)


def _build_boundary_seed_curves(
    chart: Any, representative_positions: Any, representative_index: dict,
    support: LatentSurfaceSupport, steps_per_edge: int,
) -> tuple[SeedCurve, ...]:
    if chart is None or chart.status != _ELIGIBLE_CHART_STATUS or len(chart.ordered_node_ids) < 3:
        return ()
    nodes = [node for node in chart.ordered_node_ids if node in representative_index]
    if len(nodes) < 2:
        return ()
    positions = [representative_positions[representative_index[node]] for node in nodes]
    count = len(nodes)
    curves: list[SeedCurve] = []
    for index in range(count):
        node_a, node_b = nodes[index], nodes[(index + 1) % count]
        position_a, position_b = positions[index], positions[(index + 1) % count]
        kind = None
        if index < len(chart.segments):
            kind = chart.segments[index].segment_kind
        seed_type = _CHART_SEGMENT_KIND_TO_SEED_TYPE.get(kind, SEED_OBSERVATION_FRONTIER)
        points, _fully_supported = _densify_boundary_edge(position_a, position_b, support, steps_per_edge)
        if int(points.shape[0]) < 2:
            continue
        curves.append(SeedCurve(f"boundary:{node_a}-{node_b}", seed_type, points, f"chart_edge:{node_a}-{node_b}:{kind}"))
    return tuple(curves)


def _farthest_point_anchor_indices(evidence: Any, count: int) -> list[int]:
    """Deterministic greedy farthest-point sampling: a fixed, coverage-
    preserving anchor selection, never randomized and never tuned per
    replay outcome. Starts from the point nearest the evidence centroid so
    the result is reproducible given the same evidence."""

    torch = require_torch()
    n = int(evidence.shape[0])
    count = min(count, n)
    if count <= 0:
        return []
    centroid = evidence.mean(dim=0, keepdim=True)
    first = int(torch.cdist(evidence, centroid).reshape(-1).argmin().item())
    selected = [first]
    min_distance = torch.cdist(evidence, evidence[first : first + 1]).reshape(-1)
    while len(selected) < count:
        next_index = int(min_distance.argmax().item())
        if next_index in selected:
            break
        selected.append(next_index)
        new_distance = torch.cdist(evidence, evidence[next_index : next_index + 1]).reshape(-1)
        min_distance = torch.minimum(min_distance, new_distance)
    return selected


def _build_interior_seed_curves(
    evidence: Any, support: LatentSurfaceSupport, anchor_count: int,
) -> tuple[SeedCurve, ...]:
    from osn_gs.surface.torch_latent_surface_curve_tracer import trace_bidirectional

    torch = require_torch()
    anchor_indices = _farthest_point_anchor_indices(evidence, anchor_count)
    curves: list[SeedCurve] = []
    for anchor_index in anchor_indices:
        anchor_position = evidence[anchor_index]
        result = support.query_batch(anchor_position.reshape(1, 3))
        if not bool(result.supported[0]):
            continue  # cannot seed a curve from an unsupported anchor -- fail closed
        # Deterministic primary direction: the local frame's own tangent_u
        # axis at the anchor (never a raw-center-derived direction, never
        # arbitrary). tangent_v is reserved for the family-V trace built by
        # the curve-family module.
        primary_direction = result.tangent_u[0]
        traced = trace_bidirectional(result.positions[0], primary_direction, support)
        if int(traced.points.shape[0]) < 2:
            continue
        curves.append(SeedCurve(
            f"interior:{anchor_index}", SEED_INTERIOR_CONSTRUCTION, traced.points,
            f"interior_anchor_index:{anchor_index}",
        ))
    return tuple(curves)


def build_seed_curves(
    evidence: Any,
    chart: Any,
    representative_positions: Any,
    representative_index: dict,
    support: LatentSurfaceSupport,
    *,
    steps_per_boundary_edge: int = DEFAULT_STEPS_PER_BOUNDARY_EDGE,
    interior_anchor_count: int = DEFAULT_INTERIOR_ANCHOR_COUNT,
) -> tuple[SeedCurve, ...]:
    """Boundary/typed seeds are preferred and preserved whenever any
    survive; interior construction seeds are used ONLY as the fallback
    when no boundary/typed seed curve survives (chart ineligible, or every
    boundary edge lost support immediately) -- never as a silent
    additional enrichment that would blur the semantic distinction."""

    boundary_curves = _build_boundary_seed_curves(
        chart, representative_positions, representative_index, support, steps_per_boundary_edge,
    )
    if boundary_curves:
        return boundary_curves
    return _build_interior_seed_curves(evidence, support, interior_anchor_count)
