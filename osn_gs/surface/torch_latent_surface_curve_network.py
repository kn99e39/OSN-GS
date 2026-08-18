from __future__ import annotations

"""Structural curve network over a region's latent surface support.

seed curve (existing reliable sparse chart boundary, unmodified production
output) -> surface-following transversal curves -> interior rung curves ->
curve network -- feeding an unmodified NURBS fitter downstream.

Never reconstructs a dense point manifold, never requires a closed observed
boundary, never falls back to a convex hull / bounding box / PCA rectangle /
alpha shape to close a gap. Every point in every curve is individually
validated by :class:`~osn_gs.surface.torch_latent_surface_support.LatentSurfaceSupport`
before being kept; an unsupported step truncates that curve at the last
supported point rather than bridging past it.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# Fixed conventions, not tuned toward any replay outcome:
DEFAULT_STEPS_PER_SEED_EDGE = 6
DEFAULT_TRANSVERSAL_STEP_COUNT = 12
DEFAULT_TRANSVERSAL_STEP_SCALE = 1.0  # multiple of the support's own median spacing


@dataclass(frozen=True)
class SeedCurveSegment:
    node_a: Any
    node_b: Any
    segment_kind: str
    points: Any  # (S, 3) supported points, S in [1, steps_per_edge+1]; may be a
    # truncated prefix if support was lost partway along the edge.
    fully_supported: bool


@dataclass(frozen=True)
class TransversalCurve:
    seed_point_index: int  # index into the retained seed curve point list
    points: Any  # (T, 3), T >= 1 (at least the starting seed point)


@dataclass
class CurveNetworkResult:
    region_id: int
    status: str
    reason: str
    seed_segments: tuple[SeedCurveSegment, ...] = field(default_factory=tuple)
    transversal_curves: tuple[TransversalCurve, ...] = field(default_factory=tuple)
    rung_curves: tuple[Any, ...] = field(default_factory=tuple)  # each (R, 3)
    all_points: Any | None = None  # (M, 3) concatenation of every retained curve point

    @property
    def has_curve_network(self) -> bool:
        return self.status == STATUS_CURVE_NETWORK and self.all_points is not None and int(self.all_points.shape[0]) >= 4


STATUS_CURVE_NETWORK = "curve_network"
STATUS_NO_ELIGIBLE_SEED_CHART = "no_eligible_seed_chart"
STATUS_NO_SUPPORTED_SEED_POINTS = "no_supported_seed_points"


def _densify_seed_edge(
    position_a: Any, position_b: Any, support: LatentSurfaceSupport, steps: int,
) -> tuple[Any, bool]:
    """Walk the straight chord from a to b in ``steps`` increments, MLS-
    correcting each step onto the latent surface. Stops (truncates) the
    moment a step is unsupported -- never bridges past it."""

    torch = require_torch()
    kept: list[Any] = [position_a.reshape(1, 3)]
    fully_supported = True
    for step in range(1, steps + 1):
        t = step / steps
        predicted = (1.0 - t) * position_a + t * position_b
        result = support.query_batch(predicted.reshape(1, 3))
        if not bool(result.supported[0]):
            fully_supported = False
            break
        kept.append(result.positions[0].reshape(1, 3))
    return torch.cat(kept, dim=0), fully_supported


def _build_seed_curve(
    chart: Any, representative_positions: Any, representative_index: dict, support: LatentSurfaceSupport,
    steps_per_edge: int,
) -> tuple[SeedCurveSegment, ...]:
    segments: list[SeedCurveSegment] = []
    nodes = [node for node in chart.ordered_node_ids if node in representative_index]
    if len(nodes) < 2:
        return ()
    positions = [representative_positions[representative_index[node]] for node in nodes]
    count = len(nodes)
    for index in range(count):
        node_a, node_b = nodes[index], nodes[(index + 1) % count]
        position_a, position_b = positions[index], positions[(index + 1) % count]
        kind = None
        if index < len(chart.segments):
            kind = chart.segments[index].segment_kind
        points, fully_supported = _densify_seed_edge(position_a, position_b, support, steps_per_edge)
        if int(points.shape[0]) < 2:
            continue  # even the start point failed to stay on this edge's walk
        segments.append(SeedCurveSegment(node_a, node_b, kind or "unknown", points, fully_supported))
    return tuple(segments)


def _pick_transversal_direction(
    seed_direction: Any, tangent_u: Any, tangent_v: Any, inward_hint: Any,
) -> Any:
    """Choose whichever local tangent-basis vector is more orthogonal to the
    seed curve's own direction (the transversal direction), signed toward
    the seed curve's own centroid so curves walk inward rather than off the
    observed patch."""

    torch = require_torch()
    seed_direction = seed_direction / seed_direction.norm().clamp_min(_EPS)
    align_u = torch.abs((tangent_u * seed_direction).sum())
    align_v = torch.abs((tangent_v * seed_direction).sum())
    candidate = tangent_v if align_v < align_u else tangent_u
    sign = torch.sign((inward_hint * candidate).sum())
    if float(sign.item()) == 0.0:
        sign = torch.tensor(1.0, device=candidate.device, dtype=candidate.dtype)
    return candidate * sign


def _trace_transversal_curve(
    start_position: Any, seed_direction: Any, centroid: Any, support: LatentSurfaceSupport,
    *, step_count: int, step_size: float,
) -> Any:
    torch = require_torch()
    current = start_position
    points = [current.reshape(1, 3)]
    for _step in range(step_count):
        result = support.query_batch(current.reshape(1, 3))
        if not bool(result.supported[0]):
            break
        inward_hint = centroid - current
        direction = _pick_transversal_direction(
            seed_direction, result.tangent_u[0], result.tangent_v[0], inward_hint,
        )
        predicted = current + step_size * direction
        stepped = support.query_batch(predicted.reshape(1, 3))
        if not bool(stepped.supported[0]):
            break
        current = stepped.positions[0]
        points.append(current.reshape(1, 3))
    return torch.cat(points, dim=0)


def build_latent_surface_curve_network(
    region_id: int,
    chart: Any,
    representative_positions: Any,
    representative_index: dict,
    support: LatentSurfaceSupport,
    *,
    steps_per_seed_edge: int = DEFAULT_STEPS_PER_SEED_EDGE,
    transversal_step_count: int = DEFAULT_TRANSVERSAL_STEP_COUNT,
    transversal_step_scale: float = DEFAULT_TRANSVERSAL_STEP_SCALE,
) -> CurveNetworkResult:
    """Region-level curve network: seed curve (existing sparse chart
    boundary, densified and MLS-corrected onto the latent surface) plus
    surface-following transversal curves walked inward from retained seed
    points, plus interior rung curves connecting transversal curves at
    equal step depth. Fails closed at every stage -- a region with no
    eligible seed chart, or whose entire seed curve loses support, produces
    no curve network rather than falling back to any hull/box/rectangle
    construction.
    """

    torch = require_torch()
    ELIGIBLE_STATUS = "eligible_parametric_chart_boundary"
    if chart is None or chart.status != ELIGIBLE_STATUS or len(chart.ordered_node_ids) < 3:
        return CurveNetworkResult(region_id, STATUS_NO_ELIGIBLE_SEED_CHART, "no_eligible_parametric_chart_boundary")

    seed_segments = _build_seed_curve(chart, representative_positions, representative_index, support, steps_per_seed_edge)
    if not seed_segments:
        return CurveNetworkResult(region_id, STATUS_NO_SUPPORTED_SEED_POINTS, "seed_curve_lost_support_immediately")

    seed_points = torch.cat([segment.points for segment in seed_segments], dim=0)
    if int(seed_points.shape[0]) < 3:
        return CurveNetworkResult(
            region_id, STATUS_NO_SUPPORTED_SEED_POINTS, "insufficient_supported_seed_points",
            seed_segments=seed_segments,
        )
    centroid = seed_points.mean(dim=0)

    transversal_curves: list[TransversalCurve] = []
    step_size = transversal_step_scale * support.median_spacing
    for segment in seed_segments:
        segment_points = segment.points
        if int(segment_points.shape[0]) < 2:
            continue
        seed_direction = segment_points[-1] - segment_points[0]
        if float(seed_direction.norm().item()) < _EPS:
            continue
        for local_index in range(int(segment_points.shape[0])):
            start = segment_points[local_index]
            curve_points = _trace_transversal_curve(
                start, seed_direction, centroid, support,
                step_count=transversal_step_count, step_size=step_size,
            )
            if int(curve_points.shape[0]) >= 2:
                transversal_curves.append(TransversalCurve(len(transversal_curves), curve_points))

    rung_curves: list[Any] = []
    if transversal_curves:
        max_depth = max(int(curve.points.shape[0]) for curve in transversal_curves)
        for depth in range(max_depth):
            rung_points = [
                curve.points[depth].reshape(1, 3) for curve in transversal_curves
                if int(curve.points.shape[0]) > depth
            ]
            if len(rung_points) >= 3:
                rung_curves.append(torch.cat(rung_points, dim=0))

    all_point_sets = [seed_points] + [curve.points for curve in transversal_curves] + rung_curves
    all_points = torch.cat(all_point_sets, dim=0) if all_point_sets else None

    status = STATUS_CURVE_NETWORK if all_points is not None and int(all_points.shape[0]) >= 4 else STATUS_NO_SUPPORTED_SEED_POINTS
    reason = "" if status == STATUS_CURVE_NETWORK else "insufficient_curve_network_points"
    return CurveNetworkResult(
        region_id, status, reason, seed_segments=seed_segments,
        transversal_curves=tuple(transversal_curves), rung_curves=tuple(rung_curves), all_points=all_points,
    )
