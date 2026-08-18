from __future__ import annotations

"""Worklog 96 -- surface-following curve tracing on a
:class:`~osn_gs.surface.torch_latent_surface_support.LatentSurfaceSupport`.

Two primitives shared by seed construction and curve-family construction:

- :func:`trace_curve` walks a coherent surface-following path using
  parallel-transport-style tangent continuation: the walking direction is
  projected onto each new local tangent plane (never re-derived by picking
  an arbitrary PCA-axis sign at every step), then re-grounded by the actual
  realized displacement so it cannot silently invert. Every step is
  validated by the latent surface estimator; loss of support terminates
  the walk immediately.
- :func:`sample_segment_continuous_support` densifies a straight chord
  between two ALREADY-SUPPORTED points and requires every intermediate
  sample to also be supported -- two supported endpoints alone are never
  sufficient to accept a segment.

Neither function ever mutates ``support.support_points`` or falls back to
convex hull / bounding box / PCA rectangle / alpha shape reasoning.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

DEFAULT_TRACE_STEP_COUNT = 12
DEFAULT_TRACE_STEP_SCALE = 1.0  # multiple of the support's own median spacing
DEFAULT_SEGMENT_STEPS = 4


def propagate_tangent_onto_plane(previous_direction: Any, normal: Any) -> Any:
    """Parallel-transport-style continuation: project ``previous_direction``
    onto the tangent plane defined by ``normal`` and renormalize. This is
    what keeps a traced curve coherent -- the walking direction carries
    forward from the previous step rather than being independently
    reselected (with an arbitrary sign) from the new local frame's
    eigenvectors at every step."""

    torch = require_torch()
    normal = normal / normal.norm().clamp_min(_EPS)
    projected = previous_direction - (previous_direction * normal).sum() * normal
    norm = projected.norm()
    if float(norm.item()) < _EPS:
        # Direction became degenerate (previous direction was ~parallel to
        # the new normal) -- fail closed rather than inventing a direction.
        return None
    return projected / norm


@dataclass(frozen=True)
class TracedCurve:
    points: Any  # (N, 3), N >= 1 (at least the start point)
    terminated_reason: str  # "step_count_reached" | "unsupported" | "degenerate_direction"


def trace_curve(
    start_position: Any,
    start_direction: Any,
    support: LatentSurfaceSupport,
    *,
    step_count: int = DEFAULT_TRACE_STEP_COUNT,
    step_size: float | None = None,
) -> TracedCurve:
    """Walk a single surface-following curve from ``start_position`` along
    ``start_direction`` (assumed already tangent to the surface at that
    point; callers are responsible for deriving it from the local frame).
    Terminates the moment a step loses support -- never bridges past it."""

    torch = require_torch()
    if step_size is None:
        step_size = DEFAULT_TRACE_STEP_SCALE * support.median_spacing

    current = start_position
    direction = start_direction / start_direction.norm().clamp_min(_EPS)
    points = [current.reshape(1, 3)]
    reason = "step_count_reached"
    for _step in range(step_count):
        result = support.query_batch(current.reshape(1, 3))
        if not bool(result.supported[0]):
            reason = "unsupported"
            break
        propagated = propagate_tangent_onto_plane(direction, result.normals[0])
        if propagated is None:
            reason = "degenerate_direction"
            break
        predicted = current + step_size * propagated
        stepped = support.query_batch(predicted.reshape(1, 3))
        if not bool(stepped.supported[0]):
            reason = "unsupported"
            break
        new_position = stepped.positions[0]
        realized = new_position - current
        if float(realized.norm().item()) < _EPS:
            reason = "degenerate_direction"
            break
        # Ground the next direction in the actual realized displacement
        # rather than the raw parallel-transported vector -- this is what
        # prevents drift/sign ambiguity from accumulating step over step.
        direction = realized / realized.norm().clamp_min(_EPS)
        current = new_position
        points.append(current.reshape(1, 3))
    return TracedCurve(torch.cat(points, dim=0), reason)


def trace_bidirectional(
    start_position: Any,
    start_direction: Any,
    support: LatentSurfaceSupport,
    *,
    step_count: int = DEFAULT_TRACE_STEP_COUNT,
    step_size: float | None = None,
) -> TracedCurve:
    """Trace both signs of ``start_direction`` and splice the two halves
    around the shared start point into one continuous curve."""

    torch = require_torch()
    forward = trace_curve(start_position, start_direction, support, step_count=step_count, step_size=step_size)
    backward = trace_curve(start_position, -start_direction, support, step_count=step_count, step_size=step_size)
    # backward.points[0] == forward.points[0] == start_position; drop the
    # duplicate and reverse the backward half so the whole curve is ordered
    # monotonically from one end to the other.
    backward_prefix = torch.flip(backward.points[1:], dims=(0,)) if backward.points.shape[0] > 1 else backward.points[:0]
    points = torch.cat([backward_prefix, forward.points], dim=0)
    reason = f"forward={forward.terminated_reason},backward={backward.terminated_reason}"
    return TracedCurve(points, reason)


def sample_segment_continuous_support(
    support: LatentSurfaceSupport,
    a: Any,
    b: Any,
    steps: int = DEFAULT_SEGMENT_STEPS,
) -> tuple[Any, bool]:
    """Densify the straight chord from ``a`` to ``b`` in ``steps``
    increments, MLS-correcting each intermediate sample. Returns the kept
    prefix and whether the FULL segment (including the endpoint ``b``)
    stayed supported -- a segment with only its two endpoints supported is
    never accepted; every intermediate sample must also be supported."""

    torch = require_torch()
    kept = [a.reshape(1, 3)]
    fully_supported = True
    for step in range(1, steps + 1):
        t = step / steps
        predicted = (1.0 - t) * a + t * b
        result = support.query_batch(predicted.reshape(1, 3))
        if not bool(result.supported[0]):
            fully_supported = False
            break
        kept.append(result.positions[0].reshape(1, 3))
    return torch.cat(kept, dim=0), fully_supported
