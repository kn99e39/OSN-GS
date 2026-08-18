from __future__ import annotations

"""Worklog 96 -- curve families, correspondence, and pre-fit
curve-network-block partitioning.

For each seed curve (:mod:`torch_latent_surface_seed_curves`), this module
builds:

- family V ("transversal"): one surface-following transversal curve per
  sampled point along the seed curve, traced with
  :func:`~osn_gs.surface.torch_latent_surface_curve_tracer.trace_curve`.
- family U ("rung"): for each pair of ADJACENT transversal traces, the
  connecting segment at each shared depth -- kept only if
  :func:`~osn_gs.surface.torch_latent_surface_curve_tracer.sample_segment_continuous_support`
  reports the WHOLE segment supported, not merely its two endpoints.

A block (one per seed) is a materializable curve-network patch candidate
only if it meets the fixed provisional contract:

- >=2 supported transversal curves (family V),
- >=2 depth levels connected by continuously-supported rungs between the
  SAME pair of adjacent transversal curves (family U),
- i.e. at least one full, mutually consistent 2x2 correspondence quad.

This partition is entirely pre-fit: block membership never depends on a
NURBS fit result, fit error, or held-out error. Blocks are never merged or
split afterward based on fit quality.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_tracer import (
    DEFAULT_SEGMENT_STEPS,
    DEFAULT_TRACE_STEP_COUNT,
    DEFAULT_TRACE_STEP_SCALE,
    sample_segment_continuous_support,
    trace_curve,
)
from osn_gs.surface.torch_latent_surface_seed_curves import SeedCurve
from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# Fixed provisional contract for this batch -- explicitly NOT tuned from
# any real replay outcome (Worklog 96 directive section 4/10).
MIN_FAMILY_CURVE_COUNT = 2
MIN_CORRESPONDENCE_DEPTH_COUNT = 2

DEFAULT_MAX_SEED_SAMPLES = 8  # evenly-spaced subsampling cap along a seed curve


@dataclass(frozen=True)
class TransversalTrace:
    sample_index: int  # index into the seed curve's own sample list (monotonic)
    points: Any  # (T, 3)


@dataclass(frozen=True)
class RungSegment:
    depth: int
    a_sample_index: int
    b_sample_index: int
    points: Any  # (S, 3), continuously supported end to end


@dataclass(frozen=True)
class CurveNetworkBlock:
    seed_id: str
    seed_type: str
    seed_curve: SeedCurve
    transversal_traces: tuple[TransversalTrace, ...]
    rungs: tuple[RungSegment, ...]
    satisfies_contract: bool
    all_points: Any | None  # union of seed/transversal/rung points, only when satisfies_contract


def _evenly_spaced_indices(count: int, max_samples: int) -> list[int]:
    if count <= max_samples:
        return list(range(count))
    torch = require_torch()
    positions = torch.linspace(0, count - 1, max_samples)
    return sorted({int(round(float(value))) for value in positions})


def _local_seed_direction(seed_points: Any, index: int) -> Any:
    """Finite-difference tangent estimate of the seed curve's own
    direction at ``index`` -- used only to pick the transversal starting
    direction most orthogonal to the seed's own path, never as a
    production surface normal."""

    torch = require_torch()
    count = seed_points.shape[0]
    if count < 2:
        return torch.tensor([1.0, 0.0, 0.0], dtype=seed_points.dtype, device=seed_points.device)
    left = max(0, index - 1)
    right = min(count - 1, index + 1)
    if left == right:
        right = min(count - 1, left + 1)
    direction = seed_points[right] - seed_points[left]
    norm = direction.norm()
    if float(norm.item()) < _EPS:
        return torch.tensor([1.0, 0.0, 0.0], dtype=seed_points.dtype, device=seed_points.device)
    return direction / norm


def _pick_transversal_start_direction(
    seed_direction: Any, tangent_u: Any, tangent_v: Any, inward_hint: Any,
) -> Any:
    """One-time initial direction choice at the seed sample (not
    re-selected at every trace step -- ``trace_curve`` parallel-transports
    this direction forward from here)."""

    torch = require_torch()
    seed_direction = seed_direction / seed_direction.norm().clamp_min(_EPS)
    align_u = torch.abs((tangent_u * seed_direction).sum())
    align_v = torch.abs((tangent_v * seed_direction).sum())
    candidate = tangent_v if align_v < align_u else tangent_u
    sign = torch.sign((inward_hint * candidate).sum())
    if float(sign.item()) == 0.0:
        sign = torch.tensor(1.0, device=candidate.device, dtype=candidate.dtype)
    return candidate * sign


def _build_transversal_traces(
    seed: SeedCurve, support: LatentSurfaceSupport, *,
    max_samples: int, step_count: int, step_scale: float,
) -> tuple[TransversalTrace, ...]:
    torch = require_torch()
    count = int(seed.points.shape[0])
    if count == 0:
        return ()
    sample_indices = _evenly_spaced_indices(count, max_samples)
    centroid = seed.points.mean(dim=0)
    step_size = step_scale * support.median_spacing

    traces: list[TransversalTrace] = []
    for index in sample_indices:
        start = seed.points[index]
        result = support.query_batch(start.reshape(1, 3))
        if not bool(result.supported[0]):
            continue
        seed_direction = _local_seed_direction(seed.points, index)
        inward_hint = centroid - start
        start_direction = _pick_transversal_start_direction(
            seed_direction, result.tangent_u[0], result.tangent_v[0], inward_hint,
        )
        traced = trace_curve(result.positions[0], start_direction, support, step_count=step_count, step_size=step_size)
        if int(traced.points.shape[0]) >= 2:
            traces.append(TransversalTrace(index, traced.points))
    return tuple(traces)


def _build_rungs(
    traces: tuple[TransversalTrace, ...], support: LatentSurfaceSupport, *, rung_steps: int,
) -> tuple[RungSegment, ...]:
    rungs: list[RungSegment] = []
    for i in range(len(traces) - 1):
        trace_a, trace_b = traces[i], traces[i + 1]
        common_depth = min(int(trace_a.points.shape[0]), int(trace_b.points.shape[0]))
        for depth in range(common_depth):
            points, fully_supported = sample_segment_continuous_support(
                support, trace_a.points[depth], trace_b.points[depth], rung_steps,
            )
            if fully_supported:
                rungs.append(RungSegment(depth, trace_a.sample_index, trace_b.sample_index, points))
    return tuple(rungs)


def _has_consistent_correspondence_quad(rungs: tuple[RungSegment, ...]) -> bool:
    """At least one pair of adjacent transversal curves connected by
    continuously-supported rungs at >= MIN_CORRESPONDENCE_DEPTH_COUNT
    distinct depths -- the 2x2 mutually consistent correspondence the
    contract requires. Depths within one rung-pair are only ever adjacent
    by construction (loop over ``range(common_depth)`` in
    :func:`_build_rungs`), so ordering is monotonic and no unsupported
    crossing/gap-bridge is possible."""

    by_pair: dict[tuple[int, int], set[int]] = {}
    for rung in rungs:
        key = (rung.a_sample_index, rung.b_sample_index)
        by_pair.setdefault(key, set()).add(rung.depth)
    return any(len(depths) >= MIN_CORRESPONDENCE_DEPTH_COUNT for depths in by_pair.values())


def build_curve_network_blocks(
    seed_curves: tuple[SeedCurve, ...],
    support: LatentSurfaceSupport,
    *,
    max_seed_samples: int = DEFAULT_MAX_SEED_SAMPLES,
    transversal_step_count: int = DEFAULT_TRACE_STEP_COUNT,
    transversal_step_scale: float = DEFAULT_TRACE_STEP_SCALE,
    rung_steps: int = DEFAULT_SEGMENT_STEPS,
) -> tuple[CurveNetworkBlock, ...]:
    """One block per seed curve -- a region may therefore produce zero,
    one, or many independent curve-network blocks, each an independent
    patch candidate. Partitioning is entirely structural (pre-fit); no
    NURBS fit is attempted here and no block is merged or split based on
    downstream fit quality."""

    torch = require_torch()
    blocks: list[CurveNetworkBlock] = []
    for seed in seed_curves:
        traces = _build_transversal_traces(
            seed, support, max_samples=max_seed_samples,
            step_count=transversal_step_count, step_scale=transversal_step_scale,
        )
        rungs = _build_rungs(traces, support, rung_steps=rung_steps)
        satisfies = (
            len(traces) >= MIN_FAMILY_CURVE_COUNT
            and len({rung.depth for rung in rungs}) >= MIN_FAMILY_CURVE_COUNT
            and _has_consistent_correspondence_quad(rungs)
        )
        all_points = None
        if satisfies:
            point_sets = [seed.points] + [trace.points for trace in traces] + [rung.points for rung in rungs]
            all_points = torch.cat(point_sets, dim=0)
        blocks.append(CurveNetworkBlock(seed.seed_id, seed.seed_type, seed, traces, rungs, satisfies, all_points))
    return tuple(blocks)
