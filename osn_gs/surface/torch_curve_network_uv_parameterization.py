from __future__ import annotations

"""Worklog 97 -- curve-network-native ``(u, v)`` derivation for a Worklog 96
:class:`~osn_gs.surface.torch_latent_surface_curve_families.CurveNetworkBlock`.

Never calls PCA point-cloud parameterization. Instead:

- ``u`` per family-V curve (``TransversalTrace``) is the cumulative
  chord-length position of that curve's own seed sample along the SEED
  curve, normalized to ``[0, 1]``. Fixed per curve -- every point on one
  transversal trace shares the same ``u`` (this is exactly ``C_i(v) ~=
  S(u_i, v)``).
- ``v`` per depth level is the cumulative chord-length position along each
  transversal trace at that depth, normalized by that trace's own total
  chord length, RECONCILED across every trace reaching that depth (their
  mean) into one shared value used by every point at that depth -- this is
  what keeps the U-family and V-family referring to the same parametric
  domain at their correspondence (rung) locations. Reconciliation only
  uses the depth range every retained trace can reach (the block's shared
  correspondence depth), so the reconciled sequence stays monotonic by
  construction (the same fixed trace set contributes at every depth).
- Rung (family-U, ``D_j(u) ~= S(u, v_j)``) interior points get ``u``
  linearly interpolated between their two endpoint traces' ``u`` values by
  chord-length position along the rung itself, and the depth's shared
  ``v_j``.

Every degenerate/non-monotonic/contradictory case fails closed -- never
repaired by falling back to PCA-UV, never silently reordered.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_families import (
    MIN_CORRESPONDENCE_DEPTH_COUNT,
    MIN_FAMILY_CURVE_COUNT,
    CurveNetworkBlock,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


@dataclass(frozen=True)
class CurveNetworkUV:
    valid: bool
    invalid_reason: str | None
    points: Any | None = None  # (N, 3)
    uv: Any | None = None  # (N, 2)
    provenance: tuple[str, ...] = ()  # per-point tag, e.g. "trace:3" / "rung:depth2"


def _cumulative_chord_length(points: Any) -> Any:
    torch = require_torch()
    if points.shape[0] < 2:
        return torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    deltas = (points[1:] - points[:-1]).norm(dim=1)
    cumulative = torch.cat([torch.zeros(1, dtype=points.dtype, device=points.device), torch.cumsum(deltas, dim=0)])
    return cumulative


def _strictly_increasing(values: list[float], tol: float = _EPS) -> bool:
    return all(values[index + 1] - values[index] > tol for index in range(len(values) - 1))


def build_curve_network_uv(block: CurveNetworkBlock) -> CurveNetworkUV:
    """Derive network-native ``(u, v)`` for every retained point in
    ``block``. Fails closed (returns ``valid=False``) rather than
    repairing any inconsistency -- callers must not fall back to PCA-UV on
    an invalid result."""

    torch = require_torch()
    if not block.satisfies_contract:
        return CurveNetworkUV(False, "block_does_not_satisfy_contract")

    traces = block.transversal_traces
    if len(traces) < MIN_FAMILY_CURVE_COUNT:
        return CurveNetworkUV(False, "insufficient_family_v_curves")

    seed_points = block.seed_curve.points
    seed_chord = _cumulative_chord_length(seed_points)
    seed_total = float(seed_chord[-1].item())
    if seed_total <= _EPS:
        return CurveNetworkUV(False, "degenerate_seed_chord_length")

    max_seed_index = int(seed_points.shape[0]) - 1
    u_values: list[float] = []
    for trace in traces:
        index = min(trace.sample_index, max_seed_index)
        u_values.append(float(seed_chord[index].item()) / seed_total)
    if not _strictly_increasing(u_values):
        return CurveNetworkUV(False, "nonmonotonic_or_duplicate_u_family_ordering")
    if u_values[-1] - u_values[0] <= _EPS:
        return CurveNetworkUV(False, "degenerate_u_parameter_extent")

    per_trace_chord = [_cumulative_chord_length(trace.points) for trace in traces]
    per_trace_total = [float(chord[-1].item()) for chord in per_trace_chord]
    if any(total <= _EPS for total in per_trace_total):
        return CurveNetworkUV(False, "degenerate_v_parameter_extent")

    max_shared_depth = min(int(trace.points.shape[0]) for trace in traces)
    if max_shared_depth < MIN_CORRESPONDENCE_DEPTH_COUNT:
        return CurveNetworkUV(False, "insufficient_shared_correspondence_depth")

    # Direction-consistency check: two ADJACENT transversal traces (the
    # only pairs Worklog 96 ever connects with a rung) must walk in
    # broadly the same transverse direction. Worklog 96's own
    # inward-hint sign selection is a per-seed heuristic and is not
    # guaranteed consistent trace-to-trace; if adjacent traces walk in
    # opposing directions, the "shared" depth/v they are reconciled
    # against does not represent the same physical direction and the
    # correspondence is contradictory -- fail closed rather than silently
    # averaging opposed displacements into one v value.
    for index in range(len(traces) - 1):
        displacement_a = traces[index].points[max_shared_depth - 1] - traces[index].points[0]
        displacement_b = traces[index + 1].points[max_shared_depth - 1] - traces[index + 1].points[0]
        norm_a = float(displacement_a.norm().item())
        norm_b = float(displacement_b.norm().item())
        if norm_a <= _EPS or norm_b <= _EPS:
            continue
        cosine = float((displacement_a * displacement_b).sum().item()) / (norm_a * norm_b)
        if cosine <= 0.0:
            return CurveNetworkUV(False, "inconsistent_transversal_curve_direction")

    v_per_depth: list[float] = []
    for depth in range(max_shared_depth):
        local_values = [
            float(per_trace_chord[i][depth].item()) / per_trace_total[i] for i in range(len(traces))
        ]
        v_per_depth.append(sum(local_values) / len(local_values))
    if not all(v_per_depth[index + 1] - v_per_depth[index] >= -_EPS for index in range(len(v_per_depth) - 1)):
        return CurveNetworkUV(False, "nonmonotonic_v_family_ordering")
    if v_per_depth[-1] - v_per_depth[0] <= _EPS:
        return CurveNetworkUV(False, "degenerate_v_parameter_extent")

    trace_index_by_sample = {trace.sample_index: index for index, trace in enumerate(traces)}

    points_list: list[Any] = []
    uv_list: list[tuple[float, float]] = []
    provenance: list[str] = []
    for trace_index, trace in enumerate(traces):
        for depth in range(max_shared_depth):
            points_list.append(trace.points[depth].reshape(1, 3))
            uv_list.append((u_values[trace_index], v_per_depth[depth]))
            provenance.append(f"trace_family:{trace.sample_index}")

    for rung in block.rungs:
        if rung.depth >= max_shared_depth:
            continue
        if rung.a_sample_index not in trace_index_by_sample or rung.b_sample_index not in trace_index_by_sample:
            continue
        u_a = u_values[trace_index_by_sample[rung.a_sample_index]]
        u_b = u_values[trace_index_by_sample[rung.b_sample_index]]
        v_j = v_per_depth[rung.depth]
        rung_chord = _cumulative_chord_length(rung.points)
        rung_total = float(rung_chord[-1].item())
        if rung_total <= _EPS:
            continue
        for sample_index in range(int(rung.points.shape[0])):
            t = float(rung_chord[sample_index].item()) / rung_total
            u_k = (1.0 - t) * u_a + t * u_b
            points_list.append(rung.points[sample_index].reshape(1, 3))
            uv_list.append((u_k, v_j))
            provenance.append(f"rung_family:depth{rung.depth}")

    if len(points_list) < 4:
        return CurveNetworkUV(False, "insufficient_reconciled_points")

    points_tensor = torch.cat(points_list, dim=0)
    uv_tensor = torch.tensor(uv_list, dtype=points_tensor.dtype, device=points_tensor.device)

    # Duplicated-parameter-location check: two points that round to the
    # same (u, v) cell but disagree geometrically indicate a contradictory
    # correspondence -- fail closed rather than letting the LSQ solve
    # silently average them away.
    spacing_scale = max(seed_total / max(len(seed_points) - 1, 1), _EPS)
    rounded = torch.round(uv_tensor * 1000).to(torch.long)
    seen: dict[tuple[int, int], Any] = {}
    for index in range(int(rounded.shape[0])):
        key = (int(rounded[index, 0].item()), int(rounded[index, 1].item()))
        candidate = points_tensor[index]
        if key in seen:
            if float((seen[key] - candidate).norm().item()) > 3.0 * spacing_scale:
                return CurveNetworkUV(False, "duplicated_parameter_location_incompatible_geometry")
        else:
            seen[key] = candidate

    return CurveNetworkUV(True, None, points_tensor, uv_tensor, tuple(provenance))
