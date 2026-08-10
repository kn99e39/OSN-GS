from __future__ import annotations

"""Worklog 83: chart-scale topology/assembly over Worklog 82 micro-components.

Worklog 82 built evidence-scale surface-consistency MICRO-components inside a
region (bounded-degree kNN + normal/residual same_surface criterion + typed
crease veto). Real result: 364 micro-components across 7 regions, 16 reaching
`valid_supported` -- proof that coherent surface-consistent evidence exists --
but median size 3-6 points against 92-1035 owned evidence, 91% ending
`no_chart`. Those micro-components are too fragmented to be final chart units
on their own.

This module does NOT touch worklog 82's point-level criterion or thresholds.
It treats worklog 82's own components (excluding any it already flagged
`non_manifold_suspected`) as CONSERVATIVE ATOMIC SUPPORT, and asks a coarser
question: which pairs of micro-components are jointly, redundantly supported
as parts of the SAME surface sheet, using AGGREGATE evidence rather than any
single pointwise edge.

A component-pair is only merged when it is a scale-gated CANDIDATE (nearest
points within a proximity multiple of the pair's own local spacing -- the
same 2.5x multiplier `torch_region_owned_dense_boundary_support._connect`
already uses, reused not reinvented) AND passes typed-crease absence AND at
least two of three independent aggregate signals, each itself built from
already-existing production primitives:

  1. aggregate normal/tangent compatibility -- mean normal alignment over the
     candidate cross-component point pairs (same alignment computation
     worklog 82 already uses per-point, aggregated here, same 0.85 bound).
  2. repeated same_surface correspondence -- how many INDIVIDUAL cross-
     component point pairs already satisfy worklog 82's own same_surface
     criterion (normal alignment + mutual tangent residual, same thresholds,
     same per-point residual scale) despite having been split apart purely by
     the point-level kNN degree cap. A single passing pair is coincidence; a
     handful of independently-computed passing pairs is redundant support.
  3. observed support occupancy between the components -- reuses
     `measure_edge_support_occupancy` (Worklog 76, disclosure-only there,
     used here as an ACCEPTANCE signal for the first time: a synthetic edge
     between the two components' nearest points must not span an empty
     interior bin, i.e. must not bridge unsupported evidence).

A candidate pair whose nearest points fall on opposite sides of an already-
typed Worklog 80 crease/frontier arc is VETOED unconditionally, regardless of
how many aggregate signals pass -- exactly Worklog 82's own veto rule, reused
at component scale.

Assembled CHART UNITS are the connected components of the ACCEPTED
component-pair graph. A candidate pair that fails the joint-signal vote is
disclosed as AMBIGUOUS, never silently merged and never silently dropped. A
micro-component with no accepted pair keeps standing alone as its own
single-micro-component chart unit -- this is not a failure state, it is
exactly what Worklog 82 already validated as internally coherent.

Nothing here ever merges components because a combined NURBS fit would look
better -- fit quality is not a signal this module reads at all.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_boundary_support_spacing import measure_edge_support_occupancy
from osn_gs.surface.torch_dense_surface_consistency_components import (
    DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    _nearest_arc_side,
    _residual_scale,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

RELATION_ACCEPTED = "chart_unit_assembly_accepted"
RELATION_CREASE_VETOED = "chart_unit_assembly_crease_vetoed"
RELATION_AMBIGUOUS = "chart_unit_assembly_ambiguous"
RELATION_NOT_CANDIDATE = "chart_unit_assembly_not_candidate"

# Same 2.5x multiplier `torch_region_owned_dense_boundary_support._connect`
# already uses for its own distance gate -- reused, not a new constant.
PROXIMITY_SCALE_MULTIPLIER = 2.5
DEFAULT_MIN_SAME_SURFACE_CORRESPONDENCE_COUNT = 3
DEFAULT_MIN_REQUIRED_SIGNALS = 2


@dataclass(frozen=True)
class ComponentPairEdge:
    component_a: int
    component_b: int
    relation: str
    normal_signal: bool
    correspondence_signal: bool
    occupancy_signal: bool
    same_surface_correspondence_count: int
    mean_normal_alignment: float | None
    unsupported_edge_fraction: float | None


@dataclass(frozen=True)
class ChartUnit:
    micro_component_indices: tuple[int, ...]  # indices into the input micro-component list
    member_indices: tuple[int, ...]  # flattened evidence-point indices (union of members)


@dataclass(frozen=True)
class ChartUnitAssemblyResult:
    region_id: int
    micro_component_count: int
    chart_unit_count: int
    chart_units: tuple[ChartUnit, ...]
    edges: tuple[ComponentPairEdge, ...]
    excluded_non_manifold_component_count: int


def _component_local_scale(positions: Any, member: Sequence[int]) -> float:
    torch = require_torch()
    if len(member) < 2:
        return 0.0
    sub = positions[torch.tensor(member, dtype=torch.long, device=positions.device)]
    d = torch.cdist(sub, sub)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.median())


def build_chart_unit_assembly(
    region_id: int,
    positions: Any,
    *,
    covariance: Any,
    micro_components: Sequence[tuple[int, ...]],
    non_manifold_flags: Sequence[bool],
    full_evidence_spacing: float,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
    proximity_scale_multiplier: float = PROXIMITY_SCALE_MULTIPLIER,
    same_surface_min_normal_alignment: float = DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    same_surface_max_mutual_residual: float = DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    min_same_surface_correspondence_count: int = DEFAULT_MIN_SAME_SURFACE_CORRESPONDENCE_COUNT,
    min_required_signals: int = DEFAULT_MIN_REQUIRED_SIGNALS,
) -> ChartUnitAssemblyResult:
    """Assemble Worklog 82 micro-components into chart-scale units. Components
    flagged ``non_manifold_flags[i]`` are excluded from assembly entirely
    (carried forward as already-excluded, never a merge partner)."""

    torch = require_torch()
    eligible = [i for i, flag in enumerate(non_manifold_flags) if not flag]
    excluded_count = len(micro_components) - len(eligible)

    if len(eligible) == 0:
        return ChartUnitAssemblyResult(region_id, len(micro_components), 0, (), (), excluded_count)

    frame = extract_covariance_frame(covariance)
    normals = frame.normal_candidate
    residual_scale = _residual_scale(positions, k=8)

    scales = {i: _component_local_scale(positions, micro_components[i]) or full_evidence_spacing for i in eligible}

    arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        arc_side = _nearest_arc_side(positions, arc_starts, arc_ends, arc_kinds)

    edges: list[ComponentPairEdge] = []
    accepted_adjacency: dict[int, set[int]] = {i: set() for i in eligible}

    for idx_a in range(len(eligible)):
        for idx_b in range(idx_a + 1, len(eligible)):
            a, b = eligible[idx_a], eligible[idx_b]
            member_a, member_b = micro_components[a], micro_components[b]
            pos_a = positions[torch.tensor(member_a, dtype=torch.long, device=positions.device)]
            pos_b = positions[torch.tensor(member_b, dtype=torch.long, device=positions.device)]
            cross = torch.cdist(pos_a, pos_b)
            min_dist = float(cross.min())
            local_scale = (scales[a] + scales[b]) / 2.0
            gate = proximity_scale_multiplier * max(local_scale, _EPS)
            if min_dist > gate:
                continue  # not even a candidate -- no edge recorded, mirrors worklog 82's kNN candidate gate

            nearest_local = int(cross.argmin())
            nearest_a_local, nearest_b_local = divmod(nearest_local, int(pos_b.shape[0]))
            nearest_a, nearest_b = member_a[nearest_a_local], member_b[nearest_b_local]

            # Typed-provenance veto: unconditional, checked before any signal.
            if arc_side is not None and arc_side[nearest_a] and arc_side[nearest_b] and arc_side[nearest_a] != arc_side[nearest_b]:
                edges.append(ComponentPairEdge(a, b, RELATION_CREASE_VETOED, False, False, False, 0, None, None))
                continue

            # Candidate cross-component pairs within the same proximity gate,
            # for signals 1 and 2 (aggregate normal alignment, same_surface count).
            within = torch.nonzero(cross <= gate, as_tuple=False)
            if int(within.shape[0]) == 0:
                within = torch.tensor([[nearest_a_local, nearest_b_local]])
            local_a_idx = within[:, 0]
            local_b_idx = within[:, 1]
            global_a = torch.tensor([member_a[int(i)] for i in local_a_idx], dtype=torch.long, device=positions.device)
            global_b = torch.tensor([member_b[int(i)] for i in local_b_idx], dtype=torch.long, device=positions.device)

            normal_a, normal_b = normals[global_a], normals[global_b]
            alignments = (normal_a * normal_b).sum(dim=1).abs()
            mean_alignment = float(alignments.mean())
            normal_signal = mean_alignment >= same_surface_min_normal_alignment

            displacement = positions[global_b] - positions[global_a]
            residual_a = (displacement * normal_a).sum(dim=1).abs() / residual_scale[global_a]
            residual_b = (-displacement * normal_b).sum(dim=1).abs() / residual_scale[global_b]
            mutual_residual = torch.maximum(residual_a, residual_b)
            same_surface_mask = (alignments >= same_surface_min_normal_alignment) & (mutual_residual <= same_surface_max_mutual_residual)
            correspondence_count = int(same_surface_mask.sum())
            correspondence_signal = correspondence_count >= min_same_surface_correspondence_count

            occupancy = measure_edge_support_occupancy(
                [(0, 1)], positions[[nearest_a, nearest_b]], positions,
                full_evidence_spacing=full_evidence_spacing,
            )
            unsupported_fraction = occupancy["unsupported_edge_fraction"]
            occupancy_signal = unsupported_fraction == 0.0

            signal_count = int(normal_signal) + int(correspondence_signal) + int(occupancy_signal)
            if signal_count >= min_required_signals:
                relation = RELATION_ACCEPTED
                accepted_adjacency[a].add(b)
                accepted_adjacency[b].add(a)
            else:
                relation = RELATION_AMBIGUOUS

            edges.append(
                ComponentPairEdge(
                    a, b, relation, normal_signal, correspondence_signal, occupancy_signal,
                    correspondence_count, mean_alignment, unsupported_fraction,
                )
            )

    # Connected components of the ACCEPTED graph = chart units. A
    # micro-component with no accepted partner is its own chart unit.
    visited: set[int] = set()
    chart_units: list[ChartUnit] = []
    for start in eligible:
        if start in visited:
            continue
        stack, group = [start], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            stack.extend(accepted_adjacency[node] - visited)
        merged_members: list[int] = []
        for comp_idx in group:
            merged_members.extend(micro_components[comp_idx])
        chart_units.append(ChartUnit(tuple(sorted(group)), tuple(merged_members)))

    return ChartUnitAssemblyResult(
        region_id, len(micro_components), len(chart_units), tuple(chart_units), tuple(edges), excluded_count,
    )
