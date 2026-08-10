from __future__ import annotations

"""Chart-unit boundaries from full-region observed-face incidence.

This is the corrected Worklog 88 construction.  The Worklog 82
``same_surface`` graph is embedded once for the full region in vertex-local
covariance tangent frames.  Observed faces are recovered before chart-unit
membership is consulted.  A chart boundary is then exactly the set of
single-unit-face half-edges.  No PCA coordinates, sparse representative
geometry, boundary candidate, or stable-ID chain participates in topology.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_boundary_support_spacing import measure_edge_support_occupancy
from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import (
    NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND,
    STATE_AMBIGUOUS_OR_OVER_MERGED,
    STATE_COVERAGE_FAILED,
    STATE_MATERIALIZED,
    STATE_SELF_INTERSECTING,
    STATE_UNSUPPORTED_CLOSURE,
    ChartUnitCoherence,
    assess_chart_unit_coherence,
)
from osn_gs.surface.torch_dense_surface_consistency_components import (
    DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
    DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
    DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    _nearest_arc_side,
    build_same_surface_adjacency,
)
from osn_gs.surface.torch_full_region_surface_face_topology import (
    FullRegionSurfaceFaceTopology,
    build_full_region_surface_face_topology,
    edge_key,
    loop_orientation_score,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_region_owned_dense_boundary_support import (
    estimate_full_evidence_sampling_scale,
    extract_dense_boundary_support,
)
from osn_gs.surface.torch_region_owned_full_evidence import (
    MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
    evidence_outside_chart_domain_fraction,
)
from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import evaluate_closed_loop_geometry
from osn_gs.utils.torch_ops import require_torch

PHYSICAL_ONLY = "physical_only"
MIXED_PHYSICAL_PARTITION_SEAM = "mixed_physical_partition_seam"
SEAM_ONLY = "seam_only"

SEGMENT_PHYSICAL_TERMINATION = "physical_termination"
SEGMENT_CREASE = "crease"
SEGMENT_OBSERVATION_FRONTIER = "observation_frontier"
SEGMENT_PARTITION_SEAM = "partition_seam"

ROLE_OUTER_BOUNDARY = "outer_boundary"
ROLE_INNER_BOUNDARY = "interior_boundary"

STATE_INSUFFICIENT_TOPOLOGY = "chart_unit_cut_insufficient_topology"
STATE_TOPOLOGY_OPEN_OR_BRANCHING = "chart_unit_cut_topology_open_or_branching"
STATE_NON_MANIFOLD = "chart_unit_cut_non_manifold"
STATE_UNTYPED_PHYSICAL_BOUNDARY = "chart_unit_cut_untyped_physical_boundary"
STATE_INNER_BOUNDARY_REVIEW_REQUIRED = "chart_unit_cut_inner_boundary_review_required"
STATE_NO_CLOSED_CUT_BOUNDARY = "chart_unit_cut_no_closed_boundary"


@dataclass(frozen=True)
class ChartUnitFaceTopologyContext:
    positions: Any
    covariance: Any
    stable_ids: tuple[Any, ...]
    normals: Any
    arc_side: tuple[str, ...] | None
    same_surface_adjacency: tuple[frozenset[int], ...]
    surface_faces: FullRegionSurfaceFaceTopology
    full_region_physical_candidate_ids: frozenset[Any]
    full_evidence_spacing: float


@dataclass(frozen=True)
class ChartUnitCutBoundarySegment:
    stable_id_a: Any
    stable_id_b: Any
    segment_kind: str
    crease_inconsistent: bool


@dataclass(frozen=True)
class RecoveredBoundaryLoop:
    role: str
    ordered_region_indices: tuple[int, ...]
    segments: tuple[ChartUnitCutBoundarySegment, ...]
    orientation_score: float


@dataclass(frozen=True)
class ChartUnitCutDomainResult:
    state: str
    boundary_role: str
    ordered_region_indices: tuple[int, ...]
    ordered_stable_ids: tuple[Any, ...]
    ordered_positions: Any | None
    segments: tuple[ChartUnitCutBoundarySegment, ...]
    evidence_outside_domain_fraction: float | None
    boundary_composition: str
    physical_segment_count: int
    partition_seam_segment_count: int
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


@dataclass(frozen=True)
class ChartUnitCutBoundaryResult:
    coherence: ChartUnitCoherence | None
    domains: tuple[ChartUnitCutDomainResult, ...]
    boundary_loops: tuple[RecoveredBoundaryLoop, ...]
    admitted_boundary_candidate_count: int
    induced_same_surface_edge_count: int
    membership_crossing_edge_count: int
    topology_component_count: int
    unit_supported_face_count: int
    unresolved_reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return any(domain.materialized for domain in self.domains)


def build_chart_unit_face_topology_context(
    region_positions: Any,
    region_covariance: Any,
    region_stable_ids: Sequence[Any],
    *,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
) -> ChartUnitFaceTopologyContext:
    """Build unchanged same-surface adjacency and full-region faces once."""

    normals = extract_covariance_frame(region_covariance).normal_candidate
    arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        arc_side = tuple(_nearest_arc_side(region_positions, arc_starts, arc_ends, arc_kinds))
    _edges, adjacency, _crease_vetoed = build_same_surface_adjacency(
        region_positions,
        normals,
        arc_side=arc_side,
        candidate_neighbor_count=DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
        max_candidate_count_per_node=DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
        same_surface_min_normal_alignment=DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
        same_surface_max_mutual_residual=DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    )
    frozen_adjacency = tuple(frozenset(row) for row in adjacency)
    surface_faces = build_full_region_surface_face_topology(
        region_positions, region_covariance, region_stable_ids, frozen_adjacency,
    )
    support = extract_dense_boundary_support(region_positions, normals, region_stable_ids)
    return ChartUnitFaceTopologyContext(
        positions=region_positions,
        covariance=region_covariance,
        stable_ids=tuple(region_stable_ids),
        normals=normals,
        arc_side=arc_side,
        same_surface_adjacency=frozen_adjacency,
        surface_faces=surface_faces,
        full_region_physical_candidate_ids=frozenset(candidate.stable_id for candidate in support.candidates),
        full_evidence_spacing=estimate_full_evidence_sampling_scale(region_positions),
    )


def _normalized_arc_kind(kind: str) -> str | None:
    lowered = kind.lower()
    if not lowered or "partition_seam" in lowered:
        return None
    if "physical" in lowered or "termination" in lowered or "observed_support" in lowered:
        return SEGMENT_PHYSICAL_TERMINATION
    if "crease" in lowered:
        return SEGMENT_CREASE
    return SEGMENT_OBSERVATION_FRONTIER


def _provenance_kind(
    a: int,
    b: int,
    context: ChartUnitFaceTopologyContext,
) -> tuple[str | None, bool]:
    sid_a, sid_b = context.stable_ids[a], context.stable_ids[b]
    if sid_a in context.full_region_physical_candidate_ids or sid_b in context.full_region_physical_candidate_ids:
        return SEGMENT_PHYSICAL_TERMINATION, False
    if context.arc_side is None:
        return None, False
    raw_a, raw_b = context.arc_side[a], context.arc_side[b]
    kind_a, kind_b = _normalized_arc_kind(raw_a), _normalized_arc_kind(raw_b)
    inconsistent = bool(
        raw_a and raw_b and raw_a != raw_b
        and (kind_a == SEGMENT_CREASE or kind_b == SEGMENT_CREASE)
    )
    for kind in (SEGMENT_PHYSICAL_TERMINATION, SEGMENT_CREASE, SEGMENT_OBSERVATION_FRONTIER):
        if kind in (kind_a, kind_b):
            return kind, inconsistent
    return None, inconsistent


def _make_segment(
    a: int,
    b: int,
    kind: str,
    inconsistent: bool,
    context: ChartUnitFaceTopologyContext,
) -> ChartUnitCutBoundarySegment:
    return ChartUnitCutBoundarySegment(
        context.stable_ids[a], context.stable_ids[b], kind, inconsistent,
    )


def _trace_all_boundary_loops(
    halfedges: dict[tuple[int, int], ChartUnitCutBoundarySegment],
    context: ChartUnitFaceTopologyContext,
) -> tuple[tuple[RecoveredBoundaryLoop, ...], tuple[str, ...]]:
    if not halfedges:
        return (), (STATE_NO_CLOSED_CUT_BOUNDARY,)
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for a, b in halfedges:
        outgoing.setdefault(a, []).append(b)
        incoming.setdefault(b, []).append(a)
    vertices = set(outgoing) | set(incoming)
    bad = sorted(
        node for node in vertices
        if len(outgoing.get(node, ())) != 1 or len(incoming.get(node, ())) != 1
    )
    if bad:
        return (), (
            f"{STATE_TOPOLOGY_OPEN_OR_BRANCHING}:vertex_count={len(bad)}",
        )

    unvisited = set(halfedges)
    loops: list[RecoveredBoundaryLoop] = []
    while unvisited:
        start = min(
            unvisited,
            key=lambda edge: (str(context.stable_ids[edge[0]]), str(context.stable_ids[edge[1]])),
        )
        current = start
        ordered: list[int] = []
        segments: list[ChartUnitCutBoundarySegment] = []
        seen: set[tuple[int, int]] = set()
        for _ in range(len(halfedges) + 1):
            if current in seen:
                if current != start:
                    return (), (STATE_TOPOLOGY_OPEN_OR_BRANCHING,)
                break
            seen.add(current)
            unvisited.discard(current)
            a, b = current
            ordered.append(a)
            segments.append(halfedges[current])
            current = (b, outgoing[b][0])
        else:
            return (), (STATE_TOPOLOGY_OPEN_OR_BRANCHING,)
        if current != start or len(ordered) < 3 or len(set(ordered)) != len(ordered):
            return (), (STATE_TOPOLOGY_OPEN_OR_BRANCHING,)
        score = loop_orientation_score(
            ordered, context.positions, context.surface_faces.oriented_normals,
        )
        if abs(score) <= 1e-12:
            return (), (STATE_TOPOLOGY_OPEN_OR_BRANCHING, "zero_loop_orientation")
        role = ROLE_OUTER_BOUNDARY if score > 0.0 else ROLE_INNER_BOUNDARY
        loops.append(RecoveredBoundaryLoop(role, tuple(ordered), tuple(segments), score))
    loops.sort(
        key=lambda loop: (
            0 if loop.role == ROLE_OUTER_BOUNDARY else 1,
            min(str(context.stable_ids[index]) for index in loop.ordered_region_indices),
        )
    )
    return tuple(loops), ()


def _validate_outer_loop(
    loop: RecoveredBoundaryLoop,
    context: ChartUnitFaceTopologyContext,
    unit_positions: Any,
    max_evidence_outside_domain_fraction: float,
) -> ChartUnitCutDomainResult:
    torch = require_torch()
    indices = loop.ordered_region_indices

    def fail(state: str, *reasons: str) -> ChartUnitCutDomainResult:
        return ChartUnitCutDomainResult(
            state, loop.role, indices, (), None, loop.segments, None, "", 0, 0, tuple(reasons),
        )

    selector = torch.tensor(indices, dtype=torch.long, device=context.positions.device)
    ordered_positions = context.positions[selector]
    ordered_ids = tuple(context.stable_ids[index] for index in indices)
    geometry = evaluate_closed_loop_geometry(
        [tuple(float(value) for value in row) for row in ordered_positions.detach().cpu().tolist()]
    )
    if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
        return fail(STATE_SELF_INTERSECTING, f"proper_crossing_count={geometry.proper_crossing_count}")

    edge_pairs = [(index, (index + 1) % len(indices)) for index in range(len(indices))]
    occupancy = measure_edge_support_occupancy(
        edge_pairs,
        ordered_positions,
        context.positions,
        full_evidence_spacing=max(context.full_evidence_spacing, 1e-9),
    )
    if occupancy["edges_with_empty_interior_bin"] > 0:
        return fail(
            STATE_UNSUPPORTED_CLOSURE,
            f"edges_with_empty_interior_bin={occupancy['edges_with_empty_interior_bin']}/{occupancy['edge_count']}",
        )

    partition_count = sum(segment.segment_kind == SEGMENT_PARTITION_SEAM for segment in loop.segments)
    physical_count = len(loop.segments) - partition_count
    if partition_count == 0:
        composition = PHYSICAL_ONLY
    elif physical_count == 0:
        composition = SEAM_ONLY
    else:
        composition = MIXED_PHYSICAL_PARTITION_SEAM
    outside = evidence_outside_chart_domain_fraction(ordered_positions, unit_positions)
    if outside is not None and outside > max_evidence_outside_domain_fraction:
        return ChartUnitCutDomainResult(
            STATE_COVERAGE_FAILED, loop.role, indices, ordered_ids, ordered_positions,
            loop.segments, outside, composition, physical_count, partition_count,
            (f"evidence_outside_chart_domain_fraction={outside:.4f}>{max_evidence_outside_domain_fraction}",),
        )
    return ChartUnitCutDomainResult(
        STATE_MATERIALIZED, loop.role, indices, ordered_ids, ordered_positions,
        loop.segments, outside, composition, physical_count, partition_count, (),
    )


def materialize_chart_unit_face_incidence_boundaries(
    context: ChartUnitFaceTopologyContext,
    member_region_indices: Sequence[int],
    *,
    max_evidence_outside_domain_fraction: float = MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
) -> ChartUnitCutBoundaryResult:
    """Apply chart membership to full-region face incidence and fail closed."""

    torch = require_torch()
    members = tuple(dict.fromkeys(int(index) for index in member_region_indices))
    member_set = set(members)
    if any(index < 0 or index >= len(context.stable_ids) for index in members):
        raise IndexError("chart-unit member index outside topology context")
    if not members:
        return ChartUnitCutBoundaryResult(None, (), (), 0, 0, 0, 0, 0, (STATE_INSUFFICIENT_TOPOLOGY,))

    selector = torch.tensor(members, dtype=torch.long, device=context.positions.device)
    unit_positions = context.positions[selector]
    coherence = assess_chart_unit_coherence(context.covariance[selector], list(range(len(members))))
    candidate_count = sum(
        context.stable_ids[index] in context.full_region_physical_candidate_ids for index in members
    )
    induced_edge_count = sum(
        b in member_set for a in members for b in context.same_surface_adjacency[a]
    ) // 2
    if not coherence.coherent:
        return ChartUnitCutBoundaryResult(
            coherence, (), (), candidate_count, induced_edge_count, 0, 0, 0,
            (
                f"{STATE_AMBIGUOUS_OR_OVER_MERGED}:internal_normal_disagreement_fraction="
                f"{coherence.internal_normal_disagreement_fraction:.4f}>"
                f"{NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND}",
            ),
        )
    invalid_members = member_set & set(context.surface_faces.invalid_topology_nodes)
    if invalid_members:
        return ChartUnitCutBoundaryResult(
            coherence, (), (), candidate_count, induced_edge_count, 0, 0, 0,
            (f"{STATE_NON_MANIFOLD}:invalid_member_count={len(invalid_members)}",),
        )

    unit_faces = tuple(
        face for face in context.surface_faces.observed_faces
        if set(face.ordered_region_indices) <= member_set
    )
    if not unit_faces:
        return ChartUnitCutBoundaryResult(
            coherence, (), (), candidate_count, induced_edge_count, 0, 0, 0,
            (STATE_INSUFFICIENT_TOPOLOGY,),
        )
    unit_incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in unit_faces:
        for halfedge in face.halfedges:
            unit_incidence.setdefault(edge_key(*halfedge), []).append(halfedge)

    boundary_halfedges: dict[tuple[int, int], ChartUnitCutBoundarySegment] = {}
    seam_count = 0
    reasons: list[str] = []
    for edge, halfedges in unit_incidence.items():
        if len(halfedges) == 2:
            continue
        if len(halfedges) != 1:
            reasons.append(f"{STATE_NON_MANIFOLD}:unit_face_incidence={len(halfedges)}")
            continue
        a, b = halfedges[0]
        full_incidence = len(context.surface_faces.face_incidence_by_edge.get(edge, ()))
        provenance, inconsistent = _provenance_kind(a, b, context)
        if provenance is not None:
            kind = provenance
        elif full_incidence == 2:
            kind = SEGMENT_PARTITION_SEAM
            seam_count += 1
        elif full_incidence == 1:
            reasons.append(
                f"{STATE_UNTYPED_PHYSICAL_BOUNDARY}:stable_ids="
                f"{context.stable_ids[a]},{context.stable_ids[b]}"
            )
            continue
        else:
            reasons.append(f"{STATE_NON_MANIFOLD}:full_face_incidence={full_incidence}")
            continue
        boundary_halfedges[(a, b)] = _make_segment(a, b, kind, inconsistent, context)
    if reasons:
        return ChartUnitCutBoundaryResult(
            coherence, (), (), candidate_count, induced_edge_count, seam_count, 0,
            len(unit_faces), tuple(reasons),
        )

    loops, loop_reasons = _trace_all_boundary_loops(boundary_halfedges, context)
    if loop_reasons:
        return ChartUnitCutBoundaryResult(
            coherence, (), (), candidate_count, induced_edge_count, seam_count, 0,
            len(unit_faces), loop_reasons,
        )
    # The current PCA-UV / untrimmed 6x6 NURBS path cannot silently discard a
    # topologically proven hole.  Preserve every loop in the result and stop.
    if any(loop.role == ROLE_INNER_BOUNDARY for loop in loops):
        return ChartUnitCutBoundaryResult(
            coherence, (), loops, candidate_count, induced_edge_count, seam_count,
            len(loops), len(unit_faces), (STATE_INNER_BOUNDARY_REVIEW_REQUIRED,),
        )

    domains: list[ChartUnitCutDomainResult] = []
    validation_reasons: list[str] = []
    for loop in loops:
        domain = _validate_outer_loop(
            loop, context, unit_positions, max_evidence_outside_domain_fraction,
        )
        if domain.materialized:
            domains.append(domain)
        else:
            detail = "|".join(domain.reasons)
            validation_reasons.append(
                f"topology_loop_rejected:{domain.state}:{detail}"
                if detail else f"topology_loop_rejected:{domain.state}"
            )
    if not domains and not validation_reasons:
        validation_reasons.append(STATE_NO_CLOSED_CUT_BOUNDARY)
    return ChartUnitCutBoundaryResult(
        coherence, tuple(domains), loops, candidate_count, induced_edge_count,
        seam_count, len(loops), len(unit_faces), tuple(validation_reasons),
    )


# Replay-facing aliases make the construction swap explicit without changing
# downstream coverage/PCA-UV/NURBS/held-out evaluation code.
build_chart_unit_topology_context = build_chart_unit_face_topology_context
materialize_chart_unit_cut_boundaries = materialize_chart_unit_face_incidence_boundaries
