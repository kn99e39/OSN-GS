from __future__ import annotations

"""Worklog 84: chart-unit coherence audit + evidence-scale boundary topology.

Worklog 83 assembled Worklog 82 micro-components into chart-scale units via
aggregate, redundant evidence (75-93% of region evidence recovered per
region, zero unsupported gap bridging audited). But two questions stayed
coupled and unresolved: is an assembled unit genuinely one coherent
parametric-chart unit rather than an over-merge, and can it derive its own
boundary topology instead of depending on 3-7 sparse representative nodes
(Worklog 80's macro topology) to type/geometrically bound tens to hundreds
of evidence points. This module closes both, reusing existing contracts only.

COHERENCE AUDIT (evidence-only, never fit quality):
Reuses `internal_normal_disagreement_fraction` (Worklog 82's own per-
micro-component non-manifold check, the SAME formula and the SAME 0.15
bound, unchanged) applied to the WHOLE assembled unit's evidence instead of
one micro-component's. A unit whose accepted-edge assembly chain folded
together evidence with real internal orientation disagreement is exactly
what that check already exists to catch -- applying it again at assembled
scale is reuse, not a new/tuned criterion. Units over the bound are
`AMBIGUOUS_OR_OVER_MERGED` and never reach boundary materialization.

EVIDENCE-SCALE BOUNDARY TOPOLOGY (coherent units only):
Worklog 80's `build_dense_chart_support` required a sparse macro cycle (3-7
representative nodes) to supply the geometric ARC ORDER a chart's dense
support gets assigned into -- workable when the chart IS the representative-
scale region, unworkable when an assembled unit spans hundreds of evidence
points with only a handful of representative nodes touching it.

The first design tried here reused `extract_dense_boundary_support`'s own
`_connect` closed-loop recovery (mutual +/-tangent half-line selection) as
the ORDER, not just the candidate filter. Measured directly on real
baseline_compatible@2900 assembled units: 0/178 materialized -- 107
`not_closed`, 71 `no_dense_support`. This reproduces Worklog 71's own
already-documented finding (17/282 seed components ever reached
`closed_loop_recovered`, mostly degenerate triangles): `_connect`'s mutual-
match connectivity is too strict to serve as a chart-unit's PRIMARY topology
source on real evidence, independent of this redesign.

This module instead uses `extract_dense_boundary_support` for CANDIDATE
ADMISSION ONLY (Worklog 77's corrected predicate, unmodified -- which is
observed to work reliably; only the downstream `_connect` ordering was the
problem) and orders the admitted candidates by ANGLE around their own
centroid in the unit's OWN best-fit tangent plane (SVD over the unit's own
evidence -- no sparse dependency, no global/region-level projection). This
is evidence-scale by construction and requires no sparse macro node. It is
validated, not assumed, twice before acceptance:
  1. `evaluate_closed_loop_geometry` (Worklog 71, unmodified) checks the
     resulting ordering for self-intersection, so a genuinely non-star-shaped
     (concave-from-centroid) boundary fails closed as `SELF_INTERSECTING`
     rather than silently producing a crossed polygon;
  2. `measure_edge_support_occupancy` (Worklog 76, disclosure-only there,
     Worklog 83's first acceptance use) checks that EVERY resulting edge --
     including the wrap-around edge -- runs along observed evidence. An
     angular ordering always closes SOME polygon, including, for a genuinely
     OPEN arc with no far side, a straight chord across empty space; that
     chord is not a self-intersection, so nothing above would catch it. A
     unit with any such edge fails closed as `UNSUPPORTED_CLOSURE` instead of
     silently bridging the gap.
Sparse macro topology is used only AFTER that ordering exists, and only for
two things:
  * typing each boundary vertex with the nearest sparse arc's `segment_kind`
    (crease / physical_termination / observation_frontier) -- label only,
    never geometry;
  * disclosing (not silently accepting) any boundary vertex whose nearest
    macro arc is a `crease` while the loop's own dense connectivity carried
    it in from the OTHER side of that same arc -- an inconsistent
    continuation the assembly stage's crease veto should already have
    prevented between micro-components, checked again here as a closing
    audit at the unit's final boundary.

Every failure is typed and fail-closed: no hull, no PCA rectangle, no
bounding box, no alpha shape, no forced closure, no gap bridging. If the
unit's own dense evidence does not produce a closed loop, this module
reports why and produces nothing.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_dense_surface_consistency_components import (
    NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND,
    _nearest_arc_side,
    internal_normal_disagreement_fraction,
)
from osn_gs.surface.torch_boundary_support_spacing import measure_edge_support_occupancy
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_region_owned_full_evidence import (
    MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
    evidence_outside_chart_domain_fraction,
)
from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import evaluate_closed_loop_geometry
from osn_gs.utils.torch_ops import require_torch

STATE_MATERIALIZED = "chart_unit_boundary_materialized"
STATE_AMBIGUOUS_OR_OVER_MERGED = "chart_unit_ambiguous_or_over_merged"
STATE_NO_DENSE_SUPPORT = "chart_unit_boundary_no_dense_support"
STATE_SELF_INTERSECTING = "chart_unit_boundary_self_intersecting"
STATE_UNSUPPORTED_CLOSURE = "chart_unit_boundary_unsupported_closure"
STATE_COVERAGE_FAILED = "chart_unit_boundary_coverage_failed"


@dataclass(frozen=True)
class ChartUnitBoundarySegment:
    stable_id_a: Any
    stable_id_b: Any
    segment_kind: str  # nearest sparse-arc label, or "" if no macro arc coverage
    crease_inconsistent: bool  # loop crossed a crease arc without a matching veto upstream


@dataclass(frozen=True)
class ChartUnitCoherence:
    coherent: bool
    internal_normal_disagreement_fraction: float


@dataclass(frozen=True)
class ChartUnitBoundaryResult:
    state: str
    coherence: ChartUnitCoherence | None
    ordered_stable_ids: tuple[Any, ...]
    ordered_positions: Any | None
    segments: tuple[ChartUnitBoundarySegment, ...]
    evidence_outside_domain_fraction: float | None
    crease_inconsistent_segment_count: int
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


def assess_chart_unit_coherence(covariance: Any, member_local_indices: Sequence[int]) -> ChartUnitCoherence:
    """Evidence-only coherence audit -- reuses Worklog 82's own formula/bound
    at assembled chart-unit scale. Never reads fit quality."""

    normals = extract_covariance_frame(covariance).normal_candidate
    fraction = internal_normal_disagreement_fraction(normals, member_local_indices)
    return ChartUnitCoherence(fraction <= NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND, fraction)


def materialize_chart_unit_boundary(
    positions: Any,
    covariance: Any,
    stable_ids: Sequence[Any],
    full_evidence_positions: Any,
    *,
    arc_starts: Any | None = None,
    arc_ends: Any | None = None,
    arc_kinds: Sequence[str] | None = None,
    max_evidence_outside_domain_fraction: float = MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION,
) -> ChartUnitBoundaryResult:
    """Coherence-audit an assembled chart unit, then (if coherent) recover its
    boundary topology directly from its OWN dense evidence.

    ``positions``/``covariance``/``stable_ids`` describe the chart unit's
    member evidence only. ``full_evidence_positions`` is the SAME unit's
    evidence again (kept as a separate argument to make the Worklog 79
    coverage-contract call site explicit and match its existing signature).
    ``arc_starts``/``arc_ends``/``arc_kinds`` are Worklog 80's sparse macro
    arcs, in the SAME frame as ``positions`` -- optional: typing is
    best-effort, never required to produce geometry.
    """

    torch = require_torch()
    n = int(positions.shape[0])

    def _fail(state: str, coherence: ChartUnitCoherence | None, *reasons: str) -> ChartUnitBoundaryResult:
        return ChartUnitBoundaryResult(state, coherence, (), None, (), None, 0, tuple(reasons))

    coherence = assess_chart_unit_coherence(covariance, list(range(n)))
    if not coherence.coherent:
        return _fail(
            STATE_AMBIGUOUS_OR_OVER_MERGED, coherence,
            f"internal_normal_disagreement_fraction={coherence.internal_normal_disagreement_fraction:.4f}"
            f">{NON_MANIFOLD_DISAGREEMENT_FRACTION_BOUND}",
        )

    normals = extract_covariance_frame(covariance).normal_candidate
    support = extract_dense_boundary_support(positions, normals, list(stable_ids))
    if len(support.candidates) < 3:
        return _fail(STATE_NO_DENSE_SUPPORT, coherence, f"admitted_boundary_candidate_count={len(support.candidates)}<3")

    # Order the ADMITTED candidates (Worklog 77's corrected predicate,
    # unmodified -- this admission step is reliable) by ANGLE around their
    # own centroid in the unit's OWN best-fit tangent plane (SVD over the
    # unit's own evidence). This is evidence-scale and sparse-independent by
    # construction; `_connect`'s stricter mutual-match connectivity was tried
    # first and measured to fail on 178/178 real units (matching Worklog 71's
    # already-documented 17/282 closure rate) -- see module docstring.
    ordered_ids = tuple(c.stable_id for c in support.candidates)
    candidate_positions = torch.stack(
        [torch.tensor(c.position, dtype=positions.dtype, device=positions.device) for c in support.candidates],
        dim=0,
    )
    centered = positions - positions.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis_u, axis_v = vh[0], vh[1]
    candidate_centroid = candidate_positions.mean(dim=0)
    relative = candidate_positions - candidate_centroid[None, :]
    angles = torch.atan2(relative @ axis_v, relative @ axis_u)
    order = torch.argsort(angles)
    ordered_ids = tuple(ordered_ids[int(i)] for i in order.tolist())
    ordered_positions = candidate_positions[order]

    geometry = evaluate_closed_loop_geometry(
        [tuple(float(v) for v in row) for row in ordered_positions.detach().cpu().tolist()]
    )
    if geometry.crossing_check == "checked" and geometry.proper_crossing_count > 0:
        return _fail(STATE_SELF_INTERSECTING, coherence, f"proper_crossing_count={geometry.proper_crossing_count}")

    # Angular ordering always closes SOME polygon -- including, for an open
    # arc (no genuine far side), a straight chord across the open end. That
    # chord is exactly the gap-bridging this whole project forbids, and
    # nothing above would catch it (a chord across empty space is not a
    # self-intersection). Reuses `measure_edge_support_occupancy` (Worklog
    # 76, disclosure-only there, Worklog 83's first acceptance use) as a
    # closing safety gate: every edge of the accepted loop, including the
    # wrap-around edge, must run along observed evidence.
    edge_pairs = [(i, (i + 1) % len(ordered_ids)) for i in range(len(ordered_ids))]
    occupancy = measure_edge_support_occupancy(
        edge_pairs, ordered_positions, positions, full_evidence_spacing=support.full_evidence_scale,
    )
    if occupancy["edges_with_empty_interior_bin"] > 0:
        return _fail(
            STATE_UNSUPPORTED_CLOSURE, coherence,
            f"edges_with_empty_interior_bin={occupancy['edges_with_empty_interior_bin']}/{occupancy['edge_count']}",
        )

    # Typed provenance: label only, from the nearest sparse macro arc.
    arc_side = None
    if arc_starts is not None and arc_ends is not None and arc_kinds and int(arc_starts.shape[0]) > 0:
        arc_side = _nearest_arc_side(ordered_positions, arc_starts, arc_ends, arc_kinds)

    k = len(ordered_ids)
    segments = []
    crease_inconsistent_count = 0
    for i in range(k):
        kind_a = arc_side[i] if arc_side is not None else ""
        kind_b = arc_side[(i + 1) % k] if arc_side is not None else ""
        # A segment whose two endpoints sit nearest to DIFFERENT typed arcs
        # where one of them is a crease is disclosed as a closing-audit
        # inconsistency: the assembly stage's crease veto (Worklog 83) should
        # already have prevented a same-unit continuation across a crease, so
        # this should not fire in practice -- reported explicitly rather than
        # assumed.
        inconsistent = bool(kind_a and kind_b and kind_a != kind_b and ("crease" in kind_a or "crease" in kind_b))
        if inconsistent:
            crease_inconsistent_count += 1
        segments.append(ChartUnitBoundarySegment(ordered_ids[i], ordered_ids[(i + 1) % k], kind_a or kind_b, inconsistent))

    outside = evidence_outside_chart_domain_fraction(ordered_positions, full_evidence_positions)
    if outside is not None and outside > max_evidence_outside_domain_fraction:
        return ChartUnitBoundaryResult(
            STATE_COVERAGE_FAILED, coherence, ordered_ids, ordered_positions, tuple(segments), outside,
            crease_inconsistent_count,
            (f"evidence_outside_chart_domain_fraction={outside:.4f}>{max_evidence_outside_domain_fraction}",),
        )

    return ChartUnitBoundaryResult(
        STATE_MATERIALIZED, coherence, ordered_ids, ordered_positions, tuple(segments), outside,
        crease_inconsistent_count, (),
    )
