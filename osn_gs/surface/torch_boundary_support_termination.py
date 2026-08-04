from __future__ import annotations

"""Topology-local, reference-rotation-invariant support termination evidence."""

import math
from typing import Any, Mapping, Sequence

from osn_gs.surface.torch_full_cloud_continuation_shell import (
    STATE_NO_GAP,
    STATE_OBSERVED_TERMINATION,
    ContinuationTerminationQuery,
)
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    CANDIDATE_STATUS_CANDIDATE,
    RELATION_CREASE,
    RELATION_PARALLEL_SEPARATE,
    RELATION_SAME_SURFACE,
)
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationResult

# Confidence per continuation state (worklog 130) -- mirrors the existing
# 0.7/0.4 observed/sampling-gap split, extended to the richer state set.
_CONTINUATION_CONFIDENCE = {
    STATE_OBSERVED_TERMINATION: 0.7,
    "reliability_frontier": 0.5,
    "unresolved_sampling_gap": 0.4,
    "crease_discontinuity": 0.5,
    "parallel_sheet_conflict": 0.5,
    "ambiguous_continuation": 0.3,
}


def _missing_sector_runs(occupied: set[int], sectors: int) -> tuple[tuple[int, ...], ...]:
    """Circular contiguous runs; sector zero cannot split one physical gap."""
    if len(occupied) == sectors:
        return ()
    if not occupied:
        return (tuple(range(sectors)),)
    start = next(item for item in range(sectors) if item in occupied and (item + 1) % sectors not in occupied)
    ordered = [(start + 1 + offset) % sectors for offset in range(sectors - 1)]
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for item in ordered:
        if item in occupied:
            if current:
                runs.append(tuple(current))
                current = []
        else:
            current.append(item)
    if current:
        runs.append(tuple(current))
    return tuple(item for item in runs if len(item) >= 2)


def _unit(vector: Any):
    return vector / vector.norm().clamp_min(1e-12)


def _largest_geometric_gap(local: Sequence[Any], axis_u: Any, axis_v: Any):
    """Return largest support-free arc and its world direction, independent of bin origin."""
    angles = sorted(math.atan2(float(vector @ axis_v), float(vector @ axis_u)) for vector in local)
    gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) for i in range(len(angles))]
    index = max(range(len(gaps)), key=lambda item: gaps[item])
    # The support resultant is a geometry-only outward direction. Unlike a
    # sector-bin midpoint it is unchanged when the tangent-frame reference or
    # the unoriented normal is reversed.
    resultant = sum(local[1:], local[0].clone())
    if float(resultant.norm()) > 1e-8:
        return gaps[index], -_unit(resultant)
    center = angles[index] + gaps[index] * 0.5
    return gaps[index], axis_u * math.cos(center) + axis_v * math.sin(center)


def _candidate_from_continuation_query(region_id: int, node_id: Any, position: Any, canonical: Any, query: ContinuationTerminationQuery) -> WorldSpaceBoundaryHalfEdgeCandidate | None:
    """Build a candidate from full-cloud continuation evidence (worklog 130).

    Emits one diagnostic candidate per EVERY non-``no_gap`` state, mirroring
    how ``torch_world_space_boundary_halfedges.py`` already emits diagnostic-
    only crease/parallel/ambiguous candidates -- only
    ``observed_support_termination`` gets ``ordering_state="locally_chainable"``
    and therefore only that one ever reaches directed ordering (see
    ``torch_directed_boundary_ordering.py``'s input filter). A
    ``reliability_frontier``/``unresolved_sampling_gap`` candidate is real,
    observable, and useful for diagnostics, but is explicitly NEVER used to
    close a boundary loop.
    """
    if query.state == STATE_NO_GAP or query.outward_direction is None:
        return None
    normal = canonical.oriented_normal
    outward = query.outward_direction
    boundary_tangent = normal.cross(outward, dim=0)
    tangent_norm = float(boundary_tangent.norm())
    if tangent_norm <= 1e-8:
        return None
    boundary_tangent = boundary_tangent / tangent_norm
    return WorldSpaceBoundaryHalfEdgeCandidate(
        half_edge_id=f"region:{region_id}:gaussian:{node_id}:continuation:{query.state}",
        source_region_id=region_id,
        source_gaussian_id=node_id,
        adjacent_gaussian_id=None,
        world_position=tuple(float(x) for x in position),
        local_normal=tuple(float(x) for x in normal),
        local_tangent_direction=tuple(float(x) for x in boundary_tangent),
        boundary_direction=tuple(float(x) for x in boundary_tangent),
        boundary_reason=query.state,
        source_pair_ids=None,
        confidence=_CONTINUATION_CONFIDENCE.get(query.state, 0.3),
        ordering_state="locally_chainable" if query.state == STATE_OBSERVED_TERMINATION else "ambiguous_ordering",
        review_reasons=("full_cloud_continuation_shell_gap",),
        gap_width_degrees=query.gap_width_degrees,
        same_mode_support_count=query.same_mode_support_count,
        same_mode_opacity_mass=query.same_mode_opacity_mass,
        ambiguous_continuation_mass=query.ambiguous_continuation_mass,
        competing_mode_mass=query.competing_mode_mass,
        support_radius=query.support_radius,
        reliability_frontier=query.state == "reliability_frontier",
        sampling_gap=query.state == "unresolved_sampling_gap",
        source_full_cloud_fingerprint=query.source_full_cloud_fingerprint,
        policy_version=query.policy_version,
    )


SMOOTH_CROSS_REGION_CONTINUATION = "smooth_cross_region_continuation"

# Worklog 41 (task section 5): explicit region-pair relation homogeneity.
# A pair whose relation evidence is spatially mixed must never be approved as
# smooth continuation on the strength of the aggregate alone.
PAIR_HOMOGENEOUS_SMOOTH = "homogeneous_smooth_continuation"
PAIR_HOMOGENEOUS_CREASE = "homogeneous_crease_adjacent"
PAIR_HOMOGENEOUS_PARALLEL = "homogeneous_parallel_separate"
PAIR_SPATIALLY_MIXED = "spatially_mixed_relation"
PAIR_INSUFFICIENT_EVIDENCE = "insufficient_local_relation_evidence"


def classify_cross_region_pairs(region_result: RegionFormationResult, affinity_graph: Any) -> dict:
    """Worklog 40 (task section 3/4): per REGION PAIR, what does the affinity
    graph say about the relation between two regions that touch?

    Returns ``{(low_region, high_region): verdict}`` where verdict is one of
    ``crease_adjacent`` / ``parallel_separate`` / ``smooth_continuation`` /
    ``ambiguous``. Pure aggregation of relations the manifold affinity graph
    has ALREADY computed -- no new geometry, no new threshold, bounded by the
    existing candidate-edge set.

    Why region-pair level and not per-Gaussian-pair: a candidate's outward
    arc is filled by out-of-region Gaussians that frequently have no direct
    affinity edge to the candidate itself (bounded-kNN drops it). Measured on
    the sphere, all 22 seam candidates individually look like "cross-region
    support with no relation evidence", yet the region PAIR carries 12
    ``same_surface`` edges and zero ``crease`` -- the graph does know the two
    hemispheres are one smooth surface.

    Measured verdicts (worklog 40): sphere (0,1) same_surface=12 crease=0 ->
    smooth_continuation; every box face pair crease=32-33 -> crease_adjacent;
    cylinder side/cap crease=88-90 -> crease_adjacent; thin_slab (0,1)
    parallel_but_separate=57 -> parallel_separate. The sphere is the ONLY
    fixture whose touching regions are crease-free and same_surface-bearing,
    which is exactly the distinction this function has to make.
    """
    counts: dict = {}
    for edge in affinity_graph.edges:
        if edge.candidate_status != CANDIDATE_STATUS_CANDIDATE:
            continue
        left_region = region_result.node_region_id[edge.source]
        right_region = region_result.node_region_id[edge.target]
        if left_region < 0 or right_region < 0 or left_region == right_region:
            continue
        key = (min(left_region, right_region), max(left_region, right_region))
        bucket = counts.setdefault(key, {"crease": 0, "parallel": 0, "same_surface": 0})
        if edge.manifold_relation == RELATION_CREASE:
            bucket["crease"] += 1
        elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
            bucket["parallel"] += 1
        elif edge.manifold_relation == RELATION_SAME_SURFACE:
            bucket["same_surface"] += 1

    verdicts: dict = {}
    for key, bucket in counts.items():
        # Precedence is deliberately conservative: ANY crease evidence, or
        # parallel evidence that is not outweighed by same_surface evidence,
        # blocks the smooth-continuation verdict. Only a pair that is
        # crease-free AND same_surface-dominant is treated as one surface.
        if bucket["crease"] > 0:
            verdicts[key] = "crease_adjacent"
        elif bucket["parallel"] >= bucket["same_surface"] and bucket["parallel"] > 0:
            verdicts[key] = "parallel_separate"
        elif bucket["same_surface"] > 0:
            verdicts[key] = "smooth_continuation"
        else:
            verdicts[key] = "ambiguous"
    return verdicts


def collect_cross_region_relation_sources(region_result: RegionFormationResult, affinity_graph: Any) -> dict:
    """Worklog 41 (task section 4): per region pair, the ENDPOINT NODES that
    carry each cross-region relation class.

    The region-pair verdict alone answers "somewhere on this pair, what do the
    two regions look like". A candidate needs the stricter question: "is there
    same-surface evidence NEAR ME, and no crease/parallel evidence near me".
    Returning the source nodes lets the candidate check locality against the
    same bounded support radius it already uses -- no new geometry and no new
    threshold.

    Measured motivation (worklog 41): on the sphere the aggregate verdict is
    applied to candidates up to 0.375 away from the nearest supporting
    ``same_surface`` edge, which exceeds the sphere's own 0.30 radius. No
    current fixture turns that into a false suppression, but the aggregate is
    genuinely non-local, so suppression is additionally gated on local
    evidence below.
    """
    sources: dict = {}
    for edge in affinity_graph.edges:
        if edge.candidate_status != CANDIDATE_STATUS_CANDIDATE:
            continue
        left_region = region_result.node_region_id[edge.source]
        right_region = region_result.node_region_id[edge.target]
        if left_region < 0 or right_region < 0 or left_region == right_region:
            continue
        key = (min(left_region, right_region), max(left_region, right_region))
        bucket = sources.setdefault(key, {"crease": set(), "parallel": set(), "same_surface": set()})
        if edge.manifold_relation == RELATION_CREASE:
            bucket["crease"].update((edge.source, edge.target))
        elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
            bucket["parallel"].update((edge.source, edge.target))
        elif edge.manifold_relation == RELATION_SAME_SURFACE:
            bucket["same_surface"].update((edge.source, edge.target))
    return sources


def extract_support_termination_candidates(positions: Any, normals: Any, tangent_scales: Any, region_result: RegionFormationResult, *, ids: Sequence[Any] | None = None, sectors: int = 8, canonical_frames: Sequence[Any | None] | None = None, continuation: Mapping[Any, ContinuationTerminationQuery] | None = None, affinity_graph: Any | None = None) -> tuple[WorldSpaceBoundaryHalfEdgeCandidate, ...]:
    """Produce at most one support-termination candidate per accepted node.

    Sector occupancy remains a robustness guard, but candidate direction comes
    from the actual circular support gap. Consequently a valid tangent-frame
    reference can rotate sector labels without changing boundary evidence.

    ``continuation`` (worklog 130, optional): a per-node
    :class:`~osn_gs.surface.torch_full_cloud_continuation_shell.ContinuationTerminationQuery`
    lookup. When a node has an entry here, its candidate comes from the
    full-cloud continuous circular gap query instead of the representative-
    only sector histogram below -- the representative-only path is otherwise
    completely unchanged (existing synthetic-fixture behavior is preserved
    whenever ``continuation`` is not supplied, e.g. every pre-worklog-130
    caller and test).
    """
    count = len(region_result.node_region_id)
    ids = tuple(range(count)) if ids is None else tuple(ids)
    index = {item: node for node, item in enumerate(ids)}
    cross_region_verdicts = (
        classify_cross_region_pairs(region_result, affinity_graph)
        if affinity_graph is not None else {}
    )
    cross_region_relation_sources = (
        collect_cross_region_relation_sources(region_result, affinity_graph)
        if affinity_graph is not None else {}
    )
    adjacency = {node: [] for node in range(count)}
    for region in region_result.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                adjacency[index[left]].append(index[right])
                adjacency[index[right]].append(index[left])

    candidates = []
    width = 2 * math.pi / sectors
    for source in range(count):
        region_id = region_result.node_region_id[source]
        if region_id < 0 or region_result.node_membership_state[source] not in ("core_member", "consensus_attached"):
            continue
        canonical = canonical_frames[source] if canonical_frames is not None else None
        if canonical is None:
            continue
        node_id = ids[source]
        query = continuation.get(node_id) if continuation is not None else None
        if query is not None:
            candidate = _candidate_from_continuation_query(region_id, node_id, positions[source], canonical, query)
            if candidate is not None:
                candidates.append(candidate)
            continue
        normal = canonical.oriented_normal
        axis_u = canonical.tangent_axis_0
        axis_v = canonical.tangent_axis_1
        local = []
        for target in adjacency[source]:
            delta = positions[target] - positions[source]
            tangent = delta - normal * (delta @ normal)
            distance = float(tangent.norm())
            if 1e-8 < distance <= float(tangent_scales[source]) * 4.0:
                local.append(_unit(tangent))
        if len(local) < 2:
            continue

        occupied: set[int] = set()
        for vector in local:
            angle = math.atan2(float(vector @ axis_v), float(vector @ axis_u))
            value = (angle + math.pi) / width
            primary = int(math.floor(value)) % sectors
            occupied.add(primary)
            # Share a small angular margin across bin boundaries so a tiny
            # representation-level perturbation cannot create a new run.
            fraction = value - math.floor(value)
            if fraction < 0.15:
                occupied.add((primary - 1) % sectors)
            if fraction > 0.85:
                occupied.add((primary + 1) % sectors)
        runs = _missing_sector_runs(occupied, sectors)
        gap, outward = _largest_geometric_gap(local, axis_u, axis_v)
        # Worklog 38 (task section 13) DIAGNOSIS -- deliberately NOT patched
        # here. The geometric gap is the canonical termination measurement
        # (see this function's docstring), but the sector histogram can veto
        # it: `occupied` is deliberately smeared (+-0.15 of a bin) so
        # bin-boundary jitter cannot invent a run, and with enough neighbours
        # that smearing can mark all `sectors` bins occupied even when the
        # measured `gap` clearly clears `width * 1.5`. Measured on the
        # cylinder cap fixture: nodes 257/266 have gap 1.568/1.590 rad
        # against a 1.178 rad threshold yet occupied == all 8 sectors, so
        # `runs == ()` drops them and leaves a ~50-degree hole in the cap's
        # candidate ring. One extra accepted neighbour is enough to flip a
        # node into that state, which is why improved core seeding surfaced
        # this as the worklog 37 "cylinder cap regression".
        #
        # Every reconciliation tried in worklog 38 (an independent
        # gap-dominates-histogram threshold, deriving the run from the
        # measured gap span) came down to choosing a new constant that
        # happened to sit just above this fixture's 1.568 rad -- i.e. tuning
        # a threshold to a scene, which section 13 explicitly forbids. The
        # defect is real and precisely located, but the correct repair is a
        # principled reconciliation of the smeared histogram with the
        # geometric measurement, which needs its own scoped round rather
        # than a constant picked to fit one cap. Left unpatched and reported.
        if not runs or gap < width * 1.5:
            continue

        outward = _unit(outward)
        boundary_tangent = _unit(normal.cross(outward, dim=0))
        reason = "observed_support_termination" if len(local) >= 3 else "unresolved_sampling_gap"

        # Worklog 40 (task section 4/5/15 Case A): cross-region smooth
        # continuation certificate. The angular gap above is measured over
        # SAME-REGION accepted adjacency, so a node on a region frontier
        # reports a "support-free" direction that is in fact occupied by
        # observed Gaussians belonging to a neighbouring region. Whether that
        # makes the candidate nonphysical depends entirely on WHAT that
        # neighbouring region is:
        #
        #   crease_adjacent   -> a real surface really does end here (box
        #                        face at a crease, cylinder side/cap). KEEP.
        #   parallel_separate -> the neighbour is the opposite/parallel sheet,
        #                        not a continuation (thin slab). KEEP.
        #   ambiguous         -> insufficient/conflicting evidence. KEEP and
        #                        leave for review; never silently suppressed.
        #   smooth_continuation -> the affinity graph itself says the two
        #                        regions are one smooth surface, so this
        #                        "boundary" is a region-fragmentation
        #                        artifact, not physical termination.
        #
        # Only the last case is reclassified, and only to the existing
        # non-physical `reliability_frontier` state -- the candidate is still
        # emitted with full provenance, it simply stops claiming to be a
        # physical boundary and therefore never reaches directed ordering.
        # Verdicts come from `classify_cross_region_pairs`, a bounded
        # aggregation of relations the affinity graph already computed; no
        # new geometry, no new threshold, no scene-specific constant.
        #
        # Worklog 39 attempted the cruder version of this (suppress whenever
        # ANY out-of-region support occupies the arc) and it destroyed every
        # genuine candidate on box/cylinder/thin_slab, because those are
        # exactly the crease/parallel cases. Consulting the relation class is
        # what separates them.
        # Worklog 41 (task section 2/4/5): candidate-local hardening of this
        # suppression was implemented, measured, and NOT adopted -- with the
        # audit recorded here because the aggregate's non-locality is real.
        #
        # Measured scope of the region-pair prior: on the sphere it is applied
        # to candidates up to 0.375 from the nearest supporting `same_surface`
        # edge (the sphere's own radius is 0.30), and the pair is itself
        # mixed (12 same_surface + 2 parallel, the two classes only 0.037
        # apart). So the aggregate genuinely is non-local.
        #
        # Two candidate-local certificates were built and measured:
        #   (1) require same_surface evidence within the candidate's own
        #       support radius -- regressed sphere 0 -> 11 physical
        #       candidates, because 11 of 22 seam candidates have no
        #       same_surface source node inside their radius even though the
        #       seam demonstrably is one surface (bounded-kNN distributes the
        #       evidence unevenly along the seam).
        #   (2) additionally veto on locally-present crease/parallel evidence
        #       -- still regressed sphere 0 -> 8, firing near the pair's 4
        #       parallel-evidence nodes.
        # A "nearest evidence class wins" variant was also measured (17/22)
        # and rejected: it was chosen for scoring better, not for being
        # principled, which is the scene-tuning this task forbids.
        #
        # Crucially, neither variant protected anything: every mixed-relation
        # fixture built for this round (smooth-to-crease composite;
        # mostly-smooth seam with a localized fold at 15/25/40% of its
        # length; smooth-to-gap; two surfaces touching at a single point) is
        # already resolved upstream by the crease-precedence rule in
        # `classify_cross_region_pairs`, which returns crease_adjacent (or no
        # verdict at all) and never reaches this branch. The locality gates
        # therefore cost real sphere accuracy while adding no measured
        # safety, so the worklog 40 behaviour is kept and the residual
        # non-locality is reported as a known, currently-unexercised risk.
        if reason == "observed_support_termination" and cross_region_verdicts:
            for target in range(count):
                target_region = region_result.node_region_id[target]
                if target_region < 0 or target_region == region_id:
                    continue
                delta = positions[target] - positions[source]
                tangent = delta - normal * (delta @ normal)
                distance = float(tangent.norm())
                if not (1e-8 < distance <= float(tangent_scales[source]) * 4.0):
                    continue
                if float(_unit(tangent) @ outward) < 0.0:
                    continue
                key = (min(region_id, target_region), max(region_id, target_region))
                if cross_region_verdicts.get(key) == "smooth_continuation":
                    reason = SMOOTH_CROSS_REGION_CONTINUATION
                    break

        # Worklog 39 (task section 10) DIAGNOSIS -- deliberately NOT patched.
        # The angular gap above is measured over `adjacency`, which is
        # `internal_accepted_edge_ids`: REGION-topology evidence (bounded-kNN
        # affinity edges that survived region formation), not a record of
        # which directions actually carry observed support. A node on a
        # REGION FRONTIER therefore shows a large "gap" pointing at the
        # neighbouring region even where the surface plainly continues, and
        # that frontier is emitted as `observed_support_termination`.
        #
        # Measured on the closed sphere (no physical boundary anywhere): it
        # fragments into two ~hemispheres (99/90) and emits 22
        # `observed_support_termination` candidates, ALL on the seam. Each
        # has ~25 same-region AND ~26 other-region observed neighbours inside
        # its own support radius -- the "gap" is fully occupied by real
        # Gaussians that merely carry a different region id. Recomputing the
        # gap over all spatially-observed neighbours drops sphere 22 -> 0
        # while box_face keeps all 32.
        #
        # The obvious guard -- demote to `reliability_frontier` when
        # out-of-region observed support lies in the chosen outward direction
        # -- was implemented and measured, and it is WRONG: it also erases
        # every genuine candidate on box (110 -> 0), cylinder (74 -> 0,
        # closed 2 -> 0) and thin_slab (48 -> 3), because on a multi-region
        # solid a real physical patch boundary legitimately abuts another
        # region across a real crease. Out-of-region support alone cannot
        # distinguish "this surface continues here" from "a DIFFERENT surface
        # meets here"; that needs the crease/parallel relation evidence the
        # affinity graph already computes, wired through to this stage. That
        # is a real repair but a structural one, out of scope for a guard
        # patch, so it is reported rather than approximated.
        # Source Gaussian provenance is canonical; raw sector indexes and
        # iteration order never appear in the candidate identity.
        candidates.append(WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"region:{region_id}:gaussian:{ids[source]}:support-termination:{reason}",
            source_region_id=region_id,
            source_gaussian_id=ids[source],
            adjacent_gaussian_id=None,
            world_position=tuple(float(x) for x in positions[source]),
            local_normal=tuple(float(x) for x in normal),
            local_tangent_direction=tuple(float(x) for x in boundary_tangent),
            boundary_direction=tuple(float(x) for x in boundary_tangent),
            boundary_reason=reason,
            source_pair_ids=None,
            confidence=0.7 if reason == "observed_support_termination" else 0.4,
            ordering_state="locally_chainable" if reason == "observed_support_termination" else "ambiguous_ordering",
            review_reasons=("local_tangent_sector_missing_continuation",),
            reliability_frontier=reason == "reliability_frontier",
            sampling_gap=reason == "unresolved_sampling_gap",
        ))
    return tuple(sorted(candidates, key=lambda item: item.half_edge_id))


def normalize_continuation_candidates(
    candidates: Sequence[WorldSpaceBoundaryHalfEdgeCandidate],
    *,
    origin_proximity_ratio: float = 0.5,
    direction_alignment_min: float = 0.8,
) -> tuple[WorldSpaceBoundaryHalfEdgeCandidate, ...]:
    """Merge near-duplicate continuation candidates (worklog 130 item 9).

    Adjacent reliable representatives in the same region can independently
    detect the SAME physical boundary segment. Only candidates that both
    carry continuation provenance (``support_radius is not None`` --
    sector-based candidates are left untouched, so this is a no-op whenever
    ``continuation`` was never supplied) AND share the same
    ``(source_region_id, boundary_reason)`` AND lie within
    ``origin_proximity_ratio`` of the SMALLER of the two candidates'
    ``support_radius`` AND whose ``boundary_direction``s are aligned above
    ``direction_alignment_min`` are merged. Distance-only merging would wrongly
    fuse genuinely separate creases/parallel-conflicts/physical boundaries
    that merely happen to be close -- the direction-alignment and same-reason
    gates exist specifically to prevent that.

    The kept representative is the one with the larger ``same_mode_support_count``
    (ties broken by ``half_edge_id`` for determinism); its provenance
    fingerprint is extended with the merged candidates' fingerprints.
    """
    torch_free_candidates = [c for c in candidates if c.support_radius is None]
    continuation_candidates = [c for c in candidates if c.support_radius is not None]
    if len(continuation_candidates) < 2:
        return tuple(sorted(candidates, key=lambda item: item.half_edge_id))

    groups: dict[tuple[int, str], list[int]] = {}
    for index, candidate in enumerate(continuation_candidates):
        groups.setdefault((candidate.source_region_id, candidate.boundary_reason), []).append(index)

    parent = list(range(len(continuation_candidates)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for _, indices in groups.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                left, right = continuation_candidates[indices[i]], continuation_candidates[indices[j]]
                origin_left = math.sqrt(sum((a - b) ** 2 for a, b in zip(left.world_position, right.world_position)))
                radius = min(left.support_radius, right.support_radius)
                if radius <= 0 or origin_left > radius * origin_proximity_ratio:
                    continue
                alignment = sum(a * b for a, b in zip(left.boundary_direction, right.boundary_direction))
                if alignment < direction_alignment_min:
                    continue
                union(indices[i], indices[j])

    clusters: dict[int, list[int]] = {}
    for index in range(len(continuation_candidates)):
        clusters.setdefault(find(index), []).append(index)

    merged = []
    for members in clusters.values():
        if len(members) == 1:
            merged.append(continuation_candidates[members[0]])
            continue
        best = max(members, key=lambda i: (continuation_candidates[i].same_mode_support_count, continuation_candidates[i].half_edge_id))
        kept = continuation_candidates[best]
        fingerprint = tuple(
            item
            for i in sorted(members)
            for item in continuation_candidates[i].source_full_cloud_fingerprint
        )
        merged.append(
            WorldSpaceBoundaryHalfEdgeCandidate(
                half_edge_id=kept.half_edge_id, source_region_id=kept.source_region_id,
                source_gaussian_id=kept.source_gaussian_id, adjacent_gaussian_id=kept.adjacent_gaussian_id,
                world_position=kept.world_position, local_normal=kept.local_normal,
                local_tangent_direction=kept.local_tangent_direction, boundary_direction=kept.boundary_direction,
                boundary_reason=kept.boundary_reason, source_pair_ids=kept.source_pair_ids,
                confidence=kept.confidence, ordering_state=kept.ordering_state,
                review_reasons=kept.review_reasons + ("normalized_duplicate_merge",),
                gap_width_degrees=kept.gap_width_degrees, same_mode_support_count=kept.same_mode_support_count,
                same_mode_opacity_mass=kept.same_mode_opacity_mass,
                ambiguous_continuation_mass=kept.ambiguous_continuation_mass,
                competing_mode_mass=kept.competing_mode_mass, support_radius=kept.support_radius,
                reliability_frontier=kept.reliability_frontier, sampling_gap=kept.sampling_gap,
                source_full_cloud_fingerprint=fingerprint, policy_version=kept.policy_version,
            )
        )
    return tuple(sorted(list(torch_free_candidates) + merged, key=lambda item: item.half_edge_id))
