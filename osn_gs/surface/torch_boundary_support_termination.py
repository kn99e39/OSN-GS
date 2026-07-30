from __future__ import annotations

"""Topology-local, reference-rotation-invariant support termination evidence."""

import math
from typing import Any, Mapping, Sequence

from osn_gs.surface.torch_full_cloud_continuation_shell import (
    STATE_NO_GAP,
    STATE_OBSERVED_TERMINATION,
    ContinuationTerminationQuery,
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


def extract_support_termination_candidates(positions: Any, normals: Any, tangent_scales: Any, region_result: RegionFormationResult, *, ids: Sequence[Any] | None = None, sectors: int = 8, canonical_frames: Sequence[Any | None] | None = None, continuation: Mapping[Any, ContinuationTerminationQuery] | None = None) -> tuple[WorldSpaceBoundaryHalfEdgeCandidate, ...]:
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
        if not runs or gap < width * 1.5:
            continue

        outward = _unit(outward)
        boundary_tangent = _unit(normal.cross(outward, dim=0))
        reason = "observed_support_termination" if len(local) >= 3 else "unresolved_sampling_gap"
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
