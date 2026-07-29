from __future__ import annotations

"""Topology-local, reference-rotation-invariant support termination evidence."""

import math
from typing import Any, Sequence

from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationResult


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
    center = angles[index] + gaps[index] * 0.5
    return gaps[index], axis_u * math.cos(center) + axis_v * math.sin(center)


def extract_support_termination_candidates(positions: Any, normals: Any, tangent_scales: Any, region_result: RegionFormationResult, *, ids: Sequence[Any] | None = None, sectors: int = 8, canonical_frames: Sequence[Any | None] | None = None) -> tuple[WorldSpaceBoundaryHalfEdgeCandidate, ...]:
    """Produce at most one support-termination candidate per accepted node.

    Sector occupancy remains a robustness guard, but candidate direction comes
    from the actual circular support gap. Consequently a valid tangent-frame
    reference can rotate sector labels without changing boundary evidence.
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

