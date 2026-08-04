"""Trace real physical-boundary candidate chains on frozen checkpoints.

This diagnostic keeps production candidate generation unchanged.  It exposes
the raw-to-normalized lineage and the directed compatibility graph so a failed
loop can be attributed to candidate generation, normalization, topology, or
geometry instead of being treated as an opaque ordering failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from osn_gs.surface.torch_boundary_support_termination import (
    extract_support_termination_candidates,
    normalize_continuation_candidates,
)
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_directed_boundary_ordering import (
    _build_accepted_adjacency,
    _compatible_directed_edges,
    _decompose_into_paths_and_cycles,
    _max_weight_one_in_one_out_matching,
)
from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput, build_continuation_shells_from_input
from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions
from osn_gs.surface.torch_visible_surface_construction import _orient_normals_along_accepted_topology


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _norm(a):
    return max(sum(x * x for x in a) ** 0.5, 1e-12)


def _candidate_sets(state):
    regions = form_surface_regions(
        state.rep_points, state.rep_frame, state.reliability, state.graph,
        ids=state.rep_stable_ids,
    )
    canonical = construct_canonical_region_tangent_frames(
        state.rep_points, state.rep_frame, state.reliability, regions, ids=state.rep_stable_ids,
    )
    continuation_input = ContinuationShellInput(
        full_positions=state.full_points, full_frame=state.full_frame,
        full_intrinsic=state.full_intrinsic, full_opacity=state.full_opacity,
        full_stable_ids=state.full_stable_ids,
        nearest_representative_index=state.nearest_representative_index,
        representative_mean_spacing=state.representative_mean_spacing,
    )
    continuation = build_continuation_shells_from_input(
        continuation_input, state.rep_points, state.rep_frame, state.rep_stable_ids, regions, canonical,
    )
    accepted = tuple(sorted(
        (edge for region in regions.regions for edge in region.internal_accepted_edge_ids),
        key=lambda pair: (str(pair[0]), str(pair[1])),
    ))
    oriented_normals = _orient_normals_along_accepted_topology(
        state.rep_frame.normal_candidate, accepted, state.rep_stable_ids,
    )
    raw = extract_support_termination_candidates(
        state.rep_points, oriented_normals, state.candidate_scale, regions,
        ids=state.rep_stable_ids, sectors=8, canonical_frames=canonical,
        continuation=continuation, affinity_graph=state.graph,
    )
    return regions, continuation, raw, normalize_continuation_candidates(raw)


def _gate_reason(source, target, accepted_pairs, accepted_adjacency, candidate_ids, spacing):
    source_id, target_id = source.source_gaussian_id, target.source_gaussian_id
    direct = frozenset((source_id, target_id)) in accepted_pairs
    shared = set(accepted_adjacency.get(source_id, ())) & set(accepted_adjacency.get(target_id, ()))
    if not direct and not (shared - candidate_ids):
        return "topology_support_missing"
    delta = _sub(target.world_position, source.world_position)
    distance = _norm(delta)
    forward = sum(a * b for a, b in zip(delta, source.boundary_direction))
    if forward <= 1e-8:
        return "non_forward_direction"
    if distance > spacing * 1.6:
        return "distance_exceeds_local_scale"
    lateral = max(distance * distance - forward * forward, 0.0) ** 0.5
    if lateral > spacing * 0.9:
        return "lateral_geometry_incompatible"
    normal = abs(sum(a * b for a, b in zip(source.local_normal, target.local_normal)))
    if normal < .45:
        return "normal_geometry_incompatible"
    return "compatible"


def _region_trace(region_id, candidates, accepted_pairs, accepted_adjacency):
    if len(candidates) < 2:
        return {"candidate_count": len(candidates), "classification": "physical_candidate_not_generated"}
    by_id = {candidate.half_edge_id: candidate for candidate in candidates}
    distances = sorted(
        _norm(_sub(left.world_position, right.world_position))
        for left in candidates for right in candidates if left.half_edge_id != right.half_edge_id
    )
    spacing = distances[len(distances) // 2]
    candidate_ids = frozenset(candidate.source_gaussian_id for candidate in candidates)
    edges = _compatible_directed_edges(candidates, accepted_pairs, spacing, accepted_adjacency, candidate_ids)
    out_degree = Counter(source for source, _ in edges)
    in_degree = Counter(target for _, target in edges)
    matched = _max_weight_one_in_one_out_matching(sorted(by_id), edges)
    cycles, paths, isolated = _decompose_into_paths_and_cycles(matched, sorted(by_id))

    # The closest spatial neighbour is the least-assumptive local perimeter
    # proxy available without filling a gap.  Its first failed direction is
    # recorded for every open-chain endpoint.
    endpoint_rows = []
    endpoint_ids = sorted(set(by_id) - set(matched)) + sorted(set(by_id) - set(matched.values()))
    for endpoint_id in dict.fromkeys(endpoint_ids):
        source = by_id[endpoint_id]
        nearest = min(
            (candidate for candidate in candidates if candidate.half_edge_id != endpoint_id),
            key=lambda candidate: _norm(_sub(source.world_position, candidate.world_position)),
        )
        forward_reason = _gate_reason(source, nearest, accepted_pairs, accepted_adjacency, candidate_ids, spacing)
        reverse_reason = _gate_reason(nearest, source, accepted_pairs, accepted_adjacency, candidate_ids, spacing)
        endpoint_rows.append({
            "source_id": source.source_gaussian_id,
            "half_edge_id": endpoint_id,
            "out_degree": out_degree[endpoint_id],
            "in_degree": in_degree[endpoint_id],
            "nearest_source_id": nearest.source_gaussian_id,
            "nearest_distance_over_spacing": round(_norm(_sub(source.world_position, nearest.world_position)) / spacing, 3),
            "first_gate": forward_reason if forward_reason != "non_forward_direction" else reverse_reason,
        })

    degree_histogram = Counter((in_degree[c.half_edge_id], out_degree[c.half_edge_id]) for c in candidates)
    chain_lengths = sorted([len(chain) for chain in cycles], reverse=True) + sorted([len(chain) for chain in paths], reverse=True)
    maximum_missing_interval = max((row["nearest_distance_over_spacing"] for row in endpoint_rows), default=0.0)
    classification = (
        "closed" if cycles else
        "topology_support_missing" if endpoint_rows and all(row["first_gate"] == "topology_support_missing" for row in endpoint_rows) else
        "geometry_compatibility_error" if endpoint_rows and all("geometry" in row["first_gate"] or "scale" in row["first_gate"] for row in endpoint_rows) else
        "fragmented_physical_candidates"
    )
    return {
        "candidate_count": len(candidates),
        "local_spacing": round(spacing, 6),
        "compatible_directed_edges": len(edges),
        "compatibility_degree_histogram": {f"in{key[0]}_out{key[1]}": value for key, value in sorted(degree_histogram.items())},
        "cycles": [list(chain) for chain in cycles],
        "open_chains": [list(chain) for chain in paths],
        "isolated_half_edge_ids": isolated,
        "open_chain_endpoints": endpoint_rows,
        "maximum_missing_interval_over_spacing": maximum_missing_interval,
        "classification": classification,
    }


def trace(checkpoint: Path, cap: int) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state

    state = build_frozen_state(checkpoint, cap)
    regions, continuation, raw, normalized = _candidate_sets(state)
    raw_physical = [item for item in raw if item.boundary_reason == "observed_support_termination"]
    physical = [item for item in normalized if item.boundary_reason == "observed_support_termination"]
    raw_ids = {item.half_edge_id for item in raw_physical}
    normalized_ids = {item.half_edge_id for item in physical}
    accepted = tuple(edge for region in regions.regions for edge in region.internal_accepted_edge_ids)
    accepted_pairs = {frozenset(pair) for pair in accepted}
    accepted_adjacency = _build_accepted_adjacency(accepted)
    by_region = defaultdict(list)
    raw_by_region = defaultdict(list)
    for candidate in raw_physical:
        raw_by_region[candidate.source_region_id].append(candidate)
    for candidate in physical:
        by_region[candidate.source_region_id].append(candidate)
    raw_traces = {
        region_id: _region_trace(region_id, candidates, accepted_pairs, accepted_adjacency)
        for region_id, candidates in raw_by_region.items()
    }
    traces = {
        region_id: _region_trace(region_id, candidates, accepted_pairs, accepted_adjacency)
        for region_id, candidates in by_region.items()
    }
    major = sorted(traces, key=lambda region_id: (-traces[region_id]["candidate_count"], region_id))[:12]
    surface_major = []
    for region in sorted(regions.regions, key=lambda item: (-len(item.member_ids), item.region_id))[:12]:
        states = Counter(continuation[node_id].state for node_id in region.member_ids if node_id in continuation)
        surface_major.append({
            "region_id": region.region_id,
            "member_count": len(region.member_ids),
            "continuation_state_counts": dict(states),
            "raw_reason_counts": dict(Counter(item.boundary_reason for item in raw if item.source_region_id == region.region_id)),
            "physical_normalized_count": len(by_region.get(region.region_id, ())),
        })
    evidence_rows = []
    id_index = {node_id: index for index, node_id in enumerate(state.rep_stable_ids)}
    for region in sorted(regions.regions, key=lambda item: (-len(item.member_ids), item.region_id))[:12]:
        for node_id in region.member_ids:
            query = continuation.get(node_id)
            if query is None or query.state not in ("no_gap", "parallel_sheet_conflict", "crease_discontinuity", "ambiguous_continuation"):
                continue
            node_index = id_index[node_id]
            incident = [edge for edge in state.graph.edges if edge.source_id == node_id or edge.target_id == node_id]
            relations = Counter(edge.manifold_relation for edge in incident)
            residuals = [edge.mutual_tangent_residual for edge in incident if edge.metrics is not None]
            evidence_rows.append({
                "region_id": region.region_id, "source_stable_id": node_id, "state": query.state,
                "world_position": [round(float(value), 6) for value in state.rep_points[node_index]],
                "normal": [round(float(value), 6) for value in state.rep_frame.normal_candidate[node_index]],
                "outward_arc": None if query.outward_direction is None else [round(float(value), 6) for value in query.outward_direction],
                "support_radius": round(query.support_radius, 6), "same_mode_support_count": query.same_mode_support_count,
                "same_mode_opacity_mass": round(query.same_mode_opacity_mass, 6), "competing_mode_mass": round(query.competing_mode_mass, 6),
                "ambiguous_mass": round(query.ambiguous_continuation_mass, 6), "supporting_stable_ids": list(query.source_full_cloud_fingerprint),
                "bounded_affinity_relation_counts": dict(relations),
                "bounded_affinity_residual_range": None if not residuals else [round(min(residuals), 6), round(max(residuals), 6)],
            })
    return {
        "checkpoint": str(checkpoint),
        "physical_raw": len(raw_physical),
        "physical_normalized": len(physical),
        "physical_lost_by_normalization": len(raw_ids - normalized_ids),
        "raw_closed_regions": sorted(region_id for region_id, item in raw_traces.items() if item.get("cycles")),
        "normalized_closed_regions": sorted(region_id for region_id, item in traces.items() if item.get("cycles")),
        "normalization_loss_by_region": {str(region_id): len(items) - len(by_region.get(region_id, ())) for region_id, items in raw_by_region.items() if len(items) != len(by_region.get(region_id, ()))},
        "raw_reason_counts": dict(Counter(item.boundary_reason for item in raw)),
        "continuation_state_counts": dict(Counter(item.state for item in continuation.values())),
        "major_regions": {str(region_id): traces[region_id] for region_id in major},
        "major_surface_regions": surface_major,
        "boundary_proximate_evidence_rows": evidence_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps({path.stem + "_" + path.parent.name: trace(path, args.cap) for path in args.checkpoints}, indent=2))


if __name__ == "__main__":
    main()
