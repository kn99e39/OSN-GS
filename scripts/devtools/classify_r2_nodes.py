"""Worklog 37 (task section 6): full R2 node classification.

R2 = ambiguous_unassigned node with same_surface degree>0 but none of its
same_surface neighbors are themselves in any region. Classify each into a
precise sub-category using raw graph topology + the actual veto reasons that
apply to its edges.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from osn_gs.surface.torch_gaussian_structural_reliability import INTRINSIC_RELIABLE
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    BRIDGE_WEAK_CANDIDATE,
    CONSENSUS_CONTRADICTED,
    PATH_PHASE_ALIAS,
    RegionFormationConfig,
    _build_relation_adjacency,
    _seed_core_components,
    _pair_key,
    form_surface_regions,
)

from frozen_core_seeding_replay import build_frozen_state


def _connected_component_of(start: int, adjacency: list[set[int]]) -> set[int]:
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def classify(checkpoint: Path, cap: int) -> dict:
    state = build_frozen_state(checkpoint, cap)
    config = RegionFormationConfig()
    count = int(state.reliability.intrinsic.conditioning_score.shape[0])
    same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, state.graph)
    intrinsic_class = state.reliability.intrinsic.intrinsic_class

    regions = form_surface_regions(state.rep_points, state.rep_frame, state.reliability, state.graph, config=config, ids=state.rep_stable_ids)
    node_region_id = regions.node_region_id
    node_membership_state = regions.node_membership_state

    uf, consensus_by_pair, bridge_by_pair, path_by_pair, boundary_conflict_edges = _seed_core_components(
        count, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, state.reliability, state.rep_frame, config,
    )

    raw_adjacency = [set() for _ in range(count)]
    for n in range(count):
        if intrinsic_class[n] != INTRINSIC_RELIABLE:
            continue
        for nb in same_surface[n]:
            if intrinsic_class[nb] == INTRINSIC_RELIABLE:
                raw_adjacency[n].add(nb)

    category_counts = Counter()
    category_component_sizes: dict[str, list[int]] = {}
    r2_count = 0

    for n in range(count):
        if node_membership_state[n] != "ambiguous_unassigned":
            continue
        same_surface_degree = len(same_surface[n])
        if same_surface_degree == 0:
            continue  # R1, not R2
        neighbor_regions = {node_region_id[nb] for nb in same_surface[n] if node_region_id[nb] >= 0}
        if neighbor_regions:
            continue  # R3/R4, not R2
        r2_count += 1

        component = _connected_component_of(n, raw_adjacency) if n in set(raw_adjacency[n]) or raw_adjacency[n] else {n}
        # Even isolated-in-raw-adjacency nodes (degree 0 in the RELIABLE-only
        # raw graph) can still show up here if their same_surface neighbor is
        # reliability-ineligible -- handle that case explicitly below.

        if not raw_adjacency[n] and same_surface[n]:
            category = "unseeded_component_reliability_ineligible"
        else:
            # Check every same_surface edge from n for its veto reason.
            edge_reasons = []
            for nb in same_surface[n]:
                key = _pair_key(n, nb)
                if key not in by_pair or by_pair[key].manifold_relation != "same_surface":
                    continue
                if key in boundary_conflict_edges:
                    consensus = consensus_by_pair.get(key)
                    path_result = path_by_pair.get(key)
                    bridge = bridge_by_pair.get(key)
                    if consensus is not None and consensus.consensus_state == CONSENSUS_CONTRADICTED:
                        edge_reasons.append("consensus")
                    elif path_result is not None and path_result.path_status == PATH_PHASE_ALIAS:
                        edge_reasons.append("path")
                    elif bridge is not None and bridge.bridge_state == BRIDGE_WEAK_CANDIDATE:
                        edge_reasons.append("weak_bridge")
                    else:
                        edge_reasons.append("parallel_or_other")
                else:
                    edge_reasons.append("not_core_eligible_or_unchecked")

            member_set = component
            has_cycle = sum(len(raw_adjacency[m] & member_set) for m in member_set) // 2 >= len(member_set)

            if not edge_reasons:
                category = "unseeded_component_no_core_eligible_edge"
            elif all(r == "weak_bridge" for r in edge_reasons):
                category = "unseeded_component_weak_bridge_only"
            elif "consensus" in edge_reasons:
                category = "unseeded_component_path_or_consensus_veto"
            elif "path" in edge_reasons:
                category = "unseeded_component_path_or_consensus_veto"
            elif "parallel_or_other" in edge_reasons:
                category = "unseeded_component_parallel_veto"
            elif has_cycle:
                category = "unseeded_component_cycle_but_seed_rejected"
            else:
                category = "unseeded_component_tree_without_supported_core"

        category_counts[category] += 1
        category_component_sizes.setdefault(category, []).append(len(component))

    category_summary = {}
    for category, sizes in category_component_sizes.items():
        sizes_sorted = sorted(sizes)
        category_summary[category] = {
            "node_count": category_counts[category],
            "component_size_median": sizes_sorted[len(sizes_sorted) // 2] if sizes_sorted else 0,
            "component_size_max": max(sizes_sorted) if sizes_sorted else 0,
        }

    return {
        "checkpoint": str(checkpoint),
        "cap": cap,
        "r2_total_count": r2_count,
        "category_counts": dict(category_counts),
        "category_summary": category_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    result = classify(args.checkpoint, args.cap)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
