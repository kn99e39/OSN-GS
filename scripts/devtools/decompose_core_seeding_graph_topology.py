"""Worklog 37 (task sections 5-6): decompose the same_surface graph into
raw / bridge-veto-applied / final-seeded-core states, and classify every R2
ambiguous_unassigned node into a precise sub-category.

Uses the frozen core-seeding replay harness (no re-selection/re-affinity).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from osn_gs.surface.torch_gaussian_manifold_affinity import RELATION_SAME_SURFACE
from osn_gs.surface.torch_gaussian_structural_reliability import INTRINSIC_RELIABLE
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    BRIDGE_WEAK_CANDIDATE,
    BRIDGE_CONTRADICTED,
    CONSENSUS_CONTRADICTED,
    PATH_PHASE_ALIAS,
    RegionFormationConfig,
    _build_relation_adjacency,
    _seed_core_components,
)

from frozen_core_seeding_replay import build_frozen_state, replay_region_formation


def _connected_components(n: int, adjacency: list[set[int]]) -> list[list[int]]:
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start] or not adjacency[start]:
            continue
        stack = [start]
        visited[start] = True
        members = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(members)
    # Also record true singletons (degree 0) as their own components for coverage math.
    for i in range(n):
        if not visited[i]:
            components.append([i])
            visited[i] = True
    return components


def _degree_histogram(n: int, adjacency: list[set[int]]) -> dict:
    degrees = [len(adjacency[i]) for i in range(n)]
    return {
        "degree_0": sum(1 for d in degrees if d == 0),
        "degree_1": sum(1 for d in degrees if d == 1),
        "degree_2": sum(1 for d in degrees if d == 2),
        "degree_3_plus": sum(1 for d in degrees if d >= 3),
    }


def _component_stats(components: list[list[int]]) -> dict:
    sizes = sorted((len(c) for c in components), reverse=True)
    real_sizes = [s for s in sizes if s > 1]  # exclude pure isolated singletons from median/p90
    return {
        "component_count": len(components),
        "singleton_count": sum(1 for s in sizes if s == 1),
        "component_size_median": (real_sizes[len(real_sizes) // 2] if real_sizes else 0),
        "component_size_p90": (sorted(real_sizes)[int(0.9 * len(real_sizes))] if real_sizes else 0),
        "component_size_max": max(sizes) if sizes else 0,
        "largest_component_coverage": (max(sizes) / sum(sizes)) if sizes else 0.0,
    }


def _has_cycle(members: list[int], adjacency: list[set[int]]) -> bool:
    member_set = set(members)
    edge_count = sum(len(adjacency[m] & member_set) for m in members) // 2
    return edge_count >= len(members)  # tree has exactly n-1 edges; >=n means a cycle exists


def decompose(checkpoint: Path, cap: int) -> dict:
    state = build_frozen_state(checkpoint, cap)
    config = RegionFormationConfig()
    count = int(state.reliability.intrinsic.conditioning_score.shape[0])

    same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, state.graph)

    # --- Raw same_surface graph (restricted to intrinsic-reliable nodes, matching seed eligibility scope) ---
    intrinsic_class = state.reliability.intrinsic.intrinsic_class
    reliable_same_surface = [
        (same_surface[n] if intrinsic_class[n] == INTRINSIC_RELIABLE else set())
        for n in range(count)
    ]
    # Only keep edges where BOTH endpoints are intrinsic-reliable.
    raw_adjacency = [set() for _ in range(count)]
    for n in range(count):
        if intrinsic_class[n] != INTRINSIC_RELIABLE:
            continue
        for nb in same_surface[n]:
            if intrinsic_class[nb] == INTRINSIC_RELIABLE:
                raw_adjacency[n].add(nb)

    raw_components = _connected_components(count, raw_adjacency)
    raw_edge_count = sum(len(raw_adjacency[n]) for n in range(count)) // 2
    raw_degree_hist = _degree_histogram(count, raw_adjacency)
    raw_stats = _component_stats(raw_components)
    raw_cycle_components = sum(1 for c in raw_components if len(c) > 1 and _has_cycle(c, raw_adjacency))
    raw_tree_components = sum(1 for c in raw_components if len(c) > 1) - raw_cycle_components

    # --- Bridge-veto-applied graph: run _seed_core_components and see which edges got vetoed ---
    uf, consensus_by_pair, bridge_by_pair, path_by_pair, boundary_conflict_edges = _seed_core_components(
        count, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, state.reliability, state.rep_frame, config,
    )

    veto_adjacency = [set() for _ in range(count)]
    removed_reason_counts = Counter()
    for n in range(count):
        for nb in raw_adjacency[n]:
            if n >= nb:
                continue
            key = (n, nb)
            if key in boundary_conflict_edges:
                consensus = consensus_by_pair.get(key)
                path_result = path_by_pair.get(key)
                bridge = bridge_by_pair.get(key)
                if consensus is not None and consensus.consensus_state == CONSENSUS_CONTRADICTED:
                    removed_reason_counts["consensus_contradicted"] += 1
                elif path_result is not None and path_result.path_status == PATH_PHASE_ALIAS:
                    removed_reason_counts["phase_alias"] += 1
                elif bridge is not None and bridge.bridge_state == BRIDGE_WEAK_CANDIDATE:
                    if "too_few_shared_same_surface_neighbors_for_confident_bridge" in bridge.reasons:
                        removed_reason_counts["too_few_shared_neighbor"] += 1
                    elif "local_tangent_frame_divergence_between_endpoints_neighborhoods" in bridge.reasons:
                        removed_reason_counts["tangent_frame_divergence"] += 1
                    elif "removing_edge_splits_local_neighborhood_into_two_large_components" in bridge.reasons:
                        removed_reason_counts["local_cut_splits"] += 1
                    else:
                        removed_reason_counts["weak_bridge_other"] += 1
                elif bridge is not None and bridge.bridge_state == BRIDGE_CONTRADICTED:
                    removed_reason_counts["bridge_contradicted"] += 1
                else:
                    removed_reason_counts["oversized_footprint_or_other"] += 1
                continue
            veto_adjacency[n].add(nb)
            veto_adjacency[nb].add(n)

    veto_components = _connected_components(count, veto_adjacency)
    veto_edge_count = sum(len(veto_adjacency[n]) for n in range(count)) // 2
    veto_stats = _component_stats(veto_components)

    # --- Final seeded core graph: from union-find result ---
    core_components_map: dict[int, list[int]] = {}
    for n in range(count):
        core_components_map.setdefault(uf.find(n), []).append(n)
    core_component_sizes = sorted((len(v) for v in core_components_map.values() if len(v) > 1), reverse=True)
    core_degree = [0] * count
    for (a, b), consensus in consensus_by_pair.items():
        if consensus.consensus_state != CONSENSUS_CONTRADICTED and uf.find(a) == uf.find(b):
            core_degree[a] += 1
            core_degree[b] += 1
    core_seed_count = sum(
        1 for n in range(count)
        if intrinsic_class[n] == INTRINSIC_RELIABLE and core_degree[n] >= config.core_min_same_surface_degree
    )

    return {
        "checkpoint": str(checkpoint),
        "cap": cap,
        "representative_count": count,
        "raw_graph": {
            "node_count_intrinsic_reliable": sum(1 for n in range(count) if intrinsic_class[n] == INTRINSIC_RELIABLE),
            "edge_count": raw_edge_count,
            "degree_histogram": raw_degree_hist,
            **raw_stats,
            "cycle_containing_components": raw_cycle_components,
            "tree_components": raw_tree_components,
        },
        "bridge_veto_graph": {
            "edge_count": veto_edge_count,
            "removed_edge_count": raw_edge_count - veto_edge_count,
            **veto_stats,
            "removed_edge_reason_counts": dict(removed_reason_counts),
        },
        "final_seeded_core_graph": {
            "core_seed_count": core_seed_count,
            "core_component_count": len(core_component_sizes),
            "core_component_size_median": core_component_sizes[len(core_component_sizes) // 2] if core_component_sizes else 0,
            "core_component_size_max": max(core_component_sizes) if core_component_sizes else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    result = decompose(args.checkpoint, args.cap)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
