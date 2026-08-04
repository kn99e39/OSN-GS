"""Worklog 52: exact audit of why a specific region's directed one-in/one-out
matching does not close a loop -- full compatible-edge list, per-edge score
components, the matching's actual assignment, and an exhaustive check of
whether ANY feasible cyclic assignment exists and, if so, its total score
against the matching's chosen (possibly non-cyclic) assignment.

Reuses production `_compatible_directed_edges` / `_max_weight_one_in_one_out_matching`
/ `_decompose_into_paths_and_cycles` unchanged -- this only adds bookkeeping
and a small-region exhaustive cycle search (region sizes here are <=6, so
brute-force permutation search over feasible edges is cheap and exact).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path


def _sid(half_edge_id: str) -> str:
    return half_edge_id.split(":")[3]


def _best_cycle(node_ids, edges):
    """Exhaustive search (region is tiny) for the highest-total-score simple
    cycle (length >= 3) using only feasible edges. Returns (score, cycle) or
    (None, None) if no cycle of length >= 3 exists at all."""
    best = (None, None)
    n = len(node_ids)
    for length in range(3, n + 1):
        for subset in itertools.combinations(node_ids, length):
            for perm in itertools.permutations(subset):
                total = 0.0
                ok = True
                for i in range(length):
                    pair = (perm[i], perm[(i + 1) % length])
                    if pair not in edges:
                        ok = False
                        break
                    total += edges[pair].score
                if ok and (best[0] is None or total > best[0]):
                    best = (total, list(perm))
    return best


def trace(checkpoint: Path, cap: int, region_ids: list[int]) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state

    from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
    from osn_gs.surface.torch_directed_boundary_ordering import (
        _build_accepted_adjacency, _compatible_directed_edges, _decompose_into_paths_and_cycles,
        _max_weight_one_in_one_out_matching,
    )
    from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput, build_continuation_shells_from_input
    from osn_gs.surface.torch_gaussian_surface_region_formation import form_surface_regions
    from osn_gs.surface.torch_visible_surface_construction import _orient_normals_along_accepted_topology
    from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates, normalize_continuation_candidates

    state = build_frozen_state(checkpoint, cap)
    regions = form_surface_regions(state.rep_points, state.rep_frame, state.reliability, state.graph, ids=state.rep_stable_ids)
    canonical = construct_canonical_region_tangent_frames(state.rep_points, state.rep_frame, state.reliability, regions, ids=state.rep_stable_ids)
    continuation_input = ContinuationShellInput(
        full_positions=state.full_points, full_frame=state.full_frame, full_intrinsic=state.full_intrinsic,
        full_opacity=state.full_opacity, full_stable_ids=state.full_stable_ids,
        nearest_representative_index=state.nearest_representative_index, representative_mean_spacing=state.representative_mean_spacing,
    )
    continuation = build_continuation_shells_from_input(continuation_input, state.rep_points, state.rep_frame, state.rep_stable_ids, regions, canonical)
    accepted = tuple(sorted((edge for region in regions.regions for edge in region.internal_accepted_edge_ids), key=lambda pair: (str(pair[0]), str(pair[1]))))
    oriented_normals = _orient_normals_along_accepted_topology(state.rep_frame.normal_candidate, accepted, state.rep_stable_ids)
    raw = extract_support_termination_candidates(state.rep_points, oriented_normals, state.candidate_scale, regions, ids=state.rep_stable_ids, sectors=8, canonical_frames=canonical, continuation=continuation, affinity_graph=state.graph)
    normalized = normalize_continuation_candidates(raw)
    physical = [c for c in normalized if c.boundary_reason == "observed_support_termination"]
    accepted_pairs = {frozenset(pair) for pair in accepted}
    accepted_adjacency = _build_accepted_adjacency(accepted)

    by_region = defaultdict(list)
    for candidate in physical:
        by_region[candidate.source_region_id].append(candidate)

    results = {}
    for region_id in region_ids:
        candidates = by_region.get(region_id, [])
        if not candidates:
            results[region_id] = {"error": "no physical candidates in this region under current state"}
            continue
        nearest = [
            ((a.world_position[0] - b.world_position[0]) ** 2 + (a.world_position[1] - b.world_position[1]) ** 2 + (a.world_position[2] - b.world_position[2]) ** 2) ** 0.5
            for a in candidates for b in candidates if a.half_edge_id != b.half_edge_id
        ]
        local_spacing = sorted(nearest)[len(nearest) // 2] if nearest else 1.0
        boundary_ids = frozenset(c.source_gaussian_id for c in candidates)
        edges = _compatible_directed_edges(candidates, accepted_pairs, local_spacing, accepted_adjacency, boundary_ids)
        node_ids = sorted(c.half_edge_id for c in candidates)
        matched = _max_weight_one_in_one_out_matching(node_ids, edges)
        cycles, paths, isolated = _decompose_into_paths_and_cycles(matched, node_ids)
        matched_score = sum(edges[(source, target)].score for source, target in matched.items())
        best_cycle_score, best_cycle = _best_cycle(node_ids, edges)

        results[region_id] = {
            "candidate_stable_ids": [c.source_gaussian_id for c in candidates],
            "edges": [
                {
                    "source": _sid(s), "target": _sid(t), "score": round(e.score, 4),
                    "forward_distance": round(e.forward_distance, 4), "lateral_residual": round(e.lateral_residual, 4),
                    "normalized_distance": round(e.normalized_distance, 4), "tangent_alignment": round(e.tangent_alignment, 4),
                    "normal_alignment": round(e.normal_alignment, 4),
                }
                for (s, t), e in edges.items()
            ],
            "matched_assignment": [{"source": _sid(s), "target": _sid(t), "score": round(edges[(s, t)].score, 4)} for s, t in matched.items()],
            "matched_total_score": round(matched_score, 4),
            "cycles": [[_sid(x) for x in c] for c in cycles],
            "paths": [[_sid(x) for x in p] for p in paths],
            "isolated": [_sid(x) for x in isolated],
            "best_feasible_cycle_score": round(best_cycle_score, 4) if best_cycle_score is not None else None,
            "best_feasible_cycle": [_sid(x) for x in best_cycle] if best_cycle else None,
            "cycle_score_deficit_vs_matched": (round(matched_score - best_cycle_score, 4) if best_cycle_score is not None else None),
        }
    return {"checkpoint": str(checkpoint), "regions": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--regions", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(trace(args.checkpoint, args.cap, args.regions), indent=2))


if __name__ == "__main__":
    main()
