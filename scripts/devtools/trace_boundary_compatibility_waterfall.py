"""Worklog 39 (task section 11/12): boundary compatibility rejection
waterfall.

For every potential boundary candidate pair inside a region, replay the exact
gate sequence `_compatible_directed_edges` uses and record the FIRST gate that
rejects it -- with special attention to pairs that are ANALYTICALLY ADJACENT
on the true perimeter (consecutive along the physical boundary), since those
are the pairs a closed ring actually needs.

Diagnostic only. The ordering solver is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _norm(a): return max(sum(x * x for x in a) ** .5, 1e-12)


def waterfall(scene_name: str, seed: int = 0) -> dict:
    scene = make_gaussian_reliability_scene(scene_name, seed=seed)
    ids = tuple(range(scene.positions.shape[0]))
    construction = construct_visible_nurbs_from_gaussians(
        scene.positions, covariance=scene.covariances, stable_ids=ids,
    )
    candidates = [
        h for h in construction.boundary_halfedge_candidates
        if h.boundary_reason == "observed_support_termination"
    ]
    accepted_pairs = {frozenset(p) for p in construction.accepted_local_topology}

    by_region: dict[int, list] = {}
    for c in candidates:
        by_region.setdefault(c.source_region_id, []).append(c)

    region_rows = []
    for region_id, region_candidates in sorted(by_region.items()):
        nearest = [
            _norm(_sub(s.world_position, t.world_position))
            for s in region_candidates for t in region_candidates
            if s.half_edge_id != t.half_edge_id
        ]
        if not nearest:
            continue
        local_spacing = sorted(nearest)[len(nearest) // 2]
        max_distance, max_lateral = local_spacing * 1.6, local_spacing * 0.9

        # "Analytically adjacent" proxy: the pair is among each other's two
        # nearest same-region candidates. On a true perimeter ring these are
        # exactly the consecutive pairs a closed loop must connect.
        nearest_two: dict[str, set[str]] = {}
        for s in region_candidates:
            ranked = sorted(
                (t for t in region_candidates if t.half_edge_id != s.half_edge_id),
                key=lambda t: _norm(_sub(s.world_position, t.world_position)),
            )
            nearest_two[s.half_edge_id] = {t.half_edge_id for t in ranked[:2]}

        first_failure = Counter()
        adjacent_first_failure = Counter()
        adjacent_pairs = 0
        for s in region_candidates:
            for t in region_candidates:
                if s.half_edge_id == t.half_edge_id:
                    continue
                is_adjacent = t.half_edge_id in nearest_two[s.half_edge_id]
                if is_adjacent:
                    adjacent_pairs += 1

                def record(reason):
                    first_failure[reason] += 1
                    if is_adjacent:
                        adjacent_first_failure[reason] += 1

                if frozenset((s.source_gaussian_id, t.source_gaussian_id)) not in accepted_pairs:
                    record("no_direct_accepted_core_pair")
                    continue
                delta = _sub(t.world_position, s.world_position)
                distance = _norm(delta)
                forward = _dot(delta, s.boundary_direction)
                if forward <= 1e-8:
                    record("non_forward_direction")
                    continue
                if distance > max_distance:
                    record("distance_exceeds_max")
                    continue
                lateral = (max(distance * distance - forward * forward, 0.0)) ** .5
                if lateral > max_lateral:
                    record("lateral_exceeds_max")
                    continue
                tan_align = _dot(s.boundary_direction, t.boundary_direction)
                if tan_align < -.15:
                    record("tangent_misaligned")
                    continue
                normal_align = abs(_dot(s.local_normal, t.local_normal))
                if normal_align < .45:
                    record("normal_misaligned")
                    continue
                record("accepted_compatible_edge")

        region_rows.append({
            "region_id": region_id,
            "candidate_count": len(region_candidates),
            "ordered_pair_count": len(region_candidates) * (len(region_candidates) - 1),
            "analytically_adjacent_pair_count": adjacent_pairs,
            "first_failure_all_pairs": dict(first_failure),
            "first_failure_adjacent_pairs": dict(adjacent_first_failure),
        })

    return {"scene": scene_name, "regions": region_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=["box", "cylinder", "box_face"])
    args = parser.parse_args()
    for scene_name in args.scenes:
        print(json.dumps(waterfall(scene_name), indent=2, default=str))


if __name__ == "__main__":
    main()
