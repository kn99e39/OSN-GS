"""Worklog 35: frozen boundary-candidate replay + stage-by-stage decomposition
of the directed ordering failure on the post-ADC box_face positive control.

Offline-only, read/diagnose only, does not modify production code. Runs
`construct_visible_nurbs_from_gaussians` up to the halfedge/accepted-topology
stage, then decomposes `_recover_directed_boundary_components`'s internal
funnel (candidate -> same-region pair -> accepted core edge -> distance ->
lateral -> tangent -> normal -> forward-compatible -> backward-compatible ->
mutual -> greedy -> used edge -> open/closed) to classify O1-O8.
"""

from __future__ import annotations

import json
from collections import Counter

import argparse

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _norm(a): return max(sum(x * x for x in a) ** 0.5, 1e-12)


def decompose(candidates, accepted_topology):
    candidates = [c for c in candidates if c.boundary_reason == "observed_support_termination"]
    by_id = {c.half_edge_id: c for c in candidates}
    accepted_pairs = {frozenset(pair) for pair in accepted_topology}

    stage_counts = Counter()
    stage_counts["physical_boundary_candidate"] = len(candidates)

    same_region_pairs = 0
    accepted_core_pairs = 0
    distance_ok = 0
    lateral_ok = 0
    tangent_ok = 0
    normal_ok = 0
    forward_edges = {}
    for source in candidates:
        nearest = [
            _norm(_sub(source.world_position, t.world_position))
            for t in candidates if t.half_edge_id != source.half_edge_id and t.source_region_id == source.source_region_id
        ]
    nearest_all = []
    for source in candidates:
        for t in candidates:
            if t.half_edge_id != source.half_edge_id and t.source_region_id == source.source_region_id:
                nearest_all.append(_norm(_sub(source.world_position, t.world_position)))
    local_spacing = sorted(nearest_all)[len(nearest_all) // 2] if nearest_all else 1.0
    max_distance, max_lateral = local_spacing * 1.6, local_spacing * 0.9

    per_node_forward_count = Counter()
    per_node_backward_count = Counter()
    per_node_best_margin = {}

    for source in candidates:
        options = []
        for target in candidates:
            if target.half_edge_id == source.half_edge_id or target.source_region_id != source.source_region_id:
                continue
            same_region_pairs += 1
            if frozenset((source.source_gaussian_id, target.source_gaussian_id)) not in accepted_pairs:
                continue
            accepted_core_pairs += 1
            delta = _sub(target.world_position, source.world_position)
            distance = _norm(delta)
            forward = _dot(delta, source.boundary_direction)
            if forward <= 1e-8 or distance > max_distance:
                continue
            distance_ok += 1
            lateral = (max(distance * distance - forward * forward, 0.0)) ** 0.5
            if lateral > max_lateral:
                continue
            lateral_ok += 1
            tan_align = _dot(source.boundary_direction, target.boundary_direction)
            if tan_align < -0.15:
                continue
            tangent_ok += 1
            normal_align = abs(_dot(source.local_normal, target.local_normal))
            if normal_align < 0.45:
                continue
            normal_ok += 1
            outward_align = normal_align * max(tan_align, 0.0)
            score = forward / distance + tan_align + normal_align + outward_align - lateral / max_lateral
            options.append((score, str(target.source_gaussian_id), target))
        per_node_forward_count[source.half_edge_id] = len(options)
        if options:
            options.sort(key=lambda item: (-item[0], item[1], item[2].half_edge_id))
            margin = options[0][0] - options[1][0] if len(options) > 1 else float("inf")
            per_node_best_margin[source.half_edge_id] = margin
            if len(options) == 1 or margin >= 1e-6:
                forward_edges[source.half_edge_id] = options[0][2].half_edge_id

    backward_edges = {}
    for target in candidates:
        options = []
        for source in candidates:
            if source.half_edge_id == target.half_edge_id or source.source_region_id != target.source_region_id:
                continue
            if frozenset((source.source_gaussian_id, target.source_gaussian_id)) not in accepted_pairs:
                continue
            delta = _sub(source.world_position, target.world_position)
            distance = _norm(delta)
            backward = -_dot(delta, target.boundary_direction)
            if backward <= 1e-8 or distance > max_distance:
                continue
            lateral = (max(distance * distance - backward * backward, 0.0)) ** 0.5
            tangent = _dot(source.boundary_direction, target.boundary_direction)
            if lateral > max_lateral or tangent < -0.15:
                continue
            normal = abs(_dot(source.local_normal, target.local_normal))
            if normal < 0.45:
                continue
            score = backward / distance + tangent + normal + normal * max(tangent, 0.0) - lateral / max_lateral
            options.append((score, str(source.source_gaussian_id), source))
        per_node_backward_count[target.half_edge_id] = len(options)
        if options:
            options.sort(key=lambda item: (-item[0], item[1], item[2].half_edge_id))
            if len(options) == 1 or abs(options[0][0] - options[1][0]) >= 1e-6:
                backward_edges[target.half_edge_id] = options[0][2].half_edge_id

    forward_compatible_edges = len(forward_edges)
    backward_compatible_edges = len(backward_edges)
    mutual_edges = sum(1 for s, t in forward_edges.items() if backward_edges.get(t) == s)

    # O1: forward/backward disagree even though both exist
    both_exist_disagree = sum(
        1 for s, t in forward_edges.items()
        if t in by_id and backward_edges.get(t) is not None and backward_edges.get(t) != s
    )
    # O4: node has >1 viable forward or backward candidate (competing successors)
    multi_forward = sum(1 for v in per_node_forward_count.values() if v > 1)
    multi_backward = sum(1 for v in per_node_backward_count.values() if v > 1)
    # score-tie ambiguous (excluded from forward_edges despite having options)
    tie_ambiguous = sum(
        1 for k, v in per_node_forward_count.items()
        if v > 1 and k not in forward_edges
    )

    return {
        "total_genuine_candidates": len(candidates),
        "same_region_pair_count": same_region_pairs,
        "accepted_core_edge_pair_count": accepted_core_pairs,
        "distance_compatible_count": distance_ok,
        "lateral_compatible_count": lateral_ok,
        "tangent_compatible_count": tangent_ok,
        "normal_compatible_count": normal_ok,
        "forward_compatible_directed_edge_count": forward_compatible_edges,
        "backward_compatible_directed_edge_count": backward_compatible_edges,
        "mutual_agreement_edge_count": mutual_edges,
        "nodes_with_multiple_forward_candidates": multi_forward,
        "nodes_with_multiple_backward_candidates": multi_backward,
        "nodes_with_score_tie_ambiguous": tie_ambiguous,
        "forward_backward_disagree_when_both_exist": both_exist_disagree,
        "local_spacing": local_spacing,
        "max_distance": max_distance,
        "max_lateral": max_lateral,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="box_face")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cap", type=int, default=0, help="0 = no forced downsample (direct construct call)")
    args = parser.parse_args()

    scene = make_gaussian_reliability_scene(args.scene, seed=args.seed)
    stable_ids = tuple(range(scene.positions.shape[0]))

    if args.cap <= 0:
        result = construct_visible_nurbs_from_gaussians(
            scene.positions, covariance=scene.covariances, stable_ids=stable_ids,
        )
    else:
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=int(args.cap))
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        result = bundle.construction

    decomposition = decompose(result.boundary_halfedge_candidates, result.accepted_local_topology)
    output = {
        "summary": result.diagnostic_summary,
        "component_ordering_states": Counter(c.ordering_state for c in result.ordered_boundary_components).most_common(),
        "component_node_counts": [len(c.ordered_source_ids) for c in result.ordered_boundary_components],
        "decomposition": decomposition,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
