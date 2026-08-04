"""Worklog 34: instrumented replay of `_recover_directed_boundary_components`'s
forward-successor search, to classify precisely why each genuine boundary
termination candidate fails to get a successor edge (read-only, does not
modify torch_directed_boundary_ordering.py -- reproduces its exact logic
with per-candidate failure-reason tracking added).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def _sh_degree_from_checkpoint(raw: dict) -> int:
    rest_dim = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim:
        degree += 1
    return degree


def load_model(checkpoint_path: Path, device: str) -> TorchGaussianModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw = payload["model_raw"]
    model = TorchGaussianModel(sh_degree=_sh_degree_from_checkpoint(raw), device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"),
        surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    return model


def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _norm(a): return max(sum(x * x for x in a) ** 0.5, 1e-12)


def instrumented_forward_search(candidates, accepted_topology):
    """Exact reproduction of `_recover_directed_boundary_components`'s forward
    pass (torch_directed_boundary_ordering.py), with a failure-reason
    classification added per (source, target) pair and per source node."""
    candidates = [c for c in candidates if c.boundary_reason == "observed_support_termination"]
    nearest = []
    for source in candidates:
        distances = [_norm(_sub(source.world_position, target.world_position)) for target in candidates if target.half_edge_id != source.half_edge_id and target.source_region_id == source.source_region_id]
        if distances:
            nearest.append(min(distances))
    local_spacing = sorted(nearest)[len(nearest) // 2] if nearest else 1.0
    max_distance, max_lateral = local_spacing * 1.6, local_spacing * 0.9
    accepted_pairs = {frozenset(pair) for pair in accepted_topology}

    per_node_failure = {}
    pair_failure_counts = Counter()
    node_has_any_same_region_partner = Counter()
    for source in candidates:
        options = []
        reasons_this_source = Counter()
        same_region_targets = 0
        for target in candidates:
            if target.half_edge_id == source.half_edge_id:
                continue
            if target.source_region_id != source.source_region_id:
                continue
            same_region_targets += 1
            if frozenset((source.source_gaussian_id, target.source_gaussian_id)) not in accepted_pairs:
                reasons_this_source["not_accepted_core_edge"] += 1
                pair_failure_counts["not_accepted_core_edge"] += 1
                continue
            delta = _sub(target.world_position, source.world_position)
            distance = _norm(delta)
            tangent = source.boundary_direction
            forward = _dot(delta, tangent)
            if forward <= 1e-8:
                reasons_this_source["non_forward_direction"] += 1
                pair_failure_counts["non_forward_direction"] += 1
                continue
            if distance > max_distance:
                reasons_this_source["distance_exceeds_max"] += 1
                pair_failure_counts["distance_exceeds_max"] += 1
                continue
            lateral = (max(distance * distance - forward * forward, 0.0)) ** 0.5
            tan_align = _dot(source.boundary_direction, target.boundary_direction)
            normal_align = abs(_dot(source.local_normal, target.local_normal))
            if lateral > max_lateral:
                reasons_this_source["lateral_exceeds_max"] += 1
                pair_failure_counts["lateral_exceeds_max"] += 1
                continue
            if tan_align < -0.15:
                reasons_this_source["tangent_misaligned"] += 1
                pair_failure_counts["tangent_misaligned"] += 1
                continue
            if normal_align < 0.45:
                reasons_this_source["normal_misaligned"] += 1
                pair_failure_counts["normal_misaligned"] += 1
                continue
            outward_align = normal_align * max(tan_align, 0.0)
            score = forward / distance + tan_align + normal_align + outward_align - lateral / max_lateral
            options.append((score, str(target.source_gaussian_id), target))
        node_has_any_same_region_partner[source.half_edge_id] = same_region_targets
        if not options:
            if reasons_this_source:
                dominant = reasons_this_source.most_common(1)[0][0]
            else:
                dominant = "no_same_region_partner_at_all"
            per_node_failure[source.half_edge_id] = dominant
            continue
        options.sort(key=lambda item: (-item[0], item[1], item[2].half_edge_id))
        if len(options) > 1 and abs(options[0][0] - options[1][0]) < 1e-6:
            per_node_failure[source.half_edge_id] = "score_tie_ambiguous"
            continue
        per_node_failure[source.half_edge_id] = "succeeded"

    return {
        "local_spacing": local_spacing,
        "max_distance": max_distance,
        "max_lateral": max_lateral,
        "total_candidates": len(candidates),
        "per_node_failure_distribution": dict(Counter(per_node_failure.values())),
        "pair_level_gate_failure_counts": dict(pair_failure_counts),
        "nodes_with_zero_same_region_partners": sum(1 for v in node_has_any_same_region_partner.values() if v == 0),
        "same_region_partner_count_distribution_median": (
            sorted(node_has_any_same_region_partner.values())[len(node_has_any_same_region_partner) // 2]
            if node_has_any_same_region_partner else 0
        ),
    }


def trace(checkpoint_path: Path, cap: int, device: str) -> dict:
    model = load_model(checkpoint_path, device)
    config = TorchPipelineConfig(canonical_construction_max_points=int(cap))
    pipeline = TorchOSNGSPipeline(config, device=device)
    with torch.no_grad():
        eligible_mask = (~model.is_uncertain) & (model.surface_owner_kind != 2)
        eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
        points = model.get_xyz.detach()[eligible_indices]
        activated_scale = model.get_scaling.detach()[eligible_indices]
        normalized_rotation = model.get_rotation.detach()[eligible_indices]
        covariance = covariance_from_scale_rotation(activated_scale, normalized_rotation)
        opacity = model.get_opacity.detach()[eligible_indices, 0]
        stable_ids = tuple(int(v) for v in model.stable_gaussian_ids[eligible_indices].detach().cpu().tolist())
        bundle = pipeline._construct_canonical_with_full_evidence(points, covariance, opacity, stable_ids)

    construction = bundle.construction
    halfedges = construction.boundary_halfedge_candidates
    accepted = construction.accepted_local_topology
    result = instrumented_forward_search(halfedges, accepted)
    result["region_count"] = construction.diagnostic_summary["region_count"]
    result["genuine_termination_candidate_count"] = construction.diagnostic_summary["boundary_genuine_termination_candidate_count"]
    result["accepted_local_topology_edge_count"] = len(accepted)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = trace(args.checkpoint, args.cap, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
