"""Worklog 35: for each cross-component pair with >=2 same_surface cross
edges, check ALL edges (not just the first-sampled one) to see whether the
processing ORDER in `_seed_core_components` -- which evaluates edges strictly
in priority order and immediately unions on the FIRST well-supported bridge,
STOPPING further consideration once components merge -- is masking edges that
would individually have passed the bridge veto. This directly tests R4 (does
G1 evidence exist but fail to reach merge evaluation) and R6 (residual
mergeable components after growth)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_gaussian_manifold_affinity import RELATION_SAME_SURFACE
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    RegionFormationConfig,
    _build_relation_adjacency,
    _compute_edge_consensus,
    _evaluate_bridge_veto,
    _evaluate_path_consistency,
    _seed_core_components,
    _pair_key,
    BRIDGE_WELL_SUPPORTED,
    CONSENSUS_CONTRADICTED,
    PATH_PHASE_ALIAS,
)


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
    graph = construction.manifold_affinity
    reliability = construction.reliability
    frame = construction.covariance_frame
    count = int(reliability.intrinsic.conditioning_score.shape[0])
    config_rf = RegionFormationConfig()

    same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, graph)
    uf, consensus_by_pair, bridge_by_pair, path_by_pair, boundary_conflict_edges = _seed_core_components(
        count, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, reliability, frame, config_rf,
    )

    components: dict[int, list[int]] = {}
    for node in range(count):
        components.setdefault(uf.find(node), []).append(node)

    cross_component_same_surface_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b), edge in by_pair.items():
        if edge.manifold_relation != RELATION_SAME_SURFACE:
            continue
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        cross_component_same_surface_edges.setdefault(_pair_key(ra, rb), []).append((a, b))

    # For every pair with >=2 cross edges still separate at the end, RE-EVALUATE
    # every individual edge's bridge veto against the FINAL component membership
    # (post-hoc, diagnostic only) to see if ANY edge in the pair would have
    # passed if it had been the one evaluated (order-dependence check).
    any_edge_would_pass_but_none_did = 0
    all_edges_genuinely_weak = 0
    total_multi_edge_pairs = 0
    order_dependence_examples = []
    for (ra, rb), edges in cross_component_same_surface_edges.items():
        if len(edges) < 2:
            continue
        total_multi_edge_pairs += 1
        any_would_pass = False
        edge_details = []
        for (a, b) in edges:
            consensus = _compute_edge_consensus(a, b, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, reliability, frame, config_rf)
            if consensus.consensus_state == CONSENSUS_CONTRADICTED:
                edge_details.append({"pair": (a, b), "consensus": "contradicted"})
                continue
            path_result = _evaluate_path_consistency(a, b, same_surface, frame, config_rf)
            if path_result.path_status == PATH_PHASE_ALIAS:
                edge_details.append({"pair": (a, b), "consensus": consensus.consensus_state, "path": "phase_alias"})
                continue
            bridge = _evaluate_bridge_veto(a, b, consensus, same_surface, frame, len(edges), config_rf)
            edge_details.append({"pair": (a, b), "consensus": consensus.consensus_state, "bridge": bridge.bridge_state, "reasons": bridge.reasons})
            if bridge.bridge_state == BRIDGE_WELL_SUPPORTED:
                any_would_pass = True
        if any_would_pass:
            any_edge_would_pass_but_none_did += 1
            if len(order_dependence_examples) < 5:
                order_dependence_examples.append({"component_sizes": (len(components.get(ra, [])), len(components.get(rb, []))), "edges": edge_details})
        else:
            all_edges_genuinely_weak += 1

    return {
        "checkpoint": str(checkpoint_path),
        "cap": cap,
        "total_multi_edge_cross_component_pairs": total_multi_edge_pairs,
        "pairs_where_some_individual_edge_would_pass_bridge_veto": any_edge_would_pass_but_none_did,
        "pairs_where_all_edges_genuinely_fail_bridge_veto": all_edges_genuinely_weak,
        "order_dependence_examples": order_dependence_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = trace(args.checkpoint, args.cap, args.device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
