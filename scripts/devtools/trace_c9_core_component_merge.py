"""Worklog 35: instrumented replay of `_seed_core_components` / region merge
to answer R1-R7 -- why real long-horizon core components stay small (C9).

Offline-only, read-only reproduction of the exact production functions
(`_build_relation_adjacency`, `_seed_core_components`, the region-merge pass
in `form_surface_regions`), with per-component-pair diagnostics recorded
along the way. Does not modify production code.
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
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    BRIDGE_CONTRADICTED,
    BRIDGE_WEAK_CANDIDATE,
    BRIDGE_WELL_SUPPORTED,
    CONSENSUS_CONTRADICTED,
    RegionFormationConfig,
    _build_relation_adjacency,
    _evaluate_bridge_veto,
    _seed_core_components,
    _pair_key,
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

    bridge_state_counts = Counter(b.bridge_state for b in bridge_by_pair.values())
    weak_bridge_reasons = Counter()
    for b in bridge_by_pair.values():
        if b.bridge_state == BRIDGE_WEAK_CANDIDATE:
            for r in b.reasons:
                weak_bridge_reasons[r] += 1

    # R2: is each weak-bridge component pair's cross-support really JUST one edge?
    # Reconstruct component membership from the union-find result and count same_surface
    # cross edges between distinct FINAL components (not just what was checked during seeding,
    # since union order matters -- this re-derives it post-hoc for diagnosis only).
    components: dict[int, list[int]] = {}
    for node in range(count):
        components.setdefault(uf.find(node), []).append(node)
    component_sizes = sorted((len(v) for v in components.values()), reverse=True)

    cross_component_same_surface_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b), edge in by_pair.items():
        from osn_gs.surface.torch_gaussian_manifold_affinity import RELATION_SAME_SURFACE
        if edge.manifold_relation != RELATION_SAME_SURFACE:
            continue
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        cross_component_same_surface_edges.setdefault(_pair_key(ra, rb), []).append((a, b))

    single_bridge_only_pairs = sum(1 for edges in cross_component_same_surface_edges.values() if len(edges) == 1)
    multi_bridge_pairs = sum(1 for edges in cross_component_same_surface_edges.values() if len(edges) >= 2)

    # R3: for pairs with >=2 cross edges but still not merged (per bridge veto during seeding),
    # what specifically blocked them? Re-run the veto post-hoc for the TOP few largest-crossing pairs.
    multi_edge_blocked_examples = []
    for (ra, rb), edges in sorted(cross_component_same_surface_edges.items(), key=lambda kv: -len(kv[1]))[:8]:
        if len(edges) < 2:
            continue
        sample_a, sample_b = edges[0]
        consensus = consensus_by_pair.get((sample_a, sample_b))
        bridge = bridge_by_pair.get((sample_a, sample_b))
        multi_edge_blocked_examples.append({
            "component_sizes": (len(components.get(ra, [])), len(components.get(rb, []))),
            "cross_edge_count": len(edges),
            "sample_consensus_state": consensus.consensus_state if consensus else None,
            "sample_bridge_state": bridge.bridge_state if bridge else None,
            "sample_bridge_reasons": list(bridge.reasons) if bridge else None,
        })

    return {
        "checkpoint": str(checkpoint_path),
        "cap": cap,
        "representative_count": count,
        "region_count_from_uf": len(components),
        "component_size_top10": component_sizes[:10],
        "component_size_median": component_sizes[len(component_sizes) // 2] if component_sizes else 0,
        "core_eligible_edge_count": len(consensus_by_pair),
        "bridge_state_counts": dict(bridge_state_counts),
        "weak_bridge_reason_counts": dict(weak_bridge_reasons),
        "boundary_conflict_edge_count": len(boundary_conflict_edges),
        "cross_component_pair_count": len(cross_component_same_surface_edges),
        "cross_component_pairs_with_exactly_one_edge": single_bridge_only_pairs,
        "cross_component_pairs_with_two_plus_edges": multi_bridge_pairs,
        "multi_edge_still_blocked_examples": multi_edge_blocked_examples,
        "diagnostic_summary": dict(construction.diagnostic_summary),
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
