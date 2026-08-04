"""Worklog 36 (task section 14): classify every ambiguous_unassigned
representative on a real checkpoint into a precise sub-reason (R1-R6)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_gaussian_manifold_affinity import RELATION_SAME_SURFACE
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    RegionFormationConfig,
    _build_relation_adjacency,
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
    regions_result = construction.surface_regions
    graph = construction.manifold_affinity
    count = len(regions_result.node_region_id)
    same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, graph)

    node_region_id = regions_result.node_region_id
    node_membership_state = regions_result.node_membership_state

    sub_reason_counts = Counter()
    same_surface_degree_of_ambiguous = []
    for n in range(count):
        if node_membership_state[n] != "ambiguous_unassigned":
            continue
        same_surface_degree = len(same_surface[n])
        same_surface_degree_of_ambiguous.append(same_surface_degree)

        if same_surface_degree == 0:
            sub_reason_counts["R1_no_same_surface_degree"] += 1
            continue

        # Which regions do same_surface neighbors belong to?
        neighbor_regions = Counter()
        for nb in same_surface[n]:
            rid = node_region_id[nb]
            if rid >= 0:
                neighbor_regions[rid] += 1

        if not neighbor_regions:
            # Has same_surface degree, but none of those neighbors are
            # themselves in ANY region yet -- consensus/support has nothing
            # to attach to.
            sub_reason_counts["R2_same_surface_neighbors_not_yet_in_any_region"] += 1
            continue

        if len(neighbor_regions) >= 2:
            top_two = neighbor_regions.most_common(2)
            if top_two[0][1] < 1.5 * top_two[1][1]:
                sub_reason_counts["R3_competing_regions_no_clear_majority"] += 1
                continue

        sub_reason_counts["R4_or_growth_threshold_not_met"] += 1

    return {
        "checkpoint": str(checkpoint_path),
        "cap": cap,
        "representative_count": count,
        "membership_state_counts": dict(Counter(node_membership_state)),
        "ambiguous_unassigned_sub_reasons": dict(sub_reason_counts),
        "ambiguous_same_surface_degree_distribution": {
            "zero": sum(1 for d in same_surface_degree_of_ambiguous if d == 0),
            "one": sum(1 for d in same_surface_degree_of_ambiguous if d == 1),
            "two_plus": sum(1 for d in same_surface_degree_of_ambiguous if d >= 2),
            "median": sorted(same_surface_degree_of_ambiguous)[len(same_surface_degree_of_ambiguous) // 2] if same_surface_degree_of_ambiguous else 0,
        },
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
