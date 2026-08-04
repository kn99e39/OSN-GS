"""Worklog 36 (task section 15): per-major-region physical boundary
candidate coverage on real 3k/5k/10k -- distinguishes region-too-small vs.
low candidate recall vs. ordering-solver failure."""

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
    regions = construction.surface_regions.regions
    halfedges = [h for h in construction.boundary_halfedge_candidates if h.boundary_reason == "observed_support_termination"]
    candidates_by_region = Counter(h.source_region_id for h in halfedges)
    components = construction.ordered_boundary_components
    closed_by_region = Counter(c.region_id for c in components if c.ordering_state == "ordered_closed_loop")
    component_states_by_region: dict[int, Counter] = {}
    for c in components:
        component_states_by_region.setdefault(c.region_id, Counter())[c.ordering_state] += 1

    region_rows = []
    for region in sorted(regions, key=lambda r: -len(r.member_ids))[:10]:
        n_candidates = candidates_by_region.get(region.region_id, 0)
        n_closed = closed_by_region.get(region.region_id, 0)
        classification = (
            "1_region_too_small_for_perimeter" if n_candidates < 3
            else "4_candidates_and_compatibility_sufficient_but_ordering_failed" if n_candidates >= 3 and n_closed == 0
            else "resolved_or_partial" if n_closed > 0
            else "unclassified"
        )
        region_rows.append({
            "region_id": region.region_id,
            "member_count": len(region.member_ids),
            "genuine_termination_candidate_count": n_candidates,
            "candidate_per_member_ratio": round(n_candidates / max(len(region.member_ids), 1), 3),
            "closed_component_count": n_closed,
            "component_states": dict(component_states_by_region.get(region.region_id, {})),
            "classification": classification,
        })

    classification_counts = Counter(row["classification"] for row in [
        {
            "classification": (
                "1_region_too_small_for_perimeter" if candidates_by_region.get(r.region_id, 0) < 3
                else "4_candidates_and_compatibility_sufficient_but_ordering_failed" if candidates_by_region.get(r.region_id, 0) >= 3 and closed_by_region.get(r.region_id, 0) == 0
                else "resolved_or_partial"
            )
        }
        for r in regions
    ])

    ordering_failure_rows = []
    for region in regions:
        n_candidates = candidates_by_region.get(region.region_id, 0)
        n_closed = closed_by_region.get(region.region_id, 0)
        if n_candidates >= 3 and n_closed == 0:
            ordering_failure_rows.append({
                "region_id": region.region_id,
                "member_count": len(region.member_ids),
                "genuine_termination_candidate_count": n_candidates,
                "component_states": dict(component_states_by_region.get(region.region_id, {})),
            })

    return {
        "checkpoint": str(checkpoint_path),
        "cap": cap,
        "region_count": len(regions),
        "top10_region_rows": region_rows,
        "ordering_failure_rows": ordering_failure_rows,
        "classification_counts_all_regions": dict(classification_counts),
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
