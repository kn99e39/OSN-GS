"""Worklog 34: region-quality + boundary-component diagnostic trace.

Offline-only. Loads a checkpoint, runs the exact production
``_construct_canonical_with_full_evidence`` path (same as worklog 30/31/33),
and extracts region membership/size/purity statistics plus a per-component
breakdown of WHY each boundary component is not a closed simple loop --
without changing any production code or threshold.
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


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


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
    rep_points = points[bundle.representative_indices]
    stable_to_local = {sid: i for i, sid in enumerate(bundle.representative_stable_ids)}

    membership_counts = Counter(regions_result.node_membership_state)

    region_rows = []
    member_counts = []
    for region in regions_result.regions:
        member_ids = region.member_ids
        local_idx = [stable_to_local[sid] for sid in member_ids if sid in stable_to_local]
        n = len(local_idx)
        member_counts.append(n)
        if n >= 2:
            member_positions = rep_points[torch.tensor(local_idx)]
            centroid = member_positions.mean(dim=0)
            diameter = float(torch.cdist(member_positions, member_positions).max().item())
        else:
            diameter = 0.0
        # normal dispersion from covariance_frame at those indices
        normals = construction.covariance_frame.normal_candidate[torch.tensor(local_idx)] if local_idx else None
        if normals is not None and normals.shape[0] >= 2:
            # orientation-insensitive: align signs to first normal before averaging
            ref = normals[0]
            signs = torch.where((normals @ ref) < 0, -1.0, 1.0).unsqueeze(-1)
            aligned = normals * signs
            mean_normal_len = float(torch.linalg.norm(aligned.mean(dim=0)).item())
            normal_dispersion = 1.0 - mean_normal_len  # 0 = perfectly consistent, 1 = fully dispersed
        else:
            normal_dispersion = 0.0
        region_rows.append({
            "region_id": region.region_id,
            "member_count": n,
            "core_member_count": len(region.core_member_ids),
            "attached_ambiguous_count": len(region.attached_ambiguous_member_ids),
            "rejected_excluded_count": len(region.rejected_excluded_ids),
            "world_space_diameter": diameter,
            "normal_dispersion": normal_dispersion,
            "region_state": region.region_state,
            "region_confidence": region.region_confidence,
        })

    total_rep_count = len(bundle.representative_stable_ids)
    in_any_region = sum(1 for r in regions_result.node_region_id if r >= 0)

    # --- boundary component breakdown ---
    components = construction.ordered_boundary_components
    halfedges = construction.boundary_halfedge_candidates
    component_rows = []
    ordering_state_counts = Counter()
    for component in components:
        ordering_state_counts[component.ordering_state] += 1
        node_ids = list(component.ordered_source_ids)
        component_rows.append({
            "region_id": component.region_id,
            "ordering_state": component.ordering_state,
            "role_candidate": getattr(component, "role_candidate", None),
            "node_count": len(node_ids),
            "branch_node_count": len(getattr(component, "branch_node_ids", ()) or ()),
        })

    halfedge_reason_counts = Counter(h.boundary_reason for h in halfedges)

    return {
        "checkpoint": str(checkpoint_path),
        "cap": cap,
        "representative_count": total_rep_count,
        "membership_state_counts": dict(membership_counts),
        "representatives_in_any_region": in_any_region,
        "representatives_in_any_region_fraction": in_any_region / max(total_rep_count, 1),
        "region_count": len(regions_result.regions),
        "region_member_count_median": _percentile(member_counts, 0.5),
        "region_member_count_p90": _percentile(member_counts, 0.9),
        "region_member_count_max": max(member_counts) if member_counts else 0,
        "singleton_region_count": sum(1 for c in member_counts if c == 1),
        "two_member_region_count": sum(1 for c in member_counts if c == 2),
        "three_member_region_count": sum(1 for c in member_counts if c == 3),
        "micro_region_count_le3": sum(1 for c in member_counts if c <= 3),
        "major_region_count_gt10": sum(1 for c in member_counts if c > 10),
        "region_rows": region_rows,
        "boundary_component_count": len(components),
        "boundary_component_ordering_state_counts": dict(ordering_state_counts),
        "boundary_component_rows": component_rows,
        "halfedge_boundary_reason_counts": dict(halfedge_reason_counts),
        "construction_state": construction.construction_state,
        "diagnostic_summary": dict(construction.diagnostic_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = trace(args.checkpoint, args.cap, args.device)
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
