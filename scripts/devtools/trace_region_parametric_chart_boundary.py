"""Worklog 61: real-checkpoint before/after report for region-local
parametric chart boundary materialization.

Reuses the same checkpoint-to-full-evidence extraction as
``authoritative_replay_fingerprint.py`` -- reports the pre-existing
`eligible_closed_boundary` physical count (before/unchanged) alongside the
new `eligible_parametric_chart_boundary` count/materialized surfaces (after).
"""

from __future__ import annotations

import argparse
import json
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


def trace(checkpoint: Path, cap: int, device: str = "cpu") -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
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
        s = construction.diagnostic_summary

    return {
        "checkpoint": str(checkpoint),
        "region_count": s["region_count"],
        "before_physical_eligible_closed_count": s["region_boundary_eligible_closed_count"],
        "before_physical_materialized_surface_count": s["materialized_surface_count"],
        "after_parametric_chart_eligible_count": s["parametric_chart_eligible_count"],
        "after_parametric_chart_materialized_surface_count": s["parametric_chart_materialized_surface_count"],
        "combined_materialized_surface_count": (
            s["materialized_surface_count"] + s["parametric_chart_materialized_surface_count"]
        ),
        "parametric_chart_insufficient_topology_count": s["parametric_chart_insufficient_topology_count"],
        "parametric_chart_open_or_branching_count": s["parametric_chart_open_or_branching_count"],
        "parametric_chart_self_intersecting_count": s["parametric_chart_self_intersecting_count"],
        "parametric_chart_no_tangent_frame_count": s["parametric_chart_no_tangent_frame_count"],
        "parametric_chart_partition_seam_segment_count": s["parametric_chart_partition_seam_segment_count"],
        "parametric_chart_physical_termination_segment_count": s["parametric_chart_physical_termination_segment_count"],
        "per_region": [
            {
                "region_id": item["region_id"],
                "status": item["status"],
                "reason": item["reason"],
                "segment_kind_counts": item.get("segment_kind_counts"),
            }
            for item in s["region_parametric_chart_boundaries"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(
        {path.stem + "_" + path.parent.name: trace(path, args.cap) for path in args.checkpoints}, indent=2,
    ))


if __name__ == "__main__":
    main()
