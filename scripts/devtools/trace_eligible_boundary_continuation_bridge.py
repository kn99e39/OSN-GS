"""Worklog 56: real-checkpoint trace for the eligible-boundary -> continuation
-domain -> occluded-region-candidate production bridge.

Reuses the same checkpoint-to-full-evidence extraction as
``authoritative_replay_fingerprint.py`` / ``frozen_core_seeding_replay.py``,
then calls the canonical construction path and the new bridge, reporting only
counts and per-attempt status -- no threshold/topology re-derivation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_eligible_boundary_continuation_bridge import build_eligible_boundary_continuation_bridge
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
        bridge_result = build_eligible_boundary_continuation_bridge(construction)

    return {
        "checkpoint": str(checkpoint),
        "eligible_materialized_surface_count": len(construction.eligible_materialized_surfaces()),
        "bridge_diagnostic_summary": bridge_result.diagnostic_summary(),
        "attempts": [attempt.payload() for attempt in bridge_result.attempts],
        "continuation_domain_ids": [d.domain_id for d in bridge_result.continuation_domains],
        "occluded_region_candidate_ids": [c.candidate_id for c in bridge_result.occluded_region_candidates],
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
