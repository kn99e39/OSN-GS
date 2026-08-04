"""Worklog 36: authoritative frozen replay with a full fingerprint, resolving
the worklog 34/35 baseline discrepancy.

Root cause (confirmed): worklog 35's ablation harness captured its "baseline"
via `git show HEAD:<file>` -- but HEAD (d359c5e) predates BOTH worklog 34's
growth-loop fix and worklog 35's own C9 fix (neither was ever committed, only
present in the uncommitted working tree). So worklog 35's "A_baseline" was
silently a PRE-worklog-34 state, not worklog 34's own reported baseline --
explaining region_count 70/84/63 (ablation A) vs worklog 34's reported
75/85/64, and core_member 362/431/344 vs 392/447/357.

This script replaces file-swapping with explicit config flags
(`RegionFormationConfig.allow_weak_bridge_only_growth_support`,
`require_nearby_parallel_evidence_for_parallel_veto`) so all four ablation
states run on the IDENTICAL code path and file content -- no drift possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationConfig


def _sh_degree_from_checkpoint(raw: dict) -> int:
    rest_dim = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim:
        degree += 1
    return degree


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_fingerprint() -> dict:
    def run(args):
        return subprocess.run(
            args, cwd=Path(__file__).resolve().parents[2], capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()

    commit = run(["git", "rev-parse", "HEAD"])
    dirty_files = run(["git", "status", "--short"])
    dirty_diff_hash = hashlib.sha256(run(["git", "diff", "HEAD"]).encode("utf-8")).hexdigest()
    return {"commit": commit, "dirty_file_count": len(dirty_files.splitlines()), "dirty_diff_sha256": dirty_diff_hash}


def load_model(checkpoint_path: Path, device: str) -> tuple[TorchGaussianModel, dict]:
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
    return model, payload


def replay(checkpoint_path: Path, cap: int, device: str, region_config: RegionFormationConfig) -> dict:
    torch.manual_seed(0)
    model, payload = load_model(checkpoint_path, device)
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

        # Inject the region_config override by monkeypatching the pipeline's
        # default construction call -- `_construct_canonical_with_full_evidence`
        # doesn't currently accept a RegionFormationConfig override, so we
        # replicate its body's region-formation call with the override applied.
        # (See section 2 note: kept diagnostic-only, not a production signature change.)
        import time
        from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
        from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_intrinsic_reliability, evaluate_structural_reliability_from_full_evidence
        from osn_gs.surface.torch_density_preserving_representative_selection import select_density_preserving_representatives
        from osn_gs.surface.torch_full_neighborhood_evidence import compute_full_neighborhood_evidence, assign_nearest_representative
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians, VisibleSurfaceConstructionConfig
        from osn_gs.core.torch_pipeline import _representative_knn_spacing
        from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput

        t0 = time.time()
        frame_full = extract_covariance_frame(covariance)
        intrinsic_full = evaluate_intrinsic_reliability(frame_full)
        stable_ids_list = list(stable_ids)
        selection = select_density_preserving_representatives(
            points, frame_full, opacity, stable_ids_list, max_points=int(pipeline.config.canonical_construction_max_points),
        )
        rep_indices = selection.representative_indices
        rep_points = points[rep_indices]
        rep_covariance = covariance[rep_indices]
        from osn_gs.core.torch_pipeline import _slice_covariance_frame
        rep_frame = _slice_covariance_frame(frame_full, rep_indices)
        rep_stable_ids = tuple(stable_ids_list[i] for i in rep_indices.detach().cpu().tolist())
        downsampled = int(rep_indices.numel()) != int(points.shape[0])
        precomputed_assignment = assign_nearest_representative(points, rep_points) if downsampled else None

        import math
        local_evidence_scale = None
        if downsampled:
            budget = max(16, int(pipeline.config.canonical_construction_max_points))
            resolution = max(2, int(math.ceil(budget ** 0.5)))
            centered = points - points.mean(dim=0, keepdim=True)
            variance_trace = (centered.square().sum(dim=0) / max(int(points.shape[0]), 1)).sum().clamp_min(1e-12)
            characteristic_length = 2.0 * torch.sqrt(variance_trace / 3.0)
            cell_volume = float((characteristic_length / resolution).clamp_min(1e-9) ** 3)
            source_counts = torch.tensor([rep.source_count for rep in selection.representatives], dtype=points.dtype, device=points.device)
            local_evidence_scale = (cell_volume / source_counts.clamp_min(1)).pow(1.0 / 3.0)

        representative_graph_scale = _representative_knn_spacing(rep_points)
        evidence = compute_full_neighborhood_evidence(
            points, frame_full, opacity, intrinsic_full, rep_points, rep_frame, rep_stable_ids,
            precomputed_assignment=precomputed_assignment, local_evidence_scale=local_evidence_scale,
        )
        construction_config = VisibleSurfaceConstructionConfig(regions=region_config)
        if not downsampled:
            from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
            reliability = evaluate_structural_reliability(rep_points, rep_frame)
            construction = construct_visible_nurbs_from_gaussians(
                rep_points, covariance=rep_covariance, stable_ids=rep_stable_ids, reliability=reliability,
                candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
                config=construction_config,
            )
        else:
            reliability = evaluate_structural_reliability_from_full_evidence(rep_frame, evidence)
            nearest_representative_index, _distance = precomputed_assignment
            continuation_input = ContinuationShellInput(
                full_positions=points, full_frame=frame_full, full_intrinsic=intrinsic_full, full_opacity=opacity,
                full_stable_ids=stable_ids_list, nearest_representative_index=nearest_representative_index,
                representative_mean_spacing=evidence.mean_spacing,
            )
            construction = construct_visible_nurbs_from_gaussians(
                rep_points, covariance=rep_covariance, stable_ids=rep_stable_ids, reliability=reliability,
                continuation_input=continuation_input,
                candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
                config=construction_config,
            )
        elapsed = time.time() - t0

    regions = construction.surface_regions
    member_counts = [len(r.member_ids) for r in regions.regions]
    core_member = sum(1 for s in regions.node_membership_state if s == "core_member")
    consensus_attached = sum(1 for s in regions.node_membership_state if s == "consensus_attached")
    ambiguous_unassigned = sum(1 for s in regions.node_membership_state if s == "ambiguous_unassigned")
    conflict_boundary = sum(1 for s in regions.node_membership_state if s == "conflict_boundary_candidate")
    rejected = sum(1 for s in regions.node_membership_state if s == "rejected_structural_node")

    summary = dict(construction.diagnostic_summary)
    summary.update({
        "core_member": core_member,
        "consensus_attached": consensus_attached,
        "ambiguous_unassigned": ambiguous_unassigned,
        "conflict_boundary": conflict_boundary,
        "rejected": rejected,
        "region_member_median": sorted(member_counts)[len(member_counts) // 2] if member_counts else 0,
        "region_member_p90": sorted(member_counts)[int(0.9 * len(member_counts))] if member_counts else 0,
        "region_member_max": max(member_counts) if member_counts else 0,
        "micro_region_le3": sum(1 for c in member_counts if c <= 3),
        "major_region_gt10": sum(1 for c in member_counts if c > 10),
        "runtime_seconds": elapsed,
    })

    fingerprint = {
        "checkpoint_absolute_path": str(checkpoint_path.resolve()),
        "checkpoint_content_sha256": _file_hash(checkpoint_path),
        "iteration": payload.get("iteration"),
        "full_gaussian_count": int(points.shape[0]),
        "representative_cap": int(pipeline.config.canonical_construction_max_points),
        "representative_stable_id_sha256": hashlib.sha256(json.dumps(sorted(rep_stable_ids)).encode()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps({
            "cap": pipeline.config.canonical_construction_max_points,
            "region_config": {k: str(v) for k, v in region_config.__dict__.items()},
        }, sort_keys=True).encode()).hexdigest(),
        "local_evidence_scale_active": downsampled,
        "g1_graph_scale_active": True,
        "worklog34_growth_fix_enabled": region_config.allow_weak_bridge_only_growth_support,
        "worklog35_c9_fix_enabled": region_config.require_nearby_parallel_evidence_for_parallel_veto,
        "ordering_implementation": "hungarian_one_in_one_out_worklog35",
        "random_seed": 0,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": device,
        **_git_fingerprint(),
    }

    return {"fingerprint": fingerprint, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--worklog34", type=int, default=1)
    parser.add_argument("--worklog35", type=int, default=1)
    args = parser.parse_args()
    region_config = RegionFormationConfig(
        allow_weak_bridge_only_growth_support=bool(args.worklog34),
        require_nearby_parallel_evidence_for_parallel_veto=bool(args.worklog35),
    )
    result = replay(args.checkpoint, args.cap, args.device, region_config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
