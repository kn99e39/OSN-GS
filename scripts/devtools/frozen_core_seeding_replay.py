"""Worklog 37: frozen core-seeding replay harness.

Computes representative selection, full-neighborhood evidence, and the
manifold affinity graph ONCE per checkpoint, then exposes everything
`_seed_core_components` / `form_surface_regions` / boundary-candidate
generation need to be re-run repeatedly (e.g. across a threshold sweep)
without re-running selection/evidence/affinity. This is an in-process
equivalent of a serialized "frozen artifact" -- selection and the affinity
graph are the expensive, non-deterministic-under-perturbation steps; core
seeding/region formation/boundary generation are the cheap, deterministic
steps this task needs to sweep repeatedly.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, _representative_knn_spacing, _slice_covariance_frame
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_density_preserving_representative_selection import select_density_preserving_representatives
from osn_gs.surface.torch_full_neighborhood_evidence import assign_nearest_representative, compute_full_neighborhood_evidence
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import ManifoldAffinityConfig, build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import (
    evaluate_intrinsic_reliability,
    evaluate_structural_reliability_from_full_evidence,
)
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationConfig, form_surface_regions
from osn_gs.surface.torch_world_space_boundary_halfedges import extract_world_space_boundary_halfedge_candidates
from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates, normalize_continuation_candidates
from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput, build_continuation_shells_from_input
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames


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


@dataclass
class FrozenCoreSeedingState:
    rep_points: Any
    rep_covariance: Any
    rep_frame: Any
    rep_stable_ids: tuple
    reliability: Any
    graph: Any
    candidate_scale: Any
    residual_scale: Any
    full_points: Any
    full_frame: Any
    full_intrinsic: Any
    full_opacity: Any
    full_stable_ids: list
    nearest_representative_index: Any
    representative_mean_spacing: Any
    fingerprint: dict


def build_frozen_state(checkpoint_path: Path, cap: int, device: str = "cpu") -> FrozenCoreSeedingState:
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

        frame_full = extract_covariance_frame(covariance)
        intrinsic_full = evaluate_intrinsic_reliability(frame_full)
        stable_ids_list = list(stable_ids)
        selection = select_density_preserving_representatives(
            points, frame_full, opacity, stable_ids_list, max_points=int(pipeline.config.canonical_construction_max_points),
        )
        rep_indices = selection.representative_indices
        rep_points = points[rep_indices]
        rep_covariance = covariance[rep_indices]
        rep_frame = _slice_covariance_frame(frame_full, rep_indices)
        rep_stable_ids = tuple(stable_ids_list[i] for i in rep_indices.detach().cpu().tolist())
        downsampled = int(rep_indices.numel()) != int(points.shape[0])
        precomputed_assignment = assign_nearest_representative(points, rep_points) if downsampled else None

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
        reliability = evaluate_structural_reliability_from_full_evidence(rep_frame, evidence)
        graph = build_manifold_affinity_graph(
            rep_points, rep_frame, reliability, ids=rep_stable_ids,
            candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
        )
        nearest_representative_index, _distance = precomputed_assignment

    fingerprint = {
        "checkpoint_content_sha256": _file_hash(checkpoint_path),
        "representative_cap": cap,
        "representative_stable_id_sha256": hashlib.sha256(json.dumps(sorted(rep_stable_ids)).encode()).hexdigest(),
    }

    return FrozenCoreSeedingState(
        rep_points=rep_points, rep_covariance=rep_covariance, rep_frame=rep_frame, rep_stable_ids=rep_stable_ids,
        reliability=reliability, graph=graph,
        candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
        full_points=points, full_frame=frame_full, full_intrinsic=intrinsic_full, full_opacity=opacity,
        full_stable_ids=stable_ids_list, nearest_representative_index=nearest_representative_index,
        representative_mean_spacing=evidence.mean_spacing, fingerprint=fingerprint,
    )


def replay_region_formation(state: FrozenCoreSeedingState, region_config: RegionFormationConfig):
    """Re-run ONLY `form_surface_regions` (which internally calls
    `_seed_core_components`) against the frozen graph -- no selection/
    evidence/affinity recomputation."""
    return form_surface_regions(
        state.rep_points, state.rep_frame, state.reliability, state.graph,
        config=region_config, ids=state.rep_stable_ids,
    )


def replay_boundary_candidates(state: FrozenCoreSeedingState, regions):
    """Re-run boundary candidate generation against a frozen graph + a given
    region-formation result, using the full-cloud continuation shell (same
    path production uses when downsampled)."""
    oriented_normals = state.rep_frame.normal_candidate
    canonical_frames = construct_canonical_region_tangent_frames(state.rep_points, state.rep_frame, state.reliability, regions, ids=state.rep_stable_ids)
    continuation_input = ContinuationShellInput(
        full_positions=state.full_points, full_frame=state.full_frame, full_intrinsic=state.full_intrinsic,
        full_opacity=state.full_opacity, full_stable_ids=state.full_stable_ids,
        nearest_representative_index=state.nearest_representative_index,
        representative_mean_spacing=state.representative_mean_spacing,
    )
    continuation = build_continuation_shells_from_input(continuation_input, state.rep_points, state.rep_frame, state.rep_stable_ids, regions, canonical_frames)
    # Full-cloud continuation replaces the angular evidence source, but the
    # representative topology neighborhood scale still follows the production
    # contract: use RepresentativeGraphScale, not Gaussian footprint scale.
    termination_halfedges = extract_support_termination_candidates(
        state.rep_points, oriented_normals, state.candidate_scale, regions,
        ids=state.rep_stable_ids, sectors=8, canonical_frames=canonical_frames, continuation=continuation,
    )
    return normalize_continuation_candidates(termination_halfedges)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()
    t0 = time.time()
    state = build_frozen_state(args.checkpoint, args.cap)
    print(f"built frozen state in {time.time()-t0:.1f}s, fingerprint={json.dumps(state.fingerprint)}")
    t0 = time.time()
    result = replay_region_formation(state, RegionFormationConfig())
    print(f"replayed region formation in {time.time()-t0:.3f}s, region_count={len(result.regions)}")

