"""Worklog 48: A/B comparison of the fold-signature same-mode locality gate
on the standard synthetic negative-control fixtures (box/cylinder/sphere/
thin_slab), by rerunning the exact `_construct_canonical_with_full_evidence`
downsampled path with two ``ContinuationShellConfig`` instances: production
default vs. the fold check effectively disabled (ratio threshold raised to
1e9). Does not modify production code.
"""

from __future__ import annotations

import math

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import _representative_knn_spacing, _slice_covariance_frame
from osn_gs.surface.torch_density_preserving_representative_selection import select_density_preserving_representatives
from osn_gs.surface.torch_full_neighborhood_evidence import assign_nearest_representative, compute_full_neighborhood_evidence
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_structural_reliability import (
    evaluate_intrinsic_reliability,
    evaluate_structural_reliability_from_full_evidence,
)
from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellConfig, ContinuationShellInput
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def run(name: str, cap: int, config: ContinuationShellConfig) -> dict:
    scene = make_gaussian_reliability_scene(name)
    points = torch.as_tensor(scene.positions, dtype=torch.float32)
    covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
    opacity = torch.ones(points.shape[0])
    stable_ids_list = list(range(points.shape[0]))

    frame_full = extract_covariance_frame(covariance)
    intrinsic_full = evaluate_intrinsic_reliability(frame_full)
    selection = select_density_preserving_representatives(points, frame_full, opacity, stable_ids_list, max_points=cap)
    rep_indices = selection.representative_indices
    rep_points = points[rep_indices]
    rep_covariance = covariance[rep_indices]
    rep_frame = _slice_covariance_frame(frame_full, rep_indices)
    rep_stable_ids = tuple(stable_ids_list[i] for i in rep_indices.detach().cpu().tolist())
    downsampled = int(rep_indices.numel()) != int(points.shape[0])
    if not downsampled:
        return {"name": name, "downsampled": False}
    precomputed_assignment = assign_nearest_representative(points, rep_points)
    evidence = compute_full_neighborhood_evidence(
        points, frame_full, opacity, intrinsic_full, rep_points, rep_frame, rep_stable_ids,
        precomputed_assignment=precomputed_assignment,
    )
    reliability = evaluate_structural_reliability_from_full_evidence(rep_frame, evidence)
    nearest_representative_index, _distance = precomputed_assignment
    representative_graph_scale = _representative_knn_spacing(rep_points)
    continuation_input = ContinuationShellInput(
        full_positions=points, full_frame=frame_full, full_intrinsic=intrinsic_full,
        full_opacity=opacity, full_stable_ids=stable_ids_list,
        nearest_representative_index=nearest_representative_index,
        representative_mean_spacing=evidence.mean_spacing, config=config,
    )
    construction = construct_visible_nurbs_from_gaussians(
        rep_points, covariance=rep_covariance, stable_ids=rep_stable_ids, reliability=reliability,
        continuation_input=continuation_input,
        candidate_scale=representative_graph_scale, residual_scale=representative_graph_scale,
    )
    s = construction.diagnostic_summary
    return {
        "name": name, "downsampled": True,
        "physical": s.get("boundary_genuine_termination_candidate_count"),
        "closed": s.get("boundary_component_closed_count"),
        "materialized": s.get("materialized_surface_count"),
    }


def main():
    default_config = ContinuationShellConfig()
    disabled_config = ContinuationShellConfig(fold_signature_min_hops=1_000_000, fold_signature_path_ratio_min=1e18)
    for cap in (64,):
        for name in ("box", "cylinder", "sphere", "thin_slab"):
            before = run(name, cap, disabled_config)
            after = run(name, cap, default_config)
            print(f"cap={cap} {name}: fold_check_disabled={before} fold_check_default={after}")


if __name__ == "__main__":
    main()
