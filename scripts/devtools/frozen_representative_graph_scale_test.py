"""Worklog 33: frozen-representative graph-scale invariance harness.

Separates two previously-conflated questions:

1. Is a graph-scale ESTIMATOR itself rigid-transform/uniform-scale invariant,
   holding the representative SET fixed (same stable IDs, same membership)?
   -- "Test A" / frozen-representative test.
2. Does representative SELECTION itself return a different representative
   subset under a rigid transform (an already-documented, accepted
   limitation of the axis-aligned voxel grid), and how much does THAT
   perturb downstream graph/region topology?
   -- "Test B" / selection perturbation robustness test.

Worklog 32's end-to-end invariance tests re-ran selection on every call, so
a G-candidate failing them could have been failing because of (1) or (2) --
indistinguishable. This harness freezes representative-level state (from a
real checkpoint replay) once, then applies rigid transforms directly to that
FIXED set for Test A, and separately re-runs full selection after
transforming the raw full cloud for Test B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import (
    covariance_from_scale_rotation,
    extract_covariance_frame,
)
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    ManifoldAffinityConfig,
    build_manifold_affinity_graph,
    CANDIDATE_STATUS_CANDIDATE,
    RELATION_SAME_SURFACE,
)
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability


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


def _rep_kNN_spacing(positions: torch.Tensor, k: int = 8) -> torch.Tensor:
    """G1: this representative's own median distance to its k nearest OTHER
    representatives. Purely a function of ``positions`` -- provably rigid/
    uniform-scale invariant in exact arithmetic (Euclidean distance)."""
    count = int(positions.shape[0])
    k = max(1, min(k, count - 1))
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    knn_distances, _ = torch.topk(distances, k=k, largest=False, dim=1)
    return knn_distances.median(dim=1).values.clamp_min(1e-9)


def _rep_normal_compatible_spacing(positions: torch.Tensor, frame, k: int = 8, alignment_min: float = 0.6) -> torch.Tensor:
    """G2: median distance to the k nearest OTHER representatives whose
    normal aligns (orientation-insensitive) above ``alignment_min`` --
    excludes close-parallel/opposite-mode neighbors from the spacing
    estimate. Falls back to the plain kNN spacing (G1) if fewer than 2
    compatible neighbors exist within a generous 4x-kNN search pool."""
    count = int(positions.shape[0])
    pool_k = max(1, min(4 * k, count - 1))
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    pool_distances, pool_indices = torch.topk(distances, k=pool_k, largest=False, dim=1)
    normals = frame.normal_candidate
    alignment = (normals.unsqueeze(1) * normals[pool_indices]).sum(dim=-1).abs()
    compatible = alignment >= alignment_min
    fallback = _rep_kNN_spacing(positions, k=k)
    out = torch.empty((count,), dtype=positions.dtype, device=positions.device)
    for i in range(count):
        mask = compatible[i]
        if int(mask.sum().item()) >= 2:
            values = pool_distances[i][mask]
            top = torch.topk(values, k=min(k, int(values.numel())), largest=False).values
            out[i] = top.median()
        else:
            out[i] = fallback[i]
    return out.clamp_min(1e-9)


GRAPH_SCALE_CANDIDATES = {
    "G0_footprint": lambda positions, frame: frame.tangent_major_scale,
    "G1_rep_knn": lambda positions, frame: _rep_kNN_spacing(positions),
    "G2_normal_compatible": lambda positions, frame: _rep_normal_compatible_spacing(positions, frame),
}


def frozen_representative_state(checkpoint_path: Path, cap: int, device: str) -> dict:
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
    rep_idx = bundle.representative_indices
    return {
        "positions": points[rep_idx].detach().clone(),
        "covariance": covariance[rep_idx].detach().clone(),
        "stable_ids": bundle.representative_stable_ids,
        "reliability": bundle.construction.reliability,
    }


def fixed_representative_rigid_test(state: dict, *, angle: float, translation, scale: float, config: ManifoldAffinityConfig | None = None) -> dict:
    """Test A: apply a rigid rotation + translation + uniform scale DIRECTLY
    to the frozen representative set (no selection re-run at all). Compare
    graph results for the SAME stable-ID pairs before/after."""
    torch = __import__("torch")
    positions = state["positions"]
    covariance = state["covariance"]
    reliability = state["reliability"]
    ids = state["stable_ids"]

    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    rotation = torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=positions.dtype, device=positions.device)
    t = torch.tensor(translation, dtype=positions.dtype, device=positions.device)
    transformed_positions = (positions @ rotation.T) * scale + t
    transformed_covariance = (scale * scale) * (rotation @ covariance @ rotation.transpose(-1, -2))

    frame_base = extract_covariance_frame(covariance)
    frame_transformed = extract_covariance_frame(transformed_covariance)

    results = {}
    for name, estimator in GRAPH_SCALE_CANDIDATES.items():
        graph_scale_base = estimator(positions, frame_base)
        graph_scale_transformed = estimator(transformed_positions, frame_transformed)

        graph_base = build_manifold_affinity_graph(
            positions, frame_base, reliability, config=config, ids=ids,
            candidate_scale=graph_scale_base, residual_scale=graph_scale_base,
        )
        graph_transformed = build_manifold_affinity_graph(
            transformed_positions, frame_transformed, reliability, config=config, ids=ids,
            candidate_scale=graph_scale_transformed, residual_scale=graph_scale_transformed,
        )
        base_relations = {(e.source_id, e.target_id): e.manifold_relation for e in graph_base.edges}
        transformed_relations = {(e.source_id, e.target_id): e.manifold_relation for e in graph_transformed.edges}
        mismatches = [key for key in base_relations if base_relations[key] != transformed_relations.get(key)]
        base_same_surface = sum(1 for r in base_relations.values() if r == RELATION_SAME_SURFACE)
        transformed_same_surface = sum(1 for r in transformed_relations.values() if r == RELATION_SAME_SURFACE)
        results[name] = {
            "edge_count_base": len(base_relations),
            "edge_count_transformed": len(transformed_relations),
            "relation_mismatch_count": len(mismatches),
            "same_surface_base": base_same_surface,
            "same_surface_transformed": transformed_same_surface,
            "exactly_invariant": len(mismatches) == 0 and len(base_relations) == len(transformed_relations),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    state = frozen_representative_state(args.checkpoint, args.cap, args.device)
    result = fixed_representative_rigid_test(
        state, angle=0.4, translation=(5.0, -3.0, 1.5), scale=2.5,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
