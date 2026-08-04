"""Per-representative reliability-gate trace (worklog 31, diagnostic-only).

Runs the EXACT production ``reconstruct_visible_after_adc`` code path against
a saved checkpoint (no retraining) and, for every selected representative,
records identity/intrinsic/contextual/final/region fields plus explicit
pass/fail + signed margin for every gate the production contextual-
consistency function actually evaluates -- all read from the ALREADY-COMPUTED
production result objects (``bundle.selection``, ``bundle.evidence``,
``bundle.construction.reliability/.covariance_frame/.manifold_affinity/.surface_regions``).
No reliability policy, threshold, normalization, or region-admission logic is
reimplemented or changed here -- every gate comparison below mirrors
``evaluate_contextual_consistency_from_full_evidence``
(osn_gs/surface/torch_gaussian_structural_reliability.py) read-only, using
the SAME config object the production call used.

Usage:
    python -m scripts.devtools.trace_representative_reliability_gates \
        --checkpoint output/osn_gs_scene/3000/checkpoint.pt --cap 2048 \
        --output-jsonl <path> --output-summary <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    CANDIDATE_STATUS_CANDIDATE,
    RELATION_SAME_SURFACE,
)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    CONTEXTUAL_CONSISTENT,
    INTRINSIC_RELIABLE,
    StructuralReliabilityConfig,
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


def _gate(measured: float, threshold: float, direction: str) -> dict:
    """direction: 'min' means measured must be >= threshold to pass;
    'max' means measured must be <= threshold to pass. Margin is always
    signed so that margin > 0 means PASS, matching the production
    comparison exactly (no new semantics introduced)."""
    if direction == "min":
        passed = measured >= threshold
        margin = measured - threshold
    else:
        passed = measured <= threshold
        margin = threshold - measured
    return {
        "measured": measured, "threshold": threshold, "direction": direction,
        "pass": bool(passed), "signed_margin": margin,
    }


def trace(checkpoint_path: Path, cap: int, device: str, snapshot_name: str, iteration: int) -> dict:
    model = load_model(checkpoint_path, device)
    config = TorchPipelineConfig(canonical_construction_max_points=int(cap))
    pipeline = TorchOSNGSPipeline(config, device=device)
    contextual_config = StructuralReliabilityConfig().contextual

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

    reps = bundle.selection.representatives
    evidence = bundle.evidence
    reliability = bundle.construction.reliability
    frame = bundle.construction.covariance_frame
    regions_result = bundle.construction.surface_regions
    graph = bundle.construction.manifold_affinity
    m = len(reps)

    # Same-surface degree per representative, purely a READ over the
    # already-computed affinity graph (no new classification logic).
    same_surface_degree = [0] * m
    for edge in graph.edges:
        if edge.candidate_status == CANDIDATE_STATUS_CANDIDATE and edge.manifold_relation == RELATION_SAME_SURFACE:
            same_surface_degree[edge.source] += 1
            same_surface_degree[edge.target] += 1

    region_member_kind: dict[int, str] = {}
    for region in regions_result.regions:
        for sid in region.core_member_ids:
            region_member_kind[sid] = "core_member"
        for sid in region.attached_ambiguous_member_ids:
            region_member_kind[sid] = "attached_ambiguous_member"
        for sid in region.rejected_excluded_ids:
            region_member_kind[sid] = "rejected_excluded"

    rows = []
    for i in range(m):
        rep = reps[i]
        intrinsic_class = reliability.intrinsic.intrinsic_class[i]
        contextual_class = reliability.contextual.contextual_class[i]
        final_class = reliability.reliability_class[i]

        # --- contextual gates, mirrored read-only from
        # evaluate_contextual_consistency_from_full_evidence's own logic ---
        support_count = int(evidence.support_count[i])
        has_support = support_count > 0
        gates = {}
        if not has_support or support_count < contextual_config.insufficient_min_valid_neighbor_count:
            gates["support_present"] = _gate(float(support_count), float(contextual_config.insufficient_min_valid_neighbor_count), "min")
        else:
            non_rejected_support = support_count * (1.0 - float(evidence.rejected_neighbor_mass[i]))
            span = max(contextual_config.full_evidence_saturating_support_count - contextual_config.full_evidence_min_support_count, 1)
            support_sufficiency = max(0.0, min(1.0, (non_rejected_support - contextual_config.full_evidence_min_support_count) / span))
            gates["normal_consensus"] = _gate(float(evidence.normal_consensus[i]), contextual_config.consistent_min_neighbor_normal_agreement, "min")
            gates["tangent_residual"] = _gate(float(evidence.tangent_residual_mean[i]), contextual_config.consistent_max_mutual_tangent_residual, "max")
            gates["support_sufficiency"] = _gate(support_sufficiency, contextual_config.consistent_min_support_sufficiency, "min")
            gates["competing_mode_mass"] = _gate(float(evidence.competing_mode_mass[i]), contextual_config.consistent_max_multi_surface_ambiguity, "max")
            gates["rejected_neighbor_mass"] = _gate(float(evidence.rejected_neighbor_mass[i]), contextual_config.full_evidence_max_rejected_mass, "max")

        failed_gates = [name for name, g in gates.items() if not g["pass"]]

        row = {
            # --- identity ---
            "snapshot": snapshot_name, "iteration": iteration, "representative_index": i,
            "representative_stable_id": rep.representative_stable_id,
            "source_gaussian_index": rep.representative_gaussian_index,
            "cell_id": rep.cell_id, "mode_id": rep.mode_id,
            "source_count": rep.source_count, "source_opacity_mass": rep.source_opacity_mass,
            # --- intrinsic ---
            "intrinsic_class": intrinsic_class,
            "intrinsic_reasons": list(reliability.intrinsic.reasons[i]),
            "conditioning_score": float(reliability.intrinsic.conditioning_score[i]),
            "planar_likelihood": float(reliability.intrinsic.planar_likelihood[i]),
            "needle_likelihood": float(reliability.intrinsic.needle_likelihood[i]),
            "isotropic_likelihood": float(reliability.intrinsic.isotropic_likelihood[i]),
            "scale_validity": float(reliability.intrinsic.scale_validity[i]),
            "eigenvalues": frame.eigenvalues[i].detach().cpu().tolist(),
            "tangent_major_scale": float(frame.tangent_major_scale[i]),
            "tangent_minor_scale": float(frame.tangent_minor_scale[i]),
            "normal_thickness": float(frame.normal_thickness[i]),
            "elongation": float(frame.elongation[i]),
            "planarity": float(frame.planarity[i]),
            "isotropy": float(frame.isotropy[i]),
            "shape_class": frame.shape_class[i],
            # --- contextual evidence ---
            "support_count": support_count,
            "out_of_local_radius_count": int(evidence.out_of_local_radius_count[i]),
            "opacity_sum": float(evidence.opacity_sum[i]),
            "mean_spacing": float(evidence.mean_spacing[i]),
            "spacing_std": float(evidence.spacing_std[i]),
            "normal_consensus": float(evidence.normal_consensus[i]),
            "tangent_residual_mean": float(evidence.tangent_residual_mean[i]),
            "tangent_residual_std": float(evidence.tangent_residual_std[i]),
            "eigenvalue_ratio_mean": float(evidence.eigenvalue_ratio_mean[i]),
            "eigenvalue_ratio_std": float(evidence.eigenvalue_ratio_std[i]),
            "competing_mode_mass": float(evidence.competing_mode_mass[i]),
            "rejected_neighbor_mass": float(evidence.rejected_neighbor_mass[i]),
            "local_density": float(evidence.local_density[i]),
            "normalization_denominator_tangent_major_scale": float(frame.tangent_major_scale[i]),
            "contextual_class": contextual_class,
            "contextual_reasons": list(reliability.contextual.reasons[i]),
            "neighborhood_support_sufficiency": float(reliability.contextual.neighborhood_support_sufficiency[i]),
            "scale_consistency": float(reliability.contextual.scale_consistency[i]),
            # --- contextual gate results ---
            "gates": gates,
            "all_failed_gates": failed_gates,
            "first_failed_gate": failed_gates[0] if failed_gates else None,
            # --- final admission ---
            "final_reliability_class": final_class,
            "final_reasons": list(reliability.reasons[i]),
            "same_surface_degree": same_surface_degree[i],
            "region_seed_state": regions_result.node_membership_state[i],
            "final_region_id": regions_result.node_region_id[i],
            "region_member_kind": region_member_kind.get(rep.representative_stable_id, "not_in_any_region"),
        }
        rows.append(row)

    # --- gate waterfall summary ---
    def _count(pred):
        return sum(1 for r in rows if pred(r))

    waterfall = {
        "total_representatives": m,
        "intrinsic_reliable": _count(lambda r: r["intrinsic_class"] == INTRINSIC_RELIABLE),
        "intrinsic_ambiguous": _count(lambda r: r["intrinsic_class"] == "intrinsic_ambiguous"),
        "intrinsic_rejected": _count(lambda r: r["intrinsic_class"] == "intrinsic_rejected"),
        "contextual_consistent": _count(lambda r: r["contextual_class"] == CONTEXTUAL_CONSISTENT),
        "contextual_mixed": _count(lambda r: r["contextual_class"] == "contextual_mixed"),
        "contextual_insufficient": _count(lambda r: r["contextual_class"] == "contextual_insufficient"),
        "final_reliable": _count(lambda r: r["final_reliability_class"] == "reliable_structural_evidence"),
        "region_seed_core": _count(lambda r: r["region_seed_state"] == "core_member"),
        "region_consensus_attached": _count(lambda r: r["region_seed_state"] == "consensus_attached"),
        "region_ambiguous_unassigned": _count(lambda r: r["region_seed_state"] == "ambiguous_unassigned"),
        "region_conflict_boundary": _count(lambda r: r["region_seed_state"] == "conflict_boundary"),
        "region_rejected": _count(lambda r: r["region_seed_state"] == "rejected"),
        "final_region_member": _count(lambda r: r["final_region_id"] >= 0),
    }

    # Dominant single-gate failure histogram among representatives whose
    # intrinsic evidence was reliable but contextual class was NOT consistent
    # (i.e. exactly the "B: intrinsic ok, contextual collapses" population).
    dominant_failure_histogram: dict[str, int] = {}
    for row in rows:
        if row["intrinsic_class"] == INTRINSIC_RELIABLE and row["contextual_class"] != CONTEXTUAL_CONSISTENT:
            for gate_name in row["all_failed_gates"]:
                dominant_failure_histogram[gate_name] = dominant_failure_histogram.get(gate_name, 0) + 1
    single_failed_gate_count = sum(1 for row in rows if len(row["all_failed_gates"]) == 1)
    multi_failed_gate_count = sum(1 for row in rows if len(row["all_failed_gates"]) > 1)

    return {
        "snapshot": snapshot_name, "iteration": iteration, "cap": cap,
        "waterfall": waterfall,
        "dominant_failure_histogram_among_intrinsic_reliable_contextual_not_consistent": dominant_failure_histogram,
        "representatives_with_exactly_one_failed_gate": single_failed_gate_count,
        "representatives_with_multiple_failed_gates": multi_failed_gate_count,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snapshot-name", default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    args = parser.parse_args()

    snapshot_name = args.snapshot_name or args.checkpoint.parent.name
    iteration = args.iteration if args.iteration is not None else int(args.checkpoint.parent.name)
    result = trace(args.checkpoint, args.cap, args.device, snapshot_name, iteration)

    rows = result.pop("rows")
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    summary_text = json.dumps(result, indent=2)
    if args.output_summary is not None:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
