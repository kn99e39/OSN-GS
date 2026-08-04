"""Full-fidelity offline replay of ``reconstruct_visible_after_adc`` against a
saved checkpoint (worklog 135). No trainer, no renderer, no re-training --
loads raw checkpoint tensors into a bare ``TorchGaussianModel`` and calls the
EXACT SAME production code path production training would call after an ADC
commit, so the result is directly comparable to (and should reproduce) the
``nurbs_surface.json`` diagnostics already recorded during real training.

Usage:
    python -m scripts.devtools.replay_long_horizon_snapshot \
        --checkpoint output/osn_gs_scene/3000/checkpoint.pt --cap 2048
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_gaussian_structural_reliability import (
    evaluate_intrinsic_reliability,
    INTRINSIC_RELIABLE,
    INTRINSIC_AMBIGUOUS,
    INTRINSIC_REJECTED,
)


def _sh_degree_from_checkpoint(raw: dict) -> int:
    rest_dim = int(raw["features_rest"].shape[-2])
    # rest_dim = (degree+1)^2 - 1
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim:
        degree += 1
    return degree


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


def replay(checkpoint_path: Path, cap: int, device: str, run_full_construction: bool = True) -> dict:
    model, payload = load_model(checkpoint_path, device)
    report: dict = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_iteration": int(payload["iteration"]),
        "source_gaussian_count": len(model),
        "cap": int(cap),
    }

    with torch.no_grad():
        eligible_mask = (~model.is_uncertain) & (model.surface_owner_kind != 2)
        eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
        report["eligible_count"] = int(eligible_indices.numel())

        points = model.get_xyz.detach()[eligible_indices]
        activated_scale = model.get_scaling.detach()[eligible_indices]
        normalized_rotation = model.get_rotation.detach()[eligible_indices]
        covariance = covariance_from_scale_rotation(activated_scale, normalized_rotation)
        report["raw_state_all_finite"] = bool(torch.isfinite(model._xyz).all().item()
            and torch.isfinite(model._scaling).all().item()
            and torch.isfinite(model._rotation).all().item())
        report["covariance_all_finite"] = bool(torch.isfinite(covariance).all().item())

        opacity = model.get_opacity.detach()[eligible_indices, 0]
        stable_ids = tuple(int(v) for v in model.stable_gaussian_ids[eligible_indices].detach().cpu().tolist())

        # --- Full-cloud intrinsic reliability waterfall (section 4, "Full
        # observed Gaussian intrinsic state") ---
        from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
        t0 = time.perf_counter()
        frame_full = extract_covariance_frame(covariance)
        intrinsic_full = evaluate_intrinsic_reliability(frame_full)
        full_counts = {INTRINSIC_RELIABLE: 0, INTRINSIC_AMBIGUOUS: 0, INTRINSIC_REJECTED: 0}
        for cls in intrinsic_full.intrinsic_class:
            full_counts[cls] += 1
        report["full_cloud_intrinsic_reliability"] = full_counts
        report["full_cloud_eigenframe_seconds"] = time.perf_counter() - t0

        if not run_full_construction:
            return report

        config = TorchPipelineConfig(canonical_construction_max_points=int(cap))
        pipeline = TorchOSNGSPipeline(config, device=device)

        t0 = time.perf_counter()
        bundle = pipeline._construct_canonical_with_full_evidence(points, covariance, opacity, stable_ids)
        report["construction_seconds"] = time.perf_counter() - t0

        sel_diag = bundle.selection.diagnostics
        report["selection_diagnostics"] = {
            "input_gaussian_count": sel_diag.input_gaussian_count,
            "occupied_cell_count": sel_diag.occupied_cell_count,
            "total_candidate_mode_count": sel_diag.total_candidate_mode_count,
            "modes_per_cell_mean": sel_diag.modes_per_cell_mean,
            "modes_per_cell_max": sel_diag.modes_per_cell_max,
            "multi_mode_cell_count": sel_diag.multi_mode_cell_count,
            "selected_representative_count": sel_diag.selected_representative_count,
            "representative_source_count_mean": sel_diag.representative_source_count_mean,
            "representative_source_count_min": sel_diag.representative_source_count_min,
            "representative_source_count_max": sel_diag.representative_source_count_max,
            "selection_mode": sel_diag.selection_mode,
        }

        evidence = bundle.evidence
        support_count = evidence.support_count.float()

        def _summ(tensor: torch.Tensor) -> dict:
            tensor = tensor.float()
            return {
                "mean": float(tensor.mean().item()),
                "median": float(tensor.median().item()),
                "p10": float(tensor.quantile(0.10).item()),
                "p90": float(tensor.quantile(0.90).item()),
                "max": float(tensor.max().item()),
            }

        report["evidence_stats"] = {
            "support_count": _summ(support_count),
            "zero_support_count": int((support_count == 0).sum().item()),
            "mean_spacing": _summ(evidence.mean_spacing),
            "normal_consensus": _summ(evidence.normal_consensus),
            "tangent_residual_mean": _summ(evidence.tangent_residual_mean),
            "competing_mode_mass": _summ(evidence.competing_mode_mass),
            "rejected_neighbor_mass": _summ(evidence.rejected_neighbor_mass),
            "local_density": _summ(evidence.local_density),
        }

        rep_frame = frame_full  # placeholder overwritten below if available
        try:
            rep_indices = bundle.representative_indices
            rep_intrinsic_full_cloud = evaluate_intrinsic_reliability(
                extract_covariance_frame(covariance[rep_indices])
            )
            rep_counts = {INTRINSIC_RELIABLE: 0, INTRINSIC_AMBIGUOUS: 0, INTRINSIC_REJECTED: 0}
            for cls in rep_intrinsic_full_cloud.intrinsic_class:
                rep_counts[cls] += 1
            report["representative_intrinsic_reliability"] = rep_counts
        except Exception as exc:  # noqa: BLE001
            report["representative_intrinsic_reliability_error"] = str(exc)

        construction = bundle.construction
        report["construction_diagnostic_summary"] = dict(construction.diagnostic_summary)
        report["construction_state"] = construction.construction_state

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--intrinsic-only", action="store_true")
    args = parser.parse_args()

    report = replay(args.checkpoint, args.cap, args.device, run_full_construction=not args.intrinsic_only)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
