"""Worklog 67: Region-Owned Full-Evidence Patch Support validation.

For baseline_compatible@2900/3100 and the Graphdeco baseline reference,
compares representative-only vs region-owned full-evidence NURBS fitting
support per materialized patch (production wiring: `osn_gs/core/
torch_pipeline.py::TorchOSNGSPipeline._collect_region_owned_full_evidence_fits`,
`osn_gs/surface/torch_region_owned_full_evidence.py`).

Terminology correction from worklog 66: `validate_simple_closed_loop`
(reused here unmodified) checks that the patch's 2D BOUNDARY LOOP is a
simple, non-self-intersecting polygon in its own tangent plane. It says
nothing about whether the fitted 3D PARAMETRIC SURFACE folds over itself
away from that boundary -- that is a different, harder property this
pipeline does not check anywhere. Fields/labels below say "boundary-loop"
explicitly, never bare "self-intersection".

Explicit 5-way classification priority (checked in this exact order, first
match wins; `duplicate_or_overlapping` is a whole-scene post-pass applied
after every patch's own state is otherwise decided):
  1. unsafe_geometry      -- boundary-loop violation OR Jacobian-degenerate
                              full-evidence fit OR full-evidence fit_failed
  2. duplicate_or_overlapping (post-pass, never downgrades an unsafe_geometry patch)
  3. under_supported      -- full-evidence support count below the minimum
                              contract (MIN_FULL_EVIDENCE_SUPPORT)
  4. extrapolative        -- point-to-surface or surface-to-evidence p95
                              (normalized) exceeds the bound
  5. valid_supported      -- passes all of the above

Worklog 66's per-condition gap percentage must be read as "share of THIS
condition's own accepted representative evidence left uncovered" -- it is
NOT a common ground-truth surface-coverage metric, and is never compared
across conditions here as if it were one (different conditions have
different accepted evidence sets to begin with).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_parametric_diagnostics import compute_orientation_consistency
from osn_gs.surface.torch_region_owned_full_evidence import MIN_FULL_EVIDENCE_SUPPORT

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

UNDER_SUPPORTED_MIN_EVIDENCE = MIN_FULL_EVIDENCE_SUPPORT  # same contract, single source of truth
EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND = 4.0  # RegionFormationConfig.local_backbone_max_normalized_distance, reused (worklog 66)
DUPLICATE_ID_JACCARD_THRESHOLD = 0.3
DUPLICATE_SPATIAL_OVERLAP_FRACTION = 0.5
SURFACE_SAMPLE_RESOLUTION = 24


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def _local_evidence_scale(points: torch.Tensor) -> float:
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    value = float(d.min(dim=1).values.median())
    return value if value > 0 else 1e-6


def _sample_surface(surface, resolution: int = SURFACE_SAMPLE_RESOLUTION):
    device = surface.control_grid.device
    dtype = surface.control_grid.dtype
    grid = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(grid, grid, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    return surface.evaluate(uv)


def _orientation_consistency(surface, resolution: int = SURFACE_SAMPLE_RESOLUTION) -> dict:
    device = surface.control_grid.device
    dtype = surface.control_grid.dtype
    grid = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(grid, grid, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    _, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    normals = torch.cross(deriv_u, deriv_v, dim=1)
    result = compute_orientation_consistency(normals)
    return {
        "orientation_flip_count": result["orientation_flip_count"],
        "orientation_valid_sample_count": result["valid_sample_count"],
    }


def _surface_area(sample_points: torch.Tensor, resolution: int) -> float:
    pts = sample_points.reshape(resolution, resolution, 3)
    p00, p01, p10, p11 = pts[:-1, :-1], pts[:-1, 1:], pts[1:, :-1], pts[1:, 1:]
    a1 = 0.5 * torch.cross(p01 - p00, p10 - p00, dim=-1).norm(dim=-1)
    a2 = 0.5 * torch.cross(p11 - p01, p10 - p01, dim=-1).norm(dim=-1)
    return float((a1 + a2).sum())


def _distance_metrics(evidence: torch.Tensor, surface) -> dict:
    sample_points = _sample_surface(surface)
    scale = _local_evidence_scale(evidence)
    d_fwd = torch.cdist(evidence, sample_points).min(dim=1).values
    d_bwd = torch.cdist(sample_points, evidence).min(dim=1).values
    fwd_pct = _percentiles(d_fwd.detach().cpu().numpy())
    bwd_pct = _percentiles(d_bwd.detach().cpu().numpy())
    fwd_norm = {k: (v / scale if v is not None else None) for k, v in fwd_pct.items()}
    bwd_norm = {k: (v / scale if v is not None else None) for k, v in bwd_pct.items()}
    return {
        "local_evidence_scale": scale,
        "point_to_surface_distance_normalized": fwd_norm,
        "surface_to_evidence_distance_normalized": bwd_norm,
        "patch_area": _surface_area(sample_points, SURFACE_SAMPLE_RESOLUTION),
    }


def _classify(
    boundary_loop_violation: bool, full_evidence_state: str, support_count: int,
    fwd_norm: dict, bwd_norm: dict,
) -> tuple[str, list[str]]:
    """Priority order documented in the module docstring: unsafe_geometry
    first (boundary-loop violation OR degenerate/failed full-evidence fit),
    then under_supported, then extrapolative, else valid_supported.
    duplicate_or_overlapping is applied by the caller as a scene-wide
    post-pass, after this per-patch decision."""

    if boundary_loop_violation:
        return "unsafe_geometry", ["boundary_loop_simple_polygon_violation"]
    if full_evidence_state == "unsafe_geometry":
        return "unsafe_geometry", ["full_evidence_jacobian_degenerate"]
    if full_evidence_state == "fit_failed":
        return "unsafe_geometry", ["full_evidence_fit_failed"]
    if full_evidence_state == "under_supported" or support_count < UNDER_SUPPORTED_MIN_EVIDENCE:
        return "under_supported", [f"full_evidence_support_count={support_count}<{UNDER_SUPPORTED_MIN_EVIDENCE}"]
    bwd_p95 = bwd_norm.get("p95") or 0.0
    fwd_p95 = fwd_norm.get("p95") or 0.0
    if bwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND:
        return "extrapolative", [f"surface_to_evidence_p95_normalized={bwd_p95:.2f}>{EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND}"]
    if fwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND:
        return "extrapolative", [f"evidence_to_surface_p95_normalized={fwd_p95:.2f}>{EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND}"]
    return "valid_supported", []


def analyze_condition(model, cap: int, device: str, label: str) -> dict:
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction

    items = [("physical", item) for item in construction.materialized_visible_nurbs_surfaces if item.surface is not None]
    items += [("parametric", item) for item in construction.materialized_parametric_chart_surfaces if item.surface is not None]

    patches = []
    for chart_type, item in items:
        key = (chart_type, item.input.source_region_id)
        fit = bundle.region_owned_full_evidence_fits.get(key)
        boundary_pts = item.input.ordered_boundary_points
        n_boundary = int(boundary_pts.shape[0])

        world_points = [tuple(float(v) for v in row) for row in boundary_pts.detach().cpu().tolist()]
        si_report = validate_simple_closed_loop(world_points) if n_boundary >= 3 else None
        boundary_loop_violation = (si_report is not None) and (not si_report.is_simple_polygon)

        representative_support_count = n_boundary + (
            int(item.input.interior_points.shape[0]) if item.input.interior_points is not None else 0
        )

        empty_metrics = {
            "local_evidence_scale": None, "point_to_surface_distance_normalized": {},
            "surface_to_evidence_distance_normalized": {}, "patch_area": None,
        }
        empty_orientation = {"orientation_flip_count": None, "orientation_valid_sample_count": None}
        if fit is None:
            full_evidence_state, full_evidence_support_count = "missing", 0
            full_metrics, orientation = empty_metrics, empty_orientation
        else:
            full_evidence_state = fit.state
            full_evidence_support_count = fit.full_evidence_support_count
            if fit.surface is not None:
                # `stable_ids` is `list(range(N))` here (row index == stable
                # id in this analysis context), so the stable IDs are
                # directly usable as row indices -- no O(N) lookup needed.
                if fit.full_evidence_stable_ids:
                    evidence_index = torch.tensor(list(fit.full_evidence_stable_ids), dtype=torch.long, device=boundary_pts.device)
                    full_evidence = torch.cat((boundary_pts, model.get_xyz.detach()[evidence_index]), dim=0)
                else:
                    full_evidence = boundary_pts
                full_metrics = _distance_metrics(full_evidence, fit.surface)
                orientation = _orientation_consistency(fit.surface)
            else:
                full_metrics, orientation = empty_metrics, empty_orientation

        classification, reasons = _classify(
            boundary_loop_violation, full_evidence_state, full_evidence_support_count,
            full_metrics["point_to_surface_distance_normalized"], full_metrics["surface_to_evidence_distance_normalized"],
        )

        patches.append({
            "chart_type": chart_type, "source_region_id": item.input.source_region_id,
            "representative_support_count": representative_support_count,
            "full_evidence_support_count": full_evidence_support_count,
            "full_evidence_state": full_evidence_state,
            "boundary_loop_simple_polygon_violation": boundary_loop_violation,
            "jacobian_near_degenerate_count": fit.jacobian_near_degenerate_count if fit else None,
            "boundary_residual": fit.boundary_residual if fit else None,
            "full_evidence_interior_residual": fit.full_evidence_interior_residual if fit else None,
            **full_metrics,
            **orientation,
            "classification": classification,
            "classification_reasons": reasons,
            "_sample_points": _sample_surface(fit.surface) if (fit and fit.surface is not None) else None,
        })

    # scene-wide duplicate/overlap post-pass (same convention as worklog 66)
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            a, b = patches[i], patches[j]
            if a["_sample_points"] is None or b["_sample_points"] is None:
                continue
            scale_a = a["local_evidence_scale"] or 1e-6
            scale_b = b["local_evidence_scale"] or 1e-6
            local_scale = min(scale_a, scale_b)
            d = torch.cdist(a["_sample_points"], b["_sample_points"])
            close_a = (d.min(dim=1).values < local_scale).float().mean().item()
            close_b = (d.min(dim=0).values < local_scale).float().mean().item()
            if max(close_a, close_b) > DUPLICATE_SPATIAL_OVERLAP_FRACTION:
                for idx in (i, j):
                    if patches[idx]["classification"] != "unsafe_geometry":
                        patches[idx]["classification"] = "duplicate_or_overlapping"
                        patches[idx]["classification_reasons"] = patches[idx]["classification_reasons"] + ["overlaps_another_materialized_patch"]

    total_accepted_evidence = sum(len(r.member_ids) for r in construction.surface_regions.regions)
    from collections import Counter
    by_classification = Counter(p["classification"] for p in patches)

    return {
        "label": label,
        "region_count": len(construction.surface_regions.regions),
        "patch_count": len(patches),
        "by_classification": dict(by_classification),
        "total_accepted_representative_evidence": total_accepted_evidence,
        "patches": [{k: v for k, v in p.items() if not k.startswith("_")} for p in patches],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val67/region_owned_evidence_report.json"))
    args = parser.parse_args()
    device = "cuda"

    from osn_gs.gaussian.torch_model import TorchGaussianModel

    def load_ckpt(ckpt_dir: Path):
        payload = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
        raw = payload["model_raw"]
        rest_dim = int(raw["features_rest"].shape[-2])
        degree = 0
        while (degree + 1) ** 2 - 1 < rest_dim:
            degree += 1
        model = TorchGaussianModel(sh_degree=degree, device=device)
        model.replace_tensors(
            xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
            opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
            uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
            surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
            surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
            stable_gaussian_ids=raw.get("stable_gaussian_ids"),
        )
        return model

    report: dict = {"baseline_compatible": {}, "baseline": {}}
    for it in args.iterations:
        ckpt = args.baseline_compatible_run_dir / str(it)
        if (ckpt / "checkpoint.pt").exists():
            print(f"analyzing baseline_compatible@{it} ...", flush=True)
            model = load_ckpt(ckpt)
            report["baseline_compatible"][str(it)] = analyze_condition(model, args.cap, device, f"baseline_compatible@{it}")
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            print(f"analyzing baseline@{it} ...", flush=True)
            model = baseline_ply_analysis.load_baseline_ply_as_model(ply, device)
            report["baseline"][str(it)] = analyze_condition(model, args.cap, device, f"baseline@{it}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
