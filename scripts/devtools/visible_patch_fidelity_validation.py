"""Worklog 66: Visible Patch Coverage and Fidelity Validation.

Per-patch and scene-level geometric-fidelity metrics for the materialized
visible NURBS patches (both physical `eligible_closed_boundary` and worklog
61's parametric-chart path), comparing baseline_compatible OSN-GS vs
Graphdeco baseline vs covariance_knn (over-segmentation reference only) at
iteration 2900/3100.

Read-only analysis. No algorithm/threshold is changed here or in anything it
imports; classification thresholds below are borrowed from EXISTING
production conventions (cited inline), never fit to this round's results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_parametric_diagnostics import compute_orientation_consistency, compute_parametric_jacobian_metrics

import fixed_loader_replay_analysis as osn_ckpt_analysis  # noqa: E402
import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

# Thresholds borrowed from EXISTING production conventions -- never tuned to
# this round's results (see worklog 66 for citations):
#   - RegionFormationConfig.core_region_typical_min_size == 4 (already the
#     codebase's own "core_region/stable_region vs small_review_region" cut).
#   - RegionFormationConfig.local_backbone_max_normalized_distance == 4.0
#     (already the codebase's own normalized-distance sanity bound used by
#     the bridge/consensus backbone check).
UNDER_SUPPORTED_MIN_EVIDENCE = 4
EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND = 4.0
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
    """Median nearest-neighbor spacing within a patch's own observed evidence."""
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    value = float(nn.median())
    return value if value > 0 else 1e-6


def _sample_surface(surface, resolution: int = SURFACE_SAMPLE_RESOLUTION):
    device = surface.control_grid.device
    dtype = surface.control_grid.dtype
    grid = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(grid, grid, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    points, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    normals = torch.cross(deriv_u, deriv_v, dim=1)
    return points, deriv_u, deriv_v, normals


def _surface_area(sample_points: torch.Tensor, resolution: int) -> float:
    pts = sample_points.reshape(resolution, resolution, 3)
    p00, p01, p10, p11 = pts[:-1, :-1], pts[:-1, 1:], pts[1:, :-1], pts[1:, 1:]
    a1 = 0.5 * torch.cross(p01 - p00, p10 - p00, dim=-1).norm(dim=-1)
    a2 = 0.5 * torch.cross(p11 - p01, p10 - p01, dim=-1).norm(dim=-1)
    return float((a1 + a2).sum())


def _classify_patch(
    supporting_count: int, forward_norm: dict, backward_norm: dict,
    jacobian: dict, self_intersecting: bool,
) -> tuple[str, list[str]]:
    if self_intersecting:
        return "unsafe_geometry", ["self_intersection_detected"]
    if jacobian["near_degenerate_count"] > 0:
        return "unsafe_geometry", [f"jacobian_near_degenerate_count={jacobian['near_degenerate_count']}"]
    if supporting_count < UNDER_SUPPORTED_MIN_EVIDENCE:
        return "under_supported", [f"supporting_count={supporting_count}<{UNDER_SUPPORTED_MIN_EVIDENCE}"]
    bwd_p95 = backward_norm.get("p95") or 0.0
    fwd_p95 = forward_norm.get("p95") or 0.0
    if bwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND:
        return "extrapolative", [f"surface_to_evidence_p95_normalized={bwd_p95:.2f}>{EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND}"]
    if fwd_p95 > EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND:
        return "extrapolative", [f"evidence_to_surface_p95_normalized={fwd_p95:.2f}>{EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND}"]
    return "valid_supported", []


def _analyze_patch(item, chart_type: str, region_lookup: dict, param_chart_lookup: dict) -> dict:
    inp = item.input
    surface = item.surface
    boundary_pts = inp.ordered_boundary_points
    interior_pts = inp.interior_points
    has_interior = interior_pts is not None and int(interior_pts.shape[0]) > 0
    evidence_raw = torch.cat((boundary_pts, interior_pts), dim=0) if has_interior else boundary_pts
    n_boundary = int(boundary_pts.shape[0])
    n_interior = int(interior_pts.shape[0]) if has_interior else 0

    # Worklog 66 implementation defect (found via this validation, not a
    # production defect -- this analysis script's own bug): for very small
    # regions, `interior_points` and `ordered_boundary_points` are the exact
    # same underlying Gaussians (a 3-member region has no distinct
    # boundary/interior split). Concatenating them verbatim puts EXACT
    # duplicate rows into `evidence`, which collapses the nearest-neighbor
    # `_local_evidence_scale` to 0 (clamped to the 1e-6 floor) and explodes
    # every normalized distance into meaningless five/six-digit values.
    # De-duplicating before any distance/scale computation is required for
    # the metric to mean anything; `supporting_evidence_count` is likewise
    # reported on the de-duplicated set so the same physical Gaussian is not
    # double-counted just because it appears in both lists.
    evidence = torch.unique(evidence_raw, dim=0)
    supporting_count = int(evidence.shape[0])

    region = region_lookup.get(inp.source_region_id)
    region_accepted_count = len(region.member_ids) if region is not None else None

    sample_points, deriv_u, deriv_v, normals = _sample_surface(surface)
    finite = bool(torch.isfinite(sample_points).all())

    scale = _local_evidence_scale(evidence)
    d_fwd = torch.cdist(evidence, sample_points).min(dim=1).values
    d_bwd = torch.cdist(sample_points, evidence).min(dim=1).values
    fwd_pct = _percentiles(d_fwd.detach().cpu().numpy())
    bwd_pct = _percentiles(d_bwd.detach().cpu().numpy())
    fwd_norm = {k: (v / scale if v is not None else None) for k, v in fwd_pct.items()}
    bwd_norm = {k: (v / scale if v is not None else None) for k, v in bwd_pct.items()}

    area = _surface_area(sample_points, SURFACE_SAMPLE_RESOLUTION)
    jacobian = compute_parametric_jacobian_metrics(deriv_u, deriv_v, scale=scale)
    orientation = compute_orientation_consistency(normals)

    world_points = [tuple(float(v) for v in row) for row in boundary_pts.detach().cpu().tolist()]
    si_report = validate_simple_closed_loop(world_points) if n_boundary >= 3 else None
    self_intersecting = (si_report is not None) and (not si_report.is_simple_polygon)
    if not finite:
        self_intersecting = True  # non-finite surface is always unsafe, route through the same gate

    if chart_type == "physical":
        provenance = {"physical_termination": 1.0}
        partition_seam_ratio = 0.0
    else:
        chart = param_chart_lookup.get(inp.source_region_id)
        if chart is not None:
            counts = chart.segment_kind_counts()
            total = sum(counts.values()) or 1
            provenance = {k: v / total for k, v in counts.items()}
            partition_seam_ratio = counts.get("partition_seam", 0) / total
        else:
            provenance, partition_seam_ratio = {}, None

    classification, reasons = _classify_patch(supporting_count, fwd_norm, bwd_norm, jacobian, self_intersecting)

    covered_ids = frozenset(
        list(inp.ordered_boundary_point_ids) + (list(inp.interior_reliable_point_ids) if has_interior else [])
    )

    return {
        "chart_type": chart_type,
        "source_region_id": inp.source_region_id,
        "region_status": inp.region_status,
        "boundary_role_scope": inp.boundary_role_scope,
        "supporting_evidence_count": supporting_count,
        "boundary_point_count": n_boundary,
        "interior_point_count": n_interior,
        "region_accepted_evidence_count": region_accepted_count,
        "local_evidence_scale": scale,
        "point_to_surface_distance": fwd_pct,
        "point_to_surface_distance_normalized": fwd_norm,
        "surface_to_evidence_distance": bwd_pct,
        "surface_to_evidence_distance_normalized": bwd_norm,
        "patch_area": area,
        "jacobian": {
            "min_area_jacobian": jacobian["min_area_jacobian"],
            "min_singular_value_normalized": jacobian["min_jacobian_singular_value_normalized"],
            "condition_mean": jacobian["jacobian_condition_mean"],
            "condition_p95": jacobian["jacobian_condition_p95"],
            "near_degenerate_count": jacobian["near_degenerate_count"],
        },
        "orientation_flip_count": orientation["orientation_flip_count"],
        "orientation_valid_sample_count": orientation["valid_sample_count"],
        "self_intersecting": self_intersecting,
        "boundary_provenance": provenance,
        "partition_seam_ratio": partition_seam_ratio,
        "classification": classification,
        "classification_reasons": reasons,
        "_sample_points": sample_points,
        "_boundary_points": boundary_pts,
        "_interior_points": interior_pts if has_interior else None,
        "_covered_ids": covered_ids,
    }


def _detect_overlaps(patches: list[dict]) -> list[tuple[int, int, float, str]]:
    pairs = []
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            a, b = patches[i], patches[j]
            union = a["_covered_ids"] | b["_covered_ids"]
            inter = a["_covered_ids"] & b["_covered_ids"]
            jaccard = (len(inter) / len(union)) if union else 0.0
            if jaccard > DUPLICATE_ID_JACCARD_THRESHOLD:
                pairs.append((i, j, jaccard, "id_overlap"))
                continue
            local_scale = min(a["local_evidence_scale"], b["local_evidence_scale"])
            d = torch.cdist(a["_sample_points"], b["_sample_points"])
            close_a = (d.min(dim=1).values < local_scale).float().mean().item()
            close_b = (d.min(dim=0).values < local_scale).float().mean().item()
            spatial = max(close_a, close_b)
            if spatial > DUPLICATE_SPATIAL_OVERLAP_FRACTION:
                pairs.append((i, j, spatial, "spatial_overlap"))
    return pairs


def analyze_condition(model, cap: int, source_points_for_scale: torch.Tensor | None, device: str, label: str) -> dict:
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction
    region_lookup = {r.region_id: r for r in construction.surface_regions.regions}
    param_chart_lookup = {c.region_id: c for c in construction.region_parametric_chart_boundaries}

    patches = []
    for item in construction.materialized_visible_nurbs_surfaces:
        if item.surface is None:
            continue
        patches.append(_analyze_patch(item, "physical", region_lookup, param_chart_lookup))
    for item in construction.materialized_parametric_chart_surfaces:
        if item.surface is None:
            continue
        patches.append(_analyze_patch(item, "parametric", region_lookup, param_chart_lookup))

    overlap_pairs = _detect_overlaps(patches)
    overlapping_indices = {i for i, j, _, _ in overlap_pairs} | {j for i, j, _, _ in overlap_pairs}
    for idx in overlapping_indices:
        if patches[idx]["classification"] != "unsafe_geometry":
            patches[idx]["classification"] = "duplicate_or_overlapping"
            patches[idx]["classification_reasons"] = patches[idx]["classification_reasons"] + ["overlaps_another_materialized_patch"]

    total_accepted_evidence = sum(len(r.member_ids) for r in construction.surface_regions.regions)
    covered_ids: set = set()
    for p in patches:
        covered_ids |= p["_covered_ids"]
    # region member_ids are representative-local indices (ints); covered_ids
    # come from the same index space via ordered_boundary_point_ids/
    # interior_reliable_point_ids, so direct set membership is valid.
    all_member_ids: set = set()
    for r in construction.surface_regions.regions:
        all_member_ids |= set(r.member_ids)
    gap_ids = all_member_ids - covered_ids
    gap_fraction = (len(gap_ids) / len(all_member_ids)) if all_member_ids else None

    by_classification = Counter(p["classification"] for p in patches)
    by_chart_type = Counter(p["chart_type"] for p in patches)

    exportable_patches = [{k: v for k, v in p.items() if not k.startswith("_")} for p in patches]

    return {
        "label": label,
        "region_count": len(construction.surface_regions.regions),
        "patch_count": len(patches),
        "by_classification": dict(by_classification),
        "by_chart_type": dict(by_chart_type),
        "total_patch_area": sum(p["patch_area"] for p in patches),
        "total_accepted_evidence": total_accepted_evidence,
        "covered_evidence_count": len(covered_ids),
        "gap_evidence_count": len(gap_ids),
        "gap_fraction": gap_fraction,
        "overlap_pair_count": len(overlap_pairs),
        "overlap_pairs": [{"i": i, "j": j, "overlap": o, "kind": k} for i, j, o, k in overlap_pairs],
        "patches": exportable_patches,
        "_patches_raw": patches,
    }


def _load_osn_checkpoint(ckpt_dir: Path, device: str):
    payload = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    from osn_gs.gaussian.torch_model import TorchGaussianModel
    model = TorchGaussianModel(sh_degree=osn_ckpt_analysis._sh_degree_from_checkpoint(raw), device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--covariance_knn_run_dir", type=Path, default=Path("output/extent_ab/val64/covariance_knn"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val66/patch_fidelity_report.json"))
    parser.add_argument("--viz_out", type=Path, default=Path("output/extent_ab/val66/patch_fidelity_viz.pt"))
    args = parser.parse_args()
    device = "cuda"

    report: dict = {}
    viz_bundle: dict = {}
    for it in args.iterations:
        for cond, label in (("baseline_compatible", "baseline_compatible"), ("covariance_knn", "covariance_knn")):
            run_dir = args.baseline_compatible_run_dir if cond == "baseline_compatible" else args.covariance_knn_run_dir
            ckpt = run_dir / str(it)
            if not (ckpt / "checkpoint.pt").exists():
                continue
            print(f"analyzing {label}@{it} ...", flush=True)
            model = _load_osn_checkpoint(ckpt, device)
            result = analyze_condition(model, args.cap, None, device, f"{label}@{it}")
            report.setdefault(label, {})[str(it)] = {k: v for k, v in result.items() if k != "_patches_raw"}
            viz_bundle[f"{label}@{it}"] = result["_patches_raw"]

        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            print(f"analyzing baseline@{it} ...", flush=True)
            model = baseline_ply_analysis.load_baseline_ply_as_model(ply, device)
            result = analyze_condition(model, args.cap, None, device, f"baseline@{it}")
            report.setdefault("baseline", {})[str(it)] = {k: v for k, v in result.items() if k != "_patches_raw"}
            viz_bundle[f"baseline@{it}"] = result["_patches_raw"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)

    # Save raw tensors (sample/boundary/interior points) for the visualization
    # step, keyed by condition@iteration -> list of patch dicts (with tensors
    # moved to CPU so the file loads without CUDA).
    cpu_bundle = {}
    for key, patches in viz_bundle.items():
        cpu_patches = []
        for p in patches:
            cpu_patches.append({
                "chart_type": p["chart_type"], "source_region_id": p["source_region_id"],
                "classification": p["classification"], "classification_reasons": p["classification_reasons"],
                "sample_points": p["_sample_points"].detach().cpu(),
                "boundary_points": p["_boundary_points"].detach().cpu(),
                "interior_points": p["_interior_points"].detach().cpu() if p["_interior_points"] is not None else None,
                "point_to_surface_distance_normalized": p["point_to_surface_distance_normalized"],
                "surface_to_evidence_distance_normalized": p["surface_to_evidence_distance_normalized"],
            })
        cpu_bundle[key] = cpu_patches
    args.viz_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_bundle, args.viz_out)
    print(f"wrote {args.viz_out}", flush=True)


if __name__ == "__main__":
    main()
