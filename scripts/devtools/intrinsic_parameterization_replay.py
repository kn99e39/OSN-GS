"""Worklog 81: PCA-UV vs intrinsic boundary-conditioned parameterization replay.

Worklog 80 fixed chart-domain coverage (0/5 -> 4/5 real regions) by separating
sparse topology (order + provenance) from dense evidence-backed geometry. All
4 passing charts remained `extrapolative` and their raw evidence showed
21-36% UV-adjacent triangle normal-sign disagreement under
`pca_parameterize_points` -- a single global affine projection.

This script re-fits the SAME 4 real chart domains (regions 0/1/2/3,
baseline_compatible@2900) with two parameterizations, through the identical
downstream chain (coverage -> UV validity -> 6x6 NURBS fit -> held-out eval):

  PCA-UV:      `pca_parameterize_points` (worklog 61-80's existing global
               projection, reused completely unmodified).
  INTRINSIC:   `build_intrinsic_boundary_parameterization` (worklog 81, new)
               -- boundary fixed to worklog 80's own validated dense chart
               loop, interior solved via a boundary-conditioned discrete
               harmonic (Tutte) embedding over the region's own kNN graph.

Region formation/ownership, the worklog 80 dense chart support, covariance_
normal, full_evidence_spacing, the worklog 77 predicate, the chart-domain
coverage contract, and the 6x6 NURBS grid are all unchanged. Regions 4/5/6
are not touched (worklog 80 already leaves them as no_chart).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_dense_parametric_chart_support import (
    STATE_MATERIALIZED as CHART_STATE_MATERIALIZED,
    build_dense_chart_support,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_intrinsic_boundary_parameterization import (
    STATE_MATERIALIZED as UV_STATE_MATERIALIZED,
    build_intrinsic_boundary_parameterization,
)
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_region_owned_full_evidence import evidence_outside_chart_domain_fraction
from osn_gs.surface.torch_single_chart_uv_validity import (
    interior_within_boundary,
    neighborhood_preservation,
    uv_duplicate_diagnostics,
    uv_triangulation_diagnostics,
)

BASE_GRID = 6
SAMPLE_RESOLUTION = 24
HOLDOUT_K = 4
EXTRAPOLATION_BOUND = 4.0


def _pct(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p95": None, "max": None}
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _median_nn(points: torch.Tensor) -> float:
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    v = float(d.min(dim=1).values.median())
    return v if v > 0 else 1e-6


def _holdout_mask_from_uv(uv: torch.Tensor) -> torch.Tensor:
    """Deterministic spatial checkerboard holdout in a GIVEN UV layout (worklog
    66-80 convention), so both parameterizations get a like-for-like held-out
    split defined in their OWN chart-relative coordinates."""
    cu = (uv[:, 0] * HOLDOUT_K).clamp(0, HOLDOUT_K - 1e-6).floor().long()
    cv = (uv[:, 1] * HOLDOUT_K).clamp(0, HOLDOUT_K - 1e-6).floor().long()
    return ((cu + cv) % 2) == 0


def _sample(surface, resolution=SAMPLE_RESOLUTION):
    device, dtype = surface.control_grid.device, surface.control_grid.dtype
    g = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(g, g, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    pts, du, dv = surface.evaluate_with_derivatives(uv)
    return pts, du, dv, torch.cross(du, dv, dim=1)


def evaluate_with_uv(
    boundary: torch.Tensor, evidence: torch.Tensor, uv_all: torch.Tensor,
    positions_all: torch.Tensor, label: str,
) -> dict:
    """Shared downstream chain (coverage already checked by caller before
    this is invoked): UV validity -> 6x6 NURBS fit -> held-out evaluation.
    Fits ``positions_all`` at parameter ``uv_all`` -- SAME fitting call
    (`fit_torch_visible_surface_lsq`, unmodified) both parameterizations use,
    only the initial UV differs, so any behavior difference is attributable
    to the parameterization alone."""
    scale = _median_nn(evidence)
    record: dict = {"label": label}

    n = int(positions_all.shape[0])
    boundary_n = int(boundary.shape[0])
    dup = uv_duplicate_diagnostics(uv_all)
    nb = neighborhood_preservation(positions_all, uv_all, k=8)
    tri = uv_triangulation_diagnostics(positions_all, uv_all)
    boundary_uv_ordered = uv_all[:boundary_n]
    interior_uv = uv_all[boundary_n:]
    interior_inside = interior_within_boundary(interior_uv, boundary_uv_ordered)
    record["uv_validity"] = {
        "uv_near_collision_count": dup["uv_near_collision_count"],
        "neighborhood_preservation_mean": nb["neighborhood_preservation_mean"],
        "triangle_fold_fraction": (tri["triangle_fold_count"] / tri["triangle_total_count"]) if tri["triangle_total_count"] else None,
        "interior_outside_boundary_count": interior_inside["interior_outside_boundary_count"],
        "interior_total_count": interior_inside["interior_total_count"],
    }

    holdout_mask = _holdout_mask_from_uv(uv_all)
    train = positions_all[~holdout_mask]
    held = positions_all[holdout_mask]
    train_uv = uv_all[~holdout_mask]
    if int(train.shape[0]) < 8:
        train, train_uv, held = positions_all, uv_all, positions_all[:0]

    try:
        surface, _ = fit_torch_visible_surface_lsq(
            train, resolution_u=BASE_GRID, resolution_v=BASE_GRID,
            degree_u=2, degree_v=2, initial_uv=train_uv,
        )
    except Exception as exc:  # noqa: BLE001
        record["classification"] = "fit_failed"
        record["fit_error"] = f"{type(exc).__name__}: {exc}"
        return record

    pts, du, dv, normals = _sample(surface)
    jac = compute_parametric_jacobian_metrics(du, dv, scale=scale)
    fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)

    def err(subset):
        if int(subset.shape[0]) == 0:
            return _pct(np.array([]))
        return _pct(torch.cdist(pts, subset).min(dim=1).values.detach().cpu().numpy() / scale)

    held_err = err(held)
    full_err = err(evidence)
    p95 = (held_err["p95"] if held_err["p95"] is not None else full_err["p95"]) or 0.0
    record.update({
        "heldout_surface_to_evidence": held_err,
        "full_surface_to_evidence": full_err,
        "extrapolation_p95": p95,
        "jacobian_near_degenerate_count": jac["near_degenerate_count"],
        "local_fold_fraction": fold["local_fold_fraction"],
    })
    unsafe = jac["near_degenerate_count"] > 0 or fold["local_fold_fraction"] > 0.01
    if unsafe:
        record["classification"] = "unsafe_geometry"
    elif p95 <= EXTRAPOLATION_BOUND:
        record["classification"] = "valid_supported"
    else:
        record["classification"] = "extrapolative"
    return record


def analyze(checkpoint: Path, cap: int, device: str) -> dict:
    from osn_gs.gaussian.torch_model import TorchGaussianModel

    payload = torch.load(checkpoint / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    rest = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest:
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
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    points = model.get_xyz.detach()
    stable_ids = list(range(int(points.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            points, covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction
    regions = construction.surface_regions
    rep_ids = bundle.representative_stable_ids
    rep_pos = points[bundle.representative_indices]
    rep_index = {s: i for i, s in enumerate(rep_ids)}

    frames = construct_canonical_region_tangent_frames(
        rep_pos, construction.covariance_frame, construction.reliability, regions, ids=rep_ids,
    )
    frame_by_region = {}
    for frame in frames:
        if frame is not None and frame.region_id not in frame_by_region:
            frame_by_region[frame.region_id] = frame
    chart_by_region = {c.region_id: c for c in construction.region_parametric_chart_boundaries}

    cluster = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _ = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)

    target_regions = {0, 1, 2, 3}  # the worklog 80 coverage-passing regions only
    rows = []
    for region in regions.regions:
        region_id = region.region_id
        if region_id not in target_regions:
            continue
        indices = owned.get(region_id, [])
        row = {"region": region_id, "owned_evidence": len(indices)}
        if len(indices) < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        chart = chart_by_region.get(region_id)
        frame = frame_by_region.get(region_id)
        if chart is None or not chart.ordered_node_ids or frame is None:
            row["skip_reason"] = "no_sparse_chart_or_frame"
            rows.append(row)
            continue

        # Rebuild worklog 80's dense chart support exactly as its own replay does.
        normals = extract_covariance_frame(covariance[selector]).normal_candidate
        support = extract_dense_boundary_support(evidence, normals, [stable_ids[i] for i in indices])
        dense = support.candidates
        if not dense:
            row["skip_reason"] = "no_dense_boundary_support"
            rows.append(row)
            continue
        sparse_positions = torch.stack([
            rep_pos[rep_index[node]] for node in chart.ordered_node_ids if node in rep_index
        ], dim=0)
        kinds = [s.segment_kind for s in chart.segments]
        dense_positions = torch.tensor([c.position for c in dense], dtype=points.dtype, device=points.device)
        origin = rep_pos[rep_index[frame.gaussian_id]]
        chart_support = build_dense_chart_support(
            region_id, sparse_positions, kinds, [c.stable_id for c in dense], dense_positions, evidence,
            axis_u=frame.tangent_axis_0, axis_v=frame.tangent_axis_1, origin=origin,
            full_evidence_spacing=support.full_evidence_scale,
        )
        if chart_support.state != CHART_STATE_MATERIALIZED:
            row["skip_reason"] = f"dense_chart_support_state={chart_support.state}"
            rows.append(row)
            continue

        boundary = chart_support.ordered_positions
        boundary_id_set = set(int(i) for i in chart_support.ordered_ids)
        interior_selector = [i for i in indices if stable_ids[i] not in boundary_id_set]
        interior = points[torch.tensor(interior_selector, dtype=torch.long, device=points.device)] \
            if interior_selector else evidence[:0]

        row["boundary_vertex_count"] = int(boundary.shape[0])
        row["interior_evidence_count"] = int(interior.shape[0])
        row["evidence_outside_domain_fraction"] = evidence_outside_chart_domain_fraction(boundary, evidence)

        # ---------------- PCA-UV (worklog 61-80's existing global projection)
        all_points_pca_order = torch.cat((boundary, interior), dim=0)
        pca_uv_all = pca_parameterize_points(all_points_pca_order)
        row["pca_uv"] = evaluate_with_uv(boundary, evidence, pca_uv_all, all_points_pca_order, "pca_uv")

        # ---------------- INTRINSIC boundary-conditioned harmonic embedding -
        intrinsic = build_intrinsic_boundary_parameterization(boundary, interior)
        row["intrinsic_state"] = intrinsic.state
        row["intrinsic_disconnected_interior_count"] = intrinsic.disconnected_interior_count
        if intrinsic.state == UV_STATE_MATERIALIZED:
            row["intrinsic_uv"] = evaluate_with_uv(
                boundary, evidence, intrinsic.uv, intrinsic.ordered_positions, "intrinsic_uv",
            )
        else:
            row["intrinsic_uv"] = {"classification": "parameterization_failed", "state": intrinsic.state}

        rows.append(row)
    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val81/intrinsic_parameterization_replay.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}\n", flush=True)

    print(f"{'reg':>3} {'evid':>5} {'bnd':>4} | "
          f"{'PCA cls':<20} {'PCA p95':>8} {'PCA nbp':>8} {'PCA fold%':>9} {'PCA jacdeg':>10} | "
          f"{'INT cls':<20} {'INT p95':>8} {'INT nbp':>8} {'INT fold%':>9} {'INT jacdeg':>10}")
    for row in report["regions"]:
        if "skip_reason" in row:
            print(f"{row['region']:>3} {row['owned_evidence']:>5} skipped: {row['skip_reason']}")
            continue
        p = row.get("pca_uv", {})
        i = row.get("intrinsic_uv", {})
        puv = p.get("uv_validity", {})
        iuv = i.get("uv_validity", {})
        print(
            f"{row['region']:>3} {row['owned_evidence']:>5} {row.get('boundary_vertex_count', 0):>4} | "
            f"{p.get('classification', '?'):<20} {(p.get('extrapolation_p95') or 0):>8.2f} "
            f"{(puv.get('neighborhood_preservation_mean') or 0):>8.2f} "
            f"{(puv.get('triangle_fold_fraction') or 0) * 100:>9.1f} "
            f"{(p.get('jacobian_near_degenerate_count') or 0):>10} | "
            f"{i.get('classification', '?'):<20} {(i.get('extrapolation_p95') or 0):>8.2f} "
            f"{(iuv.get('neighborhood_preservation_mean') or 0):>8.2f} "
            f"{(iuv.get('triangle_fold_fraction') or 0) * 100:>9.1f} "
            f"{(i.get('jacobian_near_degenerate_count') or 0):>10}"
        )

    for side in ("pca_uv", "intrinsic_uv"):
        counts: dict[str, int] = {}
        for row in report["regions"]:
            key = row.get(side, {}).get("classification", "?")
            counts[key] = counts.get(key, 0) + 1
        print(f"\n{side.upper()} totals: {counts}")


if __name__ == "__main__":
    main()
