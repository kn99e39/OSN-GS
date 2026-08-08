"""Worklog 80: before/after replay of the redesigned parametric chart representation.

BEFORE = worklog 78/79 production: the sparse accepted representative cycle is
the chart's geometric boundary, paired with region-owned full evidence.
AFTER  = worklog 80 redesign: the sparse cycle supplies only the cyclic order
and the typed frontier provenance, while the geometry comes from the region's
own dense boundary-support candidates (worklog 77 predicate, unmodified).

Both sides then run the identical downstream chain so the comparison is
like-for-like:
    chart-domain coverage -> parameterization validity -> 6x6 NURBS fitting
    -> held-out / full-evidence evaluation.

Region formation, ownership, covariance_normal, full_evidence_spacing, the
worklog 77 correction, and the physical-boundary connectivity path are all
untouched. Multiple charts are attempted only where the region's OWN accepted
topology proves the separation (disjoint 2-core components).
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
    STATE_MATERIALIZED,
    build_dense_chart_support,
    independent_chart_components,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_region_owned_full_evidence import evidence_outside_chart_domain_fraction
from osn_gs.surface.torch_single_chart_uv_validity import (
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


def _holdout(evidence: torch.Tensor):
    if int(evidence.shape[0]) < 8:
        return evidence, evidence[:0]
    uv = pca_parameterize_points(evidence)
    cu = (uv[:, 0] * HOLDOUT_K).clamp(0, HOLDOUT_K - 1e-6).floor().long()
    cv = (uv[:, 1] * HOLDOUT_K).clamp(0, HOLDOUT_K - 1e-6).floor().long()
    mask = ((cu + cv) % 2) == 0
    if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
        return evidence, evidence[:0]
    return evidence[~mask], evidence[mask]


def _sample(surface, resolution=SAMPLE_RESOLUTION):
    device, dtype = surface.control_grid.device, surface.control_grid.dtype
    g = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(g, g, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    pts, du, dv = surface.evaluate_with_derivatives(uv)
    return pts, du, dv, torch.cross(du, dv, dim=1)


def evaluate_chart(boundary: torch.Tensor, evidence: torch.Tensor, label: str) -> dict:
    """The identical downstream chain for BEFORE and AFTER."""
    scale = _median_nn(evidence)
    outside = evidence_outside_chart_domain_fraction(boundary, evidence)
    record = {
        "label": label,
        "boundary_vertex_count": int(boundary.shape[0]),
        "evidence_count": int(evidence.shape[0]),
        "boundary_extent": float((boundary - boundary.mean(0)).norm(dim=1).max() * 2) if int(boundary.shape[0]) > 1 else 0.0,
        "evidence_extent": float((evidence - evidence.mean(0)).norm(dim=1).max() * 2) if int(evidence.shape[0]) > 1 else 0.0,
        "evidence_outside_domain_fraction": outside,
    }
    if outside is not None and outside > 0.5:
        record["classification"] = "chart_domain_does_not_cover_evidence"
        return record

    if int(evidence.shape[0]) >= 4:
        uv = pca_parameterize_points(evidence)
        dup = uv_duplicate_diagnostics(uv)
        nb = neighborhood_preservation(evidence, uv, k=8)
        tri = uv_triangulation_diagnostics(evidence, uv)
        record["uv_validity"] = {
            "uv_near_collision_count": dup["uv_near_collision_count"],
            "neighborhood_preservation_mean": nb["neighborhood_preservation_mean"],
            "triangle_fold_fraction": (tri["triangle_fold_count"] / tri["triangle_total_count"]) if tri["triangle_total_count"] else None,
        }

    train, held = _holdout(evidence)
    try:
        surface, _ = fit_torch_visible_surface_lsq(
            torch.cat((boundary, train), dim=0), resolution_u=BASE_GRID, resolution_v=BASE_GRID,
            degree_u=2, degree_v=2,
        )
    except Exception as exc:  # noqa: BLE001
        record["classification"] = "fit_failed"
        record["fit_error"] = type(exc).__name__
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
        "train_surface_to_evidence": err(train),
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
    materialized = {i.input.source_region_id: i for i in construction.materialized_parametric_chart_surfaces}

    cluster = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _ = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)

    rows = []
    for region in regions.regions:
        region_id = region.region_id
        indices = owned.get(region_id, [])
        row = {"region": region_id, "owned_evidence": len(indices)}
        if len(indices) < 4:
            row["after"] = {"classification": "no_chart", "reason": "insufficient_owned_evidence"}
            rows.append(row)
            continue
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        members = [s for s in region.member_ids if s in rep_index]
        accepted = list(region.internal_accepted_edge_ids)
        components = independent_chart_components(members, accepted)
        row["independent_chart_components"] = len(components)

        # ---------------- BEFORE: sparse representative cycle as geometry ----
        item = materialized.get(region_id)
        chart = chart_by_region.get(region_id)
        if item is not None and item.surface is not None:
            row["before"] = evaluate_chart(item.input.ordered_boundary_points, evidence, "before")
            row["before"]["chart_status"] = chart.status if chart else ""
        else:
            row["before"] = {
                "classification": "no_chart",
                "reason": chart.reason if chart else "no_chart_record",
                "chart_status": chart.status if chart else "",
            }

        # ---------------- AFTER: dense evidence-backed chart support ---------
        normals = extract_covariance_frame(covariance[selector]).normal_candidate
        support = extract_dense_boundary_support(evidence, normals, [stable_ids[i] for i in indices])
        dense = support.candidates
        frame = frame_by_region.get(region_id)
        if not dense or frame is None or chart is None or not chart.ordered_node_ids:
            row["after"] = {
                "classification": "no_chart",
                "reason": ("no_dense_boundary_support" if not dense else
                           "no_tangent_frame" if frame is None else
                           "no_sparse_chart_cycle"),
                "dense_candidate_count": len(dense),
                "sparse_chart_status": chart.status if chart else "",
            }
            rows.append(row)
            continue

        sparse_positions = torch.stack([
            rep_pos[rep_index[node]] for node in chart.ordered_node_ids if node in rep_index
        ], dim=0)
        kinds = [s.segment_kind for s in chart.segments]
        dense_positions = torch.tensor(
            [c.position for c in dense], dtype=points.dtype, device=points.device,
        )
        origin = rep_pos[rep_index[frame.gaussian_id]]
        result = build_dense_chart_support(
            region_id, sparse_positions, kinds, [c.stable_id for c in dense], dense_positions, evidence,
            axis_u=frame.tangent_axis_0, axis_v=frame.tangent_axis_1, origin=origin,
            full_evidence_spacing=support.full_evidence_scale,
        )
        after = {
            "dense_support_state": result.state,
            "dense_candidate_count": len(dense),
            "sparse_topology_node_count": result.sparse_topology_node_count,
            "chart_vertex_count": len(result.ordered_ids),
            "arc_support_counts": list(result.arc_support_counts),
            "unsupported_arc_count": result.unsupported_arc_count,
            "reasons": list(result.reasons),
        }
        if result.state != STATE_MATERIALIZED:
            after["classification"] = "no_chart"
            after["evidence_outside_domain_fraction"] = result.evidence_outside_domain_fraction
        else:
            after.update(evaluate_chart(result.ordered_positions, evidence, "after"))
        row["after"] = after
        rows.append(row)
    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val80/dense_chart_replay.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}\n", flush=True)

    print(f"{'reg':>3} {'evid':>5} | {'BEFORE bnd':>10} {'out%':>6} {'p95':>7} {'class':<34} | "
          f"{'AFTER bnd':>9} {'out%':>6} {'p95':>7} {'class':<24} {'nbp':>5} {'fold':>6}")
    for row in report["regions"]:
        b = row.get("before", {})
        a = row.get("after", {})
        bo = b.get("evidence_outside_domain_fraction")
        ao = a.get("evidence_outside_domain_fraction")
        uv = a.get("uv_validity") or {}
        print(
            f"{row['region']:>3} {row['owned_evidence']:>5} | "
            f"{b.get('boundary_vertex_count', 0):>10} {(bo * 100 if bo is not None else -1):>6.1f} "
            f"{(b.get('extrapolation_p95') or 0):>7.2f} {b.get('classification', '?'):<34} | "
            f"{a.get('chart_vertex_count', 0):>9} {(ao * 100 if ao is not None else -1):>6.1f} "
            f"{(a.get('extrapolation_p95') or 0):>7.2f} {a.get('classification', '?'):<24} "
            f"{(uv.get('neighborhood_preservation_mean') or 0):>5.2f} "
            f"{(a.get('local_fold_fraction') or 0):>6.4f}"
        )

    for side in ("before", "after"):
        counts: dict[str, int] = {}
        for row in report["regions"]:
            key = row.get(side, {}).get("classification", "?")
            counts[key] = counts.get(key, 0) + 1
        print(f"\n{side.upper()} totals: {counts}")


if __name__ == "__main__":
    main()
