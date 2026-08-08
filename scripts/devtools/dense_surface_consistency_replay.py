"""Worklog 82: evidence-scale surface-consistency chart-unit decomposition replay.

Worklog 81 closed the parameterization-only hypothesis: an injectivity-
guaranteed alternative (Tutte embedding) still fails worse than PCA-UV on
worklog 80's 4 coverage-passing charts, and the root cause is that the
region's OWN evidence is not locally flat (local normal disagreement
16.3-37.4%, thickness ratio 17-55%). This script tests whether that
non-flatness resolves into DEFENSIBLE separate chart-units once evidence-
scale surface-consistency (normal alignment + tangent residual + typed
crease/frontier veto, `torch_dense_surface_consistency_components.py`) is
applied INSIDE each of the 7 real regions, instead of assuming region ==
chart as every prior round (61-81) did.

Pipeline per region: region-owned evidence -> dense surface-consistency
components (new) -> per-component dense chart boundary (worklog 80,
unmodified, applied per component instead of per region) -> worklog 79
coverage contract (unmodified) -> PCA-UV (unmodified, worklog 81 confirmed
no better alternative exists) -> 6x6 NURBS fit (unmodified) -> held-out eval.

Region formation/ownership and the sparse representative topology
(macro-topology + typed provenance) are read but never altered.
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
from osn_gs.surface.torch_dense_surface_consistency_components import (
    build_dense_surface_consistency_components,
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


def _local_normal_thickness_stats(positions: torch.Tensor, normal_axis: torch.Tensor, k: int = 10) -> dict:
    """Worklog 81's own diagnostic, reused to report before/after componentization."""
    n = int(positions.shape[0])
    neighbors = min(k, max(1, n - 1))
    if n < 4:
        return {"local_normal_disagreement_fraction": None, "normal_thickness_ratio": None}
    d = torch.cdist(positions, positions)
    d.fill_diagonal_(float("inf"))
    knn = d.topk(neighbors, largest=False, dim=1).indices
    disagreements = []
    for i in range(n):
        neighbor_points = positions[knn[i]]
        centered = neighbor_points - neighbor_points.mean(0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        local_normal = vh[-1]
        disagreements.append(float(local_normal @ normal_axis))
    disagreements_t = torch.tensor(disagreements).abs()
    fraction = float((disagreements_t < 0.5).float().mean())
    centered = positions - positions.mean(0, keepdim=True)
    normal_extent = float((centered @ normal_axis).abs().max())
    tangent = centered - (centered @ normal_axis)[:, None] * normal_axis[None, :]
    tangent_extent = float(tangent.norm(dim=1).max().clamp_min(1e-9))
    return {
        "local_normal_disagreement_fraction": fraction,
        "normal_thickness_ratio": normal_extent / tangent_extent,
    }


def evaluate_chart(boundary: torch.Tensor, evidence: torch.Tensor, label: str) -> dict:
    """Identical downstream chain worklog 80/81 already used (unmodified)."""
    scale = _median_nn(evidence)
    outside = evidence_outside_chart_domain_fraction(boundary, evidence)
    record = {
        "label": label,
        "boundary_vertex_count": int(boundary.shape[0]),
        "evidence_count": int(evidence.shape[0]),
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

    rows = []
    for region in regions.regions:
        region_id = region.region_id
        indices = owned.get(region_id, [])
        row = {"region": region_id, "owned_evidence": len(indices)}
        if len(indices) < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        evidence_covariance = covariance[selector]
        chart = chart_by_region.get(region_id)
        frame = frame_by_region.get(region_id)

        # ---- region-level BEFORE stats (worklog 81's own diagnostic) -------
        if frame is not None:
            region_normal = torch.nn.functional.normalize(
                torch.cross(frame.tangent_axis_0, frame.tangent_axis_1, dim=0), dim=0
            )
            row["before_local_consistency"] = _local_normal_thickness_stats(evidence, region_normal)
        else:
            row["before_local_consistency"] = {"local_normal_disagreement_fraction": None, "normal_thickness_ratio": None}

        # ---- dense surface-consistency componentization (worklog 82, NEW) --
        # Typed crease/frontier arcs come from worklog 80's own sparse-arc
        # segments, projected in the region's canonical frame -- reused
        # provenance, not invented separators.
        arc_starts = arc_ends = None
        arc_kinds: list[str] = []
        if chart is not None and chart.ordered_node_ids and frame is not None:
            nodes_with_pos = [n for n in chart.ordered_node_ids if n in rep_index]
            if len(nodes_with_pos) >= 2:
                sparse_positions = torch.stack([rep_pos[rep_index[n]] for n in nodes_with_pos], dim=0)
                kinds = [s.segment_kind for s in chart.segments][: len(nodes_with_pos)]
                n_arc = int(sparse_positions.shape[0])
                arc_starts = sparse_positions
                arc_ends = torch.stack([sparse_positions[(i + 1) % n_arc] for i in range(n_arc)], dim=0)
                arc_kinds = kinds

        consistency = build_dense_surface_consistency_components(
            region_id, evidence, covariance=evidence_covariance,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        row["dense_components"] = {
            "component_count": len(consistency.components),
            "component_sizes": [len(c.member_indices) for c in consistency.components],
            "non_manifold_suspected": [c.non_manifold_suspected for c in consistency.components],
            "internal_normal_disagreement_fraction": [
                round(c.internal_normal_disagreement_fraction, 4) for c in consistency.components
            ],
            "unresolved_count": len(consistency.unresolved_indices),
            "unresolved_fraction": len(consistency.unresolved_indices) / max(1, consistency.point_count),
            "crease_vetoed_edge_count": consistency.crease_vetoed_edge_count,
        }

        if not consistency.components:
            row["charts"] = []
            rows.append(row)
            continue

        # ---- per-component chart materialization (worklog 80, applied per
        # component instead of per region) -> worklog 79 coverage -> fit ----
        charts = []
        for comp_idx, component in enumerate(consistency.components):
            member_local = list(component.member_indices)
            comp_evidence = evidence[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            comp_covariance = evidence_covariance[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            comp_stable_ids = [stable_ids[indices[i]] for i in member_local]

            comp_record = {
                "component_index": comp_idx,
                "member_count": len(member_local),
                "non_manifold_suspected": component.non_manifold_suspected,
            }
            if len(member_local) < 4:
                comp_record["classification"] = "no_chart"
                comp_record["reason"] = "component_too_small"
                charts.append(comp_record)
                continue
            if component.non_manifold_suspected:
                comp_record["classification"] = "unresolved_non_manifold"
                charts.append(comp_record)
                continue

            comp_normals = extract_covariance_frame(comp_covariance).normal_candidate
            support = extract_dense_boundary_support(comp_evidence, comp_normals, comp_stable_ids)
            dense = support.candidates
            if not dense:
                comp_record["classification"] = "no_chart"
                comp_record["reason"] = "no_dense_boundary_support"
                charts.append(comp_record)
                continue

            # Component-own local tangent frame (PCA over the component's own
            # evidence) -- used purely as the arc-assignment/UV PROJECTION
            # frame worklog 80's build_dense_chart_support needs, never as
            # the chart's geometric extent itself.
            centered = comp_evidence - comp_evidence.mean(0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            axis_u, axis_v = vh[0], vh[1]
            origin = comp_evidence.mean(0)

            if chart is not None and chart.ordered_node_ids and len(
                [n for n in chart.ordered_node_ids if n in rep_index]
            ) >= 3:
                nodes_with_pos = [n for n in chart.ordered_node_ids if n in rep_index]
                sparse_positions = torch.stack([rep_pos[rep_index[n]] for n in nodes_with_pos], dim=0)
                kinds = [s.segment_kind for s in chart.segments][: len(nodes_with_pos)]
            else:
                comp_record["classification"] = "no_chart"
                comp_record["reason"] = "no_sparse_macro_topology_for_arc_typing"
                charts.append(comp_record)
                continue

            dense_positions = torch.tensor([c.position for c in dense], dtype=points.dtype, device=points.device)
            chart_support = build_dense_chart_support(
                region_id, sparse_positions, kinds, [c.stable_id for c in dense], dense_positions, comp_evidence,
                axis_u=axis_u, axis_v=axis_v, origin=origin,
                full_evidence_spacing=support.full_evidence_scale,
            )
            comp_record["dense_chart_support_state"] = chart_support.state
            if chart_support.state != CHART_STATE_MATERIALIZED:
                comp_record["classification"] = "no_chart"
                comp_record["reason"] = f"dense_chart_support_state={chart_support.state}"
                charts.append(comp_record)
                continue

            evaluation = evaluate_chart(chart_support.ordered_positions, comp_evidence, f"region{region_id}_component{comp_idx}")
            comp_record.update(evaluation)

            # ---- after-componentization local consistency (same evidence,
            # now scoped to just this component) ----
            comp_frame = extract_covariance_frame(comp_covariance)
            comp_mean_normal = torch.nn.functional.normalize(comp_frame.normal_candidate.mean(dim=0), dim=0)
            comp_record["after_local_consistency"] = _local_normal_thickness_stats(comp_evidence, comp_mean_normal)
            charts.append(comp_record)

        row["charts"] = charts
        rows.append(row)
    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val82/dense_surface_consistency_replay.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}\n", flush=True)

    print(f"{'reg':>3} {'evid':>5} {'comp':>4} {'unresolved%':>11} | before-nbp-disagree% before-thickratio | charts")
    for row in report["regions"]:
        if "skip_reason" in row:
            print(f"{row['region']:>3} {row['owned_evidence']:>5} skipped: {row['skip_reason']}")
            continue
        dc = row.get("dense_components", {})
        before = row.get("before_local_consistency", {})
        n_comp = dc.get("component_count", 0)
        unresolved_pct = (dc.get("unresolved_fraction") or 0) * 100
        before_disagree = (before.get("local_normal_disagreement_fraction") or 0) * 100
        before_thick = before.get("normal_thickness_ratio") or 0
        chart_classes = [c.get("classification", "?") for c in row.get("charts", [])]
        print(
            f"{row['region']:>3} {row['owned_evidence']:>5} {n_comp:>4} {unresolved_pct:>10.1f}% | "
            f"{before_disagree:>6.1f}% {before_thick:>6.3f} | {chart_classes}"
        )

    all_charts = [c for row in report["regions"] for c in row.get("charts", [])]
    counts: dict[str, int] = {}
    for c in all_charts:
        key = c.get("classification", "?")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nALL-CHART totals across 7 regions: {counts}")
    print(f"total components materialized as charts: {len(all_charts)}")


if __name__ == "__main__":
    main()
