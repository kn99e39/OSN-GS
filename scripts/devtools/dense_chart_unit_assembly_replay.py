"""Worklog 83: chart-scale topology/assembly replay over worklog 82 micro-components.

Worklog 82 built evidence-scale surface-consistency MICRO-components inside
each region. Real result: 364 components across 7 regions, 16 reached
`valid_supported` -- proof coherent surface-consistent evidence exists -- but
91% ended `no_chart`, median size 3-6 points. This script treats those
micro-components as conservative atomic support and inserts a NEW
component-level assembly stage (`torch_dense_chart_unit_assembly.py`) between
them and chart materialization:

    region-owned evidence
    -> worklog 82 micro-components
    -> worklog 83 component-level assembly (THIS)
    -> per chart-unit worklog 80 dense chart support
    -> worklog 79 coverage contract
    -> PCA-UV (worklog 81 confirmed no better alternative)
    -> 6x6 NURBS fit
    -> held-out evaluation

All of worklog 80/81/82's own primitives are reused unmodified; only the
NEW assembly stage is worklog 83's actual contribution.
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
from osn_gs.surface.torch_dense_chart_unit_assembly import (
    RELATION_ACCEPTED,
    RELATION_AMBIGUOUS,
    RELATION_CREASE_VETOED,
    build_chart_unit_assembly,
)
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


def evaluate_chart(boundary: torch.Tensor, evidence: torch.Tensor, label: str) -> dict:
    """Identical downstream chain worklog 80/81/82 already used (unmodified)."""
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

        # ---- worklog-80 sparse macro-topology arcs, for crease veto (unchanged
        # inputs, reused at both the micro-component AND the assembly stage) --
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

        # ---- worklog 82 micro-components (unchanged) -----------------------
        consistency = build_dense_surface_consistency_components(
            region_id, evidence, covariance=evidence_covariance,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        row["micro_components"] = {
            "component_count": len(consistency.components),
            "component_sizes": [len(c.member_indices) for c in consistency.components],
            "unresolved_fraction": len(consistency.unresolved_indices) / max(1, consistency.point_count),
        }

        if not consistency.components:
            row["chart_units"] = []
            rows.append(row)
            continue

        micro_components = tuple(c.member_indices for c in consistency.components)
        non_manifold_flags = tuple(c.non_manifold_suspected for c in consistency.components)
        full_evidence_scale = _median_nn(evidence)

        # ---- worklog 83 assembly (NEW) --------------------------------------
        assembly = build_chart_unit_assembly(
            region_id, evidence, covariance=evidence_covariance,
            micro_components=micro_components, non_manifold_flags=non_manifold_flags,
            full_evidence_spacing=full_evidence_scale,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        edge_relation_counts: dict[str, int] = {}
        for edge in assembly.edges:
            edge_relation_counts[edge.relation] = edge_relation_counts.get(edge.relation, 0) + 1
        row["assembly"] = {
            "micro_component_count": assembly.micro_component_count,
            "chart_unit_count": assembly.chart_unit_count,
            "chart_unit_sizes": [len(u.member_indices) for u in assembly.chart_units],
            "chart_unit_micro_component_counts": [len(u.micro_component_indices) for u in assembly.chart_units],
            "edge_relation_counts": edge_relation_counts,
            "excluded_non_manifold_component_count": assembly.excluded_non_manifold_component_count,
        }

        # ---- per chart-unit materialization: worklog 80 -> worklog 79 -> fit
        charts = []
        for unit_idx, unit in enumerate(assembly.chart_units):
            member_local = list(unit.member_indices)
            unit_evidence = evidence[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            unit_covariance = evidence_covariance[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            unit_stable_ids = [stable_ids[indices[i]] for i in member_local]

            chart_record = {
                "unit_index": unit_idx,
                "micro_component_count": len(unit.micro_component_indices),
                "member_count": len(member_local),
            }
            if len(member_local) < 4:
                chart_record["classification"] = "no_chart"
                chart_record["reason"] = "chart_unit_too_small"
                charts.append(chart_record)
                continue

            unit_normals = extract_covariance_frame(unit_covariance).normal_candidate
            support = extract_dense_boundary_support(unit_evidence, unit_normals, unit_stable_ids)
            dense = support.candidates
            if not dense:
                chart_record["classification"] = "no_chart"
                chart_record["reason"] = "no_dense_boundary_support"
                charts.append(chart_record)
                continue

            centered = unit_evidence - unit_evidence.mean(0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            axis_u, axis_v = vh[0], vh[1]
            origin = unit_evidence.mean(0)

            if chart is not None and chart.ordered_node_ids and len(
                [n for n in chart.ordered_node_ids if n in rep_index]
            ) >= 3:
                nodes_with_pos = [n for n in chart.ordered_node_ids if n in rep_index]
                sparse_positions = torch.stack([rep_pos[rep_index[n]] for n in nodes_with_pos], dim=0)
                kinds = [s.segment_kind for s in chart.segments][: len(nodes_with_pos)]
            else:
                chart_record["classification"] = "no_chart"
                chart_record["reason"] = "no_sparse_macro_topology_for_arc_typing"
                charts.append(chart_record)
                continue

            dense_positions = torch.tensor([c.position for c in dense], dtype=points.dtype, device=points.device)
            chart_support = build_dense_chart_support(
                region_id, sparse_positions, kinds, [c.stable_id for c in dense], dense_positions, unit_evidence,
                axis_u=axis_u, axis_v=axis_v, origin=origin,
                full_evidence_spacing=support.full_evidence_scale,
            )
            chart_record["dense_chart_support_state"] = chart_support.state
            if chart_support.state != CHART_STATE_MATERIALIZED:
                chart_record["classification"] = "no_chart"
                chart_record["reason"] = f"dense_chart_support_state={chart_support.state}"
                charts.append(chart_record)
                continue

            evaluation = evaluate_chart(chart_support.ordered_positions, unit_evidence, f"region{region_id}_unit{unit_idx}")
            chart_record.update(evaluation)
            charts.append(chart_record)

        row["chart_units"] = charts
        rows.append(row)
    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val83/dense_chart_unit_assembly_replay.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}\n", flush=True)

    print(f"{'reg':>3} {'evid':>5} {'micro':>5} {'units':>5} {'unresolved%':>11} | edge relations | chart classes")
    for row in report["regions"]:
        if "skip_reason" in row:
            print(f"{row['region']:>3} {row['owned_evidence']:>5} skipped: {row['skip_reason']}")
            continue
        mc = row.get("micro_components", {})
        asm = row.get("assembly", {})
        classes = [c.get("classification", "?") for c in row.get("chart_units", [])]
        from collections import Counter
        cnt = Counter(classes)
        print(
            f"{row['region']:>3} {row['owned_evidence']:>5} {mc.get('component_count', 0):>5} "
            f"{asm.get('chart_unit_count', 0):>5} {(mc.get('unresolved_fraction') or 0) * 100:>10.1f}% | "
            f"{asm.get('edge_relation_counts', {})} | {dict(cnt)}"
        )

    all_charts = [c for row in report["regions"] for c in row.get("chart_units", [])]
    counts: dict[str, int] = {}
    for c in all_charts:
        key = c.get("classification", "?")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nALL-CHART-UNIT totals across 7 regions: {counts}")
    print(f"total chart units evaluated: {len(all_charts)}")


if __name__ == "__main__":
    main()
