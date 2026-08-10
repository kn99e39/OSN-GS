"""Worklog 84: chart-unit coherence audit + evidence-scale boundary replay.

Worklog 83 assembled Worklog 82 micro-components into chart-scale units:
364 -> 178, 75-93% of region evidence recovered per region into size>=4
candidates, zero unsupported gap bridging audited. But no_chart stayed ~91%
because per-unit chart materialization still relied on Worklog 80's sparse
macro cycle (3-7 representative nodes) to type/geometrically bound units
spanning tens to hundreds of evidence points, and valid_supported DROPPED
16 -> 4 for reasons that were not yet disentangled (legitimate exposure vs.
over-merging).

This script closes both coupled questions in one pipeline:

    region-owned evidence
    -> worklog 82 micro-components (UNCHANGED)
    -> worklog 83 assembly (UNCHANGED -- preserved as the current proposal,
       not tuned toward fit quality)
    -> worklog 84 per-unit coherence audit (NEW: reuses worklog 82's own
       0.15 internal-disagreement bound at assembled-unit scale)
    -> worklog 84 evidence-scale boundary materialization for COHERENT units
       only (NEW: reuses `extract_dense_boundary_support`'s own dense
       connectivity -- worklog 72/76/77, unmodified -- instead of sparse-arc
       assignment; sparse macro arcs are consulted only for typed-segment
       labels and a closing crease-consistency disclosure)
    -> worklog 79 coverage contract (UNCHANGED, applied inside worklog 84's
       module)
    -> PCA-UV (UNCHANGED, worklog 81 confirmed no better alternative)
    -> 6x6 NURBS fit (UNCHANGED)
    -> held-out evaluation (UNCHANGED)

Reports both chart-count and evidence-WEIGHTED metrics per region, and
attributes worklog 83's valid-chart loss (16 -> 4) directly: for every unit
that WAS valid_supported as a worklog-82 micro-component, this script checks
whether the worklog-83-assembled unit containing it was judged coherent or
ambiguous/over-merged, and whether it reached materialization.
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
from osn_gs.surface.torch_chart_unit_evidence_scale_boundary import (
    STATE_MATERIALIZED as BOUNDARY_STATE_MATERIALIZED,
    materialize_chart_unit_boundary,
)
from osn_gs.surface.torch_dense_chart_unit_assembly import build_chart_unit_assembly
from osn_gs.surface.torch_dense_surface_consistency_components import (
    build_dense_surface_consistency_components,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_parametric_jacobian_metrics
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


def evaluate_fit(boundary: torch.Tensor, evidence: torch.Tensor, label: str) -> dict:
    """Identical downstream fit chain worklog 80/81/82/83 already used
    (unmodified): UV validity -> 6x6 NURBS -> held-out. Coverage is already
    checked upstream by `materialize_chart_unit_boundary`."""
    scale = _median_nn(evidence)
    record: dict = {"label": label}

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
        total_evidence = len(indices)
        row = {"region": region_id, "total_evidence": total_evidence}
        if total_evidence < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        evidence_covariance = covariance[selector]
        chart = chart_by_region.get(region_id)
        frame = frame_by_region.get(region_id)

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

        # ---- worklog 82 micro-components (UNCHANGED) ------------------------
        consistency = build_dense_surface_consistency_components(
            region_id, evidence, covariance=evidence_covariance,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        if not consistency.components:
            row["assembled_units"] = []
            rows.append(row)
            continue

        micro_components = tuple(c.member_indices for c in consistency.components)
        non_manifold_flags = tuple(c.non_manifold_suspected for c in consistency.components)
        full_evidence_scale = _median_nn(evidence)

        # ---- worklog 83 assembly (UNCHANGED, preserved as-is) ----------------
        assembly = build_chart_unit_assembly(
            region_id, evidence, covariance=evidence_covariance,
            micro_components=micro_components, non_manifold_flags=non_manifold_flags,
            full_evidence_spacing=full_evidence_scale,
            arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )

        # Which worklog-82 micro-components were themselves valid_supported,
        # to attribute the 16->4 loss precisely (evaluated by refitting each
        # micro-component alone through the SAME evaluate_fit chain, exactly
        # as worklog 82's own replay already measured -- reused, not redone
        # differently).
        micro_valid_flags = []
        for local_members in micro_components:
            if len(local_members) < 4:
                micro_valid_flags.append(False)
                continue
            sub_positions = evidence[torch.tensor(local_members, dtype=torch.long, device=points.device)]
            sub_covariance = evidence_covariance[torch.tensor(local_members, dtype=torch.long, device=points.device)]
            sub_ids = [stable_ids[indices[i]] for i in local_members]
            sub_boundary = materialize_chart_unit_boundary(
                sub_positions, sub_covariance, sub_ids, sub_positions,
                arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
            )
            if sub_boundary.state != BOUNDARY_STATE_MATERIALIZED:
                micro_valid_flags.append(False)
                continue
            sub_eval = evaluate_fit(sub_boundary.ordered_positions, sub_positions, "micro")
            micro_valid_flags.append(sub_eval.get("classification") == "valid_supported")

        units = []
        evidence_in_assembled = 0
        evidence_coherent = 0
        evidence_ambiguous = 0
        evidence_materialized = 0
        classification_evidence = {"valid_supported": 0, "extrapolative": 0, "unsafe_geometry": 0, "no_chart": 0}
        boundary_failure_reasons: dict[str, int] = {}
        was_valid_micro_now = {"still_valid": 0, "ambiguous_over_merged": 0, "boundary_failed_or_extrapolative": 0}

        for unit_idx, unit in enumerate(assembly.chart_units):
            member_local = list(unit.member_indices)
            unit_size = len(member_local)
            evidence_in_assembled += unit_size
            unit_evidence = evidence[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            unit_covariance = evidence_covariance[torch.tensor(member_local, dtype=torch.long, device=points.device)]
            unit_stable_ids = [stable_ids[indices[i]] for i in member_local]

            unit_record = {
                "unit_index": unit_idx,
                "micro_component_count": len(unit.micro_component_indices),
                "member_count": unit_size,
                "contains_previously_valid_micro": any(
                    micro_valid_flags[i] for i in unit.micro_component_indices if i < len(micro_valid_flags)
                ),
            }

            boundary = materialize_chart_unit_boundary(
                unit_evidence, unit_covariance, unit_stable_ids, unit_evidence,
                arc_starts=arc_starts, arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
            )
            unit_record["coherent"] = boundary.coherence.coherent if boundary.coherence else None
            unit_record["internal_normal_disagreement_fraction"] = (
                boundary.coherence.internal_normal_disagreement_fraction if boundary.coherence else None
            )
            unit_record["boundary_state"] = boundary.state
            unit_record["crease_inconsistent_segment_count"] = boundary.crease_inconsistent_segment_count

            if boundary.coherence and boundary.coherence.coherent:
                evidence_coherent += unit_size
            else:
                evidence_ambiguous += unit_size

            boundary_failure_reasons[boundary.state] = boundary_failure_reasons.get(boundary.state, 0) + 1

            if boundary.state != BOUNDARY_STATE_MATERIALIZED:
                unit_record["classification"] = "no_chart"
                classification_evidence["no_chart"] += unit_size
                if unit_record["contains_previously_valid_micro"]:
                    if boundary.state == "chart_unit_ambiguous_or_over_merged":
                        was_valid_micro_now["ambiguous_over_merged"] += 1
                    else:
                        was_valid_micro_now["boundary_failed_or_extrapolative"] += 1
                units.append(unit_record)
                continue

            evidence_materialized += unit_size
            fit = evaluate_fit(boundary.ordered_positions, unit_evidence, f"region{region_id}_unit{unit_idx}")
            unit_record.update(fit)
            classification = fit.get("classification", "no_chart")
            classification_evidence[classification] = classification_evidence.get(classification, 0) + unit_size
            if unit_record["contains_previously_valid_micro"]:
                if classification == "valid_supported":
                    was_valid_micro_now["still_valid"] += 1
                else:
                    was_valid_micro_now["boundary_failed_or_extrapolative"] += 1
            units.append(unit_record)

        row["assembly"] = {
            "micro_component_count": len(micro_components),
            "chart_unit_count": len(assembly.chart_units),
            "micro_valid_supported_count": sum(micro_valid_flags),
        }
        row["evidence_fractions"] = {
            "entering_assembled_units": evidence_in_assembled / total_evidence,
            "coherent": evidence_coherent / total_evidence,
            "ambiguous_or_over_merged": evidence_ambiguous / total_evidence,
            "reaching_materialized_chart": evidence_materialized / total_evidence,
            "valid_supported": classification_evidence["valid_supported"] / total_evidence,
            "extrapolative": classification_evidence["extrapolative"] / total_evidence,
            "unsafe_geometry": classification_evidence["unsafe_geometry"] / total_evidence,
            "no_chart": classification_evidence["no_chart"] / total_evidence,
        }
        row["boundary_failure_reasons"] = boundary_failure_reasons
        row["valid_micro_attribution"] = was_valid_micro_now
        row["assembled_units"] = units
        rows.append(row)
    return {"checkpoint": str(checkpoint), "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val84/dense_chart_unit_boundary_replay.json"))
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, "cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}\n", flush=True)

    print(f"{'reg':>3} {'evid':>5} {'units':>5} | evid% assembled/coherent/materialized | evid% valid/extrap/unsafe/no_chart | attribution")
    for row in report["regions"]:
        if "skip_reason" in row:
            print(f"{row['region']:>3} {row['total_evidence']:>5} skipped: {row['skip_reason']}")
            continue
        asm = row.get("assembly", {})
        ef = row.get("evidence_fractions", {})
        attr = row.get("valid_micro_attribution", {})
        print(
            f"{row['region']:>3} {row['total_evidence']:>5} {asm.get('chart_unit_count', 0):>5} | "
            f"{ef.get('entering_assembled_units', 0)*100:>5.1f}/{ef.get('coherent', 0)*100:>5.1f}/{ef.get('reaching_materialized_chart', 0)*100:>5.1f} | "
            f"{ef.get('valid_supported', 0)*100:>5.1f}/{ef.get('extrapolative', 0)*100:>5.1f}/{ef.get('unsafe_geometry', 0)*100:>5.1f}/{ef.get('no_chart', 0)*100:>5.1f} | "
            f"{attr}"
        )

    all_units = [u for row in report["regions"] for u in row.get("assembled_units", [])]
    counts: dict[str, int] = {}
    for u in all_units:
        key = u.get("classification", "?")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nALL-UNIT totals across 7 regions: {counts}")
    total_attr = {"still_valid": 0, "ambiguous_over_merged": 0, "boundary_failed_or_extrapolative": 0}
    for row in report["regions"]:
        for k, v in row.get("valid_micro_attribution", {}).items():
            total_attr[k] = total_attr.get(k, 0) + v
    print(f"16->4 valid-chart-loss attribution (units containing a previously-valid micro-component): {total_attr}")


if __name__ == "__main__":
    main()
