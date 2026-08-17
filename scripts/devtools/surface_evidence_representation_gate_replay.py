"""Worklog 94 -- bounded architecture gate replay.

Replays the SAME unmodified downstream constructor contract (Worklog 82
micro-components -> Worklog 83 assembly -> Worklog 89 face-incidence
boundary -> Worklog 79 coverage (inside boundary validation) -> PCA-UV ->
6x6 NURBS -> held-out evaluation -> valid_supported/extrapolative/
unsafe/unresolved classification, all via the exact same
``evaluate_fit``/``materialize_chart_unit_cut_boundaries`` functions worklog
89's own replay already used) for four fixed surface-evidence
representations, on the same 7 real regions and the same real checkpoint.

Region formation, region ownership, Worklog 82 relation thresholds, NURBS
capacity, PCA-UV, ADC, and visible Gaussian training are all computed ONCE
on raw evidence and are IDENTICAL across all four representations -- only
the per-region-owned ``(evidence, evidence_covariance)`` pair handed to the
downstream constructor is swapped, via
``osn_gs.surface.torch_surface_evidence_representation_gate.build_representation_evidence``.

No fallback chain: a representation that fails a unit stays unresolved/
unsafe/extrapolative for that unit -- it is never silently retried with a
different representation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _median_nn, evaluate_fit
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import (
    MIXED_PHYSICAL_PARTITION_SEAM,
    PHYSICAL_ONLY,
    SEAM_ONLY,
    build_chart_unit_topology_context,
    materialize_chart_unit_cut_boundaries,
)
from osn_gs.surface.torch_dense_chart_unit_assembly import build_chart_unit_assembly
from osn_gs.surface.torch_dense_surface_consistency_components import build_dense_surface_consistency_components
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_surface_evidence_representation_gate import (
    REPRESENTATIONS,
    build_representation_evidence,
)


def _weighted_percentile(values: list[tuple[float, float]], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    target = sum(weight for _value, weight in ordered) * percentile / 100.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def _load_region_context(checkpoint: Path, cap: int, device: str):
    """Everything computed ONCE on raw evidence, identical for all four
    representations: model load, canonical region construction, region
    ownership propagation, and the sparse chart/frame lookup used only for
    arc typing. None of this differs by representation."""

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
    representative_ids = bundle.representative_stable_ids
    representative_positions = points[bundle.representative_indices]
    representative_index = {stable_id: index for index, stable_id in enumerate(representative_ids)}
    frames = construct_canonical_region_tangent_frames(
        representative_positions, construction.covariance_frame, construction.reliability,
        regions, ids=representative_ids,
    )
    frame_by_region = {frame.region_id: frame for frame in frames if frame is not None}
    chart_by_region = {chart.region_id: chart for chart in construction.region_parametric_chart_boundaries}
    cluster = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _ = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)
    return (
        regions, points, covariance, stable_ids, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    )


def analyze_representation(
    representation: str, region_context, device: str,
) -> dict:
    (
        regions, points, covariance, stable_ids, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = region_context

    rows = []
    overall_counts = Counter()
    overall_composition = Counter()
    overall_classification = Counter()
    overall_p95: list[tuple[float, float]] = []
    overall_displacement: list[tuple[float, float]] = []
    overall_curvature_metrics: list[tuple[float, float]] = []

    adapter_time_total = 0.0

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        total_evidence = len(full_indices)
        row: dict = {"region": region_id, "total_evidence": total_evidence}
        overall_counts["total_evidence"] += total_evidence
        if total_evidence < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue

        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        raw_evidence = points[selector]
        raw_evidence_covariance = covariance[selector]
        region_stable_ids = [stable_ids[index] for index in full_indices]

        adapter_start = time.perf_counter()
        adapted = build_representation_evidence(representation, raw_evidence, raw_evidence_covariance)
        adapter_time_total += time.perf_counter() - adapter_start
        evidence = adapted.positions
        evidence_covariance = adapted.covariance
        if adapted.displacement_over_spacing:
            overall_displacement.append((adapted.displacement_over_spacing, total_evidence))

        chart = chart_by_region.get(region_id)
        frame = frame_by_region.get(region_id)
        arc_starts = arc_ends = None
        arc_kinds: list[str] = []
        if chart is not None and chart.ordered_node_ids and frame is not None:
            nodes = [node for node in chart.ordered_node_ids if node in representative_index]
            if len(nodes) >= 2:
                sparse_positions = torch.stack(
                    [representative_positions[representative_index[node]] for node in nodes], dim=0,
                )
                arc_kinds = [segment.segment_kind for segment in chart.segments][:len(nodes)]
                arc_starts = sparse_positions
                arc_ends = torch.stack(
                    [sparse_positions[(index + 1) % len(nodes)] for index in range(len(nodes))], dim=0,
                )

        consistency = build_dense_surface_consistency_components(
            region_id, evidence, covariance=evidence_covariance, arc_starts=arc_starts,
            arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        if not consistency.components:
            row["skip_reason"] = "no_worklog82_components"
            rows.append(row)
            continue
        micro_components = tuple(component.member_indices for component in consistency.components)
        non_manifold_flags = tuple(component.non_manifold_suspected for component in consistency.components)
        assembly = build_chart_unit_assembly(
            region_id, evidence, covariance=evidence_covariance,
            micro_components=micro_components, non_manifold_flags=non_manifold_flags,
            full_evidence_spacing=_median_nn(evidence), arc_starts=arc_starts, arc_ends=arc_ends,
            arc_kinds=arc_kinds if arc_kinds else None,
        )
        context = build_chart_unit_topology_context(
            evidence, evidence_covariance, region_stable_ids, arc_starts=arc_starts,
            arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )

        counts = Counter()
        composition = Counter()
        classification = Counter()
        heldout_p95: list[tuple[float, float]] = []
        units = []
        for unit_index, unit in enumerate(assembly.chart_units):
            members = list(unit.member_indices)
            unit_size = len(members)
            result = materialize_chart_unit_cut_boundaries(context, members)
            coherent = bool(result.coherence and result.coherence.coherent)
            if coherent:
                counts["coherent_evidence"] += unit_size

            materialized_domains = [domain for domain in result.domains if domain.materialized]
            if not materialized_domains:
                classification["unresolved"] += unit_size
                units.append({
                    "unit_index": unit_index, "member_count": unit_size, "coherent": coherent,
                    "unresolved_reasons": list(result.unresolved_reasons),
                })
                continue

            counts["cut_recoverable_evidence"] += unit_size
            share = unit_size / len(materialized_domains)
            member_selector = torch.tensor(members, dtype=torch.long, device=points.device)
            unit_evidence = evidence[member_selector]
            for domain_index, domain in enumerate(materialized_domains):
                fit = evaluate_fit(
                    domain.ordered_positions, unit_evidence,
                    f"region{region_id}_unit{unit_index}_domain{domain_index}_{representation}",
                )
                fit_class = fit.get("classification", "unresolved")
                classification[fit_class] += share
                composition[domain.boundary_composition] += share
                p95 = fit.get("extrapolation_p95")
                if p95 is not None:
                    heldout_p95.append((float(p95), share))
            units.append({"unit_index": unit_index, "member_count": unit_size, "coherent": coherent})

        unresolved_evidence = classification["unresolved"] + classification["fit_failed"]
        row.update({
            "chart_unit_count": len(assembly.chart_units),
            "evidence_counts": dict(counts),
            "evidence_fractions": {
                "coherent_chart_unit_evidence": counts["coherent_evidence"] / total_evidence,
                "chart_unit_cut_boundary_recoverable_evidence": counts["cut_recoverable_evidence"] / total_evidence,
                "physical_only_domain_evidence": composition[PHYSICAL_ONLY] / total_evidence,
                "mixed_domain_evidence": composition[MIXED_PHYSICAL_PARTITION_SEAM] / total_evidence,
                "seam_only_domain_evidence": composition[SEAM_ONLY] / total_evidence,
                "valid_supported": classification["valid_supported"] / total_evidence,
                "extrapolative": classification["extrapolative"] / total_evidence,
                "unsafe_geometry": classification["unsafe_geometry"] / total_evidence,
                "unresolved": unresolved_evidence / total_evidence,
            },
            "heldout_p95_evidence_weighted": _weighted_percentile(heldout_p95, 95.0),
            "assembled_units": units,
        })
        rows.append(row)
        overall_counts.update(counts)
        overall_composition.update(composition)
        overall_classification.update(classification)
        overall_p95.extend(heldout_p95)

    total = overall_counts["total_evidence"] or 1
    coherent = overall_counts["coherent_evidence"] or 1
    overall_unresolved = overall_classification["unresolved"] + overall_classification["fit_failed"]
    summary = {
        "representation": representation,
        "total_evidence": overall_counts["total_evidence"],
        "evidence_fractions": {
            "coherent_chart_unit_evidence": overall_counts["coherent_evidence"] / total,
            "chart_unit_cut_boundary_recoverable_evidence": overall_counts["cut_recoverable_evidence"] / total,
            "worklog79_coverage_pass_rate_of_coherent": overall_counts["cut_recoverable_evidence"] / coherent,
            "physical_only_domain_evidence": overall_composition[PHYSICAL_ONLY] / total,
            "mixed_domain_evidence": overall_composition[MIXED_PHYSICAL_PARTITION_SEAM] / total,
            "seam_only_domain_evidence": overall_composition[SEAM_ONLY] / total,
            "valid_supported": overall_classification["valid_supported"] / total,
            "extrapolative": overall_classification["extrapolative"] / total,
            "unsafe_geometry": overall_classification["unsafe_geometry"] / total,
            "unresolved": overall_unresolved / total,
        },
        "nurbs_materialization_rate": overall_counts["cut_recoverable_evidence"] / total,
        "heldout_p95_evidence_weighted": _weighted_percentile(overall_p95, 95.0),
        "geometry_displacement_over_spacing_evidence_weighted": (
            _weighted_percentile(overall_displacement, 50.0) if overall_displacement else 0.0
        ),
        "adapter_runtime_seconds": adapter_time_total,
    }
    return {"summary": summary, "regions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/extent_ab/val94/surface_evidence_representation_gate_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    load_start = time.perf_counter()
    region_context = _load_region_context(args.checkpoint, args.cap, args.device)
    load_time = time.perf_counter() - load_start

    reports = {}
    for representation in REPRESENTATIONS:
        tracemalloc.start()
        start = time.perf_counter()
        report = analyze_representation(representation, region_context, args.device)
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["summary"]["total_runtime_seconds"] = elapsed
        report["summary"]["peak_python_memory_bytes"] = peak
        reports[representation] = report

    output = {
        "checkpoint": str(args.checkpoint), "cap": args.cap,
        "region_load_time_seconds": load_time,
        "representations": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps(
        {representation: report["summary"] for representation, report in reports.items()},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
