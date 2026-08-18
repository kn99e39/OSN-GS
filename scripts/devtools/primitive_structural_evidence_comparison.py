"""exp/2dgs-nurbs-surface-evidence -- structural evidence comparison.

Answers the branch's research question with ONE evaluator applied to BOTH
arms: does the actual 2DGS training formulation produce Gaussian surface
evidence that is materially more suitable for OSN-GS curve-constrained NURBS
construction than vanilla `baseline_compatible` 3DGS evidence?

Both arms enter through
`osn_gs.gaussian.torch_primitive_evidence_adapter.load_primitive_evidence`,
which dispatches on the primitive the checkpoint itself records and hands the
identical `(positions, covariance, opacity)` triple to the identical,
unmodified downstream chain:

    canonical region construction (`_construct_canonical_with_full_evidence`)
      -> region-owned full evidence (`_propagate_with_evidence_gating`)
      -> Worklog 82 micro-components
      -> Worklog 83 chart-unit assembly
      -> Worklog 89 face-incidence cut boundaries
      -> Worklog 79 coverage + PCA-UV + 6x6 NURBS + held-out evaluation
      -> valid_supported / extrapolative / unsafe_geometry / unresolved

That chain, and every threshold in it, is exactly what
`surface_evidence_representation_gate_replay.py` (Worklog 94) ran on the
volumetric baseline; this script reuses that script's own functions rather
than reimplementing them, so the 2DGS arm gets no easier constructor.

On top of it, the per-region STRUCTURAL metrics the branch contract asks for
are computed with `attribute_local_center_geometry` (Worklog 92, unmodified,
CENTER-POSITIONS-ONLY so it cannot favour either primitive) plus
primitive-agnostic orientation/coverage measures.

Usage:
    python3 scripts/devtools/primitive_structural_evidence_comparison.py \
        --arm vanilla=output/vanilla_30k/30000 \
        --arm 2dgs=output/2dgs_30k/30000 \
        --output reports/primitive_evidence_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from osn_gs.gaussian.torch_surfel_analysis_adapter import EPSILON_REGULARIZED, EXACT_RANK2
from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
    LOCAL_CLASSES,
    attribute_local_center_geometry,
)
from surface_evidence_representation_gate_replay import (
    _load_region_context,
    analyze_representation,
)
from osn_gs.surface.torch_surface_evidence_representation_gate import (
    REPRESENTATION_RAW_CENTER_BASELINE,
)

# Bounded so the O(n) python attribution loop stays tractable on multi-million
# primitive checkpoints. Deterministic stride, identical rule for both arms.
_STRUCTURAL_SAMPLE_CAP = 4000
_ORIENTATION_NEIGHBORS = 8


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float(0.5 * (ordered[middle - 1] + ordered[middle]))


def _deterministic_sample(count: int, cap: int, device) -> torch.Tensor:
    if count <= cap:
        return torch.arange(count, dtype=torch.long, device=device)
    return torch.linspace(0, count - 1, cap, device=device).round().long().unique()


def _knn(points: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    neighbors = min(k, points.shape[0] - 1)
    values, indices = torch.topk(distances, neighbors, dim=1, largest=False)
    return values, indices


def _orientation_metrics(points: torch.Tensor, normals: torch.Tensor) -> dict[str, float | None]:
    """Primitive-agnostic orientation quality of the evidence itself.

    `normals` is whatever unit orientation the primitive genuinely offers:
    the intrinsic tangent-plane normal t_w for a 2DGS surfel, the minor
    principal axis of the 3D covariance for a volumetric Gaussian. Neither is
    privileged -- each arm is measured on the orientation its own primitive
    actually carries.
    """

    count = int(points.shape[0])
    if count < _ORIENTATION_NEIGHBORS + 1:
        return {
            "normal_coherence": None,
            "tangent_plane_residual_over_spacing": None,
            "neighbor_spacing": None,
        }
    distances, indices = _knn(points, _ORIENTATION_NEIGHBORS)
    spacing = distances.mean(dim=1).clamp_min(1e-12)

    neighbor_normals = normals[indices]
    # Unoriented: covariance/tangent normals are lines, so compare |cos|.
    coherence = (neighbor_normals * normals[:, None, :]).sum(dim=-1).abs().mean(dim=1)

    offsets = points[indices] - points[:, None, :]
    out_of_plane = (offsets * normals[:, None, :]).sum(dim=-1).abs().mean(dim=1)
    residual = out_of_plane / spacing

    return {
        "normal_coherence": float(coherence.median()),
        "tangent_plane_residual_over_spacing": float(residual.median()),
        "neighbor_spacing": float(spacing.median()),
    }


def _coverage_metrics(points: torch.Tensor, spacing: float | None) -> dict[str, float | None]:
    """Spatial evidence coverage: occupied fraction of a spacing-sized grid."""

    if spacing is None or spacing <= 0 or points.shape[0] == 0:
        return {"occupied_cells": None, "evidence_per_occupied_cell": None}
    minimum = points.amin(dim=0)
    extent = (points.amax(dim=0) - minimum).clamp_min(1e-9)
    resolution = torch.clamp((extent / spacing).floor(), min=1.0, max=256.0)
    cells = torch.floor((points - minimum) / extent * resolution).clamp_min(0)
    cells = torch.minimum(cells, resolution - 1).long()
    keys = (cells[:, 0] * int(resolution[1]) + cells[:, 1]) * int(resolution[2]) + cells[:, 2]
    occupied = int(torch.unique(keys).numel())
    return {
        "occupied_cells": occupied,
        "evidence_per_occupied_cell": float(points.shape[0]) / max(occupied, 1),
    }


def structural_metrics(region_context, device: str) -> dict:
    """Per-region structural evidence quality, primitive-agnostic."""

    (
        regions, points, _covariance, _stable_ids, owned, _representative_positions,
        _representative_index, _frame_by_region, chart_by_region, evidence, _construction,
    ) = region_context

    normals = evidence.normals
    rows = []
    class_totals = Counter()
    weighted_total = 0
    band_thickness: list[float] = []
    coherence_values: list[float] = []
    residual_values: list[float] = []
    spacing_values: list[float] = []
    coverage_density: list[float] = []

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        total_evidence = len(full_indices)
        row: dict = {"region": region_id, "total_evidence": total_evidence}
        if total_evidence < 8:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue

        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        sample = selector[_deterministic_sample(total_evidence, _STRUCTURAL_SAMPLE_CAP, points.device)]
        region_points = points[sample]
        region_normals = normals[sample]

        attribution = attribute_local_center_geometry(
            region_points, list(range(int(region_points.shape[0])))
        )
        sampled = int(region_points.shape[0])
        for name, fraction in attribution.class_node_fractions.items():
            class_totals[name] += fraction * sampled
        weighted_total += sampled

        spreads = [
            item.local_mode_spread_over_spacing
            for item in attribution.local_geometry_by_member
            if item.local_mode_spread_over_spacing is not None
        ]
        region_band = _median(spreads)
        if region_band is not None:
            band_thickness.append(region_band)

        orientation = _orientation_metrics(region_points, region_normals)
        if orientation["normal_coherence"] is not None:
            coherence_values.append(orientation["normal_coherence"])
            residual_values.append(orientation["tangent_plane_residual_over_spacing"])
            spacing_values.append(orientation["neighbor_spacing"])
        coverage = _coverage_metrics(region_points, orientation["neighbor_spacing"])
        if coverage["evidence_per_occupied_cell"] is not None:
            coverage_density.append(coverage["evidence_per_occupied_cell"])

        chart = chart_by_region.get(region_id)
        row.update({
            "sampled_evidence": sampled,
            "local_class_fractions": attribution.class_node_fractions,
            "primary_local_class": attribution.primary_class,
            "persistent_layer_count": attribution.persistent_layer_count,
            "local_band_thickness_over_spacing": region_band,
            **orientation,
            **coverage,
            "chart_boundary_status": None if chart is None else chart.status,
            "chart_boundary_segments": 0 if chart is None else len(chart.segments),
            "chart_segment_kinds": None if chart is None else chart.segment_kind_counts(),
        })
        rows.append(row)

    total = max(weighted_total, 1)
    return {
        "summary": {
            "region_count": len(regions.regions),
            "structurally_measured_regions": sum(1 for row in rows if "skip_reason" not in row),
            "sampled_evidence": weighted_total,
            "local_class_fractions": {
                name: class_totals[name] / total for name in LOCAL_CLASSES
            },
            "persistent_multilayer_fraction": (
                class_totals["TRUE_PERSISTENT_TWO_LAYER"] + class_totals["TRUE_PERSISTENT_MULTI_LAYER"]
            ) / total,
            "single_sheet_fraction": (
                class_totals["LOCALLY_SINGLE_CURVED_SHEET"] + class_totals["LOCALLY_THICK_UNIMODAL_SHEET"]
            ) / total,
            "median_local_band_thickness_over_spacing": _median(band_thickness),
            "median_normal_coherence": _median(coherence_values),
            "median_tangent_plane_residual_over_spacing": _median(residual_values),
            "median_neighbor_spacing": _median(spacing_values),
            "median_evidence_per_occupied_cell": _median(coverage_density),
        },
        "regions": rows,
    }


def curve_network_metrics(region_context) -> dict:
    """Structural curve / curve-network availability, from the same construction.

    Segment endpoints are the construction's own representative positions, so
    curve LENGTH is measured in the scene's world units and is comparable
    between arms. `regions_with_usable_curve_network` counts regions whose
    parametric chart boundary carries at least a triangle's worth of segments
    -- the minimum from which a chart domain could be built at all.
    """

    (
        _regions, _points, _covariance, _stable_ids, _owned, representative_positions,
        representative_index, _frame_by_region, chart_by_region, _evidence, _construction,
    ) = region_context

    status_counts = Counter()
    segment_kind_counts = Counter()
    total_segments = 0
    lengths: list[float] = []
    usable_regions = 0
    spacing = None

    for chart in chart_by_region.values():
        status_counts[chart.status] += 1
        for kind, count in chart.segment_kind_counts().items():
            segment_kind_counts[kind] += count
        total_segments += len(chart.segments)
        if len(chart.segments) >= 3:
            usable_regions += 1
        for segment in chart.segments:
            start = representative_index.get(segment.node_a)
            end = representative_index.get(segment.node_b)
            if start is None or end is None:
                continue
            lengths.append(
                float((representative_positions[start] - representative_positions[end]).norm())
            )

    if int(representative_positions.shape[0]) > 8:
        distances, _ = _knn(representative_positions, _ORIENTATION_NEIGHBORS)
        spacing = float(distances.mean(dim=1).median())

    return {
        "chart_boundary_status_counts": dict(status_counts),
        "structural_curve_segments": total_segments,
        "structural_curve_segment_kinds": dict(segment_kind_counts),
        "structural_curve_total_length": float(sum(lengths)) if lengths else 0.0,
        "structural_curve_median_segment_length": _median(lengths),
        "structural_curve_measured_segments": len(lengths),
        # Length in units of the construction's own representative spacing, so
        # the two arms stay comparable even if they sample space differently.
        "structural_curve_total_length_over_spacing": (
            float(sum(lengths)) / spacing if lengths and spacing else None
        ),
        "representative_spacing": spacing,
        "regions_with_usable_curve_network": usable_regions,
        "regions_with_any_chart_boundary": len(chart_by_region),
    }


def relation_metrics(region_context) -> dict:
    """Manifold-relation composition of the SAME affinity graph both arms build.

    Reported because one legacy input to that classifier -- the
    `normal_direction_separation_over_thickness` ratio -- is ill-posed for a
    true 2DGS surfel, whose per-primitive normal thickness is exactly zero and
    is therefore floored at `sqrt(1e-12)` by `extract_covariance_frame`. The
    ratio then saturates, which can only push pairs OUT of `ambiguous` and INTO
    `parallel_but_separate` (the `same_surface` branch is evaluated first and
    does not read it). Quantifying the shift is the point: it is a property of
    the OSN-GS criterion meeting rank-2 evidence, not of the 2DGS training.
    """

    (
        _regions, _points, _covariance, _stable_ids, _owned, _rep_positions,
        _rep_index, _frame_by_region, _chart_by_region, _evidence, construction,
    ) = region_context
    graph = construction.manifold_affinity
    counts = Counter(edge.manifold_relation for edge in graph.edges)
    total = max(sum(counts.values()), 1)
    return {
        "affinity_edge_count": sum(counts.values()),
        "relation_counts": dict(counts),
        "relation_fractions": {name: value / total for name, value in counts.items()},
    }


def shape_class_metrics(region_context) -> dict:
    """Covariance-frame shape classification of the evidence both arms feed in."""

    (
        _regions, _points, covariance, _stable_ids, _owned, _rep_positions,
        _rep_index, _frame_by_region, _chart_by_region, evidence, _construction,
    ) = region_context
    from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame

    sample = _deterministic_sample(int(covariance.shape[0]), _STRUCTURAL_SAMPLE_CAP, covariance.device)
    frame = extract_covariance_frame(covariance[sample])
    counts = Counter(frame.shape_class)
    total = max(sum(counts.values()), 1)
    return {
        "sampled_primitives": int(sample.numel()),
        "shape_class_fractions": {name: value / total for name, value in counts.items()},
        "planarity_median": float(frame.planarity.median()),
        "normal_thickness_median": float(frame.normal_thickness.median()),
        "equivalent_tangent_scale_median": float(frame.equivalent_tangent_scale.median()),
        # For a rank-2 surfel this is the `sqrt(degenerate_eps)` floor, i.e. the
        # measurement is saturated rather than informative -- see the module
        # docstring of `torch_primitive_evidence_adapter`.
        "normal_thickness_is_at_the_degenerate_floor": bool(
            float(frame.normal_thickness.median()) <= 1.01e-6
        ),
    }


def analyze_arm(name: str, checkpoint: Path, cap: int, device: str, surfel_mode: str) -> dict:
    start = time.perf_counter()
    context = _load_region_context(
        checkpoint, cap, device, surfel_covariance_mode=surfel_mode
    )
    load_seconds = time.perf_counter() - start

    evidence = context[-1]
    downstream = analyze_representation(REPRESENTATION_RAW_CENTER_BASELINE, context, device)
    structural = structural_metrics(context, device)
    curves = curve_network_metrics(context)

    return {
        "arm": name,
        "evidence": evidence.describe(),
        "primitive_scale_statistics": _scale_statistics(evidence),
        "covariance_shape_classes": shape_class_metrics(context),
        "manifold_relations": relation_metrics(context),
        "region_context_seconds": load_seconds,
        "downstream": downstream["summary"],
        "downstream_regions": downstream["regions"],
        "structural": structural["summary"],
        "structural_regions": structural["regions"],
        "curve_network": curves,
    }


def _scale_statistics(evidence) -> dict:
    tangent = evidence.tangent_scales
    normal = evidence.normal_scale
    anisotropy = tangent[:, 0] / tangent[:, 1].clamp_min(1e-12)
    return {
        "primitive_count": int(tangent.shape[0]),
        "tangent_major_median": float(tangent[:, 0].median()),
        "tangent_minor_median": float(tangent[:, 1].median()),
        "tangent_anisotropy_median": float(anisotropy.median()),
        "tangent_anisotropy_p99": float(torch.quantile(anisotropy.float(), 0.99)),
        "per_primitive_normal_scale_median": float(normal.median()),
        "per_primitive_normal_scale_max": float(normal.max()),
        # For a true 2DGS surfel this is exactly 0 by construction: the
        # primitive has no normal-direction extent at all.
        "per_primitive_thickness_is_exactly_zero": bool(float(normal.abs().max()) == 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", action="append", required=True,
        help="name=checkpoint_dir, repeatable. Example: --arm 2dgs=output/2dgs_30k/30000",
    )
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--surfel_covariance_mode", type=str, default=EXACT_RANK2,
        choices=(EXACT_RANK2, EPSILON_REGULARIZED),
        help=(
            "Covariance view for surfel checkpoints. exact_rank2 (default) is the "
            "true 2DGS geometry with a zero normal eigenvalue. epsilon_regularized "
            "manufactures thickness for legacy ratio-based metrics and MUST be "
            "disclosed wherever its numbers are quoted."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = []
    for item in args.arm:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--arm expects name=checkpoint_dir, got {item!r}")
        arms.append((name, Path(path)))

    report = {
        "cap": args.cap,
        "device": args.device,
        "surfel_covariance_mode": args.surfel_covariance_mode,
        "downstream_representation": REPRESENTATION_RAW_CENTER_BASELINE,
        "arms": [],
    }
    for name, checkpoint in arms:
        print(f"[comparison] analyzing arm {name} <- {checkpoint}", flush=True)
        report["arms"].append(
            analyze_arm(name, checkpoint, args.cap, args.device, args.surfel_covariance_mode)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[comparison] wrote {args.output}", flush=True)

    for arm in report["arms"]:
        summary = arm["downstream"]["evidence_fractions"]
        structural = arm["structural"]
        print(
            f"  {arm['arm']:<10} primitives={arm['primitive_scale_statistics']['primitive_count']:>9} "
            f"valid_supported={summary['valid_supported']:.4f} "
            f"extrapolative={summary['extrapolative']:.4f} "
            f"unsafe={summary['unsafe_geometry']:.4f} "
            f"unresolved={summary['unresolved']:.4f} "
            f"coherent={summary['coherent_chart_unit_evidence']:.4f} "
            f"band={structural['median_local_band_thickness_over_spacing']} "
            f"coherence={structural['median_normal_coherence']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
