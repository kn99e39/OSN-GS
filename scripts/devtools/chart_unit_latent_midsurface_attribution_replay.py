"""Worklog 93 -- read-only latent-midsurface recoverability replay.

For every Worklog 92 LOCALLY_THICK_UNIMODAL_SHEET / LOCALLY_SINGLE_CURVED_SHEET
member (the ~87-97% majority of Worklog 90's MULTILAYER_OR_VOLUMETRIC
evidence, per Worklog 92), this asks whether that thick/curved center band
contains a recoverable latent 2D midsurface -- using center positions only,
never Gaussian covariance normals as target geometry, never mutating model
state.

Fixed and unmodified: Worklog 89 boundary constructor, Worklog 82 relation
semantics, ADC, visible Gaussian training, NURBS fitting, and Worklog 90/91/92's
own attribution logic (imported, not reimplemented). No new production
boundary or NURBS path. No threshold tuned toward a favorable outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _median_nn
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig  # noqa: F401
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames  # noqa: F401
from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import (
    STATE_NON_MANIFOLD,
    build_chart_unit_topology_context,
    materialize_chart_unit_cut_boundaries,
)
from osn_gs.surface.torch_chart_unit_latent_midsurface_attribution import (
    attribute_latent_midsurface_recoverability,
)
from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
    LOCALLY_SINGLE_CURVED_SHEET,
    LOCALLY_THICK_UNIMODAL_SHEET,
    attribute_local_center_geometry,
)
from osn_gs.surface.torch_chart_unit_surface_topology_attribution import (
    MULTILAYER_OR_VOLUMETRIC,
    attribute_failed_chart_unit_surface_topology,
)
from osn_gs.surface.torch_dense_chart_unit_assembly import build_chart_unit_assembly
from osn_gs.surface.torch_dense_surface_consistency_components import (
    DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
    DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
    DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
    DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    build_dense_surface_consistency_components,
    build_same_surface_adjacency,
)

TARGET_CLASSES = (LOCALLY_THICK_UNIMODAL_SHEET, LOCALLY_SINGLE_CURVED_SHEET)
_MIN_SUBSET_SIZE = 6  # matches the module's own _MIN_POINTS_FOR_QUADRATIC_FIT


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    if not values:
        return None
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / total if total else None


def analyze_checkpoint(checkpoint_dir: Path, iteration, cap: int, device: str) -> dict:
    model, stable_ids = _load_model(checkpoint_dir, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    rows: list[dict] = []
    metric_values: dict[str, list[tuple[float, int]]] = {
        "local_thickness_over_spacing": [],
        "neighbor_position_agreement": [],
        "neighbor_tangent_agreement": [],
        "neighbor_curvature_agreement": [],
        "projection_displacement_over_spacing": [],
        "projection_displacement_over_extent": [],
        "raw_open_or_nonmanifold_fraction": [],
        "diagnostic_open_or_nonmanifold_fraction": [],
        "raw_valid_face_incidence_fraction": [],
        "valid_face_incidence_fraction": [],
        "neighborhood_preservation_fraction": [],
        "curvature_before": [],
        "curvature_after": [],
        "support_band_fidelity": [],
    }
    total_evidence = 0
    curvature_preserved_evidence = 0
    curvature_not_preserved_evidence = 0
    manifold_improved_evidence = 0
    manifold_not_improved_evidence = 0
    by_class_evidence: dict[str, int] = {cls: 0 for cls in TARGET_CLASSES}

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        if len(full_indices) < 4:
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence, evidence_covariance = points[selector], covariance[selector]
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
        assembly = build_chart_unit_assembly(
            region_id, evidence, covariance=evidence_covariance,
            micro_components=tuple(component.member_indices for component in consistency.components),
            non_manifold_flags=tuple(component.non_manifold_suspected for component in consistency.components),
            full_evidence_spacing=_median_nn(evidence), arc_starts=arc_starts, arc_ends=arc_ends,
            arc_kinds=arc_kinds if arc_kinds else None,
        )
        context = build_chart_unit_topology_context(
            evidence, evidence_covariance, list(range(len(full_indices))), arc_starts=arc_starts,
            arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        relation_edges, _adjacency, _vetoes = build_same_surface_adjacency(
            evidence, context.normals, arc_side=context.arc_side,
            candidate_neighbor_count=DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
            max_candidate_count_per_node=DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
            same_surface_min_normal_alignment=DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
            same_surface_max_mutual_residual=DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
        )

        for unit_index, unit in enumerate(assembly.chart_units):
            members = list(unit.member_indices)
            boundary = materialize_chart_unit_cut_boundaries(context, members)
            if not (boundary.coherence and boundary.coherence.coherent):
                continue
            if not any(reason.startswith(STATE_NON_MANIFOLD) for reason in boundary.unresolved_reasons):
                continue
            attribution = attribute_failed_chart_unit_surface_topology(
                evidence, evidence_covariance, members, relation_edges,
            )
            if attribution.primary_cause != MULTILAYER_OR_VOLUMETRIC:
                continue

            local_result = attribute_local_center_geometry(evidence, members)
            for target_class in TARGET_CLASSES:
                subset = [
                    members[local_index] for local_index, cls in enumerate(local_result.class_by_member)
                    if cls == target_class
                ]
                if len(subset) < _MIN_SUBSET_SIZE:
                    continue
                report = attribute_latent_midsurface_recoverability(evidence, subset)
                subset_size = len(subset)
                total_evidence += subset_size
                by_class_evidence[target_class] += subset_size

                curvature_before = report.mean_curvature_before
                curvature_after = report.mean_curvature_after
                if report.curvature_preserved:
                    curvature_preserved_evidence += subset_size
                else:
                    curvature_not_preserved_evidence += subset_size

                manifold_improved = (
                    report.valid_local_face_incidence_fraction
                    >= report.raw_valid_local_face_incidence_fraction
                )
                if manifold_improved:
                    manifold_improved_evidence += subset_size
                else:
                    manifold_not_improved_evidence += subset_size

                for key, value in (
                    ("local_thickness_over_spacing", report.local_thickness_over_spacing_evidence_weighted),
                    ("neighbor_position_agreement", report.neighbor_position_agreement),
                    ("neighbor_tangent_agreement", report.neighbor_tangent_agreement),
                    ("neighbor_curvature_agreement", report.neighbor_curvature_agreement),
                    (
                        "projection_displacement_over_spacing",
                        report.projection_displacement_over_spacing_median,
                    ),
                    (
                        "projection_displacement_over_extent",
                        report.projection_displacement_over_extent_median,
                    ),
                    ("raw_open_or_nonmanifold_fraction", report.raw_open_or_nonmanifold_fraction),
                    (
                        "diagnostic_open_or_nonmanifold_fraction",
                        report.diagnostic_open_or_nonmanifold_fraction,
                    ),
                    (
                        "raw_valid_face_incidence_fraction",
                        report.raw_valid_local_face_incidence_fraction,
                    ),
                    ("valid_face_incidence_fraction", report.valid_local_face_incidence_fraction),
                    (
                        "neighborhood_preservation_fraction",
                        report.neighborhood_preservation_fraction,
                    ),
                    ("curvature_before", curvature_before),
                    ("curvature_after", curvature_after),
                    ("support_band_fidelity", report.observed_support_band_fidelity_fraction),
                ):
                    if value is not None:
                        metric_values[key].append((float(value), subset_size))

                rows.append({
                    "region": region_id, "unit_index": unit_index, "target_class": target_class,
                    "subset_size": subset_size,
                    "local_thickness_over_spacing": report.local_thickness_over_spacing_evidence_weighted,
                    "raw_open_or_nonmanifold_fraction": report.raw_open_or_nonmanifold_fraction,
                    "diagnostic_open_or_nonmanifold_fraction": report.diagnostic_open_or_nonmanifold_fraction,
                    "raw_valid_face_incidence_fraction": report.raw_valid_local_face_incidence_fraction,
                    "valid_face_incidence_fraction": report.valid_local_face_incidence_fraction,
                    "manifold_improved": manifold_improved,
                    "curvature_before": curvature_before,
                    "curvature_after": curvature_after,
                    "curvature_preserved": report.curvature_preserved,
                    "support_band_fidelity": report.observed_support_band_fidelity_fraction,
                })

    return {
        "checkpoint": str(checkpoint_dir),
        "iteration": iteration,
        "summary": {
            "total_evidence": total_evidence,
            "evidence_by_target_class": by_class_evidence,
            "curvature_preserved_fraction": (
                curvature_preserved_evidence / total_evidence if total_evidence else None
            ),
            "manifold_improved_fraction": (
                manifold_improved_evidence / total_evidence if total_evidence else None
            ),
            "metrics_evidence_weighted": {
                key: _weighted_mean(values) for key, values in metric_values.items()
            },
        },
        "unit_reports": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/extent_ab/val93/chart_unit_latent_midsurface_attribution_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoints", nargs="+", default=["2900", "final"])
    args = parser.parse_args()

    reports = []
    for iteration in args.checkpoints:
        checkpoint_dir = args.run_dir / str(iteration)
        if not checkpoint_dir.exists():
            continue
        reports.append(analyze_checkpoint(checkpoint_dir, iteration, args.cap, args.device))

    output = {"run_dir": str(args.run_dir), "cap": args.cap, "checkpoints": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps(
        {"checkpoint_summaries": [{"iteration": r["iteration"], **r["summary"]} for r in reports]},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
