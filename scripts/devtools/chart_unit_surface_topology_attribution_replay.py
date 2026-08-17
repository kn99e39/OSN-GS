"""Read-only root-cause attribution for Worklog 89 topology failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _median_nn
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import (
    STATE_NON_MANIFOLD,
    build_chart_unit_topology_context,
    materialize_chart_unit_cut_boundaries,
)
from osn_gs.surface.torch_chart_unit_surface_topology_attribution import (
    ATTRIBUTION_CLASSES,
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
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    if not values:
        return None
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / total if total else None


def _region_inputs(checkpoint: Path, cap: int, device: str):
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
    representative_ids = bundle.representative_stable_ids
    representative_positions = points[bundle.representative_indices]
    representative_index = {stable_id: index for index, stable_id in enumerate(representative_ids)}
    frames = construct_canonical_region_tangent_frames(
        representative_positions, construction.covariance_frame, construction.reliability,
        construction.surface_regions, ids=representative_ids,
    )
    frame_by_region = {frame.region_id: frame for frame in frames if frame is not None}
    chart_by_region = {
        chart.region_id: chart for chart in construction.region_parametric_chart_boundaries
    }
    cluster = torch.tensor(construction.surface_regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _ = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)
    return (
        construction.surface_regions, points, covariance, stable_ids, owned,
        representative_positions, representative_index, frame_by_region, chart_by_region,
    )


def analyze(checkpoint: Path, cap: int, device: str) -> dict:
    (
        regions, points, covariance, stable_ids, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_inputs(checkpoint, cap, device)
    rows: list[dict] = []
    overall_causes = Counter()
    overall_cause_units = Counter()
    overall_secondary_node_evidence = Counter()
    overall_pair_counts = Counter()
    overall_metric_values: dict[str, list[tuple[float, int]]] = {
        "center_spacing_over_tangent_scale_median": [],
        "compatible_footprint_overlap_coverage": [],
        "missing_same_surface_edge_fraction_despite_footprint": [],
        "relation_false_negative_fraction": [],
        "layer_ambiguity_fraction": [],
    }
    overall_failed_evidence = 0
    overall_failed_units = 0
    overall_plausible_evidence = 0
    examples: dict[str, dict] = {}

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        row: dict = {"region": region_id, "total_evidence": len(full_indices), "failed_units": []}
        if len(full_indices) < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence, evidence_covariance = points[selector], covariance[selector]
        region_stable_ids = [stable_ids[index] for index in full_indices]
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
            evidence, evidence_covariance, region_stable_ids, arc_starts=arc_starts,
            arc_ends=arc_ends, arc_kinds=arc_kinds if arc_kinds else None,
        )
        relation_edges, _adjacency, _vetoes = build_same_surface_adjacency(
            evidence, context.normals, arc_side=context.arc_side,
            candidate_neighbor_count=DEFAULT_CANDIDATE_NEIGHBOR_COUNT,
            max_candidate_count_per_node=DEFAULT_MAX_CANDIDATE_COUNT_PER_NODE,
            same_surface_min_normal_alignment=DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
            same_surface_max_mutual_residual=DEFAULT_SAME_SURFACE_MAX_MUTUAL_RESIDUAL,
        )
        region_causes = Counter()
        region_metric_values = {key: [] for key in overall_metric_values}
        plausible_evidence = failed_evidence = 0
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
            payload = asdict(attribution)
            unit_size = len(members)
            record = {
                "unit_index": unit_index,
                "member_count": unit_size,
                "primary_cause": attribution.primary_cause,
                "metrics": payload,
            }
            row["failed_units"].append(record)
            region_causes[attribution.primary_cause] += unit_size
            overall_causes[attribution.primary_cause] += unit_size
            overall_cause_units[attribution.primary_cause] += 1
            for cause, fraction in attribution.cause_node_fractions.items():
                overall_secondary_node_evidence[cause] += unit_size * fraction
            overall_pair_counts["footprint_compatible"] += attribution.footprint_compatible_pair_count
            overall_pair_counts["missing_center_graph"] += attribution.missing_center_graph_pair_count
            overall_pair_counts["relation_rejected"] += attribution.rejected_relation_pair_count
            overall_pair_counts["accepted_same_surface"] += attribution.accepted_same_surface_pair_count
            overall_pair_counts["layer_conflict"] += attribution.layer_conflict_pair_count
            overall_pair_counts["provenance_veto"] += attribution.provenance_veto_pair_count
            failed_evidence += unit_size
            overall_failed_evidence += unit_size
            overall_failed_units += 1
            if attribution.valid_local_surface_complex_plausible:
                plausible_evidence += unit_size
                overall_plausible_evidence += unit_size
            for key, values in region_metric_values.items():
                value = payload[key]
                if value is not None:
                    values.append((float(value), unit_size))
                    overall_metric_values[key].append((float(value), unit_size))
            previous = examples.get(attribution.primary_cause)
            candidate = {
                "region": region_id, "unit_index": unit_index, "member_count": unit_size,
                "primary_cause": attribution.primary_cause, "metrics": payload,
            }
            strength = payload["cause_node_fractions"][attribution.primary_cause]
            if previous is None or (strength, unit_size, -region_id, -unit_index) > (
                previous["metrics"]["cause_node_fractions"][previous["primary_cause"]],
                previous["member_count"], -previous["region"], -previous["unit_index"],
            ):
                examples[attribution.primary_cause] = candidate
        row["summary"] = {
            "failed_unit_count": len(row["failed_units"]),
            "failed_evidence": failed_evidence,
            "cause_evidence": dict(region_causes),
            "valid_local_surface_complex_plausible_evidence": plausible_evidence,
            "metrics_evidence_weighted": {
                key: _weighted_mean(values) for key, values in region_metric_values.items()
            },
        }
        rows.append(row)
    return {
        "checkpoint": str(checkpoint), "cap": cap, "regions": rows,
        "summary": {
            "failed_topology_unit_count": overall_failed_units,
            "failed_topology_evidence": overall_failed_evidence,
            "cause_evidence": {cause: overall_causes[cause] for cause in ATTRIBUTION_CLASSES},
            "cause_unit_count": {cause: overall_cause_units[cause] for cause in ATTRIBUTION_CLASSES},
            "secondary_cause_node_evidence": {
                cause: overall_secondary_node_evidence[cause] for cause in ATTRIBUTION_CLASSES
            },
            "footprint_pair_counts": dict(overall_pair_counts),
            "cause_evidence_fraction": {
                cause: overall_causes[cause] / overall_failed_evidence if overall_failed_evidence else 0.0
                for cause in ATTRIBUTION_CLASSES
            },
            "valid_local_surface_complex_plausible_evidence": overall_plausible_evidence,
            "valid_local_surface_complex_plausible_fraction": (
                overall_plausible_evidence / overall_failed_evidence if overall_failed_evidence else 0.0
            ),
            "metrics_evidence_weighted": {
                key: _weighted_mean(values) for key, values in overall_metric_values.items()
            },
            "representative_examples": examples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val90/chart_unit_surface_topology_attribution_replay.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.cap, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
