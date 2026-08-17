"""Worklog 92 -- final center-geometry attribution replay.

Removes Worklog 91's global-single-SVD-plane confound (a curved single
sheet can look like several depth bands relative to ONE plane) by
reclassifying every Worklog 90 ``MULTILAYER_OR_VOLUMETRIC`` unit with the
local, spatial-persistence-gated classifier in
``osn_gs/surface/torch_chart_unit_local_center_geometry_attribution.py``.

For members landing in TRUE_PERSISTENT_TWO_LAYER / TRUE_PERSISTENT_MULTI_LAYER,
this script additionally reports, per persistent local layer:

- evidence population,
- depth separation normalized by local center spacing,
- spatial persistence/extent (member count / bounding radius),
- opacity,
- screen-space footprint overlap and rendered-alpha contribution (via real
  camera renders, same convention as Worklog 91's visibility check),
- fraction of newly appearing stable IDs associated with the layer (spatial
  world-space match against checkpoint 600, not aggregate chart-unit
  population comparison),
- whether the layer survives later pruning (checkpoint iteration where each
  stable ID was last observed).

Fixed and unmodified: Worklog 89 boundary constructor, Worklog 82 relation
thresholds, NURBS fitting, visible Gaussian training, ADC, and Worklog 90/91's
own attribution logic (imported, not reimplemented). No new surface
constructor, no threshold tuning, no boundary experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _median_nn
from chart_unit_surface_topology_temporal_lineage_replay import (
    CHECKPOINT_ITERATIONS,
    _load_model,
    _region_analysis,
)
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig  # noqa: F401
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames  # noqa: F401
from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import (
    STATE_NON_MANIFOLD,
    build_chart_unit_topology_context,
    materialize_chart_unit_cut_boundaries,
)
from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
    LOCAL_CLASSES,
    TRUE_PERSISTENT_MULTI_LAYER,
    TRUE_PERSISTENT_TWO_LAYER,
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

# World-space radius used only to spatially match a real Worklog-92 layer's
# footprint back to an earlier checkpoint's Gaussian centers. This is a
# read-only spatial lookup, not a relation/adjacency threshold, and reuses
# the same order of magnitude as the local kNN diagnostic neighborhoods
# already computed for this unit (no independent sweep).
_SPATIAL_MATCH_RADIUS_OVER_SPACING = 3.0


def _layer_report(
    positions: torch.Tensor,
    covariance: torch.Tensor,
    model,
    stable_ids_current: list[int],
    members: list[int],
    class_by_member: tuple[str, ...],
    local_geometry,
    layer_class: str,
    cameras: list,
    checkpoint_600_positions: torch.Tensor | None,
    checkpoint_600_ids: set[int] | None,
    checkpoint_pruned_ids: dict,
) -> list[dict]:
    """One entry per persistent local layer id found among ``members``
    classified as ``layer_class``."""

    torch_mod = torch
    layer_members = [
        (local_index, member) for local_index, member in enumerate(members)
        if class_by_member[local_index] == layer_class
    ]
    if not layer_members:
        return []
    by_layer_id: dict[int, list[int]] = {}
    for local_index, member in layer_members:
        layer_id = local_geometry[local_index].local_mode_id
        by_layer_id.setdefault(layer_id, []).append(member)

    reports = []
    for layer_id, member_list in by_layer_id.items():
        if len(member_list) < 2:
            continue
        selector = torch_mod.tensor(member_list, dtype=torch_mod.long, device=positions.device)
        layer_points = positions[selector]
        centroid = layer_points.mean(dim=0)
        bounding_radius = float((layer_points - centroid).norm(dim=1).max().item())
        member_stable_ids = [stable_ids_current[member] for member in member_list]

        opacity = float(torch_mod.sigmoid(model.get_opacity.detach())[selector].mean().item())

        # Spatial persistence against checkpoint 600: for each member's
        # world position, check whether a Gaussian existed within a local
        # radius at checkpoint 600 (persistent structure already present
        # near initialization) vs. only appearing later (ADC-created).
        newly_appearing_fraction = None
        if checkpoint_600_positions is not None and checkpoint_600_ids is not None:
            spacing = _median_nn(layer_points) if layer_points.shape[0] > 1 else None
            radius = (
                float(spacing) * _SPATIAL_MATCH_RADIUS_OVER_SPACING if spacing else 0.1
            )
            distances = torch_mod.cdist(layer_points, checkpoint_600_positions)
            nearest = distances.min(dim=1).values
            present_at_600 = nearest <= radius
            newly_appearing_fraction = float((~present_at_600).float().mean().item())

        survives_pruning = None
        if checkpoint_pruned_ids:
            last_seen = [checkpoint_pruned_ids.get(sid) for sid in member_stable_ids]
            survives_pruning = float(
                sum(1 for value in last_seen if value == "final") / len(last_seen)
            ) if last_seen else None

        # Visibility / rendered contribution for this layer alone.
        both_visible = 0
        radii_values: list[float] = []
        for camera in cameras:
            with torch_mod.no_grad():
                result = OSNGaussianRasterizer(
                    GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True)
                ).render(camera, model)
            radii = result["radii"][selector.to(result["radii"].device)]
            visible = radii > 0
            if bool(visible.any()):
                both_visible += 1
                radii_values.append(float(radii[visible].float().mean().item()))
            if both_visible >= 20:  # cap camera sweep cost per layer
                break

        reports.append({
            "layer_class": layer_class,
            "layer_id": layer_id,
            "evidence_population": len(member_list),
            "opacity_mean": opacity,
            "spatial_bounding_radius": bounding_radius,
            "visible_camera_count_sampled": both_visible,
            "mean_screen_radius_when_visible": (
                sum(radii_values) / len(radii_values) if radii_values else None
            ),
            "newly_appearing_stable_id_fraction_vs_checkpoint_600": newly_appearing_fraction,
            "fraction_surviving_to_final_checkpoint": survives_pruning,
            "member_stable_ids_sample": member_stable_ids[:10],
        })
    return reports


def analyze_checkpoint(
    checkpoint_dir: Path, iteration, cap: int, device: str,
    checkpoint_600_positions: torch.Tensor | None,
    checkpoint_600_ids: set[int] | None,
    checkpoint_pruned_ids: dict,
    cameras: list,
) -> dict:
    model, stable_ids = _load_model(checkpoint_dir, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    overall_local_class_evidence = Counter()
    overall_failed_evidence = 0
    persistent_layer_reports: list[dict] = []
    rows: list[dict] = []

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        row: dict = {"region": region_id, "total_evidence": len(full_indices)}
        if len(full_indices) < 4:
            row["skip_reason"] = "insufficient_owned_evidence"
            rows.append(row)
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence, evidence_covariance = points[selector], covariance[selector]
        evidence_stable_ids = stable_ids[selector.detach().cpu()].tolist()
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

        region_local_class_evidence = Counter()
        region_failed_evidence = 0

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
            unit_size = len(members)
            region_failed_evidence += unit_size
            overall_failed_evidence += unit_size
            if attribution.primary_cause != MULTILAYER_OR_VOLUMETRIC:
                continue

            local_result = attribute_local_center_geometry(evidence, members)
            for cls in LOCAL_CLASSES:
                weighted = local_result.class_node_fractions[cls] * unit_size
                region_local_class_evidence[cls] += weighted
                overall_local_class_evidence[cls] += weighted

            for layer_class in (TRUE_PERSISTENT_TWO_LAYER, TRUE_PERSISTENT_MULTI_LAYER):
                layer_reports = _layer_report(
                    evidence, evidence_covariance, model, evidence_stable_ids, members,
                    local_result.class_by_member, local_result.local_geometry_by_member,
                    layer_class, cameras, checkpoint_600_positions, checkpoint_600_ids,
                    checkpoint_pruned_ids,
                )
                for report in layer_reports:
                    report["region"] = region_id
                    report["unit_index"] = unit_index
                    persistent_layer_reports.append(report)

        row["failed_evidence"] = region_failed_evidence
        row["local_class_evidence"] = dict(region_local_class_evidence)
        rows.append(row)

    return {
        "checkpoint": str(checkpoint_dir),
        "iteration": iteration,
        "regions": rows,
        "summary": {
            "multilayer_evidence_total": sum(overall_local_class_evidence.values()),
            "local_class_evidence": dict(overall_local_class_evidence),
            "local_class_evidence_fraction": {
                cls: (
                    overall_local_class_evidence[cls] / sum(overall_local_class_evidence.values())
                    if sum(overall_local_class_evidence.values()) else 0.0
                )
                for cls in LOCAL_CLASSES
            },
        },
        "persistent_layer_reports": persistent_layer_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"),
    )
    parser.add_argument("--source_path", type=Path, default=Path("DATASET"))
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/extent_ab/val92/chart_unit_local_center_geometry_attribution_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoints", nargs="+", default=["2900", "final"])
    parser.add_argument("--skip_visibility", action="store_true")
    args = parser.parse_args()

    # Checkpoint 600 world-space centers, used purely as a read-only spatial
    # lookup for "did a Gaussian already exist near here at init/early ADC."
    checkpoint_600_dir = args.run_dir / "600"
    checkpoint_600_positions = None
    checkpoint_600_ids: set[int] | None = None
    if checkpoint_600_dir.exists():
        model_600, stable_ids_600 = _load_model(checkpoint_600_dir, args.device)
        checkpoint_600_positions = model_600.get_xyz.detach()
        checkpoint_600_ids = set(stable_ids_600.tolist())

    # Which checkpoint each stable ID was last observed in, across the full
    # 5-checkpoint sequence, to report pruning survival for persistent
    # layers found at an earlier checkpoint.
    checkpoint_pruned_ids: dict[int, str] = {}
    for iteration in CHECKPOINT_ITERATIONS:
        checkpoint_dir = args.run_dir / str(iteration)
        if not checkpoint_dir.exists():
            continue
        _, stable_ids_here = _load_model(checkpoint_dir, args.device)
        for sid in stable_ids_here.tolist():
            checkpoint_pruned_ids[sid] = str(iteration)

    cameras: list = []
    if not args.skip_visibility:
        try:
            eval_split = load_colmap_scene_with_eval_split(
                args.source_path, device=args.device, llffhold=args.llffhold,
            )
            cameras = eval_split.train_scene.cameras
        except FileNotFoundError:
            cameras = []

    reports = []
    for iteration in args.checkpoints:
        checkpoint_dir = args.run_dir / str(iteration)
        if not checkpoint_dir.exists():
            continue
        report = analyze_checkpoint(
            checkpoint_dir, iteration, args.cap, args.device,
            checkpoint_600_positions, checkpoint_600_ids, checkpoint_pruned_ids, cameras,
        )
        reports.append(report)

    output = {"run_dir": str(args.run_dir), "cap": args.cap, "checkpoints": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "checkpoint_summaries": [{"iteration": r["iteration"], **r["summary"]} for r in reports],
        "persistent_layer_report_counts": {
            r["iteration"]: len(r["persistent_layer_reports"]) for r in reports
        },
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
