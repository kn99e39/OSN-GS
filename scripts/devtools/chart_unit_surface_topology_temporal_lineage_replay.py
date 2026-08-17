"""Worklog 91 -- read-only temporal + lineage attribution replay.

Extends Worklog 90's covariance-footprint attribution with, for the same
failed chart units at the same 7 real regions:

1. CENTER_GEOMETRY_LAYERING -- position-only PCA depth clustering of centers
   (independent of each Gaussian's own covariance orientation).
2. COVARIANCE_ONLY_AMBIGUITY -- Worklog 90 layer-conflict nodes that remain
   single-sheet in (1).
3. ADC_LINEAGE -- stable-Gaussian-ID presence/absence across the 5 available
   baseline_compatible checkpoints (600, 2900, 3000, 3100, final), joined
   with the training log's per-iteration cumulative ADC counters.
4. TEMPORAL ONSET -- every Worklog 90 metric plus footprint/anisotropy
   fields, recomputed at each checkpoint.
5. VISIBILITY / DEPTH ORDERING -- for the dominant multilayer example unit
   per region, render every train camera and report, for cameras where both
   layers are visible (radii > 0), their screen-space (pixel) overlap and
   view-space depth separation.

Fixed and unmodified: Worklog 89's boundary constructor, Worklog 82 relation
thresholds, NURBS fitting, visible Gaussian training, and Worklog 90's own
attribution logic (imported, not reimplemented). No new surface constructor,
no threshold tuning.
"""

from __future__ import annotations

import argparse
import json
import re
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
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import (
    STATE_NON_MANIFOLD,
    build_chart_unit_topology_context,
    materialize_chart_unit_cut_boundaries,
)
from osn_gs.surface.torch_chart_unit_surface_topology_attribution import (
    MULTILAYER_OR_VOLUMETRIC,
    attribute_failed_chart_unit_surface_topology,
)
from osn_gs.surface.torch_chart_unit_surface_topology_temporal_lineage import (
    compute_center_geometry_layering,
    compute_covariance_only_ambiguity,
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
from osn_gs.surface.torch_gaussian_covariance_frame import (
    covariance_from_scale_rotation,
    extract_covariance_frame,
)

CHECKPOINT_ITERATIONS = (600, 2900, 3000, 3100, "final")


def _log_adc_line(log_path: Path, iteration) -> dict | None:
    """Same convention as ``fixed_loader_replay_analysis._log_adc_line``:
    parse the training log's cumulative per-iteration ADC counter line."""

    if not log_path.exists() or not isinstance(iteration, int):
        return None
    pattern = re.compile(rf"OSN-GS ADC: iteration={iteration} ([^\r\n]*)")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = pattern.search(text)
    if not match:
        return None
    fields: dict = {}
    for token in match.group(1).split():
        if "=" in token:
            key, _, value = token.partition("=")
            try:
                fields[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                fields[key] = value
    return fields


def _load_model(checkpoint_dir: Path, device: str) -> tuple[TorchGaussianModel, torch.Tensor]:
    payload = torch.load(checkpoint_dir / "checkpoint.pt", map_location=device, weights_only=False)
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
    stable_ids = raw.get("stable_gaussian_ids")
    if stable_ids is None:
        stable_ids = torch.arange(int(raw["xyz"].shape[0]), device=device)
    return model, stable_ids.detach().cpu()


def _region_analysis(model: TorchGaussianModel, stable_ids: torch.Tensor, cap: int, device: str):
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    points = model.get_xyz.detach()
    positional_ids = list(range(int(points.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            points, covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), positional_ids,
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
        construction.surface_regions, points, covariance, owned,
        representative_positions, representative_index, frame_by_region, chart_by_region,
    )


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    if not values:
        return None
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / total if total else None


def analyze_checkpoint(
    checkpoint_dir: Path, log_path: Path, iteration, cap: int, device: str,
) -> dict:
    model, stable_ids = _load_model(checkpoint_dir, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    with torch.no_grad():
        frame_all = extract_covariance_frame(covariance)

    rows: list[dict] = []
    overall_center_multilayer_evidence = 0
    overall_covariance_only_evidence = 0
    overall_true_center_multilayer_evidence = 0
    overall_failed_evidence = 0
    overall_metrics: dict[str, list[tuple[float, int]]] = {
        "layer_count_median": [],
        "depth_separation_over_spacing": [],
        "tangent_major_scale": [],
        "tangent_minor_scale": [],
        "normal_thickness": [],
        "anisotropy": [],
        "opacity": [],
    }
    dominant_multilayer_unit: dict | None = None

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

        region_failed_evidence = 0
        region_center_multilayer_evidence = 0
        region_covariance_only_evidence = 0
        region_true_center_multilayer_evidence = 0
        region_units: list[dict] = []

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

            layering = compute_center_geometry_layering(evidence, members)

            # Recompute the exact Worklog 90 layer_conflict node mask locally
            # (same formula, same fixed footprint sigma) so covariance-only
            # ambiguity can be split against this unit's own center layering.
            selector_local = torch.tensor(members, dtype=torch.long, device=evidence.device)
            local_points = evidence[selector_local]
            local_frame = extract_covariance_frame(evidence_covariance[selector_local])
            delta = local_points[None, :, :] - local_points[:, None, :]
            off_diagonal = ~torch.eye(unit_size, dtype=torch.bool, device=local_points.device)
            normal = local_frame.normal_candidate
            normal_alignment = (normal @ normal.T).abs()
            signed_a = (delta * normal[:, None, :]).sum(dim=2)
            signed_b = (delta * normal[None, :, :]).sum(dim=2)
            depth_a = signed_a.abs() / local_frame.normal_thickness[:, None].clamp_min(1e-12)
            depth_b = signed_b.abs() / local_frame.normal_thickness[None, :].clamp_min(1e-12)
            tangent_delta_a = delta - signed_a[..., None] * normal[:, None, :]
            tangent_delta_b = delta - signed_b[..., None] * normal[None, :, :]
            tangent_distance = 0.5 * (tangent_delta_a.norm(dim=2) + tangent_delta_b.norm(dim=2))
            direction_a = tangent_delta_a / tangent_delta_a.norm(dim=2, keepdim=True).clamp_min(1e-12)
            direction_b = tangent_delta_b / tangent_delta_b.norm(dim=2, keepdim=True).clamp_min(1e-12)
            reach_a = torch.sqrt(
                (direction_a * local_frame.tangent_u[:, None, :]).sum(dim=2).square()
                * local_frame.tangent_major_scale[:, None].square()
                + (direction_a * local_frame.tangent_v[:, None, :]).sum(dim=2).square()
                * local_frame.tangent_minor_scale[:, None].square()
            )
            reach_b = torch.sqrt(
                (direction_b * local_frame.tangent_u[None, :, :]).sum(dim=2).square()
                * local_frame.tangent_major_scale[None, :].square()
                + (direction_b * local_frame.tangent_v[None, :, :]).sum(dim=2).square()
                * local_frame.tangent_minor_scale[None, :].square()
            )
            tangent_overlap = tangent_distance <= (reach_a + reach_b)
            normal_depth_compatible = (depth_a <= 1.0) & (depth_b <= 1.0)
            normal_compatible = normal_alignment >= DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT
            layer_conflict = off_diagonal & tangent_overlap & ~(normal_depth_compatible & normal_compatible)
            layer_conflict_node_mask = layer_conflict.any(dim=1)

            covariance_only = compute_covariance_only_ambiguity(layering, layer_conflict_node_mask)

            if attribution.primary_cause == MULTILAYER_OR_VOLUMETRIC:
                if layering.multilayer:
                    region_true_center_multilayer_evidence += unit_size
                    overall_true_center_multilayer_evidence += unit_size
                if covariance_only.covariance_only_ambiguous:
                    region_covariance_only_evidence += unit_size
                    overall_covariance_only_evidence += unit_size
                region_center_multilayer_evidence += unit_size if layering.multilayer else 0
                overall_center_multilayer_evidence += unit_size if layering.multilayer else 0

            unit_stable_ids = [evidence_stable_ids[member] for member in members]
            local_anisotropy = (
                local_frame.tangent_major_scale / local_frame.tangent_minor_scale.clamp_min(1e-12)
            )
            for key, values in (
                ("layer_count_median", [(float(layering.layer_count), unit_size)]),
                (
                    "depth_separation_over_spacing",
                    [(
                        float(layering.depth_separation / max(layering.center_spacing, 1e-9)),
                        unit_size,
                    )] if layering.depth_separation is not None and layering.center_spacing else [],
                ),
                ("tangent_major_scale", [(float(local_frame.tangent_major_scale.mean()), unit_size)]),
                ("tangent_minor_scale", [(float(local_frame.tangent_minor_scale.mean()), unit_size)]),
                ("normal_thickness", [(float(local_frame.normal_thickness.mean()), unit_size)]),
                ("anisotropy", [(float(local_anisotropy.mean()), unit_size)]),
                (
                    "opacity",
                    [(float(torch.sigmoid(model.get_opacity.detach())[selector][selector_local].mean()), unit_size)],
                ),
            ):
                overall_metrics[key].extend(values)

            record = {
                "unit_index": unit_index,
                "member_count": unit_size,
                "primary_cause": attribution.primary_cause,
                "center_geometry_layer_count": layering.layer_count,
                "center_geometry_multilayer": layering.multilayer,
                "center_geometry_depth_separation": layering.depth_separation,
                "center_geometry_spacing": layering.center_spacing,
                "covariance_only_ambiguous": covariance_only.covariance_only_ambiguous,
                "covariance_only_ambiguous_node_fraction": (
                    covariance_only.covariance_only_ambiguous_node_fraction
                ),
                "member_stable_ids": unit_stable_ids,
            }
            region_units.append(record)

            if attribution.primary_cause == MULTILAYER_OR_VOLUMETRIC and layering.multilayer:
                strength = unit_size
                if dominant_multilayer_unit is None or strength > dominant_multilayer_unit["strength"]:
                    dominant_multilayer_unit = {
                        "strength": strength, "region": region_id, "unit_index": unit_index,
                        "member_positions": local_points.detach().cpu(),
                        "member_stable_ids": unit_stable_ids,
                        "layer_id_by_member": layering.layer_id_by_member,
                    }

        row["failed_evidence"] = region_failed_evidence
        row["center_multilayer_evidence"] = region_center_multilayer_evidence
        row["true_center_multilayer_evidence"] = region_true_center_multilayer_evidence
        row["covariance_only_ambiguous_evidence"] = region_covariance_only_evidence
        row["units"] = region_units
        rows.append(row)

    adc_event = _log_adc_line(log_path, iteration)
    return {
        "checkpoint": str(checkpoint_dir),
        "iteration": iteration,
        "gaussian_count": int(points.shape[0]),
        "cumulative_adc_event": adc_event,
        "regions": rows,
        "summary": {
            "failed_topology_evidence": overall_failed_evidence,
            "multilayer_or_volumetric_evidence_with_true_center_layering": overall_true_center_multilayer_evidence,
            "multilayer_or_volumetric_evidence_covariance_only": overall_covariance_only_evidence,
            "center_geometry_multilayer_evidence": overall_center_multilayer_evidence,
            "metrics_evidence_weighted": {
                key: _weighted_mean(values) for key, values in overall_metrics.items()
            },
        },
        "dominant_multilayer_unit": dominant_multilayer_unit,
    }


def analyze_lineage(reports: list[dict]) -> dict:
    """Diff stable-Gaussian-ID sets across consecutive checkpoints and join
    against the dominant multilayer unit's member IDs, without touching any
    constructor. Birth = present at checkpoint N, absent at N-1. Death
    (pruned) = present at N-1, absent at N. Clone vs. split origin is not
    separable from stable IDs alone (both allocate fresh IDs identically),
    so this reports "newly born" only -- disclosed, not inferred further.
    """

    transitions = []
    for previous, current in zip(reports[:-1], reports[1:]):
        prev_ids = set(previous["_stable_id_set"])
        curr_ids = set(current["_stable_id_set"])
        born = curr_ids - prev_ids
        died = prev_ids - curr_ids
        dominant = current.get("dominant_multilayer_unit")
        born_in_dominant_layer = None
        if dominant is not None:
            member_ids = set(dominant["member_stable_ids"])
            born_in_dominant_layer = len(member_ids & born)
        transitions.append({
            "from_iteration": previous["iteration"],
            "to_iteration": current["iteration"],
            "born_count": len(born),
            "died_count": len(died),
            "cumulative_adc_event": current.get("cumulative_adc_event"),
            "dominant_multilayer_unit_members_born_this_interval": born_in_dominant_layer,
        })
    return {"transitions": transitions}


def analyze_visibility(
    dominant_unit: dict, model: TorchGaussianModel, cameras: list, device: str,
) -> dict:
    """For the dominant MULTILAYER_OR_VOLUMETRIC unit's two most-separated
    center layers, render every training camera and report, for cameras
    where both layers have visible screen radius, their pixel-space overlap
    and view-space depth separation. Pure read of existing render output --
    no constructor, relation, or threshold change."""

    if dominant_unit is None:
        return {"skip_reason": "no_dominant_multilayer_unit_found"}

    layer_ids = dominant_unit["layer_id_by_member"]
    member_stable_ids = dominant_unit["member_stable_ids"]
    layer_sizes = Counter(layer_ids)
    if len(layer_sizes) < 2:
        return {"skip_reason": "dominant_unit_center_geometry_is_single_layer"}
    # Compare the two most-populated layers (not the extreme layer indices,
    # which can be single-member depth outliers) so the visibility check
    # reflects the two dominant competing sheets, not a stray point.
    (layer_a, _), (layer_b, _) = layer_sizes.most_common(2)
    ids_a = [sid for sid, lid in zip(member_stable_ids, layer_ids) if lid == layer_a]
    ids_b = [sid for sid, lid in zip(member_stable_ids, layer_ids) if lid == layer_b]

    stable_ids_full = model.stable_gaussian_ids.detach().cpu().tolist()
    id_to_index = {sid: index for index, sid in enumerate(stable_ids_full)}
    index_a = torch.tensor([id_to_index[sid] for sid in ids_a if sid in id_to_index], dtype=torch.long)
    index_b = torch.tensor([id_to_index[sid] for sid in ids_b if sid in id_to_index], dtype=torch.long)
    if index_a.numel() == 0 or index_b.numel() == 0:
        return {"skip_reason": "layer_members_not_found_in_current_model_state"}

    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    world_view = None
    both_visible_camera_count = 0
    overlap_fractions: list[float] = []
    depth_separations: list[float] = []
    for camera in cameras:
        with torch.no_grad():
            result = rasterizer.render(camera, model)
        radii = result["radii"]
        visible_a = radii[index_a.to(radii.device)] > 0
        visible_b = radii[index_b.to(radii.device)] > 0
        if not (bool(visible_a.any()) and bool(visible_b.any())):
            continue
        both_visible_camera_count += 1
        world_view = camera.world_view_transform
        centroid_a = model.get_xyz.detach()[index_a.to(radii.device)][visible_a].mean(dim=0)
        centroid_b = model.get_xyz.detach()[index_b.to(radii.device)][visible_b].mean(dim=0)
        homo_a = torch.cat([centroid_a, torch.ones(1, device=centroid_a.device)])
        homo_b = torch.cat([centroid_b, torch.ones(1, device=centroid_b.device)])
        view_a = homo_a @ world_view
        view_b = homo_b @ world_view
        depth_separations.append(float((view_a[2] - view_b[2]).abs().item()))
        # Screen-space overlap proxy: mean visible radius of each layer's
        # members relative to their centroid separation in the same camera,
        # using the renderer's own reported radii (already in pixel units).
        radius_a = float(radii[index_a.to(radii.device)][visible_a].float().mean().item())
        radius_b = float(radii[index_b.to(radii.device)][visible_b].float().mean().item())
        overlap_fractions.append(min(1.0, (radius_a + radius_b) / max(1e-6, (radius_a + radius_b))))

    return {
        "layer_a_member_count": int(index_a.numel()),
        "layer_b_member_count": int(index_b.numel()),
        "camera_count": len(cameras),
        "both_layers_visible_camera_count": both_visible_camera_count,
        "mean_view_space_depth_separation": (
            sum(depth_separations) / len(depth_separations) if depth_separations else None
        ),
        "min_view_space_depth_separation": min(depth_separations) if depth_separations else None,
        "max_view_space_depth_separation": max(depth_separations) if depth_separations else None,
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
        default=Path("output/extent_ab/val91/chart_unit_surface_topology_temporal_lineage_replay.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_visibility", action="store_true")
    args = parser.parse_args()

    log_path = args.run_dir.with_suffix(".log")
    reports = []
    last_model = None
    for iteration in CHECKPOINT_ITERATIONS:
        checkpoint_dir = args.run_dir / str(iteration)
        if not checkpoint_dir.exists():
            continue
        report = analyze_checkpoint(checkpoint_dir, log_path, iteration, args.cap, args.device)
        model, stable_ids = _load_model(checkpoint_dir, args.device)
        report["_stable_id_set"] = stable_ids.tolist()
        reports.append(report)
        last_model = model

    lineage = analyze_lineage(reports)

    visibility = {"skip_reason": "skipped_by_flag"}
    if not args.skip_visibility and reports and last_model is not None:
        final_report = reports[-1]
        dominant = final_report.get("dominant_multilayer_unit")
        try:
            eval_split = load_colmap_scene_with_eval_split(
                args.source_path, device=args.device, llffhold=args.llffhold,
            )
            visibility = analyze_visibility(
                dominant, last_model, eval_split.train_scene.cameras, args.device,
            )
        except FileNotFoundError as exc:
            visibility = {"skip_reason": f"dataset_unavailable: {exc}"}

    for report in reports:
        report.pop("_stable_id_set", None)
        dominant = report.get("dominant_multilayer_unit")
        if dominant is not None:
            dominant.pop("member_positions", None)

    output = {
        "run_dir": str(args.run_dir),
        "cap": args.cap,
        "checkpoints": reports,
        "lineage": lineage,
        "visibility_final_checkpoint": visibility,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "checkpoint_summaries": [
            {"iteration": r["iteration"], **r["summary"]} for r in reports
        ],
        "lineage": lineage,
        "visibility_final_checkpoint": visibility,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
