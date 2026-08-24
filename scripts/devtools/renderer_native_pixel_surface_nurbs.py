"""Worklog 112 -- Renderer-Native Pixel Surface as NURBS Fitting Geometry.

Controlled A/B replay of Worklog 111. Preserves EXACTLY: the WL107/109
canonical topology (`torch_camera_induced_visible_adjacency.py`, unmodified),
WL111's image-space chart connectivity (`label_same_component_blobs`, reused
unmodified), and WL111's fixed NURBS config (degree 2, 8x4 control grid).
Changes ONLY the 3D fitting target: instead of collapsing each representative
surfel's pixels to one mean sample at its OWN CENTER (WL111 / Worklog 110's
frozen constraint kept the wrong geometry assumption alive), every valid
renderer pixel keeps its OWN renderer-native unprojected surface position --
the same `median_depth` channel WL107's `median_surfel_id` is captured from
(`out_others[MIDDEPTH_OFFSET]`), unprojected via the already-established,
OFFICIAL_CODE_FAITHFUL `osn_gs.render.surfel_geometry.depths_to_points`
(used elsewhere for the training-time depth-normal consistency loss -- not a
new mechanism invented for this batch).

Does NOT use Worklog 110's non-representative evidence (still AMBIGUOUS/
LAYERED SUPPORT). Does NOT modify torch_camera_induced_visible_adjacency.py,
torch_camera_observed_chart_domains.py's existing WL111 functions, or
torch_nurbs.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import (  # noqa: E402
    load_primitive_model, checkpoint_primitive, PRIMITIVE_SURFEL_2D,
    _hsv_to_rgb, _rgb_to_f_dc, write_surfel_ply, write_ppm,
)
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402
from osn_gs.render.surfel_geometry import depths_to_points
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative
from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    CameraInducedAdjacencyConfig,
    accumulate_image_space_pairs,
    apply_secondary_geometric_gate,
    filter_by_3d_locality,
)
from osn_gs.surface.torch_camera_observed_chart_domains import build_view_chart_pixel_samples, valid_pixel_chart_mask
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"
_EPS = 1e-9
_MIDDEPTH_OFFSET = 5
_UNCUT_RGB = (0.08, 0.09, 0.11)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

# Worklog 111's fixed configuration, PRESERVED exactly (directive section 1).
RESOLUTION_U = 8
RESOLUTION_V = 4
DEGREE_U = 2
DEGREE_V = 2
# Directive section 7: 32 is the support requirement of THIS fixed 8x4
# model's control-point count -- reinterpreted this batch as a PIXEL-SAMPLE
# count (not representative-component size, section 8).
MIN_PIXEL_SAMPLES = RESOLUTION_U * RESOLUTION_V

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_WL111_BASELINE = "WL111_REPRESENTATIVE_CENTER_NURBS"
VIEW_PIXEL_SURFACE = "RENDERER_NATIVE_PIXEL_SURFACE"
VIEW_CHART_DOMAINS = "PIXEL_SURFACE_CHART_DOMAINS"
VIEW_NURBS = "PIXEL_SURFACE_NURBS"
VIEW_COVERAGE = "PIXEL_SURFACE_COVERAGE"
VIEW_RESIDUAL_COMPARE = "WL111_VS_PIXEL_SURFACE_RESIDUAL"
VIEW_OVERLAP = "OVERLAP_DISAGREEMENT"
VIEW_TABLE = "TABLE_PIXEL_SURFACE_NURBS"
VIEW_CURVED = "CURVED_STRUCTURE_PIXEL_SURFACE_NURBS"
VIEW_HEDGE = "HEDGE_PIXEL_SURFACE_NURBS"

ANCHOR_FRACTIONS = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[pixel-surface nurbs] {message}", flush=True)


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    if int(values.shape[0]) == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    sorted_values = torch.sort(values.to(torch.float64)).values

    def _percentile(fraction: float) -> float:
        position = min(int(sorted_values.shape[0]) - 1, max(0, int(round(fraction * (int(sorted_values.shape[0]) - 1)))))
        return float(sorted_values[position].item())

    return {
        "min": _percentile(0.0), "median": _percentile(0.5), "mean": float(sorted_values.mean().item()),
        "p95": _percentile(0.95), "max": _percentile(1.0), "count": int(sorted_values.shape[0]),
    }


def _ramp(ratio: torch.Tensor, low_rgb, high_rgb) -> torch.Tensor:
    low = torch.tensor(low_rgb, device=ratio.device).reshape(1, 3)
    high = torch.tensor(high_rgb, device=ratio.device).reshape(1, 3)
    return low + ratio.clamp(0.0, 1.0).reshape(-1, 1) * (high - low)


def _hash_colors(identifiers: torch.Tensor) -> torch.Tensor:
    identifiers = identifiers.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--preview-camera-images", default=None)
    parser.add_argument("--wl111-report", type=Path, default=Path("output/osn_gs_rep_only_nurbs/representative_only_visible_nurbs_report.json"))
    parser.add_argument("--wl111-export-dir", type=Path, default=Path("output/osn_gs_rep_only_nurbs"))
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--max-charts", type=int, default=0)
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)

    wl111_reference: dict[str, Any] | None = None
    if arguments.wl111_report.exists():
        wl111_reference = json.loads(arguments.wl111_report.read_text(encoding="utf-8"))
        _progress(f"loaded WL111 reference report: {arguments.wl111_report}")
    else:
        _progress(f"WARNING: WL111 reference report not found at {arguments.wl111_report}; comparison sections will be empty")

    _progress(f"loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint (primitive={primitive!r}).")
    total_model_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    visible_count = int(visible_selector.shape[0])
    _progress(f"model surfels={total_model_count} visible={visible_count} iteration={payload.get('iteration')}")

    _progress("loading all train cameras")
    cameras, camera_meta = load_all_train_cameras(arguments.source_path, arguments.images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device)
    if int(arguments.max_views) > 0:
        cameras = cameras[: int(arguments.max_views)]
        camera_meta = {**camera_meta, "smoke_test_max_views": int(arguments.max_views)}
    _progress(f"train cameras: {camera_meta}")

    device = model.device
    full_to_visible = torch.full((total_model_count,), -1, dtype=torch.int64, device=device)
    full_to_visible[visible_selector] = torch.arange(visible_count, dtype=torch.int64, device=device)

    with torch.no_grad():
        full_orientation = derive_surface_orientation_from_surfel(model)
        orientation = _dc_replace(
            full_orientation,
            gaussian_ids=full_orientation.gaussian_ids[visible_selector],
            positions=full_orientation.positions[visible_selector],
            tangent_axis_u=full_orientation.tangent_axis_u[visible_selector],
            tangent_axis_v=full_orientation.tangent_axis_v[visible_selector],
            surface_normal=full_orientation.surface_normal[visible_selector],
            tangent_scale_u=full_orientation.tangent_scale_u[visible_selector],
            tangent_scale_v=full_orientation.tangent_scale_v[visible_selector],
        )
        positions = orientation.positions
        count = int(positions.shape[0])

        local_config = CoverageFirstPartitionConfig()
        config = CameraInducedAdjacencyConfig(local=local_config)
        _progress("[WL107/109 replay, unchanged] build_candidate_graph")
        graph = build_candidate_graph(orientation, config.local, progress=_progress)

    # --- sweep: representative maps AND renderer-native median-depth unprojection ---
    _progress("[sweep] representative_id + median_depth unprojection per view")
    per_view_rep_remapped: list[torch.Tensor] = []
    per_view_world_points: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=device)
    total_valid_pixels_all_views = 0

    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        represented_ids = torch.unique(rep_full[valid])
        ever_representative_full[represented_ids] = True
        total_valid_pixels_all_views += int(valid.sum().item())

        depth_map = diag["out_others"][_MIDDEPTH_OFFSET]
        with torch.no_grad():
            world_points = depths_to_points(camera, depth_map.unsqueeze(0)).reshape(*depth_map.shape, 3)

        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_rep_remapped.append(rep_remapped.detach().cpu())
        per_view_world_points.append(world_points.detach().cpu())
        del diag
        if index % 20 == 0:
            _progress(f"sweep view {index + 1}/{len(cameras)}")

    ever_representative = ever_representative_full[visible_selector]
    representative_count = int(ever_representative.sum().item())
    _progress(f"[accounting] median_surface_representatives={representative_count}")

    # --- replay WL107/109 topology exactly (identical, read-only functions) ---
    with torch.no_grad():
        _progress("[WL107/109 replay, unchanged] accumulate_image_space_pairs")
        per_view_rep_gpu = [t.to(device) for t in per_view_rep_remapped]
        raw_pairs, _raw_view_support = accumulate_image_space_pairs(count, per_view_rep_gpu, progress=_progress)
        local_pairs, _local_mask = filter_by_3d_locality(raw_pairs, count, graph)
        _progress("[WL107/109 replay, unchanged] apply_secondary_geometric_gate")
        geometry = apply_secondary_geometric_gate(local_pairs, orientation, config, progress=_progress)
        positive_edges = local_pairs[geometry["kept_mask"]]
        roots = _connected_component_roots(count, positive_edges, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
        subset_ids = subset_id_of_position[inverse]
        subset_sizes = counts[order]
        subset_count = int(order.shape[0])
        wl107_replay_stats = {
            "visible_component_count": subset_count,
            "largest_component_surfel_fraction": float(subset_sizes[0]) / count,
            "singleton_surfel_count": int((subset_sizes == 1).sum()),
        }
        _progress(f"[replay consistency check] {wl107_replay_stats}")

    # --- chart candidates + NURBS fitting over DENSE renderer-native pixel surface ---
    ever_in_valid_chart = torch.zeros((count,), dtype=torch.bool, device=device)
    valid_chart_membership_count = torch.zeros((count,), dtype=torch.int64, device=device)
    raw_chart_count = 0
    valid_chart_count = 0
    covered_valid_pixels_all_views = 0

    all_member_ids: list[torch.Tensor] = []
    all_chart_ids: list[torch.Tensor] = []
    all_view_ids: list[torch.Tensor] = []
    all_fitted_points: list[torch.Tensor] = []
    all_normals: list[torch.Tensor] = []
    residual_values: list[torch.Tensor] = []
    residual_pixel_rep_ids: list[torch.Tensor] = []

    global_chart_id = 0
    max_charts = int(arguments.max_charts) if int(arguments.max_charts) > 0 else None
    stop = False
    for view_index, (rep_remapped_cpu, world_points_cpu) in enumerate(zip(per_view_rep_remapped, per_view_world_points)):
        if stop:
            break
        rep_gpu = rep_remapped_cpu.to(device)
        world_gpu = world_points_cpu.to(device)
        valid_pix = rep_gpu >= 0
        comp_map = torch.where(valid_pix, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1))
        vs = build_view_chart_pixel_samples(view_index, comp_map, rep_gpu, world_gpu)
        if vs.blob_count == 0:
            continue
        mask = valid_pixel_chart_mask(vs, MIN_PIXEL_SAMPLES)
        raw_chart_count += vs.blob_count
        valid_chart_count += int(mask.sum().item())
        covered_valid_pixels_all_views += int(vs.blob_pixel_total[mask].sum().item()) if int(mask.sum().item()) > 0 else 0

        valid_blob_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        for local_blob in valid_blob_ids:
            if max_charts is not None and global_chart_id >= max_charts:
                stop = True
                break
            pixel_sel = vs.pixel_blob_id == local_blob
            pixel_uv = vs.pixel_uv[pixel_sel]
            pixel_xyz = vs.pixel_xyz[pixel_sel]
            pixel_rep_ids = vs.pixel_representative_id[pixel_sel]
            component_id = int(vs.blob_component_id[local_blob].item())

            with torch.no_grad():
                surface, uv = fit_torch_visible_surface_lsq(
                    pixel_xyz, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                    degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=pixel_uv,
                    correction_rounds=2, projection_iterations=3,
                )
                fitted = surface.evaluate(uv)
                normals = surface.normals(uv)
                residual = (fitted - pixel_xyz).norm(dim=-1)

            distinct_reps, inverse_rep = torch.unique(pixel_rep_ids, return_inverse=True)
            ever_in_valid_chart[distinct_reps] = True
            valid_chart_membership_count.index_add_(0, distinct_reps, torch.ones_like(distinct_reps))

            # Per-DISTINCT-representative mean fitted point/normal within this
            # chart -- a diagnostic-only aggregate for cross-chart OVERLAP
            # comparison (directive section 12), never fed back into fitting
            # (which already used every dense pixel above). One aligned row
            # per representative this chart touches.
            rep_count = int(distinct_reps.shape[0])
            sum_point = torch.zeros((rep_count, 3), dtype=torch.float32, device=device)
            sum_normal = torch.zeros((rep_count, 3), dtype=torch.float32, device=device)
            member_counts = torch.zeros((rep_count,), dtype=torch.float32, device=device)
            sum_point.index_add_(0, inverse_rep, fitted)
            sum_normal.index_add_(0, inverse_rep, normals)
            member_counts.index_add_(0, inverse_rep, torch.ones_like(inverse_rep, dtype=torch.float32))
            mean_point = sum_point / member_counts.clamp_min(1.0)[:, None]
            mean_normal = torch.nn.functional.normalize(sum_normal, dim=-1, eps=1e-8)

            all_member_ids.append(distinct_reps.detach().cpu())
            all_chart_ids.append(torch.full((rep_count,), global_chart_id, dtype=torch.int64))
            all_view_ids.append(torch.full((rep_count,), view_index, dtype=torch.int64))
            all_fitted_points.append(mean_point.detach().cpu())
            all_normals.append(mean_normal.detach().cpu())
            residual_values.append(residual.detach().cpu())
            residual_pixel_rep_ids.append(pixel_rep_ids.detach().cpu())
            global_chart_id += 1
        if view_index % 10 == 0:
            _progress(f"chart-fit view {view_index + 1}/{len(per_view_rep_remapped)} raw_charts={raw_chart_count} valid_charts={valid_chart_count} fitted={global_chart_id}")

    _progress(f"[chart accounting] raw_chart_count={raw_chart_count} valid_chart_count={valid_chart_count} fitted={global_chart_id}")

    ever_in_valid_chart_visible = ever_in_valid_chart & ever_representative
    no_valid_chart = ever_representative & ~ever_in_valid_chart
    multi_valid_chart = ever_representative & (valid_chart_membership_count >= 2)
    single_valid_chart = ever_representative & (valid_chart_membership_count == 1)

    residual_all = torch.cat(residual_values) if residual_values else torch.zeros((0,))
    residual_distribution = _distribution(residual_all)

    # --- overlap: consecutive-pair sampling per representative (same technique as WL111,
    # directive section 9/12) -- one row per (chart, representative) is exactly what the
    # fit loop above just appended (per-representative mean point/normal within that chart). ---
    position_discrepancies: list[torch.Tensor] = []
    normal_discrepancies: list[torch.Tensor] = []
    if all_member_ids:
        cat_member = torch.cat(all_member_ids)
        cat_points = torch.cat(all_fitted_points)
        cat_normals = torch.cat(all_normals)
        order = torch.argsort(cat_member)
        sorted_member = cat_member[order]
        sorted_points = cat_points[order]
        sorted_normals = cat_normals[order]
        same_as_next = sorted_member[:-1] == sorted_member[1:]
        if bool(same_as_next.any()):
            diff = (sorted_points[1:] - sorted_points[:-1])[same_as_next]
            position_discrepancies.append(diff.norm(dim=-1))
            cos = (sorted_normals[1:] * sorted_normals[:-1]).sum(dim=-1)[same_as_next].clamp(-1.0, 1.0)
            normal_discrepancies.append(torch.rad2deg(torch.acos(cos)))
    position_discrepancy_distribution = _distribution(torch.cat(position_discrepancies) if position_discrepancies else torch.zeros((0,)))
    normal_discrepancy_distribution = _distribution(torch.cat(normal_discrepancies) if normal_discrepancies else torch.zeros((0,)))
    _progress(f"[overlap] position_discrepancy={position_discrepancy_distribution} normal_discrepancy_degrees={normal_discrepancy_distribution}")

    # --- per-component chartability ---
    representative_per_component = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    representative_per_component.index_add_(0, subset_ids[ever_representative], torch.ones((int(ever_representative.sum().item()),), dtype=torch.int64, device=device))
    covered_per_component = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    covered_ids = torch.nonzero(ever_in_valid_chart_visible, as_tuple=False).reshape(-1)
    if int(covered_ids.shape[0]) > 0:
        covered_per_component.index_add_(0, subset_ids[covered_ids], torch.ones((int(covered_ids.shape[0]),), dtype=torch.int64, device=device))
    nonzero_component_mask = representative_per_component > 0
    per_component_fraction = torch.zeros((subset_count,), dtype=torch.float32, device=device)
    per_component_fraction[nonzero_component_mask] = covered_per_component[nonzero_component_mask].to(torch.float32) / representative_per_component[nonzero_component_mask].to(torch.float32)
    per_component_coverage_distribution = _distribution(per_component_fraction[nonzero_component_mask])

    # --- region masks (WORKING INTERPRETATION ONLY, same anchor mechanism as WL108-111) ---
    preview_images = arguments.preview_camera_images or arguments.images
    preview_cameras, _preview_meta = load_all_train_cameras(arguments.source_path, preview_images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device)
    preview_camera = min(preview_cameras, key=lambda c: c.image_name)
    preview_diag = render_with_pixel_representative(preview_camera, model)
    preview_rep = preview_diag["representative_id"].to(torch.int64)
    preview_h, preview_w = preview_rep.shape

    anchor_visible_ids: dict[str, list[int]] = {}
    for label, fractions in ANCHOR_FRACTIONS.items():
        ids = []
        for fx, fy in fractions:
            px = min(preview_w - 1, int(fx * preview_w))
            py = min(preview_h - 1, int(fy * preview_h))
            full_id = int(preview_rep[py, px].item())
            if full_id >= 0:
                ids.append(int(full_to_visible[full_id].item()))
        anchor_visible_ids[label] = ids

    region_masks: dict[str, torch.Tensor] = {}
    for label in ANCHOR_FRACTIONS:
        ids = anchor_visible_ids.get(label, [])
        mask = torch.zeros((count,), dtype=torch.bool, device=device)
        if ids:
            anchor_positions = positions[torch.tensor(ids, device=device)]
            dist_to_this = torch.cdist(positions, anchor_positions).min(dim=1).values
            other_ids = sum((anchor_visible_ids.get(l, []) for l in ANCHOR_FRACTIONS if l != label), [])
            if other_ids:
                dist_to_other = torch.cdist(positions, positions[torch.tensor(other_ids, device=device)]).min(dim=1).values
                mask = dist_to_this < dist_to_other
            else:
                mask = torch.ones((count,), dtype=torch.bool, device=device)
        region_masks[label] = mask

    region_results = {}
    for label, mask in region_masks.items():
        rep_pop = ever_representative & mask
        covered_pop = ever_in_valid_chart_visible & mask
        region_results[label] = {
            "representative_count": int(rep_pop.sum().item()),
            "covered_by_valid_chart_count": int(covered_pop.sum().item()),
            "coverage_fraction": (float(covered_pop.sum().item()) / float(rep_pop.sum().item())) if int(rep_pop.sum().item()) > 0 else 0.0,
            "no_valid_chart_count": int((rep_pop & no_valid_chart).sum().item()),
            "multi_valid_chart_count": int((rep_pop & multi_valid_chart).sum().item()),
        }
    _progress(f"[region results] {region_results}")

    accounting = {
        "total_trained_surfels": total_model_count,
        "visible_domain_surfels": visible_count,
        "median_surface_representatives": representative_count,
        "raw_chart_count": raw_chart_count,
        "valid_chart_count": valid_chart_count,
        "fitted_chart_count": global_chart_id,
        "min_pixel_samples_required": MIN_PIXEL_SAMPLES,
        "renderer_surface_pixel_coverage_fraction": (float(covered_valid_pixels_all_views) / float(total_valid_pixels_all_views)) if total_valid_pixels_all_views else 0.0,
        "representatives_in_ge1_valid_chart": int(ever_in_valid_chart_visible.sum().item()),
        "representatives_no_valid_chart": int(no_valid_chart.sum().item()),
        "representatives_single_valid_chart": int(single_valid_chart.sum().item()),
        "representatives_multi_valid_chart": int(multi_valid_chart.sum().item()),
        "representative_membership_coverage_fraction": (float(ever_in_valid_chart_visible.sum().item()) / float(representative_count)) if representative_count else 0.0,
    }
    _progress(f"[accounting] {accounting}")

    wl111_comparison: dict[str, Any] = {}
    if wl111_reference is not None:
        wl111_comparison = {
            "wl111_accounting": wl111_reference.get("accounting", {}),
            "wl111_fitting_residual_distribution": wl111_reference.get("fitting_residual_distribution", {}),
            "wl111_overlap_position_discrepancy_distribution": wl111_reference.get("overlap_position_discrepancy_distribution", {}),
            "wl111_overlap_normal_discrepancy_degrees_distribution": wl111_reference.get("overlap_normal_discrepancy_degrees_distribution", {}),
            "wl111_per_component_chart_coverage_fraction_distribution": wl111_reference.get("per_component_chart_coverage_fraction_distribution", {}),
            "wl111_region_results": wl111_reference.get("region_results_WORKING_INTERPRETATION_ONLY", {}),
        }

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 112",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "camera_meta": camera_meta,
        "nurbs_config": {"resolution_u": RESOLUTION_U, "resolution_v": RESOLUTION_V, "degree_u": DEGREE_U, "degree_v": DEGREE_V, "min_pixel_samples": MIN_PIXEL_SAMPLES},
        "accounting": accounting,
        "wl107_109_replay_consistency_check": wl107_replay_stats,
        "per_component_chart_coverage_fraction_distribution_LABEL": "unweighted count of components at 100% vs 0% coverage -- NOT a scene surface-area metric; tiny and large components count equally here",
        "per_component_chart_coverage_fraction_distribution": per_component_coverage_distribution,
        "fitting_residual_distribution": residual_distribution,
        "overlap_position_discrepancy_distribution": position_discrepancy_distribution,
        "overlap_normal_discrepancy_degrees_distribution": normal_discrepancy_distribution,
        "region_results_WORKING_INTERPRETATION_ONLY": region_results,
        "wl111_comparison": wl111_comparison,
        "runtime_seconds": {"total": time.time() - started},
    }

    # --- colors / exports ---
    original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]

    backbone_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    backbone_colors[ever_representative] = _hash_colors(subset_ids[ever_representative])

    first_chart_of_member = torch.full((count,), -1, dtype=torch.int64, device=device)
    first_view_of_member = torch.full((count,), -1, dtype=torch.int64, device=device)
    if all_member_ids:
        cat_member = torch.cat(all_member_ids).to(device)
        cat_chart = torch.cat(all_chart_ids).to(device)
        cat_view = torch.cat(all_view_ids).to(device)
        order2 = torch.argsort(cat_member * (global_chart_id + 1) + cat_chart)
        sm, sc, sv = cat_member[order2], cat_chart[order2], cat_view[order2]
        first_mask = torch.ones_like(sm, dtype=torch.bool)
        first_mask[1:] = sm[1:] != sm[:-1]
        first_chart_of_member[sm[first_mask]] = sc[first_mask]
        first_view_of_member[sm[first_mask]] = sv[first_mask]

    domain_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    has_chart = first_chart_of_member >= 0
    domain_colors[has_chart] = _hash_colors(first_view_of_member[has_chart])

    coverage_colors = _ramp(ever_in_valid_chart.to(torch.float32), _UNCUT_RGB, (0.2, 0.95, 0.3))
    nurbs_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    nurbs_colors[has_chart] = _hash_colors(first_chart_of_member[has_chart])

    # residual-per-representative: mean pixel-fit residual over every pixel that
    # representative contributed across all its (chart, view) occurrences.
    mean_residual = torch.zeros((count,), dtype=torch.float32, device=device)
    if residual_pixel_rep_ids:
        cat_res_rep = torch.cat(residual_pixel_rep_ids).to(device)
        cat_res_val = torch.cat(residual_values).to(device)
        sum_res = torch.zeros((count,), dtype=torch.float32, device=device)
        cnt_res = torch.zeros((count,), dtype=torch.float32, device=device)
        sum_res.index_add_(0, cat_res_rep, cat_res_val)
        cnt_res.index_add_(0, cat_res_rep, torch.ones_like(cat_res_val))
        mean_residual = sum_res / cnt_res.clamp_min(1.0)
    residual_norm = mean_residual / mean_residual.clamp_min(1e-6).max() if bool((mean_residual > 0).any()) else mean_residual
    residual_colors = _ramp(residual_norm.clamp(0.0, 1.0), _UNCUT_RGB, (1.0, 0.85, 0.0))

    # overlap disagreement per representative: max consecutive-pair position
    # discrepancy across the charts it belongs to (same technique as WL111).
    overlap_ratio = torch.zeros((count,), dtype=torch.float32, device=device)
    if all_member_ids and position_discrepancies:
        cat_member_o = torch.cat(all_member_ids).to(device)
        cat_points_o = torch.cat(all_fitted_points).to(device)
        order_o = torch.argsort(cat_member_o)
        sm_o = cat_member_o[order_o]
        pts_o = cat_points_o[order_o]
        same_next_o = torch.zeros_like(sm_o, dtype=torch.bool)
        same_next_o[:-1] = sm_o[:-1] == sm_o[1:]
        diffs_o = torch.zeros((int(sm_o.shape[0]),), dtype=torch.float32, device=device)
        diffs_o[:-1] = torch.where(same_next_o[:-1], (pts_o[1:] - pts_o[:-1]).norm(dim=-1), torch.zeros((int(sm_o.shape[0]) - 1,), device=device))
        member_max_o = torch.zeros((count,), dtype=torch.float32, device=device)
        member_max_o.scatter_reduce_(0, sm_o, diffs_o, reduce="amax", include_self=True)
        overlap_ratio = member_max_o
    overlap_norm = overlap_ratio / overlap_ratio.clamp_min(1e-6).max() if bool((overlap_ratio > 0).any()) else overlap_ratio
    overlap_colors = _ramp(overlap_norm.clamp(0.0, 1.0), _UNCUT_RGB, (1.0, 0.2, 0.85))

    uncovered_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    uncovered_colors[no_valid_chart] = torch.tensor((1.0, 0.1, 0.05), dtype=torch.float32, device=device)
    uncovered_colors[ever_in_valid_chart_visible] = torch.tensor((0.1, 0.3, 0.15), dtype=torch.float32, device=device)

    def _region_view_colors(region_label: str) -> torch.Tensor:
        mask = region_masks.get(region_label, torch.zeros((count,), dtype=torch.bool, device=device)) & ever_representative
        colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        colors[mask & ever_in_valid_chart] = torch.tensor((0.2, 0.95, 0.3), dtype=torch.float32, device=device)
        colors[mask & no_valid_chart] = torch.tensor((1.0, 0.1, 0.05), dtype=torch.float32, device=device)
        return colors

    table_colors = _region_view_colors("table_top")
    curved_colors = _region_view_colors("table_side_curved")
    hedge_colors = _region_view_colors("hedge")

    views = {
        VIEW_ORIGINAL_SCENE: original_f_dc,
        VIEW_PIXEL_SURFACE: _rgb_to_f_dc(backbone_colors),
        VIEW_CHART_DOMAINS: _rgb_to_f_dc(domain_colors),
        VIEW_NURBS: _rgb_to_f_dc(nurbs_colors),
        VIEW_COVERAGE: _rgb_to_f_dc(coverage_colors),
        VIEW_RESIDUAL_COMPARE: _rgb_to_f_dc(residual_colors),
        VIEW_OVERLAP: _rgb_to_f_dc(overlap_colors),
        VIEW_TABLE: _rgb_to_f_dc(table_colors),
        VIEW_CURVED: _rgb_to_f_dc(curved_colors),
        VIEW_HEDGE: _rgb_to_f_dc(hedge_colors),
    }

    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())

    view_paths: dict[str, Any] = {}
    for view_name, f_dc in views.items():
        view_dir = output_root / view_name
        ply_path = view_dir / _ITERATION_DIR / "point_cloud.ply"
        n = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
        view_paths[view_name] = {"point_cloud_ply": str(ply_path), "gaussian_count": n}
        _progress(f"wrote {view_name} ({n} surfels)")

    # copy the WL111 baseline export directly for side-by-side comparison (never re-run WL111)
    if arguments.wl111_export_dir.exists():
        import shutil
        src_dir = arguments.wl111_export_dir / "VISIBLE_NURBS_PATCHES"
        dst_dir = output_root / VIEW_WL111_BASELINE
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            view_paths[VIEW_WL111_BASELINE] = {"copied_from": str(src_dir)}
            _progress(f"copied WL111 baseline export from {src_dir}")

    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    render_report: dict[str, Any] = {"enabled": True}
    try:
        _progress(f"rendering previews from camera {preview_camera.image_name}")
        with torch.no_grad():
            for view_name, f_dc in views.items():
                full_dc = torch.zeros_like(model._features_dc)
                full_dc[visible_selector, 0, :] = f_dc
                model._features_dc.data.copy_(full_dc)
                model._features_rest.data.zero_()
                model.active_sh_degree = 0
                del full_dc
                package = rasterizer.render(preview_camera, model)
                write_ppm(output_root / view_name / "render.ppm", package["render"])
                view_paths.setdefault(view_name, {})["render_ppm"] = str(output_root / view_name / "render.ppm")
                _progress(f"rendered {view_name}")
                del package
        render_report.update({"camera": preview_camera.image_name})
    except Exception as error:
        render_report.update({"failed": True, "reason": f"{type(error).__name__}: {error}"})

    report["views"] = view_paths
    report["render_ppm"] = render_report
    report_path = output_root / "renderer_native_pixel_surface_nurbs_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
