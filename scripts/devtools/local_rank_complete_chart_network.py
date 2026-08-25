"""Worklog 114 -- Local Rank-Complete NURBS Chart Network.

Worklog 113 (accepted as a successful diagnostic) traced the B/C failure
signatures to the chart UNIT itself: "one camera-connected blob == one fixed
8x4 NURBS chart" operates at the wrong scale for large blobs. This batch
changes ONLY that unit -- everything else (WL107/109 canonical topology,
WL112 renderer-native per-pixel surface geometry, per-view same-component
blob connectivity, the fixed 8x4/degree-2 NURBS config, visible/occluded
separation) is preserved exactly, reusing the same frozen functions.

PRIMARY CONTROL EXPERIMENT (directive section 8): for an identical bounded
subset of views, compare
    A. WL112: one camera blob -> one fixed 8x4 NURBS (recomputed here on the
       SAME view subset for a fair A/B, NOT WL112's saved full-161-view
       report -- different sample sizes must never be compared directly).
    B. NEW:   same camera blob -> local rank-complete domains
       (`grow_local_rank_complete_charts`) -> multiple fixed 8x4 NURBS
       patches.

SCOPE REDUCTION (disclosed, not hidden): the topology/representative sweep
and canonical-topology replay always use ALL 161 training views (identical
to WL107-113, fully reproducible). The expensive NEW per-view chart-growth
and per-chart NURBS-fitting stage -- for BOTH method A and method B, so the
comparison stays fair -- runs on a bounded, deterministic, strided subset of
views (`--chart-view-stride`, `--chart-max-views`), because the module-level
synthetic test already shows a single giant blob can require ~10-1000x more
local chart fits than one global fit; a full 161-view sweep at that combinatorial
rate was not tractable in this session. This is stated explicitly in every
report section that could otherwise be misread as full-scene coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

import numpy as np
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
from chart_representation_contract_diagnostic import (  # noqa: E402 -- WL113, frozen, reused read-only
    blob_domain_shape,
    design_matrix_rank_diagnostics,
    _bin_by_quantile,
    _distribution,
)
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
from osn_gs.surface.torch_local_rank_complete_chart_growth import (
    REASON_INSUFFICIENT_RANK_CLOSURE,
    REASON_RUNTIME_CAP_SKIPPED,
    REASON_TOO_FEW_PIXELS,
    grow_local_rank_complete_charts,
)
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"
_MIDDEPTH_OFFSET = 5
_UNCUT_RGB = (0.08, 0.09, 0.11)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

# Frozen exactly as WL111/112/113 (directive section 2: do not tune).
RESOLUTION_U = 8
RESOLUTION_V = 4
DEGREE_U = 2
DEGREE_V = 2
MIN_PIXEL_SAMPLES = RESOLUTION_U * RESOLUTION_V  # 32, WL112's blob-level threshold, reused for the baseline arm only

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_WL112_BASELINE_SUBSET = "WL112_BASELINE_SAME_SUBSET"
VIEW_LOCAL_CHART_NETWORK = "LOCAL_CHART_NETWORK"
VIEW_UNRESOLVED_EVIDENCE = "UNRESOLVED_VISIBLE_EVIDENCE"
VIEW_DOMAIN_OCCUPANCY = "LOCAL_CHART_DOMAIN_OCCUPANCY"
VIEW_RESIDUAL_COMPARE = "LOCAL_VS_WL112_RESIDUAL"
VIEW_OVERLAP = "LOCAL_OVERLAP_DISAGREEMENT"
VIEW_D_OUTLIERS = "D_OUTLIER_PERSISTENCE"
VIEW_TABLE = "TABLE_LOCAL_CHART_NETWORK"
VIEW_CURVED = "CURVED_LOCAL_CHART_NETWORK"
VIEW_HEDGE = "HEDGE_LOCAL_CHART_NETWORK"

ANCHOR_FRACTIONS = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[local-chart-network] {message}", flush=True)


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


def _overlap_stats(all_member_ids, all_chart_ids, all_fitted_points, all_normals):
    if not all_member_ids:
        empty = _distribution(np.zeros((0,)))
        return empty, empty, {}
    cat_member = torch.cat(all_member_ids)
    cat_points = torch.cat(all_fitted_points)
    cat_normals = torch.cat(all_normals)
    cat_chart = torch.cat(all_chart_ids)
    order = torch.argsort(cat_member)
    sm, sp, sn, sc = cat_member[order], cat_points[order], cat_normals[order], cat_chart[order]
    same_next = sm[:-1] == sm[1:]
    worst_by_chart: dict[int, float] = {}
    if bool(same_next.any()):
        diff = (sp[1:] - sp[:-1])[same_next]
        pos = diff.norm(dim=-1)
        cos = (sn[1:] * sn[:-1]).sum(dim=-1)[same_next].clamp(-1.0, 1.0)
        normal_deg = torch.rad2deg(torch.acos(cos))
        chart_a = sc[:-1][same_next].tolist()
        chart_b = sc[1:][same_next].tolist()
        for ca, cb, d in zip(chart_a, chart_b, pos.tolist()):
            worst_by_chart[ca] = max(worst_by_chart.get(ca, 0.0), d)
            worst_by_chart[cb] = max(worst_by_chart.get(cb, 0.0), d)
        return _distribution(pos.numpy()), _distribution(normal_deg.numpy()), worst_by_chart
    return _distribution(np.zeros((0,))), _distribution(np.zeros((0,))), worst_by_chart


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
    parser.add_argument("--max-views", type=int, default=0, help="truncate the FULL topology/representative sweep (smoke testing only)")
    parser.add_argument("--chart-view-stride", type=int, default=1, help="process every Nth view for the expensive chart-growth+fit stage (both arms)")
    parser.add_argument("--chart-max-views", type=int, default=0, help="cap the number of (strided) views processed for the chart-growth+fit stage, 0=all strided views")
    parser.add_argument("--max-patches-per-blob", type=int, default=2000, help="runtime safety valve for grow_local_rank_complete_charts, NOT an architecture parameter")
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)

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

    # --- FULL-SCENE sweep: representative maps + renderer-native median-depth (identical to WL112/113) ---
    _progress("[full-scene sweep] representative_id + median_depth unprojection per view")
    per_view_rep_remapped: list[torch.Tensor] = []
    per_view_world_points: list[torch.Tensor] = []
    per_view_depth: list[torch.Tensor] = []
    per_view_camera_center: list[torch.Tensor] = []
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
            c2w = (camera.world_view_transform.T).inverse()
            camera_center = c2w[:3, 3].detach()

        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_rep_remapped.append(rep_remapped.detach().cpu())
        per_view_world_points.append(world_points.detach().cpu())
        per_view_depth.append(depth_map.detach().cpu())
        per_view_camera_center.append(camera_center.cpu())
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

    # --- region masks (WORKING INTERPRETATION ONLY, same anchor mechanism as WL108-113) ---
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
    region_of_representative = torch.full((count,), -1, dtype=torch.int64, device=device)
    region_labels_list = list(ANCHOR_FRACTIONS.keys())
    for region_index, label in enumerate(region_labels_list):
        region_of_representative[region_masks[label]] = region_index

    # --- select the bounded view subset for the expensive chart-growth+fit stage (BOTH arms) ---
    stride = max(1, int(arguments.chart_view_stride))
    chart_view_indices = list(range(0, len(per_view_rep_remapped), stride))
    if int(arguments.chart_max_views) > 0:
        chart_view_indices = chart_view_indices[: int(arguments.chart_max_views)]
    _progress(f"[scope] processing {len(chart_view_indices)}/{len(per_view_rep_remapped)} views for chart-growth+fit (stride={stride}): {chart_view_indices}")

    # =================================================================
    # ARM A: WL112 baseline (one blob -> one fixed 8x4 NURBS), recomputed
    # on the SAME view subset as ARM B, for a fair controlled comparison.
    # =================================================================
    baseline_ever_covered = torch.zeros((count,), dtype=torch.bool, device=device)
    baseline_chart_count = 0
    baseline_residuals: list[np.ndarray] = []
    baseline_domain_shapes: list[dict[str, Any]] = []
    baseline_member_ids: list[torch.Tensor] = []
    baseline_chart_ids: list[torch.Tensor] = []
    baseline_fitted_points: list[torch.Tensor] = []
    baseline_normals: list[torch.Tensor] = []
    baseline_global_chart_id = 0

    from osn_gs.surface.torch_camera_observed_chart_domains import label_same_component_blobs

    for view_index in chart_view_indices:
        rep_gpu = per_view_rep_remapped[view_index].to(device)
        world_gpu = per_view_world_points[view_index].to(device)
        valid_pix = rep_gpu >= 0
        comp_map = torch.where(valid_pix, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1))
        vs = build_view_chart_pixel_samples(view_index, comp_map, rep_gpu, world_gpu)
        if vs.blob_count == 0:
            continue
        mask = valid_pixel_chart_mask(vs, MIN_PIXEL_SAMPLES)
        blob_labels_np = label_same_component_blobs(comp_map).detach().cpu().numpy()
        valid_blob_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        for local_blob in valid_blob_ids:
            pixel_sel = vs.pixel_blob_id == local_blob
            pixel_uv = vs.pixel_uv[pixel_sel]
            pixel_xyz = vs.pixel_xyz[pixel_sel]
            pixel_rep_ids = vs.pixel_representative_id[pixel_sel]
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
            baseline_ever_covered[distinct_reps] = True
            rep_count = int(distinct_reps.shape[0])
            sum_point = torch.zeros((rep_count, 3), device=device)
            sum_normal = torch.zeros((rep_count, 3), device=device)
            member_counts = torch.zeros((rep_count,), device=device)
            sum_point.index_add_(0, inverse_rep, fitted)
            sum_normal.index_add_(0, inverse_rep, normals)
            member_counts.index_add_(0, inverse_rep, torch.ones_like(inverse_rep, dtype=torch.float32))
            mean_point = sum_point / member_counts.clamp_min(1.0)[:, None]
            mean_normal = torch.nn.functional.normalize(sum_normal, dim=-1, eps=1e-8)
            baseline_member_ids.append(distinct_reps.detach().cpu())
            baseline_chart_ids.append(torch.full((rep_count,), baseline_global_chart_id, dtype=torch.int64))
            baseline_fitted_points.append(mean_point.detach().cpu())
            baseline_normals.append(mean_normal.detach().cpu())
            baseline_residuals.append(residual.detach().cpu().numpy())
            blob_mask_np = blob_labels_np == local_blob
            baseline_domain_shapes.append(blob_domain_shape(blob_mask_np))
            baseline_global_chart_id += 1
            baseline_chart_count += 1
    baseline_position_dist, baseline_normal_dist, _baseline_worst = _overlap_stats(
        baseline_member_ids, baseline_chart_ids, baseline_fitted_points, baseline_normals
    )
    baseline_residual_all = np.concatenate(baseline_residuals) if baseline_residuals else np.zeros((0,))
    _progress(f"[ARM A / WL112 baseline, same subset] charts={baseline_chart_count} covered_reps={int(baseline_ever_covered.sum().item())}")

    # =================================================================
    # ARM B: NEW local rank-complete chart network.
    # =================================================================
    local_ever_covered = torch.zeros((count,), dtype=torch.bool, device=device)
    local_chart_records: list[dict[str, Any]] = []
    local_member_ids: list[torch.Tensor] = []
    local_chart_ids: list[torch.Tensor] = []
    local_fitted_points: list[torch.Tensor] = []
    local_normals: list[torch.Tensor] = []
    local_residuals: list[np.ndarray] = []
    unresolved_records: list[dict[str, Any]] = []
    unresolved_representative_ids = torch.zeros((count,), dtype=torch.bool, device=device)
    local_global_chart_id = 0

    for progress_i, view_index in enumerate(chart_view_indices):
        rep_gpu = per_view_rep_remapped[view_index].to(device)
        world_gpu = per_view_world_points[view_index].to(device)
        depth_cpu = per_view_depth[view_index]
        camera_center_cpu = per_view_camera_center[view_index]
        valid_pix = rep_gpu >= 0
        comp_map = torch.where(valid_pix, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1))

        charts, unresolved = grow_local_rank_complete_charts(
            view_index, comp_map, rep_gpu, world_gpu,
            RESOLUTION_U, RESOLUTION_V, DEGREE_U, DEGREE_V,
            max_patches_per_blob=int(arguments.max_patches_per_blob),
        )

        for region in unresolved:
            rep_ids_here = torch.as_tensor(rep_gpu.detach().cpu().numpy()[region.pixel_rows, region.pixel_cols], dtype=torch.int64)
            valid_reps = rep_ids_here[rep_ids_here >= 0]
            if int(valid_reps.numel()) > 0:
                unresolved_representative_ids[valid_reps.to(device)] = True
            unresolved_records.append({
                "view_index": view_index, "component_id": region.component_id,
                "pixel_count": int(region.pixel_rows.shape[0]), "reason": region.reason,
            })

        for chart in charts:
            with torch.no_grad():
                surface, uv = fit_torch_visible_surface_lsq(
                    chart.xyz, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                    degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=chart.uv,
                    correction_rounds=2, projection_iterations=3,
                )
                fitted = surface.evaluate(uv)
                normals = surface.normals(uv)
                residual = (fitted - chart.xyz).norm(dim=-1)

            rep_ids_np = chart.representative_ids
            valid_mask_np = rep_ids_np >= 0
            rep_ids_t = torch.as_tensor(rep_ids_np[valid_mask_np], dtype=torch.int64, device=device)
            if int(rep_ids_t.numel()) == 0:
                continue
            local_ever_covered[rep_ids_t] = True
            distinct_reps, inverse_rep = torch.unique(rep_ids_t, return_inverse=True)
            fitted_valid = fitted[torch.as_tensor(valid_mask_np, device=device)]
            normals_valid = normals[torch.as_tensor(valid_mask_np, device=device)]
            rep_count = int(distinct_reps.shape[0])
            sum_point = torch.zeros((rep_count, 3), device=device)
            sum_normal = torch.zeros((rep_count, 3), device=device)
            member_counts = torch.zeros((rep_count,), device=device)
            sum_point.index_add_(0, inverse_rep, fitted_valid)
            sum_normal.index_add_(0, inverse_rep, normals_valid)
            member_counts.index_add_(0, inverse_rep, torch.ones_like(inverse_rep, dtype=torch.float32))
            mean_point = sum_point / member_counts.clamp_min(1.0)[:, None]
            mean_normal = torch.nn.functional.normalize(sum_normal, dim=-1, eps=1e-8)
            local_member_ids.append(distinct_reps.detach().cpu())
            local_chart_ids.append(torch.full((rep_count,), local_global_chart_id, dtype=torch.int64))
            local_fitted_points.append(mean_point.detach().cpu())
            local_normals.append(mean_normal.detach().cpu())
            local_residuals.append(residual.detach().cpu().numpy())

            h, w = comp_map.shape
            blob_mask_np = np.zeros((h, w), dtype=bool)
            blob_mask_np[chart.pixel_rows, chart.pixel_cols] = True
            shape_diag = blob_domain_shape(blob_mask_np)
            capacity_diag = design_matrix_rank_diagnostics(surface, uv, seed=local_global_chart_id)

            rows_t = torch.from_numpy(chart.pixel_rows).to(torch.int64)
            cols_t = torch.from_numpy(chart.pixel_cols).to(torch.int64)
            chart_depths = depth_cpu[rows_t, cols_t]
            view_dirs = chart.xyz.detach().cpu() - camera_center_cpu.reshape(1, 3)
            view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)
            incidence_cos = (view_dirs * normals.detach().cpu()).sum(dim=-1).abs()

            region_votes = region_of_representative[distinct_reps]
            valid_votes = region_votes[region_votes >= 0]
            region_index = int(torch.mode(valid_votes).values.item()) if int(valid_votes.numel()) > 0 else -1

            residual_np = residual.detach().cpu().numpy()
            local_chart_records.append({
                "chart_id": local_global_chart_id, "view_index": view_index,
                "camera_name": cameras[view_index].image_name, "component_id": chart.component_id,
                "pixel_count": int(chart.pixel_rows.shape[0]),
                "representative_count": rep_count,
                **{k: shape_diag[k] for k in ("bbox_h", "bbox_w", "bbox_area", "occupancy_ratio", "aspect_ratio", "hole_count", "hole_area")},
                "rank": capacity_diag["rank"], "full_capacity": capacity_diag["full_capacity"], "cond_number": capacity_diag["cond_number"],
                "residual_median": float(np.median(residual_np)) if residual_np.shape[0] else 0.0,
                "residual_p95": float(np.percentile(residual_np, 95)) if residual_np.shape[0] else 0.0,
                "residual_max": float(residual_np.max()) if residual_np.shape[0] else 0.0,
                "depth_std": float(chart_depths.std().item()) if chart_depths.shape[0] > 1 else 0.0,
                "grazing_incidence_min": float(incidence_cos.min().item()) if incidence_cos.shape[0] else float("nan"),
                "region_label": region_labels_list[region_index] if region_index >= 0 else None,
            })
            local_global_chart_id += 1

        if progress_i % 2 == 0:
            _progress(f"chart-growth view {view_index} ({progress_i + 1}/{len(chart_view_indices)}) charts_so_far={local_global_chart_id} unresolved_so_far={len(unresolved_records)}")

    local_position_dist, local_normal_dist, local_worst_overlap = _overlap_stats(
        local_member_ids, local_chart_ids, local_fitted_points, local_normals
    )
    for record in local_chart_records:
        record["max_overlap_position_discrepancy"] = local_worst_overlap.get(record["chart_id"], 0.0)
    local_residual_all = np.concatenate(local_residuals) if local_residuals else np.zeros((0,))
    _progress(f"[ARM B / local chart network] charts={local_global_chart_id} covered_reps={int(local_ever_covered.sum().item())} unresolved_regions={len(unresolved_records)}")

    # --- unresolved-visible-evidence coverage (directive section 7) ---
    representatives_in_processed_views = torch.zeros((count,), dtype=torch.bool, device=device)
    for view_index in chart_view_indices:
        rep_cpu = per_view_rep_remapped[view_index]
        valid = rep_cpu >= 0
        ids = torch.unique(rep_cpu[valid])
        if int(ids.numel()) > 0:
            representatives_in_processed_views[ids.to(device)] = True
    never_processed = ever_representative & ~representatives_in_processed_views
    unresolved_only = ever_representative & representatives_in_processed_views & unresolved_representative_ids & ~local_ever_covered
    accounting = {
        "median_surface_representatives": representative_count,
        "representatives_in_processed_view_subset": int(representatives_in_processed_views.sum().item()),
        "representatives_never_in_processed_subset": int(never_processed.sum().item()),
        "representatives_note": "the last two rows are a SCOPE artifact of the bounded view subset, not an architecture property -- see module docstring",
        "arm_a_wl112_baseline_same_subset_chart_count": baseline_chart_count,
        "arm_a_wl112_baseline_same_subset_covered_representatives": int(baseline_ever_covered.sum().item()),
        "arm_b_local_chart_count": local_global_chart_id,
        "arm_b_local_covered_representatives": int(local_ever_covered.sum().item()),
        "arm_b_unresolved_region_count": len(unresolved_records),
        "arm_b_representatives_touched_by_unresolved_only": int(unresolved_only.sum().item()),
    }
    _progress(f"[accounting] {accounting}")

    # --- patch-count / complexity audit (directive section 12) ---
    patches_per_view: dict[int, int] = {}
    patches_per_component: dict[int, int] = {}
    for r in local_chart_records:
        patches_per_view[r["view_index"]] = patches_per_view.get(r["view_index"], 0) + 1
        patches_per_component[r["component_id"]] = patches_per_component.get(r["component_id"], 0) + 1
    covered_pixel_total = sum(r["pixel_count"] for r in local_chart_records)
    complexity_audit = {
        "total_local_patches": local_global_chart_id,
        "wl112_baseline_same_subset_chart_count": baseline_chart_count,
        "patches_per_view_distribution": _distribution(np.array(list(patches_per_view.values()), dtype=np.float64)),
        "patches_per_component_distribution": _distribution(np.array(list(patches_per_component.values()), dtype=np.float64)),
        "patches_per_10000_covered_pixels": (float(local_global_chart_id) / (covered_pixel_total / 10000.0)) if covered_pixel_total else 0.0,
        "sample_count_per_patch_distribution": _distribution(np.array([r["pixel_count"] for r in local_chart_records], dtype=np.float64)),
        "representative_count_per_patch_distribution": _distribution(np.array([r["representative_count"] for r in local_chart_records], dtype=np.float64)),
        "unresolved_reason_counts": {
            REASON_TOO_FEW_PIXELS: sum(1 for u in unresolved_records if u["reason"] == REASON_TOO_FEW_PIXELS),
            REASON_INSUFFICIENT_RANK_CLOSURE: sum(1 for u in unresolved_records if u["reason"] == REASON_INSUFFICIENT_RANK_CLOSURE),
            REASON_RUNTIME_CAP_SKIPPED: sum(1 for u in unresolved_records if u["reason"] == REASON_RUNTIME_CAP_SKIPPED),
        },
    }
    _progress(f"[complexity audit] {complexity_audit}")

    # --- domain-shape comparison (directive section 9) ---
    local_occupancy = np.array([r["occupancy_ratio"] for r in local_chart_records], dtype=np.float64)
    local_holes = np.array([r["hole_count"] for r in local_chart_records], dtype=np.float64)
    local_aspect = np.array([r["aspect_ratio"] for r in local_chart_records], dtype=np.float64)
    baseline_occupancy = np.array([d["occupancy_ratio"] for d in baseline_domain_shapes], dtype=np.float64)
    baseline_holes = np.array([d["hole_count"] for d in baseline_domain_shapes], dtype=np.float64)
    baseline_aspect = np.array([d["aspect_ratio"] for d in baseline_domain_shapes], dtype=np.float64)
    domain_shape_comparison = {
        "wl112_baseline_same_subset": {
            "occupancy_ratio": _distribution(baseline_occupancy), "hole_count": _distribution(baseline_holes),
            "aspect_ratio": _distribution(baseline_aspect),
            "fraction_with_ge1_hole": float((baseline_holes > 0).mean()) if baseline_holes.shape[0] else 0.0,
        },
        "local_chart_network": {
            "occupancy_ratio": _distribution(local_occupancy), "hole_count": _distribution(local_holes),
            "aspect_ratio": _distribution(local_aspect),
            "fraction_with_ge1_hole": float((local_holes > 0).mean()) if local_holes.shape[0] else 0.0,
        },
    }
    _progress(f"[domain shape comparison] {domain_shape_comparison}")

    # --- residual / overlap comparison (directive sections 10/11) ---
    fit_quality_comparison = {
        "wl112_baseline_same_subset_residual": _distribution(baseline_residual_all),
        "local_chart_network_residual": _distribution(local_residual_all),
        "wl112_baseline_same_subset_overlap_position": baseline_position_dist,
        "local_chart_network_overlap_position": local_position_dist,
        "wl112_baseline_same_subset_overlap_normal_degrees": baseline_normal_dist,
        "local_chart_network_overlap_normal_degrees": local_normal_dist,
    }
    _progress(f"[fit quality comparison] {fit_quality_comparison}")

    # --- D-outlier persistence (directive section 11 -- track only, never reject/clamp) ---
    top_overlap = sorted(local_chart_records, key=lambda r: r["max_overlap_position_discrepancy"], reverse=True)[:15]
    top_residual = sorted(local_chart_records, key=lambda r: r["residual_max"], reverse=True)[:15]
    d_outlier_persistence = {
        "top_15_by_overlap_position_discrepancy": [
            {k: r[k] for k in ("chart_id", "view_index", "component_id", "pixel_count", "depth_std", "grazing_incidence_min", "max_overlap_position_discrepancy", "region_label")}
            for r in top_overlap
        ],
        "top_15_by_residual_max": [
            {k: r[k] for k in ("chart_id", "view_index", "component_id", "pixel_count", "hole_count", "rank", "depth_std", "residual_max", "region_label")}
            for r in top_residual
        ],
    }
    _progress(f"[D-outlier persistence] worst overlap chart: {d_outlier_persistence['top_15_by_overlap_position_discrepancy'][:1]}")

    # --- region breakdown (directive section 13) ---
    region_results = {}
    for label, mask in region_masks.items():
        rep_pop = ever_representative & mask & representatives_in_processed_views
        local_covered = local_ever_covered & mask
        baseline_covered = baseline_ever_covered & mask
        region_chart_records = [r for r in local_chart_records if r["region_label"] == label]
        region_results[label] = {
            "visible_representative_evidence_in_processed_subset": int(rep_pop.sum().item()),
            "arm_a_wl112_baseline_covered": int(baseline_covered.sum().item()),
            "arm_b_local_covered": int(local_covered.sum().item()),
            "arm_b_local_patch_count": len(region_chart_records),
            "arm_b_residual_median": _distribution(np.array([r["residual_median"] for r in region_chart_records], dtype=np.float64)),
            "arm_b_occupancy_ratio_median": float(np.median([r["occupancy_ratio"] for r in region_chart_records])) if region_chart_records else 0.0,
        }
    _progress(f"[region results] {json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in region_results.items()}, indent=2)}")

    # --- architecture judgment (directive section 16, evidence-derived) ---
    # Every factor directive section 16 actually lists is checked here --
    # NOT just residual/domain-shape. A method that fits better locally but
    # covers fewer representatives or degrades cross-chart normal agreement
    # is not automatically "viable" (directive: "Do NOT claim architecture
    # success... simply because fitting residual becomes small").
    residual_improved = (fit_quality_comparison["local_chart_network_residual"]["max"] < fit_quality_comparison["wl112_baseline_same_subset_residual"]["max"]) and \
        (fit_quality_comparison["local_chart_network_residual"]["p95"] <= fit_quality_comparison["wl112_baseline_same_subset_residual"]["p95"] * 1.5)
    domain_improved = (domain_shape_comparison["local_chart_network"]["fraction_with_ge1_hole"] < domain_shape_comparison["wl112_baseline_same_subset"]["fraction_with_ge1_hole"])
    patch_explosion = complexity_audit["total_local_patches"] > 20 * max(1, complexity_audit["wl112_baseline_same_subset_chart_count"])
    baseline_covered = accounting["arm_a_wl112_baseline_same_subset_covered_representatives"]
    local_covered = accounting["arm_b_local_covered_representatives"]
    coverage_ratio = (float(local_covered) / float(baseline_covered)) if baseline_covered else 1.0
    coverage_materially_dropped = coverage_ratio < 0.9  # >10% fewer representatives covered than the baseline arm, same subset
    overlap_normal_baseline = fit_quality_comparison["wl112_baseline_same_subset_overlap_normal_degrees"]
    overlap_normal_local = fit_quality_comparison["local_chart_network_overlap_normal_degrees"]
    overlap_normal_materially_worse = (
        overlap_normal_local["median"] > overlap_normal_baseline["median"] * 1.5
        and overlap_normal_local["p95"] > overlap_normal_baseline["p95"] * 1.25
    )
    judgment = {
        "residual_materially_improved": bool(residual_improved),
        "domain_shape_materially_improved": bool(domain_improved),
        "patch_count_explosion_vs_baseline": bool(patch_explosion),
        "patch_count_ratio_local_over_baseline": (float(complexity_audit["total_local_patches"]) / max(1, complexity_audit["wl112_baseline_same_subset_chart_count"])),
        "coverage_ratio_local_over_baseline": coverage_ratio,
        "coverage_materially_dropped": bool(coverage_materially_dropped),
        "overlap_normal_materially_worse": bool(overlap_normal_materially_worse),
    }
    if patch_explosion:
        judgment["verdict"] = "LOCAL_CHART_UNIT_NOT_VIABLE"
        judgment["reason"] = "patch count exploded far beyond the WL112 baseline on the identical view subset -- locality traded residual for extreme fragmentation"
    elif not (residual_improved and domain_improved):
        judgment["verdict"] = "LOCAL_CHART_UNIT_NOT_VIABLE"
        judgment["reason"] = "did not materially improve both fit quality and domain shape on the identical view subset"
    elif coverage_materially_dropped or overlap_normal_materially_worse:
        judgment["verdict"] = "LOCAL_CHART_UNIT_NOT_VIABLE"
        judgment["reason"] = (
            f"fit quality/domain shape improved, but this came at a material cost the directive requires weighing: "
            f"representative coverage ratio {coverage_ratio:.3f} (baseline={baseline_covered}, local={local_covered})"
            f"{' [material drop]' if coverage_materially_dropped else ''}, "
            f"overlap normal-disagreement median {overlap_normal_baseline['median']:.2f}deg->{overlap_normal_local['median']:.2f}deg, "
            f"p95 {overlap_normal_baseline['p95']:.2f}deg->{overlap_normal_local['p95']:.2f}deg"
            f"{' [material worsening]' if overlap_normal_materially_worse else ''} -- "
            "more/smaller representation seams measurably increase cross-chart normal disagreement and strand more "
            "boundary pixels as unresolved, so this is not an unqualified success"
        )
    else:
        judgment["verdict"] = "LOCAL_CHART_UNIT_VIABLE"
        judgment["reason"] = "fit quality and domain shape materially improved without patch-count explosion, coverage drop, or overlap-normal degradation"
    _progress(f"[judgment] {judgment}")

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 114",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "camera_meta": camera_meta,
        "nurbs_config": {"resolution_u": RESOLUTION_U, "resolution_v": RESOLUTION_V, "degree_u": DEGREE_U, "degree_v": DEGREE_V},
        "scope_reduction": {
            "topology_and_representative_sweep": "FULL scene, all loaded training cameras (identical to WL107-113)",
            "chart_growth_and_fit_stage": f"BOUNDED to {len(chart_view_indices)} views (stride={stride}, indices={chart_view_indices}) for BOTH arm A and arm B -- disclosed scope reduction, see module docstring",
        },
        "accounting": accounting,
        "wl107_109_replay_consistency_check": wl107_replay_stats,
        "patch_count_complexity_audit": complexity_audit,
        "domain_shape_comparison_vs_wl112_same_subset": domain_shape_comparison,
        "fit_quality_comparison_vs_wl112_same_subset": fit_quality_comparison,
        "d_outlier_persistence": d_outlier_persistence,
        "region_results_WORKING_INTERPRETATION_ONLY": region_results,
        "architecture_judgment": judgment,
        "runtime_seconds": {"total": time.time() - started},
    }

    # --- colors / exports ---
    original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]

    baseline_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    baseline_colors[baseline_ever_covered] = torch.tensor((0.2, 0.95, 0.3), dtype=torch.float32, device=device)

    local_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    first_chart_of_member = torch.full((count,), -1, dtype=torch.int64, device=device)
    if local_member_ids:
        cat_member = torch.cat(local_member_ids).to(device)
        cat_chart = torch.cat(local_chart_ids).to(device)
        order2 = torch.argsort(cat_member * (local_global_chart_id + 1) + cat_chart)
        sm, sc = cat_member[order2], cat_chart[order2]
        first_mask = torch.ones_like(sm, dtype=torch.bool)
        first_mask[1:] = sm[1:] != sm[:-1]
        first_chart_of_member[sm[first_mask]] = sc[first_mask]
    has_chart = first_chart_of_member >= 0
    local_colors[has_chart] = _hash_colors(first_chart_of_member[has_chart])

    unresolved_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    unresolved_colors[local_ever_covered] = torch.tensor((0.1, 0.3, 0.15), dtype=torch.float32, device=device)
    unresolved_colors[unresolved_only] = torch.tensor((1.0, 0.6, 0.0), dtype=torch.float32, device=device)

    occupancy_by_chart = torch.zeros((local_global_chart_id,), dtype=torch.float32, device=device)
    for r in local_chart_records:
        occupancy_by_chart[r["chart_id"]] = r["occupancy_ratio"]
    occupancy_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    occupancy_colors[has_chart] = _ramp(occupancy_by_chart[first_chart_of_member[has_chart]], (0.9, 0.15, 0.05), (0.2, 0.95, 0.3))

    mean_residual_local = torch.zeros((count,), dtype=torch.float32, device=device)
    if local_member_ids:
        cat_member_r = torch.cat(local_member_ids).to(device)
        cat_chart_r = torch.cat(local_chart_ids)
        residual_by_chart = torch.zeros((local_global_chart_id,), dtype=torch.float32)
        for r in local_chart_records:
            residual_by_chart[r["chart_id"]] = r["residual_median"]
        chart_res_per_member = residual_by_chart[cat_chart_r].to(device)
        sum_res = torch.zeros((count,), dtype=torch.float32, device=device)
        cnt_res = torch.zeros((count,), dtype=torch.float32, device=device)
        sum_res.index_add_(0, cat_member_r, chart_res_per_member)
        cnt_res.index_add_(0, cat_member_r, torch.ones_like(chart_res_per_member))
        mean_residual_local = sum_res / cnt_res.clamp_min(1.0)
    residual_norm = mean_residual_local / mean_residual_local.clamp_min(1e-6).max() if bool((mean_residual_local > 0).any()) else mean_residual_local
    residual_colors = _ramp(residual_norm.clamp(0.0, 1.0), _UNCUT_RGB, (1.0, 0.85, 0.0))

    overlap_ratio = torch.zeros((count,), dtype=torch.float32, device=device)
    if local_member_ids and local_worst_overlap:
        cat_member_o = torch.cat(local_member_ids).to(device)
        cat_chart_o = torch.cat(local_chart_ids)
        worst_by_chart_t = torch.zeros((local_global_chart_id,), dtype=torch.float32)
        for cid, val in local_worst_overlap.items():
            worst_by_chart_t[cid] = val
        chart_worst_per_member = worst_by_chart_t[cat_chart_o].to(device)
        member_max = torch.zeros((count,), dtype=torch.float32, device=device)
        member_max.scatter_reduce_(0, cat_member_o, chart_worst_per_member, reduce="amax", include_self=True)
        overlap_ratio = member_max
    overlap_norm = overlap_ratio / overlap_ratio.clamp_min(1e-6).max() if bool((overlap_ratio > 0).any()) else overlap_ratio
    overlap_colors = _ramp(overlap_norm.clamp(0.0, 1.0), _UNCUT_RGB, (1.0, 0.2, 0.85))

    d_outlier_chart_ids = {r["chart_id"] for r in top_overlap} | {r["chart_id"] for r in top_residual}
    d_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    d_colors[has_chart] = torch.tensor((0.15, 0.2, 0.25), dtype=torch.float32, device=device)
    if d_outlier_chart_ids and local_member_ids:
        cat_member_d = torch.cat(local_member_ids).to(device)
        cat_chart_d = torch.cat(local_chart_ids).to(device)
        outlier_t = torch.tensor(sorted(d_outlier_chart_ids), dtype=torch.int64, device=device)
        is_outlier_row = torch.isin(cat_chart_d, outlier_t)
        d_colors[cat_member_d[is_outlier_row]] = torch.tensor((1.0, 0.0, 1.0), dtype=torch.float32, device=device)

    def _region_view_colors(region_label: str) -> torch.Tensor:
        mask = region_masks.get(region_label, torch.zeros((count,), dtype=torch.bool, device=device)) & ever_representative
        colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        colors[mask & local_ever_covered] = torch.tensor((0.2, 0.95, 0.3), dtype=torch.float32, device=device)
        colors[mask & ~local_ever_covered & representatives_in_processed_views] = torch.tensor((1.0, 0.6, 0.0), dtype=torch.float32, device=device)
        return colors

    table_colors = _region_view_colors("table_top")
    curved_colors = _region_view_colors("table_side_curved")
    hedge_colors = _region_view_colors("hedge")

    views = {
        VIEW_ORIGINAL_SCENE: original_f_dc,
        VIEW_WL112_BASELINE_SUBSET: _rgb_to_f_dc(baseline_colors),
        VIEW_LOCAL_CHART_NETWORK: _rgb_to_f_dc(local_colors),
        VIEW_UNRESOLVED_EVIDENCE: _rgb_to_f_dc(unresolved_colors),
        VIEW_DOMAIN_OCCUPANCY: _rgb_to_f_dc(occupancy_colors),
        VIEW_RESIDUAL_COMPARE: _rgb_to_f_dc(residual_colors),
        VIEW_OVERLAP: _rgb_to_f_dc(overlap_colors),
        VIEW_D_OUTLIERS: _rgb_to_f_dc(d_colors),
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
    report_path = output_root / "local_rank_complete_chart_network_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
