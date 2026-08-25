"""Worklog 113 -- Chart Representation Contract Diagnostic.

Worklog 112 rejected representative-center/pixel-surface mismatch as the
dominant cause of Worklog 111's failure (residual and overlap got WORSE at
the tails, not better). This batch does NOT tune Worklog 112 and does NOT
implement chart subdivision. It is a diagnostic-only replay that FREEZES:

    - Worklog 107/109 canonical visible topology
      (`torch_camera_induced_visible_adjacency.py`, unmodified);
    - Worklog 111's camera-observed blob construction
      (`label_same_component_blobs`, unmodified);
    - Worklog 112's renderer-native pixel-surface geometry
      (`build_view_chart_pixel_samples`, `depths_to_points`, unmodified);
    - the fixed 8x4 degree-2 NURBS configuration (unmodified).

and asks a single question: WHY does "one camera-observed connected blob ==
one rectangular tensor-product NURBS chart" fail, distinguishing between
(A) insufficient pixel support, (B) non-rectangular/holed blob domains being
forced into a rectangular parameter domain, (C) fixed 8x4 NURBS capacity, and
(D) numerical/grazing-view pathologies in renderer-native median-depth
samples. No new representation mechanism (subdivision, adaptive grids,
variable degree, merging, non-representative attachment) is implemented
here.
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
from osn_gs.render.surfel_geometry import depths_to_points
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative
from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    CameraInducedAdjacencyConfig,
    accumulate_image_space_pairs,
    apply_secondary_geometric_gate,
    filter_by_3d_locality,
)
from osn_gs.surface.torch_camera_observed_chart_domains import (
    build_view_chart_pixel_samples,
    label_same_component_blobs,
    valid_pixel_chart_mask,
)
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

# Frozen exactly as Worklog 111/112 (directive: do not tune).
RESOLUTION_U = 8
RESOLUTION_V = 4
DEGREE_U = 2
DEGREE_V = 2
MIN_PIXEL_SAMPLES = RESOLUTION_U * RESOLUTION_V  # 32
_RANK_SUBSAMPLE_CAP = 2048  # diagnostic-only cost bound, disclosed in report

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_SUPPORT_ATTRIBUTION = "COMPONENT_SUPPORT_ATTRIBUTION"
VIEW_ZERO_COVERAGE_CAUSE = "ZERO_COVERAGE_CAUSE"
VIEW_DOMAIN_OCCUPANCY = "CHART_DOMAIN_OCCUPANCY"
VIEW_DOMAIN_HOLES = "CHART_DOMAIN_HOLES"
VIEW_CAPACITY_RANK = "NURBS_CAPACITY_RANK_DEFICIT"
VIEW_OUTLIER_PROVENANCE = "EXTREME_RESIDUAL_PROVENANCE"
VIEW_TABLE = "TABLE_CONTRACT_DIAGNOSTIC"
VIEW_CURVED = "CURVED_CONTRACT_DIAGNOSTIC"
VIEW_HEDGE = "HEDGE_CONTRACT_DIAGNOSTIC"

ANCHOR_FRACTIONS = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[chart-contract-diag] {message}", flush=True)


def _distribution(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    sorted_values = np.sort(values)

    def _percentile(fraction: float) -> float:
        position = min(sorted_values.shape[0] - 1, max(0, int(round(fraction * (sorted_values.shape[0] - 1)))))
        return float(sorted_values[position])

    return {
        "min": _percentile(0.0), "median": _percentile(0.5), "mean": float(sorted_values.mean()),
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


def _bin_by_quantile(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Deterministic quantile bin index per value (diagnostic grouping only,
    not an acceptance threshold -- directive section 3/5)."""

    if values.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]) if values.shape[0] > n_bins else np.array([])
    return np.searchsorted(edges, values, side="right")


def blob_domain_shape(blob_mask_np: np.ndarray) -> dict[str, Any]:
    """Raster-domain shape diagnostics for ONE connected blob mask (directive
    section 3): bounding-box extent, occupancy ratio, aspect ratio, and
    enclosed-hole count/area. Pure numpy/scipy, no torch dependency, so it is
    directly unit-testable against hand-built masks."""

    rows_np, cols_np = np.nonzero(blob_mask_np)
    if rows_np.shape[0] == 0:
        return {"bbox_h": 0, "bbox_w": 0, "bbox_area": 0, "pixel_count": 0, "occupancy_ratio": 0.0, "aspect_ratio": 0.0, "hole_count": 0, "hole_area": 0}
    rmin, rmax = int(rows_np.min()), int(rows_np.max())
    cmin, cmax = int(cols_np.min()), int(cols_np.max())
    bbox_h, bbox_w = (rmax - rmin + 1), (cmax - cmin + 1)
    bbox_area = bbox_h * bbox_w
    pixel_count = int(rows_np.shape[0])
    occupancy_ratio = float(pixel_count) / float(bbox_area) if bbox_area > 0 else 0.0
    aspect_ratio = float(bbox_w) / float(bbox_h) if bbox_h > 0 else 0.0
    hole_count, hole_area = 0, 0
    from scipy.ndimage import binary_fill_holes, label as ndi_label
    sub = blob_mask_np[rmin:rmax + 1, cmin:cmax + 1]
    filled = binary_fill_holes(sub)
    holes = filled & ~sub
    if bool(holes.any()):
        _hole_labels, hole_count = ndi_label(holes)
        hole_area = int(holes.sum())
    return {
        "bbox_h": bbox_h, "bbox_w": bbox_w, "bbox_area": bbox_area, "pixel_count": pixel_count,
        "occupancy_ratio": occupancy_ratio, "aspect_ratio": aspect_ratio,
        "hole_count": int(hole_count), "hole_area": int(hole_area),
    }


def design_matrix_rank_diagnostics(surface: Any, uv: Any, seed: int = 0, subsample_cap: int = _RANK_SUBSAMPLE_CAP) -> dict[str, Any]:
    """Fixed 8x4 (or whatever `surface`'s actual grid is) NURBS-model capacity
    diagnostic (directive section 6): design-matrix numerical rank and
    condition number of the tensor-product basis at the chart's own fitted
    `uv` samples. Deterministic per `seed` when subsampling is needed for
    cost. Never used to change the model -- diagnostic-only."""

    uv_for_rank = uv
    if int(uv_for_rank.shape[0]) > subsample_cap:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pick = torch.randperm(int(uv_for_rank.shape[0]), generator=gen)[:subsample_cap].to(uv_for_rank.device)
        uv_for_rank = uv_for_rank[pick]
    with torch.no_grad():
        basis_u, basis_v, _du, _dv = surface._basis_tables(uv_for_rank)
        design = (basis_u[:, :, None] * basis_v[:, None, :]).reshape(basis_u.shape[0], -1)
        try:
            rank = int(torch.linalg.matrix_rank(design).item())
            svals = torch.linalg.svdvals(design)
            nz = svals[svals > 1e-10]
            cond = float((svals.max() / nz.min()).item()) if nz.numel() > 0 else float("inf")
        except Exception:
            rank, cond = -1, float("nan")
    return {"rank": rank, "full_capacity": int(design.shape[1]), "cond_number": cond}


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
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--max-charts", type=int, default=0)
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

    # --- sweep: representative maps + renderer-native median-depth unprojection ---
    # (identical mechanism to WL112; additionally retains per-view depth maps and
    # camera centers -- diagnostic-only, needed for section 7 outlier provenance)
    _progress("[sweep] representative_id + median_depth unprojection per view")
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

    # --- region masks (WORKING INTERPRETATION ONLY, same anchor mechanism as WL108-112) ---
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
    for region_index, label in enumerate(ANCHOR_FRACTIONS):
        region_of_representative[region_masks[label]] = region_index
    region_labels_list = list(ANCHOR_FRACTIONS.keys())

    # --- second pass: chart candidates + dense pixel-surface fitting + diagnostics ---
    ever_in_valid_chart = torch.zeros((count,), dtype=torch.bool, device=device)
    valid_chart_membership_count = torch.zeros((count,), dtype=torch.int64, device=device)
    raw_chart_count = 0
    valid_chart_count = 0
    covered_valid_pixels_all_views = 0

    # section 2: corrected per-component support accounting
    component_total_valid_pixels = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    component_max_blob_pixels = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    component_blobs_ge32 = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    component_raw_blob_count = torch.zeros((subset_count,), dtype=torch.int64, device=device)

    all_member_ids: list[torch.Tensor] = []
    all_chart_ids: list[torch.Tensor] = []
    all_view_ids: list[torch.Tensor] = []
    all_fitted_points: list[torch.Tensor] = []
    all_normals: list[torch.Tensor] = []
    residual_values: list[torch.Tensor] = []
    residual_pixel_rep_ids: list[torch.Tensor] = []
    chart_records: list[dict[str, Any]] = []  # one row per fitted (valid) chart -- sections 3/5/6/7/8

    global_chart_id = 0
    max_charts = int(arguments.max_charts) if int(arguments.max_charts) > 0 else None
    stop = False
    for view_index, (rep_remapped_cpu, world_points_cpu) in enumerate(zip(per_view_rep_remapped, per_view_world_points)):
        if stop:
            break
        rep_gpu = rep_remapped_cpu.to(device)
        world_gpu = world_points_cpu.to(device)
        depth_cpu = per_view_depth[view_index]
        camera_center_cpu = per_view_camera_center[view_index]
        valid_pix = rep_gpu >= 0
        comp_map = torch.where(valid_pix, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1))
        vs = build_view_chart_pixel_samples(view_index, comp_map, rep_gpu, world_gpu)
        if vs.blob_count == 0:
            continue
        mask = valid_pixel_chart_mask(vs, MIN_PIXEL_SAMPLES)
        raw_chart_count += vs.blob_count
        valid_chart_count += int(mask.sum().item())
        covered_valid_pixels_all_views += int(vs.blob_pixel_total[mask].sum().item()) if int(mask.sum().item()) > 0 else 0

        # section 2: aggregate support to CANONICAL COMPONENT, for EVERY raw
        # blob (not only valid ones) -- a component's total observed pixel
        # support spans however many blobs across however many views it has.
        comp_per_blob = vs.blob_component_id
        pix_per_blob = vs.blob_pixel_total
        ones_blob = torch.ones_like(comp_per_blob)
        component_total_valid_pixels.index_add_(0, comp_per_blob, pix_per_blob)
        component_raw_blob_count.index_add_(0, comp_per_blob, ones_blob)
        component_max_blob_pixels.scatter_reduce_(0, comp_per_blob, pix_per_blob, reduce="amax", include_self=True)
        ge32 = pix_per_blob >= MIN_PIXEL_SAMPLES
        if bool(ge32.any()):
            component_blobs_ge32.index_add_(0, comp_per_blob[ge32], ones_blob[ge32])

        # section 3: raw raster blob-label map, reused directly (frozen
        # function) for bbox/occupancy/hole diagnostics -- independent of
        # build_view_chart_pixel_samples's own internal call to the same
        # function (deterministic, so blob numbering matches byte-for-byte).
        blob_labels_np = label_same_component_blobs(comp_map).detach().cpu().numpy()

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

            # --- section 3: raster domain shape (bbox / occupancy / holes) ---
            blob_mask_np = blob_labels_np == local_blob
            rows_np, cols_np = np.nonzero(blob_mask_np)
            shape_diag = blob_domain_shape(blob_mask_np)
            bbox_h, bbox_w = shape_diag["bbox_h"], shape_diag["bbox_w"]
            pixel_count = shape_diag["pixel_count"]
            occupancy_ratio, aspect_ratio = shape_diag["occupancy_ratio"], shape_diag["aspect_ratio"]
            hole_count, hole_area = shape_diag["hole_count"], shape_diag["hole_area"]

            # --- section 6: fixed 8x4 capacity -- design-matrix rank/conditioning ---
            capacity_diag = design_matrix_rank_diagnostics(surface, uv, seed=global_chart_id)
            rank, cond, full_capacity = capacity_diag["rank"], capacity_diag["cond_number"], capacity_diag["full_capacity"]

            # --- section 7 support data: depth / grazing-incidence within this chart ---
            rows_t = torch.from_numpy(rows_np).to(torch.int64)
            cols_t = torch.from_numpy(cols_np).to(torch.int64)
            # only the subset belonging to THIS blob AND passing validity (rep>=0);
            # blob_mask already guarantees validity by construction of label_same_component_blobs.
            chart_depths = depth_cpu[rows_t, cols_t]
            view_dirs = pixel_xyz.detach().cpu() - camera_center_cpu.reshape(1, 3)
            view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)
            chart_normals_cpu = normals.detach().cpu()
            # normals correspond to `uv` (all pixels), align by same ordering as pixel_xyz
            incidence_cos = (view_dirs * chart_normals_cpu).sum(dim=-1).abs()

            residual_np = residual.detach().cpu().numpy()
            region_votes = region_of_representative[distinct_reps]
            region_index = -1
            valid_votes = region_votes[region_votes >= 0]
            if int(valid_votes.numel()) > 0:
                region_index = int(torch.mode(valid_votes).values.item())

            chart_records.append({
                "chart_id": global_chart_id,
                "view_index": view_index,
                "camera_name": cameras[view_index].image_name,
                "component_id": component_id,
                "pixel_count": pixel_count,
                "bbox_h": bbox_h, "bbox_w": bbox_w, "bbox_area": shape_diag["bbox_area"],
                "occupancy_ratio": occupancy_ratio, "aspect_ratio": aspect_ratio,
                "hole_count": int(hole_count), "hole_area": int(hole_area),
                "rank": rank, "full_capacity": full_capacity, "cond_number": cond,
                "residual_median": float(np.median(residual_np)),
                "residual_p95": float(np.percentile(residual_np, 95)),
                "residual_max": float(residual_np.max()) if residual_np.shape[0] else 0.0,
                "depth_min": float(chart_depths.min().item()), "depth_max": float(chart_depths.max().item()),
                "depth_std": float(chart_depths.std().item()) if chart_depths.shape[0] > 1 else 0.0,
                "grazing_incidence_median": float(incidence_cos.median().item()) if incidence_cos.shape[0] else float("nan"),
                "grazing_incidence_min": float(incidence_cos.min().item()) if incidence_cos.shape[0] else float("nan"),
                "region_label": region_labels_list[region_index] if region_index >= 0 else None,
            })

            global_chart_id += 1
        if view_index % 10 == 0:
            _progress(f"chart-fit view {view_index + 1}/{len(per_view_rep_remapped)} raw_charts={raw_chart_count} valid_charts={valid_chart_count} fitted={global_chart_id}")

    _progress(f"[chart accounting] raw_chart_count={raw_chart_count} valid_chart_count={valid_chart_count} fitted={global_chart_id}")

    ever_in_valid_chart_visible = ever_in_valid_chart & ever_representative
    no_valid_chart = ever_representative & ~ever_in_valid_chart
    multi_valid_chart = ever_representative & (valid_chart_membership_count >= 2)
    single_valid_chart = ever_representative & (valid_chart_membership_count == 1)

    residual_all = torch.cat(residual_values) if residual_values else torch.zeros((0,))
    residual_distribution = _distribution(residual_all.numpy())

    # --- overlap: consecutive-pair sampling per representative (identical technique to WL111/112) ---
    position_discrepancies: list[torch.Tensor] = []
    normal_discrepancies: list[torch.Tensor] = []
    # per-representative max overlap discrepancy AND which chart pair produced it (for section 7)
    worst_overlap_by_chart: dict[int, float] = {}
    if all_member_ids:
        cat_member = torch.cat(all_member_ids)
        cat_points = torch.cat(all_fitted_points)
        cat_normals = torch.cat(all_normals)
        cat_chart = torch.cat(all_chart_ids)
        order = torch.argsort(cat_member)
        sorted_member = cat_member[order]
        sorted_points = cat_points[order]
        sorted_normals = cat_normals[order]
        sorted_chart = cat_chart[order]
        same_as_next = sorted_member[:-1] == sorted_member[1:]
        if bool(same_as_next.any()):
            diff = (sorted_points[1:] - sorted_points[:-1])[same_as_next]
            pos_disc = diff.norm(dim=-1)
            position_discrepancies.append(pos_disc)
            cos = (sorted_normals[1:] * sorted_normals[:-1]).sum(dim=-1)[same_as_next].clamp(-1.0, 1.0)
            normal_discrepancies.append(torch.rad2deg(torch.acos(cos)))
            chart_a = sorted_chart[:-1][same_as_next]
            chart_b = sorted_chart[1:][same_as_next]
            pos_disc_np = pos_disc.numpy()
            for ca, cb, d in zip(chart_a.tolist(), chart_b.tolist(), pos_disc_np.tolist()):
                worst_overlap_by_chart[ca] = max(worst_overlap_by_chart.get(ca, 0.0), d)
                worst_overlap_by_chart[cb] = max(worst_overlap_by_chart.get(cb, 0.0), d)
    position_discrepancy_distribution = _distribution(torch.cat(position_discrepancies).numpy() if position_discrepancies else np.zeros((0,)))
    normal_discrepancy_distribution = _distribution(torch.cat(normal_discrepancies).numpy() if normal_discrepancies else np.zeros((0,)))
    _progress(f"[overlap] position_discrepancy={position_discrepancy_distribution} normal_discrepancy_degrees={normal_discrepancy_distribution}")

    for record in chart_records:
        record["max_overlap_position_discrepancy"] = worst_overlap_by_chart.get(record["chart_id"], 0.0)

    # --- section 1/2: per-component chartability + zero-coverage attribution ---
    representative_per_component = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    representative_per_component.index_add_(0, subset_ids[ever_representative], torch.ones((int(ever_representative.sum().item()),), dtype=torch.int64, device=device))
    covered_per_component = torch.zeros((subset_count,), dtype=torch.int64, device=device)
    covered_ids = torch.nonzero(ever_in_valid_chart_visible, as_tuple=False).reshape(-1)
    if int(covered_ids.shape[0]) > 0:
        covered_per_component.index_add_(0, subset_ids[covered_ids], torch.ones((int(covered_ids.shape[0]),), dtype=torch.int64, device=device))
    nonzero_component_mask = representative_per_component > 0
    per_component_fraction = torch.zeros((subset_count,), dtype=torch.float32, device=device)
    per_component_fraction[nonzero_component_mask] = covered_per_component[nonzero_component_mask].to(torch.float32) / representative_per_component[nonzero_component_mask].to(torch.float32)
    per_component_coverage_distribution = _distribution(per_component_fraction[nonzero_component_mask].cpu().numpy())

    zero_coverage_mask = nonzero_component_mask & (covered_per_component == 0)
    zero_coverage_ids = torch.nonzero(zero_coverage_mask, as_tuple=False).reshape(-1)
    zero_no_view_reaches_32 = zero_coverage_ids[component_blobs_ge32[zero_coverage_ids] == 0]
    zero_has_ge32_but_uncovered = zero_coverage_ids[component_blobs_ge32[zero_coverage_ids] > 0]
    zero_coverage_attribution = {
        "zero_coverage_component_count": int(zero_coverage_ids.shape[0]),
        "NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES_count": int(zero_no_view_reaches_32.shape[0]),
        "HAS_GE32_BLOB_BUT_STILL_UNCOVERED_count": int(zero_has_ge32_but_uncovered.shape[0]),
        "note": "current fit loop has no failure/skip branch once a blob passes the >=32 pixel-sample mask, so HAS_GE32_BLOB_BUT_STILL_UNCOVERED should be 0 unless max-charts early-stop truncated the run",
    }
    _progress(f"[zero-coverage attribution] {zero_coverage_attribution}")

    # --- section 2: corrected component support accounting table (summary stats only; full table would be 155k rows) ---
    rep_count_np = representative_per_component.cpu().numpy()
    total_pix_np = component_total_valid_pixels.cpu().numpy()
    max_blob_pix_np = component_max_blob_pixels.cpu().numpy()
    blobs_ge32_np = component_blobs_ge32.cpu().numpy()
    covered_np = (covered_per_component > 0).cpu().numpy()
    nz = rep_count_np > 0
    corrected_support_accounting = {
        "components_with_representatives": int(nz.sum()),
        "representative_count_distribution": _distribution(rep_count_np[nz]),
        "total_valid_pixel_observations_distribution": _distribution(total_pix_np[nz]),
        "max_single_blob_pixel_count_distribution": _distribution(max_blob_pix_np[nz]),
        "blobs_with_ge32_pixels_distribution": _distribution(blobs_ge32_np[nz]),
        "components_with_ge1_blob_ge32_pixels": int((blobs_ge32_np[nz] > 0).sum()),
        "components_with_ge1_blob_ge32_pixels_AND_covered": int(((blobs_ge32_np[nz] > 0) & covered_np[nz]).sum()),
        "components_with_zero_blob_ge32_pixels_but_small_rep_count_lt32": int(((blobs_ge32_np[nz] == 0) & (rep_count_np[nz] < 32)).sum()),
        "components_with_zero_blob_ge32_pixels_but_LARGE_rep_count_ge32": int(((blobs_ge32_np[nz] == 0) & (rep_count_np[nz] >= 32)).sum()),
        "note": "the last line answers directive section 2's core distinction: components whose REPRESENTATIVE count already exceeds the fixed model's 32-sample requirement, yet no single camera view ever produced a >=32-pixel BLOB for them (i.e. observations exist but are split thinly across many small per-view blobs) -- this is a genuinely different failure mode from simple small-population starvation",
    }
    _progress(f"[corrected support accounting] {corrected_support_accounting}")

    # --- section 3/5: domain-shape distributions + relation to fit quality (valid charts only) ---
    if chart_records:
        pixel_counts = np.array([r["pixel_count"] for r in chart_records], dtype=np.float64)
        occupancy = np.array([r["occupancy_ratio"] for r in chart_records], dtype=np.float64)
        aspect = np.array([r["aspect_ratio"] for r in chart_records], dtype=np.float64)
        hole_counts = np.array([r["hole_count"] for r in chart_records], dtype=np.float64)
        residual_median_arr = np.array([r["residual_median"] for r in chart_records], dtype=np.float64)
        residual_p95_arr = np.array([r["residual_p95"] for r in chart_records], dtype=np.float64)
        overlap_arr = np.array([r["max_overlap_position_discrepancy"] for r in chart_records], dtype=np.float64)
        rank_arr = np.array([r["rank"] for r in chart_records], dtype=np.float64)
        cond_arr = np.array([r["cond_number"] for r in chart_records], dtype=np.float64)
    else:
        pixel_counts = occupancy = aspect = hole_counts = residual_median_arr = residual_p95_arr = overlap_arr = rank_arr = cond_arr = np.zeros((0,))

    blob_domain_shape_distributions = {
        "pixel_count": _distribution(pixel_counts),
        "bbox_occupancy_ratio": _distribution(occupancy),
        "aspect_ratio_width_over_height": _distribution(aspect),
        "hole_count": _distribution(hole_counts),
        "fraction_of_charts_with_ge1_hole": float((hole_counts > 0).mean()) if hole_counts.shape[0] else 0.0,
        "unsupported_rectangular_domain_fraction_median": float(np.median(1.0 - occupancy)) if occupancy.shape[0] else 0.0,
        "unsupported_rectangular_domain_fraction_p95": float(np.percentile(1.0 - occupancy, 95)) if occupancy.shape[0] else 0.0,
        "note": "unsupported_rectangular_domain_fraction = 1 - occupancy_ratio: the share of each fitted chart's bounding rectangle with NO observed renderer-surface pixel at all -- this is what the rectangular tensor-product NURBS domain is asked to model past its actually-observed support",
    }
    _progress(f"[domain shape] {blob_domain_shape_distributions}")

    def _bin_report(values: np.ndarray, label: str) -> dict[str, Any]:
        if values.shape[0] < 8:
            return {"note": "insufficient charts for quantile binning"}
        bins = _bin_by_quantile(values, n_bins=4)
        out = {}
        for b in sorted(set(bins.tolist())):
            sel = bins == b
            out[f"{label}_bin_{b}"] = {
                "value_range": [float(values[sel].min()), float(values[sel].max())],
                "chart_count": int(sel.sum()),
                "residual_median": _distribution(residual_median_arr[sel]),
                "residual_p95_of_p95": float(np.percentile(residual_p95_arr[sel], 95)) if sel.sum() else 0.0,
                "overlap_position_median": float(np.median(overlap_arr[sel])) if sel.sum() else 0.0,
                "overlap_position_p95": float(np.percentile(overlap_arr[sel], 95)) if sel.sum() else 0.0,
            }
        return out

    fit_quality_vs_domain_shape = {
        "by_pixel_count_quantile": _bin_report(pixel_counts, "sample_count"),
        "by_occupancy_ratio_quantile": _bin_report(occupancy, "occupancy"),
        "by_aspect_ratio_quantile": _bin_report(aspect, "aspect"),
        "by_hole_count_present": {
            "no_holes": {
                "chart_count": int((hole_counts == 0).sum()),
                "residual_median": _distribution(residual_median_arr[hole_counts == 0]),
                "overlap_position_median": float(np.median(overlap_arr[hole_counts == 0])) if (hole_counts == 0).sum() else 0.0,
            },
            "has_holes": {
                "chart_count": int((hole_counts > 0).sum()),
                "residual_median": _distribution(residual_median_arr[hole_counts > 0]),
                "overlap_position_median": float(np.median(overlap_arr[hole_counts > 0])) if (hole_counts > 0).sum() else 0.0,
            },
        },
    }
    _progress(f"[fit quality vs domain shape] computed {len(fit_quality_vs_domain_shape)} groupings")

    # --- section 6: fixed 8x4 capacity diagnostics ---
    capacity_diagnostics = {
        "rank_distribution": _distribution(rank_arr[rank_arr >= 0]),
        "full_capacity": int(RESOLUTION_U * RESOLUTION_V),
        "fraction_full_rank": float((rank_arr == (RESOLUTION_U * RESOLUTION_V)).mean()) if rank_arr.shape[0] else 0.0,
        "condition_number_distribution": _distribution(cond_arr[np.isfinite(cond_arr)]),
        "residual_by_rank_deficiency": {
            "full_rank": {
                "chart_count": int((rank_arr == (RESOLUTION_U * RESOLUTION_V)).sum()),
                "residual_median": _distribution(residual_median_arr[rank_arr == (RESOLUTION_U * RESOLUTION_V)]),
            },
            "rank_deficient": {
                "chart_count": int((rank_arr < (RESOLUTION_U * RESOLUTION_V)).sum()),
                "residual_median": _distribution(residual_median_arr[rank_arr < (RESOLUTION_U * RESOLUTION_V)]),
            },
        },
        "residual_vs_sample_count_at_full_rank_correlation_hint": (
            float(np.corrcoef(pixel_counts[rank_arr == (RESOLUTION_U * RESOLUTION_V)], residual_median_arr[rank_arr == (RESOLUTION_U * RESOLUTION_V)])[0, 1])
            if int((rank_arr == (RESOLUTION_U * RESOLUTION_V)).sum()) > 8 else None
        ),
        "note": "rank subsampled to at most 2048 pixel rows per chart for cost (diagnostic-only, deterministic per chart_id seed); this changes the estimate's precision, not its qualitative meaning",
    }
    _progress(f"[capacity diagnostics] fraction_full_rank={capacity_diagnostics['fraction_full_rank']}")

    # --- section 7: extreme outlier provenance ---
    top_residual = sorted(chart_records, key=lambda r: r["residual_max"], reverse=True)[:15]
    top_overlap = sorted(chart_records, key=lambda r: r["max_overlap_position_discrepancy"], reverse=True)[:15]

    def _provenance_row(r: dict[str, Any]) -> dict[str, Any]:
        return {k: r[k] for k in (
            "chart_id", "view_index", "camera_name", "component_id", "pixel_count",
            "bbox_h", "bbox_w", "occupancy_ratio", "aspect_ratio", "hole_count",
            "rank", "full_capacity", "cond_number", "residual_median", "residual_p95", "residual_max",
            "depth_min", "depth_max", "depth_std", "grazing_incidence_median", "grazing_incidence_min",
            "max_overlap_position_discrepancy", "region_label",
        )}

    extreme_outlier_provenance = {
        "top_15_by_residual_max": [_provenance_row(r) for r in top_residual],
        "top_15_by_overlap_position_discrepancy": [_provenance_row(r) for r in top_overlap],
    }
    if top_residual:
        _progress(f"[worst residual chart] {_provenance_row(top_residual[0])}")

    # --- section 8: per-region breakdown using chart-level records ---
    region_results = {}
    for label, mask in region_masks.items():
        rep_pop = ever_representative & mask
        covered_pop = ever_in_valid_chart_visible & mask
        region_chart_records = [r for r in chart_records if r["region_label"] == label]
        rc_pixel_counts = np.array([r["pixel_count"] for r in region_chart_records], dtype=np.float64)
        rc_occupancy = np.array([r["occupancy_ratio"] for r in region_chart_records], dtype=np.float64)
        rc_hole = np.array([r["hole_count"] for r in region_chart_records], dtype=np.float64)
        rc_residual = np.array([r["residual_median"] for r in region_chart_records], dtype=np.float64)
        region_components = torch.unique(subset_ids[rep_pop])
        region_zero_cov = int(((representative_per_component[region_components] > 0) & (covered_per_component[region_components] == 0)).sum().item()) if int(region_components.shape[0]) else 0
        region_zero_cov_ids = region_components[(representative_per_component[region_components] > 0) & (covered_per_component[region_components] == 0)] if int(region_components.shape[0]) else torch.zeros((0,), dtype=torch.int64, device=device)
        region_zero_no_view32 = int((component_blobs_ge32[region_zero_cov_ids] == 0).sum().item()) if int(region_zero_cov_ids.shape[0]) else 0
        region_results[label] = {
            "representative_count": int(rep_pop.sum().item()),
            "covered_by_valid_chart_count": int(covered_pop.sum().item()),
            "coverage_fraction": (float(covered_pop.sum().item()) / float(rep_pop.sum().item())) if int(rep_pop.sum().item()) > 0 else 0.0,
            "no_valid_chart_count": int((rep_pop & no_valid_chart).sum().item()),
            "multi_valid_chart_count": int((rep_pop & multi_valid_chart).sum().item()),
            "component_count_in_region": int(region_components.shape[0]),
            "zero_coverage_component_count_in_region": region_zero_cov,
            "zero_coverage_NO_VIEW_BLOB_REACHES_32_in_region": region_zero_no_view32,
            "fitted_chart_count_in_region": int(len(region_chart_records)),
            "chart_pixel_count_distribution": _distribution(rc_pixel_counts),
            "chart_occupancy_ratio_distribution": _distribution(rc_occupancy),
            "chart_hole_count_distribution": _distribution(rc_hole),
            "chart_residual_median_distribution": _distribution(rc_residual),
        }
    _progress(f"[region results] {json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in region_results.items()}, indent=2)}")

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

    # --- section 11/12: classification (derived from measured evidence, no tuning) ---
    frac_zero_no_view32 = (
        float(zero_coverage_attribution["NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES_count"]) / float(zero_coverage_attribution["zero_coverage_component_count"])
        if zero_coverage_attribution["zero_coverage_component_count"] else 0.0
    )
    frac_unsupported_domain_p95 = blob_domain_shape_distributions["unsupported_rectangular_domain_fraction_p95"]
    frac_full_rank = capacity_diagnostics["fraction_full_rank"]
    classification = {
        "A_SUPPORT_LIMITED": {
            "applies": frac_zero_no_view32 > 0.9,
            "evidence": f"{frac_zero_no_view32:.4f} fraction of zero-coverage components have NO camera blob ever reaching {MIN_PIXEL_SAMPLES} valid pixel samples",
        },
        "B_CHART_UNIT_RECTANGULAR_DOMAIN_FAILURE": {
            "applies": frac_unsupported_domain_p95 > 0.3,
            "evidence": f"p95 unsupported-rectangular-domain-fraction across fitted charts is {frac_unsupported_domain_p95:.4f}; fraction with >=1 hole is {blob_domain_shape_distributions['fraction_of_charts_with_ge1_hole']:.4f}",
        },
        "C_FIXED_NURBS_CAPACITY_FAILURE": {
            "applies": frac_full_rank > 0.5 and residual_distribution["p95"] > 0.3,
            "evidence": f"fraction of fitted charts at full design-matrix rank is {frac_full_rank:.4f} (i.e. NOT capacity-starved numerically) while residual p95 is {residual_distribution['p95']:.4f}",
        },
        "D_NUMERICAL_GRAZING_SURFACE_FAILURE": {
            "applies": residual_distribution["max"] > 50.0,
            "evidence": f"residual max={residual_distribution['max']:.2f} vs p95={residual_distribution['p95']:.4f}; see extreme_outlier_provenance for traced source",
        },
    }
    _progress(f"[classification] {classification}")

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 113",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "camera_meta": camera_meta,
        "nurbs_config": {"resolution_u": RESOLUTION_U, "resolution_v": RESOLUTION_V, "degree_u": DEGREE_U, "degree_v": DEGREE_V, "min_pixel_samples": MIN_PIXEL_SAMPLES},
        "accounting": accounting,
        "wl107_109_replay_consistency_check": wl107_replay_stats,
        "corrected_component_support_accounting": corrected_support_accounting,
        "zero_coverage_component_attribution": zero_coverage_attribution,
        "per_component_chart_coverage_fraction_distribution_LABEL": "unweighted count of components at 100% vs 0% coverage -- NOT a scene surface-area metric; retained ONLY as a replay-consistency check against WL111/112, not reopened as a topology failure per directive section 9",
        "per_component_chart_coverage_fraction_distribution": per_component_coverage_distribution,
        "blob_domain_shape_distributions": blob_domain_shape_distributions,
        "fit_quality_vs_domain_shape": fit_quality_vs_domain_shape,
        "fixed_nurbs_capacity_diagnostics": capacity_diagnostics,
        "fitting_residual_distribution": residual_distribution,
        "overlap_position_discrepancy_distribution": position_discrepancy_distribution,
        "overlap_normal_discrepancy_degrees_distribution": normal_discrepancy_distribution,
        "extreme_outlier_provenance": extreme_outlier_provenance,
        "region_results_WORKING_INTERPRETATION_ONLY": region_results,
        "dominant_failure_classification": classification,
        "runtime_seconds": {"total": time.time() - started},
    }

    # --- colors / exports ---
    original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]

    support_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    # per-representative: fraction of its component's blobs that reach >=32 pixels
    comp_support_ratio = torch.zeros((subset_count,), dtype=torch.float32, device=device)
    has_raw = component_raw_blob_count > 0
    comp_support_ratio[has_raw] = component_blobs_ge32[has_raw].to(torch.float32) / component_raw_blob_count[has_raw].to(torch.float32)
    support_colors[ever_representative] = _ramp(comp_support_ratio[subset_ids[ever_representative]], (0.7, 0.1, 0.05), (0.2, 0.95, 0.3))

    zero_cause_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    zero_cause_colors[ever_in_valid_chart_visible] = torch.tensor((0.1, 0.3, 0.15), dtype=torch.float32, device=device)
    no_view32_reps = torch.isin(subset_ids, zero_no_view_reaches_32) & ever_representative & no_valid_chart
    ge32_uncovered_reps = torch.isin(subset_ids, zero_has_ge32_but_uncovered) & ever_representative & no_valid_chart
    zero_cause_colors[no_view32_reps] = torch.tensor((1.0, 0.15, 0.05), dtype=torch.float32, device=device)
    zero_cause_colors[ge32_uncovered_reps] = torch.tensor((1.0, 0.85, 0.0), dtype=torch.float32, device=device)

    # domain occupancy / holes: color by first-chart-of-member's occupancy/hole stats
    first_chart_of_member = torch.full((count,), -1, dtype=torch.int64, device=device)
    if all_member_ids:
        cat_member = torch.cat(all_member_ids).to(device)
        cat_chart = torch.cat(all_chart_ids).to(device)
        order2 = torch.argsort(cat_member * (global_chart_id + 1) + cat_chart)
        sm, sc = cat_member[order2], cat_chart[order2]
        first_mask = torch.ones_like(sm, dtype=torch.bool)
        first_mask[1:] = sm[1:] != sm[:-1]
        first_chart_of_member[sm[first_mask]] = sc[first_mask]
    has_chart = first_chart_of_member >= 0

    occupancy_by_chart = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
    hole_by_chart = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
    rank_deficit_by_chart = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
    for r in chart_records:
        occupancy_by_chart[r["chart_id"]] = r["occupancy_ratio"]
        hole_by_chart[r["chart_id"]] = 1.0 if r["hole_count"] > 0 else 0.0
        rank_deficit_by_chart[r["chart_id"]] = 1.0 if (r["rank"] >= 0 and r["rank"] < r["full_capacity"]) else 0.0

    occupancy_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    occupancy_colors[has_chart] = _ramp(occupancy_by_chart[first_chart_of_member[has_chart]], (0.9, 0.15, 0.05), (0.2, 0.95, 0.3))
    hole_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    hole_colors[has_chart] = _ramp(hole_by_chart[first_chart_of_member[has_chart]], (0.2, 0.95, 0.3), (1.0, 0.5, 0.0))
    rank_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    rank_colors[has_chart] = _ramp(rank_deficit_by_chart[first_chart_of_member[has_chart]], (0.2, 0.95, 0.3), (0.85, 0.1, 0.75))

    outlier_ids = {r["chart_id"] for r in top_residual} | {r["chart_id"] for r in top_overlap}
    outlier_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    outlier_colors[has_chart] = torch.tensor((0.15, 0.2, 0.25), dtype=torch.float32, device=device)
    if outlier_ids and all_member_ids:
        cat_member_o = torch.cat(all_member_ids).to(device)
        cat_chart_o = torch.cat(all_chart_ids).to(device)
        outlier_chart_tensor = torch.tensor(sorted(outlier_ids), dtype=torch.int64, device=device)
        is_outlier_row = torch.isin(cat_chart_o, outlier_chart_tensor)
        outlier_colors[cat_member_o[is_outlier_row]] = torch.tensor((1.0, 0.0, 1.0), dtype=torch.float32, device=device)

    def _region_view_colors(region_label: str) -> torch.Tensor:
        mask = region_masks.get(region_label, torch.zeros((count,), dtype=torch.bool, device=device)) & ever_representative
        colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        colors[mask & ever_in_valid_chart] = torch.tensor((0.2, 0.95, 0.3), dtype=torch.float32, device=device)
        colors[mask & no_valid_chart & no_view32_reps] = torch.tensor((1.0, 0.15, 0.05), dtype=torch.float32, device=device)
        colors[mask & no_valid_chart & ~no_view32_reps] = torch.tensor((1.0, 0.85, 0.0), dtype=torch.float32, device=device)
        return colors

    table_colors = _region_view_colors("table_top")
    curved_colors = _region_view_colors("table_side_curved")
    hedge_colors = _region_view_colors("hedge")

    views = {
        VIEW_ORIGINAL_SCENE: original_f_dc,
        VIEW_SUPPORT_ATTRIBUTION: _rgb_to_f_dc(support_colors),
        VIEW_ZERO_COVERAGE_CAUSE: _rgb_to_f_dc(zero_cause_colors),
        VIEW_DOMAIN_OCCUPANCY: _rgb_to_f_dc(occupancy_colors),
        VIEW_DOMAIN_HOLES: _rgb_to_f_dc(hole_colors),
        VIEW_CAPACITY_RANK: _rgb_to_f_dc(rank_colors),
        VIEW_OUTLIER_PROVENANCE: _rgb_to_f_dc(outlier_colors),
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
    report_path = output_root / "chart_representation_contract_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
