"""Worklog 118 -- Visible-NURBS Evidence Contract Closure.

Worklog 117 is accepted conditionally: its strongest valid result is that
hole/unsupported-boundary proximity does NOT explain the dominant giant-patio
residual failure under the current WL112 fitting path -- NOT that B2 does not
exist, and NOT that capacity is proven dominant. A new audit found that
WL117's hole-distance analysis used POST-FIT foot-point UV occupancy, while
WL113's original "holey chart" observation was defined in the CAMERA-RASTER
chart domain -- a real domain mismatch this batch closes explicitly.

This batch freezes WL107/109 topology, WL112 camera-blob membership, and the
fixed 8x4/degree-2 model (FOR CONTROL ONLY). It does not implement any new
NURBS mechanism. It separates and measures, in one real-scene pass:

  (2) uv_camera vs uv_footpoint displacement;
  (3) camera-domain vs fitted-domain support/hole disagreement (correcting
      WL117's domain conflation);
  (4) a fixed-camera-UV vs corrected-UV (foot-point) fitting A/B, reusing
      the existing regularized solver's fixed-UV capability directly
      (`_solve_control_grid_lsq`) rather than inventing a new fitter;
  (5) median-event low-pass provenance (rho3d/rho2d/s), via the sibling
      diagnostic CUDA build's new (worklog 118) output fields -- canonical
      renderer untouched;
  (6) a rasterizer-pixel-center-convention unprojection control, as a
      sibling diagnostic function -- `depths_to_points` itself untouched;
  (7) signed vs sign-invariant (acos(abs(dot))) normal disagreement;
  (8) representative finite-support spatial spread vs cross-chart mean
      displacement, decomposing "overlap position error";
  (9) equal-retained-count synthetic full/hole/dispersed-removal controls.

No topology change. No new representation mechanism.
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
from chart_representation_contract_diagnostic import _bin_by_quantile, _distribution, blob_domain_shape  # noqa: E402 -- WL113, frozen
from holey_chart_fitting_coupling_attribution import hole_and_edge_masks  # noqa: E402 -- WL117, frozen
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline  # noqa: E402 -- existing occupancy-mask semantics, reused read-only
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
from osn_gs.surface.torch_nurbs import _solve_control_grid_lsq, fit_torch_visible_surface, fit_torch_visible_surface_lsq
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel
from osn_gs.utils.torch_ops import require_torch

_ITERATION_DIR = "iteration_0000001"
_MIDDEPTH_OFFSET = 5
_UNCUT_RGB = (0.08, 0.09, 0.11)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

# Frozen exactly as WL112 (directive: control only, not canonical).
RESOLUTION_U = 8
RESOLUTION_V = 4
DEGREE_U = 2
DEGREE_V = 2
MIN_PIXEL_SAMPLES = RESOLUTION_U * RESOLUTION_V  # WL112's own gate, preserved for control fidelity only
TRIM_RESOLUTION = 24  # existing torch_pipeline.py config default, reused verbatim
TRIM_DILATION = 1

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_UV_DISPLACEMENT = "UV_CAMERA_VS_FOOTPOINT_DISPLACEMENT"
VIEW_DOMAIN_DISAGREEMENT = "CAMERA_VS_FITTED_DOMAIN_DISAGREEMENT"
VIEW_FIXED_UV_RESIDUAL = "FIXED_UV_RESIDUAL"
VIEW_CORRECTED_UV_RESIDUAL = "CORRECTED_UV_RESIDUAL"
VIEW_NORMAL_SIGN_EFFECT = "NORMAL_SIGN_INVARIANCE_EFFECT"
VIEW_LOW_PASS_PROVENANCE = "LOW_PASS_DOMINATED_EVENTS"
VIEW_HALF_PIXEL_DISPLACEMENT = "HALF_PIXEL_UNPROJECTION_DISPLACEMENT"

ANCHOR_FRACTIONS = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[nurbs-evidence-closure] {message}", flush=True)


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


# --------------------------------------------------------------------------
# Section 6: rasterizer-pixel-center-convention unprojection -- a SIBLING
# diagnostic function. `osn_gs/render/surfel_geometry.py::depths_to_points`
# (OFFICIAL_CODE_FAITHFUL) is NEVER modified. The only change here is the
# ndc2pix half-pixel offset: `W/2, H/2` (official 2DGS convention, what
# depths_to_points already uses) vs `(W-1)/2, (H-1)/2` (the CUDA rasterizer's
# OWN `compute_transmat` convention -- see surfel_geometry.py's own docstring,
# which already names this exact inconsistency as deliberately preserved).
# --------------------------------------------------------------------------

def depths_to_points_rasterizer_pixel_center(view: Any, depthmap: Any) -> Any:
    torch = require_torch()
    device = depthmap.device
    c2w = (view.world_view_transform.T).inverse()
    W, H = int(view.image_width), int(view.image_height)
    ndc2pix = torch.tensor(
        [
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    ).T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


# --------------------------------------------------------------------------
# Section 7: sign-invariant normal comparison.
# --------------------------------------------------------------------------

def sign_invariant_normal_discrepancy_degrees(normal_a: torch.Tensor, normal_b: torch.Tensor) -> torch.Tensor:
    cos = (normal_a * normal_b).sum(dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# --------------------------------------------------------------------------
# Section 9: equal-retained-count synthetic controls.
# --------------------------------------------------------------------------

def _grid_uv(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
    return ii / max(rows - 1, 1), jj / max(cols - 1, 1)


def _geometry_xyz(u: torch.Tensor, v: torch.Tensor, curved: bool) -> torch.Tensor:
    x = u * 2.0
    y = v * 1.0
    z = 0.4 * torch.sin(torch.pi * u) * torch.sin(torch.pi * v) if curved else torch.zeros_like(u)
    return torch.stack([x, y, z], dim=-1)


def run_equal_count_synthetic_contracts() -> dict[str, Any]:
    """A. FULL support; B. CENTER-HOLE (enclosed) support; C. SAME retained
    count as B but removed as scattered/dispersed samples (no enclosed
    hole). Keeps geometry, NURBS config, and retained sample count between
    B and C identical -- isolates enclosed-hole TOPOLOGY from mere sample
    count."""

    results: dict[str, Any] = {}
    rows, cols = 24, 24
    for label, curved in (("planar", False), ("curved", True)):
        u, v = _grid_uv(rows, cols)
        xyz_full = _geometry_xyz(u, v, curved)
        u_flat, v_flat, xyz_flat = u.reshape(-1), v.reshape(-1), xyz_full.reshape(-1, 3)

        hole_keep = ~((u_flat > 0.35) & (u_flat < 0.65) & (v_flat > 0.35) & (v_flat < 0.65))
        removed_count = int((~hole_keep).sum())

        # C: remove the SAME COUNT, but as a deterministic dispersed pattern
        # (every Nth sample in raster order) rather than one enclosed block.
        total = u_flat.shape[0]
        stride = max(1, total // max(removed_count, 1))
        dispersed_remove_idx = torch.arange(0, total, stride)[:removed_count]
        dispersed_keep = torch.ones((total,), dtype=torch.bool)
        dispersed_keep[dispersed_remove_idx] = False

        def _fit_and_score(keep_mask: torch.Tensor, fixed_uv: bool) -> dict[str, Any]:
            uv_k = torch.stack([u_flat[keep_mask], v_flat[keep_mask]], dim=-1)
            xyz_k = xyz_flat[keep_mask]
            with torch.no_grad():
                if fixed_uv:
                    surface = fit_torch_visible_surface(xyz_k, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V, degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=uv_k)
                    surface.control_grid = _solve_control_grid_lsq(xyz_k, uv_k, surface, 1e-4, 1e-4, 4096, None)
                    fitted = surface.evaluate(uv_k)
                else:
                    surface, fit_uv = fit_torch_visible_surface_lsq(
                        xyz_k, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V, degree_u=DEGREE_U, degree_v=DEGREE_V,
                        initial_uv=uv_k, correction_rounds=2, projection_iterations=3,
                    )
                    fitted = surface.evaluate(fit_uv)
                residual = (fitted - xyz_k).norm(dim=-1)
            return {"retained_count": int(keep_mask.sum()), "residual": _distribution(residual.numpy())}

        results[label] = {
            "removed_count_in_B_and_C": removed_count,
            "A_full_footpoint": _fit_and_score(torch.ones((total,), dtype=torch.bool), fixed_uv=False),
            "B_enclosed_hole_footpoint": _fit_and_score(hole_keep, fixed_uv=False),
            "C_dispersed_removal_footpoint": _fit_and_score(dispersed_keep, fixed_uv=False),
            "A_full_fixed_uv": _fit_and_score(torch.ones((total,), dtype=torch.bool), fixed_uv=True),
            "B_enclosed_hole_fixed_uv": _fit_and_score(hole_keep, fixed_uv=True),
            "C_dispersed_removal_fixed_uv": _fit_and_score(dispersed_keep, fixed_uv=True),
        }
    return results


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

    _progress("[section 9] running equal-count synthetic controls")
    synthetic_results = run_equal_count_synthetic_contracts()
    _progress(f"[section 9] {json.dumps(synthetic_results, indent=2)[:1500]}")

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

    _progress("[full-scene sweep] representative_id + median_depth + low-pass provenance + alt-unprojection per view")
    per_view_rep_remapped: list[torch.Tensor] = []
    per_view_world_points: list[torch.Tensor] = []
    per_view_world_points_alt: list[torch.Tensor] = []
    per_view_rho3d: list[torch.Tensor] = []
    per_view_rho2d: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=device)
    half_pixel_displacement_samples: list[torch.Tensor] = []

    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        represented_ids = torch.unique(rep_full[valid])
        ever_representative_full[represented_ids] = True

        depth_map = diag["out_others"][_MIDDEPTH_OFFSET]
        with torch.no_grad():
            world_points = depths_to_points(camera, depth_map.unsqueeze(0)).reshape(*depth_map.shape, 3)
            world_points_alt = depths_to_points_rasterizer_pixel_center(camera, depth_map.unsqueeze(0)).reshape(*depth_map.shape, 3)

        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_rep_remapped.append(rep_remapped.detach().cpu())
        per_view_world_points.append(world_points.detach().cpu())
        per_view_world_points_alt.append(world_points_alt.detach().cpu())
        per_view_rho3d.append(diag["median_rho3d"].detach().cpu())
        per_view_rho2d.append(diag["median_rho2d"].detach().cpu())

        if bool(valid.any()):
            disp = (world_points[valid] - world_points_alt[valid]).norm(dim=-1)
            if disp.shape[0] > 2000:
                perm = torch.randperm(disp.shape[0], device=disp.device)[:2000]
                disp = disp[perm]
            half_pixel_displacement_samples.append(disp.detach().cpu())
        del diag
        if index % 20 == 0:
            _progress(f"sweep view {index + 1}/{len(cameras)}")

    ever_representative = ever_representative_full[visible_selector]
    representative_count = int(ever_representative.sum().item())
    _progress(f"[accounting] median_surface_representatives={representative_count}")

    half_pixel_displacement = torch.cat(half_pixel_displacement_samples) if half_pixel_displacement_samples else torch.zeros((0,))
    half_pixel_displacement_summary = _distribution(half_pixel_displacement.numpy())
    _progress(f"[section 6] half-pixel unprojection displacement (sampled): {half_pixel_displacement_summary}")

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

    # --- main loop ---
    chart_records: list[dict[str, Any]] = []
    all_member_ids_a: list[torch.Tensor] = []
    all_chart_ids_a: list[torch.Tensor] = []
    all_fitted_points_a: list[torch.Tensor] = []
    all_normals_a: list[torch.Tensor] = []

    global_chart_id = 0
    max_charts = int(arguments.max_charts) if int(arguments.max_charts) > 0 else None
    stop = False

    for view_index, (rep_remapped_cpu, world_points_cpu) in enumerate(zip(per_view_rep_remapped, per_view_world_points)):
        if stop:
            break
        rep_gpu = rep_remapped_cpu.to(device)
        world_gpu = world_points_cpu.to(device)
        rho3d_cpu = per_view_rho3d[view_index]
        rho2d_cpu = per_view_rho2d[view_index]
        valid_pix = rep_gpu >= 0
        comp_map = torch.where(valid_pix, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1))
        vs = build_view_chart_pixel_samples(view_index, comp_map, rep_gpu, world_gpu)
        if vs.blob_count == 0:
            continue
        mask_valid = valid_pixel_chart_mask(vs, MIN_PIXEL_SAMPLES)
        valid_blob_ids = torch.nonzero(mask_valid, as_tuple=False).reshape(-1).tolist()
        blob_labels_np = label_same_component_blobs(comp_map).detach().cpu().numpy()

        for local_blob in valid_blob_ids:
            if max_charts is not None and global_chart_id >= max_charts:
                stop = True
                break
            pixel_sel = vs.pixel_blob_id == local_blob
            uv_camera = vs.pixel_uv[pixel_sel]
            pixel_xyz = vs.pixel_xyz[pixel_sel]
            pixel_rep_ids = vs.pixel_representative_id[pixel_sel]
            component_id = int(vs.blob_component_id[local_blob].item())

            with torch.no_grad():
                # ARM A -- CURRENT (camera UV init -> LSQ -> foot-point correction)
                surface_a, uv_footpoint = fit_torch_visible_surface_lsq(
                    pixel_xyz, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                    degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=uv_camera,
                    correction_rounds=2, projection_iterations=3,
                )
                fitted_a = surface_a.evaluate(uv_footpoint)
                residual_a = (fitted_a - pixel_xyz).norm(dim=-1)
                normals_a = surface_a.normals(uv_footpoint)

                # ARM B -- FIXED CAMERA UV (same IDW seed, ONE regularized
                # solve at the externally supplied fixed uv_camera, no
                # foot-point reparameterization loop at all).
                surface_b = fit_torch_visible_surface(
                    pixel_xyz, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                    degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=uv_camera,
                )
                surface_b.control_grid = _solve_control_grid_lsq(pixel_xyz, uv_camera, surface_b, 1e-4, 1e-4, 4096, None)
                fitted_b = surface_b.evaluate(uv_camera)
                residual_b = (fitted_b - pixel_xyz).norm(dim=-1)

            uv_displacement = (uv_footpoint - uv_camera).norm(dim=-1)

            # --- section 3: camera-domain (raster) vs fitted-domain (uv_footpoint) support ---
            blob_mask_np = blob_labels_np == local_blob
            camera_domain_shape = blob_domain_shape(blob_mask_np)  # WL113 raster-native

            camera_mask = TorchOSNGSPipeline._uv_occupancy_mask(uv_camera.detach(), TRIM_RESOLUTION, TRIM_DILATION)
            fitted_mask = TorchOSNGSPipeline._uv_occupancy_mask(uv_footpoint.detach(), TRIM_RESOLUTION, TRIM_DILATION)
            camera_mask_np, fitted_mask_np = camera_mask.cpu().numpy(), fitted_mask.cpu().numpy()
            camera_holes_np, _ = hole_and_edge_masks(camera_mask_np)
            fitted_holes_np, _ = hole_and_edge_masks(fitted_mask_np)
            intersection = int((camera_mask_np & fitted_mask_np).sum())
            union = int((camera_mask_np | fitted_mask_np).sum())
            iou = (intersection / union) if union > 0 else 1.0

            # --- section 5/D-outlier: low-pass provenance for this chart's own pixels ---
            rows_t = torch.from_numpy(np.nonzero(blob_mask_np)[0]).to(torch.int64)
            cols_t = torch.from_numpy(np.nonzero(blob_mask_np)[1]).to(torch.int64)
            chart_rho3d = rho3d_cpu[rows_t, cols_t]
            chart_rho2d = rho2d_cpu[rows_t, cols_t]
            low_pass_valid = chart_rho3d >= 0
            low_pass_dominated_fraction = float((chart_rho2d[low_pass_valid] < chart_rho3d[low_pass_valid]).float().mean()) if bool(low_pass_valid.any()) else float("nan")

            # --- section 8: representative finite-support spread (within this chart) ---
            # Vectorized (no per-representative Python loop -- the original
            # version looped over every distinct representative id inside
            # every chart, which serialized on the CPU for charts with many
            # thousands of representatives and left the GPU mostly idle;
            # caught and fixed mid-run via scatter/index_add_ group-mean +
            # group-max-deviation, identical result, no Python-level loop).
            distinct_reps, inverse_rep = torch.unique(pixel_rep_ids, return_inverse=True)
            rep_count = int(distinct_reps.shape[0])
            group_sum = torch.zeros((rep_count, 3), device=device)
            group_counts = torch.zeros((rep_count,), device=device)
            group_sum.index_add_(0, inverse_rep, pixel_xyz)
            group_counts.index_add_(0, inverse_rep, torch.ones_like(inverse_rep, dtype=torch.float32))
            group_mean = group_sum / group_counts.clamp_min(1.0)[:, None]
            deviation = (pixel_xyz - group_mean[inverse_rep]).norm(dim=-1)
            group_max_deviation = torch.zeros((rep_count,), device=device)
            group_max_deviation.scatter_reduce_(0, inverse_rep, deviation, reduce="amax", include_self=True)
            multi_member = group_counts > 1
            # np.median (not torch's .median(), which returns the lower of
            # the two middle values for even-length input instead of
            # averaging them) for consistency with every other median in
            # this script's reporting.
            within_chart_spread_median = float(np.median(group_max_deviation[multi_member].cpu().numpy())) if bool(multi_member.any()) else 0.0

            sum_point = torch.zeros((rep_count, 3), device=device)
            sum_normal = torch.zeros((rep_count, 3), device=device)
            member_counts = torch.zeros((rep_count,), device=device)
            sum_point.index_add_(0, inverse_rep, fitted_a)
            sum_normal.index_add_(0, inverse_rep, normals_a)
            member_counts.index_add_(0, inverse_rep, torch.ones_like(inverse_rep, dtype=torch.float32))
            mean_point = sum_point / member_counts.clamp_min(1.0)[:, None]
            mean_normal = torch.nn.functional.normalize(sum_normal, dim=-1, eps=1e-8)

            all_member_ids_a.append(distinct_reps.detach().cpu())
            all_chart_ids_a.append(torch.full((rep_count,), global_chart_id, dtype=torch.int64))
            all_fitted_points_a.append(mean_point.detach().cpu())
            all_normals_a.append(mean_normal.detach().cpu())

            region_votes = region_of_representative[distinct_reps]
            valid_votes = region_votes[region_votes >= 0]
            region_index = int(torch.mode(valid_votes).values.item()) if int(valid_votes.numel()) > 0 else -1

            uv_disp_np = uv_displacement.detach().cpu().numpy()
            chart_records.append({
                "chart_id": global_chart_id, "view_index": view_index, "camera_name": cameras[view_index].image_name,
                "component_id": component_id, "region_label": region_labels_list[region_index] if region_index >= 0 else None,
                "pixel_count": int(pixel_xyz.shape[0]), "representative_count": rep_count,
                "uv_displacement_median": float(np.median(uv_disp_np)), "uv_displacement_p95": float(np.percentile(uv_disp_np, 95)),
                "uv_displacement_max": float(uv_disp_np.max()) if uv_disp_np.shape[0] else 0.0,
                "camera_domain_hole_count": int(camera_domain_shape["hole_count"]),
                "camera_uv_hole_count": (1 if bool(camera_holes_np.any()) else 0),
                "fitted_domain_hole_count": (1 if bool(fitted_holes_np.any()) else 0),
                "camera_vs_fitted_support_iou": iou,
                "residual_a_median": float(residual_a.median().item()), "residual_a_p95": float(residual_a.quantile(0.95).item()), "residual_a_max": float(residual_a.max().item()),
                "residual_b_median": float(residual_b.median().item()), "residual_b_p95": float(residual_b.quantile(0.95).item()), "residual_b_max": float(residual_b.max().item()),
                "low_pass_dominated_fraction": low_pass_dominated_fraction,
                "within_chart_representative_spread_median": within_chart_spread_median,
            })
            global_chart_id += 1
        if view_index % 10 == 0:
            _progress(f"chart-fit view {view_index + 1}/{len(per_view_rep_remapped)} fitted={global_chart_id}")

    _progress(f"[chart accounting] fitted={global_chart_id}")

    # --- section 2: UV displacement summary ---
    uv_disp_medians = np.array([r["uv_displacement_median"] for r in chart_records], dtype=np.float64)
    uv_disp_maxes = np.array([r["uv_displacement_max"] for r in chart_records], dtype=np.float64)
    pixel_counts = np.array([r["pixel_count"] for r in chart_records], dtype=np.float64)
    has_camera_hole = np.array([r["camera_domain_hole_count"] > 0 for r in chart_records], dtype=bool)
    residual_a_medians = np.array([r["residual_a_median"] for r in chart_records], dtype=np.float64)

    uv_displacement_summary = {
        "per_chart_median_displacement_distribution": _distribution(uv_disp_medians),
        "per_chart_max_displacement_distribution": _distribution(uv_disp_maxes),
        "giant_chart_median_displacement": _distribution(uv_disp_medians[_bin_by_quantile(pixel_counts, 4) == 3]),
        "holey_chart_median_displacement": _distribution(uv_disp_medians[has_camera_hole]) if bool(has_camera_hole.any()) else None,
        "unholey_chart_median_displacement": _distribution(uv_disp_medians[~has_camera_hole]) if bool((~has_camera_hole).any()) else None,
        "correlation_uv_displacement_vs_residual_a": float(np.corrcoef(uv_disp_medians, residual_a_medians)[0, 1]) if uv_disp_medians.shape[0] > 2 else float("nan"),
    }
    _progress(f"[section 2] {uv_displacement_summary}")

    # --- section 3: camera-domain vs fitted-domain disagreement ---
    # Two DISTINCT camera-domain measurements, never conflated:
    #  (a) raster-native (WL113's own definition, native pixel resolution);
    #  (b) uv-binned at the SAME 24x24 resolution as the fitted-domain mask,
    #      the only fair basis for IoU / hole-presence agreement.
    camera_hole_counts_raster = np.array([r["camera_domain_hole_count"] for r in chart_records], dtype=np.float64)
    camera_hole_flags_uv = np.array([r["camera_uv_hole_count"] for r in chart_records], dtype=np.float64)
    fitted_hole_flags = np.array([r["fitted_domain_hole_count"] for r in chart_records], dtype=np.float64)
    ious = np.array([r["camera_vs_fitted_support_iou"] for r in chart_records], dtype=np.float64)
    domain_disagreement_summary = {
        "camera_domain_raster_native_hole_count_distribution": _distribution(camera_hole_counts_raster),
        "camera_domain_raster_native_has_hole_fraction": float((camera_hole_counts_raster > 0).mean()) if camera_hole_counts_raster.shape[0] else 0.0,
        "camera_domain_uv_binned_has_hole_fraction": float(camera_hole_flags_uv.mean()) if camera_hole_flags_uv.shape[0] else 0.0,
        "fitted_domain_uv_binned_has_hole_fraction": float(fitted_hole_flags.mean()) if fitted_hole_flags.shape[0] else 0.0,
        "camera_vs_fitted_iou_distribution_SAME_RESOLUTION": _distribution(ious),
        "fraction_charts_domains_disagree_on_hole_presence_SAME_RESOLUTION": float((camera_hole_flags_uv != fitted_hole_flags).mean()) if camera_hole_flags_uv.shape[0] else 0.0,
        "note": "raster-native and uv-binned camera-domain hole counts are DIFFERENT quantities (different resolution/domain) and must not be compared directly to each other; only the two uv-binned (SAME 24x24 resolution) masks are IoU/agreement-comparable",
    }
    _progress(f"[section 3] {domain_disagreement_summary}")

    # --- section 4: fixed-UV vs corrected-UV residual ---
    residual_b_medians = np.array([r["residual_b_median"] for r in chart_records], dtype=np.float64)
    residual_a_p95 = np.array([r["residual_a_p95"] for r in chart_records], dtype=np.float64)
    residual_b_p95 = np.array([r["residual_b_p95"] for r in chart_records], dtype=np.float64)
    residual_a_max = np.array([r["residual_a_max"] for r in chart_records], dtype=np.float64)
    residual_b_max = np.array([r["residual_b_max"] for r in chart_records], dtype=np.float64)
    fixed_vs_corrected_summary = {
        "arm_a_current_residual_median": _distribution(residual_a_medians), "arm_b_fixed_uv_residual_median": _distribution(residual_b_medians),
        "arm_a_current_residual_p95": _distribution(residual_a_p95), "arm_b_fixed_uv_residual_p95": _distribution(residual_b_p95),
        "arm_a_current_residual_max": _distribution(residual_a_max), "arm_b_fixed_uv_residual_max": _distribution(residual_b_max),
    }
    _progress(f"[section 4] {fixed_vs_corrected_summary}")

    # --- section 7: signed vs sign-invariant normal disagreement ---
    signed_pos, signed_normal, sign_invariant_normal = [], [], []
    if all_member_ids_a:
        cat_member = torch.cat(all_member_ids_a)
        cat_points = torch.cat(all_fitted_points_a)
        cat_normals = torch.cat(all_normals_a)
        order = torch.argsort(cat_member)
        sm, sp, sn = cat_member[order], cat_points[order], cat_normals[order]
        same_next = sm[:-1] == sm[1:]
        if bool(same_next.any()):
            diff = (sp[1:] - sp[:-1])[same_next]
            signed_pos = diff.norm(dim=-1).numpy()
            cos = (sn[1:] * sn[:-1]).sum(dim=-1)[same_next].clamp(-1.0, 1.0)
            signed_normal = torch.rad2deg(torch.acos(cos)).numpy()
            sign_invariant_normal = sign_invariant_normal_discrepancy_degrees(sn[1:], sn[:-1])[same_next].numpy()
    normal_comparison_summary = {
        "overlap_position_discrepancy": _distribution(np.asarray(signed_pos)),
        "signed_normal_discrepancy_degrees": _distribution(np.asarray(signed_normal)),
        "sign_invariant_normal_discrepancy_degrees": _distribution(np.asarray(sign_invariant_normal)),
        "fraction_of_signed_disagreement_explained_by_sign_flip": (
            float(np.mean(np.asarray(signed_normal) > 90.0)) if len(signed_normal) else 0.0
        ),
    }
    _progress(f"[section 7] {normal_comparison_summary}")

    # --- section 8: representative finite-support spread vs cross-chart displacement ---
    spreads = np.array([r["within_chart_representative_spread_median"] for r in chart_records if r["within_chart_representative_spread_median"] > 0], dtype=np.float64)
    position_correspondence_summary = {
        "within_chart_representative_footprint_spread_distribution": _distribution(spreads),
        "cross_chart_mean_displacement_distribution": _distribution(np.asarray(signed_pos)),
        "footprint_spread_median_vs_cross_chart_displacement_median_ratio": (
            float(np.median(spreads) / max(np.median(np.asarray(signed_pos)), 1e-12)) if spreads.shape[0] and len(signed_pos) else None
        ),
    }
    _progress(f"[section 8] {position_correspondence_summary}")

    # --- section 5/D: low-pass provenance summary + top outliers ---
    low_pass_fractions = np.array([r["low_pass_dominated_fraction"] for r in chart_records if not np.isnan(r["low_pass_dominated_fraction"])], dtype=np.float64)
    top_residual_charts = sorted(chart_records, key=lambda r: r["residual_a_max"], reverse=True)[:10]
    low_pass_provenance_summary = {
        "chart_mean_low_pass_dominated_fraction_distribution": _distribution(low_pass_fractions),
        "top10_worst_residual_charts_low_pass_fraction": [
            {"chart_id": r["chart_id"], "region_label": r["region_label"], "residual_a_max": r["residual_a_max"], "low_pass_dominated_fraction": r["low_pass_dominated_fraction"], "pixel_count": r["pixel_count"]}
            for r in top_residual_charts
        ],
    }
    _progress(f"[section 5] {low_pass_provenance_summary}")

    # --- region breakdown ---
    region_results = {}
    for label in ANCHOR_FRACTIONS:
        region_charts = [r for r in chart_records if r["region_label"] == label]
        if not region_charts:
            region_results[label] = None
            continue
        region_results[label] = {
            "chart_count": len(region_charts),
            "uv_displacement_median": _distribution(np.array([r["uv_displacement_median"] for r in region_charts])),
            "residual_a_median": _distribution(np.array([r["residual_a_median"] for r in region_charts])),
            "residual_b_median": _distribution(np.array([r["residual_b_median"] for r in region_charts])),
            "camera_domain_has_hole_fraction": float(np.mean([r["camera_domain_hole_count"] > 0 for r in region_charts])),
            "low_pass_dominated_fraction_median": float(np.median([r["low_pass_dominated_fraction"] for r in region_charts if not np.isnan(r["low_pass_dominated_fraction"])])) if any(not np.isnan(r["low_pass_dominated_fraction"]) for r in region_charts) else None,
        }
    _progress(f"[region results] {json.dumps({k: v for k, v in region_results.items()}, indent=2)[:2000]}")

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 118",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "camera_meta": camera_meta,
        "nurbs_config_FOR_CONTROL_ONLY": {"resolution_u": RESOLUTION_U, "resolution_v": RESOLUTION_V, "degree_u": DEGREE_U, "degree_v": DEGREE_V},
        "wl107_109_replay_consistency_check": wl107_replay_stats,
        "accounting": {"total_trained_surfels": total_model_count, "median_surface_representatives": representative_count, "fitted_chart_count": global_chart_id},
        "synthetic_equal_count_contracts": synthetic_results,
        "half_pixel_unprojection_displacement": half_pixel_displacement_summary,
        "uv_camera_vs_footpoint_displacement": uv_displacement_summary,
        "camera_vs_fitted_domain_disagreement": domain_disagreement_summary,
        "fixed_uv_vs_corrected_uv_residual": fixed_vs_corrected_summary,
        "normal_signed_vs_sign_invariant": normal_comparison_summary,
        "representative_position_correspondence": position_correspondence_summary,
        "low_pass_provenance": low_pass_provenance_summary,
        "region_results_WORKING_INTERPRETATION_ONLY": region_results,
        "runtime_seconds": {"total": time.time() - started},
    }

    # --- colors / exports ---
    original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]

    first_chart_of_member = torch.full((count,), -1, dtype=torch.int64, device=device)
    if all_member_ids_a:
        cat_member = torch.cat(all_member_ids_a).to(device)
        cat_chart = torch.cat(all_chart_ids_a).to(device)
        order2 = torch.argsort(cat_member * (global_chart_id + 1) + cat_chart)
        sm, sc = cat_member[order2], cat_chart[order2]
        first_mask = torch.ones_like(sm, dtype=torch.bool)
        first_mask[1:] = sm[1:] != sm[:-1]
        first_chart_of_member[sm[first_mask]] = sc[first_mask]
    has_chart = first_chart_of_member >= 0

    def _chart_scalar_color(field: str, lo_high) -> torch.Tensor:
        values = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
        for r in chart_records:
            v = r[field]
            values[r["chart_id"]] = 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        vmax = values.clamp_min(1e-9).max()
        colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        colors[has_chart] = _ramp((values / vmax)[first_chart_of_member[has_chart]], *lo_high)
        return colors

    uv_disp_colors = _chart_scalar_color("uv_displacement_median", ((0.2, 0.95, 0.3), (1.0, 0.15, 0.05)))
    domain_disagree_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    disagree_flag = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
    for r in chart_records:
        disagree_flag[r["chart_id"]] = 1.0 if (r["camera_domain_hole_count"] > 0) != (r["fitted_domain_hole_count"] > 0) else 0.0
    domain_disagree_colors[has_chart] = _ramp(disagree_flag[first_chart_of_member[has_chart]], (0.2, 0.95, 0.3), (1.0, 0.5, 0.0))

    residual_a_colors = _chart_scalar_color("residual_a_median", ((0.2, 0.95, 0.3), (1.0, 0.85, 0.0)))
    residual_b_colors = _chart_scalar_color("residual_b_median", ((0.2, 0.95, 0.3), (1.0, 0.85, 0.0)))
    low_pass_colors = _chart_scalar_color("low_pass_dominated_fraction", ((0.15, 0.4, 1.0), (1.0, 0.15, 0.05)))

    half_pixel_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    half_pixel_colors[ever_representative] = torch.tensor((0.6, 0.6, 0.6), dtype=torch.float32, device=device)

    normal_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    normal_colors[has_chart] = torch.tensor((0.3, 0.3, 0.9), dtype=torch.float32, device=device)

    views = {
        VIEW_ORIGINAL_SCENE: original_f_dc,
        VIEW_UV_DISPLACEMENT: _rgb_to_f_dc(uv_disp_colors),
        VIEW_DOMAIN_DISAGREEMENT: _rgb_to_f_dc(domain_disagree_colors),
        VIEW_FIXED_UV_RESIDUAL: _rgb_to_f_dc(residual_b_colors),
        VIEW_CORRECTED_UV_RESIDUAL: _rgb_to_f_dc(residual_a_colors),
        VIEW_NORMAL_SIGN_EFFECT: _rgb_to_f_dc(normal_colors),
        VIEW_LOW_PASS_PROVENANCE: _rgb_to_f_dc(low_pass_colors),
        VIEW_HALF_PIXEL_DISPLACEMENT: _rgb_to_f_dc(half_pixel_colors),
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
    report_path = output_root / "visible_nurbs_evidence_contract_closure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
