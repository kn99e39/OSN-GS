"""Worklog 117 -- Holey-Chart Fitting-Coupling Attribution.

Worklog 116 established that the existing `uv_support_mask` mechanism solves
unsupported-domain MATERIALIZATION (never touches the actual LSQ fit) but
does not by itself prove or disprove whether irregular/holey UV support
independently damages the existing global tensor-product fit (B2), as
opposed to merely correlating with chart scale/complexity (WL113's raw
hole-vs-no-hole comparison could not distinguish these).

This batch is a bounded ATTRIBUTION batch, not a new chart-construction
batch. It freezes exactly: WL107/109 visible topology, WL112 renderer-native
per-pixel surface geometry, WL112's one-camera-blob fitting baseline, and
the fixed 8x4/degree-2 NURBS model (FOR CONTROL ONLY -- not claimed
canonical). It does NOT reintroduce >=32 as an intrinsic NURBS requirement,
full-rank closure, or Worklog 114's local extraction. It does NOT implement
coupled fitting.

Central question: after controlling for chart scale (pixel count,
representative count), does proximity to unsupported/hole UV regions
independently predict fitting residual, WITHIN the same chart? Answered via:

  (1) B1 verification: uv_support_mask assignment provably does not alter
      the already-computed control grid or fitted output (bitwise check,
      not just code inspection).
  (2) B2 within-chart attribution: residual vs. distance-to-nearest-
      unsupported-cell and distance-to-nearest-hole-cell, derived from the
      existing `TorchOSNGSPipeline._uv_occupancy_mask` semantics applied to
      each WL112 chart's own fitted UV distribution.
  (3) Matched (scale-stratified) hole/no-hole comparison.
  (4) Giant-chart-specific spatial attribution (patio/table).
  (5) Synthetic full-vs-hole ground-truth controls (planar and curved),
      using the unmodified fitter.
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
from chart_representation_contract_diagnostic import _bin_by_quantile, _distribution  # noqa: E402 -- WL113, frozen, reused read-only
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline  # noqa: E402 -- existing OSN-GS occupancy-mask semantics, reused read-only
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
MIN_PIXEL_SAMPLES = RESOLUTION_U * RESOLUTION_V  # WL112's OWN gate, preserved for control fidelity only

# Existing OSN-GS occupancy-mask defaults (torch_pipeline.py's own config
# defaults, reused verbatim -- not new numbers invented for this batch).
TRIM_RESOLUTION = 24
TRIM_DILATION = 1

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_UNTRIMMED = "WL112_UNTRIMMED_SURFACE"
VIEW_TRIMMED = "WL112_TRIMMED_SUPPORT"
VIEW_UNSUPPORTED_REMOVED = "UNSUPPORTED_DOMAIN_REMOVED"
VIEW_DIST_UNSUPPORTED = "RESIDUAL_VS_DIST_UNSUPPORTED"
VIEW_DIST_HOLE = "RESIDUAL_VS_DIST_HOLE"
VIEW_GIANT_CHARTS = "GIANT_CHART_ATTRIBUTION"

ANCHOR_FRACTIONS = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[holey-chart-attribution] {message}", flush=True)


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
# Pure-logic helpers (module-level, testable): mask geometry / distance
# accounting, built ONLY from the existing OSN-GS occupancy-mask semantics.
# --------------------------------------------------------------------------

def hole_and_edge_masks(mask_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the unsupported region of an occupancy mask into:

    - enclosed HOLES (unsupported cells fully surrounded by supported cells,
      via `scipy.ndimage.binary_fill_holes`);
    - EDGE-CONNECTED unsupported cells (everything else unsupported -- i.e.
      connected to the [0,1]^2 domain's own border, such as unused corners
      of a rectangular chart whose data does not fill it).

    These are directive-required, distinct diagnostic categories: most of a
    typical chart's "unsupported" area is edge-connected (e.g. corners), not
    a genuine interior hole.
    """

    from scipy.ndimage import binary_fill_holes

    filled = binary_fill_holes(mask_np)
    holes = filled & ~mask_np
    edge_unsupported = (~mask_np) & (~holes)
    return holes, edge_unsupported


def sample_uv_to_cell(uv: Any, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin `(N, 2)` UV samples into `(resolution, resolution)` grid cells,
    using the IDENTICAL convention `TorchOSNGSPipeline._uv_occupancy_mask`
    uses to build the mask in the first place (so cell lookups agree with
    mask construction by definition, not by coincidence)."""

    uv_np = uv.detach().cpu().numpy() if hasattr(uv, "detach") else np.asarray(uv)
    cell_u = np.clip((uv_np[:, 0] * resolution).astype(np.int64), 0, resolution - 1)
    cell_v = np.clip((uv_np[:, 1] * resolution).astype(np.int64), 0, resolution - 1)
    return cell_u, cell_v


def distance_to_unsupported_grid(mask_np: np.ndarray) -> np.ndarray:
    """`(R, R)` grid: for every SUPPORTED cell, its distance (in grid cells)
    to the nearest UNSUPPORTED cell. Values at unsupported cells are 0 by
    `distance_transform_edt` convention and are never queried at those
    positions by this module (only observed/supported samples are scored)."""

    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(mask_np)


def distance_to_hole_grid(holes_np: np.ndarray) -> np.ndarray | None:
    """`(R, R)` grid: distance (in grid cells) to the nearest enclosed-HOLE
    cell specifically (as opposed to any unsupported cell). `None` when the
    chart has no enclosed holes at all."""

    from scipy.ndimage import distance_transform_edt

    if not bool(holes_np.any()):
        return None
    return distance_transform_edt(~holes_np)


def within_chart_distance_correlation(residual: np.ndarray, distance: np.ndarray) -> float:
    """Pearson correlation between per-sample residual and per-sample
    distance-to-unsupported/hole, within one chart. Negative correlation is
    the B2-consistent direction (residual falls as distance from the
    unsupported boundary grows). Returns `nan` if degenerate (constant
    distance or fewer than 3 samples)."""

    if residual.shape[0] < 3 or float(np.std(distance)) < 1e-9:
        return float("nan")
    matrix = np.corrcoef(residual, distance)
    return float(matrix[0, 1])


def near_far_median_split(residual: np.ndarray, distance: np.ndarray) -> tuple[float, float]:
    """Median residual in this chart's OWN nearest tercile of distance vs.
    its OWN farthest tercile -- a within-chart, scale-free near/far
    comparison (no cross-chart threshold)."""

    if residual.shape[0] < 6:
        return float("nan"), float("nan")
    order = np.argsort(distance)
    third = max(1, residual.shape[0] // 3)
    near = residual[order[:third]]
    far = residual[order[-third:]]
    return float(np.median(near)), float(np.median(far))


def _make_planar_grid(rows: int, cols: int, hole: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic contract A/B fixture: a flat rectangular observation grid,
    optionally with a central rectangular observation hole cut out. Returns
    `(points, uv)` -- ground-truth UV is exact (no PCA/foot-point ambiguity
    is introduced as a confound)."""

    ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
    u = ii / max(rows - 1, 1)
    v = jj / max(cols - 1, 1)
    keep = torch.ones_like(u, dtype=torch.bool)
    if hole:
        keep = ~((u > 0.35) & (u < 0.65) & (v > 0.35) & (v < 0.65))
    u_flat, v_flat, keep_flat = u.reshape(-1), v.reshape(-1), keep.reshape(-1)
    x = u_flat * 2.0
    y = v_flat * 1.0
    z = torch.zeros_like(x)
    points = torch.stack([x, y, z], dim=-1)[keep_flat]
    uv = torch.stack([u_flat, v_flat], dim=-1)[keep_flat]
    return points, uv


def _make_curved_grid(rows: int, cols: int, hole: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic contract C/D fixture: a smoothly curved observation grid
    (single bump), optionally with the same central observation hole."""

    ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
    u = ii / max(rows - 1, 1)
    v = jj / max(cols - 1, 1)
    keep = torch.ones_like(u, dtype=torch.bool)
    if hole:
        keep = ~((u > 0.35) & (u < 0.65) & (v > 0.35) & (v < 0.65))
    u_flat, v_flat, keep_flat = u.reshape(-1), v.reshape(-1), keep.reshape(-1)
    x = u_flat * 2.0
    y = v_flat * 1.0
    z = 0.4 * torch.sin(torch.pi * u_flat) * torch.sin(torch.pi * v_flat)
    points = torch.stack([x, y, z], dim=-1)[keep_flat]
    uv = torch.stack([u_flat, v_flat], dim=-1)[keep_flat]
    return points, uv


def run_synthetic_contracts() -> dict[str, Any]:
    """Directive section 6: A/B/C/D full-vs-hole ground-truth controls,
    using the unmodified fitter and the SAME fixed 8x4/degree-2 config.
    Compares residual on the RETAINED points only (the hole variant's own
    observed points, a strict subset of the full variant's), so the
    comparison isolates whether removing the hole region's data damages the
    fit on points the model still has to explain -- not merely "hole variant
    has fewer points so of course its aggregate differs"."""

    results: dict[str, Any] = {}
    for label, maker in (("planar", _make_planar_grid), ("curved", _make_curved_grid)):
        rows, cols = 24, 24
        full_points, full_uv = maker(rows, cols, hole=False)
        hole_points, hole_uv = maker(rows, cols, hole=True)

        with torch.no_grad():
            full_surface, full_fit_uv = fit_torch_visible_surface_lsq(
                full_points, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=full_uv,
                correction_rounds=2, projection_iterations=3,
            )
            hole_surface, hole_fit_uv = fit_torch_visible_surface_lsq(
                hole_points, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=hole_uv,
                correction_rounds=2, projection_iterations=3,
            )
            full_fitted = full_surface.evaluate(full_fit_uv)
            hole_fitted = hole_surface.evaluate(hole_fit_uv)
            full_residual_on_own_points = (full_fitted - full_points).norm(dim=-1)
            hole_residual_on_own_points = (hole_fitted - hole_points).norm(dim=-1)

            # Apples-to-apples: evaluate the FULL-fit surface's own residual
            # restricted to exactly the retained (non-hole) points, so both
            # numbers describe the identical point set under the identical
            # underlying geometry -- only the fitting data differs. Match by
            # UV (exact, since both grids share the identical (u, v) lattice
            # before any hole is cut).
            full_uv_np = full_uv.numpy()
            hole_uv_np = hole_uv.numpy()
            full_uv_keys = {(round(float(u_), 6), round(float(v_), 6)): i for i, (u_, v_) in enumerate(full_uv_np)}
            retained_indices = [full_uv_keys[(round(float(u_), 6), round(float(v_), 6))] for u_, v_ in hole_uv_np]
            retained_indices_t = torch.tensor(retained_indices, dtype=torch.int64)

            full_residual_on_retained = full_residual_on_own_points[retained_indices_t]

        results[label] = {
            "full_point_count": int(full_points.shape[0]),
            "hole_point_count": int(hole_points.shape[0]),
            "full_fit_residual_on_retained_points": _distribution(full_residual_on_retained.numpy()),
            "hole_fit_residual_on_retained_points": _distribution(hole_residual_on_own_points.numpy()),
            "full_fit_residual_all_points": _distribution(full_residual_on_own_points.numpy()),
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

    _progress("[synthetic contracts] running A/B/C/D full-vs-hole controls (independent of real-scene data)")
    synthetic_results = run_synthetic_contracts()
    _progress(f"[synthetic contracts] {json.dumps(synthetic_results, indent=2)[:2000]}")

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

    _progress("[full-scene sweep] representative_id + median_depth unprojection per view")
    per_view_rep_remapped: list[torch.Tensor] = []
    per_view_world_points: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=device)

    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        represented_ids = torch.unique(rep_full[valid])
        ever_representative_full[represented_ids] = True

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

    # --- main loop: WL112-identical chart construction + fitting, PLUS
    # post-fit-only support-mask / distance-to-unsupported-or-hole diagnostics ---
    chart_records: list[dict[str, Any]] = []
    materialization_violations = 0
    global_chart_id = 0
    max_charts = int(arguments.max_charts) if int(arguments.max_charts) > 0 else None
    stop = False

    all_member_ids: list[torch.Tensor] = []
    all_chart_ids: list[torch.Tensor] = []

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
        mask_valid = valid_pixel_chart_mask(vs, MIN_PIXEL_SAMPLES)
        valid_blob_ids = torch.nonzero(mask_valid, as_tuple=False).reshape(-1).tolist()

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
                fitted_before = surface.evaluate(uv)
                residual = (fitted_before - pixel_xyz).norm(dim=-1)
                control_grid_before = surface.control_grid.detach().clone()

                # --- B1: assign the existing occupancy-mask semantics, verify
                # it changes NOTHING about the already-computed fit. ---
                mask = TorchOSNGSPipeline._uv_occupancy_mask(uv.detach(), TRIM_RESOLUTION, TRIM_DILATION)
                surface.uv_support_mask = mask
                control_grid_after = surface.control_grid.detach()
                fitted_after = surface.evaluate(uv)
                materialization_only = bool(torch.equal(control_grid_before, control_grid_after)) and bool(torch.equal(fitted_before, fitted_after))
                if not materialization_only:
                    materialization_violations += 1

            mask_np = mask.cpu().numpy()
            holes_np, edge_unsupported_np = hole_and_edge_masks(mask_np)
            hole_count = 0
            if bool(holes_np.any()):
                from scipy.ndimage import label as ndi_label
                _hole_labels, hole_count = ndi_label(holes_np)

            dist_unsupported_grid_np = distance_to_unsupported_grid(mask_np)
            dist_hole_grid_np = distance_to_hole_grid(holes_np)

            cell_u, cell_v = sample_uv_to_cell(uv, TRIM_RESOLUTION)
            residual_np = residual.detach().cpu().numpy()
            dist_unsupported_per_sample = dist_unsupported_grid_np[cell_u, cell_v] / float(TRIM_RESOLUTION)
            corr_unsupported = within_chart_distance_correlation(residual_np, dist_unsupported_per_sample)
            near_unsupported, far_unsupported = near_far_median_split(residual_np, dist_unsupported_per_sample)

            corr_hole = float("nan")
            near_hole, far_hole = float("nan"), float("nan")
            if dist_hole_grid_np is not None:
                dist_hole_per_sample = dist_hole_grid_np[cell_u, cell_v] / float(TRIM_RESOLUTION)
                corr_hole = within_chart_distance_correlation(residual_np, dist_hole_per_sample)
                near_hole, far_hole = near_far_median_split(residual_np, dist_hole_per_sample)

            distinct_reps = torch.unique(pixel_rep_ids)
            all_member_ids.append(distinct_reps.detach().cpu())
            all_chart_ids.append(torch.full((int(distinct_reps.shape[0]),), global_chart_id, dtype=torch.int64))

            region_votes = region_of_representative[distinct_reps]
            valid_votes = region_votes[region_votes >= 0]
            region_index = int(torch.mode(valid_votes).values.item()) if int(valid_votes.numel()) > 0 else -1

            chart_records.append({
                "chart_id": global_chart_id, "view_index": view_index,
                "camera_name": cameras[view_index].image_name, "component_id": component_id,
                "pixel_count": int(pixel_xyz.shape[0]), "representative_count": int(distinct_reps.shape[0]),
                "has_holes": bool(holes_np.any()), "hole_count": int(hole_count),
                "occupied_cell_count": int(mask_np.sum()),
                "residual_median": float(np.median(residual_np)), "residual_p95": float(np.percentile(residual_np, 95)),
                "residual_max": float(residual_np.max()) if residual_np.shape[0] else 0.0,
                "materialization_only_verified": materialization_only,
                "corr_residual_vs_dist_unsupported": corr_unsupported,
                "residual_median_near_unsupported_boundary": near_unsupported,
                "residual_median_far_from_unsupported_boundary": far_unsupported,
                "corr_residual_vs_dist_hole": corr_hole,
                "residual_median_near_hole_boundary": near_hole,
                "residual_median_far_from_hole_boundary": far_hole,
                "region_label": region_labels_list[region_index] if region_index >= 0 else None,
            })
            global_chart_id += 1
        if view_index % 10 == 0:
            _progress(f"chart-fit view {view_index + 1}/{len(per_view_rep_remapped)} fitted={global_chart_id}")

    _progress(f"[chart accounting] fitted={global_chart_id} materialization_violations={materialization_violations}")

    # --- section 4: matched (scale-stratified) hole/no-hole comparison ---
    pixel_counts = np.array([r["pixel_count"] for r in chart_records], dtype=np.float64)
    has_holes = np.array([r["has_holes"] for r in chart_records], dtype=bool)
    residual_medians = np.array([r["residual_median"] for r in chart_records], dtype=np.float64)
    rep_counts = np.array([r["representative_count"] for r in chart_records], dtype=np.float64)

    def _stratified_hole_comparison(scale_values: np.ndarray, label: str) -> dict[str, Any]:
        bins = _bin_by_quantile(scale_values, n_bins=4)
        out: dict[str, Any] = {}
        for b in sorted(set(bins.tolist())):
            in_bin = bins == b
            holed = in_bin & has_holes
            unholed = in_bin & ~has_holes
            out[f"{label}_bin_{b}"] = {
                "scale_range": [float(scale_values[in_bin].min()), float(scale_values[in_bin].max())] if bool(in_bin.any()) else None,
                "holed_chart_count": int(holed.sum()), "unholed_chart_count": int(unholed.sum()),
                "holed_residual_median": _distribution(residual_medians[holed]),
                "unholed_residual_median": _distribution(residual_medians[unholed]),
                "ratio_holed_over_unholed_median": (
                    float(np.median(residual_medians[holed]) / max(np.median(residual_medians[unholed]), 1e-12))
                    if bool(holed.any()) and bool(unholed.any()) else None
                ),
            }
        return out

    matched_analysis = {
        "unweighted_overall_ratio_holed_over_unholed": (
            float(np.median(residual_medians[has_holes]) / max(np.median(residual_medians[~has_holes]), 1e-12))
            if bool(has_holes.any()) and bool((~has_holes).any()) else None
        ),
        "by_pixel_count_quantile": _stratified_hole_comparison(pixel_counts, "pixel_count"),
        "by_representative_count_quantile": _stratified_hole_comparison(rep_counts, "rep_count"),
    }
    _progress(f"[matched analysis] overall unweighted ratio = {matched_analysis['unweighted_overall_ratio_holed_over_unholed']}")

    # --- section 3 headline: pooled within-chart correlation distributions ---
    corr_unsupported_all = np.array([r["corr_residual_vs_dist_unsupported"] for r in chart_records if not np.isnan(r["corr_residual_vs_dist_unsupported"])], dtype=np.float64)
    corr_hole_all = np.array([r["corr_residual_vs_dist_hole"] for r in chart_records if not np.isnan(r["corr_residual_vs_dist_hole"])], dtype=np.float64)
    within_chart_summary = {
        "corr_residual_vs_dist_unsupported": _distribution(corr_unsupported_all),
        "fraction_charts_with_negative_corr_unsupported": float((corr_unsupported_all < 0).mean()) if corr_unsupported_all.shape[0] else 0.0,
        "corr_residual_vs_dist_hole": _distribution(corr_hole_all),
        "fraction_charts_with_negative_corr_hole": float((corr_hole_all < 0).mean()) if corr_hole_all.shape[0] else 0.0,
        "near_vs_far_unsupported_ratio_distribution": _distribution(np.array([
            r["residual_median_near_unsupported_boundary"] / max(r["residual_median_far_from_unsupported_boundary"], 1e-12)
            for r in chart_records if not (np.isnan(r["residual_median_near_unsupported_boundary"]) or np.isnan(r["residual_median_far_from_unsupported_boundary"]))
        ], dtype=np.float64)),
    }
    _progress(f"[within-chart summary] {within_chart_summary}")

    # --- section 5: giant-chart attribution (top 10 by pixel count) ---
    giant_charts = sorted(chart_records, key=lambda r: r["pixel_count"], reverse=True)[:10]
    giant_chart_attribution = [
        {k: r[k] for k in (
            "chart_id", "view_index", "component_id", "region_label", "pixel_count", "has_holes", "hole_count",
            "residual_median", "residual_max", "corr_residual_vs_dist_unsupported",
            "residual_median_near_unsupported_boundary", "residual_median_far_from_unsupported_boundary",
        )}
        for r in giant_charts
    ]
    _progress(f"[giant chart attribution] top chart: {giant_chart_attribution[0] if giant_chart_attribution else None}")

    # --- decision ---
    b2_signals = {
        "median_corr_unsupported_negative": bool(within_chart_summary["corr_residual_vs_dist_unsupported"]["median"] < -0.05),
        "majority_charts_negative_corr": bool(within_chart_summary["fraction_charts_with_negative_corr_unsupported"] > 0.55),
        "matched_ratio_survives": None,
        "giant_chart_far_from_boundary_still_high": None,
    }
    # matched_ratio_survives: true if every pixel-count-stratified bin still shows holed > unholed median
    stratum_ratios = [v["ratio_holed_over_unholed_median"] for v in matched_analysis["by_pixel_count_quantile"].values() if v["ratio_holed_over_unholed_median"] is not None]
    b2_signals["matched_ratio_survives"] = bool(stratum_ratios and all(r > 1.1 for r in stratum_ratios))
    # giant-chart check: is far-from-boundary residual still comparable to near?
    giant_far_high = []
    for r in giant_charts:
        if not (np.isnan(r["residual_median_near_unsupported_boundary"]) or np.isnan(r["residual_median_far_from_unsupported_boundary"])):
            giant_far_high.append(r["residual_median_far_from_unsupported_boundary"] >= 0.5 * r["residual_median_near_unsupported_boundary"])
    b2_signals["giant_chart_far_from_boundary_still_high"] = bool(giant_far_high and (sum(giant_far_high) / len(giant_far_high)) > 0.5)

    supported_votes = sum(1 for v in (b2_signals["median_corr_unsupported_negative"], b2_signals["majority_charts_negative_corr"], b2_signals["matched_ratio_survives"]) if v)
    against_votes = 1 if b2_signals["giant_chart_far_from_boundary_still_high"] else 0
    if supported_votes >= 2 and against_votes == 0:
        decision = "B2_SUPPORTED"
    elif supported_votes <= 1 or against_votes >= 1:
        decision = "B2_NOT_SUPPORTED"
    else:
        decision = "MIXED_INCONCLUSIVE"
    _progress(f"[decision] {decision} signals={b2_signals}")

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 117",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "camera_meta": camera_meta,
        "nurbs_config_FOR_CONTROL_ONLY": {"resolution_u": RESOLUTION_U, "resolution_v": RESOLUTION_V, "degree_u": DEGREE_U, "degree_v": DEGREE_V, "min_pixel_samples_WL112_gate_not_intrinsic": MIN_PIXEL_SAMPLES},
        "trim_semantics": {"resolution": TRIM_RESOLUTION, "dilation": TRIM_DILATION, "source": "TorchOSNGSPipeline config defaults, reused verbatim"},
        "wl107_109_replay_consistency_check": wl107_replay_stats,
        "accounting": {"total_trained_surfels": total_model_count, "median_surface_representatives": representative_count, "fitted_chart_count": global_chart_id, "materialization_violations": materialization_violations},
        "synthetic_contracts": synthetic_results,
        "matched_hole_vs_noHole_analysis": matched_analysis,
        "within_chart_distance_correlation_summary": within_chart_summary,
        "giant_chart_attribution_top10": giant_chart_attribution,
        "b2_decision_signals": b2_signals,
        "decision": decision,
        "runtime_seconds": {"total": time.time() - started},
    }

    # --- colors / exports ---
    original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]

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

    untrimmed_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    untrimmed_colors[has_chart] = _hash_colors(first_chart_of_member[has_chart])

    hole_by_chart = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
    corr_by_chart = torch.zeros((global_chart_id,), dtype=torch.float32, device=device)
    for r in chart_records:
        hole_by_chart[r["chart_id"]] = 1.0 if r["has_holes"] else 0.0
        corr_by_chart[r["chart_id"]] = 0.0 if np.isnan(r["corr_residual_vs_dist_unsupported"]) else r["corr_residual_vs_dist_unsupported"]

    trimmed_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    trimmed_colors[has_chart] = _ramp(hole_by_chart[first_chart_of_member[has_chart]], (0.2, 0.95, 0.3), (1.0, 0.5, 0.0))

    dist_corr_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    corr_norm = (corr_by_chart + 1.0) / 2.0
    dist_corr_colors[has_chart] = _ramp(corr_norm[first_chart_of_member[has_chart]], (1.0, 0.15, 0.05), (0.15, 0.4, 1.0))

    giant_ids = {r["chart_id"] for r in giant_charts}
    giant_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    giant_colors[has_chart] = torch.tensor((0.15, 0.2, 0.25), dtype=torch.float32, device=device)
    if giant_ids and all_member_ids:
        cat_member_g = torch.cat(all_member_ids).to(device)
        cat_chart_g = torch.cat(all_chart_ids).to(device)
        giant_t = torch.tensor(sorted(giant_ids), dtype=torch.int64, device=device)
        is_giant_row = torch.isin(cat_chart_g, giant_t)
        giant_colors[cat_member_g[is_giant_row]] = torch.tensor((1.0, 0.0, 1.0), dtype=torch.float32, device=device)

    views = {
        VIEW_ORIGINAL_SCENE: original_f_dc,
        VIEW_UNTRIMMED: _rgb_to_f_dc(untrimmed_colors),
        VIEW_TRIMMED: _rgb_to_f_dc(trimmed_colors),
        VIEW_UNSUPPORTED_REMOVED: _rgb_to_f_dc(trimmed_colors),
        VIEW_DIST_UNSUPPORTED: _rgb_to_f_dc(dist_corr_colors),
        VIEW_DIST_HOLE: _rgb_to_f_dc(dist_corr_colors),
        VIEW_GIANT_CHARTS: _rgb_to_f_dc(giant_colors),
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
    report_path = output_root / "holey_chart_fitting_coupling_attribution_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
