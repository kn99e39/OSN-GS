"""Worklog 119 -- Visible-NURBS Geometry / UV Control Correction.

Worklog 118 is accepted with an important methodological correction: its
FIXED-UV A/B was not a true one-variable control. ARM A ran the normal
multi-round LSQ -> foot-point-UV-reprojection loop (2 solves), while ARM B
performed only ONE direct regularized solve -- an unequal solve count. Worse,
ARM A's residual was measured at foot-point-corrected UV while ARM B's was
measured at fixed camera UV, so the two numbers never measured the same
semantic quantity. Both problems are corrected here before any architecture
conclusion is drawn from the comparison.

This batch freezes WL107/109 topology, WL112 camera-blob membership, and the
fixed 8x4/degree-2 model (FOR CONTROL ONLY, exactly as WL112-118). It does not
implement adaptive capacity, domain-aware fitting, coupled fitting, new
charts, or topology changes. It only corrects:

  (2) the UV A/B control itself -- ARM B now performs the SAME NUMBER of LSQ
      solves as ARM A (2), with the sole difference being whether UV is
      reprojected between rounds;
  (3) evaluation semantics -- METRIC G (geometric point-to-surface error,
      evaluated via a closest-point foot-point projection computed ONLY for
      evaluation, on BOTH final surfaces) and METRIC C (camera-correspondence
      error, both final surfaces evaluated at the SAME original immutable
      camera UV) are reported SEPARATELY and never cross-compared;
  (5/6) renderer geometry provenance -- three geometry sources (G0 official
      depth unprojection, G1 rasterizer-pixel-center unprojection, G2 direct
      median-surfel local-plane intersection reconstructed from the trained
      surfel's own center/tangent frame/tangent scales and the renderer's own
      `s_u`/`s_v`) are compared, without assuming G2 is correct a priori;
  (7) low-pass events (`rho2d < rho3d`) are classified separately, never
      rejected outright;
  (8) pixel-level (not chart-mean) D-outlier attribution;
  (9/10) corrected wording for correspondence accounting and normal
      interpretation (comparable SCALE, not "percent explained"; remaining
      normal disagreement after sign correction is NOT automatically "true
      NURBS disagreement");
  (11) an optional boundary-connected contiguous-notch synthetic control,
      alongside WL118's enclosed-hole/dispersed-removal pair, same retained
      count.
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
from visible_nurbs_evidence_contract_closure import (  # noqa: E402 -- WL118, frozen; reused read-only
    depths_to_points_rasterizer_pixel_center,
    sign_invariant_normal_discrepancy_degrees,
    _grid_uv,
    _geometry_xyz,
)
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
    valid_pixel_chart_mask,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph
from osn_gs.surface.torch_nurbs import (
    _lsq_normal_system,
    _solve_control_grid_lsq,
    fit_torch_visible_surface,
    fit_torch_visible_surface_lsq,
    project_torch_points_to_nurbs,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel
from osn_gs.utils.torch_ops import require_torch

_ITERATION_DIR = "iteration_0000001"
_MIDDEPTH_OFFSET = 5
_UNCUT_RGB = (0.08, 0.09, 0.11)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

# Frozen exactly as WL112-118 (directive: control only, not canonical).
RESOLUTION_U = 8
RESOLUTION_V = 4
DEGREE_U = 2
DEGREE_V = 2
MIN_PIXEL_SAMPLES = RESOLUTION_U * RESOLUTION_V
TRIM_RESOLUTION = 24
TRIM_DILATION = 1
CORRECTION_ROUNDS = 2  # solve count for BOTH arms, held equal per this batch's correction
PROJECTION_ITERATIONS = 3
SMOOTHNESS_LAMBDA = 1e-4
TIKHONOV_LAMBDA = 1e-4
PIXEL_SAMPLE_STRIDE_TARGET = 200  # bounded deterministic per-chart pixel-level D-attribution sample

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_SUBSET_MEMBERSHIP = "CANONICAL_SUBSET_MEMBERSHIP"
VIEW_GEOMETRIC_ERROR_A = "METRIC_G_GEOMETRIC_ERROR_ARM_A"
VIEW_GEOMETRIC_ERROR_B = "METRIC_G_GEOMETRIC_ERROR_ARM_B"
VIEW_CAMERA_ERROR_A = "METRIC_C_CAMERA_CORRESPONDENCE_ARM_A"
VIEW_CAMERA_ERROR_B = "METRIC_C_CAMERA_CORRESPONDENCE_ARM_B"
VIEW_G0_G2_DISAGREEMENT = "G0_VS_G2_GEOMETRY_DISAGREEMENT"
VIEW_BRANCH_CLASSIFICATION = "RHO3D_VS_RHO2D_BRANCH"

ANCHOR_FRACTIONS = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[nurbs-geometry-uv-control] {message}", flush=True)


def _build_pixel_records_vectorized(
    chart_id: int,
    view_index: int,
    rows_np: np.ndarray,
    cols_np: np.ndarray,
    residual_np: np.ndarray,
    rep_np: np.ndarray,
    rho3d_np: np.ndarray,
    rho2d_np: np.ndarray,
    branch_np: np.ndarray,
    s_magnitude_np: np.ndarray,
    depth_np: np.ndarray,
    g0_g2_dist_np: np.ndarray,
    g2_finite_np: np.ndarray,
) -> list[dict[str, Any]]:
    """Build bounded pixel bookkeeping records after bulk host transfer.

    The arrays are already selected in the original row-major/sample order.
    Keeping the small record-construction loop here makes the transfer boundary
    explicit and gives the old scalar implementation a direct exact-equivalence
    target in the focused Worklog 119 tests.
    """

    branch_labels = ["rho3d" if b == 1 else ("rho2d" if b == 0 else "none") for b in branch_np]
    records: list[dict[str, Any]] = []
    for row, col, residual, rep_id, rho3d_v, rho2d_v, branch_label, s_mag, depth_v, g2_dist, g2_ok in zip(
        rows_np, cols_np, residual_np, rep_np, rho3d_np, rho2d_np, branch_labels, s_magnitude_np, depth_np, g0_g2_dist_np, g2_finite_np
    ):
        records.append({
            "chart_id": chart_id, "view_index": view_index, "row": int(row), "col": int(col),
            "residual_camera_correspondence_arm_a": float(residual),
            "representative_id_full": int(rep_id),
            "rho3d": float(rho3d_v), "rho2d": float(rho2d_v),
            "branch": branch_label,
            "s_magnitude": float(s_mag),
            "depth": float(depth_v),
            "g0_vs_g2_distance": float(g2_dist) if bool(g2_ok) else None,
            "region_label": None,
        })
    return records


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
# Section 2: corrected fixed-UV ARM B -- SAME solve count as ARM A (2), UV
# is NEVER reprojected between rounds. The only difference vs ARM A is
# whether `project_torch_points_to_nurbs` is called between solves.
# --------------------------------------------------------------------------

def fit_fixed_uv_equal_solves(
    points: torch.Tensor,
    uv_fixed: torch.Tensor,
    resolution_u: int = RESOLUTION_U,
    resolution_v: int = RESOLUTION_V,
    degree_u: int = DEGREE_U,
    degree_v: int = DEGREE_V,
    smoothness_lambda: float = SMOOTHNESS_LAMBDA,
    tikhonov_lambda: float = TIKHONOV_LAMBDA,
    correction_rounds: int = CORRECTION_ROUNDS,
    chunk_size: int = 4096,
):
    """ARM B (corrected): identical IDW seed and identical solve count to
    ARM A, but `uv_fixed` is passed to every `_solve_control_grid_lsq` call
    unchanged -- never refreshed via foot-point projection. Returns
    ``(surface, uv_fixed)`` -- the SAME `uv_fixed` tensor object, so a caller
    can assert (not merely believe) that ARM B never changes UV."""

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    uv_fixed = torch.as_tensor(uv_fixed, dtype=torch.float32, device=points.device)
    surface = fit_torch_visible_surface(
        points, resolution_u=resolution_u, resolution_v=resolution_v,
        chunk_size=chunk_size, degree_u=degree_u, degree_v=degree_v, initial_uv=uv_fixed,
    )
    if int(points.shape[0]) <= 1:
        return surface, uv_fixed
    with torch.no_grad():
        normal_system = _lsq_normal_system(
            points, uv_fixed, surface, smoothness_lambda, tikhonov_lambda, chunk_size, None
        )
        for _ in range(max(1, int(correction_rounds))):
            surface.control_grid = _solve_control_grid_lsq(
                points,
                uv_fixed,
                surface,
                smoothness_lambda,
                tikhonov_lambda,
                chunk_size,
                None,
                preassembled_normal_system=normal_system,
            )
    return surface, uv_fixed


# --------------------------------------------------------------------------
# Section 3: common evaluation metrics, applied identically to BOTH arms'
# FINAL surfaces. Neither function mutates `surface` (evaluation only).
# --------------------------------------------------------------------------

def geometric_point_to_surface_error(surface, points: torch.Tensor, chunk_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """METRIC G: closest-point (foot-point) UV computed FOR EVALUATION ONLY
    on the given final surface, then point-to-surface residual. Returns
    ``(residual, uv_eval)``. Does not write `surface.control_grid`."""

    uv_eval = project_torch_points_to_nurbs(points, surface, iterations=PROJECTION_ITERATIONS, chunk_size=chunk_size)
    residual = (surface.evaluate(uv_eval) - points).norm(dim=-1)
    return residual, uv_eval


def camera_correspondence_error(surface, points: torch.Tensor, uv_camera: torch.Tensor) -> torch.Tensor:
    """METRIC C: residual at the ORIGINAL, immutable camera UV -- never
    reprojected. Does not write `surface.control_grid`."""

    residual = (surface.evaluate(uv_camera) - points).norm(dim=-1)
    return residual


# --------------------------------------------------------------------------
# Section 5/6: direct median-surfel local-plane intersection (G2).
#
# `compute_transmat` (forward.cu) builds `splat2world` directly from the
# surfel's WORLD-space center `p_orig` and WORLD-space columns `L = R * S`
# (no view-matrix applied to L itself) -- `T = transpose(splat2world) *
# world2ndc * ndc2pix` maps a LOCAL homogeneous (s_u, s_v, 1) directly to
# pixel-homogeneous coordinates. The world-space point corresponding to the
# renderer's own local intersection `(s_u, s_v)` is therefore
#   world = center + s_u * scale_u * tangent_u + s_v * scale_v * tangent_v
# using EXACTLY the trained `t_u`/`t_v`/`s_u`/`s_v` fields
# `derive_surface_orientation_from_surfel` already reads off the model
# (`model.get_tangent_u`/`get_tangent_v`/`get_scaling`), not a re-derived
# covariance axis.
# --------------------------------------------------------------------------

def reconstruct_direct_surfel_intersection_world_point(
    positions_full: torch.Tensor,
    tangent_u_full: torch.Tensor,
    tangent_v_full: torch.Tensor,
    scale_u_full: torch.Tensor,
    scale_v_full: torch.Tensor,
    representative_id_full: torch.Tensor,
    s_u: torch.Tensor,
    s_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(world_point, valid)`` for a per-pixel (or per-sample) batch.

    ``representative_id_full`` is the FULL (unfiltered) model index, -1
    where the pixel has no median representative -- matching
    `render_with_pixel_representative`'s own convention. Invalid entries get
    an arbitrary (clamped) but VALID index so gather never errors; callers
    must mask by the returned ``valid``."""

    valid = representative_id_full >= 0
    idx = representative_id_full.clamp(min=0)
    center = positions_full[idx]
    tangent_u = tangent_u_full[idx]
    tangent_v = tangent_v_full[idx]
    scale_u = scale_u_full[idx]
    scale_v = scale_v_full[idx]
    world = center + (s_u * scale_u).unsqueeze(-1) * tangent_u + (s_v * scale_v).unsqueeze(-1) * tangent_v
    return world, valid


# --------------------------------------------------------------------------
# Section 7: rho3d/rho2d branch classification -- semantic, not a filter.
# --------------------------------------------------------------------------

def classify_median_event_branch(rho3d: torch.Tensor, rho2d: torch.Tensor) -> torch.Tensor:
    """Return an int8 tensor: 1 = "rho3d" (true 3D surfel-footprint branch,
    `rho3d <= rho2d`), 0 = "rho2d" (screen-space low-pass branch,
    `rho2d < rho3d`), -1 = no median event (`rho3d < 0`, matching the -1
    fill convention). Ties (`rho3d == rho2d`) resolve to the "rho3d" branch,
    matching the kernel's own `rho = min(rho3d, rho2d)` and `<=` comparison
    semantics at the acceptance test."""

    torch = require_torch()
    no_event = rho3d < 0
    is_rho3d = rho3d <= rho2d
    return torch.where(no_event, torch.full_like(rho3d, -1, dtype=torch.int8), is_rho3d.to(torch.int8))


# --------------------------------------------------------------------------
# Section 9 (corrected): equal-retained-count synthetic controls, now with
# an OPTIONAL boundary-connected contiguous notch alongside the enclosed
# hole and dispersed removal, same retained count throughout.
# --------------------------------------------------------------------------

def run_equal_count_synthetic_contracts_corrected() -> dict[str, Any]:
    results: dict[str, Any] = {}
    rows, cols = 24, 24
    for label, curved in (("planar", False), ("curved", True)):
        u, v = _grid_uv(rows, cols)
        xyz_full = _geometry_xyz(u, v, curved)
        u_flat, v_flat, xyz_flat = u.reshape(-1), v.reshape(-1), xyz_full.reshape(-1, 3)
        total = u_flat.shape[0]

        hole_keep = ~((u_flat > 0.35) & (u_flat < 0.65) & (v_flat > 0.35) & (v_flat < 0.65))
        removed_count = int((~hole_keep).sum())

        stride = max(1, total // max(removed_count, 1))
        dispersed_remove_idx = torch.arange(0, total, stride)[:removed_count]
        dispersed_keep = torch.ones((total,), dtype=torch.bool)
        dispersed_keep[dispersed_remove_idx] = False

        # Section 11: boundary-connected contiguous notch -- a full-width
        # contiguous block touching the u=0 edge, matched to the SAME
        # removed_count (rounded to whole rows; the actual count is
        # reported, not silently assumed equal).
        rows_needed = max(1, round(removed_count / cols))
        notch_mask_2d = torch.zeros((rows, cols), dtype=torch.bool)
        notch_mask_2d[:rows_needed, :] = True
        notch_keep = ~notch_mask_2d.reshape(-1)
        notch_removed_count = int((~notch_keep).sum())

        def _fit_and_score(keep_mask: torch.Tensor, fixed_uv: bool) -> dict[str, Any]:
            uv_k = torch.stack([u_flat[keep_mask], v_flat[keep_mask]], dim=-1)
            xyz_k = xyz_flat[keep_mask]
            with torch.no_grad():
                if fixed_uv:
                    surface, _ = fit_fixed_uv_equal_solves(xyz_k, uv_k, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V, degree_u=DEGREE_U, degree_v=DEGREE_V)
                    fitted = surface.evaluate(uv_k)
                else:
                    surface, fit_uv = fit_torch_visible_surface_lsq(
                        xyz_k, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V, degree_u=DEGREE_U, degree_v=DEGREE_V,
                        initial_uv=uv_k, correction_rounds=CORRECTION_ROUNDS, projection_iterations=PROJECTION_ITERATIONS,
                    )
                    fitted = surface.evaluate(fit_uv)
                residual = (fitted - xyz_k).norm(dim=-1)
            return {"retained_count": int(keep_mask.sum()), "residual": _distribution(residual.numpy())}

        results[label] = {
            "removed_count_in_B_and_C": removed_count,
            "notch_removed_count_in_D": notch_removed_count,
            "A_full_footpoint": _fit_and_score(torch.ones((total,), dtype=torch.bool), fixed_uv=False),
            "B_enclosed_hole_footpoint": _fit_and_score(hole_keep, fixed_uv=False),
            "C_dispersed_removal_footpoint": _fit_and_score(dispersed_keep, fixed_uv=False),
            "D_boundary_notch_footpoint": _fit_and_score(notch_keep, fixed_uv=False),
            "A_full_fixed_uv": _fit_and_score(torch.ones((total,), dtype=torch.bool), fixed_uv=True),
            "B_enclosed_hole_fixed_uv": _fit_and_score(hole_keep, fixed_uv=True),
            "C_dispersed_removal_fixed_uv": _fit_and_score(dispersed_keep, fixed_uv=True),
            "D_boundary_notch_fixed_uv": _fit_and_score(notch_keep, fixed_uv=True),
        }
    return results



def _write_performance_chart_corpus(
    output_path: Path,
    per_view_rep_remapped: list[torch.Tensor],
    per_view_world_points_g0: list[torch.Tensor],
    subset_ids: torch.Tensor,
    max_charts: int | None,
    source_metadata: dict[str, Any],
) -> None:
    """Emit an ordered WL119 chart corpus for the independent Performance Track.

    This opt-in branch does not execute or alter the serial scientific path.
    Membership, IDs, row-major pixel order, camera UV, and world points are
    retained so both performance arms consume one immutable input artifact.
    """

    charts: list[dict[str, Any]] = []
    stop = False
    for view_index, (rep_cpu, world_cpu) in enumerate(
        zip(per_view_rep_remapped, per_view_world_points_g0)
    ):
        if stop:
            break
        rep_gpu = rep_cpu.to(subset_ids.device)
        world_gpu = world_cpu.to(subset_ids.device)
        valid = rep_gpu >= 0
        component_map = torch.where(
            valid, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1)
        )
        samples = build_view_chart_pixel_samples(
            view_index, component_map, rep_gpu, world_gpu
        )
        if samples.blob_count == 0:
            continue
        valid_blob_ids = torch.nonzero(
            valid_pixel_chart_mask(samples, MIN_PIXEL_SAMPLES), as_tuple=False
        ).reshape(-1).tolist()
        order = samples.pixel_order_by_blob
        grouped_uv = samples.pixel_uv[order]
        grouped_xyz = samples.pixel_xyz[order]
        grouped_rep = samples.pixel_representative_id[order]
        grouped_row = samples.pixel_row[order].detach().cpu()
        grouped_col = samples.pixel_col[order].detach().cpu()
        offsets = samples.blob_pixel_offset.detach().cpu()
        components = samples.blob_component_id.detach().cpu()
        for local_blob in valid_blob_ids:
            if max_charts is not None and len(charts) >= max_charts:
                stop = True
                break
            start, end = int(offsets[local_blob]), int(offsets[local_blob + 1])
            charts.append({
                "chart_id": len(charts),
                "view_index": view_index,
                "local_blob_id": int(local_blob),
                "component_id": int(components[local_blob]),
                "pixel_count": end - start,
                "pixel_row": grouped_row[start:end].clone(),
                "pixel_col": grouped_col[start:end].clone(),
                "representative_id": grouped_rep[start:end].detach().cpu().clone(),
                "camera_uv": grouped_uv[start:end].detach().cpu().clone(),
                "world_points": grouped_xyz[start:end].detach().cpu().clone(),
            })
    lengths = [int(chart["pixel_count"]) for chart in charts]
    payload = {
        "schema": "wl119-performance-chart-corpus-v1",
        "source": source_metadata,
        "chart_count": len(charts),
        "point_count": sum(lengths),
        "chart_lengths": lengths,
        "charts": charts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    _progress(
        f"performance corpus -> {output_path} charts={len(charts)} "
        f"points={sum(lengths)}"
    )

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
    parser.add_argument(
        "--performance-corpus-out", type=Path, default=None,
        help="opt-in Performance Track corpus export; serial WL119 fitting is not run",
    )
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)

    _progress("[section 9/11] running corrected equal-count synthetic controls (+ boundary notch)")
    synthetic_results = run_equal_count_synthetic_contracts_corrected()
    _progress(f"[section 9/11] {json.dumps(synthetic_results, indent=2)[:1500]}")

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

    # Full-model tensors for G2 reconstruction -- representative_id is a FULL
    # model index (-1 fill), so these must NOT be visible-filtered.
    positions_full = model.get_xyz.detach()
    tangent_u_full = model.get_tangent_u.detach()
    tangent_v_full = model.get_tangent_v.detach()
    scaling_full = model.get_scaling.detach()
    scale_u_full = scaling_full[:, 0]
    scale_v_full = scaling_full[:, 1]

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
        graph = build_candidate_graph(
            orientation, config.local, retain_neighbor_index=True, progress=_progress
        )

    _progress("[full-scene sweep] representative_id + median_depth + rho3d/rho2d/s + G0/G1/G2 world points per view")
    per_view_rep_remapped: list[torch.Tensor] = []
    per_view_rep_full: list[torch.Tensor] = []
    per_view_world_points_g0: list[torch.Tensor] = []
    per_view_world_points_g1: list[torch.Tensor] = []
    per_view_world_points_g2: list[torch.Tensor] = []
    per_view_rho3d: list[torch.Tensor] = []
    per_view_rho2d: list[torch.Tensor] = []
    per_view_s_u: list[torch.Tensor] = []
    per_view_s_v: list[torch.Tensor] = []
    per_view_depth: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=device)

    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        represented_ids = torch.unique(rep_full[valid])
        ever_representative_full[represented_ids] = True

        depth_map = diag["out_others"][_MIDDEPTH_OFFSET]
        s_u_map = diag["median_s_u"]
        s_v_map = diag["median_s_v"]
        with torch.no_grad():
            world_g0 = depths_to_points(camera, depth_map.unsqueeze(0)).reshape(*depth_map.shape, 3)
            world_g1 = depths_to_points_rasterizer_pixel_center(camera, depth_map.unsqueeze(0)).reshape(*depth_map.shape, 3)
            world_g2, g2_valid = reconstruct_direct_surfel_intersection_world_point(
                positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
                rep_full, s_u_map, s_v_map,
            )
            world_g2 = torch.where(g2_valid.unsqueeze(-1), world_g2, torch.full_like(world_g2, float("nan")))

        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_rep_remapped.append(rep_remapped.detach().cpu())
        per_view_rep_full.append(rep_full.detach().cpu())
        per_view_world_points_g0.append(world_g0.detach().cpu())
        per_view_world_points_g1.append(world_g1.detach().cpu())
        per_view_world_points_g2.append(world_g2.detach().cpu())
        per_view_rho3d.append(diag["median_rho3d"].detach().cpu())
        per_view_rho2d.append(diag["median_rho2d"].detach().cpu())
        per_view_s_u.append(s_u_map.detach().cpu())
        per_view_s_v.append(s_v_map.detach().cpu())
        per_view_depth.append(depth_map.detach().cpu())
        del diag
        if index % 20 == 0:
            _progress(f"sweep view {index + 1}/{len(cameras)}")

    ever_representative = ever_representative_full[visible_selector]
    representative_count = int(ever_representative.sum().item())
    _progress(f"[accounting] median_surface_representatives={representative_count}")

    # --- section 6: G0 vs G1 vs G2 disagreement, split by rho3d/rho2d branch ---
    g0_g1_disp_rho3d, g0_g1_disp_rho2d = [], []
    g0_g2_disp_rho3d, g0_g2_disp_rho2d = [], []
    for view_index in range(len(cameras)):
        rho3d_v = per_view_rho3d[view_index]
        rho2d_v = per_view_rho2d[view_index]
        branch = classify_median_event_branch(rho3d_v, rho2d_v)
        g2_v = per_view_world_points_g2[view_index]
        g2_finite = torch.isfinite(g2_v).all(dim=-1)
        g0_v = per_view_world_points_g0[view_index]
        g1_v = per_view_world_points_g1[view_index]

        rho3d_mask = (branch == 1) & g2_finite
        rho2d_mask = (branch == 0) & g2_finite

        def _bounded_indices(mask: torch.Tensor, cap: int = 3000) -> torch.Tensor:
            idx = torch.nonzero(mask, as_tuple=False)  # (K, 2) row/col pairs
            if int(idx.shape[0]) > cap:
                idx = idx[torch.randperm(idx.shape[0])[:cap]]
            return idx

        rho3d_idx = _bounded_indices(rho3d_mask)
        if int(rho3d_idx.shape[0]) > 0:
            rows, cols = rho3d_idx[:, 0], rho3d_idx[:, 1]
            g0_g1_disp_rho3d.append((g0_v[rows, cols] - g1_v[rows, cols]).norm(dim=-1))
            g0_g2_disp_rho3d.append((g0_v[rows, cols] - g2_v[rows, cols]).norm(dim=-1))
        rho2d_idx = _bounded_indices(rho2d_mask)
        if int(rho2d_idx.shape[0]) > 0:
            rows, cols = rho2d_idx[:, 0], rho2d_idx[:, 1]
            g0_g1_disp_rho2d.append((g0_v[rows, cols] - g1_v[rows, cols]).norm(dim=-1))
            g0_g2_disp_rho2d.append((g0_v[rows, cols] - g2_v[rows, cols]).norm(dim=-1))

    def _cat_np(chunks: list[torch.Tensor]) -> np.ndarray:
        return torch.cat(chunks).numpy() if chunks else np.zeros((0,), dtype=np.float64)

    geometry_source_comparison = {
        "rho3d_dominated": {
            "g0_vs_g1_displacement": _distribution(_cat_np(g0_g1_disp_rho3d)),
            "g0_vs_g2_displacement": _distribution(_cat_np(g0_g2_disp_rho3d)),
        },
        "rho2d_dominated": {
            "g0_vs_g1_displacement": _distribution(_cat_np(g0_g1_disp_rho2d)),
            "g0_vs_g2_displacement": _distribution(_cat_np(g0_g2_disp_rho2d)),
        },
        "note": "rho3d_dominated events (rho3d<=rho2d) are where the true 3D surfel-plane intersection selected `rho`; rho2d_dominated events (rho2d<rho3d) are where the screen-space low-pass floor selected it instead -- G2 is reconstructed from the SAME s_u/s_v the kernel always derives `depth` from regardless of branch, so disagreement here is diagnostic, not an error in either quantity.",
    }
    _progress(f"[section 6] {geometry_source_comparison}")

    with torch.no_grad():
        _progress("[WL107/109 replay, unchanged] accumulate_image_space_pairs")
        per_view_rep_gpu = [t.to(device) for t in per_view_rep_remapped]
        raw_pairs, _raw_view_support = accumulate_image_space_pairs(count, per_view_rep_gpu, progress=_progress)
        local_pairs, _local_mask = filter_by_3d_locality(raw_pairs, count, graph)
        _progress("[WL107/109 replay, unchanged] apply_secondary_geometric_gate")
        geometry = apply_secondary_geometric_gate(
            local_pairs,
            orientation,
            config,
            neighbor_index=graph.neighbor_index,
            progress=_progress,
        )
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
    region_of_representative_cpu = region_of_representative.detach().cpu()

    if arguments.performance_corpus_out is not None:
        _write_performance_chart_corpus(
            arguments.performance_corpus_out,
            per_view_rep_remapped,
            per_view_world_points_g0,
            subset_ids,
            int(arguments.max_charts) if int(arguments.max_charts) > 0 else None,
            {
                "checkpoint": str(arguments.checkpoint),
                "source_path": str(arguments.source_path),
                "images": arguments.images,
                "max_views": int(arguments.max_views),
                "max_charts": int(arguments.max_charts),
            },
        )
        return

    # --- main chart loop ---
    chart_records: list[dict[str, Any]] = []
    pixel_records: list[dict[str, Any]] = []  # bounded per-chart deterministic subsample, section 8
    all_member_ids_a: list[torch.Tensor] = []
    all_chart_ids_a: list[torch.Tensor] = []
    all_fitted_points_a: list[torch.Tensor] = []
    all_normals_a: list[torch.Tensor] = []

    global_chart_id = 0
    max_charts = int(arguments.max_charts) if int(arguments.max_charts) > 0 else None
    stop = False
    chart_loop_started = time.perf_counter()
    chart_loop_start_epoch = time.time()
    _progress(f"[chart timing] start_epoch={chart_loop_start_epoch:.6f}")

    for view_index, (rep_remapped_cpu, world_points_cpu) in enumerate(zip(per_view_rep_remapped, per_view_world_points_g0)):
        if stop:
            break
        rep_gpu = rep_remapped_cpu.to(device)
        world_gpu = world_points_cpu.to(device)
        rho3d_cpu = per_view_rho3d[view_index]
        rho2d_cpu = per_view_rho2d[view_index]
        s_u_cpu = per_view_s_u[view_index]
        s_v_cpu = per_view_s_v[view_index]
        depth_cpu = per_view_depth[view_index]
        g2_cpu = per_view_world_points_g2[view_index]
        rep_full_cpu = per_view_rep_full[view_index]
        valid_pix = rep_gpu >= 0
        comp_map = torch.where(valid_pix, subset_ids[rep_gpu.clamp(min=0)], torch.full_like(rep_gpu, -1))
        vs = build_view_chart_pixel_samples(view_index, comp_map, rep_gpu, world_gpu)
        if vs.blob_count == 0:
            continue
        mask_valid = valid_pixel_chart_mask(vs, MIN_PIXEL_SAMPLES)
        valid_blob_ids = torch.nonzero(mask_valid, as_tuple=False).reshape(-1).tolist()
        grouped_order = vs.pixel_order_by_blob
        grouped_uv = vs.pixel_uv[grouped_order]
        grouped_xyz = vs.pixel_xyz[grouped_order]
        grouped_representative_id = vs.pixel_representative_id[grouped_order]
        grouped_row_cpu = vs.pixel_row[grouped_order].detach().cpu()
        grouped_col_cpu = vs.pixel_col[grouped_order].detach().cpu()
        blob_offset_cpu = vs.blob_pixel_offset.detach().cpu()
        blob_component_id_cpu = vs.blob_component_id.detach().cpu()

        for local_blob in valid_blob_ids:
            if max_charts is not None and global_chart_id >= max_charts:
                stop = True
                break
            pixel_start = int(blob_offset_cpu[local_blob])
            pixel_end = int(blob_offset_cpu[local_blob + 1])
            uv_camera = grouped_uv[pixel_start:pixel_end]
            pixel_xyz = grouped_xyz[pixel_start:pixel_end]
            pixel_rep_ids = grouped_representative_id[pixel_start:pixel_end]
            rows_t = grouped_row_cpu[pixel_start:pixel_end]
            cols_t = grouped_col_cpu[pixel_start:pixel_end]
            component_id = int(blob_component_id_cpu[local_blob])

            with torch.no_grad():
                # ARM A -- CURRENT (camera UV init -> LSQ -> foot-point correction)
                surface_a, uv_footpoint = fit_torch_visible_surface_lsq(
                    pixel_xyz, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                    degree_u=DEGREE_U, degree_v=DEGREE_V, initial_uv=uv_camera,
                    correction_rounds=CORRECTION_ROUNDS, projection_iterations=PROJECTION_ITERATIONS,
                )
                fitted_a_at_footpoint, normals_a = surface_a.evaluate_with_normals(uv_footpoint)

                # ARM B -- CORRECTED fixed camera UV, SAME solve count, UV NEVER reprojected
                surface_b, uv_fixed_returned = fit_fixed_uv_equal_solves(
                    pixel_xyz, uv_camera, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
                    degree_u=DEGREE_U, degree_v=DEGREE_V, correction_rounds=CORRECTION_ROUNDS,
                )
                assert uv_fixed_returned is uv_camera or torch.equal(uv_fixed_returned, uv_camera)

                # METRIC G -- geometric point-to-surface error, both final surfaces, common evaluation
                # The final ARM A correction round already projected these exact
                # points on this exact final surface.
                uv_geo_a = uv_footpoint
                residual_g_a = (surface_a.evaluate(uv_geo_a) - pixel_xyz).norm(dim=-1)
                residual_g_b, uv_geo_b = geometric_point_to_surface_error(surface_b, pixel_xyz)

                # METRIC C -- camera-correspondence error, both final surfaces, SAME original uv_camera
                residual_c_a = camera_correspondence_error(surface_a, pixel_xyz, uv_camera)
                residual_c_b = camera_correspondence_error(surface_b, pixel_xyz, uv_camera)

            uv_displacement = (uv_footpoint - uv_camera).norm(dim=-1)
            control_grid_diff = (surface_a.control_grid - surface_b.control_grid).norm(dim=-1).mean()
            scalar_values = torch.stack([
                residual_g_a.median(), residual_g_a.quantile(0.95), residual_g_a.max(),
                residual_g_b.median(), residual_g_b.quantile(0.95), residual_g_b.max(),
                residual_c_a.median(), residual_c_a.quantile(0.95), residual_c_a.max(),
                residual_c_b.median(), residual_c_b.quantile(0.95), residual_c_b.max(),
                control_grid_diff, surface_a.smoothness(), surface_b.smoothness(),
            ]).detach().cpu().numpy()
            (
                residual_g_a_median, residual_g_a_p95, residual_g_a_max,
                residual_g_b_median, residual_g_b_p95, residual_g_b_max,
                residual_c_a_median, residual_c_a_p95, residual_c_a_max,
                residual_c_b_median, residual_c_b_p95, residual_c_b_max,
                control_grid_diff_mean, smoothness_a, smoothness_b,
            ) = [float(value) for value in scalar_values]

            # --- domain/support behavior (as WL118, using ARM A's foot-point domain) ---
            rows_np_all = rows_t.numpy()
            cols_np_all = cols_t.numpy()
            row_min, row_max = int(rows_np_all.min()), int(rows_np_all.max())
            col_min, col_max = int(cols_np_all.min()), int(cols_np_all.max())
            blob_mask_np = np.zeros(
                (row_max - row_min + 1, col_max - col_min + 1), dtype=np.bool_
            )
            blob_mask_np[rows_np_all - row_min, cols_np_all - col_min] = True
            camera_domain_shape = blob_domain_shape(blob_mask_np)
            camera_mask = TorchOSNGSPipeline._uv_occupancy_mask(uv_camera.detach(), TRIM_RESOLUTION, TRIM_DILATION)
            fitted_mask = TorchOSNGSPipeline._uv_occupancy_mask(uv_footpoint.detach(), TRIM_RESOLUTION, TRIM_DILATION)
            camera_mask_np, fitted_mask_np = camera_mask.cpu().numpy(), fitted_mask.cpu().numpy()
            camera_holes_np, _ = hole_and_edge_masks(camera_mask_np)
            fitted_holes_np, _ = hole_and_edge_masks(fitted_mask_np)
            intersection = int((camera_mask_np & fitted_mask_np).sum())
            union = int((camera_mask_np | fitted_mask_np).sum())
            iou = (intersection / union) if union > 0 else 1.0

            # --- section 7/8: low-pass provenance + pixel-level D attribution ---
            chart_rho3d = rho3d_cpu[rows_t, cols_t]
            chart_rho2d = rho2d_cpu[rows_t, cols_t]
            chart_branch = classify_median_event_branch(chart_rho3d, chart_rho2d)
            low_pass_valid = chart_branch >= 0
            low_pass_dominated_fraction = float((chart_branch[low_pass_valid] == 0).float().mean()) if bool(low_pass_valid.any()) else float("nan")

            n_pixels = int(chart_rho3d.shape[0])
            assert n_pixels == int(pixel_xyz.shape[0]), "row-major nonzero(blob_mask) must align 1:1 with pixel_xyz's own construction order"
            stride = max(1, n_pixels // PIXEL_SAMPLE_STRIDE_TARGET)
            sample_idx = torch.arange(0, n_pixels, stride)[:PIXEL_SAMPLE_STRIDE_TARGET]
            chart_depth = depth_cpu[rows_t, cols_t]
            chart_s_u = s_u_cpu[rows_t, cols_t]
            chart_s_v = s_v_cpu[rows_t, cols_t]
            chart_g2 = g2_cpu[rows_t, cols_t]
            chart_g0 = per_view_world_points_g0[view_index][rows_t, cols_t]
            chart_rep_full = rep_full_cpu[rows_t, cols_t]
            g2_finite = torch.isfinite(chart_g2).all(dim=-1)
            g0_g2_dist = torch.where(g2_finite, (chart_g0 - chart_g2).norm(dim=-1), torch.full((n_pixels,), float("nan")))
            residual_c_a_cpu = residual_c_a.detach().cpu()

            # Vectorized (no per-pixel Python-level tensor indexing -- the
            # original version called `float(tensor[i])`/`int(tensor[i])`
            # once per field per sampled pixel inside this per-chart loop,
            # which serialized on the CPU across up to ~15,000 charts and
            # starved the per-chart GPU fitting work of wall-clock share
            # (~15% measured GPU utilization); caught via direct user
            # question, fixed by gathering each sampled column ONCE into a
            # plain numpy array, then zipping columns into per-pixel dicts).
            sel_np = sample_idx.numpy()
            s_magnitude_np = torch.sqrt(chart_s_u ** 2 + chart_s_v ** 2).numpy()[sel_np]
            branch_np = chart_branch.numpy()[sel_np]
            g0_g2_dist_np = g0_g2_dist.numpy()[sel_np]
            g2_finite_np = g2_finite.numpy()[sel_np]
            rows_np = rows_t.numpy()[sel_np]
            cols_np = cols_t.numpy()[sel_np]
            residual_np = residual_c_a_cpu.numpy()[sel_np]
            rep_np = chart_rep_full.numpy()[sel_np]
            rho3d_np = chart_rho3d.numpy()[sel_np]
            rho2d_np = chart_rho2d.numpy()[sel_np]
            depth_np = chart_depth.numpy()[sel_np]
            pixel_records.extend(_build_pixel_records_vectorized(
                global_chart_id, view_index,
                rows_np, cols_np, residual_np, rep_np, rho3d_np, rho2d_np,
                branch_np, s_magnitude_np, depth_np, g0_g2_dist_np, g2_finite_np,
            ))

            # --- section 8 (WL118 residual, reworded): representative finite-support spread ---
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
            spread_values = group_max_deviation[multi_member].detach().cpu().numpy()
            within_chart_spread_median = float(np.median(spread_values)) if spread_values.size else 0.0

            sum_point = torch.zeros((rep_count, 3), device=device)
            sum_normal = torch.zeros((rep_count, 3), device=device)
            member_counts = torch.zeros((rep_count,), device=device)
            sum_point.index_add_(0, inverse_rep, fitted_a_at_footpoint)
            sum_normal.index_add_(0, inverse_rep, normals_a)
            member_counts.index_add_(0, inverse_rep, torch.ones_like(inverse_rep, dtype=torch.float32))
            mean_point = sum_point / member_counts.clamp_min(1.0)[:, None]
            mean_normal = torch.nn.functional.normalize(sum_normal, dim=-1, eps=1e-8)

            distinct_reps_cpu = distinct_reps.detach().cpu()
            all_member_ids_a.append(distinct_reps_cpu)
            all_chart_ids_a.append(torch.full((rep_count,), global_chart_id, dtype=torch.int64))
            all_fitted_points_a.append(mean_point.detach().cpu())
            all_normals_a.append(mean_normal.detach().cpu())

            region_votes = region_of_representative_cpu[distinct_reps_cpu]
            valid_votes = region_votes[region_votes >= 0]
            region_index = int(torch.mode(valid_votes).values.item()) if int(valid_votes.numel()) > 0 else -1
            region_label = region_labels_list[region_index] if region_index >= 0 else None
            for rec in pixel_records[-len(sample_idx):]:
                rec["region_label"] = region_label

            uv_disp_np = uv_displacement.detach().cpu().numpy()
            chart_records.append({
                "chart_id": global_chart_id, "view_index": view_index, "camera_name": cameras[view_index].image_name,
                "component_id": component_id, "region_label": region_label,
                "pixel_count": n_pixels, "representative_count": rep_count,
                "uv_displacement_median": float(np.median(uv_disp_np)), "uv_displacement_p95": float(np.percentile(uv_disp_np, 95)),
                "uv_displacement_max": float(uv_disp_np.max()) if uv_disp_np.shape[0] else 0.0,
                "camera_domain_hole_count": int(camera_domain_shape["hole_count"]),
                "camera_uv_hole_count": (1 if bool(camera_holes_np.any()) else 0),
                "fitted_domain_hole_count": (1 if bool(fitted_holes_np.any()) else 0),
                "camera_vs_fitted_support_iou": iou,
                "residual_g_arm_a_median": residual_g_a_median, "residual_g_arm_a_p95": residual_g_a_p95, "residual_g_arm_a_max": residual_g_a_max,
                "residual_g_arm_b_median": residual_g_b_median, "residual_g_arm_b_p95": residual_g_b_p95, "residual_g_arm_b_max": residual_g_b_max,
                "residual_c_arm_a_median": residual_c_a_median, "residual_c_arm_a_p95": residual_c_a_p95, "residual_c_arm_a_max": residual_c_a_max,
                "residual_c_arm_b_median": residual_c_b_median, "residual_c_arm_b_p95": residual_c_b_p95, "residual_c_arm_b_max": residual_c_b_max,
                "control_grid_diff_mean": control_grid_diff_mean,
                "smoothness_arm_a": smoothness_a, "smoothness_arm_b": smoothness_b,
                "low_pass_dominated_fraction": low_pass_dominated_fraction,
                "within_chart_representative_spread_median": within_chart_spread_median,
            })
            global_chart_id += 1
        if view_index % 10 == 0:
            _progress(f"chart-fit view {view_index + 1}/{len(per_view_rep_remapped)} fitted={global_chart_id}")

    chart_loop_seconds = time.perf_counter() - chart_loop_started
    chart_loop_end_epoch = time.time()
    _progress(
        f"[chart accounting] fitted={global_chart_id} pixel_records={len(pixel_records)} "
        f"chart_loop_seconds={chart_loop_seconds:.6f} end_epoch={chart_loop_end_epoch:.6f}"
    )

    # --- section 3/4: METRIC G vs METRIC C, both arms, reported separately ---
    def _agg(field: str) -> dict[str, Any]:
        return _distribution(np.array([r[field] for r in chart_records], dtype=np.float64))

    metric_comparison_summary = {
        "metric_g_geometric_error": {
            "arm_a_median": _agg("residual_g_arm_a_median"), "arm_b_median": _agg("residual_g_arm_b_median"),
            "arm_a_p95": _agg("residual_g_arm_a_p95"), "arm_b_p95": _agg("residual_g_arm_b_p95"),
            "arm_a_max": _agg("residual_g_arm_a_max"), "arm_b_max": _agg("residual_g_arm_b_max"),
        },
        "metric_c_camera_correspondence_error": {
            "arm_a_median": _agg("residual_c_arm_a_median"), "arm_b_median": _agg("residual_c_arm_b_median"),
            "arm_a_p95": _agg("residual_c_arm_a_p95"), "arm_b_p95": _agg("residual_c_arm_b_p95"),
            "arm_a_max": _agg("residual_c_arm_a_max"), "arm_b_max": _agg("residual_c_arm_b_max"),
        },
        "control_grid_difference_mean_distribution": _agg("control_grid_diff_mean"),
        "smoothness_arm_a_distribution": _agg("smoothness_arm_a"),
        "smoothness_arm_b_distribution": _agg("smoothness_arm_b"),
        "note": "METRIC G uses a closest-point evaluation UV computed independently for EACH arm's own final surface (never ARM A's foot-point UV applied to ARM B or vice versa). METRIC C uses the SAME original camera UV for both arms. Do not compare METRIC G of one arm against METRIC C of the other.",
    }
    _progress(f"[section 3/4] {metric_comparison_summary}")

    # --- section 8: pixel-level D attribution ---
    residual_arr = np.array([r["residual_camera_correspondence_arm_a"] for r in pixel_records], dtype=np.float64)
    branch_arr = np.array([r["branch"] for r in pixel_records])
    rho3d_dominated_residual = residual_arr[branch_arr == "rho3d"]
    rho2d_dominated_residual = residual_arr[branch_arr == "rho2d"]
    top_k = min(1000, residual_arr.shape[0])
    top_order = np.argsort(-residual_arr)[:top_k]
    top_branch = branch_arr[top_order]
    pixel_level_d_attribution = {
        "sample_note": f"deterministic per-chart stride subsample, target {PIXEL_SAMPLE_STRIDE_TARGET} pixels/chart, {len(pixel_records)} total sampled pixels across {global_chart_id} charts",
        "rho3d_dominated_residual_distribution": _distribution(rho3d_dominated_residual),
        "rho2d_dominated_residual_distribution": _distribution(rho2d_dominated_residual),
        f"fraction_top{top_k}_residual_pixels_rho2d_dominated": float(np.mean(top_branch == "rho2d")) if top_k else 0.0,
        "top20_extreme_pixels": sorted(pixel_records, key=lambda r: r["residual_camera_correspondence_arm_a"], reverse=True)[:20],
    }
    _progress(f"[section 8] {pixel_level_d_attribution['rho3d_dominated_residual_distribution']} vs {pixel_level_d_attribution['rho2d_dominated_residual_distribution']}")

    # --- section 2: UV displacement summary (unchanged from WL118) ---
    uv_disp_medians = np.array([r["uv_displacement_median"] for r in chart_records], dtype=np.float64)
    uv_displacement_summary = {"per_chart_median_displacement_distribution": _distribution(uv_disp_medians)}

    # --- section 7: signed vs sign-invariant normal disagreement (unchanged computation, reworded interpretation in worklog) ---
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
        "cross_chart_position_discrepancy": _distribution(np.asarray(signed_pos)),
        "signed_normal_discrepancy_degrees": _distribution(np.asarray(signed_normal)),
        "sign_invariant_normal_discrepancy_degrees": _distribution(np.asarray(sign_invariant_normal)),
        "fraction_of_signed_disagreement_explained_by_sign_flip": (float(np.mean(np.asarray(signed_normal) > 90.0)) if len(signed_normal) else 0.0),
        "interpretation_caveat": "remaining discrepancy after sign correction may still contain: different physical-point correspondence, finite surfel footprint, real curvature, parameterization effects, or genuine NURBS disagreement -- this batch does not decompose which.",
    }

    # --- section 9 (WL118 residual, reworded correspondence accounting) ---
    spreads = np.array([r["within_chart_representative_spread_median"] for r in chart_records if r["within_chart_representative_spread_median"] > 0], dtype=np.float64)
    position_correspondence_summary = {
        "within_chart_representative_footprint_spread_distribution": _distribution(spreads),
        "cross_chart_position_discrepancy_distribution": _distribution(np.asarray(signed_pos)),
        "ratio_of_medians": (float(np.median(spreads) / max(np.median(np.asarray(signed_pos)), 1e-12)) if spreads.shape[0] and len(signed_pos) else None),
        "interpretation": "the two distributions have COMPARABLE characteristic scale (see ratio_of_medians) -- this is NOT an 'explained fraction' of the cross-chart discrepancy; it indicates within-chart finite-support ambiguity is of the same ORDER OF MAGNITUDE as the historically-reported cross-chart displacement, not a decomposition of it.",
    }

    # --- region breakdown ---
    region_results = {}
    for label in ANCHOR_FRACTIONS:
        region_charts = [r for r in chart_records if r["region_label"] == label]
        if not region_charts:
            region_results[label] = None
            continue
        region_results[label] = {
            "chart_count": len(region_charts),
            "residual_g_arm_a_median": _distribution(np.array([r["residual_g_arm_a_median"] for r in region_charts])),
            "residual_g_arm_b_median": _distribution(np.array([r["residual_g_arm_b_median"] for r in region_charts])),
            "residual_c_arm_a_median": _distribution(np.array([r["residual_c_arm_a_median"] for r in region_charts])),
            "residual_c_arm_b_median": _distribution(np.array([r["residual_c_arm_b_median"] for r in region_charts])),
        }

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 119",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "camera_meta": camera_meta,
        "nurbs_config_FOR_CONTROL_ONLY": {"resolution_u": RESOLUTION_U, "resolution_v": RESOLUTION_V, "degree_u": DEGREE_U, "degree_v": DEGREE_V, "correction_rounds_BOTH_ARMS": CORRECTION_ROUNDS},
        "wl107_109_replay_consistency_check": wl107_replay_stats,
        "accounting": {"total_trained_surfels": total_model_count, "median_surface_representatives": representative_count, "fitted_chart_count": global_chart_id},
        "synthetic_equal_count_contracts_corrected": synthetic_results,
        "uv_camera_vs_footpoint_displacement": uv_displacement_summary,
        "corrected_uv_ab_metric_comparison": metric_comparison_summary,
        "geometry_source_comparison_G0_G1_G2": geometry_source_comparison,
        "pixel_level_d_attribution": pixel_level_d_attribution,
        "normal_signed_vs_sign_invariant_corrected_interpretation": normal_comparison_summary,
        "representative_position_correspondence_corrected_interpretation": position_correspondence_summary,
        "region_results_WORKING_INTERPRETATION_ONLY": region_results,
        "runtime_seconds": {"total": time.time() - started, "chart_fitting_loop": chart_loop_seconds},
    }

    # --- colors / exports ---
    original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]

    subset_colors = _hash_colors(subset_ids.detach().to(torch.float64))
    subset_colors = torch.where(ever_representative.unsqueeze(-1), subset_colors, torch.tensor(_UNCUT_RGB, device=device).reshape(1, 3))

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

    geo_a_colors = _chart_scalar_color("residual_g_arm_a_median", ((0.2, 0.95, 0.3), (1.0, 0.85, 0.0)))
    geo_b_colors = _chart_scalar_color("residual_g_arm_b_median", ((0.2, 0.95, 0.3), (1.0, 0.85, 0.0)))
    cam_a_colors = _chart_scalar_color("residual_c_arm_a_median", ((0.2, 0.95, 0.3), (1.0, 0.85, 0.0)))
    cam_b_colors = _chart_scalar_color("residual_c_arm_b_median", ((0.2, 0.95, 0.3), (1.0, 0.85, 0.0)))
    branch_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    branch_colors[ever_representative] = torch.tensor((0.6, 0.6, 0.6), dtype=torch.float32, device=device)
    g0g2_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    g0g2_colors[ever_representative] = torch.tensor((0.3, 0.3, 0.9), dtype=torch.float32, device=device)

    views = {
        VIEW_ORIGINAL_SCENE: original_f_dc,
        VIEW_SUBSET_MEMBERSHIP: _rgb_to_f_dc(subset_colors),
        VIEW_GEOMETRIC_ERROR_A: _rgb_to_f_dc(geo_a_colors),
        VIEW_GEOMETRIC_ERROR_B: _rgb_to_f_dc(geo_b_colors),
        VIEW_CAMERA_ERROR_A: _rgb_to_f_dc(cam_a_colors),
        VIEW_CAMERA_ERROR_B: _rgb_to_f_dc(cam_b_colors),
        VIEW_G0_G2_DISAGREEMENT: _rgb_to_f_dc(g0g2_colors),
        VIEW_BRANCH_CLASSIFICATION: _rgb_to_f_dc(branch_colors),
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
    report_path = output_root / "visible_nurbs_geometry_uv_control_correction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
