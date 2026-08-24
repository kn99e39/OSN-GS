"""Worklog 107 -- Camera-Induced Visible Adjacency + review export.

Does NOT modify torch_positive_visible_adjacency.py (Worklog 103),
torch_node_level_observability_accounting.py (Worklog 104),
torch_surfel_contribution_diagnostics.py (Worklog 105), or
torch_renderer_grounded_visible_adjacency.py (Worklog 106). Worklog 106's own
committed real-scene numbers (`output/osn_gs_renderer_grounded_visible_
adjacency/renderer_grounded_visible_adjacency_report.json`) are cited directly
as the PAIRWISE_CAMERA_APPROVAL_BASELINE rather than re-running that
(unmodified, deterministic, already-measured) pipeline a second time.

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-106:

    A. ORIGINAL_2DGS_SCENE
    B. RENDERER_SURFACE_REPRESENTATIVE_VIEW
    C. CAMERA_INDUCED_PER_VIEW_ADJACENCY (raw image-space pairs, pre-filter)
    D. CAMERA_INDUCED_GLOBAL_ADJACENCY (final positive edges)
    E. CAMERA_INDUCED_VISIBLE_COMPONENTS
    F. GEOMETRIC_REJECTION_VIEW
    G. NON_REPRESENTATIVE_CONTRIBUTOR_VIEW
    H. WL106_PAIRWISE_BASELINE (copied from the committed Worklog 106 export)

Neither Trust, latent surface, NURBS fitting, NURBS decomposition, occluded
surface generation, nor uncertain Gaussian proposal is implemented here.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from osn_gs.render.torch_surfel_contribution_diagnostics import accumulate_renderer_contribution_evidence
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative
from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    REASON_GEOMETRIC_DISCONTINUITY,
    REASON_POSITIONAL_SHEET_SEPARATION,
    CameraInducedAdjacencyConfig,
    camera_induced_visible_adjacency_accounting,
    partition_camera_induced_visible_adjacency,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_REPRESENTATIVE = "RENDERER_SURFACE_REPRESENTATIVE_VIEW"
VIEW_PER_VIEW_ADJACENCY = "CAMERA_INDUCED_PER_VIEW_ADJACENCY"
VIEW_GLOBAL_ADJACENCY = "CAMERA_INDUCED_GLOBAL_ADJACENCY"
VIEW_COMPONENTS = "CAMERA_INDUCED_VISIBLE_COMPONENTS"
VIEW_GEOMETRIC_REJECTION = "GEOMETRIC_REJECTION_VIEW"
VIEW_NON_REPRESENTATIVE = "NON_REPRESENTATIVE_CONTRIBUTOR_VIEW"
VIEW_WL106_BASELINE = "WL106_PAIRWISE_BASELINE"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532
_UNCUT_RGB = (0.08, 0.09, 0.11)
_REPRESENTATIVE_RGB = (0.15, 0.75, 0.95)
_PER_VIEW_RGB = (0.7, 0.6, 0.1)
_POSITIVE_RGB = (0.2, 0.9, 0.3)
_GEOMETRIC_RGB = (1.0, 0.15, 0.55)
_NON_REPRESENTATIVE_RGB = (1.0, 0.55, 0.0)


def _progress(message: str) -> None:
    print(f"[camera-induced visible adjacency] {message}", flush=True)


def _subset_partition_colors(subset_ids: torch.Tensor) -> torch.Tensor:
    identifiers = subset_ids.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def _ramp(ratio: torch.Tensor, low_rgb, high_rgb) -> torch.Tensor:
    low = torch.tensor(low_rgb, device=ratio.device).reshape(1, 3)
    high = torch.tensor(high_rgb, device=ratio.device).reshape(1, 3)
    return low + ratio.clamp(0.0, 1.0).reshape(-1, 1) * (high - low)


def _edge_highlight_colors(count: int, edges: torch.Tensor, base_rgb, highlight_rgb, device) -> torch.Tensor:
    degree = torch.zeros((count,), dtype=torch.float32, device=device)
    if int(edges.shape[0]) > 0:
        ones = torch.ones((int(edges.shape[0]),), dtype=torch.float32, device=device)
        degree.index_add_(0, edges[:, 0], ones)
        degree.index_add_(0, edges[:, 1], ones)
    ratio = (degree > 0).to(torch.float32)
    return _ramp(ratio, base_rgb, highlight_rgb)


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
    parser.add_argument(
        "--wl106-report", type=Path,
        default=REPO_ROOT / "output" / "osn_gs_renderer_grounded_visible_adjacency" / "renderer_grounded_visible_adjacency_report.json",
    )
    parser.add_argument(
        "--wl106-components-ply", type=Path,
        default=REPO_ROOT / "output" / "osn_gs_renderer_grounded_visible_adjacency" / "RENDERER_GROUNDED_VISIBLE_COMPONENTS" / _ITERATION_DIR / "point_cloud.ply",
    )
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
    _progress(f"train cameras: {camera_meta}")

    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())

    full_to_visible = torch.full((total_model_count,), -1, dtype=torch.int64, device=model.device)
    full_to_visible[visible_selector] = torch.arange(visible_count, dtype=torch.int64, device=model.device)

    _progress("[contribution] renderer-contribution evidence (Worklog 105, unmodified)")
    started_contrib = time.time()
    contribution = accumulate_renderer_contribution_evidence(cameras, model, rasterizer, progress=_progress)
    ever_contributed = contribution.ever_contributed[visible_selector]
    seconds_contrib = time.time() - started_contrib
    _progress(f"[contribution] done in {seconds_contrib:.1f}s, contributing={int(ever_contributed.sum())}/{visible_count}")

    _progress("[representative] per-view renderer surface-representative maps (Worklog 107, new diagnostic build)")
    started_rep = time.time()
    per_view_representative_ids: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        ever_representative_full[rep_full[valid]] = True
        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_representative_ids.append(rep_remapped.detach())
        del diag
        if index % 20 == 0:
            _progress(f"representative view {index + 1}/{len(cameras)}")
    seconds_rep = time.time() - started_rep
    ever_representative = ever_representative_full[visible_selector]
    _progress(f"[representative] done in {seconds_rep:.1f}s, representative={int(ever_representative.sum())}/{visible_count}")

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
        device = positions.device
        count = int(positions.shape[0])

        local_config = CoverageFirstPartitionConfig()
        config = CameraInducedAdjacencyConfig(local=local_config)

        _progress("[B] camera-induced visible adjacency (new)")
        started_b = time.time()
        result = partition_camera_induced_visible_adjacency(orientation, per_view_representative_ids, config, progress=_progress)
        accounting = camera_induced_visible_adjacency_accounting(result)
        seconds_b = time.time() - started_b
        _progress(f"[B] done in {seconds_b:.1f}s -> {accounting['visible_component_count']} components "
                  f"largest={accounting['largest_component_surfel_fraction']:.4f} singleton={accounting['singleton_surfel_fraction']:.4f}")

        # --- WL106 baseline: cite the committed report, do not recompute ---
        wl106_accounting = None
        if arguments.wl106_report.exists():
            wl106_accounting = json.loads(arguments.wl106_report.read_text(encoding="utf-8"))["B_renderer_grounded"]
            _progress(f"[A, cited] WL106 committed: {wl106_accounting['visible_component_count']} components "
                      f"largest={wl106_accounting['largest_component_surfel_fraction']:.4f} "
                      f"singleton={wl106_accounting['singleton_surfel_fraction']:.4f}")
        else:
            _progress(f"[A] WL106 committed report not found at {arguments.wl106_report}; baseline comparison omitted")

        # --- representative coverage (directive section 12) ---
        never_representative_but_contributing = ever_contributed & ~ever_representative
        representative_coverage = {
            "total_visible_domain_surfels": count,
            "renderer_contributing_surfel_count": int(ever_contributed.sum()),
            "renderer_surface_representative_surfel_count": int(ever_representative.sum()),
            "renderer_contributing_but_never_representative_count": int(never_representative_but_contributing.sum()),
            "renderer_contributing_but_never_representative_fraction": (
                int(never_representative_but_contributing.sum()) / int(ever_contributed.sum()) if int(ever_contributed.sum()) > 0 else 0.0
            ),
        }
        _progress(f"[representative coverage] {representative_coverage}")

        report = {
            "batch": "arch/2dgs-coverage-first-surface, Worklog 107",
            "checkpoint": str(arguments.checkpoint),
            "primitive": primitive,
            "iteration": int(payload.get("iteration", 0)),
            "primitive_accounting": {
                "total_model_surfel_count": total_model_count,
                "visible_domain_surfel_count": visible_count,
            },
            "camera_meta": camera_meta,
            "representative_coverage": representative_coverage,
            "B_camera_induced": accounting,
            "A_wl106_pairwise_baseline_cited": wl106_accounting,
            "runtime_seconds": {"contribution_pass": seconds_contrib, "representative_pass": seconds_rep, "adjacency_pass": seconds_b, "total": time.time() - started},
        }

        # --- colors ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        representative_colors = _ramp(ever_representative.to(torch.float32), _UNCUT_RGB, _REPRESENTATIVE_RGB)
        component_colors = _subset_partition_colors(result.subset_ids)
        non_representative_colors = _ramp(never_representative_but_contributing.to(torch.float32), _UNCUT_RGB, _NON_REPRESENTATIVE_RGB)

        # raw per-view (pre-filter) pairs, recomputed cheaply for the review view only
        from osn_gs.surface.torch_camera_induced_visible_adjacency import accumulate_image_space_pairs

        raw_pairs, _raw_support = accumulate_image_space_pairs(count, per_view_representative_ids, progress=None)
        per_view_adjacency_colors = _edge_highlight_colors(count, raw_pairs, _UNCUT_RGB, _PER_VIEW_RGB, device)
        global_adjacency_colors = _edge_highlight_colors(count, result.positive_visible_edges, _UNCUT_RGB, _POSITIVE_RGB, device)
        geometric_rejection_colors = _edge_highlight_colors(count, result.geometric_rejected_pairs, _UNCUT_RGB, _GEOMETRIC_RGB, device)

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_REPRESENTATIVE: _rgb_to_f_dc(representative_colors),
            VIEW_PER_VIEW_ADJACENCY: _rgb_to_f_dc(per_view_adjacency_colors),
            VIEW_GLOBAL_ADJACENCY: _rgb_to_f_dc(global_adjacency_colors),
            VIEW_COMPONENTS: _rgb_to_f_dc(component_colors),
            VIEW_GEOMETRIC_REJECTION: _rgb_to_f_dc(geometric_rejection_colors),
            VIEW_NON_REPRESENTATIVE: _rgb_to_f_dc(non_representative_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        if arguments.wl106_components_ply.exists():
            dest = output_root / VIEW_WL106_BASELINE / _ITERATION_DIR / "point_cloud.ply"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(arguments.wl106_components_ply, dest)
            view_paths[VIEW_WL106_BASELINE] = {"point_cloud_ply": str(dest), "copied_from": str(arguments.wl106_components_ply)}
            _progress(f"copied {VIEW_WL106_BASELINE} from committed Worklog 106 export")

    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    render_report: dict[str, Any] = {"enabled": True}
    try:
        preview_images = arguments.preview_camera_images or arguments.images
        preview_cameras, preview_meta = load_all_train_cameras(arguments.source_path, preview_images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device)
        preview_camera = min(preview_cameras, key=lambda c: c.image_name)
        _progress(f"rendering previews from camera {preview_camera.image_name}")
        with torch.no_grad():
            for name, f_dc in views.items():
                full_dc = torch.zeros_like(model._features_dc)
                full_dc[visible_selector, 0, :] = f_dc
                model._features_dc.data.copy_(full_dc)
                model._features_rest.data.zero_()
                model.active_sh_degree = 0
                del full_dc
                package = rasterizer.render(preview_camera, model)
                ppm_path = output_root / name / "render.ppm"
                write_ppm(ppm_path, package["render"])
                view_paths[name]["render_ppm"] = str(ppm_path)
                _progress(f"rendered {name}")
                del package
        render_report.update({"camera": preview_camera.image_name})
    except Exception as error:
        render_report.update({"failed": True, "reason": f"{type(error).__name__}: {error}"})
        _progress(f"render.ppm generation FAILED: {type(error).__name__}: {error}")

    report["views"] = view_paths
    report["render_ppm"] = render_report
    report_path = output_root / "camera_induced_visible_adjacency_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
