"""Worklog 102 -- Maximal Visible Surface Components + review export.

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-101:

    A. ORIGINAL_2DGS_SCENE
    B. OBSERVATION_STATE_VIEW
    C. OCCLUDED_DOMAIN_BOUNDARY_VIEW
    D. KNOWN_FREE_SPACE_CONTRADICTION_VIEW
    E. VISIBLE_DISCONTINUITY_CUT_VIEW
    F. MAXIMAL_VISIBLE_SURFACE_COMPONENTS
    G. WL100_BILATERAL_BASELINE

Neither Trustable-surfel estimation, latent surface construction, nor NURBS
fitting is implemented here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    _hsv_to_rgb, _rgb_to_f_dc, write_surfel_ply, write_cut_edge_curves, write_ppm,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig
from osn_gs.surface.torch_bilateral_interface_region_merge import (
    BilateralInterfaceMergeConfig,
    bilateral_interface_accounting,
    partition_surfels_bilateral_interface,
)
from osn_gs.surface.torch_maximal_visible_connectivity import (
    CUT_KNOWN_FREE_SPACE,
    CUT_OCCLUDED_DOMAIN,
    CUT_POSITIONAL_SHEET_SEPARATION,
    CUT_VISIBLE_GEOMETRIC_DISCONTINUITY,
    UNRESOLVED_OBSERVATION_CONFLICT,
    MaximalVisibleConnectivityConfig,
    maximal_visible_connectivity_accounting,
    partition_maximal_visible_components,
)
from osn_gs.surface.torch_observation_evidence import CameraViewEvidence, ObservationEvidence
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_OBSERVATION_STATE = "OBSERVATION_STATE_VIEW"
VIEW_OCCLUDED_BOUNDARY = "OCCLUDED_DOMAIN_BOUNDARY_VIEW"
VIEW_FREE_SPACE_CONTRADICTION = "KNOWN_FREE_SPACE_CONTRADICTION_VIEW"
VIEW_DISCONTINUITY_CUT = "VISIBLE_DISCONTINUITY_CUT_VIEW"
VIEW_MAXIMAL_COMPONENTS = "MAXIMAL_VISIBLE_SURFACE_COMPONENTS"
VIEW_WL100_BASELINE = "WL100_BILATERAL_BASELINE"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532
_UNCUT_RGB = (0.12, 0.13, 0.16)
_OCCLUDED_RGB = (1.0, 0.15, 0.05)
_FREE_SPACE_RGB = (0.15, 0.55, 1.0)
_DISCONTINUITY_RGB = (1.0, 0.85, 0.1)


def _progress(message: str) -> None:
    print(f"[maximal visible connectivity] {message}", flush=True)


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


def load_all_train_cameras(source_path: Path, image_dir_name: str, sparse_dir_name: str, resolution: int, llffhold: int, device: str):
    """ALL training cameras (held-out test split excluded, matching
    `TorchOSNGSTrainer`'s own split) -- geometry only, no image tensors
    loaded (observation evidence needs depth renders, not the photos)."""

    from PIL import Image as PILImage

    from osn_gs.data.colmap_scene import camera_fovs, camera_matrices, read_colmap_cameras, read_colmap_images, resolve_image_path
    from osn_gs.data.vendor.graphdeco_scene_split import resolve_graphdeco_resolution, select_llff_holdout_test_names
    from osn_gs.render.torch_fallback import TorchCamera

    image_root = source_path / image_dir_name
    sparse_root = source_path / sparse_dir_name
    cameras = read_colmap_cameras(sparse_root)
    images = read_colmap_images(sparse_root)
    ordered = sorted(images.values(), key=lambda image: image.name)
    test_names = set(select_llff_holdout_test_names([image.name for image in ordered], scene_path=source_path, eval=True, llffhold=llffhold))
    train = [image for image in ordered if image.name not in test_names]
    if not train:
        raise ValueError(f"No training cameras remain under {image_root} (llffhold={llffhold}).")

    torch_cameras = []
    for image in train:
        colmap_camera = cameras[image.camera_id]
        with PILImage.open(resolve_image_path(image_root, image.name)) as probe:
            original_width, original_height = probe.size
        target_width, target_height, _ = resolve_graphdeco_resolution(original_width, original_height, resolution=resolution, resolution_scale=1.0)
        fovx, fovy = camera_fovs(colmap_camera, width=colmap_camera.width, height=colmap_camera.height)
        world_view, full_projection, center = camera_matrices(image.qvec, image.tvec, fovx, fovy, device=device)
        torch_cameras.append(TorchCamera(
            image_height=target_height, image_width=target_width, world_view_transform=world_view,
            full_proj_transform=full_projection, camera_center=center, FoVx=fovx, FoVy=fovy, image_name=image.name,
        ))
    return torch_cameras, {"train_camera_count": len(train), "held_out_camera_count": len(ordered) - len(train), "llffhold": llffhold}


def build_surfel_observation_evidence(cameras, model, rasterizer, *, near=1e-3, far=1e6, depth_epsilon=1e-2, progress=None):
    """Minimal 2DGS-surfel counterpart of
    `torch_observation_evidence.build_observation_evidence` -- that function
    checks `rasterizer.config.prefer_cuda`, a field `OSNSurfelRasterizer`'s
    config does not have (it has no non-CUDA fallback at all). Reuses the
    SAME `CameraViewEvidence`/`ObservationEvidence` dataclasses UNCHANGED;
    the only new code is the per-camera render-and-wrap loop itself, not any
    observation-state semantics.
    """

    views = []
    for index, camera in enumerate(cameras):
        package = rasterizer.render(camera, model)
        view_depth = package["depth"].detach().squeeze(0)
        valid_depth_mask = package["valid_depth_mask"].detach()
        if valid_depth_mask.dim() == 3:
            valid_depth_mask = valid_depth_mask.squeeze(0)
        views.append(CameraViewEvidence(
            camera_index=index, image_height=int(camera.image_height), image_width=int(camera.image_width),
            world_view_transform=camera.world_view_transform, full_proj_transform=camera.full_proj_transform,
            view_depth=view_depth, valid_depth_mask=valid_depth_mask, coverage_alpha=None,
            backend_source=rasterizer.backend_source, coverage_kind="binary_contribution_mask",
            depth_kind="direct_linear", depth_is_approximate=True,
        ))
        if progress is not None and index % 20 == 0:
            progress(f"rendered observation evidence {index + 1}/{len(cameras)}")
    return ObservationEvidence(views=views, near=near, far=far, depth_epsilon=depth_epsilon, topology_version="checkpoint", camera_set_version=f"{len(cameras)}_train_cameras")


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
    parser.add_argument("--depth-epsilon", type=float, default=1e-2)
    parser.add_argument("--preview-camera-images", default=None, help="images dir for the single preview render (defaults to --images)")
    parser.add_argument("--curve-cap", type=int, default=50_000)
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)

    _progress(f"loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint (primitive={primitive!r}).")
    total_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    visible_count = int(visible_selector.shape[0])
    _progress(f"model surfels={total_count} visible={visible_count} iteration={payload.get('iteration')}")

    _progress("loading all train cameras")
    cameras, camera_meta = load_all_train_cameras(arguments.source_path, arguments.images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device)
    _progress(f"train cameras: {camera_meta}")

    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    with torch.no_grad():
        observation_evidence = build_surfel_observation_evidence(cameras, model, rasterizer, depth_epsilon=arguments.depth_epsilon, progress=_progress)
    _progress(f"observation evidence built over {len(observation_evidence.views)} views")

    with torch.no_grad():
        from dataclasses import replace as _dc_replace

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

        local_config = CoverageFirstPartitionConfig()
        region_config = RegionCoherenceConfig(local=local_config, require_positional_continuity=True)

        _progress("[F] maximal visible surface components (new)")
        started_f = time.time()
        config_f = MaximalVisibleConnectivityConfig(local=local_config)
        result_f = partition_maximal_visible_components(orientation, observation_evidence, config_f, progress=_progress)
        accounting_f = maximal_visible_connectivity_accounting(result_f)
        seconds_f = time.time() - started_f
        _progress(f"[F] done in {seconds_f:.1f}s -> {accounting_f['visible_component_count']} components "
                  f"largest={accounting_f['largest_component_surfel_fraction']:.4f} "
                  f"cuts={accounting_f['boundary_cut_reason_counts']}")

        _progress("[G] Worklog 100 bilateral baseline")
        started_g = time.time()
        config_g = BilateralInterfaceMergeConfig(local=local_config, region=region_config)
        result_g = partition_surfels_bilateral_interface(orientation, config_g, progress=_progress)
        accounting_g = bilateral_interface_accounting(result_g)
        seconds_g = time.time() - started_g
        _progress(f"[G] done in {seconds_g:.1f}s -> largest={accounting_g['largest_subset_surfel_fraction']:.4f}")

        # --- colors ---
        maximal_colors = _subset_partition_colors(result_f.subset_ids)
        wl100_colors = _subset_partition_colors(result_g.subset_ids)

        occluded_edges = result_f.graph.candidate_edges[result_f.graph.spatial_edge_mask & result_f.cut_occluded_domain]
        free_edges = result_f.graph.candidate_edges[result_f.graph.spatial_edge_mask & result_f.cut_known_free_space]
        discontinuity_edges = result_f.graph.candidate_edges[
            result_f.graph.spatial_edge_mask & (result_f.cut_visible_geometric_discontinuity | result_f.cut_positional_sheet_separation)
        ]
        conflict_edges = result_f.graph.candidate_edges[result_f.graph.spatial_edge_mask & result_f.cut_unresolved_observation_conflict]

        occluded_colors = _edge_highlight_colors(visible_count, occluded_edges, _UNCUT_RGB, _OCCLUDED_RGB, device)
        free_colors = _edge_highlight_colors(visible_count, free_edges, _UNCUT_RGB, _FREE_SPACE_RGB, device)
        discontinuity_colors = _edge_highlight_colors(visible_count, discontinuity_edges, _UNCUT_RGB, _DISCONTINUITY_RGB, device)

        # observation-state view: per-surfel, was it evaluated by ANY camera
        # co-observation (green) or never evaluated (dark)?
        evaluated_node = torch.zeros((visible_count,), dtype=torch.bool, device=device)
        eval_edges = result_f.graph.candidate_edges[result_f.observation_evaluated]
        if int(eval_edges.shape[0]) > 0:
            evaluated_node[eval_edges[:, 0]] = True
            evaluated_node[eval_edges[:, 1]] = True
        observation_state_colors = _ramp(evaluated_node.to(torch.float32), _UNCUT_RGB, (0.2, 0.9, 0.3))

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_OBSERVATION_STATE: _rgb_to_f_dc(observation_state_colors),
            VIEW_OCCLUDED_BOUNDARY: _rgb_to_f_dc(occluded_colors),
            VIEW_FREE_SPACE_CONTRADICTION: _rgb_to_f_dc(free_colors),
            VIEW_DISCONTINUITY_CUT: _rgb_to_f_dc(discontinuity_colors),
            VIEW_MAXIMAL_COMPONENTS: _rgb_to_f_dc(maximal_colors),
            VIEW_WL100_BASELINE: _rgb_to_f_dc(wl100_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        for name, edges in ((VIEW_OCCLUDED_BOUNDARY, occluded_edges), (VIEW_FREE_SPACE_CONTRADICTION, free_edges), (VIEW_DISCONTINUITY_CUT, discontinuity_edges)):
            total = int(edges.shape[0])
            if total > arguments.curve_cap > 0:
                stride = (total + arguments.curve_cap - 1) // arguments.curve_cap
                curve_edges = edges[::stride]
            else:
                curve_edges = edges
            segments = (
                torch.stack([positions[curve_edges[:, 0]], positions[curve_edges[:, 1]]], dim=1)
                if int(curve_edges.shape[0]) > 0 else torch.zeros((0, 2, 3), device=positions.device)
            )
            json_path = output_root / name / _ITERATION_DIR / "nurbs_surface.json"
            write_cut_edge_curves(json_path, segments, {"representation": name.lower(), "edge_total": total, "edge_rendered": int(curve_edges.shape[0])})
            view_paths[name]["boundary_curves_json"] = str(json_path)

    del full_orientation, orientation, result_g
    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    render_report: dict[str, Any] = {"enabled": True}
    try:
        from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer as _R, SurfelRasterizerConfig as _C

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

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 102",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "input_domain": {"model_surfel_count": total_count, "visible_surfel_count": visible_count},
        "camera_meta": camera_meta,
        "F_maximal_visible_connectivity": accounting_f,
        "G_wl100_bilateral_baseline": accounting_g,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"observation_evidence": time.time() - started - seconds_f - seconds_g, "partition_f": seconds_f, "partition_g": seconds_g, "total": time.time() - started},
    }
    report_path = output_root / "maximal_visible_connectivity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "F_component_count": accounting_f["visible_component_count"], "F_largest_fraction": accounting_f["largest_component_surfel_fraction"],
        "F_cut_reason_counts": accounting_f["boundary_cut_reason_counts"],
        "G_subset_count": accounting_g["subset_count"], "G_largest_fraction": accounting_g["largest_subset_surfel_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
