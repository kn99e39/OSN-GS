"""Worklog 98 -- discontinuity-first vs region-concentration Surfel Subset partition + review export.

Runs the discontinuity-first partition
(osn_gs.surface.torch_discontinuity_first_surfel_partition) alongside
Worklog 97's region-concentration partition on the SAME trained 2DGS
checkpoint and the SAME local candidate graph, and produces the 7 matched
full-scene views the architecture directive requires:

    A. ORIGINAL_2DGS_SCENE
    B. RAW_INTRINSIC_NORMAL
    C. NORMAL_GRADIENT_MAGNITUDE
    D. SMOOTH_SURFACE_MODEL_RESIDUAL
    E. DETECTED_DISCONTINUITY_BOUNDARY
    F. WL97_REGION_CONCENTRATION_PARTITION
    G. DISCONTINUITY_FIRST_PARTITION

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
    build_preview_camera,
    load_primitive_model,
    checkpoint_primitive,
    PRIMITIVE_SURFEL_2D,
    _hsv_to_rgb,
    _rgb_to_f_dc,
    _percentile,
    write_surfel_ply,
    write_cut_edge_curves,
    write_ppm,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_discontinuity_first_surfel_partition import (
    DiscontinuityFirstConfig,
    discontinuity_first_accounting,
    partition_surfels_discontinuity_first,
)
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    RegionCoherenceConfig,
    partition_surfels_region_coherent,
    region_coherent_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_RAW_NORMAL = "RAW_INTRINSIC_NORMAL"
VIEW_GRADIENT_MAGNITUDE = "NORMAL_GRADIENT_MAGNITUDE"
VIEW_MODEL_RESIDUAL = "SMOOTH_SURFACE_MODEL_RESIDUAL"
VIEW_DISCONTINUITY_BOUNDARY = "DETECTED_DISCONTINUITY_BOUNDARY"
VIEW_WL97_PARTITION = "WL97_REGION_CONCENTRATION_PARTITION"
VIEW_DISCONTINUITY_PARTITION = "DISCONTINUITY_FIRST_PARTITION"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

_LOW_RGB = (0.10, 0.35, 0.95)
_HIGH_RGB = (1.0, 0.55, 0.05)
_UNCUT_RGB = (0.12, 0.13, 0.16)
_CUT_RGB = (1.0, 0.15, 0.05)


def _progress(message: str) -> None:
    print(f"[discontinuity-first partition] {message}", flush=True)


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


def _p95_ratio(values: torch.Tensor) -> torch.Tensor:
    reference = max(float(torch.quantile(values.float(), 0.95)), 1e-8)
    return values / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--neighbor-count", type=int, default=CoverageFirstPartitionConfig.neighbor_count)
    parser.add_argument("--spacing-multiplier", type=float, default=CoverageFirstPartitionConfig.spatial_connect_spacing_multiplier)
    parser.add_argument("--normal-alignment", type=float, default=CoverageFirstPartitionConfig.normal_compatibility_min_alignment)
    parser.add_argument("--knn-chunk", type=int, default=0)
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

        local_config = CoverageFirstPartitionConfig(
            neighbor_count=int(arguments.neighbor_count),
            spatial_connect_spacing_multiplier=float(arguments.spacing_multiplier),
            normal_compatibility_min_alignment=float(arguments.normal_alignment),
            knn_chunk_size=int(arguments.knn_chunk),
        )

        _progress("[F] WL97 region-concentration partition")
        started_f = time.time()
        partition_f = partition_surfels_region_coherent(orientation, RegionCoherenceConfig(local=local_config), progress=_progress)
        accounting_f = region_coherent_accounting(partition_f)
        seconds_f = time.time() - started_f
        _progress(f"[F] done in {seconds_f:.1f}s -> {accounting_f['subset_count']} subsets, "
                  f"largest_fraction={accounting_f['largest_subset_surfel_fraction']:.4f}")

        _progress("[G] discontinuity-first partition")
        started_g = time.time()
        discontinuity_config = DiscontinuityFirstConfig(local=local_config)
        partition_g = partition_surfels_discontinuity_first(orientation, discontinuity_config, progress=_progress)
        accounting_g = discontinuity_first_accounting(partition_g)
        seconds_g = time.time() - started_g
        _progress(f"[G] done in {seconds_g:.1f}s -> {accounting_g['subset_count']} subsets, "
                  f"largest_fraction={accounting_g['largest_subset_surfel_fraction']:.4f}, "
                  f"boundary_cut_edges={accounting_g['boundary_cut_edge_count']}")

        # --- colors ---
        raw_normal_colors = orientation.surface_normal.abs().clamp(0.0, 1.0)

        gradient_ratio = _p95_ratio(partition_g.normal_gradient_magnitude)
        gradient_colors = _ramp(gradient_ratio, _LOW_RGB, _HIGH_RGB)

        spatial_mask = partition_g.graph.spatial_edge_mask
        residual_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        residual_weight_sum = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        spatial_edges_all = partition_g.graph.candidate_edges[spatial_mask]
        spatial_residual = partition_g.edge_residual[spatial_mask]
        if int(spatial_edges_all.shape[0]) > 0:
            residual_weight_sum.index_add_(0, spatial_edges_all[:, 0], spatial_residual)
            residual_weight_sum.index_add_(0, spatial_edges_all[:, 1], spatial_residual)
            ones = torch.ones((int(spatial_edges_all.shape[0]),), dtype=torch.float32, device=positions.device)
            residual_degree.index_add_(0, spatial_edges_all[:, 0], ones)
            residual_degree.index_add_(0, spatial_edges_all[:, 1], ones)
        per_surfel_residual = torch.where(
            residual_degree > 0, residual_weight_sum / residual_degree.clamp_min(1.0), torch.zeros_like(residual_degree)
        )
        residual_colors = _ramp(_p95_ratio(per_surfel_residual), _LOW_RGB, _HIGH_RGB)

        boundary_edges = partition_g.boundary_edges
        cut_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        spatial_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        for target, source in ((cut_degree, boundary_edges), (spatial_degree, spatial_edges_all)):
            if int(source.shape[0]) > 0:
                ones = torch.ones((int(source.shape[0]),), dtype=torch.float32, device=positions.device)
                target.index_add_(0, source[:, 0], ones)
                target.index_add_(0, source[:, 1], ones)
        cut_ratio = torch.where(spatial_degree > 0, cut_degree / spatial_degree.clamp_min(1.0), torch.zeros_like(cut_degree))
        boundary_colors = _ramp(cut_ratio, _UNCUT_RGB, _CUT_RGB)

        wl97_colors = _subset_partition_colors(partition_f.subset_ids)
        discontinuity_colors = _subset_partition_colors(partition_g.subset_ids)

        boundary_total = int(boundary_edges.shape[0])
        if boundary_total > arguments.curve_cap > 0:
            stride = (boundary_total + arguments.curve_cap - 1) // arguments.curve_cap
            curve_edges = boundary_edges[::stride]
        else:
            curve_edges = boundary_edges
        boundary_segments = (
            torch.stack([positions[curve_edges[:, 0]], positions[curve_edges[:, 1]]], dim=1)
            if int(curve_edges.shape[0]) > 0 else torch.zeros((0, 2, 3), device=positions.device)
        )

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_RAW_NORMAL: _rgb_to_f_dc(raw_normal_colors),
            VIEW_GRADIENT_MAGNITUDE: _rgb_to_f_dc(gradient_colors),
            VIEW_MODEL_RESIDUAL: _rgb_to_f_dc(residual_colors),
            VIEW_DISCONTINUITY_BOUNDARY: _rgb_to_f_dc(boundary_colors),
            VIEW_WL97_PARTITION: _rgb_to_f_dc(wl97_colors),
            VIEW_DISCONTINUITY_PARTITION: _rgb_to_f_dc(discontinuity_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        boundary_json_path = output_root / VIEW_DISCONTINUITY_BOUNDARY / _ITERATION_DIR / "nurbs_surface.json"
        write_cut_edge_curves(
            boundary_json_path, boundary_segments,
            {
                "representation": "discontinuity_first_boundary_cut_edges",
                "boundary_total": boundary_total, "boundary_rendered": int(curve_edges.shape[0]),
                "selection": "uniform stride over the canonical spatial-edge order, not a spatial crop",
                "reason_counts": accounting_g["boundary_cut_reason_counts"],
            },
        )
        view_paths[VIEW_DISCONTINUITY_BOUNDARY]["boundary_curves_json"] = str(boundary_json_path)

    del full_orientation, orientation, partition_f, partition_g
    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    render_report: dict[str, Any] = {"enabled": arguments.source_path is not None}
    if arguments.source_path is not None:
        try:
            from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

            camera, camera_metadata = build_preview_camera(
                arguments.source_path, arguments.images, arguments.sparse_dir,
                arguments.resolution, arguments.llffhold, arguments.device,
            )
            rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
            _progress(f"rendering previews from camera {camera_metadata['image_name']} backend={rasterizer.backend_source}")
            trained_sh_degree = int(model.active_sh_degree)
            with torch.no_grad():
                for name, f_dc in views.items():
                    if name == VIEW_ORIGINAL_SCENE:
                        model.active_sh_degree = trained_sh_degree
                    else:
                        full_dc = torch.zeros_like(model._features_dc)
                        full_dc[visible_selector, 0, :] = f_dc
                        model._features_dc.data.copy_(full_dc)
                        model._features_rest.data.zero_()
                        model.active_sh_degree = 0
                        del full_dc
                    package = rasterizer.render(camera, model)
                    ppm_path = output_root / name / "render.ppm"
                    write_ppm(ppm_path, package["render"])
                    view_paths[name]["render_ppm"] = str(ppm_path)
                    _progress(f"rendered {name}")
                    del package
            render_report.update({"camera": camera_metadata, "backend": rasterizer.backend_source})
        except Exception as error:
            render_report.update({"failed": True, "reason": f"{type(error).__name__}: {error}"})
            _progress(f"render.ppm generation FAILED: {type(error).__name__}: {error}")
    else:
        render_report["reason"] = "--source-path not provided"

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 98",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "input_domain": {"model_surfel_count": total_count, "visible_surfel_count": visible_count},
        "F_wl97_region_concentration": accounting_f,
        "G_discontinuity_first": accounting_g,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"partition_f": seconds_f, "partition_g": seconds_g, "total": time.time() - started},
    }
    report_path = output_root / "discontinuity_first_partition_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "F_subset_count": accounting_f["subset_count"], "G_subset_count": accounting_g["subset_count"],
        "F_largest_fraction": accounting_f["largest_subset_surfel_fraction"],
        "G_largest_fraction": accounting_g["largest_subset_surfel_fraction"],
    }))


if __name__ == "__main__":
    main()
