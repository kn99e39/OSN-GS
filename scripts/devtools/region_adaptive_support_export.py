"""Worklog 101 -- FIXED_MASKED_KNN vs ADAPTIVE_SAME_REGION_LOCAL review export."""

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
    build_preview_camera, load_primitive_model, checkpoint_primitive, PRIMITIVE_SURFEL_2D,
    _hsv_to_rgb, _rgb_to_f_dc, write_surfel_ply, write_ppm,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig
from osn_gs.surface.torch_region_adaptive_support_merge import (
    AdaptiveSupportConfig, SUPPORT_MODE_FIXED_MASKED_KNN, SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL,
    partition_surfels_region_adaptive_support, region_adaptive_support_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"
VIEW_FIXED = "FIXED_MASKED_KNN_PARTITION"
VIEW_ADAPTIVE = "ADAPTIVE_SAME_REGION_LOCAL_PARTITION"
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532


def _progress(message: str) -> None:
    print(f"[adaptive support export] {message}", flush=True)


def _subset_partition_colors(subset_ids: torch.Tensor) -> torch.Tensor:
    identifiers = subset_ids.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)

    _progress(f"loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint.")
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    visible_count = int(visible_selector.shape[0])

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

        local_config = CoverageFirstPartitionConfig()
        region_config = RegionCoherenceConfig(local=local_config, require_positional_continuity=True)

        _progress("[FIXED] baseline (Worklog 100)")
        started_f = time.time()
        config_f = AdaptiveSupportConfig(local=local_config, region=region_config, support_mode=SUPPORT_MODE_FIXED_MASKED_KNN)
        partition_f = partition_surfels_region_adaptive_support(orientation, config_f, progress=_progress)
        accounting_f = region_adaptive_support_accounting(partition_f)
        seconds_f = time.time() - started_f
        _progress(f"[FIXED] done in {seconds_f:.1f}s largest={accounting_f['largest_subset_surfel_fraction']:.4f}")

        _progress("[ADAPTIVE] new support acquisition")
        started_a = time.time()
        config_a = AdaptiveSupportConfig(local=local_config, region=region_config, support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
        partition_a = partition_surfels_region_adaptive_support(orientation, config_a, progress=_progress)
        accounting_a = region_adaptive_support_accounting(partition_a)
        seconds_a = time.time() - started_a
        _progress(f"[ADAPTIVE] done in {seconds_a:.1f}s largest={accounting_a['largest_subset_surfel_fraction']:.4f}")

        fixed_colors = _subset_partition_colors(partition_f.subset_ids)
        adaptive_colors = _subset_partition_colors(partition_a.subset_ids)

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]

        views = {VIEW_FIXED: _rgb_to_f_dc(fixed_colors), VIEW_ADAPTIVE: _rgb_to_f_dc(adaptive_colors)}
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

    del full_orientation, orientation, partition_f, partition_a
    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    render_report: dict[str, Any] = {"enabled": arguments.source_path is not None}
    if arguments.source_path is not None:
        try:
            from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

            camera, camera_metadata = build_preview_camera(
                arguments.source_path, arguments.images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device,
            )
            rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
            _progress(f"rendering previews from camera {camera_metadata['image_name']} backend={rasterizer.backend_source}")
            with torch.no_grad():
                for name, f_dc in views.items():
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
        "batch": "arch/2dgs-coverage-first-surface, Worklog 101",
        "checkpoint": str(arguments.checkpoint),
        "FIXED_MASKED_KNN": accounting_f,
        "ADAPTIVE_SAME_REGION_LOCAL": accounting_a,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"fixed": seconds_f, "adaptive": seconds_a, "total": time.time() - started},
    }
    report_path = output_root / "region_adaptive_support_export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
