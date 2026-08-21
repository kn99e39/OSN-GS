"""Worklog 100 -- region-conditioned bilateral interface Surfel Region merge
+ review export.

Runs, on the SAME trained 2DGS checkpoint and the SAME local candidate
graph/initialization as Worklog 99:

    A. WL99_INTERFACE_COHERENT_PARTITION   (baseline, unmodified default)
    B. BILATERAL_INTERFACE_PARTITION       (new: region-conditioned, bilateral)
    C. ACCEPTED_BILATERAL_INTERFACE_MERGES
    D. REJECTED_BILATERAL_INTERFACES
    E. MERGE_PROVENANCE_DEPTH

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
    write_surfel_ply,
    write_cut_edge_curves,
    write_ppm,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig
from osn_gs.surface.torch_interface_coherent_region_merge import (
    InterfaceCoherentMergeConfig,
    interface_coherent_accounting,
    partition_surfels_interface_coherent,
)
from osn_gs.surface.torch_bilateral_interface_region_merge import (
    BilateralInterfaceMergeConfig,
    bilateral_interface_accounting,
    partition_surfels_bilateral_interface,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_WL99_PARTITION = "WL99_INTERFACE_COHERENT_PARTITION"
VIEW_BILATERAL_PARTITION = "BILATERAL_INTERFACE_PARTITION"
VIEW_ACCEPTED = "ACCEPTED_BILATERAL_INTERFACE_MERGES"
VIEW_REJECTED = "REJECTED_BILATERAL_INTERFACES"
VIEW_MERGE_DEPTH = "MERGE_PROVENANCE_DEPTH"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

_UNCUT_RGB = (0.12, 0.13, 0.16)
_ACCEPTED_RGB = (0.15, 0.85, 0.25)
_REJECTED_RGB = (1.0, 0.15, 0.05)
_LOW_RGB = (0.10, 0.35, 0.95)
_HIGH_RGB = (1.0, 0.55, 0.05)


def _progress(message: str) -> None:
    print(f"[bilateral interface merge] {message}", flush=True)


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
        device = positions.device

        local_config = CoverageFirstPartitionConfig(
            neighbor_count=int(arguments.neighbor_count),
            spatial_connect_spacing_multiplier=float(arguments.spacing_multiplier),
            normal_compatibility_min_alignment=float(arguments.normal_alignment),
            knn_chunk_size=int(arguments.knn_chunk),
        )
        region_config = RegionCoherenceConfig(local=local_config, require_positional_continuity=True)

        _progress("[A] Worklog 99 interface-coherent merge (baseline)")
        started_a = time.time()
        config_a = InterfaceCoherentMergeConfig(local=local_config, region=region_config)
        partition_a = partition_surfels_interface_coherent(orientation, config_a, progress=_progress)
        accounting_a = interface_coherent_accounting(partition_a)
        seconds_a = time.time() - started_a
        _progress(f"[A] done in {seconds_a:.1f}s -> initial={accounting_a['initial_region_count']} "
                  f"final={accounting_a['final_region_count']} largest={accounting_a['largest_subset_surfel_fraction']:.4f} "
                  f"merges_applied={accounting_a['merges_applied']}")

        _progress("[B] region-conditioned bilateral interface merge (new)")
        started_b = time.time()
        config_b = BilateralInterfaceMergeConfig(local=local_config, region=region_config)
        partition_b = partition_surfels_bilateral_interface(orientation, config_b, progress=_progress)
        accounting_b = bilateral_interface_accounting(partition_b)
        seconds_b = time.time() - started_b
        _progress(f"[B] done in {seconds_b:.1f}s -> initial={accounting_b['initial_region_count']} "
                  f"final={accounting_b['final_region_count']} largest={accounting_b['largest_subset_surfel_fraction']:.4f} "
                  f"merges_applied={accounting_b['merges_applied']}")

        # --- colors ---
        wl99_colors = _subset_partition_colors(partition_a.subset_ids)
        bilateral_colors = _subset_partition_colors(partition_b.subset_ids)

        accepted_edges = partition_b.graph.candidate_edges[partition_b.accepted_merge_edges_mask]
        rejected_edges = partition_b.graph.candidate_edges[partition_b.rejected_interface_edges_mask]
        accepted_colors = _edge_highlight_colors(visible_count, accepted_edges, _UNCUT_RGB, _ACCEPTED_RGB, device)
        rejected_colors = _edge_highlight_colors(visible_count, rejected_edges, _UNCUT_RGB, _REJECTED_RGB, device)

        final_of_initial = partition_b.final_region_of_initial
        _, fragment_inverse, fragment_counts = torch.unique(final_of_initial, return_inverse=True, return_counts=True)
        fragments_per_final_root_index = (fragment_counts - 1).clamp_min(0).to(torch.float32)
        depth_per_initial_region = fragments_per_final_root_index[fragment_inverse]
        depth_per_surfel = depth_per_initial_region[partition_b.initial_region_ids]
        depth_reference = max(float(depth_per_surfel.max()), 1.0)
        merge_depth_colors = _ramp(depth_per_surfel / depth_reference, _LOW_RGB, _HIGH_RGB)

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]

        views = {
            VIEW_WL99_PARTITION: _rgb_to_f_dc(wl99_colors),
            VIEW_BILATERAL_PARTITION: _rgb_to_f_dc(bilateral_colors),
            VIEW_ACCEPTED: _rgb_to_f_dc(accepted_colors),
            VIEW_REJECTED: _rgb_to_f_dc(rejected_colors),
            VIEW_MERGE_DEPTH: _rgb_to_f_dc(merge_depth_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        for name, mask_edges in ((VIEW_ACCEPTED, accepted_edges), (VIEW_REJECTED, rejected_edges)):
            total = int(mask_edges.shape[0])
            if total > arguments.curve_cap > 0:
                stride = (total + arguments.curve_cap - 1) // arguments.curve_cap
                curve_edges = mask_edges[::stride]
            else:
                curve_edges = mask_edges
            segments = (
                torch.stack([positions[curve_edges[:, 0]], positions[curve_edges[:, 1]]], dim=1)
                if int(curve_edges.shape[0]) > 0 else torch.zeros((0, 2, 3), device=positions.device)
            )
            json_path = output_root / name / _ITERATION_DIR / "nurbs_surface.json"
            write_cut_edge_curves(
                json_path, segments,
                {"representation": name.lower(), "edge_total": total, "edge_rendered": int(curve_edges.shape[0])},
            )
            view_paths[name]["boundary_curves_json"] = str(json_path)

    del full_orientation, orientation, partition_a
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
        "batch": "arch/2dgs-coverage-first-surface, Worklog 100",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "input_domain": {"model_surfel_count": total_count, "visible_surfel_count": visible_count},
        "A_wl99_interface_coherent": accounting_a,
        "B_bilateral_interface": accounting_b,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"partition_a": seconds_a, "partition_b": seconds_b, "total": time.time() - started},
    }
    report_path = output_root / "bilateral_interface_region_merge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "A_initial": accounting_a["initial_region_count"], "A_final": accounting_a["final_region_count"],
        "A_largest_fraction": accounting_a["largest_subset_surfel_fraction"],
        "B_initial": accounting_b["initial_region_count"], "B_final": accounting_b["final_region_count"],
        "B_largest_fraction": accounting_b["largest_subset_surfel_fraction"],
    }))


if __name__ == "__main__":
    main()
