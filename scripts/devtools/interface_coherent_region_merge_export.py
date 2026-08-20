"""Worklog 99 -- interface-coherent Surfel Region merge + review export.

Runs, on the SAME trained 2DGS checkpoint and the SAME local candidate
graph, all three partition semantics under comparison:

    A. ORIGINAL_2DGS_SCENE
    B. WL97_REGION_CONCENTRATION            (standalone, unmodified default)
    C. WL98_DISCONTINUITY_FIRST             (standalone, unmodified default)
    D. INTERFACE_COHERENT_PARTITION         (new: WL97 init -> interface merge)
    E. ACCEPTED_REGION_INTERFACE_MERGES     (which cross-region edges merged)
    F. REJECTED_REGION_INTERFACES           (which cross-region edges did not)
    G. MERGE_PROVENANCE_DEPTH               (how many WL97 fragments per final region)

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
from osn_gs.surface.torch_interface_coherent_region_merge import (
    InterfaceCoherentMergeConfig,
    interface_coherent_accounting,
    partition_surfels_interface_coherent,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_WL97_PARTITION = "WL97_REGION_CONCENTRATION_PARTITION"
VIEW_WL98_PARTITION = "WL98_DISCONTINUITY_FIRST_PARTITION"
VIEW_INTERFACE_COHERENT_PARTITION = "INTERFACE_COHERENT_PARTITION"
VIEW_ACCEPTED_INTERFACES = "ACCEPTED_REGION_INTERFACE_MERGES"
VIEW_REJECTED_INTERFACES = "REJECTED_REGION_INTERFACES"
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
    print(f"[interface-coherent merge] {message}", flush=True)


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

        _progress("[B] WL97 region-concentration partition (standalone baseline)")
        started_b = time.time()
        partition_b = partition_surfels_region_coherent(orientation, RegionCoherenceConfig(local=local_config), progress=_progress)
        accounting_b = region_coherent_accounting(partition_b)
        seconds_b = time.time() - started_b
        _progress(f"[B] done in {seconds_b:.1f}s -> {accounting_b['subset_count']} subsets, "
                  f"largest_fraction={accounting_b['largest_subset_surfel_fraction']:.4f}")

        _progress("[C] WL98 discontinuity-first partition (standalone baseline)")
        started_c = time.time()
        partition_c = partition_surfels_discontinuity_first(orientation, DiscontinuityFirstConfig(local=local_config), progress=_progress)
        accounting_c = discontinuity_first_accounting(partition_c)
        seconds_c = time.time() - started_c
        _progress(f"[C] done in {seconds_c:.1f}s -> {accounting_c['subset_count']} subsets, "
                  f"largest_fraction={accounting_c['largest_subset_surfel_fraction']:.4f}")

        _progress("[D] interface-coherent region merge (new)")
        started_d = time.time()
        merge_config = InterfaceCoherentMergeConfig(
            local=local_config,
            region=RegionCoherenceConfig(local=local_config, require_positional_continuity=True),
        )
        partition_d = partition_surfels_interface_coherent(orientation, merge_config, progress=_progress)
        accounting_d = interface_coherent_accounting(partition_d)
        seconds_d = time.time() - started_d
        _progress(f"[D] done in {seconds_d:.1f}s -> initial_regions={accounting_d['initial_region_count']} "
                  f"final_regions={accounting_d['final_region_count']} "
                  f"largest_fraction={accounting_d['largest_subset_surfel_fraction']:.4f} "
                  f"merges_applied={accounting_d['merges_applied']}")

        # --- colors ---
        wl97_colors = _subset_partition_colors(partition_b.subset_ids)
        wl98_colors = _subset_partition_colors(partition_c.subset_ids)
        interface_partition_colors = _subset_partition_colors(partition_d.subset_ids)

        accepted_edges = partition_d.graph.candidate_edges[partition_d.accepted_merge_edges_mask]
        rejected_edges = partition_d.graph.candidate_edges[partition_d.rejected_interface_edges_mask]
        accepted_colors = _edge_highlight_colors(visible_count, accepted_edges, _UNCUT_RGB, _ACCEPTED_RGB, device)
        rejected_colors = _edge_highlight_colors(visible_count, rejected_edges, _UNCUT_RGB, _REJECTED_RGB, device)

        # merge-depth: for every final region, how many INITIAL Worklog 97
        # regions were stitched together to form it (0 or 1 fragment merged
        # in => the region needed no interface evidence at all).
        final_of_initial = partition_d.final_region_of_initial
        _, fragment_inverse, fragment_counts = torch.unique(final_of_initial, return_inverse=True, return_counts=True)
        fragments_per_final_root_index = (fragment_counts - 1).clamp_min(0).to(torch.float32)
        depth_per_initial_region = fragments_per_final_root_index[fragment_inverse]
        depth_per_surfel = depth_per_initial_region[partition_d.initial_region_ids]
        depth_reference = max(float(depth_per_surfel.max()), 1.0)
        merge_depth_colors = _ramp(depth_per_surfel / depth_reference, _LOW_RGB, _HIGH_RGB)

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_WL97_PARTITION: _rgb_to_f_dc(wl97_colors),
            VIEW_WL98_PARTITION: _rgb_to_f_dc(wl98_colors),
            VIEW_INTERFACE_COHERENT_PARTITION: _rgb_to_f_dc(interface_partition_colors),
            VIEW_ACCEPTED_INTERFACES: _rgb_to_f_dc(accepted_colors),
            VIEW_REJECTED_INTERFACES: _rgb_to_f_dc(rejected_colors),
            VIEW_MERGE_DEPTH: _rgb_to_f_dc(merge_depth_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        for name, mask_edges in (
            (VIEW_ACCEPTED_INTERFACES, accepted_edges),
            (VIEW_REJECTED_INTERFACES, rejected_edges),
        ):
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

    del full_orientation, orientation, partition_b, partition_c, partition_d
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
        "batch": "arch/2dgs-coverage-first-surface, Worklog 99",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "input_domain": {"model_surfel_count": total_count, "visible_surfel_count": visible_count},
        "B_wl97_region_concentration": accounting_b,
        "C_wl98_discontinuity_first": accounting_c,
        "D_interface_coherent": accounting_d,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {
            "partition_b": seconds_b, "partition_c": seconds_c, "partition_d": seconds_d,
            "total": time.time() - started,
        },
    }
    report_path = output_root / "interface_coherent_region_merge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "B_subset_count": accounting_b["subset_count"], "C_subset_count": accounting_c["subset_count"],
        "D_initial_region_count": accounting_d["initial_region_count"], "D_final_region_count": accounting_d["final_region_count"],
        "B_largest_fraction": accounting_b["largest_subset_surfel_fraction"],
        "C_largest_fraction": accounting_c["largest_subset_surfel_fraction"],
        "D_largest_fraction": accounting_d["largest_subset_surfel_fraction"],
    }))


if __name__ == "__main__":
    main()
