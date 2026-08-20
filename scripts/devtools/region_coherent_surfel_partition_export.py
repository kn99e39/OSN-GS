"""Worklog 97 -- WL96 pairwise-CC vs region-coherent Surfel Subset partition + review export.

Runs BOTH partition semantics on the SAME trained 2DGS checkpoint and the SAME
local candidate graph, isolating exactly one variable (architecture directive
section 11):

    A. WL96_PAIRWISE_CC
       intrinsic 2DGS normals + local pairwise accepted edges + plain
       connected components (osn_gs.surface.torch_coverage_first_subset_partition)

    B. REGION_COHERENT_2DGS_PARTITION
       same intrinsic 2DGS normals + same local candidate graph +
       region-level anti-chaining + coverage ownership propagation
       (osn_gs.surface.torch_region_coherent_surfel_partition)

Neither Trustable-surfel estimation, latent surface construction, nor NURBS
fitting is implemented here. See both partition modules' docstrings for the
contracts this script only replays and exports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import (  # noqa: E402 -- sys.path set above
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
from osn_gs.surface.torch_coverage_first_subset_partition import (
    CoverageFirstPartitionConfig,
    partition_accounting,
    partition_gaussian_subsets,
)
from osn_gs.surface.torch_gaussian_surface_orientation import unsigned_normal_alignment
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    PARTITION_ROLES,
    ROLE_ISOLATED_FALLBACK,
    ROLE_OWNERSHIP_PROPAGATED,
    ROLE_STRUCTURAL_CORE,
    RegionCoherenceConfig,
    partition_surfels_region_coherent,
    region_coherent_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "2DGS_ORIGINAL_SCENE"
VIEW_WL96_PARTITION = "WL96_PAIRWISE_CC_PARTITION"
VIEW_REGION_PARTITION = "REGION_COHERENT_PARTITION"
VIEW_DISPERSION = "REGION_ORIENTATION_DISPERSION_VIEW"
VIEW_OWNERSHIP_ROLE = "OWNERSHIP_ROLE_VIEW"
VIEW_ANTI_CHAINING_BOUNDARY = "ANTI_CHAINING_BOUNDARY_VIEW"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

# REGION_ORIENTATION_DISPERSION_VIEW: linear ramp, low dispersion (concentrated,
# single coherent surface) -> high dispersion (a region whose scatter is close
# to the concentration floor, i.e. it is barely coherent).
_LOW_DISPERSION_RGB = (0.10, 0.35, 0.95)
_HIGH_DISPERSION_RGB = (1.0, 0.55, 0.05)

# OWNERSHIP_ROLE_VIEW: three fixed, maximally distinguishable colors -- never
# chosen to make the result "look good".
_ROLE_COLORS = {
    ROLE_STRUCTURAL_CORE: (0.15, 0.75, 0.25),  # green
    ROLE_OWNERSHIP_PROPAGATED: (0.95, 0.85, 0.10),  # yellow
    ROLE_ISOLATED_FALLBACK: (0.85, 0.10, 0.15),  # red
}

# ANTI_CHAINING_BOUNDARY_VIEW: same ramp family as WL96's NORMAL_CUT_VIEW.
_FULLY_REJECTED_RGB = (1.0, 0.15, 0.05)
_UNREJECTED_RGB = (0.12, 0.13, 0.16)


def _progress(message: str) -> None:
    print(f"[region-coherent partition] {message}", flush=True)


def _subset_partition_colors(subset_ids: torch.Tensor) -> torch.Tensor:
    identifiers = subset_ids.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


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
    parser.add_argument(
        "--spacing-multiplier", type=float, default=CoverageFirstPartitionConfig.spatial_connect_spacing_multiplier
    )
    parser.add_argument(
        "--normal-alignment", type=float, default=CoverageFirstPartitionConfig.normal_compatibility_min_alignment
    )
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
        raise ValueError(
            f"{arguments.checkpoint} is not a 2DGS surfel checkpoint (primitive={primitive!r}, "
            f"scale_dim={getattr(model, 'scale_dim', None)!r}). Refusing to substitute a volumetric checkpoint."
        )
    total_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    visible_count = int(visible_selector.shape[0])
    _progress(f"model surfels={total_count} visible={visible_count} primitive={primitive} iteration={payload.get('iteration')}")

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
        region_config = RegionCoherenceConfig(local=local_config)

        _progress(f"[A] WL96 pairwise-CC partition, local={local_config.payload()}")
        started_a = time.time()
        partition_a = partition_gaussian_subsets(orientation, local_config, progress=_progress)
        accounting_a = partition_accounting(partition_a)
        seconds_a = time.time() - started_a
        _progress(f"[A] done in {seconds_a:.1f}s -> {accounting_a['subset_count']} subsets, "
                  f"largest_fraction={accounting_a['largest_subset_gaussian_fraction']:.4f}")

        _progress(f"[B] region-coherent partition, concentration_floor={region_config.concentration_floor():.6f}")
        started_b = time.time()
        partition_b = partition_surfels_region_coherent(orientation, region_config, progress=_progress)
        accounting_b = region_coherent_accounting(partition_b)
        seconds_b = time.time() - started_b
        _progress(f"[B] done in {seconds_b:.1f}s -> {accounting_b['subset_count']} subsets, "
                  f"largest_fraction={accounting_b['largest_subset_surfel_fraction']:.4f}")

        # --- former-giant-component decomposition (section 7/10) -----------
        giant_subset_id = int(torch.argmax(partition_a.subset_sizes))
        giant_member_mask = partition_a.subset_ids == giant_subset_id
        giant_member_count = int(giant_member_mask.sum())
        descendant_ids, descendant_sizes = torch.unique(partition_b.subset_ids[giant_member_mask], return_counts=True)
        descendant_order = torch.argsort(descendant_sizes, descending=True, stable=True)
        giant_decomposition = {
            "wl96_giant_subset_size": giant_member_count,
            "wl96_giant_subset_fraction": float(giant_member_count) / visible_count,
            "descendant_final_subset_count": int(descendant_ids.shape[0]),
            "descendant_sizes_desc": descendant_sizes[descendant_order].tolist()[:32],
            "largest_descendant_fraction_of_original_giant": float(descendant_sizes.max()) / giant_member_count
            if giant_member_count else 0.0,
            "largest_descendant_fraction_of_scene": float(descendant_sizes.max()) / visible_count
            if giant_member_count else 0.0,
        }

        # --- WL96 singleton/fallback fate (section 8/9) ---------------------
        wl96_singleton_subset_ids = torch.nonzero(partition_a.subset_sizes == 1, as_tuple=False).reshape(-1)
        wl96_singleton_mask = torch.isin(partition_a.subset_ids, wl96_singleton_subset_ids)
        singleton_role_counts = torch.bincount(
            partition_b.partition_role[wl96_singleton_mask].reshape(-1).to(torch.int64), minlength=len(PARTITION_ROLES)
        )
        singleton_fate = {
            "wl96_singleton_surfel_count": int(wl96_singleton_mask.sum()),
            "became_structural_core": int(singleton_role_counts[PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)]),
            "became_ownership_propagated": int(singleton_role_counts[PARTITION_ROLES.index(ROLE_OWNERSHIP_PROPAGATED)]),
            "remained_isolated_fallback": int(singleton_role_counts[PARTITION_ROLES.index(ROLE_ISOLATED_FALLBACK)]),
        }

        # --- review colors ---------------------------------------------------
        wl96_colors = _subset_partition_colors(partition_a.subset_ids)
        region_colors = _subset_partition_colors(partition_b.subset_ids)

        concentration_per_surfel = partition_b.region_concentration[partition_b.subset_ids]
        low_rgb = torch.tensor(_LOW_DISPERSION_RGB, device=positions.device).reshape(1, 3)
        high_rgb = torch.tensor(_HIGH_DISPERSION_RGB, device=positions.device).reshape(1, 3)
        # Color scale reference = the UNWEIGHTED per-region concentration
        # p05/p95 already reported in accounting_b (region_orientation), not a
        # value picked from this rendering pass: the same two numbers the
        # user sees in the JSON report define where the ramp's endpoints
        # sit. Using the per-surfel population directly would let a handful
        # of huge regions dominate the reference range; the per-REGION
        # statistic treats every region as one data point, matching what
        # "how much do regions differ in coherence" actually means.
        conc_low = float(accounting_b["region_orientation"]["concentration_p05"])
        conc_high = float(accounting_b["region_orientation"]["concentration_p95"])
        conc_span = max(conc_high - conc_low, 1e-6)
        dispersion_ratio = (1.0 - (concentration_per_surfel - conc_low) / conc_span).clamp(0.0, 1.0)
        dispersion_colors = low_rgb + dispersion_ratio.reshape(-1, 1) * (high_rgb - low_rgb)

        role_colors = torch.zeros((visible_count, 3), device=positions.device)
        for role_name, rgb in _ROLE_COLORS.items():
            mask = partition_b.partition_role == PARTITION_ROLES.index(role_name)
            role_colors[mask] = torch.tensor(rgb, device=positions.device)

        rejected_edges = partition_b.anti_chaining_boundary_edges
        accepted_edges_b = partition_b.graph.accepted_edges
        rejected_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        accepted_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        for target, source in ((rejected_degree, rejected_edges), (accepted_degree, accepted_edges_b)):
            if int(source.shape[0]) > 0:
                ones = torch.ones((int(source.shape[0]),), dtype=torch.float32, device=positions.device)
                target.index_add_(0, source[:, 0], ones)
                target.index_add_(0, source[:, 1], ones)
        rejected_ratio = torch.where(
            accepted_degree > 0, rejected_degree / accepted_degree.clamp_min(1.0), torch.zeros_like(rejected_degree)
        )
        unrej_rgb = torch.tensor(_UNREJECTED_RGB, device=positions.device).reshape(1, 3)
        rej_rgb = torch.tensor(_FULLY_REJECTED_RGB, device=positions.device).reshape(1, 3)
        boundary_colors = unrej_rgb + rejected_ratio.reshape(-1, 1) * (rej_rgb - unrej_rgb)

        rejected_total = int(rejected_edges.shape[0])
        if rejected_total > arguments.curve_cap > 0:
            stride = (rejected_total + arguments.curve_cap - 1) // arguments.curve_cap
            curve_edges = rejected_edges[::stride]
        else:
            curve_edges = rejected_edges
        boundary_segments = torch.stack([positions[curve_edges[:, 0]], positions[curve_edges[:, 1]]], dim=1) \
            if int(curve_edges.shape[0]) > 0 else torch.zeros((0, 2, 3), device=positions.device)

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_WL96_PARTITION: _rgb_to_f_dc(wl96_colors),
            VIEW_REGION_PARTITION: _rgb_to_f_dc(region_colors),
            VIEW_DISPERSION: _rgb_to_f_dc(dispersion_colors),
            VIEW_OWNERSHIP_ROLE: _rgb_to_f_dc(role_colors),
            VIEW_ANTI_CHAINING_BOUNDARY: _rgb_to_f_dc(boundary_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        boundary_json_path = output_root / VIEW_ANTI_CHAINING_BOUNDARY / _ITERATION_DIR / "nurbs_surface.json"
        write_cut_edge_curves(
            boundary_json_path, boundary_segments,
            {
                "representation": "anti_chaining_rejected_merge_edges",
                "rejected_total": rejected_total, "rejected_rendered": int(curve_edges.shape[0]),
                "selection": "uniform stride over the canonical accepted-edge order, not a spatial crop",
            },
        )
        view_paths[VIEW_ANTI_CHAINING_BOUNDARY]["boundary_curves_json"] = str(boundary_json_path)

    del full_orientation, orientation, partition_a, partition_b
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
        "batch": "arch/2dgs-coverage-first-surface, Worklog 97",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "input_domain": {"model_surfel_count": total_count, "visible_surfel_count": visible_count},
        "A_wl96_pairwise_cc": accounting_a,
        "B_region_coherent": accounting_b,
        "former_giant_component_decomposition": giant_decomposition,
        "wl96_singleton_fallback_fate": singleton_fate,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"partition_a": seconds_a, "partition_b": seconds_b, "total": time.time() - started},
    }
    report_path = output_root / "region_coherent_partition_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "A_subset_count": accounting_a["subset_count"], "B_subset_count": accounting_b["subset_count"],
        "A_largest_fraction": accounting_a["largest_subset_gaussian_fraction"],
        "B_largest_fraction": accounting_b["largest_subset_surfel_fraction"],
    }))


if __name__ == "__main__":
    main()
