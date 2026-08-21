"""Worklog 103 -- Positively Observation-Supported Visible Adjacency + review export.

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-102:

    A. ORIGINAL_2DGS_SCENE
    B. POSITIVE_VISIBLE_ADJACENCY_SUPPORT
    C. UNKNOWN_SPATIAL_RELATIONS
    D. OCCLUDED_VISIBLE_TERMINATIONS
    E. KNOWN_FREE_SPACE_TERMINATIONS
    F. VISIBLE_GEOMETRIC_DISCONTINUITIES
    G. POSITIVE_OBSERVATION_VISIBLE_COMPONENTS
    H. WORKLOG102_MAXIMAL_SPATIAL_BASELINE

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
from maximal_visible_connectivity_export import (  # noqa: E402
    load_all_train_cameras,
    build_surfel_observation_evidence,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_maximal_visible_connectivity import (
    CUT_KNOWN_FREE_SPACE,
    CUT_OCCLUDED_DOMAIN,
    CUT_POSITIONAL_SHEET_SEPARATION,
    CUT_VISIBLE_GEOMETRIC_DISCONTINUITY,
    MaximalVisibleConnectivityConfig,
    maximal_visible_connectivity_accounting,
    partition_maximal_visible_components,
)
from osn_gs.surface.torch_positive_visible_adjacency import (
    RELATION_STATES,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_POSITIVE_VISIBLE_CONTINUATION,
    STATE_UNKNOWN_NO_POSITIVE_OBSERVATION,
    STATE_UNRESOLVED_CONFLICT,
    PositiveVisibleAdjacencyConfig,
    partition_positive_visible_adjacency,
    positive_visible_adjacency_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_POSITIVE_SUPPORT = "POSITIVE_VISIBLE_ADJACENCY_SUPPORT"
VIEW_UNKNOWN_RELATIONS = "UNKNOWN_SPATIAL_RELATIONS"
VIEW_OCCLUDED_TERMINATIONS = "OCCLUDED_VISIBLE_TERMINATIONS"
VIEW_FREE_SPACE_TERMINATIONS = "KNOWN_FREE_SPACE_TERMINATIONS"
VIEW_DISCONTINUITIES = "VISIBLE_GEOMETRIC_DISCONTINUITIES"
VIEW_POSITIVE_COMPONENTS = "POSITIVE_OBSERVATION_VISIBLE_COMPONENTS"
VIEW_WL102_BASELINE = "WORKLOG102_MAXIMAL_SPATIAL_BASELINE"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532
_UNCUT_RGB = (0.12, 0.13, 0.16)
_POSITIVE_RGB = (0.2, 0.9, 0.3)
_UNKNOWN_RGB = (0.5, 0.5, 0.5)
_OCCLUDED_RGB = (1.0, 0.15, 0.05)
_FREE_SPACE_RGB = (0.15, 0.55, 1.0)
_DISCONTINUITY_RGB = (1.0, 0.85, 0.1)


def _progress(message: str) -> None:
    print(f"[positive visible adjacency] {message}", flush=True)


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

        _progress("[B] positively observation-supported visible adjacency (new)")
        started_b = time.time()
        config_b = PositiveVisibleAdjacencyConfig(local=local_config)
        result_b = partition_positive_visible_adjacency(orientation, observation_evidence, config_b, progress=_progress)
        accounting_b = positive_visible_adjacency_accounting(result_b)
        seconds_b = time.time() - started_b
        _progress(f"[B] done in {seconds_b:.1f}s -> {accounting_b['visible_component_count']} components "
                  f"largest={accounting_b['largest_component_surfel_fraction']:.4f} "
                  f"states={accounting_b['relation_state_counts']}")

        _progress("[H] Worklog 102 maximal-spatial baseline (unmodified, for comparison only)")
        started_h = time.time()
        config_h = MaximalVisibleConnectivityConfig(local=local_config)
        result_h = partition_maximal_visible_components(orientation, observation_evidence, config_h, progress=_progress)
        accounting_h = maximal_visible_connectivity_accounting(result_h)
        seconds_h = time.time() - started_h
        _progress(f"[H] done in {seconds_h:.1f}s -> largest={accounting_h['largest_component_surfel_fraction']:.4f}")

        # --- colors ---
        positive_component_colors = _subset_partition_colors(result_b.subset_ids)
        wl102_colors = _subset_partition_colors(result_h.subset_ids)

        spatial_mask = result_b.graph.spatial_edge_mask
        state = result_b.relation_state
        positive_edges = result_b.graph.candidate_edges[spatial_mask & (state == RELATION_STATES.index(STATE_POSITIVE_VISIBLE_CONTINUATION))]
        unknown_edges = result_b.graph.candidate_edges[spatial_mask & (state == RELATION_STATES.index(STATE_UNKNOWN_NO_POSITIVE_OBSERVATION))]
        occluded_edges = result_b.graph.candidate_edges[spatial_mask & (state == RELATION_STATES.index(STATE_CUT_OCCLUDED_DOMAIN))]
        free_edges = result_b.graph.candidate_edges[spatial_mask & (state == RELATION_STATES.index(STATE_CUT_KNOWN_FREE_SPACE))]
        discontinuity_edges = result_b.graph.candidate_edges[
            spatial_mask & (
                (state == RELATION_STATES.index(STATE_CUT_VISIBLE_DISCONTINUITY))
                | (state == RELATION_STATES.index(STATE_CUT_POSITIONAL_SHEET_SEPARATION))
            )
        ]
        conflict_edges = result_b.graph.candidate_edges[spatial_mask & (state == RELATION_STATES.index(STATE_UNRESOLVED_CONFLICT))]

        positive_colors = _edge_highlight_colors(visible_count, positive_edges, _UNCUT_RGB, _POSITIVE_RGB, device)
        unknown_colors = _edge_highlight_colors(visible_count, unknown_edges, _UNCUT_RGB, _UNKNOWN_RGB, device)
        occluded_colors = _edge_highlight_colors(visible_count, occluded_edges, _UNCUT_RGB, _OCCLUDED_RGB, device)
        free_colors = _edge_highlight_colors(visible_count, free_edges, _UNCUT_RGB, _FREE_SPACE_RGB, device)
        discontinuity_colors = _edge_highlight_colors(visible_count, discontinuity_edges, _UNCUT_RGB, _DISCONTINUITY_RGB, device)

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_POSITIVE_SUPPORT: _rgb_to_f_dc(positive_colors),
            VIEW_UNKNOWN_RELATIONS: _rgb_to_f_dc(unknown_colors),
            VIEW_OCCLUDED_TERMINATIONS: _rgb_to_f_dc(occluded_colors),
            VIEW_FREE_SPACE_TERMINATIONS: _rgb_to_f_dc(free_colors),
            VIEW_DISCONTINUITIES: _rgb_to_f_dc(discontinuity_colors),
            VIEW_POSITIVE_COMPONENTS: _rgb_to_f_dc(positive_component_colors),
            VIEW_WL102_BASELINE: _rgb_to_f_dc(wl102_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        for name, edges in (
            (VIEW_POSITIVE_SUPPORT, positive_edges), (VIEW_UNKNOWN_RELATIONS, unknown_edges),
            (VIEW_OCCLUDED_TERMINATIONS, occluded_edges), (VIEW_FREE_SPACE_TERMINATIONS, free_edges),
            (VIEW_DISCONTINUITIES, discontinuity_edges),
        ):
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

    del full_orientation, orientation, result_h
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

    spatial_edge_count = int(result_b.graph.spatial_edge_mask.sum())
    accounting_b["candidate_edges_with_at_least_one_co_observation"] = spatial_edge_count - accounting_b["relation_state_counts"][STATE_UNKNOWN_NO_POSITIVE_OBSERVATION] - accounting_b["relation_state_counts"][STATE_CUT_VISIBLE_DISCONTINUITY] - accounting_b["relation_state_counts"][STATE_CUT_POSITIONAL_SHEET_SEPARATION]

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 103",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "input_domain": {"model_surfel_count": total_count, "visible_surfel_count": visible_count},
        "camera_meta": camera_meta,
        "B_positive_visible_adjacency": accounting_b,
        "H_worklog102_maximal_spatial_baseline": accounting_h,
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"observation_evidence": time.time() - started - seconds_b - seconds_h, "partition_b": seconds_b, "partition_h": seconds_h, "total": time.time() - started},
    }
    report_path = output_root / "positive_visible_adjacency_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "B_component_count": accounting_b["visible_component_count"], "B_largest_fraction": accounting_b["largest_component_surfel_fraction"],
        "B_relation_state_counts": accounting_b["relation_state_counts"],
        "H_component_count": accounting_h["visible_component_count"], "H_largest_fraction": accounting_h["largest_component_surfel_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
