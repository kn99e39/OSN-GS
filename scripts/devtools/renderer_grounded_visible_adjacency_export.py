"""Worklog 106 -- Renderer-Grounded Visible Adjacency + controlled comparison.

Does NOT modify torch_positive_visible_adjacency.py (Worklog 103),
torch_node_level_observability_accounting.py (Worklog 104), or
torch_surfel_contribution_diagnostics.py (Worklog 105). All three are
replayed/reused unmodified. Builds the spatial candidate graph exactly ONCE
and evaluates it under two DISTINCT endpoint-eligibility rules:

    A. WL103_CENTER_GROUNDED (Phase-C center on_observed_surface, unmodified
       Worklog 103 replay)
    B. RENDERER_GROUNDED (Worklog 105 official-renderer accepted
       alpha-compositing contribution, this batch's new module)

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-105:

    A. ORIGINAL_2DGS_SCENE
    B. WL103_CENTER_GROUNDED_COMPONENTS
    C. RENDERER_CONTRIBUTING_PRIMITIVES
    D. SAME_VIEW_CO_CONTRIBUTION_RELATIONS
    E. RENDERER_GROUNDED_VISIBLE_ADJACENCY
    F. RENDERER_GROUNDED_VISIBLE_COMPONENTS
    G. REMAINING_SINGLETON_CAUSE_VIEW
    H. OCCLUDED_FREE_SPACE_TERMINATION_VIEW

Neither Trust, latent surface, NURBS fitting, NURBS decomposition, occluded
surface generation, nor uncertain Gaussian proposal is implemented here.
"""

from __future__ import annotations

import argparse
import json
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
from maximal_visible_connectivity_export import build_surfel_observation_evidence, load_all_train_cameras  # noqa: E402
from osn_gs.render.torch_surfel_contribution_diagnostics import compute_renderer_contribution_for_view
from osn_gs.surface.torch_coverage_first_subset_partition import (
    CoverageFirstPartitionConfig,
    build_candidate_graph,
    _connected_component_roots,
)
from osn_gs.surface.torch_positive_visible_adjacency import (
    PositiveVisibleAdjacencyConfig,
    PositiveVisibleAdjacencyResult,
    compute_positive_visible_adjacency_evidence,
    positive_visible_adjacency_accounting,
)
from osn_gs.surface.torch_renderer_grounded_visible_adjacency import (
    RELATION_STATES,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION,
    STATE_UNRESOLVED_CONFLICT,
    RendererGroundedVisibleAdjacencyConfig,
    RendererGroundedVisibleAdjacencyResult,
    compute_renderer_grounded_visible_adjacency_evidence,
    renderer_grounded_visible_adjacency_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_WL103_COMPONENTS = "WL103_CENTER_GROUNDED_COMPONENTS"
VIEW_CONTRIBUTING_PRIMITIVES = "RENDERER_CONTRIBUTING_PRIMITIVES"
VIEW_CO_CONTRIBUTION_RELATIONS = "SAME_VIEW_CO_CONTRIBUTION_RELATIONS"
VIEW_ADJACENCY = "RENDERER_GROUNDED_VISIBLE_ADJACENCY"
VIEW_COMPONENTS = "RENDERER_GROUNDED_VISIBLE_COMPONENTS"
VIEW_SINGLETON_CAUSE = "REMAINING_SINGLETON_CAUSE_VIEW"
VIEW_TERMINATION = "OCCLUDED_FREE_SPACE_TERMINATION_VIEW"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532
_UNCUT_RGB = (0.08, 0.09, 0.11)
_CONTRIBUTING_RGB = (1.0, 0.55, 0.0)
_POSITIVE_RGB = (0.2, 0.9, 0.3)
_OCCLUDED_RGB = (1.0, 0.15, 0.05)
_FREE_SPACE_RGB = (0.15, 0.55, 1.0)

# remaining-singleton causes for renderer-contributing WL103 singletons
CAUSE_NO_COCONTRIBUTING_NEIGHBOR = "NO_SAME_VIEW_COCONTRIBUTING_SPATIAL_NEIGHBOR"
CAUSE_COCONTRIBUTING_BUT_CORRIDOR_FAILS = "COCONTRIBUTING_NEIGHBOR_EXISTS_BUT_CORRIDOR_FAILS"
CAUSE_HARD_CONTRADICTION = "HARD_OBSERVATION_CONTRADICTION"
CAUSE_GEOMETRIC_DISCONTINUITY = "GEOMETRIC_DISCONTINUITY"
CAUSE_POSITIONAL_SEPARATION = "POSITIONAL_SHEET_SEPARATION"
CAUSE_CONFLICT = "OBSERVATION_CONFLICT"
CAUSE_NOT_RENDERER_CONTRIBUTING = "NOT_RENDERER_CONTRIBUTING_AT_ALL"
CAUSE_CATEGORIES = (
    CAUSE_NOT_RENDERER_CONTRIBUTING, CAUSE_NO_COCONTRIBUTING_NEIGHBOR, CAUSE_COCONTRIBUTING_BUT_CORRIDOR_FAILS,
    CAUSE_HARD_CONTRADICTION, CAUSE_GEOMETRIC_DISCONTINUITY, CAUSE_POSITIONAL_SEPARATION, CAUSE_CONFLICT,
)
_CAUSE_RGB = {
    CAUSE_NOT_RENDERER_CONTRIBUTING: (0.6, 0.05, 0.05),
    CAUSE_NO_COCONTRIBUTING_NEIGHBOR: (0.95, 0.55, 0.05),
    CAUSE_COCONTRIBUTING_BUT_CORRIDOR_FAILS: (0.95, 0.9, 0.1),
    CAUSE_HARD_CONTRADICTION: (0.15, 0.55, 1.0),
    CAUSE_GEOMETRIC_DISCONTINUITY: (1.0, 0.15, 0.55),
    CAUSE_POSITIONAL_SEPARATION: (0.7, 0.15, 0.9),
    CAUSE_CONFLICT: (0.55, 0.2, 0.85),
}
_NOT_SINGLETON_RGB = (0.08, 0.09, 0.11)


def _progress(message: str) -> None:
    print(f"[renderer grounded visible adjacency] {message}", flush=True)


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


def _category_colors(category: torch.Tensor, palette: dict, categories: tuple, base_rgb) -> torch.Tensor:
    count = int(category.shape[0])
    device = category.device
    colors = torch.tensor(base_rgb, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
    for index, name in enumerate(categories):
        rgb = palette.get(name)
        if rgb is None:
            continue
        colors[category == index] = torch.tensor(rgb, dtype=torch.float32, device=device)
    return colors


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
    parser.add_argument("--preview-camera-images", default=None)
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
    with torch.no_grad():
        observation_evidence = build_surfel_observation_evidence(cameras, model, rasterizer, depth_epsilon=arguments.depth_epsilon, progress=_progress)
    _progress(f"observation evidence built over {len(observation_evidence.views)} views")

    _progress("[contribution] per-view renderer contribution masks (Worklog 105, unmodified)")
    started_contrib = time.time()
    contributing_masks_full: list[torch.Tensor] = []
    ever_contributed_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    contributing_view_count_full = torch.zeros((total_model_count,), dtype=torch.int32, device=model.device)
    for index, camera in enumerate(cameras):
        contributed, _weight, _package = compute_renderer_contribution_for_view(camera, model, rasterizer)
        contributing_masks_full.append(contributed.detach())
        ever_contributed_full |= contributed
        contributing_view_count_full += contributed.to(torch.int32)
        del _package
        if index % 20 == 0:
            _progress(f"contribution view {index + 1}/{len(cameras)}")
    seconds_contrib = time.time() - started_contrib
    _progress(f"[contribution] done in {seconds_contrib:.1f}s, ever_contributed={int(ever_contributed_full.sum())}/{total_model_count}")

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
        contributing_masks = [mask[visible_selector] for mask in contributing_masks_full]
        ever_contributed = ever_contributed_full[visible_selector]
        contributing_view_count = contributing_view_count_full[visible_selector]

        local_config = CoverageFirstPartitionConfig()

        _progress("[shared] build_candidate_graph ONCE (unmodified, shared by both A and B)")
        graph = build_candidate_graph(orientation, local_config, progress=_progress)

        # --- A: Worklog 103 replay (unmodified, center-grounded) ---
        _progress("[A] WL103 center-grounded replay")
        config_a = PositiveVisibleAdjacencyConfig(local=local_config)
        evidence_a = compute_positive_visible_adjacency_evidence(orientation, observation_evidence, graph, config_a, progress=_progress)
        kept_a = graph.candidate_edges[evidence_a["positive_visible_edges_mask"]]
        roots_a = _connected_component_roots(count, kept_a, local_config)
        unique_a, inverse_a, counts_a = torch.unique(roots_a, return_inverse=True, return_counts=True)
        order_a = torch.argsort(counts_a, descending=True, stable=True)
        subset_id_of_position_a = torch.empty_like(order_a)
        subset_id_of_position_a[order_a] = torch.arange(int(order_a.shape[0]), dtype=order_a.dtype, device=device)
        subset_ids_a = subset_id_of_position_a[inverse_a]
        subset_sizes_a = counts_a[order_a]
        result_a = PositiveVisibleAdjacencyResult(
            subset_ids=subset_ids_a, subset_count=int(order_a.shape[0]), subset_sizes=subset_sizes_a,
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            relation_state=evidence_a["relation_state"], positive_visible_edges_mask=evidence_a["positive_visible_edges_mask"],
            normal_gradient_magnitude=evidence_a["normal_gradient_magnitude"], residual_threshold=evidence_a["residual_threshold"],
            config=config_a,
        )
        accounting_a = positive_visible_adjacency_accounting(result_a)
        singleton_mask_a = subset_sizes_a[subset_ids_a] == 1
        _progress(f"[A] {accounting_a['visible_component_count']} components largest={accounting_a['largest_component_surfel_fraction']:.4f}")

        # --- B: renderer-grounded (this batch, new module) ---
        _progress("[B] renderer-grounded visible adjacency")
        config_b = RendererGroundedVisibleAdjacencyConfig(local=local_config)
        evidence_b = compute_renderer_grounded_visible_adjacency_evidence(orientation, observation_evidence, contributing_masks, graph, config_b, progress=_progress)
        kept_b = graph.candidate_edges[evidence_b["positive_visible_edges_mask"]]
        roots_b = _connected_component_roots(count, kept_b, local_config)
        unique_b, inverse_b, counts_b = torch.unique(roots_b, return_inverse=True, return_counts=True)
        order_b = torch.argsort(counts_b, descending=True, stable=True)
        subset_id_of_position_b = torch.empty_like(order_b)
        subset_id_of_position_b[order_b] = torch.arange(int(order_b.shape[0]), dtype=order_b.dtype, device=device)
        subset_ids_b = subset_id_of_position_b[inverse_b]
        subset_sizes_b = counts_b[order_b]
        result_b = RendererGroundedVisibleAdjacencyResult(
            subset_ids=subset_ids_b, subset_count=int(order_b.shape[0]), subset_sizes=subset_sizes_b,
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            relation_state=evidence_b["relation_state"], positive_visible_edges_mask=evidence_b["positive_visible_edges_mask"],
            normal_gradient_magnitude=evidence_b["normal_gradient_magnitude"], residual_threshold=evidence_b["residual_threshold"],
            config=config_b,
        )
        accounting_b = renderer_grounded_visible_adjacency_accounting(result_b)
        singleton_mask_b = subset_sizes_b[subset_ids_b] == 1
        _progress(f"[B] {accounting_b['visible_component_count']} components largest={accounting_b['largest_component_surfel_fraction']:.4f} states={accounting_b['relation_state_counts']}")

        # singleton stats split by renderer-contributing vs not
        singleton_contributing_a_count = int((singleton_mask_a & ever_contributed).sum())
        singleton_noncontributing_a_count = int((singleton_mask_a & ~ever_contributed).sum())
        singleton_contributing_b_count = int((singleton_mask_b & ever_contributed).sum())
        singleton_noncontributing_b_count = int((singleton_mask_b & ~ever_contributed).sum())

        # --- key causal test (directive section 12) ---
        wl103_singleton_and_contributing = singleton_mask_a & ever_contributed
        wl103_singleton_and_contributing_count = int(wl103_singleton_and_contributing.sum())
        gained_edge_mask = wl103_singleton_and_contributing & (~singleton_mask_b)
        gained_edge_count = int(gained_edge_mask.sum())
        still_singleton_mask = wl103_singleton_and_contributing & singleton_mask_b
        still_singleton_count = int(still_singleton_mask.sum())
        _progress(f"[causal] WL103-singleton & renderer-contributing = {wl103_singleton_and_contributing_count}; "
                  f"gained edge in B = {gained_edge_count}; still singleton in B = {still_singleton_count}")

        # --- remaining-singleton cause attribution (this batch's own, over WL103-singleton & contributing surfels still singleton in B) ---
        spatial_mask = graph.spatial_edge_mask
        edges = graph.candidate_edges

        def _scatter_or(bool_edge_array: torch.Tensor) -> torch.Tensor:
            node_flag = torch.zeros((count,), dtype=torch.bool, device=device)
            selected = spatial_mask & bool_edge_array
            if not bool(selected.any()):
                return node_flag
            picked = edges[selected]
            node_flag[picked[:, 0]] = True
            node_flag[picked[:, 1]] = True
            return node_flag

        node_ever_evaluated_b = _scatter_or(evidence_b["ever_evaluated"])
        node_any_positive_pre_geometry_b = _scatter_or(evidence_b["any_positive"])
        node_geometric_cut_b = _scatter_or(evidence_b["cut_reason_residual"] | evidence_b["cut_reason_positional"])
        node_positional_cut_b = _scatter_or(evidence_b["cut_reason_positional"])
        conflict_edge_mask_b = evidence_b["relation_state"] == RELATION_STATES.index(STATE_UNRESOLVED_CONFLICT)
        node_conflict_b = _scatter_or(conflict_edge_mask_b)
        hard_cut_edge_mask_b = (
            (evidence_b["relation_state"] == RELATION_STATES.index(STATE_CUT_KNOWN_FREE_SPACE))
            | (evidence_b["relation_state"] == RELATION_STATES.index(STATE_CUT_OCCLUDED_DOMAIN))
        )
        node_hard_contradiction_b = _scatter_or(hard_cut_edge_mask_b)

        cause = torch.full((count,), CAUSE_CATEGORIES.index(CAUSE_NOT_RENDERER_CONTRIBUTING), dtype=torch.int64, device=device)
        cause = torch.where(ever_contributed, torch.full_like(cause, CAUSE_CATEGORIES.index(CAUSE_NO_COCONTRIBUTING_NEIGHBOR)), cause)
        cause = torch.where(ever_contributed & node_ever_evaluated_b & ~node_any_positive_pre_geometry_b & ~node_conflict_b & ~node_hard_contradiction_b,
                             torch.full_like(cause, CAUSE_CATEGORIES.index(CAUSE_COCONTRIBUTING_BUT_CORRIDOR_FAILS)), cause)
        cause = torch.where(ever_contributed & node_hard_contradiction_b, torch.full_like(cause, CAUSE_CATEGORIES.index(CAUSE_HARD_CONTRADICTION)), cause)
        cause = torch.where(ever_contributed & node_any_positive_pre_geometry_b & node_geometric_cut_b & ~node_positional_cut_b,
                             torch.full_like(cause, CAUSE_CATEGORIES.index(CAUSE_GEOMETRIC_DISCONTINUITY)), cause)
        cause = torch.where(ever_contributed & node_positional_cut_b, torch.full_like(cause, CAUSE_CATEGORIES.index(CAUSE_POSITIONAL_SEPARATION)), cause)
        cause = torch.where(ever_contributed & node_conflict_b & ~node_geometric_cut_b, torch.full_like(cause, CAUSE_CATEGORIES.index(CAUSE_CONFLICT)), cause)

        still_singleton_cause_counts = {
            name: int((cause[still_singleton_mask] == index).sum()) for index, name in enumerate(CAUSE_CATEGORIES)
        }
        _progress(f"[remaining singleton cause] {still_singleton_cause_counts}")

        accounting = {
            "batch": "arch/2dgs-coverage-first-surface, Worklog 106",
            "checkpoint": str(arguments.checkpoint),
            "primitive": primitive,
            "iteration": int(payload.get("iteration", 0)),
            "primitive_accounting": {
                "total_model_surfel_count": total_model_count,
                "visible_domain_surfel_count": visible_count,
                "renderer_contributing_surfel_count": int(ever_contributed.sum()),
                "renderer_noncontributing_surfel_count": int((~ever_contributed).sum()),
                "renderer_grounded_visible_adjacency_connected_surfel_count": int((~singleton_mask_b).sum()),
                "renderer_contributing_but_topologically_isolated_surfel_count": int((ever_contributed & singleton_mask_b).sum()),
            },
            "camera_meta": camera_meta,
            "A_wl103_center_grounded": accounting_a,
            "B_renderer_grounded": accounting_b,
            "singleton_by_contribution_status": {
                "A_singleton_and_contributing": singleton_contributing_a_count,
                "A_singleton_and_noncontributing": singleton_noncontributing_a_count,
                "B_singleton_and_contributing": singleton_contributing_b_count,
                "B_singleton_and_noncontributing": singleton_noncontributing_b_count,
            },
            "causal_test_wl103_singleton_and_contributing": {
                "total": wl103_singleton_and_contributing_count,
                "gained_edge_in_B": gained_edge_count,
                "gained_edge_fraction": gained_edge_count / wl103_singleton_and_contributing_count if wl103_singleton_and_contributing_count else 0.0,
                "still_singleton_in_B": still_singleton_count,
                "still_singleton_fraction": still_singleton_count / wl103_singleton_and_contributing_count if wl103_singleton_and_contributing_count else 0.0,
            },
            "remaining_singleton_cause_breakdown": still_singleton_cause_counts,
            "runtime_seconds": {"contribution_pass": seconds_contrib, "total": time.time() - started},
        }
        _progress(json.dumps({"causal_test": accounting["causal_test_wl103_singleton_and_contributing"]}, indent=2))

        # --- colors ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        wl103_colors = _subset_partition_colors(subset_ids_a)
        b_component_colors = _subset_partition_colors(subset_ids_b)
        contributing_colors = _ramp(ever_contributed.to(torch.float32), _UNCUT_RGB, _CONTRIBUTING_RGB)

        spatial_mask_only = graph.spatial_edge_mask
        state_b = evidence_b["relation_state"]
        co_contribution_edges = graph.candidate_edges[spatial_mask_only & evidence_b["ever_evaluated"]]
        positive_edges_b = graph.candidate_edges[spatial_mask_only & (state_b == RELATION_STATES.index(STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION))]
        occluded_edges_b = graph.candidate_edges[spatial_mask_only & (state_b == RELATION_STATES.index(STATE_CUT_OCCLUDED_DOMAIN))]
        free_edges_b = graph.candidate_edges[spatial_mask_only & (state_b == RELATION_STATES.index(STATE_CUT_KNOWN_FREE_SPACE))]
        termination_edges_b = torch.cat([occluded_edges_b, free_edges_b], dim=0)

        co_contribution_colors = _edge_highlight_colors(count, co_contribution_edges, _UNCUT_RGB, (0.5, 0.5, 0.5), device)
        adjacency_colors = _edge_highlight_colors(count, positive_edges_b, _UNCUT_RGB, _POSITIVE_RGB, device)
        termination_colors_occluded = _edge_highlight_colors(count, occluded_edges_b, _UNCUT_RGB, _OCCLUDED_RGB, device)
        termination_colors_free = _edge_highlight_colors(count, free_edges_b, torch.zeros(3), _FREE_SPACE_RGB, device)
        termination_colors = torch.maximum(termination_colors_occluded, termination_colors_free)

        singleton_cause_colors = _category_colors(cause, _CAUSE_RGB, CAUSE_CATEGORIES, _NOT_SINGLETON_RGB)
        singleton_cause_colors[~still_singleton_mask] = torch.tensor(_NOT_SINGLETON_RGB, dtype=torch.float32, device=device)

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_WL103_COMPONENTS: _rgb_to_f_dc(wl103_colors),
            VIEW_CONTRIBUTING_PRIMITIVES: _rgb_to_f_dc(contributing_colors),
            VIEW_CO_CONTRIBUTION_RELATIONS: _rgb_to_f_dc(co_contribution_colors),
            VIEW_ADJACENCY: _rgb_to_f_dc(adjacency_colors),
            VIEW_COMPONENTS: _rgb_to_f_dc(b_component_colors),
            VIEW_SINGLETON_CAUSE: _rgb_to_f_dc(singleton_cause_colors),
            VIEW_TERMINATION: _rgb_to_f_dc(termination_colors),
        }
        view_paths: dict[str, dict[str, Any]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

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

    accounting["views"] = view_paths
    accounting["render_ppm"] = render_report
    report_path = output_root / "renderer_grounded_visible_adjacency_report.json"
    report_path.write_text(json.dumps(accounting, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
