"""Worklog 104 -- Node-Level Observability Accounting + review export.

Does NOT modify torch_positive_visible_adjacency.py. Replays Worklog 103's
own baseline EXACTLY (same checkpoint, cameras, candidate graph, observation
states, geometry gates -- via direct, unmodified import of
`build_candidate_graph`, `compute_positive_visible_adjacency_evidence`,
`_connected_component_roots`) and adds a SEPARATE node-level observability
accounting pass (`torch_node_level_observability_accounting.py`) plus a
singleton-cause breakdown.

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-103:

    A. ORIGINAL_2DGS_SCENE
    B. WL103_PAIRWISE_POSITIVE_COMPONENTS (exact WL103 replay)
    C. SINGLETON_CAUSE_VIEW
    D. NODE_OBSERVABILITY_CATEGORY_VIEW (A/B/C/D, all surfels)
    E. RENDERER_PROJECTABILITY_VIEW (radii>0 view-count, diagnostic only)

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
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402
from osn_gs.surface.torch_coverage_first_subset_partition import (
    CoverageFirstPartitionConfig,
    build_candidate_graph,
    _connected_component_roots,
)
from osn_gs.surface.torch_node_level_observability_accounting import (
    CATEGORY_A_NEVER_POSITIVELY_OBSERVED,
    CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE,
    CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE,
    CATEGORY_D_OBSERVED_CONFLICT_ONLY,
    NODE_OBSERVABILITY_CATEGORIES,
    SINGLETON_CAUSE_CATEGORIES,
    classify_node_observability,
    classify_singleton_causes,
    compute_node_view_observability,
    node_observability_accounting,
)
from osn_gs.surface.torch_observation_evidence import CameraViewEvidence, ObservationEvidence
from osn_gs.surface.torch_positive_visible_adjacency import (
    RELATION_STATES,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_POSITIVE_VISIBLE_CONTINUATION,
    STATE_UNRESOLVED_CONFLICT,
    PositiveVisibleAdjacencyConfig,
    PositiveVisibleAdjacencyResult,
    compute_positive_visible_adjacency_evidence,
    positive_visible_adjacency_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_WL103_BASELINE = "WL103_PAIRWISE_POSITIVE_COMPONENTS"
VIEW_SINGLETON_CAUSE = "SINGLETON_CAUSE_VIEW"
VIEW_NODE_CATEGORY = "NODE_OBSERVABILITY_CATEGORY_VIEW"
VIEW_PROJECTABILITY = "RENDERER_PROJECTABILITY_VIEW"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

_NODE_CATEGORY_RGB = {
    CATEGORY_A_NEVER_POSITIVELY_OBSERVED: (0.6, 0.05, 0.05),   # dark red -- never seen at all
    CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE: (0.9, 0.75, 0.1),    # amber -- seen, isolated
    CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE: (0.15, 0.75, 0.25),  # green -- seen and connected
    CATEGORY_D_OBSERVED_CONFLICT_ONLY: (0.55, 0.2, 0.85),      # violet -- seen, conflicted only
}

_SINGLETON_CAUSE_RGB = {
    SINGLETON_CAUSE_CATEGORIES[0]: (0.6, 0.05, 0.05),   # NODE_NEVER_POSITIVELY_VISIBLE -- dark red
    SINGLETON_CAUSE_CATEGORIES[1]: (0.95, 0.55, 0.05),  # NODE_VISIBLE_BUT_NO_COOBSERVED_CANDIDATE_EDGE -- orange
    SINGLETON_CAUSE_CATEGORIES[2]: (0.95, 0.9, 0.1),    # COOBSERVED_EDGE_EXISTS_BUT_CORRIDOR_POSITIVE_TEST_FAILS -- yellow
    SINGLETON_CAUSE_CATEGORIES[3]: (1.0, 0.15, 0.55),   # POSITIVE_OBSERVATION_EXISTS_BUT_GEOMETRIC_GATE_CUTS -- magenta
    SINGLETON_CAUSE_CATEGORIES[4]: (0.55, 0.2, 0.85),   # OBSERVATION_CONFLICT -- violet
    SINGLETON_CAUSE_CATEGORIES[5]: (0.4, 0.4, 0.4),     # OTHER -- gray
}
_NON_SINGLETON_RGB = (0.08, 0.09, 0.11)


def _progress(message: str) -> None:
    print(f"[node-level observability] {message}", flush=True)


def _subset_partition_colors(subset_ids: torch.Tensor) -> torch.Tensor:
    identifiers = subset_ids.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def _category_colors(category: torch.Tensor, palette: dict, categories: tuple) -> torch.Tensor:
    count = int(category.shape[0])
    device = category.device
    colors = torch.zeros((count, 3), dtype=torch.float32, device=device)
    for index, name in enumerate(categories):
        rgb = palette.get(name)
        if rgb is None:
            continue
        mask = category == index
        colors[mask] = torch.tensor(rgb, dtype=torch.float32, device=device)
    return colors


def build_surfel_observation_evidence_and_projectability(cameras, model, rasterizer, visible_selector, *, near=1e-3, far=1e6, depth_epsilon=1e-2, progress=None):
    """Same per-camera render-and-wrap loop as
    `maximal_visible_connectivity_export.build_surfel_observation_evidence`
    (that function is reused nowhere near this one to avoid a second render
    pass), PLUS accumulation of the renderer-native `radii > 0` projection
    signal for every training view -- reusing the SAME render() call, no
    extra rendering. `radii` is per-primitive in the model's own full index
    space; sliced by `visible_selector` to align with WL103's own domain.
    """

    views = []
    projectable_view_count = torch.zeros((int(visible_selector.shape[0]),), dtype=torch.int32, device=visible_selector.device)
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
        radii = package["radii"].detach()[visible_selector]
        projectable_view_count += (radii > 0).to(torch.int32)
        del package
        if progress is not None and index % 20 == 0:
            progress(f"rendered observation evidence + radii {index + 1}/{len(cameras)}")
    evidence = ObservationEvidence(views=views, near=near, far=far, depth_epsilon=depth_epsilon, topology_version="checkpoint", camera_set_version=f"{len(cameras)}_train_cameras")
    return evidence, projectable_view_count


def _scatter_or(count: int, edges: torch.Tensor, spatial_mask: torch.Tensor, edge_flag: torch.Tensor, device) -> torch.Tensor:
    node_flag = torch.zeros((count,), dtype=torch.bool, device=device)
    selected = spatial_mask & edge_flag
    if not bool(selected.any()):
        return node_flag
    picked = edges[selected]
    node_flag[picked[:, 0]] = True
    node_flag[picked[:, 1]] = True
    return node_flag


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
        observation_evidence, projectable_view_count = build_surfel_observation_evidence_and_projectability(
            cameras, model, rasterizer, visible_selector, depth_epsilon=arguments.depth_epsilon, progress=_progress,
        )
    _progress(f"observation evidence + renderer projectability built over {len(observation_evidence.views)} views")

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
        config = PositiveVisibleAdjacencyConfig(local=local_config)

        # --- exact Worklog 103 baseline replay (unmodified functions only) ---
        _progress("[WL103 replay] build_candidate_graph (unmodified)")
        started_graph = time.time()
        graph = build_candidate_graph(orientation, config.local, progress=_progress)
        _progress(f"[WL103 replay] compute_positive_visible_adjacency_evidence (unmodified), {time.time()-started_graph:.1f}s for graph")
        started_evidence = time.time()
        evidence_dict = compute_positive_visible_adjacency_evidence(orientation, observation_evidence, graph, config, progress=_progress)
        seconds_wl103 = time.time() - started_graph
        _progress(f"[WL103 replay] evidence done in {time.time()-started_evidence:.1f}s")

        kept = graph.candidate_edges[evidence_dict["positive_visible_edges_mask"]]
        roots = _connected_component_roots(count, kept, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
        subset_ids = subset_id_of_position[inverse]
        subset_sizes = counts[order]
        result_wl103 = PositiveVisibleAdjacencyResult(
            subset_ids=subset_ids, subset_count=int(order.shape[0]), subset_sizes=subset_sizes,
            graph=graph, gaussian_ids=orientation.gaussian_ids,
            relation_state=evidence_dict["relation_state"], positive_visible_edges_mask=evidence_dict["positive_visible_edges_mask"],
            normal_gradient_magnitude=evidence_dict["normal_gradient_magnitude"], residual_threshold=evidence_dict["residual_threshold"],
            config=config,
        )
        accounting_wl103 = positive_visible_adjacency_accounting(result_wl103)
        _progress(f"[WL103 replay] {accounting_wl103['visible_component_count']} components largest={accounting_wl103['largest_component_surfel_fraction']:.4f} (cross-check vs committed report)")

        singleton_mask = subset_sizes[subset_ids] == 1

        # --- node-level observability accounting (new, this batch) ---
        _progress("[node accounting] compute_node_view_observability")
        node_obs = compute_node_view_observability(positions, observation_evidence, progress=_progress)
        node_obs = _dc_replace(node_obs, projectable_view_count=projectable_view_count)

        spatial_mask = graph.spatial_edge_mask
        edges = graph.candidate_edges
        node_has_positive_edge = _scatter_or(count, edges, spatial_mask, evidence_dict["positive_visible_edges_mask"], device)
        node_ever_evaluated = _scatter_or(count, edges, spatial_mask, evidence_dict["ever_evaluated"], device)
        node_any_positive_pre_geometry = _scatter_or(count, edges, spatial_mask, evidence_dict["any_positive"], device)
        node_geometric_cut = _scatter_or(count, edges, spatial_mask, evidence_dict["cut_reason_residual"] | evidence_dict["cut_reason_positional"], device)
        conflict_edge_mask = evidence_dict["relation_state"] == RELATION_STATES.index(STATE_UNRESOLVED_CONFLICT)
        node_conflict = _scatter_or(count, edges, spatial_mask, conflict_edge_mask, device)

        node_category = classify_node_observability(node_obs, node_has_positive_edge, node_conflict)
        all_node_accounting = node_observability_accounting(node_category)
        singleton_node_accounting = node_observability_accounting(node_category, mask=singleton_mask)
        _progress(f"[node accounting] all: {all_node_accounting['counts']}")
        _progress(f"[node accounting] singleton-only: {singleton_node_accounting['counts']}")

        singleton_cause = classify_singleton_causes(
            singleton_mask, node_obs, node_ever_evaluated, node_any_positive_pre_geometry, node_geometric_cut, node_conflict,
        )
        singleton_cause_accounting = node_observability_accounting(singleton_cause, categories=SINGLETON_CAUSE_CATEGORIES, mask=singleton_mask)
        _progress(f"[singleton cause] {singleton_cause_accounting['counts']}")

        # --- decide branch (directive section 6) ---
        never_visible_fraction_of_singletons = singleton_cause_accounting["fractions"][SINGLETON_CAUSE_CATEGORIES[0]]
        node_visible_but_isolated_fraction_of_singletons = 1.0 - never_visible_fraction_of_singletons
        branch = "A" if never_visible_fraction_of_singletons >= 0.5 else "B"
        _progress(f"[decision] never-visible-at-node-level fraction of singletons = {never_visible_fraction_of_singletons:.4f} -> Branch {branch}")

        # --- colors ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        wl103_colors = _subset_partition_colors(subset_ids)
        node_category_colors = _category_colors(node_category, _NODE_CATEGORY_RGB, NODE_OBSERVABILITY_CATEGORIES)

        singleton_cause_colors = torch.tensor(_NON_SINGLETON_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        singleton_cause_colors[singleton_mask] = _category_colors(singleton_cause[singleton_mask], _SINGLETON_CAUSE_RGB, SINGLETON_CAUSE_CATEGORIES)

        projectability_ratio = (node_obs.projectable_view_count.to(torch.float32) / max(node_obs.total_views, 1)).clamp(0.0, 1.0)
        low = torch.tensor((0.08, 0.09, 0.11), dtype=torch.float32, device=device).reshape(1, 3)
        high = torch.tensor((0.1, 0.7, 0.95), dtype=torch.float32, device=device).reshape(1, 3)
        projectability_colors = low + projectability_ratio.reshape(-1, 1) * (high - low)

        # --- bounded center-vs-primitive visibility comparison (directive section 4) ---
        center_visible = node_obs.on_observed_surface_view_count > 0
        renderer_projectable = node_obs.projectable_view_count > 0
        both_device = center_visible.device
        renderer_projectable = renderer_projectable.to(both_device)
        agreement_count = int((center_visible == renderer_projectable).sum())
        center_negative_renderer_positive = int((~center_visible & renderer_projectable).sum())
        center_positive_renderer_negative = int((center_visible & ~renderer_projectable).sum())
        center_vs_renderer_comparison = {
            "note": (
                "renderer-native signal here is radii>0 (projection/culling only, NOT an "
                "occlusion-aware contribution proof -- see module docstring); this is a bounded "
                "diagnostic, not a replacement ground truth"
            ),
            "total_surfels": count,
            "agreement_count": agreement_count,
            "agreement_fraction": agreement_count / count if count else 0.0,
            "center_negative_renderer_positive_count": center_negative_renderer_positive,
            "center_negative_renderer_positive_fraction": center_negative_renderer_positive / count if count else 0.0,
            "center_positive_renderer_negative_count": center_positive_renderer_negative,
            "center_positive_renderer_negative_fraction": center_positive_renderer_negative / count if count else 0.0,
            "singleton_center_negative_renderer_positive_count": int((~center_visible & renderer_projectable & singleton_mask).sum()),
            "singleton_center_positive_renderer_negative_count": int((center_visible & ~renderer_projectable & singleton_mask).sum()),
            "on_observed_surface_view_count_distribution": {
                "min": int(node_obs.on_observed_surface_view_count.min()),
                "median": int(node_obs.on_observed_surface_view_count.median()),
                "mean": float(node_obs.on_observed_surface_view_count.float().mean()),
                "max": int(node_obs.on_observed_surface_view_count.max()),
            },
            "projectable_view_count_distribution": {
                "min": int(node_obs.projectable_view_count.min()),
                "median": int(node_obs.projectable_view_count.median()),
                "mean": float(node_obs.projectable_view_count.float().mean()),
                "max": int(node_obs.projectable_view_count.max()),
            },
        }
        _progress(f"[center-vs-renderer] {center_vs_renderer_comparison}")

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_WL103_BASELINE: _rgb_to_f_dc(wl103_colors),
            VIEW_SINGLETON_CAUSE: _rgb_to_f_dc(singleton_cause_colors),
            VIEW_NODE_CATEGORY: _rgb_to_f_dc(node_category_colors),
            VIEW_PROJECTABILITY: _rgb_to_f_dc(projectability_colors),
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

    report = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 104",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "iteration": int(payload.get("iteration", 0)),
        "primitive_accounting": {"total_model_surfel_count": total_model_count, "visible_domain_surfel_count": visible_count},
        "camera_meta": camera_meta,
        "wl103_replay_accounting": accounting_wl103,
        "wl103_replay_seconds": seconds_wl103,
        "visible_topology_accounting": {
            "structural_visible_surfel_count": int(node_has_positive_edge.sum()),
            "structural_visible_surfel_fraction": float(node_has_positive_edge.float().mean()),
            "singleton_surfel_count": int(singleton_mask.sum()),
            "singleton_surfel_fraction": float(singleton_mask.float().mean()),
        },
        "node_observability_all_surfels": all_node_accounting,
        "node_observability_singleton_surfels": singleton_node_accounting,
        "singleton_cause_breakdown": singleton_cause_accounting,
        "center_vs_renderer_visibility_comparison": center_vs_renderer_comparison,
        "branch_decision": {
            "never_visible_fraction_of_singletons": never_visible_fraction_of_singletons,
            "node_visible_but_isolated_fraction_of_singletons": node_visible_but_isolated_fraction_of_singletons,
            "branch": branch,
        },
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"total": time.time() - started},
    }
    report_path = output_root / "node_level_observability_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({
        "branch": branch,
        "singleton_cause_breakdown": singleton_cause_accounting["fractions"],
        "node_observability_all": all_node_accounting["fractions"],
    }, indent=2))


if __name__ == "__main__":
    main()
