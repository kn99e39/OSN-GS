"""Worklog 105 -- Renderer Contribution Diagnostics + review export.

Does NOT modify torch_positive_visible_adjacency.py, torch_maximal_visible_
connectivity.py, torch_node_level_observability_accounting.py, or any
vendored rasterizer file. Replays Worklog 103 exactly (unmodified) for the
singleton mask, reuses Worklog 104's node-level Phase-C accounting and
renderer-projectability pass unmodified, and adds a NEW, genuinely
renderer-grounded per-surfel CONTRIBUTION signal via
`osn_gs/render/torch_surfel_contribution_diagnostics.py` (diagnostic-only,
uses the official vendored backward kernel's own arithmetic through
`torch.autograd.grad`, never `.backward()`, never mutates any parameter's
`.grad`).

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-104:

    A. ORIGINAL_2DGS_SCENE
    B. WL103_PAIRWISE_POSITIVE_COMPONENTS (exact WL103 replay)
    C. CENTER_VS_RENDERER_CONTRIBUTION_VIEW (this batch's central result)
    D. RENDERER_CONTRIBUTION_VIEW (view-count ramp, diagnostic only)

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
from node_level_observability_export import build_surfel_observation_evidence_and_projectability  # noqa: E402
from osn_gs.render.torch_surfel_contribution_diagnostics import accumulate_renderer_contribution_evidence
from osn_gs.surface.torch_coverage_first_subset_partition import (
    CoverageFirstPartitionConfig,
    build_candidate_graph,
    _connected_component_roots,
)
from osn_gs.surface.torch_node_level_observability_accounting import compute_node_view_observability
from osn_gs.surface.torch_positive_visible_adjacency import (
    PositiveVisibleAdjacencyConfig,
    PositiveVisibleAdjacencyResult,
    compute_positive_visible_adjacency_evidence,
    positive_visible_adjacency_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_WL103_BASELINE = "WL103_PAIRWISE_POSITIVE_COMPONENTS"
VIEW_CENTER_VS_CONTRIBUTION = "CENTER_VS_RENDERER_CONTRIBUTION_VIEW"
VIEW_CONTRIBUTION_RAMP = "RENDERER_CONTRIBUTION_VIEW"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

# 4 mutually-exclusive, priority-ordered categories covering all 5 named
# concepts from the directive's review-export requirement (section 12):
#   CENTER_POSITIVE                              -- Phase-C center-positive (any contribution status)
#   CENTER_NEGATIVE_CONTRIBUTING                 -- the critical group this batch measures
#   CENTER_NEGATIVE_PROJECTABLE_NONCONTRIBUTING  -- radii>0 only, never an accepted contributor
#   CENTER_NEGATIVE_NONCONTRIBUTING_NONPROJECTABLE -- genuinely absent by every available signal
CATEGORY_CENTER_POSITIVE = "CENTER_POSITIVE"
CATEGORY_CENTER_NEGATIVE_CONTRIBUTING = "CENTER_NEGATIVE_CONTRIBUTING"
CATEGORY_CENTER_NEGATIVE_PROJECTABLE_NONCONTRIBUTING = "CENTER_NEGATIVE_PROJECTABLE_NONCONTRIBUTING"
CATEGORY_CENTER_NEGATIVE_NONCONTRIBUTING_NONPROJECTABLE = "CENTER_NEGATIVE_NONCONTRIBUTING_NONPROJECTABLE"
CROSS_CATEGORIES = (
    CATEGORY_CENTER_POSITIVE, CATEGORY_CENTER_NEGATIVE_CONTRIBUTING,
    CATEGORY_CENTER_NEGATIVE_PROJECTABLE_NONCONTRIBUTING, CATEGORY_CENTER_NEGATIVE_NONCONTRIBUTING_NONPROJECTABLE,
)
_CROSS_CATEGORY_RGB = {
    CATEGORY_CENTER_POSITIVE: (0.15, 0.75, 0.25),                              # green
    CATEGORY_CENTER_NEGATIVE_CONTRIBUTING: (1.0, 0.55, 0.0),                   # orange -- the critical group
    CATEGORY_CENTER_NEGATIVE_PROJECTABLE_NONCONTRIBUTING: (0.15, 0.55, 1.0),   # blue
    CATEGORY_CENTER_NEGATIVE_NONCONTRIBUTING_NONPROJECTABLE: (0.6, 0.05, 0.05),  # dark red
}


def _progress(message: str) -> None:
    print(f"[renderer contribution diagnostics] {message}", flush=True)


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
        colors[category == index] = torch.tensor(rgb, dtype=torch.float32, device=device)
    return colors


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    if int(values.shape[0]) == 0:
        return {"min": 0, "median": 0, "mean": 0.0, "p95": 0.0, "max": 0}
    sorted_values = torch.sort(values.to(torch.float64)).values

    def _percentile(fraction: float) -> float:
        position = min(int(sorted_values.shape[0]) - 1, max(0, int(round(fraction * (int(sorted_values.shape[0]) - 1)))))
        return float(sorted_values[position].item())

    return {
        "min": _percentile(0.0), "median": _percentile(0.5), "mean": float(sorted_values.mean().item()),
        "p95": _percentile(0.95), "max": _percentile(1.0),
    }


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

    _progress("[contribution] accumulating renderer contribution evidence (grad-enabled pass, no .grad mutation)")
    started_contribution = time.time()
    contribution = accumulate_renderer_contribution_evidence(cameras, model, rasterizer, progress=_progress)
    seconds_contribution = time.time() - started_contribution
    _progress(f"[contribution] done in {seconds_contribution:.1f}s")
    contributing_view_count_visible = contribution.contributing_view_count[visible_selector]
    accumulated_weight_sum_visible = contribution.accumulated_weight_sum[visible_selector]
    max_single_view_weight_visible = contribution.max_single_view_accumulated_weight[visible_selector]
    ever_contributed_visible = contribution.ever_contributed[visible_selector]

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

        _progress("[WL103 replay] build_candidate_graph (unmodified)")
        graph = build_candidate_graph(orientation, config.local, progress=_progress)
        evidence_dict = compute_positive_visible_adjacency_evidence(orientation, observation_evidence, graph, config, progress=_progress)

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

        _progress("[node accounting] compute_node_view_observability (Phase-C center, unmodified reuse)")
        node_obs = compute_node_view_observability(positions, observation_evidence, progress=_progress)
        center_positive = node_obs.on_observed_surface_view_count > 0
        renderer_projectable = projectable_view_count > 0
        renderer_contributing = ever_contributed_visible

        category = torch.full((count,), CROSS_CATEGORIES.index(CATEGORY_CENTER_NEGATIVE_NONCONTRIBUTING_NONPROJECTABLE), dtype=torch.int64, device=device)
        category = torch.where(renderer_projectable, torch.full_like(category, CROSS_CATEGORIES.index(CATEGORY_CENTER_NEGATIVE_PROJECTABLE_NONCONTRIBUTING)), category)
        category = torch.where(renderer_contributing, torch.full_like(category, CROSS_CATEGORIES.index(CATEGORY_CENTER_NEGATIVE_CONTRIBUTING)), category)
        category = torch.where(center_positive, torch.full_like(category, CROSS_CATEGORIES.index(CATEGORY_CENTER_POSITIVE)), category)

        def _cross_tab(mask: torch.Tensor | None) -> dict[str, Any]:
            c = center_positive if mask is None else center_positive[mask]
            r = renderer_contributing if mask is None else renderer_contributing[mask]
            total = int(c.shape[0])
            return {
                "total": total,
                "center_positive_contribution_positive": int((c & r).sum()),
                "center_positive_contribution_negative": int((c & ~r).sum()),
                "center_negative_contribution_positive": int((~c & r).sum()),
                "center_negative_contribution_negative": int((~c & ~r).sum()),
            }

        cross_tab_all = _cross_tab(None)
        cross_tab_singletons = _cross_tab(singleton_mask)

        never_center_positive_mask = ~center_positive
        never_center_positive_count = int(never_center_positive_mask.sum())
        never_center_positive_contributing_count = int((never_center_positive_mask & renderer_contributing).sum())
        never_center_positive_contributing_fraction = (
            never_center_positive_contributing_count / never_center_positive_count if never_center_positive_count else 0.0
        )

        report = {
            "batch": "arch/2dgs-coverage-first-surface, Worklog 105",
            "checkpoint": str(arguments.checkpoint),
            "primitive": primitive,
            "iteration": int(payload.get("iteration", 0)),
            "primitive_accounting": {"total_model_surfel_count": total_model_count, "visible_domain_surfel_count": visible_count},
            "camera_meta": camera_meta,
            "wl103_replay_accounting": accounting_wl103,
            "cross_tabulation_all_surfels": cross_tab_all,
            "cross_tabulation_wl103_singletons": cross_tab_singletons,
            "never_center_positive_cohort": {
                "count": never_center_positive_count,
                "actually_contributing_count": never_center_positive_contributing_count,
                "actually_contributing_fraction": never_center_positive_contributing_fraction,
                "contributing_view_count_distribution": _distribution(contributing_view_count_visible[never_center_positive_mask].to(torch.float32)),
                "accumulated_weight_sum_distribution": _distribution(accumulated_weight_sum_visible[never_center_positive_mask]),
                "max_single_view_accumulated_weight_distribution": _distribution(max_single_view_weight_visible[never_center_positive_mask]),
            },
            "renderer_contributing_view_count_distribution_all": _distribution(contributing_view_count_visible.to(torch.float32)),
            "renderer_accumulated_weight_sum_distribution_all": _distribution(accumulated_weight_sum_visible),
            "branch_reassessment": {
                "never_center_positive_actually_contributing_fraction": never_center_positive_contributing_fraction,
                "case": (
                    "CASE_A_BRANCH_A_STRONGLY_SUPPORTED"
                    if never_center_positive_contributing_fraction < 0.10
                    else "CASE_B_BRANCH_A_INSUFFICIENT"
                ),
            },
            "runtime_seconds": {"contribution_pass": seconds_contribution, "total": time.time() - started},
        }
        _progress(f"[cross-tab all] {cross_tab_all}")
        _progress(f"[cross-tab singletons] {cross_tab_singletons}")
        _progress(f"[never-center-positive cohort] count={never_center_positive_count} actually_contributing={never_center_positive_contributing_count} fraction={never_center_positive_contributing_fraction:.4f}")
        _progress(f"[branch reassessment] {report['branch_reassessment']}")

        # --- colors ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
        wl103_colors = _subset_partition_colors(subset_ids)
        cross_category_colors = _category_colors(category, _CROSS_CATEGORY_RGB, CROSS_CATEGORIES)

        low = torch.tensor((0.08, 0.09, 0.11), dtype=torch.float32, device=device).reshape(1, 3)
        high = torch.tensor((1.0, 0.55, 0.0), dtype=torch.float32, device=device).reshape(1, 3)
        contribution_ratio = (contributing_view_count_visible.to(torch.float32) / max(len(cameras), 1)).clamp(0.0, 1.0)
        contribution_ramp_colors = low + contribution_ratio.reshape(-1, 1) * (high - low)

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_WL103_BASELINE: _rgb_to_f_dc(wl103_colors),
            VIEW_CENTER_VS_CONTRIBUTION: _rgb_to_f_dc(cross_category_colors),
            VIEW_CONTRIBUTION_RAMP: _rgb_to_f_dc(contribution_ramp_colors),
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

    report["views"] = view_paths
    report["render_ppm"] = render_report
    report_path = output_root / "renderer_contribution_diagnostics_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
