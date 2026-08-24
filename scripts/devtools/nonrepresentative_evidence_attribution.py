"""Worklog 110 -- Non-Representative Renderer Evidence: Role Attribution.

Does NOT modify torch_camera_induced_visible_adjacency.py (Worklog 107/109,
frozen canonical topology) or torch_surfel_representative_diagnostics.py's
existing fields. Reads the Worklog 107/109 representative-only component
graph as read-only input and reuses its own public functions unmodified.

Uses the worklog-110-extended diagnostic build
(osn_gs/render/vendor/diff_surfel_rasterization_diag/, canonical
diff_surfel_rasterization still untouched) to classify every same-forward
accepted contribution event as PRE_MEDIAN or POST_MEDIAN relative to its
pixel's own median crossing, and to derive contributor<->representative
co-support relations restricted to a bounded per-pixel provenance slot array
(OSN_GS_MAX_CONTRIB_SLOTS, never a full pixel x surfel matrix).

Does NOT attach any primitive to any component. This is attribution only.
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
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative
from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    CameraInducedAdjacencyConfig,
    accumulate_image_space_pairs,
    apply_secondary_geometric_gate,
    filter_by_3d_locality,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph
from osn_gs.surface.torch_discontinuity_first_surfel_partition import _auto_chunk_size, _fit_shape_operators, _knn, _predicted_delta_n_t, _tangent_plane_components
from osn_gs.surface.torch_nonrepresentative_evidence_attribution import (
    ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION,
    COMPONENT_RELATION_CATEGORIES,
    SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS,
    SUPPORTS_ONE_REPRESENTATIVE_COMPONENT,
    classify_pre_post_median,
    component_relation_category,
    finalize_component_co_support,
    view_contributor_component_pairs,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"
_EPS = 1e-9
_UNCUT_RGB = (0.08, 0.09, 0.11)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_CANONICAL_BACKBONE = "CANONICAL_REPRESENTATIVE_BACKBONE"
VIEW_NONREP_ACCEPTED = "FORWARD_ACCEPTED_NON_REPRESENTATIVES"
VIEW_PRE_MEDIAN = "PRE_MEDIAN_ACCEPTED"
VIEW_POST_MEDIAN = "POST_MEDIAN_ACCEPTED"
VIEW_POST_MEDIAN_ONLY = "POST_MEDIAN_ONLY"
VIEW_SINGLE_COMPONENT = "SINGLE_COMPONENT_COSUPPORT"
VIEW_MULTI_COMPONENT = "MULTI_COMPONENT_COSUPPORT"
VIEW_TABLE_PATIO_RELATIONS = "TABLE_PATIO_NONREP_RELATIONS"
VIEW_HEDGE_RELATIONS = "HEDGE_NONREP_RELATIONS"

ANCHOR_FRACTIONS = {
    "table": [(0.5, 0.48)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[nonrep attribution] {message}", flush=True)


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    if int(values.shape[0]) == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = torch.sort(values.to(torch.float64)).values

    def _percentile(fraction: float) -> float:
        position = min(int(sorted_values.shape[0]) - 1, max(0, int(round(fraction * (int(sorted_values.shape[0]) - 1)))))
        return float(sorted_values[position].item())

    return {"min": _percentile(0.0), "median": _percentile(0.5), "mean": float(sorted_values.mean().item()), "p95": _percentile(0.95), "max": _percentile(1.0)}


def _ramp(ratio: torch.Tensor, low_rgb, high_rgb) -> torch.Tensor:
    low = torch.tensor(low_rgb, device=ratio.device).reshape(1, 3)
    high = torch.tensor(high_rgb, device=ratio.device).reshape(1, 3)
    return low + ratio.clamp(0.0, 1.0).reshape(-1, 1) * (high - low)


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
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
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

    full_to_visible = torch.full((total_model_count,), -1, dtype=torch.int64, device=model.device)
    full_to_visible[visible_selector] = torch.arange(visible_count, dtype=torch.int64, device=model.device)

    # --- build the canonical WL107/109 representative-only topology (READ-ONLY input, unmodified functions) ---
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
        normals = orientation.surface_normal
        device = positions.device
        count = int(positions.shape[0])

        local_config = CoverageFirstPartitionConfig()
        config = CameraInducedAdjacencyConfig(local=local_config)

        _progress("[WL107/109 replay, unchanged] build_candidate_graph")
        graph = build_candidate_graph(orientation, config.local, progress=_progress)

    # --- same-forward sweep: representative, forward-accepted, pre/post-median, contributor-component pairs ---
    _progress("[same-forward sweep] representative_id + forward_accepted + contrib provenance (worklog 110 diagnostic build)")
    per_view_representative_ids: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    ever_forward_accepted_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    ever_pre_median_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    ever_post_median_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    pair_batches_visible: list[torch.Tensor] = []
    truncated_pixel_view_count = 0

    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        represented_ids = torch.unique(rep_full[valid])
        ever_representative_full[represented_ids] = True

        forward_accepted_this_view = diag["forward_accepted"].to(torch.bool).reshape(-1)
        ever_forward_accepted_full |= forward_accepted_this_view

        contrib_ids_full = diag["contrib_ids"].to(torch.int64)
        contrib_post_median = diag["contrib_post_median"].to(torch.int64)
        contrib_count = diag["contrib_count"]
        truncated_pixel_view_count += int((contrib_count > contrib_ids_full.shape[-1]).sum().item())

        ever_pre, ever_post = classify_pre_post_median(contrib_ids_full, contrib_post_median, total_model_count)
        ever_pre_median_full |= ever_pre
        ever_post_median_full |= ever_post

        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_representative_ids.append(rep_remapped.detach())

        contrib_valid = contrib_ids_full >= 0
        contrib_ids_remapped = torch.where(contrib_valid, full_to_visible[contrib_ids_full.clamp(min=0)], torch.full_like(contrib_ids_full, -1))

        del diag
        if index % 20 == 0:
            _progress(f"same-forward view {index + 1}/{len(cameras)}")

        # component pairing deferred to after subset_ids is known (needs positive_edges from this same sweep)
        pair_batches_visible.append((contrib_ids_remapped.cpu(), rep_remapped.cpu()))

    ever_representative = ever_representative_full[visible_selector]
    ever_forward_accepted = ever_forward_accepted_full[visible_selector]
    ever_pre_median = ever_pre_median_full[visible_selector]
    ever_post_median = ever_post_median_full[visible_selector]
    accepted_non_representative = ever_forward_accepted & ~ever_representative
    never_forward_accepted = ~ever_forward_accepted & ~ever_representative

    accounting = {
        "total_trained_surfels": total_model_count,
        "visible_domain_surfels": visible_count,
        "median_surface_representatives": int(ever_representative.sum()),
        "same_forward_accepted_non_representatives": int(accepted_non_representative.sum()),
        "never_forward_accepted": int(never_forward_accepted.sum()),
    }
    _progress(f"[accounting] {accounting}")

    with torch.no_grad():
        _progress("[WL107/109 replay, unchanged] accumulate_image_space_pairs")
        raw_pairs, raw_view_support = accumulate_image_space_pairs(count, per_view_representative_ids, progress=_progress)
        local_pairs, local_mask = filter_by_3d_locality(raw_pairs, count, graph)

        _progress("[WL107/109 replay, unchanged] apply_secondary_geometric_gate")
        geometry = apply_secondary_geometric_gate(local_pairs, orientation, config, progress=_progress)
        kept_mask = geometry["kept_mask"]
        positive_edges = local_pairs[kept_mask]

        roots = _connected_component_roots(count, positive_edges, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
        subset_ids = subset_id_of_position[inverse]
        subset_sizes = counts[order]
        subset_count = int(order.shape[0])
        wl107_replay_stats = {
            "visible_component_count": subset_count,
            "largest_component_surfel_fraction": float(subset_sizes[0]) / count,
            "singleton_surfel_count": int((subset_sizes == 1).sum()),
        }
        _progress(f"[replay consistency check] {wl107_replay_stats}")

        # --- component co-support pairs (item 5/7): stream per-view, restricted to visible domain ---
        _progress("[co-support] deriving contributor<->representative-component pairs per view")
        subset_ids_cpu = subset_ids.cpu()
        pair_batches: list[torch.Tensor] = []
        for view_index, (contrib_ids_remapped_cpu, rep_remapped_cpu) in enumerate(pair_batches_visible):
            pairs = view_contributor_component_pairs(contrib_ids_remapped_cpu, rep_remapped_cpu, subset_ids_cpu)
            if int(pairs.shape[0]) > 0:
                pair_batches.append(pairs.to(device))
            if view_index % 20 == 0:
                _progress(f"co-support view {view_index + 1}/{len(pair_batches_visible)}")
        del pair_batches_visible

        co_support = finalize_component_co_support(pair_batches, count, subset_count)
        distinct_component_count = co_support["distinct_component_count"]
        category = component_relation_category(distinct_component_count, accepted_non_representative)

        single_mask = accepted_non_representative & (distinct_component_count == 1)
        multi_mask = accepted_non_representative & (distinct_component_count >= 2)
        none_mask = accepted_non_representative & (distinct_component_count == 0)

        component_relation_counts = {
            ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION: int(none_mask.sum()),
            SUPPORTS_ONE_REPRESENTATIVE_COMPONENT: int(single_mask.sum()),
            SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS: int(multi_mask.sum()),
        }
        multi_component_touch_distribution = _distribution(distinct_component_count[multi_mask].to(torch.float32))
        _progress(f"[component relation] {component_relation_counts} multi-touch-count dist={multi_component_touch_distribution}")

        # --- pre/post-median distributions (item 4/9), restricted to accepted-non-representative population ---
        pre_only = accepted_non_representative & ever_pre_median & ~ever_post_median
        post_only = accepted_non_representative & ever_post_median & ~ever_pre_median
        mixed = accepted_non_representative & ever_pre_median & ever_post_median
        neither_pre_post = accepted_non_representative & ~ever_pre_median & ~ever_post_median  # should be ~0: every accepted event is pre-or-post by construction
        pre_post_counts = {
            "any_pre_median": int((accepted_non_representative & ever_pre_median).sum()),
            "any_post_median": int((accepted_non_representative & ever_post_median).sum()),
            "pre_median_only": int(pre_only.sum()),
            "post_median_only": int(post_only.sum()),
            "mixed_pre_and_post": int(mixed.sum()),
            "neither_flag_set_unexpected": int(neither_pre_post.sum()),
        }
        _progress(f"[pre/post median] {pre_post_counts}")

        # --- region masks (WORKING INTERPRETATION ONLY, same anchor heuristic as WL108/109) ---
        preview_images = arguments.preview_camera_images or arguments.images
        preview_cameras, _preview_meta = load_all_train_cameras(arguments.source_path, preview_images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device)
        preview_camera = min(preview_cameras, key=lambda c: c.image_name)
        preview_diag = render_with_pixel_representative(preview_camera, model)
        preview_rep = preview_diag["representative_id"].to(torch.int64)
        preview_h, preview_w = preview_rep.shape

        anchor_visible_ids: dict[str, list[int]] = {}
        for label, fractions in ANCHOR_FRACTIONS.items():
            ids = []
            for fx, fy in fractions:
                px = min(preview_w - 1, int(fx * preview_w))
                py = min(preview_h - 1, int(fy * preview_h))
                full_id = int(preview_rep[py, px].item())
                if full_id >= 0:
                    ids.append(int(full_to_visible[full_id].item()))
            anchor_visible_ids[label] = ids

        region_masks: dict[str, torch.Tensor] = {}
        for label in ANCHOR_FRACTIONS:
            ids = anchor_visible_ids.get(label, [])
            mask = torch.zeros((count,), dtype=torch.bool, device=device)
            if ids:
                anchor_positions = positions[torch.tensor(ids, device=device)]
                dist_to_this = torch.cdist(positions, anchor_positions).min(dim=1).values
                other_ids = sum((anchor_visible_ids.get(l, []) for l in ANCHOR_FRACTIONS if l != label), [])
                if other_ids:
                    dist_to_other = torch.cdist(positions, positions[torch.tensor(other_ids, device=device)]).min(dim=1).values
                    mask = dist_to_this < dist_to_other
                else:
                    mask = torch.ones((count,), dtype=torch.bool, device=device)
            region_masks[label] = mask

        region_breakdown = {}
        for label, mask in region_masks.items():
            pop = accepted_non_representative & mask
            region_breakdown[label] = {
                "accepted_non_representative_count": int(pop.sum()),
                "pre_median_only": int((pop & ever_pre_median & ~ever_post_median).sum()),
                "post_median_only": int((pop & ever_post_median & ~ever_pre_median).sum()),
                "mixed_pre_and_post": int((pop & ever_pre_median & ever_post_median).sum()),
                "no_representative_association": int((pop & none_mask).sum()),
                "single_component_cosupport": int((pop & single_mask).sum()),
                "multi_component_cosupport": int((pop & multi_mask).sum()),
            }
        _progress(f"[region breakdown] {region_breakdown}")

        # --- cross-component behavior (item 10): table<->patio, largest<->hedge ---
        table_ids = torch.tensor(anchor_visible_ids.get("table", []), device=device, dtype=torch.int64)
        patio_ids = torch.tensor(anchor_visible_ids.get("patio", []), device=device, dtype=torch.int64)
        hedge_ids = torch.tensor(anchor_visible_ids.get("hedge", []), device=device, dtype=torch.int64)
        anchor_components = {
            "table": set(subset_ids[table_ids].tolist()) if int(table_ids.shape[0]) else set(),
            "patio": set(subset_ids[patio_ids].tolist()) if int(patio_ids.shape[0]) else set(),
            "hedge": set(subset_ids[hedge_ids].tolist()) if int(hedge_ids.shape[0]) else set(),
        }
        multi_pairs = co_support["unique_pairs"]
        multi_contrib_ids_only = torch.unique(multi_pairs[:, 0][torch.isin(multi_pairs[:, 0], torch.nonzero(multi_mask, as_tuple=False).reshape(-1))]) if int(multi_pairs.shape[0]) else torch.zeros((0,), dtype=torch.int64, device=device)
        cross_component_examples = []
        max_examples = 30
        if int(multi_contrib_ids_only.shape[0]) > 0:
            mp_mask = torch.isin(multi_pairs[:, 0], multi_contrib_ids_only)
            relevant = multi_pairs[mp_mask]
            order_by_contrib = torch.argsort(relevant[:, 0])
            relevant = relevant[order_by_contrib]
            contrib_col = relevant[:, 0]
            unique_contribs, counts_per = torch.unique_consecutive(contrib_col, return_counts=True)
            offset = 0
            for contributor_id, group_count in zip(unique_contribs.tolist(), counts_per.tolist()):
                if len(cross_component_examples) >= max_examples:
                    break
                comps = relevant[offset:offset + group_count, 1].tolist()
                offset += group_count
                comp_set = set(comps)
                touches_table = bool(comp_set & anchor_components["table"])
                touches_patio = bool(comp_set & anchor_components["patio"])
                touches_hedge = bool(comp_set & anchor_components["hedge"])
                cross_component_examples.append({
                    "contributor_visible_id": contributor_id,
                    "position": positions[contributor_id].tolist(),
                    "component_ids": comps,
                    "component_count": len(comp_set),
                    "touches_table_component": touches_table,
                    "touches_patio_component": touches_patio,
                    "touches_hedge_component": touches_hedge,
                    "pre_median": bool(ever_pre_median[contributor_id].item()),
                    "post_median": bool(ever_post_median[contributor_id].item()),
                })
        table_patio_cross_count = sum(1 for e in cross_component_examples if e["touches_table_component"] and e["touches_patio_component"])
        largest_hedge_cross_count = sum(1 for e in cross_component_examples if 0 in e["component_ids"] and e["touches_hedge_component"])
        cross_component_summary = {
            "multi_component_contributor_count": int(multi_mask.sum()),
            "example_sample_size": len(cross_component_examples),
            "table_patio_cross_examples_in_sample": table_patio_cross_count,
            "largest_component_hedge_cross_examples_in_sample": largest_hedge_cross_count,
        }
        _progress(f"[cross-component] {cross_component_summary}")

        # --- local geometric compatibility for SINGLE-component co-support (item 8), via EXISTING candidate graph only ---
        k_shape = min(config.resolved_shape_operator_neighbor_count() or config.local.neighbor_count, max(count - 1, 1))
        chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
        neighbor_index, _ = _knn(positions, k_shape, chunk_size, _progress)
        shape_operator = _fit_shape_operators(positions, normals, orientation.tangent_axis_u, orientation.tangent_axis_v, neighbor_index, float(config.shape_operator_ridge))

        cand_edges = graph.candidate_edges[graph.spatial_edge_mask]
        u, v = cand_edges[:, 0], cand_edges[:, 1]
        u_rep, v_rep = ever_representative[u], ever_representative[v]
        mixed_edge_mask = u_rep ^ v_rep
        contributor_side = torch.where(u_rep, v, u)[mixed_edge_mask]
        representative_side = torch.where(u_rep, u, v)[mixed_edge_mask]
        edge_component = subset_ids[representative_side]
        edge_key = contributor_side.to(torch.int64) * subset_count + edge_component.to(torch.int64)

        single_pairs = co_support["unique_pairs"][torch.isin(co_support["unique_pairs"][:, 0], torch.nonzero(single_mask, as_tuple=False).reshape(-1))]
        single_keys = torch.sort(single_pairs[:, 0] * subset_count + single_pairs[:, 1]).values if int(single_pairs.shape[0]) else torch.zeros((0,), dtype=torch.int64, device=device)
        corroborated_mask = torch.isin(edge_key, single_keys)
        corroborated_edges = torch.stack([contributor_side[corroborated_mask], representative_side[corroborated_mask]], dim=1)

        if int(corroborated_edges.shape[0]) > 0:
            left, right = corroborated_edges[:, 0], corroborated_edges[:, 1]
            delta_x = positions[right] - positions[left]
            distance = delta_x.norm(dim=-1)
            spacing = (graph.local_spacing[left] + graph.local_spacing[right]) * 0.5
            delta_x_t_left = _tangent_plane_components(delta_x, orientation.tangent_axis_u[left], orientation.tangent_axis_v[left])
            sign_lr = torch.where((normals[left] * normals[right]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
            aligned_right_normal = normals[right] * sign_lr
            delta_n_left = aligned_right_normal - normals[left]
            delta_n_t_left = _tangent_plane_components(delta_n_left, orientation.tangent_axis_u[left], orientation.tangent_axis_v[left])
            predicted_left = _predicted_delta_n_t(shape_operator[left], delta_x_t_left)
            residual = (delta_n_t_left - predicted_left).norm(dim=-1)
            average_normal = torch.nn.functional.normalize(normals[left] + aligned_right_normal, dim=-1, eps=_EPS)
            signed_normal_offset = (delta_x * average_normal).sum(dim=-1)
            tangential_offset = (delta_x - signed_normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
            normal_offset_ratio = signed_normal_offset.abs() / tangential_offset.clamp_min(_EPS)
            local_geometry_summary = {
                "corroborated_edge_count": int(corroborated_edges.shape[0]),
                "single_component_cosupport_with_candidate_edge_fraction": int(corroborated_edges.shape[0]) / max(1, int(single_mask.sum())),
                "distance_distribution": _distribution(distance),
                "distance_over_local_spacing_distribution": _distribution(distance / spacing.clamp_min(_EPS)),
                "residual_distribution": _distribution(residual),
                "normal_offset_ratio_distribution": _distribution(normal_offset_ratio),
            }
        else:
            local_geometry_summary = {"corroborated_edge_count": 0}
        _progress(f"[local geometry, single-component cosupport] {local_geometry_summary}")

        report = {
            "batch": "arch/2dgs-coverage-first-surface, Worklog 110",
            "checkpoint": str(arguments.checkpoint),
            "primitive": primitive,
            "iteration": int(payload.get("iteration", 0)),
            "camera_meta": camera_meta,
            "accounting": accounting,
            "truncated_pixel_view_events": truncated_pixel_view_count,
            "wl107_109_replay_consistency_check": wl107_replay_stats,
            "component_relation_counts": component_relation_counts,
            "multi_component_touch_distribution": multi_component_touch_distribution,
            "pre_post_median_counts": pre_post_counts,
            "region_breakdown_WORKING_INTERPRETATION_ONLY": region_breakdown,
            "cross_component_summary": cross_component_summary,
            "cross_component_examples_sample": cross_component_examples,
            "local_geometric_compatibility_single_component_cosupport": local_geometry_summary,
            "runtime_seconds": {"total": time.time() - started},
        }

        # --- colors / exports ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        backbone_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        backbone_colors[ever_representative] = _subset_partition_colors(subset_ids[ever_representative])

        nonrep_colors = _ramp(accepted_non_representative.to(torch.float32), _UNCUT_RGB, (1.0, 0.55, 0.0))

        pre_colors = _ramp((accepted_non_representative & ever_pre_median).to(torch.float32), _UNCUT_RGB, (0.15, 0.75, 0.95))
        post_colors = _ramp((accepted_non_representative & ever_post_median).to(torch.float32), _UNCUT_RGB, (0.85, 0.1, 0.4))
        post_only_colors = _ramp(post_only.to(torch.float32), _UNCUT_RGB, (0.85, 0.1, 0.4))

        # color single-component contributors by the component they co-support
        single_component_id = torch.full((count,), -1, dtype=torch.int64, device=device)
        if int(single_pairs.shape[0]) > 0:
            single_component_id[single_pairs[:, 0]] = single_pairs[:, 1]
        single_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        has_single_color = single_component_id >= 0
        single_colors[has_single_color] = _subset_partition_colors(single_component_id[has_single_color])

        multi_colors = _ramp(multi_mask.to(torch.float32), _UNCUT_RGB, (1.0, 0.85, 0.0))

        table_patio_mask = accepted_non_representative & (region_masks.get("table", torch.zeros_like(accepted_non_representative)) | region_masks.get("patio", torch.zeros_like(accepted_non_representative)))
        table_patio_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        table_patio_colors[table_patio_mask & pre_only] = torch.tensor((0.15, 0.75, 0.95), dtype=torch.float32, device=device)
        table_patio_colors[table_patio_mask & post_only] = torch.tensor((0.85, 0.1, 0.4), dtype=torch.float32, device=device)
        table_patio_colors[table_patio_mask & mixed] = torch.tensor((0.9, 0.6, 0.1), dtype=torch.float32, device=device)

        hedge_mask_pop = accepted_non_representative & region_masks.get("hedge", torch.zeros_like(accepted_non_representative))
        hedge_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        hedge_colors[hedge_mask_pop & pre_only] = torch.tensor((0.15, 0.75, 0.95), dtype=torch.float32, device=device)
        hedge_colors[hedge_mask_pop & post_only] = torch.tensor((0.85, 0.1, 0.4), dtype=torch.float32, device=device)
        hedge_colors[hedge_mask_pop & mixed] = torch.tensor((0.9, 0.6, 0.1), dtype=torch.float32, device=device)

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_CANONICAL_BACKBONE: _rgb_to_f_dc(backbone_colors),
            VIEW_NONREP_ACCEPTED: _rgb_to_f_dc(nonrep_colors),
            VIEW_PRE_MEDIAN: _rgb_to_f_dc(pre_colors),
            VIEW_POST_MEDIAN: _rgb_to_f_dc(post_colors),
            VIEW_POST_MEDIAN_ONLY: _rgb_to_f_dc(post_only_colors),
            VIEW_SINGLE_COMPONENT: _rgb_to_f_dc(single_colors),
            VIEW_MULTI_COMPONENT: _rgb_to_f_dc(multi_colors),
            VIEW_TABLE_PATIO_RELATIONS: _rgb_to_f_dc(table_patio_colors),
            VIEW_HEDGE_RELATIONS: _rgb_to_f_dc(hedge_colors),
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
    report_path = output_root / "nonrepresentative_evidence_attribution_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
