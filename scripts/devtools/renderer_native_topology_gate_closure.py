"""Worklog 109 -- closing Worklog 108's two remaining caveats before the
Renderer-Native Surface Representative Graph (Worklog 107) can be accepted
as canonical.

Does NOT modify torch_camera_induced_visible_adjacency.py (Worklog 107),
torch_surfel_representative_diagnostics.py's adjacency-relevant behavior, or
any earlier worklog module's topology algorithm. This script only ADDS a new
diagnostic accumulation (per-primitive `forward_accepted`, exposed by the
worklog-108-continuation CUDA change in `diff_surfel_rasterization_diag`) and
re-derives statistics from WL107's own public functions, replayed unchanged.

CAVEAT 1 -- same-forward accepted-contributor / representative consistency:
  resolves the WL108 "36,051 representative-but-not-contributing" category
  directly, in the SAME forward CUDA execution, instead of by floating-point
  speculation across two different builds.

CAVEAT 2 -- patio/hedge overlap provenance: decomposes the largest
component's hedge-region membership, extracts the actual graph frontier
between the largest component and the hedge region, performs high-impact
bridge / split-impact analysis, and traces high-impact bridges back to their
supporting camera observations.

Neither Trust, latent surface, NURBS fitting/decomposition, occluded surface
construction, nor uncertain Gaussian proposal is implemented here. Non-
representative contributors are not attached to anything.
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
from osn_gs.render.torch_surfel_contribution_diagnostics import accumulate_renderer_contribution_evidence
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative
from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    CameraInducedAdjacencyConfig,
    accumulate_image_space_pairs,
    apply_secondary_geometric_gate,
    filter_by_3d_locality,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph
from osn_gs.surface.torch_discontinuity_first_surfel_partition import _auto_chunk_size, _fit_shape_operators, _knn, _predicted_delta_n_t, _tangent_plane_components
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"
_EPS = 1e-9

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_SAME_FORWARD = "SAME_FORWARD_CONTRIBUTOR_VS_REPRESENTATIVE"
VIEW_REPRESENTATIVE_COMPONENTS = "REPRESENTATIVE_ONLY_VISIBLE_COMPONENTS"
VIEW_LARGEST_MEMBERSHIP = "LARGEST_COMPONENT_MEMBERSHIP"
VIEW_LARGEST_HEDGE_MEMBERS = "LARGEST_COMPONENT_HEDGE_REGION_MEMBERS"
VIEW_FRONTIER = "PATIO_HEDGE_GRAPH_FRONTIER"
VIEW_HIGH_IMPACT_BRIDGES = "HIGH_IMPACT_BRIDGES"
VIEW_BRIDGE_SOURCE_VIEW = "HIGH_IMPACT_BRIDGE_SOURCE_VIEW"
VIEW_HEDGE_BACKBONE = "HEDGE_REPRESENTATIVE_BACKBONE"
VIEW_HEDGE_LARGEST_OVERLAP = "HEDGE_LARGEST_COMPONENT_OVERLAP"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532
_UNCUT_RGB = (0.08, 0.09, 0.11)

# Diagnostic-only impact bins, per directive section 6 -- NOT topology
# thresholds, never used to cut/keep any edge.
IMPACT_BINS = (10, 100, 1_000, 10_000)

ANCHOR_FRACTIONS = {
    "table": [(0.5, 0.48)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[gate closure] {message}", flush=True)


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


def _edge_highlight_colors(count: int, edges: torch.Tensor, base_rgb, highlight_rgb, device) -> torch.Tensor:
    degree = torch.zeros((count,), dtype=torch.float32, device=device)
    if int(edges.shape[0]) > 0:
        ones = torch.ones((int(edges.shape[0]),), dtype=torch.float32, device=device)
        degree.index_add_(0, edges[:, 0], ones)
        degree.index_add_(0, edges[:, 1], ones)
    ratio = (degree > 0).to(torch.float32)
    return _ramp(ratio, base_rgb, highlight_rgb)


def _find_bridges_with_split_impact(count: int, edges: torch.Tensor) -> list[dict[str, int]]:
    """Iterative Tarjan bridge-finding, augmented to report the exact
    split-impact of removing each bridge: since a DFS-tree bridge (parent,
    child) always separates the child's own subtree from the rest of the
    component, its subtree size IS one side of the split -- no separate
    per-bridge BFS/flood-fill is needed (O(V+E) total, not O(bridges * V))."""

    adjacency: list[list[int]] = [[] for _ in range(count)]
    edge_list = edges.tolist()
    for u, v in edge_list:
        adjacency[u].append(v)
        adjacency[v].append(u)

    disc = [-1] * count
    low = [0] * count
    subtree_size = [1] * count
    bridges: list[dict[str, int]] = []
    timer = 0

    for start in range(count):
        if disc[start] != -1:
            continue
        stack = [(start, -1, iter(adjacency[start]))]
        disc[start] = low[start] = timer
        timer += 1
        while stack:
            node, parent, neighbors = stack[-1]
            advanced = False
            for neighbor in neighbors:
                if neighbor == parent:
                    continue
                if disc[neighbor] == -1:
                    disc[neighbor] = low[neighbor] = timer
                    timer += 1
                    stack.append((neighbor, node, iter(adjacency[neighbor])))
                    advanced = True
                    break
                else:
                    low[node] = min(low[node], disc[neighbor])
            if not advanced:
                stack.pop()
                if stack:
                    parent_node = stack[-1][0]
                    low[parent_node] = min(low[parent_node], low[node])
                    subtree_size[parent_node] += subtree_size[node]
                    if low[node] > disc[parent_node]:
                        bridges.append({"u": parent_node, "v": node, "subtree_size": subtree_size[node]})
    return bridges


def _impact_bin(min_size: int, total: int) -> str:
    if min_size >= max(1, int(round(0.01 * total))):
        return ">=1%_of_component"
    for threshold in reversed(IMPACT_BINS):
        if min_size >= threshold:
            return f">={threshold}"
    return "<10"


def _trace_bridge_provenance(
    u: int, v: int, per_view_representative_ids: list[torch.Tensor], camera_meta_names: list[str], max_views: int = 8,
) -> list[dict[str, Any]]:
    """Directly re-scans each view's own (H, W) representative map for
    4-connectivity pixel pairs equal to (u, v) -- exactly the same relation
    `accumulate_image_space_pairs` already computed in aggregate, but here
    resolved back to individual (camera, pixel) observations for one
    specific edge. Cheap: only called for a small high-impact sample."""

    observations: list[dict[str, Any]] = []
    for view_index, rep in enumerate(per_view_representative_ids):
        if len(observations) >= max_views:
            break
        horiz = ((rep[:, :-1] == u) & (rep[:, 1:] == v)) | ((rep[:, :-1] == v) & (rep[:, 1:] == u))
        vert = ((rep[:-1, :] == u) & (rep[1:, :] == v)) | ((rep[:-1, :] == v) & (rep[1:, :] == u))
        if bool(horiz.any()):
            py, px = (horiz.nonzero(as_tuple=False)[0]).tolist()
            observations.append({"camera": camera_meta_names[view_index], "view_index": view_index, "pixel": [int(px), int(py)], "orientation": "horizontal"})
        elif bool(vert.any()):
            py, px = (vert.nonzero(as_tuple=False)[0]).tolist()
            observations.append({"camera": camera_meta_names[view_index], "view_index": view_index, "pixel": [int(px), int(py)], "orientation": "vertical"})
    return observations


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
    parser.add_argument("--max-bridge-nodes", type=int, default=600_000)
    parser.add_argument("--bridge-provenance-sample", type=int, default=20, help="how many top-impact bridges to trace to source observations")
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
    camera_names = [str(getattr(camera, "image_name", index)) for index, camera in enumerate(cameras)]
    _progress(f"train cameras: {camera_meta}")

    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())

    full_to_visible = torch.full((total_model_count,), -1, dtype=torch.int64, device=model.device)
    full_to_visible[visible_selector] = torch.arange(visible_count, dtype=torch.int64, device=model.device)

    _progress("[legacy] WL105 backward-gradient renderer-contribution evidence (unmodified)")
    contribution = accumulate_renderer_contribution_evidence(cameras, model, rasterizer, progress=_progress)
    ever_contributed_full = contribution.ever_contributed
    ever_contributed = ever_contributed_full[visible_selector]

    _progress("[same-forward] per-view representative_id + forward_accepted (worklog-108-continuation diagnostic build)")
    per_view_representative_ids: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    representative_view_count_full = torch.zeros((total_model_count,), dtype=torch.int32, device=model.device)
    ever_forward_accepted_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        represented_ids = torch.unique(rep_full[valid])
        ever_representative_full[represented_ids] = True
        representative_view_count_full[represented_ids] += 1
        forward_accepted_this_view = diag["forward_accepted"].to(torch.bool).reshape(-1)
        ever_forward_accepted_full |= forward_accepted_this_view
        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_representative_ids.append(rep_remapped.detach())
        del diag
        if index % 20 == 0:
            _progress(f"same-forward view {index + 1}/{len(cameras)}")
    ever_representative = ever_representative_full[visible_selector]
    representative_view_count = representative_view_count_full[visible_selector]
    ever_forward_accepted = ever_forward_accepted_full[visible_selector]
    _progress(f"contributing(WL105)={int(ever_contributed.sum())}/{visible_count} representative(WL107)={int(ever_representative.sum())}/{visible_count} "
              f"forward_accepted(same-execution)={int(ever_forward_accepted.sum())}/{visible_count}")

    # --- CAVEAT 1 §1-3: same-forward cross-tab, resolving the 36,051 category directly ---
    representative_without_forward_accepted = int((ever_representative & ~ever_forward_accepted).sum())
    same_forward_cross_tab = {
        "representative_and_forward_accepted": int((ever_representative & ever_forward_accepted).sum()),
        "representative_and_not_forward_accepted": representative_without_forward_accepted,
        "forward_accepted_and_not_representative": int((ever_forward_accepted & ~ever_representative).sum()),
        "neither": int((~ever_representative & ~ever_forward_accepted).sum()),
    }
    representative_implies_forward_accepted_same_execution = representative_without_forward_accepted == 0
    _progress(f"[CAVEAT 1] same-forward cross-tab {same_forward_cross_tab} "
              f"representative_implies_forward_accepted={representative_implies_forward_accepted_same_execution}")

    legacy_cross_tab = {
        "contributing_and_representative": int((ever_contributed & ever_representative).sum()),
        "contributing_and_not_representative": int((ever_contributed & ~ever_representative).sum()),
        "not_contributing_and_representative": int((~ever_contributed & ever_representative).sum()),
        "not_contributing_and_not_representative": int((~ever_contributed & ~ever_representative).sum()),
    }
    discrepancy_mask = ever_representative & ~ever_contributed  # WL108's 36,051 category
    discrepancy_count = int(discrepancy_mask.sum())
    discrepancy_is_forward_accepted = int((discrepancy_mask & ever_forward_accepted).sum())
    discrepancy_explanation = {
        "legacy_discrepancy_count_representative_but_not_wl105_contributing": discrepancy_count,
        "of_those_that_ARE_forward_accepted_same_execution": discrepancy_is_forward_accepted,
        "fraction_explained_by_forward_accepted_but_wl105_missed": (discrepancy_is_forward_accepted / discrepancy_count) if discrepancy_count else None,
        "conclusion": (
            "every representative-but-not-WL105-contributing surfel IS forward_accepted in the same execution "
            "that produced its representative status -- confirms the discrepancy is WL105's separate backward-"
            "gradient diagnostic (built on a DIFFERENT CUDA execution/build) failing to register a contribution "
            "that genuinely occurred, not a same-forward inconsistency in WL107's own signal"
        ) if discrepancy_count and discrepancy_is_forward_accepted == discrepancy_count else (
            "NOT fully explained by forward_accepted -- some representative-but-not-WL105-contributing surfels "
            "are also not forward_accepted in this same execution; this is a genuine open anomaly, not resolved"
        ),
    }
    _progress(f"[CAVEAT 1 resolution] {discrepancy_explanation}")

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

        _progress("[WL107 replay, unchanged] build_candidate_graph")
        graph = build_candidate_graph(orientation, config.local, progress=_progress)

        _progress("[WL107 replay, unchanged] accumulate_image_space_pairs")
        raw_pairs, raw_view_support = accumulate_image_space_pairs(count, per_view_representative_ids, progress=_progress)
        local_pairs, local_mask = filter_by_3d_locality(raw_pairs, count, graph)
        local_view_support = raw_view_support[local_mask]

        _progress("[WL107 replay, unchanged] apply_secondary_geometric_gate")
        geometry = apply_secondary_geometric_gate(local_pairs, orientation, config, progress=_progress)
        kept_mask = geometry["kept_mask"]
        positive_edges = local_pairs[kept_mask]
        positive_view_support = local_view_support[kept_mask]

        roots = _connected_component_roots(count, positive_edges, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
        subset_ids = subset_id_of_position[inverse]
        subset_sizes = counts[order]

        wl107_replay_stats = {
            "visible_component_count": int(order.shape[0]),
            "largest_component_surfel_fraction": float(subset_sizes[0]) / count,
            "singleton_surfel_count": int((subset_sizes == 1).sum()),
            "singleton_fraction": float((subset_sizes == 1).sum()) / count,
        }
        _progress(f"[WL107 replay stats] {wl107_replay_stats}")

        degree = torch.zeros((count,), dtype=torch.int32, device=device)
        if int(positive_edges.shape[0]) > 0:
            ones = torch.ones((int(positive_edges.shape[0]),), dtype=torch.int32, device=device)
            degree.index_add_(0, positive_edges[:, 0], ones)
            degree.index_add_(0, positive_edges[:, 1], ones)
        rep_degree = degree[ever_representative]
        rep_count = int(ever_representative.sum())
        rep_degree0 = int((rep_degree == 0).sum())
        representative_backbone_stats = {
            "representative_surfel_count": rep_count,
            "representative_degree0_count": rep_degree0,
            "representative_connected_fraction": (rep_count - rep_degree0) / rep_count if rep_count else 0.0,
        }
        _progress(f"[representative backbone replay] {representative_backbone_stats}")

        # --- deterministic pixel anchors (unchanged convention) ---
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

        largest_subset_id = int(order[0].item())
        largest_member_mask = subset_ids == largest_subset_id

        # --- CAVEAT 2 §4: hedge-region membership -- explicitly a WORKING
        # INTERPRETATION ONLY (nearest-anchor-by-3D-distance heuristic, no
        # ground/vegetation threshold, no architecture claim). ---
        hedge_region_mask = torch.zeros((count,), dtype=torch.bool, device=device)
        if anchor_visible_ids.get("hedge"):
            hedge_anchor_positions = positions[torch.tensor(anchor_visible_ids["hedge"], device=device)]
            other_ids = anchor_visible_ids.get("table", []) + anchor_visible_ids.get("patio", [])
            dist_to_hedge = torch.cdist(positions, hedge_anchor_positions).min(dim=1).values
            if other_ids:
                dist_to_other = torch.cdist(positions, positions[torch.tensor(other_ids, device=device)]).min(dim=1).values
                hedge_region_mask = dist_to_hedge < dist_to_other
            else:
                hedge_region_mask = torch.ones((count,), dtype=torch.bool, device=device)

        hedge_in_largest_mask = hedge_region_mask & largest_member_mask
        hedge_total = int(hedge_region_mask.sum())

        # per-node planarity proxy: local shape-operator norm (already the
        # exact quantity `apply_secondary_geometric_gate` computes internally
        # for the residual test -- recomputed once here, over ALL nodes, so
        # every hedge-region member gets a value, not only edge endpoints).
        k_shape = min(config.resolved_shape_operator_neighbor_count() or config.local.neighbor_count, max(count - 1, 1))
        chunk_size = int(config.local.knn_chunk_size) or _auto_chunk_size(count, device)
        neighbor_index, _ = _knn(positions, k_shape, chunk_size, _progress)
        shape_operator = _fit_shape_operators(positions, normals, orientation.tangent_axis_u, orientation.tangent_axis_v, neighbor_index, float(config.shape_operator_ridge))
        planarity_proxy = torch.linalg.matrix_norm(shape_operator)  # larger = less planar (more curvature)

        hedge_ids = torch.nonzero(hedge_region_mask, as_tuple=False).reshape(-1)
        hedge_decomposition = {
            "hedge_region_surfel_count": hedge_total,
            "hedge_region_surfels_in_largest_component_count": int(hedge_in_largest_mask.sum()),
            "hedge_region_surfels_in_largest_component_fraction_of_hedge": (int(hedge_in_largest_mask.sum()) / hedge_total) if hedge_total else 0.0,
            "position_bbox": {
                "min": positions[hedge_ids].min(dim=0).values.tolist(), "max": positions[hedge_ids].max(dim=0).values.tolist(),
            } if hedge_total else {},
            "position_axis1_height_distribution": _distribution(positions[hedge_ids, 1]) if hedge_total else {},
            "normal_x_distribution": _distribution(normals[hedge_ids, 0]) if hedge_total else {},
            "normal_y_distribution": _distribution(normals[hedge_ids, 1]) if hedge_total else {},
            "normal_z_distribution": _distribution(normals[hedge_ids, 2]) if hedge_total else {},
            "planarity_proxy_distribution": _distribution(planarity_proxy[hedge_ids]) if hedge_total else {},
            "representative_view_count_distribution": _distribution(representative_view_count[hedge_ids].to(torch.float32)) if hedge_total else {},
        }
        _progress(f"[CAVEAT 2 hedge decomposition] {hedge_decomposition}")

        # --- CAVEAT 2 §5-6: patio/hedge graph frontier -- real edges only,
        # restricted to the largest component (both endpoints already
        # members), one endpoint hedge-region, the other not. ---
        edge_left, edge_right = positive_edges[:, 0], positive_edges[:, 1]
        edge_in_largest = largest_member_mask[edge_left] & largest_member_mask[edge_right]
        edge_hedge_left = hedge_region_mask[edge_left]
        edge_hedge_right = hedge_region_mask[edge_right]
        frontier_mask = edge_in_largest & (edge_hedge_left != edge_hedge_right)
        frontier_edges = positive_edges[frontier_mask]
        frontier_view_support = positive_view_support[frontier_mask]

        # per-frontier-edge diagnostics: distance, distance/local-spacing,
        # normals, residual, positional stat -- recomputed with the exact
        # same formula `apply_secondary_geometric_gate` uses internally
        # (that function itself only returns pass/fail booleans, not the
        # raw per-edge values needed for this report).
        def _edge_diagnostics(edges: torch.Tensor) -> dict[str, torch.Tensor]:
            left, right = edges[:, 0], edges[:, 1]
            delta_x = positions[right] - positions[left]
            distance = delta_x.norm(dim=-1)
            spacing = (graph.local_spacing[left] + graph.local_spacing[right]) * 0.5
            delta_x_t_left = _tangent_plane_components(delta_x, orientation.tangent_axis_u[left], orientation.tangent_axis_v[left])
            delta_x_t_right = _tangent_plane_components(-delta_x, orientation.tangent_axis_u[right], orientation.tangent_axis_v[right])
            sign_lr = torch.where((normals[left] * normals[right]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
            aligned_right_normal = normals[right] * sign_lr
            delta_n_left = aligned_right_normal - normals[left]
            delta_n_t_left = _tangent_plane_components(delta_n_left, orientation.tangent_axis_u[left], orientation.tangent_axis_v[left])
            aligned_left_normal = normals[left] * sign_lr
            delta_n_right = aligned_left_normal - normals[right]
            delta_n_t_right = _tangent_plane_components(delta_n_right, orientation.tangent_axis_u[right], orientation.tangent_axis_v[right])
            predicted_left = _predicted_delta_n_t(shape_operator[left], delta_x_t_left)
            predicted_right = _predicted_delta_n_t(shape_operator[right], delta_x_t_right)
            residual_left = (delta_n_t_left - predicted_left).norm(dim=-1)
            residual_right = (delta_n_t_right - predicted_right).norm(dim=-1)
            edge_residual = torch.minimum(residual_left, residual_right)
            average_normal = torch.nn.functional.normalize(normals[left] + aligned_right_normal, dim=-1, eps=_EPS)
            signed_normal_offset = (delta_x * average_normal).sum(dim=-1)
            tangential_offset = (delta_x - signed_normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
            normal_offset_ratio = signed_normal_offset.abs() / tangential_offset.clamp_min(_EPS)
            return {
                "distance": distance, "distance_over_local_spacing": distance / spacing.clamp_min(_EPS),
                "residual": edge_residual, "normal_offset_ratio": normal_offset_ratio,
                "normal_left": normals[left], "normal_right": normals[right],
            }

        frontier_diag = _edge_diagnostics(frontier_edges) if int(frontier_edges.shape[0]) > 0 else None
        frontier_summary = {
            "frontier_edge_count": int(frontier_edges.shape[0]),
            "distance_distribution": _distribution(frontier_diag["distance"]) if frontier_diag else {},
            "distance_over_local_spacing_distribution": _distribution(frontier_diag["distance_over_local_spacing"]) if frontier_diag else {},
            "residual_distribution": _distribution(frontier_diag["residual"]) if frontier_diag else {},
            "normal_offset_ratio_distribution": _distribution(frontier_diag["normal_offset_ratio"]) if frontier_diag else {},
            "view_support_distribution": _distribution(frontier_view_support.to(torch.float32)) if int(frontier_edges.shape[0]) > 0 else {},
            "single_view_supported_fraction": float((frontier_view_support == 1).float().mean()) if int(frontier_edges.shape[0]) > 0 else 0.0,
        }
        # per-edge full diagnostic rows, capped for report size
        frontier_edge_rows = []
        for row_index in range(min(int(frontier_edges.shape[0]), 500)):
            u, v = int(frontier_edges[row_index, 0].item()), int(frontier_edges[row_index, 1].item())
            frontier_edge_rows.append({
                "endpoints_visible_index": [u, v],
                "endpoints_position": [positions[u].tolist(), positions[v].tolist()],
                "distance": float(frontier_diag["distance"][row_index].item()),
                "distance_over_local_spacing": float(frontier_diag["distance_over_local_spacing"][row_index].item()),
                "normal_left": normals[u].tolist(), "normal_right": normals[v].tolist(),
                "residual": float(frontier_diag["residual"][row_index].item()),
                "normal_offset_ratio": float(frontier_diag["normal_offset_ratio"][row_index].item()),
                "supporting_view_count": int(frontier_view_support[row_index].item()),
            })
        _progress(f"[frontier summary] {frontier_summary}")

        # --- CAVEAT 2 §7: high-impact bridge analysis over the largest component ---
        largest_edge_in_component = largest_member_mask[positive_edges[:, 0]] & largest_member_mask[positive_edges[:, 1]]
        largest_component_edges = positive_edges[largest_edge_in_component]
        largest_component_view_support = positive_view_support[largest_edge_in_component]
        largest_component_size = int(largest_member_mask.sum())

        bridge_analysis: dict[str, Any] = {"performed": largest_component_size <= arguments.max_bridge_nodes}
        high_impact_bridge_rows: list[dict[str, Any]] = []
        high_impact_frontier_bridge_rows: list[dict[str, Any]] = []
        bridge_provenance_rows: list[dict[str, Any]] = []
        bridge_edge_mask_global = torch.zeros((int(positive_edges.shape[0]),), dtype=torch.bool, device=device)

        if bridge_analysis["performed"] and int(largest_component_edges.shape[0]) > 0:
            local_ids = torch.unique(largest_component_edges.reshape(-1))
            remap = torch.full((count,), -1, dtype=torch.int64, device=device)
            remap[local_ids] = torch.arange(int(local_ids.shape[0]), dtype=torch.int64, device=device)
            local_edges = remap[largest_component_edges].cpu()
            _progress(f"[bridge] searching bridges over {int(local_ids.shape[0])} nodes / {int(local_edges.shape[0])} edges")
            bridges = _find_bridges_with_split_impact(int(local_ids.shape[0]), local_edges)

            split_impacts = []
            for bridge in bridges:
                size_a = bridge["subtree_size"]
                size_b = largest_component_size - size_a
                min_size = min(size_a, size_b)
                split_impacts.append({
                    "u_local": bridge["u"], "v_local": bridge["v"], "size_a": size_a, "size_b": size_b,
                    "min_size": min_size, "fraction_separated": min_size / largest_component_size,
                    "bin": _impact_bin(min_size, largest_component_size),
                })
            split_impacts.sort(key=lambda entry: entry["min_size"], reverse=True)

            bin_counts: dict[str, int] = {}
            for entry in split_impacts:
                bin_counts[entry["bin"]] = bin_counts.get(entry["bin"], 0) + 1
            min_size_values = torch.tensor([entry["min_size"] for entry in split_impacts], dtype=torch.float32) if split_impacts else torch.zeros((0,))

            # vectorized check: of ALL bridges (not just the top-N sample),
            # how many actually cross the hedge/non-hedge region boundary --
            # i.e. would removing them disconnect (part of) the hedge region
            # from (part of) the rest of the largest component. Cheap:
            # derived directly from hedge_region_mask, no per-bridge search.
            if bridges:
                bridge_u_global = local_ids[torch.tensor([b["u"] for b in bridges], dtype=torch.int64, device=device)]
                bridge_v_global = local_ids[torch.tensor([b["v"] for b in bridges], dtype=torch.int64, device=device)]
                bridge_is_frontier = hedge_region_mask[bridge_u_global] != hedge_region_mask[bridge_v_global]
                frontier_bridge_min_sizes = torch.tensor([split_impacts[i]["min_size"] for i in range(len(split_impacts))
                                                           if hedge_region_mask[local_ids[split_impacts[i]["u_local"]]].item()
                                                           != hedge_region_mask[local_ids[split_impacts[i]["v_local"]]].item()],
                                                          dtype=torch.float32) if len(split_impacts) else torch.zeros((0,))
                frontier_bridge_count = int(bridge_is_frontier.sum())
            else:
                frontier_bridge_count = 0
                frontier_bridge_min_sizes = torch.zeros((0,))

            bridge_analysis.update({
                "bridge_count": len(bridges),
                "min_size_distribution": _distribution(min_size_values),
                "impact_bin_counts": bin_counts,
                "frontier_crossing_bridge_count": frontier_bridge_count,
                "frontier_crossing_bridge_fraction_of_all_bridges": (frontier_bridge_count / len(bridges)) if bridges else 0.0,
                "frontier_crossing_bridge_min_size_distribution": _distribution(frontier_bridge_min_sizes),
            })
            _progress(f"[bridge] {len(bridges)} bridges, impact bins: {bin_counts}, "
                      f"frontier-crossing bridges: {frontier_bridge_count} ({bridge_analysis['frontier_crossing_bridge_fraction_of_all_bridges']:.4%})")

            for entry in split_impacts[: max(1, arguments.bridge_provenance_sample)]:
                u_global = int(local_ids[entry["u_local"]].item())
                v_global = int(local_ids[entry["v_local"]].item())
                edge_row_mask = (largest_component_edges[:, 0] == min(u_global, v_global)) & (largest_component_edges[:, 1] == max(u_global, v_global))
                support = int(largest_component_view_support[edge_row_mask].max().item()) if bool(edge_row_mask.any()) else None
                is_frontier = bool(hedge_region_mask[u_global].item() != hedge_region_mask[v_global].item())
                row = {
                    "endpoints_visible_index": [u_global, v_global],
                    "endpoints_position": [positions[u_global].tolist(), positions[v_global].tolist()],
                    "size_a": entry["size_a"], "size_b": entry["size_b"], "min_size": entry["min_size"],
                    "fraction_separated": entry["fraction_separated"], "bin": entry["bin"],
                    "supporting_view_count": support,
                    "is_patio_hedge_frontier_edge": is_frontier,
                    "u_in_hedge_region": bool(hedge_region_mask[u_global].item()), "v_in_hedge_region": bool(hedge_region_mask[v_global].item()),
                }
                high_impact_bridge_rows.append(row)
                global_match = (positive_edges[:, 0] == min(u_global, v_global)) & (positive_edges[:, 1] == max(u_global, v_global))
                bridge_edge_mask_global = bridge_edge_mask_global | global_match

                provenance = _trace_bridge_provenance(u_global, v_global, per_view_representative_ids, camera_names)
                bridge_provenance_rows.append({"endpoints_visible_index": [u_global, v_global], "observations": provenance, "supporting_view_count": support})

            single_view_high_impact = sum(1 for row in high_impact_bridge_rows if row["supporting_view_count"] == 1)
            multi_view_high_impact = len(high_impact_bridge_rows) - single_view_high_impact
            bridge_analysis["high_impact_sample_view_support"] = {
                "sample_size": len(high_impact_bridge_rows), "single_view_supported": single_view_high_impact, "multi_view_supported": multi_view_high_impact,
            }
            _progress(f"[high-impact bridge sample] single_view={single_view_high_impact} multi_view={multi_view_high_impact} of {len(high_impact_bridge_rows)}")

            # directive section 8: specifically trace the highest-impact
            # FRONTIER-CROSSING bridges (patio-core <-> hedge-region), not
            # just the global top-N (which turned out, per the vectorized
            # check above, to contain zero frontier-crossing edges).
            frontier_split_impacts = [
                entry for entry in split_impacts
                if bool(hedge_region_mask[local_ids[entry["u_local"]]].item() != hedge_region_mask[local_ids[entry["v_local"]]].item())
            ]
            for entry in frontier_split_impacts[: max(1, arguments.bridge_provenance_sample)]:
                u_global = int(local_ids[entry["u_local"]].item())
                v_global = int(local_ids[entry["v_local"]].item())
                edge_row_mask = (largest_component_edges[:, 0] == min(u_global, v_global)) & (largest_component_edges[:, 1] == max(u_global, v_global))
                support = int(largest_component_view_support[edge_row_mask].max().item()) if bool(edge_row_mask.any()) else None
                row = {
                    "endpoints_visible_index": [u_global, v_global],
                    "endpoints_position": [positions[u_global].tolist(), positions[v_global].tolist()],
                    "size_a": entry["size_a"], "size_b": entry["size_b"], "min_size": entry["min_size"],
                    "fraction_separated": entry["fraction_separated"], "bin": entry["bin"],
                    "supporting_view_count": support,
                    "u_in_hedge_region": bool(hedge_region_mask[u_global].item()), "v_in_hedge_region": bool(hedge_region_mask[v_global].item()),
                }
                high_impact_frontier_bridge_rows.append(row)
                global_match = (positive_edges[:, 0] == min(u_global, v_global)) & (positive_edges[:, 1] == max(u_global, v_global))
                bridge_edge_mask_global = bridge_edge_mask_global | global_match
                provenance = _trace_bridge_provenance(u_global, v_global, per_view_representative_ids, camera_names)
                bridge_provenance_rows.append({"endpoints_visible_index": [u_global, v_global], "observations": provenance, "supporting_view_count": support, "is_frontier_crossing": True})

            frontier_single_view = sum(1 for row in high_impact_frontier_bridge_rows if row["supporting_view_count"] == 1)
            bridge_analysis["high_impact_frontier_bridge_sample_view_support"] = {
                "sample_size": len(high_impact_frontier_bridge_rows),
                "single_view_supported": frontier_single_view,
                "multi_view_supported": len(high_impact_frontier_bridge_rows) - frontier_single_view,
                "max_min_size_among_frontier_bridges": frontier_split_impacts[0]["min_size"] if frontier_split_impacts else 0,
            }
            _progress(f"[high-impact frontier bridge sample] {bridge_analysis['high_impact_frontier_bridge_sample_view_support']}")
        else:
            _progress("[bridge] skipped (component too large or no edges)")

        report = {
            "batch": "arch/2dgs-coverage-first-surface, Worklog 109 (WL108 caveat closure)",
            "checkpoint": str(arguments.checkpoint),
            "primitive": primitive,
            "iteration": int(payload.get("iteration", 0)),
            "primitive_accounting": {"total_model_surfel_count": total_model_count, "visible_domain_surfel_count": visible_count},
            "camera_meta": camera_meta,
            "same_forward_cross_tab": same_forward_cross_tab,
            "representative_implies_forward_accepted_same_execution": representative_implies_forward_accepted_same_execution,
            "legacy_cross_tab_wl105_vs_wl107": legacy_cross_tab,
            "discrepancy_resolution": discrepancy_explanation,
            "wl107_replay_stats": wl107_replay_stats,
            "representative_backbone_replay_stats": representative_backbone_stats,
            "hedge_region_decomposition_WORKING_INTERPRETATION_ONLY": hedge_decomposition,
            "patio_hedge_frontier_summary": frontier_summary,
            "patio_hedge_frontier_edge_sample": frontier_edge_rows,
            "high_impact_bridge_analysis": bridge_analysis,
            "high_impact_bridge_sample": high_impact_bridge_rows,
            "high_impact_frontier_bridge_sample": high_impact_frontier_bridge_rows,
            "high_impact_bridge_provenance": bridge_provenance_rows,
            "deterministic_anchor_visible_ids": anchor_visible_ids,
            "runtime_seconds": {"total": time.time() - started},
        }

        # --- colors / exports ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        same_forward_category = torch.zeros((count,), dtype=torch.int64, device=device)
        same_forward_category = torch.where(ever_representative & ever_forward_accepted, torch.full_like(same_forward_category, 1), same_forward_category)
        same_forward_category = torch.where(ever_forward_accepted & ~ever_representative, torch.full_like(same_forward_category, 2), same_forward_category)
        same_forward_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        same_forward_colors[same_forward_category == 1] = torch.tensor((0.15, 0.75, 0.95), dtype=torch.float32, device=device)
        same_forward_colors[same_forward_category == 2] = torch.tensor((1.0, 0.55, 0.0), dtype=torch.float32, device=device)

        representative_component_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        representative_component_colors[ever_representative] = _subset_partition_colors(subset_ids[ever_representative])

        largest_membership_colors = _ramp(largest_member_mask.to(torch.float32), _UNCUT_RGB, (0.9, 0.75, 0.1))
        largest_hedge_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        largest_hedge_colors[largest_member_mask & ~hedge_region_mask] = torch.tensor((0.9, 0.75, 0.1), dtype=torch.float32, device=device)
        largest_hedge_colors[hedge_in_largest_mask] = torch.tensor((1.0, 0.1, 0.55), dtype=torch.float32, device=device)

        frontier_colors = _edge_highlight_colors(count, frontier_edges, _UNCUT_RGB, (1.0, 0.1, 0.55), device)
        bridge_colors = _edge_highlight_colors(count, positive_edges[bridge_edge_mask_global], _UNCUT_RGB, (1.0, 0.1, 0.1), device)

        provenance_highlight_ids = []
        for row in bridge_provenance_rows:
            provenance_highlight_ids.extend(row["endpoints_visible_index"])
        provenance_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        if provenance_highlight_ids:
            provenance_colors[torch.tensor(provenance_highlight_ids, device=device, dtype=torch.int64)] = torch.tensor((1.0, 0.1, 0.1), dtype=torch.float32, device=device)

        hedge_backbone_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        hedge_rep_only_mask = hedge_region_mask & ever_representative
        hedge_backbone_colors[hedge_rep_only_mask] = _subset_partition_colors(subset_ids[hedge_rep_only_mask])

        hedge_overlap_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        hedge_overlap_colors[hedge_region_mask & ~largest_member_mask] = torch.tensor((0.15, 0.75, 0.95), dtype=torch.float32, device=device)
        hedge_overlap_colors[largest_member_mask & ~hedge_region_mask] = torch.tensor((0.9, 0.75, 0.1), dtype=torch.float32, device=device)
        hedge_overlap_colors[hedge_in_largest_mask] = torch.tensor((1.0, 0.1, 0.55), dtype=torch.float32, device=device)

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_SAME_FORWARD: _rgb_to_f_dc(same_forward_colors),
            VIEW_REPRESENTATIVE_COMPONENTS: _rgb_to_f_dc(representative_component_colors),
            VIEW_LARGEST_MEMBERSHIP: _rgb_to_f_dc(largest_membership_colors),
            VIEW_LARGEST_HEDGE_MEMBERS: _rgb_to_f_dc(largest_hedge_colors),
            VIEW_FRONTIER: _rgb_to_f_dc(frontier_colors),
            VIEW_HIGH_IMPACT_BRIDGES: _rgb_to_f_dc(bridge_colors),
            VIEW_BRIDGE_SOURCE_VIEW: _rgb_to_f_dc(provenance_colors),
            VIEW_HEDGE_BACKBONE: _rgb_to_f_dc(hedge_backbone_colors),
            VIEW_HEDGE_LARGEST_OVERLAP: _rgb_to_f_dc(hedge_overlap_colors),
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
    report_path = output_root / "renderer_native_topology_gate_closure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
