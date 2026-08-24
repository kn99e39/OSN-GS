"""Worklog 108 -- Renderer-Native Surface Representative Backbone: architecture
gate / accounting audit for Worklog 107.

Does NOT modify torch_camera_induced_visible_adjacency.py (Worklog 107),
torch_surfel_representative_diagnostics.py, torch_surfel_contribution_
diagnostics.py (Worklog 105), or any earlier worklog module. Reuses Worklog
107's own public functions (`accumulate_image_space_pairs`,
`filter_by_3d_locality`, `apply_secondary_geometric_gate`,
`partition_camera_induced_visible_adjacency`) directly to obtain the exact
same final result PLUS the intermediate arrays needed for this audit's
attribution -- no new topology algorithm, no threshold change.

Runs, on the SAME trained 2DGS checkpoint as Worklogs 96-107:

    A. ORIGINAL_2DGS_SCENE
    B. ALL_RENDERER_CONTRIBUTORS
    C. RENDERER_SURFACE_REPRESENTATIVES
    D. REPRESENTATIVE_ONLY_VISIBLE_COMPONENTS
    E. REPRESENTATIVE_SINGLETON_CAUSE_VIEW
    F. LARGEST_COMPONENT_MEMBERSHIP_VIEW
    G. LARGE_COMPONENT_BRIDGE_VIEW
    H. 3D_LOCALITY_REJECTION_VIEW
    I. HEDGE_REPRESENTATIVE_BACKBONE
    J. HEDGE_NON_REPRESENTATIVE_SUPPORT

Neither Trust, latent surface, NURBS, occluded surface construction, uncertain
Gaussian proposal, nor non-representative-contributor attachment is
implemented here.
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
    REASON_GEOMETRIC_DISCONTINUITY,
    REASON_POSITIONAL_SHEET_SEPARATION,
    CameraInducedAdjacencyConfig,
    accumulate_image_space_pairs,
    apply_secondary_geometric_gate,
    filter_by_3d_locality,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_ALL_CONTRIBUTORS = "ALL_RENDERER_CONTRIBUTORS"
VIEW_REPRESENTATIVES = "RENDERER_SURFACE_REPRESENTATIVES"
VIEW_REPRESENTATIVE_COMPONENTS = "REPRESENTATIVE_ONLY_VISIBLE_COMPONENTS"
VIEW_SINGLETON_CAUSE = "REPRESENTATIVE_SINGLETON_CAUSE_VIEW"
VIEW_LARGEST_MEMBERSHIP = "LARGEST_COMPONENT_MEMBERSHIP_VIEW"
VIEW_BRIDGE = "LARGE_COMPONENT_BRIDGE_VIEW"
VIEW_LOCALITY_REJECTION = "3D_LOCALITY_REJECTION_VIEW"
VIEW_HEDGE_BACKBONE = "HEDGE_REPRESENTATIVE_BACKBONE"
VIEW_HEDGE_SUPPORT = "HEDGE_NON_REPRESENTATIVE_SUPPORT"

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532
_UNCUT_RGB = (0.08, 0.09, 0.11)

SINGLETON_CAUSE_NO_NEIGHBOR = "REPRESENTATIVE_HAS_NO_DISTINCT_PIXEL_NEIGHBOR_RELATION"
SINGLETON_CAUSE_FAILS_LOCALITY = "CAMERA_PAIR_GENERATED_BUT_FAILS_3D_LOCALITY"
SINGLETON_CAUSE_GEOMETRIC_DISCONTINUITY = "PASSES_LOCALITY_BUT_GEOMETRIC_DISCONTINUITY"
SINGLETON_CAUSE_POSITIONAL_SEPARATION = "PASSES_LOCALITY_BUT_POSITIONAL_SEPARATION"
SINGLETON_CAUSE_OTHER = "OTHER_EXPLICITLY_REPORTED_CAUSE"
SINGLETON_CAUSES = (
    SINGLETON_CAUSE_NO_NEIGHBOR, SINGLETON_CAUSE_FAILS_LOCALITY,
    SINGLETON_CAUSE_GEOMETRIC_DISCONTINUITY, SINGLETON_CAUSE_POSITIONAL_SEPARATION, SINGLETON_CAUSE_OTHER,
)
_SINGLETON_CAUSE_RGB = {
    SINGLETON_CAUSE_NO_NEIGHBOR: (0.6, 0.05, 0.05),
    SINGLETON_CAUSE_FAILS_LOCALITY: (0.95, 0.55, 0.05),
    SINGLETON_CAUSE_GEOMETRIC_DISCONTINUITY: (1.0, 0.15, 0.55),
    SINGLETON_CAUSE_POSITIONAL_SEPARATION: (0.7, 0.15, 0.9),
    SINGLETON_CAUSE_OTHER: (0.9, 0.9, 0.1),
}

# Deterministic pixel anchors on the established preview camera / resolution,
# matching Worklog 107's own color-sampling fractional coordinates exactly
# (w//2,h*0.48 for table; (0.15,0.85)/(0.85,0.9) for patio;
# (0.1,0.1)/(0.9,0.15)/(0.5,0.05) for hedge/background).
ANCHOR_FRACTIONS = {
    "table": [(0.5, 0.48)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}


def _progress(message: str) -> None:
    print(f"[representative backbone audit] {message}", flush=True)


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


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    if int(values.shape[0]) == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = torch.sort(values.to(torch.float64)).values

    def _percentile(fraction: float) -> float:
        position = min(int(sorted_values.shape[0]) - 1, max(0, int(round(fraction * (int(sorted_values.shape[0]) - 1)))))
        return float(sorted_values[position].item())

    return {"min": _percentile(0.0), "median": _percentile(0.5), "mean": float(sorted_values.mean().item()), "p95": _percentile(0.95), "max": _percentile(1.0)}


def _component_size_stats(sizes: torch.Tensor) -> dict[str, Any]:
    return _distribution(sizes.to(torch.float32))


def _scatter_or(count: int, edges: torch.Tensor, edge_flag: torch.Tensor, device) -> torch.Tensor:
    node_flag = torch.zeros((count,), dtype=torch.bool, device=device)
    if int(edges.shape[0]) == 0 or not bool(edge_flag.any()):
        return node_flag
    picked = edges[edge_flag]
    node_flag[picked[:, 0]] = True
    node_flag[picked[:, 1]] = True
    return node_flag


def _find_bridges(count: int, edges: torch.Tensor) -> list[tuple[int, int]]:
    """Iterative Tarjan bridge-finding restricted to the (small enough)
    induced subgraph passed in. Returns a list of (u, v) bridge edges."""

    adjacency: list[list[int]] = [[] for _ in range(count)]
    edge_list = edges.tolist()
    for u, v in edge_list:
        adjacency[u].append(v)
        adjacency[v].append(u)

    disc = [-1] * count
    low = [0] * count
    bridges: list[tuple[int, int]] = []
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
                    if low[node] > disc[parent_node]:
                        bridges.append((parent_node, node))
    return bridges


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
    parser.add_argument("--max-bridge-nodes", type=int, default=600_000, help="skip exact bridge-finding above this component size")
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

    _progress("[contribution] renderer-contribution evidence (Worklog 105, unmodified)")
    contribution = accumulate_renderer_contribution_evidence(cameras, model, rasterizer, progress=_progress)
    ever_contributed_full = contribution.ever_contributed
    ever_contributed = ever_contributed_full[visible_selector]

    _progress("[representative] per-view renderer surface-representative maps (Worklog 107 diagnostic build, unmodified)")
    per_view_representative_ids: list[torch.Tensor] = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    anchor_pixel_ids: dict[str, list[int]] = {}
    for index, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        ever_representative_full[rep_full[valid]] = True
        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_representative_ids.append(rep_remapped.detach())
        del diag
        if index % 20 == 0:
            _progress(f"representative view {index + 1}/{len(cameras)}")
    ever_representative = ever_representative_full[visible_selector]
    _progress(f"contributing={int(ever_contributed.sum())}/{visible_count} representative={int(ever_representative.sum())}/{visible_count}")

    # --- §1: resolve the accounting discrepancy with an exact cross-tab ---
    cross_tab = {
        "contributing_and_representative": int((ever_contributed & ever_representative).sum()),
        "contributing_and_not_representative": int((ever_contributed & ~ever_representative).sum()),
        "not_contributing_and_representative": int((~ever_contributed & ever_representative).sum()),
        "not_contributing_and_not_representative": int((~ever_contributed & ~ever_representative).sum()),
    }
    representative_is_strict_subset_of_contributing = cross_tab["not_contributing_and_representative"] == 0
    _progress(f"[cross-tab] {cross_tab} strict_subset={representative_is_strict_subset_of_contributing}")

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
        config = CameraInducedAdjacencyConfig(local=local_config)

        _progress("[WL107 replay] build_candidate_graph (unmodified)")
        graph = build_candidate_graph(orientation, config.local, progress=_progress)

        _progress("[WL107 replay] accumulate_image_space_pairs (unmodified)")
        raw_pairs, raw_view_support = accumulate_image_space_pairs(count, per_view_representative_ids, progress=_progress)
        local_pairs, local_mask = filter_by_3d_locality(raw_pairs, count, graph)
        local_view_support = raw_view_support[local_mask]
        rejected_pairs = raw_pairs[~local_mask]

        _progress("[WL107 replay] apply_secondary_geometric_gate (unmodified)")
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
        _progress(f"[WL107 replay cross-check] {int(order.shape[0])} components largest={float(subset_sizes[0])/count:.4f} "
                  f"singleton={int((subset_sizes==1).sum())}")

        # --- §2: representative-only topology accounting ---
        degree = torch.zeros((count,), dtype=torch.int32, device=device)
        if int(positive_edges.shape[0]) > 0:
            ones = torch.ones((int(positive_edges.shape[0]),), dtype=torch.int32, device=device)
            degree.index_add_(0, positive_edges[:, 0], ones)
            degree.index_add_(0, positive_edges[:, 1], ones)

        rep_degree = degree[ever_representative]
        rep_count = int(ever_representative.sum())
        rep_degree0 = int((rep_degree == 0).sum())
        rep_degree_ge1 = rep_count - rep_degree0
        rep_singleton_sizes = subset_sizes  # component sizes are shared; representative-only view below filters membership
        rep_component_ids_present = torch.unique(subset_ids[ever_representative])
        rep_component_sizes = subset_sizes[rep_component_ids_present]  # every component containing >=1 representative
        representative_only_stats = {
            "representative_surfel_count": rep_count,
            "representative_degree0_count": rep_degree0,
            "representative_degree_ge1_count": rep_degree_ge1,
            "representative_singleton_fraction": rep_degree0 / rep_count if rep_count else 0.0,
            "representative_connected_fraction": rep_degree_ge1 / rep_count if rep_count else 0.0,
            "representative_only_component_count": int(rep_component_ids_present.shape[0]),
            "component_size_stats_among_representative_containing_components": _component_size_stats(rep_component_sizes),
            "largest_component_fraction_of_representative_population": (float(rep_component_sizes.max()) / rep_count) if rep_count and int(rep_component_sizes.shape[0]) > 0 else 0.0,
        }
        _progress(f"[representative-only] {representative_only_stats}")

        # --- §3: non-representative accounting A/B/C/D ---
        category_counts = {
            "A_contributing_and_representative": cross_tab["contributing_and_representative"],
            "B_contributing_and_never_representative": cross_tab["contributing_and_not_representative"],
            "C_renderer_noncontributing": int((~ever_contributed & ~ever_representative).sum()),
            "D_noncontributing_but_representative_unexpected": cross_tab["not_contributing_and_representative"],
        }
        _progress(f"[A/B/C/D accounting] {category_counts}")

        # --- §4: remaining representative singleton causes ---
        node_in_raw = torch.zeros((count,), dtype=torch.bool, device=device)
        if int(raw_pairs.shape[0]) > 0:
            node_in_raw[raw_pairs[:, 0]] = True
            node_in_raw[raw_pairs[:, 1]] = True
        node_in_local = torch.zeros((count,), dtype=torch.bool, device=device)
        if int(local_pairs.shape[0]) > 0:
            node_in_local[local_pairs[:, 0]] = True
            node_in_local[local_pairs[:, 1]] = True
        node_residual_rejected = _scatter_or(count, local_pairs, geometry["fails_residual"], device)
        node_positional_rejected = _scatter_or(count, local_pairs, geometry["fails_positional"], device)

        singleton_mask = degree == 0
        rep_singleton_mask = singleton_mask & ever_representative

        cause = torch.full((count,), SINGLETON_CAUSES.index(SINGLETON_CAUSE_OTHER), dtype=torch.int64, device=device)
        cause = torch.where(~node_in_raw, torch.full_like(cause, SINGLETON_CAUSES.index(SINGLETON_CAUSE_NO_NEIGHBOR)), cause)
        cause = torch.where(node_in_raw & ~node_in_local, torch.full_like(cause, SINGLETON_CAUSES.index(SINGLETON_CAUSE_FAILS_LOCALITY)), cause)
        cause = torch.where(node_in_local & node_residual_rejected, torch.full_like(cause, SINGLETON_CAUSES.index(SINGLETON_CAUSE_GEOMETRIC_DISCONTINUITY)), cause)
        cause = torch.where(node_in_local & node_positional_rejected, torch.full_like(cause, SINGLETON_CAUSES.index(SINGLETON_CAUSE_POSITIONAL_SEPARATION)), cause)

        singleton_cause_counts_global = {name: int((cause[rep_singleton_mask] == index).sum()) for index, name in enumerate(SINGLETON_CAUSES)}
        _progress(f"[singleton cause, global] total={int(rep_singleton_mask.sum())} {singleton_cause_counts_global}")

        # --- §5: 3D locality rejection distribution ---
        if int(rejected_pairs.shape[0]) > 0:
            rl, rr = rejected_pairs[:, 0], rejected_pairs[:, 1]
            reject_distance = (positions[rr] - positions[rl]).norm(dim=-1)
            reject_spacing = (graph.local_spacing[rl] + graph.local_spacing[rr]) * 0.5
            reject_ratio = reject_distance / reject_spacing.clamp_min(1e-9)
        else:
            reject_distance = torch.zeros((0,), device=device)
            reject_ratio = torch.zeros((0,), device=device)
        locality_rejection_stats = {
            "rejected_pair_count": int(rejected_pairs.shape[0]),
            "distance_distribution": _distribution(reject_distance),
            "distance_over_local_spacing_distribution": _distribution(reject_ratio),
        }
        _progress(f"[locality rejection] {locality_rejection_stats}")

        # --- §6: largest component identity via deterministic pixel anchors ---
        preview_images = arguments.preview_camera_images or arguments.images
        preview_cameras, _preview_meta = load_all_train_cameras(arguments.source_path, preview_images, arguments.sparse_dir, arguments.resolution, arguments.llffhold, arguments.device)
        preview_camera = min(preview_cameras, key=lambda c: c.image_name)
        preview_diag = render_with_pixel_representative(preview_camera, model)
        preview_rep = preview_diag["representative_id"].to(torch.int64)
        preview_h, preview_w = preview_rep.shape

        anchor_results: dict[str, Any] = {}
        for label, fractions in ANCHOR_FRACTIONS.items():
            entries = []
            for fx, fy in fractions:
                px = min(preview_w - 1, int(fx * preview_w))
                py = min(preview_h - 1, int(fy * preview_h))
                full_id = int(preview_rep[py, px].item())
                if full_id < 0:
                    entries.append({"pixel": [px, py], "full_id": full_id, "visible_id": None})
                    continue
                visible_id = int(full_to_visible[full_id].item())
                entries.append({
                    "pixel": [px, py], "full_id": full_id, "visible_id": visible_id,
                    "subset_id": int(subset_ids[visible_id].item()) if visible_id >= 0 else None,
                    "degree": int(degree[visible_id].item()) if visible_id >= 0 else None,
                })
            anchor_results[label] = entries
        _progress(f"[anchors] {json.dumps(anchor_results, indent=2)}")

        largest_subset_id = int(order[0].item())
        largest_member_mask = subset_ids == largest_subset_id
        largest_positions = positions[largest_member_mask]
        largest_bbox = {
            "min": largest_positions.min(dim=0).values.tolist(),
            "max": largest_positions.max(dim=0).values.tolist(),
        } if int(largest_positions.shape[0]) > 0 else {}
        largest_component_summary = {
            "subset_id": largest_subset_id,
            "member_count": int(largest_member_mask.sum()),
            "bounding_box": largest_bbox,
            "anchor_subset_membership": {
                label: [entry.get("subset_id") == largest_subset_id for entry in entries]
                for label, entries in anchor_results.items()
            },
        }
        _progress(f"[largest component] {largest_component_summary}")

        # --- §7: bridge / connectivity robustness of the largest components ---
        bridge_reports = []
        for rank in range(min(3, int(order.shape[0]))):
            component_subset_id = rank
            member_mask = subset_ids == component_subset_id
            member_count = int(member_mask.sum())
            edge_in_component = member_mask[positive_edges[:, 0]] & member_mask[positive_edges[:, 1]]
            component_edges_global = positive_edges[edge_in_component]
            component_view_support = positive_view_support[edge_in_component]
            support_distribution = _distribution(component_view_support.to(torch.float32))
            single_view_fraction = float((component_view_support == 1).float().mean()) if int(component_view_support.shape[0]) > 0 else 0.0

            report = {
                "component_rank": rank, "member_count": member_count,
                "edge_count": int(component_edges_global.shape[0]),
                "view_support_distribution": support_distribution,
                "single_view_supported_edge_fraction": single_view_fraction,
                "bridge_search_performed": member_count <= arguments.max_bridge_nodes,
            }
            if member_count <= arguments.max_bridge_nodes and int(component_edges_global.shape[0]) > 0:
                local_ids = torch.unique(component_edges_global.reshape(-1))
                remap = torch.full((count,), -1, dtype=torch.int64, device=device)
                remap[local_ids] = torch.arange(int(local_ids.shape[0]), dtype=torch.int64, device=device)
                local_edges = remap[component_edges_global]
                _progress(f"[bridge] component rank {rank}: searching bridges over {int(local_ids.shape[0])} nodes / {int(local_edges.shape[0])} edges")
                bridges = _find_bridges(int(local_ids.shape[0]), local_edges.cpu())
                bridge_details = []
                for u_local, v_local in bridges[:50]:
                    u_global, v_global = int(local_ids[u_local].item()), int(local_ids[v_local].item())
                    edge_row_mask = ((component_edges_global[:, 0] == min(u_global, v_global)) & (component_edges_global[:, 1] == max(u_global, v_global)))
                    support = int(component_view_support[edge_row_mask].max().item()) if bool(edge_row_mask.any()) else None
                    bridge_details.append({
                        "endpoints_visible_index": [u_global, v_global],
                        "endpoints_position": [positions[u_global].tolist(), positions[v_global].tolist()],
                        "supporting_view_count": support,
                    })
                report["bridge_count"] = len(bridges)
                report["bridge_details_sample"] = bridge_details
            else:
                report["bridge_count"] = None
            bridge_reports.append(report)
            _progress(f"[bridge] rank {rank}: {report.get('bridge_count')} bridges, single-view-edge fraction={single_view_fraction:.4f}")

        # --- §8: hedge representative-backbone reassessment ---
        # crude hedge region mask: everything NOT within the bounding box of
        # the largest (patio-anchored) component's own extent, restricted to
        # points behind/around the anchor hedge pixels' own 3D neighborhood --
        # kept intentionally simple (diagnostic-only, not a new architecture
        # rule): points whose nearest anchor is a hedge anchor by 3D distance.
        anchor_visible_ids = {label: [e["visible_id"] for e in entries if e.get("visible_id") is not None] for label, entries in anchor_results.items()}
        hedge_region_mask = torch.zeros((count,), dtype=torch.bool, device=device)
        if anchor_visible_ids.get("hedge"):
            hedge_anchor_positions = positions[torch.tensor(anchor_visible_ids["hedge"], device=device)]
            table_patio_anchor_ids = (anchor_visible_ids.get("table", []) + anchor_visible_ids.get("patio", []))
            other_anchor_positions = positions[torch.tensor(table_patio_anchor_ids, device=device)] if table_patio_anchor_ids else None
            dist_to_hedge = torch.cdist(positions, hedge_anchor_positions).min(dim=1).values
            if other_anchor_positions is not None and int(other_anchor_positions.shape[0]) > 0:
                dist_to_other = torch.cdist(positions, other_anchor_positions).min(dim=1).values
                hedge_region_mask = dist_to_hedge < dist_to_other
            else:
                hedge_region_mask = torch.ones((count,), dtype=torch.bool, device=device)

        hedge_total = int(hedge_region_mask.sum())
        hedge_contributing = int((hedge_region_mask & ever_contributed).sum())
        hedge_representative = int((hedge_region_mask & ever_representative).sum())
        hedge_rep_singleton = int((hedge_region_mask & rep_singleton_mask).sum())
        hedge_rep_connected = hedge_representative - hedge_rep_singleton
        singleton_cause_counts_hedge = {name: int((cause[hedge_region_mask & rep_singleton_mask] == index).sum()) for index, name in enumerate(SINGLETON_CAUSES)}
        hedge_summary = {
            "hedge_region_surfel_count": hedge_total,
            "hedge_contributing_count": hedge_contributing,
            "hedge_representative_count": hedge_representative,
            "hedge_representative_singleton_count": hedge_rep_singleton,
            "hedge_representative_connected_count": hedge_rep_connected,
            "hedge_representative_connected_fraction": (hedge_rep_connected / hedge_representative) if hedge_representative else 0.0,
            "hedge_singleton_cause_counts": singleton_cause_counts_hedge,
            # exact overlap with the largest (patio-anchored) component --
            # added after the first run's visual review showed part of the
            # hedge/foliage region sharing the largest component's own color.
            "hedge_region_surfels_in_largest_component_count": int((hedge_region_mask & largest_member_mask).sum()),
            "hedge_region_surfels_in_largest_component_fraction_of_hedge": (
                int((hedge_region_mask & largest_member_mask).sum()) / hedge_total if hedge_total else 0.0
            ),
            "largest_component_surfels_that_are_in_hedge_region_fraction": (
                int((hedge_region_mask & largest_member_mask).sum()) / int(largest_member_mask.sum()) if int(largest_member_mask.sum()) else 0.0
            ),
        }
        _progress(f"[hedge] {hedge_summary}")

        # --- table/patio singleton-cause breakdown (§4 region split) ---
        region_masks = {"table": torch.zeros((count,), dtype=torch.bool, device=device), "patio": torch.zeros((count,), dtype=torch.bool, device=device)}
        for label in ("table", "patio"):
            ids = anchor_visible_ids.get(label, [])
            if ids:
                anchor_positions = positions[torch.tensor(ids, device=device)]
                dist_to_this = torch.cdist(positions, anchor_positions).min(dim=1).values
                all_other_labels = [l for l in ANCHOR_FRACTIONS if l != label]
                other_ids = sum((anchor_visible_ids.get(l, []) for l in all_other_labels), [])
                if other_ids:
                    dist_to_other = torch.cdist(positions, positions[torch.tensor(other_ids, device=device)]).min(dim=1).values
                    region_masks[label] = dist_to_this < dist_to_other
        singleton_cause_by_region = {
            label: {name: int((cause[mask & rep_singleton_mask] == index).sum()) for index, name in enumerate(SINGLETON_CAUSES)}
            for label, mask in region_masks.items()
        }
        _progress(f"[singleton cause by region] {singleton_cause_by_region}")

        report = {
            "batch": "arch/2dgs-coverage-first-surface, Worklog 108",
            "checkpoint": str(arguments.checkpoint),
            "primitive": primitive,
            "iteration": int(payload.get("iteration", 0)),
            "primitive_accounting": {"total_model_surfel_count": total_model_count, "visible_domain_surfel_count": visible_count},
            "camera_meta": camera_meta,
            "cross_tab_contributing_vs_representative": cross_tab,
            "representative_is_strict_subset_of_contributing": representative_is_strict_subset_of_contributing,
            "wl107_replay_cross_check": {
                "visible_component_count": int(order.shape[0]),
                "largest_component_surfel_fraction": float(subset_sizes[0]) / count,
                "singleton_surfel_count": int((subset_sizes == 1).sum()),
            },
            "representative_only_stats": representative_only_stats,
            "category_accounting_ABCD": category_counts,
            "singleton_cause_counts_global": singleton_cause_counts_global,
            "singleton_cause_by_region": singleton_cause_by_region,
            "locality_rejection_stats": locality_rejection_stats,
            "deterministic_anchor_results": anchor_results,
            "largest_component_summary": largest_component_summary,
            "large_component_bridge_reports": bridge_reports,
            "hedge_summary": hedge_summary,
            "runtime_seconds": {"total": time.time() - started},
        }

        # --- colors / exports ---
        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        all_contributors_colors = _ramp(ever_contributed.to(torch.float32), _UNCUT_RGB, (1.0, 0.55, 0.0))
        representatives_colors = _ramp(ever_representative.to(torch.float32), _UNCUT_RGB, (0.15, 0.75, 0.95))
        representative_component_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        representative_component_colors[ever_representative] = _subset_partition_colors(subset_ids[ever_representative])
        singleton_cause_colors = _category_colors(cause, _SINGLETON_CAUSE_RGB, SINGLETON_CAUSES, _UNCUT_RGB)
        singleton_cause_colors[~rep_singleton_mask] = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device)

        largest_membership_colors = _ramp(largest_member_mask.to(torch.float32), _UNCUT_RGB, (0.9, 0.75, 0.1))

        bridge_edge_mask_global = torch.zeros((int(positive_edges.shape[0]),), dtype=torch.bool, device=device)
        for bridge_report_entry in bridge_reports:
            if not bridge_report_entry.get("bridge_details_sample"):
                continue
            for detail in bridge_report_entry["bridge_details_sample"]:
                u, v = detail["endpoints_visible_index"]
                match = (positive_edges[:, 0] == u) & (positive_edges[:, 1] == v)
                bridge_edge_mask_global = bridge_edge_mask_global | match
        bridge_colors = _edge_highlight_colors(count, positive_edges[bridge_edge_mask_global], _UNCUT_RGB, (1.0, 0.1, 0.1), device)

        locality_rejection_colors = _edge_highlight_colors(count, rejected_pairs, _UNCUT_RGB, (0.7, 0.6, 0.9), device)

        hedge_backbone_colors = torch.tensor(_UNCUT_RGB, dtype=torch.float32, device=device).reshape(1, 3).repeat(count, 1)
        hedge_rep_only_mask = hedge_region_mask & ever_representative
        hedge_backbone_colors[hedge_rep_only_mask] = _subset_partition_colors(subset_ids[hedge_rep_only_mask])
        hedge_support_colors = _ramp((hedge_region_mask & ever_contributed & ~ever_representative).to(torch.float32), _UNCUT_RGB, (1.0, 0.55, 0.0))

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_ALL_CONTRIBUTORS: _rgb_to_f_dc(all_contributors_colors),
            VIEW_REPRESENTATIVES: _rgb_to_f_dc(representatives_colors),
            VIEW_REPRESENTATIVE_COMPONENTS: _rgb_to_f_dc(representative_component_colors),
            VIEW_SINGLETON_CAUSE: _rgb_to_f_dc(singleton_cause_colors),
            VIEW_LARGEST_MEMBERSHIP: _rgb_to_f_dc(largest_membership_colors),
            VIEW_BRIDGE: _rgb_to_f_dc(bridge_colors),
            VIEW_LOCALITY_REJECTION: _rgb_to_f_dc(locality_rejection_colors),
            VIEW_HEDGE_BACKBONE: _rgb_to_f_dc(hedge_backbone_colors),
            VIEW_HEDGE_SUPPORT: _rgb_to_f_dc(hedge_support_colors),
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
    report_path = output_root / "camera_induced_representative_backbone_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")


if __name__ == "__main__":
    main()
