"""Worklog 118 supplementary export -- canonical subset/component-membership
visualization, added per standing feedback (2026-08-26): always include a
subset/component visualization the same way ORIGINAL_2DGS_SCENE is always
included. Applied here as a lightweight addendum (topology-only replay, no
chart fitting) since the main WL118 run was already in progress when this
feedback arrived. From Worklog 119 onward this is a fixed view baked
directly into each batch's own main script.
"""

from __future__ import annotations

import sys
from dataclasses import replace as _dc_replace
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import (
    load_primitive_model, checkpoint_primitive, PRIMITIVE_SURFEL_2D,
    _hsv_to_rgb, _rgb_to_f_dc, write_surfel_ply, write_ppm,
)
from maximal_visible_connectivity_export import load_all_train_cameras
from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative
from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    CameraInducedAdjacencyConfig, accumulate_image_space_pairs, apply_secondary_geometric_gate, filter_by_3d_locality,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_GOLDEN, _PLASTIC, _SILVER = 0.6180339887498949, 0.7548776662466927, 0.5698402909980532


def _progress(m): print(f"[wl118-subset-addendum] {m}", flush=True)


def _hash_colors(ids):
    ids = ids.to(torch.float64)
    hue = torch.frac(ids * _GOLDEN)
    sat = 0.55 + 0.35 * torch.frac(ids * _PLASTIC)
    val = 0.60 + 0.40 * torch.frac(ids * _SILVER)
    return _hsv_to_rgb(hue, sat, val).to(torch.float32).clamp(0.0, 1.0)


def main():
    checkpoint = Path("output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt")
    out_root = Path("output/118_osn_gs_nurbs_evidence_contract_closure/CANONICAL_SUBSET_MEMBERSHIP")
    source_path = Path("DATASET")
    device = "cuda"

    model, payload = load_primitive_model(checkpoint, device=device)
    primitive = checkpoint_primitive(payload)
    assert primitive == PRIMITIVE_SURFEL_2D
    total_model_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    full_to_visible = torch.full((total_model_count,), -1, dtype=torch.int64, device=model.device)
    full_to_visible[visible_selector] = torch.arange(int(visible_selector.shape[0]), dtype=torch.int64, device=model.device)

    cameras, camera_meta = load_all_train_cameras(source_path, "images_8", "sparse/0", -1, 8, device)
    _progress(f"cameras: {camera_meta}")

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
        count = int(positions.shape[0])
        local_config = CoverageFirstPartitionConfig()
        config = CameraInducedAdjacencyConfig(local=local_config)
        _progress("build_candidate_graph")
        graph = build_candidate_graph(orientation, config.local, progress=_progress)

    per_view_rep = []
    ever_representative_full = torch.zeros((total_model_count,), dtype=torch.bool, device=model.device)
    for i, camera in enumerate(cameras):
        diag = render_with_pixel_representative(camera, model)
        rep_full = diag["representative_id"].to(torch.int64)
        valid = rep_full >= 0
        ever_representative_full[torch.unique(rep_full[valid])] = True
        rep_remapped = torch.where(valid, full_to_visible[rep_full.clamp(min=0)], torch.full_like(rep_full, -1))
        per_view_rep.append(rep_remapped.detach().cpu())
        if i % 40 == 0:
            _progress(f"sweep {i + 1}/{len(cameras)}")
    ever_representative = ever_representative_full[visible_selector]

    with torch.no_grad():
        per_view_rep_gpu = [t.to(model.device) for t in per_view_rep]
        raw_pairs, _ = accumulate_image_space_pairs(count, per_view_rep_gpu, progress=_progress)
        local_pairs, _ = filter_by_3d_locality(raw_pairs, count, graph)
        geometry = apply_secondary_geometric_gate(local_pairs, orientation, config, progress=_progress)
        positive_edges = local_pairs[geometry["kept_mask"]]
        roots = _connected_component_roots(count, positive_edges, config.local)
        unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
        order = torch.argsort(counts, descending=True, stable=True)
        subset_id_of_position = torch.empty_like(order)
        subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=model.device)
        subset_ids = subset_id_of_position[inverse]
        _progress(f"replay check: components={int(order.shape[0])} largest_fraction={float(counts[order][0]) / count:.4f}")

    colors = torch.tensor((0.08, 0.09, 0.11), dtype=torch.float32, device=model.device).reshape(1, 3).repeat(count, 1)
    colors[ever_representative] = _hash_colors(subset_ids[ever_representative])

    visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
    visible_log_scaling = model._scaling.detach()[visible_selector]
    visible_rotation = model.get_rotation.detach()[visible_selector]
    f_dc = _rgb_to_f_dc(colors)
    ply_path = out_root / "iteration_0000001" / "point_cloud.ply"
    n = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
    _progress(f"wrote {n} surfels to {ply_path}")

    preview_cameras, _ = load_all_train_cameras(source_path, "images_8", "sparse/0", -1, 8, device)
    preview_camera = min(preview_cameras, key=lambda c: c.image_name)
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    full_dc = torch.zeros_like(model._features_dc)
    full_dc[visible_selector, 0, :] = f_dc
    model._features_dc.data.copy_(full_dc)
    model._features_rest.data.zero_()
    model.active_sh_degree = 0
    package = rasterizer.render(preview_camera, model)
    write_ppm(out_root / "render.ppm", package["render"])
    _progress(f"rendered from {preview_camera.image_name}")

    from PIL import Image
    preview_dir = out_root.parent / "preview_png"
    preview_dir.mkdir(exist_ok=True)
    Image.open(out_root / "render.ppm").save(preview_dir / "CANONICAL_SUBSET_MEMBERSHIP.png")
    _progress("done")


if __name__ == "__main__":
    main()
