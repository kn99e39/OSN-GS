"""arch/2dgs-coverage-first-surface -- 2DGS Coverage-first Surfel Subset partition + review export.

Runs the NEW top-level construction path on a trained 2DGS surfel checkpoint:

    full trained 2DGS surfel scene
        -> intrinsic surfel orientation (t_u, t_v, t_w -- read, not decomposed)
        -> coverage-preserving, normal-coherent, spatially-connected
           Surfel Subset partition
        -> full-scene review export

The isolation-of-effect requirement for the first replay (architecture
directive section 6): the SAME `CoverageFirstPartitionConfig` primary
parameters (`neighbor_count`, `spatial_connect_spacing_multiplier`,
`normal_compatibility_min_alignment`) that Worklog 105/106 used on the
volumetric 3DGS scene are reused UNCHANGED here, so any difference between the
two partitions is attributable to the orientation source, not to a
re-tuned threshold. Trust estimation, latent surface, and NURBS are NOT
implemented here -- see `osn_gs/surface/torch_surfel_surface_orientation.py`
and `torch_coverage_first_subset_partition.py` for the partition/orientation
contracts this reuses unmodified.

Fails closed (never silently substitutes a volumetric checkpoint) if the
loaded checkpoint is not `primitive=="surfel_2d"` / `scale_dim==2`.
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

from osn_gs.gaussian.torch_primitive_evidence_adapter import (
    PRIMITIVE_SURFEL_2D,
    checkpoint_primitive,
    load_primitive_model,
)
from osn_gs.surface.torch_coverage_first_subset_partition import (
    OWNERSHIP_KINDS,
    CoverageFirstPartitionConfig,
    partition_accounting,
    partition_gaussian_subsets,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

_SH_C0 = 0.28209479177387814
_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "2DGS_ORIGINAL_SCENE"
VIEW_INTRINSIC_NORMAL = "2DGS_INTRINSIC_NORMAL_VIEW"
VIEW_SUBSET_PARTITION = "2DGS_COVERAGE_FIRST_SUBSET_PARTITION"
VIEW_NORMAL_CUT = "2DGS_NORMAL_CUT_VIEW"

# Same review-color scheme as Worklog 105/106's 3DGS export
# (`coverage_first_subset_partition_export.py`), reused verbatim so the two
# partitions' renders are visually comparable rather than incidentally
# different because of a second, independently chosen palette.
_FULLY_CUT_RGB = (1.0, 0.15, 0.05)
_UNCUT_RGB = (0.12, 0.13, 0.16)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532


def _progress(message: str) -> None:
    print(f"[2dgs coverage-first partition] {message}", flush=True)


def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    sector = torch.floor(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector = (sector.to(torch.int64) % 6).reshape(-1, 1)
    options = torch.stack(
        [
            torch.stack([value, t, p], dim=-1),
            torch.stack([q, value, p], dim=-1),
            torch.stack([p, value, t], dim=-1),
            torch.stack([p, q, value], dim=-1),
            torch.stack([t, p, value], dim=-1),
            torch.stack([value, p, q], dim=-1),
        ],
        dim=1,
    )
    return options.gather(1, sector.unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)


def subset_partition_colors(subset_ids: torch.Tensor) -> torch.Tensor:
    identifiers = subset_ids.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def normal_orientation_colors(surface_normal: torch.Tensor) -> torch.Tensor:
    """Unsigned `|n|` encoding -- `t_w` and `-t_w` render identically, matching
    the partition's own sign contract. No global flipping."""

    return surface_normal.abs().clamp(0.0, 1.0).to(torch.float32)


def _rgb_to_f_dc(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / _SH_C0


_PLY_HEADER_PROPERTIES = (
    "property float x\nproperty float y\nproperty float z\n"
    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\nproperty float scale_1\n"
    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
)


def write_surfel_ply(
    path: Path, xyz: torch.Tensor, f_dc: torch.Tensor, opacity_logit: torch.Tensor,
    log_scaling: torch.Tensor, rotation: torch.Tensor,
) -> int:
    """Renderer PLY with exactly TWO scale properties (`scale_0`/`scale_1`) --
    the WebRenderer's Gaussian PLY parser accepts a vertex element missing
    `scale_2`; a 2-column surfel checkpoint has no third scale to write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(xyz.shape[0])
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {count}\n" + _PLY_HEADER_PROPERTIES + "end_header\n"
    columns = np.concatenate(
        [
            xyz.detach().cpu().numpy().astype(np.float32),
            f_dc.detach().cpu().numpy().astype(np.float32),
            opacity_logit.detach().cpu().numpy().astype(np.float32).reshape(-1, 1),
            log_scaling.detach().cpu().numpy().astype(np.float32),
            rotation.detach().cpu().numpy().astype(np.float32),
        ],
        axis=1,
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(columns, dtype="<f4").tobytes())
    return count


def write_cut_edge_curves(path: Path, segments: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "nurbs_surface", "iteration": 1,
        "base_curves": segments.detach().cpu().numpy().astype(float).reshape(-1, 2, 3).tolist(),
        "occlusion_curves": [], "patches": [], "metadata": metadata,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_ppm(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image_u8 = (image * 255.0).to(torch.uint8)
    height, width = int(image_u8.shape[0]), int(image_u8.shape[1])
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image_u8.numpy().tobytes())


def build_preview_camera(
    source_path: Path, image_dir_name: str, sparse_dir_name: str, resolution: int, llffhold: int, device: str
) -> tuple[Any, dict[str, Any]]:
    """Identical to Worklog 105/106's camera selection
    (`coverage_first_subset_partition_export.py::build_preview_camera`):
    the name-sorted first TRAIN camera of the eval split, matching
    `TorchOSNGSTrainer._preview_camera` -- reused so a matched
    OPTIONAL_COMPARISON render against the WL105 3DGS output is possible from
    the same viewpoint without altering either historical output."""

    from PIL import Image as PILImage

    from osn_gs.data.colmap_scene import (
        camera_fovs, camera_matrices, read_colmap_cameras, read_colmap_images, resolve_image_path,
    )
    from osn_gs.data.vendor.graphdeco_scene_split import (
        resolve_graphdeco_resolution, select_llff_holdout_test_names,
    )
    from osn_gs.render.torch_fallback import TorchCamera

    image_root = source_path / image_dir_name
    sparse_root = source_path / sparse_dir_name
    cameras = read_colmap_cameras(sparse_root)
    images = read_colmap_images(sparse_root)
    ordered = sorted(images.values(), key=lambda image: image.name)
    test_names = set(
        select_llff_holdout_test_names([image.name for image in ordered], scene_path=source_path, eval=True, llffhold=llffhold)
    )
    train = [image for image in ordered if image.name not in test_names]
    if not train:
        raise ValueError(f"No training cameras remain under {image_root} (llffhold={llffhold}).")
    selected = min(train, key=lambda image: image.name)

    colmap_camera = cameras[selected.camera_id]
    with PILImage.open(resolve_image_path(image_root, selected.name)) as probe:
        original_width, original_height = probe.size
    target_width, target_height, downscale = resolve_graphdeco_resolution(
        original_width, original_height, resolution=resolution, resolution_scale=1.0
    )
    fovx, fovy = camera_fovs(colmap_camera, width=colmap_camera.width, height=colmap_camera.height)
    world_view, full_projection, center = camera_matrices(selected.qvec, selected.tvec, fovx, fovy, device=device)
    camera = TorchCamera(
        image_height=target_height, image_width=target_width,
        world_view_transform=world_view, full_proj_transform=full_projection, camera_center=center,
        FoVx=fovx, FoVy=fovy, image_name=selected.name,
    )
    return camera, {
        "image_name": selected.name, "train_camera_count": len(train),
        "held_out_camera_count": len(ordered) - len(train), "llffhold": llffhold,
        "resolution": [target_width, target_height], "downscale_factor": float(downscale),
        "selection_rule": "name-sorted first train camera (TorchOSNGSTrainer._preview_camera), matches WL105/106",
    }


def _percentile(values: torch.Tensor, fraction: float) -> float:
    if int(values.shape[0]) == 0:
        return 0.0
    ordered = torch.sort(values.to(torch.float64)).values
    position = min(int(ordered.shape[0]) - 1, max(0, int(round(fraction * (int(ordered.shape[0]) - 1)))))
    return float(ordered[position].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="A 2DGS surfel checkpoint (primitive=surfel_2d).")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, default=None, help="COLMAP root; enables render.ppm output.")
    parser.add_argument("--images", default="images_8", help="Matches the branch's own training resolution folder.")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    # Section 6: reuse the SAME primary parameters as Worklog 105/106 by
    # default -- do not silently re-tune them for the surfel scene.
    parser.add_argument("--neighbor-count", type=int, default=CoverageFirstPartitionConfig.neighbor_count)
    parser.add_argument(
        "--spacing-multiplier", type=float, default=CoverageFirstPartitionConfig.spatial_connect_spacing_multiplier
    )
    parser.add_argument(
        "--normal-alignment", type=float, default=CoverageFirstPartitionConfig.normal_compatibility_min_alignment
    )
    parser.add_argument("--knn-chunk", type=int, default=0)
    parser.add_argument("--cut-edge-curve-cap", type=int, default=50_000)
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
            f"scale_dim={getattr(model, 'scale_dim', None)!r}). Refusing to silently substitute a "
            "volumetric checkpoint into the 2DGS Coverage-first partition -- pass a surfel_2d checkpoint."
        )
    total_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    visible_count = int(visible_selector.shape[0])
    _progress(f"model surfels={total_count} visible={visible_count} uncertain={total_count - visible_count} "
              f"primitive={primitive} scale_dim={model.scale_dim} iteration={payload.get('iteration')}")

    with torch.no_grad():
        # Deriving orientation from the FULL model (not a pre-sliced view) so
        # `derive_surface_orientation_from_surfel` reads the model's own
        # get_tangent_u/v/get_normal exactly as trained, then restricts to the
        # visible selector for the partition -- matches WL105/106's own
        # visible-selector pattern.
        full_orientation = derive_surface_orientation_from_surfel(model)
        from dataclasses import replace as _dc_replace

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

        config = CoverageFirstPartitionConfig(
            neighbor_count=int(arguments.neighbor_count),
            spatial_connect_spacing_multiplier=float(arguments.spacing_multiplier),
            normal_compatibility_min_alignment=float(arguments.normal_alignment),
            knn_chunk_size=int(arguments.knn_chunk),
        )
        _progress(f"partitioning with {config.payload()}")
        partition_started = time.time()
        partition = partition_gaussian_subsets(orientation, config, progress=_progress)
        partition_seconds = time.time() - partition_started
        _progress(f"partition done in {partition_seconds:.1f}s -> {partition.subset_count} subsets")

        accounting = partition_accounting(partition)
        cut_edges = partition.normal_cut_edges
        spatial_edges = partition.candidate_edges[partition.spatial_edge_mask]
        spatial_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        cut_degree = torch.zeros((visible_count,), dtype=torch.float32, device=positions.device)
        for target, source in ((spatial_degree, spatial_edges), (cut_degree, cut_edges)):
            if int(source.shape[0]) > 0:
                ones = torch.ones((int(source.shape[0]),), dtype=torch.float32, device=positions.device)
                target.index_add_(0, source[:, 0], ones)
                target.index_add_(0, source[:, 1], ones)
        cut_ratio = torch.where(spatial_degree > 0, cut_degree / spatial_degree.clamp_min(1.0), torch.zeros_like(cut_degree))

        subset_colors = subset_partition_colors(partition.subset_ids)
        normal_colors = normal_orientation_colors(orientation.surface_normal)
        uncut_rgb = torch.tensor(_UNCUT_RGB, device=positions.device).reshape(1, 3)
        cut_rgb = torch.tensor(_FULLY_CUT_RGB, device=positions.device).reshape(1, 3)
        boundary_colors = uncut_rgb + cut_ratio.reshape(-1, 1) * (cut_rgb - uncut_rgb)
        cut_ratio_stats = {
            "mean": float(cut_ratio.mean().item()) if visible_count else 0.0,
            "median": _percentile(cut_ratio, 0.5), "p95": _percentile(cut_ratio, 0.95),
            "fully_cut_gaussian_count": int((cut_ratio >= 1.0).sum()),
            "uncut_gaussian_count": int((cut_ratio <= 0.0).sum()),
        }
        # Local unsigned normal agreement over the SPATIAL adjacency neighborhood
        # (section 10.D) -- independent of whether the edge was accepted.
        if int(spatial_edges.shape[0]) > 0:
            from osn_gs.surface.torch_gaussian_surface_orientation import unsigned_normal_alignment

            spatial_alignment = unsigned_normal_alignment(
                orientation.surface_normal[spatial_edges[:, 0]], orientation.surface_normal[spatial_edges[:, 1]]
            )
            normal_coherence_stats = {
                "mean": float(spatial_alignment.mean().item()), "median": _percentile(spatial_alignment, 0.5),
                "p05": _percentile(spatial_alignment, 0.05), "p95": _percentile(spatial_alignment, 0.95),
            }
        else:
            normal_coherence_stats = {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}

        cut_edge_total = int(cut_edges.shape[0])
        if cut_edge_total > arguments.cut_edge_curve_cap > 0:
            stride = (cut_edge_total + arguments.cut_edge_curve_cap - 1) // arguments.cut_edge_curve_cap
            curve_edges = cut_edges[::stride]
        else:
            curve_edges = cut_edges
        cut_segments = torch.stack([positions[curve_edges[:, 0]], positions[curve_edges[:, 1]]], dim=1)

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = model.get_rotation.detach()[visible_selector]
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_INTRINSIC_NORMAL: _rgb_to_f_dc(normal_colors),
            VIEW_SUBSET_PARTITION: _rgb_to_f_dc(subset_colors),
            VIEW_NORMAL_CUT: _rgb_to_f_dc(boundary_colors),
        }
        view_paths: dict[str, dict[str, str]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} surfels)")

        cut_json_path = output_root / VIEW_NORMAL_CUT / _ITERATION_DIR / "nurbs_surface.json"
        write_cut_edge_curves(
            cut_json_path, cut_segments,
            {
                "representation": "normal_incompatibility_cut_edges", "cut_edge_total": cut_edge_total,
                "cut_edge_rendered": int(curve_edges.shape[0]),
                "selection": "uniform stride over the canonical sorted cut-edge list, not a spatial crop",
            },
        )
        view_paths[VIEW_NORMAL_CUT]["cut_edge_curves_json"] = str(cut_json_path)

    del full_orientation, orientation, partition
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
            render_report.update(
                {"camera": camera_metadata, "backend": rasterizer.backend_source,
                 "original_scene_sh_degree": trained_sh_degree, "review_view_sh_degree": 0}
            )
        except Exception as error:
            render_report.update({"failed": True, "reason": f"{type(error).__name__}: {error}"})
            _progress(f"render.ppm generation FAILED: {type(error).__name__}: {error}")
    else:
        render_report["reason"] = "--source-path not provided; camera intrinsics/extrinsics unavailable"

    report = {
        "batch": "arch/2dgs-coverage-first-surface",
        "checkpoint": str(arguments.checkpoint),
        "primitive": primitive,
        "scale_dim": int(model.scale_dim),
        "iteration": int(payload.get("iteration", 0)),
        "device": arguments.device,
        "input_domain": {
            "model_surfel_count": total_count, "visible_surfel_count": visible_count,
            "uncertain_surfel_count": total_count - visible_count,
            "partition_input_surfel_count": visible_count,
            "restricted_to_prior_regions": False, "required_latent_support": False,
        },
        "orientation": {
            "definition": "intrinsic t_u/t_v/t_w read directly off the trained surfel rotation quaternion; "
                          "NO eigen-decomposition, NO axis reordering, NO axis-separability diagnostic",
            "source": "torch_surfel_surface_orientation.derive_surface_orientation_from_surfel",
        },
        "partition": accounting,
        "local_normal_coherence_over_spatial_neighborhood": normal_coherence_stats,
        "ownership_kinds": list(OWNERSHIP_KINDS),
        "cut_edges": {
            "total": cut_edge_total, "rendered_as_curves": int(curve_edges.shape[0]),
            "curve_cap": int(arguments.cut_edge_curve_cap), "per_surfel_cut_ratio": cut_ratio_stats,
            "boundary_view_encoding": "linear ramp from _UNCUT_RGB at cut_ratio 0 to _FULLY_CUT_RGB at cut_ratio 1",
        },
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"partition": partition_seconds, "total": time.time() - started},
    }
    report_path = output_root / "surfel_partition_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({"subset_count": accounting["subset_count"], "coverage_identity_holds": accounting["coverage_identity_holds"]}))


if __name__ == "__main__":
    main()
