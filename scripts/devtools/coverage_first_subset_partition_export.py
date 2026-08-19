"""Worklog 105 -- coverage-first Gaussian Subset partition replay and review export.

Runs the NEW top-level construction path end to end on a trained checkpoint:

    full trained visible Gaussian scene
        -> per-Gaussian surface-orientation representation
        -> coverage-preserving, normal-coherent Gaussian Subset partition
        -> full-scene review export

and nothing else. No trust estimation, no latent surface, no NURBS: this batch
validates the partition itself. The Worklog 95-104 pipeline is untouched and
still replayable through its own scripts
(`latent_surface_coverage_export.py`, `latent_surface_visualization_coverage_export.py`).

Four mandatory full-scene review representations, all in the ORIGINAL scene
coordinate frame, none cropped to "successful" areas:

    A. original_scene            -- the trained scene as it is
    B. normal_orientation_view   -- derived surface orientation, unsigned (|n|)
    C. gaussian_subset_partition -- every subset in a deterministic distinct color
    D. subset_boundary_view      -- adjacency edges cut for normal incompatibility

Each view produces a WebRenderer folder (`iteration_<N>/point_cloud.ply`, plus
one `nurbs_surface.json` for view D's cut segments) AND a `render.ppm`
rasterized from the deterministic preview camera -- the same camera
`TorchOSNGSTrainer._preview_camera` picks (name-sorted first TRAIN camera of
the eval split), so every `render.ppm` here is pixel-comparable with the
checkpoint's own `render.ppm`.

Colors and camera are fixed by construction, never chosen from the result.
No acceptance threshold and no qualitative verdict is computed here.
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

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_coverage_first_subset_partition import (
    OWNERSHIP_KINDS,
    CoverageFirstPartitionConfig,
    partition_accounting,
    partition_gaussian_subsets,
)
from osn_gs.surface.torch_gaussian_surface_orientation import (
    SEPARABILITY_CODES,
    derive_surface_orientation_from_scale_rotation,
)

_SH_C0 = 0.28209479177387814
_ITERATION_DIR = "iteration_0000001"

VIEW_ORIGINAL_SCENE = "original_scene"
VIEW_NORMAL_ORIENTATION = "normal_orientation_view"
VIEW_SUBSET_PARTITION = "gaussian_subset_partition"
VIEW_SUBSET_BOUNDARY = "subset_boundary_view"

# Fixed review colors for view D, linearly interpolated by a Gaussian's CUT
# RATIO (share of its own spatial adjacency edges rejected for normal
# incompatibility). A binary "touches at least one cut" flag was tried first
# and saturated -- 46% of all spatial edges are cut on the reference scene, so
# almost every Gaussian lit up and the view carried no information. The ratio
# encoding is strictly more informative and is not a tuned threshold: it is the
# raw per-Gaussian statistic, mapped end to end.
_FULLY_CUT_RGB = (1.0, 0.15, 0.05)
_UNCUT_RGB = (0.12, 0.13, 0.16)

# Irrational multipliers (golden ratio and two other low-discrepancy
# constants) so consecutive subset IDs never land on neighbouring colors.
# Deterministic: subset i always gets the same color in every run.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_PLASTIC_CONJUGATE = 0.7548776662466927
_SILVER_CONJUGATE = 0.5698402909980532


def _progress(message: str) -> None:
    print(f"[coverage-first partition] {message}", flush=True)


# ---------------------------------------------------------------------------
# checkpoint loading (self-contained: no Worklog 95-104 module is imported)
# ---------------------------------------------------------------------------


def load_checkpoint_model(checkpoint_dir: Path, device: str) -> tuple[TorchGaussianModel, torch.Tensor]:
    payload = torch.load(checkpoint_dir / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    rest_count = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_count:
        degree += 1
    model = TorchGaussianModel(sh_degree=degree, device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    model.active_sh_degree = int(payload.get("active_sh_degree", degree)) if isinstance(payload, dict) else degree
    stable_ids = raw.get("stable_gaussian_ids")
    if stable_ids is None:
        stable_ids = torch.arange(int(raw["xyz"].shape[0]), device=device)
    return model, stable_ids.detach().to(device).reshape(-1).to(torch.int64)


# ---------------------------------------------------------------------------
# deterministic review colors
# ---------------------------------------------------------------------------


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
    """Deterministic per-subset RGB. Subset ``i`` always maps to the same color."""

    identifiers = subset_ids.to(torch.float64)
    hue = torch.frac(identifiers * _GOLDEN_RATIO_CONJUGATE)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * _PLASTIC_CONJUGATE)
    value = 0.60 + 0.40 * torch.frac(identifiers * _SILVER_CONJUGATE)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def normal_orientation_colors(surface_normal: torch.Tensor) -> torch.Tensor:
    """``|n|`` per component -- an UNSIGNED encoding, so ``n`` and ``-n`` render
    identically, matching the partition's own sign contract. No global flipping
    is applied to make the picture look coherent."""

    return surface_normal.abs().clamp(0.0, 1.0).to(torch.float32)


def _rgb_to_f_dc(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / _SH_C0


# ---------------------------------------------------------------------------
# WebRenderer outputs
# ---------------------------------------------------------------------------


_PLY_HEADER_PROPERTIES = (
    "property float x\nproperty float y\nproperty float z\n"
    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
)


def write_gaussian_ply(
    path: Path,
    xyz: torch.Tensor,
    f_dc: torch.Tensor,
    opacity_logit: torch.Tensor,
    log_scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> int:
    """`binary_little_endian` renderer PLY (the ASCII variant is ~4x larger at
    scene scale, and the WebRenderer reads both). Scale/opacity stay in the
    Graphdeco log/logit domain the renderer expects; only ``f_dc`` differs
    between views."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(xyz.shape[0])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n" + _PLY_HEADER_PROPERTIES + "end_header\n"
    )
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
    """Cut edges as `nurbs_surface.json` ``base_curves`` polylines (the only
    line primitive the WebRenderer accepts)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "nurbs_surface",
        "iteration": 1,
        "base_curves": segments.detach().cpu().numpy().astype(float).reshape(-1, 2, 3).tolist(),
        "occlusion_curves": [],
        "patches": [],
        "metadata": metadata,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_ppm(path: Path, image: torch.Tensor) -> None:
    """Same P6 writer the trainer uses (`TorchOSNGSTrainer._save_ppm`)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image_u8 = (image * 255.0).to(torch.uint8)
    height, width = int(image_u8.shape[0]), int(image_u8.shape[1])
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image_u8.numpy().tobytes())


# ---------------------------------------------------------------------------
# deterministic preview camera (no image pixels are ever loaded)
# ---------------------------------------------------------------------------


def build_preview_camera(
    source_path: Path, image_dir_name: str, sparse_dir_name: str, resolution: int, llffhold: int, device: str
) -> tuple[Any, dict[str, Any]]:
    """Rebuild EXACTLY the camera `TorchOSNGSTrainer._preview_camera` selects.

    `_preview_camera` takes ``min(scene.cameras, key=image_name)`` over the
    eval split's TRAIN cameras, and `load_colmap_scene_with_eval_split` builds
    those from COLMAP intrinsics/extrinsics plus the Graphdeco resolution rule.
    Only that one camera is rebuilt here and no image is decoded (just a PIL
    size probe), so the render matches the checkpoint's own `render.ppm`
    viewpoint and resolution without loading the whole dataset.
    """

    from PIL import Image as PILImage

    from osn_gs.data.colmap_scene import (
        camera_fovs,
        camera_matrices,
        read_colmap_cameras,
        read_colmap_images,
        resolve_image_path,
    )
    from osn_gs.data.vendor.graphdeco_scene_split import (
        resolve_graphdeco_resolution,
        select_llff_holdout_test_names,
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
        image_height=target_height,
        image_width=target_width,
        world_view_transform=world_view,
        full_proj_transform=full_projection,
        camera_center=center,
        FoVx=fovx,
        FoVy=fovy,
        image_name=selected.name,
    )
    return camera, {
        "image_name": selected.name,
        "train_camera_count": len(train),
        "held_out_camera_count": len(ordered) - len(train),
        "llffhold": llffhold,
        "resolution": [target_width, target_height],
        "downscale_factor": float(downscale),
        "selection_rule": "name-sorted first train camera (TorchOSNGSTrainer._preview_camera)",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _percentile(values: torch.Tensor, fraction: float) -> float:
    if int(values.shape[0]) == 0:
        return 0.0
    ordered = torch.sort(values.to(torch.float64)).values
    position = min(int(ordered.shape[0]) - 1, max(0, int(round(fraction * (int(ordered.shape[0]) - 1)))))
    return float(ordered[position].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, default=None, help="COLMAP root; enables render.ppm output.")
    parser.add_argument("--images", default="images")
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
    parser.add_argument(
        "--cut-edge-curve-cap",
        type=int,
        default=50_000,
        help="Max cut-edge segments written as renderable curves (uniform stride, never a spatial crop).",
    )
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)

    _progress(f"loading checkpoint {arguments.checkpoint}")
    model, stable_ids = load_checkpoint_model(arguments.checkpoint, arguments.device)
    total_gaussian_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    visible_count = int(visible_selector.shape[0])
    _progress(f"model gaussians={total_gaussian_count} visible={visible_count} uncertain={total_gaussian_count - visible_count}")

    with torch.no_grad():
        positions = model.get_xyz.detach()[visible_selector]
        linear_scaling = model.get_scaling.detach()[visible_selector]
        rotation = model.get_rotation.detach()[visible_selector]
        _progress("deriving per-Gaussian surface orientation")
        orientation = derive_surface_orientation_from_scale_rotation(
            positions, linear_scaling, rotation, stable_ids[visible_selector]
        )

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
            "median": _percentile(cut_ratio, 0.5),
            "p95": _percentile(cut_ratio, 0.95),
            "fully_cut_gaussian_count": int((cut_ratio >= 1.0).sum()),
            "uncut_gaussian_count": int((cut_ratio <= 0.0).sum()),
        }

        # Deterministic uniform stride over the sorted cut-edge list -- keeps
        # cuts from EVERY part of the scene rather than cropping to a region.
        cut_edge_total = int(cut_edges.shape[0])
        if cut_edge_total > arguments.cut_edge_curve_cap > 0:
            stride = (cut_edge_total + arguments.cut_edge_curve_cap - 1) // arguments.cut_edge_curve_cap
            curve_edges = cut_edges[::stride]
        else:
            curve_edges = cut_edges
        cut_segments = torch.stack([positions[curve_edges[:, 0]], positions[curve_edges[:, 1]]], dim=1)

        separability_counts = orientation.separability_counts()
        local_spacing = partition.local_spacing
        spacing_stats = {
            "min": float(local_spacing.min().item()) if visible_count else 0.0,
            "median": _percentile(local_spacing, 0.5),
            "mean": float(local_spacing.mean().item()) if visible_count else 0.0,
            "p95": _percentile(local_spacing, 0.95),
            "max": float(local_spacing.max().item()) if visible_count else 0.0,
        }

        visible_opacity = model._opacity.detach().reshape(-1)[visible_selector]
        visible_log_scaling = model._scaling.detach()[visible_selector]
        visible_rotation = rotation
        original_f_dc = model._features_dc.detach()[visible_selector][:, 0, :]

        views = {
            VIEW_ORIGINAL_SCENE: original_f_dc,
            VIEW_NORMAL_ORIENTATION: _rgb_to_f_dc(normal_colors),
            VIEW_SUBSET_PARTITION: _rgb_to_f_dc(subset_colors),
            VIEW_SUBSET_BOUNDARY: _rgb_to_f_dc(boundary_colors),
        }
        view_paths: dict[str, dict[str, str]] = {}
        for name, f_dc in views.items():
            ply_path = output_root / name / _ITERATION_DIR / "point_cloud.ply"
            written = write_gaussian_ply(ply_path, positions, f_dc, visible_opacity, visible_log_scaling, visible_rotation)
            view_paths[name] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
            _progress(f"wrote {name} ({written} gaussians)")

        cut_json_path = output_root / VIEW_SUBSET_BOUNDARY / _ITERATION_DIR / "nurbs_surface.json"
        write_cut_edge_curves(
            cut_json_path,
            cut_segments,
            {
                "representation": "normal_incompatibility_cut_edges",
                "cut_edge_total": cut_edge_total,
                "cut_edge_rendered": int(curve_edges.shape[0]),
                "selection": "uniform stride over the canonical sorted cut-edge list, not a spatial crop",
            },
        )
        view_paths[VIEW_SUBSET_BOUNDARY]["cut_edge_curves_json"] = str(cut_json_path)

    del orientation, partition
    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    # --- render.ppm for every view, from one fixed camera ---
    render_report: dict[str, Any] = {"enabled": arguments.source_path is not None}
    if arguments.source_path is not None:
        try:
            from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer

            camera, camera_metadata = build_preview_camera(
                arguments.source_path, arguments.images, arguments.sparse_dir,
                arguments.resolution, arguments.llffhold, arguments.device,
            )
            rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=arguments.device == "cuda"))
            _progress(f"rendering previews from camera {camera_metadata['image_name']} backend={rasterizer.backend_source}")
            trained_sh_degree = int(model.active_sh_degree)
            with torch.no_grad():
                # ORIGINAL_SCENE renders FIRST and with the checkpoint's own
                # spherical harmonics intact, so it is directly comparable with
                # the checkpoint's own render.ppm. Only afterwards are the SH
                # bands zeroed for the flat per-Gaussian review colors.
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
                    # Positions/scales/rotations/opacities always stay the
                    # checkpoint's own, so geometry and coverage in every image
                    # are exactly the trained scene's.
                    package = rasterizer.render(camera, model)
                    ppm_path = output_root / name / "render.ppm"
                    write_ppm(ppm_path, package["render"])
                    view_paths[name]["render_ppm"] = str(ppm_path)
                    _progress(f"rendered {name}")
                    del package
            render_report.update(
                {
                    "camera": camera_metadata,
                    "backend": rasterizer.backend_source,
                    "original_scene_sh_degree": trained_sh_degree,
                    "review_view_sh_degree": 0,
                }
            )
        except Exception as error:  # reported, never silently swallowed
            render_report.update({"failed": True, "reason": f"{type(error).__name__}: {error}"})
            _progress(f"render.ppm generation FAILED: {type(error).__name__}: {error}")
    else:
        render_report["reason"] = "--source-path not provided; camera intrinsics/extrinsics unavailable"

    report = {
        "worklog": 105,
        "checkpoint": str(arguments.checkpoint),
        "device": arguments.device,
        "input_domain": {
            "model_gaussian_count": total_gaussian_count,
            "visible_gaussian_count": visible_count,
            "uncertain_gaussian_count": total_gaussian_count - visible_count,
            "partition_input_gaussian_count": visible_count,
            "restricted_to_prior_regions": False,
            "required_latent_support": False,
        },
        "orientation": {
            "definition": "principal axes of Sigma = R diag(exp(scaling)^2) R^T, ordered by descending eigenvalue; "
                          "normal = axis(lambda3), tangent_u = axis(lambda1), tangent_v = normal x tangent_u",
            "axis_separability_counts": separability_counts,
            "axis_separability_codes": list(SEPARABILITY_CODES),
        },
        "partition": accounting,
        "local_spacing": spacing_stats,
        "ownership_kinds": list(OWNERSHIP_KINDS),
        "cut_edges": {
            "total": cut_edge_total,
            "rendered_as_curves": int(curve_edges.shape[0]),
            "curve_cap": int(arguments.cut_edge_curve_cap),
            "per_gaussian_cut_ratio": cut_ratio_stats,
            "boundary_view_encoding": "linear ramp from _UNCUT_RGB at cut_ratio 0 to _FULLY_CUT_RGB at cut_ratio 1",
        },
        "views": view_paths,
        "render_ppm": render_report,
        "runtime_seconds": {"partition": partition_seconds, "total": time.time() - started},
    }
    report_path = output_root / "partition_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"report -> {report_path}")
    print(json.dumps({"subset_count": accounting["subset_count"], "coverage_identity_holds": accounting["coverage_identity_holds"]}))


if __name__ == "__main__":
    main()
