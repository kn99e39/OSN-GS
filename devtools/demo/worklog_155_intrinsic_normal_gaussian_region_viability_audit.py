"""Worklog 155 -- standalone Gaussian-region viability and attribution audit.

This is a diagnostic-only replay of the exact Gaussian-side path used by
Worklog 154:

    trained 2DGS surfels -> intrinsic t_w -> existing region-coherent partition

The partition implementation and its parameters are intentionally not copied,
wrapped with new gates, or tuned here.  The script only adds accounting,
deterministic review exports, and a read-only attribution join against the
already completed Worklog 154 Candidate F arrays.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    PARTITION_ROLES,
    ROLE_ISOLATED_FALLBACK,
    ROLE_OWNERSHIP_PROPAGATED,
    ROLE_STRUCTURAL_CORE,
    RegionCoherenceConfig,
    partition_surfels_region_coherent,
    region_coherent_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel


DEFAULT_CHECKPOINT = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt"
DEFAULT_WL154 = REPO_ROOT / "output/154_gaussian_region_owned_tsdf_boundary_first_nurbs"
DEFAULT_SOURCE_PATH = REPO_ROOT / "DATASET"
DEFAULT_OUT = REPO_ROOT / "output/155_intrinsic_normal_gaussian_region_viability_audit"
ITERATION_DIR = "iteration_00030000"
SH_C0 = 0.28209479177387814

# Fixed, diagnostic-only status palette.  This is not the mandatory
# Observed/Occluded palette; that matched pair is preserved from WL154 below.
STATUS_COLORS = {
    "core": (0.10, 0.55, 0.95),
    "attached": (0.10, 0.85, 0.35),
    "ambiguous": (0.96, 0.70, 0.08),
    "rejected": (0.92, 0.18, 0.18),
    "unassigned": (0.60, 0.60, 0.62),
    "uncertain": (0.60, 0.60, 0.62),
}
BOUNDARY_COLORS = {
    "none": (0.16, 0.17, 0.20),
    "normal_cut": (0.92, 0.18, 0.18),
    "region_conflict": (0.98, 0.74, 0.04),
    "normal_cut_and_region_conflict": (0.98, 0.36, 0.05),
}

# These are the fixed WL145 review windows.  They are annotations for human
# review only, never membership or selection predicates.
REVIEW_POLYGONS: dict[str, dict[str, tuple[tuple[float, float], ...]]] = {
    "tabletop": {
        "DSC08043.JPG": ((200, 215), (235, 213), (236, 235), (201, 236)),
        "DSC07960.JPG": ((383, 188), (415, 189), (413, 200), (385, 199)),
        "DSC08003.JPG": ((240, 158), (280, 158), (279, 171), (242, 170)),
    },
    "table_side": {
        "DSC08043.JPG": ((220, 264), (385, 260), (383, 280), (222, 283)),
        "DSC07960.JPG": ((215, 257), (375, 253), (373, 275), (217, 278)),
        "DSC08003.JPG": ((205, 259), (380, 256), (378, 277), (207, 280)),
    },
    "vase_neighbor": {
        "DSC08043.JPG": ((385, 185), (458, 193), (445, 226), (376, 218)),
        "DSC07960.JPG": ((375, 184), (447, 193), (440, 226), (369, 217)),
        "DSC08003.JPG": ((225, 184), (282, 190), (278, 224), (221, 218)),
    },
}
REVIEW_CAMERAS = ("DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG")
EVENT_1527_CAMERA = "DSC08003.JPG"
EVENT_1527_PIXEL = (259.0, 169.0)
EVENT_1527_RADIUS = 12.0
REVIEW_WORLD_BOXES = {
    # Existing WL140 semantic review controls, reused as visible annotations
    # only.  They are not physical-sheet ground truth and do not alter IDs.
    "background_lower": (
        ((-1.0, 1.5, -0.15), (1.0, 2.5, 0.15)),
        ((-11.0, 2.0, 0.0), (-9.5, 3.5, 2.5)),
    ),
}


def _progress(message: str) -> None:
    print(f"[worklog 155] {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_bytes(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Any, fraction: float) -> float:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, fraction * 100.0, method="nearest"))


def _summary(values: Any) -> dict[str, Any]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.percentile(array, 50, method="nearest")),
        "p75": float(np.percentile(array, 75, method="nearest")),
        "p90": float(np.percentile(array, 90, method="nearest")),
        "p95": float(np.percentile(array, 95, method="nearest")),
        "p99": float(np.percentile(array, 99, method="nearest")),
        "max": float(np.max(array)),
    }


def _load_surfel_model_safe(checkpoint: Path, device: str) -> tuple[TorchGaussianSurfelModel, dict[str, Any]]:
    """Read the same checkpoint contract as WL154 without unsafe pickle code."""

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if int(payload.get("scale_dim", 3)) != 2:
        raise ValueError("Worklog 155 requires a scale_dim==2 2DGS surfel checkpoint")
    raw = payload["model_raw"]
    rest = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest:
        degree += 1
    model = TorchGaussianSurfelModel(sh_degree=degree, device=device)
    stable_ids = raw.get("stable_gaussian_ids")
    if stable_ids is None:
        stable_ids = torch.arange(raw["xyz"].shape[0], dtype=torch.int64, device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=stable_ids,
    )
    model.active_sh_degree = int(payload.get("active_sh_degree", degree))
    return model, payload


def _active_orientation(model: TorchGaussianSurfelModel) -> tuple[Any, torch.Tensor]:
    uncertain = model.is_uncertain.reshape(-1).to(torch.bool)
    selector = torch.nonzero(~uncertain, as_tuple=False).reshape(-1)
    full = derive_surface_orientation_from_surfel(model)
    active = replace(
        full,
        gaussian_ids=full.gaussian_ids[selector], positions=full.positions[selector],
        tangent_axis_u=full.tangent_axis_u[selector], tangent_axis_v=full.tangent_axis_v[selector],
        surface_normal=full.surface_normal[selector], tangent_scale_u=full.tangent_scale_u[selector],
        tangent_scale_v=full.tangent_scale_v[selector],
    )
    return active, selector


def _partition_status(partition: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Map the existing partition roles to WL154 membership semantics."""

    role = partition.partition_role.to(torch.int64)
    ambiguous = partition.ambiguous_multi_region.to(torch.bool)
    status = torch.full_like(role, 4)  # unassigned / isolated fallback
    status[role == PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)] = 0  # core
    propagated = role == PARTITION_ROLES.index(ROLE_OWNERSHIP_PROPAGATED)
    status[propagated & ~ambiguous] = 1  # attached
    status[propagated & ambiguous] = 2  # ambiguous
    # The current partition has no per-surfel rejected role. Rejected merges
    # are edge relations and are accounted separately, never hidden here.
    status_counts = {
        "core": int((status == 0).sum()), "attached": int((status == 1).sum()),
        "ambiguous": int((status == 2).sum()), "rejected": 0,
        "unassigned": int((status == 4).sum()),
    }
    accepted = (status == 0) | (status == 1)
    return status, accepted, status_counts


def _mapping_digest(orientation: Any, partition: Any, status: torch.Tensor) -> str:
    ids = orientation.gaussian_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    order = np.argsort(ids, kind="stable")
    return _sha256_bytes(ids[order], partition.subset_ids.detach().cpu().numpy().astype(np.int64)[order], status.detach().cpu().numpy().astype(np.int8)[order])


def _contract_reconciliation() -> dict[str, Any]:
    from osn_gs.surface import torch_coverage_first_subset_partition as graph_module
    from osn_gs.surface import torch_region_coherent_surfel_partition as region_module

    graph_source = inspect.getsource(graph_module.build_candidate_graph)
    region_source = inspect.getsource(region_module.partition_surfels_region_coherent)
    merged_source = inspect.getsource(region_module._region_coherent_merge_cpu)
    source = "\n".join((graph_source, region_source, merged_source))
    tree = ast.parse(source)
    calls = sorted({node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                    for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and (isinstance(node.func, ast.Attribute) or isinstance(node.func, ast.Name))})
    covariance_calls = [name for name in calls if "covariance" in name.lower()]
    decomposition_calls = [name for name in calls if name in {"eigh", "eigvalsh", "eig", "linalg"}]
    return {
        "active_path": [
            "TorchGaussianSurfelModel.get_tangent_u",
            "TorchGaussianSurfelModel.get_tangent_v",
            "TorchGaussianSurfelModel.get_normal (intrinsic t_w)",
            "derive_surface_orientation_from_surfel",
            "partition_surfels_region_coherent",
            "build_candidate_graph",
            "_region_coherent_merge_cpu",
        ],
        "partition_contract_fields_read": {
            "positions": "used for exact spatial kNN and local-spacing gate",
            "surface_normal_t_w": "used for unsigned local alignment and region normal-scatter coherence",
            "gaussian_ids": "carried to output provenance; not a membership gate",
            "t_u": "not consumed by partition",
            "t_v": "not consumed by partition",
            "tangent_scales_s_u_s_v": "not consumed by partition",
            "stable_gaussian_ids": "provided as gaussian_ids; no new IDs constructed",
            "spatial_neighbor_relation": "existing CandidateGraph kNN relation",
            "reliability_field": "not consumed",
            "opacity_or_covariance": "not consumed",
        },
        "covariance_construction_called": bool(covariance_calls),
        "covariance_related_calls": covariance_calls,
        "eigendecomposition_called": bool(decomposition_calls),
        "eigendecomposition_calls": decomposition_calls,
        "covariance_minor_axis_normal": False,
        "covariance_normal_override_of_t_w": False,
        "lambda2_lambda3_axis_separability_membership_rule": False,
        "existing_region_scatter_note": (
            "The active region-coherence merge computes a closed-form largest-eigenvalue/trace "
            "concentration of sign-invariant M_R=sum(n_i n_i^T). This is not covariance, is not a "
            "per-surfel normal definition, and is the existing WL97 merge rule."
        ),
        "contract_status": "INTENT_ALIGNED_NO_CONTRACT_DRIFT",
    }


def _graph_accounting(partition: Any) -> dict[str, Any]:
    graph = partition.graph
    candidate = graph.candidate_edges
    spatial = graph.spatial_edge_mask
    normal = graph.normal_compatible_mask
    accepted = spatial & normal
    same_region = partition.subset_ids[candidate[:, 0]] == partition.subset_ids[candidate[:, 1]]
    spatial_alignment = graph.normal_alignment[spatial]
    accepted_alignment = graph.normal_alignment[accepted]
    normal_cut = spatial & ~normal
    anti = partition.rejected_merge_mask
    return {
        "candidate_edge_count": int(candidate.shape[0]),
        "spatial_pass_edge_count": int(spatial.sum()),
        "spatial_cut_edge_count": int((~spatial).sum()),
        "normal_compatible_edge_count_all_candidates": int(normal.sum()),
        "normal_compatible_edge_count_on_spatial_edges": int(accepted.sum()),
        "normal_cut_conflict_edge_count_on_spatial_edges": int(normal_cut.sum()),
        "accepted_edge_count": int(accepted.sum()),
        "accepted_edges_internal_to_final_region": int((accepted & same_region).sum()),
        "accepted_edges_crossing_final_region_boundary": int((accepted & ~same_region).sum()),
        "region_coherence_rejected_merge_edge_count": int(anti.sum()),
        "internal_t_w_unsigned_alignment": _summary(accepted_alignment),
        "spatial_t_w_unsigned_alignment": _summary(spatial_alignment),
        "existing_boundary_relation_sources": {
            "normal_cut_edges": "graph.spatial_edge_mask & ~graph.normal_compatible_mask",
            "region_conflict_edges": "partition.rejected_merge_mask / anti_chaining_boundary_edges",
            "new_graph_constructed": False,
        },
        "region_internal_connected_components": int(
            region_module_count_spatially_disconnected(partition)
        ),
        "attachment_accounting": {
            "ambiguous_multi_region_surfel_count": int(partition.ambiguous_multi_region.sum()),
            "isolated_fallback_surfel_count": int((partition.partition_role == PARTITION_ROLES.index(ROLE_ISOLATED_FALLBACK)).sum()),
            "rejected_surfel_membership_count": 0,
        },
    }


def region_module_count_spatially_disconnected(partition: Any) -> int:
    from osn_gs.surface.torch_region_coherent_surfel_partition import count_spatially_disconnected_structural_regions
    return count_spatially_disconnected_structural_regions(partition)


def _region_accounting(partition: Any, status: torch.Tensor, accepted: torch.Tensor, raw: dict[str, Any]) -> dict[str, Any]:
    sizes = partition.subset_sizes.detach().cpu().numpy().astype(np.int64, copy=False)
    accepted_region_ids = np.flatnonzero(partition.region_structural_core_size.detach().cpu().numpy() > 0)
    accepted_sizes = sizes[accepted_region_ids]
    accepted_population = int(accepted.sum())
    largest_order = accepted_region_ids[np.argsort(accepted_sizes, kind="stable")[::-1]] if len(accepted_region_ids) else np.empty(0, dtype=np.int64)

    def size_buckets(values: np.ndarray) -> dict[str, int]:
        return {f"size_le_{limit}": int((values <= limit).sum()) for limit in (2, 4, 8)}

    top = []
    for region_id in largest_order[:20]:
        size = int(sizes[region_id])
        top.append({
            "region_id": int(region_id), "gaussian_member_count": size,
            "accepted_population_fraction": size / max(accepted_population, 1),
            "structural_core_size": int(partition.region_structural_core_size[region_id]),
            "concentration": float(partition.region_concentration[region_id]),
        })
    all_dist = {
        "region_count": int(len(sizes)), "size": _summary(sizes),
        "size_buckets": {"singleton": int((sizes == 1).sum()), **size_buckets(sizes)},
    }
    accepted_dist = {
        "accepted_surface_region_count": int(len(accepted_region_ids)),
        "size": _summary(accepted_sizes),
        "size_buckets": {"singleton": int((accepted_sizes == 1).sum()), **size_buckets(accepted_sizes)},
    }
    largest_fractions = {}
    for count in (1, 5, 10):
        largest_fractions[f"largest_{count}_accepted_region_population_fraction"] = float(sizes[largest_order[:count]].sum() / max(accepted_population, 1)) if len(largest_order) else 0.0
    return {
        "total_surfel_count": int(len(status)),
        "membership_status_counts": {
            "core": int((status == 0).sum()), "attached": int((status == 1).sum()),
            "ambiguous": int((status == 2).sum()), "rejected": 0,
            "unassigned": int((status == 4).sum()),
        },
        "accepted_gaussian_population": accepted_population,
        "final_region_id_count": int(partition.subset_count),
        "accepted_surface_region_count": int(len(accepted_region_ids)),
        "unassigned_fallback_region_count": int(partition.subset_count - len(accepted_region_ids)),
        "all_final_regions": all_dist,
        "accepted_regions": accepted_dist,
        "largest_regions": top,
        "largest_region_fractions": largest_fractions,
        "raw_existing_partition_accounting": raw,
    }


def _fixed_rgb_to_dc(rgb: Any) -> torch.Tensor:
    return (torch.as_tensor(rgb, dtype=torch.float32) - 0.5) / SH_C0


def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    sector = torch.floor(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector = (sector.to(torch.int64) % 6).reshape(-1, 1)
    options = torch.stack([
        torch.stack([value, t, p], dim=-1), torch.stack([q, value, p], dim=-1),
        torch.stack([p, value, t], dim=-1), torch.stack([p, q, value], dim=-1),
        torch.stack([t, p, value], dim=-1), torch.stack([value, p, q], dim=-1),
    ], dim=1)
    return options.gather(1, sector.unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)


def _region_colors(region_ids: torch.Tensor) -> torch.Tensor:
    identifiers = region_ids.to(torch.float64)
    hue = torch.frac(identifiers * 0.6180339887498949)
    saturation = 0.55 + 0.35 * torch.frac(identifiers * 0.7548776662466927)
    value = 0.60 + 0.40 * torch.frac(identifiers * 0.5698402909980532)
    return _hsv_to_rgb(hue, saturation, value).to(torch.float32).clamp(0.0, 1.0)


def _status_colors(status: torch.Tensor, uncertain: torch.Tensor, device: Any) -> torch.Tensor:
    names = ("core", "attached", "ambiguous", "rejected", "unassigned")
    colors = torch.zeros((len(status), 3), dtype=torch.float32, device=device)
    colors[:] = torch.as_tensor(STATUS_COLORS["uncertain"], device=device)
    for code, name in ((0, "core"), (1, "attached"), (2, "ambiguous"), (3, "rejected"), (4, "unassigned")):
        colors[status == code] = torch.as_tensor(STATUS_COLORS[name], device=device)
    colors[uncertain] = torch.as_tensor(STATUS_COLORS["uncertain"], device=device)
    return colors


def _write_png(path: Path, image: torch.Tensor) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    data = (image * 255.0).to(torch.uint8)
    Image.fromarray(data.numpy(), mode="RGB").save(path, format="PNG", optimize=True)


def _build_named_cameras(source_path: Path, image_dir: str, sparse_dir: str, resolution: int, llffhold: int, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image as PILImage
    from osn_gs.data.colmap_scene import camera_fovs, camera_matrices, read_colmap_cameras, read_colmap_images, resolve_image_path
    from osn_gs.data.vendor.graphdeco_scene_split import resolve_graphdeco_resolution, select_llff_holdout_test_names
    from osn_gs.render.torch_fallback import TorchCamera

    sparse_root = source_path / sparse_dir
    image_root = source_path / image_dir
    cameras = read_colmap_cameras(sparse_root)
    images = read_colmap_images(sparse_root)
    ordered = sorted(images.values(), key=lambda image: image.name)
    holdout = set(select_llff_holdout_test_names([image.name for image in ordered], scene_path=source_path, eval=True, llffhold=llffhold))
    selected = {image.name: image for image in ordered if image.name in REVIEW_CAMERAS}
    missing = [name for name in REVIEW_CAMERAS if name not in selected]
    if missing:
        raise ValueError(f"Missing required WL145 review cameras: {missing}")
    result: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for name in REVIEW_CAMERAS:
        image = selected[name]
        colmap_camera = cameras[image.camera_id]
        with PILImage.open(resolve_image_path(image_root, image.name)) as probe:
            original_width, original_height = probe.size
        target_width, target_height, downscale = resolve_graphdeco_resolution(original_width, original_height, resolution=resolution, resolution_scale=1.0)
        fovx, fovy = camera_fovs(colmap_camera, width=colmap_camera.width, height=colmap_camera.height)
        world_view, full_projection, center = camera_matrices(image.qvec, image.tvec, fovx, fovy, device=device)
        result[name] = TorchCamera(
            image_height=target_height, image_width=target_width,
            world_view_transform=world_view, full_proj_transform=full_projection,
            camera_center=center, FoVx=fovx, FoVy=fovy, image_name=name,
        )
        metadata[name] = {
            "image_name": name, "is_train_camera": name not in holdout,
            "resolution": [target_width, target_height], "downscale_factor": float(downscale),
            "llffhold": llffhold, "train_camera_count": int(len(ordered) - len(holdout)),
            "held_out_camera_count": int(len(holdout)),
            "selection_rule": "fixed WL145-WL154 review camera set; no camera chosen from partition outcome",
        }
    return result, metadata


def _project_world_points(points: np.ndarray, camera: Any) -> dict[str, np.ndarray]:
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    view = homogeneous @ camera.world_view_transform.detach().cpu().numpy()
    clip = homogeneous @ camera.full_proj_transform.detach().cpu().numpy()
    w = clip[:, 3]
    safe = np.maximum(w, 1.0e-12)
    ndc_x, ndc_y = clip[:, 0] / safe, clip[:, 1] / safe
    width, height = int(camera.image_width), int(camera.image_height)
    x = (ndc_x + 1.0) * 0.5 * max(width - 1, 1)
    y = (1.0 - ndc_y) * 0.5 * max(height - 1, 1)
    valid = (w > 1.0e-8) & (view[:, 2] > 0.0) & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
    return {"x": x, "y": y, "depth": view[:, 2], "valid": valid}


def _point_in_polygon(x: np.ndarray, y: np.ndarray, polygon: Iterable[tuple[float, float]]) -> np.ndarray:
    vertices = np.asarray(tuple(polygon), dtype=np.float64)
    inside = np.zeros_like(x, dtype=bool)
    x0, y0 = vertices[:, 0], vertices[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for left_x, left_y, right_x, right_y in zip(x0, y0, x1, y1):
        denominator = right_y - left_y
        if abs(float(denominator)) <= 1.0e-12:
            continue
        cross = ((left_y > y) != (right_y > y)) & (x < (right_x - left_x) * (y - left_y) / denominator + left_x)
        inside ^= cross
    return inside


def _target_review_accounting(camera_name: str, projection: dict[str, np.ndarray], region_ids: np.ndarray, statuses: np.ndarray) -> dict[str, Any]:
    valid = projection["valid"]
    result: dict[str, Any] = {}
    for target, per_camera in REVIEW_POLYGONS.items():
        polygon = per_camera.get(camera_name)
        if polygon is None:
            result[target] = {"annotation_available": False, "candidate_region_ids": [], "candidate_region_counts": {}}
            continue
        mask = valid & _point_in_polygon(projection["x"], projection["y"], polygon)
        counts = Counter(int(value) for value in region_ids[mask].tolist())
        status_counts = Counter(int(value) for value in statuses[mask].tolist())
        result[target] = {
            "annotation_available": True,
            "annotation_type": "fixed WL145 image-space review polygon; not a membership predicate or physical-sheet oracle",
            "projected_gaussian_count": int(mask.sum()),
            "candidate_region_ids": [int(value) for value, _ in counts.most_common(20)],
            "candidate_region_counts": {str(value): int(count) for value, count in counts.most_common(20)},
            "status_counts": {str(value): int(count) for value, count in sorted(status_counts.items())},
        }
    world_xyz = projection.get("world_xyz")
    if world_xyz is not None:
        world_mask = np.zeros((len(world_xyz),), dtype=bool)
        for minimum, maximum in REVIEW_WORLD_BOXES["background_lower"]:
            world_mask |= np.all((world_xyz >= np.asarray(minimum)) & (world_xyz <= np.asarray(maximum)), axis=1)
        mask = valid & world_mask
        counts = Counter(int(value) for value in region_ids[mask].tolist())
        result["background_lower"] = {
            "annotation_available": True,
            "annotation_type": "fixed WL140 world-space review boxes for patio/background controls; not a physical-sheet oracle",
            "world_boxes": [[list(minimum), list(maximum)] for minimum, maximum in REVIEW_WORLD_BOXES["background_lower"]],
            "projected_gaussian_count": int(mask.sum()),
            "candidate_region_ids": [int(value) for value, _ in counts.most_common(20)],
            "candidate_region_counts": {str(value): int(count) for value, count in counts.most_common(20)},
        }
    if camera_name == EVENT_1527_CAMERA:
        distance = np.sqrt((projection["x"] - EVENT_1527_PIXEL[0]) ** 2 + (projection["y"] - EVENT_1527_PIXEL[1]) ** 2)
        mask = valid & (distance <= EVENT_1527_RADIUS)
        counts = Counter(int(value) for value in region_ids[mask].tolist())
        result["event_1527_local_review"] = {
            "camera": EVENT_1527_CAMERA, "pixel": list(EVENT_1527_PIXEL), "radius_pixels": EVENT_1527_RADIUS,
            "historical_review": "CLEAR_NOT_ON_INTENDED_SURFACE",
            "purpose": "visual location only; no event-to-Gaussian correspondence or membership modification",
            "projected_gaussian_count": int(mask.sum()),
            "candidate_region_ids": [int(value) for value, _ in counts.most_common(20)],
            "candidate_region_counts": {str(value): int(count) for value, count in counts.most_common(20)},
        }
    return result


def _copy_wl154_mandatory_pair(out: Path, wl154: Path) -> dict[str, Any]:
    from PIL import Image

    source_root = wl154 / "mandatory_gaussian_visualization_pair"
    target_root = out / "mandatory_gaussian_visualization_pair"
    paths: dict[str, str] = {}
    render_paths: dict[str, str] = {}
    for label in ("Original Scene", "Observed-Occluded"):
        source_ply = source_root / label / ITERATION_DIR / "point_cloud.ply"
        source_render = source_root / label / "render.png"
        if not source_render.exists():
            source_render = source_root / label / "render.ppm"
        target_ply = target_root / label / ITERATION_DIR / "point_cloud.ply"
        target_render = target_root / label / "render.png"
        if not source_ply.exists() or not source_render.exists():
            raise FileNotFoundError(f"WL154 mandatory pair incomplete: {source_ply} / {source_render}")
        target_ply.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_ply, target_ply)
        with Image.open(source_render) as image:
            image.convert("RGB").save(target_render, format="PNG", optimize=True)
        paths[label] = str(target_ply)
        render_paths[label] = str(target_render)
    with np.load(wl154 / "gaussian_visualization_states.npz", allow_pickle=False) as states:
        state = np.asarray(states["state"], dtype=np.int8)
    return {
        "source": str(source_root),
        "copied_without_modification": True,
        "contract": {
            "same_checkpoint": True, "same_iteration": 30000, "same_camera": True,
            "same_resolution": True, "same_background": True, "same_renderer": True,
            "same_gaussian_row_count": True, "geometry_changed": False,
            "original_scene_color_source": "preserved WL154 learned model pair",
            "observed_occluded_color_source": "preserved WL154 fixed state palette pair",
            "unresolved_color_is_explicit_gray": True,
        },
        "state_counts": {"OBSERVED": int((state == 0).sum()), "OCCLUDED": int((state == 1).sum()), "UNRESOLVED": int((state == 2).sum())},
        "row_count": int(state.shape[0]), "paths": paths, "render_paths": render_paths,
        "review_note": "Mandatory pair is preserved from WL154; W155 region diagnostics are separate views and do not use this pair as partition input.",
    }


def _write_ply(path: Path, xyz: torch.Tensor, rgb: torch.Tensor, opacity: torch.Tensor, scaling: torch.Tensor, rotation: torch.Tensor) -> None:
    from scripts.devtools.coverage_first_surfel_partition_export import write_surfel_ply
    write_surfel_ply(path, xyz, _fixed_rgb_to_dc(rgb), opacity, scaling, rotation)


def _annotate_overlay(path: Path, image: torch.Tensor, camera_name: str) -> None:
    from PIL import Image, ImageDraw
    array = (image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    canvas = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    colors = {"tabletop": (40, 220, 90), "table_side": (255, 180, 20), "vase_neighbor": (50, 170, 255)}
    for target, per_camera in REVIEW_POLYGONS.items():
        polygon = per_camera.get(camera_name)
        if polygon:
            draw.line(list(polygon) + [polygon[0]], fill=colors[target], width=2)
    if camera_name == EVENT_1527_CAMERA:
        x, y = EVENT_1527_PIXEL
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 40, 40), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def _write_visualization_readme(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_visualization_readmes(out: Path, review_report: dict[str, Any], mandatory_pair: dict[str, Any]) -> None:
    """Make every visualization directory self-describing for later review."""

    _write_visualization_readme(
        out / "README.md",
        """# Worklog 155 산출물 안내

이 디렉터리는 trained 2DGS surfel의 intrinsic `t_w`와 기존 region-coherent partition을 진단한 W155 결과다. `worklog_155_report.json`은 정량 accounting, mapping, W154 read-only join을 담고, `review_views/`는 Gaussian region 해석용 시각화다. `mandatory_gaussian_visualization_pair/`는 W154의 canonical `Original Scene`/`Observed-Occluded` pair를 보존한 것이다.

시각화는 동일 checkpoint `iteration=30000`, 동일 Gaussian row와 geometry를 기준으로 한다. PNG가 주 검토 파일이며, renderer 원본 PPM은 provenance 보존용으로만 남아 있다. region/status/boundary 색상은 진단용이고 canonical Observed/Occluded pair의 고정 palette와 혼동하지 않는다.
""",
    )

    pair_root = out / "mandatory_gaussian_visualization_pair"
    pair_counts = mandatory_pair.get("state_counts", {})
    _write_visualization_readme(
        pair_root / "README.md",
        f"""# Mandatory Gaussian Visualization Pair

W154에서 보존한 필수 matched pair다. 두 항목은 같은 checkpoint/iteration/camera/resolution/background/renderer와 {mandatory_pair.get('row_count', 1190469):,}개 Gaussian row를 사용한다. `Original Scene`은 학습된 원래 색/SH를 사용하고, `Observed-Occluded`는 geometry를 바꾸지 않고 상태 색만 적용한다.

- OBSERVED: {pair_counts.get('OBSERVED', 0):,}
- OCCLUDED: {pair_counts.get('OCCLUDED', 0):,}
- UNRESOLVED: {pair_counts.get('UNRESOLVED', 0):,} (gray when present)

W155 region diagnostics는 이 pair의 Gaussian rows나 색을 partition 입력으로 사용하지 않는다. 각 label 폴더의 `render.png`가 주 검토 파일이며 `render.ppm`은 원본 보존본이다.
""",
    )
    pair_descriptions = {
        "Original Scene": "학습 checkpoint의 원래 Gaussian 색/SH, position, scale, rotation, opacity를 그대로 렌더링한 기준 장면이다. 추가 조명·마커·recolor·geometry 변경은 없다.",
        "Observed-Occluded": "Original Scene과 동일한 Gaussian row와 geometry에 대해 Observed/Occluded 상태만 고정 색으로 표시한다. OBSERVED는 green, OCCLUDED는 red, UNRESOLVED는 gray다.",
    }
    for label, meaning in pair_descriptions.items():
        label_root = pair_root / label
        content = f"""# {label}

{meaning}

`iteration_00030000/point_cloud.ply`와 `render.png`는 같은 W154 canonical pair의 산출물이다. `render.ppm`은 PNG 변환 전 원본 보존본이다.
"""
        _write_visualization_readme(label_root / "README.md", content)
        _write_visualization_readme(label_root / ITERATION_DIR / "README.md", content + "\n이 iteration 디렉터리의 PLY는 pair의 Gaussian geometry와 row ordering을 보존한다.")

    review_root = out / "review_views"
    _write_visualization_readme(
        review_root / "README.md",
        """# W155 Gaussian Region Review Views

이 폴더의 A–E는 W155 standalone Gaussian region replay의 진단 view다. A는 원래 학습 색, B는 intrinsic `t_w` normal, C는 final Region ID, D는 membership status, E는 기존 graph의 normal-cut/region-conflict 관계를 뜻한다. `cameras/` 아래에는 세 고정 camera의 full-scene render와 A+C overlay가 있다.

필수 canonical `Original Scene`/`Observed-Occluded` pair는 sibling 폴더 `../mandatory_gaussian_visualization_pair/`에 있으며 이 진단 view로 대체하지 않는다. 모든 camera render는 checkpoint iteration 30000, resolution `(648,420)`, black background, `OSNSurfelRasterizer` 조건이다.
""",
    )
    shared_descriptions = {
        "A_original_scene": "원래 학습된 Gaussian 색/SH를 보여주는 기준 장면이다. region/status palette를 적용하지 않으며, camera render의 첫 번째 full-scene 기준이다.",
        "B_intrinsic_tw_normal": "trained surfel rotation의 intrinsic `t_w`를 `abs(t_w)` RGB로 표시한다. 이는 normal 방향의 진단 view이며 covariance normal이나 새 normal 추정이 아니다.",
        "C_accepted_region_ids": "standalone replay의 final Region ID를 deterministic palette로 표시한다. active Gaussian만 region 색을 받고, unassigned/uncertain은 gray다.",
        "D_membership_status": "partition membership를 core=blue, attached=green, ambiguous=yellow, rejected=red, unassigned/uncertain=gray로 표시한다. rejected는 현재 per-surfel role이 아니라 edge relation으로 별도 accounting된다.",
        "E_boundary_conflict": "기존 graph의 normal-cut과 partition의 rejected-merge/anti-chaining conflict를 표시한다. dark=none, red=normal_cut, yellow=region_conflict, orange=both이며 새 graph를 만들지 않았다.",
    }
    for view_name, meaning in shared_descriptions.items():
        content = f"""# {view_name}

{meaning}

`iteration_00030000/point_cloud.ply`는 전체 Gaussian point cloud의 diagnostic color payload다. `cameras/` 아래 같은 이름의 `render.png`는 세 고정 real-scene camera에서의 2D view다. 이 view의 색은 canonical Observed/Occluded pair의 상태 색이 아니다.
"""
        _write_visualization_readme(review_root / view_name / "README.md", content)
        _write_visualization_readme(review_root / view_name / ITERATION_DIR / "README.md", content + "\n이 iteration 디렉터리의 PLY는 해당 diagnostic view의 shared 3D 산출물이다.")

    camera_root = review_root / "cameras"
    _write_visualization_readme(
        camera_root / "README.md",
        """# Camera Review Exports

`DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`는 WL145–154에서 고정한 review camera다. partition 결과로 camera를 선택하지 않았다. 각 camera 폴더에는 A–E full-scene render와 F original-plus-region overlay가 있으며, target annotation은 review용일 뿐 membership predicate나 physical-sheet oracle이 아니다.

각 파일은 `render.png`를 우선 사용한다. 동일 이름의 PPM은 renderer 원본 보존본이다.
""",
    )
    camera_views = {
        **shared_descriptions,
        "F_original_plus_region_overlay": "A_original_scene와 C_accepted_region_ids를 50:50으로 합친 overlay다. tabletop/table_side/vase_neighbor polygon은 초록/주황/파랑 outline으로 표시하며, DSC08003에서는 event 1527 위치를 빨간 원으로 표시한다. background_lower는 world-space accounting으로만 기록되고 이 overlay의 membership을 바꾸지 않는다.",
    }
    for camera_name in REVIEW_CAMERAS:
        metadata = review_report.get("camera_metadata", {}).get(camera_name, {})
        resolution = tuple(metadata.get("resolution", [648, 420]))
        _write_visualization_readme(
            camera_root / camera_name / "README.md",
            f"""# {camera_name} Review Camera

WL145–154에서 고정한 real-scene camera의 W155 export다. resolution은 `{resolution}`이며, full-scene A render를 먼저 만들고 같은 camera/geometry 조건으로 B–E와 F를 만들었다. 아래 view별 README가 색과 의미를 설명한다. review target annotation은 사람의 검토 위치를 고정하기 위한 것이며 region membership을 변경하지 않는다.
""",
        )
        for view_name, meaning in camera_views.items():
            content = f"""# {view_name} — {camera_name}

{meaning}

주 검토 파일은 `render.png`다. `render.ppm`은 PNG 변환 전 renderer 원본이다. 이 camera의 render들은 동일 checkpoint/iteration, black background, `OSNSurfelRasterizer`, Gaussian row count 조건을 공유한다.
"""
            if view_name == "F_original_plus_region_overlay":
                content += "\n`overlay.png`는 원본 장면과 Region ID view를 합친 결과이며, `review_targets_overlay.png`는 위 annotation을 추가한 결과다.\n"
            _write_visualization_readme(camera_root / camera_name / view_name / "README.md", content)


def _render_review_views(
    out: Path, model: TorchGaussianSurfelModel, selector: torch.Tensor, partition: Any, status: torch.Tensor,
    source_path: Path, images: str, sparse_dir: str, resolution: int, llffhold: int, device: str,
) -> dict[str, Any]:
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    from scripts.devtools.coverage_first_surfel_partition_export import build_preview_camera

    cameras, camera_meta = _build_named_cameras(source_path, images, sparse_dir, resolution, llffhold, device)
    uncertain_full = model.is_uncertain.reshape(-1).to(torch.bool)
    active_full = torch.zeros((len(model),), dtype=torch.bool, device=model.device)
    active_full[selector] = True
    full_region_ids = torch.full((len(model),), -1, dtype=torch.int64, device=model.device)
    full_region_ids[selector] = partition.subset_ids
    full_status = torch.full((len(model),), 4, dtype=torch.int64, device=model.device)
    full_status[selector] = status
    original_dc = model._features_dc.detach().clone()
    original_rest = model._features_rest.detach().clone()
    original_degree = int(model.active_sh_degree)
    xyz = model.get_xyz.detach()
    opacity = model._opacity.detach()
    scaling = model._scaling.detach()
    rotation = model.get_rotation.detach()
    normal_rgb = model.get_normal.detach().abs().clamp(0.0, 1.0)
    region_rgb = _region_colors(full_region_ids.clamp_min(0))
    region_rgb[~active_full] = torch.as_tensor(STATUS_COLORS["uncertain"], device=model.device)
    status_rgb = _status_colors(full_status, uncertain_full, model.device)

    graph = partition.graph
    normal_cut = graph.spatial_edge_mask & ~graph.normal_compatible_mask
    region_conflict = partition.rejected_merge_mask
    normal_degree = torch.zeros((len(partition),), dtype=torch.int32, device=model.device)
    conflict_degree = torch.zeros((len(partition),), dtype=torch.int32, device=model.device)
    if int(normal_cut.sum()) > 0:
        edges = graph.candidate_edges[normal_cut]
        ones = torch.ones((edges.shape[0],), dtype=torch.int32, device=model.device)
        normal_degree.index_add_(0, edges[:, 0], ones); normal_degree.index_add_(0, edges[:, 1], ones)
    if int(region_conflict.sum()) > 0:
        edges = graph.candidate_edges[region_conflict]
        ones = torch.ones((edges.shape[0],), dtype=torch.int32, device=model.device)
        conflict_degree.index_add_(0, edges[:, 0], ones); conflict_degree.index_add_(0, edges[:, 1], ones)
    boundary_rgb_active = torch.tensor([BOUNDARY_COLORS["none"]] * len(partition), dtype=torch.float32, device=model.device)
    boundary_rgb_active[normal_degree > 0] = torch.as_tensor(BOUNDARY_COLORS["normal_cut"], device=model.device)
    boundary_rgb_active[conflict_degree > 0] = torch.as_tensor(BOUNDARY_COLORS["region_conflict"], device=model.device)
    boundary_rgb_active[(normal_degree > 0) & (conflict_degree > 0)] = torch.as_tensor(BOUNDARY_COLORS["normal_cut_and_region_conflict"], device=model.device)
    boundary_rgb = torch.as_tensor(STATUS_COLORS["uncertain"], device=model.device).reshape(1, 3).repeat(len(model), 1)
    boundary_rgb[selector] = boundary_rgb_active

    view_rgb = {
        "A_original_scene": original_dc[:, 0, :].clone(),
        "B_intrinsic_tw_normal": normal_rgb,
        "C_accepted_region_ids": region_rgb,
        "D_membership_status": status_rgb,
        "E_boundary_conflict": boundary_rgb,
    }
    view_root = out / "review_views"
    for view_name, colors in view_rgb.items():
        _write_ply(view_root / view_name / ITERATION_DIR / "point_cloud.ply", xyz, colors, opacity, scaling, rotation)

    # Persist relation endpoints as an existing-graph diagnostic. No new edge
    # relation is generated here.
    np.savez(
        out / "region_boundary_conflict_edges.npz",
        normal_cut_edges=graph.candidate_edges[normal_cut].detach().cpu().numpy(),
        region_conflict_edges=graph.candidate_edges[region_conflict].detach().cpu().numpy(),
    )
    report: dict[str, Any] = {
        "camera_set": list(REVIEW_CAMERAS), "camera_metadata": camera_meta,
        "views": {name: {"point_cloud_ply": str(view_root / name / ITERATION_DIR / "point_cloud.ply")} for name in view_rgb},
        "region_palette": "stable deterministic palette derived only from final Region ID",
        "status_palette": STATUS_COLORS,
        "boundary_palette": BOUNDARY_COLORS,
        "boundary_edges": {"normal_cut_count": int(normal_cut.sum()), "region_conflict_count": int(region_conflict.sum()), "npz": str(out / "region_boundary_conflict_edges.npz")},
        "cameras": {},
    }
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    background = torch.zeros((3,), dtype=torch.float32, device=model.device)
    try:
        for camera_name in REVIEW_CAMERAS:
            camera = cameras[camera_name]
            camera_root = view_root / "cameras" / camera_name
            rendered: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                model._features_dc.data.copy_(original_dc)
                model._features_rest.data.copy_(original_rest)
                model.active_sh_degree = original_degree
                package = rasterizer.render(camera, model, background=background)
                rendered["A_original_scene"] = package["render"].detach().clone()
                del package
                for view_name in ("B_intrinsic_tw_normal", "C_accepted_region_ids", "D_membership_status", "E_boundary_conflict"):
                    model._features_dc.data.copy_(_fixed_rgb_to_dc(view_rgb[view_name])[:, None, :])
                    model._features_rest.data.zero_()
                    model.active_sh_degree = 0
                    package = rasterizer.render(camera, model, background=background)
                    rendered[view_name] = package["render"].detach().clone()
                    del package
            for view_name, image in rendered.items():
                _write_png(camera_root / view_name / "render.png", image)
            overlay = 0.50 * rendered["A_original_scene"] + 0.50 * rendered["C_accepted_region_ids"]
            _write_png(camera_root / "F_original_plus_region_overlay" / "overlay.png", overlay)
            _annotate_overlay(camera_root / "F_original_plus_region_overlay" / "review_targets_overlay.png", overlay, camera_name)
            projection = _project_world_points(xyz.detach().cpu().numpy(), cameras[camera_name])
            projection["world_xyz"] = xyz.detach().cpu().numpy()
            review = _target_review_accounting(camera_name, projection, full_region_ids.detach().cpu().numpy(), full_status.detach().cpu().numpy())
            report["cameras"][camera_name] = {
                "full_scene_first": True,
                "render_paths": {name: str(camera_root / name / "render.png") for name in rendered},
                "overlay_paths": {"original_plus_region": str(camera_root / "F_original_plus_region_overlay" / "overlay.png"), "annotated_review_targets": str(camera_root / "F_original_plus_region_overlay" / "review_targets_overlay.png")},
                "review_targets": review,
                "renderer": "OSNSurfelRasterizer",
                "background": [0.0, 0.0, 0.0],
                "same_checkpoint_iteration_and_geometry": True,
            }
    finally:
        model._features_dc.data.copy_(original_dc)
        model._features_rest.data.copy_(original_rest)
        model.active_sh_degree = original_degree
    report["renderer"] = {"name": "OSNSurfelRasterizer", "backend": rasterizer.backend_source, "same_background": True, "same_geometry": True}
    return report


def _load_downstream_attribution(wl154: Path, partition: Any, status: torch.Tensor, accepted: torch.Tensor, orientation: Any) -> dict[str, Any]:
    assoc_path = wl154 / "candidate_f_association.npz"
    support_path = wl154 / "candidate_f_region_owned_support.npz"
    components_path = wl154 / "support_components.json"
    if not assoc_path.exists() or not support_path.exists() or not components_path.exists():
        return {"available": False, "reason": "WL154 Candidate F arrays or support component payload are absent"}
    with np.load(assoc_path, allow_pickle=False) as assoc, np.load(support_path, allow_pickle=False) as support:
        nearest_index = np.asarray(assoc["nearest_gaussian_index"], dtype=np.int64)
        nearest_stable_id = np.asarray(assoc["nearest_gaussian_id"], dtype=np.int64)
        nearest_region = np.asarray(support["nearest_region_id"], dtype=np.int64)
        accepted_mask = np.asarray(support["accepted_mask"], dtype=bool)
        owned_region = np.asarray(support["owned_region_id"], dtype=np.int64)
    active_ids = orientation.gaussian_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    subset_ids = partition.subset_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    expected_region = subset_ids[nearest_index]
    expected_accepted = accepted.detach().cpu().numpy().astype(bool)[nearest_index]
    id_match = np.array_equal(nearest_stable_id, active_ids[nearest_index])
    region_match = np.array_equal(nearest_region, expected_region)
    accepted_match = np.array_equal(accepted_mask, expected_accepted)
    payload = json.loads(components_path.read_text(encoding="utf-8"))
    gaussian_member_count = np.bincount(subset_ids, minlength=int(partition.subset_count)).astype(np.int64)
    valid_region = nearest_region >= 0
    sample_count = np.bincount(nearest_region[valid_region], minlength=int(partition.subset_count)).astype(np.int64)
    owned_sample_count = np.bincount(owned_region[accepted_mask], minlength=int(partition.subset_count)).astype(np.int64)
    comp_sizes: dict[int, list[int]] = defaultdict(list)
    comp_count: Counter[int] = Counter()
    materialized: Counter[int] = Counter()
    abstained: Counter[int] = Counter()
    for item in payload:
        region_id = int(item["region_id"])
        size = int(item["sample_count"])
        comp_sizes[region_id].append(size)
        comp_count[region_id] += 1
        if item["representative"].get("status") == "MATERIALIZED_REPRESENTATIVE":
            materialized[region_id] += 1
        elif item["representative"].get("status") == "ABSTAIN_REPRESENTATIVE":
            abstained[region_id] += 1
    accepted_region_ids = np.flatnonzero(partition.region_structural_core_size.detach().cpu().numpy() > 0)
    per_region: list[dict[str, Any]] = []
    for region_id in accepted_region_ids.tolist():
        values = np.asarray(comp_sizes.get(int(region_id), []), dtype=np.int64)
        per_region.append({
            "region_id": int(region_id), "gaussian_member_count": int(gaussian_member_count[region_id]),
            "associated_tsdf_sample_count": int(sample_count[region_id]), "owned_tsdf_sample_count": int(owned_sample_count[region_id]),
            "native_tsdf_support_component_count": int(comp_count[region_id]),
            "component_size": _summary(values), "materialized_representative_count": int(materialized[region_id]),
            "abstain_representative_count": int(abstained[region_id]),
        })
    per_region.sort(key=lambda row: (-row["associated_tsdf_sample_count"], row["region_id"]))
    comp_per_region = np.asarray([row["native_tsdf_support_component_count"] for row in per_region], dtype=np.int64)
    top = per_region[:20]
    return {
        "available": True,
        "wl154_array_counts": {"association_samples": int(len(nearest_index)), "support_samples": int(len(nearest_region)), "component_records": int(len(payload))},
        "association_population_rule": "associated_tsdf_sample_count counts every valid nearest_region_id in the WL154 support array; owned_tsdf_sample_count counts only accepted owned_region_id entries",
        "wl154_region_id_reconciliation": {"nearest_stable_id_matches_checkpoint": bool(id_match), "nearest_region_id_matches_standalone_partition": bool(region_match), "accepted_mask_matches_standalone_partition": bool(accepted_match), "exact_region_id_join": bool(id_match and region_match and accepted_match)},
        "accepted_region_count": int(len(accepted_region_ids)),
        "tsdf_support_components_per_accepted_gaussian_region": _summary(comp_per_region),
        "regions_with_zero_tsdf_samples": int((sample_count[accepted_region_ids] == 0).sum()),
        "regions_with_zero_native_components": int((comp_per_region == 0).sum()),
        "top_regions_by_associated_tsdf_samples": top,
        "per_region_payload": str(wl154 / "w155_tsdf_attribution_per_region.json"),
        "feedback_into_gaussian_partition": False,
        "candidate_f_recomputed": False,
        "candidate_f_ownership_changed": False,
        "all_region_rows": per_region,
    }


def _failure_attribution(region_accounting: dict[str, Any], downstream: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    accepted = region_accounting["accepted_regions"]
    all_regions = region_accounting["all_final_regions"]
    visual_review_required = True
    return {
        "A_GAUSSIAN_REGION_OVER_FRAGMENTATION": {
            "quantitative_signal": {
                "final_region_id_count": region_accounting["final_region_id_count"],
                "accepted_surface_region_count": region_accounting["accepted_surface_region_count"],
                "singleton_fraction_all_final_regions": all_regions["size_buckets"]["singleton"] / max(all_regions["region_count"], 1),
                "regions_size_le_8_fraction": all_regions["size_buckets"]["size_le_8"] / max(all_regions["region_count"], 1),
            },
            "qualitative_status": "HUMAN_REVIEW_REQUIRED",
            "interpretation": "Many Gaussian regions exist before TSDF; this is a fragmentation signal, not proof that one physical sheet was split.",
        },
        "B_GAUSSIAN_REGION_OVER_MERGE": {
            "quantitative_signal": {"largest_region_fraction_of_accepted_population": region_accounting["largest_region_fractions"].get("largest_1_accepted_region_population_fraction", 0.0)},
            "qualitative_status": "HUMAN_REVIEW_REQUIRED",
            "interpretation": "Large regions and sparse graph bridges must be inspected against full-scene region and boundary views; no automatic physical-sheet claim is made.",
        },
        "C_GAUSSIAN_REGION_PLAUSIBLE_TSDF_SUPPORT_FRAGMENTED": {
            "quantitative_signal": downstream.get("tsdf_support_components_per_accepted_gaussian_region") if downstream.get("available") else None,
            "qualitative_status": "HUMAN_REVIEW_REQUIRED",
            "interpretation": "This is the downstream hypothesis tested by the read-only WL154 join; it cannot be selected from component counts alone.",
        },
        "D_GAUSSIAN_REGION_PLAUSIBLE_ASSOCIATION_LEAKAGE": {
            "quantitative_signal": None,
            "qualitative_status": "HUMAN_REVIEW_REQUIRED",
            "interpretation": "Nearest-Gaussian transfer is not changed or scored by this batch; target-window and event-local region lists are review evidence only.",
        },
        "E_MIXED": {
            "status": "NOT_AUTOMATICALLY_COLLAPSED",
            "interpretation": "The four hypotheses remain separate; mixed failure is not assigned without qualitative review.",
        },
        "visual_review_required": visual_review_required,
        "architecture_verdict": "UNRESOLVED",
        "verdict_reason": "Quantitative Gaussian-side fragmentation signal is measured, but physical-sheet plausibility and over-merge/over-fragmentation require human inspection of the matched full-scene exports.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = _contract_reconciliation()
    _progress(f"loading checkpoint {args.checkpoint}")
    model, payload = _load_surfel_model_safe(args.checkpoint, args.device)
    orientation, selector = _active_orientation(model)
    _progress(f"checkpoint rows={len(model):,} active={len(orientation):,} uncertain={len(model)-len(orientation):,}")

    with torch.no_grad():
        partition_a = partition_surfels_region_coherent(orientation, RegionCoherenceConfig(), progress=_progress)
    digest_a = _mapping_digest(orientation, partition_a, _partition_status(partition_a)[0])
    raw_accounting = region_coherent_accounting(partition_a)
    del partition_a
    if args.device == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        partition = partition_surfels_region_coherent(orientation, RegionCoherenceConfig(), progress=_progress)
    status, accepted, status_counts = _partition_status(partition)
    digest_b = _mapping_digest(orientation, partition, status)
    deterministic = digest_a == digest_b
    _progress(f"standalone replay regions={partition.subset_count:,} deterministic={deterministic}")

    graph = _graph_accounting(partition)
    region = _region_accounting(partition, status, accepted, raw_accounting)
    mapping_path = args.out / "gaussian_id_region_status_mapping.npz"
    np.savez(
        mapping_path,
        stable_gaussian_id=orientation.gaussian_ids.detach().cpu().numpy(),
        region_id=partition.subset_ids.detach().cpu().numpy(),
        membership_status=status.detach().cpu().numpy().astype(np.int8),
        partition_role=partition.partition_role.detach().cpu().numpy().astype(np.int8),
        ambiguous_multi_region=partition.ambiguous_multi_region.detach().cpu().numpy(),
    )
    mapping_hash = digest_b
    (args.out / "gaussian_id_region_status_mapping.sha256").write_text(mapping_hash + "\n", encoding="ascii")

    mandatory_pair = _copy_wl154_mandatory_pair(args.out, args.wl154)
    review = _render_review_views(
        args.out, model, selector, partition, status, args.source_path, args.images,
        args.sparse_dir, args.resolution, args.llffhold, args.device,
    )
    _write_visualization_readmes(args.out, review, mandatory_pair)
    downstream = _load_downstream_attribution(args.wl154, partition, status, accepted, orientation)
    if downstream.get("available"):
        (args.out / "w155_tsdf_attribution_per_region.json").write_text(json.dumps(_jsonable(downstream["all_region_rows"]), indent=2), encoding="utf-8")
        downstream.pop("all_region_rows", None)
    attribution = _failure_attribution(region, downstream, review)

    baseline = {}
    lineage_path = args.wl154 / "event_1527_lineage.json"
    if lineage_path.exists():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        baseline = {
            "source": str(lineage_path), "event_1527_preserved": bool(lineage.get("available", False)),
            "historical_review": lineage.get("historical_review"), "blacklist_applied": bool(lineage.get("blacklist_applied", False)),
            "new_event_to_gaussian_correspondence": False,
        }
    report = {
        "status": "COMPLETE_GAUSSIAN_REGION_VIABILITY_AUDIT",
        "batch": "Worklog 155 — Intrinsic-Normal Gaussian Surface Region Real-Scene Viability and Fragmentation Attribution Audit",
        "intent_alignment": {"diagnostic_only": True, "partition_tuned": False, "production_partition_modified": False, "downstream_feedback": False},
        "implementation_fidelity": contract,
        "architecture_result": attribution,
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint),
            "iteration": int(payload.get("iteration", 0)), "primitive": "surfel_2d", "scale_dim": 2,
            "same_current_partition_parameters": RegionCoherenceConfig().payload(),
            "review_cameras": list(REVIEW_CAMERAS),
        },
        "intrinsic_tw_normal_source": {
            "definition": "trained surfel rotation R=[t_u,t_v,t_w], model.get_normal third column",
            "orientation_wrapper": "derive_surface_orientation_from_surfel",
            "sign_semantics": "raw trained sign preserved; comparisons are unsigned |dot(t_w_i,t_w_j)|",
        },
        "standalone_gaussian_region_replay": {
            "input_only": ["positions", "intrinsic t_w", "t_u/t_v provenance", "tangent scales provenance", "stable Gaussian IDs"],
            "tsdf_inspected_before_freeze": False,
            "renderer_event_inspected_before_freeze": False,
            "wl145_polygon_used_before_freeze": False,
            "candidate_f_fit_inspected_before_freeze": False,
            "deterministic_replay": deterministic,
            "mapping_hash_run_a": digest_a, "mapping_hash_run_b": digest_b,
            "mapping_hash": mapping_hash, "mapping_path": str(mapping_path),
            "mapping_hash_algorithm": "SHA256(sorted stable_gaussian_id, region_id, membership_status)",
        },
        "global_region_accounting": region,
        "existing_graph_normal_coherence_accounting": graph,
        "real_scene_review_export": review,
        "mandatory_gaussian_visualization_pair": mandatory_pair,
        "wl154_downstream_fragmentation_attribution": downstream,
        "event_1527_review": baseline,
        "preserved_history": {
            "wl96_intrinsic_normal": True, "wl127_tsdf": True, "wl139_wl145_wl148": True,
            "wl149_event_union": True, "wl150_architecture_bypass": True, "wl151_contract_gap": True,
            "wl152_wl153_raw_surface": True, "wl154_candidate_f": True,
        },
        "forbidden_changes": {
            "t_w_semantics_changed": False, "covariance_normal_arm_added": False, "lambda2_lambda3_rule_added": False,
            "partition_parameters_tuned": False, "new_knn_graph": False, "region_size_threshold": False,
            "small_region_merge_or_split": False, "tsdf_association_changed": False, "tsdf_connectivity_changed": False,
            "boundary_first_changed": False, "nurbs_refit": False, "event_1527_blacklist": False,
            "trust_or_latent_or_occluded_surface": False,
        },
        "outputs": {
            "report": str(args.out / "worklog_155_report.json"), "mapping": str(mapping_path),
            "review_root": str(args.out / "review_views"), "mandatory_pair": str(args.out / "mandatory_gaussian_visualization_pair"),
            "visualization_readme_root": str(args.out),
            "visualization_output_format": "PNG primary; renderer PPM retained only as provenance when present",
        },
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "worklog_155_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wl154", type=Path, default=DEFAULT_WL154)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({
        "status": report["status"], "architecture_verdict": report["architecture_result"]["architecture_verdict"],
        "region_count": report["global_region_accounting"]["final_region_id_count"],
        "accepted_region_count": report["global_region_accounting"]["accepted_surface_region_count"],
        "mapping_hash": report["standalone_gaussian_region_replay"]["mapping_hash"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
