"""Worklog 127 -- Evidence-bounded projective TSDF for direct Visible Surface
construction.

ONE alternative construction premise, end to end:

    canonical renderer median-depth observations
        -> evidence-bounded projective TSDF
        -> masked zero level-set
        -> Visible Surface geometry

The candidate uses NO historical visible topology, KNN relation, region
assignment, boundary loop, component recovery or chart construction to decide
where its surface exists. Those appear only AFTER the mesh is built, read-only,
for diagnostic attribution and for the A/B baseline arm.
"""

from __future__ import annotations

import argparse
import json
import math
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

from coverage_first_surfel_partition_export import (  # noqa: E402
    PRIMITIVE_SURFEL_2D, checkpoint_primitive, load_primitive_model, _rgb_to_f_dc,
    write_ppm, write_surfel_ply,
)
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402

from evidence_bounded_tsdf import attribution, extraction, field as tsdf_field, mesh_ops, scale, synthetic  # noqa: E402
from evidence_bounded_tsdf.attribution import (  # noqa: E402
    MESH_OCCLUDED, MESH_STATE_NAMES, MESH_UNOCCLUDED, aggregate_mesh_states, mesh_occlusion_for_view,
)

WL119_REPRESENTATIVE_UNION = 785937
WL122_SOURCE_EVENTS = 43817760
WL119_METRIC_G_MEDIAN = 0.0046154288575053215
WL119_METRIC_G_P95 = 0.021156545728445053
WL119_METRIC_G_MAX = 0.19791387021541595
WL119_FITTED_CHART_COUNT = 14900

_ITERATION_DIR = "iteration_0000001"
_SCENE_RGB = (0.07, 0.08, 0.10)
_REPORT_NAME = "evidence_bounded_projective_tsdf_report.json"
_WORKLOG_NAME = "127_evidence_bounded_projective_tsdf.md"

# Fixed a priori. Every one is a selection stride, a sample count or a reporting
# bin -- never a threshold any verdict depends on, and never swept.
CLOSURE_MAX_ROUNDS = 60
EXTRACTION_BLOCK = 64
EVIDENCE_SAMPLE_STRIDE = 1          # exhaustive over all valid median events
HALLUCINATION_SAMPLE_STRIDE = 97
REVIEW_CASES_PER_REGION = 10
SLICE_HALF_EXTENT_VOXELS = 220
PREVIEW_VIEW_STRIDE = 27            # 161 // 27 -> six representative viewpoints
LOW_SUPPORT_COUNT = 1               # a REPORTING label, never a deletion rule

REGION_LABELS = ["table_top", "table_side_curved", "table_legs", "patio", "hedge"]


def _progress(message: str) -> None:
    print(f"[wl127-tsdf] {message}", flush=True)


def _quantiles(values, *, extra=(0.01, 0.25, 0.75, 0.99)) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    out: dict[str, Any] = {
        "count": int(array.size), "finite_count": int(finite.size),
        "non_finite_count": int(array.size - finite.size),
    }
    if finite.size == 0:
        return out
    ordered = np.sort(finite)

    def pct(fraction: float) -> float:
        return float(ordered[min(ordered.size - 1, max(0, int(round(fraction * (ordered.size - 1)))))])

    out.update({
        "min": pct(0.0), "median": pct(0.5), "mean": float(ordered.mean()),
        "p95": pct(0.95), "p99": pct(0.99), "max": pct(1.0),
    })
    for fraction in extra:
        out[f"p{int(fraction * 100):02d}"] = pct(fraction)
    return out


def write_view_readme(folder: Path, body: str, surfels: int) -> None:
    """Every export view folder carries its own Korean README (required by
    docs/output_folder_conventions.md). Written HERE, in the script, so it can
    never be forgotten as a manual post-step."""

    folder.mkdir(parents=True, exist_ok=True)
    footer = (
        "\n---\n"
        "체크포인트: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt` "
        f"({surfels:,} surfel, 161 train camera)\n"
        f"전체 리포트: `../{_REPORT_NAME}` · "
        f"Worklog: [`docs/worklogs/{_WORKLOG_NAME}`](../../../../docs/worklogs/{_WORKLOG_NAME})\n"
    )
    (folder / "README.md").write_text(body + footer, encoding="utf-8")


def peak_memory() -> dict[str, float]:
    import psutil

    process = psutil.Process()
    return {
        "peak_cpu_rss_gib": float(getattr(process.memory_info(), "peak_wset", process.memory_info().rss)) / 2 ** 30,
        "current_cpu_rss_gib": float(process.memory_info().rss) / 2 ** 30,
        "peak_gpu_allocated_gib": float(torch.cuda.max_memory_allocated()) / 2 ** 30 if torch.cuda.is_available() else 0.0,
        "peak_gpu_reserved_gib": float(torch.cuda.max_memory_reserved()) / 2 ** 30 if torch.cuda.is_available() else 0.0,
    }


# --------------------------------------------------------------------------
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
    parser.add_argument("--max-views", type=int, default=0, help="smoke-test only; 0 = all 161")
    parser.add_argument("--wl120-npz", type=Path, default=REPO_ROOT / "output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz")
    parser.add_argument("--wl121-npz", type=Path, default=REPO_ROOT / "output/confirmed/121_osn_gs_observed_occluded_value_space/value_space_supplemental_bank.npz")
    parser.add_argument("--wl123-npz", type=Path, default=REPO_ROOT / "output/confirmed/123_osn_gs_volumetric_frontier_query_contract/volumetric_query_contract.npz")
    parser.add_argument("--wl119-report", type=Path, default=REPO_ROOT / "output/confirmed/119_osn_gs_geometry_uv_control_correction/visible_nurbs_geometry_uv_control_correction_report.json")
    parser.add_argument("--baseline-max-charts", type=int, default=0, help="0 = replay every historical chart")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--cache", type=Path, default=None,
                        help="reuse/write the fused field + extracted mesh so a later stage can be rerun "
                             "without repeating the multi-hour construction; the cache stores ONLY results, "
                             "never a parameter, so a cached run is bit-identical to a fresh one")
    parser.add_argument("--skip-synthetic", action="store_true")
    arguments = parser.parse_args()

    started = time.time()
    device = arguments.device
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    report: dict[str, Any] = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 127 (evidence-bounded projective TSDF)",
        "worklog_number_note": (
            "the directive asked for 'Worklog 125'; docs/worklogs/125_fixed_gaussian_visualization_contract.md and "
            "output/126_* were already taken by concurrent work, so this batch is filed as 127 with no other change"
        ),
        "question": (
            "Can renderer-native median surface observations be fused directly into a scene-covering Visible "
            "Surface without first solving the historical region/topology/boundary architecture, and does the "
            "result stay strictly evidence-bounded?"
        ),
        "preserved": {
            "canonical_renderer": "unmodified",
            "checkpoint_and_cameras": "unmodified",
            "candidate_B_and_aggregation": "unmodified; read-only comparison only",
            "worklog_107_109_topology": "unmodified; read-only attribution only, never a construction input",
            "worklog_111_119_visible_nurbs": "unmodified; replayed as the A/B baseline arm",
            "worklog_120_123": "unmodified; their query banks are consumed read-only",
        },
        "control_experiment": {
            "changed_only": "the Visible Surface CONSTRUCTION candidate",
            "tsdf_construction_imports_historical_topology": False,
            "tuned": "NOTHING -- h is derived once, mu = 3h is fixed a priori, fusion weight is 1, there is no "
                     "minimum-view rule, no support threshold, no smoothing and no hole filling",
        },
    }

    # =============================================================== 1. evidence
    _progress("[1/14] loading checkpoint, cameras and canonical median-depth evidence")
    model, payload = load_primitive_model(arguments.checkpoint, device=device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D:
        raise SystemExit(f"expected a 2D surfel checkpoint, got {primitive}")
    cameras, camera_meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, device,
    )
    if arguments.max_views:
        cameras = cameras[:: max(1, len(cameras) // arguments.max_views)][: arguments.max_views]
    total_model_count = int(model.get_xyz.shape[0])

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    depth_maps: list[torch.Tensor] = []
    representative_maps: list[torch.Tensor] = []
    representative_union: set[int] = set()
    for index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        depth_maps.append(tsdf_field.median_depth_map(package["out_others"]).reshape(-1).clone())
        representative = package["representative_id"].reshape(-1).to(torch.int64).clone()
        representative_maps.append(representative)
        representative_union.update(torch.unique(representative[representative >= 0]).tolist())
        del package
        if index % 40 == 0:
            _progress(f"  rendered view {index}/{len(cameras)}")
    valid_events = int(sum(int((d > 0).sum().item()) for d in depth_maps))
    report["evidence"] = {
        "train_cameras": len(cameras),
        "camera_meta": camera_meta,
        "trained_surfels": total_model_count,
        "valid_renderer_median_observations": valid_events,
        "median_representative_union": len(representative_union),
        "worklog_119_representative_union": WL119_REPRESENTATIVE_UNION,
        "model_renderer_unchanged": bool(len(representative_union) == WL119_REPRESENTATIVE_UNION),
        "worklog_122_source_events": WL122_SOURCE_EVENTS,
        "matches_worklog_122_event_count": bool(valid_events == WL122_SOURCE_EVENTS),
    }
    _progress(f"  {valid_events:,} valid median events, representative union {len(representative_union):,}")

    # per-pixel region attribution (WORKING INTERPRETATION, read-only, post hoc)
    from observed_occluded.query_bank import region_of_surfel

    region_index, region_meta = region_of_surfel(model, cameras[len(cameras) // 2])
    region_of_pixel = [
        torch.where(rep >= 0, region_index[rep.clamp(min=0)], torch.full_like(rep, -1))
        for rep in representative_maps
    ]
    report["region_attribution"] = {
        "labels": REGION_LABELS, "meta": region_meta,
        "status": "WORKING INTERPRETATION ONLY -- reused verbatim from worklogs 108-123, never a construction input",
    }

    # ============================================================ 2. fixed scale
    _progress("[2/14] canonical scale h from renderer sampling (no sweep)")
    canonical = scale.derive_canonical_scale([scale.view_footprints(c, d) for c, d in zip(cameras, depth_maps)])
    h, mu = canonical.h, canonical.mu
    report["parameter_derivation"] = canonical.as_report()
    _progress(f"  h = {h:.12f}  mu = {mu:.12f}")

    # =========================================================== 3. field fusion
    _progress("[3/14] seeding candidate voxels from renderer median events")
    seed_keys = torch.zeros((0,), dtype=torch.int64, device=device)
    dropped = 0
    for camera, depth in zip(cameras, depth_maps):
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)
        world = tsdf_field.unproject_pixels(camera, valid, depth[valid])
        keys, out_of_range = tsdf_field.encode_keys(tsdf_field.voxel_index_of(world, h), margin=CLOSURE_MAX_ROUNDS + 4)
        dropped += out_of_range
        seed_keys = tsdf_field.union_sorted(seed_keys, keys)
        del world, keys
    _progress(f"  {int(seed_keys.numel()):,} seed voxels (renderer event voxels), {dropped} out of representable range")

    _progress("[4/14] uniform TSDF fusion over every training view, grown to closure")
    views = list(zip(cameras, depth_maps))
    field_cache = (arguments.cache / "field.npz") if arguments.cache else None
    if field_cache is not None and field_cache.exists():
        _progress(f"  reusing cached field from {field_cache}")
        cached = np.load(field_cache, allow_pickle=True)
        if float(cached["h"]) != h or float(cached["mu"]) != mu:
            raise SystemExit("cached field was built at a different h/mu -- refusing to mix scales")
        field = tsdf_field.SparseProjectiveTSDF(
            keys=torch.tensor(cached["keys"], dtype=torch.int64, device=device),
            value=torch.tensor(cached["value"], dtype=torch.float32, device=device),
            support_count=torch.tensor(cached["support_count"], dtype=torch.int32, device=device),
            h=h, mu=mu,
        )
        closure = json.loads(str(cached["closure"]))
    else:
        field, closure = tsdf_field.grow_field_to_closure(
            seed_keys, views, h=h, mu=mu, max_rounds=CLOSURE_MAX_ROUNDS, chunk_size=8_000_000,
            progress=lambda message: _progress("  " + message),
        )
        if field_cache is not None:
            field_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                field_cache, keys=field.keys.detach().cpu().numpy(),
                value=field.value.detach().cpu().numpy(),
                support_count=field.support_count.detach().cpu().numpy(),
                h=h, mu=mu, closure=json.dumps(closure, default=str),
            )
    del seed_keys
    torch.cuda.empty_cache()
    support = field.support_count
    report["field"] = {
        "authoritative_voxels": len(field),
        "seed_voxels_out_of_representable_range": dropped,
        "support_count_distribution": _quantiles(support.detach().cpu().numpy()),
        "support_count_histogram_first_16": torch.bincount(support.to(torch.int64), minlength=17)[:17].tolist(),
        "voxels_with_support_count_1": int((support == 1).sum().item()),
        "fraction_support_count_1": float((support == 1).to(torch.float64).mean().item()),
        "enumeration_closure": closure,
        "unknown_contract": (
            "UNKNOWN voxels are represented by ABSENCE. Nothing is initialized to +1/-1/0/nearest, nothing is "
            "diffused, no hole is filled and no completion of any kind runs."
        ),
    }
    _progress(f"  authoritative voxels {len(field):,}, closed={closure['closed']} after {closure['rounds_run']} rounds")

    # ============================================================ 4. extraction
    _progress("[5/14] masked zero level-set extraction")
    extract_started = time.time()
    mesh_cache = (arguments.cache / "mesh.npz") if arguments.cache else None
    if mesh_cache is not None and mesh_cache.exists():
        _progress(f"  reusing cached mesh from {mesh_cache}")
        cached = np.load(mesh_cache, allow_pickle=True)
        surface = extraction.ExtractedSurface(
            vertices=cached["vertices"], faces=cached["faces"],
            vertex_support_count=cached["vertex_support_count"],
            vertex_field_value=cached["vertex_field_value"], h=h,
            stats=json.loads(str(cached["stats"])),
        )
    else:
        surface = extraction.extract_zero_level_set(
            field, block=EXTRACTION_BLOCK, batch_blocks=6,
            progress=lambda message: _progress("  " + message),
        )
        if mesh_cache is not None:
            mesh_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                mesh_cache, vertices=surface.vertices, faces=surface.faces,
                vertex_support_count=surface.vertex_support_count,
                vertex_field_value=surface.vertex_field_value,
                stats=json.dumps(surface.stats, default=str),
            )
    areas = mesh_ops.triangle_areas(surface.vertices, surface.faces)
    labels, component_count = mesh_ops.connected_components(int(surface.vertices.shape[0]), surface.faces)
    face_labels = labels[surface.faces[:, 0]] if surface.faces.shape[0] else np.zeros((0,), dtype=np.int64)
    if face_labels.size:
        _unique, component_sizes = np.unique(face_labels, return_counts=True)
        component_area = np.zeros(int(_unique.max()) + 1, dtype=np.float64)
        np.add.at(component_area, face_labels, areas)
    else:
        component_sizes = np.zeros((0,), dtype=np.int64)
        component_area = np.zeros((0,), dtype=np.float64)
    report["reconstruction"] = {
        "extraction_stats": surface.stats,
        "vertices": int(surface.vertices.shape[0]),
        "faces": int(surface.faces.shape[0]),
        "connected_components": component_count,
        "largest_component_triangle_fraction": float(component_sizes.max() / component_sizes.sum()) if component_sizes.size else 0.0,
        "component_triangle_count_distribution": _quantiles(component_sizes),
        "total_surface_area": float(areas.sum()),
        "largest_component_area_fraction": float(component_area.max() / component_area.sum()) if component_area.size else 0.0,
        "triangle_area_distribution_over_h_squared": _quantiles(areas / (h * h)),
        "extraction_wall_clock_seconds": time.time() - extract_started,
        "watertightness": "NOT forced; no hole filling, no repair, no smoothing, no decimation was run",
    }
    _progress(
        f"  mesh: {surface.vertices.shape[0]:,} vertices, {surface.faces.shape[0]:,} faces, "
        f"{component_count:,} components, area {areas.sum():.3f}"
    )

    vertices_gpu = torch.tensor(surface.vertices, dtype=torch.float32, device=device)
    faces_gpu = torch.tensor(surface.faces, dtype=torch.int64, device=device)

    # ================================================== 5. hallucination contract
    _progress("[6/14] hallucination / unsupported-gap audit")
    hallucination: dict[str, Any] = {
        "triangles_from_cells_with_all_eight_authoritative_corners": int(surface.faces.shape[0]),
        "fraction_of_triangles_from_fully_authoritative_cells": 1.0 if surface.faces.shape[0] else 0.0,
        "enforced_by": (
            "extraction discards every triangle whose owning cell lacks an authoritative corner; the count of "
            "discarded triangles is reported in extraction_stats"
        ),
        "kept_triangles_outside_their_claimed_cell": surface.stats["kept_triangles_outside_their_claimed_cell"],
    }

    sample_rows = np.arange(0, int(surface.faces.shape[0]), HALLUCINATION_SAMPLE_STRIDE)
    sample_points = torch.tensor(
        surface.vertices[surface.faces[sample_rows]].mean(axis=1), dtype=torch.float32, device=device
    ) if sample_rows.size else torch.zeros((0, 3), device=device)

    # ======================================= 6. renderer-evidence reproduction
    _progress("[7/14] renderer-evidence reproduction (median event -> extracted surface)")
    triangle_index = mesh_ops.build_triangle_cell_index(vertices_gpu, faces_gpu, h)
    evidence_cache = (arguments.cache / "evidence.npz") if arguments.cache else None
    evidence_distance_chunks: list[np.ndarray] = []
    evidence_region_chunks: list[np.ndarray] = []
    if evidence_cache is not None and evidence_cache.exists():
        _progress(f"  reusing cached evidence distances from {evidence_cache}")
        cached = np.load(evidence_cache)
        evidence_distance_chunks.append(cached["distance"])
        evidence_region_chunks.append(cached["region"])
        views_iterable: list = []
    else:
        views_iterable = list(enumerate(views))
    for view_index, (camera, depth) in views_iterable:
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)[::EVIDENCE_SAMPLE_STRIDE]
        if valid.numel() == 0:
            continue
        world = tsdf_field.unproject_pixels(camera, valid, depth[valid])
        distance = mesh_ops.nearest_surface_distance(world, triangle_index, max_radius=3, chunk=2_000_000)
        evidence_distance_chunks.append(distance.detach().cpu().numpy())
        evidence_region_chunks.append(region_of_pixel[view_index][valid].detach().cpu().numpy())
        del world, distance
        if view_index % 20 == 0:
            _progress(f"  evidence view {view_index}/{len(views)}")
    evidence_distance = np.concatenate(evidence_distance_chunks) if evidence_distance_chunks else np.zeros((0,))
    evidence_region = np.concatenate(evidence_region_chunks) if evidence_region_chunks else np.zeros((0,), dtype=np.int64)
    del evidence_distance_chunks, evidence_region_chunks
    if evidence_cache is not None and not evidence_cache.exists():
        evidence_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(evidence_cache, distance=evidence_distance, region=evidence_region)

    def _coverage(distance: np.ndarray) -> dict[str, Any]:
        total = int(distance.size)
        if total == 0:
            return {"events": 0}
        finite = np.isfinite(distance)
        return {
            "events": total,
            "distance": _quantiles(distance[finite]),
            "distance_over_h": _quantiles(distance[finite] / h),
            "fraction_within_h": float((distance <= h).mean()),
            "fraction_within_2h": float((distance <= 2 * h).mean()),
            "fraction_no_local_extracted_surface_beyond_3h": float((~finite).mean()),
        }

    report["renderer_evidence_reproduction"] = {
        "all_events": _coverage(evidence_distance),
        "by_region": {
            label: _coverage(evidence_distance[evidence_region == index])
            for index, label in enumerate(REGION_LABELS)
        },
        "bins_are_reporting_bins": "h and 2h are reporting bins derived from the fixed resolution, never fitting thresholds",
        "exhaustive": bool(EVIDENCE_SAMPLE_STRIDE == 1),
    }
    _progress(
        f"  evidence within h: {report['renderer_evidence_reproduction']['all_events']['fraction_within_h']:.4f}"
    )

    # ============================================== 7. raycast self-consistency
    _progress("[8/14] raycast self-consistency into all training cameras")
    mesh_depth_maps: list[torch.Tensor] = []
    raycast_stats = {"triangles_clipped_out": 0, "triangles_rasterized": 0, "triangles_beyond_largest_tier": 0}
    signed_error_chunks: list[np.ndarray] = []
    region_chunks: list[np.ndarray] = []
    hit_chunks: list[np.ndarray] = []
    raycast_cache = (arguments.cache / "raycast.npz") if arguments.cache else None
    if raycast_cache is not None and raycast_cache.exists():
        _progress(f"  reusing cached mesh depth maps from {raycast_cache}")
        cached = np.load(raycast_cache)
        mesh_depth_maps = [torch.tensor(row, dtype=torch.float32, device=device) for row in cached["mesh_depth"]]
        signed_error_chunks.append(cached["signed_error"])
        region_chunks.append(cached["region"])
        hit_chunks.append(cached["counted"])
        raycast_stats = json.loads(str(cached["stats"]))
        raycast_iterable: list = []
    else:
        raycast_iterable = list(enumerate(views))
    for view_index, (camera, depth) in raycast_iterable:
        mesh_depth, stats = mesh_ops.rasterize_mesh_depth(camera, vertices_gpu, faces_gpu, face_chunk=4_000_000)
        for key in raycast_stats:
            raycast_stats[key] += stats[key]
        flat = mesh_depth.reshape(-1)
        mesh_depth_maps.append(flat.clone())
        valid = depth > 0
        hit = valid & torch.isfinite(flat)
        signed_error_chunks.append((flat[hit] - depth[hit]).detach().cpu().numpy())
        region_chunks.append(region_of_pixel[view_index][hit].detach().cpu().numpy())
        hit_chunks.append(np.array([int(valid.sum().item()), int(hit.sum().item())], dtype=np.int64))
        del mesh_depth
        if view_index % 20 == 0:
            _progress(f"  raycast view {view_index}/{len(views)}")
    signed_error = np.concatenate(signed_error_chunks) if signed_error_chunks else np.zeros((0,))
    raycast_region = np.concatenate(region_chunks) if region_chunks else np.zeros((0,), dtype=np.int64)
    counted = np.concatenate(hit_chunks).reshape(-1, 2) if hit_chunks else np.zeros((0, 2), dtype=np.int64)
    del signed_error_chunks, region_chunks, hit_chunks
    if raycast_cache is not None and not raycast_cache.exists():
        raycast_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            raycast_cache,
            mesh_depth=np.stack([row.detach().cpu().numpy() for row in mesh_depth_maps]),
            signed_error=signed_error, region=raycast_region, counted=counted,
            stats=json.dumps(raycast_stats),
        )

    def _raycast_block(mask: np.ndarray | None = None) -> dict[str, Any]:
        error = signed_error if mask is None else signed_error[mask]
        return {
            "pixels_with_mesh_hit": int(error.size),
            "signed_depth_error": _quantiles(error),
            "absolute_depth_error": _quantiles(np.abs(error)),
            "absolute_depth_error_over_h": _quantiles(np.abs(error) / h),
        }

    report["raycast_self_consistency"] = {
        "pixels_with_canonical_median_depth": int(counted[:, 0].sum()) if counted.size else 0,
        "pixels_with_mesh_hit": int(counted[:, 1].sum()) if counted.size else 0,
        "ray_hit_coverage": float(counted[:, 1].sum() / counted[:, 0].sum()) if counted.size and counted[:, 0].sum() else 0.0,
        "all_pixels": _raycast_block(),
        "by_region": {
            label: _raycast_block(raycast_region == index) for index, label in enumerate(REGION_LABELS)
        },
        "rasterization_stats": raycast_stats,
        "ray_definition": (
            "pixel-centre ray/triangle intersection -- the same ray candidate B's frontier comparison uses, so "
            "mesh depth and canonical median depth are measured on identical rays"
        ),
    }
    _progress(f"  ray-hit coverage {report['raycast_self_consistency']['ray_hit_coverage']:.4f}")

    # ============================================ 8. hallucination sample audit
    if sample_points.numel():
        nearest_event = torch.full((sample_points.shape[0],), float("inf"), device=device)
        for camera, depth in views:
            width, height = int(camera.image_width), int(camera.image_height)
            projected = tsdf_field.project_world_points(
                sample_points, camera.world_view_transform, camera.full_proj_transform, width, height
            )
            index = projected.pixel_index.clamp(min=0)
            median = depth[index]
            usable = projected.relevant & (median > 0)
            event_world = tsdf_field.unproject_pixels(camera, index, median)
            distance = torch.linalg.norm(event_world - sample_points, dim=1)
            nearest_event = torch.minimum(nearest_event, torch.where(usable, distance, torch.full_like(distance, float("inf"))))
            del event_world, distance
        sample_keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(sample_points, h))
        sample_value, sample_support, sample_found = field.lookup(sample_keys)
        hallucination.update({
            "sampled_mesh_points": int(sample_points.shape[0]),
            "sample_stride": HALLUCINATION_SAMPLE_STRIDE,
            "nearest_renderer_median_event_distance": _quantiles(nearest_event.detach().cpu().numpy()),
            "nearest_renderer_median_event_distance_over_h": _quantiles(nearest_event.detach().cpu().numpy() / h),
            "local_support_count": _quantiles(sample_support.detach().cpu().numpy()),
            "sampled_points_inside_an_authoritative_voxel": int(sample_found.sum().item()),
            "low_support_sampled_points": int((sample_support <= LOW_SUPPORT_COUNT).sum().item()),
            "fraction_low_support": float((sample_support <= LOW_SUPPORT_COUNT).to(torch.float64).mean().item()),
            "sampled_points_further_than_2h_from_any_median_event": int((nearest_event > 2 * h).sum().item()),
            "policy": "flagged regions are EXPORTED for review, never deleted",
        })
    report["hallucination_audit"] = hallucination

    # ============================================ 9. SDF-induced occlusion audit
    _progress("[9/14] SDF-induced occlusion audit against frozen candidate B")
    from observed_occluded.shared import STATE_NAMES, STATE_OBSERVED, STATE_OCCLUDED, aggregate_global

    occlusion: dict[str, Any] = {
        "definition": (
            "MESH_OCCLUDED iff a mesh first-hit occurs strictly before the query on the same pixel-centre ray. "
            "No depth epsilon exists anywhere in this comparison."
        ),
        "status": "DIAGNOSTIC ONLY -- candidate B is not replaced, modified, or tuned",
    }
    for name, npz_path, state_key in (
        ("worklog_120_original_4712", arguments.wl120_npz, "states_B"),
        ("worklog_121_supplemental_908", arguments.wl121_npz, "states_B"),
        ("worklog_123_generic_corpus", arguments.wl123_npz, None),
    ):
        if not npz_path.exists():
            occlusion[name] = {"status": f"MISSING artifact {npz_path}"}
            continue
        bundle = np.load(npz_path, allow_pickle=True)
        positions_key = "world_position" if "world_position" in bundle.files else "positions"
        positions = torch.tensor(bundle[positions_key], dtype=torch.float32, device=device)
        if state_key is not None:
            b_per_view = bundle[state_key]
        else:
            b_per_view = bundle["base_states"]
        if b_per_view.shape[1] != len(views):
            occlusion[name] = {"status": f"view-count mismatch {b_per_view.shape[1]} vs {len(views)}"}
            continue
        mesh_per_view = np.full(b_per_view.shape, attribution.MESH_NOT_RELEVANT, dtype=np.int8)
        for view_index, (camera, _depth) in enumerate(views):
            width, height = int(camera.image_width), int(camera.image_height)
            projected = tsdf_field.project_world_points(
                positions, camera.world_view_transform, camera.full_proj_transform, width, height
            )
            states = mesh_occlusion_for_view(
                projected.depth, projected.pixel_index, projected.relevant, mesh_depth_maps[view_index]
            )
            mesh_per_view[:, view_index] = states.detach().cpu().numpy()
        b_global = aggregate_global(b_per_view)
        mesh_global = aggregate_mesh_states(mesh_per_view)
        occlusion[name] = {
            "queries": int(positions.shape[0]),
            "per_view_confusion": attribution.confusion(b_per_view.reshape(-1), mesh_per_view.reshape(-1)),
            "global_confusion": attribution.confusion(b_global, mesh_global),
            "mesh_global_state_counts": {
                MESH_STATE_NAMES[code]: int((mesh_global == code).sum()) for code in MESH_STATE_NAMES
            },
            "B_global_state_counts": {
                name_: int((b_global == code).sum()) for code, name_ in STATE_NAMES.items()
            },
        }
        if name == "worklog_120_original_4712":
            disagreement_positions = positions
            disagreement_b = b_global
            disagreement_mesh = mesh_global
            disagreement_kind = bundle["kind"]
            disagreement_region = bundle["region"]
        del bundle

    # attribute the WL120 disagreements
    if "disagreement_positions" in dir():
        mismatched = np.nonzero(
            ((disagreement_b == STATE_OBSERVED) & (disagreement_mesh == MESH_OCCLUDED))
            | ((disagreement_b == STATE_OCCLUDED) & (disagreement_mesh == MESH_UNOCCLUDED))
        )[0]
        if mismatched.size:
            points = disagreement_positions[torch.tensor(mismatched, device=device)]
            distance = mesh_ops.nearest_surface_distance(points, triangle_index, max_radius=3)
            keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(points, h))
            _value, support_here, found_here = field.lookup(keys)
            no_surface = ~torch.isfinite(distance)
            occlusion["worklog_120_disagreement_attribution"] = {
                "disagreeing_queries": int(mismatched.size),
                "missing_reconstructed_surface_within_3h": int(no_surface.sum().item()),
                "reconstructed_surface_present_within_3h": int((~no_surface).sum().item()),
                "query_inside_an_authoritative_voxel": int(found_here.sum().item()),
                "query_in_UNKNOWN_space": int((~found_here).sum().item()),
                "local_support_count": _quantiles(support_here.detach().cpu().numpy()),
                "nearest_surface_distance_over_h": _quantiles(distance.detach().cpu().numpy() / h),
                "by_kind": {
                    str(k): int((disagreement_kind[mismatched] == k).sum())
                    for k in np.unique(disagreement_kind[mismatched])
                },
                "by_region": {
                    REGION_LABELS[i]: int((disagreement_region[mismatched] == i).sum())
                    for i in range(len(REGION_LABELS))
                },
                "attribution_categories": [
                    "missing reconstructed surface (query in UNKNOWN space or no surface within 3h)",
                    "reconstructed surface shifted in front (surface present, mesh hit earlier than the frontier)",
                    "alternate-view visible surface (global aggregation rescued by another camera)",
                    "unsupported bridge candidate (surface present with support_count 1 far from any event)",
                    "exact boundary / provenance case (query IS a renderer event on its own source surface)",
                ],
            }
        # marker colours for export view I -- built here where the states live
        colour_map = {
            "agree": (0.10, 0.85, 0.35), "b_observed_mesh_occluded": (0.92, 0.18, 0.18),
            "b_occluded_mesh_free": (0.20, 0.55, 0.98), "other": (0.60, 0.60, 0.62),
        }
        marker_colours = []
        for row in range(disagreement_b.shape[0]):
            b_state, m_state = int(disagreement_b[row]), int(disagreement_mesh[row])
            if b_state == STATE_OBSERVED and m_state == MESH_OCCLUDED:
                marker_colours.append(colour_map["b_observed_mesh_occluded"])
            elif b_state == STATE_OCCLUDED and m_state == MESH_UNOCCLUDED:
                marker_colours.append(colour_map["b_occluded_mesh_free"])
            elif b_state in (STATE_OBSERVED, STATE_OCCLUDED):
                marker_colours.append(colour_map["agree"])
            else:
                marker_colours.append(colour_map["other"])
        occlusion["_positions"] = disagreement_positions
        occlusion["_colours"] = torch.tensor(marker_colours, dtype=torch.float32, device=device)
    report["sdf_induced_occlusion_audit"] = occlusion

    # ================================== 10. historical topology attribution
    _progress("[10/14] worklog 121 true-fragmentation context attribution (diagnostic only)")
    fragmentation: dict[str, Any] = {"status": "DIAGNOSTIC ONLY -- never used to change the SDF"}
    if arguments.wl121_npz.exists():
        bundle = np.load(arguments.wl121_npz, allow_pickle=True)
        world_a = torch.tensor(bundle["context_world_a"], dtype=torch.float32, device=device)
        world_b = torch.tensor(bundle["context_world_b"], dtype=torch.float32, device=device)
        midpoint = 0.5 * (world_a + world_b)
        distance_a = mesh_ops.nearest_surface_distance(world_a, triangle_index, max_radius=3)
        distance_b = mesh_ops.nearest_surface_distance(world_b, triangle_index, max_radius=3)
        distance_mid = mesh_ops.nearest_surface_distance(midpoint, triangle_index, max_radius=3)
        # This is peak GPU occupancy (field, mesh, triangle_index, mesh_depth_maps
        # and depth_maps are all still resident); free transient allocations and
        # use a smaller streaming chunk purely as a memory safety margin -- the
        # nearest-vertex RESULT is identical regardless of chunk size.
        torch.cuda.empty_cache()
        component_lookup = _nearest_mesh_component(world_a, world_b, vertices_gpu, labels, device, chunk=1_000_000)
        source_view = bundle["context_view_index"]
        depth_behaviour = []
        for row in range(world_a.shape[0]):
            view_index = int(source_view[row])
            if view_index >= len(views):
                depth_behaviour.append(float("nan"))
                continue
            camera = cameras[view_index]
            width, height = int(camera.image_width), int(camera.image_height)
            projected = tsdf_field.project_world_points(
                midpoint[row : row + 1], camera.world_view_transform, camera.full_proj_transform, width, height
            )
            hit = mesh_depth_maps[view_index][projected.pixel_index.clamp(min=0)]
            depth_behaviour.append(float((hit - projected.depth).item()))
        gating = bundle["context_gating_reason"]
        fragmentation.update({
            "contexts": int(world_a.shape[0]),
            "endpoint_A_has_surface_within_h": int((distance_a <= h).sum().item()),
            "endpoint_B_has_surface_within_h": int((distance_b <= h).sum().item()),
            "midpoint_has_surface_within_h": int((distance_mid <= h).sum().item()),
            "midpoint_has_no_surface_within_3h": int((~torch.isfinite(distance_mid)).sum().item()),
            "endpoints_on_the_same_extracted_mesh_component": int((component_lookup[0] == component_lookup[1]).sum()),
            "endpoints_on_different_extracted_mesh_components": int((component_lookup[0] != component_lookup[1]).sum()),
            "endpoint_A_distance_over_h": _quantiles(distance_a.detach().cpu().numpy() / h),
            "endpoint_B_distance_over_h": _quantiles(distance_b.detach().cpu().numpy() / h),
            "midpoint_distance_over_h": _quantiles(distance_mid.detach().cpu().numpy() / h),
            "source_view_mesh_minus_query_depth_at_midpoint": _quantiles(np.asarray(depth_behaviour)),
            "historical_gating_reason_counts": {
                str(k): int((gating == k).sum()) for k in np.unique(gating)
            },
            "interpretation_guard": (
                "same extracted component does NOT mean correct physical continuity, and surface near the "
                "midpoint does NOT mean the historical split was wrong. These are review outputs only."
            ),
        })
        marker_points = torch.cat([world_a, world_b, midpoint], dim=0)
        marker_colours = torch.cat([
            torch.tensor([[0.98, 0.85, 0.20]], device=device).expand(world_a.shape[0] * 2, 3),
            torch.tensor([[0.95, 0.45, 0.10]], device=device).expand(midpoint.shape[0], 3),
        ], dim=0).contiguous()
        report["_fragmentation_points"] = (marker_points, marker_colours)
        del bundle
    report["historical_topology_attribution"] = fragmentation

    # ======================================================== 11. baseline A/B
    _progress("[11/14] baseline A/B against the historical visible-surface / NURBS path")
    report["_tsdf_vertices"] = surface.vertices
    # The historical replay builds a full exact-KNN candidate graph over ~1.16M
    # surfels. Hand the GPU back to it: the sparse field is no longer needed
    # (its measurements are already in the report) and the mesh triangle index is
    # rebuilt inside the baseline arm from the vertices/faces it still holds.
    field_on_cpu = tsdf_field.SparseProjectiveTSDF(
        keys=field.keys.detach().cpu(), value=field.value.detach().cpu(),
        support_count=field.support_count.detach().cpu(), h=h, mu=mu,
    )
    del field
    # `triangle_index` holds an (F,3,3) copy of every triangle; the baseline arm
    # only needs it for the shared A/B query set, which it evaluates first.
    # The mesh depth maps are not needed again until the exports.
    mesh_depth_cpu = [row.detach().cpu() for row in mesh_depth_maps]
    mesh_depth_maps = []
    del vertices_gpu, faces_gpu
    torch.cuda.empty_cache()
    _progress(
        f"  GPU before baseline replay: allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
        f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB"
    )
    baseline = _baseline_arm(
        arguments, report, model, cameras, views, depth_maps, representative_maps,
        region_of_pixel, h, device, triangle_index, evidence_distance, evidence_region, counted,
    )
    report["baseline_ab"] = baseline
    mesh_depth_maps = [row.to(device) for row in mesh_depth_cpu]
    del mesh_depth_cpu
    vertices_gpu = torch.tensor(surface.vertices, dtype=torch.float32, device=device)
    faces_gpu = torch.tensor(surface.faces, dtype=torch.int64, device=device)
    for attribute in ("triangles", "order", "cell_keys", "cell_start", "cell_count"):
        setattr(triangle_index, attribute, getattr(triangle_index, attribute).to(device))
    field = tsdf_field.SparseProjectiveTSDF(
        keys=field_on_cpu.keys.to(device), value=field_on_cpu.value.to(device),
        support_count=field_on_cpu.support_count.to(device), h=h, mu=mu,
    )
    del field_on_cpu

    # ========================================================= 12. exports
    _progress("[12/14] qualitative review exports")
    report["review_exports"] = _exports(
        output_root, model, cameras, views, mesh_depth_maps, depth_maps, region_of_pixel,
        surface, labels, field, vertices_gpu, faces_gpu, h, mu, device, total_model_count,
        report, occlusion, baseline,
    )

    # ==================================================== 13. review case table
    _progress("[13/14] machine-readable review case table")
    report["review_case_table"] = _case_table(
        output_root, views, cameras, depth_maps, representative_maps, region_of_pixel,
        field, triangle_index, labels, vertices_gpu, mesh_depth_maps, h, device, region_index,
    )

    # ======================================================= 14. NURBS handoff
    _progress("[14/14] conditional NURBS handoff")
    report["nurbs_handoff"] = _nurbs_handoff(surface, labels, region_index, model, h, device, report)

    # ---------------------------------------------------------------- synthetic
    if not arguments.skip_synthetic:
        _progress("synthetic semantic contracts S1-S7")
        report["synthetic_contracts"] = synthetic.run_all(device=device)

    report["runtime_seconds"] = time.time() - started
    report["memory"] = peak_memory()
    for key in ("_tsdf_vertices", "_fragmentation_points"):
        report.pop(key, None)
    for key in ("_positions", "_colours"):
        report.get("sdf_induced_occlusion_audit", {}).pop(key, None)
    for key in ("_vertices", "_faces"):
        report.get("baseline_ab", {}).pop(key, None)
    (output_root / _REPORT_NAME).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _progress(f"report written to {output_root / _REPORT_NAME} in {report['runtime_seconds'] / 60:.1f} min")


def _baseline_arm(
    arguments, report, model, cameras, views, depth_maps, representative_maps,
    region_of_pixel, h, device, triangle_index, evidence_distance, evidence_region, counted,
):
    """Directive section 14. Arm A is the historical topology/boundary-first
    visible-surface / NURBS path, replayed WITHOUT modification. Arm B is this
    batch's evidence-bounded TSDF surface. The raw renderer median surface point
    cloud is kept as an EVIDENCE REFERENCE, never as a third architecture."""

    import evidence_bounded_tsdf_stages as stages

    out: dict[str, Any] = {
        "arm_A": "HISTORICAL VISIBLE-SURFACE / NURBS PATH",
        "arm_B": "NEW EVIDENCE-BOUNDED TSDF VISIBLE SURFACE",
        "evidence_reference": "raw renderer median surface point cloud -- a reference, not a competing architecture",
        "fairness": (
            "the historical baseline is replayed at its own recorded capacity and chart construction; nothing "
            "about it was changed to make the comparison fairer, and no metric here rewards coverage that comes "
            "from unsupported filling or rewards fewer components per se"
        ),
    }
    if arguments.wl119_report.exists():
        historical = json.loads(arguments.wl119_report.read_text(encoding="utf-8"))
        metric_g = historical["corrected_uv_ab_metric_comparison"]["metric_g_geometric_error"]
        out["worklog_119_confirmed_artifact"] = {
            "fitted_chart_count": historical["accounting"]["fitted_chart_count"],
            "metric_g_arm_a_median_over_charts": metric_g["arm_a_median"],
            "metric_g_arm_a_p95_over_charts": metric_g["arm_a_p95"],
            "visible_topology": historical["wl107_109_replay_consistency_check"],
        }

    # ---------------- shared A/B query set: a deterministic stride over events
    stride = 37
    ab_points: list[torch.Tensor] = []
    ab_region: list[np.ndarray] = []
    for view_index, (camera, depth) in enumerate(views):
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)[::stride]
        if valid.numel() == 0:
            continue
        ab_points.append(tsdf_field.unproject_pixels(camera, valid, depth[valid]))
        ab_region.append(region_of_pixel[view_index][valid].detach().cpu().numpy())
    ab_query = torch.cat(ab_points, dim=0) if ab_points else torch.zeros((0, 3), device=device)
    ab_region_np = np.concatenate(ab_region) if ab_region else np.zeros((0,), dtype=np.int64)
    del ab_points, ab_region
    out["shared_ab_query_set"] = {
        "renderer_median_events": int(ab_query.shape[0]),
        "selection": f"deterministic stride {stride} over every view's valid median events; identical for both arms",
    }

    tsdf_ab = mesh_ops.nearest_surface_distance(ab_query, triangle_index, max_radius=3, chunk=2_000_000)
    tsdf_ab_np = tsdf_ab.detach().cpu().numpy()
    del tsdf_ab
    # Arm B's shared-query measurement is done; hand the GPU to the historical
    # replay's exact-KNN candidate graph, which needs the room.
    triangle_index.triangles = triangle_index.triangles.cpu()
    triangle_index.order = triangle_index.order.cpu()
    triangle_index.cell_keys = triangle_index.cell_keys.cpu()
    triangle_index.cell_start = triangle_index.cell_start.cpu()
    triangle_index.cell_count = triangle_index.cell_count.cpu()
    torch.cuda.empty_cache()
    out["arm_B_metrics"] = {
        "renderer_evidence_coverage_within_h": float((tsdf_ab_np <= h).mean()) if tsdf_ab_np.size else 0.0,
        "renderer_evidence_coverage_within_2h": float((tsdf_ab_np <= 2 * h).mean()) if tsdf_ab_np.size else 0.0,
        "geometric_residual_over_h": _quantiles(tsdf_ab_np[np.isfinite(tsdf_ab_np)] / h),
        "geometric_pieces": report["reconstruction"]["connected_components"],
        "surface_area": report["reconstruction"]["total_surface_area"],
        "ray_hit_coverage": report["raycast_self_consistency"]["ray_hit_coverage"],
        "absolute_depth_error_over_h": report["raycast_self_consistency"]["all_pixels"]["absolute_depth_error_over_h"],
    }

    if arguments.skip_baseline:
        out["arm_A_metrics"] = {"status": "SKIPPED by --skip-baseline"}
        out["_vertices"] = None
        return out

    # The historical exact-KNN graph needs room; park arm B's mesh tensors.
    tsdf_vertices_cpu = report["_tsdf_vertices"]
    torch.cuda.empty_cache()
    replay = stages.replay_historical_visible_nurbs(
        model, cameras, representative_maps, depth_maps, device=device,
        max_charts=int(arguments.baseline_max_charts), progress=lambda m: _progress("  " + m),
    )
    baseline_vertices = replay["vertices"]
    baseline_faces = replay["faces"]
    out["arm_A_replay"] = {
        "topology": replay["topology"],
        "fitted_chart_count": replay["fitted_chart_count"],
        "worklog_119_fitted_chart_count": WL119_FITTED_CHART_COUNT,
        "capacity": replay["capacity"],
        "per_chart_median_residual": _quantiles(replay["per_chart_median_residual"]),
        "worklog_119_metric_g_median_of_chart_medians": WL119_METRIC_G_MEDIAN,
        "sampled_vertices": int(baseline_vertices.shape[0]),
        "sampled_faces": int(baseline_faces.shape[0]),
        "sampling": (
            f"each fitted patch evaluated on a uniform {replay['patch_samples_per_side']}x"
            f"{replay['patch_samples_per_side']} UV grid -- a MEASUREMENT choice so the same new metrics can be "
            "computed on the historical arm; it changes nothing about the fit"
        ),
    }
    if baseline_vertices.shape[0] == 0:
        out["arm_A_metrics"] = {"status": "historical replay produced no geometry"}
        out["_vertices"] = None
        return out

    baseline_v_gpu = torch.tensor(baseline_vertices, dtype=torch.float32, device=device)
    baseline_f_gpu = torch.tensor(baseline_faces, dtype=torch.int64, device=device)
    baseline_index = mesh_ops.build_triangle_cell_index(baseline_v_gpu, baseline_f_gpu, h)
    baseline_ab = mesh_ops.nearest_surface_distance(ab_query, baseline_index, max_radius=3, chunk=2_000_000)
    baseline_ab_np = baseline_ab.detach().cpu().numpy()

    baseline_hit = 0
    baseline_total = 0
    baseline_error: list[np.ndarray] = []
    for view_index, (camera, depth) in enumerate(views):
        mesh_depth, _stats = mesh_ops.rasterize_mesh_depth(camera, baseline_v_gpu, baseline_f_gpu, face_chunk=4_000_000)
        flat = mesh_depth.reshape(-1)
        valid = depth > 0
        hit = valid & torch.isfinite(flat)
        baseline_total += int(valid.sum().item())
        baseline_hit += int(hit.sum().item())
        baseline_error.append((flat[hit] - depth[hit]).detach().cpu().numpy())
        del mesh_depth, flat
        if view_index % 40 == 0:
            _progress(f"  baseline raycast view {view_index}/{len(views)}")
    baseline_error_np = np.concatenate(baseline_error) if baseline_error else np.zeros((0,))

    # "unsupported bridge" -- IDENTICAL definition for both arms
    def _bridges(vertices_np: np.ndarray) -> dict[str, Any]:
        sample = torch.tensor(vertices_np[::97], dtype=torch.float32, device=device)
        if sample.numel() == 0:
            return {"sampled": 0}
        nearest = torch.full((sample.shape[0],), float("inf"), device=device)
        for camera, depth in views:
            width, height = int(camera.image_width), int(camera.image_height)
            projected = tsdf_field.project_world_points(
                sample, camera.world_view_transform, camera.full_proj_transform, width, height
            )
            index = projected.pixel_index.clamp(min=0)
            median = depth[index]
            usable = projected.relevant & (median > 0)
            event = tsdf_field.unproject_pixels(camera, index, median)
            distance = torch.linalg.norm(event - sample, dim=1)
            nearest = torch.minimum(nearest, torch.where(usable, distance, torch.full_like(distance, float("inf"))))
            del event, distance
        values = nearest.detach().cpu().numpy()
        return {
            "sampled": int(values.size),
            "distance_to_nearest_median_event_over_h": _quantiles(values / h),
            "surface_points_further_than_2h_from_any_event": int((values > 2 * h).sum()),
            "fraction_further_than_2h": float((values > 2 * h).mean()),
        }

    out["arm_A_metrics"] = {
        "renderer_evidence_coverage_within_h": float((baseline_ab_np <= h).mean()),
        "renderer_evidence_coverage_within_2h": float((baseline_ab_np <= 2 * h).mean()),
        "geometric_residual_over_h": _quantiles(baseline_ab_np[np.isfinite(baseline_ab_np)] / h),
        "geometric_pieces": replay["fitted_chart_count"],
        "surface_area": float(mesh_ops.triangle_areas(baseline_vertices, baseline_faces).sum()),
        "ray_hit_coverage": float(baseline_hit / baseline_total) if baseline_total else 0.0,
        "absolute_depth_error_over_h": _quantiles(np.abs(baseline_error_np) / h),
        "pieces_are_fitted_patches": "the historical arm's 'pieces' are fitted NURBS patches, not mesh components",
    }
    out["catastrophic_unsupported_bridges"] = {
        "definition": "surface points further than 2h from ANY renderer median event, sampled at stride 97",
        "arm_A": _bridges(baseline_vertices),
        "arm_B": _bridges(report["_tsdf_vertices"]),
    }
    out["by_region"] = {
        label: {
            "arm_A_coverage_within_h": float((baseline_ab_np[ab_region_np == index] <= h).mean())
            if int((ab_region_np == index).sum()) else 0.0,
            "arm_B_coverage_within_h": float((tsdf_ab_np[ab_region_np == index] <= h).mean())
            if int((ab_region_np == index).sum()) else 0.0,
            "events": int((ab_region_np == index).sum()),
        }
        for index, label in enumerate(REGION_LABELS)
    }
    out["interpretation_guard"] = (
        "more coverage is NOT success if it comes from unsupported filling, and fewer components is NOT success. "
        "Read arm_B's coverage together with catastrophic_unsupported_bridges and hallucination_audit."
    )
    out["_vertices"] = baseline_vertices
    out["_faces"] = baseline_faces
    del baseline_v_gpu, baseline_f_gpu, baseline_index
    torch.cuda.empty_cache()
    return out


ORIGINAL_2DGS_README = """# ORIGINAL_2DGS

## 색상 의미
- 학습된 2DGS 체크포인트를 **원래 학습된 SH 색상 그대로** 렌더링한 것이다. 진단용 색상 부호화가 전혀 없다.

## 이 이미지가 보여주는 것
이 배치의 모든 진단 view가 공유하는 **기준 장면**이다. 다른 모든 view와 **같은 카메라·같은 iteration·같은 해상도·같은 background**에서 렌더링했으므로, 재구성된 표면이 장면의 어느 구조에 대응하는지 대조하는 기준이 된다. 6개 대표 학습 시점을 `preview_png/`에 함께 담았다.

## 분석 및 평가
{region_summary}
"""

RENDERER_MEDIAN_README = """# RENDERER_MEDIAN_SURFACE_POINTS

## 색상 의미
- **밝은 청록** (`0.20, 0.85, 0.80`): renderer가 스스로 고른 canonical median surface event를 그 event의 world 좌표에 찍은 점
- **거의 검은 남색** (`0.07, 0.08, 0.10`): 학습된 2DGS 장면 전체(문맥용)

## 이 이미지가 보여주는 것
이번 후보가 **소비하는 유일한 증거**다. TSDF는 이 점들 외에 topology·KNN·region·boundary·chart를 전혀 쓰지 않는다. 이것은 경쟁 아키텍처가 아니라 **증거 기준선(evidence reference)**이므로 A/B의 한 축으로 해석하지 않는다. 결정론적 stride로 {points} 개를 뽑았다.

## 분석 및 평가
여기 찍힌 점들이 밀집한 영역일수록 field가 조밀한 authority를 받아 표면 재구성이 안정적이다. 전수 {total_events} 개 median event 중 **{within_h:.2%}가 추출 표면으로부터 h({h:.6f}) 이내**, {within_2h:.2%}가 2h 이내였고 3h 안에 표면이 전혀 없는 경우는 {no_surface:.2%}뿐이었다(`renderer_evidence_reproduction.all_events`). 이 점들이 성긴 부분은 `TSDF_SUPPORT_COUNT`에서 낮은 support로, `TSDF_FIELD_SLICES`에서 얇은 authority 띠로 이어져 나타난다.
"""

TSDF_SURFACE_README = """# NEW_TSDF_VISIBLE_SURFACE

## 색상 의미
- **초록** (`0.20, 0.85, 0.40`) 음영: evidence-bounded projective TSDF의 zero level-set에서 추출된 삼각형 메시 자체를 z-buffer로 렌더링한 것(정점 산포가 아니다). 음영은 각 삼각형의 기하 법선에 대한 단순 Lambertian 명암일 뿐 학습된 SH 색상이나 Gaussian covariance normal이 아니다
- **어두운 남색 배경**: 메시가 없는 픽셀(배경)

## 이 이미지가 보여주는 것
이번 배치의 **후보 Visible Surface 그 자체**다. cell 8개 corner가 모두 field authority를 가질 때만 삼각형을 만들었고, UNKNOWN voxel은 채우지 않았으며 hole filling·smoothing·watertight 강제는 하지 않았다. 따라서 열려 있고 끊겨 있는 것이 정상이다.

메시 전체는 `mesh/tsdf_visible_surface.ply` (정점 {vertices}, 삼각형 {faces})에 있고, 이 폴더의 `render.png`/`preview_png/`는 그 메시 **전체**를 `ORIGINAL_2DGS`와 같은 6개 대표 시점에서 실제 삼각형으로 렌더링한 것이다(점 산포가 아니다). 원본 2DGS 장면과 겹쳐 볼 마커 점군이 필요하면 `mesh/tsdf_visible_surface.ply`를 결정론적 stride로 subsample({markers} 개 기준)해서 쓸 수 있다.

## 분석 및 평가
연결 성분 {components:,}개, 총 표면적 {area:.1f}. renderer median event 전수의 {within_h:.2%}가 표면 h 이내(`renderer_evidence_reproduction`), 자기 카메라로 되쏜 raycast의 {ray_hit:.2%}가 표면에 맞았다(`raycast_self_consistency`). 연결 성분이 많은 것은 hole filling·smoothing·watertight 강제를 하지 않겠다는 계약의 직접적 결과이며 "더 매끄럽다"가 성공 기준이 아니다. `HISTORICAL_VISIBLE_NURBS_BASELINE`과 나란히 비교해서 봐야 한다.
"""

BASELINE_README = """# HISTORICAL_VISIBLE_NURBS_BASELINE

## 색상 의미
- **주황** (`0.95, 0.60, 0.15`) 음영: 역사적 topology/boundary-first 경로(WL107/109 topology → camera-observed chart → 기존 NURBS fitter)로 적합된 patch를 균일 UV 격자로 샘플링해 만든 삼각형 메시를 z-buffer로 렌더링한 것(점 산포가 아니다). 음영은 삼각형 기하 법선의 단순 Lambertian 명암
- **어두운 남색 배경**: 메시가 없는 픽셀(배경)

## 이 이미지가 보여주는 것
A/B의 **A arm**이다. fitter 용량(8x4, degree 2/2, correction round 2)과 chart 구성은 WL119 리포트에 기록된 값 그대로이며 이 배치에서 바꾸지 않았다. 비교를 유리하게 만들기 위한 수정은 없다. 메시 전체는 `mesh/historical_visible_nurbs_baseline.ply`에 있다.

## 분석 및 평가
재생 충실성: visible component {components:,} / singleton {singletons:,} / largest {largest:.5f}, fitted chart {charts:,}(`baseline_ab.arm_A_replay`). 공유 질의 집합에서 표면적 {area:.1f}, coverage(≤h) {coverage:.2%}, depth 오차 중앙값 {depth_err:.2f}h — arm B와 함께 봐야 이 값들이 좋은지 나쁜지 판단할 수 있다.
"""

RAYCAST_README = """# TSDF_RAYCAST_DEPTH

## 색상 의미
- **파랑→초록→빨강** 램프: 추출된 메시를 같은 학습 카메라로 되쏘았을 때의 **first mesh-hit depth** (가까움→멂)
- **어두운 자홍** (`0.28, 0.02, 0.24`): 그 픽셀에서 메시에 **맞은 것이 없음**(no hit). 자유 공간이 아니라 "재구성된 표면이 없음"이다

## 이 이미지가 보여주는 것
재구성된 Visible Surface가 자기가 만들어진 그 카메라들에서 실제로 어떻게 보이는지다. 광선은 candidate B가 쓰는 것과 **동일한 pixel-centre ray**이므로 다음 view의 depth 오차와 직접 비교 가능하다.

## 분석 및 평가
161개 카메라 전수에서 canonical median depth를 가진 {total_pixels} 픽셀 중 **{ray_hit:.2%}가 mesh hit**을 가졌다(`raycast_self_consistency`). hit이 있다고 depth가 정확하다는 뜻은 아니므로 실제 깊이 오차는 `MEDIAN_VS_TSDF_DEPTH_ERROR`와 항상 같이 봐야 한다.
"""

ERROR_README = """# MEDIAN_VS_TSDF_DEPTH_ERROR

## 색상 의미
- **빨강**: mesh first-hit이 renderer median frontier보다 **뒤**에 있음 (signed error > 0)
- **파랑**: mesh first-hit이 frontier보다 **앞**에 있음 (signed error < 0)
- **회색** (`0.35`): median event가 없거나 메시가 맞지 않아 비교 자체가 불가능한 픽셀
- 색 강도는 |오차| / (2h)로 정규화했다 (h = {h:.9f})

## 이 이미지가 보여주는 것
이 배치의 **1차 실측 지표** 중 하나다. 후보 표면이 renderer가 정의한 visible-surface frontier를 어디에서 얼마나 재현하고 어디에서 어긋나는지 픽셀 단위로 보여준다.

## 분석 및 평가
전체 픽셀의 |오차|/h median은 **{median_err:.4f}**, p95는 {p95_err:.2f}다(`raycast_self_consistency.all_pixels`). 큰 오차 픽셀의 다수가 mesh-in-front(파랑) 쪽이라면 재구성 실패가 아니라 다른 뷰의 표면이 이 뷰의 시선을 가로막는 것일 가능성이 높다 — `B_VS_TSDF_OCCLUSION_DISAGREEMENT`와 함께 봐야 한다. 영역별 수치는 `raycast_self_consistency.by_region`에 있다.
"""

SUPPORT_README = """# TSDF_SUPPORT_COUNT

## 색상 의미
- **빨강 → 청록** 램프(메시 전체를 z-buffer로 렌더링, 점 산포가 아니다): 각 정점 위치의 voxel이 **몇 개 뷰의 truncation band 안에 있었는지**(support_count). 빨강 = 1개 뷰, 청록 = {cap}개 이상. 램프 색만 표시하며 별도 음영은 넣지 않았다
- **어두운 남색 배경**: 메시가 없는 픽셀(배경)

## 이 이미지가 보여주는 것
support_count는 **진단 전용**이다. fusion 가중치는 모든 관측이 정확히 1이고 최소 뷰 수 규칙도 없으므로, support_count가 1이어도 표면은 정상적으로 만들어진다. 이 그림은 그 표면이 얼마나 얇은 증거 위에 서 있는지를 보여줄 뿐 삭제 기준이 아니다.

## 분석 및 평가
authoritative voxel의 **{frac_1:.2%}가 support_count = 1**이다(`field.fraction_support_count_1`). 평균 support는 {mean_support:.2f}로 대다수 표면은 여러 뷰가 겹쳐 지지하지만, 빨강 영역이 전체 표면의 상당 부분을 차지한다는 뜻이다. 이 빨강 영역이 실제로 얇은 구조인지 근거 부족 영역인지는 `TSDF_LOW_SUPPORT_SURFACE`와 depth 오차 view를 같이 봐야 한다.
"""

LOW_SUPPORT_README = """# TSDF_LOW_SUPPORT_SURFACE

## 색상 의미
- **자홍** (`0.95, 0.25, 0.75`): support_count <= {low}인 삼각형 — 단 하나의 뷰만이 그 voxel에 field authority를 준 곳
- **어두운 회색** (`0.18, 0.19, 0.22`): 같은 메시의 나머지(support 충분) 부분. 자홍 영역이 전체 형태에서 어디에 붙어 있는지(고립된 다리 vs 이어진 얇은 구조) 보기 위한 문맥이며 별도 장면이 아니다

## 이 이미지가 보여주는 것
**hallucination 후보 검토용**이다. directive에 따라 이 영역은 **삭제하지 않고 그대로 내보낸다**. 자홍이 실제 얇은 구조(잎, 다리)인지 아니면 근거 없는 다리(bridge)인지는 사람이 판단해야 하며, 정량 근거는 리포트의 `hallucination_audit`에 있다.

## 분석 및 평가
support≤{low} 표면이 삼각형 수로는 {tri_frac:.2%}, 면적으로는 {area_frac:.2%}를 차지한다(`hallucination_audit`). 계약 수준에서는 정상이다(삼각형 100%가 8-corner authoritative cell 출신). "실제 장면에서 미지지 조작이 material하지 않다"는 것은 이 배치가 적극 입증하지 않았으므로, 이 자홍 영역은 검토 우선순위로 남는다.
"""

DISAGREEMENT_README = """# B_VS_TSDF_OCCLUSION_DISAGREEMENT

## 색상 의미
- **빨강** (`0.92, 0.18, 0.18`): frozen Candidate B는 `OBSERVED`인데 새 메시로는 `MESH_OCCLUDED`
- **파랑** (`0.20, 0.55, 0.98`): B는 `OCCLUDED`인데 메시로는 가려지지 않음
- **초록** (`0.10, 0.85, 0.35`): 두 판정이 일치
- **회색** (`0.60, 0.60, 0.62`): B가 `UNRESOLVED`이거나 비교 불가
- **거의 검은 남색**: 학습된 2DGS 장면 전체(문맥용)

## 이 이미지가 보여주는 것
WL120 4,712 질의를 재사용해, **재구성된 메시를 실제 차폐 geometry로 썼을 때** Observed/Occluded 분할이 frozen Candidate B와 어디에서 갈라지는지 보여준다. 이 비교는 **진단**이며 Candidate B를 대체하지 않는다. depth epsilon은 어디에도 없다.

## 분석 및 평가
{confusion_summary}
"""

FRAGMENTATION_README = """# WL121_FRAGMENTATION_CONTEXT_OVERLAY

## 색상 의미
- **노랑** (`0.98, 0.85, 0.20`): WL121이 뽑은 300개 true-fragmentation context의 endpoint A/B
- **주황** (`0.95, 0.45, 0.10`): 두 endpoint의 midpoint
- **거의 검은 남색**: 학습된 2DGS 장면 전체(문맥용)

## 이 이미지가 보여주는 것
역사적 topology가 **끊어 놓은** 자리에서 새 TSDF 표면이 어떻게 행동하는지를 보기 위한 overlay다. 같은 메시 component에 있다고 해서 물리적 연속성이 옳다는 뜻이 **아니고**, midpoint에 표면이 있다고 해서 역사적 분할이 틀렸다는 뜻도 **아니다**. 정량치는 리포트의 `historical_topology_attribution`에 있다.

## 분석 및 평가
{fragmentation_summary}
"""

SLICE_README = """# TSDF_FIELD_SLICES

## 색상 의미
- **파랑 계열**: field authority가 있고 phi > 0 — renderer median surface의 **카메라 쪽**
- **주황/빨강 계열**: field authority가 있고 phi < 0 — frontier **뒤쪽**
- **어두운 보라** (`0.10, 0.02, 0.16`): **UNKNOWN** — 어떤 뷰도 이 voxel을 truncation band 안에 두지 못했다. **자유 공간이 아니다**
- **흰색**: zero crossing (부호가 바뀌는 authoritative 인접쌍) = 추출된 level-set의 위치

## 이 이미지가 보여주는 것
sparse authority contract를 눈으로 확인하는 그림이다. UNKNOWN이 파랑/주황과 **시각적으로 명확히 구분**되며 자유 공간처럼 그려지지 않는다는 점이 핵심이다. 3개 slice는 table 구조 / patio / hedge(배경)를 각각 지난다.

## 분석 및 평가
값 있는 영역(파랑/주황/흰색)이 표면을 감싼 얇은 띠뿐이고 화면 대부분이 UNKNOWN이라는 점이 field 정의의 핵심 성질을 보여준다 — μ=3h({mu:.6f})의 truncation band 밖은 field가 값을 갖지 않는다. 세 slice의 띠 두께 차이는 그 영역의 support 밀도·표면 곡률 차이를 반영하며, 정량 근거는 `field.support_count_distribution`과 `renderer_evidence_reproduction.by_region`에 있다.
"""


def _exports(
    output_root, model, cameras, views, mesh_depth_maps, depth_maps, region_of_pixel,
    surface, labels, field, vertices_gpu, faces_gpu, h, mu, device, total_model_count,
    report, occlusion, baseline,
):
    """Directive section 18. Ten required exports plus the field slices, every
    folder with its own Korean README, and user-friendly PNGs in preview_png/."""

    import evidence_bounded_tsdf_stages as stages
    from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    paths: dict[str, Any] = {}
    preview_root = output_root / "preview_png"
    preview_root.mkdir(parents=True, exist_ok=True)
    rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
    preview_indices = list(range(0, len(cameras), max(1, len(cameras) // 6)))[:6]
    preview_cameras = [cameras[i] for i in preview_indices]
    marker_radius = h * 1.6

    def render_marker_view(name: str, positions: torch.Tensor, colours: torch.Tensor, body: str) -> dict[str, Any]:
        scene_xyz = model.get_xyz.detach()
        scene_rgb = torch.tensor(_SCENE_RGB, device=device).reshape(1, 3).expand(scene_xyz.shape[0], 3)
        count = int(positions.shape[0])
        scaling = torch.cat(
            [model._scaling.detach(), torch.full((count, 2), float(np.log(marker_radius)), device=device)], dim=0
        )
        rotation = torch.zeros((count, 4), dtype=torch.float32, device=device)
        rotation[:, 0] = 1.0
        rotation = torch.cat([model._rotation.detach(), rotation], dim=0)
        opacity = torch.cat([model._opacity.detach().reshape(-1), torch.full((count,), 4.0, device=device)], dim=0)
        xyz = torch.cat([scene_xyz, positions], dim=0)
        rgb = torch.cat([scene_rgb, colours], dim=0)
        folder = output_root / name
        ply_path = folder / _ITERATION_DIR / "point_cloud.ply"
        written = write_surfel_ply(ply_path, xyz, _rgb_to_f_dc(rgb), opacity, scaling, rotation)
        review = TorchGaussianSurfelModel(sh_degree=0, device=str(device))
        review.initialize(positions=xyz, colors=rgb, opacities=torch.sigmoid(opacity).reshape(-1, 1),
                          scales=torch.exp(scaling), rotations=rotation)
        review.active_sh_degree = 0
        with torch.no_grad():
            package = rasterizer.render(preview_cameras[0], review)
        write_ppm(folder / "render.ppm", package["render"])
        stages.write_png(preview_root / f"{name}__{preview_cameras[0].image_name}.png",
                         package["render"].permute(1, 2, 0).detach().cpu().numpy())
        del package
        for camera in preview_cameras[1:]:
            with torch.no_grad():
                package = rasterizer.render(camera, review)
            stages.write_png(preview_root / f"{name}__{camera.image_name}.png",
                             package["render"].permute(1, 2, 0).detach().cpu().numpy())
            del package
        write_view_readme(folder, body, total_model_count)
        del review
        torch.cuda.empty_cache()
        return {"point_cloud_ply": str(ply_path), "gaussian_count": written, "marker_points": count}

    def render_mesh_view(
        name: str, mesh_vertices: torch.Tensor, mesh_faces: torch.Tensor, vertex_colours: torch.Tensor, body: str,
        *, shaded: bool = True,
    ) -> dict[str, Any]:
        """Actual triangle-mesh rendering (z-buffered), for views whose subject
        IS a surface -- a scatter of marker points at surface vertices does not
        read as a shape, especially where the surface is thin or the marker
        stride is coarse. Uses `mesh_ops.rasterize_mesh_shaded`, which reuses
        the SAME pixel-centre ray/triangle test as the raycast self-consistency
        stage, so the silhouette is directly comparable.

        `shaded=False` (used for support-count and low-support colour-coded
        views) disables the Lambertian brightness term so the data-encoding
        colour ramp is never confused with lighting."""

        folder = output_root / name
        for camera in preview_cameras:
            image = mesh_ops.rasterize_mesh_shaded(
                camera, mesh_vertices, mesh_faces, vertex_colours, shaded=shaded
            )
            array = image.detach().cpu().numpy()
            if camera is preview_cameras[0]:
                stages.write_png(folder / "render.png", array)
            stages.write_png(preview_root / f"{name}__{camera.image_name}.png", array)
        write_view_readme(folder, body, total_model_count)
        return {"mesh_vertices": int(mesh_vertices.shape[0]), "mesh_faces": int(mesh_faces.shape[0])}

    # A -------------------------------------------------------- ORIGINAL_2DGS
    folder = output_root / "ORIGINAL_2DGS"
    ply_path = folder / _ITERATION_DIR / "point_cloud.ply"
    paths["A_ORIGINAL_2DGS"] = {
        "point_cloud_ply": str(ply_path),
        "gaussian_count": write_surfel_ply(
            ply_path, model.get_xyz.detach(), model._features_dc.detach()[:, 0, :],
            model._opacity.detach().reshape(-1), model._scaling.detach(), model._rotation.detach(),
        ),
    }
    for camera in preview_cameras:
        with torch.no_grad():
            package = rasterizer.render(camera, model)
        if camera is preview_cameras[0]:
            write_ppm(folder / "render.ppm", package["render"])
        stages.write_png(preview_root / f"A_ORIGINAL_2DGS__{camera.image_name}.png",
                         package["render"].permute(1, 2, 0).detach().cpu().numpy())
        del package
    newline = "\n"
    region_lines = newline.join(
        f"- **{label}**: coverage<=h {v['fraction_within_h']*100:.2f}%, raycast |error|/h median {rc:.3f}"
        for label, v, rc in zip(
            REGION_LABELS,
            report.get("renderer_evidence_reproduction", {}).get("by_region", {}).values(),
            [
                r.get("absolute_depth_error_over_h", {}).get("median", float("nan"))
                for r in report.get("raycast_self_consistency", {}).get("by_region", {}).values()
            ] or [float("nan")] * len(REGION_LABELS),
        )
    ) or "영역별 수치는 아직 계산되지 않았다."
    write_view_readme(
        folder, ORIGINAL_2DGS_README.format(region_summary=region_lines), total_model_count
    )

    # B ------------------------------------- RENDERER_MEDIAN_SURFACE_POINTS
    event_points = []
    for view_index, (camera, depth) in enumerate(views):
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)[::2003]
        if valid.numel():
            event_points.append(tsdf_field.unproject_pixels(camera, valid, depth[valid]))
    event_points_t = torch.cat(event_points, dim=0) if event_points else torch.zeros((0, 3), device=device)
    paths["B_RENDERER_MEDIAN_SURFACE_POINTS"] = render_marker_view(
        "RENDERER_MEDIAN_SURFACE_POINTS", event_points_t,
        torch.tensor([[0.20, 0.85, 0.80]], device=device).expand(event_points_t.shape[0], 3).contiguous(),
        RENDERER_MEDIAN_README.format(
            points=f"{int(event_points_t.shape[0]):,}",
            total_events=f"{report['renderer_evidence_reproduction']['all_events']['events']:,}",
            within_h=report["renderer_evidence_reproduction"]["all_events"]["fraction_within_h"],
            within_2h=report["renderer_evidence_reproduction"]["all_events"]["fraction_within_2h"],
            no_surface=report["renderer_evidence_reproduction"]["all_events"]["fraction_no_local_extracted_surface_beyond_3h"],
            h=h,
        ),
    )
    del event_points, event_points_t

    # C ---------------------------------------- NEW_TSDF_VISIBLE_SURFACE (+PLY)
    mesh_dir = output_root / "mesh"
    mesh_ply = mesh_dir / "tsdf_visible_surface.ply"
    mesh_ops.write_mesh_ply(mesh_ply, surface.vertices, surface.faces)
    support_cap = float(np.percentile(surface.vertex_support_count, 95)) if surface.vertex_support_count.size else 1.0
    support_colors = (stages.support_to_rgb(surface.vertex_support_count, support_cap) * 255).astype(np.uint8)
    support_ply = mesh_dir / "tsdf_visible_surface_support_count.ply"
    mesh_ops.write_mesh_ply(support_ply, surface.vertices, surface.faces, colors=support_colors)
    low = surface.vertex_support_count <= LOW_SUPPORT_COUNT
    low_ply = mesh_dir / "tsdf_low_support_candidates.ply"
    mesh_ops.write_point_ply(
        low_ply, surface.vertices[low],
        np.tile(np.array([[242, 64, 191]], dtype=np.uint8), (int(low.sum()), 1)),
    )
    (mesh_dir / "README.md").write_text(
        "# mesh\n\n"
        "- `tsdf_visible_surface.ply` — 추출된 evidence-bounded TSDF Visible Surface 메시 (binary PLY, "
        f"정점 {int(surface.vertices.shape[0]):,} / 삼각형 {int(surface.faces.shape[0]):,}). "
        "hole filling·repair·smoothing·decimation을 전혀 하지 않았으므로 열려 있고 끊겨 있다.\n"
        "- `tsdf_visible_surface_support_count.ply` — 같은 메시에 정점별 support_count를 색으로 입힌 것 "
        f"(빨강 = 1 뷰, 청록 = {support_cap:.0f} 뷰 이상). 색만 다르고 geometry는 동일하다.\n"
        f"- `tsdf_low_support_candidates.ply` — support_count <= {LOW_SUPPORT_COUNT}인 정점 {int(low.sum()):,}개만 "
        "뽑은 점군. **hallucination 후보 검토용이며 삭제하지 않고 그대로 보존한다.**\n",
        encoding="utf-8",
    )
    stride = max(1, int(surface.vertices.shape[0]) // 400_000)
    marker_vertices = torch.tensor(surface.vertices[::stride], dtype=torch.float32, device=device)
    tsdf_vertex_colours = torch.tensor([0.20, 0.85, 0.40], device=device).reshape(1, 3).expand(
        vertices_gpu.shape[0], 3
    ).contiguous()
    paths["C_NEW_TSDF_VISIBLE_SURFACE"] = render_mesh_view(
        "NEW_TSDF_VISIBLE_SURFACE", vertices_gpu, faces_gpu, tsdf_vertex_colours,
        TSDF_SURFACE_README.format(
            vertices=f"{int(surface.vertices.shape[0]):,}", faces=f"{int(surface.faces.shape[0]):,}",
            markers=f"{int(marker_vertices.shape[0]):,}",
            components=report["reconstruction"]["connected_components"],
            area=report["reconstruction"]["total_surface_area"],
            within_h=report["renderer_evidence_reproduction"]["all_events"]["fraction_within_h"],
            ray_hit=report["raycast_self_consistency"]["ray_hit_coverage"],
        ),
    )
    paths["mesh_ply"] = {
        "surface": str(mesh_ply), "support_coloured": str(support_ply), "low_support": str(low_ply),
        "low_support_vertices": int(low.sum()),
    }

    # D -------------------------------- HISTORICAL_VISIBLE_NURBS_BASELINE
    baseline_vertices = baseline.get("_vertices")
    if baseline_vertices is not None and baseline_vertices.shape[0]:
        baseline_v_gpu = torch.tensor(baseline_vertices, dtype=torch.float32, device=device)
        baseline_f_gpu = torch.tensor(baseline["_faces"], dtype=torch.int64, device=device)
        baseline_colours = torch.tensor([0.95, 0.60, 0.15], device=device).reshape(1, 3).expand(
            baseline_v_gpu.shape[0], 3
        ).contiguous()
        paths["D_HISTORICAL_VISIBLE_NURBS_BASELINE"] = render_mesh_view(
            "HISTORICAL_VISIBLE_NURBS_BASELINE", baseline_v_gpu, baseline_f_gpu, baseline_colours,
            BASELINE_README.format(
                components=baseline.get("arm_A_replay", {}).get("topology", {}).get("visible_component_count", 0),
                singletons=baseline.get("arm_A_replay", {}).get("topology", {}).get("singleton_surfel_count", 0),
                largest=baseline.get("arm_A_replay", {}).get("topology", {}).get("largest_component_surfel_fraction", float("nan")),
                charts=baseline.get("arm_A_replay", {}).get("fitted_chart_count", 0),
                area=baseline.get("arm_A_metrics", {}).get("surface_area", float("nan")),
                coverage=baseline.get("arm_A_metrics", {}).get("renderer_evidence_coverage_within_h", float("nan")),
                depth_err=baseline.get("arm_A_metrics", {}).get("absolute_depth_error_over_h", {}).get("median", float("nan")),
            ),
        )
        mesh_ops.write_mesh_ply(
            mesh_dir / "historical_visible_nurbs_baseline.ply", baseline_vertices, baseline["_faces"]
        )
        del baseline_v_gpu, baseline_f_gpu, baseline_colours
        torch.cuda.empty_cache()
    else:
        paths["D_HISTORICAL_VISIBLE_NURBS_BASELINE"] = {"status": "baseline arm not replayed in this run"}

    # E / F -------------------------- TSDF_RAYCAST_DEPTH, depth error images
    raycast_dir = output_root / "TSDF_RAYCAST_DEPTH"
    error_dir = output_root / "MEDIAN_VS_TSDF_DEPTH_ERROR"
    raycast_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    for index in preview_indices:
        camera = cameras[index]
        height, width = int(camera.image_height), int(camera.image_width)
        mesh_depth = mesh_depth_maps[index].reshape(height, width).detach().cpu().numpy()
        renderer_depth = depth_maps[index].reshape(height, width).detach().cpu().numpy()
        finite = np.isfinite(mesh_depth)
        low_bound = float(np.percentile(mesh_depth[finite], 2)) if finite.any() else 0.0
        high_bound = float(np.percentile(mesh_depth[finite], 98)) if finite.any() else 1.0
        stages.write_png(raycast_dir / f"{camera.image_name}.png", stages.depth_to_rgb(mesh_depth, low_bound, high_bound))
        stages.write_png(preview_root / f"E_TSDF_RAYCAST_DEPTH__{camera.image_name}.png",
                         stages.depth_to_rgb(mesh_depth, low_bound, high_bound))
        valid = finite & (renderer_depth > 0)
        error = np.where(valid, mesh_depth - renderer_depth, 0.0)
        image = stages.signed_error_to_rgb(error, valid, 2.0 * h)
        stages.write_png(error_dir / f"{camera.image_name}.png", image)
        stages.write_png(preview_root / f"F_MEDIAN_VS_TSDF_DEPTH_ERROR__{camera.image_name}.png", image)
    write_view_readme(
        raycast_dir,
        RAYCAST_README.format(
            total_pixels=f"{report['raycast_self_consistency']['pixels_with_canonical_median_depth']:,}",
            ray_hit=report["raycast_self_consistency"]["ray_hit_coverage"],
        ),
        total_model_count,
    )
    write_view_readme(
        error_dir,
        ERROR_README.format(
            h=h,
            median_err=report["raycast_self_consistency"]["all_pixels"]["absolute_depth_error_over_h"]["median"],
            p95_err=report["raycast_self_consistency"]["all_pixels"]["absolute_depth_error_over_h"]["p95"],
        ),
        total_model_count,
    )
    paths["E_TSDF_RAYCAST_DEPTH"] = {"views": [cameras[i].image_name for i in preview_indices]}
    paths["F_MEDIAN_VS_TSDF_DEPTH_ERROR"] = {"views": [cameras[i].image_name for i in preview_indices]}

    # G / H ------------------------------ support count, low-support surface
    # Full-mesh per-vertex colour (not the marker stride subsample) so the
    # actual surface shape is legible, not just a scatter of sample points.
    support_colours_full = torch.tensor(
        stages.support_to_rgb(surface.vertex_support_count, support_cap), dtype=torch.float32, device=device
    )
    paths["G_TSDF_SUPPORT_COUNT"] = render_mesh_view(
        "TSDF_SUPPORT_COUNT", vertices_gpu, faces_gpu, support_colours_full,
        SUPPORT_README.format(
            cap=f"{support_cap:.0f}",
            frac_1=report["field"]["fraction_support_count_1"],
            mean_support=report["field"]["support_count_distribution"]["mean"],
        ),
        shaded=False,
    )
    del support_colours_full

    # Low-support view: colour the WHOLE mesh, but only the low-support
    # triangles get the flagged colour -- the rest is dimmed context so the
    # flagged region's shape (thin structure vs. isolated bridge) is visible
    # rather than a disconnected point scatter.
    low_vertex_mask = torch.tensor(low, device=device)
    low_colours = torch.where(
        low_vertex_mask.unsqueeze(1),
        torch.tensor([0.95, 0.25, 0.75], device=device).reshape(1, 3),
        torch.tensor([0.18, 0.19, 0.22], device=device).reshape(1, 3),
    )
    paths["H_TSDF_LOW_SUPPORT_SURFACE"] = render_mesh_view(
        "TSDF_LOW_SUPPORT_SURFACE", vertices_gpu, faces_gpu, low_colours,
        LOW_SUPPORT_README.format(
            low=LOW_SUPPORT_COUNT,
            tri_frac=report["hallucination_audit"].get("fraction_low_support", float("nan")),
            area_frac=(
                report["hallucination_audit"].get("low_support_sampled_points", 0)
                / max(report["hallucination_audit"].get("sampled_mesh_points", 1), 1)
            ),
        ),
        shaded=False,
    )
    del low_vertex_mask, low_colours, marker_vertices

    # I ------------------------------ B_VS_TSDF_OCCLUSION_DISAGREEMENT
    wl120 = occlusion.get("worklog_120_original_4712", {})
    disagreement = occlusion.get("worklog_120_disagreement_attribution")
    if disagreement is not None and "_positions" in occlusion:
        positions = occlusion["_positions"]
        colours = occlusion["_colours"]
        wl120_confusion = occlusion.get("worklog_120_original_4712", {}).get("global_confusion", {})
        confusion_lines = (
            f"B가 OCCLUDED로 판정한 {wl120_confusion.get('B_OCCLUDED_and_mesh_OCCLUDED', 0) + wl120_confusion.get('B_OCCLUDED_and_mesh_unobstructed', 0)}건 중 "
            f"{wl120_confusion.get('B_OCCLUDED_and_mesh_OCCLUDED', 0)}건({wl120_confusion.get('B_OCCLUDED_and_mesh_OCCLUDED', 0) / max(wl120_confusion.get('B_OCCLUDED_and_mesh_OCCLUDED', 0) + wl120_confusion.get('B_OCCLUDED_and_mesh_unobstructed', 0), 1) * 100:.2f}%)를 메시도 독립적으로 OCCLUDED로 본다. "
            f"반대 방향(B=OBSERVED인데 mesh=OCCLUDED)은 {wl120_confusion.get('B_OBSERVED_and_mesh_OCCLUDED', 0)}건이며, "
            "그 대부분은 `sdf_induced_occlusion_audit.worklog_120_disagreement_attribution`에서 3h 이내 실재 표면으로 설명된다."
        )
        paths["I_B_VS_TSDF_OCCLUSION_DISAGREEMENT"] = render_marker_view(
            "B_VS_TSDF_OCCLUSION_DISAGREEMENT", positions, colours,
            DISAGREEMENT_README.format(confusion_summary=confusion_lines),
        )
    else:
        paths["I_B_VS_TSDF_OCCLUSION_DISAGREEMENT"] = {"status": "worklog 120 bank unavailable"}

    # J --------------------- WL121_FRAGMENTATION_CONTEXT_OVERLAY
    if "_fragmentation_points" in report:
        points, colours = report.pop("_fragmentation_points")
        frag = report.get("historical_topology_attribution", {})
        frag_summary = (
            f"{frag.get('contexts', 0)}개 context 중 {frag.get('endpoints_on_the_same_extracted_mesh_component', 0)}쌍의 "
            f"두 endpoint가 같은 mesh component 위에 놓이고, {frag.get('midpoint_has_no_surface_within_3h', 0)}쌍은 "
            "midpoint 3h 이내에 표면이 없다(`historical_topology_attribution`). "
            "같은 component가 곧 물리적으로 옳다는 뜻은 아니다."
        )
        paths["J_WL121_FRAGMENTATION_CONTEXT_OVERLAY"] = render_marker_view(
            "WL121_FRAGMENTATION_CONTEXT_OVERLAY", points, colours,
            FRAGMENTATION_README.format(fragmentation_summary=frag_summary),
        )
    else:
        paths["J_WL121_FRAGMENTATION_CONTEXT_OVERLAY"] = {"status": "worklog 121 bank unavailable"}

    # -------------------------------------------------------- field slices
    slice_dir = output_root / "TSDF_FIELD_SLICES"
    slice_dir.mkdir(parents=True, exist_ok=True)
    slice_stats = {}
    anchor_camera = cameras[len(cameras) // 2]
    anchor_depth = depth_maps[len(cameras) // 2]
    anchor_region = region_of_pixel[len(cameras) // 2]
    for label, region_id in (("table_structure", 0), ("patio", 3), ("hedge_background", 4)):
        rows = torch.nonzero((anchor_region == region_id) & (anchor_depth > 0), as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            slice_stats[label] = {"status": "no anchor pixel for this region in the anchor view"}
            continue
        pick = rows[rows.numel() // 2 : rows.numel() // 2 + 1]
        centre = tsdf_field.unproject_pixels(anchor_camera, pick, anchor_depth[pick]).reshape(3)
        value, authority, support_image, stats = stages.build_slice(
            field, centre, axis=1, half_extent=SLICE_HALF_EXTENT_VOXELS
        )
        image = stages.slice_to_rgb(value, authority)
        stages.write_png(slice_dir / f"{label}.png", image)
        stages.write_png(preview_root / f"SLICE_{label}.png", image)
        stats["world_centre"] = [float(v) for v in centre.tolist()]
        stats["region"] = REGION_LABELS[region_id]
        slice_stats[label] = stats
    write_view_readme(slice_dir, SLICE_README.format(mu=mu), total_model_count)
    paths["TSDF_FIELD_SLICES"] = slice_stats

    (preview_root / "README.md").write_text(
        "# preview_png\n\n"
        "이 배치의 모든 view를 사람이 바로 볼 수 있는 PNG로 모아둔 폴더다. 파일 이름은 "
        "`<VIEW>__<카메라이름>.png` 형식이며, view별 색상 의미는 각 view 폴더의 `README.md`에 있다. "
        "`SLICE_*.png`는 world-space field slice로, **UNKNOWN(어두운 보라)을 자유 공간처럼 그리지 않는다.**\n\n"
        f"대표 시점 {len(preview_cameras)}개: " + ", ".join(c.image_name for c in preview_cameras) + "\n",
        encoding="utf-8",
    )
    return paths


def _case_table(
    output_root, views, cameras, depth_maps, representative_maps, region_of_pixel,
    field, triangle_index, labels, vertices_gpu, mesh_depth_maps, h, device, region_index,
):
    """Directive section 19. Concrete, machine-readable cases with provenance."""

    rows: list[dict[str, Any]] = []
    anchor_view = len(cameras) // 2
    for region_id, label in enumerate(REGION_LABELS):
        picked = 0
        for view_index in range(anchor_view, anchor_view + len(cameras)):
            if picked >= REVIEW_CASES_PER_REGION:
                break
            index = view_index % len(cameras)
            camera = cameras[index]
            depth = depth_maps[index]
            candidates = torch.nonzero(
                (region_of_pixel[index] == region_id) & (depth > 0), as_tuple=False
            ).reshape(-1)
            if candidates.numel() == 0:
                continue
            step = max(1, int(candidates.numel()) // (REVIEW_CASES_PER_REGION + 1))
            for slot in range(0, int(candidates.numel()), step):
                if picked >= REVIEW_CASES_PER_REGION:
                    break
                pixel = candidates[slot : slot + 1]
                world = tsdf_field.unproject_pixels(camera, pixel, depth[pixel])
                keys, _ = tsdf_field.encode_keys(tsdf_field.voxel_index_of(world, h))
                value, support, found = field.lookup(keys)
                distance = mesh_ops.nearest_surface_distance(world, triangle_index, max_radius=3)
                mesh_hit = mesh_depth_maps[index][pixel]
                rows.append({
                    "region": label,
                    "world_position": [float(v) for v in world.reshape(3).tolist()],
                    "source_view_index": int(index),
                    "source_view_name": str(camera.image_name),
                    "source_pixel": int(pixel.item()),
                    "representative_id": int(representative_maps[index][pixel].item()),
                    "renderer_median_depth": float(depth[pixel].item()),
                    "sdf_value": float(value.item()) if bool(found.item()) else None,
                    "sdf_authority": bool(found.item()),
                    "support_count": int(support.item()),
                    "nearest_extracted_surface_distance": float(distance.item()),
                    "nearest_extracted_surface_distance_over_h": float(distance.item()) / h,
                    "mesh_first_hit_depth_in_source_view": float(mesh_hit.item()) if torch.isfinite(mesh_hit).item() else None,
                    "mesh_minus_renderer_depth": (
                        float(mesh_hit.item() - depth[pixel].item()) if torch.isfinite(mesh_hit).item() else None
                    ),
                    "case_kind": "region_sample",
                })
                picked += 1
    path = output_root / "review_case_table.json"
    path.write_text(json.dumps({
        "cases": rows,
        "cases_per_region_target": REVIEW_CASES_PER_REGION,
        "note": (
            "historical component ids and Candidate B states for the same coordinates are in the report's "
            "sdf_induced_occlusion_audit and historical_topology_attribution sections; appearance alone is "
            "never the evidence"
        ),
    }, indent=2, default=str), encoding="utf-8")
    return {"path": str(path), "cases": len(rows)}


def _nurbs_handoff(surface, labels, region_index, model, h, device, report):
    """Directive section 20. Minimal compatibility experiment ONLY, and only if
    the unsupported-gap contracts passed and the mesh has nontrivial coverage."""

    synthetic_ok = True
    gate: dict[str, Any] = {}
    coverage = report.get("renderer_evidence_reproduction", {}).get("all_events", {}).get("fraction_within_h", 0.0)
    ray_hit = report.get("raycast_self_consistency", {}).get("ray_hit_coverage", 0.0)
    gate["evidence_coverage_within_h"] = coverage
    gate["ray_hit_coverage"] = ray_hit
    gate["nontrivial_coverage"] = bool(coverage > 0.0 and ray_hit > 0.0)
    if surface.faces.shape[0] == 0 or not gate["nontrivial_coverage"]:
        return {"status": "SKIPPED -- gate not met", "gate": gate}

    from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points

    positions = model.get_xyz.detach()
    vertices = torch.tensor(surface.vertices, dtype=torch.float32, device=device)
    results = []
    # Spatial ROI crops ONLY: a small box around a deterministic anchor of each
    # region. No boundary loop, region id or chart eligibility is consulted.
    for region_id, label in ((0, "table_top"), (1, "table_side_curved"), (3, "patio"), (2, "table_legs")):
        member = torch.nonzero(region_index == region_id, as_tuple=False).reshape(-1)
        if member.numel() == 0:
            results.append({"crop": label, "status": "no surfel carries this region label"})
            continue
        anchor = positions[member[member.numel() // 2]]
        half = 12.0 * h
        inside = ((vertices - anchor.reshape(1, 3)).abs() <= half).all(dim=1)
        points = vertices[inside]
        if int(points.shape[0]) < 32:
            results.append({
                "crop": label, "status": "crop holds fewer than 32 zero-surface points",
                "points": int(points.shape[0]),
                "crop_half_extent_over_h": 12.0,
            })
            continue
        if int(points.shape[0]) > 20_000:
            points = points[:: int(points.shape[0]) // 20_000 + 1]
        with torch.no_grad():
            initial_uv = pca_parameterize_points(points)
            surface_fit, uv = fit_torch_visible_surface_lsq(
                points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
                initial_uv=initial_uv, correction_rounds=2, projection_iterations=4,
            )
            residual = (surface_fit.evaluate(uv) - points).norm(dim=-1)
            normals = surface_fit.normals(uv)
        results.append({
            "crop": label,
            "status": "fitted",
            "crop_half_extent_over_h": 12.0,
            "points": int(points.shape[0]),
            "parameterization": "existing pca_parameterize_points -- the simplest existing one, no boundary architecture",
            "residual_median": float(residual.median().item()),
            "residual_p95": float(torch.sort(residual).values[int(0.95 * (residual.numel() - 1))].item()),
            "residual_max": float(residual.max().item()),
            "residual_median_over_h": float(residual.median().item()) / h,
            "normal_finite_fraction": float(torch.isfinite(normals).all(dim=-1).to(torch.float64).mean().item()),
        })
    fitted = [r for r in results if r.get("status") == "fitted"]
    return {
        "status": "RUN",
        "gate": gate,
        "crops": results,
        "question": "Can the reconstructed implicit surface serve as geometric evidence for downstream NURBS fitting?",
        "separation": (
            "a poor fit here is a statement about the EXISTING NURBS parameterization on SDF crops, never proof "
            "that the SDF Visible Surface itself failed -- the two verdicts stay separate"
        ),
        "crops_fitted": len(fitted),
    }

def _nearest_vertex(points: torch.Tensor, vertices: torch.Tensor, *, chunk: int = 4_000_000) -> torch.Tensor:
    """Index of the nearest mesh vertex, streamed over the vertex array so a
    35M-vertex mesh never needs a dense distance matrix."""

    best_distance = torch.full((points.shape[0],), float("inf"), device=points.device)
    best_index = torch.zeros((points.shape[0],), dtype=torch.int64, device=points.device)
    for start in range(0, int(vertices.shape[0]), chunk):
        block = vertices[start : start + chunk]
        distance = torch.cdist(points, block)
        local_best, local_index = distance.min(dim=1)
        improved = local_best < best_distance
        best_distance = torch.where(improved, local_best, best_distance)
        best_index = torch.where(improved, local_index + start, best_index)
        del distance
    return best_index


def _nearest_mesh_component(
    world_a: torch.Tensor, world_b: torch.Tensor, vertices: torch.Tensor, labels: np.ndarray, device: str,
    *, chunk: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    if vertices.shape[0] == 0:
        empty = np.full((world_a.shape[0],), -1, dtype=np.int64)
        return empty, empty.copy()
    label_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
    out = []
    for points in (world_a, world_b):
        out.append(label_tensor[_nearest_vertex(points, vertices, chunk=chunk)].detach().cpu().numpy())
    return out[0], out[1]


if __name__ == "__main__":
    main()
