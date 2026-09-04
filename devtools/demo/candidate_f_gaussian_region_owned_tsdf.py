"""Worklog 154 Candidate F replay on the real 2DGS checkpoint.

Pipeline
--------

    2DGS surfels -> intrinsic Gaussian normal -> region-coherent Gaussian IDs
    renderer median-depth TSDF -> authoritative zero-surface samples
    nearest Gaussian -> existing Gaussian region ID
    native TSDF cell adjacency -> observed support components
    support-derived boundary chart -> WL139 NURBS representative

The script is deliberately a candidate replay.  It does not alter the
production visible-surface constructor, does not call the legacy mesh
extractor, and never uses historical event 1527 as a blacklist or selection
signal.  ``torch.load(weights_only=True)`` keeps the read-only checkpoint
replay free of arbitrary checkpoint-code execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.surface.torch_gaussian_region_owned_tsdf import (
    ABSTAIN_REPRESENTATIVE,
    MEMBERSHIP_AMBIGUOUS,
    MEMBERSHIP_ATTACHED,
    MEMBERSHIP_CORE,
    MEMBERSHIP_REJECTED,
    MEMBERSHIP_UNASSIGNED,
    MATERIALIZED_REPRESENTATIVE,
    OCCLUDED,
    OBSERVED,
    UNRESOLVED,
    EvidenceBoundedTSDFField,
    TSDFVisibleSurfaceSamples,
    representative_to_json,
    run_candidate_f,
    extract_tsdf_zero_surface_samples,
)
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    RegionCoherenceConfig,
    region_coherent_accounting,
    partition_surfels_region_coherent,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel


DEFAULT_CHECKPOINT = REPO_ROOT / "output" / "arch_2dgs_coverage_first_surface" / "2dgs_run1" / "30000" / "checkpoint.pt"
DEFAULT_FIELD = REPO_ROOT / "output" / "153_raw_visible_surface_replay_construction_provenance_audit" / "replay_cache" / "field.npz"
DEFAULT_SOURCE_PATH = REPO_ROOT / "DATASET"
DEFAULT_OUT = REPO_ROOT / "output" / "154_gaussian_region_owned_tsdf_boundary_first_nurbs"
BASELINE_RECONCILIATION = REPO_ROOT / "output" / "149_physical_sheet_evidence_vs_chart_extent_failure_attribution" / "baseline_reconciliation.json"
WL139_REPORT_CANDIDATES = (
    REPO_ROOT / "output" / "confirmed" / "139_physical_chart_surface_representative" / "physical_chart_surface_representative_report.json",
    REPO_ROOT / "output" / "confirmed" / "138_scale_separated_visible_surface_representative" / "scale_separated_visible_surface_representative_report.json",
)
WL148_REPORT = REPO_ROOT / "output" / "149_physical_sheet_evidence_vs_chart_extent_failure_attribution" / "baseline_reconciliation.json"
SH_C0 = 0.28209479177387814
VISUAL_ITERATION_DIR = "iteration_00030000"
FIXED_COLORS = {
    OBSERVED: (0.10, 0.85, 0.35),
    OCCLUDED: (0.92, 0.18, 0.18),
    UNRESOLVED: (0.60, 0.60, 0.62),
}


def _progress(message: str) -> None:
    print(f"[candidate F] {message}", flush=True)


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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_surfel_model_safe(checkpoint: Path, device: str) -> tuple[TorchGaussianSurfelModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if int(payload.get("scale_dim", 3)) != 2:
        raise ValueError("Candidate F requires a scale_dim==2 2DGS surfel checkpoint")
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
        xyz=raw["xyz"],
        features_dc=raw["features_dc"],
        features_rest=raw["features_rest"],
        opacity=raw["opacity"],
        scaling=raw["scaling"],
        rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"],
        uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"],
        cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"),
        surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=stable_ids,
    )
    model.active_sh_degree = int(payload.get("active_sh_degree", degree))
    return model, payload


def _fixed_rgb_to_f_dc(rgb: Any) -> torch.Tensor:
    return (torch.as_tensor(rgb, dtype=torch.float32) - 0.5) / SH_C0


def _state_colors(states: torch.Tensor, device: Any) -> torch.Tensor:
    colors = torch.empty((states.shape[0], 3), dtype=torch.float32, device=device)
    for code, name in enumerate((OBSERVED, OCCLUDED, UNRESOLVED)):
        colors[states == code] = torch.as_tensor(FIXED_COLORS[name], dtype=torch.float32, device=device)
    return colors


def _write_visual_pair(
    output_root: Path,
    model: TorchGaussianSurfelModel,
    states: torch.Tensor,
    *,
    source_path: Path | None,
    images: str,
    sparse_dir: str,
    resolution: int,
    llffhold: int,
    device: str,
    skip_render: bool,
) -> dict[str, Any]:
    from scripts.devtools.coverage_first_surfel_partition_export import build_preview_camera, write_ppm, write_surfel_ply

    original_dc = model._features_dc.detach()[:, 0, :]
    original_rest = model._features_rest.detach().clone()
    original_degree = int(model.active_sh_degree)
    colors = _state_colors(states, model.device)
    observed_dc = _fixed_rgb_to_f_dc(colors).to(model.device)
    visual_root = output_root / "mandatory_gaussian_visualization_pair"
    original_ply = visual_root / "Original Scene" / VISUAL_ITERATION_DIR / "point_cloud.ply"
    observed_ply = visual_root / "Observed-Occluded" / VISUAL_ITERATION_DIR / "point_cloud.ply"
    original_ply.parent.mkdir(parents=True, exist_ok=True)
    observed_ply.parent.mkdir(parents=True, exist_ok=True)
    write_surfel_ply(original_ply, model.get_xyz.detach(), original_dc, model._opacity.detach(), model._scaling.detach(), model.get_rotation.detach())
    write_surfel_ply(observed_ply, model.get_xyz.detach(), observed_dc, model._opacity.detach(), model._scaling.detach(), model.get_rotation.detach())
    pair_report: dict[str, Any] = {
        "contract": {
            "same_checkpoint": True,
            "same_iteration": 30000,
            "same_camera": True,
            "same_resolution": True,
            "same_background": True,
            "same_renderer": True,
            "same_gaussian_row_count": True,
            "geometry_changed": False,
            "original_scene_color_source": "trained_model_features_dc_and_features_rest",
            "observed_occluded_color_source": "fixed_state_palette_only",
            "unresolved_color_is_explicit_gray": True,
        },
        "state_counts": {
            OBSERVED: int((states == 0).sum()),
            OCCLUDED: int((states == 1).sum()),
            UNRESOLVED: int((states == 2).sum()),
        },
        "row_count": int(states.shape[0]),
        "paths": {"Original Scene": str(original_ply), "Observed-Occluded": str(observed_ply)},
        "fixed_palette": FIXED_COLORS,
    }
    if skip_render or source_path is None:
        pair_report["render"] = {"enabled": False, "reason": "--skip-render or --source-path omitted"}
        return pair_report
    try:
        camera, camera_metadata = build_preview_camera(source_path, images, sparse_dir, resolution, llffhold, device)
        from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        background = torch.zeros((3,), dtype=torch.float32, device=model.device)
        package = rasterizer.render(camera, model, background=background)
        original_render = visual_root / "Original Scene" / "render.ppm"
        write_ppm(original_render, package["render"])
        del package
        model._features_dc.data.copy_(observed_dc[:, None, :])
        model._features_rest.data.zero_()
        model.active_sh_degree = 0
        package = rasterizer.render(camera, model, background=background)
        observed_render = visual_root / "Observed-Occluded" / "render.ppm"
        write_ppm(observed_render, package["render"])
        del package
        model._features_dc.data.copy_(original_dc[:, None, :])
        model._features_rest.data.copy_(original_rest)
        model.active_sh_degree = original_degree
        pair_report["render"] = {
            "enabled": True,
            "camera": camera_metadata,
            "renderer": "OSNSurfelRasterizer",
            "backend": rasterizer.backend_source,
            "background": [0.0, 0.0, 0.0],
            "clamp_render": bool(rasterizer.config.clamp_render),
            "original_scene_sh_degree": original_degree,
            "observed_occluded_sh_degree": 0,
            "paths": {"Original Scene": str(original_render), "Observed-Occluded": str(observed_render)},
        }
    except Exception as error:
        model._features_dc.data.copy_(original_dc[:, None, :])
        model._features_rest.data.copy_(original_rest)
        model.active_sh_degree = original_degree
        pair_report["render"] = {"enabled": False, "failed": True, "reason": f"{type(error).__name__}: {error}"}
        _progress(f"matched render pair unavailable: {type(error).__name__}: {error}")
    return pair_report


def _component_json(result: Any, boundary: Any, representative: Any) -> dict[str, Any]:
    return {
        "component_id": result.component_id,
        "region_id": result.region_id,
        "sample_count": int(result.sample_indices.numel()),
        "min_cell": list(result.min_cell),
        "max_cell": list(result.max_cell),
        "boundary": {
            "closed": boundary.closed,
            "eligible": boundary.eligible,
            "reason": boundary.reason,
            "loop_count": len(boundary.loops),
            "boundary_vertex_count": int(boundary.boundary_world.shape[0]),
            "provenance": _jsonable(boundary.provenance),
        },
        "representative": representative_to_json(representative),
    }


def _save_result_arrays(output_root: Path, result: Any) -> None:
    np.savez(
        output_root / "candidate_f_association.npz",
        nearest_gaussian_index=result.association.nearest_gaussian_index.numpy(),
        nearest_gaussian_id=result.association.nearest_gaussian_id.numpy(),
        nearest_distance=result.association.nearest_distance.numpy(),
    )
    np.savez(
        output_root / "candidate_f_region_owned_support.npz",
        nearest_region_id=result.support.nearest_region_id.numpy(),
        owned_region_id=result.support.owned_region_id.numpy(),
        accepted_mask=result.support.accepted_mask.numpy(),
        component_id=result.component_ids.numpy(),
    )


def _save_full_sample_arrays(output_root: Path, samples: TSDFVisibleSurfaceSamples) -> None:
    """Persist provenance-heavy fields before topology drops them from RAM."""

    np.savez(
        output_root / "candidate_f_tsdf_surface_samples.npz",
        source_cell_keys=samples.source_cell_keys.numpy(),
        cell_indices=samples.cell_indices.numpy(),
        world_xyz=samples.world_xyz.numpy(),
        normals=samples.normals.numpy(),
        corner_values=samples.corner_values.numpy(),
        corner_support_count=samples.corner_support_count.numpy(),
    )


def _load_saved_sample_arrays(path: Path, h: float) -> TSDFVisibleSurfaceSamples:
    with np.load(path, allow_pickle=False) as data:
        keys = np.asarray(data["source_cell_keys"], dtype=np.int64)
        return TSDFVisibleSurfaceSamples(
            source_cell_keys=keys,
            cell_indices=np.asarray(data["cell_indices"], dtype=np.int64),
            world_xyz=np.asarray(data["world_xyz"], dtype=np.float32),
            normals=np.asarray(data["normals"], dtype=np.float32),
            corner_values=np.asarray(data["corner_values"], dtype=np.float32),
            corner_support_count=np.asarray(data["corner_support_count"], dtype=np.int32),
            h=h,
            stats={
                "surface_sample_count": int(keys.shape[0]),
                "replay_source": "candidate_f_tsdf_surface_samples.npz",
                "mesh_intermediate": False,
            },
        )


def _save_representative_controls(output_root: Path, result: Any) -> None:
    control_root = output_root / "representatives"
    control_root.mkdir(parents=True, exist_ok=True)
    for representative in result.representatives:
        if representative.surface is None:
            continue
        np.savez(
            control_root / f"component_{representative.component_id:06d}.npz",
            control_grid=representative.surface.control_grid.detach().cpu().numpy(),
            weights=representative.surface.weights.detach().cpu().numpy(),
            uv=representative.uv.detach().cpu().numpy(),
        )


def _recursive_event_1527(value: Any) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) == "1527":
                return child
            found = _recursive_event_1527(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict) and (child.get("event_id") == 1527 or child.get("id") == 1527):
                return child
            found = _recursive_event_1527(child)
            if found is not None:
                return found
    return None


def _event_1527_lineage() -> dict[str, Any]:
    if not BASELINE_RECONCILIATION.exists():
        return {"available": False, "reason": "WL149 baseline reconciliation is absent"}
    baseline = json.loads(BASELINE_RECONCILIATION.read_text(encoding="utf-8"))
    return {
        "available": True,
        "historical_baseline": _recursive_event_1527(baseline),
        "historical_baseline_path": str(BASELINE_RECONCILIATION),
        "historical_review": "CLEAR_NOT_ON_INTENDED_SURFACE",
        "candidate_f_use": {
            "field_construction": "not isolatable from the WL153 fused field cache because it has no per-event provenance; no blacklist was applied",
            "nearest_association": "no direct event input; association consumes TSDF zero-surface samples and Gaussian rows",
            "region_ownership": "no direct event input; ownership consumes nearest Gaussian and pre-existing region ID",
            "boundary_and_fit": "no event coordinate pooling, global PCA, extrema, or event-specific selection",
            "semantic_review": "historical review retained for audit only; it does not select, reject, or color Candidate F geometry",
        },
        "blacklist_applied": False,
    }


def _review_clouds(result: Any, active_xyz: torch.Tensor, active_partition: Any, active_ids: torch.Tensor) -> dict[str, Any]:
    """Attach frozen WL145 clouds as diagnostic review only.

    These clouds are never fed into Candidate F.  They only provide a matched
    qualitative panel for tabletop/table-side/vase-neighbour/background review.
    """

    root = REPO_ROOT / "output" / "confirmed" / "145_genuine_physical_sheet_oracle_clean_support_representative_audit"
    cases = {
        "tabletop": "tabletop_broad_planar_clean",
        "table_side": "tabletop_near_vase_boundary_candidate",
        "vase_neighbor": "table_rim_curved_interior_candidate",
    }
    active_xyz_cpu = active_xyz.detach().cpu()
    active_ids_cpu = active_ids.detach().cpu()
    reviews: dict[str, Any] = {
        "review_only": True,
        "source": str(root),
        "candidate_f_input": False,
        "cases": {},
    }
    from osn_gs.surface.torch_gaussian_region_owned_tsdf import associate_tsdf_samples_to_gaussians

    for label, case in cases.items():
        case_root = root / case / "per_view_renderer_median_events"
        files = sorted(case_root.rglob("event_cloud_with_provenance.npz")) if case_root.exists() else []
        clouds = []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                key = next((name for name in ("event_points_xyz", "points_xyz", "world_xyz", "xyz") if name in data), None)
                if key is None:
                    continue
                points = np.asarray(data[key], dtype=np.float32).reshape(-1, 3)
            if points.shape[0] == 0:
                continue
            assoc = associate_tsdf_samples_to_gaussians(points, active_xyz_cpu, active_ids_cpu)
            region = active_partition.subset_ids.detach().cpu()[assoc.nearest_gaussian_index]
            counts = Counter(int(value) for value in region.tolist())
            clouds.append({
                "path": str(path),
                "point_count": int(points.shape[0]),
                "candidate_region_ids_by_nearest_gaussian": counts.most_common(12),
                "association_distance": assoc.stats,
            })
        reviews["cases"][label] = {
            "frozen_case": case,
            "cloud_count": len(clouds),
            "clouds": clouds,
            "classification": "USER_REVIEW_REQUIRED",
            "interpretation": "Gaussian-region/TSDF correspondence diagnostic only; no semantic promotion",
        }
    reviews["cases"]["background_lower"] = {
        "frozen_case": None,
        "cloud_count": 0,
        "classification": "USER_REVIEW_REQUIRED",
        "interpretation": "No frozen WL145 event cloud was used as an input; inspect the matched Gaussian pair and component exports",
    }
    return reviews


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    _progress(f"loading safe checkpoint {args.checkpoint}")
    model, payload = _load_surfel_model_safe(args.checkpoint, args.device)
    total_count = len(model)
    uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
    active_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
    _progress(f"checkpoint rows={total_count:,} active={int(active_selector.numel()):,} uncertain={total_count - int(active_selector.numel()):,}")

    with torch.no_grad():
        full_orientation = derive_surface_orientation_from_surfel(model)
        active_orientation = replace(
            full_orientation,
            gaussian_ids=full_orientation.gaussian_ids[active_selector],
            positions=full_orientation.positions[active_selector],
            tangent_axis_u=full_orientation.tangent_axis_u[active_selector],
            tangent_axis_v=full_orientation.tangent_axis_v[active_selector],
            surface_normal=full_orientation.surface_normal[active_selector],
            tangent_scale_u=full_orientation.tangent_scale_u[active_selector],
            tangent_scale_v=full_orientation.tangent_scale_v[active_selector],
        )
        partition = partition_surfels_region_coherent(active_orientation, RegionCoherenceConfig(), progress=_progress)
    gaussian_partition_accounting = region_coherent_accounting(partition)

    saved_samples = args.out / "candidate_f_tsdf_surface_samples.npz"
    if args.reuse_samples and saved_samples.exists():
        with np.load(args.field, allow_pickle=False) as field_meta:
            field_h = float(np.asarray(field_meta["h"]).reshape(-1)[0])
            field_mu = float(np.asarray(field_meta["mu"]).reshape(-1)[0])
        if abs(field_mu - 3.0 * field_h) > max(1e-7, abs(field_h) * 1e-6):
            raise ValueError(f"WL127 TSDF contract violated: mu={field_mu} h={field_h}; expected mu=3h")
        samples = _load_saved_sample_arrays(saved_samples, field_h)
        _progress(f"reusing saved direct TSDF samples={samples.source_cell_keys.numel():,}")
    else:
        field = EvidenceBoundedTSDFField.from_npz(str(args.field))
        if abs(field.mu - 3.0 * field.h) > max(1e-7, abs(field.h) * 1e-6):
            raise ValueError(f"WL127 TSDF contract violated: mu={field.mu} h={field.h}; expected mu=3h")
        _progress(f"loading field keys={field.keys.numel():,} h={field.h:g} mu={field.mu:g}")
        samples = extract_tsdf_zero_surface_samples(field, chunk_size=args.field_chunk, device="cpu")
        _progress(f"direct TSDF samples={samples.source_cell_keys.numel():,}")
        del field
        _save_full_sample_arrays(args.out, samples)
    # Topology and NURBS need only source cells, positions, and normals.  Keep
    # the corner provenance on disk so the connected-components peak does not
    # retain two additional 21M-row arrays in memory.
    samples = TSDFVisibleSurfaceSamples(
        source_cell_keys=samples.source_cell_keys,
        cell_indices=samples.cell_indices,
        world_xyz=samples.world_xyz,
        normals=samples.normals,
        corner_values=torch.empty((0, 8), dtype=torch.float32),
        corner_support_count=torch.empty((0, 8), dtype=torch.int32),
        h=samples.h,
        stats=samples.stats,
    )

    fit_kwargs = {
        "resolution_u": 8,
        "resolution_v": 4,
        "degree_u": 2,
        "degree_v": 2,
        "smoothness_lambda": 1e-4,
        "tikhonov_lambda": 1e-4,
        "correction_rounds": 2,
        "chunk_size": 8192,
        "projection_iterations": 2,
    }
    result = run_candidate_f(
        samples,
        active_orientation.positions,
        active_orientation.gaussian_ids,
        partition,
        association_chunk_size=args.association_chunk,
        torch_pair_limit=args.torch_pair_limit,
        fit_kwargs=fit_kwargs,
        progress=_progress,
    )
    _progress(f"support components={len(result.components):,} materialized representatives={sum(item.status == MATERIALIZED_REPRESENTATIVE for item in result.representatives):,}")

    _save_result_arrays(args.out, result)
    _save_representative_controls(args.out, result)
    components_payload = [
        _component_json(component, boundary, representative)
        for component, boundary, representative in zip(result.components, result.boundaries, result.representatives)
    ]
    (args.out / "support_components.json").write_text(json.dumps(_jsonable(components_payload), indent=2), encoding="utf-8")
    (args.out / "representatives.json").write_text(json.dumps(_jsonable([representative_to_json(item) for item in result.representatives]), indent=2), encoding="utf-8")
    (args.out / "gaussian_branch_accounting.json").write_text(json.dumps(_jsonable(gaussian_partition_accounting), indent=2), encoding="utf-8")
    (args.out / "tsdf_surface_accounting.json").write_text(json.dumps(_jsonable(result.accounting["tsdf_branch"]), indent=2), encoding="utf-8")
    (args.out / "association_accounting.json").write_text(json.dumps(_jsonable(result.accounting["association"]), indent=2), encoding="utf-8")
    (args.out / "region_owned_support_accounting.json").write_text(json.dumps(_jsonable(result.accounting["region_owned_support"]), indent=2), encoding="utf-8")

    observed_active = torch.zeros((active_selector.numel(),), dtype=torch.bool, device=active_selector.device)
    if result.support.accepted_mask.any():
        observed_active[result.association.nearest_gaussian_index[result.support.accepted_mask].to(observed_active.device)] = True
    states = torch.full((total_count,), 2, dtype=torch.int8, device=model.device)
    states[active_selector] = 1
    states[active_selector[observed_active]] = 0
    np.savez(args.out / "gaussian_visualization_states.npz", stable_gaussian_ids=model.stable_gaussian_ids.detach().cpu().numpy(), state=states.detach().cpu().numpy())
    visualization = _write_visual_pair(
        args.out, model, states,
        source_path=args.source_path,
        images=args.images,
        sparse_dir=args.sparse_dir,
        resolution=args.resolution,
        llffhold=args.llffhold,
        device=args.device,
        skip_render=args.skip_render,
    )

    reviews = _review_clouds(result, active_orientation.positions, partition, active_orientation.gaussian_ids)
    (args.out / "qualitative_review.json").write_text(json.dumps(_jsonable(reviews), indent=2), encoding="utf-8")
    lineage = _event_1527_lineage()
    (args.out / "event_1527_lineage.json").write_text(json.dumps(_jsonable(lineage), indent=2), encoding="utf-8")

    historical_wl139 = next((path for path in WL139_REPORT_CANDIDATES if path.exists()), None)
    baseline_payload = json.loads(BASELINE_RECONCILIATION.read_text(encoding="utf-8")) if BASELINE_RECONCILIATION.exists() else None
    baseline_report = {
        "source": str(BASELINE_RECONCILIATION),
        "copied_event_count": baseline_payload.get("event_count") if baseline_payload else None,
        "copied_event_union_sha256": baseline_payload.get("event_union_sha256") if baseline_payload else None,
        "event_1527_review_retained": True,
        "historical_wl139_report": str(historical_wl139) if historical_wl139 else None,
        "historical_wl148_reconciliation": str(WL148_REPORT),
        "candidate_f_does_not_rewrite_historical_baseline": True,
    }
    (args.out / "baseline_reconciliation.json").write_text(json.dumps(_jsonable(baseline_report), indent=2), encoding="utf-8")

    report = {
        "status": "COMPLETE_CANDIDATE_REPLAY",
        "batch": "Worklog 154 — Gaussian-Region-Owned TSDF Surface Support and Boundary-First NURBS Construction",
        "candidate": "F",
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "field": str(args.field.resolve()),
            "field_contract": {"source": "WL127 field.npz", "mu_equals_3h": True, "unknown_absence_preserved": True},
            "source_path": str(args.source_path.resolve()) if args.source_path else None,
            "iteration": int(payload.get("iteration", 0)),
            "primitive": "surfel_2d",
            "scale_dim": 2,
        },
        "gaussian_branch": {
            "normal_source": "derive_surface_orientation_from_surfel -> trained intrinsic t_w",
            "region_id_source": "partition_surfels_region_coherent using existing WL96 local candidate graph",
            "accounting": gaussian_partition_accounting,
            "candidate_f_membership": result.accounting["gaussian_branch"],
        },
        "observation_branch": result.accounting["tsdf_branch"],
        "association": result.accounting["association"],
        "region_owned_support": result.accounting["region_owned_support"],
        "topology_and_boundary": {
            "components": len(result.components),
            "boundary_first": True,
            "boundary_source": "native_tsdf_cell_face_adjacency",
            "component_payload": str(args.out / "support_components.json"),
        },
        "representative": result.accounting["representative"],
        "wl139_fit_family": fit_kwargs,
        "visualization": visualization,
        "qualitative_review": reviews,
        "event_1527": lineage,
        "historical_baseline": baseline_report,
        "forbidden_paths": result.accounting["forbidden_paths"],
        "outputs": {
            "surface_samples": str(args.out / "candidate_f_tsdf_surface_samples.npz"),
            "association": str(args.out / "candidate_f_association.npz"),
            "support": str(args.out / "candidate_f_region_owned_support.npz"),
            "representatives": str(args.out / "representatives.json"),
            "matched_gaussian_pair": visualization["paths"],
        },
        "runtime_seconds": {"total": time.time() - started},
    }
    (args.out / "candidate_f_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--field-chunk", type=int, default=1_000_000)
    parser.add_argument("--association-chunk", type=int, default=131_072)
    parser.add_argument("--torch-pair-limit", type=int, default=8_000_000)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--reuse-samples", action="store_true", help="Reuse a previously persisted direct TSDF sample cache under --out.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "surface_sample_count": report["observation_branch"].get("surface_sample_count"),
        "component_count": report["topology_and_boundary"]["components"],
        "materialized_representative_count": report["representative"]["materialized_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
