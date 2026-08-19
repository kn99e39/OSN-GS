"""Worklog 103 -- factual latent-surface spatial-coverage audit and export.

Restores the geometry-provenance fix (Worklog 98's ``component.positions``
now genuinely carries ``LatentSurfaceSupport.query_batch(...).positions``,
not raw Gaussian centers -- see ``osn_gs/surface/torch_latent_surface_tangent_frame_field.py``)
and exports, for EVERY region-owned source observation, the full latent
geometry the Worklog 95 estimator actually produces -- regardless of
whether any downstream stage (Worklog 98 coherence, Worklog 100 UV
validity, Worklog 101 chart membership, Worklog 102 patch identifiability)
would accept it.

This script does NOT advance Worklog 98-102 architecture decisions. It
reuses Worklog 102's existing candidate-C pipeline ONLY to produce the
`worklog102_existing_nurbs` comparison representation -- nothing about
that pipeline is modified, tuned, or re-decided here.

WebRenderer directory contract (explicit user correction): each
``iteration_<N>`` folder holds EXACTLY ONE ``point_cloud.ply`` -- the
renderer reads only one PLY per iteration folder. A folder with a
``point_cloud.ply`` pairs with either exactly one ``nurbs_surface.json``
or none. Every distinct point set therefore gets its OWN folder; nothing
is ever combined into a shared PLY.

No qualitative/architecture conclusion is computed or printed by this
script -- only factual counts, displacement statistics, and file paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from osn_gs.surface.torch_adaptive_patch_capacity import select_support_adaptive_capacity
from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_coverage_audit import audit_region_latent_coverage
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, build_seed_curves
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_latent_surface_visualization_nurbs import fit_visualization_nurbs
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv

_SH_DC = 0.28209479177387814
_ITER = "iteration_0000001"


def _color_to_f_dc(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple((component - 0.5) / _SH_DC for component in rgb)


def _write_diagnostic_ply(path: Path, positions: Any, rgb: tuple[float, float, float], point_scale: float) -> int:
    """Minimal renderer-compatible Gaussian ``point_cloud.ply`` for a plain
    diagnostic point set (no OSN-GS-specific columns) -- every point gets
    the SAME flat color and a small isotropic scale so it renders as a
    visible dot, not an interpretable trained Gaussian. ``path`` must be
    the sole PLY in its own folder (WebRenderer directory contract)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(positions.shape[0])
    header_lines = [
        "ply", "format ascii 1.0", f"element vertex {count}",
        "property float x", "property float y", "property float z",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"
    if count == 0:
        path.write_text(header, encoding="utf-8")
        return 0
    positions_np = positions.detach().cpu().numpy()
    f_dc = _color_to_f_dc(rgb)
    log_scale = float(torch.log(torch.tensor(max(point_scale, 1e-6))).item())
    rows = [
        f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
        f"{f_dc[0]:.6f} {f_dc[1]:.6f} {f_dc[2]:.6f} 10.0 "
        f"{log_scale:.6f} {log_scale:.6f} {log_scale:.6f} 1.0 0.0 0.0 0.0"
        for row in positions_np
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return count


def _write_nurbs_json(path: Path, iteration: int, patches: list[dict], base_curves: list | None = None, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "nurbs_surface", "iteration": iteration,
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "base_curves": base_curves or [], "occlusion_curves": [],
        "patches": patches, "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _surface_to_patch_dict(patch_id: int, surface, extra: dict) -> dict:
    return {
        "patch_id": patch_id,
        "control_grid_shape": [int(value) for value in surface.control_grid.shape],
        "control_grid": surface.control_grid.detach().cpu().tolist(),
        "weights": surface.weights.detach().cpu().tolist(),
        "degree_u": int(surface.degree_u), "degree_v": int(surface.degree_v),
        "observed_v_max": float(surface.observed_v_max),
        **extra,
    }


def _pick_field_anchor(seeds) -> tuple[object, object] | tuple[None, None]:
    for seed in seeds:
        if seed.seed_type != SEED_INTERIOR_CONSTRUCTION and int(seed.points.shape[0]) >= 2:
            anchor_position = seed.points[0]
            anchor_hint = seed.points[1] - seed.points[0]
            if float(anchor_hint.norm().item()) > 1e-9:
                return anchor_position, anchor_hint
    return None, None


def _worklog102_patches_for_region(train_evidence, region_chart, representative_positions, representative_index, support) -> list[dict]:
    """Reuses Worklog 102's existing candidate-C pipeline UNCHANGED, purely
    to populate the `worklog102_existing_nurbs` comparison representation.
    No architecture decision is made or re-evaluated here."""

    seeds = build_seed_curves(train_evidence, region_chart, representative_positions, representative_index, support)
    anchor_position, anchor_hint = _pick_field_anchor(seeds)
    field_result = build_tangent_frame_field(
        train_evidence, support, anchor_position=anchor_position, anchor_hint_direction=anchor_hint,
    )
    coherent_components = [component for component in field_result.components if component.coherent]

    patches = []
    for component in coherent_components:
        atlas = build_local_chart_atlas(component, support.median_spacing)
        for chart in atlas.charts:
            selection = select_support_adaptive_capacity(chart.integration.uv)
            if not selection.selected:
                continue
            try:
                surface = fit_torch_visible_surface_from_uv(
                    chart.component.positions, chart.integration.uv,
                    resolution_u=selection.control_grid_u, resolution_v=selection.control_grid_v,
                    degree_u=selection.degree_u, degree_v=selection.degree_v,
                )
            except Exception:  # noqa: BLE001
                continue
            patches.append(_surface_to_patch_dict(len(patches), surface, {
                "osn_gs_representation_kind": "worklog102_existing_nurbs",
                "osn_gs_chart_size": len(chart.node_indices),
            }))
    return patches


def run(checkpoint: Path, cap: int, device: str, out_dir: Path, displacement_stride: int) -> dict:
    model, stable_ids = _load_model(checkpoint, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    region_reports = []
    all_latent_samples = []
    all_viz_patches: list[dict] = []
    all_worklog102_patches: list[dict] = []
    all_displacement_segments: list[list[list[float]]] = []

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        if len(full_indices) < 4:
            region_reports.append({"region": region_id, "skip_reason": "insufficient_owned_evidence"})
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        raw_evidence = points[selector]  # ALL region-owned raw Gaussian centers, no holdout split for this audit

        support = build_latent_surface_support(raw_evidence)
        audit = audit_region_latent_coverage(region_id, raw_evidence, support)

        viz_results = [fit_visualization_nurbs(unit.unit_id, unit.latent_positions) for unit in audit.units]
        viz_patches_region = []
        for unit, result in zip(audit.units, viz_results):
            if not result.materialized:
                continue
            patch = _surface_to_patch_dict(len(all_viz_patches) + len(viz_patches_region), result.surface, {
                "osn_gs_representation_kind": "latent_surface_coverage_visualization_nurbs",
                "osn_gs_region_id": region_id, "osn_gs_unit_id": unit.unit_id,
                "osn_gs_unit_size": len(unit.node_indices),
                "osn_gs_mean_residual": result.mean_residual,
            })
            viz_patches_region.append(patch)
        all_viz_patches.extend(viz_patches_region)

        region_chart = chart_by_region.get(region_id)
        train_evidence = raw_evidence
        worklog102_patches_region = []
        try:
            worklog102_patches_region = _worklog102_patches_for_region(
                train_evidence, region_chart, representative_positions, representative_index, support,
            )
        except Exception as exc:  # noqa: BLE001
            region_reports.append({"region": region_id, "worklog102_comparison_error": f"{type(exc).__name__}: {exc}"})
        all_worklog102_patches.extend(worklog102_patches_region)

        latent_samples_region = (
            torch.cat([unit.latent_positions for unit in audit.units], dim=0)
            if audit.units else torch.zeros((0, 3))
        )
        all_latent_samples.append(latent_samples_region)

        displacement_segments_region = []
        for unit in audit.units:
            for row in range(0, int(unit.raw_positions.shape[0]), max(1, displacement_stride)):
                displacement_segments_region.append([
                    unit.raw_positions[row].detach().cpu().tolist(),
                    unit.latent_positions[row].detach().cpu().tolist(),
                ])
        all_displacement_segments.extend(displacement_segments_region)

        # Region-isolated exports -- ONE point_cloud.ply per folder, at
        # most ONE nurbs_surface.json alongside it.
        region_root = out_dir / "regions" / f"region_{region_id}"
        raw_dir = region_root / "raw_region_evidence" / _ITER
        _write_diagnostic_ply(raw_dir / "point_cloud.ply", raw_evidence, (1.0, 1.0, 1.0), support.median_spacing * 0.3)

        unsupported_dir = region_root / "unsupported_evidence" / _ITER
        _write_diagnostic_ply(unsupported_dir / "point_cloud.ply", audit.unsupported_raw_positions, (0.9, 0.2, 0.2), support.median_spacing * 0.3)

        displacement_dir = region_root / "latent_projected_samples_with_displacement" / _ITER
        _write_diagnostic_ply(displacement_dir / "point_cloud.ply", latent_samples_region, (0.2, 0.9, 0.9), support.median_spacing * 0.3)
        _write_nurbs_json(
            displacement_dir / "nurbs_surface.json", 1, [], base_curves=displacement_segments_region,
            metadata={"osn_gs_representation_kind": "latent_surface_projection_displacement", "region_id": region_id},
        )

        viz_dir = region_root / "latent_projected_samples_with_visualization_nurbs" / _ITER
        _write_diagnostic_ply(viz_dir / "point_cloud.ply", latent_samples_region, (0.2, 0.9, 0.9), support.median_spacing * 0.3)
        _write_nurbs_json(
            viz_dir / "nurbs_surface.json", 1, viz_patches_region,
            metadata={"osn_gs_representation_kind": "latent_surface_coverage_visualization_nurbs", "region_id": region_id},
        )

        region_reports.append({
            "region": region_id,
            "raw_evidence_count": audit.raw_evidence_count,
            "latent_supported_count": audit.latent_supported_count,
            "latent_unsupported_count": audit.latent_unsupported_count,
            "supported_fraction": (audit.latent_supported_count / audit.raw_evidence_count) if audit.raw_evidence_count else None,
            "latent_support_unit_count": len(audit.units),
            "projected_latent_position_count": sum(len(unit.node_indices) for unit in audit.units),
            "projection_displacement_over_spacing": {
                "median": float((audit.projection_displacement_all_supported.norm(dim=1) / max(audit.median_spacing, 1e-9)).median().item()) if audit.latent_supported_count else None,
                "p95": float((audit.projection_displacement_all_supported.norm(dim=1) / max(audit.median_spacing, 1e-9)).quantile(0.95).item()) if audit.latent_supported_count else None,
                "max": float((audit.projection_displacement_all_supported.norm(dim=1) / max(audit.median_spacing, 1e-9)).max().item()) if audit.latent_supported_count else None,
            },
            "visualization_nurbs_attempted": len(audit.units),
            "visualization_nurbs_materialized": len(viz_patches_region),
            "visualization_nurbs_failed": len(audit.units) - len(viz_patches_region),
            "worklog102_comparison_patch_count": len(worklog102_patches_region),
            "export_dirs": {
                "raw_region_evidence": str(raw_dir), "unsupported_evidence": str(unsupported_dir),
                "latent_projected_samples_with_displacement": str(displacement_dir),
                "latent_projected_samples_with_visualization_nurbs": str(viz_dir),
            },
        })

    all_raw_region_points = torch.cat(
        [points[torch.tensor(owned[r.region_id], dtype=torch.long)] for r in regions.regions if len(owned.get(r.region_id, [])) >= 4],
        dim=0,
    ) if any(len(owned.get(r.region_id, [])) >= 4 for r in regions.regions) else torch.zeros((0, 3))
    all_latent_cat = torch.cat(all_latent_samples, dim=0) if all_latent_samples else torch.zeros((0, 3))

    # Global stages -- each its OWN folder, one point_cloud.ply, 0 or 1 nurbs_surface.json.
    full_scene_dir = out_dir / "full_scene" / _ITER
    model.save_ply(full_scene_dir / "point_cloud.ply")

    region_evidence_dir = out_dir / "region_owned_evidence" / _ITER
    _write_diagnostic_ply(region_evidence_dir / "point_cloud.ply", all_raw_region_points, (1.0, 0.85, 0.2), 0.02)

    displacement_dir = out_dir / "latent_projected_samples_with_displacement" / _ITER
    _write_diagnostic_ply(displacement_dir / "point_cloud.ply", all_latent_cat, (0.2, 0.9, 0.9), 0.02)
    _write_nurbs_json(
        displacement_dir / "nurbs_surface.json", 1, [], base_curves=all_displacement_segments,
        metadata={"osn_gs_representation_kind": "latent_surface_projection_displacement", "segment_count": len(all_displacement_segments)},
    )

    viz_dir = out_dir / "latent_projected_samples_with_visualization_nurbs" / _ITER
    _write_diagnostic_ply(viz_dir / "point_cloud.ply", all_latent_cat, (0.2, 0.9, 0.9), 0.02)
    _write_nurbs_json(
        viz_dir / "nurbs_surface.json", 1, all_viz_patches,
        metadata={"osn_gs_representation_kind": "latent_surface_coverage_visualization_nurbs", "count": len(all_viz_patches)},
    )

    worklog102_dir = out_dir / "full_scene_with_worklog102_nurbs" / _ITER
    model.save_ply(worklog102_dir / "point_cloud.ply")
    _write_nurbs_json(
        worklog102_dir / "nurbs_surface.json", 1, all_worklog102_patches,
        metadata={"osn_gs_representation_kind": "worklog102_existing_nurbs", "count": len(all_worklog102_patches)},
    )

    report = {
        "checkpoint": str(checkpoint), "cap": cap,
        "full_scene_gaussian_count": len(model),
        "global_export_dirs": {
            "full_scene": str(full_scene_dir),
            "region_owned_evidence": str(region_evidence_dir),
            "latent_projected_samples_with_displacement": str(displacement_dir),
            "latent_projected_samples_with_visualization_nurbs": str(viz_dir),
            "full_scene_with_worklog102_nurbs": str(worklog102_dir),
        },
        "region_isolated_dir": str(out_dir / "regions"),
        "regions": region_reports,
        "global": {
            "raw_evidence_total": sum(r.get("raw_evidence_count", 0) for r in region_reports),
            "latent_supported_total": sum(r.get("latent_supported_count", 0) for r in region_reports),
            "latent_unsupported_total": sum(r.get("latent_unsupported_count", 0) for r in region_reports),
            "latent_support_unit_total": sum(r.get("latent_support_unit_count", 0) for r in region_reports),
            "projected_latent_position_total": sum(r.get("projected_latent_position_count", 0) for r in region_reports),
            "visualization_nurbs_attempted_total": sum(r.get("visualization_nurbs_attempted", 0) for r in region_reports),
            "visualization_nurbs_materialized_total": sum(r.get("visualization_nurbs_materialized", 0) for r in region_reports),
            "visualization_nurbs_failed_total": sum(r.get("visualization_nurbs_failed", 0) for r in region_reports),
            "worklog102_comparison_patch_total": sum(r.get("worklog102_comparison_patch_count", 0) for r in region_reports),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/final"))
    parser.add_argument("--out", type=Path, default=Path("output/osn_gs_scene_latent_coverage_audit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--displacement-stride", type=int, default=1, help="Write every Nth displacement segment (1 = all).")
    args = parser.parse_args()

    start = time.perf_counter()
    report = run(args.checkpoint, args.cap, args.device, args.out, args.displacement_stride)
    report["runtime_seconds"] = time.perf_counter() - start

    report_path = args.out / "coverage_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["global"], indent=2, default=str))
    print("full report:", report_path)


if __name__ == "__main__":
    main()
