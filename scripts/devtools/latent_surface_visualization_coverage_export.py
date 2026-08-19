"""Worklog 104 -- visualization-completeness regeneration for Worklog 103's
ALL_LATENT_SURFACES (D) stage and region-isolated exports.

Worklog 103 attempted exactly ONE visualization-only NURBS per latent
support unit (86 units, 49 materialized, 37 failed outright) -- meaning
37/86 units' latent-projected geometry was invisible in the NURBS overlay
even though the underlying supported samples were correctly exported as
raw point data. This script changes ONLY the visualization unit (via
:func:`~osn_gs.surface.torch_latent_surface_visualization_coverage.materialize_unit_with_subdivision`,
deterministic connectivity-only subdivision) -- it does NOT touch the
Worklog 95 estimator, Worklog 98 support connectivity, chart construction,
UV parameterization, or production patch fitting, and does NOT replay
Worklog 98-102 architecture (fold rate / UV validity / chart coverage /
identifiability / A-B-C comparisons).

Regenerates ONLY:
- D. ALL_LATENT_SURFACES (full scene + all projected latent samples + all
  post-subdivision visualization NURBS + explicit unrepresented-fragment
  markers)
- region-isolated exports (raw / latent / unsupported / displacement /
  subdivided visualization NURBS / unrepresented fragments)

Writes to a SEPARATE output directory from Worklog 103's own export so the
before/after visualization can be compared directly without overwriting
historical artifacts. A/B/C/E stages are unaffected by this batch (they do
not depend on per-unit visualization NURBS materialization) and are not
regenerated.

No qualitative/architecture conclusion is computed or printed.
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
from osn_gs.surface.torch_latent_surface_coverage_audit import audit_region_latent_coverage
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_visualization_coverage import materialize_unit_with_subdivision

STAGE_D_ALL_LATENT_SURFACES = 103
_SH_DC = 0.28209479177387814


def _color_to_f_dc(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple((component - 0.5) / _SH_DC for component in rgb)


def _write_diagnostic_ply(path: Path, positions: Any, rgb: tuple[float, float, float], point_scale: float) -> int:
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


def run(checkpoint: Path, cap: int, device: str, out_dir: Path, displacement_stride: int) -> dict:
    model, stable_ids = _load_model(checkpoint, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    region_reports = []
    all_latent_samples = []
    all_viz_patches: list[dict] = []
    all_unrepresented_markers: list[dict] = []
    all_unrepresented_position_parts: list[Any] = []

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        if len(full_indices) < 4:
            region_reports.append({"region": region_id, "skip_reason": "insufficient_owned_evidence"})
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        raw_evidence = points[selector]

        support = build_latent_surface_support(raw_evidence)
        audit = audit_region_latent_coverage(region_id, raw_evidence, support)

        region_materialized_nodes: set[int] = set()
        region_unrepresented_nodes: set[int] = set()
        viz_patches_region: list[dict] = []
        unrepresented_region: list[dict] = []
        unrepresented_positions_parts: list[Any] = []
        fully_represented_units = 0
        partially_represented_units = 0
        completely_unrepresented_units = 0

        for unit in audit.units:
            node_to_local = {node: local for local, node in enumerate(unit.node_indices)}
            materialized, unrepresented = materialize_unit_with_subdivision(unit)
            for fragment in materialized:
                region_materialized_nodes.update(fragment.node_indices)
                viz_patches_region.append(_surface_to_patch_dict(len(all_viz_patches) + len(viz_patches_region), fragment.result.surface, {
                    "osn_gs_representation_kind": "latent_surface_coverage_visualization_nurbs",
                    "osn_gs_region_id": region_id, "osn_gs_unit_id": unit.unit_id,
                    "osn_gs_fragment_id": fragment.fragment_id, "osn_gs_fragment_node_count": len(fragment.node_indices),
                    "osn_gs_mean_residual": fragment.result.mean_residual,
                }))
            for fragment in unrepresented:
                region_unrepresented_nodes.update(fragment.node_indices)
                unrepresented_region.append({
                    "region_id": region_id, "unit_id": unit.unit_id,
                    "node_indices": list(fragment.node_indices), "reason": fragment.reason,
                })
                local_indices = [node_to_local[node] for node in fragment.node_indices]
                unrepresented_positions_parts.append(unit.latent_positions[local_indices])
            if not unrepresented:
                fully_represented_units += 1
            elif materialized:
                partially_represented_units += 1
            else:
                completely_unrepresented_units += 1

        all_viz_patches.extend(viz_patches_region)
        all_unrepresented_markers.extend(unrepresented_region)
        all_unrepresented_position_parts.extend(unrepresented_positions_parts)

        latent_samples_region = (
            torch.cat([unit.latent_positions for unit in audit.units], dim=0)
            if audit.units else torch.zeros((0, 3))
        )
        all_latent_samples.append(latent_samples_region)

        # Region-isolated regeneration.
        region_dir = out_dir / "regions" / f"region_{region_id}" / "iteration_0000001"
        _write_diagnostic_ply(region_dir / "raw_region_evidence.ply", raw_evidence, (1.0, 1.0, 1.0), support.median_spacing * 0.3)
        _write_diagnostic_ply(region_dir / "latent_projected_samples.ply", latent_samples_region, (0.2, 0.9, 0.9), support.median_spacing * 0.3)
        _write_diagnostic_ply(region_dir / "unsupported_evidence.ply", audit.unsupported_raw_positions, (0.9, 0.2, 0.2), support.median_spacing * 0.3)
        # Unrepresented fragments (post-subdivision): distinct color/kind so
        # they read as "latent geometry exists but no proxy could be fit"
        # rather than absence.
        unrepresented_positions_region = (
            torch.cat(unrepresented_positions_parts, dim=0) if unrepresented_positions_parts else torch.zeros((0, 3))
        )
        _write_diagnostic_ply(region_dir / "unrepresented_latent_fragments.ply", unrepresented_positions_region, (1.0, 0.5, 0.0), support.median_spacing * 0.3)
        displacement_segments_region = []
        for unit in audit.units:
            for row in range(0, int(unit.raw_positions.shape[0]), max(1, displacement_stride)):
                displacement_segments_region.append([
                    unit.raw_positions[row].detach().cpu().tolist(),
                    unit.latent_positions[row].detach().cpu().tolist(),
                ])
        _write_nurbs_json(
            region_dir / "nurbs_surface.json", 1, viz_patches_region, base_curves=displacement_segments_region,
            metadata={
                "osn_gs_representation_kind": "region_isolated_latent_surface_coverage_subdivided",
                "region_id": region_id, "unrepresented_fragment_count": len(unrepresented_region),
            },
        )

        supported_total = audit.latent_supported_count
        represented_count = len(region_materialized_nodes)
        unrepresented_count = len(region_unrepresented_nodes)
        region_reports.append({
            "region": region_id,
            "latent_supported_node_count": supported_total,
            "visualization_represented_node_count": represented_count,
            "visualization_unrepresented_node_count": unrepresented_count,
            "visualization_represented_node_fraction": (represented_count / supported_total) if supported_total else None,
            "latent_support_unit_count": len(audit.units),
            "fully_represented_unit_count": fully_represented_units,
            "partially_represented_unit_count": partially_represented_units,
            "completely_unrepresented_unit_count": completely_unrepresented_units,
            "visualization_nurbs_patch_count": len(viz_patches_region),
            "unrepresented_fragments": unrepresented_region,
            "node_accounting_ok": (represented_count + unrepresented_count == supported_total),
        })

    # D. ALL_LATENT_SURFACES regeneration.
    all_latent_dir = out_dir / f"iteration_{STAGE_D_ALL_LATENT_SURFACES:07d}"
    model.save_ply(all_latent_dir / "full_scene.ply")
    all_latent_cat = torch.cat(all_latent_samples, dim=0) if all_latent_samples else torch.zeros((0, 3))
    _write_diagnostic_ply(all_latent_dir / "latent_projected_samples.ply", all_latent_cat, (0.2, 0.9, 0.9), 0.02)
    all_unrepresented_positions = (
        torch.cat(all_unrepresented_position_parts, dim=0) if all_unrepresented_position_parts else torch.zeros((0, 3))
    )
    _write_diagnostic_ply(all_latent_dir / "unrepresented_latent_fragments.ply", all_unrepresented_positions, (1.0, 0.5, 0.0), 0.02)
    _write_nurbs_json(
        all_latent_dir / "nurbs_surface.json", STAGE_D_ALL_LATENT_SURFACES, all_viz_patches,
        metadata={
            "osn_gs_representation_kind": "latent_surface_coverage_visualization_nurbs",
            "count": len(all_viz_patches), "unrepresented_fragment_count": len(all_unrepresented_markers),
        },
    )

    global_supported = sum(r.get("latent_supported_node_count", 0) for r in region_reports)
    global_represented = sum(r.get("visualization_represented_node_count", 0) for r in region_reports)
    global_unrepresented = sum(r.get("visualization_unrepresented_node_count", 0) for r in region_reports)

    report = {
        "checkpoint": str(checkpoint), "cap": cap,
        "stage_regenerated": str(all_latent_dir),
        "region_isolated_dir": str(out_dir / "regions"),
        "regions": region_reports,
        "global": {
            "latent_supported_node_count": global_supported,
            "visualization_represented_node_count": global_represented,
            "visualization_unrepresented_node_count": global_unrepresented,
            "visualization_represented_node_fraction": (global_represented / global_supported) if global_supported else None,
            "latent_support_unit_count": sum(r.get("latent_support_unit_count", 0) for r in region_reports),
            "fully_represented_unit_count": sum(r.get("fully_represented_unit_count", 0) for r in region_reports),
            "partially_represented_unit_count": sum(r.get("partially_represented_unit_count", 0) for r in region_reports),
            "completely_unrepresented_unit_count": sum(r.get("completely_unrepresented_unit_count", 0) for r in region_reports),
            "visualization_nurbs_patch_count": sum(r.get("visualization_nurbs_patch_count", 0) for r in region_reports),
            "node_accounting_ok": all(r.get("node_accounting_ok", False) for r in region_reports if "node_accounting_ok" in r),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/final"))
    parser.add_argument("--out", type=Path, default=Path("output/osn_gs_scene_latent_coverage_audit_subdivided"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--displacement-stride", type=int, default=1)
    args = parser.parse_args()

    start = time.perf_counter()
    report = run(args.checkpoint, args.cap, args.device, args.out, args.displacement_stride)
    report["runtime_seconds"] = time.perf_counter() - start

    report_path = args.out / "visualization_coverage_certificate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["global"], indent=2, default=str))
    print("full report:", report_path)


if __name__ == "__main__":
    main()
