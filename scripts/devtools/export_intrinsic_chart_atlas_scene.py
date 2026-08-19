"""Export the Worklog 101/102 local intrinsic chart atlas as a
WebRenderer-loadable scene (``nurbs_surface.json`` + ``point_cloud.ply``),
in exactly the same directory/file format as ``output/osn_gs_scene`` (see
``WebRenderer/RENDERER_INPUT_FORMAT.md``).

This is a VIEWING export only -- it reuses the exact same chart-atlas
construction and candidate-C support-adaptive capacity selection that
Worklog 102's real replay was built from (identical
``build_local_chart_atlas`` + ``select_support_adaptive_capacity`` +
``fit_torch_visible_surface_from_uv`` call sites), so what is written here
is the same population of charts/patches the worklog's numbers describe.
It does not train, does not modify the checkpoint, and is not wired into
the training loop or CLI -- run it against an existing checkpoint whenever
you want to look at the current chart atlas.

Every identifiable chart (candidate C: degree/capacity chosen solely by
pre-fit identifiability, exactly as in Worklog 102) is fit and included as
one ``patches[]`` entry, regardless of its downstream
safe/unsafe/extrapolative classification -- so what you see is the whole
constructed atlas, not just the safe subset. Each patch's
``patch_id``/metadata also carries its Worklog 102 category and chart
size so the renderer's per-patch inspector shows why a patch looks the
way it does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401
from chart_unit_general_partition_seam_replay import _holdout, _median_nn
from chart_unit_surface_topology_temporal_lineage_replay import _load_model, _region_analysis
from curve_network_native_fit_replay import classify_fitted_surface
from osn_gs.surface.torch_adaptive_patch_capacity import select_support_adaptive_capacity
from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, build_seed_curves
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv


def _pick_field_anchor(seeds) -> tuple[object, object] | tuple[None, None]:
    for seed in seeds:
        if seed.seed_type != SEED_INTERIOR_CONSTRUCTION and int(seed.points.shape[0]) >= 2:
            anchor_position = seed.points[0]
            anchor_hint = seed.points[1] - seed.points[0]
            if float(anchor_hint.norm().item()) > 1e-9:
                return anchor_position, anchor_hint
    return None, None


def _patch_payload(patch_id: int, surface, category: str, chart_size: int, region_id: int) -> dict:
    return {
        "patch_id": patch_id,
        "control_grid_shape": [int(value) for value in surface.control_grid.shape],
        "control_grid": surface.control_grid.detach().cpu().tolist(),
        "weights": surface.weights.detach().cpu().tolist(),
        "degree_u": int(surface.degree_u),
        "degree_v": int(surface.degree_v),
        "observed_v_max": float(surface.observed_v_max),
        # Extra, renderer-ignored diagnostic fields -- lets the inspector
        # show which Worklog 102 category/region/chart size produced this
        # patch without needing a second file.
        "osn_gs_chart_region_id": region_id,
        "osn_gs_chart_size": chart_size,
        "osn_gs_fit_category": category,
    }


def export(checkpoint: Path, cap: int, device: str, out_dir: Path, iteration_label: int) -> dict:
    model, stable_ids = _load_model(checkpoint, device)
    (
        regions, points, covariance, owned, representative_positions,
        representative_index, frame_by_region, chart_by_region,
    ) = _region_analysis(model, stable_ids, cap, device)

    patches: list[dict] = []
    stats = {"regions": 0, "coherent_components": 0, "charts": 0, "identifiable": 0, "patches_written": 0}

    for region in regions.regions:
        region_id = region.region_id
        full_indices = owned.get(region_id, [])
        if len(full_indices) < 4:
            continue
        selector = torch.tensor(full_indices, dtype=torch.long, device=points.device)
        evidence = points[selector]
        region_chart = chart_by_region.get(region_id)

        train_evidence, held_evidence = _holdout(evidence)
        if int(train_evidence.shape[0]) < 4:
            continue
        stats["regions"] += 1

        support = build_latent_surface_support(train_evidence)
        held_out_target = held_evidence if int(held_evidence.shape[0]) > 0 else evidence
        scale = _median_nn(held_out_target) if int(held_out_target.shape[0]) >= 2 else _median_nn(evidence)

        seeds = build_seed_curves(train_evidence, region_chart, representative_positions, representative_index, support)
        anchor_position, anchor_hint = _pick_field_anchor(seeds)
        field_result = build_tangent_frame_field(
            train_evidence, support, anchor_position=anchor_position, anchor_hint_direction=anchor_hint,
        )
        coherent_components = [component for component in field_result.components if component.coherent]
        stats["coherent_components"] += len(coherent_components)

        for component in coherent_components:
            atlas = build_local_chart_atlas(component, support.median_spacing)
            stats["charts"] += len(atlas.charts)
            for chart in atlas.charts:
                selection = select_support_adaptive_capacity(chart.integration.uv)
                if not selection.selected:
                    continue
                stats["identifiable"] += 1
                try:
                    surface = fit_torch_visible_surface_from_uv(
                        chart.component.positions, chart.integration.uv,
                        resolution_u=selection.control_grid_u, resolution_v=selection.control_grid_v,
                        degree_u=selection.degree_u, degree_v=selection.degree_v,
                    )
                except Exception:  # noqa: BLE001
                    continue
                record = classify_fitted_surface(surface, chart.component.positions, held_out_target, scale)
                category = record.get("classification", "unknown")
                patches.append(_patch_payload(
                    len(patches), surface, category, len(chart.node_indices), region_id,
                ))
                stats["patches_written"] += 1

    payload = {
        "type": "nurbs_surface",
        "iteration": iteration_label,
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "base_curves": [],
        "occlusion_curves": [],
        "patches": patches,
        "metadata": {
            "source": "worklog_101_102_intrinsic_chart_atlas_export",
            "checkpoint": str(checkpoint),
            "capacity_candidate": "C_support_adaptive_local_nurbs",
            **stats,
        },
    }

    scene_dir = out_dir / f"iteration_{iteration_label:07d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "nurbs_surface.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model.save_ply(scene_dir / "point_cloud.ply")

    return {"scene_dir": str(scene_dir), **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/final"),
    )
    parser.add_argument("--iteration", type=int, default=999999, help="Numeric label for iteration_<N> directory naming.")
    parser.add_argument(
        "--out", type=Path, default=Path("output/osn_gs_scene_chart_atlas"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = export(args.checkpoint, args.cap, args.device, args.out, args.iteration)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
