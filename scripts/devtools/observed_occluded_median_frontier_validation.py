"""Worklog 122 -- Renderer-defined median surface frontier validation (Candidate B only).

Worklog 120 (candidate comparison) and worklog 121 (value-space supplement) are
historical baselines and are not rewritten here. A is retired, C is retired, D is
not viable as stated and is reported only as historical context -- D is never used
as ground truth for B, and worklog 121's directional claim about physical-depth
reordering of D is explicitly NOT carried forward.

This batch validates ONE claim, and a deliberately narrow one:

    NOT  "median depth is the physical first ray/surface hit"
    BUT  "the renderer's own selected visible-surface event provides a coherent,
          closed, sufficiently non-contradictory frontier separating the
          camera-facing observed domain from the behind-surface domain"

Candidate B's decision function is imported and called unmodified. No epsilon, no
tolerance, no threshold sweep, no hybrid, no topology change.
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

from coverage_first_surfel_partition_export import (  # noqa: E402
    PRIMITIVE_SURFEL_2D, checkpoint_primitive, load_primitive_model,
    _rgb_to_f_dc, write_ppm, write_surfel_ply,
)
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402

from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded import frontier_synthetic_contracts, frontier_validation  # noqa: E402
from observed_occluded import query_bank as bank_module  # noqa: E402
from observed_occluded import topology_gap_bank  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    RELEVANCE_NAMES, RELEVANCE_OK, STATE_NAMES, STATE_NON_RELEVANT, STATE_OBSERVED,
    STATE_OCCLUDED, STATE_UNRESOLVED, aggregate_global, distribution, project_queries,
    reconstruct_direct_surfel_intersection_world_point, state_fractions,
)

_ITERATION_DIR = "iteration_0000001"
_SCENE_RGB = (0.07, 0.08, 0.10)
STATE_RGB = {
    STATE_OBSERVED: (0.10, 0.85, 0.35),
    STATE_OCCLUDED: (0.92, 0.18, 0.18),
    STATE_UNRESOLVED: (0.60, 0.60, 0.62),
}

WL119_REPRESENTATIVE_UNION = 785937
WL107_COMPONENT_COUNT = 559989
WL107_SINGLETON_COUNT = 535910

# Fixed a priori -- selection strides only, no decision depends on them.
DISOCCLUSION_VIEW_STRIDE = 10      # 161 views -> 17 anchor views
DISOCCLUSION_PER_VIEW = 200        # deterministic raster-order stride within each anchor view
CLOSURE_SAMPLE_VIEW_STRIDE = 20    # views from which qualitative closure samples are retained
POST_MEDIAN_PIXEL_SAMPLE_STRIDE = 977  # prime stride for the per-pixel mass-fraction sample


def _progress(message: str) -> None:
    print(f"[wl122-frontier] {message}", flush=True)


def _quantiles(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    out = distribution(finite)
    if finite.size:
        for fraction in (0.01, 0.05, 0.25, 0.75, 0.99):
            out[f"p{int(fraction * 100):02d}"] = float(np.quantile(finite, fraction))
    out["non_finite_excluded"] = int(values.size - finite.size)
    return out


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
    parser.add_argument("--wl121-supplemental-npz", type=Path,
                        default=Path("output/confirmed/121_osn_gs_observed_occluded_value_space/value_space_supplemental_bank.npz"))
    parser.add_argument("--wl120-npz", type=Path,
                        default=Path("output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"))
    parser.add_argument("--max-views", type=int, default=0, help="smoke-test only")
    parser.add_argument("--skip-exports", action="store_true")
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 122 (candidate B median frontier validation)",
        "historical_baselines_preserved": {
            "worklog_120_commit": "fdfb8ad60b6233ea8364a09ea3467c18e600a246",
            "worklog_121": "value-space supplement, preserved as historical",
        },
        "claim_under_test": (
            "the renderer's own selected visible-surface event provides a coherent, closed, sufficiently "
            "non-contradictory frontier separating the camera-facing observed domain from the behind-surface "
            "domain -- NOT that median depth is the physical first ray/surface hit"
        ),
        "candidate_D_status": (
            "historical context only. Worklog 121's claim that physical-depth reordering can only INCREASE "
            "D OCCLUDED is NOT carried forward: worklog 121 measured late-front events after REACHED "
            "resolution but never the symmetric early-behind events that may already have reduced T before a "
            "termination event. The direction of a hypothetical correction of D is NOT established, and D is "
            "not used as ground truth for B anywhere in this batch."
        ),
    }

    _progress(f"[1/7] loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint.")
    device = model.device
    total_model_count = len(model)
    cameras, camera_meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, arguments.device,
    )
    if int(arguments.max_views) > 0:
        cameras = cameras[: int(arguments.max_views)]
        camera_meta = {**camera_meta, "smoke_test_max_views": int(arguments.max_views)}
    report["total_trained_surfels"] = total_model_count
    report["camera_meta"] = camera_meta

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe
    from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel

    with torch.no_grad():
        orientation = derive_surface_orientation_from_surfel(model)
        positions_full = model.get_xyz.detach()
        rotation_full = model.get_rotation_matrix.detach()
        tangent_u_full = rotation_full[:, :, 0].contiguous()
        tangent_v_full = rotation_full[:, :, 1].contiguous()
        scaling_full = model.get_scaling.detach()
        scale_u_full = scaling_full[:, 0].contiguous()
        scale_v_full = scaling_full[:, 1].contiguous()

    # ---------------------------------------------------------------- sweep 1
    _progress("[2/7] sweep 1/2: representative maps + deterministic disocclusion bank")
    per_view_rep_cpu: list[torch.Tensor] = []
    ever_representative = torch.zeros((total_model_count,), dtype=torch.bool, device=device)
    disocclusion_points: list[torch.Tensor] = []
    disocclusion_view: list[int] = []
    disocclusion_surfel: list[int] = []
    for index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        representative = package["representative_id"].to(torch.int64)
        ever_representative[torch.unique(representative[representative >= 0])] = True
        per_view_rep_cpu.append(representative.to(torch.int32).cpu())
        if index % DISOCCLUSION_VIEW_STRIDE == 0:
            flat = representative.reshape(-1)
            valid = torch.nonzero(flat >= 0, as_tuple=False).reshape(-1)
            if valid.numel():
                world = reconstruct_direct_surfel_intersection_world_point(
                    flat, package["median_s_u"], package["median_s_v"],
                    positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
                )
                finite = torch.isfinite(world).all(dim=1)
                valid = valid[finite[valid]]
                picks = topology_gap_bank.deterministic_stride(int(valid.numel()), DISOCCLUSION_PER_VIEW, device)
                chosen = valid[picks]
                disocclusion_points.append(world[chosen].clone())
                disocclusion_view.extend([index] * int(chosen.numel()))
                disocclusion_surfel.extend(flat[chosen].tolist())
                del world
        del package
        if index % 40 == 0:
            _progress(f"  sweep1 view {index + 1}/{len(cameras)}")
    representative_union = int(ever_representative.sum().item())
    report["frozen_state_fingerprint"] = {
        "median_surface_representatives_union": representative_union,
        "worklog_119_120_121_reference": WL119_REPRESENTATIVE_UNION,
        "matches": bool(representative_union == WL119_REPRESENTATIVE_UNION),
    }

    _progress("[3/7] frozen WL107/109 topology replay (read-only, unmodified)")
    replay = topology_gap_bank.replay_frozen_topology(
        orientation, [t.to(device=device, dtype=torch.int64) for t in per_view_rep_cpu], progress=_progress,
    )
    report["frozen_topology_replay"] = {
        **replay.stats,
        "matches_component_count": bool(replay.stats["visible_component_count"] == WL107_COMPONENT_COUNT),
        "matches_singleton_count": bool(replay.stats["singleton_surfel_count"] == WL107_SINGLETON_COUNT),
    }
    component_of_surfel = replay.subset_ids
    component_int32 = component_of_surfel.to(torch.int32).contiguous()
    preview_camera = min(cameras, key=lambda c: str(c.image_name))
    region_index, region_meta = bank_module.region_of_surfel(model, preview_camera)
    report["region_anchor_mechanism"] = region_meta

    disocclusion_positions = (
        torch.cat(disocclusion_points, dim=0) if disocclusion_points else torch.zeros((0, 3), device=device)
    )
    disocclusion_view_np = np.asarray(disocclusion_view, dtype=np.int64)
    disocclusion_surfel_np = np.asarray(disocclusion_surfel, dtype=np.int64)
    del disocclusion_points

    # ------------------------------------------- worklog 121 context replay bank
    supplemental_positions = torch.zeros((0, 3), device=device)
    supplemental_meta: dict[str, Any] = {"available": False}
    if arguments.wl121_supplemental_npz.exists():
        stored = np.load(arguments.wl121_supplemental_npz, allow_pickle=True)
        supplemental_positions = torch.as_tensor(stored["positions"], dtype=torch.float32, device=device)
        supplemental_meta = {
            "available": True,
            "queries": int(supplemental_positions.shape[0]),
            "kinds": {k: int(v) for k, v in zip(*np.unique(stored["kind"], return_counts=True))},
            "contexts": int(stored["context_gating_reason"].shape[0]),
            "gating_attribution": {
                topology_gap_bank.GATING_NAMES[int(code)]: int((stored["context_gating_reason"] == code).sum())
                for code in sorted(topology_gap_bank.GATING_NAMES)
                if int((stored["context_gating_reason"] == code).sum())
            },
            "stored_kind": stored["kind"],
            "stored_region": stored["region"],
            "context_gating_reason": stored["context_gating_reason"],
            "context_component_a": stored["context_component_a"],
            "context_component_b": stored["context_component_b"],
            "context_view_index": stored["context_view_index"],
            "context_representative_a": stored["context_representative_a"],
            "context_representative_b": stored["context_representative_b"],
            "stored_global_B": stored["global_B"],
        }
        _progress(f"  worklog 121 supplemental bank loaded: {supplemental_meta['queries']} queries")

    # ---------------------------------------------------------------- sweep 2
    _progress("[4/7] sweep 2/2: exhaustive frontier self-closure + post-median accounting")
    closure = frontier_validation.ClosureAccumulator()
    post_median = frontier_validation.PostMedianAccumulator()
    disocclusion_states = np.full((int(disocclusion_positions.shape[0]), len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    supplemental_states = np.full((int(supplemental_positions.shape[0]), len(cameras)), STATE_NON_RELEVANT, dtype=np.int8)
    supplemental_margin_sum = np.zeros((int(supplemental_positions.shape[0]),), dtype=np.float64)

    for index, camera in enumerate(cameras):
        this_view_rep = per_view_rep_cpu[index].to(device=device, dtype=torch.int64).reshape(-1)
        representative_class = torch.zeros((total_model_count,), dtype=torch.int32, device=device)
        representative_class[ever_representative] = 1
        present = torch.unique(this_view_rep[this_view_rep >= 0])
        representative_class[present] = 2
        package = render_with_query_depth_probe(
            camera, model, query_depths=None,
            primitive_component=component_int32,
            primitive_representative_class=representative_class,
        )
        representative = package["representative_id"].reshape(-1).to(torch.int64)

        frontier_validation.evaluate_frontier_closure_for_view(
            index, camera, package, positions_full, tangent_u_full, tangent_v_full,
            scale_u_full, scale_v_full, closure, region_of_surfel=region_index,
            sample_stride=1 if index % CLOSURE_SAMPLE_VIEW_STRIDE == 0 else 0,
        )
        post_median.accumulate(package, representative, region_index, POST_MEDIAN_PIXEL_SAMPLE_STRIDE)

        median_flat = candidate_b.median_depth_map(package["out_others"]).reshape(-1)
        if int(disocclusion_positions.shape[0]):
            geometry = project_queries(camera, disocclusion_positions)
            disocclusion_states[:, index] = candidate_b.classify_view(geometry, median_flat)["states"].cpu().numpy()
        if int(supplemental_positions.shape[0]):
            geometry = project_queries(camera, supplemental_positions)
            result = candidate_b.classify_view(geometry, median_flat)
            supplemental_states[:, index] = result["states"].cpu().numpy()
            margin = (geometry.depth - result["median_depth"]).cpu().numpy()
            supplemental_margin_sum += np.nan_to_num(margin, nan=0.0)
        del package, representative_class
        if index % 20 == 0:
            _progress(f"  sweep2 view {index + 1}/{len(cameras)}")

    report["frontier_self_closure"] = closure.summary()
    report["frontier_self_closure"]["per_view"] = closure.per_view
    report["post_median_accounting"] = post_median.summary()
    report["post_median_per_pixel_mass_fraction"] = _quantiles(post_median.per_pixel_fraction_samples)
    report["region_level"] = frontier_validation.region_table(closure, post_median, bank_module.REGION_LABELS)

    # --------------------------------------------- post-median semantic attribution
    summary = report["post_median_accounting"]
    mass = summary["contribution_mass_by_category"]
    counts = summary["counts_by_category"]
    post_mass = max(mass["all"], 1e-20)
    report["post_median_semantic_attribution"] = {
        "A_same_component_redundant_representation": {
            "contributor_count": counts["same_component"],
            "count_share": counts["same_component"] / max(counts["all"], 1),
            "contribution_mass": mass["same_component"],
            "mass_share_of_post_median": mass["same_component"] / post_mass,
            "mass_share_of_total_scene_contribution": mass["same_component"] / max(summary["total_accepted_contribution_mass"], 1e-20),
        },
        "B_cross_component_contribution": {
            "contributor_count": counts["cross_component"],
            "count_share": counts["cross_component"] / max(counts["all"], 1),
            "contribution_mass": mass["cross_component"],
            "mass_share_of_post_median": mass["cross_component"] / post_mass,
            "mass_share_of_total_scene_contribution": mass["cross_component"] / max(summary["total_accepted_contribution_mass"], 1e-20),
            "note": "NOT automatically a visible secondary surface -- see the representative-provenance split below.",
        },
        "C_unresolved_provenance": {
            "contributor_count": counts["unresolved_component"],
            "count_share": counts["unresolved_component"] / max(counts["all"], 1),
            "contribution_mass": mass["unresolved_component"],
            "mass_share_of_post_median": mass["unresolved_component"] / post_mass,
        },
        "representative_provenance_of_post_median_contributors": {
            "median_representative_in_THIS_view": {
                "count": counts["representative_this_view"], "mass": mass["representative_this_view"],
                "mass_share_of_post_median": mass["representative_this_view"] / post_mass,
            },
            "median_representative_in_ANOTHER_view_only": {
                "count": counts["representative_other_view"], "mass": mass["representative_other_view"],
                "mass_share_of_post_median": mass["representative_other_view"] / post_mass,
            },
            "never_a_median_representative_anywhere": {
                "count": counts["never_representative"], "mass": mass["never_representative"],
                "mass_share_of_post_median": mass["never_representative"] / post_mass,
            },
        },
        "rho2d_low_pass_branch_share_of_post_median": {
            "count": counts["rho2d_low_pass"], "mass": mass["rho2d_low_pass"],
            "mass_share_of_post_median": mass["rho2d_low_pass"] / post_mass,
        },
    }

    # ------------------------------------------------- cross-view disocclusion
    _progress("[5/7] cross-view disocclusion accounting")
    if int(disocclusion_positions.shape[0]):
        global_states = aggregate_global(disocclusion_states)
        rows = np.arange(disocclusion_states.shape[0])
        source_state = disocclusion_states[rows, disocclusion_view_np]
        occluded_views = (disocclusion_states == STATE_OCCLUDED).sum(axis=1)
        observed_views = (disocclusion_states == STATE_OBSERVED).sum(axis=1)
        relevant_views = (disocclusion_states != STATE_NON_RELEVANT).sum(axis=1)
        hidden_somewhere = occluded_views > 0
        region_of_anchor = region_index[torch.as_tensor(disocclusion_surfel_np, dtype=torch.int64, device=device)].cpu().numpy()
        component_of_anchor = component_of_surfel[torch.as_tensor(disocclusion_surfel_np, dtype=torch.int64, device=device)].cpu().numpy()
        report["cross_view_disocclusion"] = {
            "anchor_count": int(disocclusion_states.shape[0]),
            "anchor_construction": (
                f"every {DISOCCLUSION_VIEW_STRIDE}th training view, deterministic raster-order stride of "
                f"{DISOCCLUSION_PER_VIEW} valid median events; each anchor IS a renderer median-surface event "
                "in its own source view"
            ),
            "source_view_state": state_fractions(source_state),
            "relevant_views_per_anchor": distribution(relevant_views),
            "observed_views_per_anchor": distribution(observed_views),
            "occluded_views_per_anchor": distribution(occluded_views),
            "anchors_hidden_in_at_least_one_view": int(hidden_somewhere.sum()),
            "anchors_hidden_in_at_least_one_view_fraction": float(hidden_somewhere.mean()),
            "global_state_of_those_anchors": state_fractions(global_states[hidden_somewhere]),
            "global_OBSERVED_retention": float((global_states[hidden_somewhere] == STATE_OBSERVED).mean()) if hidden_somewhere.any() else 0.0,
            "global_states_all_anchors": state_fractions(global_states),
            "by_region": {
                label: {
                    "anchors": int((region_of_anchor == region_id).sum()),
                    "hidden_in_at_least_one_view": int((hidden_somewhere & (region_of_anchor == region_id)).sum()),
                    "global_states": state_fractions(global_states[region_of_anchor == region_id]),
                }
                for region_id, label in enumerate(bank_module.REGION_LABELS)
                if int((region_of_anchor == region_id).sum())
            },
            "note": "No view-count threshold anywhere; the frozen aggregation is used unchanged.",
        }
        picks = np.linspace(0, disocclusion_states.shape[0] - 1, num=min(24, disocclusion_states.shape[0])).round().astype(np.int64)
        report["cross_view_disocclusion_cases"] = [
            {
                "anchor_index": int(row),
                "source_view_index": int(disocclusion_view_np[row]),
                "source_view_name": str(getattr(cameras[int(disocclusion_view_np[row])], "image_name", "")),
                "representative_id": int(disocclusion_surfel_np[row]),
                "component_id": int(component_of_anchor[row]),
                "region": bank_module.REGION_LABELS[int(region_of_anchor[row])] if region_of_anchor[row] >= 0 else None,
                "world_position": [float(v) for v in disocclusion_positions[row].tolist()],
                "source_view_B_state": STATE_NAMES[int(source_state[row])],
                "relevant_views": int(relevant_views[row]),
                "OBSERVED_views": int(observed_views[row]),
                "OCCLUDED_views": int(occluded_views[row]),
                "global_state": STATE_NAMES[int(global_states[row])],
            }
            for row in np.unique(picks)
        ]

    # ------------------------------------- worklog 121 true-fragmentation replay
    _progress("[6/7] worklog 121 true-fragmentation replay")
    if supplemental_meta["available"]:
        stored_kind = supplemental_meta.pop("stored_kind")
        stored_region = supplemental_meta.pop("stored_region")
        gating = supplemental_meta.pop("context_gating_reason")
        component_a = supplemental_meta.pop("context_component_a")
        component_b = supplemental_meta.pop("context_component_b")
        context_view = supplemental_meta.pop("context_view_index")
        representative_a = supplemental_meta.pop("context_representative_a")
        representative_b = supplemental_meta.pop("context_representative_b")
        stored_global_b = supplemental_meta.pop("stored_global_B")
        global_b = aggregate_global(supplemental_states)
        context_count = int(gating.shape[0])
        replayed = {}
        for kind in sorted(set(stored_kind.tolist())):
            rows = np.nonzero(stored_kind == kind)[0]
            replayed[str(kind)] = {
                "queries": int(rows.size),
                "B_global": state_fractions(global_b[rows]),
                "worklog_121_stored_B_global": state_fractions(stored_global_b[rows]),
                "identical_to_worklog_121": bool(np.array_equal(global_b[rows], stored_global_b[rows])),
            }
        report["worklog_121_true_fragmentation_replay"] = {
            "contexts": context_count,
            "gating_attribution": supplemental_meta["gating_attribution"],
            "by_query_kind": replayed,
            "B_global_identical_to_worklog_121_overall": bool(np.array_equal(global_b, stored_global_b)),
            "endpoint_closure_reproduction": {
                "worklog_121_endpoint_A": {"OBSERVED": 290, "OCCLUDED": 10},
                "worklog_121_endpoint_B": {"OBSERVED": 296, "OCCLUDED": 4},
                "this_batch_endpoint_A": replayed.get("T1_TOPOLOGY_GAP_ENDPOINT_A", {}).get("B_global", {}).get("counts"),
                "this_batch_endpoint_B": replayed.get("T1_TOPOLOGY_GAP_ENDPOINT_B", {}).get("B_global", {}).get("counts"),
            },
            "midpoint_interpretation_guard": (
                "B(midpoint) = OBSERVED does NOT prove the surface continues through the midpoint -- observed "
                "space may be free space. It is NOT used as a visible-component merge criterion, and topology "
                "is unchanged. The only valid reading is whether B provides evidence that the component "
                "separation is not explained by global occlusion."
            ),
            "component_separation_not_explained_by_global_occlusion": {
                "midpoints": int((stored_kind == "T1_TOPOLOGY_GAP_MIDPOINT").sum()),
                "midpoints_globally_OCCLUDED": int(
                    (global_b[stored_kind == "T1_TOPOLOGY_GAP_MIDPOINT"] == STATE_OCCLUDED).sum()
                ),
            },
            "by_region": {
                label: {
                    "queries": int((stored_region == region_id).sum()),
                    "B_global": state_fractions(global_b[stored_region == region_id]),
                }
                for region_id, label in enumerate(bank_module.REGION_LABELS)
                if int((stored_region == region_id).sum())
            },
        }
        midpoint_start = 2 * context_count
        picks = np.linspace(0, context_count - 1, num=min(20, context_count)).round().astype(np.int64)
        report["true_fragmentation_cases"] = [
            {
                "context_index": int(row),
                "source_view_index": int(context_view[row]),
                "representative_a": int(representative_a[row]),
                "representative_b": int(representative_b[row]),
                "component_a": int(component_a[row]),
                "component_b": int(component_b[row]),
                "gating_reason": topology_gap_bank.GATING_NAMES[int(gating[row])],
                "endpoint_A_B_global": STATE_NAMES[int(global_b[row])],
                "endpoint_B_B_global": STATE_NAMES[int(global_b[context_count + row])],
                "midpoint_B_global": STATE_NAMES[int(global_b[midpoint_start + row])],
                "midpoint_mean_signed_margin": float(
                    supplemental_margin_sum[midpoint_start + row]
                    / max(int((supplemental_states[midpoint_start + row] != STATE_NON_RELEVANT).sum()), 1)
                ),
            }
            for row in np.unique(picks)
        ]
    report["worklog_121_supplemental_bank"] = {
        k: v for k, v in supplemental_meta.items() if not isinstance(v, np.ndarray)
    }

    _progress("[7/7] synthetic known-geometry contracts + exports")
    report["synthetic_frontier_contracts"] = frontier_synthetic_contracts.run_frontier_contracts(device=arguments.device)
    report["qualitative_cases"] = {
        "frontier_closure_exact": closure.closed_samples,
        "frontier_closure_contradiction": closure.contradiction_samples,
    }

    view_paths: dict[str, Any] = {}
    if not arguments.skip_exports and int(supplemental_positions.shape[0]):
        from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
        from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        marker_radius = 0.036220766603946686 * 1.5

        def _write(view_name: str, positions: torch.Tensor, colors: torch.Tensor) -> dict[str, Any]:
            scene_xyz = model.get_xyz.detach()
            scene_rgb = torch.tensor(_SCENE_RGB, device=device).reshape(1, 3).expand(scene_xyz.shape[0], 3)
            count = int(positions.shape[0])
            scaling = torch.cat([model._scaling.detach(), torch.full((count, 2), float(np.log(marker_radius)), device=device)], dim=0)
            rotation = torch.zeros((count, 4), dtype=torch.float32, device=device)
            rotation[:, 0] = 1.0
            rotation = torch.cat([model._rotation.detach(), rotation], dim=0)
            opacity = torch.cat([model._opacity.detach().reshape(-1), torch.full((count,), 4.0, device=device)], dim=0)
            xyz = torch.cat([scene_xyz, positions], dim=0)
            rgb = torch.cat([scene_rgb, colors], dim=0)
            ply_path = output_root / view_name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, xyz, _rgb_to_f_dc(rgb), opacity, scaling, rotation)
            review = TorchGaussianSurfelModel(sh_degree=0, device=str(device))
            review.initialize(positions=xyz, colors=rgb, opacities=torch.sigmoid(opacity).reshape(-1, 1),
                              scales=torch.exp(scaling), rotations=rotation)
            review.active_sh_degree = 0
            with torch.no_grad():
                package = rasterizer.render(preview_camera, review)
            write_ppm(output_root / view_name / "render.ppm", package["render"])
            del review, package
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            return {"point_cloud_ply": str(ply_path), "gaussian_count": written, "marker_points": count}

        with torch.no_grad():
            package = rasterizer.render(preview_camera, model)
        write_ppm(output_root / "ORIGINAL_2DGS_SCENE" / "render.ppm", package["render"])
        ply_path = output_root / "ORIGINAL_2DGS_SCENE" / _ITERATION_DIR / "point_cloud.ply"
        view_paths["ORIGINAL_2DGS_SCENE"] = {
            "point_cloud_ply": str(ply_path),
            "gaussian_count": write_surfel_ply(
                ply_path, model.get_xyz.detach(), model._features_dc.detach()[:, 0, :],
                model._opacity.detach().reshape(-1), model._scaling.detach(), model._rotation.detach(),
            ),
        }
        del package

        global_b = aggregate_global(supplemental_states)
        view_paths["FRAGMENTATION_B_GLOBAL_STATE"] = _write(
            "FRAGMENTATION_B_GLOBAL_STATE", supplemental_positions,
            torch.tensor([STATE_RGB[int(s)] for s in global_b], dtype=torch.float32, device=device),
        )
        if int(disocclusion_positions.shape[0]):
            disocclusion_global = aggregate_global(disocclusion_states)
            occluded_views = (disocclusion_states == STATE_OCCLUDED).sum(axis=1)
            colours = torch.tensor(
                [(0.95, 0.65, 0.15) if occluded_views[row] > 0 else STATE_RGB[int(disocclusion_global[row])]
                 for row in range(disocclusion_states.shape[0])],
                dtype=torch.float32, device=device,
            )
            view_paths["DISOCCLUSION_ANCHORS"] = _write("DISOCCLUSION_ANCHORS", disocclusion_positions, colours)
            view_paths["DISOCCLUSION_ANCHORS"]["colour_encodes"] = (
                "orange = median event hidden (B=OCCLUDED) in >=1 other view; green/red/grey = global B state"
            )
    report["review_exports"] = view_paths

    np.savez_compressed(
        output_root / "median_frontier_validation.npz",
        disocclusion_positions=disocclusion_positions.detach().cpu().numpy(),
        disocclusion_view=disocclusion_view_np, disocclusion_surfel=disocclusion_surfel_np,
        disocclusion_states=disocclusion_states,
        supplemental_states=supplemental_states,
        post_median_fraction_samples=np.asarray(post_median.per_pixel_fraction_samples, dtype=np.float32),
        view_names=np.asarray([str(getattr(c, "image_name", i)) for i, c in enumerate(cameras)]),
    )
    report["artifacts"] = {"npz": str(output_root / "median_frontier_validation.npz")}
    report["total_seconds"] = time.time() - started
    report_path = output_root / "median_frontier_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _progress(f"wrote {report_path} ({report['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
