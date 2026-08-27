"""Worklog 121 -- Worklog 120 value-space comparison and fidelity supplement.

This is NOT a new Observed/Occluded classifier experiment. Worklog 120 is a
historical baseline that must stay reproducible, and this batch treats it that
way: the four candidate decision functions and the frozen global aggregation are
imported and called unmodified, and the run STOPS before interpreting any value
if the historical state arrays do not replay bit-identically.

What it adds:
  * a corrected value/provenance layer over the EXACT original 4,712-query bank;
  * a supplemental query bank built on ACTUAL frozen WL107/109 visible-component
    fragmentation, replacing worklog 120's same-region midpoint approximation;
  * a verified zero-relevant-view control set;
  * three synthetic value contracts (S-D1 / S-C1 / S-B1).

Corrected provenance (all in the reporting layer only):
  * candidate C reports camera-nearest (min t) AND query-nearest (max t)
    blockers separately -- worklog 120's `nearest_blocker_t` was max(t) alone;
  * candidate C's primitive is named the `rho3d geometric footprint derived from
    the canonical alpha cutoff`, never the renderer's complete support;
  * candidate D is named `canonical traversal-order reachability`, its `query_T`
    is the `pre-update traversal transmittance at the recorded resolution
    event`, and the quantity compared against 1e-4 is `T_pre * (1 - alpha)`.
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

from observed_occluded import query_bank as bank_module  # noqa: E402
from observed_occluded import synthetic_contracts, synthetic_value_contracts, topology_gap_bank  # noqa: E402
from observed_occluded.engine import build_geometric_support  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    KIND_R1_ANCHOR_RHO2D, KIND_R1_ANCHOR_RHO3D, KIND_R3_BEHIND, KIND_R4_FRONT,
    RELEVANCE_NAMES, RELEVANCE_OK, STATE_NAMES, STATE_NON_RELEVANT, STATE_OBSERVED,
    STATE_OCCLUDED, STATE_UNRESOLVED, distribution, state_fractions,
)
from observed_occluded.value_diagnostics import (  # noqa: E402
    CANDIDATE_NAMES, CANONICAL_TERMINATION_TEST_T, REASON_NAMES,
    assert_historical_state_replay, bank_replay_check, evaluate_with_values,
)

_ITERATION_DIR = "iteration_0000001"
_SCENE_RGB = (0.07, 0.08, 0.10)
STATE_RGB = {
    STATE_OBSERVED: (0.10, 0.85, 0.35),
    STATE_OCCLUDED: (0.92, 0.18, 0.18),
    STATE_UNRESOLVED: (0.60, 0.60, 0.62),
}
GATING_RGB = {
    topology_gap_bank.GATING_LOCALITY_REJECTED: (0.95, 0.55, 0.15),
    topology_gap_bank.GATING_GEOMETRIC_REJECTED: (0.30, 0.60, 0.95),
    topology_gap_bank.GATING_POSITIVE_EDGE_BUT_SPLIT: (0.85, 0.30, 0.85),
    topology_gap_bank.GATING_UNKNOWN: (0.60, 0.60, 0.60),
}

# Worklog 119/120 reference values for the frozen replay.
WL119_REPRESENTATIVE_UNION = 785937
WL107_COMPONENT_COUNT = 559989
WL107_SINGLETON_COUNT = 535910


def _progress(message: str) -> None:
    print(f"[wl121-value] {message}", flush=True)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    """distribution() plus an explicit quantile table (directive section 14)."""

    finite = _finite(values)
    out = distribution(finite)
    if finite.size:
        for fraction in (0.01, 0.05, 0.25, 0.75, 0.99):
            out[f"p{int(fraction * 100):02d}"] = float(np.quantile(finite, fraction))
    out["non_finite_excluded"] = int(np.asarray(values).size - finite.size)
    return out


def relevant_mask(result) -> np.ndarray:
    return result.relevance_code == RELEVANCE_OK


def value_by_state(result, candidate: str, field: str) -> dict[str, Any]:
    states = result.per_view_states[candidate]
    values = result.values[field]
    out: dict[str, Any] = {}
    for code, name in STATE_NAMES.items():
        if code == STATE_NON_RELEVANT:
            continue
        out[name] = _quantiles(values[states == code])
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
    parser.add_argument("--wl120-npz", type=Path,
                        default=Path("output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"))
    parser.add_argument("--max-views", type=int, default=0, help="smoke-test only")
    parser.add_argument("--skip-exports", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--allow-replay-failure", action="store_true",
                        help="smoke-test only: continue past the baseline gate (never for a reported run)")
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_bytes = int(arguments.chunk_mib) * 1024 * 1024
    report: dict[str, Any] = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 121 (worklog 120 value-space supplement)",
        "historical_baseline_commit": "fdfb8ad60b6233ea8364a09ea3467c18e600a246",
        "checkpoint": str(arguments.checkpoint),
        "corrected_terminology": {
            "candidate_B": "renderer-defined median-surface event under the canonical pre-update T > 0.5 rule",
            "candidate_C": "rho3d geometric footprint derived from the canonical alpha cutoff",
            "candidate_D": "canonical traversal-order reachability",
            "query_T": "pre-update traversal transmittance at the recorded resolution event",
            "termination_test_T": "T_pre * (1 - alpha), the quantity the canonical kernel compares against 1e-4",
        },
    }

    _progress(f"[1/8] loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint.")
    device = model.device
    total_model_count = len(model)
    uncertain_count = int(model.is_uncertain.reshape(-1).to(torch.bool).sum())
    cameras, camera_meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, arguments.device,
    )
    if int(arguments.max_views) > 0:
        cameras = cameras[: int(arguments.max_views)]
        camera_meta = {**camera_meta, "smoke_test_max_views": int(arguments.max_views)}
    report["total_trained_surfels"] = total_model_count
    report["uncertain_surfels_in_checkpoint"] = uncertain_count
    report["camera_meta"] = camera_meta

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe
    from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel
    from observed_occluded.shared import reconstruct_direct_surfel_intersection_world_point

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
    _progress("[2/8] sweep 1/2: per-view representative maps (frozen renderer)")
    per_view_rep_cpu: list[torch.Tensor] = []
    ever_representative = torch.zeros((total_model_count,), dtype=torch.bool, device=device)
    for index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        rep = package["representative_id"].to(torch.int64)
        ever_representative[torch.unique(rep[rep >= 0])] = True
        per_view_rep_cpu.append(rep.to(torch.int32).cpu())
        del package
        if index % 40 == 0:
            _progress(f"  sweep1 view {index + 1}/{len(cameras)}")
    representative_union = int(ever_representative.sum().item())
    report["frozen_state_fingerprint"] = {
        "median_surface_representatives_union": representative_union,
        "worklog_119_120_reference_value": WL119_REPRESENTATIVE_UNION,
        "matches": bool(representative_union == WL119_REPRESENTATIVE_UNION),
    }
    _progress(f"  representative union = {representative_union} (WL119/120 = {WL119_REPRESENTATIVE_UNION})")

    # ------------------------------------------------- frozen topology replay
    _progress("[3/8] frozen WL107/109 topology replay (read-only, unmodified)")
    replay = topology_gap_bank.replay_frozen_topology(
        orientation, [t.to(device=device, dtype=torch.int64) for t in per_view_rep_cpu], progress=_progress,
    )
    report["frozen_topology_replay"] = {
        **replay.stats,
        "worklog_107_109_reference_component_count": WL107_COMPONENT_COUNT,
        "worklog_107_109_reference_singleton_count": WL107_SINGLETON_COUNT,
        "matches_component_count": bool(replay.stats["visible_component_count"] == WL107_COMPONENT_COUNT),
        "matches_singleton_count": bool(replay.stats["singleton_surfel_count"] == WL107_SINGLETON_COUNT),
    }
    component_of_surfel = replay.subset_ids

    preview_camera = min(cameras, key=lambda c: str(c.image_name))
    region_index, region_meta = bank_module.region_of_surfel(model, preview_camera)

    # ---------------------------------------------------------------- sweep 2
    _progress("[4/8] sweep 2/2: cross-component raster adjacency contexts (true fragmentation)")
    context_chunks: list[dict[str, torch.Tensor]] = []
    total_cross_component = 0
    for index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        rep = package["representative_id"].to(torch.int64)
        event_world = reconstruct_direct_surfel_intersection_world_point(
            rep, package["median_s_u"], package["median_s_v"],
            positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
        ).reshape(rep.shape[0], rep.shape[1], 3)
        found = topology_gap_bank.collect_cross_component_contexts(index, rep, component_of_surfel, event_world)
        available = int(found["view_index"].shape[0])
        total_cross_component += available
        if available:
            picks = topology_gap_bank.deterministic_stride(available, topology_gap_bank.PER_VIEW_CONTEXT_CAP, device)
            context_chunks.append({key: value[picks] for key, value in found.items()})
        del package, event_world, found
        if index % 40 == 0:
            _progress(f"  sweep2 view {index + 1}/{len(cameras)} (cross-component adjacencies so far: {total_cross_component})")

    merged = {key: torch.cat([chunk[key] for chunk in context_chunks], dim=0) for key in context_chunks[0]} if context_chunks else {}
    del context_chunks
    if not merged:
        raise RuntimeError("no cross-component raster adjacency contexts found -- cannot build the supplemental bank")

    finite_world = torch.isfinite(merged["world_a"]).all(dim=1) & torch.isfinite(merged["world_b"]).all(dim=1)
    merged = {key: value[finite_world] for key, value in merged.items()}
    gating = topology_gap_bank.attribute_gating(
        merged["representative_a"], merged["representative_b"], total_model_count, replay
    )
    merged["gating_reason"] = gating
    merged["region"] = region_index[merged["representative_a"]]

    selected_rows: list[torch.Tensor] = []
    per_region_available: dict[str, int] = {}
    for region_id, label in enumerate(bank_module.REGION_LABELS):
        rows = torch.nonzero(merged["region"] == region_id, as_tuple=False).reshape(-1)
        per_region_available[label] = int(rows.numel())
        if rows.numel() == 0:
            continue
        picks = topology_gap_bank.deterministic_stride(int(rows.numel()), topology_gap_bank.CONTEXTS_PER_REGION, device)
        selected_rows.append(rows[picks])
    selected = torch.cat(selected_rows) if selected_rows else torch.zeros((0,), dtype=torch.int64, device=device)
    selected = torch.sort(selected).values
    contexts_np = {
        key: (value[selected].detach().cpu().numpy() if value.ndim == 1 else value[selected].detach().cpu().numpy())
        for key, value in merged.items()
    }
    del merged

    scene_lower = positions_full.min(dim=0).values
    scene_upper = positions_full.max(dim=0).values
    scene_centre = (scene_lower + scene_upper) * 0.5
    scene_extent = float((scene_upper - scene_lower).max().item())
    _progress("[5/8] verified zero-relevant-view controls")
    controls, control_meta = topology_gap_bank.build_verified_out_of_frustum_controls(
        cameras, scene_centre, scene_extent, device
    )
    supplemental_bank, supplemental_sidecar = topology_gap_bank.build_supplemental_bank(
        contexts_np, controls, region_index, device
    )
    report["supplemental_bank"] = {
        "total_cross_component_raster_adjacencies_observed": total_cross_component,
        "per_view_stride_cap": topology_gap_bank.PER_VIEW_CONTEXT_CAP,
        "contexts_per_region_target": topology_gap_bank.CONTEXTS_PER_REGION,
        "contexts_available_per_region": per_region_available,
        "contexts_selected": int(selected.numel()),
        "queries": len(supplemental_bank),
        "by_kind": {k: int(v) for k, v in zip(*np.unique(np.asarray(supplemental_bank.kind), return_counts=True))},
        "gating_attribution": {
            topology_gap_bank.GATING_NAMES[code]: int((contexts_np["gating_reason"] == code).sum())
            for code in sorted(topology_gap_bank.GATING_NAMES)
        },
        "verified_out_of_frustum_controls": control_meta,
        "construction": (
            "observed raster-local 4-neighbour adjacency whose two representatives lie in DIFFERENT final "
            "WL107/109 visible components; never nearest-3D pairs, never region labels, never proximity"
        ),
    }
    report["region_anchor_mechanism"] = region_meta

    # -------------------------------------------------- ORIGINAL BANK + GATE
    _progress("[6/8] rebuilding the EXACT worklog 120 bank and replaying its states")
    original_bank, original_bank_report = bank_module.build_bank(model, cameras, region_index, progress=None)
    reference = np.load(arguments.wl120_npz, allow_pickle=True) if arguments.wl120_npz.exists() else None
    if reference is None:
        raise FileNotFoundError(f"worklog 120 reference artifact not found: {arguments.wl120_npz}")
    report["baseline_bank_replay"] = bank_replay_check(original_bank, reference)
    _progress(f"  bank replay gate: {report['baseline_bank_replay']['gate']}")

    support = build_geometric_support(model)
    original = evaluate_with_values(
        model, cameras, original_bank.positions, support=support,
        component_of_surfel=component_of_surfel, source_surfel=original_bank.source_surfel,
        chunk_bytes=chunk_bytes, progress=_progress,
    )
    report["baseline_state_replay"] = assert_historical_state_replay(original, reference)
    report["value_pass_diagnostics"] = original.diagnostics
    gate_ok = (
        report["baseline_bank_replay"]["gate"] == "PASS"
        and report["baseline_state_replay"]["gate"] == "PASS"
        and original.diagnostics["candidate_C_value_vs_decision_mismatches"] == 0
    )
    report["baseline_replay_gate"] = "PASS" if gate_ok else "FAIL"
    _progress(f"  state replay gate: {report['baseline_state_replay']['gate']}")
    if not gate_ok and not arguments.allow_replay_failure:
        report["stopped"] = (
            "Historical baseline did not reproduce bit-identically. Per directive section 2 the "
            "value-comparison interpretation is NOT continued; attribute the replay failure first."
        )
        (output_root / "observed_occluded_value_space_comparison_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise SystemExit("BASELINE REPLAY GATE FAILED -- see report; value interpretation deliberately not produced.")

    # ------------------------------------------------------ supplemental eval
    _progress("[7/8] evaluating the supplemental true-fragmentation bank")
    supplemental = evaluate_with_values(
        model, cameras, supplemental_bank.positions, support=support,
        component_of_surfel=component_of_surfel, source_surfel=supplemental_bank.source_surfel,
        chunk_bytes=chunk_bytes, progress=_progress,
    )

    _progress("[8/8] synthetic value contracts + metrics")
    report["worklog_120_synthetic_contracts_replayed"] = synthetic_contracts.contract_summary(
        synthetic_contracts.run_contracts(device=arguments.device)
    )
    report["synthetic_value_contracts"] = synthetic_value_contracts.run_value_contracts(device=arguments.device)

    # ================================================================ METRICS
    def candidate_state_counts(result) -> dict[str, Any]:
        return {
            name: {
                "per_view_pair": state_fractions(result.per_view_states[name]),
                "global": state_fractions(result.global_states[name]),
            }
            for name in CANDIDATE_NAMES
        }

    report["state_space"] = {
        "original_bank": candidate_state_counts(original),
        "supplemental_bank": candidate_state_counts(supplemental),
    }

    kinds = np.asarray(original_bank.kind)
    anchor_mask = (kinds == KIND_R1_ANCHOR_RHO3D) | (kinds == KIND_R1_ANCHOR_RHO2D)
    anchor_rows = np.nonzero(anchor_mask)[0]
    anchor_cols = original_bank.source_view[anchor_rows]
    relevant = relevant_mask(original)

    # ------------------------------------------------------------- A values
    a_states = original.per_view_states["A"]
    b_states = original.per_view_states["B"]
    c_states = original.per_view_states["C"]
    d_states = original.per_view_states["D"]
    signed_depth_delta_a = original.query_depth - original.values["A_event_depth"]
    signed_median_margin = original.query_depth - original.values["B_median_depth"]

    report["candidate_A_values"] = {
        "primitive": "worklog 119 G2 direct median-surfel local intersection (unchanged)",
        "signed_depth_delta_A_by_state": {
            STATE_NAMES[code]: _quantiles(signed_depth_delta_a[a_states == code])
            for code in (STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED)
        },
        "world_hit_distance_A_by_state": value_by_state(original, "A", "A_hit_distance"),
        "event_depth_A_by_state": value_by_state(original, "A", "A_event_depth"),
        "branch_composition_by_state": {
            STATE_NAMES[code]: {
                "rho3d": int(((a_states == code) & (original.values["A_event_branch"] == 0)).sum()),
                "rho2d": int(((a_states == code) & (original.values["A_event_branch"] == 1)).sum()),
                "no_event": int(((a_states == code) & (original.values["A_event_branch"] == -1)).sum()),
            }
            for code in (STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED)
        },
        "disagreement_classes": {
            "A_UNRESOLVED_and_B_OBSERVED": {
                "pairs": int(((a_states == STATE_UNRESOLVED) & (b_states == STATE_OBSERVED)).sum()),
                "B_signed_median_margin": _quantiles(signed_median_margin[(a_states == STATE_UNRESOLVED) & (b_states == STATE_OBSERVED)]),
            },
            "A_UNRESOLVED_and_D_OBSERVED": {
                "pairs": int(((a_states == STATE_UNRESOLVED) & (d_states == STATE_OBSERVED)).sum()),
                "D_traversal_T_pre": _quantiles(original.values["D_traversal_T_pre"][(a_states == STATE_UNRESOLVED) & (d_states == STATE_OBSERVED)]),
            },
            "A_OCCLUDED_and_B_OCCLUDED": {
                "pairs": int(((a_states == STATE_OCCLUDED) & (b_states == STATE_OCCLUDED)).sum()),
                "signed_depth_delta_A": _quantiles(signed_depth_delta_a[(a_states == STATE_OCCLUDED) & (b_states == STATE_OCCLUDED)]),
            },
            "A_OCCLUDED_and_D_OBSERVED": {
                "pairs": int(((a_states == STATE_OCCLUDED) & (d_states == STATE_OBSERVED)).sum()),
                "D_traversal_T_pre": _quantiles(original.values["D_traversal_T_pre"][(a_states == STATE_OCCLUDED) & (d_states == STATE_OBSERVED)]),
                "signed_depth_delta_A": _quantiles(signed_depth_delta_a[(a_states == STATE_OCCLUDED) & (d_states == STATE_OBSERVED)]),
            },
        },
    }

    # ------------------------------------------------------------- B values
    def margin_by(group_values: np.ndarray, labels) -> dict[str, Any]:
        out = {}
        for index, label in labels:
            mask = np.zeros_like(relevant)
            mask[group_values == index, :] = True
            out[label] = _quantiles(signed_median_margin[mask & relevant])
        return out

    anchor_margin = signed_median_margin[anchor_rows, anchor_cols]
    anchor_depth = original.query_depth[anchor_rows, anchor_cols]
    report["candidate_B_values"] = {
        "primitive": "renderer-defined median-surface event under the canonical pre-update T > 0.5 rule",
        "signed_median_margin_by_state": {
            STATE_NAMES[code]: _quantiles(signed_median_margin[b_states == code])
            for code in (STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED)
        },
        "signed_median_margin_by_query_kind": {
            kind: _quantiles(signed_median_margin[kinds == kind][relevant[kinds == kind]])
            for kind in sorted(set(kinds.tolist()))
        },
        "signed_median_margin_by_region": margin_by(original_bank.region, list(enumerate(bank_module.REGION_LABELS))),
        "R1_source_anchor_boundary_analysis": {
            "anchors": int(anchor_rows.size),
            "absolute_delta": _quantiles(np.abs(anchor_margin)),
            "signed_delta": _quantiles(anchor_margin),
            "relative_delta": _quantiles(anchor_margin / np.maximum(anchor_depth, 1e-9)),
            "exactly_zero": int((anchor_margin == 0).sum()),
            "strictly_positive_so_B_says_OCCLUDED": int((anchor_margin > 0).sum()),
            "strictly_negative": int((anchor_margin < 0).sum()),
            "note": (
                "Preserved from worklog 120 and NOT repaired: candidate B's dividing surface has zero "
                "thickness, so an anchor that IS the median event lands on either side by float32 round-off. "
                "No tolerance is introduced anywhere in this batch."
            ),
        },
    }

    # ------------------------------------------------------------- C values
    c_blocked = c_states == STATE_OCCLUDED
    anchor_c_states = c_states[anchor_rows, anchor_cols]
    anchor_blocked = anchor_c_states == STATE_OCCLUDED
    report["candidate_C_values"] = {
        "primitive": "rho3d geometric footprint derived from the canonical alpha cutoff (NOT the renderer's complete contribution support)",
        "primitive_caveat": (
            "Canonical acceptance is rho = min(rho3d, rho2d); a rho2d low-pass event can be accepted outside "
            "the rho3d footprint, so this footprint is a strict geometric subset of what the renderer composites."
        ),
        "camera_nearest_blocker_t": _quantiles(original.values["C_camera_nearest_blocker_t"][c_blocked]),
        "query_nearest_blocker_t": _quantiles(original.values["C_query_nearest_blocker_t"][c_blocked]),
        "camera_nearest_blocker_world_gap": _quantiles(original.values["C_camera_nearest_blocker_world_gap"][c_blocked]),
        "query_nearest_blocker_world_gap": _quantiles(original.values["C_query_nearest_blocker_world_gap"][c_blocked]),
        "blocker_region_thickness": _quantiles(original.values["C_blocker_region_thickness"][c_blocked]),
        "blocker_count": _quantiles(original.values["C_blocker_count"][c_blocked]),
        "max_blocker_opacity": _quantiles(original.values["C_max_blocker_opacity"][c_blocked]),
        "opacity_of_camera_nearest_blocker": _quantiles(original.values["C_camera_nearest_blocker_opacity"][c_blocked]),
        "opacity_of_query_nearest_blocker": _quantiles(original.values["C_query_nearest_blocker_opacity"][c_blocked]),
        "worklog_120_reported_value_was": "MAX(t) only, i.e. the query_nearest_blocker_t column above",
        "source_view_R1_anchors": {
            "anchors": int(anchor_rows.size),
            "classified_OCCLUDED_in_their_own_source_view": int(anchor_blocked.sum()),
            "fraction": float(anchor_blocked.mean()) if anchor_rows.size else 0.0,
            "camera_nearest_blocker_world_gap": _quantiles(original.values["C_camera_nearest_blocker_world_gap"][anchor_rows, anchor_cols][anchor_blocked]),
            "query_nearest_blocker_world_gap": _quantiles(original.values["C_query_nearest_blocker_world_gap"][anchor_rows, anchor_cols][anchor_blocked]),
            "blocker_region_thickness": _quantiles(original.values["C_blocker_region_thickness"][anchor_rows, anchor_cols][anchor_blocked]),
            "blocker_count": _quantiles(original.values["C_blocker_count"][anchor_rows, anchor_cols][anchor_blocked]),
            "same_component_blocker_count": _quantiles(original.values["C_same_component_blocker_count"][anchor_rows, anchor_cols][anchor_blocked]),
            "same_component_fraction_of_blockers": _quantiles(
                original.values["C_same_component_blocker_count"][anchor_rows, anchor_cols][anchor_blocked]
                / np.maximum(original.values["C_blocker_count"][anchor_rows, anchor_cols][anchor_blocked], 1)
            ),
            "camera_nearest_blocker_in_same_component": int(
                (original.values["C_camera_nearest_blocker_component"][anchor_rows, anchor_cols][anchor_blocked]
                 == component_of_surfel[torch.as_tensor(original_bank.source_surfel[anchor_rows], dtype=torch.int64, device=device)].cpu().numpy()[anchor_blocked]).sum()
            ),
            "attribution_note": (
                "Diagnostic only. Same-component blockers are NEVER ignored by the decision and no "
                "self-occlusion tolerance is introduced."
            ),
        },
    }

    # ------------------------------------------------------------- D values
    reason = original.values["D_resolution_reason"]
    terminated_mask = reason == 2
    test_t = original.values["D_termination_test_T"]
    report["candidate_D_values"] = {
        "primitive": "canonical traversal-order reachability (canonical test_T < 1e-4 termination event)",
        "resolution_reason_counts": {
            REASON_NAMES[code]: int((reason[relevant] == code).sum()) for code in sorted(REASON_NAMES)
        },
        "traversal_T_pre_by_state": value_by_state(original, "D", "D_traversal_T_pre"),
        "traversal_T_pre_by_resolution_reason": {
            REASON_NAMES[code]: _quantiles(original.values["D_traversal_T_pre"][reason == code])
            for code in sorted(REASON_NAMES) if int((reason == code).sum())
        },
        "resolution_event_depth": _quantiles(original.values["D_resolution_event_depth"][relevant]),
        "accepted_prefix_count_by_state": value_by_state(original, "D", "D_prefix_count"),
        "termination_events": {
            "count": int(terminated_mask.sum()),
            "termination_alpha": _quantiles(original.values["D_termination_alpha"][terminated_mask]),
            "termination_T_pre": _quantiles(original.values["D_traversal_T_pre"][terminated_mask]),
            "termination_test_T": _quantiles(test_t[terminated_mask]),
            "contract_test_T_below_1e-4_violations": int(original.diagnostics["D_termination_contract_violations"]),
            "T_pre_itself_below_1e-4_count": int((original.values["D_traversal_T_pre"][terminated_mask] < CANONICAL_TERMINATION_TEST_T).sum()),
            "T_pre_itself_below_1e-4_fraction": float((original.values["D_traversal_T_pre"][terminated_mask] < CANONICAL_TERMINATION_TEST_T).mean()) if terminated_mask.any() else 0.0,
            "correction_note": (
                "The canonical condition is test_T = T_pre * (1 - alpha) < 1e-4. T_pre itself is NOT bounded "
                "by 1e-4; worklog 120's phrasing implied it was."
            ),
        },
    }

    # --------------------------------------------- D depth-order fidelity audit
    late_front = original.values["D_late_front_count"]
    inversions = original.values["pixel_inversion_count"]
    resolved_reached = reason == 1
    report["candidate_D_depth_order_fidelity"] = {
        "accounting_scope": "FULL -- every relevant (query, view) pair, no subset was needed",
        "relevant_pairs": int(relevant.sum()),
        "pixel_inversion_count": _quantiles(inversions[relevant]),
        "pixels_with_at_least_one_inversion": int((inversions[relevant] >= 1).sum()),
        "fraction_of_relevant_pairs_whose_pixel_has_an_inversion": float((inversions[relevant] >= 1).mean()),
        "pixel_inversion_count_where_nonzero": _quantiles(inversions[relevant & (inversions >= 1)]),
        "pixel_max_backward_jump": _quantiles(original.values["pixel_max_backward_jump"][relevant]),
        "pixel_max_backward_jump_where_inverted": _quantiles(original.values["pixel_max_backward_jump"][relevant & (inversions >= 1)]),
        "late_front_event_count": {
            "scope": "queries resolved by REACHED_ACCEPTED_EVENT",
            "pairs": int(resolved_reached.sum()),
            "distribution": _quantiles(late_front[resolved_reached]),
            "pairs_with_at_least_one_late_front_event": int((late_front[resolved_reached] >= 1).sum()),
            "fraction_with_at_least_one": float((late_front[resolved_reached] >= 1).mean()) if resolved_reached.any() else 0.0,
            "meaning": (
                "accepted events processed AFTER the query resolved whose own per-pixel depth is still in "
                "front of the query -- the direct gap between 'first traversal event reaching query depth' "
                "and 'all geometrically front-of-query accepted events have been seen'"
            ),
        },
    }

    # ------------------------------------------------ B vs D value comparison
    def disagreement_table(left: str, right: str, left_state: int, right_state: int) -> dict[str, Any]:
        rows = np.nonzero((original.global_states[left] == left_state) & (original.global_states[right] == right_state))[0]
        if rows.size == 0:
            return {"queries": 0}
        mask = np.zeros_like(relevant)
        mask[rows, :] = True
        mask = mask & relevant
        out: dict[str, Any] = {
            "queries": int(rows.size),
            "query_view_pairs": int(mask.sum()),
            "query_depth": _quantiles(original.query_depth[mask]),
            "B_median_depth": _quantiles(original.values["B_median_depth"][mask]),
            "B_signed_median_margin": _quantiles(signed_median_margin[mask]),
            "D_traversal_T_pre": _quantiles(original.values["D_traversal_T_pre"][mask]),
            "D_resolution_event_depth": _quantiles(original.values["D_resolution_event_depth"][mask]),
            "D_accepted_prefix_count": _quantiles(original.values["D_prefix_count"][mask]),
            "D_termination_test_T": _quantiles(test_t[mask]),
            "D_resolution_reason_counts": {
                REASON_NAMES[code]: int((reason[mask] == code).sum()) for code in sorted(REASON_NAMES)
            },
            "D_late_front_event_count": _quantiles(late_front[mask & resolved_reached]),
            "pixel_inversion_count": _quantiles(inversions[mask]),
            "by_query_kind": {},
            "by_region": {},
        }
        for kind in sorted(set(kinds[rows].tolist())):
            kind_rows = rows[kinds[rows] == kind]
            kind_mask = np.zeros_like(relevant)
            kind_mask[kind_rows, :] = True
            kind_mask = kind_mask & relevant
            out["by_query_kind"][kind] = {
                "queries": int(kind_rows.size),
                "B_signed_median_margin": _quantiles(signed_median_margin[kind_mask]),
                "D_traversal_T_pre": _quantiles(original.values["D_traversal_T_pre"][kind_mask]),
            }
        for region_id, label in enumerate(bank_module.REGION_LABELS):
            region_rows = rows[original_bank.region[rows] == region_id]
            if region_rows.size == 0:
                continue
            region_mask = np.zeros_like(relevant)
            region_mask[region_rows, :] = True
            region_mask = region_mask & relevant
            out["by_region"][label] = {
                "queries": int(region_rows.size),
                "B_signed_median_margin": _quantiles(signed_median_margin[region_mask]),
                "D_traversal_T_pre": _quantiles(original.values["D_traversal_T_pre"][region_mask]),
            }
        return out

    report["B_vs_D_value_space"] = {
        "state_relationship": {
            "B_OCCLUDED_and_D_OBSERVED": int(((original.global_states["B"] == STATE_OCCLUDED) & (original.global_states["D"] == STATE_OBSERVED)).sum()),
            "B_OBSERVED_and_D_OCCLUDED": int(((original.global_states["B"] == STATE_OBSERVED) & (original.global_states["D"] == STATE_OCCLUDED)).sum()),
            "worklog_120_reported": {"B_OCCLUDED_and_D_OBSERVED": 654, "reverse": 0},
        },
        "B_OCCLUDED_D_OBSERVED": disagreement_table("B", "D", STATE_OCCLUDED, STATE_OBSERVED),
        "per_view_pair_disagreement": {
            "B_OCCLUDED_D_OBSERVED_pairs": int(((b_states == STATE_OCCLUDED) & (d_states == STATE_OBSERVED)).sum()),
            "B_OBSERVED_D_OCCLUDED_pairs": int(((b_states == STATE_OBSERVED) & (d_states == STATE_OCCLUDED)).sum()),
            "B_signed_median_margin_where_B_OCC_D_OBS": _quantiles(signed_median_margin[(b_states == STATE_OCCLUDED) & (d_states == STATE_OBSERVED)]),
            "D_T_pre_where_B_OCC_D_OBS": _quantiles(original.values["D_traversal_T_pre"][(b_states == STATE_OCCLUDED) & (d_states == STATE_OBSERVED)]),
            "B_signed_median_margin_where_B_OBS_D_OCC": _quantiles(signed_median_margin[(b_states == STATE_OBSERVED) & (d_states == STATE_OCCLUDED)]),
            "D_T_pre_where_B_OBS_D_OCC": _quantiles(original.values["D_traversal_T_pre"][(b_states == STATE_OBSERVED) & (d_states == STATE_OCCLUDED)]),
        },
        "interpretation_guard": (
            "Reported as two different renderer-level questions -- B is the renderer-defined visible-surface "
            "frontier, D is canonical traversal termination/reachability. No threshold between them is "
            "searched for, and no claim is made that a true boundary lies between T=0.5 and test_T=1e-4."
        ),
    }

    # ---------------------------------------------------- A vs B / C vs B
    a_unresolved = a_states == STATE_UNRESOLVED
    cb_mask = (b_states == STATE_OBSERVED) & (c_states == STATE_OCCLUDED)
    report["A_vs_B_and_C_vs_B"] = {
        "A_vs_B": {
            "A_UNRESOLVED_pairs": int(a_unresolved.sum()),
            "B_signed_median_margin_on_A_UNRESOLVED": _quantiles(signed_median_margin[a_unresolved]),
            "B_state_composition_on_A_UNRESOLVED": {
                STATE_NAMES[code]: int((a_unresolved & (b_states == code)).sum())
                for code in (STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED)
            },
        },
        "C_vs_B": {
            "B_OBSERVED_and_C_OCCLUDED_pairs": int(cb_mask.sum()),
            "B_signed_median_margin": _quantiles(signed_median_margin[cb_mask]),
            "C_camera_nearest_blocker_t": _quantiles(original.values["C_camera_nearest_blocker_t"][cb_mask]),
            "C_query_nearest_blocker_t": _quantiles(original.values["C_query_nearest_blocker_t"][cb_mask]),
            "C_camera_nearest_blocker_world_gap": _quantiles(original.values["C_camera_nearest_blocker_world_gap"][cb_mask]),
            "C_query_nearest_blocker_world_gap": _quantiles(original.values["C_query_nearest_blocker_world_gap"][cb_mask]),
            "C_blocker_region_thickness": _quantiles(original.values["C_blocker_region_thickness"][cb_mask]),
            "C_blocker_count": _quantiles(original.values["C_blocker_count"][cb_mask]),
            "C_same_component_blocker_count": _quantiles(original.values["C_same_component_blocker_count"][cb_mask]),
        },
    }

    # ------------------------------------------------- supplemental gap table
    supplemental_relevant = relevant_mask(supplemental)
    supplemental_kinds = np.asarray(supplemental_bank.kind)
    supplemental_margin = supplemental.query_depth - supplemental.values["B_median_depth"]
    gap_table: dict[str, Any] = {}
    for kind in sorted(set(supplemental_kinds.tolist())):
        rows = np.nonzero(supplemental_kinds == kind)[0]
        mask = np.zeros_like(supplemental_relevant)
        mask[rows, :] = True
        mask = mask & supplemental_relevant
        entry: dict[str, Any] = {
            "queries": int(rows.size),
            "relevant_pairs": int(mask.sum()),
            "global_states": {
                name: state_fractions(supplemental.global_states[name][rows]) for name in CANDIDATE_NAMES
            },
            "B_signed_median_margin": _quantiles(supplemental_margin[mask]),
            "C_camera_nearest_blocker_world_gap": _quantiles(supplemental.values["C_camera_nearest_blocker_world_gap"][mask]),
            "C_query_nearest_blocker_world_gap": _quantiles(supplemental.values["C_query_nearest_blocker_world_gap"][mask]),
            "C_blocker_count": _quantiles(supplemental.values["C_blocker_count"][mask]),
            "D_traversal_T_pre": _quantiles(supplemental.values["D_traversal_T_pre"][mask]),
            "D_late_front_count": _quantiles(supplemental.values["D_late_front_count"][mask]),
            "A_hit_distance": _quantiles(supplemental.values["A_hit_distance"][mask]),
        }
        gap_table[kind] = entry
    region_gap_table: dict[str, Any] = {}
    for region_id, label in enumerate(bank_module.REGION_LABELS):
        rows = np.nonzero(supplemental_bank.region == region_id)[0]
        if rows.size == 0:
            continue
        mask = np.zeros_like(supplemental_relevant)
        mask[rows, :] = True
        mask = mask & supplemental_relevant
        region_gap_table[label] = {
            "queries": int(rows.size),
            "global_states": {name: state_fractions(supplemental.global_states[name][rows]) for name in CANDIDATE_NAMES},
            "B_signed_median_margin": _quantiles(supplemental_margin[mask]),
            "C_blocker_region_thickness": _quantiles(supplemental.values["C_blocker_region_thickness"][mask]),
            "D_traversal_T_pre": _quantiles(supplemental.values["D_traversal_T_pre"][mask]),
        }
    control_rows = np.nonzero(supplemental_kinds == topology_gap_bank.KIND_VERIFIED_OUT_OF_FRUSTUM)[0]
    report["supplemental_gap_results"] = {
        "by_query_kind": gap_table,
        "by_region": region_gap_table,
        "by_gating_reason": {
            topology_gap_bank.GATING_NAMES[code]: {
                "contexts": int((contexts_np["gating_reason"] == code).sum()),
                "midpoint_global_states": {
                    name: state_fractions(
                        supplemental.global_states[name][
                            supplemental_sidecar["midpoint_rows"][0] + np.nonzero(contexts_np["gating_reason"] == code)[0]
                        ]
                    )
                    for name in CANDIDATE_NAMES
                },
            }
            for code in sorted(topology_gap_bank.GATING_NAMES)
            if int((contexts_np["gating_reason"] == code).sum())
        },
        "verified_out_of_frustum_controls": {
            "queries": int(control_rows.size),
            "relevant_views_per_control": distribution((supplemental.relevance_code[control_rows] == RELEVANCE_OK).sum(axis=1)),
            "global_states": {name: state_fractions(supplemental.global_states[name][control_rows]) for name in CANDIDATE_NAMES},
            "all_UNRESOLVED": bool(
                all((supplemental.global_states[name][control_rows] == STATE_UNRESOLVED).all() for name in CANDIDATE_NAMES)
            ) if control_rows.size else None,
        },
    }

    report["relevant_view_contract"] = {
        "original_bank": {
            RELEVANCE_NAMES[code]: int((original.relevance_code == code).sum()) for code in sorted(RELEVANCE_NAMES)
        },
        "supplemental_bank": {
            RELEVANCE_NAMES[code]: int((supplemental.relevance_code == code).sum()) for code in sorted(RELEVANCE_NAMES)
        },
    }

    # ------------------------------------------------------------- exports
    view_paths: dict[str, Any] = {}
    if not arguments.skip_exports:
        from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
        from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        marker_radius = 0.036220766603946686  # identical to worklog 120's export marker size

        def _write(view_name: str, bank, colors: torch.Tensor, radius: float) -> dict[str, Any]:
            scene_xyz = model.get_xyz.detach()
            scene_rgb = torch.tensor(_SCENE_RGB, device=device).reshape(1, 3).expand(scene_xyz.shape[0], 3)
            count = len(bank)
            log_scaling = torch.full((count, 2), float(np.log(radius)), dtype=torch.float32, device=device)
            opacity_logit = torch.full((count,), 4.0, dtype=torch.float32, device=device)
            rotation = torch.zeros((count, 4), dtype=torch.float32, device=device)
            rotation[:, 0] = 1.0
            xyz = torch.cat([scene_xyz, bank.positions], dim=0)
            f_dc = torch.cat([_rgb_to_f_dc(scene_rgb), _rgb_to_f_dc(colors)], dim=0)
            opacity = torch.cat([model._opacity.detach().reshape(-1), opacity_logit], dim=0)
            scaling = torch.cat([model._scaling.detach(), log_scaling], dim=0)
            rot = torch.cat([model._rotation.detach(), rotation], dim=0)
            ply_path = output_root / view_name / _ITERATION_DIR / "point_cloud.ply"
            written = write_surfel_ply(ply_path, xyz, f_dc, opacity, scaling, rot)
            review = TorchGaussianSurfelModel(sh_degree=0, device=str(device))
            review.initialize(
                positions=xyz, colors=torch.cat([scene_rgb, colors], dim=0),
                opacities=torch.sigmoid(opacity).reshape(-1, 1), scales=torch.exp(scaling), rotations=rot,
            )
            review.active_sh_degree = 0
            with torch.no_grad():
                package = rasterizer.render(preview_camera, review)
            write_ppm(output_root / view_name / "render.ppm", package["render"])
            del review, package
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            return {"point_cloud_ply": str(ply_path), "gaussian_count": written,
                    "render_ppm": str(output_root / view_name / "render.ppm"), "query_points": count}

        with torch.no_grad():
            package = rasterizer.render(preview_camera, model)
        ply_path = output_root / "ORIGINAL_2DGS_SCENE" / _ITERATION_DIR / "point_cloud.ply"
        written = write_surfel_ply(
            ply_path, model.get_xyz.detach(), model._features_dc.detach()[:, 0, :],
            model._opacity.detach().reshape(-1), model._scaling.detach(), model._rotation.detach(),
        )
        write_ppm(output_root / "ORIGINAL_2DGS_SCENE" / "render.ppm", package["render"])
        view_paths["ORIGINAL_2DGS_SCENE"] = {"point_cloud_ply": str(ply_path), "gaussian_count": written}
        del package

        gating_colors = torch.tensor(
            [GATING_RGB[int(code)] for code in contexts_np["gating_reason"]] * 3
            + [(0.95, 0.95, 0.95)] * int(control_rows.size),
            dtype=torch.float32, device=device,
        )
        view_paths["TOPOLOGY_GAP_BY_GATING_REASON"] = _write(
            "TOPOLOGY_GAP_BY_GATING_REASON", supplemental_bank, gating_colors, marker_radius * 1.5
        )
        for name in CANDIDATE_NAMES:
            colors = torch.tensor(
                [STATE_RGB[int(state)] for state in supplemental.global_states[name]],
                dtype=torch.float32, device=device,
            )
            view_paths[f"TOPOLOGY_GAP_CANDIDATE_{name}"] = _write(
                f"TOPOLOGY_GAP_CANDIDATE_{name}", supplemental_bank, colors, marker_radius * 1.5
            )
    report["review_exports"] = view_paths

    # -------------------------------------------- annotated qualitative cases
    midpoint_start = supplemental_sidecar["midpoint_rows"][0]
    annotated: list[dict[str, Any]] = []
    context_count = supplemental_sidecar["context_count"]
    picks = np.linspace(0, max(context_count - 1, 0), num=min(30, context_count)).round().astype(np.int64)
    for context in np.unique(picks):
        context = int(context)
        view = int(contexts_np["view_index"][context])
        rows = {"endpoint_a": context, "endpoint_b": context_count + context, "midpoint": midpoint_start + context}
        record: dict[str, Any] = {
            "context_index": context,
            "source_camera_index": view,
            "source_camera_name": supplemental.view_names[view] if view < len(supplemental.view_names) else None,
            "pixel_a": [int(contexts_np["row_a"][context]), int(contexts_np["col_a"][context])],
            "pixel_b": [int(contexts_np["row_b"][context]), int(contexts_np["col_b"][context])],
            "representative_a": int(contexts_np["representative_a"][context]),
            "representative_b": int(contexts_np["representative_b"][context]),
            "component_a": int(contexts_np["component_a"][context]),
            "component_b": int(contexts_np["component_b"][context]),
            "gating_reason": topology_gap_bank.GATING_NAMES[int(contexts_np["gating_reason"][context])],
            "region": bank_module.REGION_LABELS[int(contexts_np["region"][context])] if contexts_np["region"][context] >= 0 else None,
            "queries": {},
        }
        for label, row in rows.items():
            record["queries"][label] = {
                "query_id": int(row),
                "world_position": [float(v) for v in supplemental_bank.positions[row].tolist()],
                "global_states": {name: STATE_NAMES[int(supplemental.global_states[name][row])] for name in CANDIDATE_NAMES},
                "in_source_camera": {
                    "state": {name: STATE_NAMES[int(supplemental.per_view_states[name][row, view])] for name in CANDIDATE_NAMES},
                    "query_depth": float(supplemental.query_depth[row, view]),
                    "B_median_depth": float(supplemental.values["B_median_depth"][row, view]),
                    "B_signed_median_margin": float(supplemental_margin[row, view]),
                    "C_camera_nearest_blocker_t": float(supplemental.values["C_camera_nearest_blocker_t"][row, view]),
                    "C_query_nearest_blocker_t": float(supplemental.values["C_query_nearest_blocker_t"][row, view]),
                    "C_camera_nearest_blocker_world_gap": float(supplemental.values["C_camera_nearest_blocker_world_gap"][row, view]),
                    "C_query_nearest_blocker_world_gap": float(supplemental.values["C_query_nearest_blocker_world_gap"][row, view]),
                    "C_blocker_count": int(supplemental.values["C_blocker_count"][row, view]),
                    "C_same_component_blocker_count": int(supplemental.values["C_same_component_blocker_count"][row, view]),
                    "D_resolution_reason": REASON_NAMES[int(supplemental.values["D_resolution_reason"][row, view])],
                    "D_traversal_T_pre": float(supplemental.values["D_traversal_T_pre"][row, view]),
                    "D_resolution_event_depth": float(supplemental.values["D_resolution_event_depth"][row, view]),
                    "D_termination_test_T": float(supplemental.values["D_termination_test_T"][row, view]),
                    "D_late_front_count": int(supplemental.values["D_late_front_count"][row, view]),
                    "A_hit_distance": float(supplemental.values["A_hit_distance"][row, view]),
                },
            }
        annotated.append(record)
    report["annotated_topology_gap_cases"] = annotated

    # ------------------------------------------------------------- artifacts
    np.savez_compressed(
        output_root / "value_space_original_bank.npz",
        positions=original_bank.positions.detach().cpu().numpy(),
        kind=np.asarray(original_bank.kind), region=original_bank.region,
        source_view=original_bank.source_view, source_surfel=original_bank.source_surfel,
        relevance_code=original.relevance_code, query_depth=original.query_depth,
        view_names=np.asarray(original.view_names),
        **{f"states_{n}": original.per_view_states[n] for n in CANDIDATE_NAMES},
        **{f"global_{n}": original.global_states[n] for n in CANDIDATE_NAMES},
        **original.values,
    )
    np.savez_compressed(
        output_root / "value_space_supplemental_bank.npz",
        positions=supplemental_bank.positions.detach().cpu().numpy(),
        kind=np.asarray(supplemental_bank.kind), region=supplemental_bank.region,
        source_view=supplemental_bank.source_view, source_surfel=supplemental_bank.source_surfel,
        relevance_code=supplemental.relevance_code, query_depth=supplemental.query_depth,
        view_names=np.asarray(supplemental.view_names),
        **{f"context_{k}": v for k, v in contexts_np.items()},
        **{f"states_{n}": supplemental.per_view_states[n] for n in CANDIDATE_NAMES},
        **{f"global_{n}": supplemental.global_states[n] for n in CANDIDATE_NAMES},
        **supplemental.values,
    )
    (output_root / "topology_gap_contexts.json").write_text(json.dumps(supplemental_sidecar), encoding="utf-8")
    report["artifacts"] = {
        "original_bank_npz": str(output_root / "value_space_original_bank.npz"),
        "supplemental_bank_npz": str(output_root / "value_space_supplemental_bank.npz"),
        "topology_gap_contexts_json": str(output_root / "topology_gap_contexts.json"),
    }
    report["total_seconds"] = time.time() - started
    report_path = output_root / "observed_occluded_value_space_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"wrote {report_path} ({report['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
