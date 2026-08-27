"""Worklog 120 -- Observed / Occluded volumetric operationalization audit.

A controlled comparison of FOUR INDEPENDENT architecture hypotheses for
partitioning the camera-supported 3D reconstruction domain into OBSERVED and
OCCLUDED, with UNRESOLVED allowed only as a fail-closed implementation state:

    A. DIRECT SURFACE OBSERVATION / SURFACE-HIT   scripts/devtools/observed_occluded/candidate_a_surface_hit.py
    B. MEDIAN-DEPTH PARTITION                     scripts/devtools/observed_occluded/candidate_b_median_depth.py
    C. GEOMETRIC VISIBILITY                       scripts/devtools/observed_occluded/candidate_c_geometric_visibility.py
    D. RENDERER REACHABILITY                      scripts/devtools/observed_occluded/candidate_d_renderer_reachability.py

They are NOT four tunable variants of one implementation. This driver only
orchestrates: it builds one deterministic query bank, hands it unchanged to the
shared engine, and reports what came back. It does not optimize a candidate, it
does not combine them, and it does not force a winner.

Nothing in this batch modifies the visible topology (worklog 107/109), the
NURBS fitter, chart construction, the canonical production renderer, opacity,
Gaussian scale, or any renderer threshold. The only new CUDA is a THIRD
vendored sibling build, `diff_surfel_rasterization_qdepth` (alongside the
canonical package and worklog 107's `diff_surfel_rasterization_diag`), whose
additions are purely observational; both existing packages are left untouched so
every earlier replay stays bit-identical.
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
from observed_occluded import synthetic_contracts  # noqa: E402
from observed_occluded.candidate_a_surface_hit import FLOAT32_IDENTITY_RELATIVE_EPSILON  # noqa: E402
from observed_occluded.candidate_c_geometric_visibility import SEGMENT_EPSILON  # noqa: E402
from observed_occluded.engine import CANDIDATE_NAMES, build_geometric_support, evaluate  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    KIND_R1_ANCHOR_RHO2D, KIND_R1_ANCHOR_RHO3D, KIND_R3_BEHIND, KIND_R4_FRONT,
    KIND_R5_REGION_GAP, KIND_R6_OUT_OF_FRUSTUM,
    RELEVANCE_NAMES, RELEVANCE_OK, STATE_NAMES, STATE_NON_RELEVANT, STATE_OBSERVED,
    STATE_OCCLUDED, STATE_UNRESOLVED, agreement, canonical_constants_from_source,
    distribution, state_fractions,
)

_ITERATION_DIR = "iteration_0000001"
_SCENE_RGB = (0.07, 0.08, 0.10)
STATE_RGB = {
    STATE_OBSERVED: (0.10, 0.85, 0.35),
    STATE_OCCLUDED: (0.92, 0.18, 0.18),
    STATE_UNRESOLVED: (0.60, 0.60, 0.62),
}
KIND_RGB = {
    KIND_R1_ANCHOR_RHO3D: (0.20, 0.75, 0.95),
    KIND_R1_ANCHOR_RHO2D: (0.95, 0.75, 0.15),
    KIND_R3_BEHIND: (0.90, 0.30, 0.55),
    KIND_R4_FRONT: (0.45, 0.90, 0.60),
    KIND_R5_REGION_GAP: (0.70, 0.45, 0.95),
    KIND_R6_OUT_OF_FRUSTUM: (0.95, 0.95, 0.95),
}
REGION_RGB = [
    (0.95, 0.35, 0.20), (0.25, 0.65, 0.95), (0.95, 0.85, 0.25),
    (0.35, 0.85, 0.45), (0.75, 0.45, 0.95),
]

VIEW_ORIGINAL_SCENE = "ORIGINAL_2DGS_SCENE"
VIEW_QUERY_BANK = "QUERY_BANK_BY_KIND"
VIEW_REGION = "QUERY_BANK_BY_REGION"


def _progress(message: str) -> None:
    print(f"[observed-occluded] {message}", flush=True)


# --------------------------------------------------------------------------
# Metrics (directive section 10). Every axis reported separately; there is no
# single weighted "win score" anywhere in this file (directive section 11).
# --------------------------------------------------------------------------

def coverage_accounting(result, name: str) -> dict[str, Any]:
    states = result.per_view_states[name]
    relevant = states != STATE_NON_RELEVANT
    pairs = int(states.size)
    return {
        "total_query_view_pairs": pairs,
        "relevant_query_view_pairs": int(relevant.sum()),
        "relevant_fraction_of_all_pairs": float(relevant.sum()) / pairs if pairs else 0.0,
        "per_view_pair_states": state_fractions(states),
        "per_view_pair_states_including_non_relevant": state_fractions(states, include_non_relevant=True),
        "global_states": state_fractions(result.global_states[name]),
    }


def positive_observation_retention(result, bank, name: str) -> dict[str, Any]:
    kinds = np.asarray(bank.kind)
    out: dict[str, Any] = {}
    for label, mask in (
        ("all_R1_anchors", (kinds == KIND_R1_ANCHOR_RHO3D) | (kinds == KIND_R1_ANCHOR_RHO2D)),
        ("rho3d_true_footprint_anchors", kinds == KIND_R1_ANCHOR_RHO3D),
        ("rho2d_low_pass_anchors", kinds == KIND_R1_ANCHOR_RHO2D),
    ):
        if not mask.any():
            continue
        globals_ = result.global_states[name][mask]
        entry = state_fractions(globals_)
        entry["severe_semantic_contradiction_count"] = int((globals_ == STATE_OCCLUDED).sum())
        entry["severe_semantic_contradiction_fraction"] = float((globals_ == STATE_OCCLUDED).mean()) if globals_.size else 0.0
        # Source-view contradiction: the anchor's OWN generating view calling it occluded.
        rows = np.nonzero(mask)[0]
        source = bank.source_view[rows]
        per_view = result.per_view_states[name][rows]
        source_state = per_view[np.arange(rows.size), source]
        entry["source_view_states"] = state_fractions(source_state)
        entry["source_view_occluded_count"] = int((source_state == STATE_OCCLUDED).sum())
        out[label] = entry
    return out


def cross_view_accounting(result, bank, name: str) -> dict[str, Any]:
    kinds = np.asarray(bank.kind)
    mask = (kinds == KIND_R1_ANCHOR_RHO3D) | (kinds == KIND_R1_ANCHOR_RHO2D)
    rows = np.nonzero(mask)[0]
    if rows.size == 0:
        return {}
    per_view = result.per_view_states[name][rows]
    relevant = (per_view != STATE_NON_RELEVANT).sum(axis=1)
    observed = (per_view == STATE_OBSERVED).sum(axis=1)
    occluded = (per_view == STATE_OCCLUDED).sum(axis=1)
    unresolved = (per_view == STATE_UNRESOLVED).sum(axis=1)
    mixed = (observed > 0) & (occluded > 0)
    globals_ = result.global_states[name][rows]
    return {
        "anchor_count": int(rows.size),
        "relevant_views_per_anchor": distribution(relevant),
        "observed_views_per_anchor": distribution(observed),
        "occluded_views_per_anchor": distribution(occluded),
        "unresolved_views_per_anchor": distribution(unresolved),
        "anchors_with_both_observed_and_occluded_views": int(mixed.sum()),
        "global_state_of_those_mixed_anchors": state_fractions(globals_[mixed]) if mixed.any() else None,
        "anchors_observed_in_no_view": int((observed == 0).sum()),
    }


def ray_order_diagnostics(result, bank, name: str) -> dict[str, Any]:
    """Directive 10G: MEASURE how the state evolves with query depth along a
    ray. Monotonicity is never imposed by post-processing."""

    kinds = np.asarray(bank.kind)
    ladder = (kinds == KIND_R3_BEHIND) | (kinds == KIND_R4_FRONT)
    rows = np.nonzero(ladder)[0]
    if rows.size == 0:
        return {}
    keys = list(zip(bank.source_view[rows].tolist(), bank.source_surfel[rows].tolist()))
    grouped: dict[tuple[int, int], list[int]] = {}
    for row, key in zip(rows.tolist(), keys):
        grouped.setdefault(key, []).append(row)

    monotone = 0
    non_monotone = 0
    step_states: dict[float, list[int]] = {}
    examples: list[dict[str, Any]] = []
    for key, members in grouped.items():
        members_sorted = sorted(members, key=lambda row: float(bank.ladder_step[row]))
        sequence = [int(result.global_states[name][row]) for row in members_sorted]
        for row in members_sorted:
            step_states.setdefault(float(bank.ladder_step[row]), []).append(int(result.global_states[name][row]))
        # "Monotone" = once the ladder leaves OBSERVED going deeper it never
        # returns; UNRESOLVED is treated as its own value and any OBSERVED after
        # a non-OBSERVED counts as non-monotone.
        seen_non_observed = False
        ok = True
        for state in sequence:
            if state != STATE_OBSERVED:
                seen_non_observed = True
            elif seen_non_observed:
                ok = False
                break
        if ok:
            monotone += 1
        else:
            non_monotone += 1
            if len(examples) < 20:
                examples.append({
                    "source_view": int(key[0]), "source_surfel": int(key[1]),
                    "steps": [float(bank.ladder_step[row]) for row in members_sorted],
                    "states": [STATE_NAMES[state] for state in sequence],
                })
    return {
        "ladder_count": len(grouped),
        "monotone_ladders": monotone,
        "non_monotone_ladders": non_monotone,
        "monotone_fraction": monotone / len(grouped) if grouped else 0.0,
        "state_by_ladder_step": {
            f"{step:+.1f}x_support_radius": state_fractions(np.asarray(values, dtype=np.int8))
            for step, values in sorted(step_states.items())
        },
        "non_monotone_examples": examples,
    }


def region_accounting(result, bank, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for index, label in enumerate(bank_module.REGION_LABELS):
        mask = bank.region == index
        if not mask.any():
            continue
        out[label] = {
            "query_count": int(mask.sum()),
            "global_states": state_fractions(result.global_states[name][mask]),
            "per_view_pair_states": state_fractions(result.per_view_states[name][mask]),
        }
    unlabelled = bank.region < 0
    if unlabelled.any():
        out["UNLABELLED"] = {
            "query_count": int(unlabelled.sum()),
            "global_states": state_fractions(result.global_states[name][unlabelled]),
        }
    return out


def kind_accounting(result, bank, name: str) -> dict[str, Any]:
    kinds = np.asarray(bank.kind)
    return {
        kind: {
            "query_count": int((kinds == kind).sum()),
            "global_states": state_fractions(result.global_states[name][kinds == kind]),
        }
        for kind in sorted(set(kinds.tolist()))
    }


def disagreement_cases(result, bank, left: str, right: str, limit: int = 40) -> list[dict[str, Any]]:
    """Directive 18: explicit disagreement cases with enough provenance to
    inspect the primitive each side actually used."""

    left_states = result.global_states[left]
    right_states = result.global_states[right]
    differing = np.nonzero(
        ((left_states == STATE_OBSERVED) & (right_states == STATE_OCCLUDED))
        | ((left_states == STATE_OCCLUDED) & (right_states == STATE_OBSERVED))
    )[0]
    if differing.size == 0:
        differing = np.nonzero(left_states != right_states)[0]
    if differing.size == 0:
        return []
    picks = differing[np.linspace(0, differing.size - 1, num=min(limit, differing.size)).round().astype(np.int64)]
    cases: list[dict[str, Any]] = []
    for row in np.unique(picks):
        row = int(row)
        source = int(bank.source_view[row])
        camera_row = source if source >= 0 else int(np.argmax(result.relevance_code[row] == RELEVANCE_OK))
        cases.append({
            "query_id": row,
            "kind": bank.kind[row],
            "region": bank_module.REGION_LABELS[int(bank.region[row])] if bank.region[row] >= 0 else None,
            "world_position": [float(v) for v in bank.positions[row].tolist()],
            "source_view_index": source,
            "source_view_name": result.view_names[source] if source >= 0 else None,
            "inspected_camera_index": camera_row,
            "inspected_camera_name": result.view_names[camera_row] if camera_row < len(result.view_names) else None,
            "ladder_step_in_support_radii": float(bank.ladder_step[row]),
            "query_depth_in_inspected_camera": float(result.query_depth[row, camera_row]),
            "relevance_in_inspected_camera": RELEVANCE_NAMES[int(result.relevance_code[row, camera_row])],
            "global_states": {name: STATE_NAMES[int(result.global_states[name][row])] for name in CANDIDATE_NAMES},
            "state_in_inspected_camera": {
                name: STATE_NAMES[int(result.per_view_states[name][row, camera_row])] for name in CANDIDATE_NAMES
            },
            "primitive_in_inspected_camera": {
                "A_surface_event_depth": float(result.provenance["A_event_depth"][row, camera_row]),
                "A_hit_distance": float(result.provenance["A_hit_distance"][row, camera_row]),
                "A_representative_surfel_id": int(result.provenance["A_event_surfel"][row, camera_row]),
                "A_rho_branch": {0: "rho3d", 1: "rho2d", -1: None}[int(result.provenance["A_event_branch"][row, camera_row])],
                "B_median_depth": float(result.provenance["B_median_depth"][row, camera_row]),
                "C_geometric_blocker_count": int(result.provenance["C_blocker_count"][row, camera_row]),
                "C_nearest_blocker_t": float(result.provenance["C_nearest_blocker_t"][row, camera_row]),
                "C_max_blocker_opacity": float(result.provenance["C_max_blocker_opacity"][row, camera_row]),
                "D_transmittance_before_query_depth": float(result.provenance["D_transmittance"][row, camera_row]),
                "D_traversal_reached_query_depth": int(result.provenance["D_reached"][row, camera_row]),
                "D_accepted_prefix_count": int(result.provenance["D_prefix_count"][row, camera_row]),
            },
        })
    return cases


# --------------------------------------------------------------------------
# Review exports
# --------------------------------------------------------------------------

def _query_surfel_arrays(colors: torch.Tensor, radius: float, device: str):
    count = int(colors.shape[0])
    log_scaling = torch.full((count, 2), float(np.log(max(radius, 1e-5))), dtype=torch.float32, device=device)
    opacity_logit = torch.full((count,), 4.0, dtype=torch.float32, device=device)
    rotation = torch.zeros((count, 4), dtype=torch.float32, device=device)
    rotation[:, 0] = 1.0
    return _rgb_to_f_dc(colors), opacity_logit, log_scaling, rotation


def write_review_view(
    output_root: Path, view_name: str, model: Any, bank, colors: torch.Tensor,
    *, radius: float, rasterizer=None, camera=None, subset: np.ndarray | None = None,
) -> dict[str, Any]:
    """One review export: the whole trained scene in near-black plus the query
    points coloured by whatever `colors` encodes."""

    device = model.device
    rows = np.arange(len(bank)) if subset is None else subset
    scene_xyz = model.get_xyz.detach()
    scene_dc = _rgb_to_f_dc(torch.tensor(_SCENE_RGB, device=device).reshape(1, 3).expand(scene_xyz.shape[0], 3).contiguous())
    scene_opacity = model._opacity.detach().reshape(-1)
    scene_log_scaling = model._scaling.detach()
    scene_rotation = model._rotation.detach()

    picked = torch.as_tensor(rows, dtype=torch.int64, device=device)
    query_xyz = bank.positions[picked]
    query_dc, query_opacity, query_scaling, query_rotation = _query_surfel_arrays(
        colors[picked], radius, str(device)
    )
    xyz = torch.cat([scene_xyz, query_xyz], dim=0)
    f_dc = torch.cat([scene_dc, query_dc], dim=0)
    opacity = torch.cat([scene_opacity, query_opacity], dim=0)
    log_scaling = torch.cat([scene_log_scaling, query_scaling], dim=0)
    rotation = torch.cat([scene_rotation, query_rotation], dim=0)

    ply_path = output_root / view_name / _ITERATION_DIR / "point_cloud.ply"
    written = write_surfel_ply(ply_path, xyz, f_dc, opacity, log_scaling, rotation)
    entry: dict[str, Any] = {"point_cloud_ply": str(ply_path), "gaussian_count": written, "query_points": int(len(rows))}

    if rasterizer is not None and camera is not None:
        from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel

        review = TorchGaussianSurfelModel(sh_degree=0, device=str(device))
        review.initialize(
            positions=xyz,
            colors=torch.cat([
                torch.tensor(_SCENE_RGB, device=device).reshape(1, 3).expand(scene_xyz.shape[0], 3),
                colors[picked],
            ], dim=0),
            opacities=torch.sigmoid(opacity).reshape(-1, 1),
            scales=torch.exp(log_scaling),
            rotations=rotation,
        )
        review.active_sh_degree = 0
        with torch.no_grad():
            package = rasterizer.render(camera, review)
        ppm_path = output_root / view_name / "render.ppm"
        write_ppm(ppm_path, package["render"])
        entry["render_ppm"] = str(ppm_path)
        del review, package
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return entry


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
    parser.add_argument("--max-views", type=int, default=0, help="smoke-test only: truncate the camera set")
    parser.add_argument("--skip-exports", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=384, help="Candidate C per-chunk tensor budget")
    parser.add_argument("--query-marker-radius", type=float, default=0.0, help="review exports only; 0 = derive from the bank's own support radii")
    arguments = parser.parse_args()

    started = time.time()
    output_root: Path = arguments.out
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "batch": "arch/2dgs-coverage-first-surface, Worklog 120",
        "checkpoint": str(arguments.checkpoint),
        "canonical_constants_reread_from_production_source": canonical_constants_from_source(REPO_ROOT),
        "agent_introduced_numerical_rules": {
            "candidate_A_float32_identity_relative_epsilon": FLOAT32_IDENTITY_RELATIVE_EPSILON,
            "candidate_C_open_segment_epsilon": SEGMENT_EPSILON,
            "query_pixel_rounding": "nearest pixel (round-half-to-even on the rasterizer's own continuous pixel coordinate)",
        },
    }

    _progress("[1/6] synthetic contracts S1-S7")
    synthetic = synthetic_contracts.run_contracts(device=arguments.device, progress=_progress)
    report["synthetic_contracts"] = synthetic
    report["synthetic_contract_summary"] = synthetic_contracts.contract_summary(synthetic)
    if arguments.device == "cuda":
        torch.cuda.empty_cache()

    _progress(f"[2/6] loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint (primitive={primitive!r}).")
    total_model_count = len(model)
    uncertain_count = int(model.is_uncertain.reshape(-1).to(torch.bool).sum())
    report["primitive"] = primitive
    report["iteration"] = payload.get("iteration")
    report["total_trained_surfels"] = total_model_count
    report["uncertain_surfels_in_checkpoint"] = uncertain_count

    cameras, camera_meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, arguments.device,
    )
    if int(arguments.max_views) > 0:
        cameras = cameras[: int(arguments.max_views)]
        camera_meta = {**camera_meta, "smoke_test_max_views": int(arguments.max_views)}
    report["camera_meta"] = camera_meta
    _progress(f"model surfels={total_model_count} uncertain={uncertain_count} cameras={len(cameras)}")

    _progress("[3/6] region labels + deterministic query bank")
    preview_camera = min(cameras, key=lambda c: str(c.image_name))
    region_index, region_meta = bank_module.region_of_surfel(model, preview_camera)
    bank, bank_report = bank_module.build_bank(model, cameras, region_index, progress=_progress)
    report["query_bank"] = {
        **bank.as_metadata(bank_module.REGION_LABELS),
        "region_anchor_mechanism": region_meta,
        "construction": bank_report.notes,
        "anchor_views": bank_report.anchor_views,
        "anchor_view_names": bank_report.anchor_view_names,
        "anchor_view_valid_median_pixels": bank_report.per_view_valid_pixels,
        "anchor_view_rho3d_pixels": bank_report.per_view_rho3d_pixels,
        "anchor_view_rho2d_pixels": bank_report.per_view_rho2d_pixels,
    }
    _progress(f"query bank: {len(bank)} queries -> {report['query_bank']['by_kind']}")

    _progress("[4/6] evaluating A/B/C/D over every view (identical query bank)")
    support = build_geometric_support(model)
    report["candidate_C_support_summary"] = {
        "surfels_with_nonempty_geometric_support": int(support.nonempty.sum()),
        "support_radius_world_units": distribution(support.support_radius.detach().cpu().numpy()),
        "rho_max": distribution(support.rho_max.detach().cpu().numpy()),
    }
    evaluation_started = time.time()
    result = evaluate(
        model, cameras, bank.positions, support=support,
        chunk_bytes=int(arguments.chunk_mib) * 1024 * 1024, progress=_progress,
    )
    report["evaluation_seconds"] = time.time() - evaluation_started
    report["engine_diagnostics"] = result.diagnostics
    report["frozen_state_fingerprint"] = {
        "median_surface_representatives_union": result.diagnostics["median_surface_representatives_union"],
        "worklog_119_reference_value": 785937,
        "matches_worklog_119": bool(result.diagnostics["median_surface_representatives_union"] == 785937),
        "note": (
            "Union over all training views of surfels that were ever the renderer's median surface "
            "representative. Reproducing worklog 119's value exactly is direct evidence the model, the camera "
            "set and the renderer are unchanged by this batch. Only meaningful on the full 161-view run."
        ),
    }

    _progress("[5/6] metrics")
    relevance_codes = result.relevance_code
    report["relevant_view_contract"] = {
        "pair_counts": {
            RELEVANCE_NAMES[code]: int((relevance_codes == code).sum()) for code in sorted(RELEVANCE_NAMES)
        },
        "queries_with_zero_relevant_views": int((relevance_codes != RELEVANCE_OK).all(axis=1).sum()),
        "relevant_views_per_query": distribution((relevance_codes == RELEVANCE_OK).sum(axis=1)),
    }

    per_candidate: dict[str, Any] = {}
    for name in CANDIDATE_NAMES:
        per_candidate[name] = {
            "coverage": coverage_accounting(result, name),
            "positive_observation_retention": positive_observation_retention(result, bank, name),
            "cross_view": cross_view_accounting(result, bank, name),
            "ray_order": ray_order_diagnostics(result, bank, name),
            "by_query_kind": kind_accounting(result, bank, name),
            "by_region": region_accounting(result, bank, name),
        }
    per_candidate["A"]["event_branch_metadata"] = {
        "rho3d_dominated_query_view_pairs": int((result.provenance["A_event_branch"] == 0).sum()),
        "rho2d_dominated_query_view_pairs": int((result.provenance["A_event_branch"] == 1).sum()),
        "no_event_query_view_pairs": int((result.provenance["A_event_branch"] == -1).sum()),
        "hit_distance_when_event_exists": distribution(result.provenance["A_hit_distance"]),
        "note": "Branch provenance is metadata only; candidate A never uses it as a second decision rule.",
    }
    per_candidate["C"]["blocker_metadata"] = {
        "blocker_count_over_relevant_pairs": distribution(
            result.provenance["C_blocker_count"][result.per_view_states["C"] != STATE_NON_RELEVANT]
        ),
        "nearest_blocker_t": distribution(result.provenance["C_nearest_blocker_t"]),
        "max_blocker_opacity_on_occluded_pairs": distribution(
            result.provenance["C_max_blocker_opacity"][result.per_view_states["C"] == STATE_OCCLUDED]
        ),
        "note": "Metadata only; C's decision is opacity-blind by hypothesis. See candidate_c module docstring.",
    }
    observed_mask = result.per_view_states["D"] == STATE_OBSERVED
    occluded_mask = result.per_view_states["D"] == STATE_OCCLUDED
    per_candidate["D"]["transmittance_metadata"] = {
        "T_before_query_depth_on_OBSERVED_pairs": distribution(result.provenance["D_transmittance"][observed_mask]),
        "T_before_query_depth_on_OCCLUDED_pairs": distribution(result.provenance["D_transmittance"][occluded_mask]),
        "traversal_reached_query_depth_fraction_on_OBSERVED": float(
            (result.provenance["D_reached"][observed_mask] == 1).mean()
        ) if observed_mask.any() else 0.0,
        "accepted_prefix_count": distribution(result.provenance["D_prefix_count"][observed_mask | occluded_mask]),
        "note": "Metadata only; candidate D never compares T against anything. Its only decision is the canonical termination event.",
    }
    report["per_candidate"] = per_candidate

    matrix: dict[str, Any] = {}
    for left_index, left in enumerate(CANDIDATE_NAMES):
        for right in CANDIDATE_NAMES[left_index + 1:]:
            matrix[f"{left}_vs_{right}"] = {
                "global": agreement(result.global_states[left], result.global_states[right]),
                "per_view_pair": agreement(
                    result.per_view_states[left][result.per_view_states[left] != STATE_NON_RELEVANT],
                    result.per_view_states[right][result.per_view_states[right] != STATE_NON_RELEVANT],
                ),
            }
    report["candidate_agreement_matrix"] = matrix
    report["disagreement_cases"] = {
        pair: disagreement_cases(result, bank, pair.split("_vs_")[0], pair.split("_vs_")[1])
        for pair in ("B_vs_D", "A_vs_D", "C_vs_D", "A_vs_B", "B_vs_C")
    }

    _progress("[6/6] review exports")
    view_paths: dict[str, Any] = {}
    if not arguments.skip_exports:
        from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        # Query markers are drawn at roughly ONE trained surfel's own tangent
        # scale, so they read as distinct points against the scene rather than
        # smearing over it: the canonical support radius is sqrt(rho_max) *
        # max(scale_u, scale_v) and sqrt(rho_max) is ~3.3 at typical opacity.
        radius = float(np.nanmedian(bank.support_radius[np.isfinite(bank.support_radius)])) / 3.0
        radius = float(arguments.query_marker_radius) if arguments.query_marker_radius > 0 else (radius if radius > 0 else 0.02)
        report["review_export_query_marker_radius"] = radius
        device = model.device

        trained_sh_degree = int(model.active_sh_degree)
        original_dc = model._features_dc.detach().clone()
        original_rest = model._features_rest.detach().clone()
        with torch.no_grad():
            package = rasterizer.render(preview_camera, model)
        ply_path = output_root / VIEW_ORIGINAL_SCENE / _ITERATION_DIR / "point_cloud.ply"
        written = write_surfel_ply(
            ply_path, model.get_xyz.detach(), original_dc[:, 0, :], model._opacity.detach().reshape(-1),
            model._scaling.detach(), model._rotation.detach(),
        )
        write_ppm(output_root / VIEW_ORIGINAL_SCENE / "render.ppm", package["render"])
        view_paths[VIEW_ORIGINAL_SCENE] = {
            "point_cloud_ply": str(ply_path), "gaussian_count": written,
            "render_ppm": str(output_root / VIEW_ORIGINAL_SCENE / "render.ppm"),
            "sh_degree": trained_sh_degree,
        }
        del package
        if arguments.device == "cuda":
            torch.cuda.empty_cache()

        kind_colors = torch.tensor(
            [KIND_RGB[kind] for kind in bank.kind], dtype=torch.float32, device=device
        )
        view_paths[VIEW_QUERY_BANK] = write_review_view(
            output_root, VIEW_QUERY_BANK, model, bank, kind_colors,
            radius=radius, rasterizer=rasterizer, camera=preview_camera,
        )
        region_colors = torch.tensor(
            [REGION_RGB[int(r)] if r >= 0 else (0.5, 0.5, 0.5) for r in bank.region],
            dtype=torch.float32, device=device,
        )
        view_paths[VIEW_REGION] = write_review_view(
            output_root, VIEW_REGION, model, bank, region_colors,
            radius=radius, rasterizer=rasterizer, camera=preview_camera,
        )
        for name in CANDIDATE_NAMES:
            colors = torch.tensor(
                [STATE_RGB[int(state)] for state in result.global_states[name]],
                dtype=torch.float32, device=device,
            )
            view_paths[f"CANDIDATE_{name}_GLOBAL_STATE"] = write_review_view(
                output_root, f"CANDIDATE_{name}_GLOBAL_STATE", model, bank, colors,
                radius=radius, rasterizer=rasterizer, camera=preview_camera,
            )
        for pair in ("B_vs_D", "A_vs_D", "C_vs_D"):
            left, right = pair.split("_vs_")
            differing = np.nonzero(result.global_states[left] != result.global_states[right])[0]
            if differing.size == 0:
                view_paths[f"DISAGREEMENT_{pair}"] = {"skipped": "no global disagreement"}
                continue
            colors = torch.tensor(
                [STATE_RGB[int(result.global_states[left][row])] for row in differing],
                dtype=torch.float32, device=device,
            )
            full_colors = torch.zeros((len(bank), 3), dtype=torch.float32, device=device)
            full_colors[torch.as_tensor(differing, dtype=torch.int64, device=device)] = colors
            view_paths[f"DISAGREEMENT_{pair}"] = write_review_view(
                output_root, f"DISAGREEMENT_{pair}", model, bank, full_colors,
                radius=radius * 1.5, rasterizer=rasterizer, camera=preview_camera, subset=differing,
            )
            view_paths[f"DISAGREEMENT_{pair}"]["colour_encodes"] = f"the {left} state of each query where {left} and {right} disagree"

        with torch.no_grad():
            model._features_dc.data.copy_(original_dc)
            model._features_rest.data.copy_(original_rest)
            model.active_sh_degree = trained_sh_degree
    report["review_exports"] = view_paths

    report["query_table_columns"] = [
        "query_id", "kind", "region", "x", "y", "z", "source_view", "source_surfel",
        "ladder_step_support_radii", "relevant_views", "A", "B", "C", "D",
    ]
    kinds = np.asarray(bank.kind)
    relevant_counts = (relevance_codes == RELEVANCE_OK).sum(axis=1)
    table = []
    for row in range(len(bank)):
        table.append([
            row, kinds[row],
            bank_module.REGION_LABELS[int(bank.region[row])] if bank.region[row] >= 0 else None,
            *[float(v) for v in bank.positions[row].tolist()],
            int(bank.source_view[row]), int(bank.source_surfel[row]),
            float(bank.ladder_step[row]), int(relevant_counts[row]),
            *[STATE_NAMES[int(result.global_states[name][row])] for name in CANDIDATE_NAMES],
        ])
    table_path = output_root / "observed_occluded_query_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    report["query_table_path"] = str(table_path)

    npz_path = output_root / "observed_occluded_per_view_states.npz"
    np.savez_compressed(
        npz_path,
        positions=bank.positions.detach().cpu().numpy(),
        kind=np.asarray(bank.kind),
        region=bank.region,
        source_view=bank.source_view,
        source_surfel=bank.source_surfel,
        ladder_step=bank.ladder_step,
        support_radius=bank.support_radius,
        relevance_code=relevance_codes,
        query_depth=result.query_depth,
        view_names=np.asarray(result.view_names),
        **{f"states_{name}": result.per_view_states[name] for name in CANDIDATE_NAMES},
        **{f"global_{name}": result.global_states[name] for name in CANDIDATE_NAMES},
        **result.provenance,
    )
    report["per_view_state_archive"] = str(npz_path)
    report["total_seconds"] = time.time() - started

    report_path = output_root / "observed_occluded_volumetric_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"wrote {report_path} ({report['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
