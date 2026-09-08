"""Worklog 179: renderer-native three-state observation evidence contract gate.

The script is intentionally a small deterministic audit, not a renderer
replacement.  It records the exact subject/event schema already exposed by
the diagnostic rasterizers, a controlled analytic blocker scene, and the
negative semantic gate.  Live qdepth replay is CUDA-dependent; the existing
W165/W168 CUDA controls are referenced as prior measured evidence rather than
recreated with a synthetic score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from osn_gs.render.torch_observation_contract_audit import (
    RAW_REACHED,
    RAW_TERMINATED,
    RAW_UNRESOLVED,
    RawQueryEvent,
    aggregate_raw_query_events,
    evaluate_contract_gate,
    existing_evidence_subject_audit,
)


STATE_COLORS = {
    "OBSERVED": (0.10, 0.85, 0.35),
    "OCCLUDED": (0.92, 0.18, 0.18),
    "UNRESOLVED": (0.60, 0.60, 0.62),
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def controlled_scene() -> dict[str, Any]:
    """Return exact mesh/camera/query metadata for the minimal analytic scene."""

    return {
        "scene_type": "analytic_foreground_blocker_background_wall",
        "geometry": {
            "foreground_blocker": {
                "type": "rectangle_mesh",
                "z": 2.0,
                "x_range": [-0.5, 0.5],
                "y_range": [-0.5, 0.5],
                "primitive_ids": [0],
            },
            "background_wall": {
                "type": "rectangle_mesh",
                "z": 4.0,
                "x_range": [-2.0, 2.0],
                "y_range": [-2.0, 2.0],
                "primitive_ids": [1],
            },
        },
        "cameras": [
            {
                "name": "front_camera",
                "center": [0.0, 0.0, 0.0],
                "look_at": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
                "resolution": [64, 64],
                "near": 0.01,
                "far": 100.0,
                "background": [0.0, 0.0, 0.0],
            },
            {
                "name": "offset_camera",
                "center": [3.0, 0.0, 1.0],
                "look_at": [0.0, 0.0, 3.0],
                "up": [0.0, 1.0, 0.0],
                "resolution": [64, 64],
                "near": 0.01,
                "far": 100.0,
                "background": [0.0, 0.0, 0.0],
            },
        ],
        "gt_definition": {
            "observed": "analytic camera-to-query segment has no earlier mesh intersection",
            "occluded": "analytic first mesh intersection is strictly before the query",
            "unresolved": "query has no target surface or admissible renderer event; no state is inferred from absence",
            "independent_from_renderer": True,
        },
    }


def renderer_event_records() -> list[dict[str, Any]]:
    """Return field-complete contract fixtures in the qdepth output schema."""

    records = [
        {
            "id": "blocker_center",
            "subject_type": "camera-ray point query",
            "world_query": [0.0, 0.0, 4.0],
            "camera": "front_camera",
            "gt_state": "OCCLUDED",
            "relevant_ray": {"origin": [0.0, 0.0, 0.0], "analytic_first_blocker_z": 2.0},
            "evidence": RawQueryEvent(
                query_terminated=1,
                query_reached=0,
                query_resolution_depth=2.0,
                query_termination_alpha=0.99,
                query_T=1.0,
                query_prefix_count=0,
                query_late_front_count=0,
                pixel_inversion_count=0,
            ).as_dict(),
            "evidence_origin": "qdepth field schema; positive termination witness",
        },
        {
            "id": "background_clear_side",
            "subject_type": "camera-ray point query",
            "world_query": [1.2, 0.0, 4.0],
            "camera": "front_camera",
            "gt_state": "OBSERVED",
            "relevant_ray": {"origin": [0.0, 0.0, 0.0], "analytic_first_blocker_z": None},
            "evidence": RawQueryEvent(
                query_terminated=0,
                query_reached=1,
                query_resolution_depth=4.0,
                query_T=0.8,
                query_prefix_count=1,
                query_late_front_count=0,
                pixel_inversion_count=0,
            ).as_dict(),
            "evidence_origin": "qdepth field schema; reached accepted event witness",
        },
        {
            "id": "unsupported_empty_query",
            "subject_type": "camera-ray point query",
            "world_query": [0.0, 0.0, 1.0],
            "camera": "offset_camera",
            "gt_state": "UNRESOLVED",
            "relevant_ray": {"origin": [3.0, 0.0, 1.0], "analytic_first_blocker_z": None},
            "evidence": RawQueryEvent(
                query_terminated=0,
                query_reached=0,
                query_resolution_depth=-1.0,
                query_T=0.0,
                query_prefix_count=0,
                query_late_front_count=-1,
                pixel_inversion_count=0,
            ).as_dict(),
            "evidence_origin": "qdepth unresolved/exhausted fill; no state inferred",
        },
        {
            "id": "adversarial_renderer_termination",
            "subject_type": "camera-ray point query",
            "world_query": [1.2, 0.0, 4.0],
            "camera": "front_camera",
            "gt_state": "OBSERVED",
            "relevant_ray": {"origin": [0.0, 0.0, 0.0], "analytic_first_blocker_z": None},
            "evidence": RawQueryEvent(
                query_terminated=1,
                query_reached=0,
                query_resolution_depth=2.0,
                query_termination_alpha=0.99,
                query_T=1.0,
                query_prefix_count=0,
                query_late_front_count=0,
                pixel_inversion_count=0,
            ).as_dict(),
            "evidence_origin": "adversarial event fixture; tests semantic soundness, not a measured renderer replay",
        },
    ]
    for record in records:
        record["raw_event"] = record["evidence"]["raw_event"]
    return records


def _write_readme(path: Path, title: str, meaning: str, limitation: str) -> None:
    path.write_text(
        "# " + title + "\n\n"
        "의미: " + meaning + "\n\n"
        "팔레트/범례: OBSERVED=(0.10, 0.85, 0.35), OCCLUDED=(0.92, 0.18, 0.18), "
        "UNRESOLVED=(0.60, 0.60, 0.62). Raw qdepth event는 별도 표기로 구분한다.\n\n"
        "공통 조건: Worklog 179 analytic blocker/wall scene, fixed camera metadata, "
        "64x64 conceptual review grid, no score threshold or learned quantity.\n\n"
        "검토 한계: " + limitation + "\n",
        encoding="utf-8",
    )


def _render_visuals(out: Path, records: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        _write_json(out / "visualization_error.json", {"error": repr(exc)})
        return []

    visual_specs = {
        "gt_state_map": (
            "GT_STATE_MAP",
            "Independent analytic GT state map",
            "독립 analytic mesh/camera-ray GT 상태를 보여준다.",
            "This is a query-state control, not a Gaussian visualization.",
        ),
        "renderer_event_partition": (
            "RENDERER_EVENT_PARTITION",
            "Renderer-native raw event partition",
            "qdepth의 reached/terminated/unresolved raw event를 최종 F 라벨 없이 보여준다.",
            "Raw events are not promoted to OBSERVED/OCCLUDED F states.",
        ),
        "evidence_reason_map": (
            "EVIDENCE_REASON_MAP",
            "Evidence reason map",
            "각 subject가 direct access, positive termination, insufficient event 중 무엇인지 보여준다.",
            "A termination event lacks an independent physical blocker identity.",
        ),
        "renderer_event_gt_disagreement": (
            "RENDERER_EVENT_GT_DISAGREEMENT",
            "Raw renderer event versus independent GT",
            "adversarial termination과 독립 GT의 semantic disagreement를 보존한다.",
            "The adversarial row is a contract fixture, not a live CUDA measurement.",
        ),
    }
    for directory, (title, plot_title, meaning, limitation) in visual_specs.items():
        folder = out / directory
        folder.mkdir(parents=True, exist_ok=True)
        _write_readme(folder / "README.md", title, meaning, limitation)

        fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=140)
        for record in records:
            x, y, _z = record["world_query"]
            if directory == "gt_state_map":
                color = STATE_COLORS[record["gt_state"]]
                label = record["gt_state"]
                marker = "o"
            elif directory == "renderer_event_partition":
                raw = record["raw_event"]
                color = {RAW_REACHED: STATE_COLORS["OBSERVED"], RAW_TERMINATED: STATE_COLORS["OCCLUDED"], RAW_UNRESOLVED: STATE_COLORS["UNRESOLVED"]}[raw]
                label = raw
                marker = "s"
            elif directory == "evidence_reason_map":
                raw = record["raw_event"]
                color = {RAW_REACHED: STATE_COLORS["OBSERVED"], RAW_TERMINATED: STATE_COLORS["OCCLUDED"], RAW_UNRESOLVED: STATE_COLORS["UNRESOLVED"]}[raw]
                label = "direct/reached" if raw == RAW_REACHED else "positive blocker" if raw == RAW_TERMINATED else "insufficient"
                marker = "^"
            else:
                disagree = record["gt_state"] == "OBSERVED" and record["raw_event"] == RAW_TERMINATED
                color = STATE_COLORS["OCCLUDED"] if disagree else STATE_COLORS["OBSERVED"]
                label = "DISAGREEMENT" if disagree else "matched control"
                marker = "X" if disagree else "o"
            ax.scatter(x, y, c=[color], marker=marker, s=100, edgecolors="black", linewidths=0.5, label=label)
            ax.annotate(record["id"], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(plot_title)
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"{directory}.png", format="png")
        plt.close(fig)
    return [str((out / name / f"{name}.png").resolve()) for name in visual_specs]


def run(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    scene = controlled_scene()
    records = renderer_event_records()
    gate = evaluate_contract_gate()
    aggregate = aggregate_raw_query_events([record["raw_event"] for record in records[:3]])
    _write_json(out / "controlled_scene.json", scene)
    _write_json(out / "renderer_event_records.json", records)
    _write_json(out / "subject_type_audit.json", [item.as_dict() for item in existing_evidence_subject_audit()])
    visual_paths = _render_visuals(out, records)

    report = {
        "batch": "Worklog 179 — Renderer-Native Three-State Observation Evidence Contract",
        "architecture_verdict": gate["verdict"],
        "stop_conditions": gate["stop_conditions"],
        "1. INTENT ALIGNMENT": {
            "status": "PASS",
            "diagnostic_only": True,
            "worklog_178_continued": False,
            "surface_nurbs_tsdf_continuation_modified": False,
        },
        "2. SUBJECT-TYPE AUDIT": {
            "status": "PASS",
            "subjects": [item.as_dict() for item in existing_evidence_subject_audit()],
            "primitive_point_query_relabeling": False,
        },
        "3. EXISTING OBSERVATION-EVIDENCE IMPLEMENTATION": {
            "torch_observation_evidence": {
                "subject": "arbitrary world sample queried against per-camera rendered depth",
                "event": "valid depth plus depth_epsilon ordering",
                "continuous_or_discrete": "continuous input with chosen epsilon band",
                "uses_tuned_magnitude_threshold": True,
            },
            "contributor_diagnostic": "forward_accepted[g] plus pixel contributor identity",
            "query_diagnostic": "qdepth query_reached/query_terminated and provenance fields",
        },
        "4. RENDERER EVIDENCE SEMANTICS": {
            "forward_accepted": "primitive-camera forward accepted contribution",
            "representative_id": "pixel median-event contributor identity",
            "qdepth_termination": "positive renderer traversal termination before query",
            "qdepth_reached": "accepted traversal event at or beyond query depth; not a surface hit at query",
            "median_depth": "pixel scalar proxy; CUDA inverted expected reciprocal depth",
        },
        "5. FORMAL OBSERVED / OCCLUDED / UNRESOLVED CONTRACT": {
            "status": "NOT_DEFINED",
            "primitive": "OBSERVED witness exists; OCCLUDED witness absent; false contribution remains UNRESOLVED",
            "point_query": "raw three-way event partition exists, but it is not promoted to epistemic F",
            "mutual_exclusivity_raw_events": True,
            "deterministic_raw_events": True,
        },
        "6. THRESHOLD-FREE CONTRACT AUDIT": gate["threshold_free"],
        "7. DIAGNOSTIC RASTERIZER CHANGES": {
            "status": "NONE_REQUIRED",
            "existing_qdepth_fields_sufficient_for_audit": True,
            "canonical_renderer_changed": False,
            "trainer_changed": False,
            "new_threshold_added": False,
        },
        "8. SYNTHETIC TRUE-GT SCENE": {
            "status": "COMPLETE_CONTROL_ONLY",
            "scene": scene,
            "renderer_event_records": str((out / "renderer_event_records.json").resolve()),
            "live_cuda_replay": "not repeated; prior W165/W168 CUDA controls retained as measured evidence",
        },
        "9. PER-CAMERA EVIDENCE": {
            "per_camera_rule": "record raw event per query slot; no cross-camera collapse in the event record",
            "front_camera_records": 3,
            "offset_camera_records": 1,
        },
        "10. GLOBAL AGGREGATION": {
            "raw_rule": "reached witness dominates; termination global only if every relevant view terminates; unresolved blocks global termination",
            "majority_vote": False,
            "confidence_weighted_vote": False,
            "control_aggregate_first_three_records": aggregate,
        },
        "11. DEPTH BASELINE": {
            "status": "RETAINED_HISTORICAL_REFERENCE_ONLY",
            "rule": "historical median-depth ordering uses depth_epsilon and is not threshold-free F",
            "independent_counterexample_source": "Worklog 165 strict m < z_query < z_star and valid-median/no-blocker cases",
        },
        "12. TRANSMITTANCE BASELINE": {
            "status": "NOT_PROMOTED",
            "rule": "raw qdepth termination is retained as renderer event; no independent transmittance score threshold was introduced",
            "limitation": "canonical termination constant belongs to renderer traversal and does not identify physical blocker geometry",
        },
        "13. THREE-STATE CONFUSION / ACCOUNTING": {
            "status": "N/A_F_REJECTED_BEFORE_METRICS",
            "reason": "No promoted F exists; forcing a 3x3 confusion matrix would misrepresent rejected event semantics",
            "raw_event_counts": {event: sum(record["raw_event"] == event for record in records) for event in (RAW_REACHED, RAW_TERMINATED, RAW_UNRESOLVED)},
            "unsupported_forced_decision_count": 0,
            "unresolved_to_occluded_collapse_count": 0,
            "unresolved_to_observed_collapse_count": 0,
            "observed_to_occluded_error": "N/A",
            "occluded_to_observed_error": "N/A",
            "evidence_coverage": {"subjects_with_relevant_camera": len(records), "direct_contribution_evidence": 1, "blocker_event_evidence": 2, "lacking_sufficient_evidence": 1},
        },
        "14. COUNTEREXAMPLES": {
            "status": "FAILS_SEMANTIC_PROMOTION",
            "controlled_adversarial_count": 1,
            "categories": {
                "depth_says_occluded_renderer_event_says_observed": "Worklog 165 curved-sphere strict Candidate-B counterexamples retained",
                "depth_says_observed_renderer_event_says_occluded": "not claimed from this audit; no favorable selection",
                "transmittance_forces_state_where_raw_event_unresolved": "not promoted; unresolved raw event retained",
                "raw_event_failure_against_independent_gt": 1,
            },
            "representative_records": [record for record in records if record["id"] == "adversarial_renderer_termination"],
        },
        "15. REVIEW EXPORTS": {
            "visualizations": visual_paths,
            "png_primary": True,
            "ppm_count": 0,
            "readme_per_visualization": True,
            "f_map_exported": False,
        },
        "16. TESTING": {
            "focused_contract_tests": "implemented in tests/test_worklog_179_renderer_native_three_state_observation_evidence_contract.py",
            "canonical_renderer_regression": "not required; no renderer source or production API changed",
            "surface_dependency_check": True,
        },
        "17. ARCHITECTURE VERDICT": {
            "answer": "NO",
            "verdict": gate["verdict"],
            "exact_blockers": gate["stop_conditions"],
            "reason": gate["reason"],
            "paper_novelty_claim": False,
        },
        "18. PROMOTED / RETAINED / REJECTED / OPEN": {
            "promoted": [],
            "retained": ["W160/W164 primitive observation separation", "qdepth raw event fields", "W165/W168 independent counterexamples", "all historical surface/NURBS/TSDF behavior"],
            "rejected": ["primitive contribution as point-query occlusion", "median depth as physical blocker", "zero contribution as OCCLUDED", "new tuned F score/threshold", "F implementation before semantic closure"],
            "open": ["independent blocker identity or sound point-query observation witness", "future architecture review", "no automatic continuation to NURBS/completion"],
        },
    }
    _write_json(out / "worklog_179_report.json", report)
    _write_readme(
        out / "README.md",
        "Worklog 179 output",
        "기존 renderer diagnostic event를 subject-type별로 감사하고, analytic blocker control에서 F 승격 여부를 판정한다.",
        "F는 semantic gate 실패로 정의하지 않았다. renderer event raw partition은 기록용이며 physical surface existence나 latent eligibility를 뜻하지 않는다.",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/179_renderer_native_three_state_observation_evidence_contract"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

