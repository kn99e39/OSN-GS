from __future__ import annotations

"""Worklog 178 -- diagnostic attribution for the frozen W97 partition.

The W97 implementation is imported as a baseline only.  This script never
changes its configuration, never replaces its subset IDs, and never feeds a
diagnostic result back into a partition.  It runs the required synthetic
fixtures, replays the current W97 checkpoint when available, and writes one
matched diagnostic report plus review exports.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.surface.torch_coverage_first_subset_partition import (
    CoverageFirstPartitionConfig,
    build_candidate_graph,
)
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    RegionCoherenceConfig,
    partition_surfels_region_coherent,
    region_coherent_accounting,
)
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel
from osn_gs.surface.worklog_178_region_merge_diagnostics import (
    REAL_TABLE_RIM_PROVENANCE_GAP,
    REGION_CONCENTRATION_BREAKS_CURVATURE,
    LOCAL_GRAPH_BREAKS_CURVATURE,
    MergeChronology,
    deterministic_shortest_path,
    diagnostic_contract_summary,
    graph_support_diagnostic,
    chronology_to_npz,
    path_diagnostic,
    plain_w96_subset_ids,
    replay_w97_region_growth,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt"
DEFAULT_WL155 = REPO_ROOT / "output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit"
DEFAULT_OUT = REPO_ROOT / "output/178_region_level_anti_chaining_contract_discovery"


@dataclass(frozen=True)
class Fixture:
    name: str
    category: str
    positions: torch.Tensor
    normals: torch.Tensor
    path_indices: tuple[int, int]
    description: str


class Orientation:
    def __init__(self, positions: torch.Tensor, normals: torch.Tensor, ids: torch.Tensor | None = None):
        self.positions = positions
        self.surface_normal = normals
        self.gaussian_ids = torch.arange(len(positions), dtype=torch.int64) if ids is None else ids


def _mesh_sheet(rows: int, columns: int, pitch: float, *, z: float = 0.0, x_offset: float = 0.0) -> torch.Tensor:
    u = torch.arange(rows, dtype=torch.float32) * pitch + float(x_offset)
    v = (torch.arange(columns, dtype=torch.float32) - (columns - 1) / 2.0) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1), torch.full((rows * columns,), z)], dim=1)


def _planar_fixture() -> Fixture:
    positions = _mesh_sheet(14, 14, 0.10)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(len(positions), 1)
    return Fixture("planar_positive", "PLANAR_POSITIVE", positions, normals, (0, 13), "one coherent planar surfel sheet")


def _cylinder_fixture(name: str, degrees: float, category: str, description: str) -> Fixture:
    theta = torch.linspace(0.0, math.radians(degrees), 25)
    width = torch.linspace(-0.22, 0.22, 7)
    tt, yy = torch.meshgrid(theta, width, indexing="ij")
    positions = torch.stack([torch.cos(tt), yy, torch.sin(tt)], dim=-1).reshape(-1, 3)
    normals = torch.stack([torch.cos(tt), torch.zeros_like(tt), torch.sin(tt)], dim=-1).reshape(-1, 3)
    centre = 3
    start = centre
    end = (len(theta) - 1) * len(width) + centre
    return Fixture(name, category, positions, normals, (start, end), description)


def _pathological_chain_fixture() -> Fixture:
    left = _mesh_sheet(12, 9, 0.10, x_offset=0.0)
    right = _mesh_sheet(12, 9, 0.10, x_offset=2.10)
    left_normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(len(left), 1)
    right_angle = math.radians(120.0)
    right_normals = torch.tensor([[math.sin(right_angle), 0.0, math.cos(right_angle)]]).repeat(len(right), 1)
    bridge_x = torch.linspace(1.10, 2.00, 10)
    bridge = torch.stack([bridge_x, torch.zeros_like(bridge_x), torch.zeros_like(bridge_x)], dim=1)
    bridge_theta = torch.linspace(0.0, right_angle, len(bridge))
    bridge_normals = torch.stack([torch.sin(bridge_theta), torch.zeros_like(bridge_theta), torch.cos(bridge_theta)], dim=1)
    positions = torch.cat([left, bridge, right], dim=0)
    normals = torch.cat([left_normals, bridge_normals, right_normals], dim=0)
    start = len(left) + 0
    end = len(left) + len(bridge) - 1
    return Fixture(
        "pathological_chain_negative",
        "PATHOLOGICAL_CHAIN_NEGATIVE",
        positions,
        normals,
        (start, end),
        "two broad structurally distinct sheets joined by a narrow locally plausible normal chain",
    )


def _parallel_shortcut_fixture() -> Fixture:
    first = _mesh_sheet(12, 12, 0.10, z=0.0)
    second = _mesh_sheet(12, 12, 0.10, z=0.15, x_offset=0.03)
    positions = torch.cat([first, second], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(len(positions), 1)
    return Fixture(
        "parallel_shortcut_negative",
        "PARALLEL_SHORTCUT_NEGATIVE",
        positions,
        normals,
        (0, 11),
        "historical W97 positional-shortcut fixture: nearby parallel sheets",
    )


def fixtures() -> list[Fixture]:
    return [
        _planar_fixture(),
        _cylinder_fixture("smooth_curved_positive", 120.0, "SMOOTH_CURVED_POSITIVE", "continuous cylinder sheet with 120 degree total normal span"),
        _cylinder_fixture("strongly_bent_continuous_positive", 90.0, "STRONGLY_BENT_CONTINUOUS_POSITIVE", "continuous quarter-cylinder, tabletop-to-side-style accumulated bend"),
        _pathological_chain_fixture(),
        _parallel_shortcut_fixture(),
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subset_summary(subset_ids: np.ndarray) -> dict[str, Any]:
    if len(subset_ids) == 0:
        return {"subset_count": 0, "largest_subset_size": 0, "largest_subset_fraction": 0.0, "subset_sizes": []}
    _, counts = np.unique(subset_ids, return_counts=True)
    counts = np.sort(counts)[::-1]
    return {
        "subset_count": int(len(counts)),
        "largest_subset_size": int(counts[0]),
        "largest_subset_fraction": float(counts[0] / len(subset_ids)),
        "subset_sizes_desc": counts[:16].astype(int).tolist(),
    }


def _first_critical_event(chronology: MergeChronology) -> dict[str, Any] | None:
    event = chronology.first_rejection()
    return None if event is None else event.to_dict()


def _failure_stage(
    fixture: Fixture,
    plain_ids: np.ndarray,
    w97_ids: np.ndarray,
    chronology: MergeChronology,
) -> str:
    if len(plain_ids) > 0 and len(np.unique(plain_ids)) > 1 and fixture.category != "PARALLEL_SHORTCUT_NEGATIVE":
        return LOCAL_GRAPH_BREAKS_CURVATURE
    if len(np.unique(w97_ids)) > 1 and chronology.rejected_events:
        return REGION_CONCENTRATION_BREAKS_CURVATURE
    if len(np.unique(w97_ids)) == 1:
        return "RETAINED_ONE_REGION_OR_W97_DID_NOT_STOP_CHAIN"
    return "NO_REGION_LEVEL_REJECTION_OBSERVED"


def _write_readme(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _plot_fixture_views(root: Path, fixture: Fixture, partition: Any, chronology: MergeChronology, support: dict[str, Any] | None) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    positions = fixture.positions.detach().cpu().numpy()
    normals = fixture.normals.detach().cpu().numpy()
    subset_ids = partition.subset_ids.detach().cpu().numpy().astype(np.int64)
    accepted = partition.graph.accepted_edges.detach().cpu().numpy().astype(np.int64)
    views: dict[str, str] = {}

    def setup(name: str, title: str) -> tuple[Any, Any, Path]:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        _write_readme(directory / "README.md", f"""# {name}

{title}

입력은 fixture의 원본 position/normal row와 frozen W97 subset이다. 색은 진단용이며 membership을 바꾸지 않는다. RGB_REFERENCE는 구조 연속성의 증명이 아니다.
""")
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        return figure, axis, directory

    # A. final W97 subset membership
    figure, axis, directory = setup("GAUSSIAN_SUBSET_WORLD", "W97 final subset IDs in fixture world coordinates.")
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=subset_ids, s=16, cmap="turbo")
    axis.set_title(f"{fixture.name}: W97 subset IDs")
    figure.tight_layout(); output = directory / "subset_world.png"; figure.savefig(output, dpi=150); plt.close(figure); views["GAUSSIAN_SUBSET_WORLD"] = str(output)

    # B. exact accepted edge crop (the fixture itself is the frozen crop).
    figure, axis, directory = setup("LOCAL_ACCEPTED_GRAPH", "Every accepted local edge in the fixture; no scene-wide edge is added.")
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="black", s=10)
    for left, right in accepted.tolist():
        axis.plot(*positions[[left, right]].T, color="0.65", linewidth=0.35, alpha=0.35)
    axis.set_title(f"{fixture.name}: accepted local graph")
    figure.tight_layout(); output = directory / "accepted_graph.png"; figure.savefig(output, dpi=150); plt.close(figure); views["LOCAL_ACCEPTED_GRAPH"] = str(output)

    # C. first concentration rejection boundary.
    figure, axis, directory = setup("REGION_REJECTION_EVENT", "The first recorded W97 region-concentration rejection, if one exists.")
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="0.72", s=10)
    critical = chronology.first_rejection()
    if critical is not None:
        left, right = critical.endpoint_indices
        axis.plot(*positions[[left, right]].T, color="red", linewidth=3.0)
        axis.scatter(positions[[left, right], 0], positions[[left, right], 1], positions[[left, right], 2], c="red", s=40)
    axis.set_title(f"{fixture.name}: first rejection")
    figure.tight_layout(); output = directory / "first_rejection.png"; figure.savefig(output, dpi=150); plt.close(figure); views["REGION_REJECTION_EVENT"] = str(output)

    # D. deterministic sparse intrinsic normal field.
    figure, axis, directory = setup("NORMAL_FIELD", "Intrinsic fixture normals; arrows are diagnostic display only.")
    stride = max(1, len(positions) // 240)
    selected = np.arange(0, len(positions), stride)
    axis.quiver(positions[selected, 0], positions[selected, 1], positions[selected, 2], normals[selected, 0], normals[selected, 1], normals[selected, 2], length=0.12, normalize=True, color="tab:blue", linewidth=0.5)
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="0.15", s=7)
    axis.set_title(f"{fixture.name}: intrinsic normal field")
    figure.tight_layout(); output = directory / "normal_field.png"; figure.savefig(output, dpi=150); plt.close(figure); views["NORMAL_FIELD"] = str(output)

    # E. trajectory of the full scatter spectrum for recorded events.
    figure, axis = plt.subplots(figsize=(8, 4))
    q = np.asarray([event.hypothetical_post_spectrum for event in chronology.events], dtype=np.float64)
    if len(q):
        axis.plot(q[:, 0], label="q1")
        axis.plot(q[:, 1], label="q2")
        axis.plot(q[:, 2], label="q3")
        rejected = [index for index, event in enumerate(chronology.events) if event.result == "rejected_region_concentration"]
        if rejected:
            axis.scatter(rejected, q[rejected, 0], c="red", s=12, label="rejected")
    axis.set_xlabel("recorded inter-component merge event")
    axis.set_ylabel("normalized scatter eigenvalue")
    axis.legend(loc="best")
    directory = root / "MERGE_TRAJECTORY"; directory.mkdir(parents=True, exist_ok=True)
    _write_readme(directory / "README.md", "W97 merge-growth q1/q2/q3 trajectory. Red points are W97 concentration rejections; no q threshold is introduced here.")
    figure.tight_layout(); output = directory / "merge_trajectory.png"; figure.savefig(output, dpi=150); plt.close(figure); views["MERGE_TRAJECTORY"] = str(output)

    # F. neutral synthetic RGB context.
    figure, axis, directory = setup("RGB_REFERENCE", "Neutral coordinate-derived context only; not a continuity proof.")
    colour = positions[:, 2] if np.ptp(positions[:, 2]) > 1e-9 else positions[:, 0]
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colour, s=12, cmap="gray")
    axis.set_title(f"{fixture.name}: RGB reference placeholder")
    figure.tight_layout(); output = directory / "rgb_reference.png"; figure.savefig(output, dpi=150); plt.close(figure); views["RGB_REFERENCE"] = str(output)
    return views


def _run_fixture(fixture: Fixture, out: Path) -> dict[str, Any]:
    orientation = Orientation(fixture.positions, fixture.normals)
    config = RegionCoherenceConfig()
    with torch.no_grad():
        partition = partition_surfels_region_coherent(orientation, config)
    graph = partition.graph
    baseline_ids = partition.subset_ids.detach().cpu().numpy().astype(np.int64)
    plain_ids = plain_w96_subset_ids(graph, config.local)
    chronology = replay_w97_region_growth(orientation, config, graph=graph, record_policy="all_fixture_inter_component_merges")
    # Replaying merge chronology is observation only.  The baseline partition
    # is the sole source of the reported W97 final IDs.
    contract = diagnostic_contract_summary(baseline_ids, baseline_ids.copy(), chronology)
    path_indices = deterministic_shortest_path(graph, fixture.path_indices[0], fixture.path_indices[1], bounded_nodes=10000)
    path = path_diagnostic(orientation, path_indices) if path_indices is not None else {"available": False, "reason": "no_path_in_frozen_accepted_graph"}
    support = graph_support_diagnostic(graph, chronology, chronology.first_rejection())
    case_root = out / "synthetic" / fixture.name
    case_root.mkdir(parents=True, exist_ok=True)
    chronology_to_npz(case_root / "merge_chronology.npz", chronology)
    (case_root / "merge_chronology.json").write_text(json.dumps([event.to_dict() for event in chronology.events], indent=2), encoding="utf-8")
    views = _plot_fixture_views(case_root / "review_views", fixture, partition, chronology, support)
    accounting = region_coherent_accounting(partition)
    stage = _failure_stage(fixture, plain_ids, baseline_ids, chronology)
    result = {
        "fixture": {"name": fixture.name, "category": fixture.category, "description": fixture.description, "row_count": len(fixture.positions)},
        "w97_final_outcome": _subset_summary(baseline_ids),
        "w96_plain_cc_outcome": _subset_summary(plain_ids),
        "failure_success_stage": stage,
        "w97_accounting": accounting,
        "first_critical_merge_event": _first_critical_event(chronology),
        "chronology": chronology.summary(),
        "full_scatter_spectrum_available": True,
        "local_normal_evolution": path,
        "graph_support": support,
        "diagnostic_invariant": contract,
        "review_views": views,
        "architectural_interpretation": "measured_fixture_evidence_only; synthetic PASS is not real-scene architecture viability",
    }
    _write_readme(case_root / "README.md", f"""# {fixture.name}

이 fixture는 Worklog 178 synthetic contract `{fixture.category}`다. 원본 row의 world position과 intrinsic normal을 unchanged W97에 넣고, region merge chronology만 읽기 전용으로 관찰했다. W97 subset membership은 진단량으로 재판정하지 않았다.

`GAUSSIAN_SUBSET_WORLD`, `LOCAL_ACCEPTED_GRAPH`, `REGION_REJECTION_EVENT`, `NORMAL_FIELD`, `MERGE_TRAJECTORY`, `RGB_REFERENCE`는 각각 final subset, accepted local graph, first concentration rejection, normal field, q-spectrum trajectory, neutral context다. 색과 선은 membership/physical continuity의 증명이 아니다.

Synthetic fixture는 semantics separation 용도이며, 이 결과만으로 real-scene viability를 선언하지 않는다.
""")
    return result


def _load_model_orientation(checkpoint: Path, device: str) -> tuple[Any, dict[str, Any], torch.Tensor]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
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
    full = derive_surface_orientation_from_surfel(model)
    selector = torch.nonzero(~model.is_uncertain.reshape(-1).to(torch.bool), as_tuple=False).reshape(-1)
    orientation = replace(
        full,
        gaussian_ids=full.gaussian_ids[selector], positions=full.positions[selector],
        tangent_axis_u=full.tangent_axis_u[selector], tangent_axis_v=full.tangent_axis_v[selector],
        surface_normal=full.surface_normal[selector], tangent_scale_u=full.tangent_scale_u[selector],
        tangent_scale_v=full.tangent_scale_v[selector],
    )
    return model, payload, orientation


def _real_review_manifest(out: Path, wl155: Path) -> dict[str, Any]:
    source_review = wl155 / "review_views"
    target = out / "real_scene_qualitative_review"
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for case_name in ("table_rim_side_context", "patio_hedge_context"):
        case_root = target / case_name
        for camera in ("DSC08043.JPG", "DSC07960.JPG", "DSC08003.JPG"):
            for source_name, target_name in (
                ("A_original_scene", "RGB_REFERENCE"),
                ("C_accepted_region_ids", "GAUSSIAN_SUBSET_WORLD"),
                ("E_boundary_conflict", "REGION_REJECTION_EVENT"),
            ):
                source = source_review / "cameras" / camera / source_name / "render.png"
                if source.exists():
                    destination = case_root / target_name / f"{camera}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    copied.append(str(destination))
        for directory in case_root.rglob("*"):
            if directory.is_dir():
                _write_readme(directory / "README.md", "이 real-scene PNG는 W155의 동일 checkpoint/camera/renderer review를 provenance context로 재사용한 것이다. Table Rim/side stable-ID lineage가 없으므로 이 view는 membership 또는 physical continuity의 증명이 아니다.")
    _write_readme(target / "README.md", "W178 real-scene review manifest. Table Rim/side는 historical W97 stable-ID crosswalk가 없어 context-only다. patio→hedge는 W96 giant component/W97 replay provenance를 우선한다. RGB는 secondary context이며 continuity proof가 아니다.")
    return {
        "available": bool(copied),
        "source": str(source_review),
        "copied_context_png_count": len(copied),
        "copied_context_pngs": copied,
        "table_rim_context_is_not_positive_control": True,
        "rgb_is_not_continuity_proof": True,
    }


def _run_real(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    if not args.checkpoint.exists():
        return {"available": False, "reason": "historical/current checkpoint is absent", "table_rim_verdict": REAL_TABLE_RIM_PROVENANCE_GAP}
    model, payload, orientation = _load_model_orientation(args.checkpoint, args.device)
    config = RegionCoherenceConfig()
    with torch.no_grad():
        partition = partition_surfels_region_coherent(orientation, config)
    graph = partition.graph
    baseline_ids = partition.subset_ids.detach().cpu().numpy().astype(np.int64)
    plain_ids = plain_w96_subset_ids(graph, config.local)
    giant_id = int(np.bincount(plain_ids).argmax())
    negative_mask = plain_ids == giant_id
    chronology = replay_w97_region_growth(
        orientation,
        config,
        graph=graph,
        lineage_mask=negative_mask,
        record_policy="all_inter_component_merges_touching_reproducible_W96_giant_component",
        capture_state_statistics=False,
    )
    chronology_to_npz(out / "real_negative_patio_hedge_merge_chronology.npz", chronology)
    negative_members = np.flatnonzero(negative_mask)
    descendant, descendant_counts = np.unique(baseline_ids[negative_mask], return_counts=True)
    order = np.argsort(-descendant_counts, kind="stable")
    accepted_edges = graph.accepted_edges.detach().cpu().numpy().astype(np.int64)
    internal_edges = accepted_edges[negative_mask[accepted_edges[:, 0]] & negative_mask[accepted_edges[:, 1]]]
    stable_ids = orientation.gaussian_ids.detach().cpu().numpy().astype(np.int64)
    first_rejection = chronology.first_rejection()
    negative = {
        "available": True,
        "provenance_status": "PROVENANCE_LIMITED",
        "historical_exact_lineage_recovered": False,
        "historical_artifact_level_evidence": {
            "W96_giant_member_count": 894378,
            "W96_giant_fraction": 0.7470,
            "W97_candidate_edge_count": 6048719,
            "W97_spatial_edge_count": 5156342,
            "W97_accepted_local_edge_count": 4015325,
            "W97_region_coherence_rejected_merge_count": 553357,
            "W97_descendant_region_count": 31564,
            "W97_largest_descendant_member_count": 253853
        },
        "historical_label": "patio→hedge/background giant lineage from W96/W97 qualitative review",
        "semantic_crosswalk_to_every_stable_id": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "checkpoint_iteration": int(payload.get("iteration", 0)),
        "active_orientation_row_count": int(len(orientation.positions)),
        "current_checkpoint_replay_membership_relationship": {"giant_subset_id": giant_id, "giant_member_count": int(negative_mask.sum()), "giant_fraction": float(negative_mask.mean())},
        "w97_membership_relationship": {
            "descendant_region_count": int(len(descendant)),
            "largest_descendants": [{"region_id": int(descendant[index]), "member_count": int(descendant_counts[index])} for index in order[:32]],
        },
        "accepted_local_graph_edges_internal_to_current_replay_giant": int(len(internal_edges)),
        "current_replay_region_coherence_rejection_boundary_event_count": int(len(chronology.rejected_events)),
        "first_region_coherence_rejection": None if first_rejection is None else first_rejection.to_dict(),
        "stable_gaussian_id_provenance": {
            "available": True,
            "count": int(len(negative_members)),
            "min": int(stable_ids[negative_members].min()),
            "max": int(stable_ids[negative_members].max()),
            "sample_first_32_by_row": stable_ids[negative_members[:32]].tolist(),
        },
        "chronology": chronology.summary(),
        "graph_support_at_first_rejection": graph_support_diagnostic(graph, chronology, first_rejection),
        "diagnostic_invariant": diagnostic_contract_summary(baseline_ids, baseline_ids.copy(), chronology),
        "interpretation": "W96/W97 historical artifact-level counts are preserved, but the current checkpoint replay has a different population (historical 1,197,331 vs current 1,190,469), so current stable IDs and chronology are diagnostic-only and are not claimed as an exact historical semantic crosswalk",
    }
    manifest = _real_review_manifest(out, args.wl155)
    return {
        "available": True,
        "table_rim_side": {
            "verdict": REAL_TABLE_RIM_PROVENANCE_GAP,
            "available_historical_stable_id_lineage": False,
            "reason": "W145/W155 retained a fixed image-space table_side annotation and PARTIAL/MIXED qualitative review, but no historical W97 stable-ID lineage or region-rejection join",
            "invalid_substitute_used": False,
            "context_review_only": True,
        },
        "patio_hedge_background": negative,
        "review_manifest": manifest,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    _write_readme(args.out / "README.md", """# Worklog 178 — W97 Region-Level Anti-Chaining Contract Discovery

이 output은 immutable W97 baseline의 read-only merge chronology diagnostic이다. q2/q3, angular span, path, graph support는 어떤 membership threshold도 만들지 않는다. 모든 visualization directory에는 입력/상태/legend/한계 README가 있다.

Real Table Rim/side는 historical stable-ID crosswalk 부재로 `REAL_TABLE_RIM_PROVENANCE_GAP`이며 W171 curved/vase를 positive substitute로 사용하지 않는다. Synthetic fixture PASS는 real architecture viability가 아니다.
""")
    synthetic_results = [_run_fixture(fixture, args.out) for fixture in fixtures()]
    real_result = {"available": False, "skipped": True, "reason": "--skip-real"}
    if not args.skip_real:
        real_result = _run_real(args, args.out)
    comparison = []
    for result in synthetic_results:
        comparison.append({
            "case": result["fixture"]["name"],
            "category": result["fixture"]["category"],
            "w97_final_outcome": result["w97_final_outcome"],
            "failure_success_stage": result["failure_success_stage"],
            "first_critical_merge_event": result["first_critical_merge_event"],
            "full_scatter_spectrum": result["first_critical_merge_event"].get("hypothetical_post_spectrum") if result["first_critical_merge_event"] else None,
            "local_angle_statistics": result["first_critical_merge_event"].get("hypothetical_post_local_angle_statistics") if result["first_critical_merge_event"] else None,
            "pathwise_normal_evolution": result["local_normal_evolution"],
            "accepted_cross_edge_count": result["graph_support"].get("accepted_local_cross_edge_count"),
            "distinct_endpoint_support": result["graph_support"].get("distinct_endpoint_count_each_side"),
            "alternate_path_support": result["graph_support"].get("alternate_accepted_edge_support_count"),
            "articulation_like": result["graph_support"].get("articulation_like"),
            "region_sizes": result["w97_final_outcome"]["subset_sizes_desc"],
            "interpretation": "measured quantities above; no replacement contract implemented",
        })
    report = {
        "status": "COMPLETE_W178_DIAGNOSTIC_ONLY",
        "batch": "Worklog 178 — W97 Region-Level Anti-Chaining Contract Discovery: Genuine Curvature vs Pathological Chaining Attribution",
        "intent_alignment": {
            "w97_baseline_immutable": True,
            "parameters_tuned": False,
            "diagnostics_feed_back_into_partition": False,
            "new_region_level_contract_implemented": False,
            "synthetic_results_are_not_real_viability": True,
        },
        "implementation_fidelity": {
            "local_candidate_graph": "existing CandidateGraph from W97; no new graph",
            "knn_and_spacing_gate": "unchanged W97 graph",
            "pairwise_normal_relation": "unsigned existing alignment",
            "merge_order": "descending local alignment then ascending edge endpoints, same as W97",
            "concentration_floor": "RegionCoherenceConfig.concentration_floor() only",
            "ownership_propagation": "W97 baseline partition remains source of subset IDs; diagnostic replay stops at structural merge state",
            "stable_ids": "carried from orientation, no new IDs",
        },
        "real_positive_control_grounding_table_rim_side": real_result.get("table_rim_side", {"verdict": REAL_TABLE_RIM_PROVENANCE_GAP}),
        "historical_negative_control_grounding_patio_hedge_background": real_result.get("patio_hedge_background", real_result),
        "table_rim_failure_stage_attribution": {"verdict": real_result.get("table_rim_side", {}).get("verdict", REAL_TABLE_RIM_PROVENANCE_GAP), "stage": None, "reason": "real positive lineage is not recoverable"},
        "w97_region_merge_chronology": {"synthetic": [{"case": result["fixture"]["name"], "chronology": result["chronology"]} for result in synthetic_results], "real": real_result.get("patio_hedge_background")},
        "global_concentration_trajectory": {"source": "chronology event hypothetical_post_spectrum q1; no q2/q3 threshold"},
        "full_scatter_spectrum_diagnostic": {"source": "every materialized synthetic inter-component event; real negative NPZ when replayed", "fields": ["q1", "q2", "q3"]},
        "local_normal_evolution_diagnostic": {"source": "deterministic accepted-graph shortest path per synthetic fixture", "real_positive": "OPEN_PROVENANCE"},
        "graph_support_diagnostic": {"source": "critical recorded event endpoint neighborhood; diagnostic-only", "no_bridge_veto": True, "no_consensus_gate": True},
        "synthetic_contracts": synthetic_results,
        "matched_positive_vs_negative_comparison": comparison,
        "real_scene_qualitative_review": real_result.get("review_manifest", {"available": False, "reason": "real review unavailable"}),
        "architecture_attribution": {
            "real": [REAL_TABLE_RIM_PROVENANCE_GAP, "PROVENANCE_LIMITED"],
            "synthetic": "diagnostic evidence only; no architecture promotion",
            "candidate_next_hypothesis": "OPEN_PENDING_REAL_TABLE_RIM_LINEAGE",
        },
        "promoted_retained_rejected_open": {
            "promoted": [],
            "retained": ["W96/W97 historical implementation", "intrinsic t_w semantics", "W154/W155/W171-W177 artifacts", "WHO/WHERE separation", "current production behavior"],
            "rejected": ["all replacement region-level algorithms in this batch", "using W171 curved/vase as Table Rim positive"],
            "open": [REAL_TABLE_RIM_PROVENANCE_GAP, "real positive failure-stage attribution", "one minimal next contract hypothesis"],
        },
        "preserved_inputs": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256_file(args.checkpoint) if args.checkpoint.exists() else None,
            "wl155_source": str(args.wl155),
        },
    }
    report_path = args.out / "worklog_178_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wl155", type=Path, default=DEFAULT_WL155)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-real", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "synthetic_case_count": len(report["synthetic_contracts"]), "real_available": report["historical_negative_control_grounding_patio_hedge_background"].get("available", False)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
