"""Frozen A/B replay for representative termination-neighborhood scale only.

The replay freezes representatives, reliability, regions, accepted topology,
canonical frames, and downstream configuration. It then re-evaluates only the
read-only support-termination extraction radius under two scales:

* ``equivalent_tangent_scale`` (legacy footprint branch)
* ``candidate_scale`` (RepresentativeGraphScale branch)

No production state is mutated by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


PHYSICAL_REASON = "observed_support_termination"
BRANCH_NAMES = ("footprint", "candidate")


def _digest(ids: Sequence[Any]) -> str:
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()[:16]


def _edge_id(edge: Any) -> str:
    return f"{edge.source_half_edge_id}->{edge.target_half_edge_id}"


def _component_bucket(component: Any | None) -> str:
    if component is None:
        return "rejected"
    if component.ordering_state == "ordered_closed_loop":
        return "closed_loop"
    if component.ordering_state in ("ordered_open_chain", "ambiguous_ordering", "isolated_boundary_candidate"):
        return "open_chain"
    if component.branch_node_ids or component.ordering_state == "branching_boundary_graph":
        return "branch"
    return "rejected"


def _candidate_identity(candidate: Any) -> str:
    return str(candidate.half_edge_id)


def _physical_assertion_key(candidate: Any) -> tuple[Any, str, tuple[int, int, int]]:
    quantized = tuple(int(round(value * 1000.0)) for value in candidate.world_position)
    return (candidate.source_region_id, candidate.boundary_reason, quantized)


def _supporting_ids(positions: Any, ids: Sequence[Any], regions: Any, frames: Sequence[Any | None], scales: Any) -> dict[Any, tuple[Any, ...]]:
    index = {sid: i for i, sid in enumerate(ids)}
    adjacency = {i: [] for i in range(len(ids))}
    for region in regions.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                a, b = index[left], index[right]
                adjacency[a].append(b)
                adjacency[b].append(a)
    out: dict[Any, tuple[Any, ...]] = {}
    for source, sid in enumerate(ids):
        frame = frames[source]
        if frame is None:
            out[sid] = ()
            continue
        normal = frame.oriented_normal
        support = []
        for target in adjacency[source]:
            delta = positions[target] - positions[source]
            tangent = delta - normal * (delta @ normal)
            distance = float(tangent.norm())
            if 1e-8 < distance <= float(scales[source]) * 4.0:
                support.append(ids[target])
        out[sid] = tuple(sorted(support, key=str))
    return out


def _coverage(positions: Any, ids: Sequence[Any], construction: Any, frames: Sequence[Any | None], scales: Any) -> dict[str, Any]:
    regions = construction.surface_regions
    index = {sid: i for i, sid in enumerate(ids)}
    adjacency = {i: [] for i in range(len(ids))}
    for region in regions.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                a, b = index[left], index[right]
                adjacency[a].append(b)
                adjacency[b].append(a)
    eligible = degree_two = below_two = accepted_total = accepted_seen = 0
    for source in range(len(ids)):
        if regions.node_region_id[source] < 0 or frames[source] is None:
            continue
        eligible += 1
        neighbors = adjacency[source]
        if len(neighbors) >= 2:
            degree_two += 1
        radius = float(scales[source]) * 4.0
        seen = sum(float((positions[target] - positions[source]).norm()) <= radius for target in neighbors)
        accepted_total += len(neighbors)
        accepted_seen += seen
        if len(neighbors) >= 2 and seen < 2:
            below_two += 1
    return {
        "eligible_region_members": eligible,
        "accepted_degree_ge_2": degree_two,
        "accepted_degree_ge_2_with_radius_neighbor_count_lt_2": below_two,
        "accepted_neighbor_recall": accepted_seen / max(accepted_total, 1),
    }


def _component_maps(components: Sequence[Any]) -> tuple[dict[str, Any], dict[Any, Any]]:
    by_halfedge = {}
    by_source = {}
    for component in components:
        for halfedge_id in component.ordered_half_edge_ids:
            by_halfedge[halfedge_id] = component
        for source_id in component.ordered_source_ids:
            by_source[source_id] = component
    return by_halfedge, by_source


def _materialization_map(attempts: Sequence[Any]) -> dict[str, Any]:
    return {item.input.source_boundary_component_id: item for item in attempts}


def _materialize_attempts(positions: Any, ids: Sequence[Any], regions: Any, components: Sequence[Any]) -> tuple[Any, ...]:
    from osn_gs.surface.torch_visible_boundary_materialization_adapter import materialize_visible_boundary_component

    id_to_index = {item: index for index, item in enumerate(ids)}
    attempts = []
    for component in components:
        boundary_ids = tuple(item for item in component.ordered_source_ids if item in id_to_index)
        if not boundary_ids:
            continue
        boundary_points = positions[torch.tensor([id_to_index[item] for item in boundary_ids], device=positions.device)]
        region = next((item for item in regions.regions if item.region_id == component.region_id), None)
        interior_ids = tuple(
            item for item in (region.core_member_ids if region else ())
            if item not in set(boundary_ids) and item in id_to_index
        )
        if not interior_ids:
            interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item in id_to_index)
        interior_points = positions[torch.tensor([id_to_index[item] for item in interior_ids], device=positions.device)] if interior_ids else boundary_points
        attempts.append(materialize_visible_boundary_component(component, boundary_points, interior_points, boundary_ids=boundary_ids, interior_ids=interior_ids))
    return tuple(attempts)


def _lineage_records(raw: Sequence[Any], normalized: Sequence[Any], support_ids: dict[Any, tuple[Any, ...]], compatibility: Sequence[Any], components: Sequence[Any], attempts: Sequence[Any], *, scale_name: str, numeric_radius_by_source: dict[Any, float]) -> list[dict[str, Any]]:
    raw_by_id = {_candidate_identity(candidate): candidate for candidate in raw}
    normalized_by_id = {_candidate_identity(candidate): candidate for candidate in normalized}
    component_by_halfedge, _ = _component_maps(components)
    materialization_by_component = _materialization_map(attempts)
    compat_by_halfedge: dict[str, list[str]] = {}
    for edge in compatibility:
        eid = _edge_id(edge)
        compat_by_halfedge.setdefault(edge.source_half_edge_id, []).append(eid)
        compat_by_halfedge.setdefault(edge.target_half_edge_id, []).append(eid)
    records = []
    for raw_id, candidate in sorted(raw_by_id.items()):
        normalized_candidate = normalized_by_id.get(raw_id)
        component = component_by_halfedge.get(raw_id)
        materialization = materialization_by_component.get(component.component_id) if component is not None else None
        seed_admitted = bool(
            component is not None
            and component.ordering_state == "ordered_closed_loop"
            and component.role_candidate == "outer_boundary_candidate"
            and not component.branch_node_ids
        )
        rejection = None
        if normalized_candidate is None:
            rejection = "removed_by_normalization"
        elif component is None:
            rejection = "not_in_directed_ordering_component"
        elif not seed_admitted:
            rejection = ";".join(component.unresolved_reasons or (component.ordering_state, component.role_candidate))
        elif materialization is not None and materialization.state != "materialized":
            rejection = ";".join(materialization.review_reasons or (materialization.state,))
        records.append({
            "source_representative_stable_id": candidate.source_gaussian_id,
            "supporting_representative_stable_ids": list(support_ids.get(candidate.source_gaussian_id, ())),
            "region_id": candidate.source_region_id,
            "extraction_scale": scale_name,
            "numeric_radius": numeric_radius_by_source.get(candidate.source_gaussian_id),
            "raw_candidate_id": raw_id,
            "normalized_candidate_id": raw_id if normalized_candidate is not None else None,
            "typed_reason": candidate.boundary_reason,
            "sector_angular_evidence": {
                "gap_width_degrees": candidate.gap_width_degrees,
                "same_mode_support_count": candidate.same_mode_support_count,
                "same_mode_opacity_mass": candidate.same_mode_opacity_mass,
                "ambiguous_continuation_mass": candidate.ambiguous_continuation_mass,
                "competing_mode_mass": candidate.competing_mode_mass,
            },
            "directed_ordering_input_id": raw_id if candidate.boundary_reason == PHYSICAL_REASON else None,
            "compatibility_edge_ids": sorted(compat_by_halfedge.get(raw_id, ())),
            "ordered_component_id": component.component_id if component is not None else None,
            "component_state": _component_bucket(component),
            "seed_admission_result": "admitted" if seed_admitted else "rejected",
            "nurbs_materialization_result": materialization.state if materialization is not None else "not_attempted",
            "rejection_reason": rejection,
        })
    return records


def _branch_report(positions: Any, ids: Sequence[Any], construction: Any, frames: Sequence[Any | None], scales: Any, scale_name: str) -> dict[str, Any]:
    from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates, normalize_continuation_candidates
    from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
    from osn_gs.surface.torch_ordered_world_boundary_graph import build_boundary_compatibility

    raw = extract_support_termination_candidates(
        positions,
        None,
        scales,
        construction.surface_regions,
        ids=ids,
        canonical_frames=frames,
        affinity_graph=construction.manifold_affinity,
    )
    normalized = normalize_continuation_candidates(raw)
    compatibility = build_boundary_compatibility(normalized)
    directed_edges, components = recover_directed_boundary_components(normalized, construction.accepted_local_topology)
    attempts = _materialize_attempts(positions, ids, construction.surface_regions, components)

    raw_ids = [_candidate_identity(candidate) for candidate in raw]
    normalized_ids = [_candidate_identity(candidate) for candidate in normalized]
    physical = [candidate for candidate in normalized if candidate.boundary_reason == PHYSICAL_REASON]
    physical_keys = [_physical_assertion_key(candidate) for candidate in physical]
    duplicates = len(physical_keys) - len(set(physical_keys))
    typed = Counter(candidate.boundary_reason for candidate in normalized)
    rejection_histogram = Counter()
    component_by_halfedge, _ = _component_maps(components)
    materialization_by_component = _materialization_map(attempts)
    for candidate in normalized:
        component = component_by_halfedge.get(candidate.half_edge_id)
        materialization = materialization_by_component.get(component.component_id) if component is not None else None
        if component is None:
            rejection_histogram["not_in_directed_ordering_component"] += 1
        elif component.ordering_state != "ordered_closed_loop" or component.role_candidate != "outer_boundary_candidate" or component.branch_node_ids:
            reason = ";".join(component.unresolved_reasons or (component.ordering_state, component.role_candidate))
            rejection_histogram[reason] += 1
        elif materialization is not None and materialization.state != "materialized":
            rejection_histogram[";".join(materialization.review_reasons or (materialization.state,))] += 1
    removed = sorted(set(raw_ids) - set(normalized_ids))
    support_ids = _supporting_ids(positions, ids, construction.surface_regions, frames, scales)
    numeric_radius_by_source = {sid: float(scales[index]) * 4.0 for index, sid in enumerate(ids)}

    return {
        "representative_count": len(ids),
        "raw_termination_count": len(raw),
        "normalized_termination_count": len(normalized),
        "physical_assertion_count": len(physical),
        "duplicate_count": duplicates + len(removed),
        "directed_compatibility_edge_count": len(directed_edges),
        "ordered_component_count": len(components),
        "closed_loop_count": sum(c.ordering_state == "ordered_closed_loop" for c in components),
        "open_chain_count": sum(c.ordering_state in ("ordered_open_chain", "ambiguous_ordering", "isolated_boundary_candidate") for c in components),
        "branch_component_count": sum(c.branch_node_ids != () or c.ordering_state == "branching_boundary_graph" for c in components),
        "seed_admission_count": sum(c.ordering_state == "ordered_closed_loop" and c.role_candidate == "outer_boundary_candidate" and not c.branch_node_ids for c in components),
        "nurbs_materialization_count": sum(a.state == "materialized" for a in attempts),
        "rejection_reason_histogram": dict(sorted(rejection_histogram.items())),
        "raw_emitted": {"count": len(raw), "stable_id_set_hash": _digest([c.source_gaussian_id for c in raw])},
        "normalized": {"count": len(normalized), "stable_id_set_hash": _digest([c.source_gaussian_id for c in normalized]), "removed_candidate_ids": removed},
        "typed_provenance": {"counts": dict(sorted(typed.items())), "count": len(normalized), "stable_id_set_hash": _digest([c.source_gaussian_id for c in normalized])},
        "accepted_neighbor_coverage": _coverage(positions, ids, construction, frames, scales),
        "components": [
            {
                "component_id": c.component_id,
                "state": c.ordering_state,
                "component_state": _component_bucket(c),
                "stable_ids": list(c.ordered_source_ids),
                "branch_node_ids": list(c.branch_node_ids),
                "reasons": list(c.unresolved_reasons),
                "role_candidate": c.role_candidate,
            }
            for c in components
        ],
        "candidate_lineage": _lineage_records(
            raw,
            normalized,
            support_ids,
            directed_edges,
            components,
            attempts,
            scale_name=scale_name,
            numeric_radius_by_source=numeric_radius_by_source,
        ),
    }


def replay(positions: Any, ids: Sequence[Any], construction: Any, candidate_scale: Any) -> dict[str, Any]:
    from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames

    frames = construct_canonical_region_tangent_frames(
        positions,
        construction.covariance_frame,
        construction.reliability,
        construction.surface_regions,
        ids=ids,
    )
    footprint = construction.covariance_frame.equivalent_tangent_scale
    branches = {
        "footprint": _branch_report(positions, ids, construction, frames, footprint, "equivalent_tangent_scale"),
        "candidate": _branch_report(positions, ids, construction, frames, candidate_scale, "candidate_scale"),
    }
    fp_ids = {item["raw_candidate_id"] for item in branches["footprint"]["candidate_lineage"]}
    cand_ids = {item["raw_candidate_id"] for item in branches["candidate"]["candidate_lineage"]}
    branches["stable_id_diff"] = {
        "candidate_only_raw_candidate_ids": sorted(cand_ids - fp_ids),
        "footprint_only_raw_candidate_ids": sorted(fp_ids - cand_ids),
        "shared_raw_candidate_ids": sorted(cand_ids & fp_ids),
    }
    return branches


def _known_false_support(scene_name: str, candidate: dict[str, Any], labels: Sequence[str] | None) -> str | None:
    sid = int(candidate["source_representative_stable_id"])
    support = set(int(item) for item in candidate["supporting_representative_stable_ids"])
    label = labels[sid] if labels and sid < len(labels) else ""
    support_labels = {labels[item] for item in support if labels and item < len(labels)}
    if scene_name == "sphere" and candidate["typed_reason"] == PHYSICAL_REASON:
        return "closed_boundary_free_surface"
    if scene_name in ("thin_slab", "close_parallel_sheets", "thin_strip") and support_labels and any(item != label for item in support_labels):
        return "reaching_opposite_side_of_narrow_strip"
    if scene_name in ("box_with_bridge", "accepted_topology_bridge_contamination") and ("bridge" in support_labels or label == "bridge"):
        return "relying_on_contaminated_bridge"
    if scene_name == "genuine_narrow_connection":
        if (label == "left" and "right" in support_labels) or (label == "right" and "left" in support_labels):
            return "crossing_thin_neck"
    return None


def classify_false_support(scene_name: str, branch: dict[str, Any], labels: Sequence[str] | None = None) -> dict[str, Any]:
    true_positive = false_positive = ambiguous = 0
    reasons = Counter()
    seen_physical = set()
    for candidate in branch["candidate_lineage"]:
        if candidate["typed_reason"] != PHYSICAL_REASON:
            continue
        key = (
            candidate["region_id"],
            tuple(round(float(v), 3) for v in candidate.get("world_position", ())),
            candidate["source_representative_stable_id"],
        )
        reason = _known_false_support(scene_name, candidate, labels)
        if reason is not None:
            false_positive += 1
            reasons[reason] += 1
        elif scene_name in ("sphere", "thin_slab", "close_parallel_sheets", "thin_strip", "box_with_bridge", "accepted_topology_bridge_contamination"):
            true_positive += 1
        else:
            ambiguous += 1
        if key in seen_physical:
            false_positive += 1
            reasons["duplicate_physical_assertion"] += 1
        seen_physical.add(key)
    grounded = true_positive + false_positive
    precision = true_positive / grounded if grounded else None
    coverage = grounded / max(branch["physical_assertion_count"], 1) if branch["physical_assertion_count"] else None
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "ambiguous": ambiguous,
        "precision": precision,
        "coverage": coverage,
        "false_support_histogram": dict(sorted(reasons.items())),
        "ground_truth_note": "analytic labels/predicates used where available" if grounded else "fixture lacks sufficient ground truth for this classification",
    }


@dataclass(frozen=True)
class FixtureSpec:
    report_name: str
    factory_name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None


def _fixture_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec("close_parallel_sheets", "make_gaussian_reliability_scene", ("thin_slab",)),
        FixtureSpec("U_shaped_concavity", "make_missing_support_gap_scene", (0.35,)),
        FixtureSpec("narrow_neck", "make_genuine_narrow_connection_scene"),
        FixtureSpec("thin_strip", "make_gaussian_reliability_scene", ("thin_slab",)),
        FixtureSpec("high_valence_branching_topology", "make_gaussian_reliability_scene", ("box",)),
        FixtureSpec("accepted_topology_bridge_contamination", "make_gaussian_reliability_scene", ("box_with_bridge",)),
        FixtureSpec("box_faces_and_corners", "make_gaussian_reliability_scene", ("box",)),
        FixtureSpec("cylinder", "make_gaussian_reliability_scene", ("cylinder",)),
        FixtureSpec("sphere", "make_gaussian_reliability_scene", ("sphere",)),
    )


def _make_fixture(spec: FixtureSpec):
    from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene, make_missing_support_gap_scene
    from nurbs_constructor_benchmark.surface_region_adversarial_scenes import make_genuine_narrow_connection_scene

    factories = {
        "make_gaussian_reliability_scene": make_gaussian_reliability_scene,
        "make_missing_support_gap_scene": make_missing_support_gap_scene,
        "make_genuine_narrow_connection_scene": make_genuine_narrow_connection_scene,
    }
    return factories[spec.factory_name](*spec.args, **(spec.kwargs or {}))


def replay_fixture(scene: Any) -> dict[str, Any]:
    from osn_gs.core.torch_pipeline import _representative_knn_spacing
    from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
    from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
    from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

    positions = torch.as_tensor(scene.positions, dtype=torch.float32)
    covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
    ids = tuple(range(int(positions.shape[0])))
    frame = extract_covariance_frame(covariance)
    reliability = evaluate_structural_reliability(positions, frame)
    candidate_scale = _representative_knn_spacing(positions)
    construction = construct_visible_nurbs_from_gaussians(
        positions,
        covariance=covariance,
        stable_ids=ids,
        reliability=reliability,
        candidate_scale=candidate_scale,
        residual_scale=candidate_scale,
    )
    report = replay(positions, ids, construction, candidate_scale)
    labels = getattr(scene, "group_labels", None)
    for branch_name in BRANCH_NAMES:
        report[branch_name]["false_support"] = classify_false_support(scene.name, report[branch_name], labels)
    report["production_path_composition"] = _continuation_composition_from_construction(construction)
    report["construction"] = {
        "scene_name": scene.name,
        "region_count": len(construction.surface_regions.regions),
        "accepted_topology_edge_count": len(construction.accepted_local_topology),
        "construction_state": construction.construction_state,
    }
    return report


def replay_fixtures() -> dict[str, Any]:
    fixtures = {}
    for spec in _fixture_specs():
        scene = _make_fixture(spec)
        fixture_report = replay_fixture(scene)
        for branch_name in BRANCH_NAMES:
            fixture_report[branch_name]["fixture_report_name"] = spec.report_name
        fixtures[spec.report_name] = fixture_report
    return fixtures


def _continuation_composition_from_construction(construction: Any) -> dict[str, Any]:
    candidates = construction.boundary_halfedge_candidates
    sector = {c.half_edge_id for c in candidates if c.support_radius is None}
    continuation = {c.half_edge_id for c in candidates if c.support_radius is not None}
    normalized = {c.half_edge_id for c in construction.boundary_halfedge_candidates}
    component_by_halfedge, _ = _component_maps(construction.ordered_boundary_components)
    changed = []
    for candidate in candidates:
        component = component_by_halfedge.get(candidate.half_edge_id)
        if component is not None:
            changed.append({
                "candidate_id": candidate.half_edge_id,
                "source_representative_stable_id": candidate.source_gaussian_id,
                "path": "continuation" if candidate.support_radius is not None else "sector",
                "component_id": component.component_id,
                "component_state": _component_bucket(component),
                "ordering_state": component.ordering_state,
            })
    return {
        "sector_only_candidates": len(sector - continuation),
        "continuation_only_candidates": len(continuation - sector),
        "both_paths_candidates": len(sector & continuation),
        "removed_by_normalization": 0,
        "candidates_changing_downstream_component_state": changed,
        "note": "Production construction stores normalized candidates; use replay branch lineage for raw-to-normalized diffs.",
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()

    if args.fixtures or args.checkpoint is None:
        print(json.dumps({"fixtures": replay_fixtures()}, indent=2, sort_keys=True))
        return

    from frozen_core_seeding_replay import build_frozen_state
    from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

    state = build_frozen_state(args.checkpoint, args.cap)
    construction = construct_visible_nurbs_from_gaussians(
        state.rep_points,
        covariance=state.rep_covariance,
        stable_ids=state.rep_stable_ids,
        reliability=state.reliability,
        candidate_scale=state.candidate_scale,
        residual_scale=state.residual_scale,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "fixed_region_count": len(construction.surface_regions.regions),
        "dual_scale": replay(state.rep_points, state.rep_stable_ids, construction, state.candidate_scale),
        "production_path_composition": _continuation_composition_from_construction(construction),
        "note": "Representative-sector dual-scale replay freezes production topology and changes only extraction scale.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

