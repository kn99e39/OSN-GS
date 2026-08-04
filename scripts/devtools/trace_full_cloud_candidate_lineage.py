"""Full-cloud production candidate lineage trace for real frozen checkpoints.

Connects representative-sector raw candidates, full-cloud continuation raw
candidates, production raw/normalized candidates, directed ordering,
components, and materialization by candidate ID and source representative
stable ID. Diagnostic-only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

PHYSICAL = "observed_support_termination"


def _component_bucket(component: Any | None) -> str:
    if component is None:
        return "rejected"
    if component.ordering_state == "ordered_closed_loop":
        return "closed_loop"
    if component.branch_node_ids or component.ordering_state == "branching_boundary_graph":
        return "branch"
    if component.ordering_state in ("ordered_open_chain", "ambiguous_ordering", "isolated_boundary_candidate"):
        return "open_chain"
    return "rejected"


def _id(candidate: Any) -> str:
    return str(candidate.half_edge_id)


def _count_block(candidates: Sequence[Any]) -> dict[str, Any]:
    by_source: dict[Any, list[str]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.source_gaussian_id].append(_id(candidate))
    duplicate_sources = {str(k): sorted(v) for k, v in by_source.items() if len(v) > 1}
    physical = [c for c in candidates if c.boundary_reason == PHYSICAL]
    physical_sources: dict[Any, list[str]] = defaultdict(list)
    for candidate in physical:
        physical_sources[candidate.source_gaussian_id].append(_id(candidate))
    duplicate_physical_sources = {str(k): sorted(v) for k, v in physical_sources.items() if len(v) > 1}
    return {
        "candidate_count": len(candidates),
        "source_stable_id_count": len(by_source),
        "duplicate_source_count": len(duplicate_sources),
        "typed_counts": dict(sorted(Counter(c.boundary_reason for c in candidates).items())),
        "physical_candidate_count": len(physical),
        "physical_source_stable_id_count": len(physical_sources),
        "duplicate_physical_source_count": len(duplicate_physical_sources),
        "duplicate_sources": duplicate_sources,
        "duplicate_physical_sources": duplicate_physical_sources,
    }


def _coverage(positions: Any, ids: Sequence[Any], regions: Any, frames: Sequence[Any | None], scales: Any) -> dict[str, Any]:
    index = {sid: i for i, sid in enumerate(ids)}
    adjacency = {i: [] for i in range(len(ids))}
    for region in regions.regions:
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index:
                a, b = index[left], index[right]
                adjacency[a].append(b)
                adjacency[b].append(a)
    eligible = degree_two = no_neighbor = below_two = accepted_total = accepted_seen = 0
    for source in range(len(ids)):
        if regions.node_region_id[source] < 0 or frames[source] is None:
            continue
        eligible += 1
        radius = float(scales[source]) * 4.0
        seen = sum(float((positions[target] - positions[source]).norm()) <= radius for target in adjacency[source])
        accepted_total += len(adjacency[source])
        accepted_seen += seen
        if seen == 0:
            no_neighbor += 1
        if len(adjacency[source]) >= 2:
            degree_two += 1
            if seen < 2:
                below_two += 1
    return {
        "eligible_region_members": eligible,
        "accepted_degree_ge_2": degree_two,
        "no_neighbor_failure": no_neighbor,
        "accepted_degree_ge_2_with_radius_neighbor_count_lt_2": below_two,
        "accepted_neighbor_recall": accepted_seen / max(accepted_total, 1),
    }


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
        interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item not in set(boundary_ids) and item in id_to_index)
        if not interior_ids:
            interior_ids = tuple(item for item in (region.core_member_ids if region else ()) if item in id_to_index)
        interior_points = positions[torch.tensor([id_to_index[item] for item in interior_ids], device=positions.device)] if interior_ids else boundary_points
        attempts.append(materialize_visible_boundary_component(component, boundary_points, interior_points, boundary_ids=boundary_ids, interior_ids=interior_ids))
    return tuple(attempts)


def _raw_candidates(state: Any, construction: Any, frames: Sequence[Any | None], *, scales: Any, continuation: dict[Any, Any] | None) -> tuple[Any, ...]:
    from osn_gs.surface.torch_boundary_support_termination import extract_support_termination_candidates

    return extract_support_termination_candidates(
        state.rep_points,
        construction.covariance_frame.normal_candidate,
        scales,
        construction.surface_regions,
        ids=state.rep_stable_ids,
        sectors=8,
        canonical_frames=frames,
        continuation=continuation,
        affinity_graph=construction.manifold_affinity,
    )


def _build_context(checkpoint: Path, cap: int, device: str):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from frozen_core_seeding_replay import build_frozen_state
    from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
    from osn_gs.surface.torch_full_cloud_continuation_shell import ContinuationShellInput, build_continuation_shells_from_input
    from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

    state = build_frozen_state(checkpoint, cap, device=device)
    continuation_input = ContinuationShellInput(
        full_positions=state.full_points,
        full_frame=state.full_frame,
        full_intrinsic=state.full_intrinsic,
        full_opacity=state.full_opacity,
        full_stable_ids=state.full_stable_ids,
        nearest_representative_index=state.nearest_representative_index,
        representative_mean_spacing=state.representative_mean_spacing,
    )
    construction = construct_visible_nurbs_from_gaussians(
        state.rep_points,
        covariance=state.rep_covariance,
        stable_ids=state.rep_stable_ids,
        reliability=state.reliability,
        continuation_input=continuation_input,
        candidate_scale=state.candidate_scale,
        residual_scale=state.residual_scale,
    )
    frames = construct_canonical_region_tangent_frames(
        state.rep_points,
        construction.covariance_frame,
        construction.reliability,
        construction.surface_regions,
        ids=state.rep_stable_ids,
    )
    continuation = build_continuation_shells_from_input(
        continuation_input,
        state.rep_points,
        construction.covariance_frame,
        state.rep_stable_ids,
        construction.surface_regions,
        frames,
    )
    return state, construction, frames, continuation


def trace_checkpoint(checkpoint: Path, cap: int = 2048, device: str = "cpu") -> dict[str, Any]:
    from osn_gs.surface.torch_boundary_support_termination import normalize_continuation_candidates
    from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
    from osn_gs.surface.torch_ordered_world_boundary_graph import build_boundary_compatibility

    state, construction, frames, continuation = _build_context(checkpoint, cap, device)
    footprint_scale = construction.covariance_frame.equivalent_tangent_scale
    candidate_scale = state.candidate_scale

    sector_raw_footprint = _raw_candidates(state, construction, frames, scales=footprint_scale, continuation=None)
    sector_raw_candidate = _raw_candidates(state, construction, frames, scales=candidate_scale, continuation=None)
    production_raw_footprint = _raw_candidates(state, construction, frames, scales=footprint_scale, continuation=continuation)
    production_raw_candidate = _raw_candidates(state, construction, frames, scales=candidate_scale, continuation=continuation)
    normalized_candidate = normalize_continuation_candidates(production_raw_candidate)

    ordering_input = tuple(c for c in normalized_candidate if c.boundary_reason == PHYSICAL and c.ordering_state == "locally_chainable")
    compatibility = build_boundary_compatibility(normalized_candidate)
    directed_edges, components = recover_directed_boundary_components(normalized_candidate, construction.accepted_local_topology)
    attempts = _materialize_attempts(state.rep_points, state.rep_stable_ids, construction.surface_regions, components)

    sector_physical_sources = {c.source_gaussian_id for c in sector_raw_candidate if c.boundary_reason == PHYSICAL}
    continuation_physical_sources = {c.source_gaussian_id for c in production_raw_candidate if c.support_radius is not None and c.boundary_reason == PHYSICAL}
    production_raw_ids = {_id(c) for c in production_raw_candidate}
    normalized_ids = {_id(c) for c in normalized_candidate}
    ordering_input_ids = {_id(c) for c in ordering_input}
    component_by_candidate = {}
    for component in components:
        for cid in component.ordered_half_edge_ids:
            component_by_candidate[cid] = component
    materialized_by_component = {attempt.input.source_boundary_component_id: attempt for attempt in attempts}

    rows = []
    for candidate in sorted(production_raw_candidate, key=lambda c: _id(c)):
        cid = _id(candidate)
        component = component_by_candidate.get(cid)
        attempt = materialized_by_component.get(component.component_id) if component is not None else None
        rows.append({
            "candidate_id": cid,
            "source_representative_stable_id": candidate.source_gaussian_id,
            "provenance": "continuation" if candidate.support_radius is not None else "sector",
            "typed_reason": candidate.boundary_reason,
            "normalized": cid in normalized_ids,
            "directed_ordering_input": cid in ordering_input_ids,
            "component_id": component.component_id if component is not None else None,
            "component_state": _component_bucket(component),
            "materialized_boundary": attempt.state if attempt is not None else "not_attempted",
            "final_typed_state_count": 1 if isinstance(candidate.boundary_reason, str) and candidate.boundary_reason else 0,
        })

    return {
        "checkpoint": str(checkpoint),
        "cap": cap,
        "representative_count": len(state.rep_stable_ids),
        "region_count": len(construction.surface_regions.regions),
        "scale_comparison": {
            "footprint": {
                "accepted_neighbor_coverage": _coverage(state.rep_points, state.rep_stable_ids, construction.surface_regions, frames, footprint_scale),
                "sector_raw": _count_block(sector_raw_footprint),
                "production_raw": _count_block(production_raw_footprint),
            },
            "candidate": {
                "accepted_neighbor_coverage": _coverage(state.rep_points, state.rep_stable_ids, construction.surface_regions, frames, candidate_scale),
                "sector_raw": _count_block(sector_raw_candidate),
                "production_raw": _count_block(production_raw_candidate),
                "normalized": _count_block(normalized_candidate),
                "ordering_input": _count_block(ordering_input),
                "directed_compatibility_edge_count": len(directed_edges),
                "boundary_compatibility_edge_count": len(compatibility),
                "closed_component_count": sum(c.ordering_state == "ordered_closed_loop" for c in components),
                "materialized_boundary_count": sum(a.state == "materialized" for a in attempts),
            },
        },
        "production_composition": {
            "sector_only_physical_source_count": len(sector_physical_sources - continuation_physical_sources),
            "continuation_only_physical_source_count": len(continuation_physical_sources - sector_physical_sources),
            "both_physical_source_count": len(sector_physical_sources & continuation_physical_sources),
            "sector_physical_source_count": len(sector_physical_sources),
            "continuation_physical_source_count": len(continuation_physical_sources),
        },
        "trace_count_explanation": {
            "node_level_generated_physical_source_count": len({c.source_gaussian_id for c in normalized_candidate if c.boundary_reason == PHYSICAL}),
            "waterfall_physical_candidate_object_count": sum(c.boundary_reason == PHYSICAL for c in normalized_candidate),
            "difference_is_duplicate_source_count": sum(c.boundary_reason == PHYSICAL for c in normalized_candidate) - len({c.source_gaussian_id for c in normalized_candidate if c.boundary_reason == PHYSICAL}),
            "duplicate_physical_sources": _count_block(tuple(c for c in normalized_candidate if c.boundary_reason == PHYSICAL))["duplicate_physical_sources"],
        },
        "stage_exact_match": {
            "production_raw_vs_normalized_removed_candidate_ids": sorted(production_raw_ids - normalized_ids),
            "normalized_extra_candidate_ids": sorted(normalized_ids - production_raw_ids),
            "ordering_input_minus_component_ids": sorted(ordering_input_ids - set(component_by_candidate)),
            "component_minus_ordering_input_ids": sorted(set(component_by_candidate) - ordering_input_ids),
        },
        "all_candidates_have_exactly_one_typed_state": all(row["final_typed_state_count"] == 1 for row in rows),
        "candidate_lineage": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoints = args.checkpoint or [
        Path("output/osn_gs_scene/3000/checkpoint.pt"),
        Path("output/osn_gs_scene/5000/checkpoint.pt"),
        Path("output/osn_gs_scene/10000/checkpoint.pt"),
    ]
    reports = [trace_checkpoint(path, args.cap, args.device) for path in checkpoints]
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
