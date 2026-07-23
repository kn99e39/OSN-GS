from __future__ import annotations

"""Diagnostics-only merge-only proxy surface decomposition.

This module implements Stage 3 of the proxy-based surface decomposition plan.
It deliberately does not create or modify ``SurfaceComponentSet`` and is not
called by the production boundary-first constructor.  Atomic adaptive-voxel
leaves are agglomerated only inside this diagnostic result.

Every candidate evaluation computes and preserves all raw proxy, support, and
layer diagnostics before an ordered admissibility reason is selected.  Merge
priority is separate from admissibility: the prototype uses merged normalized
quadratic RMS as its sole primary ordering metric and canonical region
provenance as the deterministic tie-break.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping

from osn_gs.surface.torch_surface_candidate_graph import (
    SurfaceCandidateGraph,
    build_surface_cell_candidate_graph,
)
from osn_gs.surface.torch_surface_proxy import (
    ProxyMergeDiagnostics,
    QuadraticSurfaceProxy,
    fit_quadratic_surface_proxy,
    merge_proxy_diagnostics,
)
from osn_gs.surface.torch_voxel_hierarchy import TorchVoxelGaussianHierarchy
from osn_gs.utils.torch_ops import require_torch


PRIORITY_METRIC = "merged_normalized_quadratic_rms"
GATE_ORDER = (
    "invalid_proxy",
    "insufficient_support",
    "disconnected_support",
    "multi_layer_inconsistency",
    "excessive_proxy_distortion",
    "excessive_error_increase",
)


@dataclass(frozen=True)
class ProxySurfaceDecompositionConfig:
    """Explicit provisional Stage 3 thresholds.

    These defaults are feasibility-prototype values, not production values.
    All geometric thresholds are dimensionless and scale-normalized.
    """

    max_normalized_proxy_rms: float = 0.1
    max_normalized_error_increase: float = 0.01
    max_support_gap_over_spacing: float = 4.0
    max_layer_separation: float = 0.2
    min_layer_rms_ratio: float = 1.0
    min_layer_normalized_error_increase: float = 0.001
    max_layer_residual_concentration: float = 2.0
    minimum_support: int = 6
    proxy_regularization: float = 1e-6
    candidate_radius_factor: float = 0.25
    candidate_max_neighbors: int = 0
    support_gap_quantile: float = 0.02
    max_proxy_condition_number: float = 1e10

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "max_normalized_proxy_rms",
            "max_normalized_error_increase",
            "max_support_gap_over_spacing",
            "max_layer_separation",
            "min_layer_rms_ratio",
            "min_layer_normalized_error_increase",
            "max_layer_residual_concentration",
            "proxy_regularization",
            "candidate_radius_factor",
            "max_proxy_condition_number",
        )
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.minimum_support) < 1:
            raise ValueError("minimum_support must be >= 1")
        if int(self.candidate_max_neighbors) < 0:
            raise ValueError("candidate_max_neighbors must be >= 0")
        if not 0.0 <= float(self.support_gap_quantile) <= 1.0:
            raise ValueError("support_gap_quantile must be in [0, 1]")

    def payload(self) -> dict[str, Any]:
        return {
            "max_normalized_proxy_rms": float(self.max_normalized_proxy_rms),
            "max_normalized_error_increase": float(
                self.max_normalized_error_increase
            ),
            "max_support_gap_over_spacing": float(
                self.max_support_gap_over_spacing
            ),
            "max_layer_separation": float(self.max_layer_separation),
            "min_layer_rms_ratio": float(self.min_layer_rms_ratio),
            "min_layer_normalized_error_increase": float(
                self.min_layer_normalized_error_increase
            ),
            "max_layer_residual_concentration": float(
                self.max_layer_residual_concentration
            ),
            "minimum_support": int(self.minimum_support),
            "proxy_regularization": float(self.proxy_regularization),
            "candidate_radius_factor": float(self.candidate_radius_factor),
            "candidate_max_neighbors": int(self.candidate_max_neighbors),
            "support_gap_quantile": float(self.support_gap_quantile),
            "max_proxy_condition_number": float(
                self.max_proxy_condition_number
            ),
        }


@dataclass
class ProxySurfaceRegion:
    component_id: int
    member_leaf_ids: list[str]
    gaussian_indices: Any
    proxy: QuadraticSurfaceProxy

    def payload(self) -> dict[str, Any]:
        return {
            "component_id": int(self.component_id),
            "member_leaf_ids": list(self.member_leaf_ids),
            "member_leaf_count": len(self.member_leaf_ids),
            "point_count": int(self.gaussian_indices.numel()),
            "support_mass": float(self.proxy.effective_weight_sum),
            "proxy": self.proxy.payload(),
        }


@dataclass
class ProxyPairEvaluation:
    evaluation_index: int
    region_a: str
    region_b: str
    member_leaf_ids_a: list[str]
    member_leaf_ids_b: list[str]
    candidate_leaf_pairs: list[list[str]]
    priority_metric: str
    priority_value: float
    admissible: bool
    decision_reason: str
    failed_gates: list[str]
    gate_results: dict[str, bool]
    diagnostics: ProxyMergeDiagnostics

    def payload(self) -> dict[str, Any]:
        return {
            "evaluation_index": int(self.evaluation_index),
            "region_a": self.region_a,
            "region_b": self.region_b,
            "member_leaf_ids_a": list(self.member_leaf_ids_a),
            "member_leaf_ids_b": list(self.member_leaf_ids_b),
            "candidate_leaf_pairs": [list(pair) for pair in self.candidate_leaf_pairs],
            "priority_metric": self.priority_metric,
            "priority_value": float(self.priority_value),
            "admissible": bool(self.admissible),
            "decision_reason": self.decision_reason,
            "failed_gates": list(self.failed_gates),
            "gate_results": dict(self.gate_results),
            "diagnostics": self.diagnostics.payload(),
        }


@dataclass
class ProxySurfaceDecompositionDiagnostics:
    config: ProxySurfaceDecompositionConfig
    initial_leaf_ids: list[str]
    candidate_edge_count: int
    candidate_unique_edge_count: int
    final_regions: list[ProxySurfaceRegion]
    leaf_component_id: dict[str, int]
    pair_evaluations: list[ProxyPairEvaluation]
    merge_history: list[dict[str, Any]]
    stale_queue_entry_count: int
    decision_reason_counts: dict[str, int]
    termination_reason: str

    def component_count(self) -> int:
        return len(self.final_regions)

    def payload(self) -> dict[str, Any]:
        raw = {
            "schema_version": 1,
            "stage": "proxy_decomposition_stage3_merge_only_diagnostics",
            "production_membership_changed": False,
            "config": self.config.payload(),
            "priority": {
                "primary_metric": PRIORITY_METRIC,
                "tie_break": "canonical_member_leaf_ids",
                "admissibility_is_separate": True,
            },
            "gate_order": list(GATE_ORDER),
            "initial_leaf_ids": list(self.initial_leaf_ids),
            "initial_region_count": len(self.initial_leaf_ids),
            "candidate_edge_count": int(self.candidate_edge_count),
            "candidate_unique_edge_count": int(self.candidate_unique_edge_count),
            "final_component_count": self.component_count(),
            "leaf_component_id": dict(sorted(self.leaf_component_id.items())),
            "final_regions": [region.payload() for region in self.final_regions],
            "pair_evaluation_count": len(self.pair_evaluations),
            "pair_evaluations": [item.payload() for item in self.pair_evaluations],
            "merge_count": len(self.merge_history),
            "merge_history": list(self.merge_history),
            "stale_queue_entry_count": int(self.stale_queue_entry_count),
            "decision_reason_counts": dict(sorted(self.decision_reason_counts.items())),
            "termination_reason": self.termination_reason,
        }
        return _json_safe(raw)


@dataclass
class _RegionState:
    internal_id: int
    member_leaf_ids: tuple[str, ...]
    gaussian_indices: Any
    proxy: QuadraticSurfaceProxy
    active: bool = True

    @property
    def key(self) -> str:
        return "+".join(self.member_leaf_ids)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_int_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _canonical_leaf_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _coerce_config(
    config: ProxySurfaceDecompositionConfig | Mapping[str, Any] | None,
) -> ProxySurfaceDecompositionConfig:
    if config is None:
        return ProxySurfaceDecompositionConfig()
    if isinstance(config, ProxySurfaceDecompositionConfig):
        return config
    if isinstance(config, Mapping):
        return ProxySurfaceDecompositionConfig(**dict(config))
    raise TypeError("config must be ProxySurfaceDecompositionConfig, mapping, or None")


def _validate_candidate_config(
    graph: SurfaceCandidateGraph,
    config: ProxySurfaceDecompositionConfig,
) -> None:
    graph_radius = graph.config.get("radius_factor")
    graph_neighbors = graph.config.get("max_neighbors")
    if graph_radius is not None and not math.isclose(
        float(graph_radius), float(config.candidate_radius_factor), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "candidate graph radius_factor does not match config.candidate_radius_factor"
        )
    if graph_neighbors is not None and int(graph_neighbors) != int(
        config.candidate_max_neighbors
    ):
        raise ValueError(
            "candidate graph max_neighbors does not match config.candidate_max_neighbors"
        )


def _evaluate_pair(
    evaluation_index: int,
    region_a: _RegionState,
    region_b: _RegionState,
    candidate_leaf_pairs: tuple[tuple[str, str], ...],
    points: Any,
    config: ProxySurfaceDecompositionConfig,
) -> ProxyPairEvaluation:
    region_a, region_b = sorted(
        (region_a, region_b), key=lambda region: region.member_leaf_ids
    )
    diagnostics = merge_proxy_diagnostics(
        points[region_a.gaussian_indices],
        points[region_b.gaussian_indices],
        regularization=float(config.proxy_regularization),
        support_gap_quantile=float(config.support_gap_quantile),
    )

    # Compute every gate from the already-complete raw diagnostic record.
    # No earlier failure suppresses later diagnostic computation.
    condition_numbers = (
        diagnostics.proxy_a.condition_number,
        diagnostics.proxy_b.condition_number,
        diagnostics.merged_proxy.condition_number,
    )
    gate_failed = {
        "invalid_proxy": (
            not diagnostics.valid
            or any(
                (not math.isfinite(value))
                or value > float(config.max_proxy_condition_number)
                for value in condition_numbers
            )
        ),
        "insufficient_support": (
            diagnostics.proxy_a.point_count < int(config.minimum_support)
            or diagnostics.proxy_b.point_count < int(config.minimum_support)
            or diagnostics.proxy_a.effective_weight_sum < float(config.minimum_support)
            or diagnostics.proxy_b.effective_weight_sum < float(config.minimum_support)
        ),
        "disconnected_support": (
            diagnostics.scale_normalized_support_gap
            > float(config.max_support_gap_over_spacing)
        ),
        "multi_layer_inconsistency": (
            diagnostics.layer_separation_score
            > float(config.max_layer_separation)
            and diagnostics.merged_to_child_rms_ratio
            > float(config.min_layer_rms_ratio)
            and (
                diagnostics.normalized_error_increase
                > float(config.min_layer_normalized_error_increase)
                or diagnostics.merged_proxy.residual_concentration
                < float(config.max_layer_residual_concentration)
            )
        ),
        "excessive_proxy_distortion": (
            diagnostics.merged_proxy.normalized_rms_residual
            > float(config.max_normalized_proxy_rms)
        ),
        "excessive_error_increase": (
            diagnostics.normalized_error_increase
            > float(config.max_normalized_error_increase)
        ),
    }
    failed_gates = [name for name in GATE_ORDER if gate_failed[name]]
    admissible = not failed_gates
    decision_reason = "merge" if admissible else failed_gates[0]
    return ProxyPairEvaluation(
        evaluation_index=evaluation_index,
        region_a=region_a.key,
        region_b=region_b.key,
        member_leaf_ids_a=list(region_a.member_leaf_ids),
        member_leaf_ids_b=list(region_b.member_leaf_ids),
        candidate_leaf_pairs=[list(pair) for pair in candidate_leaf_pairs],
        priority_metric=PRIORITY_METRIC,
        priority_value=float(diagnostics.merged_proxy.normalized_rms_residual),
        admissible=admissible,
        decision_reason=decision_reason,
        failed_gates=failed_gates,
        gate_results={name: not gate_failed[name] for name in GATE_ORDER},
        diagnostics=diagnostics,
    )


def build_proxy_surface_components_diagnostics(
    cells: TorchVoxelGaussianHierarchy,
    points: Any,
    candidate_graph: SurfaceCandidateGraph | None = None,
    config: ProxySurfaceDecompositionConfig | Mapping[str, Any] | None = None,
) -> ProxySurfaceDecompositionDiagnostics:
    """Run deterministic diagnostics-only merge-only agglomeration.

    ``cells`` is the existing adaptive voxel hierarchy.  Ground-truth labels,
    scene names, topology, and production component membership are neither
    accepted nor consulted.
    """

    torch = require_torch()
    resolved_config = _coerce_config(config)
    if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    if candidate_graph is None:
        candidate_graph = build_surface_cell_candidate_graph(
            cells,
            points,
            radius_factor=float(resolved_config.candidate_radius_factor),
            max_neighbors=int(resolved_config.candidate_max_neighbors),
        )
    else:
        _validate_candidate_config(candidate_graph, resolved_config)

    node_lookup = {node.node_id: node for node in cells.nodes}
    initial_leaf_ids = sorted(set(candidate_graph.node_ids))
    missing = [leaf_id for leaf_id in initial_leaf_ids if leaf_id not in node_lookup]
    if missing:
        raise ValueError(f"candidate graph references unknown cells: {missing}")

    regions: dict[int, _RegionState] = {}
    leaf_region: dict[str, int] = {}
    next_region_id = 0
    for leaf_id in initial_leaf_ids:
        node = node_lookup[leaf_id]
        if node.gaussian_indices is None or int(node.gaussian_indices.numel()) == 0:
            raise ValueError(f"candidate cell {leaf_id} has no Gaussian support")
        indices = node.gaussian_indices.to(dtype=torch.long)
        proxy = fit_quadratic_surface_proxy(
            points[indices], regularization=float(resolved_config.proxy_regularization)
        )
        region = _RegionState(
            internal_id=next_region_id,
            member_leaf_ids=(leaf_id,),
            gaussian_indices=indices,
            proxy=proxy,
        )
        regions[next_region_id] = region
        leaf_region[leaf_id] = next_region_id
        next_region_id += 1

    input_edge_count = len(candidate_graph.edges)
    atomic_pairs = sorted(
        {
            _canonical_leaf_pair(edge.cell_a, edge.cell_b)
            for edge in candidate_graph.edges
            if edge.cell_a != edge.cell_b
        }
    )
    unknown_edge_nodes = sorted(
        {
            leaf_id
            for pair in atomic_pairs
            for leaf_id in pair
            if leaf_id not in leaf_region
        }
    )
    if unknown_edge_nodes:
        raise ValueError(
            f"candidate edges reference nodes outside graph.node_ids: {unknown_edge_nodes}"
        )

    adjacency: dict[int, set[int]] = {region_id: set() for region_id in regions}
    edge_sources: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for leaf_a, leaf_b in atomic_pairs:
        region_a, region_b = leaf_region[leaf_a], leaf_region[leaf_b]
        if region_a == region_b:
            continue
        pair = _canonical_int_pair(region_a, region_b)
        edge_sources.setdefault(pair, set()).add((leaf_a, leaf_b))
        adjacency[region_a].add(region_b)
        adjacency[region_b].add(region_a)

    evaluations: list[ProxyPairEvaluation] = []
    queue: list[tuple[float, tuple[str, ...], tuple[str, ...], int, int, int]] = []

    def evaluate_and_enqueue(region_a_id: int, region_b_id: int) -> None:
        pair = _canonical_int_pair(region_a_id, region_b_id)
        sources = tuple(sorted(edge_sources[pair]))
        evaluation = _evaluate_pair(
            len(evaluations),
            regions[pair[0]],
            regions[pair[1]],
            sources,
            points,
            resolved_config,
        )
        evaluations.append(evaluation)
        if evaluation.admissible:
            state_a, state_b = regions[pair[0]], regions[pair[1]]
            key_a, key_b = sorted(
                (state_a.member_leaf_ids, state_b.member_leaf_ids)
            )
            heapq.heappush(
                queue,
                (
                    evaluation.priority_value,
                    key_a,
                    key_b,
                    evaluation.evaluation_index,
                    pair[0],
                    pair[1],
                ),
            )

    for region_pair in sorted(edge_sources):
        evaluate_and_enqueue(*region_pair)

    merge_history: list[dict[str, Any]] = []
    stale_count = 0
    while queue:
        _, _, _, evaluation_index, region_a_id, region_b_id = heapq.heappop(queue)
        pair = _canonical_int_pair(region_a_id, region_b_id)
        region_a = regions[region_a_id]
        region_b = regions[region_b_id]
        if (
            not region_a.active
            or not region_b.active
            or pair not in edge_sources
        ):
            stale_count += 1
            continue

        evaluation = evaluations[evaluation_index]
        if not evaluation.admissible:
            raise RuntimeError("non-admissible pair reached merge queue")
        merged_members = tuple(
            sorted(set(region_a.member_leaf_ids) | set(region_b.member_leaf_ids))
        )
        merged_indices = torch.cat(
            [region_a.gaussian_indices, region_b.gaussian_indices]
        )
        merged_indices = torch.sort(merged_indices).values
        merged_region = _RegionState(
            internal_id=next_region_id,
            member_leaf_ids=merged_members,
            gaussian_indices=merged_indices,
            proxy=evaluation.diagnostics.merged_proxy,
        )
        regions[next_region_id] = merged_region
        adjacency[next_region_id] = set()

        neighbor_ids = sorted(
            (adjacency[region_a_id] | adjacency[region_b_id])
            - {region_a_id, region_b_id},
            key=lambda item: regions[item].member_leaf_ids,
        )
        new_sources: dict[int, set[tuple[str, str]]] = {}
        for neighbor_id in neighbor_ids:
            sources: set[tuple[str, str]] = set()
            for old_id in (region_a_id, region_b_id):
                old_pair = _canonical_int_pair(old_id, neighbor_id)
                sources.update(edge_sources.pop(old_pair, set()))
                adjacency[neighbor_id].discard(old_id)
            if sources:
                new_sources[neighbor_id] = sources

        edge_sources.pop(pair, None)
        region_a.active = False
        region_b.active = False
        adjacency[region_a_id].clear()
        adjacency[region_b_id].clear()

        for neighbor_id, sources in new_sources.items():
            new_pair = _canonical_int_pair(next_region_id, neighbor_id)
            edge_sources[new_pair] = sources
            adjacency[next_region_id].add(neighbor_id)
            adjacency[neighbor_id].add(next_region_id)

        merge_history.append(
            {
                "merge_index": len(merge_history),
                "pair_evaluation_index": evaluation_index,
                "region_a": evaluation.region_a,
                "region_b": evaluation.region_b,
                "merged_region": merged_region.key,
                "member_leaf_ids": list(merged_members),
                "point_count": int(merged_indices.numel()),
                "support_mass": float(merged_region.proxy.effective_weight_sum),
                "priority_metric": PRIORITY_METRIC,
                "priority_value": evaluation.priority_value,
            }
        )
        new_region_id = next_region_id
        next_region_id += 1
        for neighbor_id in sorted(
            adjacency[new_region_id], key=lambda item: regions[item].member_leaf_ids
        ):
            evaluate_and_enqueue(new_region_id, neighbor_id)

    active_regions = sorted(
        (region for region in regions.values() if region.active),
        key=lambda region: region.member_leaf_ids,
    )
    final_regions: list[ProxySurfaceRegion] = []
    leaf_component_id: dict[str, int] = {}
    for component_id, region in enumerate(active_regions):
        final_regions.append(
            ProxySurfaceRegion(
                component_id=component_id,
                member_leaf_ids=list(region.member_leaf_ids),
                gaussian_indices=region.gaussian_indices,
                proxy=region.proxy,
            )
        )
        for leaf_id in region.member_leaf_ids:
            leaf_component_id[leaf_id] = component_id

    reason_counts: dict[str, int] = {}
    for evaluation in evaluations:
        reason_counts[evaluation.decision_reason] = (
            reason_counts.get(evaluation.decision_reason, 0) + 1
        )
    termination_reason = (
        "one_region"
        if len(final_regions) == 1
        else "no_admissible_candidate_pairs"
    )
    return ProxySurfaceDecompositionDiagnostics(
        config=resolved_config,
        initial_leaf_ids=initial_leaf_ids,
        candidate_edge_count=input_edge_count,
        candidate_unique_edge_count=len(atomic_pairs),
        final_regions=final_regions,
        leaf_component_id=leaf_component_id,
        pair_evaluations=evaluations,
        merge_history=merge_history,
        stale_queue_entry_count=stale_count,
        decision_reason_counts=reason_counts,
        termination_reason=termination_reason,
    )


__all__ = [
    "GATE_ORDER",
    "PRIORITY_METRIC",
    "ProxyPairEvaluation",
    "ProxySurfaceDecompositionConfig",
    "ProxySurfaceDecompositionDiagnostics",
    "ProxySurfaceRegion",
    "build_proxy_surface_components_diagnostics",
]
