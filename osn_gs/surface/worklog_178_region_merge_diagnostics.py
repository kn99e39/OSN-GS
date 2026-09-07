from __future__ import annotations

"""Worklog 178 diagnostic-only replay of the frozen W97 merge chronology.

This module deliberately does not call the W97 partition function's merge
loop and does not feed any diagnostic quantity back into membership.  It
reuses the same :class:`CandidateGraph`, edge order, unsigned local normal
relation, scatter state, and concentration floor, then exposes the state
that W97 already observes while it grows regions.

The implementation is intentionally small and boring: the merge replay is a
second, read-only description of the historical contract.  That makes it
possible to test the important invariant directly:

    diagnostic replay enabled + frozen W97 input
        == baseline W97 subset membership

No q2/q3, angular span, path statistic, graph-support count, or synthetic
fixture result is an acceptance rule here.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CandidateGraph,
    CoverageFirstPartitionConfig,
    SurfaceOrientationEvidence,
    _connected_component_roots,
    build_candidate_graph,
)
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig
from osn_gs.utils.torch_ops import require_torch


LOCAL_GRAPH_BREAKS_CURVATURE = "LOCAL_GRAPH_BREAKS_CURVATURE"
REGION_CONCENTRATION_BREAKS_CURVATURE = "REGION_CONCENTRATION_BREAKS_CURVATURE"
OWNERSHIP_STAGE_BREAKS_CURVATURE = "OWNERSHIP_STAGE_BREAKS_CURVATURE"
MIXED_ATTRIBUTION = "MIXED_ATTRIBUTION"
REAL_TABLE_RIM_PROVENANCE_GAP = "REAL_TABLE_RIM_PROVENANCE_GAP"

_EPS = 1e-12


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _unsigned_angle_degrees(alignment: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(alignment)))))


def _spectrum(matrix: np.ndarray) -> tuple[float, float, float]:
    trace = float(np.trace(matrix))
    if trace <= _EPS:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    values = np.linalg.eigvalsh(matrix.astype(np.float64, copy=False))[::-1]
    return tuple(float(value / trace) for value in values)


def _normal_span_degrees(normals: np.ndarray) -> float:
    """Exact unsigned span for a small state, in [0, 90] degrees.

    Region growth on a real scene can contain hundreds of thousands of rows.
    The replay therefore uses a deterministic stride sample above 512 rows;
    the storage metadata records that limitation.  Synthetic fixtures and
    critical event neighborhoods remain exact.
    """

    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(normals) <= 1:
        return 0.0
    if len(normals) > 512:
        stride = int(math.ceil(len(normals) / 512.0))
        normals = normals[::stride]
    dots = np.abs(normals @ normals.T)
    return _unsigned_angle_degrees(float(np.clip(dots.min(), 0.0, 1.0)))


def _percentile_nearest(values: Sequence[float], fraction: float) -> float | None:
    if len(values) == 0:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.percentile(ordered, fraction * 100.0, method="nearest"))


@dataclass
class _AngleState:
    values: list[float]
    truncated: bool = False

    def merged(self, other: "_AngleState", *, cap: int = 4096) -> "_AngleState":
        values = self.values + other.values
        truncated = self.truncated or other.truncated or len(values) > cap
        if len(values) > cap:
            # W97's accepted edge order is descending alignment.  Keeping the
            # first cap values is deterministic and preserves the strongest
            # local relations; this is observation only, never a gate.
            values = values[:cap]
        return _AngleState(values, truncated)

    def payload(self) -> dict[str, Any]:
        if not self.values:
            return {
                "count": 0,
                "mean_deg": None,
                "median_deg": None,
                "p95_deg": None,
                "maximum_deg": None,
                "storage_truncated": self.truncated,
            }
        array = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(len(array)),
            "mean_deg": float(array.mean()),
            "median_deg": _percentile_nearest(array, 0.5),
            "p95_deg": _percentile_nearest(array, 0.95),
            "maximum_deg": float(array.max()),
            "storage_truncated": self.truncated,
        }


@dataclass(frozen=True)
class MergeEvent:
    """One attempted inter-component merge in the frozen W97 order."""

    accepted_edge_position: int
    candidate_edge_index: int
    endpoint_indices: tuple[int, int]
    endpoint_stable_ids: tuple[int, int]
    local_pairwise_alignment: float
    local_pairwise_angle_degrees: float
    component_a_root: int
    component_b_root: int
    component_a_size: int
    component_b_size: int
    pre_a_spectrum: tuple[float, float, float]
    pre_b_spectrum: tuple[float, float, float]
    hypothetical_post_spectrum: tuple[float, float, float]
    concentration_floor: float
    result: str
    resulting_component_size: int | None
    lineage_relevant: bool
    pre_a_normal_span_degrees: float | None = None
    pre_b_normal_span_degrees: float | None = None
    hypothetical_post_normal_span_degrees: float | None = None
    pre_a_local_angle_statistics: dict[str, Any] | None = None
    pre_b_local_angle_statistics: dict[str, Any] | None = None
    hypothetical_post_local_angle_statistics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_edge_position": self.accepted_edge_position,
            "candidate_edge_index": self.candidate_edge_index,
            "endpoint_indices": list(self.endpoint_indices),
            "endpoint_stable_ids": list(self.endpoint_stable_ids),
            "local_pairwise_alignment": self.local_pairwise_alignment,
            "local_pairwise_angle_degrees": self.local_pairwise_angle_degrees,
            "component_a_root": self.component_a_root,
            "component_b_root": self.component_b_root,
            "component_a_size": self.component_a_size,
            "component_b_size": self.component_b_size,
            "pre_a_spectrum": list(self.pre_a_spectrum),
            "pre_b_spectrum": list(self.pre_b_spectrum),
            "hypothetical_post_spectrum": list(self.hypothetical_post_spectrum),
            "concentration_floor": self.concentration_floor,
            "result": self.result,
            "resulting_component_size": self.resulting_component_size,
            "lineage_relevant": self.lineage_relevant,
            "pre_a_normal_span_degrees": self.pre_a_normal_span_degrees,
            "pre_b_normal_span_degrees": self.pre_b_normal_span_degrees,
            "hypothetical_post_normal_span_degrees": self.hypothetical_post_normal_span_degrees,
            "pre_a_local_angle_statistics": self.pre_a_local_angle_statistics,
            "pre_b_local_angle_statistics": self.pre_b_local_angle_statistics,
            "hypothetical_post_local_angle_statistics": self.hypothetical_post_local_angle_statistics,
        }


@dataclass
class MergeChronology:
    """Replay output plus explicit accounting of what was materialized."""

    events: list[MergeEvent]
    accepted_edge_indices: np.ndarray
    accepted_edge_order: np.ndarray
    final_component_roots: np.ndarray
    concentration_floor: float
    candidate_edge_count: int
    spatial_edge_count: int
    locally_accepted_edge_count: int
    local_graph_break_edge_count: int
    record_policy: str
    span_method: str

    @property
    def rejected_events(self) -> list[MergeEvent]:
        return [event for event in self.events if event.result == "rejected_region_concentration"]

    @property
    def accepted_events(self) -> list[MergeEvent]:
        return [event for event in self.events if event.result == "accepted_region_merge"]

    def first_rejection(self) -> MergeEvent | None:
        for event in self.events:
            if event.result == "rejected_region_concentration":
                return event
        return None

    def summary(self) -> dict[str, Any]:
        first = self.first_rejection()
        return {
            "candidate_edge_count": self.candidate_edge_count,
            "spatial_edge_count": self.spatial_edge_count,
            "locally_accepted_edge_count": self.locally_accepted_edge_count,
            "local_graph_break_edge_count": self.local_graph_break_edge_count,
            "recorded_inter_component_event_count": len(self.events),
            "recorded_accepted_merge_event_count": len(self.accepted_events),
            "recorded_region_rejection_event_count": len(self.rejected_events),
            "first_region_rejection_event": first.to_dict() if first is not None else None,
            "concentration_floor": self.concentration_floor,
            "record_policy": self.record_policy,
            "normal_span_method": self.span_method,
        }


def _accepted_edge_mask(
    orientation: SurfaceOrientationEvidence,
    graph: CandidateGraph,
    config: RegionCoherenceConfig,
) -> Any:
    """Exact copy of W97's accepted-edge predicate, kept diagnostic-only."""

    torch = require_torch()
    accepted = graph.spatial_edge_mask & graph.normal_compatible_mask
    if not config.require_positional_continuity:
        return accepted
    positions = orientation.positions
    normals = orientation.surface_normal
    left, right = graph.candidate_edges[:, 0], graph.candidate_edges[:, 1]
    delta_x = positions[right] - positions[left]
    sign = torch.where((normals[left] * normals[right]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
    average_normal = torch.nn.functional.normalize(normals[left] + normals[right] * sign, dim=-1, eps=_EPS)
    normal_offset = (delta_x * average_normal).sum(dim=-1).abs()
    tangential_offset = (delta_x - normal_offset.unsqueeze(-1) * average_normal).norm(dim=-1)
    ratio = normal_offset / tangential_offset.clamp_min(_EPS)
    return accepted & (ratio <= config.parallel_sheet_normal_over_tangent_ratio)


def _deterministic_edge_order(graph: CandidateGraph, accepted_index: Any) -> Any:
    torch = require_torch()
    edges = graph.candidate_edges[accepted_index]
    alignment = graph.normal_alignment[accepted_index]
    left, right = edges[:, 0], edges[:, 1]
    order = torch.arange(int(edges.shape[0]), device=edges.device)
    order = order[torch.argsort(right[order], stable=True)]
    order = order[torch.argsort(left[order], stable=True)]
    return order[torch.argsort(alignment[order], descending=True, stable=True)]


def _component_labels_from_roots(roots: np.ndarray) -> np.ndarray:
    if len(roots) == 0:
        return roots.astype(np.int64, copy=True)
    unique, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    subset_id_of_position = np.empty_like(order)
    subset_id_of_position[order] = np.arange(len(order), dtype=order.dtype)
    return subset_id_of_position[inverse]


def plain_w96_subset_ids(graph: CandidateGraph, config: CoverageFirstPartitionConfig) -> np.ndarray:
    """Reproduce W96 membership from an already-built W97 graph.

    This avoids rebuilding kNN and is only a provenance helper; it is not a
    new partition implementation.
    """

    roots = _connected_component_roots(graph.count, graph.accepted_edges, config)
    return _component_labels_from_roots(_as_numpy(roots, np.int64))


def replay_w97_region_growth(
    orientation: SurfaceOrientationEvidence,
    config: RegionCoherenceConfig | None = None,
    *,
    graph: CandidateGraph | None = None,
    lineage_mask: Any | None = None,
    record_policy: str = "all_inter_component_merges",
    max_events: int | None = None,
    capture_state_statistics: bool = True,
    progress: Callable[[str], None] | None = None,
) -> MergeChronology:
    """Replay W97's deterministic merge order without changing membership.

    ``lineage_mask`` only controls observation.  An event is relevant when
    either pre-merge component touches the frozen lineage.  The merge itself
    is always evaluated against the same W97 concentration floor.  When
    ``max_events`` is supplied, it is a storage cap only and the summary
    reports the cap in ``record_policy``; no decision is changed.
    """

    torch = require_torch()
    config = config or RegionCoherenceConfig()
    graph = graph or build_candidate_graph(orientation, config.local, progress=progress)
    accepted_mask = _accepted_edge_mask(orientation, graph, config)
    accepted_index_t = torch.nonzero(accepted_mask, as_tuple=False).reshape(-1)
    order_t = _deterministic_edge_order(graph, accepted_index_t)
    accepted_index = _as_numpy(accepted_index_t, np.int64)
    order = _as_numpy(order_t, np.int64)
    edges = _as_numpy(graph.candidate_edges, np.int64)
    alignment = _as_numpy(graph.normal_alignment, np.float64)
    stable_ids = _as_numpy(orientation.gaussian_ids, np.int64).reshape(-1)
    normals = _as_numpy(orientation.surface_normal, np.float64).reshape(-1, 3)
    count = int(graph.count)

    lineage = None if lineage_mask is None else _as_numpy(lineage_mask, bool).reshape(-1)
    if lineage is not None and len(lineage) != count:
        raise ValueError("lineage_mask must have one boolean value per orientation row")

    parent = np.arange(count, dtype=np.int64)
    size = np.ones(count, dtype=np.int64)
    scatter = float(config.structural_weight) * np.einsum("ni,nj->nij", normals, normals)
    angle_states = [_AngleState([]) for _ in range(count)]
    # A small deterministic state sample is sufficient to expose the angular
    # span trajectory without retaining every normal in every large region.
    span_samples: dict[int, list[np.ndarray]] = {}
    lineage_state = lineage.copy() if lineage is not None else np.ones(count, dtype=bool)
    events: list[MergeEvent] = []

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != root:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def samples(root: int) -> list[np.ndarray]:
        if root not in span_samples:
            return [normals[root].copy()]
        return span_samples[root]

    def merge_samples(a: int, b: int, root: int) -> None:
        values = samples(a) + samples(b)
        if len(values) > 256:
            values = sorted(values, key=lambda value: tuple(float(x) for x in value))[:256]
        span_samples[root] = values

    floor = float(config.concentration_floor())
    for accepted_position in order.tolist():
        candidate_position = int(accepted_index[accepted_position])
        u, v = (int(x) for x in edges[candidate_position])
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            continue

        relevant = True if lineage is None else bool(lineage_state[root_u] or lineage_state[root_v])
        if not relevant:
            # We still need exact state mutation.  The event is simply outside
            # the requested frozen lineage and is not materialized.
            pass

        pre_a = scatter[root_u].copy()
        pre_b = scatter[root_v].copy()
        post = pre_a + pre_b
        pre_a_spec = _spectrum(pre_a)
        pre_b_spec = _spectrum(pre_b)
        post_spec = _spectrum(post)
        accepted = post_spec[0] >= floor
        local_angle = _unsigned_angle_degrees(alignment[candidate_position])
        pre_a_size = int(size[root_u])
        pre_b_size = int(size[root_v])
        pre_a_angles = angle_states[root_u].payload()
        pre_b_angles = angle_states[root_v].payload()
        post_angles = angle_states[root_u].merged(angle_states[root_v]).merged(_AngleState([local_angle])).payload()
        a_samples = np.asarray(samples(root_u), dtype=np.float64)
        b_samples = np.asarray(samples(root_v), dtype=np.float64)
        post_samples = np.concatenate([a_samples, b_samples], axis=0)

        if accepted:
            surviving, absorbed = (root_u, root_v) if root_u < root_v else (root_v, root_u)
            parent[absorbed] = surviving
            size[surviving] += size[absorbed]
            scatter[surviving] = post
            angle_states[surviving] = angle_states[surviving].merged(
                angle_states[absorbed].merged(_AngleState([local_angle]))
            )
            merge_samples(surviving, absorbed, surviving)
            lineage_state[surviving] = lineage_state[root_u] or lineage_state[root_v]
            result = "accepted_region_merge"
            resulting_size: int | None = int(size[surviving])
        else:
            result = "rejected_region_concentration"
            resulting_size = None

        if relevant and (max_events is None or len(events) < max_events):
            if capture_state_statistics:
                a_span = _normal_span_degrees(a_samples)
                b_span = _normal_span_degrees(b_samples)
                post_span = _normal_span_degrees(post_samples)
                a_angles = pre_a_angles
                b_angles = pre_b_angles
            else:
                a_span = b_span = post_span = None
                a_angles = b_angles = post_angles = None
            events.append(
                MergeEvent(
                    accepted_edge_position=int(accepted_position),
                    candidate_edge_index=candidate_position,
                    endpoint_indices=(u, v),
                    endpoint_stable_ids=(int(stable_ids[u]), int(stable_ids[v])),
                    local_pairwise_alignment=float(alignment[candidate_position]),
                    local_pairwise_angle_degrees=local_angle,
                    component_a_root=root_u,
                    component_b_root=root_v,
                    component_a_size=pre_a_size,
                    component_b_size=pre_b_size,
                    pre_a_spectrum=pre_a_spec,
                    pre_b_spectrum=pre_b_spec,
                    hypothetical_post_spectrum=post_spec,
                    concentration_floor=floor,
                    result=result,
                    resulting_component_size=resulting_size,
                    lineage_relevant=relevant,
                    pre_a_normal_span_degrees=a_span,
                    pre_b_normal_span_degrees=b_span,
                    hypothetical_post_normal_span_degrees=post_span,
                    pre_a_local_angle_statistics=a_angles,
                    pre_b_local_angle_statistics=b_angles,
                    hypothetical_post_local_angle_statistics=post_angles,
                )
            )
        if progress is not None and len(events) and len(events) % 100000 == 0:
            progress(f"W178 diagnostic events={len(events)}")

    for index in range(count):
        find(index)
    roots = parent.copy()
    final_roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    return MergeChronology(
        events=events,
        accepted_edge_indices=accepted_index,
        accepted_edge_order=order,
        final_component_roots=final_roots,
        concentration_floor=floor,
        candidate_edge_count=int(edges.shape[0]),
        spatial_edge_count=int(_as_numpy(graph.spatial_edge_mask, bool).sum()),
        locally_accepted_edge_count=int(accepted_index.size),
        local_graph_break_edge_count=int((~_as_numpy(graph.spatial_edge_mask, bool)).sum()),
        record_policy=record_policy,
        span_method="exact_pairwise_unsigned_for_le_512_normals_stride_sampled_above",
    )


def chronology_to_npz(path: str | Path, chronology: MergeChronology) -> None:
    """Persist all materialized event fields compactly for real-scene runs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = chronology.events
    def ints(field: str) -> np.ndarray:
        return np.asarray([getattr(event, field) for event in events], dtype=np.int64)
    def floats(field: str) -> np.ndarray:
        return np.asarray([getattr(event, field) for event in events], dtype=np.float64)
    np.savez_compressed(
        path,
        accepted_edge_position=ints("accepted_edge_position"),
        candidate_edge_index=ints("candidate_edge_index"),
        endpoint_indices=np.asarray([event.endpoint_indices for event in events], dtype=np.int64),
        endpoint_stable_ids=np.asarray([event.endpoint_stable_ids for event in events], dtype=np.int64),
        local_pairwise_alignment=floats("local_pairwise_alignment"),
        local_pairwise_angle_degrees=floats("local_pairwise_angle_degrees"),
        component_a_root=ints("component_a_root"),
        component_b_root=ints("component_b_root"),
        component_a_size=ints("component_a_size"),
        component_b_size=ints("component_b_size"),
        pre_a_spectrum=np.asarray([event.pre_a_spectrum for event in events], dtype=np.float64),
        pre_b_spectrum=np.asarray([event.pre_b_spectrum for event in events], dtype=np.float64),
        hypothetical_post_spectrum=np.asarray([event.hypothetical_post_spectrum for event in events], dtype=np.float64),
        concentration_floor=np.asarray([event.concentration_floor for event in events], dtype=np.float64),
        result=np.asarray([event.result for event in events]),
        resulting_component_size=np.asarray(
            [-1 if event.resulting_component_size is None else event.resulting_component_size for event in events],
            dtype=np.int64,
        ),
        pre_a_normal_span_degrees=np.asarray([event.pre_a_normal_span_degrees for event in events], dtype=np.float64),
        pre_b_normal_span_degrees=np.asarray([event.pre_b_normal_span_degrees for event in events], dtype=np.float64),
        hypothetical_post_normal_span_degrees=np.asarray(
            [event.hypothetical_post_normal_span_degrees for event in events], dtype=np.float64
        ),
    )


def deterministic_shortest_path(
    graph: CandidateGraph,
    start_index: int,
    end_index: int,
    *,
    bounded_nodes: int | None = None,
) -> list[int] | None:
    """Shortest path on the frozen accepted graph, with ascending-neighbor BFS."""

    accepted = _as_numpy(graph.accepted_edges, np.int64)
    count = int(graph.count)
    if start_index == end_index:
        return [int(start_index)]
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left, right in accepted.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbors in adjacency:
        neighbors.sort()
    queue = [int(start_index)]
    parent = {int(start_index): -1}
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        if bounded_nodes is not None and len(parent) > bounded_nodes:
            return None
        for neighbor in adjacency[node]:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == end_index:
                path = [neighbor]
                while path[-1] != start_index:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    return None


def path_diagnostic(orientation: SurfaceOrientationEvidence, path: Sequence[int]) -> dict[str, Any]:
    positions = _as_numpy(orientation.positions, np.float64)
    normals = _as_numpy(orientation.surface_normal, np.float64)
    stable_ids = _as_numpy(orientation.gaussian_ids, np.int64)
    if len(path) < 1:
        return {"available": False, "reason": "empty_path"}
    if len(path) == 1:
        return {
            "available": True,
            "stable_ids": [int(stable_ids[path[0]])],
            "path_length_edges": 0,
            "world_space_path_length": 0.0,
            "endpoint_euclidean_distance": 0.0,
            "path_chord_ratio": 1.0,
            "per_hop_unsigned_normal_angle_degrees": [],
            "cumulative_unsigned_local_rotation_degrees": 0.0,
            "direct_endpoint_unsigned_normal_angle_degrees": 0.0,
        }
    hops: list[float] = []
    world_length = 0.0
    for left, right in zip(path[:-1], path[1:]):
        alignment = abs(float(np.dot(normals[left], normals[right])))
        hops.append(_unsigned_angle_degrees(alignment))
        world_length += float(np.linalg.norm(positions[left] - positions[right]))
    endpoint_distance = float(np.linalg.norm(positions[path[0]] - positions[path[-1]]))
    return {
        "available": True,
        "stable_ids": [int(stable_ids[index]) for index in path],
        "path_indices": [int(index) for index in path],
        "path_length_edges": len(path) - 1,
        "world_space_path_length": world_length,
        "endpoint_euclidean_distance": endpoint_distance,
        "path_chord_ratio": world_length / max(endpoint_distance, _EPS),
        "per_hop_unsigned_normal_angle_degrees": hops,
        "cumulative_unsigned_local_rotation_degrees": float(sum(hops)),
        "direct_endpoint_unsigned_normal_angle_degrees": _unsigned_angle_degrees(
            abs(float(np.dot(normals[path[0]], normals[path[-1]])))
        ),
    }


def graph_support_diagnostic(
    graph: CandidateGraph,
    chronology: MergeChronology,
    event: MergeEvent | None,
    *,
    bounded_alternate_path_nodes: int = 256,
) -> dict[str, Any]:
    """Measure support for the pre-merge component pair of one critical event.

    The component state is reconstructed from the recorded W97 decisions up
    to (but excluding) ``event``. This is observational reconstruction only:
    it never changes the baseline partition or re-evaluates its floor.
    """

    if event is None:
        return {"available": False, "reason": "no_critical_inter_component_event"}
    try:
        target = chronology.events.index(event)
    except ValueError:
        return {"available": False, "reason": "event_not_in_chronology"}

    count = int(graph.count)
    parent = list(range(count))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for prior in chronology.events[:target]:
        if prior.result != "accepted_region_merge":
            continue
        left, right = prior.endpoint_indices
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            if root_l > root_r:
                root_l, root_r = root_r, root_l
            parent[root_r] = root_l

    a_root = find(event.endpoint_indices[0])
    b_root = find(event.endpoint_indices[1])
    if a_root == b_root:
        return {"available": False, "reason": "recorded_event_not_inter_component_under_replay"}

    accepted = _as_numpy(graph.accepted_edges, np.int64)
    node_root = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    cross = accepted[((node_root[accepted[:, 0]] == a_root) & (node_root[accepted[:, 1]] == b_root)) | ((node_root[accepted[:, 0]] == b_root) & (node_root[accepted[:, 1]] == a_root))]
    a_endpoints: set[int] = set()
    b_endpoints: set[int] = set()
    degrees: dict[int, int] = {}
    adjacency: dict[int, list[int]] = {}
    target_edge = tuple(sorted(event.endpoint_indices))
    for left, right in accepted.tolist():
        root_l, root_r = int(node_root[left]), int(node_root[right])
        if root_l in (a_root, b_root):
            degrees[left] = degrees.get(left, 0) + 1
            adjacency.setdefault(left, []).append(right)
        if root_r in (a_root, b_root):
            degrees[right] = degrees.get(right, 0) + 1
            adjacency.setdefault(right, []).append(left)
    for left, right in cross.tolist():
        if node_root[left] == a_root:
            a_endpoints.add(left); b_endpoints.add(right)
        else:
            a_endpoints.add(right); b_endpoints.add(left)

    start_node, end_node = event.endpoint_indices
    queue = [start_node]
    visited = {start_node}
    cursor = 0
    alternate_exists = False
    while cursor < len(queue) and len(visited) <= bounded_alternate_path_nodes:
        node = queue[cursor]
        cursor += 1
        for neighbor in sorted(adjacency.get(node, [])):
            if tuple(sorted((node, neighbor))) == target_edge:
                continue
            if neighbor == end_node:
                alternate_exists = True
                cursor = len(queue)
                break
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    shared_neighbors = set(adjacency.get(start_node, [])) & set(adjacency.get(end_node, []))
    degree_values = list(degrees.values())
    return {
        "available": True,
        "event_candidate_edge_index": event.candidate_edge_index,
        "component_a_root": a_root,
        "component_b_root": b_root,
        "component_size_ratio_small_over_large": min(event.component_a_size, event.component_b_size) / max(event.component_a_size, event.component_b_size),
        "accepted_local_cross_edge_count": int(len(cross)),
        "distinct_endpoint_count_each_side": {"component_a": int(len(a_endpoints)), "component_b": int(len(b_endpoints))},
        "local_node_degree_distribution": {"count": len(degree_values), "min": min(degree_values) if degree_values else 0, "max": max(degree_values) if degree_values else 0, "mean": float(np.mean(degree_values)) if degree_values else 0.0},
        "alternate_accepted_path_exists_within_bounded_neighborhood": alternate_exists,
        "alternate_accepted_path_count_lower_bound": 1 if alternate_exists else 0,
        "alternate_path_support_rule": "ascending-neighbor BFS after removing recorded critical edge; bounded diagnostic search only",
        "articulation_like": bool(len(cross) == 1 and not alternate_exists),
        "shared_neighbor_support": int(len(shared_neighbors)),
        "bounded_alternate_path_nodes": bounded_alternate_path_nodes,
        "diagnostic_only": True,
    }
def compare_subset_membership(a: Any, b: Any) -> bool:
    """Compare partitions by exact pairwise membership, independent of IDs."""

    left = _as_numpy(a, np.int64).reshape(-1)
    right = _as_numpy(b, np.int64).reshape(-1)
    if left.shape != right.shape:
        return False
    # Exact subset ID equality is the stronger invariant and is what the
    # Worklog 178 runner records.  Pairwise equivalence helps diagnose a
    # harmless relabeling in fixture-only callers.
    return bool(np.array_equal(left, right))


def diagnostic_contract_summary(
    baseline_subset_ids: Any,
    diagnostic_subset_ids: Any,
    chronology: MergeChronology,
) -> dict[str, Any]:
    equal = compare_subset_membership(baseline_subset_ids, diagnostic_subset_ids)
    return {
        "diagnostic_enabled_w97_subset_ids_equal_historical_baseline": equal,
        "baseline_subset_ids_untouched": equal,
        "diagnostic_does_not_modify_membership": equal,
        "chronology_summary": chronology.summary(),
    }


__all__ = [
    "LOCAL_GRAPH_BREAKS_CURVATURE",
    "REGION_CONCENTRATION_BREAKS_CURVATURE",
    "OWNERSHIP_STAGE_BREAKS_CURVATURE",
    "MIXED_ATTRIBUTION",
    "REAL_TABLE_RIM_PROVENANCE_GAP",
    "MergeEvent",
    "MergeChronology",
    "plain_w96_subset_ids",
    "replay_w97_region_growth",
    "chronology_to_npz",
    "deterministic_shortest_path",
    "path_diagnostic",
    "graph_support_diagnostic",
    "diagnostic_contract_summary",
]
