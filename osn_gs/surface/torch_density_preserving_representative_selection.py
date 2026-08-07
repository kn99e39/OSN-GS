from __future__ import annotations

"""Density-preserving, mode-aware canonical representative selection (worklog 129).

Problem this replaces: the prior voxel-nearest-to-cell-center sampler
(``TorchOSNGSPipeline._canonical_construction_indices``) keeps exactly ONE
Gaussian per occupied voxel cell -- whichever happens to sit closest to the
cell's geometric center. Two consequences on a real ADC-trained scene:

1. When a cell contains multiple structurally distinct surface modes (close
   parallel sheets, a box corner's several faces, a cylinder cap meeting its
   side, a thin structure's front/back), only one mode survives into the
   representative set; the others are silently discarded from topology
   entirely (never even reach reliability/affinity/region formation).
2. The single surviving representative's local density/support is not
   recorded anywhere -- a cell with 400 real Gaussians and a cell with 4 both
   contribute exactly one representative, and nothing downstream can tell the
   difference (this is what ``torch_full_neighborhood_evidence.py`` fixes on
   the RELIABILITY side; this module fixes it on the SELECTION side).

This module still bounds the representative count to
``canonical_construction_max_points`` (the O(N^2) topology stages this feeds
can never scale past that), but selects representatives from a MODE-AWARE
candidate set (voxel cell x locally-consistent normal/offset cluster) via
deterministic, support/opacity-weighted farthest-point sampling, instead of
one nearest-to-cell-center pick per cell.

Voxel cells here are used ONLY for candidate grouping/acceleration -- never as
a surface-region or topology unit (a cell ID is not a region ID; see worklog
129 non-goals).
"""

import math

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame
from osn_gs.utils.torch_ops import require_torch

SCHEMA_VERSION = "density_preserving_representative_selection_worklog129_v1"


@dataclass(frozen=True)
class RepresentativeSelectionConfig:
    """Configurable policy, not a confirmed canonical threshold set."""

    max_modes_per_cell: int = 4
    mode_normal_alignment_min: float = 0.6
    mode_offset_max_thickness_ratio: float = 3.0

    # --- worklog 49: boundary-evidence swap-in ---
    # `_split_cell_into_modes` already separates one voxel cell's members into
    # locally-consistent normal/offset modes; a cell with >1 mode is, by that
    # split's own compatibility gate, evidence of a genuinely different local
    # surface orientation sharing the cell. Global weighted-farthest-point
    # selection under a fixed budget can still drop one of those modes purely
    # on distance/weight competition with unrelated bulk-interior candidates
    # -- that is real full-cloud evidence lost to budget pressure, not a
    # measurement gap. Measured on real ADC-trained 3k/10k checkpoints: most
    # same-cell sibling drops have near-1.0 normal alignment (just noisy
    # near-duplicate mode splits, not a real orientation difference) --
    # `boundary_evidence_alignment_max` is set well BELOW the 0.6 mode-split
    # gate itself so only genuinely divergent (near-orthogonal or sharper)
    # orientations qualify, not borderline splits. `boundary_evidence_min_source_count`
    # excludes single/few-Gaussian noise fragments from being swapped in.
    boundary_evidence_alignment_max: float = 0.3
    boundary_evidence_min_source_count: int = 3

    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CanonicalSurfaceRepresentative:
    """One selected representative's selection-time provenance (worklog 129 item 7).

    Reliability/contextual-evidence fields are intentionally NOT here --
    those are computed downstream from
    ``torch_full_neighborhood_evidence.FullNeighborhoodEvidence`` (a batched,
    tensor-shaped sibling structure keyed by the same representative order)
    to avoid duplicating tensor-vs-python-object storage for the same data.
    """

    representative_stable_id: Any
    representative_gaussian_index: int  # index into the full observed cloud
    source_count: int
    source_opacity_mass: float
    world_centroid: tuple[float, float, float]
    cell_id: int
    mode_id: int
    policy_schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class SelectionDiagnostics:
    """Stage-by-stage counts (worklog 129 item 2) -- never collapse to one final number."""

    input_gaussian_count: int
    occupied_cell_count: int
    total_candidate_mode_count: int
    modes_per_cell_mean: float
    modes_per_cell_max: int
    multi_mode_cell_count: int
    selected_representative_count: int
    representative_source_count_mean: float
    representative_source_count_min: int
    representative_source_count_max: int
    selection_mode: str  # "full_coverage" (candidates <= budget) or "weighted_farthest_point"
    boundary_evidence_swap_in_count: int = 0  # worklog 49


@dataclass(frozen=True)
class RepresentativeSelectionResult:
    representative_indices: Any  # (M,) long indices into the full cloud, sorted
    cell_ids: Any  # (M,) long
    mode_ids: Any  # (M,) long
    representatives: tuple[CanonicalSurfaceRepresentative, ...]
    diagnostics: SelectionDiagnostics


def _voxel_cells(points: Any, budget: int) -> tuple[Any, int]:
    torch = require_torch()
    resolution = max(2, int(math.ceil(budget ** 0.5)))
    minimum = points.amin(dim=0)
    span = (points.amax(dim=0) - minimum).clamp_min(1e-9)
    normalized = (points - minimum) / span
    cells = torch.floor(normalized * resolution).long().clamp(0, resolution - 1)
    keys = cells[:, 0] * resolution * resolution + cells[:, 1] * resolution + cells[:, 2]
    return keys, resolution


def _split_cell_into_modes(
    member_local_indices: list[int],
    normals_np: "np.ndarray",
    thickness_np: "np.ndarray",
    positions_np: "np.ndarray",
    stable_id_keys: list[str],
    *,
    config: RepresentativeSelectionConfig,
) -> list[list[int]]:
    """Greedy, deterministic normal/offset clustering within one voxel cell.

    Pure-numpy/python (no per-element torch calls): this runs once per
    Gaussian across the whole scene, so per-call tensor overhead would
    dominate runtime on real ADC-trained scenes (~1e5 Gaussians).

    Deterministic: members are processed in ascending stable-ID order so the
    result never depends on the input array's original ordering. Bounded:
    caps at ``max_modes_per_cell`` modes; any member that would start a
    (max_modes_per_cell + 1)-th mode is instead assigned to whichever
    existing mode it aligns with best -- an explicit, documented bound, not a
    per-scene tuning knob.
    """
    ordered = sorted(member_local_indices, key=lambda idx: stable_id_keys[idx])
    modes: list[dict[str, Any]] = []
    for idx in ordered:
        normal = normals_np[idx]
        thickness = float(thickness_np[idx])
        position = positions_np[idx]
        best_mode = None
        best_alignment = -1.0
        for mode in modes:
            alignment = float(abs(np.dot(normal, mode["normal"])))
            offset = float(abs(np.dot(position - mode["centroid"], mode["normal"])))
            compatible = (
                alignment >= config.mode_normal_alignment_min
                and offset <= config.mode_offset_max_thickness_ratio * max(thickness, mode["thickness"])
            )
            if compatible and alignment > best_alignment:
                best_alignment = alignment
                best_mode = mode
        if best_mode is None and len(modes) < config.max_modes_per_cell:
            modes.append({
                "normal": normal,
                "centroid": position,
                "thickness": max(thickness, 1e-9),
                "members": [idx],
            })
            continue
        if best_mode is None:
            # Bounded fallback: forced-merge into the best-aligned existing
            # mode regardless of the compatibility gate above.
            best_mode = max(modes, key=lambda mode: float(abs(np.dot(normal, mode["normal"]))))
        n = len(best_mode["members"])
        best_mode["centroid"] = (best_mode["centroid"] * n + position) / (n + 1)
        best_mode["members"].append(idx)
    return [mode["members"] for mode in modes]


def _boundary_evidence_swap_in(
    candidates: list[dict[str, Any]],
    selected_local: set[int],
    positions_tensor: Any,
    normals_np: "np.ndarray",
    stable_ids: Sequence[Any],
    *,
    config: RepresentativeSelectionConfig,
) -> tuple[set[int], int]:
    """Worklog 49: deterministically restore same-cell sibling modes that the
    weighted-FPS budget competition dropped despite genuinely diverging in
    orientation from the sibling mode that WAS selected -- real full-cloud
    evidence lost purely to budget pressure, not a measurement gap (see the
    ``RepresentativeSelectionConfig`` docstring above). Never changes the
    total selected count: each swap-in evicts exactly one currently-selected
    representative, chosen from the SIBLING mode's own local neighbourhood
    (same orientation as whatever already won this cell) as the most
    redundant one there -- never from the swap-in's own neighbourhood or an
    unrelated area, and never a representative this repair itself depends on
    as a sibling.
    """
    torch = require_torch()
    by_cell: dict[int, list[int]] = {}
    for local_index, candidate in enumerate(candidates):
        by_cell.setdefault(candidate["cell_id"], []).append(local_index)

    swap_in_candidates: list[dict[str, int]] = []
    protected: set[int] = set()
    for indices in by_cell.values():
        if len(indices) < 2:
            continue
        selected_here = [i for i in indices if i in selected_local]
        dropped_here = [i for i in indices if i not in selected_local]
        if not selected_here or not dropped_here:
            continue
        for dropped_index in dropped_here:
            dropped_normal = normals_np[candidates[dropped_index]["representative_index"]]
            sibling_index = max(
                selected_here,
                key=lambda i: float(abs(np.dot(dropped_normal, normals_np[candidates[i]["representative_index"]]))),
            )
            best_alignment = float(abs(np.dot(
                dropped_normal, normals_np[candidates[sibling_index]["representative_index"]],
            )))
            if (
                best_alignment > config.boundary_evidence_alignment_max
                or candidates[dropped_index]["source_count"] < config.boundary_evidence_min_source_count
            ):
                continue
            swap_in_candidates.append({"dropped": dropped_index, "sibling": sibling_index})
            protected.update(selected_here)

    if not swap_in_candidates:
        return selected_local, 0

    # Deterministic order: ascending stable ID of the candidate's own
    # representative Gaussian -- never depends on dict/set iteration order.
    swap_in_candidates.sort(key=lambda item: str(stable_ids[candidates[item["dropped"]]["representative_index"]]))
    seen_dropped: set[int] = set()
    deduplicated = []
    for item in swap_in_candidates:
        if item["dropped"] in seen_dropped:
            continue
        seen_dropped.add(item["dropped"])
        deduplicated.append(item)
    swap_in_candidates = deduplicated

    # A real edge spans many voxel cells, each independently proposing its
    # own swap-in -- accepting all of them clusters evictions along one
    # short stretch of one face and disconnects it from the rest of that
    # face's own representatives (measured on the box fixture: region_count
    # 6 -> 8, one face split into two graph components). One representative
    # per roughly one representative-spacing's worth of edge is already
    # enough to mark the edge as boundary evidence; accepted swap-ins are
    # therefore greedily kept mutually farther apart than the ORIGINAL
    # selection's own median nearest-neighbor spacing -- a property of the
    # selection itself, not a new scene-tuned constant.
    original_positions = positions_tensor[sorted(selected_local)]
    if original_positions.shape[0] >= 2:
        original_pairwise = torch.cdist(original_positions, original_positions)
        original_pairwise.fill_diagonal_(float("inf"))
        spacing_values = original_pairwise.min(dim=1).values
        median_spacing = float(spacing_values.median())
    else:
        median_spacing = 0.0
    accepted_positions: list[Any] = []
    spaced_out_candidates = []
    for item in swap_in_candidates:
        position = positions_tensor[item["dropped"]]
        if accepted_positions:
            nearest = min(float(torch.linalg.vector_norm(position - other)) for other in accepted_positions)
            if nearest < median_spacing * 3.0:
                continue
        accepted_positions.append(position)
        spaced_out_candidates.append(item)
    swap_in_candidates = spaced_out_candidates

    # Eviction targets the SIBLING's own orientation, not the swap-in's: the
    # sibling is whichever mode already won this cell's budget competition,
    # and the box fixture proved that evicting near the swap-in's OWN
    # position (which sits ON the edge, equally close to both orientations)
    # can remove the very representatives connecting the swap-in to its
    # correct face, isolating it and its edge-siblings into a spurious
    # micro-region. The candidate is drawn from the SIBLING's own orientation
    # pool (representatives currently selected that still align with the
    # SIBLING's normal), never the swap-in's own neighbourhood or an
    # unrelated face.
    #
    # Redundancy (a close nearest neighbor) alone is not a safe eviction
    # criterion either: a representative can be close to one neighbor yet
    # still be the sole connector between two halves of its face's coverage
    # (measured on the box fixture -- naive redundancy-only eviction
    # disconnected a face into two graph components, region_count 6 -> 7/8/9
    # depending on density). Each candidate is therefore only evicted if it
    # is NOT an articulation point of the pool's own proximity graph (an
    # explicit connectivity check, not a distance proxy) -- removing it must
    # never increase the pool's own component count.
    updated = set(selected_local)
    swap_count = 0
    for item in swap_in_candidates:
        swap_in_index, sibling_index = item["dropped"], item["sibling"]
        if swap_in_index in updated:
            continue
        sibling_normal = normals_np[candidates[sibling_index]["representative_index"]]
        pool = sorted(
            i for i in updated
            if i not in protected
            and float(abs(np.dot(
                sibling_normal, normals_np[candidates[i]["representative_index"]],
            ))) >= config.mode_normal_alignment_min
        )
        if len(pool) < 15:
            # Below this the sibling orientation's own local coverage is too
            # thin for a proxy-graph safety check to be trustworthy -- the
            # box fixture proved a generous-radius proxy can still miss real
            # fragmentation the stricter downstream affinity/region-formation
            # graph produces. A well-populated pool is required before this
            # repair touches it at all; a thin one is left untouched rather
            # than risking it.
            continue
        pool_positions = positions_tensor[pool]
        pool_pairwise = torch.cdist(pool_positions, pool_positions)
        pool_pairwise.fill_diagonal_(float("inf"))
        pool_nearest_distance = pool_pairwise.min(dim=1).values
        # The pool's OWN median nearest-neighbor spacing (same-orientation
        # subset, not the mixed-orientation `median_spacing` above) sets the
        # proximity-graph edge radius -- generous enough (2.5x) to match the
        # connectivity a bounded-kNN affinity graph would itself find, so an
        # articulation point here reliably predicts a real disconnection.
        pool_median_spacing = float(pool_nearest_distance.median())
        edge_radius = pool_median_spacing * 4.0
        adjacency: dict[int, list[int]] = {local: [] for local in range(len(pool))}
        # Worklog 66: this used to compare `float(pool_pairwise[a, b])` one
        # pair at a time inside a Python double loop. Each `float(...)` on a
        # CUDA tensor forces a device-to-host sync, so the O(pool^2) loop
        # became O(pool^2) individual GPU synchronizations -- profiled as the
        # dominant cost of a real-checkpoint run (minutes of wall-clock per
        # condition while GPU utilization stayed near 10%, i.e. latency-bound
        # on sync stalls, not compute). A single vectorized comparison
        # produces the EXACT same boolean adjacency relation (same
        # `edge_radius`, same strict a<b pairing) in one pass; only the
        # (typically far sparser) resulting edge list is then transferred to
        # Python, once, so this changes nothing about which edges exist.
        if len(pool) > 1:
            within_radius = pool_pairwise <= edge_radius
            triu = torch.triu_indices(len(pool), len(pool), offset=1, device=pool_pairwise.device)
            edge_mask = within_radius[triu[0], triu[1]]
            for a, b in zip(triu[0][edge_mask].tolist(), triu[1][edge_mask].tolist()):
                adjacency[a].append(b)
                adjacency[b].append(a)

        def _component_count(exclude: int | None) -> int:
            remaining = [n for n in range(len(pool)) if n != exclude]
            remaining_set = set(remaining)
            visited: set[int] = set()
            components = 0
            for start in remaining:
                if start in visited:
                    continue
                components += 1
                stack = [start]
                visited.add(start)
                while stack:
                    node = stack.pop()
                    for neighbor in adjacency[node]:
                        if neighbor != exclude and neighbor in remaining_set and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
            return components

        # Compare against the pool's OWN existing component count, not
        # "reaches every other node" -- a same-orientation pool spanning a
        # real (non-toy) surface is not always a single connected component
        # to begin with (bounded-kNN sparsity, real noise), and requiring
        # full reachability made every candidate look unsafe, silencing the
        # repair everywhere including real checkpoints. Only an eviction that
        # would INCREASE the component count (a genuine articulation point)
        # is rejected.
        original_component_count = _component_count(exclude=None)
        # Same one-sync-per-element issue as the adjacency loop above: index
        # the CUDA tensor once into a plain Python list instead of inside the
        # sort key (called once per pool element by Timsort).
        pool_nearest_distance_list = pool_nearest_distance.tolist()
        redundancy_order = sorted(
            range(len(pool)),
            key=lambda local: (pool_nearest_distance_list[local], str(stable_ids[candidates[pool[local]]["representative_index"]])),
        )
        # Component-count non-increase alone still under-protects a node that
        # remains "connected" only through a single thin remaining path (the
        # proxy graph's generous radius can call that safe while the real,
        # stricter downstream affinity graph would not) -- measured on the
        # box fixture, a lingering fragmentation case survived the component
        # check alone. Requiring degree >= 4 in the proxy graph additionally
        # restricts eviction to nodes that are well embedded in a genuinely
        # dense cluster, not merely non-critical for bare reachability.
        min_safe_degree = 4
        evict_index = None
        for local in redundancy_order:
            if len(adjacency[local]) < min_safe_degree:
                continue
            if _component_count(exclude=local) <= original_component_count:
                evict_index = pool[local]
                break
        if evict_index is None:
            continue
        updated.discard(evict_index)
        updated.add(swap_in_index)
        swap_count += 1
    return updated, swap_count


def select_density_preserving_representatives(
    points: Any,
    frame: GaussianCovarianceFrame,
    opacity: Any,
    stable_ids: Sequence[Any],
    *,
    max_points: int,
    config: RepresentativeSelectionConfig | None = None,
) -> RepresentativeSelectionResult:
    """Select <= ``max_points`` representatives preserving structural diversity.

    Falls through to plain ``arange`` (every Gaussian is its own
    representative) when the input already fits the budget, matching the
    prior sampler's behavior for small scenes.
    """
    torch = require_torch()
    config = config or RepresentativeSelectionConfig()
    count = int(points.shape[0])
    budget = max(16, int(max_points))
    opacity = torch.as_tensor(opacity).reshape(-1)

    if count <= budget:
        indices = torch.arange(count, dtype=torch.long, device=points.device)
        representatives = tuple(
            CanonicalSurfaceRepresentative(
                representative_stable_id=stable_ids[i],
                representative_gaussian_index=i,
                source_count=1,
                source_opacity_mass=float(opacity[i]),
                world_centroid=tuple(float(v) for v in points[i].detach().cpu().tolist()),
                cell_id=-1,
                mode_id=0,
                policy_schema_version=config.schema_version,
            )
            for i in range(count)
        )
        diagnostics = SelectionDiagnostics(
            input_gaussian_count=count,
            occupied_cell_count=count,
            total_candidate_mode_count=count,
            modes_per_cell_mean=1.0,
            modes_per_cell_max=1,
            multi_mode_cell_count=0,
            selected_representative_count=count,
            representative_source_count_mean=1.0,
            representative_source_count_min=1,
            representative_source_count_max=1,
            selection_mode="full_coverage",
        )
        return RepresentativeSelectionResult(indices, torch.full((count,), -1, dtype=torch.long, device=points.device), torch.zeros((count,), dtype=torch.long, device=points.device), representatives, diagnostics)

    keys, _resolution = _voxel_cells(points, budget)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    boundaries = torch.ones_like(sorted_keys, dtype=torch.bool)
    boundaries[1:] = sorted_keys[1:] != sorted_keys[:-1]
    cell_starts = torch.nonzero(boundaries, as_tuple=False).reshape(-1).tolist()
    cell_starts.append(int(order.numel()))

    normals_np = frame.normal_candidate.detach().cpu().numpy()
    thickness_np = frame.normal_thickness.detach().cpu().numpy()
    positions_np = points.detach().cpu().numpy()
    stable_id_keys = [str(item) for item in stable_ids]

    candidates: list[dict[str, Any]] = []
    # Keep the existing per-mode Torch arithmetic intact, but defer extracting
    # nearest-member distances until every mode has been evaluated. This avoids
    # one host synchronization per source Gaussian while preserving the exact
    # CPU min/tie-break policy below.
    pending_candidates: list[tuple[int, int, list[int], float, torch.Tensor, torch.Tensor]] = []
    modes_per_cell: list[int] = []
    for cell_index in range(len(cell_starts) - 1):
        start, end = cell_starts[cell_index], cell_starts[cell_index + 1]
        member_local_indices = order[start:end].tolist()
        cell_id = int(sorted_keys[start])
        modes = _split_cell_into_modes(
            member_local_indices, normals_np, thickness_np, positions_np, stable_id_keys, config=config
        )
        modes_per_cell.append(len(modes))
        for mode_id, member_indices in enumerate(modes):
            member_tensor = torch.tensor(member_indices, dtype=torch.long, device=points.device)
            member_opacity = opacity[member_tensor]
            opacity_sum = float(member_opacity.sum())
            if opacity_sum > 1e-9:
                centroid = (points[member_tensor] * member_opacity.unsqueeze(-1)).sum(dim=0) / opacity_sum
            else:
                centroid = points[member_tensor].mean(dim=0)
            distances_to_centroid = torch.linalg.norm(points[member_tensor] - centroid, dim=-1)
            pending_candidates.append(
                (cell_id, mode_id, member_indices, opacity_sum, centroid, distances_to_centroid)
            )

    if pending_candidates:
        all_distances = torch.cat([entry[5] for entry in pending_candidates]).detach().cpu().tolist()
        distance_offset = 0
        for cell_id, mode_id, member_indices, opacity_sum, centroid, _distances_to_centroid in pending_candidates:
            member_count = len(member_indices)
            # Deterministic tie-break: nearest-to-centroid member, ties broken
            # by ascending stable ID. The values are exactly the same tensor
            # values that were formerly extracted one-by-one above.
            order_key = [
                (all_distances[distance_offset + i], str(stable_ids[member_indices[i]]))
                for i in range(member_count)
            ]
            best_local = min(range(member_count), key=lambda i: order_key[i])
            representative_index = member_indices[best_local]
            candidates.append({
                "cell_id": cell_id,
                "mode_id": mode_id,
                "representative_index": representative_index,
                "source_count": member_count,
                "source_opacity_mass": opacity_sum,
                "centroid": centroid,
            })
            distance_offset += member_count

    total_candidates = len(candidates)
    boundary_evidence_swap_in_count = 0
    if total_candidates <= budget:
        selected = candidates
        selection_mode = "full_coverage"
    else:
        selection_mode = "weighted_farthest_point"
        positions_tensor = torch.stack([points[c["representative_index"]] for c in candidates], dim=0)
        support = torch.tensor([c["source_count"] for c in candidates], dtype=torch.float32, device=points.device)
        opacity_mass = torch.tensor([c["source_opacity_mass"] for c in candidates], dtype=torch.float32, device=points.device)
        weight = 0.5 * (support / support.max().clamp_min(1e-9)) + 0.5 * (opacity_mass / opacity_mass.max().clamp_min(1e-9))
        stable_key = [str(stable_ids[c["representative_index"]]) for c in candidates]

        # The prior implementation issued one tiny ``cdist(1, C)`` call per
        # FPS iteration and brought every tie candidate back to Python.  For
        # bounded real-scene candidate sets, one CxC CUDA matrix is modest in
        # memory and removes thousands of small kernel launches.  An exact
        # vector-norm fallback protects unusually large candidate sets.
        pairwise_bytes = total_candidates * total_candidates * positions_tensor.element_size()
        pairwise_distance = torch.cdist(positions_tensor, positions_tensor) if pairwise_bytes <= 256 * 1024 * 1024 else None

        def distances_from(local_index: int) -> Any:
            if pairwise_distance is not None:
                return pairwise_distance[local_index]
            return torch.linalg.vector_norm(positions_tensor - positions_tensor[local_index], dim=1)

        # Deterministic seed: highest (support, opacity_mass), tie-broken by
        # ascending stable ID. Only selected indices cross to Python.
        stable_rank = torch.empty((total_candidates,), dtype=torch.long, device=points.device)
        for rank, index in enumerate(sorted(range(total_candidates), key=lambda i: stable_key[i])):
            stable_rank[index] = rank
        support_best = support.max()
        seed_pool = torch.nonzero(support == support_best, as_tuple=False).reshape(-1)
        opacity_best = opacity_mass[seed_pool].max()
        seed_candidates = seed_pool[opacity_mass[seed_pool] == opacity_best]
        seed = int(seed_candidates[stable_rank[seed_candidates].argmin()].item())
        selected_local = [seed]
        selected_mask = torch.zeros((total_candidates,), dtype=torch.bool, device=points.device)
        selected_mask[seed] = True
        min_distance = distances_from(seed).masked_fill(selected_mask, -1.0)
        for _ in range(budget - 1):
            score = min_distance.clamp_min(0.0) * weight
            score = score.masked_fill(selected_mask, -1.0)
            best_score = score.max()
            tied = torch.nonzero(score >= best_score - 1e-9, as_tuple=False).reshape(-1)
            next_pick = int(tied[stable_rank[tied].argmin()].item())
            selected_local.append(next_pick)
            selected_mask[next_pick] = True
            min_distance = torch.minimum(min_distance, distances_from(next_pick)).masked_fill(selected_mask, -1.0)
        selected_local_set, boundary_evidence_swap_in_count = _boundary_evidence_swap_in(
            candidates, set(selected_local), positions_tensor, normals_np, stable_ids, config=config,
        )
        selected = [candidates[i] for i in sorted(selected_local_set)]

    representative_indices = sorted(c["representative_index"] for c in selected)
    by_index = {c["representative_index"]: c for c in selected}
    representatives = tuple(
        CanonicalSurfaceRepresentative(
            representative_stable_id=stable_ids[idx],
            representative_gaussian_index=idx,
            source_count=by_index[idx]["source_count"],
            source_opacity_mass=by_index[idx]["source_opacity_mass"],
            world_centroid=tuple(float(v) for v in by_index[idx]["centroid"].detach().cpu().tolist()),
            cell_id=by_index[idx]["cell_id"],
            mode_id=by_index[idx]["mode_id"],
            policy_schema_version=config.schema_version,
        )
        for idx in representative_indices
    )
    source_counts = [r.source_count for r in representatives]
    diagnostics = SelectionDiagnostics(
        input_gaussian_count=count,
        occupied_cell_count=len(cell_starts) - 1,
        total_candidate_mode_count=total_candidates,
        modes_per_cell_mean=(sum(modes_per_cell) / max(1, len(modes_per_cell))),
        modes_per_cell_max=max(modes_per_cell, default=0),
        multi_mode_cell_count=sum(1 for m in modes_per_cell if m > 1),
        selected_representative_count=len(representatives),
        representative_source_count_mean=(sum(source_counts) / max(1, len(source_counts))),
        representative_source_count_min=min(source_counts, default=0),
        representative_source_count_max=max(source_counts, default=0),
        selection_mode=selection_mode,
        boundary_evidence_swap_in_count=boundary_evidence_swap_in_count,
    )
    index_tensor = torch.tensor(representative_indices, dtype=torch.long, device=points.device)
    cell_id_tensor = torch.tensor([by_index[idx]["cell_id"] for idx in representative_indices], dtype=torch.long, device=points.device)
    mode_id_tensor = torch.tensor([by_index[idx]["mode_id"] for idx in representative_indices], dtype=torch.long, device=points.device)
    return RepresentativeSelectionResult(index_tensor, cell_id_tensor, mode_id_tensor, representatives, diagnostics)
