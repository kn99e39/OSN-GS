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
            # Deterministic tie-break: nearest-to-centroid member, ties broken
            # by ascending stable ID.
            order_key = [
                (float(distances_to_centroid[i]), str(stable_ids[member_indices[i]]))
                for i in range(len(member_indices))
            ]
            best_local = min(range(len(member_indices)), key=lambda i: order_key[i])
            representative_index = member_indices[best_local]
            candidates.append({
                "cell_id": cell_id,
                "mode_id": mode_id,
                "representative_index": representative_index,
                "source_count": len(member_indices),
                "source_opacity_mass": opacity_sum,
                "centroid": centroid,
            })

    total_candidates = len(candidates)
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
        selected = [candidates[i] for i in sorted(selected_local)]

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
    )
    index_tensor = torch.tensor(representative_indices, dtype=torch.long, device=points.device)
    cell_id_tensor = torch.tensor([by_index[idx]["cell_id"] for idx in representative_indices], dtype=torch.long, device=points.device)
    mode_id_tensor = torch.tensor([by_index[idx]["mode_id"] for idx in representative_indices], dtype=torch.long, device=points.device)
    return RepresentativeSelectionResult(index_tensor, cell_id_tensor, mode_id_tensor, representatives, diagnostics)
