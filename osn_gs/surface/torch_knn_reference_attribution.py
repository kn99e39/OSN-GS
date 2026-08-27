"""Diagnostic-only attribution of the immutable WL119 ``torch.cdist`` KNN.

Nothing in this module is connected to a production/default path.  It records
the reference's actual raw ranking values, K-boundary margins, and candidate
distances without changing the authoritative implementation in
``torch_coverage_first_subset_partition._knn``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

from osn_gs.utils.torch_ops import require_torch


CLASS_ORDER_ONLY = 1
CLASS_EXACT_DISTANCE_TIE = 2
CLASS_FLOAT32_REFERENCE_TIE = 3
CLASS_NEAR_TIE_ERROR_BOUND = 4
CLASS_MATERIAL_DISAGREEMENT = 5
CLASS_DUPLICATE_OR_SELF = 6
CLASS_UNATTRIBUTED = 7

CLASS_NAMES = {
    CLASS_ORDER_ONLY: "A_ORDER_ONLY",
    CLASS_EXACT_DISTANCE_TIE: "C_EXACT_COORDINATE_OR_DISTANCE_TIE",
    CLASS_FLOAT32_REFERENCE_TIE: "D_FLOAT32_REFERENCE_TIE",
    CLASS_NEAR_TIE_ERROR_BOUND: "E_NEAR_TIE_FLOAT32_ERROR_BOUND",
    CLASS_MATERIAL_DISAGREEMENT: "F_MATERIAL_DISTANCE_DISAGREEMENT",
    CLASS_DUPLICATE_OR_SELF: "G_SELF_EXCLUSION_OR_DUPLICATE_POINT",
    CLASS_UNATTRIBUTED: "H_UNATTRIBUTED",
}


@dataclass
class ReferenceKNNObservation:
    neighbor_index: Any
    neighbor_distance: Any
    boundary_index: Any
    boundary_raw_distance: Any
    probe_raw_distance: Any | None
    elapsed_seconds: float
    compute_mode: str | None


def observe_reference_knn(
    positions: Any,
    k: int,
    chunk_size: int,
    *,
    compute_mode: str | None = None,
    probe_index: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> ReferenceKNNObservation:
    """Observe raw ``topk(K+1)`` values using the reference call shape/order."""

    torch = require_torch()
    count = int(positions.shape[0])
    boundary_indices = []
    boundary_distances = []
    probe_distances = []
    exact_distances = []
    started = time.perf_counter()
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        if compute_mode is None:
            distance = torch.cdist(positions[start:end], positions)
        else:
            distance = torch.cdist(
                positions[start:end], positions, compute_mode=compute_mode
            )
        rows = torch.arange(end - start, device=positions.device)
        distance[rows, torch.arange(start, end, device=positions.device)] = float("inf")
        raw, indices = torch.topk(
            distance, int(k) + 1, dim=1, largest=False, sorted=True
        )
        if probe_index is not None:
            probe_distances.append(distance.gather(1, probe_index[start:end]))
        exact_distances.append(
            (positions[start:end, None, :] - positions[indices[:, :k]]).norm(dim=-1)
        )
        boundary_indices.append(indices)
        boundary_distances.append(raw)
        del distance
        if progress is not None and (start // int(chunk_size)) % 200 == 0:
            progress(f"reference observation rows {end}/{count} mode={compute_mode or 'default'}")
    if positions.device.type == "cuda":
        torch.cuda.synchronize(positions.device)
    return ReferenceKNNObservation(
        neighbor_index=torch.cat(boundary_indices, dim=0)[:, :k],
        neighbor_distance=torch.cat(exact_distances, dim=0),
        boundary_index=torch.cat(boundary_indices, dim=0),
        boundary_raw_distance=torch.cat(boundary_distances, dim=0),
        probe_raw_distance=torch.cat(probe_distances, dim=0) if probe_distances else None,
        elapsed_seconds=time.perf_counter() - started,
        compute_mode=compute_mode,
    )


def _first_masked_rank(mask: Any) -> tuple[Any, Any]:
    """Return first true rank and whether one exists for each row."""

    torch = require_torch()
    has = mask.any(dim=1)
    rank = mask.to(torch.int64).argmax(dim=1)
    return rank, has


def classify_neighbor_mismatches(
    positions: Any,
    reference_index: Any,
    reference_boundary_raw: Any,
    candidate_index: Any,
    reference_raw_for_candidate: Any,
) -> dict[str, Any]:
    """Classify every ordered-row mismatch under predeclared A~H rules.

    ``candidate_index`` and ``reference_raw_for_candidate`` must include at
    least K candidates in the historical candidate's mathematical order.
    The returned tensors remain on the input device so the caller can persist
    a complete row-level attribution artifact without Python row loops.
    """

    torch = require_torch()
    k = int(reference_index.shape[1])
    candidate_k = candidate_index[:, :k]
    ordered_equal = (reference_index == candidate_k).all(dim=1)
    mismatch = ~ordered_equal
    reference_sorted = reference_index.sort(dim=1).values
    candidate_sorted = candidate_k.sort(dim=1).values
    set_equal = (reference_sorted == candidate_sorted).all(dim=1)
    order_only = mismatch & set_equal
    membership_mismatch = mismatch & ~set_equal

    reference_in_candidate = (
        reference_index[:, :, None] == candidate_k[:, None, :]
    ).any(dim=2)
    candidate_in_reference = (
        candidate_k[:, :, None] == reference_index[:, None, :]
    ).any(dim=2)
    reference_only_mask = ~reference_in_candidate
    candidate_only_mask = ~candidate_in_reference
    # Reference-only: use its largest reference rank (closest to K boundary).
    reverse_rank, has_reference_only = _first_masked_rank(reference_only_mask.flip(1))
    reference_only_rank = (k - 1) - reverse_rank
    candidate_only_rank, has_candidate_only = _first_masked_rank(candidate_only_mask)
    rows = torch.arange(int(reference_index.shape[0]), device=positions.device)
    reference_only_index = reference_index[rows, reference_only_rank]
    candidate_only_index = candidate_k[rows, candidate_only_rank]
    reference_only_raw = reference_boundary_raw[rows, reference_only_rank]
    candidate_only_raw = reference_raw_for_candidate[rows, candidate_only_rank]

    query = positions.to(torch.float64)
    ref_point = positions[reference_only_index].to(torch.float64)
    cand_point = positions[candidate_only_index].to(torch.float64)
    ref_delta = query - ref_point
    cand_delta = query - cand_point
    ref_squared = (ref_delta * ref_delta).sum(dim=1)
    cand_squared = (cand_delta * cand_delta).sum(dim=1)
    math_gap = (ref_squared - cand_squared).abs()
    exact_tie = membership_mismatch & (ref_squared == cand_squared)
    raw_tie = membership_mismatch & ~exact_tie & (
        reference_only_raw.view(torch.int32) == candidate_only_raw.view(torch.int32)
    )

    query32 = positions
    ref32 = positions[reference_only_index]
    cand32 = positions[candidate_only_index]
    unit_roundoff = 2.0 ** -24
    gamma9 = (9.0 * unit_roundoff) / (1.0 - 9.0 * unit_roundoff)

    def mm_squared_error_scale(other: Any) -> Any:
        scale = (
            query32.square().sum(dim=1)
            + other.square().sum(dim=1)
            + 2.0 * (query32 * other).abs().sum(dim=1)
        ).to(torch.float64)
        return gamma9 * scale

    error_bound = mm_squared_error_scale(ref32) + mm_squared_error_scale(cand32)
    near_tie = membership_mismatch & ~exact_tie & ~raw_tie & (math_gap <= error_bound)

    query_duplicate = (
        (positions[reference_index] == positions[:, None, :]).all(dim=2).any(dim=1)
        | (positions[candidate_k] == positions[:, None, :]).all(dim=2).any(dim=1)
    )
    duplicate_effect = mismatch & query_duplicate
    material = membership_mismatch & ~exact_tie & ~raw_tie & ~near_tie
    evidence_available = has_reference_only & has_candidate_only

    primary = torch.zeros_like(rows, dtype=torch.int8)
    primary[mismatch] = CLASS_UNATTRIBUTED
    primary[material] = CLASS_MATERIAL_DISAGREEMENT
    primary[near_tie] = CLASS_NEAR_TIE_ERROR_BOUND
    primary[raw_tie] = CLASS_FLOAT32_REFERENCE_TIE
    primary[exact_tie] = CLASS_EXACT_DISTANCE_TIE
    primary[order_only] = CLASS_ORDER_ONLY
    primary[duplicate_effect] = CLASS_DUPLICATE_OR_SELF
    primary[membership_mismatch & ~evidence_available & ~duplicate_effect] = CLASS_UNATTRIBUTED

    first_k_minus_one_equal = (
        reference_index[:, : max(k - 1, 0)].sort(dim=1).values
        == candidate_k[:, : max(k - 1, 0)].sort(dim=1).values
    ).all(dim=1) if k > 1 else torch.ones_like(mismatch)
    symmetric_difference_count = reference_only_mask.sum(dim=1) + candidate_only_mask.sum(dim=1)
    boundary_only = membership_mismatch & first_k_minus_one_equal & (symmetric_difference_count == 2)

    raw_gap = (reference_only_raw - candidate_only_raw).abs()
    ulp_ref = torch.nextafter(
        reference_only_raw, torch.full_like(reference_only_raw, float("inf"))
    ) - reference_only_raw
    ulp_cand = torch.nextafter(
        candidate_only_raw, torch.full_like(candidate_only_raw, float("inf"))
    ) - candidate_only_raw
    ulp_scale = torch.maximum(ulp_ref.abs(), ulp_cand.abs()).clamp_min(torch.finfo(positions.dtype).tiny)
    ulp_separation = raw_gap / ulp_scale
    return {
        "ordered_equal": ordered_equal,
        "mismatch": mismatch,
        "set_equal": set_equal,
        "order_only": order_only,
        "membership_mismatch": membership_mismatch,
        "boundary_only": boundary_only,
        "primary_class": primary,
        "reference_only_index": reference_only_index,
        "candidate_only_index": candidate_only_index,
        "reference_only_raw": reference_only_raw,
        "candidate_only_raw": candidate_only_raw,
        "reference_squared_float64": ref_squared,
        "candidate_squared_float64": cand_squared,
        "mathematical_squared_gap": math_gap,
        "derived_squared_error_bound": error_bound,
        "raw_distance_gap": raw_gap,
        "ulp_separation": ulp_separation,
        "duplicate_effect": duplicate_effect,
    }


def boundary_margins(boundary_raw_distance: Any, k: int) -> tuple[Any, Any]:
    absolute = boundary_raw_distance[:, k] - boundary_raw_distance[:, k - 1]
    relative = absolute / boundary_raw_distance[:, k - 1].clamp_min(1e-30)
    return absolute, relative


def distance_arithmetic_variants(positions: Any, row: int, candidate: int) -> dict[str, float]:
    """Recompute one pair without claiming any variant is authoritative."""

    torch = require_torch()
    left = positions[int(row)]
    right = positions[int(candidate)]
    delta = left - right
    pair_left = left.reshape(1, 1, 3)
    pair_right = right.reshape(1, 1, 3)
    return {
        "explicit_norm_float32": float(delta.norm()),
        "squared_reduce_sqrt_float32": float((delta * delta).sum().sqrt()),
        "linalg_vector_norm_float32": float(torch.linalg.vector_norm(delta)),
        "explicit_norm_float64": float((left.to(torch.float64) - right.to(torch.float64)).norm()),
        "cdist_mm_pair_shape": float(torch.cdist(
            pair_left, pair_right, compute_mode="use_mm_for_euclid_dist"
        ).reshape(())),
        "cdist_direct_pair_shape": float(torch.cdist(
            pair_left, pair_right, compute_mode="donot_use_mm_for_euclid_dist"
        ).reshape(())),
    }


def adversarial_knn_fixtures(device: Any = "cpu") -> dict[str, tuple[Any, int]]:
    """S1~S8 fixtures frozen before full-scene attribution interpretation."""

    torch = require_torch()
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(1194)
    s1 = torch.rand((64, 3), generator=generator, dtype=dtype, device=device) * 100.0
    s2 = torch.tensor([
        [0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, 0, 3],
    ], dtype=dtype, device=device)
    s3 = torch.tensor([
        [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1], [2, 0, 0],
    ], dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)
    one_up = torch.nextafter(one, torch.tensor(float("inf"), dtype=dtype, device=device))
    one_up2 = torch.nextafter(one_up, torch.tensor(float("inf"), dtype=dtype, device=device))
    s4 = torch.stack([
        torch.zeros(3, dtype=dtype, device=device),
        torch.stack((one, one.new_zeros(()), one.new_zeros(()))),
        torch.stack((one_up, one.new_zeros(()), one.new_zeros(()))),
        torch.stack((one_up2, one.new_zeros(()), one.new_zeros(()))),
        torch.tensor([-1.0, 0.0, 0.0], dtype=dtype, device=device),
    ])
    s5 = torch.tensor([
        [0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0],
        [0, 0, 0.5], [0, 0, -0.5], [1, 0, 0], [-1, 0, 0],
    ], dtype=dtype, device=device)
    offset = torch.tensor([1_000_000.0, -1_000_000.0, 500_000.0], dtype=dtype, device=device)
    local = torch.tensor([
        [0, 0, 0], [0.0625, 0, 0], [0, 0.0625, 0], [0, 0, 0.0625],
        [0.125, 0, 0], [0, 0.125, 0], [0, 0, 0.125], [0.125, 0.125, 0],
    ], dtype=dtype, device=device)
    s6 = offset + local
    axis = torch.linspace(-1.0, 1.0, 32, dtype=dtype, device=device)
    s7 = torch.stack((1e4 * axis, axis.square(), 1e-3 * torch.sin(axis * 5.0)), dim=1)
    centers = torch.tensor([
        [-2, -2, 0], [2, -2, 0], [-2, 2, 0], [2, 2, 0],
        [0, 0, -2], [0, 0, 2], [3, 0, 0], [-3, 0, 0],
    ], dtype=dtype, device=device)
    cluster_id = torch.arange(128, device=device) % len(centers)
    jitter = 0.01 * torch.randn((128, 3), generator=generator, dtype=dtype, device=device)
    s8 = centers[cluster_id] + jitter
    return {
        "S1_well_separated_random": (s1, 8),
        "S2_exact_duplicate_coordinates": (s2, 3),
        "S3_exact_equal_distance_neighbors": (s3, 4),
        "S4_float32_ulp_near_equal": (s4, 2),
        "S5_k_boundary_tie": (s5, 4),
        "S6_large_offset_small_spacing": (s6, 3),
        "S7_anisotropic_magnitudes": (s7, 4),
        "S8_clustered_local_geometry": (s8, 8),
    }
