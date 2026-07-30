from __future__ import annotations

"""Full-cloud continuation shell and continuous circular support-gap query (worklog 130).

Problem this addresses: ``torch_boundary_support_termination.py`` decides
whether a representative node sits at a genuine physical surface edge using
ONLY its same-region ACCEPTED-TOPOLOGY neighbors -- i.e. only other
*representatives*, never the much denser full observed cloud each
representative was drawn from (worklog 129). On a real ADC-trained scene, a
representative's accepted-topology neighborhood can look sparse simply
because representatives are capped far below the true Gaussian count, not
because the physical surface actually ends there. The old sector-based query
therefore cannot distinguish:

1. ``observed_support_termination`` -- the surface genuinely ends here.
2. ``reliability_frontier`` -- the surface keeps going, but only ambiguous
   (not yet core-admitted) evidence backs it in that direction.
3. ``unresolved_sampling_gap`` -- density is low in every sense (few full
   Gaussians observed at all near this node), so termination cannot be
   confirmed OR denied.
4. ``crease_discontinuity`` / ``parallel_sheet_conflict`` -- a DIFFERENT
   surface mode occupies that angular direction (not empty space at all).
5. ``ambiguous_continuation`` -- some non-rejected evidence exists but does
   not cleanly fit any of the above.

This module builds a READ-ONLY "continuation support shell" per query node:
every full-cloud Gaussian assigned (via worklog 129's existing
nearest-representative Voronoi partition) to this representative OR to a
same-region representative within an adaptive radius. It reuses that
existing assignment and the already-computed
:class:`~osn_gs.surface.torch_full_neighborhood_evidence.FullNeighborhoodEvidence`
(mean spacing, local density) -- no new O(N) eigen-decomposition or O(N*M)
distance computation is introduced beyond one bounded (M x M) representative-
to-representative distance matrix (M is the already-capped representative
count).

Shell membership NEVER changes region ownership, cluster assignment,
accepted-topology membership, or NURBS fitting support -- it is consumed
exclusively by :mod:`torch_boundary_support_termination` to classify a
node's own sector-gap direction, and to compute that gap CONTINUOUSLY
(fine-angular-bin occupancy with each Gaussian contributing an angular
FOOTPRINT interval, not a single point direction) instead of via a coarse
fixed 8-sector histogram.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame
from osn_gs.surface.torch_gaussian_structural_reliability import (
    INTRINSIC_REJECTED,
    IntrinsicReliabilityResult,
)
from osn_gs.utils.torch_ops import require_torch

POLICY_VERSION = "full_cloud_continuation_shell_worklog130_v1"

STATE_OBSERVED_TERMINATION = "observed_support_termination"
STATE_RELIABILITY_FRONTIER = "reliability_frontier"
STATE_SAMPLING_GAP = "unresolved_sampling_gap"
STATE_CREASE = "crease_discontinuity"
STATE_PARALLEL_CONFLICT = "parallel_sheet_conflict"
STATE_AMBIGUOUS = "ambiguous_continuation"
STATE_REJECTED_ADJACENCY = "rejected_neighbor_adjacency"
STATE_NO_GAP = "no_gap"


@dataclass(frozen=True)
class ContinuationShellInput:
    """Bundles the full-cloud data a continuation-shell query needs, all
    reused (never recomputed) from worklog 129's existing one-shot full-cloud
    pass -- see ``TorchOSNGSPipeline._construct_canonical_with_full_evidence``.
    """

    full_positions: Any
    full_frame: GaussianCovarianceFrame
    full_intrinsic: IntrinsicReliabilityResult
    full_opacity: Any
    full_stable_ids: Sequence[Any]
    nearest_representative_index: Any  # (N,) long, indices into the representative/positions array
    representative_mean_spacing: Any  # (M,) worklog 129 FullNeighborhoodEvidence.mean_spacing
    config: "ContinuationShellConfig" = field(default_factory=lambda: ContinuationShellConfig())


@dataclass(frozen=True)
class ContinuationShellConfig:
    """Configurable policy, not a confirmed canonical threshold set."""

    # --- radius (worklog 130 item 6: never a global fixed distance) ---
    radius_tangent_scale_multiplier: float = 6.0
    radius_spacing_multiplier: float = 4.0
    min_radius_neighbor_reps: int = 1  # always includes the query representative itself

    # --- same-mode filtering (item 7) ---
    same_mode_normal_alignment_min: float = 0.75
    same_mode_residual_max_ratio: float = 1.5
    crease_alignment_max: float = 0.5
    footprint_ratio_min: float = 0.15
    footprint_ratio_max: float = 6.0

    # --- continuous circular gap query (item 5) ---
    angular_bins: int = 180
    min_gap_degrees: float = 24.0
    max_footprint_halfwidth_degrees: float = 60.0

    # --- termination classification (item 4) ---
    min_same_mode_support_for_termination: int = 6
    sampling_gap_local_density_ratio: float = 0.34  # vs. this node's OWN full-neighborhood evidence density baseline
    gap_direction_mass_epsilon: float = 1e-6

    policy_version: str = POLICY_VERSION


@dataclass(frozen=True)
class ContinuationTerminationQuery:
    """Per-node classification of its own sector-gap direction (read-only diagnostic + candidate source)."""

    node_id: Any
    state: str
    gap_width_degrees: float
    outward_direction: Any | None  # (3,) world-space unit vector, or None if state == "no_gap"
    same_mode_support_count: int
    same_mode_opacity_mass: float
    ambiguous_continuation_mass: float
    competing_mode_mass: float
    support_radius: float
    neighbor_representative_count: int
    source_full_cloud_fingerprint: tuple[Any, ...]
    policy_version: str = POLICY_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "gap_width_degrees": self.gap_width_degrees,
            "same_mode_support_count": self.same_mode_support_count,
            "same_mode_opacity_mass": self.same_mode_opacity_mass,
            "ambiguous_continuation_mass": self.ambiguous_continuation_mass,
            "competing_mode_mass": self.competing_mode_mass,
            "support_radius": self.support_radius,
            "neighbor_representative_count": self.neighbor_representative_count,
        }


def _unit(vector: Any) -> Any:
    return vector / vector.norm().clamp_min(1e-12)


def _group_full_indices_by_representative(nearest_representative_index: Any, representative_count: int) -> list[Any]:
    """Return a list (length M) of long tensors: full-cloud indices assigned to each representative."""
    torch = require_torch()
    order = torch.argsort(nearest_representative_index)
    sorted_assignment = nearest_representative_index[order]
    counts = torch.bincount(sorted_assignment, minlength=representative_count)
    boundaries = torch.cumsum(counts, dim=0)
    starts = boundaries - counts
    groups = []
    for representative in range(representative_count):
        start = int(starts[representative])
        end = int(boundaries[representative])
        groups.append(order[start:end])
    return groups


def _largest_circular_gap_from_bins(occupied: Any, bins: int) -> tuple[int, int]:
    """Return (start_bin, run_length) of the largest circular contiguous unoccupied run."""
    torch = require_torch()
    if bool(occupied.all()):
        return 0, 0
    if not bool(occupied.any()):
        return 0, bins
    occupied_list = occupied.tolist()
    start = next(i for i in range(bins) if occupied_list[i] and not occupied_list[(i + 1) % bins])
    ordered = [(start + 1 + offset) % bins for offset in range(bins - 1)]
    best_start, best_length = None, 0
    current_start, current_length = None, 0
    for item in ordered:
        if occupied_list[item]:
            if current_length > best_length:
                best_start, best_length = current_start, current_length
            current_start, current_length = None, 0
        else:
            if current_length == 0:
                current_start = item
            current_length += 1
    if current_length > best_length:
        best_start, best_length = current_start, current_length
    if best_start is None:
        return 0, 0
    return best_start, best_length


def build_continuation_shells(
    full_positions: Any,
    full_frame: GaussianCovarianceFrame,
    full_intrinsic: IntrinsicReliabilityResult,
    full_opacity: Any,
    full_stable_ids: Sequence[Any],
    nearest_representative_index: Any,
    representative_positions: Any,
    representative_frame: GaussianCovarianceFrame,
    representative_ids: Sequence[Any],
    representative_region_id: Sequence[int],
    representative_membership_state: Sequence[str],
    representative_mean_spacing: Any,
    canonical_frames: Sequence[Any | None],
    *,
    config: ContinuationShellConfig | None = None,
) -> dict[Any, ContinuationTerminationQuery]:
    """Build a read-only continuation-termination classification per eligible representative node.

    Only representatives eligible as support-termination sources under the
    EXISTING gate (``region_id >= 0`` and membership state in
    ``("core_member", "consensus_attached")`` and a non-``None`` canonical
    frame) are queried -- this mirrors
    ``torch_boundary_support_termination.extract_support_termination_candidates``'s
    own eligibility gate exactly, so this function's output is a drop-in
    per-node lookup for that module.

    Reuses (never recomputes): ``nearest_representative_index`` (worklog 129's
    ``assign_nearest_representative`` output), ``full_frame``/``full_intrinsic``
    (worklog 129's one-shot O(N) full-cloud covariance/intrinsic evaluation),
    and ``representative_mean_spacing`` (worklog 129's
    ``FullNeighborhoodEvidence.mean_spacing``).
    """
    torch = require_torch()
    config = config or ContinuationShellConfig()
    m = int(representative_positions.shape[0])
    result: dict[Any, ContinuationTerminationQuery] = {}
    if m == 0:
        return result
    device = representative_positions.device

    representative_distance = torch.cdist(representative_positions, representative_positions)
    radius = torch.maximum(
        config.radius_tangent_scale_multiplier * representative_frame.tangent_major_scale,
        config.radius_spacing_multiplier * representative_mean_spacing.clamp_min(1e-9),
    )
    members_by_representative = _group_full_indices_by_representative(nearest_representative_index, m)
    region_id_tensor = torch.tensor(list(representative_region_id), dtype=torch.long, device=device)

    for source in range(m):
        region_id = int(representative_region_id[source])
        if region_id < 0 or representative_membership_state[source] not in ("core_member", "consensus_attached"):
            continue
        canonical = canonical_frames[source] if source < len(canonical_frames) else None
        if canonical is None:
            continue

        same_region_mask = region_id_tensor == region_id
        within_radius_mask = representative_distance[source] <= radius[source]
        neighbor_reps = torch.nonzero(same_region_mask & within_radius_mask, as_tuple=False).reshape(-1).tolist()
        if source not in neighbor_reps:
            neighbor_reps.append(source)

        member_index_parts = [members_by_representative[j] for j in neighbor_reps if members_by_representative[j].numel() > 0]
        if not member_index_parts:
            continue
        members = torch.cat(member_index_parts, dim=0)

        node_position = representative_positions[source]
        node_normal = canonical.oriented_normal
        axis_u = canonical.tangent_axis_0
        axis_v = canonical.tangent_axis_1
        node_tangent_scale = representative_frame.tangent_major_scale[source].clamp_min(1e-12)

        displacement = full_positions[members] - node_position
        radial_distance = torch.linalg.norm(displacement, dim=-1)
        within_radius = radial_distance <= radius[source]
        nonzero_radius = radial_distance > 1e-8
        keep = within_radius & nonzero_radius
        if not bool(keep.any()):
            continue
        members = members[keep]
        displacement = displacement[keep]
        radial_distance = radial_distance[keep]

        member_normal = full_frame.normal_candidate[members]
        sign = torch.where((member_normal * node_normal).sum(dim=-1) < 0.0, -1.0, 1.0).unsqueeze(-1)
        corrected_normal = member_normal * sign
        alignment = (corrected_normal * node_normal).sum(dim=-1).clamp(-1.0, 1.0)

        tangent_offset = (displacement * node_normal).sum(dim=-1)
        residual_ratio = tangent_offset.abs() / node_tangent_scale
        footprint_ratio = full_frame.tangent_major_scale[members] / node_tangent_scale

        rejected = torch.tensor(
            [full_intrinsic.intrinsic_class[int(i)] == INTRINSIC_REJECTED for i in members.tolist()],
            device=device,
        )

        same_mode = (
            (~rejected)
            & (alignment >= config.same_mode_normal_alignment_min)
            & (residual_ratio <= config.same_mode_residual_max_ratio)
            & (footprint_ratio >= config.footprint_ratio_min)
            & (footprint_ratio <= config.footprint_ratio_max)
        )
        parallel_conflict = (
            (~rejected)
            & (~same_mode)
            & (alignment >= config.same_mode_normal_alignment_min)
            & (residual_ratio > config.same_mode_residual_max_ratio)
        )
        crease = (~rejected) & (~same_mode) & (~parallel_conflict) & (alignment < config.crease_alignment_max)
        ambiguous = (~rejected) & (~same_mode) & (~parallel_conflict) & (~crease)

        opacity = full_opacity[members]
        same_mode_support_count = int(same_mode.sum().item())
        same_mode_opacity_mass = float(opacity[same_mode].sum().item()) if same_mode_support_count else 0.0
        ambiguous_mass = float(opacity[ambiguous].sum().item())
        competing_mass = float(opacity[parallel_conflict].sum().item() + opacity[crease].sum().item())

        tangent_vector = displacement - node_normal * tangent_offset.unsqueeze(-1)
        angle = torch.atan2((tangent_vector * axis_v).sum(dim=-1), (tangent_vector * axis_u).sum(dim=-1))
        # Reference distance for the angular footprint is the QUERY's own
        # search radius (fixed per query), not each member's own (often
        # near-zero at high sampling density) radial distance. Using the
        # per-member radial distance would let a handful of very-close
        # members subtend a near-180-degree angle each -- geometrically
        # "correct" in a strict near-field-optics sense, but it blankets the
        # whole circle and destroys real gap detection exactly where density
        # is highest. Referencing the query radius instead answers "how big
        # is this Gaussian's footprint relative to how far out this query
        # even looks", which still scales up appropriately for a genuinely
        # oversized footprint without becoming unstable for ordinary members.
        angular_halfwidth = torch.atan2(
            full_frame.equivalent_tangent_scale[members], radius[source]
        ).clamp(max=math.radians(config.max_footprint_halfwidth_degrees))

        bins = config.angular_bins
        bin_width = 2.0 * math.pi / bins
        occupied = torch.zeros((bins,), dtype=torch.bool)
        occupied_category = [[] for _ in range(bins)]  # which categories touch each bin, for gap classification

        def _mark(mask: Any, category: str) -> None:
            if not bool(mask.any()):
                return
            member_angle = angle[mask]
            member_halfwidth = angular_halfwidth[mask]
            for a, hw in zip(member_angle.tolist(), member_halfwidth.tolist()):
                lo = int(math.floor((a - hw + math.pi) / bin_width))
                hi = int(math.floor((a + hw + math.pi) / bin_width))
                for offset in range(lo, hi + 1):
                    b = offset % bins
                    occupied[b] = True
                    occupied_category[b].append(category)

        _mark(same_mode, "same_mode")
        _mark(parallel_conflict, "parallel_conflict")
        _mark(crease, "crease")
        _mark(ambiguous, "ambiguous")

        gap_threshold_bins = max(1, int(math.ceil(math.radians(config.min_gap_degrees) / bin_width)))
        best_start, best_length = _largest_circular_gap_from_bins(occupied, bins)
        gap_width_degrees = best_length * math.degrees(bin_width)

        fingerprint = tuple(full_stable_ids[int(i)] for i in members.tolist()[:64])
        node_id = representative_ids[source]

        if best_length < gap_threshold_bins:
            result[node_id] = ContinuationTerminationQuery(
                node_id=node_id, state=STATE_NO_GAP, gap_width_degrees=gap_width_degrees,
                outward_direction=None, same_mode_support_count=same_mode_support_count,
                same_mode_opacity_mass=same_mode_opacity_mass, ambiguous_continuation_mass=ambiguous_mass,
                competing_mode_mass=competing_mass, support_radius=float(radius[source]),
                neighbor_representative_count=len(neighbor_reps), source_full_cloud_fingerprint=fingerprint,
            )
            continue

        gap_center_angle = -math.pi + (best_start + best_length / 2.0) * bin_width
        outward_direction = axis_u * math.cos(gap_center_angle) + axis_v * math.sin(gap_center_angle)

        # What (if anything) borders the gap run tells us WHY it's empty --
        # a gap flanked by parallel/crease evidence is a different surface,
        # not missing observation.
        border_categories: set[str] = set()
        for offset in range(-1, best_length + 1):
            bin_index = (best_start + offset) % bins
            border_categories.update(occupied_category[bin_index])

        insufficient_support = same_mode_support_count < config.min_same_mode_support_for_termination
        if "parallel_conflict" in border_categories and not same_mode_support_count:
            state = STATE_PARALLEL_CONFLICT
        elif "crease" in border_categories and not same_mode_support_count:
            state = STATE_CREASE
        elif insufficient_support and ambiguous_mass > config.gap_direction_mass_epsilon:
            state = STATE_RELIABILITY_FRONTIER
        elif insufficient_support:
            state = STATE_SAMPLING_GAP
        elif "ambiguous" in border_categories:
            state = STATE_AMBIGUOUS
        else:
            state = STATE_OBSERVED_TERMINATION

        result[node_id] = ContinuationTerminationQuery(
            node_id=node_id, state=state, gap_width_degrees=gap_width_degrees,
            outward_direction=_unit(outward_direction), same_mode_support_count=same_mode_support_count,
            same_mode_opacity_mass=same_mode_opacity_mass, ambiguous_continuation_mass=ambiguous_mass,
            competing_mode_mass=competing_mass, support_radius=float(radius[source]),
            neighbor_representative_count=len(neighbor_reps), source_full_cloud_fingerprint=fingerprint,
        )

    return result


def build_continuation_shells_from_input(
    shell_input: ContinuationShellInput,
    representative_positions: Any,
    representative_frame: GaussianCovarianceFrame,
    representative_ids: Sequence[Any],
    region_result: Any,
    canonical_frames: Sequence[Any | None],
) -> dict[Any, ContinuationTerminationQuery]:
    """Adapter: pull region-formation output into :func:`build_continuation_shells`'s positional shape."""
    return build_continuation_shells(
        shell_input.full_positions,
        shell_input.full_frame,
        shell_input.full_intrinsic,
        shell_input.full_opacity,
        shell_input.full_stable_ids,
        shell_input.nearest_representative_index,
        representative_positions,
        representative_frame,
        representative_ids,
        region_result.node_region_id,
        region_result.node_membership_state,
        shell_input.representative_mean_spacing,
        canonical_frames,
        config=shell_input.config,
    )
