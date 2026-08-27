from __future__ import annotations

"""Worklog 122 -- CANDIDATE B ONLY. Renderer-defined median surface frontier
validation diagnostics.

Candidate B's decision function is NOT touched anywhere in this module: every
verdict comes from calling worklog 120's own unmodified
`candidate_b_median_depth.classify_view`. No epsilon, no tolerance, no
threshold sweep, no B+D hybrid. A/C/D are not revived.

The claim under test is deliberately narrow:

    NOT  "median depth is the physical first ray/surface hit"
    BUT  "the renderer's own selected visible-surface event provides a
          coherent, closed, sufficiently non-contradictory frontier separating
          the camera-facing observed domain from the behind-surface domain"

Three diagnostic families live here:

  1. FRONTIER SELF-CLOSURE -- a renderer-defined visible-surface event must
     land on the observed side of its OWN source-view frontier. Every valid
     median event in every training view is reconstructed with worklog 119's G2
     geometry, projected back into its own source camera, and classified by the
     frozen candidate B.

  2. NUMERICAL BOUNDARY ATTRIBUTION -- where closure is lost, decompose the
     dataflow (world reconstruction -> float32 storage -> camera transform ->
     raster pixel -> camera-space depth -> comparison) and attribute the cause.
     ULP distances are MEASURED, never used as a tolerance.

  3. FRONTIER IDENTITY -- whether a principled representation exists in which
     "this query IS the renderer's own median surface event" holds exactly.
     Diagnostic only; candidate B is unchanged.

Post-median contributor accounting is exhaustive and comes from the worklog 122
additive CUDA aggregates (see `torch_surfel_query_depth_diagnostics`), not from
worklog 110's bounded slot array, whose 97.4% truncation rate would bias it.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import candidate_b_median_depth as candidate_b
from .shared import (
    RELEVANCE_OK,
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    project_queries,
    reconstruct_direct_surfel_intersection_world_point,
)

# Closure failure causes. These are CLASSIFICATIONS OF MEASURED FACTS, not
# thresholds: each case is decided by exact integer/boolean quantities
# (did the raster pixel change? how many float32 ULPs apart are the two
# depths?), never by a tuned epsilon.
CAUSE_CLOSED = 0
CAUSE_PIXEL_REASSIGNMENT = 1
CAUSE_ROUNDTRIP_1ULP = 2
CAUSE_ROUNDTRIP_FEW_ULP = 3
CAUSE_ROUNDTRIP_LARGE = 4
CAUSE_NO_VALID_MEDIAN_AT_REPROJECTION = 5
CAUSE_NON_RELEVANT = 6
CAUSE_NAMES = {
    CAUSE_CLOSED: "CLOSED_OBSERVED",
    CAUSE_PIXEL_REASSIGNMENT: "RASTER_PIXEL_REASSIGNMENT",
    CAUSE_ROUNDTRIP_1ULP: "ROUND_TRIP_1_ULP",
    CAUSE_ROUNDTRIP_FEW_ULP: "ROUND_TRIP_2_TO_8_ULP",
    CAUSE_ROUNDTRIP_LARGE: "ROUND_TRIP_ABOVE_8_ULP",
    CAUSE_NO_VALID_MEDIAN_AT_REPROJECTION: "NO_VALID_MEDIAN_AT_REPROJECTED_PIXEL",
    CAUSE_NON_RELEVANT: "REPROJECTION_NOT_RELEVANT",
}

# Post-median category indices, mirroring OSN_GS_POST_MEDIAN_CATEGORIES in the
# diagnostic sibling's config.h.
POST_MEDIAN_CATEGORIES = (
    "all", "same_component", "cross_component", "unresolved_component",
    "representative_this_view", "representative_other_view", "never_representative",
    "rho2d_low_pass", "depth_in_front_of_median", "depth_at_or_behind_median",
)


def float32_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact float32 ULP distance between two positive-float arrays.

    Reinterprets the IEEE-754 bit patterns as ordered integers, which is exact
    for finite same-sign values. Used only to DESCRIBE how far apart the two
    depths are; nothing is ever accepted or rejected because of it.
    """

    a = np.asarray(left, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(right, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


@dataclass
class ClosureAccumulator:
    """Streaming exhaustive accounting over every valid median event in every
    training view. Only aggregates and a deterministic sample are retained --
    the full population is tens of millions of events."""

    total_events: int = 0
    relevant_events: int = 0
    observed: int = 0
    occluded: int = 0
    unresolved: int = 0
    pixel_preserved: int = 0
    pixel_changed: int = 0
    cause_counts: dict[int, int] = field(default_factory=lambda: {code: 0 for code in CAUSE_NAMES})
    ulp_histogram: dict[int, int] = field(default_factory=dict)
    signed_delta_sum: float = 0.0
    signed_delta_min: float = float("inf")
    signed_delta_max: float = float("-inf")
    abs_delta_max: float = 0.0
    contradiction_by_branch: dict[str, int] = field(default_factory=lambda: {"rho3d": 0, "rho2d": 0})
    events_by_branch: dict[str, int] = field(default_factory=lambda: {"rho3d": 0, "rho2d": 0})
    contradiction_depth_sum: float = 0.0
    contradiction_samples: list[dict[str, Any]] = field(default_factory=list)
    closed_samples: list[dict[str, Any]] = field(default_factory=list)
    per_view: list[dict[str, Any]] = field(default_factory=list)
    per_region_events: dict[int, int] = field(default_factory=dict)
    per_region_contradictions: dict[int, int] = field(default_factory=dict)
    identity_observed: int = 0
    identity_total: int = 0

    def summary(self) -> dict[str, Any]:
        denominator = max(self.total_events, 1)
        return {
            "total_median_events_tested": self.total_events,
            "reprojection_relevant": self.relevant_events,
            "source_view_OBSERVED": self.observed,
            "source_view_OCCLUDED": self.occluded,
            "source_view_UNRESOLVED": self.unresolved,
            "closure_contradiction_count": self.occluded + self.unresolved,
            "closure_contradiction_rate": (self.occluded + self.unresolved) / denominator,
            "source_pixel_preserved": self.pixel_preserved,
            "source_pixel_changed": self.pixel_changed,
            "source_pixel_preserved_fraction": self.pixel_preserved / denominator,
            "cause_counts": {CAUSE_NAMES[code]: count for code, count in sorted(self.cause_counts.items())},
            "ulp_histogram": {str(key): value for key, value in sorted(self.ulp_histogram.items())},
            "signed_delta_mean": self.signed_delta_sum / denominator,
            "signed_delta_min": self.signed_delta_min if np.isfinite(self.signed_delta_min) else 0.0,
            "signed_delta_max": self.signed_delta_max if np.isfinite(self.signed_delta_max) else 0.0,
            "absolute_delta_max": self.abs_delta_max,
            "events_by_rho_branch": dict(self.events_by_branch),
            "contradictions_by_rho_branch": dict(self.contradiction_by_branch),
            "contradiction_rate_by_rho_branch": {
                branch: (self.contradiction_by_branch[branch] / max(self.events_by_branch[branch], 1))
                for branch in self.events_by_branch
            },
            "mean_depth_of_contradictions": (
                self.contradiction_depth_sum / max(self.occluded + self.unresolved, 1)
            ),
            "exact_identity_representation": {
                "events_tested": self.identity_total,
                "OBSERVED": self.identity_observed,
                "contradictions": self.identity_total - self.identity_observed,
                "representation": "source camera id + source pixel id + the renderer's own stored median depth",
                "note": (
                    "Diagnostic only -- candidate B is unchanged. This measures whether an exact "
                    "identity contract exists at the representation level, not whether B should adopt one."
                ),
            },
        }


def evaluate_frontier_closure_for_view(
    view_index: int,
    camera: Any,
    package: dict[str, Any],
    positions_full: torch.Tensor,
    tangent_u_full: torch.Tensor,
    tangent_v_full: torch.Tensor,
    scale_u_full: torch.Tensor,
    scale_v_full: torch.Tensor,
    accumulator: ClosureAccumulator,
    region_of_surfel: torch.Tensor | None = None,
    sample_stride: int = 0,
) -> dict[str, Any]:
    """Exhaustive self-closure test for ONE view.

    Every valid median event of this view is reconstructed (G2), projected back
    into this SAME camera, and classified by the FROZEN candidate B. Nothing is
    filtered by magnitude and no tolerance is applied anywhere.
    """

    device = positions_full.device
    height, width = int(camera.image_height), int(camera.image_width)
    representative = package["representative_id"].reshape(-1).to(torch.int64)
    median_flat = candidate_b.median_depth_map(package["out_others"]).reshape(-1)
    valid = representative >= 0
    source_pixels = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if source_pixels.numel() == 0:
        return {"view_index": view_index, "events": 0}

    world = reconstruct_direct_surfel_intersection_world_point(
        representative, package["median_s_u"], package["median_s_v"],
        positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
    )
    finite = torch.isfinite(world).all(dim=1)
    source_pixels = source_pixels[finite[source_pixels]]
    if source_pixels.numel() == 0:
        return {"view_index": view_index, "events": 0}

    event_world = world[source_pixels]
    geometry = project_queries(camera, event_world)
    states = candidate_b.classify_view(geometry, median_flat)["states"]

    source_row = torch.div(source_pixels, width, rounding_mode="floor")
    source_col = source_pixels - source_row * width
    pixel_preserved = (geometry.pixel_row == source_row) & (geometry.pixel_col == source_col)

    stored_median = median_flat[source_pixels]
    reprojected_median = median_flat[geometry.pixel_index.clamp(min=0)]
    signed_delta = geometry.depth - reprojected_median
    delta_vs_source = geometry.depth - stored_median

    rho3d = package["median_rho3d"].reshape(-1)[source_pixels]
    rho2d = package["median_rho2d"].reshape(-1)[source_pixels]
    is_rho3d = rho3d <= rho2d

    states_np = states.detach().cpu().numpy()
    relevant_np = geometry.relevant.detach().cpu().numpy()
    preserved_np = pixel_preserved.detach().cpu().numpy()
    depth_np = geometry.depth.detach().cpu().numpy()
    stored_np = stored_median.detach().cpu().numpy()
    signed_np = signed_delta.detach().cpu().numpy()
    reprojected_valid_np = (reprojected_median > 0).detach().cpu().numpy()
    is_rho3d_np = is_rho3d.detach().cpu().numpy()

    ulp = float32_ulp_distance(depth_np, stored_np)
    contradiction = states_np != STATE_OBSERVED

    cause = np.full(states_np.shape, CAUSE_CLOSED, dtype=np.int64)
    cause[contradiction & ~relevant_np] = CAUSE_NON_RELEVANT
    cause[contradiction & relevant_np & ~reprojected_valid_np] = CAUSE_NO_VALID_MEDIAN_AT_REPROJECTION
    remaining = contradiction & relevant_np & reprojected_valid_np
    cause[remaining & ~preserved_np] = CAUSE_PIXEL_REASSIGNMENT
    kept = remaining & preserved_np
    cause[kept & (ulp <= 1)] = CAUSE_ROUNDTRIP_1ULP
    cause[kept & (ulp > 1) & (ulp <= 8)] = CAUSE_ROUNDTRIP_FEW_ULP
    cause[kept & (ulp > 8)] = CAUSE_ROUNDTRIP_LARGE

    events = int(states_np.size)
    accumulator.total_events += events
    accumulator.relevant_events += int(relevant_np.sum())
    accumulator.observed += int((states_np == STATE_OBSERVED).sum())
    accumulator.occluded += int((states_np == STATE_OCCLUDED).sum())
    accumulator.unresolved += int((states_np == STATE_UNRESOLVED).sum())
    accumulator.pixel_preserved += int(preserved_np.sum())
    accumulator.pixel_changed += int((~preserved_np).sum())
    for code in CAUSE_NAMES:
        accumulator.cause_counts[code] += int((cause == code).sum())
    for key, value in zip(*np.unique(np.minimum(ulp, 64), return_counts=True)):
        accumulator.ulp_histogram[int(key)] = accumulator.ulp_histogram.get(int(key), 0) + int(value)
    accumulator.signed_delta_sum += float(signed_np.sum())
    accumulator.signed_delta_min = min(accumulator.signed_delta_min, float(signed_np.min()))
    accumulator.signed_delta_max = max(accumulator.signed_delta_max, float(signed_np.max()))
    accumulator.abs_delta_max = max(accumulator.abs_delta_max, float(np.abs(signed_np).max()))
    accumulator.events_by_branch["rho3d"] += int(is_rho3d_np.sum())
    accumulator.events_by_branch["rho2d"] += int((~is_rho3d_np).sum())
    accumulator.contradiction_by_branch["rho3d"] += int((contradiction & is_rho3d_np).sum())
    accumulator.contradiction_by_branch["rho2d"] += int((contradiction & ~is_rho3d_np).sum())
    accumulator.contradiction_depth_sum += float(depth_np[contradiction].sum())

    # Exact-identity representation (section 5): classify B with the query
    # depth taken as the renderer's OWN stored median depth at the SOURCE pixel.
    identity_geometry = project_queries(camera, event_world)
    identity_geometry.depth.copy_(stored_median)
    identity_geometry.pixel_index.copy_(source_pixels)
    identity_states = candidate_b.classify_view(identity_geometry, median_flat)["states"]
    accumulator.identity_total += int(identity_states.shape[0])
    accumulator.identity_observed += int((identity_states == STATE_OBSERVED).sum())

    if region_of_surfel is not None:
        region = region_of_surfel[representative[source_pixels]].detach().cpu().numpy()
        for value, count in zip(*np.unique(region, return_counts=True)):
            accumulator.per_region_events[int(value)] = accumulator.per_region_events.get(int(value), 0) + int(count)
        for value, count in zip(*np.unique(region[contradiction], return_counts=True)):
            accumulator.per_region_contradictions[int(value)] = (
                accumulator.per_region_contradictions.get(int(value), 0) + int(count)
            )

    if sample_stride > 0:
        representative_np = representative[source_pixels].detach().cpu().numpy()
        source_row_np = source_row.detach().cpu().numpy()
        source_col_np = source_col.detach().cpu().numpy()
        reprojected_row = geometry.pixel_row.detach().cpu().numpy()
        reprojected_col = geometry.pixel_col.detach().cpu().numpy()
        world_np = event_world.detach().cpu().numpy()
        delta_source_np = delta_vs_source.detach().cpu().numpy()

        def _record(index: int) -> dict[str, Any]:
            return {
                "view_index": view_index,
                "camera_name": str(getattr(camera, "image_name", view_index)),
                "source_pixel": [int(source_row_np[index]), int(source_col_np[index])],
                "reprojected_pixel": [int(reprojected_row[index]), int(reprojected_col[index])],
                "pixel_preserved": bool(preserved_np[index]),
                "representative_id": int(representative_np[index]),
                "world_position": [float(v) for v in world_np[index]],
                "stored_median_depth": float(stored_np[index]),
                "reprojected_query_depth": float(depth_np[index]),
                "signed_margin_at_reprojected_pixel": float(signed_np[index]),
                "delta_vs_source_pixel_median": float(delta_source_np[index]),
                "ulp_distance": int(ulp[index]),
                "rho_branch": "rho3d" if bool(is_rho3d_np[index]) else "rho2d",
                "B_state": {STATE_OBSERVED: "OBSERVED", STATE_OCCLUDED: "OCCLUDED",
                            STATE_UNRESOLVED: "UNRESOLVED", STATE_NON_RELEVANT: "NON_RELEVANT"}[int(states_np[index])],
                "closure_cause": CAUSE_NAMES[int(cause[index])],
            }

        contradiction_rows = np.nonzero(contradiction)[0]
        for index in contradiction_rows[:: max(1, contradiction_rows.size // 4 or 1)][:4]:
            if len(accumulator.contradiction_samples) < 60:
                accumulator.contradiction_samples.append(_record(int(index)))
        closed_rows = np.nonzero(~contradiction)[0]
        if closed_rows.size and len(accumulator.closed_samples) < 20:
            accumulator.closed_samples.append(_record(int(closed_rows[closed_rows.size // 2])))

    view_record = {
        "view_index": view_index,
        "camera_name": str(getattr(camera, "image_name", view_index)),
        "events": events,
        "OBSERVED": int((states_np == STATE_OBSERVED).sum()),
        "OCCLUDED": int((states_np == STATE_OCCLUDED).sum()),
        "UNRESOLVED": int((states_np == STATE_UNRESOLVED).sum()),
        "pixel_preserved": int(preserved_np.sum()),
    }
    accumulator.per_view.append(view_record)
    return view_record


@dataclass
class PostMedianAccumulator:
    """Exhaustive post-median accounting, streamed over all views. Every number
    comes from the worklog 122 additive CUDA aggregates -- never from worklog
    110's bounded (97.4% truncated) contributor slot array."""

    pixels_with_median: int = 0
    pixels_with_post_median: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in POST_MEDIAN_CATEGORIES})
    weights: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in POST_MEDIAN_CATEGORIES})
    total_accepted_weight: float = 0.0
    depth_offset_sum: float = 0.0
    depth_offset_min: float = float("inf")
    depth_offset_max: float = float("-inf")
    fraction_histogram: dict[str, int] = field(default_factory=dict)
    per_region_counts: dict[int, dict[str, float]] = field(default_factory=dict)
    per_pixel_fraction_samples: list[float] = field(default_factory=list)

    def accumulate(
        self,
        package: dict[str, Any],
        representative: torch.Tensor,
        region_of_surfel: torch.Tensor | None,
        sample_stride: int,
    ) -> None:
        counts = package["post_median_counts"].reshape(-1, len(POST_MEDIAN_CATEGORIES))
        weights = package["post_median_weights"].reshape(-1, len(POST_MEDIAN_CATEGORIES))
        total_weight = package["total_accepted_weight"].reshape(-1)
        depth_stats = package["post_median_depth_stats"].reshape(-1, 3)
        has_median = representative >= 0
        rows = torch.nonzero(has_median, as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            return
        self.pixels_with_median += int(rows.numel())
        selected_counts = counts[rows]
        selected_weights = weights[rows]
        has_post = selected_counts[:, 0] > 0
        self.pixels_with_post_median += int(has_post.sum())
        for index, name in enumerate(POST_MEDIAN_CATEGORIES):
            self.counts[name] += int(selected_counts[:, index].sum().item())
            self.weights[name] += float(selected_weights[:, index].sum().item())
        self.total_accepted_weight += float(total_weight[rows].sum().item())

        stats = depth_stats[rows]
        self.depth_offset_sum += float(stats[:, 0].sum().item())
        if bool(has_post.any()):
            self.depth_offset_min = min(self.depth_offset_min, float(stats[has_post, 1].min().item()))
            self.depth_offset_max = max(self.depth_offset_max, float(stats[has_post, 2].max().item()))

        safe_total = torch.clamp(total_weight[rows], min=1e-20)
        fraction = (selected_weights[:, 0] / safe_total).clamp(0.0, 1.0)
        bins = torch.clamp((fraction * 10).floor().to(torch.int64), 0, 9)
        for value, count in zip(*torch.unique(bins, return_counts=True)):
            key = f"{int(value) / 10:.1f}-{(int(value) + 1) / 10:.1f}"
            self.fraction_histogram[key] = self.fraction_histogram.get(key, 0) + int(count)

        if region_of_surfel is not None:
            region = region_of_surfel[representative[rows]]
            for value in torch.unique(region):
                mask = region == value
                bucket = self.per_region_counts.setdefault(
                    int(value), {name: 0.0 for name in POST_MEDIAN_CATEGORIES} | {"pixels": 0.0, "total_weight": 0.0}
                )
                bucket["pixels"] += float(mask.sum().item())
                bucket["total_weight"] += float(total_weight[rows][mask].sum().item())
                for index, name in enumerate(POST_MEDIAN_CATEGORIES):
                    bucket[name] += float(selected_weights[mask, index].sum().item())

        if sample_stride > 0 and len(self.per_pixel_fraction_samples) < 200000:
            sampled = fraction[:: sample_stride]
            self.per_pixel_fraction_samples.extend(sampled.detach().cpu().numpy().tolist())

    def summary(self) -> dict[str, Any]:
        post_all = max(self.counts["all"], 1)
        total_mass = max(self.total_accepted_weight, 1e-20)
        post_mass = self.weights["all"]
        return {
            "pixels_with_a_median_event": self.pixels_with_median,
            "pixels_with_at_least_one_post_median_contributor": self.pixels_with_post_median,
            "fraction_of_median_pixels_with_post_median_contribution": (
                self.pixels_with_post_median / max(self.pixels_with_median, 1)
            ),
            "post_median_accepted_contributors_total": self.counts["all"],
            "post_median_contributors_per_median_pixel_mean": self.counts["all"] / max(self.pixels_with_median, 1),
            "total_accepted_contribution_mass": self.total_accepted_weight,
            "post_median_contribution_mass": post_mass,
            "post_median_fraction_of_total_contribution": post_mass / total_mass,
            "counts_by_category": dict(self.counts),
            "contribution_mass_by_category": dict(self.weights),
            "count_fraction_by_category": {
                name: self.counts[name] / post_all for name in POST_MEDIAN_CATEGORIES
            },
            "mass_fraction_of_post_median_by_category": {
                name: (self.weights[name] / max(post_mass, 1e-20)) for name in POST_MEDIAN_CATEGORIES
            },
            "depth_offset_behind_median": {
                "mean": self.depth_offset_sum / post_all,
                "min": self.depth_offset_min if np.isfinite(self.depth_offset_min) else 0.0,
                "max": self.depth_offset_max if np.isfinite(self.depth_offset_max) else 0.0,
            },
            "per_pixel_post_median_mass_fraction_histogram": dict(sorted(self.fraction_histogram.items())),
        }


def region_table(accumulator: ClosureAccumulator, post: PostMedianAccumulator, region_labels: list[str]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for index, label in enumerate(region_labels):
        events = accumulator.per_region_events.get(index, 0)
        contradictions = accumulator.per_region_contradictions.get(index, 0)
        bucket = post.per_region_counts.get(index, {})
        post_mass = bucket.get("all", 0.0)
        table[label] = {
            "median_events": events,
            "closure_contradictions": contradictions,
            "closure_contradiction_rate": contradictions / max(events, 1),
            "pixels": int(bucket.get("pixels", 0.0)),
            "total_accepted_contribution_mass": bucket.get("total_weight", 0.0),
            "post_median_contribution_mass": post_mass,
            "post_median_fraction_of_total": post_mass / max(bucket.get("total_weight", 0.0), 1e-20),
            "post_median_same_component_mass": bucket.get("same_component", 0.0),
            "post_median_cross_component_mass": bucket.get("cross_component", 0.0),
            "post_median_unresolved_component_mass": bucket.get("unresolved_component", 0.0),
            "same_component_share_of_post_median": bucket.get("same_component", 0.0) / max(post_mass, 1e-20),
            "cross_component_share_of_post_median": bucket.get("cross_component", 0.0) / max(post_mass, 1e-20),
        }
    return table
