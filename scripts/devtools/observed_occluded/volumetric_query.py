from __future__ import annotations

"""Worklog 123 -- VOLUMETRIC QUERY CONTRACT, layered above the frozen candidate B.

Candidate B's decision function is NOT modified anywhere in this module. Every
per-view verdict still comes from `candidate_b_median_depth.classify_view`. What
is added is a QUERY REPRESENTATION layer:

    VolumetricQuery:
        world_position                      <- the CANONICAL representation
        optional renderer_event_provenance:
            camera_id
            pixel_id
            stored_median_depth
            representative_id

The provenance never changes the world position, carries no confidence, trust or
ownership semantics, and answers exactly ONE question:

    "is this query THIS EXACT renderer median event, in THIS view?"

It must not answer whether other views observe the point, whether the point is
globally observed, which component owns it, whether a surface continues, or
whether anything is trusted. Every other view uses the ordinary frozen per-view
frontier evaluation, and the frozen global aggregation is untouched.

No epsilon, no ULP band, no nextafter correction, no percentage threshold is
introduced. Provenance validity is decided by EXACT (bitwise) agreement between
the carried `stored_median_depth` and the renderer's own median depth at the
carried pixel in that view -- a correctness guard, not a tolerance: a query whose
carried provenance does not bitwise match the renderer's current output simply
falls back to the ordinary frozen comparison.

The float64 reference arm here is DIAGNOSTIC ONLY. It is never canonical, and it
exists to quantify instability, not to choose prettier arithmetic.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import candidate_b_median_depth as candidate_b
from .shared import (
    CANONICAL_NEAR_N,
    RELEVANCE_DEPTH_BELOW_NEAR,
    RELEVANCE_INVALID_PROJECTION,
    RELEVANCE_OK,
    RELEVANCE_OUTSIDE_IMAGE,
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    ViewGeometry,
    project_queries,
)

# Diagnostic marker for a per-view outcome that was settled by exact renderer
# event identity rather than by a world -> camera -> depth round-trip. It maps
# onto the existing STATE_OBSERVED for aggregation -- it is NOT a new
# reconstruction state and the frozen aggregation never sees it.
IDENTITY_NOT_APPLIED = 0
IDENTITY_ON_FRONTIER = 1
IDENTITY_REJECTED_STALE = 2
IDENTITY_NAMES = {
    IDENTITY_NOT_APPLIED: "NO_PROVENANCE_FOR_THIS_VIEW",
    IDENTITY_ON_FRONTIER: "ON_FRONTIER_BY_EVENT_IDENTITY",
    IDENTITY_REJECTED_STALE: "PROVENANCE_REJECTED_STORED_MEDIAN_MISMATCH",
}


@dataclass
class VolumetricQueryBank:
    """World space is the canonical volumetric representation. Provenance is
    optional side information attached to queries that were created directly
    from a renderer median event."""

    world_position: torch.Tensor           # (N, 3) float32 -- CANONICAL
    kind: list[str]
    provenance_camera: np.ndarray          # (N,) int64, -1 = none
    provenance_pixel: np.ndarray           # (N,) int64, -1 = none
    provenance_median_depth: np.ndarray    # (N,) float32, NaN = none
    provenance_representative: np.ndarray  # (N,) int64, -1 = none
    region: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.world_position.shape[0])

    def has_provenance(self) -> np.ndarray:
        return (self.provenance_camera >= 0) & (self.provenance_pixel >= 0)

    def without_provenance(self) -> "VolumetricQueryBank":
        """Q2 control: the SAME world coordinates with provenance removed."""

        count = len(self)
        return VolumetricQueryBank(
            world_position=self.world_position,
            kind=list(self.kind),
            provenance_camera=np.full(count, -1, dtype=np.int64),
            provenance_pixel=np.full(count, -1, dtype=np.int64),
            provenance_median_depth=np.full(count, np.nan, dtype=np.float32),
            provenance_representative=np.full(count, -1, dtype=np.int64),
            region=self.region,
        )

    def metadata(self) -> dict[str, Any]:
        kinds, counts = np.unique(np.asarray(self.kind), return_counts=True)
        return {
            "queries": len(self),
            "with_renderer_event_provenance": int(self.has_provenance().sum()),
            "by_kind": {str(k): int(v) for k, v in zip(kinds, counts)},
            "canonical_representation": "world_position (float32 xyz); provenance never replaces it",
        }


def apply_event_identity(
    view_index: int,
    bank: VolumetricQueryBank,
    geometry: ViewGeometry,
    median_flat: torch.Tensor,
    base_states: torch.Tensor,
) -> dict[str, Any]:
    """Layer the exact event-identity contract over the FROZEN candidate B result.

    `base_states` must already be `candidate_b.classify_view(...)["states"]` --
    this function never re-decides anything else and never touches other views.
    """

    device = base_states.device
    count = len(bank)
    identity = torch.full((count,), IDENTITY_NOT_APPLIED, dtype=torch.int8, device=device)

    rows = np.nonzero((bank.provenance_camera == view_index) & (bank.provenance_pixel >= 0))[0]
    if rows.size == 0:
        return {"states": base_states.clone(), "identity": identity, "applied": 0, "rejected": 0}

    row_index = torch.as_tensor(rows, dtype=torch.int64, device=device)
    pixel = torch.as_tensor(bank.provenance_pixel[rows], dtype=torch.int64, device=device)
    carried = torch.as_tensor(bank.provenance_median_depth[rows], dtype=torch.float32, device=device)
    renderer_median = median_flat[pixel]
    # EXACT bitwise agreement -- a correctness guard, never a tolerance.
    valid = (renderer_median == carried) & (renderer_median > 0)

    states = base_states.clone()
    applied_rows = row_index[valid]
    states[applied_rows] = STATE_OBSERVED
    identity[applied_rows] = IDENTITY_ON_FRONTIER
    identity[row_index[~valid]] = IDENTITY_REJECTED_STALE
    return {
        "states": states,
        "identity": identity,
        "applied": int(valid.sum().item()),
        "rejected": int((~valid).sum().item()),
    }


def project_queries_float64(camera: Any, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    """DIAGNOSTIC-ONLY higher-precision reference of `shared.project_queries`.

    Identical formulas and identical conventions -- the rasterizer's own
    `ndc2Pix(v, S) = ((v + 1) * S - 1) * 0.5` and camera-space z depth -- carried
    out in float64 from the same stored float32 inputs. This arm is NEVER
    canonical; it exists solely to attribute how much of the observed
    classification instability is float32 arithmetic in the query path.
    """

    device = positions.device
    count = int(positions.shape[0])
    width, height = int(camera.image_width), int(camera.image_height)
    homogeneous = torch.cat(
        [positions.to(torch.float64), torch.ones((count, 1), dtype=torch.float64, device=device)], dim=1
    )
    world_view = camera.world_view_transform.to(torch.float64)
    full_proj = camera.full_proj_transform.to(torch.float64)

    depth = (homogeneous @ world_view)[:, 2].contiguous()
    clip = homogeneous @ full_proj
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 0, w, torch.ones_like(w))
    pixel_x = ((clip[:, 0] / safe_w + 1.0) * width - 1.0) * 0.5
    pixel_y = ((clip[:, 1] / safe_w + 1.0) * height - 1.0) * 0.5
    col = torch.round(pixel_x).to(torch.int64)
    row = torch.round(pixel_y).to(torch.int64)

    code = torch.full((count,), RELEVANCE_OK, dtype=torch.int8, device=device)
    invalid = w <= 0
    below_near = (~invalid) & (depth < CANONICAL_NEAR_N)
    outside = (~invalid) & (~below_near) & ((col < 0) | (col >= width) | (row < 0) | (row >= height))
    code[invalid] = RELEVANCE_INVALID_PROJECTION
    code[below_near] = RELEVANCE_DEPTH_BELOW_NEAR
    code[outside] = RELEVANCE_OUTSIDE_IMAGE
    relevant = code == RELEVANCE_OK
    col = torch.where(relevant, col, torch.full_like(col, -1))
    row = torch.where(relevant, row, torch.full_like(row, -1))
    return {
        "pixel_row": row, "pixel_col": col,
        "pixel_index": torch.where(relevant, row * width + col, torch.full_like(col, -1)),
        "depth": depth, "relevant": relevant, "relevance_code": code,
    }


def reference_side(reference: dict[str, torch.Tensor], median_flat: torch.Tensor) -> torch.Tensor:
    """Candidate B's OWN rule (`query_depth <= median_depth` OBSERVED, `>`
    OCCLUDED, `median <= 0` UNRESOLVED) evaluated on the float64 reference.

    This is a diagnostic recomputation, not a second classifier: the test suite
    asserts it reproduces `candidate_b.classify_view` exactly when handed the
    float32 arm's own inputs.
    """

    device = reference["depth"].device
    count = int(reference["depth"].shape[0])
    states = torch.full((count,), STATE_NON_RELEVANT, dtype=torch.int8, device=device)
    relevant = reference["relevant"]
    if not bool(relevant.any()):
        return states
    median = median_flat.to(torch.float64)[reference["pixel_index"].clamp(min=0)]
    valid = relevant & (median > 0.0)
    states = torch.where(relevant, torch.full_like(states, STATE_UNRESOLVED), states)
    states = torch.where(valid & (reference["depth"] > median), torch.full_like(states, STATE_OCCLUDED), states)
    states = torch.where(valid & (reference["depth"] <= median), torch.full_like(states, STATE_OBSERVED), states)
    return states


@dataclass
class StabilityAccumulator:
    """float32-vs-reference agreement over (query, view) pairs, streamed."""

    label: str = ""
    pairs: int = 0
    relevant_pairs: int = 0
    state_agree: int = 0
    observed_occluded_disagree: int = 0
    resolved_unresolved_disagree: int = 0
    pixel_agree: int = 0
    relevance_agree: int = 0
    disagreement_margins: list[float] = field(default_factory=list)
    disagreement_relative_margins: list[float] = field(default_factory=list)
    disagreement_float32_margins: list[float] = field(default_factory=list)
    agreement_margins_sampled: list[float] = field(default_factory=list)
    disagreement_ulp: dict[int, int] = field(default_factory=dict)
    reference_depth_ulp: dict[int, int] = field(default_factory=dict)
    pixel_disagreements: int = 0
    per_kind_pairs: dict[str, int] = field(default_factory=dict)
    per_kind_disagree: dict[str, int] = field(default_factory=dict)
    max_disagreement_abs_margin: float = 0.0
    max_disagreement_relative_margin: float = 0.0

    def accumulate(
        self,
        float32_states: torch.Tensor,
        reference_states: torch.Tensor,
        geometry: ViewGeometry,
        reference: dict[str, torch.Tensor],
        median_flat: torch.Tensor,
        kinds: np.ndarray,
        margin_sample_stride: int = 0,
    ) -> None:
        relevant = geometry.relevant & reference["relevant"]
        relevant_np = relevant.detach().cpu().numpy()
        a = float32_states.detach().cpu().numpy()
        b = reference_states.detach().cpu().numpy()
        self.pairs += int(a.size)
        self.relevant_pairs += int(relevant_np.sum())
        agree = a == b
        self.state_agree += int((agree & relevant_np).sum())
        both_resolved = np.isin(a, (STATE_OBSERVED, STATE_OCCLUDED)) & np.isin(b, (STATE_OBSERVED, STATE_OCCLUDED))
        self.observed_occluded_disagree += int((relevant_np & both_resolved & ~agree).sum())
        self.resolved_unresolved_disagree += int(
            (relevant_np & ~both_resolved & ~agree).sum()
        )
        float32_row = geometry.pixel_row.detach().cpu().numpy()
        float32_col = geometry.pixel_col.detach().cpu().numpy()
        reference_row = reference["pixel_row"].detach().cpu().numpy()
        reference_col = reference["pixel_col"].detach().cpu().numpy()
        pixel_same = (
            relevant_np
            & (float32_row == reference_row)
            & (float32_col == reference_col)
        )
        self.pixel_agree += int(pixel_same.sum())
        self.pixel_disagreements += int((relevant_np & ~pixel_same).sum())
        self.relevance_agree += int(
            (geometry.relevance_code.detach().cpu().numpy() == reference["relevance_code"].detach().cpu().numpy()).sum()
        )

        float32_median = median_flat[geometry.pixel_index.clamp(min=0)].detach().cpu().numpy().astype(np.float64)
        reference_median = median_flat[reference["pixel_index"].clamp(min=0)].detach().cpu().numpy().astype(np.float64)
        float32_depth_all = geometry.depth.detach().cpu().numpy().astype(np.float64)
        reference_depth = reference["depth"].detach().cpu().numpy().astype(np.float64)
        float32_margin = float32_depth_all - float32_median
        reference_margin = reference_depth - reference_median
        disagree_rows = np.nonzero(relevant_np & ~agree)[0]
        if disagree_rows.size:
            values = reference_margin[disagree_rows]
            float32_values = float32_margin[disagree_rows]
            relative = values / np.maximum(np.abs(reference_median[disagree_rows]), 1e-30)
            self.disagreement_margins.extend(values.tolist())
            self.disagreement_float32_margins.extend(float32_values.tolist())
            self.disagreement_relative_margins.extend(relative.tolist())
            self.max_disagreement_abs_margin = max(self.max_disagreement_abs_margin, float(np.abs(values).max()))
            self.max_disagreement_relative_margin = max(
                self.max_disagreement_relative_margin, float(np.abs(relative).max())
            )
            float32_depth = float32_depth_all[disagree_rows]
            median32 = float32_median[disagree_rows].astype(np.float32)
            ulp = np.abs(
                float32_depth.astype(np.float32).view(np.int32).astype(np.int64)
                - median32.view(np.int32).astype(np.int64)
            )
            for key, value in zip(*np.unique(np.minimum(ulp, 64), return_counts=True)):
                self.disagreement_ulp[int(key)] = self.disagreement_ulp.get(int(key), 0) + int(value)
            reference32 = reference_depth[disagree_rows].astype(np.float32)
            reference_ulp = np.abs(
                float32_depth.astype(np.float32).view(np.int32).astype(np.int64)
                - reference32.view(np.int32).astype(np.int64)
            )
            for key, value in zip(*np.unique(np.minimum(reference_ulp, 64), return_counts=True)):
                self.reference_depth_ulp[int(key)] = self.reference_depth_ulp.get(int(key), 0) + int(value)
            for kind in np.unique(kinds[disagree_rows]):
                self.per_kind_disagree[str(kind)] = self.per_kind_disagree.get(str(kind), 0) + int(
                    (kinds[disagree_rows] == kind).sum()
                )
        for kind in np.unique(kinds):
            self.per_kind_pairs[str(kind)] = self.per_kind_pairs.get(str(kind), 0) + int(
                (relevant_np & (kinds == kind)).sum()
            )
        if margin_sample_stride > 0 and len(self.agreement_margins_sampled) < 400000:
            sampled = reference_margin[relevant_np][::margin_sample_stride]
            self.agreement_margins_sampled.extend(sampled.tolist())

    def summary(self) -> dict[str, Any]:
        denominator = max(self.relevant_pairs, 1)
        margins = np.asarray(self.disagreement_margins, dtype=np.float64)
        float32_margins = np.asarray(self.disagreement_float32_margins, dtype=np.float64)
        relative = np.asarray(self.disagreement_relative_margins, dtype=np.float64)

        def _distribution(values: np.ndarray) -> dict[str, Any]:
            if values.size == 0:
                return {"count": 0}
            absolute = np.abs(values)
            return {
                "count": int(values.size),
                "signed_min": float(values.min()), "signed_max": float(values.max()),
                "abs_min": float(absolute.min()), "abs_median": float(np.median(absolute)),
                "abs_p95": float(np.quantile(absolute, 0.95)), "abs_max": float(absolute.max()),
            }

        return {
            "label": self.label,
            "query_view_pairs": self.pairs,
            "relevant_pairs": self.relevant_pairs,
            "float32_vs_reference_state_agreement": self.state_agree,
            "float32_vs_reference_state_disagreement": self.relevant_pairs - self.state_agree,
            "state_agreement_rate": self.state_agree / denominator,
            "OBSERVED_OCCLUDED_disagreement": self.observed_occluded_disagree,
            "OBSERVED_OCCLUDED_disagreement_rate": self.observed_occluded_disagree / denominator,
            "resolved_unresolved_disagreement": self.resolved_unresolved_disagree,
            "projected_pixel_agreement": self.pixel_agree,
            "projected_pixel_agreement_rate": self.pixel_agree / denominator,
            "projected_pixel_disagreement": self.pixel_disagreements,
            "relevance_code_agreement": self.relevance_agree,
            "disagreement_signed_margin_from_reference_frontier": _distribution(margins),
            "disagreement_signed_margin_from_float32_frontier": _distribution(float32_margins),
            "disagreement_relative_margin_from_reference_frontier": _distribution(relative),
            "disagreement_float32_ulp_histogram": {
                str(k): v for k, v in sorted(self.disagreement_ulp.items())
            },
            "disagreement_float32_vs_reference_depth_ulp_histogram": {
                str(k): v for k, v in sorted(self.reference_depth_ulp.items())
            },
            "max_disagreement_abs_margin": self.max_disagreement_abs_margin,
            "max_disagreement_relative_margin": self.max_disagreement_relative_margin,
            "per_kind_relevant_pairs": dict(self.per_kind_pairs),
            "per_kind_disagreements": dict(self.per_kind_disagree),
        }


@dataclass
class EventIdentityAccumulator:
    """Exhaustive audit of section 4: does the provenance contract preserve exact
    ON_FRONTIER identity for every renderer median event?"""

    total_events: int = 0
    historical_float32_observed: int = 0
    historical_float32_contradiction: int = 0
    provenance_observed: int = 0
    provenance_contradiction: int = 0
    provenance_applied: int = 0
    provenance_rejected_stale: int = 0
    reference_observed: int = 0
    reference_contradiction: int = 0

    def summary(self) -> dict[str, Any]:
        denominator = max(self.total_events, 1)
        return {
            "total_source_median_events": self.total_events,
            "historical_float32_source_OBSERVED": self.historical_float32_observed,
            "historical_float32_source_contradiction": self.historical_float32_contradiction,
            "historical_float32_contradiction_rate": self.historical_float32_contradiction / denominator,
            "provenance_preserved_OBSERVED": self.provenance_observed,
            "provenance_preserved_contradiction": self.provenance_contradiction,
            "provenance_preserved_contradiction_rate": self.provenance_contradiction / denominator,
            "provenance_applied": self.provenance_applied,
            "provenance_rejected_stored_median_mismatch": self.provenance_rejected_stale,
            "float64_reference_source_OBSERVED": self.reference_observed,
            "float64_reference_source_contradiction": self.reference_contradiction,
            "float64_reference_contradiction_rate": self.reference_contradiction / denominator,
        }
