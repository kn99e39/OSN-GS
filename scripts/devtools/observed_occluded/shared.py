from __future__ import annotations

"""Worklog 120 -- SHARED INFRASTRUCTURE for the Observed/Occluded audit.

Directive section 4 draws a hard line: shared code may contain only query
representation, camera projection, relevant-view detection, deterministic
query-bank construction, the common global aggregation, metric computation and
serialization. It MUST NOT decide where the visibility boundary is, what counts
as a blocker, what counts as a surface hit, what T means, or what median depth
means. Every one of those five decisions lives in `candidate_a..d` and only
there.

Two functions in this module deserve explicit justification, because both touch
renderer outputs:

  * `reconstruct_direct_surfel_intersection_world_point` reconstructs, for a
    pixel's median event, the world point `center + s_u*scale_u*t_u +
    s_v*scale_v*t_v`. This is worklog 119's G2 geometry source, reproduced
    unchanged. It answers "WHERE is this renderer event in 3D", never "is
    anything observed or occluded". Candidate A uses it as its surface-event
    geometry; the query bank uses it to place R1 anchors (a bank definition the
    directive itself mandates in section 9B/R1); candidates B, C and D never
    call it.

  * `canonical_geometric_support_rho_max` turns the canonical forward kernel's
    own alpha cutoff into the surfel's exact geometric support extent. It is
    placed here only because two INDEPENDENT consumers need the identical
    canonical quantity for non-overlapping reasons -- candidate C as its
    blocker extent, and the query bank as the (untuned, evidence-derived) unit
    for its ray-ladder offsets. It is a property of a surfel and a constant of
    the canonical renderer; it decides nothing.

Nothing else in this file reads a renderer output at all.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

# --------------------------------------------------------------------------
# Reconstruction states. UNRESOLVED is a fail-closed implementation state, not
# a third reconstruction contribution (directive section 1). NON_RELEVANT is
# not a state at all -- it records that a view could not query x, so that view
# is excluded from the aggregation entirely (directive section 3).
# --------------------------------------------------------------------------
STATE_NON_RELEVANT = -1
STATE_UNRESOLVED = 0
STATE_OBSERVED = 1
STATE_OCCLUDED = 2

STATE_NAMES = {
    STATE_NON_RELEVANT: "NON_RELEVANT",
    STATE_UNRESOLVED: "UNRESOLVED",
    STATE_OBSERVED: "OBSERVED",
    STATE_OCCLUDED: "OCCLUDED",
}

# Relevance failure codes (directive section 3's required distinctions).
RELEVANCE_OK = 0
RELEVANCE_INVALID_PROJECTION = 1   # w <= 0: not in front of the camera at all
RELEVANCE_DEPTH_BELOW_NEAR = 2     # camera-space z below the renderer's own near plane
RELEVANCE_OUTSIDE_IMAGE = 3        # projects outside the usable image domain

RELEVANCE_NAMES = {
    RELEVANCE_OK: "RELEVANT",
    RELEVANCE_INVALID_PROJECTION: "NON_RELEVANT_INVALID_PROJECTION",
    RELEVANCE_DEPTH_BELOW_NEAR: "NON_RELEVANT_DEPTH_BELOW_NEAR",
    RELEVANCE_OUTSIDE_IMAGE: "NON_RELEVANT_OUTSIDE_IMAGE_DOMAIN",
}

# The canonical vendored kernel's own near plane, `near_n` in
# osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/auxiliary.h.
# Read from the canonical package (NOT from a diagnostic sibling) so this
# module can never drift from the production renderer. It is the renderer's own
# constant, not a threshold introduced by this batch;
# `TestCanonicalConstants` asserts the file still says so.
CANONICAL_NEAR_N = 0.2

# The canonical kernel's own minimum alpha, `if (alpha < 1.0f / 255.0f) continue;`
# in that same package's forward.cu. Used ONLY by
# `canonical_geometric_support_rho_max` below.
CANONICAL_MIN_ALPHA = 1.0 / 255.0

# The canonical kernel's alpha ceiling, `min(0.99f, ...)` in forward.cu. Needed
# because a surfel whose peak alpha is already clamped still has its support
# boundary set by the UNCLAMPED exponential falloff.
CANONICAL_MAX_ALPHA = 0.99


def canonical_constants_from_source(repo_root: Path) -> dict[str, float]:
    """Re-read the three canonical constants above out of the CANONICAL
    vendored package's source, so a fidelity test can assert this module has
    not drifted from the production renderer."""

    auxiliary = (repo_root / "osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/auxiliary.h").read_text(encoding="utf-8")
    forward = (repo_root / "osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/forward.cu").read_text(encoding="utf-8")
    near_line = [line for line in auxiliary.splitlines() if "near_n" in line and "const float" in line]
    return {
        "near_n": float(near_line[0].split("=")[1].strip().rstrip(";")),
        "min_alpha_source_present": "if (alpha < 1.0f / 255.0f)" in forward,
        "max_alpha_source_present": "min(0.99f, opa * exp(power))" in forward,
        "termination_source_present": "if (test_T < 0.0001f)" in forward,
    }


# --------------------------------------------------------------------------
# Query representation
# --------------------------------------------------------------------------

# Query kinds -- provenance labels only, never a policy branch.
KIND_R1_ANCHOR_RHO3D = "R1_OBSERVED_ANCHOR_RHO3D"
KIND_R1_ANCHOR_RHO2D = "R1_OBSERVED_ANCHOR_RHO2D"
KIND_R3_BEHIND = "R3_BEHIND_SURFACE_PROBE"
KIND_R4_FRONT = "R4_FRONT_OF_SURFACE_PROBE"
KIND_R5_REGION_GAP = "R5_REGION_GAP_PROBE"
KIND_R6_OUT_OF_FRUSTUM = "R6_OUT_OF_FRUSTUM_CONTROL"


@dataclass
class QueryBank:
    """One immutable, deterministically ordered set of 3D queries, shared by
    all four candidates verbatim (directive section 9)."""

    positions: torch.Tensor          # (N, 3) float32 world space
    kind: list[str]                  # (N,) provenance label
    source_view: np.ndarray          # (N,) int64 -- view the query was derived from, -1 if none
    source_surfel: np.ndarray        # (N,) int64 -- full-model surfel id it was derived from, -1 if none
    region: np.ndarray               # (N,) int64 -- region index, -1 if unlabelled
    ladder_step: np.ndarray          # (N,) float32 -- signed ray-ladder offset in units of the
                                     #      anchor surfel's own canonical support radius; 0 for anchors,
                                     #      NaN for queries that are not on a ladder
    support_radius: np.ndarray       # (N,) float32 -- the anchor surfel's canonical support radius (world units)

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def as_metadata(self, region_labels: Sequence[str]) -> dict[str, Any]:
        kinds, counts = np.unique(np.asarray(self.kind), return_counts=True)
        region_counts = {}
        for index, label in enumerate(region_labels):
            region_counts[label] = int((self.region == index).sum())
        region_counts["UNLABELLED"] = int((self.region < 0).sum())
        return {
            "query_count": len(self),
            "by_kind": {str(k): int(c) for k, c in zip(kinds, counts)},
            "by_region": region_counts,
            "ordering": "deterministic; construction order is fixed by view index then raster order, no RNG anywhere",
        }


@dataclass
class ViewGeometry:
    """Per (query, view) geometry. Pure projection + the relevant-view
    contract. Contains no visibility semantics whatsoever."""

    pixel_x: torch.Tensor       # (N,) float32 continuous pixel coordinate, rasterizer convention
    pixel_y: torch.Tensor
    pixel_col: torch.Tensor     # (N,) int64 nearest pixel column, -1 where non-relevant
    pixel_row: torch.Tensor     # (N,) int64 nearest pixel row, -1 where non-relevant
    pixel_index: torch.Tensor   # (N,) int64 row * W + col, -1 where non-relevant
    depth: torch.Tensor         # (N,) float32 camera-space z (the renderer's own depth convention)
    relevant: torch.Tensor      # (N,) bool
    relevance_code: torch.Tensor  # (N,) int8


def project_queries(camera: Any, positions: torch.Tensor) -> ViewGeometry:
    """Project world queries into one camera, in the RASTERIZER's own pixel
    convention.

    Conventions, all taken from the canonical vendored kernel and NOT invented
    here:
      * homogeneous row-vector convention: `p_clip = [x, 1] @ full_proj_transform`
        (graphdeco/`camera_matrices`);
      * `ndc2Pix(v, S) = ((v + 1) * S - 1) * 0.5` (auxiliary.h) -- note this is
        the rasterizer's own half-pixel convention, the one worklog 118/119
        showed differs from `depths_to_points`' `W/2` offsets;
      * depth is camera-space z, `p_view = [x, 1] @ world_view_transform`, which
        is exactly what the render loop's own `depth` variable holds and what
        `depths_to_points` unprojects.

    Relevance (directive section 3): a camera that cannot geometrically query x
    must not count as evidence that x is occluded. Three disjoint
    non-relevance causes are recorded separately, and NO "number of relevant
    views" threshold exists anywhere.
    """

    device = positions.device
    count = int(positions.shape[0])
    width, height = int(camera.image_width), int(camera.image_height)
    ones = torch.ones((count, 1), dtype=torch.float32, device=device)
    homogeneous = torch.cat([positions.to(torch.float32), ones], dim=1)

    view_space = homogeneous @ camera.world_view_transform
    depth = view_space[:, 2].contiguous()

    clip = homogeneous @ camera.full_proj_transform
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 0, w, torch.full_like(w, 1.0))
    ndc_x = clip[:, 0] / safe_w
    ndc_y = clip[:, 1] / safe_w
    pixel_x = ((ndc_x + 1.0) * width - 1.0) * 0.5
    pixel_y = ((ndc_y + 1.0) * height - 1.0) * 0.5

    col = torch.round(pixel_x).to(torch.int64)
    row = torch.round(pixel_y).to(torch.int64)

    code = torch.full((count,), RELEVANCE_OK, dtype=torch.int8, device=device)
    invalid_projection = w <= 0
    below_near = (~invalid_projection) & (depth < CANONICAL_NEAR_N)
    outside = (~invalid_projection) & (~below_near) & (
        (col < 0) | (col >= width) | (row < 0) | (row >= height)
    )
    code[invalid_projection] = RELEVANCE_INVALID_PROJECTION
    code[below_near] = RELEVANCE_DEPTH_BELOW_NEAR
    code[outside] = RELEVANCE_OUTSIDE_IMAGE
    relevant = code == RELEVANCE_OK

    col = torch.where(relevant, col, torch.full_like(col, -1))
    row = torch.where(relevant, row, torch.full_like(row, -1))
    index = torch.where(relevant, row * width + col, torch.full_like(col, -1))
    return ViewGeometry(
        pixel_x=pixel_x, pixel_y=pixel_y, pixel_col=col, pixel_row=row,
        pixel_index=index, depth=depth, relevant=relevant, relevance_code=code,
    )


# --------------------------------------------------------------------------
# Frozen global aggregation (directive section 2). No majority vote, no
# percentage vote, no minimum-view rule, no confidence weighting, no
# multiplicity threshold. A single qualified direct observation prevents
# GLOBAL_OCCLUDED.
# --------------------------------------------------------------------------

def aggregate_global(per_view_states: np.ndarray) -> np.ndarray:
    """`per_view_states` is (N, V) int8 using the STATE_* codes above, with
    STATE_NON_RELEVANT for views that could not query the point. Returns (N,)
    int8 global states."""

    states = np.asarray(per_view_states)
    observed_any = (states == STATE_OBSERVED).any(axis=1)
    relevant = states != STATE_NON_RELEVANT
    has_relevant = relevant.any(axis=1)
    occluded_or_irrelevant = (states == STATE_OCCLUDED) | (~relevant)
    all_relevant_occluded = has_relevant & occluded_or_irrelevant.all(axis=1)

    result = np.full(states.shape[0], STATE_UNRESOLVED, dtype=np.int8)
    result[all_relevant_occluded] = STATE_OCCLUDED
    result[observed_any] = STATE_OBSERVED  # applied last: OBSERVED always wins
    return result


# --------------------------------------------------------------------------
# Renderer geometry reconstruction (worklog 119's G2, reproduced unchanged).
# Answers WHERE, never WHETHER. See this module's docstring.
# --------------------------------------------------------------------------

def reconstruct_direct_surfel_intersection_world_point(
    representative_id: torch.Tensor,
    median_s_u: torch.Tensor,
    median_s_v: torch.Tensor,
    positions_full: torch.Tensor,
    tangent_u_full: torch.Tensor,
    tangent_v_full: torch.Tensor,
    scale_u_full: torch.Tensor,
    scale_v_full: torch.Tensor,
) -> torch.Tensor:
    """Worklog 119 G2: the exact ray/surfel-plane intersection the renderer's
    own median event was computed at, rebuilt in world space from the trained
    surfel's own frame and the kernel's own `s_u`/`s_v`.

    `compute_transmat`'s `splat2world` uses the world-space center and
    `L = R * S` directly (the camera transform is never applied to the local
    plane coordinates), so `world = center + s_u * scale_u * t_u + s_v *
    scale_v * t_v`. Verified against G1 to 6 decimals in worklog 119 section 4.

    Rows whose `representative_id` is -1 (no contributor crossed T=0.5) come
    back as NaN, so no caller can mistake a missing event for a location.
    """

    flat_ids = representative_id.reshape(-1).to(torch.int64)
    valid = flat_ids >= 0
    safe = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
    s_u = median_s_u.reshape(-1).to(torch.float32)
    s_v = median_s_v.reshape(-1).to(torch.float32)
    world = (
        positions_full[safe]
        + (s_u * scale_u_full[safe]).unsqueeze(1) * tangent_u_full[safe]
        + (s_v * scale_v_full[safe]).unsqueeze(1) * tangent_v_full[safe]
    )
    return torch.where(valid.unsqueeze(1), world, torch.full_like(world, float("nan")))


def canonical_geometric_support_rho_max(opacity: torch.Tensor) -> torch.Tensor:
    """The EXACT finite geometric support the canonical forward kernel already
    defines for every surfel, expressed as a maximum `rho3d`.

    Derivation, entirely from the canonical kernel's own two lines
    (`forward.cu`, unmodified):

        alpha = min(0.99f, opa * exp(-0.5f * rho));
        if (alpha < 1.0f / 255.0f) continue;

    so a primitive contributes at a ray-plane intersection iff

        opa * exp(-0.5 * rho) >= 1/255
        <=> rho <= 2 * ln(255 * opa)                      (and opa > 1/255)

    where `rho3d = s_u^2 + s_v^2` is the squared distance from the surfel
    centre in its own scaled tangent frame. The support is therefore an
    ellipse of semi-axes `sqrt(rho_max) * scale_u` and `sqrt(rho_max) *
    scale_v` -- a finite, exactly defined geometric region, with NO k-sigma
    choice, no tuned radius and no hand-selected blocker threshold introduced
    by this batch. Surfels with `opa <= 1/255` get `rho_max = 0`: the kernel
    itself never accepts them anywhere, so their geometric support is empty.

    The 0.99 alpha ceiling clamps the PEAK alpha only; it never moves the
    boundary, because at the boundary alpha equals 1/255, far below 0.99.
    """

    opa = opacity.reshape(-1).to(torch.float32)
    rho_max = 2.0 * torch.log(torch.clamp(opa, min=1e-30) / CANONICAL_MIN_ALPHA)
    return torch.clamp(rho_max, min=0.0)


# --------------------------------------------------------------------------
# Deterministic query-depth slot assignment for the Candidate D CUDA probe.
# Pure bookkeeping: which (query, pixel) pair occupies which probe slot. It
# carries no semantics -- the probe's meaning is decided in candidate_d.
# --------------------------------------------------------------------------

def assign_query_depth_slots(pixel_index: np.ndarray, max_slots: int) -> np.ndarray:
    """Rank each query within its own pixel, in ascending query-id order.
    Returns (N,) int64 ranks (-1 for non-relevant rows). Queries whose rank is
    >= `max_slots` are handled by additional render passes by the caller --
    never dropped."""

    ranks = np.full(pixel_index.shape[0], -1, dtype=np.int64)
    relevant = np.nonzero(pixel_index >= 0)[0]
    if relevant.size == 0:
        return ranks
    order = relevant[np.argsort(pixel_index[relevant], kind="stable")]
    sorted_pixels = pixel_index[order]
    group_start = np.concatenate([[True], sorted_pixels[1:] != sorted_pixels[:-1]])
    positions = np.arange(order.size)
    starts = np.maximum.accumulate(np.where(group_start, positions, 0))
    ranks[order] = positions - starts
    return ranks


# --------------------------------------------------------------------------
# Metrics / serialization
# --------------------------------------------------------------------------

def state_fractions(states: np.ndarray, *, include_non_relevant: bool = False) -> dict[str, Any]:
    """Explicit, never-hidden denominators (directive section 10F)."""

    states = np.asarray(states).reshape(-1)
    total = int(states.size)
    counts = {name: int((states == code).sum()) for code, name in STATE_NAMES.items()}
    if not include_non_relevant:
        counts.pop(STATE_NAMES[STATE_NON_RELEVANT], None)
        total = int((states != STATE_NON_RELEVANT).sum())
    out: dict[str, Any] = {"denominator": total, "counts": counts}
    out["fractions"] = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return out


def distribution(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    ordered = np.sort(values)

    def pct(fraction: float) -> float:
        position = min(ordered.size - 1, max(0, int(round(fraction * (ordered.size - 1)))))
        return float(ordered[position])

    return {"min": pct(0.0), "median": pct(0.5), "mean": float(ordered.mean()),
            "p95": pct(0.95), "max": pct(1.0), "count": int(ordered.size)}


def agreement(states_a: np.ndarray, states_b: np.ndarray) -> dict[str, Any]:
    """Pairwise candidate agreement over identical queries (directive 10H)."""

    a = np.asarray(states_a).reshape(-1)
    b = np.asarray(states_b).reshape(-1)
    total = int(a.size)
    same = int((a == b).sum())
    resolved_a = a != STATE_UNRESOLVED
    resolved_b = b != STATE_UNRESOLVED
    observed_occluded = int((((a == STATE_OBSERVED) & (b == STATE_OCCLUDED)) | ((a == STATE_OCCLUDED) & (b == STATE_OBSERVED))).sum())
    resolved_unresolved = int((resolved_a != resolved_b).sum())
    matrix = {}
    for code_a, name_a in STATE_NAMES.items():
        if code_a == STATE_NON_RELEVANT:
            continue
        for code_b, name_b in STATE_NAMES.items():
            if code_b == STATE_NON_RELEVANT:
                continue
            value = int(((a == code_a) & (b == code_b)).sum())
            if value:
                matrix[f"{name_a}->{name_b}"] = value
    return {
        "queries": total,
        "same_state": same,
        "same_state_fraction": same / total if total else 0.0,
        "observed_occluded_disagreement": observed_occluded,
        "resolved_unresolved_disagreement": resolved_unresolved,
        "confusion": matrix,
    }
