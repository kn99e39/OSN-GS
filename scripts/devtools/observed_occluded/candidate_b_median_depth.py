from __future__ import annotations

"""Worklog 120 -- CANDIDATE B ONLY. Median-depth partition.

HYPOTHESIS B: the canonical renderer's median surface depth can serve directly
as a view-local observed/occluded dividing surface.

Per-view semantics implemented here and nowhere else:

    OBSERVED    query_depth <= median_depth
    OCCLUDED    query_depth >  median_depth
    UNRESOLVED  no valid median event at the projected pixel

The primitive: `out_others[MIDDEPTH_OFFSET]`, the canonical vendored kernel's
own `median_depth` -- RENDERER-NATIVE, taken verbatim from the canonical
forward pass. The 0.5 crossing rule is NOT modified, reinterpreted, or replaced;
this module never touches `T`.

Validity uses the renderer's OWN uninitialized sentinel rather than borrowing
Candidate A's `representative_id`: `median_depth` is declared `float
median_depth = {0}` and is only ever assigned inside `if (T > 0.5)` at a
contributor that already passed `depth >= near_n` (= 0.2). A pixel where no
contributor crossed T = 0.5 therefore reports exactly 0.0, and a real event can
never report <= 0. `median_depth > 0` is that sentinel test and introduces no
threshold. (It agrees with `representative_id >= 0` by construction -- both are
written at the identical site -- but B does not depend on that field existing.)

No claim is made anywhere in this module, or in the report, that median depth is
a physical first hit. It is the depth at which half the ray's transmittance has
been consumed, which is what the renderer means by it and nothing more. If that
turns out not to be a first-surface visibility boundary, that failure is the
measurement, not a defect to patch.
"""

from typing import Any

import torch

from .shared import (
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    ViewGeometry,
)

NAME = "B"
TITLE = "MEDIAN-DEPTH PARTITION"

# Offset of the median ("mid") depth channel inside the canonical kernel's
# `out_others` buffer -- MIDDEPTH_OFFSET in
# osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/forward.cu.
MIDDEPTH_OFFSET = 5


def median_depth_map(out_others: torch.Tensor) -> torch.Tensor:
    """The canonical median-depth channel, (H, W)."""

    return out_others[MIDDEPTH_OFFSET]


def classify_view(geometry: ViewGeometry, median_depth_flat: torch.Tensor) -> dict[str, Any]:
    """Classify every query against ONE view. `median_depth_flat` is (H*W,)."""

    count = int(geometry.depth.shape[0])
    device = geometry.depth.device
    states = torch.full((count,), STATE_NON_RELEVANT, dtype=torch.int8, device=device)
    median_at_query = torch.full((count,), float("nan"), dtype=torch.float32, device=device)

    relevant = geometry.relevant
    if not bool(relevant.any()):
        return {"states": states, "median_depth": median_at_query}

    index = geometry.pixel_index.clamp(min=0)
    median = median_depth_flat[index]
    valid = relevant & (median > 0.0)

    states = torch.where(relevant, torch.full_like(states, STATE_UNRESOLVED), states)
    observed = valid & (geometry.depth <= median)
    occluded = valid & (geometry.depth > median)
    states = torch.where(occluded, torch.full_like(states, STATE_OCCLUDED), states)
    states = torch.where(observed, torch.full_like(states, STATE_OBSERVED), states)

    median_at_query = torch.where(valid, median, median_at_query)
    return {"states": states, "median_depth": median_at_query}
