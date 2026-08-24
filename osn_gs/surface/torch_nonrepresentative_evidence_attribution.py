from __future__ import annotations

"""Worklog 110 -- Non-Representative Renderer Evidence: Role Attribution.

Worklog 109 froze the Worklog 107 camera-induced representative topology as
canonical (`torch_camera_induced_visible_adjacency.py`, GATE PASS) and proved
that `FORWARD_ACCEPTED_CONTRIBUTOR != VISIBLE SURFACE SUPPORT` -- a surfel can
pass every forward acceptance check through residual transmittance while
lying behind the pixel's median representative and never becoming a
representative itself. This module ATTRIBUTES the role of the 395,676
real-scene accepted-non-representative surfels (Worklog 109's cross-tab)
without attaching any of them to any component -- attachment is explicitly
out of scope for this batch.

Three concepts stay separate throughout this module (directive's Central
Intent): RENDERER CONTRIBUTION (forward acceptance), SURFACE REPRESENTATION
(median crossing), and VISIBLE TOPOLOGY (the WL107/109 canonical component
graph, read-only input here, never modified).

Traversal semantics (exact, from `torch_surfel_representative_diagnostics.py`
worklog 110's `contrib_ids`/`contrib_post_median`, itself captured directly
in `forward.cu`'s existing per-pixel compositing loop, same forward pass as
`median_surfel_id`):

    at each accepted contributor event, `T` (transmittance BEFORE this
    contributor's own alpha update) is read at the exact point the kernel's
    own `if (T > 0.5)` median-crossing check reads it:

        T > 0.5  -> this event occurred at-or-before the pixel's median
                    crossing ("pre-or-at-median"; `contrib_post_median=0`)
        T <= 0.5 -> this event occurred strictly after some earlier
                    contributor at this pixel already crossed T=0.5
                    ("post-median"; `contrib_post_median=1`)

For a surfel that is a MEDIAN_SURFACE_REPRESENTATIVE at some pixel, its own
representative event is itself always `contrib_post_median=0` there (being
the median crossing requires T>0.5 immediately before it, by construction of
`median_surfel_id`/`median_contributor`). For a surfel that is NEVER a
representative ANYWHERE in the dataset (the population this batch studies:
accepted-non-representative surfels, `ever_representative == False` across
all views), "pre-or-at-median" reduces unambiguously to strictly PRE_MEDIAN
-- since by definition of population membership this surfel is never itself
the median crossing at any pixel, an event with `contrib_post_median=0`
there necessarily means some LATER contributor at that same pixel became the
median. This lets pre/post classification be computed directly from the
existing per-event flag with no additional CUDA logic, for exactly the
population this batch is required to attribute.

Contributor<->representative co-support (directive section 5/7): for each
accepted contributor event at a pixel, the SAME pixel's `representative_id`
(already exposed) is the co-occurring median representative -- "same
rendered pixel / observation" per the directive's own definition. This
module never materializes a (pixel x surfel) matrix; it derives, per view,
the set of DISTINCT (contributor, representative-component) pairs directly
from the bounded (H, W, K) `contrib_ids` slot array (K =
`OSN_GS_MAX_CONTRIB_SLOTS`, see `diff_surfel_rasterization_diag/cuda_
rasterizer/config.h`), then accumulates distinct pairs across views. Because
K is a per-pixel CAP (truncation is always visible via `contrib_count`, see
that module), this is an honestly-bounded sparse/streamed representation,
not a full contributor<->representative enumeration -- documented as a
limitation, not silently presented as exhaustive.
"""

from typing import Any, Callable

_EPS = 1e-9

SUPPORTS_ONE_REPRESENTATIVE_COMPONENT = "SUPPORTS_ONE_REPRESENTATIVE_COMPONENT"
SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS = "SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS"
ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION = "ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION"
COMPONENT_RELATION_CATEGORIES = (
    ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION,
    SUPPORTS_ONE_REPRESENTATIVE_COMPONENT,
    SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS,
)


def classify_pre_post_median(contrib_ids: Any, contrib_post_median: Any, count: int) -> tuple[Any, Any]:
    """From one view's bounded (..., K) `contrib_ids`/`contrib_post_median`
    slot arrays (any leading shape, e.g. (H, W, K)), returns per-primitive
    `(ever_pre_or_at_median, ever_post_median)` bool tensors of shape
    `(count,)`, aggregated (OR) over every valid slot. `-1` entries in
    `contrib_ids` (unused slots) are ignored. Pure tensor op, no CUDA
    needed -- directly testable on small fixture tensors."""

    torch = _torch()
    device = contrib_ids.device
    valid = contrib_ids >= 0
    ever_pre = torch.zeros((count,), dtype=torch.bool, device=device)
    ever_post = torch.zeros((count,), dtype=torch.bool, device=device)
    if not bool(valid.any()):
        return ever_pre, ever_post

    flat_ids = contrib_ids[valid].reshape(-1).to(torch.int64)
    flat_post = contrib_post_median[valid].reshape(-1).to(torch.bool)

    pre_ids = flat_ids[~flat_post]
    post_ids = flat_ids[flat_post]
    if int(pre_ids.numel()) > 0:
        ever_pre[pre_ids] = True
    if int(post_ids.numel()) > 0:
        ever_post[post_ids] = True
    return ever_pre, ever_post


def view_contributor_component_pairs(contrib_ids: Any, representative_id: Any, subset_ids: Any) -> Any:
    """From one view's `contrib_ids` (H, W, K) and `representative_id`
    (H, W) (both already remapped to the same node-index space as
    `subset_ids`, `-1` = none), returns the DISTINCT `(contributor,
    component)` pairs observed in this view -- "same rendered pixel"
    co-occurrence between an accepted contributor and that pixel's median
    representative's canonical WL107/109 component id. `(N, 2)` int64,
    deduplicated within this view. A contributor that is itself sometimes
    the representative can still appear paired with its OWN component at
    pixels where it is the representative -- callers restricting to the
    accepted-non-representative population filter that out afterward via
    `ever_representative`, not here (this function has no attachment
    semantics, it only reports co-occurrence)."""

    torch = _torch()
    device = contrib_ids.device
    k = int(contrib_ids.shape[-1])
    rep_broadcast = representative_id.unsqueeze(-1).expand(*representative_id.shape, k)
    valid = (contrib_ids >= 0) & (rep_broadcast >= 0)
    if not bool(valid.any()):
        return torch.zeros((0, 2), dtype=torch.int64, device=device)

    contributors = contrib_ids[valid].reshape(-1).to(torch.int64)
    reps = rep_broadcast[valid].reshape(-1).to(torch.int64)
    components = subset_ids[reps]
    pairs = torch.stack([contributors, components], dim=1)
    return torch.unique(pairs, dim=0)


def finalize_component_co_support(pair_batches: list[Any], count: int, subset_count: int) -> dict[str, Any]:
    """Concatenates per-view `(contributor, component)` pair tensors from
    `view_contributor_component_pairs`, deduplicates globally, and returns
    per-primitive distinct-component counts plus the deduplicated pairs
    (for provenance). Encodes `key = contributor * subset_count +
    component` (int64, safe up to ~9.2e18 -- comfortably covers real-scene
    contributor/subset counts in the low millions)."""

    torch = _torch()
    device = pair_batches[0].device if pair_batches else "cpu"
    nonempty = [p for p in pair_batches if int(p.shape[0]) > 0]
    if not nonempty:
        return {
            "distinct_component_count": torch.zeros((count,), dtype=torch.int64, device=device),
            "unique_pairs": torch.zeros((0, 2), dtype=torch.int64, device=device),
        }

    combined = torch.cat(nonempty, dim=0)
    keys = combined[:, 0] * subset_count + combined[:, 1]
    unique_keys = torch.unique(keys)
    contributors = unique_keys // subset_count
    components = unique_keys % subset_count
    unique_pairs = torch.stack([contributors, components], dim=1)

    distinct_component_count = torch.bincount(contributors, minlength=count)[:count]
    return {"distinct_component_count": distinct_component_count, "unique_pairs": unique_pairs}


def component_relation_category(distinct_component_count: Any, accepted_non_representative_mask: Any) -> Any:
    """Per-primitive category index into `COMPONENT_RELATION_CATEGORIES`,
    defined only where `accepted_non_representative_mask` is True (elsewhere
    the category is meaningless -- callers must mask separately). Diagnostic
    label only -- no attachment."""

    torch = _torch()
    count = int(distinct_component_count.shape[0])
    category = torch.full((count,), COMPONENT_RELATION_CATEGORIES.index(ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION), dtype=torch.int64, device=distinct_component_count.device)
    category = torch.where(distinct_component_count == 1, torch.full_like(category, COMPONENT_RELATION_CATEGORIES.index(SUPPORTS_ONE_REPRESENTATIVE_COMPONENT)), category)
    category = torch.where(distinct_component_count >= 2, torch.full_like(category, COMPONENT_RELATION_CATEGORIES.index(SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS)), category)
    return category


def _torch():
    from osn_gs.utils.torch_ops import require_torch
    return require_torch()
