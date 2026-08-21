from __future__ import annotations

"""Worklog 104 -- Node-Level Observability Accounting.

Worklog 103 (`torch_positive_visible_adjacency.py`, left UNMODIFIED here and
reused only via its already-public functions/imports) measures visible
topology at PAIRWISE 3D-candidate-edge granularity: a relation between two
surfels is only ever POSITIVE, CUT, CONFLICTING, or UNKNOWN. It never asks
the separate question "was this surfel ITSELF ever positively observed as
visible, independent of any particular neighbor?" -- that is what this module
adds, without touching Worklog 103's own graph-construction contract.

Two things this module explicitly does NOT do:
  - It does not change what counts as a positive/cut/conflicting/unknown
    EDGE relation (all of that stays exactly Worklog 103's own
    `compute_positive_visible_adjacency_evidence` output, called unmodified).
  - It does not invent a new "middle probability" between UNKNOWN and
    VISIBLE at the node level either -- node-level observability here is
    purely a COUNT of how many training views classify the surfel's own
    CENTER as `on_observed_surface` (canonical Phase-C, reused via
    `_per_view_status_codes`/`_project_to_camera`, both imported unmodified
    from Worklog 102), plus (where available) the renderer-native
    `radii > 0` / `visibility_filter` projection signal.

Renderer-native visibility signal -- what is and is not available (directive
section 3). `OSNSurfelRasterizer.render()`'s own docstring already documents
this precisely: the per-pixel per-surfel alpha-compositing weights
`omega_i = alpha_i * T_i` (paper eqs. 12-14) exist only inside the vendored
CUDA kernel and are never returned to Python -- there is no per-surfel
"did this surfel actually win a rendered pixel" signal anywhere in the
existing API, and materializing one would mean editing the vendored kernel
(forfeiting the OFFICIAL_CODE_FAITHFUL claim), which this module does not
do. What the kernel DOES already return is `radii` (per-surfel screen-space
projection radius; `radii > 0` is `visibility_mask` / `visibility_filter`).
This is a PROJECTION/CULLING signal -- it says the surfel's covariance
projected to a non-degenerate, in-frustum screen footprint -- NOT an
occlusion-aware contribution signal: a surfel fully hidden behind 50 nearer
surfels at every pixel its footprint touches can still have `radii > 0`.
This module therefore reports `radii > 0` counts as a SEPARATE, weaker
diagnostic, explicitly never substituted for genuine visible contribution.
"""

from dataclasses import dataclass
from typing import Any, Callable

from osn_gs.surface.torch_maximal_visible_connectivity import _per_view_status_codes, _project_to_camera
from osn_gs.surface.torch_observation_evidence import ObservationEvidence
from osn_gs.utils.torch_ops import require_torch

# --- node-level observability categories (directive section 2) ---
CATEGORY_A_NEVER_POSITIVELY_OBSERVED = "NEVER_POSITIVELY_OBSERVED_AT_NODE_LEVEL"
CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE = "OBSERVED_AT_NODE_LEVEL_NO_POSITIVE_EDGE"
CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE = "OBSERVED_AT_NODE_LEVEL_WITH_POSITIVE_EDGE"
CATEGORY_D_OBSERVED_CONFLICT_ONLY = "OBSERVED_AT_NODE_LEVEL_CONFLICT_ONLY"
NODE_OBSERVABILITY_CATEGORIES = (
    CATEGORY_A_NEVER_POSITIVELY_OBSERVED,
    CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE,
    CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE,
    CATEGORY_D_OBSERVED_CONFLICT_ONLY,
)

# --- Worklog-103-singleton failure-mode categories (directive section 5) ---
SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE = "NODE_NEVER_POSITIVELY_VISIBLE"
SINGLETON_NODE_VISIBLE_NO_COOBSERVED_EDGE = "NODE_VISIBLE_BUT_NO_COOBSERVED_CANDIDATE_EDGE"
SINGLETON_COOBSERVED_CORRIDOR_FAILS = "COOBSERVED_EDGE_EXISTS_BUT_CORRIDOR_POSITIVE_TEST_FAILS"
SINGLETON_POSITIVE_BUT_GEOMETRIC_CUT = "POSITIVE_OBSERVATION_EXISTS_BUT_GEOMETRIC_GATE_CUTS"
SINGLETON_OBSERVATION_CONFLICT = "OBSERVATION_CONFLICT"
SINGLETON_OTHER = "OTHER_EXPLICITLY_REPORTED_REASON"
SINGLETON_CAUSE_CATEGORIES = (
    SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE,
    SINGLETON_NODE_VISIBLE_NO_COOBSERVED_EDGE,
    SINGLETON_COOBSERVED_CORRIDOR_FAILS,
    SINGLETON_POSITIVE_BUT_GEOMETRIC_CUT,
    SINGLETON_OBSERVATION_CONFLICT,
    SINGLETON_OTHER,
)


@dataclass(frozen=True)
class NodeViewObservability:
    """Per-surfel, per-training-set counts -- independent of any candidate
    graph or edge. `count` matches `positions.shape[0]` exactly."""

    on_observed_surface_view_count: Any  # (N,) int32 -- canonical Phase-C center classification
    in_bounds_view_count: Any  # (N,) int32 -- projects inside SOME camera's frustum, valid or not
    projectable_view_count: Any | None  # (N,) int32 or None -- renderer-native radii>0 count, if radii were supplied
    total_views: int


def compute_node_view_observability(
    positions: Any,
    observation_evidence: ObservationEvidence,
    radii_per_view: dict[int, Any] | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> NodeViewObservability:
    """Classify every surfel's own CENTER against every training view using
    the exact same canonical per-view rule Worklog 102/103 use for edges
    (`_per_view_status_codes`) -- but accumulated per NODE, never per edge,
    and with no candidate graph involved at all.

    `radii_per_view`, if given, maps `view.camera_index -> radii tensor
    (N,)` from that camera's own `OSNSurfelRasterizer.render()` call (the
    SAME render already performed to build `observation_evidence` -- no
    extra render pass). See the module docstring for exactly what `radii>0`
    does and does not prove.
    """

    torch = require_torch()
    count = int(positions.shape[0])
    device = positions.device
    on_observed = torch.zeros((count,), dtype=torch.int32, device=device)
    in_bounds_count = torch.zeros((count,), dtype=torch.int32, device=device)
    projectable = None
    if radii_per_view is not None:
        projectable = torch.zeros((count,), dtype=torch.int32, device=device)

    for view in observation_evidence.views:
        proj = _project_to_camera(positions, view)
        status = _per_view_status_codes(
            proj["view_depth"], proj["observed_depth"], proj["valid_at_pixel"], proj["in_bounds"],
            observation_evidence.near, observation_evidence.far, observation_evidence.depth_epsilon,
        )
        on_observed += (status == 3).to(torch.int32)
        in_bounds_count += proj["in_bounds"].to(torch.int32)
        if radii_per_view is not None and view.camera_index in radii_per_view:
            radii = radii_per_view[view.camera_index].to(device=device)
            projectable += (radii > 0).to(torch.int32)
        if progress is not None and view.camera_index % 20 == 0:
            progress(f"node-level observability: view {view.camera_index}")

    return NodeViewObservability(
        on_observed_surface_view_count=on_observed, in_bounds_view_count=in_bounds_count,
        projectable_view_count=projectable, total_views=len(observation_evidence.views),
    )


def classify_node_observability(
    node_view_observability: NodeViewObservability, node_has_positive_edge: Any, node_has_conflict_edge: Any
) -> Any:
    """Directive section 2's A/B/C/D partition, as an int8 category index
    into `NODE_OBSERVABILITY_CATEGORIES`. Mutually exclusive by construction
    (checked in priority order: A first, then C, then D, else B)."""

    torch = require_torch()
    on_observed = node_view_observability.on_observed_surface_view_count
    category = torch.full_like(on_observed, NODE_OBSERVABILITY_CATEGORIES.index(CATEGORY_B_OBSERVED_NO_POSITIVE_EDGE))
    never_observed = on_observed == 0
    category = torch.where(
        never_observed, torch.full_like(category, NODE_OBSERVABILITY_CATEGORIES.index(CATEGORY_A_NEVER_POSITIVELY_OBSERVED)), category
    )
    has_positive = (~never_observed) & node_has_positive_edge
    category = torch.where(
        has_positive, torch.full_like(category, NODE_OBSERVABILITY_CATEGORIES.index(CATEGORY_C_OBSERVED_WITH_POSITIVE_EDGE)), category
    )
    conflict_only = (~never_observed) & (~node_has_positive_edge) & node_has_conflict_edge
    category = torch.where(
        conflict_only, torch.full_like(category, NODE_OBSERVABILITY_CATEGORIES.index(CATEGORY_D_OBSERVED_CONFLICT_ONLY)), category
    )
    return category


def classify_singleton_causes(
    singleton_mask: Any,
    node_view_observability: NodeViewObservability,
    node_ever_evaluated: Any,
    node_any_positive_pre_geometry: Any,
    node_geometric_cut: Any,
    node_conflict: Any,
) -> Any:
    """Directive section 5's 6-way exclusive cause attribution for Worklog
    103 singleton surfels, as an int8 category index into
    `SINGLETON_CAUSE_CATEGORIES`. All five `node_*` boolean tensors are
    (N,), True if ANY spatial candidate edge incident on that node has the
    corresponding property (derived from Worklog 103's own unmodified
    `compute_positive_visible_adjacency_evidence` output -- `ever_evaluated`,
    `any_positive`, the two geometric-cut booleans, and
    `UNRESOLVED_OBSERVATION_CONFLICT` membership -- scattered from edges to
    their two endpoints). Priority order (checked top to bottom, matching
    the directive's own listed order): node-level visibility first, then
    conflict, then "positive support existed but the geometric gate cut it",
    then "co-observed but the corridor test never went positive", then
    "visible but no candidate edge was ever co-observed at all". Only
    surfels in `singleton_mask` are classified; category is 0
    (`SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE`, the first category) for any
    non-singleton index, but callers must restrict reporting to
    `singleton_mask` -- the field is never meaningful outside it.
    """

    torch = require_torch()
    on_observed = node_view_observability.on_observed_surface_view_count
    category = torch.full_like(on_observed, SINGLETON_CAUSE_CATEGORIES.index(SINGLETON_OTHER))

    never_visible = on_observed == 0
    category = torch.where(
        never_visible, torch.full_like(category, SINGLETON_CAUSE_CATEGORIES.index(SINGLETON_NODE_NEVER_POSITIVELY_VISIBLE)), category
    )

    visible = ~never_visible
    no_edge_evaluated = visible & (~node_ever_evaluated)
    category = torch.where(
        no_edge_evaluated,
        torch.full_like(category, SINGLETON_CAUSE_CATEGORIES.index(SINGLETON_NODE_VISIBLE_NO_COOBSERVED_EDGE)),
        category,
    )

    corridor_fails = visible & node_ever_evaluated & (~node_any_positive_pre_geometry) & (~node_conflict)
    category = torch.where(
        corridor_fails,
        torch.full_like(category, SINGLETON_CAUSE_CATEGORIES.index(SINGLETON_COOBSERVED_CORRIDOR_FAILS)),
        category,
    )

    geometric_cut = visible & node_any_positive_pre_geometry & node_geometric_cut
    category = torch.where(
        geometric_cut,
        torch.full_like(category, SINGLETON_CAUSE_CATEGORIES.index(SINGLETON_POSITIVE_BUT_GEOMETRIC_CUT)),
        category,
    )

    conflict = visible & node_conflict & (~geometric_cut)
    category = torch.where(
        conflict, torch.full_like(category, SINGLETON_CAUSE_CATEGORIES.index(SINGLETON_OBSERVATION_CONFLICT)), category
    )

    return category


def node_observability_accounting(
    category: Any, categories: tuple = NODE_OBSERVABILITY_CATEGORIES, mask: Any | None = None
) -> dict[str, Any]:
    """Exact counts/fractions for a category tensor, optionally restricted to
    `mask` (e.g. Worklog 103 singleton surfels only)."""

    torch = require_torch()
    selected = category if mask is None else category[mask]
    total = int(selected.shape[0])
    counts = {name: int((selected == index).sum()) for index, name in enumerate(categories)}
    fractions = {name: (value / total if total > 0 else 0.0) for name, value in counts.items()}
    return {"total": total, "counts": counts, "fractions": fractions}
