from __future__ import annotations

"""Worklog 92 -- final read-only center-geometry attribution, replacing
Worklog 91's single-global-SVD-plane layer diagnostic with a local,
curvature-aware one.

Worklog 91 fit ONE SVD plane to an entire chart unit and clustered signed
offsets from that single plane into "layers." A curved or non-planar single
sheet can trivially produce several depth bands relative to one global
plane -- that confound is exactly what this module removes. It never reads
or uses any Gaussian's own covariance normal/tangent/scale; every measure
here comes from CENTER POSITIONS ONLY. The local plane fit per neighborhood
is diagnostic only (used to measure local signed residual modes) and is
never exposed, stored, or reused as a production surface-normal source.

Five mutually exclusive local per-node classes:

- LOCALLY_SINGLE_CURVED_SHEET: node's local neighborhood has one dominant
  signed-residual mode; wider unit-level spread is explained by surface
  curvature across neighborhoods with different local plane orientation,
  not by a true competing layer at any single neighborhood.
- LOCALLY_THICK_UNIMODAL_SHEET: node's local neighborhood has one mode but
  that mode's own spread (thickness) is large relative to local in-plane
  spacing -- a thick single sheet, not curvature and not competing layers.
- TRUE_PERSISTENT_TWO_LAYER / TRUE_PERSISTENT_MULTI_LAYER: node's local
  neighborhood shows 2 (or >2) separated residual modes, AND this
  multi-modality recurs in SPATIALLY NEIGHBORING local neighborhoods (not
  an isolated single-neighborhood artifact) -- the persistence test the
  Worklog 92 directive requires.
- SPARSE_SATELLITE_OR_OUTLIER: node's local neighborhood is too sparse to
  support any of the above (too few local neighbors, or the node's own
  local layer is a tiny disconnected population), disclosed separately
  rather than folded into a "true" multilayer count.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

LOCALLY_SINGLE_CURVED_SHEET = "LOCALLY_SINGLE_CURVED_SHEET"
LOCALLY_THICK_UNIMODAL_SHEET = "LOCALLY_THICK_UNIMODAL_SHEET"
TRUE_PERSISTENT_TWO_LAYER = "TRUE_PERSISTENT_TWO_LAYER"
TRUE_PERSISTENT_MULTI_LAYER = "TRUE_PERSISTENT_MULTI_LAYER"
SPARSE_SATELLITE_OR_OUTLIER = "SPARSE_SATELLITE_OR_OUTLIER"

LOCAL_CLASSES = (
    LOCALLY_SINGLE_CURVED_SHEET,
    LOCALLY_THICK_UNIMODAL_SHEET,
    TRUE_PERSISTENT_TWO_LAYER,
    TRUE_PERSISTENT_MULTI_LAYER,
    SPARSE_SATELLITE_OR_OUTLIER,
)

# Local neighborhood size for the diagnostic-only local plane fit. This is
# Worklog 82's own existing bounded-kNN candidate count (not a new swept
# parameter) reused here purely as a positional neighborhood radius.
_LOCAL_NEIGHBOR_COUNT = 8
# A local neighborhood too small to fit a stable plane/mode split.
_MIN_NEIGHBORHOOD_FOR_MODE_SPLIT = 5
# Same robust 1-D gap-clustering rule as Worklog 91 (local median gap x N,
# floored at a fraction of the neighborhood's own extent), applied per
# LOCAL neighborhood instead of once globally. Small local neighborhoods
# (k~8) make a bare "largest gap vs. median gap" test noisy -- extreme-value
# statistics of a handful of unimodal samples routinely produce one gap
# several times the median by chance. The floor below additionally requires
# the gap to be large relative to the neighborhood's own extent, which a
# smooth unimodal (even thick) distribution rarely satisfies at this small
# a sample size.
_LAYER_GAP_RATIO = 3.0
_LAYER_GAP_FLOOR_FRACTION = 0.2
# A local single mode is "thick" (unimodal-but-thick, not curvature) when
# its own spread exceeds this multiple of local in-plane spacing.
_THICK_UNIMODAL_SPREAD_RATIO = 1.5
# A node's own local layer population must reach this size to avoid being
# called a sparse satellite when it otherwise looks like a true layer.
_MIN_LOCAL_LAYER_POPULATION = 3


@dataclass(frozen=True)
class LocalNodeGeometry:
    node: int
    neighbor_count: int
    local_mode_count: int
    local_mode_id: int
    local_mode_population: int
    local_depth_separation_over_spacing: float | None
    local_in_plane_spacing: float | None
    local_mode_spread_over_spacing: float | None


@dataclass(frozen=True)
class LocalCenterGeometryAttribution:
    member_count: int
    class_by_member: tuple[str, ...]
    class_node_fractions: dict[str, float]
    primary_class: str
    local_geometry_by_member: tuple[LocalNodeGeometry, ...]
    persistent_layer_count: int


def _knn_indices(points: Any, k: int) -> Any:
    torch = require_torch()
    count = points.shape[0]
    k = min(k, count - 1)
    if k <= 0:
        return torch.zeros((count, 0), dtype=torch.long, device=points.device)
    delta = points[None, :, :] - points[:, None, :]
    distance = delta.norm(dim=2)
    distance.fill_diagonal_(float("inf"))
    _, indices = torch.topk(distance, k, dim=1, largest=False)
    return indices


def _local_plane_normal(points: Any) -> Any:
    """SVD-fit normal of a small local neighborhood. Diagnostic only: this
    is never returned as a per-Gaussian production normal, only used inline
    to compute this neighborhood's own signed residual modes."""

    torch = require_torch()
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
    except Exception:  # pragma: no cover - defensive
        normal = torch.zeros(3, dtype=points.dtype, device=points.device)
        normal[-1] = 1.0
    return normal / normal.norm().clamp_min(_EPS)


# A tentative gap split is only accepted as a real mode boundary when the
# gap also dominates the internal spread of the two candidate sides it
# separates. Bare "largest gap vs. median gap" is unreliable at k~8: a
# smooth unimodal (even thick) distribution's order-statistic gaps are
# noisy enough at that sample size to spuriously exceed 3x the median gap
# by chance. Requiring the gap to also exceed each side's own within-side
# spread by this ratio is a coarse 1-D silhouette/two-means separation
# check, not merely a rescaled version of the same heuristic.
_SIDE_SPREAD_SEPARATION_RATIO = 1.5


def _gap_cluster_1d(sorted_values: Any) -> Any:
    """Local per-neighborhood 1-D mode split. A gap becomes a mode boundary
    only if it (a) exceeds ``_LAYER_GAP_RATIO`` times the neighborhood's
    median gap, (b) exceeds ``_LAYER_GAP_FLOOR_FRACTION`` of the
    neighborhood's own depth extent, AND (c) dominates the internal spread
    of both sides it separates (a silhouette-style tightness check) --
    condition (c) is what actually distinguishes genuine bimodality from a
    single smooth thick distribution's largest order-statistic gap.
    """

    torch = require_torch()
    count = sorted_values.shape[0]
    if count < 2:
        return torch.zeros(count, dtype=torch.long, device=sorted_values.device)
    gaps = sorted_values[1:] - sorted_values[:-1]
    median_gap = gaps.median()
    extent = (sorted_values[-1] - sorted_values[0]).clamp_min(_EPS)
    absolute_floor = _LAYER_GAP_FLOOR_FRACTION * extent
    candidate = (gaps > (_LAYER_GAP_RATIO * median_gap.clamp_min(_EPS))) & (gaps > absolute_floor)

    accepted = torch.zeros_like(candidate)
    for index in range(gaps.shape[0]):
        if not bool(candidate[index]):
            continue
        left_side = sorted_values[: index + 1]
        right_side = sorted_values[index + 1 :]
        left_spread = (
            (left_side.max() - left_side.min()) if left_side.numel() > 1 else sorted_values.new_zeros(())
        )
        right_spread = (
            (right_side.max() - right_side.min()) if right_side.numel() > 1 else sorted_values.new_zeros(())
        )
        gap_value = gaps[index]
        side_tight_enough = (
            gap_value > _SIDE_SPREAD_SEPARATION_RATIO * left_spread.clamp_min(_EPS)
            and gap_value > _SIDE_SPREAD_SEPARATION_RATIO * right_spread.clamp_min(_EPS)
        )
        accepted[index] = side_tight_enough

    ids = torch.zeros(count, dtype=torch.long, device=sorted_values.device)
    current = 0
    for index in range(1, count):
        if bool(accepted[index - 1]):
            current += 1
        ids[index] = current
    return ids


def _local_node_geometry(points: Any, node: int, neighbor_indices: Any) -> LocalNodeGeometry:
    torch = require_torch()
    neighborhood = torch.cat([torch.tensor([node], device=points.device), neighbor_indices])
    neighbor_count = int(neighbor_indices.numel())
    if neighbor_count < _MIN_NEIGHBORHOOD_FOR_MODE_SPLIT:
        return LocalNodeGeometry(node, neighbor_count, 1, 0, 1, None, None, None)

    local_points = points[neighborhood]
    normal = _local_plane_normal(local_points)
    mean = local_points.mean(dim=0)
    signed_offset = (local_points - mean) @ normal
    order = torch.argsort(signed_offset)
    sorted_offset = signed_offset[order]
    mode_id_sorted = _gap_cluster_1d(sorted_offset)
    mode_id = torch.zeros_like(mode_id_sorted)
    mode_id[order] = mode_id_sorted
    mode_count = int(mode_id_sorted.max().item()) + 1

    node_local_index = 0  # ``node`` was prepended, so it is always index 0.
    node_mode = int(mode_id[node_local_index].item())
    node_mode_population = int((mode_id == node_mode).sum().item())

    tangent_delta = local_points - mean - signed_offset[:, None] * normal[None, :]
    in_plane_pairwise = torch.cdist(tangent_delta, tangent_delta)
    off_diagonal = ~torch.eye(neighborhood.numel(), dtype=torch.bool, device=points.device)
    nearest_in_plane = in_plane_pairwise.masked_fill(~off_diagonal, float("inf")).min(dim=1).values
    local_spacing = float(nearest_in_plane.median().item())

    depth_separation = None
    if mode_count > 1:
        mode_means = torch.stack([
            sorted_offset[mode_id_sorted == mode].mean() for mode in range(mode_count)
        ])
        depth_separation = float((mode_means.max() - mode_means.min()).item())

    mode_spread = None
    if node_mode_population > 1:
        this_mode_values = signed_offset[mode_id == node_mode]
        mode_spread = float((this_mode_values.max() - this_mode_values.min()).item())

    spacing_safe = max(local_spacing, _EPS)
    return LocalNodeGeometry(
        node=node,
        neighbor_count=neighbor_count,
        local_mode_count=mode_count,
        local_mode_id=node_mode,
        local_mode_population=node_mode_population,
        local_depth_separation_over_spacing=(
            depth_separation / spacing_safe if depth_separation is not None else None
        ),
        local_in_plane_spacing=local_spacing,
        local_mode_spread_over_spacing=(
            mode_spread / spacing_safe if mode_spread is not None else None
        ),
    )


def _spatial_persistence(
    positions: Any,
    members: Sequence[int],
    local_geometry: Sequence[LocalNodeGeometry],
    neighbor_index_by_local: Sequence[Any],
) -> tuple[bool, ...]:
    """A node's multi-modality is "persistent" only if at least one of its
    own local kNN neighbors ALSO reports >1 local mode -- i.e. the split is
    not an isolated single-neighborhood artifact but recurs across
    spatially adjacent neighborhoods, matching the Worklog 92 directive's
    explicit persistence requirement."""

    multi_modal = tuple(geometry.local_mode_count > 1 for geometry in local_geometry)
    persistent = [False] * len(members)
    for local_index, is_multi in enumerate(multi_modal):
        if not is_multi:
            continue
        neighbor_locals = neighbor_index_by_local[local_index]
        neighbor_is_multi = any(
            multi_modal[int(neighbor_local)] for neighbor_local in neighbor_locals.tolist()
        )
        persistent[local_index] = bool(neighbor_is_multi)
    return tuple(persistent)


def attribute_local_center_geometry(
    positions: Any,
    member_indices: Sequence[int],
) -> LocalCenterGeometryAttribution:
    """Center-position-only, curvature-aware, spatial-persistence-gated
    classification of one Worklog 89/90/91 failed unit's members.

    Never reads covariance. The per-neighborhood plane fit is diagnostic
    only, exactly like Worklog 91's global plane, but scoped locally so
    surface curvature across the unit cannot masquerade as competing
    layers.
    """

    torch = require_torch()
    members = tuple(dict.fromkeys(int(index) for index in member_indices))
    count = len(members)
    if count == 0:
        raise ValueError("member_indices must not be empty")
    selector = torch.tensor(members, dtype=torch.long, device=positions.device)
    points = positions[selector]

    if count < _MIN_NEIGHBORHOOD_FOR_MODE_SPLIT:
        geometry = tuple(
            LocalNodeGeometry(i, count - 1, 1, 0, count, None, None, None) for i in range(count)
        )
        classes = tuple(SPARSE_SATELLITE_OR_OUTLIER for _ in range(count))
        fractions = {cls: (1.0 if cls == SPARSE_SATELLITE_OR_OUTLIER else 0.0) for cls in LOCAL_CLASSES}
        return LocalCenterGeometryAttribution(count, classes, fractions, SPARSE_SATELLITE_OR_OUTLIER, geometry, 0)

    neighbor_indices = _knn_indices(points, _LOCAL_NEIGHBOR_COUNT)
    local_geometry = tuple(
        _local_node_geometry(points, node, neighbor_indices[node]) for node in range(count)
    )
    persistent = _spatial_persistence(points, members, local_geometry, neighbor_indices)

    classes: list[str] = []
    for local_index in range(count):
        geometry = local_geometry[local_index]
        if geometry.neighbor_count < _MIN_NEIGHBORHOOD_FOR_MODE_SPLIT:
            classes.append(SPARSE_SATELLITE_OR_OUTLIER)
            continue
        if geometry.local_mode_population < _MIN_LOCAL_LAYER_POPULATION:
            classes.append(SPARSE_SATELLITE_OR_OUTLIER)
            continue
        if geometry.local_mode_count > 1 and persistent[local_index]:
            if geometry.local_mode_count == 2:
                classes.append(TRUE_PERSISTENT_TWO_LAYER)
            else:
                classes.append(TRUE_PERSISTENT_MULTI_LAYER)
            continue
        if geometry.local_mode_count > 1 and not persistent[local_index]:
            # An isolated single-neighborhood split with no spatial
            # persistence: curvature/noise at this one neighborhood, not a
            # true competing layer.
            classes.append(LOCALLY_SINGLE_CURVED_SHEET)
            continue
        # Single local mode: either a thin curved sheet (curvature absorbed
        # by the LOCAL plane, unlike Worklog 91's single global plane) or a
        # thick unimodal sheet if this mode's own spread is large.
        spread_ratio = geometry.local_mode_spread_over_spacing
        if spread_ratio is not None and spread_ratio > _THICK_UNIMODAL_SPREAD_RATIO:
            classes.append(LOCALLY_THICK_UNIMODAL_SHEET)
        else:
            classes.append(LOCALLY_SINGLE_CURVED_SHEET)

    counts = {cls: classes.count(cls) for cls in LOCAL_CLASSES}
    fractions = {cls: counts[cls] / count for cls in LOCAL_CLASSES}
    # Primary class by largest node-level mass; ties resolved toward the
    # more conservative (non-true-layer) classes first, matching Worklog
    # 90/91's presentation-only tie convention.
    order = (
        TRUE_PERSISTENT_MULTI_LAYER,
        TRUE_PERSISTENT_TWO_LAYER,
        LOCALLY_THICK_UNIMODAL_SHEET,
        LOCALLY_SINGLE_CURVED_SHEET,
        SPARSE_SATELLITE_OR_OUTLIER,
    )
    primary = max(order, key=lambda cls: (fractions[cls], -order.index(cls)))
    persistent_layer_ids = {
        local_geometry[i].local_mode_id
        for i in range(count)
        if classes[i] in (TRUE_PERSISTENT_TWO_LAYER, TRUE_PERSISTENT_MULTI_LAYER)
    }
    return LocalCenterGeometryAttribution(
        member_count=count,
        class_by_member=tuple(classes),
        class_node_fractions=fractions,
        primary_class=primary,
        local_geometry_by_member=local_geometry,
        persistent_layer_count=len(persistent_layer_ids),
    )
