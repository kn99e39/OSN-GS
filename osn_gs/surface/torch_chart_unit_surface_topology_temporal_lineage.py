from __future__ import annotations

"""Worklog 91 -- read-only temporal + lineage attribution for Worklog 90's
``MULTILAYER_OR_VOLUMETRIC`` dominance.

This module adds NO surface constructor, alters NO Worklog 82 relation, and
tunes NO threshold. It only measures, for the same failed chart units that
Worklog 90 already classified, whether the multilayer signal comes from the
Gaussian *centers themselves* (position-only evidence, independent of any
Gaussian's own covariance orientation) or from the *covariance frame*
(orientation/shape) disagreeing while centers stay compatible with one thin
positional sheet. It also attributes ADC lineage from stable-Gaussian-ID
presence/absence across checkpoints, and reports rendered
visibility/screen-space overlap for competing-layer locations.

Four independent read-only measurements, matching the Worklog 91 request:

1. ``center_geometry_layer_count`` / ``center_geometry_multilayer`` --
   local layers found by 1-D gap clustering of *center* positions projected
   onto a PCA normal fit to those same centers only (never a per-Gaussian
   covariance eigenvector).
2. ``covariance_only_ambiguity`` -- true when centers are single-sheet
   (``center_geometry_multilayer`` is False) but Worklog 90's own
   covariance-footprint layer-conflict signal is still present.
3. Lineage fields are attached externally per stable Gaussian ID by the
   replay script (checkpoint-to-checkpoint ID set membership); this module
   only exposes the per-member local layer id so the replay script can join
   ADC birth/death lineage onto layer membership.
4. Visibility/depth-ordering is computed entirely in the replay script from
   real camera renders; this module never touches rendering.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# A layer boundary is declared where the sorted signed-normal-offset gap
# exceeds this many multiples of the local median gap. This is a robust
# 1-D clustering split, not a same-surface admission threshold, and is not
# swept here.
_LAYER_GAP_RATIO = 3.0


@dataclass(frozen=True)
class CenterGeometryLayering:
    member_count: int
    layer_count: int
    multilayer: bool
    layer_id_by_member: tuple[int, ...]
    pca_normal: Any
    depth_separation: float | None
    center_spacing: float | None


def _local_pca_normal(points: Any):
    """Fit a plane normal to centers only via SVD of the mean-centered points.

    Deliberately independent of every Gaussian's own covariance eigenframe:
    this is the position-only evidence the Worklog 91 directive requires.
    """

    torch = require_torch()
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean
    # SVD of the (count, 3) centered position matrix; the smallest right
    # singular vector is the least-variance (normal) direction of the
    # center point cloud itself.
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
    except Exception:  # pragma: no cover - defensive, SVD is expected to converge
        normal = torch.zeros(3, dtype=points.dtype, device=points.device)
        normal[-1] = 1.0
    norm = normal.norm().clamp_min(_EPS)
    return normal / norm


def compute_center_geometry_layering(
    positions: Any,
    member_indices: Sequence[int],
) -> CenterGeometryLayering:
    """Cluster member centers into signed-depth layers using position-only PCA.

    This never reads a Gaussian's covariance/scale/rotation. It answers
    "do the centers themselves form multiple local sheets" independent of
    how any individual Gaussian's footprint is oriented.
    """

    torch = require_torch()
    members = tuple(dict.fromkeys(int(index) for index in member_indices))
    count = len(members)
    if count == 0:
        raise ValueError("member_indices must not be empty")
    selector = torch.tensor(members, dtype=torch.long, device=positions.device)
    points = positions[selector]
    if count < 3:
        return CenterGeometryLayering(count, 1, False, tuple(0 for _ in members), None, None, None)

    normal = _local_pca_normal(points)
    mean = points.mean(dim=0)
    signed_offset = (points - mean) @ normal
    order = torch.argsort(signed_offset)
    sorted_offset = signed_offset[order]
    gaps = sorted_offset[1:] - sorted_offset[:-1]
    median_gap = gaps.median() if gaps.numel() else torch.tensor(0.0, device=points.device)
    split_after = gaps > (_LAYER_GAP_RATIO * median_gap.clamp_min(_EPS))
    # Guard against a degenerate all-equal cloud producing a zero median gap
    # that would call every non-zero gap a split; require an absolute floor
    # tied to the point cloud's own extent instead of a swept parameter.
    extent = (sorted_offset[-1] - sorted_offset[0]).clamp_min(_EPS)
    absolute_floor = 0.02 * extent
    split_after = split_after & (gaps > absolute_floor)

    layer_id_sorted = torch.zeros(count, dtype=torch.long, device=points.device)
    current = 0
    for local_index in range(1, count):
        if bool(split_after[local_index - 1]):
            current += 1
        layer_id_sorted[local_index] = current
    layer_id = torch.zeros(count, dtype=torch.long, device=points.device)
    layer_id[order] = layer_id_sorted
    layer_count = int(layer_id_sorted.max().item()) + 1

    depth_separation = None
    if layer_count > 1:
        centers_by_layer = [
            sorted_offset[layer_id_sorted == layer].mean()
            for layer in range(layer_count)
        ]
        centers_by_layer = torch.stack(centers_by_layer)
        depth_separation = float((centers_by_layer.max() - centers_by_layer.min()).item())

    center_spacing = float(median_gap.item()) if gaps.numel() else None

    return CenterGeometryLayering(
        member_count=count,
        layer_count=layer_count,
        multilayer=layer_count > 1,
        layer_id_by_member=tuple(int(v) for v in layer_id.tolist()),
        pca_normal=normal.detach(),
        depth_separation=depth_separation,
        center_spacing=center_spacing,
    )


@dataclass(frozen=True)
class CovarianceOnlyAmbiguity:
    member_count: int
    covariance_only_ambiguous: bool
    covariance_only_ambiguous_node_fraction: float


def compute_covariance_only_ambiguity(
    layering: CenterGeometryLayering,
    layer_conflict_node_mask: Any,
) -> CovarianceOnlyAmbiguity:
    """Split Worklog 90's covariance layer-conflict signal by center layering.

    ``layer_conflict_node_mask`` is the exact boolean node mask Worklog 90
    already computes (``layer_conflict.any(dim=1)``) for the same unit and
    member ordering -- passed in, never recomputed with different logic.
    When centers are single-sheet (``not layering.multilayer``) but this
    mask still fires on some nodes, that fraction is covariance-frame-only
    disagreement: centers remain compatible with one thin positional sheet,
    yet covariance normal/tangent/thickness interpretation disagrees.
    """

    torch = require_torch()
    mask = torch.as_tensor(layer_conflict_node_mask)
    if layering.multilayer:
        # True positional multilayer is present; covariance conflict here is
        # not purely a representation artifact -- centers genuinely disagree.
        return CovarianceOnlyAmbiguity(layering.member_count, False, 0.0)
    fraction = float(mask.float().mean().item()) if mask.numel() else 0.0
    return CovarianceOnlyAmbiguity(
        member_count=layering.member_count,
        covariance_only_ambiguous=fraction > 0.0,
        covariance_only_ambiguous_node_fraction=fraction,
    )
