from __future__ import annotations

"""Non-overlap evidence checks required before multi-loop materialization."""
from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class PlanarDomainPartitionEvidence:
    state: str
    reason: str | None
    outer_label: int | None
    hole_labels: tuple[int, ...]
    provenance: dict[str, Any]


def _signed_area(points: Any) -> float:
    torch = require_torch()
    values = torch.as_tensor(points)
    if int(values.shape[0]) > 1 and bool(torch.allclose(values[0], values[-1])):
        values = values[:-1]
    if int(values.shape[0]) < 3:
        return 0.0
    shifted = torch.roll(values, -1, dims=0)
    return float((values[:, 0] * shifted[:, 1] - values[:, 1] * shifted[:, 0]).sum() * 0.5)


def assess_non_overlapping_planar_partition(boundary_result: Any) -> PlanarDomainPartitionEvidence:
    """Validate loop ownership/orientation evidence without creating a chart."""
    torch = require_torch()
    outer = list(getattr(boundary_result, "outer_loops", ()))
    holes = list(getattr(boundary_result, "hole_loops", ()))
    if len(outer) != 1 or len(holes) < 2:
        return PlanarDomainPartitionEvidence("not_applicable", "requires_one_outer_and_multiple_holes", None, (), {})
    frame = getattr(boundary_result, "frame", None)
    if frame is None:
        return PlanarDomainPartitionEvidence("unsupported", "uv_frame_required", int(outer[0].label), (), {})
    all_loops = [outer[0], *holes]
    if not all(bool(getattr(loop, "ordered_boundary_world_points", ())) for loop in all_loops):
        return PlanarDomainPartitionEvidence("unsupported", "ordered_boundary_required", int(outer[0].label), (), {})
    outer_label = int(outer[0].label)
    nested = tuple(sorted(int(loop.label) for loop in holes if getattr(loop, "nested_in_outer_label", None) == outer_label))
    if len(nested) != len(holes):
        return PlanarDomainPartitionEvidence("unsupported", "hole_nesting_evidence_incomplete", outer_label, nested, {})
    areas = {}
    for loop in all_loops:
        world = torch.as_tensor(loop.ordered_boundary_world_points)
        areas[int(loop.label)] = _signed_area(frame.apply(world, clamp=False))
    outer_area = areas[outer_label]
    if abs(outer_area) <= 1e-12 or any(abs(areas[label]) <= 1e-12 for label in nested):
        return PlanarDomainPartitionEvidence("unsupported", "degenerate_loop_area", outer_label, nested, {"signed_areas_uv": areas})
    if any(areas[label] * outer_area >= 0.0 for label in nested):
        return PlanarDomainPartitionEvidence("review_required", "loop_orientation_normalization_required", outer_label, nested, {"signed_areas_uv": areas, "outer_boundary_owner_count": 1})
    return PlanarDomainPartitionEvidence("review_required", "partition_materialization_required", outer_label, nested, {"signed_areas_uv": areas, "outer_boundary_owner_count": 1, "non_overlapping_partition_required": True})