from __future__ import annotations

"""Worklog 101 -- restrict Worklog 98's own synchronized U/V curve lattice
to one intrinsic chart (:mod:`~osn_gs.surface.torch_intrinsic_chart_atlas`).

Not a new curve-seeding rule: :func:`~osn_gs.surface.torch_latent_surface_curve_lattice.build_curve_lattice`
is called UNMODIFIED on the chart's own restricted
:class:`~osn_gs.surface.torch_latent_surface_tangent_frame_field.TangentFrameFieldComponent`
view (same synchronized ``e_u``/``e_v`` at every retained node, own
candidate-B ``(u, v)``). The only new behavior is a post-hoc truncation:
because the tracer in that module walks continuously through the FULL
latent-surface support field (not restricted to the chart's own node set),
a traced curve can wander past the chart's own boundary. Any such point is
truncated back to the last point whose nearest ORIGINAL-component node is
still a chart member, and the curve is marked as having terminated at the
chart limit -- reported with ``PARAMETRIC_CHART_SEAM`` semantics, never as
a physical boundary/crease/observation-frontier.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_lattice import (
    CurveLatticeUV,
    IntegralCurve,
    build_curve_lattice,
)
from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.surface.torch_latent_surface_tangent_frame_field import TangentFrameFieldComponent
from osn_gs.utils.torch_ops import require_torch

TERMINATION_CHART_LIMIT = "PARAMETRIC_CHART_SEAM"


@dataclass(frozen=True)
class ChartCurveLattice:
    lattice: CurveLatticeUV
    chart_limited_curve_count: int  # curves truncated because they left the chart (parametric seam only)


def _truncate_to_chart(
    curve: IntegralCurve, chart_component: TangentFrameFieldComponent,
    parent_component: TangentFrameFieldComponent, chart_node_indices: frozenset[int],
) -> tuple[IntegralCurve, bool]:
    """A traced curve is a BIDIRECTIONAL walk concatenated as
    ``[...backward, anchor, forward...]`` -- truncating by a single
    left-to-right scan would wrongly discard an entirely valid forward half
    whenever the backward half left the chart first. Locate the anchor's
    own row (nearest match to its known 3D position) and truncate each
    direction independently, outward from that row."""

    torch = require_torch()
    parent_positions = parent_component.positions

    def _in_chart(row: int) -> bool:
        point = curve.points[row]
        nearest = int(torch.cdist(point.reshape(1, 3), parent_positions).reshape(-1).argmin().item())
        return nearest in chart_node_indices

    anchor_position = chart_component.positions[curve.anchor_node_index]
    anchor_row = int(torch.cdist(anchor_position.reshape(1, 3), curve.points).reshape(-1).argmin().item())

    keep_backward = []
    for row in range(anchor_row - 1, -1, -1):
        if not _in_chart(row):
            break
        keep_backward.append(row)
    keep_backward.reverse()

    keep_forward = []
    for row in range(anchor_row + 1, int(curve.points.shape[0])):
        if not _in_chart(row):
            break
        keep_forward.append(row)

    keep = keep_backward + [anchor_row] + keep_forward
    left_chart = len(keep) != int(curve.points.shape[0])
    if not keep:
        return curve, left_chart
    if len(keep) == int(curve.points.shape[0]):
        return curve, False
    selector = torch.tensor(keep, dtype=torch.long, device=curve.points.device)
    return IntegralCurve(curve.family, curve.anchor_node_index, curve.points[selector], curve.uv[selector]), left_chart


def build_chart_curve_lattice(
    chart_component: TangentFrameFieldComponent,
    parent_component: TangentFrameFieldComponent,
    chart_node_indices: tuple[int, ...],
    support: LatentSurfaceSupport,
) -> ChartCurveLattice:
    """Restrict the existing synchronized curve lattice to one chart --
    calls :func:`build_curve_lattice` unmodified on the chart's own
    restricted component, then truncates any curve that the tracer walked
    past the chart's own node membership."""

    lattice = build_curve_lattice(chart_component, support)
    if not lattice.valid:
        return ChartCurveLattice(lattice, 0)

    node_set = frozenset(chart_node_indices)
    truncated_u = []
    truncated_v = []
    limited_count = 0
    for curve in lattice.u_curves:
        new_curve, left = _truncate_to_chart(curve, chart_component, parent_component, node_set)
        truncated_u.append(new_curve)
        limited_count += int(left)
    for curve in lattice.v_curves:
        new_curve, left = _truncate_to_chart(curve, chart_component, parent_component, node_set)
        truncated_v.append(new_curve)
        limited_count += int(left)

    restricted_lattice = CurveLatticeUV(
        valid=True, invalid_reason=None, points=lattice.points, uv=lattice.uv,
        u_curves=tuple(truncated_u), v_curves=tuple(truncated_v),
    )
    return ChartCurveLattice(restricted_lattice, limited_count)
