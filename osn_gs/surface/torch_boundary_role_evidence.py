from __future__ import annotations

"""Topology-neutral observed Boundary-first role evidence."""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_boundary_support_network import ObservedBoundaryCurve


@dataclass(frozen=True)
class BoundaryRoleEvidence:
    role: str
    curve: ObservedBoundaryCurve | None
    anchor: Any | None
    provenance: dict[str, Any]


def boundary_role_evidence(*, outer: ObservedBoundaryCurve, interior: ObservedBoundaryCurve | None = None, anchor: Any | None = None) -> tuple[BoundaryRoleEvidence, ...]:
    if interior is not None and anchor is not None:
        raise ValueError('An interior role is either an observed boundary curve or an observed anchor, never both.')
    if interior is None and anchor is None:
        raise ValueError('Boundary-first construction requires observed interior support evidence.')
    roles = [BoundaryRoleEvidence('outer_boundary', outer, None, {'source': 'observed_boundary'})]
    if interior is not None:
        roles.append(BoundaryRoleEvidence('interior_boundary', interior, None, {'source': 'observed_boundary'}))
    else:
        roles.append(BoundaryRoleEvidence('interior_anchor', None, anchor, {'source': 'observed_anchor'}))
    return tuple(roles)