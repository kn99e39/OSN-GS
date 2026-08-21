from __future__ import annotations

"""Worklog 104 -- Primitive Ownership vs Visible Topology Membership.

Worklog 103's real-scene measurement (replayed unmodified in this batch,
`torch_positive_visible_adjacency.py`) showed 63.4% of surfels become
singleton components. This batch's node-level observability accounting
(`torch_node_level_observability_accounting.py`) attributed 94.5% of those
singletons to `NODE_NEVER_POSITIVELY_VISIBLE` -- the surfel's own CENTER was
never classified `on_observed_surface` in ANY of the 161 training views.
That is a Branch-A result (directive section 6): a material fraction of
singleton surfels genuinely lack positive observed-visible evidence, not
merely an overly strict pairwise 3D-edge test.

The fix this batch makes is deliberately NOT a new adjacency mechanism. It is
a REPRESENTATIONAL SEPARATION between two contracts that Worklog 96-103 had
implicitly conflated:

    PRIMITIVE OWNERSHIP
        every trained surfel is retained and accounted for -- this contract
        is unconditional and does not change here (WL96-103's own
        `coverage_identity_holds` invariant is untouched).

    VISIBLE TOPOLOGY MEMBERSHIP
        being a member of a structural Visible Surface Component is a
        STRONGER claim than merely being owned. A surfel whose own component
        has exactly one member (itself) is NOT called a "Visible Surface
        Component" here -- it is a retained, owned primitive that currently
        has no positively-observed-visible structural relationship to any
        neighbor. It is not discarded, and it is not Trust or any future
        latent-fitting score; it is a plain, deterministic size>=2 test on
        Worklog 103's own unmodified component output.

This module adds NO new adjacency, NO new threshold, and does not touch
`torch_positive_visible_adjacency.py`. It only reclassifies WHAT a
`PositiveVisibleAdjacencyResult`'s own component sizes already mean.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_positive_visible_adjacency import PositiveVisibleAdjacencyResult
from osn_gs.utils.torch_ops import require_torch

_MIN_STRUCTURAL_COMPONENT_SIZE = 2  # a component of one member is not a structural relationship; not swept


@dataclass(frozen=True)
class PrimitiveOwnershipAccounting:
    """Unconditional: every trained (visible-domain) surfel is retained."""

    total_surfels: int
    retained_surfels: int  # always == total_surfels; explicit invariant, never discards

    def payload(self) -> dict[str, Any]:
        return {
            "total_surfels": self.total_surfels,
            "retained_surfels": self.retained_surfels,
            "ownership_complete": self.retained_surfels == self.total_surfels,
        }


@dataclass(frozen=True)
class VisibleTopologyAccounting:
    """Structural visible-surface membership -- a STRICT SUBSET of ownership.
    `non_visible_topology_owned_*` surfels are retained primitives that are
    NOT called Visible Surface Components; they are not discarded and are
    not assigned any quality/trust score here."""

    structural_component_count: int  # components with >= 2 members
    structural_visible_surfel_count: int
    structural_visible_surfel_fraction: float
    non_visible_topology_owned_surfel_count: int
    non_visible_topology_owned_surfel_fraction: float
    structural_membership_mask: Any  # (N,) bool -- True iff this surfel's own component has >= 2 members

    def payload(self) -> dict[str, Any]:
        return {
            "structural_component_count": self.structural_component_count,
            "structural_visible_surfel_count": self.structural_visible_surfel_count,
            "structural_visible_surfel_fraction": self.structural_visible_surfel_fraction,
            "non_visible_topology_owned_surfel_count": self.non_visible_topology_owned_surfel_count,
            "non_visible_topology_owned_surfel_fraction": self.non_visible_topology_owned_surfel_fraction,
        }


def derive_primitive_vs_visible_topology_accounting(
    result: PositiveVisibleAdjacencyResult,
) -> tuple[PrimitiveOwnershipAccounting, VisibleTopologyAccounting]:
    """Split Worklog 103's own (unmodified) partition result into the two
    separate contracts. `result` is read-only here -- no field of it is
    mutated, and no new subset_ids/edges are produced."""

    torch = require_torch()
    total = len(result)
    primitive = PrimitiveOwnershipAccounting(total_surfels=total, retained_surfels=total)

    if total == 0:
        empty_mask = torch.zeros((0,), dtype=torch.bool)
        return primitive, VisibleTopologyAccounting(0, 0, 0.0, 0, 0.0, empty_mask)

    sizes = result.subset_sizes
    structural_component_count = int((sizes >= _MIN_STRUCTURAL_COMPONENT_SIZE).sum())
    node_component_size = sizes[result.subset_ids]
    structural_mask = node_component_size >= _MIN_STRUCTURAL_COMPONENT_SIZE
    structural_count = int(structural_mask.sum())
    non_visible_count = total - structural_count

    topology = VisibleTopologyAccounting(
        structural_component_count=structural_component_count,
        structural_visible_surfel_count=structural_count,
        structural_visible_surfel_fraction=structural_count / total,
        non_visible_topology_owned_surfel_count=non_visible_count,
        non_visible_topology_owned_surfel_fraction=non_visible_count / total,
        structural_membership_mask=structural_mask,
    )
    return primitive, topology
