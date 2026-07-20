from __future__ import annotations

"""Phase 1 Surface-Cell Component Builder.

Implements ``OSN_GS_Final_Boundary_First_NURBS_Direction.md`` §Phase 1: merges
Stage 1 active (and, by default, complex) leaf voxels into physical surface
*components* using face adjacency plus normal/plane-offset compatibility.

This module is purely additive: it consumes a
``TorchVoxelGaussianHierarchy`` (unchanged) and produces a new
``SurfaceComponentSet`` describing physical surface regions. It does not
touch the legacy constructor, the ``voxel_patch_stage1`` NURBS fitting path,
or any trainer/ADC code. A leaf voxel is local evidence; a component is the
first-class physical surface region that boundary extraction (Phase 2) and
geometry fitting (Phase 3+) will operate on. In particular::

    leaf voxel == NURBS patch

no longer holds once components exist (it only holds for the
``voxel_patch_stage1`` ablation baseline, which this module leaves untouched).
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.surface.torch_voxel_hierarchy import (
    FACE_EXTERIOR,
    FACE_INTERIOR,
    FACE_UNRESOLVED,
    STATE_ACTIVE,
    STATE_COMPLEX,
    TorchVoxelGaussianHierarchy,
    VoxelNode,
    _fit_leaf_plane,
    compute_leaf_face_adjacency,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass
class ComponentEdgeDecision:
    """Merge/split diagnostic for one candidate face-adjacent leaf pair.

    ``leaf_a`` is always the lexicographically smaller node ID of the pair and
    supplies the reference normal/scale (``n_i``/``h_i`` in the plan's
    notation), so decisions are reproducible regardless of traversal order.
    """

    leaf_a: str
    leaf_b: str
    face: int
    normal_angle_degrees: float
    offset_ratio: float
    compatible: bool
    # "merged" | "normal" | "offset" | "missing_plane"
    reason: str


@dataclass
class SurfaceComponent:
    """One physical surface region: a connected group of compatible leaves."""

    component_id: int
    member_leaf_ids: list[str]
    gaussian_indices: Any  # (N,) long tensor, union of member leaves' points
    aabb_min: Any  # (3,)
    aabb_max: Any  # (3,)
    centroid: Any  # (3,) mean of member leaf plane centroids
    normal: Any  # (3,) sign-aligned mean of member leaf plane normals
    # Member leaves with >= 1 face whose neighbor is outside this component
    # (support boundary, crease/topology boundary, or unresolved contact).
    boundary_leaf_ids: list[str] = field(default_factory=list)


@dataclass
class SurfaceComponentSet:
    components: list[SurfaceComponent]
    # Every mergeable (active/complex) leaf's assigned component index.
    leaf_component_id: dict[str, int]
    edge_decisions: list[ComponentEdgeDecision]
    # leaf_id -> list of boundary-face descriptors (kind in support/crease/unresolved).
    component_boundary_faces: dict[str, list[dict[str, Any]]]
    config: dict[str, Any]

    def component_count(self) -> int:
        return len(self.components)

    def edge_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edge_decisions:
            counts[edge.reason] = counts.get(edge.reason, 0) + 1
        return counts


def _effective_offset_slope(centroid_a: Any, normal_a: Any, centroid_b: Any) -> float:
    """"Tilt" implied by connecting two leaf centroids, relative to leaf A's plane.

    Splits ``c_b - c_a`` into a component along ``n_a`` (the plane-offset,
    ``d_ij`` in the plan's notation) and the remaining in-plane (tangential)
    component, and returns their ratio -- i.e. ``tan`` of the angle between
    the connecting line and leaf A's own tangent plane.

    This is scale-invariant in a way a fixed reference length (point spacing,
    AABB diagonal, ...) cannot be: a smoothly curved surface's adjacent-leaf
    centroids drift apart mostly *within* the tangent plane by construction
    (they are spatially adjacent), so this ratio stays bounded by the
    surface's true local gradient regardless of leaf size or point density.
    Two genuinely separate parallel sheets sharing the same footprint,
    however, have centroids that are offset almost *purely along the
    normal* with near-zero tangential separation, so the ratio diverges --
    exactly the discriminator a fixed-scale threshold cannot provide.
    """

    torch = require_torch()
    diff = centroid_b - centroid_a
    offset = float((diff * normal_a).sum())
    tangential = diff - offset * normal_a
    tangential_distance = float(torch.linalg.norm(tangential))
    # Numerical floor only (not a semantic scale): guards a same-footprint,
    # near-zero-tangential-distance pair from a spurious divide-by-zero while
    # still yielding a very large (correctly rejected) ratio.
    return abs(offset) / max(tangential_distance, 1e-6)


def build_surface_components(
    hierarchy: TorchVoxelGaussianHierarchy,
    points: Any,
    fit_complex_leaves: bool = True,
    normal_threshold_degrees: float = 40.0,
    offset_threshold_ratio: float = 0.5,
) -> SurfaceComponentSet:
    """Merge active (+complex) leaves into physical surface components.

    Defaults tuned empirically (2026-07-19) against the required benchmark
    scenes and locked in with margin on both sides, not picked to just barely
    pass:

    - ``normal_threshold_degrees=40``: the largest pairwise leaf-to-leaf
      normal angle on any *smooth* required scene is sine's 35.0 degrees
      (density_gradient: 24.8 degrees); the crease scene's ridge pair is 48.5
      degrees. 40 clears smooth scenes with zero incompatible edges (not
      merely "still connected via another path") while staying ~8.5 degrees
      under the crease ridge.
    - ``offset_threshold_ratio=0.5``: the largest legitimate offset slope on
      any required scene is sine's 0.295; two independently-sampled parallel
      sheets sharing a footprint (close_parallel_sheets) produce a minimum
      cross-sheet slope of 0.916. 0.5 sits centered between them.

    Two face-adjacent mergeable leaves are connected when their local plane
    normals agree within ``normal_threshold_degrees`` AND their planes are
    coplanar within ``offset_threshold_ratio`` point-spacings of the
    lexicographically-first leaf. Only contacts ``compute_leaf_face_adjacency``
    already classifies ``interior`` are candidates (inactive/empty/outside
    contacts, and unresolved complex contacts, are never merge candidates —
    they become component *boundary* faces instead, see
    ``component_boundary_faces``). An incompatible interior-classified edge
    also becomes a boundary face (of kind ``"crease"``): the pairwise
    threshold naturally lets smoothly-varying normals chain into one
    component (each local edge's angle stays small) while a true crease still
    exceeds the threshold at the ridge, without any accumulation across the
    chain — no separate "smoothness" bookkeeping is needed for this to work.

    Connected components are computed via a deterministic union-find (leaves
    processed in sorted node-ID order, smaller root always wins ties), so the
    result does not depend on traversal/original insertion order.
    """

    torch = require_torch()
    config = {
        "fit_complex_leaves": bool(fit_complex_leaves),
        "normal_threshold_degrees": float(normal_threshold_degrees),
        "offset_threshold_ratio": float(offset_threshold_ratio),
    }

    mergeable_states = {STATE_ACTIVE} | ({STATE_COMPLEX} if fit_complex_leaves else set())
    mergeable = {
        leaf.node_id: leaf for leaf in hierarchy.leaves() if leaf.state in mergeable_states
    }
    if not mergeable:
        return SurfaceComponentSet(
            components=[], leaf_component_id={}, edge_decisions=[],
            component_boundary_faces={}, config=config,
        )

    # ``degenerate_axis_tolerant=True``: unlike the Stage 1-F caller, the
    # component builder must find real x/y neighbors on flat (z-degenerate)
    # scenes -- see the flag's docstring in torch_voxel_hierarchy.py. This is
    # the ONLY caller that passes it, so Stage 1 / Stage 1-F stay unaffected.
    adjacency = compute_leaf_face_adjacency(
        hierarchy, fit_complex_leaves=fit_complex_leaves, degenerate_axis_tolerant=True
    )

    parent = {node_id: node_id for node_id in mergeable}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb < ra:  # deterministic tie-break: smaller root always wins
                ra, rb = rb, ra
            parent[rb] = ra

    edge_decisions: list[ComponentEdgeDecision] = []
    seen_pairs: set[tuple[str, str]] = set()
    normal_threshold = float(normal_threshold_degrees)
    offset_threshold = float(offset_threshold_ratio)

    for leaf_id in sorted(mergeable):
        entry = adjacency.get(leaf_id)
        if entry is None:
            continue
        leaf = mergeable[leaf_id]
        for contact in entry["contacts"]:
            neighbor_id = contact["neighbor_id"]
            if neighbor_id is None or neighbor_id not in mergeable:
                continue
            if contact["classification"] != FACE_INTERIOR:
                continue
            dedup_key = (min(leaf_id, neighbor_id), max(leaf_id, neighbor_id))
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)

            neighbor = mergeable[neighbor_id]
            plane_a, plane_b = leaf.plane, neighbor.plane
            if plane_a is None or plane_b is None:
                edge_decisions.append(
                    ComponentEdgeDecision(
                        leaf_id, neighbor_id, contact["face"],
                        float("nan"), float("nan"), False, "missing_plane",
                    )
                )
                continue

            cos_angle = float((plane_a.normal * plane_b.normal).sum().abs().clamp(-1.0, 1.0))
            angle_degrees = float(
                torch.rad2deg(torch.arccos(torch.tensor(cos_angle, dtype=torch.float64)))
            )
            offset_ratio = _effective_offset_slope(
                plane_a.centroid, plane_a.normal, plane_b.centroid
            )

            compatible = angle_degrees < normal_threshold and offset_ratio < offset_threshold
            if compatible:
                union(leaf_id, neighbor_id)
                reason = "merged"
            elif angle_degrees >= normal_threshold:
                reason = "normal"
            else:
                reason = "offset"
            edge_decisions.append(
                ComponentEdgeDecision(
                    leaf_id, neighbor_id, contact["face"],
                    angle_degrees, offset_ratio, compatible, reason,
                )
            )

    groups: dict[str, list[str]] = {}
    for leaf_id in mergeable:
        groups.setdefault(find(leaf_id), []).append(leaf_id)

    leaf_component_id: dict[str, int] = {}
    for component_id, root in enumerate(sorted(groups)):
        for leaf_id in groups[root]:
            leaf_component_id[leaf_id] = component_id

    component_boundary_faces = _compute_component_boundary_faces(adjacency, leaf_component_id)

    components: list[SurfaceComponent] = []
    for component_id, root in enumerate(sorted(groups)):
        member_ids = sorted(groups[root])
        members = [mergeable[leaf_id] for leaf_id in member_ids]
        gaussian_indices = torch.cat([member.gaussian_indices for member in members])
        aabb_min = torch.stack([member.aabb_min for member in members]).min(dim=0).values
        aabb_max = torch.stack([member.aabb_max for member in members]).max(dim=0).values
        planes = [member.plane for member in members if member.plane is not None]
        # Fit centroid/normal from the component's own union of raw points
        # (not a per-leaf average, which would silently give a 10-Gaussian
        # leaf the same weight as a 150-Gaussian leaf).
        component_plane = _fit_leaf_plane(points[gaussian_indices])
        if component_plane is not None and planes:
            centroid = component_plane.centroid
            reference = planes[0].normal
            # Sign-align to the member leaves' consensus direction: PCA's
            # smallest-variance axis is only defined up to sign.
            normal = (
                component_plane.normal
                if float((component_plane.normal * reference).sum()) >= 0
                else -component_plane.normal
            )
        elif planes:
            centroid = torch.stack([plane.centroid for plane in planes]).mean(dim=0)
            reference = planes[0].normal
            aligned = torch.stack(
                [
                    plane.normal if float((plane.normal * reference).sum()) >= 0 else -plane.normal
                    for plane in planes
                ]
            )
            normal = torch.nn.functional.normalize(aligned.sum(dim=0), dim=0)
        else:
            centroid = (aabb_min + aabb_max) * 0.5
            normal = torch.tensor([0.0, 0.0, 1.0], dtype=aabb_min.dtype, device=aabb_min.device)
        boundary_ids = [leaf_id for leaf_id in member_ids if leaf_id in component_boundary_faces]
        components.append(
            SurfaceComponent(
                component_id=component_id,
                member_leaf_ids=member_ids,
                gaussian_indices=gaussian_indices,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                centroid=centroid,
                normal=normal,
                boundary_leaf_ids=boundary_ids,
            )
        )

    return SurfaceComponentSet(
        components=components,
        leaf_component_id=leaf_component_id,
        edge_decisions=edge_decisions,
        component_boundary_faces=component_boundary_faces,
        config=config,
    )


def _compute_component_boundary_faces(
    adjacency: dict[str, dict[str, Any]], leaf_component_id: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Per-leaf faces whose neighbor is outside the leaf's final component.

    This is the Phase 2 input: it unifies three distinct reasons a face is a
    component boundary into one classified list, so boundary extraction does
    not need to re-derive them from raw adjacency + component assignment.

    - ``"support"``: neighbor is inactive/empty/outside the hierarchy (the
      original ``exterior_support`` face classification) — genuine support
      boundary, e.g. a hole edge or the outer silhouette.
    - ``"crease"``: neighbor is itself mergeable (active/complex) but ended up
      in a *different* component because the normal/offset compatibility
      check on that specific edge failed — a topology boundary between two
      physically distinct surface regions, not a support gap.
    - ``"unresolved"``: neighbor is a complex leaf excluded from fitting.

    Active-active shared faces where both sides landed in the SAME component
    are never boundary faces, matching the plan's explicit prohibition on
    treating an active-active interface as a support boundary.
    """

    result: dict[str, list[dict[str, Any]]] = {}
    for leaf_id, component_id in leaf_component_id.items():
        entry = adjacency.get(leaf_id, {})
        boundary: list[dict[str, Any]] = []
        for contact in entry.get("contacts", []):
            neighbor_id = contact["neighbor_id"]
            neighbor_component_id = (
                leaf_component_id.get(neighbor_id) if neighbor_id is not None else None
            )
            if neighbor_id is not None and neighbor_component_id == component_id:
                continue  # same component: not a boundary face
            if neighbor_id is None or contact["classification"] == FACE_EXTERIOR:
                kind = "support"
            elif contact["classification"] == FACE_UNRESOLVED:
                kind = "unresolved"
            else:
                kind = "crease"
            boundary.append(
                {
                    "face": contact["face"],
                    "face_name": contact["face_name"],
                    "neighbor_id": neighbor_id,
                    "neighbor_component_id": neighbor_component_id,
                    "kind": kind,
                }
            )
        if boundary:
            result[leaf_id] = boundary
    return result


def surface_component_set_payload(component_set: SurfaceComponentSet) -> dict[str, Any]:
    """JSON-serializable provenance of the component set (export/viewer)."""

    return {
        "config": dict(component_set.config),
        "component_count": component_set.component_count(),
        "edge_reason_counts": component_set.edge_reason_counts(),
        "components": [
            {
                "component_id": component.component_id,
                "member_leaf_ids": component.member_leaf_ids,
                "member_leaf_count": len(component.member_leaf_ids),
                "gaussian_count": int(component.gaussian_indices.numel()),
                "gaussian_indices": component.gaussian_indices.detach().cpu().tolist(),
                "aabb_min": component.aabb_min.detach().cpu().tolist(),
                "aabb_max": component.aabb_max.detach().cpu().tolist(),
                "centroid": component.centroid.detach().cpu().tolist(),
                "normal": component.normal.detach().cpu().tolist(),
                "boundary_leaf_ids": component.boundary_leaf_ids,
            }
            for component in component_set.components
        ],
        "edge_decisions": [
            {
                "leaf_a": edge.leaf_a,
                "leaf_b": edge.leaf_b,
                "face": edge.face,
                "normal_angle_degrees": edge.normal_angle_degrees,
                "offset_ratio": edge.offset_ratio,
                "compatible": edge.compatible,
                "reason": edge.reason,
            }
            for edge in component_set.edge_decisions
        ],
        "component_boundary_faces": component_set.component_boundary_faces,
    }
