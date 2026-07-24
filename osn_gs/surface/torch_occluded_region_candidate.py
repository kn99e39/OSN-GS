from __future__ import annotations

"""Phase E (geometric) — pairwise bounded occluded-region candidate builder.

docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md,
impl plan docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md section 7.

Purely geometric: pairs `ContinuationDomain`s, builds a `(s,t)`-indexed
correspondence graph, canonicalizes it to a 1:1 matching, splits it into
monotonic components, and constructs a pairwise bounded-region ribbon topology
per component. Deliberately does NOT import `torch_observation_evidence` --
Phase C evidence is applied afterward by `torch_candidate_evidence.py`, keeping
the geometric builder evidence-free (design section 2.3).

Scope limits (design section 1, 13): pairwise two-sided only (no 3+-sided
aggregation), no NURBS fitting, no global selection/ranking, no self-intersection
or visible-surface-penetration checks beyond the explicit zero-area guard, no
production wiring.

Canonical geometry source of truth is `ContinuationDomain` (`world`,
`sample_valid_mask`, `closed`, source IDs) ONLY. `boundaries_by_id` is used for
provenance / confidence / source-state / ID-consistency validation, never to
overwrite or reconstruct domain geometry (design section 2.2).
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from osn_gs.surface.torch_aabb_broad_phase import sweep_and_prune_pairs
from osn_gs.surface.torch_continuation_domain import STATE_REJECTED as DOMAIN_STATE_REJECTED
from osn_gs.surface.torch_continuation_domain import ContinuationDomain
from osn_gs.surface.torch_patch_boundary import BOUNDARY_RECONCILED_INTERNAL, PatchBoundarySegment
from osn_gs.utils.torch_ops import require_torch

STATE_CANDIDATE = "candidate"
STATE_UNSUPPORTED = "unsupported"
STATE_REJECTED = "rejected"
CANDIDATE_STATES = {STATE_CANDIDATE, STATE_UNSUPPORTED, STATE_REJECTED}

_EPS = 1e-9


@dataclass
class SupportChain:
    """Ordered full ``(s, t)`` sample sequence on one domain plus its world points."""

    domain_id: str
    st_indices: list[tuple[int, int]]
    world: Any  # (M, 3)

    def payload(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "st_indices": [[int(s), int(t)] for s, t in self.st_indices],
            "world": self.world.detach().cpu().tolist(),
        }


@dataclass
class CorrespondenceEdge:
    """One canonical correspondence between an A sample and a B sample."""

    s_a: int
    t_a: int
    s_b: int
    t_b: int
    world_distance: float
    scale_normalized_distance: float
    mutual_nearest: bool
    outward_dot: float
    normal_dot: float
    tangent_dot: float
    position_kind: str  # endpoint | interior | closed_cyclic

    def payload(self) -> dict[str, Any]:
        return {
            "s_a": int(self.s_a),
            "t_a": int(self.t_a),
            "s_b": int(self.s_b),
            "t_b": int(self.t_b),
            "world_distance": self.world_distance,
            "scale_normalized_distance": self.scale_normalized_distance,
            "mutual_nearest": bool(self.mutual_nearest),
            "outward_dot": self.outward_dot,
            "normal_dot": self.normal_dot,
            "tangent_dot": self.tangent_dot,
            "position_kind": self.position_kind,
        }


@dataclass
class OccludedRegionCandidate:
    candidate_id: str
    supporting_domain_ids: list[str]
    supporting_boundary_ids: list[str]
    supporting_patch_ids: list[int]

    support_chain_a: SupportChain
    support_chain_b: SupportChain
    correspondence_edges: list[CorrespondenceEdge]
    connector_start: Any  # (2, 3)
    connector_end: Any | None  # (2, 3) | None (cyclic)
    bridge_cells: Any  # (K, 4, 3)
    aabb_min: Any
    aabb_max: Any

    raw_distance_statistics: dict[str, float]
    normalized_distance_statistics: dict[str, float]
    outward_soft_evidence: dict[str, float]
    normal_soft_evidence: dict[str, float]
    tangent_soft_evidence: dict[str, float]

    free_space_contradiction: dict[str, Any]
    behind_observation_support: dict[str, Any]
    on_surface_evidence: dict[str, Any]
    unobserved_evidence: dict[str, Any]
    conflicting_evidence: dict[str, Any]
    empty_voxel_support: dict[str, Any]

    state: str
    reason: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.state not in CANDIDATE_STATES:
            raise ValueError(f"Unknown occluded-region candidate state: {self.state!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "supporting_domain_ids": list(self.supporting_domain_ids),
            "supporting_boundary_ids": list(self.supporting_boundary_ids),
            "supporting_patch_ids": [int(p) for p in self.supporting_patch_ids],
            "support_chain_a": self.support_chain_a.payload(),
            "support_chain_b": self.support_chain_b.payload(),
            "correspondence_edges": [edge.payload() for edge in self.correspondence_edges],
            "connector_start": self.connector_start.detach().cpu().tolist(),
            "connector_end": None if self.connector_end is None else self.connector_end.detach().cpu().tolist(),
            "bridge_cells": self.bridge_cells.detach().cpu().tolist(),
            "aabb_min": self.aabb_min.detach().cpu().tolist(),
            "aabb_max": self.aabb_max.detach().cpu().tolist(),
            "raw_distance_statistics": dict(self.raw_distance_statistics),
            "normalized_distance_statistics": dict(self.normalized_distance_statistics),
            "outward_soft_evidence": dict(self.outward_soft_evidence),
            "normal_soft_evidence": dict(self.normal_soft_evidence),
            "tangent_soft_evidence": dict(self.tangent_soft_evidence),
            "free_space_contradiction": dict(self.free_space_contradiction),
            "behind_observation_support": dict(self.behind_observation_support),
            "on_surface_evidence": dict(self.on_surface_evidence),
            "unobserved_evidence": dict(self.unobserved_evidence),
            "conflicting_evidence": dict(self.conflicting_evidence),
            "empty_voxel_support": dict(self.empty_voxel_support),
            "state": self.state,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }


@dataclass
class ConflictEdge:
    candidate_a: str
    candidate_b: str
    reason: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Narrow phase helpers
# --------------------------------------------------------------------------- #


def _valid_flat(domain: ContinuationDomain) -> tuple[Any, Any]:
    """Return ``(flat_world (V,3), st_index (V,2) long)`` over valid samples only."""

    torch = require_torch()
    mask = domain.sample_valid_mask
    s_count, t_count = int(mask.shape[0]), int(mask.shape[1])
    idx = torch.nonzero(mask, as_tuple=False)  # (V, 2) -> (s, t)
    world = domain.world[idx[:, 0], idx[:, 1]]
    return world, idx


def _nearest(dist: Any) -> Any:
    torch = require_torch()
    return torch.argmin(dist, dim=1)


def _unit(vec: Any, eps: float) -> Any:
    torch = require_torch()
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _raw_correspondence_edges(
    domain_a: ContinuationDomain,
    domain_b: ContinuationDomain,
    scale: float,
    correspondence_threshold: float,
    eps: float,
) -> list[dict[str, Any]]:
    """Bidirectional nearest edges over valid samples, threshold-filtered,
    reduced to one edge per ``(s_a, s_b)`` (min normalized distance)."""

    torch = require_torch()
    world_a, idx_a = _valid_flat(domain_a)
    world_b, idx_b = _valid_flat(domain_b)
    if int(world_a.shape[0]) == 0 or int(world_b.shape[0]) == 0:
        return []

    dist = torch.cdist(world_a, world_b)  # (Va, Vb)
    nn_b_of_a = _nearest(dist)  # (Va,)
    nn_a_of_b = _nearest(dist.t())  # (Vb,)

    raw: dict[tuple[int, int], dict[str, Any]] = {}

    def _consider(ia: int, ib: int) -> None:
        d = float(dist[ia, ib])
        normalized = d / max(scale, eps)
        if normalized > correspondence_threshold:
            return
        s_a, t_a = int(idx_a[ia, 0]), int(idx_a[ia, 1])
        s_b, t_b = int(idx_b[ib, 0]), int(idx_b[ib, 1])
        mutual = int(nn_b_of_a[ia]) == ib and int(nn_a_of_b[ib]) == ia
        key = (s_a, s_b)
        prev = raw.get(key)
        if prev is None or normalized < prev["scale_normalized_distance"]:
            raw[key] = {
                "s_a": s_a,
                "t_a": t_a,
                "s_b": s_b,
                "t_b": t_b,
                "world_distance": d,
                "scale_normalized_distance": normalized,
                "mutual_nearest": bool(mutual),
            }

    for ia in range(int(world_a.shape[0])):
        _consider(ia, int(nn_b_of_a[ia]))
    for ib in range(int(world_b.shape[0])):
        _consider(int(nn_a_of_b[ib]), ib)

    return list(raw.values())


def _canonical_one_to_one(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedy 1:1 matching: mutual-nearest first, then normalized distance,
    then deterministic ``(s_a, s_b)`` ordering."""

    ordered = sorted(
        edges,
        key=lambda e: (not e["mutual_nearest"], e["scale_normalized_distance"], e["s_a"], e["s_b"]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    chosen: list[dict[str, Any]] = []
    for edge in ordered:
        if edge["s_a"] in used_a or edge["s_b"] in used_b:
            continue
        used_a.add(edge["s_a"])
        used_b.add(edge["s_b"])
        chosen.append(edge)
    return chosen


def _signed_step(prev: int, nxt: int, s_count: int, closed: bool) -> int:
    """Signed step in s_b, using modulo-nearest wrap for closed domains."""

    step = nxt - prev
    if closed and s_count > 0:
        # Map into (-s_count/2, s_count/2] so the shorter arc direction wins.
        step = (step + s_count // 2) % s_count - s_count // 2
        if step == -(s_count // 2) and s_count % 2 == 0:
            step = s_count // 2
    return step


def _monotonic_components(
    edges: list[dict[str, Any]], s_count_b: int, closed: bool
) -> list[list[dict[str, Any]]]:
    """Split canonical edges (sorted by s_a) into maximal monotonic-in-s_b runs.

    Open: linear split whenever the sign of the s_b step reverses. Closed:
    treat s_a as circular; if the whole ring is one modulo-monotonic run it
    stays a single cyclic component, otherwise rotate the start to a break
    point and split linearly (no duplicate seam edge -- each s appears once).
    """

    if len(edges) < 2:
        return [edges] if edges else []

    ordered = sorted(edges, key=lambda e: e["s_a"])

    def _linear_split(seq: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        components: list[list[dict[str, Any]]] = []
        current = [seq[0]]
        direction = 0
        for k in range(1, len(seq)):
            step = _signed_step(seq[k - 1]["s_b"], seq[k]["s_b"], s_count_b, closed)
            if step == 0:
                # Same s_b twice cannot happen after 1:1 matching; guard anyway.
                components.append(current)
                current = [seq[k]]
                direction = 0
                continue
            sign = 1 if step > 0 else -1
            if direction == 0:
                direction = sign
                current.append(seq[k])
            elif sign == direction:
                current.append(seq[k])
            else:
                components.append(current)
                current = [seq[k]]
                direction = 0
        components.append(current)
        return components

    if not closed:
        return _linear_split(ordered)

    # Closed: check for a full modulo-monotonic ring first.
    n = len(ordered)
    ring_steps = [
        _signed_step(ordered[k]["s_b"], ordered[(k + 1) % n]["s_b"], s_count_b, closed)
        for k in range(n)
    ]
    nonzero = [s for s in ring_steps if s != 0]
    if nonzero and all(s > 0 for s in nonzero):
        return [ordered]
    if nonzero and all(s < 0 for s in nonzero):
        return [ordered]

    # Rotate so a sign break sits at the seam, then split linearly.
    break_index = 0
    for k in range(n):
        prev_sign = 1 if ring_steps[(k - 1) % n] > 0 else (-1 if ring_steps[(k - 1) % n] < 0 else 0)
        this_sign = 1 if ring_steps[k] > 0 else (-1 if ring_steps[k] < 0 else 0)
        if prev_sign != 0 and this_sign != 0 and prev_sign != this_sign:
            break_index = k
            break
    rotated = ordered[break_index:] + ordered[:break_index]
    return _linear_split(rotated)


# --------------------------------------------------------------------------- #
# Candidate construction
# --------------------------------------------------------------------------- #


def _chain_arclength(world: Any) -> float:
    torch = require_torch()
    if int(world.shape[0]) < 2:
        return 0.0
    return float((world[1:] - world[:-1]).norm(dim=1).sum())


def _bridge_cell_area_proxy(cell: Any, eps: float) -> float:
    """Sum of the two triangle areas of the (A_i, A_{i+1}, B_{i+1}, B_i) quad."""

    torch = require_torch()
    a, b, c, d = cell[0], cell[1], cell[2], cell[3]
    t1 = 0.5 * float(torch.linalg.norm(torch.cross(b - a, c - a, dim=0)))
    t2 = 0.5 * float(torch.linalg.norm(torch.cross(c - a, d - a, dim=0)))
    return t1 + t2


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "count": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {"min": ordered[0], "median": median, "max": ordered[-1], "count": len(ordered)}


def _component_candidate(
    domain_a: ContinuationDomain,
    domain_b: ContinuationDomain,
    boundary_a: PatchBoundarySegment | None,
    boundary_b: PatchBoundarySegment | None,
    component: list[dict[str, Any]],
    scale: float,
    closed: bool,
    broad_pair_payload: dict[str, Any],
    degenerate_flags: dict[str, bool],
    eps: float,
) -> OccludedRegionCandidate:
    torch = require_torch()

    # Order edges along the chain (already s_a-monotonic within a component).
    ordered = sorted(component, key=lambda e: e["s_a"])
    st_a = [(e["s_a"], e["t_a"]) for e in ordered]
    st_b = [(e["s_b"], e["t_b"]) for e in ordered]
    world_a = torch.stack([domain_a.world[s, t] for s, t in st_a])
    world_b = torch.stack([domain_b.world[s, t] for s, t in st_b])

    empty_dict: dict[str, Any] = {}
    domain_ids = sorted([domain_a.domain_id, domain_b.domain_id])
    component_key = "|".join(f"{e['s_a']}-{e['s_b']}" for e in ordered)
    digest = hashlib.sha256((domain_ids[0] + "::" + domain_ids[1] + "::" + component_key).encode()).hexdigest()[:16]
    candidate_id = f"{domain_ids[0]}~{domain_ids[1]}:{digest}"

    n = len(ordered)
    is_cyclic = closed and n >= 3 and _is_full_ring(ordered, domain_a, domain_b)

    # Correspondence edges with soft evidence + position kind.
    edges: list[CorrespondenceEdge] = []
    for k, e in enumerate(ordered):
        s_a, t_a, s_b, t_b = e["s_a"], e["t_a"], e["s_b"], e["t_b"]
        outward_dot = float((domain_a.outward_tangent_world[s_a] * domain_b.outward_tangent_world[s_b]).sum())
        normal_dot = float((domain_a.normal[s_a, t_a] * domain_b.normal[s_b, t_b]).sum())
        tangent_dot = float((domain_a.tangent_s[s_a, t_a] * domain_b.tangent_s[s_b, t_b]).sum())
        if is_cyclic:
            position_kind = "closed_cyclic"
        elif k == 0 or k == n - 1:
            position_kind = "endpoint"
        else:
            position_kind = "interior"
        edges.append(
            CorrespondenceEdge(
                s_a=s_a,
                t_a=t_a,
                s_b=s_b,
                t_b=t_b,
                world_distance=e["world_distance"],
                scale_normalized_distance=e["scale_normalized_distance"],
                mutual_nearest=e["mutual_nearest"],
                outward_dot=outward_dot,
                normal_dot=normal_dot,
                tangent_dot=tangent_dot,
                position_kind=position_kind,
            )
        )

    # Bridge cells from adjacent correspondence pairs.
    cell_pairs = list(range(n - 1))
    wrap = is_cyclic
    cells = []
    for k in cell_pairs:
        cells.append(torch.stack([world_a[k], world_a[k + 1], world_b[k + 1], world_b[k]]))
    if wrap:
        cells.append(torch.stack([world_a[-1], world_a[0], world_b[0], world_b[-1]]))
    bridge_cells = torch.stack(cells) if cells else torch.zeros((0, 4, 3), dtype=world_a.dtype, device=world_a.device)

    connector_start = torch.stack([world_a[0], world_b[0]])
    connector_end = None if is_cyclic else torch.stack([world_a[-1], world_b[-1]])

    # Structural hard gates.
    reasons: list[str] = []
    n_pairs = n
    support_a_len = _chain_arclength(world_a)
    support_b_len = _chain_arclength(world_b)
    start_sep = float((world_a[0] - world_b[0]).norm())
    end_sep = start_sep if is_cyclic else float((world_a[-1] - world_b[-1]).norm())

    all_finite = bool(torch.isfinite(bridge_cells).all()) and bool(torch.isfinite(connector_start).all())
    if connector_end is not None:
        all_finite = all_finite and bool(torch.isfinite(connector_end).all())

    cell_areas = [_bridge_cell_area_proxy(cell, eps) for cell in bridge_cells]
    min_area = min(cell_areas) if cell_areas else 0.0

    state = STATE_CANDIDATE
    if n_pairs < 2 or int(bridge_cells.shape[0]) == 0:
        state = STATE_UNSUPPORTED
        reasons.append("fewer_than_two_correspondence_pairs")
    if support_a_len <= eps or support_b_len <= eps:
        if state == STATE_CANDIDATE:
            state = STATE_UNSUPPORTED
        reasons.append("zero_support_interval")
    if not all_finite:
        state = STATE_REJECTED
        reasons.append("non_finite_bridge_geometry")
    elif start_sep <= eps or end_sep <= eps:
        state = STATE_REJECTED
        reasons.append("zero_connector_separation")
    elif cell_areas and min_area <= eps:
        state = STATE_REJECTED
        reasons.append("zero_area_bridge_cell")

    reason = "ok" if not reasons else ";".join(reasons)

    all_world = torch.cat([world_a, world_b, bridge_cells.reshape(-1, 3)]) if int(bridge_cells.shape[0]) else torch.cat([world_a, world_b])
    aabb_min = all_world.min(dim=0).values.detach()
    aabb_max = all_world.max(dim=0).values.detach()

    raw_d = [e["world_distance"] for e in ordered]
    norm_d = [e["scale_normalized_distance"] for e in ordered]

    candidate = OccludedRegionCandidate(
        candidate_id=candidate_id,
        supporting_domain_ids=[domain_a.domain_id, domain_b.domain_id],
        supporting_boundary_ids=[domain_a.source_boundary_id, domain_b.source_boundary_id],
        supporting_patch_ids=[int(domain_a.source_patch_id), int(domain_b.source_patch_id)],
        support_chain_a=SupportChain(domain_a.domain_id, st_a, world_a.detach()),
        support_chain_b=SupportChain(domain_b.domain_id, st_b, world_b.detach()),
        correspondence_edges=edges,
        connector_start=connector_start.detach(),
        connector_end=None if connector_end is None else connector_end.detach(),
        bridge_cells=bridge_cells.detach(),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        raw_distance_statistics=_stats(raw_d),
        normalized_distance_statistics=_stats(norm_d),
        outward_soft_evidence=_stats([edge.outward_dot for edge in edges]),
        normal_soft_evidence=_stats([edge.normal_dot for edge in edges]),
        tangent_soft_evidence=_stats([edge.tangent_dot for edge in edges]),
        free_space_contradiction=dict(empty_dict),
        behind_observation_support=dict(empty_dict),
        on_surface_evidence=dict(empty_dict),
        unobserved_evidence=dict(empty_dict),
        conflicting_evidence=dict(empty_dict),
        empty_voxel_support=dict(empty_dict),
        state=state,
        reason=reason,
        provenance={
            "cyclic": bool(is_cyclic),
            "correspondence_pair_count": n_pairs,
            "support_arclength_a": support_a_len,
            "support_arclength_b": support_b_len,
            "min_bridge_cell_area_proxy": min_area,
            "broad_phase": broad_pair_payload,
            "domain_scale": scale,
            "domain_a_state": domain_a.state,
            "domain_b_state": domain_b.state,
            "domain_a_degenerate": bool(degenerate_flags.get(domain_a.domain_id, False)),
            "domain_b_degenerate": bool(degenerate_flags.get(domain_b.domain_id, False)),
            "boundary_a_state": None if boundary_a is None else boundary_a.state,
            "boundary_b_state": None if boundary_b is None else boundary_b.state,
            "evidence_applied": False,
        },
    )
    return candidate


def _is_full_ring(
    ordered: list[dict[str, Any]], domain_a: ContinuationDomain, domain_b: ContinuationDomain
) -> bool:
    """A component is a genuine cyclic band only if it covers (nearly) the full
    s-range of both domains -- otherwise it is an open arc that happens to live
    on a closed domain."""

    if not (domain_a.closed and domain_b.closed):
        return False
    s_a_used = {e["s_a"] for e in ordered}
    s_b_used = {e["s_b"] for e in ordered}
    return len(s_a_used) >= int(domain_a.world.shape[0]) and len(s_b_used) >= int(domain_b.world.shape[0])


def build_geometric_region_candidates(
    domains: Sequence[ContinuationDomain],
    boundaries_by_id: Mapping[str, PatchBoundarySegment],
    surfaces_by_patch_id: Mapping[int, Any] | None = None,
    *,
    broad_phase_expand_factor: float = 1.5,
    correspondence_threshold: float = 1.0,
    eps: float = _EPS,
) -> list[OccludedRegionCandidate]:
    """Build pairwise two-sided occluded-region candidates from continuation domains.

    Geometric only -- no `ObservationEvidence`. Returns every constructed
    candidate/unsupported/rejected result (never silently dropped), one per
    monotonic correspondence component per surviving domain pair.
    """

    torch = require_torch()

    # Dedup by domain_id, drop rejected domains, record degeneracy.
    seen_ids: set[str] = set()
    kept: list[ContinuationDomain] = []
    degenerate_flags: dict[str, bool] = {}
    for domain in domains:
        if domain.domain_id in seen_ids:
            continue
        seen_ids.add(domain.domain_id)
        if domain.state == DOMAIN_STATE_REJECTED:
            continue
        kept.append(domain)
        degenerate_flags[domain.domain_id] = domain.state != "valid"

    if len(kept) < 2:
        return []

    by_id = {domain.domain_id: domain for domain in kept}
    labels = [domain.domain_id for domain in kept]
    aabb_min = torch.stack([domain.aabb_min for domain in kept])
    aabb_max = torch.stack([domain.aabb_max for domain in kept])
    scales = [float(domain.local_surface_scale) for domain in kept]

    # Exclude same-source-boundary pairs from the broad phase up front.
    excluded: set[tuple[str, str]] = set()
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if kept[i].source_boundary_id == kept[j].source_boundary_id:
                pair = tuple(sorted([kept[i].domain_id, kept[j].domain_id]))
                excluded.add(pair)  # type: ignore[arg-type]

    broad_pairs = sweep_and_prune_pairs(
        labels, aabb_min, aabb_max, scales,
        expand_factor=broad_phase_expand_factor, tol=eps, excluded_pairs=excluded,
    )

    candidates: list[OccludedRegionCandidate] = []
    for pair in broad_pairs:
        domain_a = by_id[pair.label_a]
        domain_b = by_id[pair.label_b]
        pair_scale = max(float(domain_a.local_surface_scale), float(domain_b.local_surface_scale), eps)

        raw_edges = _raw_correspondence_edges(domain_a, domain_b, pair_scale, correspondence_threshold, eps)
        if not raw_edges:
            continue  # AABB overlap but no narrow-phase correspondence -> no candidate
        canonical = _canonical_one_to_one(raw_edges)
        if not canonical:
            continue
        both_closed = bool(domain_a.closed and domain_b.closed)
        components = _monotonic_components(canonical, int(domain_b.world.shape[0]), both_closed)

        boundary_a = boundaries_by_id.get(domain_a.source_boundary_id)
        boundary_b = boundaries_by_id.get(domain_b.source_boundary_id)

        for component in components:
            if not component:
                continue
            candidates.append(
                _component_candidate(
                    domain_a, domain_b, boundary_a, boundary_b, component,
                    pair_scale, both_closed, pair.payload(), degenerate_flags, eps,
                )
            )

    # Dedup identical candidate_ids deterministically (domain order reversal etc.).
    unique: dict[str, OccludedRegionCandidate] = {}
    for candidate in candidates:
        if candidate.candidate_id not in unique:
            unique[candidate.candidate_id] = candidate
    return sorted(unique.values(), key=lambda c: c.candidate_id)


def _aabb_overlaps(a_min: Any, a_max: Any, b_min: Any, b_max: Any) -> bool:
    torch = require_torch()
    return bool(torch.all(a_min <= b_max)) and bool(torch.all(b_min <= a_max))


def build_candidate_conflicts(candidates: Sequence[OccludedRegionCandidate]) -> list[ConflictEdge]:
    """Generate (but never resolve) conflict edges between candidates.

    Reads only the candidate dataclass fields (including the evidence-summary
    dicts populated by `torch_candidate_evidence`); does NOT import
    `torch_observation_evidence`. No ranking/selection/pruning.
    """

    torch = require_torch()
    conflicts: list[ConflictEdge] = []
    items = sorted(candidates, key=lambda c: c.candidate_id)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if not _aabb_overlaps(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max):
                continue
            reasons: list[str] = []
            shared_domain = set(a.supporting_domain_ids) & set(b.supporting_domain_ids)
            shared_boundary = set(a.supporting_boundary_ids) & set(b.supporting_boundary_ids)
            if shared_domain or shared_boundary:
                reasons.append("shared_source_overlapping_bridge")
            else:
                reasons.append("distinct_pairs_similar_bridge_space")
            # Rule 3: evidence summaries clearly incompatible in the same region.
            a_free = bool(a.free_space_contradiction.get("candidate_hard_contradiction", False))
            b_behind = float(b.behind_observation_support.get("behind_support_count", 0) or 0) > 0
            b_free = bool(b.free_space_contradiction.get("candidate_hard_contradiction", False))
            a_behind = float(a.behind_observation_support.get("behind_support_count", 0) or 0) > 0
            if (a_free and b_behind) or (b_free and a_behind):
                reasons.append("evidence_incompatible_same_region")
            conflicts.append(
                ConflictEdge(
                    candidate_a=a.candidate_id,
                    candidate_b=b.candidate_id,
                    reason=";".join(reasons),
                    provenance={
                        "shared_domain_ids": sorted(shared_domain),
                        "shared_boundary_ids": sorted(shared_boundary),
                    },
                )
            )
    return conflicts
