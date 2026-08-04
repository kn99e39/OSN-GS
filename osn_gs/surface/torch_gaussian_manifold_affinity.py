from __future__ import annotations

"""Covariance-guided same-surface adjacency graph (worklog 111/113/114).

Canonical principle: spatial proximity alone never creates a same-surface
edge. Worklog 114 further separates CANDIDATE GENERATION (a purely
geometric/scale question: "is this pair even worth evaluating") from
MANIFOLD RELATION CLASSIFICATION (a covariance-evidence question: "given
that it's a candidate, what is the relationship"). A single ``edge_state``
used to fold candidate distance, endpoint reliability, and manifold relation
into one priority chain; the four axes below (candidate status, endpoint
structural status, manifold relation, relation confidence) are now
orthogonal and independently reported, with the old ``state`` kept as a
backward-compatible projection.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame, orientation_insensitive_alignment
from osn_gs.surface.torch_gaussian_structural_reliability import (
    CONTEXTUAL_CONSISTENT,
    INTRINSIC_REJECTED,
    StructuralReliabilityResult,
)
from osn_gs.utils.torch_ops import require_torch

# --- Manifold relation (worklog 111/113 vocabulary, now orthogonal to candidate/endpoint status) ---
RELATION_SAME_SURFACE = "same_surface"
RELATION_CREASE = "crease_or_orientation_discontinuity"
RELATION_PARALLEL_SEPARATE = "parallel_but_separate"
RELATION_PROXIMITY_ONLY = "proximity_only"
RELATION_AMBIGUOUS = "ambiguous"
RELATION_REJECTED = "rejected"
RELATION_NOT_EVALUATED = "not_evaluated"

# Backward-compatible aliases (worklog 111/113 names) -- same string values.
EDGE_SAME_SURFACE = RELATION_SAME_SURFACE
EDGE_CREASE = RELATION_CREASE
EDGE_PARALLEL_SEPARATE = RELATION_PARALLEL_SEPARATE
EDGE_PROXIMITY_ONLY = RELATION_PROXIMITY_ONLY
EDGE_AMBIGUOUS = RELATION_AMBIGUOUS
EDGE_REJECTED = RELATION_REJECTED

# --- Candidate status (worklog 114 짠4) ---
CANDIDATE_STATUS_CANDIDATE = "candidate"
CANDIDATE_STATUS_OUTSIDE_SUPPORT = "outside_candidate_support"
CANDIDATE_STATUS_CAPPED_OUT = "capped_out"
CANDIDATE_STATUS_INVALID_ENDPOINT = "invalid_endpoint"
CANDIDATE_STATUS_NOT_EVALUATED = "not_evaluated"

CANDIDATE_REASON_MUTUAL_KNN = "mutual_knn"
CANDIDATE_REASON_WITHIN_RADIUS = "within_scale_radius"
CANDIDATE_REASON_FOOTPRINT_OVERLAP = "footprint_overlap"
CANDIDATE_REASON_DISTANCE_ONLY = "distance_only"
CANDIDATE_REASON_DETERMINISTIC_CAP = "deterministic_cap"

# --- Endpoint structural status (worklog 114 짠5) ---
ENDPOINT_BOTH_RELIABLE = "both_intrinsically_reliable"
ENDPOINT_ONE_UNRELIABLE = "one_intrinsically_unreliable"
ENDPOINT_BOTH_UNRELIABLE = "both_intrinsically_unreliable"
ENDPOINT_CONTEXTUAL_AMBIGUITY = "contextual_ambiguity_present"

# --- Relation confidence ---
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_NOT_APPLICABLE = "not_applicable"

NODE_INTERIOR_CONTINUATION = "interior_continuation"
NODE_CREASE_BOUNDARY_CANDIDATE = "crease_boundary_candidate"
NODE_OBSERVED_SUPPORT_BOUNDARY_CANDIDATE = "observed_support_boundary_candidate"
NODE_UNRESOLVED_BOUNDARY = "unresolved_boundary"


@dataclass(frozen=True)
class ManifoldAffinityConfig:
    """Configurable policy, not a confirmed canonical threshold set."""

    candidate_neighbor_count: int = 8
    max_candidate_count_per_node: int = 12
    # Candidate geometric gates -- independent signals, ANY passing makes a
    # kNN pair an actual "candidate" rather than merely "distance_only".
    scale_radius_multiplier: float = 6.0  # vs. average tangent_major_scale
    footprint_overlap_multiplier: float = 2.5  # vs. sum of equivalent_tangent_scale
    require_mutual_knn: bool = False  # if True, non-mutual kNN pairs are downgraded to outside_candidate_support
    same_surface_min_normal_alignment: float = 0.85
    same_surface_max_mutual_residual: float = 0.35
    crease_max_normal_alignment: float = 0.5
    crease_max_normalized_distance: float = 2.5
    parallel_separate_min_normal_alignment: float = 0.85
    parallel_separate_max_mutual_residual: float = 3.0
    # Close-parallel-surface detection uses the NORMAL-direction gap
    # normalized by normal thickness (not tangent scale) -- worklog 114 짠3/짠6.
    parallel_separate_min_normal_gap_over_thickness: float = 1.5
    # Oversized-planar-bridge guard: a candidate whose footprint is much
    # larger than its neighbor's is never allowed to become same_surface,
    # regardless of how well its normal/residual otherwise line up.
    max_footprint_ratio_for_same_surface: float = 4.0


@dataclass(frozen=True)
class PairAffinityMetrics:
    """All independently-preserved pairwise metrics (worklog 114 짠6). None of
    these alone decides ``manifold_relation`` -- see ``_classify_relation``."""

    normal_alignment: float
    mutual_tangent_residual: float
    tangent_direction_displacement_ratio: float
    normal_direction_separation_over_thickness: float
    tangent_footprint_ratio: float
    tangent_anisotropy_ratio: float
    normal_thickness_ratio: float
    neighbor_spacing_normalized_distance: float
    local_curvature_change_proxy: float
    normalized_distance: float

    def payload(self) -> dict[str, Any]:
        return dict(
            normal_alignment=self.normal_alignment,
            mutual_tangent_residual=self.mutual_tangent_residual,
            tangent_direction_displacement_ratio=self.tangent_direction_displacement_ratio,
            normal_direction_separation_over_thickness=self.normal_direction_separation_over_thickness,
            tangent_footprint_ratio=self.tangent_footprint_ratio,
            tangent_anisotropy_ratio=self.tangent_anisotropy_ratio,
            normal_thickness_ratio=self.normal_thickness_ratio,
            neighbor_spacing_normalized_distance=self.neighbor_spacing_normalized_distance,
            local_curvature_change_proxy=self.local_curvature_change_proxy,
            normalized_distance=self.normalized_distance,
        )


@dataclass(frozen=True)
class ManifoldAffinityEdge:
    source: int
    target: int
    source_id: Any
    target_id: Any
    candidate_status: str
    candidate_reasons: tuple[str, ...]
    endpoint_status: str
    manifold_relation: str
    relation_confidence: str
    relation_reason: str
    metrics: PairAffinityMetrics | None

    # --- backward-compatible flat fields (worklog 111/113 names) ---
    @property
    def state(self) -> str:
        # Worklog 111/113 had one axis: a candidate lacking enough scale
        # support to evaluate any relation fell back to "proximity_only".
        # Worklog 114 splits that into (candidate_status=outside_support/
        # capped_out, manifold_relation=not_evaluated) so the CAUSE is kept;
        # this projection restores the old single-bucket meaning for callers
        # that only read `.state`.
        if self.manifold_relation == RELATION_NOT_EVALUATED:
            return RELATION_PROXIMITY_ONLY
        return self.manifold_relation

    @property
    def reason(self) -> str:
        return self.relation_reason

    @property
    def confidence(self) -> str:
        return self.relation_confidence

    @property
    def normal_alignment(self) -> float:
        return self.metrics.normal_alignment if self.metrics is not None else 0.0

    @property
    def mutual_tangent_residual(self) -> float:
        return self.metrics.mutual_tangent_residual if self.metrics is not None else 0.0

    @property
    def normalized_distance(self) -> float:
        return self.metrics.normalized_distance if self.metrics is not None else float("inf")

    def payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "candidate_status": self.candidate_status,
            "candidate_reasons": list(self.candidate_reasons),
            "endpoint_status": self.endpoint_status,
            "manifold_relation": self.manifold_relation,
            "relation_confidence": self.relation_confidence,
            "relation_reason": self.relation_reason,
            "metrics": self.metrics.payload() if self.metrics is not None else None,
            # compatibility projection
            "state": self.state,
            "reason": self.reason,
            "confidence": self.confidence,
            "normal_alignment": self.normal_alignment,
            "mutual_tangent_residual": self.mutual_tangent_residual,
            "normalized_distance": self.normalized_distance,
        }


@dataclass(frozen=True)
class ManifoldAffinityGraph:
    edges: tuple[ManifoldAffinityEdge, ...]
    config: ManifoldAffinityConfig

    def payload(self) -> dict[str, Any]:
        return {"edges": [edge.payload() for edge in self.edges]}

    def same_surface_neighbors(self, count: int) -> list[set[int]]:
        neighbors: list[set[int]] = [set() for _ in range(count)]
        for edge in self.edges:
            if edge.manifold_relation == RELATION_SAME_SURFACE:
                neighbors[edge.source].add(edge.target)
                neighbors[edge.target].add(edge.source)
        return neighbors


def _classify_endpoint_status(
    intrinsic_a: str, intrinsic_b: str, contextual_a: str, contextual_b: str
) -> str:
    a_unreliable = intrinsic_a == INTRINSIC_REJECTED
    b_unreliable = intrinsic_b == INTRINSIC_REJECTED
    if a_unreliable and b_unreliable:
        return ENDPOINT_BOTH_UNRELIABLE
    if a_unreliable or b_unreliable:
        return ENDPOINT_ONE_UNRELIABLE
    if contextual_a != CONTEXTUAL_CONSISTENT or contextual_b != CONTEXTUAL_CONSISTENT:
        return ENDPOINT_CONTEXTUAL_AMBIGUITY
    return ENDPOINT_BOTH_RELIABLE


def _compute_pair_metrics(
    positions: Any, frame: GaussianCovarianceFrame, neighbor_spacing: Any,
    candidate_scale: Any, residual_scale: Any, a: int, b: int,
) -> PairAffinityMetrics:
    """``candidate_scale``/``residual_scale`` (worklog 33, ``(N,)`` each): two
    INDEPENDENT per-node scale roles (section 9 ablation).
    ``candidate_scale`` normalizes ``normalized_distance`` (the crease gate's
    distance-vs-local-scale ratio -- conceptually part of "is this pair even
    worth evaluating", same role as the caller's ``within_radius`` check).
    ``residual_scale`` normalizes ``mutual_tangent_residual`` and
    ``tangent_direction_displacement_ratio`` (the same_surface/parallel
    relation decision). Both default to ``frame.tangent_major_scale``
    (Gaussian Footprint Scale) when the caller passes it unchanged,
    preserving prior behavior exactly.
    """
    torch = require_torch()
    displacement = positions[b] - positions[a]
    distance = float(torch.linalg.vector_norm(displacement))

    candidate_scale_a, candidate_scale_b = candidate_scale[a], candidate_scale[b]
    average_candidate_scale = float((candidate_scale_a + candidate_scale_b) / 2.0)
    normalized_distance = distance / max(average_candidate_scale, 1e-12)

    normal_alignment = float((frame.normal_candidate[a] * frame.normal_candidate[b]).sum().abs())

    residual_scale_a, residual_scale_b = residual_scale[a], residual_scale[b]
    residual_from_a = float((displacement * frame.normal_candidate[a]).sum().abs() / residual_scale_a.clamp_min(1e-12))
    residual_from_b = float((-displacement * frame.normal_candidate[b]).sum().abs() / residual_scale_b.clamp_min(1e-12))
    mutual_tangent_residual = max(residual_from_a, residual_from_b)

    average_residual_scale = float((residual_scale_a + residual_scale_b) / 2.0)
    tangent_component = displacement - (displacement * frame.normal_candidate[a]).sum() * frame.normal_candidate[a]
    tangent_direction_displacement_ratio = float(torch.linalg.vector_norm(tangent_component)) / max(average_residual_scale, 1e-12)

    # Covariance normals are unoriented lines. Align before averaging so an
    # eigensolver sign flip cannot collapse the pair normal to zero.
    normal_a = frame.normal_candidate[a]
    normal_b = frame.normal_candidate[b]
    if float(normal_a @ normal_b) < 0.0:
        normal_b = -normal_b
    average_normal = torch.nn.functional.normalize(normal_a + normal_b, dim=0)
    normal_thickness_a, normal_thickness_b = frame.normal_thickness[a], frame.normal_thickness[b]
    average_thickness = float((normal_thickness_a + normal_thickness_b) / 2.0)
    normal_direction_gap = float((displacement * average_normal).sum().abs())
    normal_direction_separation_over_thickness = normal_direction_gap / max(average_thickness, 1e-12)

    footprint_a, footprint_b = float(frame.equivalent_tangent_scale[a]), float(frame.equivalent_tangent_scale[b])
    tangent_footprint_ratio = max(footprint_a, footprint_b) / max(min(footprint_a, footprint_b), 1e-12)

    elongation_a, elongation_b = float(frame.elongation[a]), float(frame.elongation[b])
    tangent_anisotropy_ratio = max(elongation_a, elongation_b) / max(min(elongation_a, elongation_b), 1e-12)

    thickness_ratio = max(float(normal_thickness_a), float(normal_thickness_b)) / max(min(float(normal_thickness_a), float(normal_thickness_b)), 1e-12)

    average_spacing = float((neighbor_spacing[a] + neighbor_spacing[b]) / 2.0)
    neighbor_spacing_normalized_distance = distance / max(average_spacing, 1e-12)

    local_curvature_change_proxy = (1.0 - normal_alignment) / max(tangent_direction_displacement_ratio, 1e-6)

    return PairAffinityMetrics(
        normal_alignment=normal_alignment,
        mutual_tangent_residual=mutual_tangent_residual,
        tangent_direction_displacement_ratio=tangent_direction_displacement_ratio,
        normal_direction_separation_over_thickness=normal_direction_separation_over_thickness,
        tangent_footprint_ratio=tangent_footprint_ratio,
        tangent_anisotropy_ratio=tangent_anisotropy_ratio,
        normal_thickness_ratio=thickness_ratio,
        neighbor_spacing_normalized_distance=neighbor_spacing_normalized_distance,
        local_curvature_change_proxy=local_curvature_change_proxy,
        normalized_distance=normalized_distance,
    )


def _classify_relation(metrics: PairAffinityMetrics, config: ManifoldAffinityConfig) -> tuple[str, str]:
    """Manifold relation given ONLY the pairwise metrics -- endpoint/candidate
    status are applied by the caller before/after this."""
    # Oversized-planar-bridge guard: never same_surface across a footprint
    # mismatch this large, no matter how well normal/residual line up.
    if (
        metrics.normal_alignment >= config.same_surface_min_normal_alignment
        and metrics.mutual_tangent_residual <= config.same_surface_max_mutual_residual
    ):
        if metrics.tangent_footprint_ratio > config.max_footprint_ratio_for_same_surface:
            return RELATION_PARALLEL_SEPARATE, "aligned_and_close_but_footprint_mismatch_too_large_for_same_surface"
        return RELATION_SAME_SURFACE, "aligned_normal_and_low_tangent_residual"
    if (
        metrics.normal_alignment >= config.parallel_separate_min_normal_alignment
        and metrics.mutual_tangent_residual <= config.parallel_separate_max_mutual_residual
        and metrics.normal_direction_separation_over_thickness >= config.parallel_separate_min_normal_gap_over_thickness
    ):
        return RELATION_PARALLEL_SEPARATE, "aligned_normal_but_normal_direction_gap_exceeds_thickness"
    if metrics.normal_alignment <= config.crease_max_normal_alignment and metrics.normalized_distance <= config.crease_max_normalized_distance:
        return RELATION_CREASE, "orientation_discontinuity_near_shared_boundary"
    return RELATION_AMBIGUOUS, "inconclusive_normal_or_residual_evidence"


def build_manifold_affinity_graph(
    positions: Any,
    frame: GaussianCovarianceFrame,
    reliability: StructuralReliabilityResult,
    *,
    config: ManifoldAffinityConfig | None = None,
    ids: Sequence[Any] | None = None,
    candidate_scale: Any | None = None,
    residual_scale: Any | None = None,
) -> ManifoldAffinityGraph:
    """Classify candidate spatial-neighbor pairs into orthogonal (candidate
    status, endpoint status, manifold relation, confidence) axes.

    Candidate generation (worklog 114 짠4) combines kNN with scale-normalized
    radius and tangent-footprint-overlap checks -- a pure kNN hit that fails
    BOTH of those scale checks is ``outside_candidate_support`` (nothing
    behind it but index proximity), never silently treated as a real
    candidate. Manifold relation is computed from FRESH pairwise metrics
    (never the coarser whole-neighborhood-averaged reliability scores), so an
    intrinsically-reliable Gaussian sitting at a real crease -- whose own
    CONTEXTUAL consistency is legitimately "mixed" -- can still be classified
    ``crease_or_orientation_discontinuity`` rather than being blocked by its
    own endpoint status. ``ids`` are stable external identities (default:
    positional index) carried on every edge so results remain comparable
    across a shuffled input order.

    ``candidate_scale``/``residual_scale`` (worklog 33, ``(N,)`` each):
    independent REPRESENTATIVE GRAPH SCALE roles -- ``candidate_scale``
    drives the ``within_radius`` candidate-radius test (this function) and
    ``normalized_distance`` (crease gate, inside ``_compute_pair_metrics``);
    ``residual_scale`` drives ``mutual_tangent_residual``/
    ``tangent_direction_displacement_ratio`` (same_surface/parallel
    decision). Both default to ``frame.tangent_major_scale`` (Gaussian
    Footprint Scale) when omitted, preserving prior behavior exactly.
    ``footprint_overlap`` (a genuinely different candidate criterion -- do
    these two Gaussians' own splats physically overlap) always keeps using
    ``frame.equivalent_tangent_scale`` regardless of these overrides.
    """
    torch = require_torch()
    config = config or ManifoldAffinityConfig()
    positions = torch.as_tensor(positions)
    count = int(positions.shape[0])
    ids = tuple(ids) if ids is not None else tuple(range(count))
    if len(ids) != count:
        raise ValueError("ids must have the same length as positions.")

    neighbor_spacing = reliability.contextual.neighbor_spacing
    candidate_scale = candidate_scale if candidate_scale is not None else frame.tangent_major_scale
    residual_scale = residual_scale if residual_scale is not None else frame.tangent_major_scale
    equivalent_tangent_scale = frame.equivalent_tangent_scale

    k_candidates = max(1, min(config.candidate_neighbor_count, count - 1))
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    _, knn_indices = torch.topk(distances, k=k_candidates, largest=False, dim=1)
    knn_set = [set(row.tolist()) for row in knn_indices]

    intrinsic_class = reliability.intrinsic.intrinsic_class
    contextual_class = reliability.contextual.contextual_class

    seen: set[tuple[int, int]] = set()
    pending: list[tuple[int, int, float, bool, bool, bool]] = []
    for i in range(count):
        for j in knn_indices[i].tolist():
            key = (min(i, j), max(i, j))
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            a, b = key
            mutual = (b in knn_set[a]) and (a in knn_set[b])
            distance = float(distances[a, b])
            average_candidate_scale = float((candidate_scale[a] + candidate_scale[b]) / 2.0)
            within_radius = distance <= config.scale_radius_multiplier * max(average_candidate_scale, 1e-12)
            footprint_sum = float(equivalent_tangent_scale[a] + equivalent_tangent_scale[b])
            footprint_overlap = distance <= config.footprint_overlap_multiplier * max(footprint_sum, 1e-12)
            pending.append((a, b, distance, mutual, within_radius, footprint_overlap))

    # Deterministic-cap ordering (worklog 114 짠8): the per-node candidate cap
    # must never depend on which positional index happens to be walked first
    # (that leaks raw array order into the result). Sort scale-backed
    # candidates by (distance, stable id pair) BEFORE applying the cap, so the
    # same Gaussians produce the same accepted/capped-out split regardless of
    # how the input array was ordered or shuffled.
    pending.sort(key=lambda row: (row[2], min(ids[row[0]], ids[row[1]], key=str), max(ids[row[0]], ids[row[1]], key=str)))

    edges: list[ManifoldAffinityEdge] = []
    per_node_candidate_count = [0] * count
    for a, b, distance, mutual, within_radius, footprint_overlap in pending:
            candidate_reasons: list[str] = []
            if mutual:
                candidate_reasons.append(CANDIDATE_REASON_MUTUAL_KNN)
            if within_radius:
                candidate_reasons.append(CANDIDATE_REASON_WITHIN_RADIUS)
            if footprint_overlap:
                candidate_reasons.append(CANDIDATE_REASON_FOOTPRINT_OVERLAP)

            endpoint_status = _classify_endpoint_status(
                intrinsic_class[a], intrinsic_class[b], contextual_class[a], contextual_class[b]
            )

            # A pure index-proximity hit (kNN found it, but neither a scale
            # radius nor a footprint overlap backs it) is NOT a real
            # candidate -- proximity alone never creates same-surface evidence.
            is_scale_backed = within_radius or footprint_overlap
            if config.require_mutual_knn:
                is_scale_backed = is_scale_backed and mutual
            if not is_scale_backed:
                candidate_reasons.append(CANDIDATE_REASON_DISTANCE_ONLY)
                edges.append(
                    ManifoldAffinityEdge(
                        a, b, ids[a], ids[b], CANDIDATE_STATUS_OUTSIDE_SUPPORT, tuple(candidate_reasons),
                        endpoint_status, RELATION_NOT_EVALUATED, CONFIDENCE_NOT_APPLICABLE,
                        "outside_candidate_support", None,
                    )
                )
                continue
            if per_node_candidate_count[a] >= config.max_candidate_count_per_node or per_node_candidate_count[b] >= config.max_candidate_count_per_node:
                candidate_reasons.append(CANDIDATE_REASON_DETERMINISTIC_CAP)
                edges.append(
                    ManifoldAffinityEdge(
                        a, b, ids[a], ids[b], CANDIDATE_STATUS_CAPPED_OUT, tuple(candidate_reasons),
                        endpoint_status, RELATION_NOT_EVALUATED, CONFIDENCE_NOT_APPLICABLE,
                        "capped_out_by_deterministic_degree_limit", None,
                    )
                )
                continue
            per_node_candidate_count[a] += 1
            per_node_candidate_count[b] += 1

            metrics = _compute_pair_metrics(positions, frame, neighbor_spacing, candidate_scale, residual_scale, a, b)
            if endpoint_status in (ENDPOINT_ONE_UNRELIABLE, ENDPOINT_BOTH_UNRELIABLE):
                edges.append(
                    ManifoldAffinityEdge(
                        a, b, ids[a], ids[b], CANDIDATE_STATUS_INVALID_ENDPOINT, tuple(candidate_reasons),
                        endpoint_status, RELATION_REJECTED, CONFIDENCE_LOW, "unreliable_endpoint", metrics,
                    )
                )
                continue

            relation, relation_reason = _classify_relation(metrics, config)
            confidence = CONFIDENCE_HIGH if endpoint_status == ENDPOINT_BOTH_RELIABLE else CONFIDENCE_MEDIUM
            edges.append(
                ManifoldAffinityEdge(
                    a, b, ids[a], ids[b], CANDIDATE_STATUS_CANDIDATE, tuple(candidate_reasons),
                    endpoint_status, relation, confidence, relation_reason, metrics,
                )
            )

    return ManifoldAffinityGraph(tuple(edges), config)


def classify_node_boundary_status(
    count: int,
    graph: ManifoldAffinityGraph,
    reliability: StructuralReliabilityResult,
    *,
    min_same_surface_neighbors_for_interior: int = 3,
) -> tuple[str, ...]:
    """Light, EXPERIMENTAL per-node boundary-status diagnostic (worklog 111/113/114).

    This is NOT an ordered world-space boundary loop/half-edge graph -- it
    only labels each node from its already-classified same_surface/crease
    edges, as a starting diagnostic for where a future ordered-loop recovery
    stage would need to operate.
    """
    same_surface = graph.same_surface_neighbors(count)
    has_crease = [False] * count
    for edge in graph.edges:
        if edge.manifold_relation == RELATION_CREASE:
            has_crease[edge.source] = True
            has_crease[edge.target] = True

    statuses = []
    for index in range(count):
        if reliability.intrinsic.intrinsic_class[index] == INTRINSIC_REJECTED:
            statuses.append(NODE_UNRESOLVED_BOUNDARY)
            continue
        same_surface_count = len(same_surface[index])
        if has_crease[index] and same_surface_count > 0:
            statuses.append(NODE_CREASE_BOUNDARY_CANDIDATE)
        elif same_surface_count >= min_same_surface_neighbors_for_interior:
            statuses.append(NODE_INTERIOR_CONTINUATION)
        elif same_surface_count > 0:
            statuses.append(NODE_OBSERVED_SUPPORT_BOUNDARY_CANDIDATE)
        else:
            statuses.append(NODE_UNRESOLVED_BOUNDARY)
    return tuple(statuses)


@dataclass(frozen=True)
class SameSurfaceRegionDiagnostic:
    """Connected-region diagnostic over ``same_surface`` edges ONLY (worklog 114 짠11).

    This is explicitly a ROBUSTNESS/EVALUATION diagnostic, not a boundary
    graph or production chart segmentation. Crease edges are never used to
    merge regions here -- a crease is evidence of a SEPARATION, not a
    same-surface connection.
    """

    region_id: tuple[int, ...]  # per-node region id, -1 if not in any same_surface region
    region_count: int
    reliable_node_coverage: float  # fraction of intrinsically-reliable nodes assigned to some region
    ambiguous_node_attachment: int  # count of non-reliable-but-non-rejected nodes attached to a region
    rejected_node_excluded_count: int
    region_sizes: tuple[int, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "region_id": list(self.region_id),
            "region_count": self.region_count,
            "reliable_node_coverage": self.reliable_node_coverage,
            "ambiguous_node_attachment": self.ambiguous_node_attachment,
            "rejected_node_excluded_count": self.rejected_node_excluded_count,
            "region_sizes": list(self.region_sizes),
        }


def diagnose_same_surface_regions(
    count: int, graph: ManifoldAffinityGraph, reliability: StructuralReliabilityResult
) -> SameSurfaceRegionDiagnostic:
    same_surface = graph.same_surface_neighbors(count)
    region_id = [-1] * count
    region_sizes: list[int] = []
    for start in range(count):
        if region_id[start] != -1 or not same_surface[start]:
            continue
        current_region = len(region_sizes)
        frontier = [start]
        region_id[start] = current_region
        size = 0
        while frontier:
            node = frontier.pop()
            size += 1
            for neighbor in same_surface[node]:
                if region_id[neighbor] == -1:
                    region_id[neighbor] = current_region
                    frontier.append(neighbor)
        region_sizes.append(size)

    reliable_mask = [c == "intrinsic_reliable" for c in reliability.intrinsic.intrinsic_class]
    rejected_mask = [c == INTRINSIC_REJECTED for c in reliability.intrinsic.intrinsic_class]
    reliable_count = sum(reliable_mask)
    reliable_covered = sum(1 for i in range(count) if reliable_mask[i] and region_id[i] != -1)
    ambiguous_attached = sum(1 for i in range(count) if not reliable_mask[i] and not rejected_mask[i] and region_id[i] != -1)

    return SameSurfaceRegionDiagnostic(
        region_id=tuple(region_id),
        region_count=len(region_sizes),
        reliable_node_coverage=(reliable_covered / reliable_count) if reliable_count else 0.0,
        ambiguous_node_attachment=ambiguous_attached,
        rejected_node_excluded_count=sum(rejected_mask),
        region_sizes=tuple(region_sizes),
    )
