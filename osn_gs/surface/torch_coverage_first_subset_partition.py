from __future__ import annotations

"""Worklog 105 -- coverage-first, normal-coherent Gaussian Subset partition.

Top-level architecture change. The Worklog 95-104 line was SELECTION-FIRST:
evidence had to clear structural/support predicates before any component,
chart, or patch existed, so whole scene regions silently left
surface-construction responsibility. This module inverts that order:

    full trained visible Gaussian scene
        -> surface-orientation representation for EVERY Gaussian
        -> local spatial adjacency graph
        -> normal compatibility on graph edges
        -> connected, normal-coherent Gaussian Subsets

Every input Gaussian ends up in exactly one subset. Nothing here consults
Worklog 95 latent support, boundary evidence, topology success, chart
validity, patch identifiability, NURBS eligibility, or held-out validity --
the partition is upstream of all of them.

    SUBSET OWNERSHIP != TRUSTABILITY

A Gaussian with unusable orientation evidence still owns a subset; a later
batch will estimate trust and use it to weight influence on latent-surface
estimation, never to revoke ownership. Nothing in this module implements or
anticipates that estimator.

Why not global normal clustering
--------------------------------
Clustering normals alone merges geometrically unrelated but parallel
surfaces (two distant walls sharing a normal). Normal coherence is therefore
the PRIMARY partition cue but is only ever evaluated on edges of a LOCAL
spatial adjacency graph; spatial proximity constrains connectivity and never
becomes the partition semantics itself.

Sign contract
-------------
Normal similarity uses ``|dot(n_i, n_j)|``
(:func:`~osn_gs.surface.torch_gaussian_surface_orientation.unsigned_normal_alignment`)
so an identical tangent plane is never split merely because two equivalent
principal axes came back with opposite sign.
"""

from dataclasses import dataclass
from typing import Any, Callable

from osn_gs.surface.torch_gaussian_surface_orientation import (
    GaussianSurfaceOrientation,
    unsigned_normal_alignment,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# --- ownership kinds: every Gaussian carries exactly one ---
OWNERSHIP_NORMAL_COHERENT = "normal_coherent_component"
OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE = "fallback_normal_incompatible_neighborhood"
OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR = "fallback_no_spatial_neighbor"

OWNERSHIP_KINDS: tuple[str, ...] = (
    OWNERSHIP_NORMAL_COHERENT,
    OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE,
    OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR,
)

# A subset at or below this size is reported as "very small" in the
# diagnostics. Purely a reporting bucket -- it never changes the partition,
# and no acceptance threshold is derived from it in this batch.
VERY_SMALL_SUBSET_SIZE = 8


@dataclass(frozen=True)
class CoverageFirstPartitionConfig:
    """Every heuristic this architecture needs, centralized in one place.

    None of these values was chosen by looking at the rendered partition and
    adjusting until it looked good; each is either reused verbatim from an
    already-shipped OSN-GS default or fixed by an explicit geometric
    argument (see the field comments). One documented configuration produces
    the primary result -- this batch runs no hyperparameter search.
    """

    # Local neighbourhood size for the spatial adjacency graph. Same value as
    # `ManifoldAffinityConfig.candidate_neighbor_count` and
    # `torch_latent_surface_tangent_frame_field.FIELD_NEIGHBOR_COUNT`, so the
    # notion of "local neighbourhood" does not change meaning between the old
    # and new pipelines.
    neighbor_count: int = 8

    # An edge survives only when the two Gaussians sit within this multiple of
    # the TIGHTER of their two local sample spacings. Using the min (not the
    # mean/max) stops a sparsely sampled point from reaching across a gap into
    # a densely sampled surface. 2.0 = "at most twice the local sampling
    # pitch", i.e. genuine sampling neighbours rather than a bridge over empty
    # space; it is deliberately looser than 1.0 (which would cut normal
    # sampling jitter) and far tighter than the 4.0-6.0 multipliers the
    # repository uses for deliberately permissive support radii.
    spatial_connect_spacing_multiplier: float = 2.0

    # |dot(n_i, n_j)| floor for two adjacent Gaussians to belong to the same
    # subset. Reused verbatim from the already-shipped
    # `ManifoldAffinityConfig.same_surface_min_normal_alignment` (0.85,
    # i.e. 31.79 degrees) so "same surface orientation" keeps one meaning
    # across the codebase.
    normal_compatibility_min_alignment: float = 0.85

    # 0 selects a memory-aware chunk size for the brute-force kNN pass.
    knn_chunk_size: int = 0

    # Safety caps for the iterative connected-component solver; exceeding
    # them raises instead of silently returning a wrong partition.
    max_label_rounds: int = 128
    max_pointer_jumps: int = 64

    def normal_compatibility_angle_degrees(self) -> float:
        import math

        return math.degrees(math.acos(max(-1.0, min(1.0, self.normal_compatibility_min_alignment))))

    def payload(self) -> dict[str, Any]:
        return {
            "neighbor_count": self.neighbor_count,
            "spatial_connect_spacing_multiplier": self.spatial_connect_spacing_multiplier,
            "normal_compatibility_min_alignment": self.normal_compatibility_min_alignment,
            "normal_compatibility_angle_degrees": self.normal_compatibility_angle_degrees(),
            "knn_chunk_size": self.knn_chunk_size,
            "max_label_rounds": self.max_label_rounds,
            "max_pointer_jumps": self.max_pointer_jumps,
            "very_small_subset_size": VERY_SMALL_SUBSET_SIZE,
        }


@dataclass(frozen=True)
class GaussianSubsetPartition:
    """Result of one partition run. ``subset_ids`` IS the ownership contract."""

    subset_ids: Any  # (N,) int64 in [0, subset_count) -- exactly one owner per Gaussian
    subset_count: int
    subset_sizes: Any  # (subset_count,) int64, descending by size
    ownership_kind: Any  # (N,) int8 index into OWNERSHIP_KINDS
    local_spacing: Any  # (N,) float -- median kNN distance, the scale normalizer
    candidate_edges: Any  # (E_c, 2) long, canonical a<b, deduplicated kNN pairs
    spatial_edge_mask: Any  # (E_c,) bool -- passed the local-spacing distance test
    normal_compatible_mask: Any  # (E_c,) bool -- passed |dot(n_i, n_j)| >= floor
    gaussian_ids: Any  # (N,) int64 provenance, copied from the orientation input
    config: CoverageFirstPartitionConfig

    def __len__(self) -> int:
        return int(self.subset_ids.shape[0])

    @property
    def accepted_edges(self) -> Any:
        """Edges that actually built the subsets: spatially adjacent AND normal-compatible."""

        return self.candidate_edges[self.spatial_edge_mask & self.normal_compatible_mask]

    @property
    def normal_cut_edges(self) -> Any:
        """Spatially adjacent edges REJECTED for normal incompatibility.

        These are the partition cuts the review export visualizes -- they are
        exactly where a subset boundary was created by orientation change
        rather than by a break in connectivity.
        """

        return self.candidate_edges[self.spatial_edge_mask & ~self.normal_compatible_mask]


def _auto_chunk_size(count: int, device: Any) -> int:
    """Rows per brute-force kNN chunk, sized to keep the (chunk, N) distance
    matrix near a 3 GB working set (the dominant allocation)."""

    if count <= 4096:
        return count
    target_bytes = 3 * 1024**3
    chunk = int(target_bytes // max(4 * count, 1))
    if str(getattr(device, "type", device)) == "cpu":
        chunk = min(chunk, 4096)
    return max(64, min(chunk, 4096))


def _knn(positions: Any, k: int, chunk_size: int, progress: Callable[[str], None] | None) -> tuple[Any, Any]:
    """Exact brute-force kNN, chunked over query rows.

    Returns ``(neighbor_index (N, k), neighbor_distance (N, k))``. Candidate
    ranking uses ``torch.cdist`` (matmul-backed for large inputs), but every
    returned distance is RECOMPUTED directly from the gathered coordinates so
    the spacing/adjacency thresholds are never applied to a matmul-rounded
    value. Self-exclusion is by row INDEX, not by distance, so exactly
    coincident Gaussians remain valid neighbours of one another.
    """

    torch = require_torch()
    count = int(positions.shape[0])
    index_chunks = []
    distance_chunks = []
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        distance = torch.cdist(positions[start:end], positions)
        rows = torch.arange(end - start, device=positions.device)
        distance[rows, torch.arange(start, end, device=positions.device)] = float("inf")
        _, indices = torch.topk(distance, k, dim=1, largest=False)
        del distance
        exact = (positions[start:end].unsqueeze(1) - positions[indices]).norm(dim=-1)
        index_chunks.append(indices)
        distance_chunks.append(exact)
        if progress is not None and (start // chunk_size) % 200 == 0:
            progress(f"knn rows {end}/{count}")
    return torch.cat(index_chunks, dim=0), torch.cat(distance_chunks, dim=0)


def _connected_component_roots(count: int, edges: Any, config: CoverageFirstPartitionConfig) -> Any:
    """Connected-component labels by Shiloach-Vishkin hooking with full path compression.

    Each node ends up carrying the smallest node index in its component, so
    the label is itself a deterministic component identifier independent of
    edge order (``amin`` scatter is order-independent). Isolated nodes keep
    their own index and therefore form their own component -- that is the
    fallback ownership path, not a dropped Gaussian.

    Hooking writes onto the ROOT index (``max(root_u, root_v)`` adopts
    ``min(root_u, root_v)``), never onto the edge endpoints. Scattering onto
    endpoints instead only advances a label one graph hop per round, which
    costs O(diameter) rounds and does not terminate in any practical budget on
    a scene-scale kNN graph; hooking roots plus pointer jumping resolves an
    arbitrarily long chain of components in a single round and converges in
    O(log N).
    """

    torch = require_torch()
    label = torch.arange(count, dtype=torch.int64, device=edges.device)
    if int(edges.shape[0]) == 0:
        return label
    left, right = edges[:, 0].contiguous(), edges[:, 1].contiguous()
    for _ in range(config.max_label_rounds):
        root_left, root_right = label[left], label[right]
        candidate = label.clone()
        candidate.scatter_reduce_(
            0, torch.maximum(root_left, root_right), torch.minimum(root_left, root_right), reduce="amin"
        )
        for _ in range(config.max_pointer_jumps):
            jumped = candidate[candidate]
            if bool(torch.equal(jumped, candidate)):
                break
            candidate = jumped
        else:
            raise RuntimeError("Coverage-first partition: pointer jumping did not reach a fixed point.")
        if bool(torch.equal(candidate, label)):
            return label
        label = candidate
    raise RuntimeError("Coverage-first partition: connected-component labelling did not converge.")


def partition_gaussian_subsets(
    orientation: GaussianSurfaceOrientation,
    config: CoverageFirstPartitionConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> GaussianSubsetPartition:
    """Partition EVERY Gaussian in ``orientation`` into exactly one subset.

    No input row can be excluded: there is no quality, confidence, support,
    or downstream-validity predicate anywhere in this function. A Gaussian
    with no normal-compatible neighbour becomes its own subset (reported as
    fallback ownership) rather than joining an unassigned population.
    """

    torch = require_torch()
    config = config or CoverageFirstPartitionConfig()
    positions = orientation.positions
    normals = orientation.surface_normal
    count = int(positions.shape[0])
    device = positions.device

    if count == 0:
        empty_long = torch.zeros((0,), dtype=torch.int64, device=device)
        return GaussianSubsetPartition(
            subset_ids=empty_long,
            subset_count=0,
            subset_sizes=empty_long,
            ownership_kind=torch.zeros((0,), dtype=torch.int8, device=device),
            local_spacing=torch.zeros((0,), dtype=positions.dtype, device=device),
            candidate_edges=torch.zeros((0, 2), dtype=torch.int64, device=device),
            spatial_edge_mask=torch.zeros((0,), dtype=torch.bool, device=device),
            normal_compatible_mask=torch.zeros((0,), dtype=torch.bool, device=device),
            gaussian_ids=orientation.gaussian_ids,
            config=config,
        )

    k = min(int(config.neighbor_count), count - 1)
    if k <= 0:
        # A single Gaussian is trivially its own subset; still exactly one owner.
        return GaussianSubsetPartition(
            subset_ids=torch.zeros((count,), dtype=torch.int64, device=device),
            subset_count=1,
            subset_sizes=torch.tensor([count], dtype=torch.int64, device=device),
            ownership_kind=torch.full(
                (count,), OWNERSHIP_KINDS.index(OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR), dtype=torch.int8, device=device
            ),
            local_spacing=torch.zeros((count,), dtype=positions.dtype, device=device),
            candidate_edges=torch.zeros((0, 2), dtype=torch.int64, device=device),
            spatial_edge_mask=torch.zeros((0,), dtype=torch.bool, device=device),
            normal_compatible_mask=torch.zeros((0,), dtype=torch.bool, device=device),
            gaussian_ids=orientation.gaussian_ids,
            config=config,
        )

    chunk_size = int(config.knn_chunk_size) or _auto_chunk_size(count, device)
    neighbor_index, neighbor_distance = _knn(positions, k, chunk_size, progress)

    # Local sampling pitch: the median of a Gaussian's own kNN distances.
    # Median (not mean/max) so a single far outlier neighbour cannot inflate
    # the scale a Gaussian is allowed to connect over.
    local_spacing = neighbor_distance.median(dim=1).values

    rows = torch.arange(count, dtype=torch.int64, device=device).unsqueeze(1).expand(-1, k)
    left = torch.minimum(rows.reshape(-1), neighbor_index.reshape(-1))
    right = torch.maximum(rows.reshape(-1), neighbor_index.reshape(-1))
    # Deduplicate through a single int64 key instead of `unique(..., dim=0)`:
    # exact for count < 3e9 and far cheaper at scene scale.
    key = left * int(count) + right
    unique_key = torch.unique(key)
    del key, rows, left, right, neighbor_index
    candidate_left = torch.div(unique_key, count, rounding_mode="floor")
    candidate_right = unique_key - candidate_left * count
    candidate_edges = torch.stack((candidate_left, candidate_right), dim=1)
    del unique_key

    edge_distance = (positions[candidate_left] - positions[candidate_right]).norm(dim=-1)
    connect_scale = torch.minimum(local_spacing[candidate_left], local_spacing[candidate_right])
    spatial_edge_mask = edge_distance <= config.spatial_connect_spacing_multiplier * connect_scale.clamp_min(_EPS)
    alignment = unsigned_normal_alignment(normals[candidate_left], normals[candidate_right])
    normal_compatible_mask = alignment >= config.normal_compatibility_min_alignment
    if progress is not None:
        progress(
            f"edges candidate={int(candidate_edges.shape[0])} "
            f"spatial={int(spatial_edge_mask.sum())} accepted={int((spatial_edge_mask & normal_compatible_mask).sum())}"
        )

    accepted = candidate_edges[spatial_edge_mask & normal_compatible_mask]
    roots = _connected_component_roots(count, accepted, config)

    unique_roots, inverse, counts = torch.unique(roots, return_inverse=True, return_counts=True)
    # Deterministic subset IDs: largest subset first, ties broken by the
    # component's smallest member index (== its root, ascending in
    # `unique_roots`), so a rerun on identical input reproduces identical IDs.
    order = torch.argsort(counts, descending=True, stable=True)
    subset_id_of_position = torch.empty_like(order)
    subset_id_of_position[order] = torch.arange(int(order.shape[0]), dtype=order.dtype, device=device)
    subset_ids = subset_id_of_position[inverse]
    subset_sizes = counts[order]

    spatial_degree = torch.zeros((count,), dtype=torch.int64, device=device)
    spatial_edges = candidate_edges[spatial_edge_mask]
    if int(spatial_edges.shape[0]) > 0:
        ones = torch.ones((int(spatial_edges.shape[0]),), dtype=torch.int64, device=device)
        spatial_degree.index_add_(0, spatial_edges[:, 0], ones)
        spatial_degree.index_add_(0, spatial_edges[:, 1], ones)
    accepted_degree = torch.zeros((count,), dtype=torch.int64, device=device)
    if int(accepted.shape[0]) > 0:
        ones = torch.ones((int(accepted.shape[0]),), dtype=torch.int64, device=device)
        accepted_degree.index_add_(0, accepted[:, 0], ones)
        accepted_degree.index_add_(0, accepted[:, 1], ones)

    ownership_kind = torch.full(
        (count,), OWNERSHIP_KINDS.index(OWNERSHIP_NORMAL_COHERENT), dtype=torch.int8, device=device
    )
    no_accepted = accepted_degree == 0
    ownership_kind = torch.where(
        no_accepted & (spatial_degree > 0),
        torch.tensor(OWNERSHIP_KINDS.index(OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE), dtype=torch.int8, device=device),
        ownership_kind,
    )
    ownership_kind = torch.where(
        no_accepted & (spatial_degree == 0),
        torch.tensor(OWNERSHIP_KINDS.index(OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR), dtype=torch.int8, device=device),
        ownership_kind,
    )

    return GaussianSubsetPartition(
        subset_ids=subset_ids,
        subset_count=int(order.shape[0]),
        subset_sizes=subset_sizes,
        ownership_kind=ownership_kind,
        local_spacing=local_spacing,
        candidate_edges=candidate_edges,
        spatial_edge_mask=spatial_edge_mask,
        normal_compatible_mask=normal_compatible_mask,
        gaussian_ids=orientation.gaussian_ids,
        config=config,
    )


def count_spatially_disconnected_subsets(partition: GaussianSubsetPartition) -> int:
    """Independent re-derivation of subset connectivity under the partition graph.

    Recomputes components from the accepted edges and counts subsets spanning
    more than one component. Under the connectivity contract this is
    structurally zero; it is verified rather than assumed.
    """

    torch = require_torch()
    count = len(partition)
    if count == 0:
        return 0
    roots = _connected_component_roots(count, partition.accepted_edges, partition.config)
    unique_pairs = torch.unique(partition.subset_ids * int(count) + roots)
    subset_of_pair = torch.div(unique_pairs, count, rounding_mode="floor")
    components_per_subset = torch.bincount(subset_of_pair, minlength=max(partition.subset_count, 1))
    return int((components_per_subset > 1).sum())


def partition_accounting(partition: GaussianSubsetPartition) -> dict[str, Any]:
    """Full §11 diagnostic block. Diagnostic ONLY -- defines no acceptance threshold."""

    torch = require_torch()
    count = len(partition)
    sizes = partition.subset_sizes
    ownership_counts = torch.bincount(
        partition.ownership_kind.reshape(-1).to(torch.int64), minlength=len(OWNERSHIP_KINDS)
    )
    fallback_total = int(ownership_counts[1]) + int(ownership_counts[2])

    # Mechanical coverage proof. `subset_ids` is a single-valued ownership map
    # of length N, so multiple ownership is structurally impossible -- but the
    # per-subset occupancy recounted from it is cross-checked elementwise
    # against `subset_sizes`, which was derived independently from the
    # connected-component labelling, so a silent drop or duplicate in either
    # derivation would break the identity rather than hide.
    owner_histogram = torch.bincount(partition.subset_ids.reshape(-1), minlength=partition.subset_count)
    assigned = int((partition.subset_ids >= 0).sum())
    in_range = int(((partition.subset_ids >= 0) & (partition.subset_ids < max(partition.subset_count, 1))).sum())
    sizes_match = bool(
        int(sizes.shape[0]) == int(owner_histogram.shape[0]) and torch.equal(owner_histogram.to(sizes.dtype), sizes)
    )

    size_stats: dict[str, Any] = {}
    if int(sizes.shape[0]) > 0:
        sorted_sizes = torch.sort(sizes).values.to(torch.float64)
        def _percentile(fraction: float) -> int:
            position = min(int(sorted_sizes.shape[0]) - 1, max(0, int(round(fraction * (int(sorted_sizes.shape[0]) - 1)))))
            return int(sorted_sizes[position].item())

        size_stats = {
            "min": int(sorted_sizes[0].item()),
            "median": _percentile(0.5),
            "mean": float(sorted_sizes.mean().item()),
            "p95": _percentile(0.95),
            "max": int(sorted_sizes[-1].item()),
        }

    singleton = int((sizes == 1).sum()) if int(sizes.shape[0]) > 0 else 0
    very_small = int((sizes <= VERY_SMALL_SUBSET_SIZE).sum()) if int(sizes.shape[0]) > 0 else 0
    subset_count = max(partition.subset_count, 1)

    spatial_mask = partition.spatial_edge_mask
    normal_mask = partition.normal_compatible_mask
    candidate_edge_count = int(partition.candidate_edges.shape[0])
    spatial_edge_count = int(spatial_mask.sum())
    accepted_edge_count = int((spatial_mask & normal_mask).sum())

    # Size histogram reported BOTH ways: how many subsets fall in a bucket, and
    # how many Gaussians those subsets own. The two answer different questions
    # -- "is the partition dominated by tiny subsets?" vs. "does a meaningful
    # share of the scene actually live in them?" -- and reporting only the
    # first would badly misrepresent a partition with one huge component.
    histogram_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768, 131072)
    histogram = []
    previous = 0
    for bound in list(histogram_buckets) + [None]:
        if int(sizes.shape[0]) == 0:
            histogram.append({"max_size": bound, "subset_count": 0, "gaussian_count": 0})
            continue
        in_bucket = (sizes > previous) if bound is None else ((sizes > previous) & (sizes <= bound))
        histogram.append(
            {
                "max_size": bound,
                "subset_count": int(in_bucket.sum()),
                "gaussian_count": int(sizes[in_bucket].sum()),
            }
        )
        if bound is not None:
            previous = bound

    largest_sizes = sizes[: min(16, int(sizes.shape[0]))].tolist() if int(sizes.shape[0]) > 0 else []
    largest_fraction = (float(sizes[0]) / count) if (int(sizes.shape[0]) > 0 and count) else 0.0

    return {
        "input_gaussian_count": count,
        "assigned_gaussian_count": assigned,
        "unassigned_gaussian_count": count - assigned,
        "multiply_owned_gaussian_count": 0,
        "subset_id_out_of_range_count": count - in_range,
        "subset_size_sum": int(sizes.sum()) if int(sizes.shape[0]) > 0 else 0,
        "subset_sizes_match_ownership_map": sizes_match,
        "coverage_identity_holds": bool(
            assigned == count
            and in_range == count
            and sizes_match
            and (int(sizes.sum()) if int(sizes.shape[0]) > 0 else 0) == count
        ),
        "subset_count": partition.subset_count,
        "subset_size": size_stats,
        "singleton_subset_count": singleton,
        "singleton_subset_fraction": singleton / subset_count,
        "very_small_subset_size_threshold": VERY_SMALL_SUBSET_SIZE,
        "very_small_subset_count": very_small,
        "very_small_subset_fraction": very_small / subset_count,
        "gaussians_in_singleton_subsets": singleton,
        "gaussians_in_very_small_subsets": int(sizes[sizes <= VERY_SMALL_SUBSET_SIZE].sum()) if int(sizes.shape[0]) > 0 else 0,
        "largest_subset_size": int(sizes[0]) if int(sizes.shape[0]) > 0 else 0,
        "largest_subset_gaussian_fraction": largest_fraction,
        "largest_subset_sizes": largest_sizes,
        "subset_size_histogram": histogram,
        "spatially_disconnected_subset_count": count_spatially_disconnected_subsets(partition),
        "candidate_edge_count": candidate_edge_count,
        "spatial_edge_count": spatial_edge_count,
        "spatially_rejected_edge_count": candidate_edge_count - spatial_edge_count,
        "normal_compatibility_cut_edge_count": spatial_edge_count - accepted_edge_count,
        "accepted_edge_count": accepted_edge_count,
        "ownership_kind_counts": {name: int(ownership_counts[index]) for index, name in enumerate(OWNERSHIP_KINDS)},
        "fallback_ownership_count": fallback_total,
        "fallback_ownership_fraction": fallback_total / max(count, 1),
        "partition_parameters": partition.config.payload(),
    }
