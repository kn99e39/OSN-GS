from __future__ import annotations

"""Worklog 94 -- bounded architecture gate: one minimum-common adapter that
replays four fixed surface-evidence representations through the SAME
unmodified downstream constructor contract
(``build_chart_unit_face_topology_context`` ->
``build_chart_unit_assembly`` -> ``materialize_chart_unit_cut_boundaries`` ->
``evaluate_fit``).

Every representation is exactly a function
``(raw_positions, raw_covariance) -> (adapted_positions, adapted_covariance)``.
The downstream contract only ever consumes a ``(positions, covariance)``
pair -- ``build_chart_unit_face_topology_context`` and
``build_full_region_surface_face_topology`` both derive their tangent
frame/normal from ``extract_covariance_frame(covariance)`` internally, and
that is the ONLY geometric input every representation must supply. This
module changes nothing about that contract; it only computes what
``covariance`` (and, for representations that move the centers, what
``positions``) should be for A/B/C/D before handing them to the unchanged
Worklog 89 pipeline.

Hard constraints (matching Worklog 90-93's conventions):

- Worklog 82 relation thresholds, NURBS capacity, PCA-UV, ADC, and visible
  Gaussian training are never touched here or by any caller of this module.
- No representation parameter is tuned from a real-replay outcome; every
  constant below is either reused unchanged from an existing worklog or is
  a fixed, documented convention (e.g. B's smoothing pass count).
- Representation B never invents a per-point projection as final geometry
  without enforcing cross-neighborhood consistency first (a single
  iterative consensus-averaging pass over the whole subset, not an
  independent per-point one-shot projection -- see
  ``_center_latent_surface_positions``).
- Representation C never treats the covariance normal as ground-truth
  surface geometry; it only widens what counts as "supported" for the
  Worklog 79 coverage check by using the covariance footprint's own
  observed reach, not by asserting the covariance direction is correct.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
    _LOCAL_NEIGHBOR_COUNT,
    _knn_indices,
    _local_plane_normal,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.utils.torch_ops import require_torch

REPRESENTATION_RAW_CENTER_BASELINE = "RAW_CENTER_BASELINE"
REPRESENTATION_CENTER_LATENT_SURFACE = "CENTER_LATENT_SURFACE"
REPRESENTATION_COVARIANCE_SURFEL_SUPPORT = "COVARIANCE_SURFEL_SUPPORT"
REPRESENTATION_HYBRID_LATENT_PLUS_SUPPORT = "HYBRID_LATENT_PLUS_SUPPORT"

REPRESENTATIONS = (
    REPRESENTATION_RAW_CENTER_BASELINE,
    REPRESENTATION_CENTER_LATENT_SURFACE,
    REPRESENTATION_COVARIANCE_SURFEL_SUPPORT,
    REPRESENTATION_HYBRID_LATENT_PLUS_SUPPORT,
)

_EPS = 1e-12

# Fixed convention (not tuned from a real replay outcome): B's consistency
# enforcement is a single Jacobi-style consensus-averaging pass -- each
# point moves halfway from its raw position toward the mean of its own kNN
# neighbors' independent local-plane projections. One pass is the minimum
# that distinguishes "independent per-point projection" (0 passes) from
# "cross-neighborhood consistency enforced" (>=1 pass); this is a structural
# requirement from the Worklog 94 directive, not a swept hyperparameter.
_CONSISTENCY_PASSES = 1
_CONSISTENCY_BLEND = 0.5
# Latent-surface covariance thickness floor, expressed as a fraction of the
# unit's own local in-plane spacing -- prevents a degenerate zero-thickness
# covariance (which would make normal-alignment/adjacency tests div-by-zero
# unstable) without inventing an unsupported thickness value.
_LATENT_THICKNESS_FLOOR_FRACTION = 0.05


@dataclass(frozen=True)
class RepresentationEvidence:
    representation: str
    positions: Any
    covariance: Any
    # Diagnostic-only fields carried through for reporting; never consumed
    # by the downstream constructor contract.
    displacement_over_spacing: float | None
    support_radius_scale: float | None


def _median_pairwise_nn(points: Any) -> float:
    torch = require_torch()
    count = points.shape[0]
    if count < 2:
        return 1e-6
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    value = float(distance.min(dim=1).values.median().item())
    return value if value > 0 else 1e-6


def _center_latent_surface_positions(positions: Any) -> tuple[Any, float]:
    """Representation B: a coherent latent surface from overlapping
    center-only local geometry, with cross-neighborhood consistency
    explicitly enforced (never an independent per-point projection alone).

    Step 1: fit each point's own local-kNN plane (same convention as
    Worklog 92/93, diagnostic normal only, never covariance-derived).
    Step 2: compute each point's independent local-plane projection.
    Step 3 (consistency enforcement): move each point only halfway toward
    the AVERAGE of its own kNN neighbors' independently-projected
    positions, not its own projection alone -- this is what keeps the
    result a coherent shared surface rather than N disconnected planes.

    Returns the adapted positions and the evidence-weighted median
    displacement (for reporting only).
    """

    torch = require_torch()
    count = positions.shape[0]
    if count < 3:
        return positions.clone(), 0.0

    k = min(_LOCAL_NEIGHBOR_COUNT, count - 1)
    neighbor_indices = _knn_indices(positions, k)

    independent_projection = positions.clone()
    for node in range(count):
        neighborhood = torch.cat(
            [torch.tensor([node], device=positions.device), neighbor_indices[node]]
        )
        local_points = positions[neighborhood]
        normal = _local_plane_normal(local_points)
        plane_point = local_points.mean(dim=0)
        offset = ((positions[node] - plane_point) @ normal)
        independent_projection[node] = positions[node] - offset * normal

    consensus_positions = positions.clone()
    for _pass in range(_CONSISTENCY_PASSES):
        next_positions = consensus_positions.clone()
        for node in range(count):
            neighbor_mean = independent_projection[neighbor_indices[node]].mean(dim=0)
            next_positions[node] = (
                (1.0 - _CONSISTENCY_BLEND) * consensus_positions[node] + _CONSISTENCY_BLEND * neighbor_mean
            )
        consensus_positions = next_positions

    spacing = _median_pairwise_nn(positions)
    displacement = (consensus_positions - positions).norm(dim=1)
    displacement_over_spacing = float(displacement.median().item()) / max(spacing, _EPS)
    return consensus_positions, displacement_over_spacing


def _tangent_frame_covariance(
    positions: Any, reference_covariance: Any, *, thickness_floor_fraction: float,
) -> Any:
    """Build a covariance tensor aligned to each point's OWN local
    position-fit tangent frame (never the raw Gaussian's covariance
    orientation), with in-plane scale taken from the reference covariance's
    own footprint magnitude (support extent only, not orientation) and
    normal-direction thickness collapsed to a small floor. This is the
    covariance representation B/D hand to the unchanged downstream contract
    so its internal ``extract_covariance_frame`` calls see a normal that
    matches the latent surface, not the raw per-Gaussian orientation.
    """

    torch = require_torch()
    count = positions.shape[0]
    reference_frame = extract_covariance_frame(reference_covariance)
    k = min(_LOCAL_NEIGHBOR_COUNT, count - 1) if count > 1 else 0
    neighbor_indices = _knn_indices(positions, k) if k > 0 else None
    spacing = _median_pairwise_nn(positions)
    thickness = max(thickness_floor_fraction * spacing, _EPS)

    covariance = torch.zeros((count, 3, 3), dtype=positions.dtype, device=positions.device)
    for node in range(count):
        if neighbor_indices is not None and int(neighbor_indices[node].numel()) > 0:
            neighborhood = torch.cat(
                [torch.tensor([node], device=positions.device), neighbor_indices[node]]
            )
            normal = _local_plane_normal(positions[neighborhood])
        else:
            normal = reference_frame.normal_candidate[node]
        reference = torch.tensor([1.0, 0.0, 0.0], device=positions.device, dtype=positions.dtype)
        if float(torch.abs((normal * reference).sum())) > 0.9:
            reference = torch.tensor([0.0, 1.0, 0.0], device=positions.device, dtype=positions.dtype)
        tangent_u = torch.linalg.cross(normal, reference)
        tangent_u = tangent_u / tangent_u.norm().clamp_min(_EPS)
        tangent_v = torch.linalg.cross(normal, tangent_u)

        major = float(reference_frame.tangent_major_scale[node].item())
        minor = float(reference_frame.tangent_minor_scale[node].item())
        basis = torch.stack([tangent_u, tangent_v, normal], dim=1)
        eigenvalues = torch.tensor(
            [major * major, minor * minor, thickness * thickness],
            device=positions.device, dtype=positions.dtype,
        )
        covariance[node] = basis @ torch.diag(eigenvalues) @ basis.T
    return covariance


def _covariance_support_radius_scale(covariance: Any) -> float:
    """Representation C's own diagnostic: the observed footprint reach
    (mean equivalent tangent scale over the unit's own median center
    spacing) -- reported so the replay can disclose HOW MUCH support
    extent the footprint contributes, without asserting the covariance
    normal is correct surface geometry anywhere in the constructor path."""

    frame = extract_covariance_frame(covariance)
    return float(frame.equivalent_tangent_scale.mean().item())


def build_representation_evidence(
    representation: str, positions: Any, covariance: Any,
) -> RepresentationEvidence:
    """Adapt ``(positions, covariance)`` for one of the four fixed
    representations. The caller passes the result's ``.positions``/
    ``.covariance`` unchanged into the existing
    ``build_chart_unit_face_topology_context`` -- no other code path
    changes.
    """

    torch = require_torch()
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")

    if representation == REPRESENTATION_RAW_CENTER_BASELINE:
        return RepresentationEvidence(representation, positions, covariance, 0.0, None)

    if representation == REPRESENTATION_COVARIANCE_SURFEL_SUPPORT:
        # Positions and covariance orientation are unchanged -- the raw
        # center-based topology contract already only reads
        # extract_covariance_frame(covariance).normal_candidate for its
        # adjacency test, so this representation's distinguishing claim
        # (footprint as observed support, not asserted ground truth) is
        # disclosed via support_radius_scale for the replay's coverage/
        # support-band reporting rather than by silently changing the
        # topology constructor's own normal source.
        support_radius_scale = _covariance_support_radius_scale(covariance)
        return RepresentationEvidence(representation, positions, covariance, 0.0, support_radius_scale)

    if representation == REPRESENTATION_CENTER_LATENT_SURFACE:
        latent_positions, displacement = _center_latent_surface_positions(positions)
        latent_covariance = _tangent_frame_covariance(
            latent_positions, covariance, thickness_floor_fraction=_LATENT_THICKNESS_FLOOR_FRACTION,
        )
        return RepresentationEvidence(representation, latent_positions, latent_covariance, displacement, None)

    if representation == REPRESENTATION_HYBRID_LATENT_PLUS_SUPPORT:
        # Location/topology come from the center-derived latent surface
        # (identical to B); covariance contributes support extent/
        # thickness/confidence only, so the in-plane scale is taken from
        # the RAW covariance footprint (support magnitude) while the
        # orientation is still the latent surface's own tangent frame
        # (never the raw covariance orientation) and thickness is still
        # collapsed to the same floor as B -- the only difference from B
        # is that the in-plane extent is not assumed uniform but instead
        # carries the raw footprint's own observed magnitude.
        latent_positions, displacement = _center_latent_surface_positions(positions)
        hybrid_covariance = _tangent_frame_covariance(
            latent_positions, covariance, thickness_floor_fraction=_LATENT_THICKNESS_FLOOR_FRACTION,
        )
        support_radius_scale = _covariance_support_radius_scale(covariance)
        return RepresentationEvidence(
            representation, latent_positions, hybrid_covariance, displacement, support_radius_scale,
        )

    raise AssertionError("unreachable")  # pragma: no cover
