from __future__ import annotations

"""Worklog 93 -- read-only latent-midsurface recoverability attribution.

Worklog 92 found that most of Worklog 90's ``MULTILAYER_OR_VOLUMETRIC``
evidence is actually ``LOCALLY_THICK_UNIMODAL_SHEET`` (~62-70%) or
``LOCALLY_SINGLE_CURVED_SHEET`` (~25-32%), with true persistent multi-layer
center geometry a small minority (~1-2%). This module asks a different
question about that thick/curved majority: does it contain a recoverable
latent 2D midsurface, or does the raw center distribution itself lack
sufficient surface geometry?

Hard constraints (all enforced structurally, not just by convention):

- Center positions only. No function here accepts or reads a covariance
  tensor; Gaussian covariance normals/tangents/scale are never the target
  geometry (verified by an AST test, matching Worklog 92's convention).
- Fully read-only: no Gaussian xyz is modified, no projected position is
  written back into any model state. Every projection computed here is a
  local NumPy/Torch value returned in a report, never persisted.
- No new production boundary or NURBS path. No threshold is tuned toward a
  favorable outcome -- every constant below is either reused from an
  existing worklog (Worklog 82's k=8, Worklog 92's local-neighborhood
  machinery) or a fixed, documented diagnostic convention.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
    _LOCAL_NEIGHBOR_COUNT,
    _knn_indices,
    _local_plane_normal,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# A local quadratic surface fit needs at least this many neighbors (5 free
# parameters: constant, 2 linear, 3 quadratic minus the constraint already
# used by the plane -- effectively ax^2+bxy+cy^2+dx+ey needs >=5 points to
# be well-posed, and Worklog 92's own k=8 neighborhood already satisfies
# this without a new sweep).
_MIN_POINTS_FOR_QUADRATIC_FIT = 6


@dataclass(frozen=True)
class LocalMidsurfacePatch:
    node: int
    neighbor_count: int
    plane_normal: Any
    plane_point: Any
    tangent_u: Any
    tangent_v: Any
    quadratic_coeffs: Any | None  # (a, b, c, d, e) for z = a*u^2+b*u*v+c*v^2+d*u+e*v
    mean_curvature_estimate: float | None
    in_plane_spacing: float | None
    raw_normal_residual_spread: float | None


def _fit_local_quadratic_patch(points: Any, node: int, neighbor_indices: Any) -> LocalMidsurfacePatch:
    """Diagnostic-only local quadratic surface patch fit to CENTERS ONLY.

    Uses the same local-plane convention as Worklog 92 (``_local_plane_normal``,
    ``_knn_indices``) to establish a tangent frame, then fits a quadratic
    height field in that frame to additionally estimate local curvature. The
    plane/quadratic here are never covariance-derived and are never written
    back to any Gaussian.
    """

    torch = require_torch()
    neighborhood = torch.cat([torch.tensor([node], device=points.device), neighbor_indices])
    neighbor_count = int(neighbor_indices.numel())
    local_points = points[neighborhood]
    plane_point = local_points.mean(dim=0)

    if neighbor_count < 2:
        return LocalMidsurfacePatch(node, neighbor_count, None, plane_point, None, None, None, None, None, None)

    normal = _local_plane_normal(local_points)
    # Build an arbitrary orthonormal tangent basis from the plane normal.
    reference = torch.tensor([1.0, 0.0, 0.0], device=points.device)
    if float(torch.abs((normal * reference).sum())) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0], device=points.device)
    tangent_u = torch.linalg.cross(normal, reference)
    tangent_u = tangent_u / tangent_u.norm().clamp_min(_EPS)
    tangent_v = torch.linalg.cross(normal, tangent_u)

    centered = local_points - plane_point
    signed_offset = centered @ normal
    u_coord = centered @ tangent_u
    v_coord = centered @ tangent_v

    off_diagonal = ~torch.eye(neighborhood.numel(), dtype=torch.bool, device=points.device)
    in_plane_delta = torch.stack([u_coord, v_coord], dim=1)
    in_plane_pairwise = torch.cdist(in_plane_delta, in_plane_delta)
    nearest_in_plane = in_plane_pairwise.masked_fill(~off_diagonal, float("inf")).min(dim=1).values
    in_plane_spacing = float(nearest_in_plane.median().item())

    raw_normal_residual_spread = float(signed_offset.std(unbiased=False).item()) if signed_offset.numel() > 1 else 0.0

    quadratic_coeffs = None
    mean_curvature = None
    if neighbor_count + 1 >= _MIN_POINTS_FOR_QUADRATIC_FIT:
        design = torch.stack(
            [u_coord.square(), u_coord * v_coord, v_coord.square(), u_coord, v_coord], dim=1,
        )
        try:
            solution = torch.linalg.lstsq(design, signed_offset.unsqueeze(1)).solution.squeeze(1)
            quadratic_coeffs = solution.detach()
            a, b, c = float(solution[0]), float(solution[1]), float(solution[2])
            # Mean curvature proxy at the patch center (u=v=0): half the
            # trace of the quadratic form's Hessian [[2a, b], [b, 2c]].
            mean_curvature = 0.5 * (2.0 * a + 2.0 * c)
        except Exception:  # pragma: no cover - defensive, lstsq expected to converge
            quadratic_coeffs = None
            mean_curvature = None

    return LocalMidsurfacePatch(
        node=node,
        neighbor_count=neighbor_count,
        plane_normal=normal.detach(),
        plane_point=plane_point.detach(),
        tangent_u=tangent_u.detach(),
        tangent_v=tangent_v.detach(),
        quadratic_coeffs=quadratic_coeffs,
        mean_curvature_estimate=mean_curvature,
        in_plane_spacing=in_plane_spacing if in_plane_spacing > 0 else None,
        raw_normal_residual_spread=raw_normal_residual_spread,
    )


@dataclass(frozen=True)
class LatentMidsurfaceUnitReport:
    member_count: int
    local_thickness_over_spacing_evidence_weighted: float | None
    neighbor_position_agreement: float | None
    neighbor_tangent_agreement: float | None
    neighbor_curvature_agreement: float | None
    projection_displacement_over_spacing_median: float | None
    projection_displacement_over_extent_median: float | None
    raw_open_or_nonmanifold_fraction: float
    diagnostic_open_or_nonmanifold_fraction: float
    raw_valid_local_face_incidence_fraction: float
    valid_local_face_incidence_fraction: float
    neighborhood_preservation_fraction: float
    mean_curvature_before: float | None
    mean_curvature_after: float | None
    curvature_preserved: bool
    observed_support_band_fidelity_fraction: float | None


def _pairwise_same_surface_positional(
    points: Any,
    patches: Sequence[LocalMidsurfacePatch],
    neighbor_indices: Any,
    *,
    normal_alignment_min: float,
    tangent_residual_max: float,
) -> Any:
    """Position-only same-surface adjacency analog: uses each node's own
    diagnostic local plane normal (fit from centers, never covariance) as
    the comparison direction. This mirrors Worklog 82's same_surface test
    shape (normal alignment + mutual tangent residual) but the "normal"
    here is a position-only PCA/quadratic-patch normal, never a Gaussian's
    covariance eigenvector -- structurally distinct inputs, same test
    shape, reused thresholds (Worklog 82's existing 0.85/0.35 defaults,
    not swept here).
    """

    torch = require_torch()
    count = points.shape[0]
    adjacency = torch.zeros((count, count), dtype=torch.bool, device=points.device)
    for node in range(count):
        patch = patches[node]
        if patch.plane_normal is None:
            continue
        for neighbor in neighbor_indices[node].tolist():
            neighbor = int(neighbor)
            other = patches[neighbor]
            if other.plane_normal is None:
                continue
            alignment = float(torch.abs((patch.plane_normal * other.plane_normal).sum()).item())
            if alignment < normal_alignment_min:
                continue
            delta = points[neighbor] - points[node]
            residual_a = float(torch.abs((delta * patch.plane_normal).sum()).item())
            residual_b = float(torch.abs((delta * other.plane_normal).sum()).item())
            spacing = max(patch.in_plane_spacing or _EPS, other.in_plane_spacing or _EPS)
            mutual_residual = 0.5 * (residual_a + residual_b) / spacing
            if mutual_residual > tangent_residual_max:
                continue
            adjacency[node, neighbor] = True
            adjacency[neighbor, node] = True
    return adjacency


def _manifold_fraction(adjacency: Any) -> tuple[float, float]:
    """Local surface-interior manifoldness disclosure (distinct from
    Worklog 85's degree-2 BOUNDARY CURVE criterion, which does not apply
    to a dense interior patch): a node has valid local face incidence when
    it participates in at least one closed triangle (3-cycle) of accepted
    same-surface adjacency -- i.e. it has genuine 2-manifold face support,
    not just isolated pairwise edges. A node with zero same-surface
    neighbors, or with neighbors that never close a triangle, is open or
    non-manifold at this location. Returns (open_or_nonmanifold_fraction,
    valid_face_incidence_fraction) -- the two are complementary by
    construction (whether a node has any valid face incidence)."""

    torch = require_torch()
    count = adjacency.shape[0]
    if count == 0:
        return 0.0, 0.0
    valid_face = torch.zeros(count, dtype=torch.bool, device=adjacency.device)
    for node in range(count):
        neighbors = torch.nonzero(adjacency[node], as_tuple=False).reshape(-1).tolist()
        for offset, a in enumerate(neighbors):
            for b in neighbors[offset + 1:]:
                if bool(adjacency[a, b]):
                    valid_face[node] = True
                    valid_face[a] = True
                    valid_face[b] = True
    open_or_nonmanifold = float((~valid_face).float().mean().item())
    return open_or_nonmanifold, float(valid_face.float().mean().item())


def attribute_latent_midsurface_recoverability(
    positions: Any,
    member_indices: Sequence[int],
    *,
    same_surface_normal_alignment_min: float = 0.85,
    same_surface_tangent_residual_max: float = 0.35,
    flattening_curvature_ratio_threshold: float = 0.1,
) -> LatentMidsurfaceUnitReport:
    """Center-position-only latent-midsurface recoverability report for one
    Worklog 92 LOCALLY_THICK_UNIMODAL_SHEET / LOCALLY_SINGLE_CURVED_SHEET
    unit. Never reads covariance. Never mutates ``positions``. Every
    projection is computed locally and discarded after this report.

    ``same_surface_normal_alignment_min``/``same_surface_tangent_residual_max``
    reuse Worklog 82's own existing default thresholds (0.85/0.35) applied
    to a positional analog of its same-surface test -- not new swept
    parameters. ``flattening_curvature_ratio_threshold`` is a fixed
    disclosure floor: if the diagnostic's own curvature estimate collapses
    to below 10% of the raw curvature estimate, this is flagged as
    over-flattening rather than genuine recovery, never hidden.
    """

    torch = require_torch()
    members = tuple(dict.fromkeys(int(index) for index in member_indices))
    count = len(members)
    if count == 0:
        raise ValueError("member_indices must not be empty")
    selector = torch.tensor(members, dtype=torch.long, device=positions.device)
    points = positions[selector]

    if count < _MIN_POINTS_FOR_QUADRATIC_FIT:
        return LatentMidsurfaceUnitReport(
            count, None, None, None, None, None, None, 1.0, 1.0, 0.0, 0.0, 0.0, None, None, False, None,
        )

    neighbor_indices = _knn_indices(points, _LOCAL_NEIGHBOR_COUNT)
    patches = tuple(
        _fit_local_quadratic_patch(points, node, neighbor_indices[node]) for node in range(count)
    )

    # 1. LOCAL_THICKNESS: evidence-weighted normal residual spread over
    #    in-plane spacing, from CENTERS ONLY (each patch's own position-fit
    #    normal, never a covariance eigenvector).
    thickness_terms = [
        (patch.raw_normal_residual_spread / patch.in_plane_spacing, 1)
        for patch in patches
        if patch.raw_normal_residual_spread is not None and patch.in_plane_spacing
    ]
    local_thickness = (
        sum(value for value, _ in thickness_terms) / len(thickness_terms) if thickness_terms else None
    )

    # 2. LATENT_MIDSURFACE_CONSISTENCY: do overlapping local patches agree
    #    with their spatial neighbors in position, tangent orientation, and
    #    curvature.
    position_agreements: list[float] = []
    tangent_agreements: list[float] = []
    curvature_agreements: list[float] = []
    for node in range(count):
        patch = patches[node]
        if patch.plane_normal is None:
            continue
        for neighbor in neighbor_indices[node].tolist():
            neighbor = int(neighbor)
            other = patches[int(neighbor)]
            if other.plane_normal is None:
                continue
            spacing = max(patch.in_plane_spacing or _EPS, _EPS)
            plane_gap = float(torch.abs(((points[neighbor] - patch.plane_point) @ patch.plane_normal)).item())
            position_agreements.append(plane_gap / spacing)
            tangent_agreements.append(float(torch.abs((patch.plane_normal * other.plane_normal).sum()).item()))
            if patch.mean_curvature_estimate is not None and other.mean_curvature_estimate is not None:
                spread = max(abs(patch.mean_curvature_estimate), abs(other.mean_curvature_estimate), _EPS)
                curvature_agreements.append(
                    1.0 - min(1.0, abs(patch.mean_curvature_estimate - other.mean_curvature_estimate) / spread)
                )
    neighbor_position_agreement = (
        sum(position_agreements) / len(position_agreements) if position_agreements else None
    )
    neighbor_tangent_agreement = (
        sum(tangent_agreements) / len(tangent_agreements) if tangent_agreements else None
    )
    neighbor_curvature_agreement = (
        sum(curvature_agreements) / len(curvature_agreements) if curvature_agreements else None
    )

    # 3. PROJECTION_DISPLACEMENT: diagnostic-only projection of each center
    #    onto ITS OWN local patch plane. Never written back anywhere.
    extent = float((points.max(dim=0).values - points.min(dim=0).values).norm().item())
    displacements: list[float] = []
    projected_points = points.clone()
    for node in range(count):
        patch = patches[node]
        if patch.plane_normal is None:
            continue
        offset = float(((points[node] - patch.plane_point) @ patch.plane_normal).item())
        displacements.append(abs(offset))
        projected_points[node] = points[node] - offset * patch.plane_normal
    displacement_tensor = torch.tensor(displacements) if displacements else None
    spacing_values = [p.in_plane_spacing for p in patches if p.in_plane_spacing]
    median_spacing = float(torch.tensor(spacing_values).median().item()) if spacing_values else None
    projection_displacement_over_spacing_median = (
        float(displacement_tensor.median().item()) / median_spacing
        if displacement_tensor is not None and median_spacing else None
    )
    projection_displacement_over_extent_median = (
        float(displacement_tensor.median().item()) / extent
        if displacement_tensor is not None and extent > 0 else None
    )

    # 4. TOPOLOGY_RECOVERABILITY: raw vs. diagnostic-projected local
    #    manifold structure, using the SAME positional adjacency test on
    #    both point sets -- chart membership is unchanged, no threshold is
    #    tuned differently between raw and diagnostic passes.
    raw_patches = patches
    raw_adjacency = _pairwise_same_surface_positional(
        points, raw_patches, neighbor_indices,
        normal_alignment_min=same_surface_normal_alignment_min,
        tangent_residual_max=same_surface_tangent_residual_max,
    )
    raw_open_fraction, raw_valid_face_fraction = _manifold_fraction(raw_adjacency)

    diagnostic_neighbor_indices = _knn_indices(projected_points, _LOCAL_NEIGHBOR_COUNT)
    diagnostic_patches = tuple(
        _fit_local_quadratic_patch(projected_points, node, diagnostic_neighbor_indices[node])
        for node in range(count)
    )
    diagnostic_adjacency = _pairwise_same_surface_positional(
        projected_points, diagnostic_patches, diagnostic_neighbor_indices,
        normal_alignment_min=same_surface_normal_alignment_min,
        tangent_residual_max=same_surface_tangent_residual_max,
    )
    diagnostic_open_fraction, diagnostic_valid_face_fraction = _manifold_fraction(diagnostic_adjacency)

    # Neighborhood preservation: did the diagnostic collapse keep each
    # node's kNN neighbor SET the same (only moved points along normals,
    # never reshuffled topology by construction, but a large collapse can
    # still change which points are nearest).
    preserved = 0
    for node in range(count):
        raw_set = set(neighbor_indices[node].tolist())
        diagnostic_set = set(diagnostic_neighbor_indices[node].tolist())
        overlap = len(raw_set & diagnostic_set) / max(1, len(raw_set))
        preserved += overlap
    neighborhood_preservation_fraction = preserved / count if count else 0.0

    # 5. CURVATURE_PRESERVATION: compare mean curvature estimates before
    #    (raw) and after (diagnostic-projected) thickness collapse.
    raw_curvatures = [p.mean_curvature_estimate for p in raw_patches if p.mean_curvature_estimate is not None]
    diagnostic_curvatures = [
        p.mean_curvature_estimate for p in diagnostic_patches if p.mean_curvature_estimate is not None
    ]
    mean_curvature_before = (
        sum(abs(v) for v in raw_curvatures) / len(raw_curvatures) if raw_curvatures else None
    )
    mean_curvature_after = (
        sum(abs(v) for v in diagnostic_curvatures) / len(diagnostic_curvatures) if diagnostic_curvatures else None
    )
    curvature_preserved = True
    if mean_curvature_before is not None and mean_curvature_before > _EPS:
        if mean_curvature_after is None:
            curvature_preserved = False
        else:
            ratio = mean_curvature_after / mean_curvature_before
            curvature_preserved = ratio >= flattening_curvature_ratio_threshold

    # 6. OBSERVED-EVIDENCE FIDELITY: projected surface must stay within the
    #    original visible-Gaussian support band, i.e. every displacement
    #    must not exceed the unit's own observed normal residual spread by
    #    more than a small margin (never invents geometry beyond what was
    #    already observed).
    support_band_fidelity = None
    if thickness_terms and displacement_tensor is not None:
        support_band = max(patch.raw_normal_residual_spread or 0.0 for patch in patches)
        within_band = (displacement_tensor <= (support_band + _EPS)).float().mean().item()
        support_band_fidelity = float(within_band)

    return LatentMidsurfaceUnitReport(
        member_count=count,
        local_thickness_over_spacing_evidence_weighted=local_thickness,
        neighbor_position_agreement=neighbor_position_agreement,
        neighbor_tangent_agreement=neighbor_tangent_agreement,
        neighbor_curvature_agreement=neighbor_curvature_agreement,
        projection_displacement_over_spacing_median=projection_displacement_over_spacing_median,
        projection_displacement_over_extent_median=projection_displacement_over_extent_median,
        raw_open_or_nonmanifold_fraction=raw_open_fraction,
        diagnostic_open_or_nonmanifold_fraction=diagnostic_open_fraction,
        raw_valid_local_face_incidence_fraction=raw_valid_face_fraction,
        valid_local_face_incidence_fraction=diagnostic_valid_face_fraction,
        neighborhood_preservation_fraction=neighborhood_preservation_fraction,
        mean_curvature_before=mean_curvature_before,
        mean_curvature_after=mean_curvature_after,
        curvature_preserved=curvature_preserved,
        observed_support_band_fidelity_fraction=support_band_fidelity,
    )
