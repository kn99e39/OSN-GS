from __future__ import annotations

"""Worklog 75: detached STRUCTURAL normal from observed Gaussian centers only.

Worklog 74 traced dense-boundary cycle destruction to the orientation stage:
of 34 distance-stage cycle edges only 16 survived to mutuality, 13 were cut by
tangent incompatibility, and the tangent is `cross(normal, outward)` where
`normal` is the covariance eigenframe's `normal_candidate`. That makes the
covariance-derived per-Gaussian orientation a direct, independent
cycle-destruction bottleneck (independent of the scale mismatch worklog 74
also measured). This module supplies the alternative orientation source for a
bounded A/B, and nothing else.

`compute_structural_normals` estimates each point's surface normal by local
PCA over the region-owned observed point POSITIONS alone -- the smallest
principal direction of the k-nearest-neighbour position covariance. It reads
no Gaussian scale, rotation, covariance, spherical harmonics, or opacity, and
touches no renderer/optimizer state, so it is fully detached from photometric
training: an orientation derived only from where observed surface samples
actually lie.

`rebuild_candidate_orientation` re-derives the boundary frame of an ALREADY
EXTRACTED, FROZEN candidate set under a new normal field. It reuses
`torch_region_owned_dense_boundary_support`'s own missing-sector /
outward-direction construction verbatim (largest angular gap among kNN
tangent-plane directions -> outward bisector -> `tangent = cross(normal,
outward)`); it never re-decides WHICH points are candidates. Candidate ids,
positions, boundary reasons, and the full-evidence sampling scale are carried
through unchanged, so an A/B built on this differs from the covariance path in
the orientation frame and nothing else.

Deliberately isolated: no production module imports this, and it is not wired
into any pipeline, trainer, renderer, or checkpoint path.
"""

import math
from dataclasses import replace
from typing import Any, Sequence

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-8


def compute_structural_normals(points: Any, *, neighbors: int = 12) -> Any:
    """Local-PCA surface normal per point, from POSITIONS ONLY.

    Returns a unit ``(N, 3)`` normal field. Sign is left arbitrary (the
    smallest-eigenvector sign is not observable from positions alone) --
    every downstream consumer in this experiment compares orientations with
    ``abs(dot)``, exactly as the covariance path already does, so no
    orientation convention is invented here.
    """

    torch = require_torch()
    n = int(points.shape[0])
    if n < 3:
        return torch.zeros((n, 3), dtype=points.dtype, device=points.device)
    k = min(int(neighbors), n - 1)
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    near = distances.topk(k, largest=False).indices  # (N, k)
    neighborhood = points[near]  # (N, k, 3)
    centered = neighborhood - neighborhood.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered  # (N, 3, 3), position covariance only
    # Symmetric by construction; `eigvalsh`-family ordering is ascending, so
    # column 0 is the smallest-variance direction = the surface normal.
    _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normals = eigenvectors[..., 0]
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _outward_and_tangent(
    points: Any, normal: Any, index: int, neighbor_indices: Any,
) -> tuple[Any, Any, float] | None:
    """The missing-sector/outward-direction construction of
    ``torch_region_owned_dense_boundary_support.extract_dense_boundary_support``,
    applied at one point under a supplied normal. Returns
    ``(outward, tangent, gap_radians)`` or ``None`` when the tangent-plane
    reference degenerates."""

    torch = require_torch()
    reference = points[neighbor_indices[0]] - points[index]
    reference = reference - normal * (reference @ normal)
    if float(reference.norm()) <= _EPS:
        return None
    reference = reference / reference.norm()
    axis = torch.linalg.cross(normal, reference)
    axis = axis / axis.norm().clamp_min(_EPS)
    delta = points[neighbor_indices] - points[index]
    delta = delta - normal[None, :] * (delta @ normal)[:, None]
    angles = torch.atan2(delta @ axis, delta @ reference).remainder(2 * math.pi).sort().values
    gaps = torch.diff(torch.cat((angles, angles[:1] + 2 * math.pi)))
    gap, gap_index = gaps.max(dim=0)
    center = angles[gap_index] + gap / 2
    outward = torch.cos(center) * reference + torch.sin(center) * axis
    tangent = torch.linalg.cross(normal, outward)
    tangent_norm = tangent.norm()
    if float(tangent_norm) <= _EPS:
        return None
    return outward, tangent / tangent_norm.clamp_min(_EPS), float(gap)


def rebuild_candidate_orientation(
    points: Any,
    normals: Any,
    stable_ids: Sequence[Any],
    frozen_candidates: Sequence[Any],
    *,
    neighbors: int = 12,
    missing_sector_radians: float = math.pi,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Re-derive ``normal``/``tangent`` for a FROZEN candidate set.

    ``frozen_candidates`` are ``DenseBoundarySupportCandidate`` instances
    already admitted by the covariance path. Their ``stable_id``, ``position``,
    ``boundary_reason`` and ``full_evidence_scale`` are carried through
    untouched -- only the orientation frame is recomputed, so candidate
    identity/geometry/reason stay frozen across the A/B exactly as required.

    The returned diagnostics record how many frozen candidates would NOT have
    passed the missing-sector admission test under the new normal
    (``would_fail_sector_admission``). That is reported, never acted on: acting
    on it would re-extract candidates for B, which the comparison forbids.
    """

    torch = require_torch()
    n = int(points.shape[0])
    index_by_id = {stable_id: i for i, stable_id in enumerate(stable_ids)}
    k = min(int(neighbors), max(n - 1, 1))
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    near = distances.topk(k, largest=False).indices

    rebuilt: list[Any] = []
    degenerate = 0
    would_fail_admission = 0
    for candidate in frozen_candidates:
        index = index_by_id.get(candidate.stable_id)
        if index is None:
            # Frozen candidate not present in this point set: carry through
            # untouched rather than dropping it (candidate set stays frozen).
            rebuilt.append(candidate)
            continue
        normal = normals[index]
        normal = normal / normal.norm().clamp_min(_EPS)
        frame = _outward_and_tangent(points, normal, index, near[index])
        if frame is None:
            degenerate += 1
            rebuilt.append(candidate)
            continue
        _outward, tangent, gap = frame
        if gap < missing_sector_radians:
            would_fail_admission += 1
        rebuilt.append(replace(
            candidate,
            normal=tuple(float(x) for x in normal),
            tangent=tuple(float(x) for x in tangent),
        ))
    return tuple(rebuilt), {
        "rebuilt_count": len(rebuilt),
        "degenerate_tangent_frame": degenerate,
        "would_fail_sector_admission": would_fail_admission,
    }


def normal_angular_disagreement_degrees(left: Any, right: Any) -> Any:
    """Per-point unsigned angle between two normal fields, in degrees.

    Uses ``abs(dot)`` because both fields carry an arbitrary eigenvector sign,
    so the meaningful quantity is the angle between the two undirected surface
    orientations (range ``[0, 90]``).
    """

    torch = require_torch()
    a = left / left.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    b = right / right.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    alignment = (a * b).sum(dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(alignment))
