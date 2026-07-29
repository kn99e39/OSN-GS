from __future__ import annotations

"""Covariance-guided structural evidence: per-Gaussian eigenframe extraction.

Canonical direction (worklog 111/113): a Gaussian's covariance eigenframe --
not its raster/KDE footprint -- is the PRIMARY structural evidence for local
surface orientation. This module only extracts and classifies that frame; it
makes no claim about surface adjacency, boundary location, or a physical
outward-normal orientation (an eigenvector's sign is ambiguous by
construction -- see :func:`orientation_insensitive_alignment`).

Deliberately isolated: every function here takes raw ``(N, 3)``/``(N, 3, 3)``
tensors, never a ``TorchGaussianModel`` or renderer/trainer state, so it can
be exercised standalone and never risks touching production/ownership/
lifecycle code.
"""

import math
from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

SHAPE_PLANAR = "planar_surfel"
SHAPE_NEEDLE = "needle_like"
SHAPE_ISOTROPIC = "isotropic"
SHAPE_AMBIGUOUS = "ambiguous_shape"


@dataclass(frozen=True)
class GaussianCovarianceFrame:
    """Per-Gaussian principal frame. ``eigenvalues`` descend: lambda1 >= lambda2 >= lambda3.

    Scale contract (worklog 114 §3): every metric that normalizes by "the"
    local scale must say WHICH of these it means -- a single undifferentiated
    scalar scale silently conflates tangent extent with normal thickness and
    produces wrong tolerances for anisotropic Gaussians (e.g. a wide, thin
    surfel vs. a small, isotropic blob can share one eigenvalue while meaning
    very different things geometrically).
    """

    eigenvalues: Any  # (N, 3)
    tangent_u: Any  # (N, 3) eigenvector of lambda1
    tangent_v: Any  # (N, 3) eigenvector of lambda2
    normal_candidate: Any  # (N, 3) eigenvector of lambda3 -- sign-ambiguous
    planarity: Any  # (N,) large lambda2/lambda3 separation -> flat disk
    elongation: Any  # (N,) large lambda1/lambda2 separation -> rod-like
    isotropy: Any  # (N,) lambda3/lambda1, close to 1 -> sphere-like
    shape_class: tuple[str, ...]  # per-Gaussian classification
    # --- explicit scale contract, never collapsed into one scalar ---
    tangent_major_scale: Any  # (N,) sqrt(lambda1) -- longest tangent extent
    tangent_minor_scale: Any  # (N,) sqrt(lambda2) -- shortest tangent extent
    normal_thickness: Any  # (N,) sqrt(lambda3) -- thickness along the normal candidate
    equivalent_tangent_scale: Any  # (N,) sqrt(tangent_major * tangent_minor) -- isotropic-radius proxy for the tangent footprint
    footprint_area: Any  # (N,) pi * tangent_major * tangent_minor -- tangent-ellipse area proxy

    def payload(self) -> dict[str, Any]:
        return {
            "eigenvalues": self.eigenvalues.detach().cpu().tolist(),
            "normal_candidate": self.normal_candidate.detach().cpu().tolist(),
            "tangent_u": self.tangent_u.detach().cpu().tolist(),
            "tangent_v": self.tangent_v.detach().cpu().tolist(),
            "planarity": self.planarity.detach().cpu().tolist(),
            "elongation": self.elongation.detach().cpu().tolist(),
            "isotropy": self.isotropy.detach().cpu().tolist(),
            "shape_class": list(self.shape_class),
            "tangent_major_scale": self.tangent_major_scale.detach().cpu().tolist(),
            "tangent_minor_scale": self.tangent_minor_scale.detach().cpu().tolist(),
            "normal_thickness": self.normal_thickness.detach().cpu().tolist(),
            "equivalent_tangent_scale": self.equivalent_tangent_scale.detach().cpu().tolist(),
            "footprint_area": self.footprint_area.detach().cpu().tolist(),
        }


def covariance_from_scale_rotation(scale: Any, rotation_quaternion: Any) -> Any:
    """Build ``(N, 3, 3)`` covariance = R diag(scale^2) R^T.

    Convenience for isolated tests/fixtures that describe a Gaussian the same
    way the production model does (linear scale + unit quaternion), without
    importing anything from ``osn_gs.gaussian``.
    """
    torch = require_torch()
    scale = torch.as_tensor(scale)
    quaternion = torch.nn.functional.normalize(torch.as_tensor(rotation_quaternion), dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )
    scale_matrix = torch.diag_embed(scale.square())
    return rotation @ scale_matrix @ rotation.transpose(-1, -2)


def extract_covariance_frame(
    covariance: Any,
    *,
    planarity_threshold: float = 3.0,
    elongation_threshold: float = 3.0,
    isotropy_threshold: float = 0.6,
    degenerate_eps: float = 1e-12,
) -> GaussianCovarianceFrame:
    """Eigen-decompose ``(N, 3, 3)`` symmetric covariance into a descending principal frame.

    Shape classification (all ratio-based, scale-invariant, none canonical/final):

    - ``planar_surfel``: lambda2 clearly separated from lambda3 (``lambda2/lambda3
      >= planarity_threshold``) while lambda1/lambda2 stays moderate -- a flat
      disk with a well-defined normal (``eigenvector(lambda3)``).
    - ``needle_like``: lambda1 clearly separated from lambda2
      (``lambda1/lambda2 >= elongation_threshold``) while lambda2 approx lambda3
      -- a rod with no single reliable normal (rotationally ambiguous around
      the long axis).
    - ``isotropic``: lambda3/lambda1 >= isotropy_threshold -- sphere-like, no
      orientation evidence at all.
    - ``ambiguous_shape``: none of the above cleanly applies.
    """
    torch = require_torch()
    covariance = torch.as_tensor(covariance)
    if covariance.ndim != 3 or tuple(covariance.shape[1:]) != (3, 3):
        raise ValueError("covariance must have shape (N, 3, 3).")
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    # torch.linalg.eigh returns ascending eigenvalues; reverse to descending.
    eigenvalues = torch.flip(eigenvalues, dims=(-1,))
    eigenvectors = torch.flip(eigenvectors, dims=(-1,))
    eigenvalues = eigenvalues.clamp_min(0.0)
    lambda1, lambda2, lambda3 = eigenvalues.unbind(-1)
    tangent_u = eigenvectors[..., 0]
    tangent_v = eigenvectors[..., 1]
    normal_candidate = eigenvectors[..., 2]

    safe_lambda2 = torch.clamp(lambda2, min=degenerate_eps)
    safe_lambda3 = torch.clamp(lambda3, min=degenerate_eps)
    safe_lambda1 = torch.clamp(lambda1, min=degenerate_eps)
    planarity = lambda2 / safe_lambda3
    elongation = lambda1 / safe_lambda2
    isotropy = lambda3 / safe_lambda1

    shape_class = []
    for p, e, iso in zip(planarity.tolist(), elongation.tolist(), isotropy.tolist()):
        if iso >= isotropy_threshold:
            shape_class.append(SHAPE_ISOTROPIC)
        elif p >= planarity_threshold and e < elongation_threshold:
            shape_class.append(SHAPE_PLANAR)
        elif e >= elongation_threshold and p < planarity_threshold:
            shape_class.append(SHAPE_NEEDLE)
        else:
            shape_class.append(SHAPE_AMBIGUOUS)

    tangent_major_scale = torch.sqrt(safe_lambda1)
    tangent_minor_scale = torch.sqrt(safe_lambda2)
    normal_thickness = torch.sqrt(safe_lambda3)
    equivalent_tangent_scale = torch.sqrt(tangent_major_scale * tangent_minor_scale)
    footprint_area = math.pi * tangent_major_scale * tangent_minor_scale

    return GaussianCovarianceFrame(
        eigenvalues=eigenvalues.detach(),
        tangent_u=tangent_u.detach(),
        tangent_v=tangent_v.detach(),
        normal_candidate=normal_candidate.detach(),
        planarity=planarity.detach(),
        elongation=elongation.detach(),
        isotropy=isotropy.detach(),
        shape_class=tuple(shape_class),
        tangent_major_scale=tangent_major_scale.detach(),
        tangent_minor_scale=tangent_minor_scale.detach(),
        normal_thickness=normal_thickness.detach(),
        equivalent_tangent_scale=equivalent_tangent_scale.detach(),
        footprint_area=footprint_area.detach(),
    )


def covariance_conditioning_score(eigenvalues: Any, covariance: Any | None = None, *, degenerate_eps: float = 1e-12) -> Any:
    """Intrinsic (no-neighbor) validity check: finite, non-degenerate covariance.

    Returns ``(N,)`` in ``[0, 1]`` -- 0 for a non-finite or effectively-zero
    (all eigenvalues near ``degenerate_eps``) covariance, 1 for a well-posed
    one. This is a DIFFERENT axis from planarity/elongation/isotropy: those
    describe SHAPE given a valid covariance; this describes whether the
    covariance is numerically trustworthy at all. ``covariance`` is optional
    (callers that only have the already-extracted frame can pass eigenvalues
    alone -- a non-finite raw covariance always produces non-finite
    eigenvalues via ``eigh``, so the eigenvalue check alone is sufficient).
    """
    torch = require_torch()
    eigenvalues = torch.as_tensor(eigenvalues)
    finite = torch.isfinite(eigenvalues).all(dim=-1)
    if covariance is not None:
        finite = finite & torch.isfinite(torch.as_tensor(covariance)).all(dim=(-1, -2))
    non_degenerate = eigenvalues[:, 0] > degenerate_eps
    return (finite & non_degenerate).float()


def orientation_insensitive_alignment(vectors_a: Any, vectors_b: Any) -> Any:
    """``abs(dot(a, b))`` per row -- eigenvector sign is ambiguous, so pairwise
    comparisons must never depend on which of the two antiparallel directions
    the eigen-solver happened to return."""
    torch = require_torch()
    a = torch.as_tensor(vectors_a)
    b = torch.as_tensor(vectors_b)
    return (a * b).sum(dim=-1).abs()
