from __future__ import annotations

"""Worklog 105 -- 2DGS-inspired per-Gaussian surface-orientation representation.

Derived, read-only geometry for the coverage-first Gaussian Subset partition
(:mod:`osn_gs.surface.torch_coverage_first_subset_partition`). This module
NEVER replaces the renderer/training representation: nothing here is fed back
into `TorchGaussianModel`, the rasterizer, or any optimizer. It only reads a
Gaussian's existing orientation/scale parameterization and exposes it as a
planar surfel -- two tangent axes plus one orthogonal normal -- so surface
partitioning can treat the tangent plane, rather than the full volumetric
Gaussian thickness, as the primary surface-orientation signal.

Where the frame comes from
--------------------------
A 3DGS Gaussian stores ``scaling`` (log-domain, three axes) and ``rotation``
(a quaternion). Its covariance is

    Sigma = R diag(s^2) R^T ,   s = exp(scaling)

so column ``j`` of ``R`` is exactly the eigenvector of ``Sigma`` with
eigenvalue ``s_j^2``:

    Sigma (R e_j) = R diag(s^2) R^T R e_j = s_j^2 (R e_j)

The Gaussian parameterization therefore already contains the three principal
axes exactly -- no eigen-decomposition is needed on the production path. The
ONLY ambiguity it carries is **axis order**: ``scale_0/1/2`` are not sorted,
so which stored axis is "the normal" is undefined until the axes are ordered.
:func:`derive_surface_orientation_from_scale_rotation` resolves this
deterministically by sorting on ``s^2`` descending
(``lambda1 >= lambda2 >= lambda3``) and defining

    surface_normal = axis(lambda3)   -- the thinnest principal direction
    tangent_axis_u = axis(lambda1)   -- the longest in-plane direction
    tangent_axis_v = normal x u      -- completes a right-handed orthonormal frame

This matches :func:`osn_gs.surface.torch_gaussian_covariance_frame.extract_covariance_frame`'s
existing convention (descending eigenvalues, ``eigenvector(lambda3)`` as the
normal candidate), so the two modules cannot silently disagree about what
"the normal" means; :func:`derive_surface_orientation_from_covariance` exists
precisely so an arbitrary covariance (fixtures, synthetic tests, non-3DGS
inputs) goes through the SAME canonicalization instead of a second definition.

Sign contract
-------------
A principal axis has no intrinsic sign. Signs are canonicalized here only so
repeated runs and both entry points produce byte-comparable vectors -- they
carry NO physical outward-orientation meaning. Any similarity comparison
between two Gaussians must therefore use the unsigned relation
:func:`unsigned_normal_alignment` (``|dot(n_i, n_j)|``), never a signed dot
product. No global normal flipping is performed.

Degeneracy is DIAGNOSTIC ONLY
-----------------------------
``axis_separability`` records where the ordering is numerically meaningless
(``lambda2 ~= lambda3`` leaves the normal direction unresolved;
``lambda1 ~= lambda2`` leaves the in-plane axes unresolved but keeps the plane
itself well defined; all three equal means no orientation evidence at all).
These labels never remove a Gaussian from the representation -- every input
row always produces exactly one output row, because subset ownership
downstream is a coverage contract, not a quality gate.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_gaussian_covariance_frame import (
    _batched_eigh,
    orientation_insensitive_alignment,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

# --- axis separability codes (diagnostic only; never a filter) ---
SEPARABILITY_WELL_DEFINED = "well_defined"
SEPARABILITY_TANGENT_AXES_DEGENERATE = "tangent_axes_degenerate"
SEPARABILITY_NORMAL_AXIS_DEGENERATE = "normal_axis_degenerate"
SEPARABILITY_ISOTROPIC = "isotropic"
SEPARABILITY_NON_FINITE = "non_finite"

SEPARABILITY_CODES: tuple[str, ...] = (
    SEPARABILITY_WELL_DEFINED,
    SEPARABILITY_TANGENT_AXES_DEGENERATE,
    SEPARABILITY_NORMAL_AXIS_DEGENERATE,
    SEPARABILITY_ISOTROPIC,
    SEPARABILITY_NON_FINITE,
)

# Reused verbatim from `extract_covariance_frame`'s own already-shipped
# defaults (``planarity_threshold``/``elongation_threshold`` = 3.0) so this
# module does not introduce a second, differently-tuned notion of "the
# eigenvalue gap is big enough to trust this axis ordering".
DEFAULT_NORMAL_SEPARATION_RATIO = 3.0
DEFAULT_TANGENT_SEPARATION_RATIO = 3.0


@dataclass(frozen=True)
class GaussianSurfaceOrientation:
    """Per-Gaussian planar-surfel representation, one row per input Gaussian.

    Every field has leading dimension ``N`` equal to the input count -- rows
    are never dropped, reordered, or merged, so ``gaussian_ids[i]`` always
    identifies the source Gaussian of row ``i``.
    """

    gaussian_ids: Any  # (N,) int64 -- provenance back to the source Gaussian
    positions: Any  # (N, 3) -- unmodified Gaussian centers
    tangent_axis_u: Any  # (N, 3) unit, axis of lambda1
    tangent_axis_v: Any  # (N, 3) unit, normal x u -- right-handed completion
    surface_normal: Any  # (N, 3) unit, axis of lambda3 -- SIGN-AMBIGUOUS
    eigenvalues: Any  # (N, 3) descending lambda1 >= lambda2 >= lambda3
    tangent_major_scale: Any  # (N,) sqrt(lambda1)
    tangent_minor_scale: Any  # (N,) sqrt(lambda2)
    normal_thickness: Any  # (N,) sqrt(lambda3) -- band thickness, NOT the surface
    axis_separability: Any  # (N,) int8 index into SEPARABILITY_CODES
    source: str  # "scale_rotation" | "covariance"

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def separability_counts(self) -> dict[str, int]:
        """Vectorized histogram over :data:`SEPARABILITY_CODES` (no per-row Python)."""

        torch = require_torch()
        counts = torch.bincount(
            self.axis_separability.reshape(-1).to(torch.int64), minlength=len(SEPARABILITY_CODES)
        )
        return {name: int(counts[index]) for index, name in enumerate(SEPARABILITY_CODES)}


def unsigned_normal_alignment(normals_a: Any, normals_b: Any) -> Any:
    """``|dot(n_a, n_b)|`` -- the ONLY admissible normal comparison here.

    Delegates to the existing
    :func:`osn_gs.surface.torch_gaussian_covariance_frame.orientation_insensitive_alignment`
    so ``n`` and ``-n`` are one local surface orientation everywhere in the
    codebase, not two.
    """

    return orientation_insensitive_alignment(normals_a, normals_b)


def _canonical_axis_sign(vectors: Any) -> Any:
    """Deterministic sign gauge: the largest-magnitude component is made positive.

    Ties resolve to the lowest component index (``argmax`` returns the first
    maximum). This is a reproducibility gauge only -- see the module docstring's
    sign contract; it must never be read as a physical outward direction.
    """

    torch = require_torch()
    pivot_index = vectors.abs().argmax(dim=-1, keepdim=True)
    pivot = vectors.gather(-1, pivot_index)
    sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
    return vectors * sign


def _fallback_in_plane_axis(normal: Any) -> Any:
    """Deterministic in-plane axis for a normal whose stored tangent axis is unusable.

    Gram-Schmidt against whichever world axis the normal is least aligned with,
    chosen by magnitude so the subtraction never cancels.
    """

    torch = require_torch()
    reference = torch.zeros_like(normal)
    least_aligned = normal.abs().argmin(dim=-1, keepdim=True)
    reference.scatter_(-1, least_aligned, 1.0)
    axis = reference - (reference * normal).sum(dim=-1, keepdim=True) * normal
    return torch.nn.functional.normalize(axis, dim=-1, eps=_EPS)


def _assemble(
    positions: Any,
    gaussian_ids: Any,
    eigenvalues: Any,
    axis_major: Any,
    axis_normal: Any,
    finite_row: Any,
    normal_separation_ratio: float,
    tangent_separation_ratio: float,
    source: str,
) -> GaussianSurfaceOrientation:
    """Shared canonicalization for BOTH entry points -- single normal definition."""

    torch = require_torch()

    default_normal = torch.zeros_like(axis_normal)
    default_normal[..., 2] = 1.0
    default_major = torch.zeros_like(axis_major)
    default_major[..., 0] = 1.0
    keep = finite_row.reshape(-1, 1)
    axis_normal = torch.where(keep, axis_normal, default_normal)
    axis_major = torch.where(keep, axis_major, default_major)
    eigenvalues = torch.where(finite_row.reshape(-1, 1), eigenvalues, torch.zeros_like(eigenvalues))

    normal = _canonical_axis_sign(torch.nn.functional.normalize(axis_normal, dim=-1, eps=_EPS))
    tangent_u = _canonical_axis_sign(torch.nn.functional.normalize(axis_major, dim=-1, eps=_EPS))
    # Re-orthogonalize numerically; for exact principal axes this is a no-op.
    tangent_u = tangent_u - (tangent_u * normal).sum(dim=-1, keepdim=True) * normal
    residual = tangent_u.norm(dim=-1, keepdim=True)
    tangent_u = torch.where(
        residual > 1e-6,
        torch.nn.functional.normalize(tangent_u, dim=-1, eps=_EPS),
        _fallback_in_plane_axis(normal),
    )
    tangent_v = torch.cross(normal, tangent_u, dim=-1)

    lambda1, lambda2, lambda3 = eigenvalues.unbind(-1)
    safe2 = lambda2.clamp_min(_EPS)
    safe3 = lambda3.clamp_min(_EPS)
    normal_separable = lambda2 >= normal_separation_ratio * safe3
    tangent_separable = lambda1 >= tangent_separation_ratio * safe2

    separability = torch.full(lambda1.shape, SEPARABILITY_CODES.index(SEPARABILITY_WELL_DEFINED), dtype=torch.int8, device=lambda1.device)
    separability = torch.where(
        ~tangent_separable,
        torch.tensor(SEPARABILITY_CODES.index(SEPARABILITY_TANGENT_AXES_DEGENERATE), dtype=torch.int8, device=lambda1.device),
        separability,
    )
    separability = torch.where(
        ~normal_separable,
        torch.tensor(SEPARABILITY_CODES.index(SEPARABILITY_NORMAL_AXIS_DEGENERATE), dtype=torch.int8, device=lambda1.device),
        separability,
    )
    separability = torch.where(
        (~normal_separable) & (~tangent_separable),
        torch.tensor(SEPARABILITY_CODES.index(SEPARABILITY_ISOTROPIC), dtype=torch.int8, device=lambda1.device),
        separability,
    )
    separability = torch.where(
        ~finite_row,
        torch.tensor(SEPARABILITY_CODES.index(SEPARABILITY_NON_FINITE), dtype=torch.int8, device=lambda1.device),
        separability,
    )

    return GaussianSurfaceOrientation(
        gaussian_ids=gaussian_ids,
        positions=positions,
        tangent_axis_u=tangent_u.detach(),
        tangent_axis_v=tangent_v.detach(),
        surface_normal=normal.detach(),
        eigenvalues=eigenvalues.detach(),
        tangent_major_scale=torch.sqrt(lambda1.clamp_min(0.0)).detach(),
        tangent_minor_scale=torch.sqrt(lambda2.clamp_min(0.0)).detach(),
        normal_thickness=torch.sqrt(lambda3.clamp_min(0.0)).detach(),
        axis_separability=separability.detach(),
        source=source,
    )


def _resolve_ids(gaussian_ids: Any, count: int, device: Any) -> Any:
    torch = require_torch()
    if gaussian_ids is None:
        return torch.arange(count, dtype=torch.int64, device=device)
    ids = torch.as_tensor(gaussian_ids, device=device).reshape(-1).to(torch.int64)
    if int(ids.shape[0]) != count:
        raise ValueError(f"gaussian_ids has {int(ids.shape[0])} entries for {count} Gaussians.")
    return ids


def derive_surface_orientation_from_scale_rotation(
    positions: Any,
    scaling: Any,
    rotation_quaternion: Any,
    gaussian_ids: Any | None = None,
    *,
    normal_separation_ratio: float = DEFAULT_NORMAL_SEPARATION_RATIO,
    tangent_separation_ratio: float = DEFAULT_TANGENT_SEPARATION_RATIO,
) -> GaussianSurfaceOrientation:
    """Production path: exact principal axes straight from the 3DGS parameterization.

    ``scaling`` is the LINEAR scale (``model.get_scaling`` = ``exp(_scaling)``),
    ``rotation_quaternion`` the normalized quaternion (``model.get_rotation``).
    No eigen-decomposition is performed -- the rotation's columns already ARE
    the principal axes (see module docstring), so the only thing resolved here
    is their order and sign.
    """

    torch = require_torch()
    positions = torch.as_tensor(positions)
    scaling = torch.as_tensor(scaling, device=positions.device, dtype=positions.dtype)
    quaternion = torch.as_tensor(rotation_quaternion, device=positions.device, dtype=positions.dtype)
    if positions.ndim != 2 or int(positions.shape[1]) != 3:
        raise ValueError("positions must have shape (N, 3).")
    if scaling.shape != positions.shape:
        raise ValueError("scaling must have shape (N, 3).")
    if quaternion.ndim != 2 or int(quaternion.shape[1]) != 4:
        raise ValueError("rotation_quaternion must have shape (N, 4).")
    count = int(positions.shape[0])

    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=_EPS)
    w, x, y, z = quaternion.unbind(-1)
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )
    raw_eigenvalues = scaling.square()
    # Descending order resolves the ONLY ambiguity the stored representation
    # has: scale_0/1/2 carry no ordering guarantee.
    order = torch.argsort(raw_eigenvalues, dim=-1, descending=True, stable=True)
    eigenvalues = raw_eigenvalues.gather(-1, order)
    axes = rotation.gather(-1, order.unsqueeze(-2).expand(-1, 3, -1))

    finite_row = torch.isfinite(scaling).all(dim=-1) & torch.isfinite(quaternion).all(dim=-1) & torch.isfinite(positions).all(dim=-1)
    return _assemble(
        positions=positions,
        gaussian_ids=_resolve_ids(gaussian_ids, count, positions.device),
        eigenvalues=eigenvalues,
        axis_major=axes[..., 0],
        axis_normal=axes[..., 2],
        finite_row=finite_row,
        normal_separation_ratio=normal_separation_ratio,
        tangent_separation_ratio=tangent_separation_ratio,
        source="scale_rotation",
    )


def derive_surface_orientation_from_covariance(
    positions: Any,
    covariance: Any,
    gaussian_ids: Any | None = None,
    *,
    normal_separation_ratio: float = DEFAULT_NORMAL_SEPARATION_RATIO,
    tangent_separation_ratio: float = DEFAULT_TANGENT_SEPARATION_RATIO,
) -> GaussianSurfaceOrientation:
    """Generic path for an arbitrary ``(N, 3, 3)`` covariance (fixtures, tests).

    Runs the SAME ordering/sign canonicalization as the production path via
    the shared :func:`_assemble`, using the already-hardened chunked
    ``eigh`` from :mod:`~osn_gs.surface.torch_gaussian_covariance_frame`
    (cuSOLVER batch ceiling). For a covariance built from a scale/quaternion
    pair the two entry points agree up to floating-point noise.
    """

    torch = require_torch()
    positions = torch.as_tensor(positions)
    covariance = torch.as_tensor(covariance, device=positions.device, dtype=positions.dtype)
    if positions.ndim != 2 or int(positions.shape[1]) != 3:
        raise ValueError("positions must have shape (N, 3).")
    if covariance.ndim != 3 or tuple(covariance.shape[1:]) != (3, 3):
        raise ValueError("covariance must have shape (N, 3, 3).")
    if int(covariance.shape[0]) != int(positions.shape[0]):
        raise ValueError("positions and covariance must have the same leading dimension.")
    count = int(positions.shape[0])

    finite_row = torch.isfinite(covariance).all(dim=-1).all(dim=-1) & torch.isfinite(positions).all(dim=-1)
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    # eigh cannot consume non-finite rows; substitute the identity for those
    # rows only -- they are re-flagged as SEPARABILITY_NON_FINITE in _assemble
    # and still produce exactly one output row (coverage contract).
    identity = torch.eye(3, dtype=symmetric.dtype, device=symmetric.device).expand_as(symmetric)
    symmetric = torch.where(finite_row.reshape(-1, 1, 1), symmetric, identity)
    eigenvalues, eigenvectors = _batched_eigh(symmetric)
    eigenvalues = torch.flip(eigenvalues, dims=(-1,)).clamp_min(0.0)
    eigenvectors = torch.flip(eigenvectors, dims=(-1,))

    return _assemble(
        positions=positions,
        gaussian_ids=_resolve_ids(gaussian_ids, count, positions.device),
        eigenvalues=eigenvalues,
        axis_major=eigenvectors[..., 0],
        axis_normal=eigenvectors[..., 2],
        finite_row=finite_row,
        normal_separation_ratio=normal_separation_ratio,
        tangent_separation_ratio=tangent_separation_ratio,
        source="covariance",
    )
