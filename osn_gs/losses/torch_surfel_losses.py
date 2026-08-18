from __future__ import annotations

"""2DGS geometric regularization: depth distortion and normal consistency.

Implements Section 5 of arXiv:2403.17888v3, matching the official
implementation `hbb1/2d-gaussian-splatting` @ 335ad61 (`train.py`) and the
vendored rasterizer `hbb1/diff-surfel-rasterization` @ e0ed020.

The total objective is the paper's eq. 16::

    L = L_c + alpha * L_d + beta * L_n

`L_c` stays OSN-GS's existing photometric term
(`osn_gs.losses.torch_losses.image_reconstruction_loss`, i.e.
``(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)``), which is the same
formula the official 2DGS `train.py` uses. Only `L_d` and `L_n` are new.


=============================================================================
DEPTH DISTORTION -- paper vs official code
=============================================================================

PAPER_FORMULATION (eq. 13)::

    L_d = sum_{i,j} omega_i omega_j |z_i - z_j|

    omega_i = alpha_i Ghat_i(u(x)) prod_{j<i} (1 - alpha_j Ghat_j(u(x)))
    z_i     = depth of the i-th perspective-correct ray-splat intersection

i.e. pairwise separation of the intersection depths themselves, weighted by
the alpha-blending contributions, summed over intersections along a ray.

OFFICIAL_CODE_FORMULATION: the same Mip-NeRF360-style quantity, but evaluated
on a NORMALIZED INVERSE-DEPTH (disparity) coordinate rather than on `z`
directly, and accumulated by the O(N) forward recursion inside the CUDA
kernel. From the vendored `cuda_rasterizer/forward.cu::renderCUDA`::

    m           = far_n / (far_n - near_n) * (1 - near_n / depth)
    A           = 1 - T
    distortion += (m*m*A + M2 - 2*m*M1) * w        # w = alpha * T = omega_i
    M1         += m * w
    M2         += m*m * w

with `near_n = 0.2`, `far_n = 100.0` fixed device constants in
`cuda_rasterizer/auxiliary.h`. That recursion is exactly
``sum_{i,j} omega_i omega_j |m_i - m_j|`` for the monotonically increasing
`m(z)`, so the structure of eq. 13 is preserved and only the depth coordinate
differs -- `m` compresses far depths, so distant intersections are penalized
less than raw `z` would penalize them. The official `train.py` then reduces
with ``rend_dist.mean()`` over pixels (the paper writes a sum).

THIS BRANCH IMPLEMENTS: **OFFICIAL_CODE_FORMULATION**.

WHY: the accumulation lives inside the official CUDA kernel, which this branch
vendors byte-identically in order to keep the perspective-correct ray-splat
rasterization itself unmodified (see
`osn_gs/render/vendor/diff_surfel_rasterization/OSN_GS_PROVENANCE.md`).
Switching to the paper's raw-`z` coordinate would mean editing
`forward.cu`/`backward.cu`, i.e. trading a renderer that is provably the
official one for a loss that matches the paper's notation more literally. The
paper itself points at the efficient implementation ("we implement this
regularization term efficiently with CUDA in a manner similar to [Sun et al.
2022b]"), so the official recursion is the paper's own intended realization.
Mixing the two -- e.g. re-deriving a raw-`z` pairwise term in Python on top of
the CUDA `m`-space one -- is exactly what the branch contract forbids.


=============================================================================
NORMAL CONSISTENCY -- paper vs official code
=============================================================================

PAPER_FORMULATION (eq. 14), per ray::

    L_n = sum_i omega_i (1 - n_i^T N)
        = A - R . N          where A = sum_i omega_i,  R = sum_i omega_i n_i

`n_i` is the splat normal oriented toward the camera, `N` the depth-derived
normal of eq. 15 evaluated at the median-intersection surface point `p_s`.

OFFICIAL_CODE_FORMULATION (`train.py` + `gaussian_renderer/__init__.py`), per
pixel::

    rend_normal = allmap[2:5]                       # = R, alpha-weighted, un-normalized
    surf_normal = depth_to_normal(cam, surf_depth) * rend_alpha.detach()   # = N * A_detached
    normal_error = 1 - (rend_normal * surf_normal).sum(dim=0)
    L_n = normal_error.mean()                       # mean over pixels

i.e. ``1 - A * (R . N)`` rather than the paper's ``A - R . N``. The two agree
exactly where the ray is saturated (`A = 1`) and differ in how partially
transparent rays are weighted: the paper scales the whole per-ray residual by
the accumulated weight, the code scales only the alignment term and leaves the
constant at 1. The code additionally reduces with a mean over pixels where the
paper writes a sum, and `surf_depth` is the `depth_ratio` mix of expected and
median depth rather than strictly the median point `p_s`
(`depth_ratio = 1` recovers the paper's median).

THIS BRANCH IMPLEMENTS: **OFFICIAL_CODE_FORMULATION**
(`normal_consistency_loss`).

WHY: it is the formulation the published 2DGS results were produced with, it
is coherent end-to-end with the vendored rasterizer's un-normalized
alpha-weighted `rend_normal` output, and the branch contract requires
reproducing ONE coherent original formulation rather than blending the two.
The paper form is implemented separately as
`normal_consistency_loss_paper_form` for DIAGNOSTIC USE ONLY -- it is never
added to the training objective, and exists so the size of the discrepancy can
be measured rather than assumed.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


# Official `train.py` staging. Documented here as data so the trainer cannot
# quietly drift from it.
OFFICIAL_DIST_FROM_ITER = 3000
OFFICIAL_NORMAL_FROM_ITER = 7000
# Official `arguments/__init__.py::OptimizationParams` defaults.
OFFICIAL_LAMBDA_NORMAL = 0.05
OFFICIAL_LAMBDA_DIST_DEFAULT = 0.0
# Paper section 6.1 / official eval scripts.
PAPER_LAMBDA_DIST_BOUNDED = 1000.0
PAPER_LAMBDA_DIST_UNBOUNDED = 100.0


@dataclass(frozen=True)
class SurfelRegularizationSchedule:
    """Weights and activation iterations for the 2DGS geometric regularizers.

    Defaults reproduce the official `train.py` schedule verbatim:
    ``lambda_dist`` active for ``iteration > 3000``, ``lambda_normal`` active
    for ``iteration > 7000``. `lambda_dist`'s default of 0.0 is the official
    `OptimizationParams` default; the official evaluation scripts pass 1000
    (DTU, bounded), 100/10 (Tanks and Temples), matching the paper's
    ``alpha = 1000`` bounded / ``alpha = 100`` unbounded.
    """

    lambda_dist: float = OFFICIAL_LAMBDA_DIST_DEFAULT
    lambda_normal: float = OFFICIAL_LAMBDA_NORMAL
    dist_from_iter: int = OFFICIAL_DIST_FROM_ITER
    normal_from_iter: int = OFFICIAL_NORMAL_FROM_ITER

    def active_lambda_dist(self, iteration: int) -> float:
        return float(self.lambda_dist) if int(iteration) > int(self.dist_from_iter) else 0.0

    def active_lambda_normal(self, iteration: int) -> float:
        return float(self.lambda_normal) if int(iteration) > int(self.normal_from_iter) else 0.0

    def matches_official_staging(self) -> bool:
        return (
            int(self.dist_from_iter) == OFFICIAL_DIST_FROM_ITER
            and int(self.normal_from_iter) == OFFICIAL_NORMAL_FROM_ITER
        )


def depth_distortion_loss(render_package: dict[str, Any]) -> Any:
    """OFFICIAL_CODE_FORMULATION of eq. 13: ``render_pkg["rend_dist"].mean()``.

    The per-pixel pairwise term is accumulated by the vendored CUDA kernel; see
    the module docstring for the paper-vs-code coordinate difference.
    """

    return render_package["rend_dist"].mean()


def normal_consistency_loss(render_package: dict[str, Any]) -> Any:
    """OFFICIAL_CODE_FORMULATION of eq. 14. Reproduces official `train.py`.

    ``mean_pixels( 1 - sum_c rend_normal_c * surf_normal_c )`` where
    `surf_normal` already carries the detached accumulated alpha applied by
    `OSNSurfelRasterizer`.
    """

    rend_normal = render_package["rend_normal"]
    surf_normal = render_package["surf_normal"]
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    return normal_error.mean()


def normal_consistency_loss_paper_form(render_package: dict[str, Any]) -> Any:
    """PAPER_FORMULATION of eq. 14. DIAGNOSTIC ONLY -- never in the objective.

    ``mean_pixels( A - R . N )`` with ``A`` the accumulated alpha, ``R`` the
    alpha-weighted rasterized normal and ``N`` the unit depth-derived normal.
    Provided so the gap to `normal_consistency_loss` can be measured; adding
    it to training would mean implementing neither formulation coherently.
    """

    torch = require_torch()
    rend_normal = render_package["rend_normal"]
    alpha = render_package["rend_alpha"]
    # Undo the detached alpha scaling the renderer applied, recovering unit N.
    surf_normal_unit = torch.nn.functional.normalize(render_package["surf_normal"], dim=0, eps=1e-12)
    alignment = (rend_normal * surf_normal_unit).sum(dim=0)[None]
    return (alpha - alignment).mean()


def surfel_regularization_terms(
    render_package: dict[str, Any],
    schedule: SurfelRegularizationSchedule,
    iteration: int,
) -> tuple[Any, Any, float, float]:
    """Weighted (dist_loss, normal_loss, active_lambda_dist, active_lambda_normal).

    Both terms are evaluated unconditionally so their raw magnitudes stay
    loggable before activation; only the weights are staged, exactly as the
    official `train.py` does (`lambda_dist = opt.lambda_dist if iteration >
    3000 else 0.0`).
    """

    lambda_dist = schedule.active_lambda_dist(iteration)
    lambda_normal = schedule.active_lambda_normal(iteration)
    dist_loss = lambda_dist * depth_distortion_loss(render_package)
    normal_loss = lambda_normal * normal_consistency_loss(render_package)
    return dist_loss, normal_loss, lambda_dist, lambda_normal
