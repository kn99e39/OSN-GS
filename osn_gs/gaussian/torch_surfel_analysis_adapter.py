from __future__ import annotations

"""READ-ONLY covariance-shaped view of 2DGS surfels for legacy OSN-GS analysis.

Almost every OSN-GS structural-evidence module
(`osn_gs/surface/torch_gaussian_covariance_frame.py` and its ~20 consumers)
takes an ``(N, 3, 3)`` covariance tensor. A 2DGS surfel has no 3D covariance:
its density lives on a 2D tangent disk. This module is the ONE place that
bridges the two, and it exists so that the bridge is explicit, auditable, and
never silently becomes the training representation.

Contract
--------

* Nothing here is differentiable-by-intent, and every returned tensor is
  detached. Training NEVER reads this module -- the trainer, rasterizer,
  density control and checkpoint paths all consume
  `TorchGaussianSurfelModel`'s two-column `_scaling` directly.
* The exact 2DGS geometry is rank-2::

      Sigma = R diag(s_u^2, s_v^2, 0) R^T

  which is what `mode="exact_rank2"` (the default) returns. The smallest
  eigenvalue is exactly zero, i.e. the measured normal-direction band
  thickness of a single primitive is exactly zero -- that is the honest
  answer for a surface element, not a numerical accident.
* `mode="epsilon_regularized"` is available only for legacy code that cannot
  consume a singular covariance. It substitutes a normal-direction sigma of
  ``epsilon_ratio * min(s_u, s_v)`` and MUST be disclosed wherever it is used,
  because it manufactures thickness that the primitive does not have.

Fidelity: `OSN_GS_ADAPTATION`. The original 2DGS method has no such adapter --
it never needs one, because its downstream stage is TSDF meshing rather than
OSN-GS's covariance-frame structural analysis. Nothing in the 2DGS training
formulation is changed by this module's existence.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel, build_rotation
from osn_gs.utils.torch_ops import require_torch


EXACT_RANK2 = "exact_rank2"
EPSILON_REGULARIZED = "epsilon_regularized"


@dataclass(frozen=True)
class SurfelEvidenceView:
    """Covariance-shaped, detached view of a surfel population."""

    positions: Any  # (N, 3)
    covariance: Any  # (N, 3, 3), rank 2 unless epsilon-regularized
    normals: Any  # (N, 3) intrinsic t_w
    tangent_u: Any  # (N, 3)
    tangent_v: Any  # (N, 3)
    scale_u: Any  # (N,) linear s_u
    scale_v: Any  # (N,) linear s_v
    opacity: Any  # (N,)
    stable_gaussian_ids: Any  # (N,) long
    mode: str
    epsilon_ratio: float

    @property
    def normal_sigma(self) -> Any:
        """Normal-direction standard deviation actually encoded in `covariance`."""

        torch = require_torch()
        if self.mode == EXACT_RANK2:
            return torch.zeros_like(self.scale_u)
        return self.epsilon_ratio * torch.minimum(self.scale_u, self.scale_v)


def surfel_analysis_scales_rotations(
    model: TorchGaussianSurfelModel,
    *,
    mode: str = EXACT_RANK2,
    epsilon_ratio: float = 1e-3,
) -> tuple[Any, Any]:
    """(N, 3) linear scales + (N, 4) quaternions, for `covariance_from_scale_rotation`.

    The third scale column is the adapter's fabrication, not a model
    parameter: exactly 0 in `exact_rank2`, `epsilon_ratio * min(s_u, s_v)` in
    `epsilon_regularized`.
    """

    torch = require_torch()
    _validate_mode(mode)
    scaling = model.get_scaling.detach()
    if scaling.shape[1] != 2:
        raise ValueError(
            "surfel_analysis_scales_rotations expects a 2-column surfel scaling; "
            f"got {tuple(scaling.shape)}. Pass a TorchGaussianSurfelModel."
        )
    if mode == EXACT_RANK2:
        third = torch.zeros_like(scaling[:, :1])
    else:
        third = float(epsilon_ratio) * scaling.min(dim=1, keepdim=True).values
    return torch.cat([scaling, third], dim=1), model.get_rotation.detach()


def surfel_analysis_covariance(
    model: TorchGaussianSurfelModel,
    *,
    mode: str = EXACT_RANK2,
    epsilon_ratio: float = 1e-3,
) -> Any:
    """(N, 3, 3) ``R diag(s^2) R^T`` covariance view. See module contract."""

    scales, rotations = surfel_analysis_scales_rotations(
        model, mode=mode, epsilon_ratio=epsilon_ratio
    )
    rotation = build_rotation(rotations)
    torch = require_torch()
    return rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)


def surfel_evidence_view(
    model: TorchGaussianSurfelModel,
    *,
    mode: str = EXACT_RANK2,
    epsilon_ratio: float = 1e-3,
) -> SurfelEvidenceView:
    """Bundle everything the OSN-GS structural-evidence stage reads."""

    _validate_mode(mode)
    rotation = build_rotation(model.get_rotation.detach())
    scaling = model.get_scaling.detach()
    return SurfelEvidenceView(
        positions=model.get_xyz.detach(),
        covariance=surfel_analysis_covariance(model, mode=mode, epsilon_ratio=epsilon_ratio),
        normals=rotation[:, :, 2],
        tangent_u=rotation[:, :, 0],
        tangent_v=rotation[:, :, 1],
        scale_u=scaling[:, 0],
        scale_v=scaling[:, 1],
        opacity=model.get_opacity.detach().reshape(-1),
        stable_gaussian_ids=model.stable_gaussian_ids.detach(),
        mode=mode,
        epsilon_ratio=float(epsilon_ratio),
    )


def _validate_mode(mode: str) -> None:
    if mode not in {EXACT_RANK2, EPSILON_REGULARIZED}:
        raise ValueError(
            f"mode must be {EXACT_RANK2!r} or {EPSILON_REGULARIZED!r}, got {mode!r}"
        )
