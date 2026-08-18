from __future__ import annotations

"""ONE evidence interface for both primitives, so both arms meet ONE evaluator.

The 2DGS branch's research question is whether 2DGS-trained surface evidence is
more suitable for OSN-GS curve-network / NURBS construction than vanilla
`baseline_compatible` 3DGS evidence. That question is only answerable if both
arms are fed into the SAME unmodified downstream constructor chain. This module
is that single entry point: it loads either kind of OSN-GS checkpoint and
returns the identical `(positions, covariance, opacity, stable_ids)` tuple the
existing chain already consumes, dispatching on the primitive the checkpoint
itself records.

Volumetric checkpoints produce exactly what the existing worklog 89/92/94
replays already compute -- `covariance_from_scale_rotation(get_scaling,
get_rotation)` -- so the vanilla arm's numbers stay directly comparable with
the published baseline figures.

Surfel checkpoints go through
`osn_gs.gaussian.torch_surfel_analysis_adapter`, whose default `exact_rank2`
mode reports the true 2DGS geometry: `Sigma = R diag(s_u^2, s_v^2, 0) R^T`,
smallest eigenvalue exactly zero.


A KNOWN CONSEQUENCE, STATED UP FRONT
------------------------------------

Several legacy OSN-GS metrics divide by the per-primitive normal-direction
thickness -- most importantly
`torch_gaussian_manifold_affinity.py`'s
``normal_direction_separation_over_thickness = gap / average_thickness``. For a
volumetric 3D Gaussian that denominator is a real number. For a true 2DGS
surfel it is ZERO, and `extract_covariance_frame` floors it at
``sqrt(1e-12) = 1e-6``, so the ratio saturates for every pair.

That is not a bug in either the adapter or the surfel model: it is a genuine
statement that a per-primitive "band thickness" is undefined for a surface
element, and therefore that any OSN-GS criterion built on it is ill-posed for
2DGS evidence. It is reported as a finding rather than papered over.

For that reason the adapter also exposes `epsilon_regularized` mode, which
substitutes a disclosed ``epsilon_ratio * min(s_u, s_v)`` thickness. Runs using
it MUST say so: it manufactures thickness the primitive does not have, and its
numbers describe the regularized surrogate, not 2DGS.

Fidelity: `OSN_GS_ADAPTATION`. Original 2DGS needs no such adapter because its
downstream stage is TSDF meshing, not covariance-frame structural analysis.
Nothing in the 2DGS training formulation is affected by this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surfel_analysis_adapter import (
    EPSILON_REGULARIZED,
    EXACT_RANK2,
    surfel_analysis_covariance,
)
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel, build_rotation
from osn_gs.utils.torch_ops import require_torch

PRIMITIVE_GAUSSIAN_3D = "gaussian_3d"
PRIMITIVE_SURFEL_2D = "surfel_2d"

VOLUMETRIC_COVARIANCE_MODE = "scale_rotation_3x3"


@dataclass(frozen=True)
class PrimitiveEvidence:
    """Primitive-agnostic evidence bundle for the OSN-GS constructor chain."""

    primitive: str
    model: TorchGaussianModel
    positions: Any  # (N, 3)
    covariance: Any  # (N, 3, 3)
    opacity: Any  # (N,) in [0, 1]
    normals: Any  # (N, 3) unit; intrinsic t_w for surfels, minor eigenvector otherwise
    tangent_scales: Any  # (N, 2) the two largest linear scales
    normal_scale: Any  # (N,) linear normal-direction sigma; exactly 0 for surfels
    stable_gaussian_ids: Any  # (N,) long
    iteration: int
    checkpoint_path: str
    covariance_mode: str

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def is_surfel(self) -> bool:
        return self.primitive == PRIMITIVE_SURFEL_2D

    def describe(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "covariance_mode": self.covariance_mode,
            "count": len(self),
            "iteration": self.iteration,
            "checkpoint": self.checkpoint_path,
        }


def checkpoint_primitive(payload: dict[str, Any]) -> str:
    """Primitive a loaded checkpoint payload describes.

    Volumetric is the default: checkpoints written before this branch existed
    carry neither field and are always 3D Gaussians.
    """

    if int(payload.get("scale_dim", 3)) == 2:
        return PRIMITIVE_SURFEL_2D
    return PRIMITIVE_GAUSSIAN_3D


def load_primitive_model(checkpoint: str | Path, device: str = "cuda") -> tuple[TorchGaussianModel, dict[str, Any]]:
    """Rebuild the trained model from an OSN-GS checkpoint directory or file.

    Deliberately reconstructs the model directly from `model_raw` rather than
    through `load_torch_checkpoint`, which additionally requires an initialized
    `TorchPipelineState`, an optimizer, and NURBS patches -- none of which
    read-only evidence analysis needs.
    """

    torch = require_torch()
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "checkpoint.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    raw = payload["model_raw"]

    primitive = checkpoint_primitive(payload)
    rest = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest:
        degree += 1

    model_cls = TorchGaussianSurfelModel if primitive == PRIMITIVE_SURFEL_2D else TorchGaussianModel
    model = model_cls(sh_degree=degree, device=device)
    model.replace_tensors(
        xyz=raw["xyz"],
        features_dc=raw["features_dc"],
        features_rest=raw["features_rest"],
        opacity=raw["opacity"],
        scaling=raw["scaling"],
        rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"],
        uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"],
        cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"),
        surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    model.active_sh_degree = int(payload.get("active_sh_degree", degree))
    return model, payload


def load_primitive_evidence(
    checkpoint: str | Path,
    device: str = "cuda",
    *,
    surfel_covariance_mode: str = EXACT_RANK2,
    surfel_epsilon_ratio: float = 1e-3,
) -> PrimitiveEvidence:
    """Load a checkpoint of either primitive as one common evidence bundle."""

    torch = require_torch()
    from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation

    model, payload = load_primitive_model(checkpoint, device=device)
    primitive = checkpoint_primitive(payload)
    positions = model.get_xyz.detach()
    opacity = model.get_opacity.detach().reshape(-1)
    stable_ids = model.stable_gaussian_ids.detach()

    with torch.no_grad():
        if primitive == PRIMITIVE_SURFEL_2D:
            covariance = surfel_analysis_covariance(
                model, mode=surfel_covariance_mode, epsilon_ratio=surfel_epsilon_ratio
            )
            rotation = build_rotation(model.get_rotation.detach())
            normals = rotation[:, :, 2]
            scaling = model.get_scaling.detach()
            tangent_scales = torch.sort(scaling, dim=1, descending=True).values
            if surfel_covariance_mode == EPSILON_REGULARIZED:
                normal_scale = float(surfel_epsilon_ratio) * scaling.min(dim=1).values
            else:
                normal_scale = torch.zeros_like(scaling[:, 0])
            covariance_mode = surfel_covariance_mode
        else:
            scaling = model.get_scaling.detach()
            covariance = covariance_from_scale_rotation(scaling, model.get_rotation.detach())
            ordered = torch.sort(scaling, dim=1, descending=True)
            tangent_scales = ordered.values[:, :2]
            normal_scale = ordered.values[:, 2]
            # Minor principal axis of the 3D covariance: the volumetric
            # analogue of the surfel's intrinsic normal.
            rotation = build_rotation(model.get_rotation.detach())
            minor_axis = ordered.indices[:, 2]
            normals = rotation[torch.arange(rotation.shape[0], device=rotation.device), :, minor_axis]
            covariance_mode = VOLUMETRIC_COVARIANCE_MODE

    return PrimitiveEvidence(
        primitive=primitive,
        model=model,
        positions=positions,
        covariance=covariance,
        opacity=opacity,
        normals=normals,
        tangent_scales=tangent_scales,
        normal_scale=normal_scale,
        stable_gaussian_ids=stable_ids,
        iteration=int(payload.get("iteration", 0)),
        checkpoint_path=str(checkpoint),
        covariance_mode=covariance_mode,
    )
