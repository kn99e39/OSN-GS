from __future__ import annotations

"""arch/2dgs-coverage-first-surface -- intrinsic 2DGS surfel orientation evidence.

A trained `TorchGaussianSurfelModel` (`osn_gs/gaussian/torch_surfel_model.py`)
already stores its own orthonormal orientation ``R = [t_u, t_v, t_w]`` as a
FIRST-CLASS trained parameter -- the quaternion `_rotation` -- not something
recovered from a covariance's principal axes. This module reads that intrinsic
frame directly off the model:

    surface_normal = model.get_normal       # t_w = t_u x t_v, R's 3rd column
    tangent_axis_u = model.get_tangent_u    # t_u, R's 1st column
    tangent_axis_v = model.get_tangent_v    # t_v, R's 2nd column

**No eigen-decomposition happens anywhere in this module** -- there is no
`eigh` call, no covariance construction, no axis reordering. A 2DGS surfel has
no axis-order ambiguity to resolve in the first place: `t_u`/`t_v`/`t_w` are
definitionally the first, second, and third columns of the trained rotation
matrix, always in that order, regardless of the relative magnitude of the two
tangent scales `s_u`/`s_v`. This is the structural difference from the
volumetric-3DGS path in `torch_gaussian_surface_orientation.py`, where the
axis order genuinely IS ambiguous (three unordered principal axes) and has to
be resolved by descending eigenvalue.

For exactly the same reason, this module does NOT reproduce
`GaussianSurfaceOrientation`'s `axis_separability` diagnostic
(`well_defined` / `tangent_axes_degenerate` / `normal_axis_degenerate` /
`isotropic`). That diagnostic exists to flag when an eigenvalue GAP is too
small to trust an eigen-decomposition's axis ordering -- a question that only
arises when the ordering itself was derived from eigenvalues. A surfel's
`t_u`/`t_v`/`t_w` assignment never depends on `s_u` vs. `s_v`, so there is no
analogous ordering failure mode to diagnose. Reintroducing a lambda2/lambda3-
style reliability check here would manufacture a problem this representation
does not have -- exactly what the architecture directive for this batch
prohibits.

Normal-direction thickness is not a diagnostic field here either: `s_normal`
is not merely small, it does not exist as a tensor entry (`scale_dim == 2`,
see `TorchGaussianSurfelModel`). Any metric that would divide by it is
ill-posed for this representation by construction (see
`torch_surfel_analysis_adapter.py`'s module docstring for the same point
raised against legacy per-primitive-thickness metrics) and is never computed
here.

Sign contract
-------------
`t_w`'s sign is whatever the trained quaternion produced -- this module never
flips it. Comparisons must use unsigned
`osn_gs.surface.torch_gaussian_surface_orientation.unsigned_normal_alignment`
(``|dot(n_i, n_j)|``), exactly as for the volumetric path, so the same surface
is never split solely because two surfels' `t_w` came out antiparallel.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

SURFEL_SCALE_DIM = 2


@dataclass(frozen=True)
class SurfelSurfaceOrientation:
    """Intrinsic per-surfel orientation evidence, one row per input surfel.

    Deliberately NOT `GaussianSurfaceOrientation` -- see module docstring for
    why an axis-separability/eigenvalue diagnostic does not apply here. Only
    the three fields `positions`/`surface_normal`/`gaussian_ids` are read by
    the coverage-first partition
    (`torch_coverage_first_subset_partition.SurfaceOrientationEvidence`); the
    remaining fields are provenance/diagnostics for the review export.
    """

    gaussian_ids: Any  # (N,) int64 -- stable_gaussian_ids, provenance to the trained surfel
    positions: Any  # (N, 3) -- unmodified surfel centers
    tangent_axis_u: Any  # (N, 3) unit, t_u -- R's 1st column, trained order
    tangent_axis_v: Any  # (N, 3) unit, t_v -- R's 2nd column, trained order
    surface_normal: Any  # (N, 3) unit, t_w = t_u x t_v -- SIGN-AMBIGUOUS, never flipped
    tangent_scale_u: Any  # (N,) linear s_u
    tangent_scale_v: Any  # (N,) linear s_v
    source: str = "surfel_intrinsic"

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def derive_surface_orientation_from_surfel(model: Any, gaussian_ids: Any | None = None) -> SurfelSurfaceOrientation:
    """Read `t_u`/`t_v`/`t_w` directly off a trained `TorchGaussianSurfelModel`.

    Fails closed on any model whose `scale_dim` is not 2 -- silently accepting
    a volumetric model here would reintroduce the covariance-minor-axis normal
    definition this module exists to avoid (a caller that wants the volumetric
    path must use `torch_gaussian_surface_orientation.py` explicitly, never a
    fallback inside this function). `gaussian_ids` defaults to the model's own
    `stable_gaussian_ids` (falling back to a positional range only if the model
    was never assigned stable IDs), not a fresh positional index -- unlike the
    volumetric entry points, a surfel's provenance is always available from the
    model itself and should not need a caller-supplied override to be correct.
    """

    torch = require_torch()
    scale_dim = int(getattr(model, "scale_dim", 3))
    if scale_dim != SURFEL_SCALE_DIM:
        raise ValueError(
            f"derive_surface_orientation_from_surfel requires a 2DGS surfel model (scale_dim=={SURFEL_SCALE_DIM}), "
            f"got scale_dim={scale_dim}. Use torch_gaussian_surface_orientation.py for a volumetric model instead "
            "of silently falling back to a covariance-derived normal here."
        )

    positions = model.get_xyz.detach()
    tangent_u = model.get_tangent_u.detach()
    tangent_v = model.get_tangent_v.detach()
    normal = model.get_normal.detach()
    scaling = model.get_scaling.detach()
    count = int(positions.shape[0])

    if gaussian_ids is not None:
        ids = torch.as_tensor(gaussian_ids, device=positions.device).reshape(-1).to(torch.int64)
    elif int(getattr(model, "stable_gaussian_ids", torch.empty(0)).numel()) == count:
        ids = model.stable_gaussian_ids.detach().to(torch.int64)
    else:
        ids = torch.arange(count, dtype=torch.int64, device=positions.device)
    if int(ids.shape[0]) != count:
        raise ValueError(f"gaussian_ids has {int(ids.shape[0])} entries for {count} surfels.")

    return SurfelSurfaceOrientation(
        gaussian_ids=ids,
        positions=positions,
        tangent_axis_u=tangent_u,
        tangent_axis_v=tangent_v,
        surface_normal=normal,
        tangent_scale_u=scaling[:, 0],
        tangent_scale_v=scaling[:, 1],
    )
