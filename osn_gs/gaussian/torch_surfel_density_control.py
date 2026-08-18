from __future__ import annotations

"""2DGS-compatible adaptive density control.

The official 2DGS implementation keeps 3DGS's adaptive density control loop
and changes only what the 2D surfel primitive forces it to change. This module
records that audit item by item, supplies the 2DGS-parameterized config, and
exposes verification helpers. The actual clone/split/prune transaction is
OSN-GS's existing `osn_gs.gaussian.torch_density_control`, which already owns
stable-ID allocation, ownership transport, Adam-state preservation and the
uncertain-Gaussian masking that the official code has no equivalent for.


=============================================================================
AUDIT: official `scene/gaussian_model.py::densify_and_prune` @ 335ad61
=============================================================================

1. Gradient statistic that triggers densification
   OFFICIAL: `xyz_gradient_accum / denom`, accumulated as
   ``norm(viewspace_point_tensor.grad[update_filter], dim=-1)``. In 2DGS
   `means2D` does NOT participate in the forward pass; the backward writes
   the SCREEN-PROJECTED GRADIENT OF THE 3D CENTER into it
   (`backward.cu`: ``dL_dmean2Ds[idx].x = dL_dtransMats[idx*9+2] * depth *
   0.5 * W``), which is paper section 6.1's "we project the gradient of 3D
   center p_k onto the screen space as an approximation". Its z component is
   never written and stays 0.
   OSN-GS: `add_densification_stats` takes ``norm(grad[:, :2])``. Identical,
   because the third column is 0. NO CHANGE NEEDED. Threshold 0.0002 matches
   the paper and the official default.

2. How tangent-plane scales are inherited or modified
   OFFICIAL clone: `_scaling` copied verbatim (both columns).
   OFFICIAL split: ``log(get_scaling[mask].repeat(N,1) / (0.8*N))`` -- both
   tangent scales shrunk by 0.8*N, identical to 3DGS except that there are
   two columns instead of three.
   OSN-GS `_shape_transaction_candidates` already does exactly this and is
   column-count agnostic. NO CHANGE NEEDED.

3. How child positions are sampled for a planar primitive
   OFFICIAL: ``stds = cat([get_scaling[mask].repeat(N,1), 0*ones_like(...)])``
   -- THE THIRD STANDARD DEVIATION IS ZERO, so children land strictly in the
   parent's tangent plane. This is the one place volumetric 3DGS semantics
   would be wrong. CHANGED: `_shape_transaction_candidates` now pads the
   2-column scale with a zero third std when `model.scale_dim == 2`
   (3-column volumetric behaviour is untouched).

4. How rotation / tangent frames propagate
   OFFICIAL: `_rotation` copied verbatim on both clone and split; the local
   offset is rotated by ``build_rotation(q)``. OSN-GS applies the algebraically
   equivalent quaternion rotation of the offset. NO CHANGE NEEDED -- the child
   inherits the parent's exact tangent frame, so it stays planar and coplanar.

5. Opacity pruning
   OFFICIAL: ``get_opacity < opt.opacity_cull`` with `opacity_cull = 0.05`
   (paper section 6.1: "remove splats with opacity lower than 0.05"). 3DGS
   and OSN-GS default to 0.005. CHANGED via `surfel_density_control_config`.

6. Screen-space size pruning
   OFFICIAL: ``max_radii2D > size_threshold`` with
   ``size_threshold = 20 if iteration > opt.opacity_reset_interval else None``,
   plus ``get_scaling.max(dim=1) > 0.1 * extent``. OSN-GS expresses the same
   thing as `max_screen_size = 20`, `max_scale_ratio = 0.1` and
   `screen_size_prune_from_iter`, which coincides with the official gate when
   it equals `opacity_reset_interval`. `surfel_density_control_config`
   enforces that coincidence.

7. Opacity reset behaviour
   OFFICIAL: ``min(get_opacity, 0.01)`` every `opacity_reset_interval` = 3000
   while `iteration < densify_until_iter`. Identical to OSN-GS
   `TorchGaussianModel.reset_opacity` and the trainer's reset gate. Note the
   reset touches ONLY opacity, so it can never reintroduce normal-direction
   scale. NO CHANGE NEEDED.

8. Low-pass-filter interaction with densification
   The rasterizer clamps every visible splat's screen radius from below:
   ``radius = ceil(max(extent.x, extent.y, cutoff * FilterSize))`` with
   `cutoff = 3.0`, `FilterSize = 0.707106`, so `max_radii2D >= 3` for any
   splat that is rasterized at all. Consequences, both verified by
   `tests/test_surfel_density_control.py`:
   (a) the screen-size prune threshold of 20 is far above that floor, so the
       filter never causes spurious size pruning;
   (b) an edge-on surfel whose true projected extent collapses to a line
       still receives radius >= 3 and therefore still receives gradient and
       densification statistics, which is precisely the degenerate-solution
       stabilization of paper eq. 11.

Fidelity: `OFFICIAL_CODE_FAITHFUL` for items 1-8. No NURBS-aware or otherwise
OSN-GS-specific densification rule is introduced. Stable-ID semantics through
clone/split/prune are OSN-GS's own pre-existing behaviour and are preserved
unchanged (the official code has no stable IDs at all).
"""

from typing import Any

from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.utils.torch_ops import require_torch


# Official `arguments/__init__.py::OptimizationParams` values for 2DGS.
OFFICIAL_OPACITY_CULL = 0.05
OFFICIAL_DENSIFY_GRAD_THRESHOLD = 0.0002
OFFICIAL_PERCENT_DENSE = 0.01
OFFICIAL_DENSIFICATION_INTERVAL = 100
OFFICIAL_DENSIFY_FROM_ITER = 500
OFFICIAL_DENSIFY_UNTIL_ITER = 15_000
OFFICIAL_OPACITY_RESET_INTERVAL = 3000
OFFICIAL_MAX_SCREEN_SIZE = 20.0
OFFICIAL_MAX_SCALE_RATIO = 0.1


def surfel_density_control_config(
    *,
    densify_from_iter: int = OFFICIAL_DENSIFY_FROM_ITER,
    densify_until_iter: int = OFFICIAL_DENSIFY_UNTIL_ITER,
    densification_interval: int = OFFICIAL_DENSIFICATION_INTERVAL,
    opacity_reset_interval: int = OFFICIAL_OPACITY_RESET_INTERVAL,
    max_gaussians: int = 0,
    preserve_adc_gradients: bool = True,
) -> TorchDensityControlConfig:
    """`TorchDensityControlConfig` carrying the official 2DGS ADC parameters.

    Everything except the four scheduling arguments is pinned to the official
    values so a caller cannot accidentally train 2DGS with 3DGS's
    `opacity_cull = 0.005`. `screen_size_prune_from_iter` is tied to
    `opacity_reset_interval`, reproducing the official
    ``size_threshold = 20 if iteration > opt.opacity_reset_interval else None``
    gate.
    """

    return TorchDensityControlConfig(
        densify_from_iter=int(densify_from_iter),
        densify_until_iter=int(densify_until_iter),
        densification_interval=int(densification_interval),
        densify_grad_threshold=OFFICIAL_DENSIFY_GRAD_THRESHOLD,
        prune_opacity_threshold=OFFICIAL_OPACITY_CULL,
        percent_dense=OFFICIAL_PERCENT_DENSE,
        split_samples=2,
        max_screen_size=OFFICIAL_MAX_SCREEN_SIZE,
        max_scale_ratio=OFFICIAL_MAX_SCALE_RATIO,
        max_gaussians=int(max_gaussians),
        opacity_reset_interval=int(opacity_reset_interval),
        screen_size_prune_from_iter=int(opacity_reset_interval),
        preserve_adc_gradients=bool(preserve_adc_gradients),
    )


def coplanarity_residual(model: Any, parent_indices: Any, child_indices: Any) -> Any:
    """|(child_center - parent_center) . t_w| for each (parent, child) pair.

    Zero for children sampled inside the parent's tangent plane. Used by the
    focused tests to verify audit item 3 rather than trusting it.
    """

    torch = require_torch()
    parent_indices = torch.as_tensor(parent_indices, dtype=torch.long, device=model.device).reshape(-1)
    child_indices = torch.as_tensor(child_indices, dtype=torch.long, device=model.device).reshape(-1)
    normals = model.get_normal.detach()[parent_indices]
    delta = model.get_xyz.detach()[child_indices] - model.get_xyz.detach()[parent_indices]
    return (delta * normals).sum(dim=1).abs()
