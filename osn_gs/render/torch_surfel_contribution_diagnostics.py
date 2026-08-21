from __future__ import annotations

"""Worklog 105 -- diagnostic-only per-surfel renderer contribution evidence.

Worklog 104 measured that 713,540 surfels never have their CENTER classified
`on_observed_surface` by the canonical Phase-C point-sample query, and found
that 99.98% of those still have `radii > 0` in many views -- a signal too
weak (projection/culling only, not occlusion-aware) to settle whether they
actually participate in the trained renderer's image formation. This module
answers that question using the RASTERIZER'S OWN operational contribution
semantics, verified directly against the vendored CUDA source below.

Exact contribution rule (verified against the vendored, UNCHANGED files
`osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/forward.cu`
and `backward.cu`, `renderCUDA`): for one (surfel, pixel) pair, a surfel is
an ACCEPTED CONTRIBUTOR iff, in per-tile front-to-back order:

    1. the ray-splat intersection plane is well-defined (`p.z != 0`);
    2. the intersection depth clears the near plane (`depth >= near_n`);
    3. the Gaussian falloff exponent is valid (`power <= 0`, i.e. `rho >= 0`);
    4. `alpha = min(0.99, opacity * exp(power)) >= 1/255` (the kernel's own
       alpha-acceptance floor -- not a value this module invents);
    5. the running transmittance has not yet saturated when this splat is
       reached: `test_T = T * (1 - alpha) >= 0.0001` (the kernel's own
       early-termination floor).

Only surfels that clear all five conditions execute `last_contributor =
contributor` (forward.cu:423) and accumulate `w = alpha * T` into the pixel
color (forward.cu:396-418); everything else is skipped and contributes
nothing to that pixel. This is exactly the `omega_i = alpha_i * T_i`
compositing weight Worklog 104 described as unavailable to Python -- and it
still is not directly returned by the canonical forward call. What IS
directly available, unmodified, is the vendored BACKWARD kernel
(`backward.cu:144-443`), which every ordinary training step already runs: it
re-derives the IDENTICAL five-condition test in reverse (`backward.cu:279,
291, 302, 311-312, 315-317`) and, only for accepted contributors, does
`atomicAdd(&dL_dcolors[global_id * C + ch], (alpha * T) * dL_dpixel[ch])`
(backward.cu:339) and `atomicAdd(&dL_dopacity[global_id], G * dL_dalpha)`
(backward.cu:443).

Diagnostic instrumentation choice (directive section 2). The directive's
preferred approach is a diagnostic COPY of the vendored CUDA extension with
new atomic per-primitive accounting buffers threaded through
`forward.cu`/`rasterizer_impl.cu`/`rasterize_points.cu`/`ext.cpp` and a
separate build target. That was inspected and is possible, but requires
touching five files across the CUDA/C++/pybind boundary and a fresh CUDA
compile, for a benefit this module gets for free a different way: the
backward kernel above (0 lines changed, already compiled, already exercised
every training step) already computes `dL_dcolors[global_id]` with EXACTLY
the accepted-contribution semantics above, using `dchannel_dcolor = alpha *
T` unconditionally (backward.cu:320-321) -- not an approximation of it, the
same arithmetic. Choosing a diagnostic loss `L = render_unclamped.sum()`
makes `dL_dpixel[ch] = 1` uniformly, so `dL_dcolors[global_id, ch]` becomes
exactly `sum over this surfel's accepted pixels of (alpha * T)` -- a
strictly positive quantity whenever the surfel accepted-contributed to at
least one pixel in this view, and structurally zero otherwise (no accepted
pixel exists to atomicAdd into). Propagated one SH band further
(`computeColorFromSH`, backward.cu:20-56), `dL_dsh[0] = SH_C0 * dL_dRGB` is
an unconditional linear map (`SH_C0` a nonzero constant, independent of view
direction or SH degree), so `model._features_dc.grad` carries the identical
zero/nonzero pattern. This module therefore uses `torch.autograd.grad`
against `model._features_dc` -- NOT `.backward()` (so no `.grad` attribute
on any model parameter is ever mutated) -- as its sole instrumentation,
adding NO new CUDA code, NO new build target, and NO change whatsoever to
`osn_gs/render/vendor/diff_surfel_rasterization/`,
`osn_gs/render/surfel_rasterizer.py`, `TorchGaussianSurfelModel`, or any
training path. `torch_pipeline.py`/`torch_trainer.py` do not import this
module (see `tests/test_surfel_contribution_diagnostics.py::
test_canonical_training_modules_do_not_import_the_diagnostic_module`).

What this DOES prove, per view: whether a surfel was an accepted contributor
at least once (boolean), and the accumulated compositing weight `sum(alpha *
T)` over the pixels it contributed to. What this does NOT recover: the exact
CONTRIBUTING PIXEL COUNT or the MAXIMUM SINGLE-PIXEL weight -- both require
per-pixel-per-primitive bookkeeping that the vendored kernel's atomicAdd
already collapses before returning to Python, and recovering them would
require the literal kernel-copy approach described above. This limitation is
reported explicitly (never silently substituted), matching Worklog 104's own
convention of documenting exactly what a signal does and does not prove.

Backward-pass floating-point note: `alpha * T` for an accepted contribution
is always `>= (1/255) * T_min`, where `T_min` at the moment of acceptance is
itself `>= 0.0001` (the kernel's own termination floor) -- so the smallest
possible accepted weight is bounded well above float32 zero (`>= 3.9e-3 *
1e-4 ~= 3.9e-7`), leaving no realistic risk of an accepted contribution
underflowing to an indistinguishable-from-zero gradient.
"""

from typing import Any, Callable

from osn_gs.utils.torch_ops import require_torch


def compute_renderer_contribution_for_view(camera: Any, model: Any, rasterizer: Any) -> tuple[Any, Any, dict[str, Any]]:
    """One training view's diagnostic contribution signal.

    Returns `(contributed_this_view, accumulated_weight_this_view, package)`:
    both tensors are `(P,)`, `P = len(model)`; `package` is the SAME dict
    `rasterizer.render()` would return for a normal (non-diagnostic) call --
    every key in it is exactly the canonical forward output, unmodified, and
    is not itself touched by the `torch.autograd.grad` call below (backward
    never mutates already-computed forward tensors).

    No model parameter's `.grad` attribute is read or written -- this uses
    `torch.autograd.grad`, not `.backward()`, so the model's persistent
    gradient state (if any exists from an unrelated training loop reusing
    the same object) is left untouched.
    """

    torch = require_torch()
    package = rasterizer.render(camera, model)
    loss = package["render_unclamped"].sum()
    (grad,) = torch.autograd.grad(loss, (model._features_dc,), retain_graph=False, create_graph=False, allow_unused=True)
    count = len(model)
    if grad is None:
        weight = torch.zeros((count,), dtype=torch.float32, device=model.device)
    else:
        # Channel 0 alone: forward.cu's `w = alpha * T` is channel-independent
        # and `dL_dpixel[ch] = 1` uniformly for a sum-loss, so all three
        # channels of `dL_dsh[0]` carry the identical value -- see module
        # docstring. Using one channel avoids a spurious 3x inflation from
        # summing three numerically-identical values.
        weight = grad.detach()[:, 0, 0].abs()
    contributed = weight > 0
    return contributed, weight, package


class RendererContributionEvidence:
    """Per-surfel aggregate contribution accounting across a set of training
    views. `count = P`, matching `len(model)` at construction time.

    `max_single_view_accumulated_weight` is the largest PER-VIEW accumulated
    weight (sum of `alpha*T` over that one view's contributing pixels) seen
    across all views -- NOT a per-pixel maximum (see module docstring for
    exactly why per-pixel granularity is not recoverable this way)."""

    def __init__(self, count: int, device: Any):
        torch = require_torch()
        self.count = count
        self.contributing_view_count = torch.zeros((count,), dtype=torch.int32, device=device)
        self.accumulated_weight_sum = torch.zeros((count,), dtype=torch.float32, device=device)
        self.max_single_view_accumulated_weight = torch.zeros((count,), dtype=torch.float32, device=device)
        self.ever_contributed = torch.zeros((count,), dtype=torch.bool, device=device)

    def update(self, contributed_this_view: Any, weight_this_view: Any) -> None:
        torch = require_torch()
        self.contributing_view_count += contributed_this_view.to(self.contributing_view_count.dtype)
        self.accumulated_weight_sum += weight_this_view
        self.max_single_view_accumulated_weight = torch.maximum(self.max_single_view_accumulated_weight, weight_this_view)
        self.ever_contributed = self.ever_contributed | contributed_this_view


def accumulate_renderer_contribution_evidence(
    cameras: list, model: Any, rasterizer: Any, *, progress: Callable[[str], None] | None = None
) -> RendererContributionEvidence:
    """Drives `compute_renderer_contribution_for_view` over every camera and
    aggregates into `RendererContributionEvidence`. Diagnostic-only: never
    called by `osn_gs/core/torch_pipeline.py` or `osn_gs/core/torch_trainer.py`.
    """

    count = len(model)
    evidence = RendererContributionEvidence(count, model.device)
    for index, camera in enumerate(cameras):
        contributed, weight, package = compute_renderer_contribution_for_view(camera, model, rasterizer)
        evidence.update(contributed, weight)
        del package
        if progress is not None and index % 20 == 0:
            progress(f"renderer contribution diagnostics: view {index + 1}/{len(cameras)}")
    return evidence
