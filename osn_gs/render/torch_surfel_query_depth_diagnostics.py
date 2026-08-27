from __future__ import annotations

"""Worklog 120 -- diagnostic-only probe of the canonical 2DGS traversal state
at ARBITRARY query depths, for Candidate D (renderer reachability).

Why a probe was needed at all (directive section 8: "First inspect whether
existing diagnostic outputs can reconstruct the exact query-depth prefix
state"). They cannot:

  * `final_T` is only the FINAL transmittance after the whole contributor
    list, not T at an arbitrary depth;
  * worklog 110's `contrib_ids`/`contrib_post_median` slot arrays are capped at
    `OSN_GS_MAX_CONTRIB_SLOTS` = 16 and worklog 110 measured 97.4% of real
    pixel-view slots as TRUNCATED, so no prefix can be replayed from them; and
  * even untruncated they carry no per-contributor `alpha` and no
    per-contributor `depth`, which a transmittance prefix needs.

So this module uses a SECOND diagnostic-only CUDA build,
`osn_gs/render/vendor/diff_surfel_rasterization_qdepth/` -- a sibling copy of
worklog 107's `diff_surfel_rasterization_diag` (itself a sibling of the
canonical vendored package). A separate directory, extension name, and build
directory were chosen deliberately over editing `_diag` in place so that every
earlier worklog's replay (107/109/110/112-119) keeps calling a bit-identical,
untouched build and cannot be perturbed by this batch.

The addition is one optional input and four outputs, all marked
`OSN-GS DIAGNOSTIC ADDITION (worklog 120, Candidate D)` inline in that
directory's own `.cu`/`.h` files:

  input   `query_depths`          (H, W, K) camera-space z, <= 0 = unused slot
  output  `query_T`               (H, W, K) running transmittance T at the
                                  moment canonical traversal first reaches an
                                  ACCEPTED contributor whose own `depth` >= the
                                  slot's query depth -- i.e. T accumulated from
                                  every accepted contributor strictly before it
  output  `query_terminated`      (H, W, K) 1 iff the canonical termination
                                  condition `T * (1 - alpha) < 0.0001` (the
                                  kernel's own, unmodified constant) fired at a
                                  contributor before the query depth was reached
  output  `query_reached`         (H, W, K) 1 iff traversal actually reached an
                                  accepted contributor at or beyond the query depth
  output  `query_prefix_count`    (H, W, K) accepted contributors composited
                                  before resolution (provenance only)

Worklog 121 added five further PURELY ADDITIVE outputs. They are diagnostics of
the four above, never inputs to them, and `TestQDepthWorklog121Additivity`
asserts that every pre-existing output stays bit-identical with them present:

  output  `query_resolution_depth`  (H, W, K) the per-pixel `depth` of the
                                    accepted (or termination) event that
                                    resolved the slot; -1 if never resolved
  output  `query_termination_alpha` (H, W, K) the canonical `alpha` at the
                                    termination event, written ONLY for slots
                                    whose verdict is terminated = 1, so the
                                    canonical `test_T = T_pre * (1 - alpha)`
                                    can be rebuilt host-side; -1 otherwise
  output  `query_late_front_count`  (H, W, K) accepted events processed AFTER
                                    the slot resolved whose own per-pixel depth
                                    is STILL in front of the query depth --
                                    the direct measure of the traversal-order
                                    vs physical-depth gap; 0 when resolved with
                                    none, -1 when never resolved
  output  `pixel_inversion_count`   (H, W) accepted events whose depth fell
                                    below the running maximum accepted depth
  output  `pixel_max_backward_jump` (H, W) the largest such backward step

Worklog 122 added an EXHAUSTIVE post-median contributor accounting, again purely
additive, for the candidate B frontier validation. Two optional per-primitive
inputs (`primitive_component`, `primitive_representative_class`) and four outputs:

  output  `post_median_counts`   (H, W, 8) per-category counts of accepted
                                 contributors occurring AFTER the canonical
                                 median-surface event, using worklog 110's own
                                 post-median test (`T <= 0.5` at acceptance,
                                 T pre-update). Categories, see config.h:
                                 0 all / 1 same frozen visible component as the
                                 median representative / 2 different component /
                                 3 unresolved component / 4 representative in
                                 THIS view / 5 in another view only / 6 never a
                                 representative / 7 rho2d low-pass branch
  output  `post_median_weights`  (H, W, 8) matching sums of the canonical
                                 compositing weight `w = alpha * T`
  output  `total_accepted_weight`(H, W) sum of `w` over ALL accepted
                                 contributors, so a post-median CONTRIBUTION
                                 FRACTION is computable
  output  `post_median_depth_stats` (H, W, 3) sum / min / max of
                                 (contributor depth - median depth)

This is deliberately an exhaustive aggregate rather than a bounded list: worklog
110 measured 97.4% of its 16-slot contributor arrays as truncated, and that
truncation would bias any post-median mass measurement.

IMPORTANT semantics, stated precisely because worklog 120 was imprecise about
them: `query_T` is the PRE-UPDATE traversal transmittance at the recorded
resolution event. When `query_terminated == 1`, the quantity the canonical kernel
actually compared against 0.0001f is `query_T * (1 - query_termination_alpha)`,
NOT `query_T` itself -- `query_T` is not bounded by 1e-4 at a termination event.

A probe is resolved ONLY at a contributor the canonical kernel ACCEPTS (one that
passed `depth >= near_n`, `power <= 0`, `alpha >= 1/255` and the termination
test), never at an arbitrary tile-list candidate. The reason is a real property
of 2DGS, not a convenience: `depth` is the intersection of the pixel ray with
the surfel's UNBOUNDED plane and is computed for every candidate before any
acceptance check, so a candidate that never contributes can report an arbitrarily
large intersection depth. An earlier revision of this probe resolved on any
candidate past the near plane, and on the real scene that made ~99% of probes
resolve at the very first list entry with T = 1.0 -- an artifact of tile-list
order, not renderer reachability. It was corrected before any measurement was
reported; see the worklog's Implementation Fidelity Statement.

Residual caveat, disclosed rather than hidden: the canonical tile list is sorted
by each surfel's CENTRE camera-space z, so even accepted contributors' per-pixel
`depth` values are not exactly monotone along the traversal, and a low-pass
(rho2d-branch) acceptance can carry a `depth` far from its own footprint. That is
the canonical renderer's own depth semantics -- the same one `median_depth` is
built from -- and is reported, not corrected.

NO new threshold is introduced anywhere: `terminated` is exactly the canonical
`test_T < 0.0001f` event, at exactly the canonical site, with the canonical
constant. The probe never reads a canonical value into a canonical
computation and never writes into a canonical buffer -- it only observes. The
equivalence tests in `tests/test_observed_occluded_volumetric_audit.py`
(`TestQDepthCanonicalEquivalence`) assert that this build reproduces the
`_diag` build's `render`, `out_others` (median depth included),
`representative_id`, `median_rho3d/rho2d/s_u/s_v` and `radii` exactly, both
with the probe disabled and with it enabled.

Always runs under `torch.no_grad()` -- no gradients, no autograd Function, no
`.grad` state touched anywhere; the raw extension function is called directly.
"""

import math
import tempfile
from pathlib import Path
from typing import Any

from osn_gs.utils.torch_ops import require_torch

# Mirrors OSN_GS_MAX_QUERY_SLOTS in that package's cuda_rasterizer/config.h.
# A pixel needing more simultaneous probes than this is handled by the caller
# issuing additional render passes -- never by dropping queries.
MAX_QUERY_SLOTS = 8

_EXTENSION = None
_LOAD_ERROR: Exception | None = None


def _qdepth_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "diff_surfel_rasterization_qdepth"


def _jit_build_qdepth_extension():
    import torch.utils.cpp_extension as cpp_extension

    package_root = _qdepth_root()
    build_root = Path(tempfile.gettempdir()) / "osn_gs_diff_surfel_rasterization_qdepth"
    build_root.mkdir(parents=True, exist_ok=True)
    include_dir = package_root.parent / "diff_gaussian_rasterization" / "third_party" / "glm"
    sources = [
        str(package_root / "cuda_rasterizer" / "rasterizer_impl.cu"),
        str(package_root / "cuda_rasterizer" / "forward.cu"),
        str(package_root / "cuda_rasterizer" / "backward.cu"),
        str(package_root / "rasterize_points.cu"),
        str(package_root / "ext.cpp"),
    ]
    return cpp_extension.load(
        name="osn_gs_diff_surfel_rasterization_qdepth_c",
        sources=sources,
        extra_cuda_cflags=[f"-I{include_dir}"],
        build_directory=str(build_root),
        verbose=True,
        with_cuda=True,
        is_python_module=True,
    )


def get_qdepth_extension():
    """Resolution order mirrors `torch_surfel_representative_diagnostics.
    get_diag_extension`: an already pip-installed package first, then a JIT
    build. Raises on failure -- no silent fallback, callers must know if the
    diagnostic signal is unavailable."""

    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    try:
        from diff_surfel_rasterization_qdepth import _C as installed_extension
        _EXTENSION = installed_extension
        return _EXTENSION
    except Exception:
        pass
    try:
        _EXTENSION = _jit_build_qdepth_extension()
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOAD_ERROR = exc
        raise
    return _EXTENSION


def render_with_query_depth_probe(
    camera: Any,
    model: Any,
    query_depths: Any | None = None,
    background: Any | None = None,
    primitive_component: Any | None = None,
    primitive_representative_class: Any | None = None,
) -> dict[str, Any]:
    """One diagnostic forward render.

    `query_depths` is either ``None`` (probe disabled -- the kernel takes
    exactly the canonical path and the four query outputs come back at their
    -1 fill) or an ``(H, W, MAX_QUERY_SLOTS)`` float32 CUDA tensor of
    camera-space z values, ``<= 0`` marking an unused slot. Camera-space z is
    the same quantity the canonical render loop's own `depth` variable holds
    (`depths_to_points` unprojects it as a z-depth, not a ray distance).

    Returns every field `render_with_pixel_representative` returns, under the
    same keys and with the same semantics, PLUS `query_T`, `query_terminated`,
    `query_reached`, `query_prefix_count`, and the worklog 121 additions
    `query_resolution_depth`, `query_termination_alpha`,
    `query_late_front_count`, `pixel_inversion_count` and
    `pixel_max_backward_jump`.
    """

    torch = require_torch()
    extension = get_qdepth_extension()
    if background is None:
        background = torch.zeros((3,), dtype=torch.float32, device=model.device)

    empty = torch.tensor([], device=model.device)
    if query_depths is None:
        query_tensor = empty
    else:
        query_tensor = query_depths.to(device=model.device, dtype=torch.float32).contiguous()
    component_tensor = (
        empty if primitive_component is None
        else primitive_component.to(device=model.device, dtype=torch.int32).contiguous()
    )
    representative_tensor = (
        empty if primitive_representative_class is None
        else primitive_representative_class.to(device=model.device, dtype=torch.int32).contiguous()
    )

    tanfovx = math.tan(float(camera.FoVx) * 0.5)
    tanfovy = math.tan(float(camera.FoVy) * 0.5)
    with torch.no_grad():
        (
            _num_rendered, color, out_others, radii, _geom, _binning, _img,
            representative_id, forward_accepted, contrib_ids, contrib_post_median, contrib_count,
            median_rho3d, median_rho2d, median_s_u, median_s_v,
            query_T, query_terminated, query_reached, query_prefix_count,
            query_resolution_depth, query_termination_alpha, query_late_front_count,
            pixel_inversion_count, pixel_max_backward_jump,
            post_median_counts, post_median_weights, total_accepted_weight, post_median_depth_stats,
        ) = extension.rasterize_gaussians(
            background,
            model.get_xyz,
            empty,  # colors_precomp -- unused, SH path
            model.get_opacity,
            model.get_scaling,
            model.get_rotation,
            1.0,  # scale_modifier
            empty,  # transMat_precomp / cov3D_precomp -- unused
            camera.world_view_transform,
            camera.full_proj_transform,
            tanfovx,
            tanfovy,
            int(camera.image_height),
            int(camera.image_width),
            model.get_features,
            model.active_sh_degree,
            camera.camera_center,
            False,  # prefiltered
            False,  # debug
            query_tensor,
            component_tensor,
            representative_tensor,
        )

    return {
        "render": color,
        "out_others": out_others,
        "radii": radii,
        "contrib_ids": contrib_ids,
        "contrib_post_median": contrib_post_median,
        "contrib_count": contrib_count,
        "representative_id": representative_id,
        "forward_accepted": forward_accepted,
        "median_rho3d": median_rho3d,
        "median_rho2d": median_rho2d,
        "median_s_u": median_s_u,
        "median_s_v": median_s_v,
        "query_T": query_T,
        "query_terminated": query_terminated,
        "query_reached": query_reached,
        "query_prefix_count": query_prefix_count,
        "query_resolution_depth": query_resolution_depth,
        "query_termination_alpha": query_termination_alpha,
        "query_late_front_count": query_late_front_count,
        "pixel_inversion_count": pixel_inversion_count,
        "pixel_max_backward_jump": pixel_max_backward_jump,
        "post_median_counts": post_median_counts,
        "post_median_weights": post_median_weights,
        "total_accepted_weight": total_accepted_weight,
        "post_median_depth_stats": post_median_depth_stats,
    }
