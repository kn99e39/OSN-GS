from __future__ import annotations

"""Worklog 107 -- diagnostic-only per-pixel renderer surface-representative
identity, for camera-induced visible adjacency.

Investigated first (directive section 2): does the official vendored 2DGS
forward kernel (`osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/
forward.cu`, UNMODIFIED, canonical) already know, per pixel, which surfel its
own `median_depth`/`surf_depth` output actually came from? Yes: `renderCUDA`
already computes `median_contributor` -- the (per-tile, 1-indexed) position in
the sorted contributor list at the exact point running transmittance `T`
crosses 0.5 (forward.cu:407-411, `if (T > 0.5) { median_depth = depth; ...
median_contributor = contributor; }`), and stores it via `n_contrib[pix_id +
H*W]` (forward.cu:437). This is precisely the surfel that `surf_depth`
(`depth_ratio=1.0`) and `depth_median` already report a depth for -- the
renderer's own notion of "the surface at this pixel", not a new heuristic.

What is NOT available: `median_contributor` is a per-TILE list position, and
neither it nor the sorted `point_list`/per-tile `ranges` needed to convert it
into a GLOBAL surfel index are returned to Python by the canonical binding
(`rasterize_points.cu`'s `RasterizeGaussiansCUDA` returns only `(rendered,
out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer)` -- see
that file, unmodified). `n_contrib` is packed inside the opaque `imgBuffer`
blob, consumed only internally by the backward pass.

Per directive section 3: since the exact contributor identity exists
internally but is not exposed, this module uses a SEPARATE DIAGNOSTIC-ONLY
CUDA build -- `osn_gs/render/vendor/diff_surfel_rasterization_diag/` -- a
sibling copy of the vendored package with exactly one addition threaded
through `forward.cu` -> `rasterizer_impl.cu` -> `rasterize_points.cu`: at the
SAME point `median_contributor` is set, also capture `collected_id[j]` (the
GLOBAL surfel index already sitting in a register at that exact loop
iteration -- no `point_list`/`ranges` reconstruction needed) into a new local
`median_surfel_id`, written out as one new (H, W) int32 output tensor,
`-1` where no contributor ever crossed T=0.5. See that directory's own
`forward.cu`/`rasterize_points.cu` for the exact diff (every changed line is
marked `OSN-GS DIAGNOSTIC ADDITION`).

The canonical vendored package (`diff_surfel_rasterization/`), `OSNSurfelRasterizer`,
`TorchGaussianSurfelModel`, and the training path are UNTOUCHED -- this
diagnostic copy is a distinct JIT-built extension (own module name, own build
directory), loaded only by this module, never imported by
`osn_gs/core/torch_pipeline.py` or `osn_gs/core/torch_trainer.py`. Backward.cu
is copied unmodified (this diagnostic never needs gradients); the forward
call here is always issued under `torch.no_grad()`.

Worklog 108 addition -- SAME-FORWARD accepted-contributor accounting. Worklog
104/105's separate backward-gradient contribution diagnostic and this
module's forward-only representative signal come from two DIFFERENT CUDA
executions, which left an unresolved 36,051-surfel discrepancy category
(representative without backward-diagnosed contribution). To settle whether
that gap is renderer semantics or a cross-path artifact, this module now
ALSO captures, in the exact SAME forward execution as `median_surfel_id`
above, a per-PRIMITIVE (not per-pixel) 0/1 flag `forward_accepted`: set
directly at the line `float w = alpha * T;` in `forward.cu` (immediately
after the primitive has passed every forward acceptance check: depth >=
near, power <= 0, alpha >= 1/255, test_T >= 0.0001 -- the canonical
kernel's own accepted-contribution semantics, no new threshold). A benign
race (every writer stores the same value 1) needs no atomic. This answers,
from ONE execution rather than two, whether `MEDIAN_SURFACE_REPRESENTATIVE`
can occur without `FORWARD_ACCEPTED_CONTRIBUTOR` in the same forward pass.
"""

import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EXTENSION = None
_LOAD_ERROR: Exception | None = None


def _diag_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "diff_surfel_rasterization_diag"


def _jit_build_diag_extension():
    import torch.utils.cpp_extension as cpp_extension

    package_root = _diag_root()
    build_root = Path(tempfile.gettempdir()) / "osn_gs_diff_surfel_rasterization_diag"
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
        name="osn_gs_diff_surfel_rasterization_diag_c",
        sources=sources,
        extra_cuda_cflags=[f"-I{include_dir}"],
        build_directory=str(build_root),
        verbose=True,
        with_cuda=True,
        is_python_module=True,
    )


def get_diag_extension():
    """Resolution order (mirrors `diff_surfel_loader.py`): an already pip-
    installed `diff_surfel_rasterization_diag` package (built via
    `scripts/build_surfel_extension_diag.bat`) first, then a JIT build.
    Raises on failure -- no silent fallback, callers must know if the
    diagnostic signal is unavailable."""

    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    try:
        from diff_surfel_rasterization_diag import _C as installed_extension
        _EXTENSION = installed_extension
        return _EXTENSION
    except Exception:
        pass
    try:
        _EXTENSION = _jit_build_diag_extension()
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOAD_ERROR = exc
        raise
    return _EXTENSION


def render_with_pixel_representative(camera: Any, model: Any, background: Any | None = None) -> dict[str, Any]:
    """Diagnostic-only forward render exposing the per-pixel median-depth
    contributor's GLOBAL surfel index, `-1` where no contributor crossed
    T=0.5, and (worklog 108) a per-primitive `forward_accepted` 0/1 flag
    captured in the SAME forward execution. Always runs under
    `torch.no_grad()` -- no gradients, no autograd Function, no `.grad`
    state touched anywhere. Calls the raw extension function directly
    (bypassing the canonical package's autograd wrapper entirely, since no
    backward pass is ever needed here).
    """

    torch = require_torch()
    extension = get_diag_extension()
    if background is None:
        background = torch.zeros((3,), dtype=torch.float32, device=model.device)

    empty = torch.tensor([], device=model.device)
    tanfovx = math.tan(float(camera.FoVx) * 0.5)
    tanfovy = math.tan(float(camera.FoVy) * 0.5)
    with torch.no_grad():
        (
            _num_rendered, color, out_others, radii, _geom, _binning, _img,
            representative_id, forward_accepted,
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
        )

    return {
        "render": color,
        "out_others": out_others,
        "radii": radii,
        "representative_id": representative_id,
        "forward_accepted": forward_accepted,
    }
