from __future__ import annotations

"""OSN-GS-local loader for the vendored 2DGS diff-surfel rasterizer.

Mirrors `osn_gs/render/diff_gaussian_loader.py` for the 2D surfel rasterizer
vendored at `osn_gs/render/vendor/diff_surfel_rasterization` (upstream
`hbb1/diff-surfel-rasterization` @ e0ed020, see that directory's
`OSN_GS_PROVENANCE.md`).

Resolution order, first success wins:

1. an already-installed `diff_surfel_rasterization` package (what the official
   repository's `pip install submodules/diff-surfel-rasterization` produces);
2. the vendored source tree, if its `_C` extension is already importable;
3. a JIT build of the vendored CUDA sources via
   `torch.utils.cpp_extension.load`.

There is intentionally NO torch fallback renderer here. A 2DGS render is
defined by its perspective-correct ray-splat intersection; approximating it in
Python would silently invalidate the whole experiment, so this loader fails
loudly instead.
"""

from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
import sys
import tempfile
import traceback

from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class DiffSurfelBackend:
    settings_cls: type
    rasterizer_cls: type
    source: str


_BACKEND: DiffSurfelBackend | None = None
_LOAD_ERROR: Exception | None = None


def get_diff_surfel_backend() -> DiffSurfelBackend | None:
    global _BACKEND, _LOAD_ERROR

    if _BACKEND is not None:
        return _BACKEND

    for loader in (_load_installed_backend, _load_vendored_backend, _build_vendored_backend):
        try:
            backend = loader()
        except Exception as exc:
            print(f"[OSN-GS] Diff surfel backend loader failed: {loader.__name__}", flush=True)
            traceback.print_exc()
            _LOAD_ERROR = exc
            continue
        if backend is not None:
            _BACKEND = backend
            _LOAD_ERROR = None
            return backend
    return None


def diff_surfel_load_error() -> Exception | None:
    return _LOAD_ERROR


def _load_installed_backend() -> DiffSurfelBackend | None:
    module = _safe_import("diff_surfel_rasterization")
    if module is None:
        return None
    return DiffSurfelBackend(
        module.GaussianRasterizationSettings, module.GaussianRasterizer, "installed package"
    )


def _load_vendored_backend() -> DiffSurfelBackend | None:
    package_root = vendored_surfel_root()
    if not package_root.exists():
        return None
    parent = str(package_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = _safe_import("diff_surfel_rasterization")
    if module is None:
        return None
    return DiffSurfelBackend(
        module.GaussianRasterizationSettings,
        module.GaussianRasterizer,
        f"vendored source ({package_root})",
    )


def _build_vendored_backend() -> DiffSurfelBackend | None:
    package_root = vendored_surfel_root()
    if not package_root.exists():
        return None
    torch = require_torch()
    if not torch.cuda.is_available():
        return None
    extension = _jit_build_extension(package_root)
    package_module = _load_local_python_wrapper(extension)
    return DiffSurfelBackend(
        package_module.GaussianRasterizationSettings,
        package_module.GaussianRasterizer,
        f"vendored JIT build ({package_root})",
    )


def _jit_build_extension(package_root: Path):
    import torch.utils.cpp_extension as cpp_extension

    build_root = Path(tempfile.gettempdir()) / "osn_gs_diff_surfel_rasterization"
    build_root.mkdir(parents=True, exist_ok=True)
    # PACKAGING-ONLY: glm is shared with the vendored 3DGS rasterizer; the two
    # header trees are byte-identical. See the vendored tree's
    # third_party/GLM_SHARED_WITH_3DGS.md.
    include_dir = package_root.parent / "diff_gaussian_rasterization" / "third_party" / "glm"
    sources = [
        str(package_root / "cuda_rasterizer" / "rasterizer_impl.cu"),
        str(package_root / "cuda_rasterizer" / "forward.cu"),
        str(package_root / "cuda_rasterizer" / "backward.cu"),
        str(package_root / "rasterize_points.cu"),
        str(package_root / "ext.cpp"),
    ]
    return cpp_extension.load(
        name="osn_gs_diff_surfel_rasterization_c",
        sources=sources,
        extra_cuda_cflags=[f"-I{include_dir}"],
        build_directory=str(build_root),
        verbose=True,
        with_cuda=True,
        is_python_module=True,
    )


def _load_local_python_wrapper(extension) -> ModuleType:
    module_name = "osn_gs.render._local_diff_surfel_wrapper"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    wrapper_path = Path(__file__).with_name("_diff_surfel_wrapper.py")
    spec = util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {wrapper_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.set_extension(extension)
    return module


def _safe_import(name: str):
    try:
        return import_module(name)
    except Exception:
        return None


def vendored_surfel_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "diff_surfel_rasterization"
