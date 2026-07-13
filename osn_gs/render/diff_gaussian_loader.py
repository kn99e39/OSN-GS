from __future__ import annotations
import traceback

"""OSN-GS-local loader for the vendored diff Gaussian rasterizer."""

from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
import os
import shutil
import sys
import tempfile

from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class DiffGaussianBackend:
    settings_cls: type
    rasterizer_cls: type
    source: str


_BACKEND: DiffGaussianBackend | None = None
_LOAD_ERROR: Exception | None = None


def get_diff_gaussian_backend() -> DiffGaussianBackend | None:
    global _BACKEND, _LOAD_ERROR

    if _BACKEND is not None:
        return _BACKEND

    if _LOAD_ERROR is not None:
        return None

    for loader in (_load_installed_backend, _load_vendored_backend, _build_vendored_backend):
        try:
            backend = loader()

        except Exception as exc:
            print(f"[OSN-GS] Diff Gaussian backend loader failed: {loader.__name__}", flush=True)
            traceback.print_exc()
            _LOAD_ERROR = exc
            continue

        if backend is not None:
            _BACKEND = backend
            _LOAD_ERROR = None
            return backend
    return None


def diff_gaussian_load_error() -> Exception | None:
    return _LOAD_ERROR


def _load_installed_backend() -> DiffGaussianBackend | None:
    module = _safe_import("diff_gaussian_rasterization")
    if module is None:
        return None
    return DiffGaussianBackend(module.GaussianRasterizationSettings, module.GaussianRasterizer, "installed package")


def _load_vendored_backend() -> DiffGaussianBackend | None:
    package_root = _vendored_root()
    if not package_root.exists():
        return None
    parent = str(package_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = _safe_import("diff_gaussian_rasterization")
    if module is None:
        return None
    return DiffGaussianBackend(module.GaussianRasterizationSettings, module.GaussianRasterizer, f"vendored source ({package_root})")


def _build_vendored_backend() -> DiffGaussianBackend | None:
    package_root = _vendored_root()
    if not package_root.exists():
        return None
    torch = require_torch()
    if not torch.cuda.is_available():
        return None
    extension = _jit_build_extension(package_root)
    package_module = _load_local_python_wrapper(extension)
    return DiffGaussianBackend(
        package_module.GaussianRasterizationSettings,
        package_module.GaussianRasterizer,
        f"vendored JIT build ({package_root})",
    )


def _jit_build_extension(package_root: Path):
    import torch.utils.cpp_extension as cpp_extension

    # MSVC may print localized output. PyTorch decodes compiler probes using
    # the process OEM codec, which can fail before the actual build starts.
    os.environ.setdefault("VSLANG", "1033")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    scripts_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
    _activate_msvc_environment()
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    load = cpp_extension.load

    build_root = Path(tempfile.gettempdir()) / "osn_gs_diff_gaussian_rasterization"
    build_root.mkdir(parents=True, exist_ok=True)
    include_dir = package_root / "third_party" / "glm"
    sources = [
        str(package_root / "cuda_rasterizer" / "rasterizer_impl.cu"),
        str(package_root / "cuda_rasterizer" / "forward.cu"),
        str(package_root / "cuda_rasterizer" / "backward.cu"),
        str(package_root / "rasterize_points.cu"),
        str(package_root / "ext.cpp"),
    ]
    extra_cflags: list[str] = []
    extra_cuda_cflags = [f"-I{include_dir}"]
    if os.name == "nt":
        # These are MSVC options; passing them to nvcc on Linux prevents the
        # vendored extension from compiling inside the CUDA container.
        extra_cflags.append("/Zc:preprocessor")
        extra_cuda_cflags.append("-Xcompiler=/Zc:preprocessor")
    return load(
        name="osn_gs_diff_gaussian_rasterization_c",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_root),
        verbose=True,
        with_cuda=True,
        is_python_module=True,
    )


def _load_local_python_wrapper(extension) -> ModuleType:
    module_name = "osn_gs.render._local_diff_gaussian_wrapper"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    wrapper_path = Path(__file__).with_name("_diff_gaussian_wrapper.py")
    spec = util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {wrapper_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.set_extension(extension)
    return module


def _activate_msvc_environment() -> None:
    if os.name != "nt" or shutil.which("cl") is not None:
        return
    try:
        from setuptools._distutils._msvccompiler import _get_vc_env
    except Exception:
        return
    try:
        env = _get_vc_env("x64")
    except Exception:
        return
    for key, value in env.items():
        os.environ[key.upper()] = value


def _safe_import(name: str):
    try:
        return import_module(name)
    except Exception:
        return None


def _vendored_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "diff_gaussian_rasterization"
