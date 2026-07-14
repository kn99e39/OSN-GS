from __future__ import annotations
import traceback

"""OSN-GS-local loader for the vendored diff Gaussian rasterizer."""

from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
import os
import shutil
import subprocess
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


def validate_diff_gaussian_build_environment() -> dict[str, str]:
    """Activate and validate the native toolchain before a CUDA JIT build.

    This intentionally does not compile anything. It fails before scene loading
    when the vendored rasterizer could not be built in the current process.
    """

    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; the diff Gaussian rasterizer requires a CUDA-enabled PyTorch runtime."
        )

    activation = _activate_msvc_environment()
    compiler = shutil.which("cl")
    if compiler is None:
        detail = activation or "MSVC activation did not provide a diagnostic."
        raise RuntimeError(
            "MSVC C++ compiler cl.exe is unavailable in this training process. "
            f"{detail} Open a VS x64 developer environment or install the Desktop C++ workload."
        )
    compiler_dir = str(Path(compiler).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0].lower() != compiler_dir.lower():
        os.environ["PATH"] = compiler_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        where_output = subprocess.check_output(
            ["where", "cl"], text=True, encoding="utf-8", errors="replace"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "MSVC activation found cl.exe through Python, but PyTorch's exact "
            f"compiler probe ('where cl') still failed: {exc}."
        ) from exc
    where_compilers = [line.strip() for line in where_output.splitlines() if line.strip()]
    if not where_compilers:
        raise RuntimeError("MSVC activation produced an empty result for PyTorch's 'where cl' probe.")

    missing_env = [key for key in ("INCLUDE", "LIB") if not os.environ.get(key)]
    if missing_env:
        raise RuntimeError(
            "MSVC activation is incomplete; missing "
            + ", ".join(missing_env)
            + ". Restart the notebook kernel after installing Visual Studio Build Tools."
        )

    from torch.utils.cpp_extension import CUDA_HOME

    cuda_home = Path(CUDA_HOME) if CUDA_HOME else None
    nvcc = (cuda_home / "bin" / "nvcc.exe") if cuda_home is not None else None
    if nvcc is None or not nvcc.exists():
        raise RuntimeError(
            "CUDA toolkit nvcc.exe is unavailable to PyTorch's extension builder. "
            f"CUDA_HOME={CUDA_HOME!r}. Install a toolkit compatible with the PyTorch CUDA runtime."
        )
    try:
        import ninja  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'ninja' is required for the diff Gaussian rasterizer JIT build."
        ) from exc

    return {
        "compiler": where_compilers[0],
        "cuda_home": str(cuda_home),
        "nvcc": str(nvcc),
        "msvc_activation": activation or "already active",
    }


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
    validate_diff_gaussian_build_environment()
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
    return load(
        name="osn_gs_diff_gaussian_rasterization_c",
        sources=sources,
        extra_cflags=["/Zc:preprocessor"],
        extra_cuda_cflags=[
            f"-I{include_dir}",
            "-Xcompiler=/Zc:preprocessor",
        ],
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


def _activate_msvc_environment() -> str | None:
    if os.name != "nt":
        return "MSVC activation is only required on Windows."
    if shutil.which("cl") is not None:
        return "MSVC compiler was already present in PATH."
    try:
        from setuptools._distutils._msvccompiler import _get_vc_env
    except Exception as exc:
        return f"Could not import setuptools MSVC activation helper: {exc}"
    try:
        env = _get_vc_env("x64")
    except Exception as exc:
        return f"Could not load the x64 MSVC environment: {exc}"
    for key, value in env.items():
        os.environ[key.upper()] = value
    if shutil.which("cl") is None:
        return "The x64 MSVC environment was loaded, but cl.exe is still absent from PATH."
    return "Activated the x64 MSVC environment for this training process."


def _safe_import(name: str):
    try:
        return import_module(name)
    except Exception:
        return None


def _vendored_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "diff_gaussian_rasterization"

