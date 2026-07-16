@echo off
REM Build the Graphdeco gaussian-splatting CUDA extensions (diff_gaussian_rasterization,
REM simple_knn) into the active .venv on this Windows system so FRAMEWORK_MODE='graphdeco_3dgs'
REM (and gaussian-splatting/train.py directly) can run. fused_ssim is optional; train.py
REM falls back to a pure-torch SSIM when it is absent.
REM
REM Usage:   scripts\build_baseline_extensions.bat  [compute_capability]
REM   e.g.   scripts\build_baseline_extensions.bat  12.0      (RTX 50-series / Blackwell, default)
REM          scripts\build_baseline_extensions.bat  8.9       (RTX 40-series / Ada)
REM
REM Requires Visual Studio 2022 (Desktop C++ workload) and a CUDA toolkit matching the
REM installed PyTorch build. CL=/Zc:preprocessor is required by CUDA's CCCL headers.

setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%"
if errorlevel 1 (echo [ERROR] Could not activate the VS x64 developer environment. & exit /b 1)

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=12.0"
set "TORCH_CUDA_ARCH_LIST=%ARCH%"
set "CL=/Zc:preprocessor"
set "DISTUTILS_USE_SDK=1"
set "VSLANG=1033"

REM Resolve repo root as the parent of this script's directory.
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "SUB=%ROOT%\gaussian-splatting\submodules"

echo Using compiler: & where cl
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

for %%E in (simple-knn diff-gaussian-rasterization) do (
    echo(
    echo ===== building %%E =====
    "%PY%" -m pip install --no-build-isolation "%SUB%\%%E" < nul
    if errorlevel 1 (echo [ERROR] build failed for %%E & exit /b 1)
)
echo(
echo All baseline CUDA extensions built and installed.
endlocal
