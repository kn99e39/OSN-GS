@echo off
REM Build the vendored 2DGS diff_surfel_rasterization CUDA extension into the
REM active .venv on this Windows system (arch/2dgs-coverage-first-surface,
REM Worklog 96). Mirrors scripts\build_baseline_extensions.bat's VS2022 +
REM TORCH_CUDA_ARCH_LIST pattern for the surfel rasterizer instead of the
REM Graphdeco baseline extensions.
REM
REM Usage:   scripts\build_surfel_extension.bat  [compute_capability]
REM   e.g.   scripts\build_surfel_extension.bat  12.0      (RTX 50-series / Blackwell, default)
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

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "PKG=%ROOT%\osn_gs\render\vendor\diff_surfel_rasterization"

echo Using compiler: & where cl
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

echo(
echo ===== building diff_surfel_rasterization =====
"%PY%" -m pip install --no-build-isolation "%PKG%" < nul
if errorlevel 1 (echo [ERROR] build failed for diff_surfel_rasterization & exit /b 1)

echo(
echo diff_surfel_rasterization built and installed.
endlocal
