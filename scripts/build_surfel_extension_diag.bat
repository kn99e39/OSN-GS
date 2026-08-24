@echo off
REM Build the DIAGNOSTIC-ONLY 2DGS diff_surfel_rasterization_diag CUDA
REM extension (arch/2dgs-coverage-first-surface, Worklog 107). Mirrors
REM scripts\build_surfel_extension.bat exactly, pointed at the diagnostic
REM sibling package instead of the canonical vendored one.
REM
REM Usage:   scripts\build_surfel_extension_diag.bat  [compute_capability]

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
set "PKG=%ROOT%\osn_gs\render\vendor\diff_surfel_rasterization_diag"

echo Using compiler: & where cl
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

echo(
echo ===== building diff_surfel_rasterization_diag =====
"%PY%" -m pip install --no-build-isolation "%PKG%" < nul
if errorlevel 1 (echo [ERROR] build failed for diff_surfel_rasterization_diag & exit /b 1)

echo(
echo diff_surfel_rasterization_diag built and installed.
endlocal
