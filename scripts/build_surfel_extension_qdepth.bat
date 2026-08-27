@echo off
REM Build (JIT, into the shared torch extension build directory) the
REM DIAGNOSTIC-ONLY 2DGS diff_surfel_rasterization_qdepth CUDA extension
REM (arch/2dgs-coverage-first-surface, Worklog 120, Candidate D). Mirrors
REM scripts\build_surfel_extension_diag.bat's environment setup, but does NOT
REM pip-install: osn_gs\render\torch_surfel_query_depth_diagnostics.py loads
REM this extension through torch.utils.cpp_extension.load, and an installed
REM copy would only shadow it.
REM
REM Usage:   scripts\build_surfel_extension_qdepth.bat  [compute_capability]

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

echo Using compiler: & where cl
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

echo(
echo ===== building diff_surfel_rasterization_qdepth (JIT) =====
"%PY%" -c "from osn_gs.render.torch_surfel_query_depth_diagnostics import get_qdepth_extension; get_qdepth_extension(); print('diff_surfel_rasterization_qdepth JIT build OK')"
if errorlevel 1 (echo [ERROR] build failed for diff_surfel_rasterization_qdepth & exit /b 1)
endlocal
