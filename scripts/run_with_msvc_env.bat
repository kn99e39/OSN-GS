@echo off
REM Run a command with the MSVC x64 developer environment active.
REM
REM Both diagnostic CUDA siblings (`diff_surfel_rasterization_diag`, worklog
REM 107, and `diff_surfel_rasterization_qdepth`, worklog 120) are loaded through
REM torch.utils.cpp_extension.load, which invokes `where cl` on EVERY load --
REM including cache hits -- so any script touching them must run under vcvars64.
REM
REM Usage:   scripts\run_with_msvc_env.bat <command> [args...]
REM Example: scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_volumetric_audit.py -q

setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
if errorlevel 1 (echo [ERROR] Could not activate the VS x64 developer environment. & exit /b 1)

set "TORCH_CUDA_ARCH_LIST=12.0"
set "CL=/Zc:preprocessor"
set "DISTUTILS_USE_SDK=1"
set "VSLANG=1033"

%*
exit /b %errorlevel%
