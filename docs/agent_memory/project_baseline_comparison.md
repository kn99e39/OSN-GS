---
name: project-baseline-comparison
description: "How the local gaussian-splatting baseline is built/run for OSN-GS comparison on this Windows/CUDA13/RTX5080 system, and the fairness caveats"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f58b1e8-0abf-4c0b-a3b4-3b8396c6006c
---

The repo bundles the original Graphdeco 3DGS at `gaussian-splatting/` as a comparison baseline. As of 2026-07-15 it is buildable and runnable on this system (torch 2.12+cu130, RTX 5080 = sm_120, Windows, VS2022). The notebook selects it via `FRAMEWORK_MODE='graphdeco_3dgs'` (now resolves `GS_ROOT` to the local `gaussian-splatting/` folder); `gaussian-splatting/train.py` also runs directly.

**Building its CUDA extensions** (`diff_gaussian_rasterization`, `simple_knn`; `fused_ssim` optional — train.py falls back to torch SSIM): run `scripts/build_baseline_extensions.bat [compute_capability]` (default 12.0 for the 5080). The critical flag is `CL=/Zc:preprocessor` — CUDA 13's CCCL headers reject MSVC's traditional preprocessor with `fatal error C1189` otherwise; also set `TORCH_CUDA_ARCH_LIST` to the GPU's compute capability. Same `/Zc:preprocessor` fix OSN-GS's JIT rasterizer build already uses. Windows gotcha: `import torch` must precede importing any torch C++ extension or you get `DLL load failed`.

Installing `diff_gaussian_rasterization` into the venv makes OSN-GS's `diff_gaussian_loader` prefer it (installed backend) over the vendored JIT build — same source (`forward.cu` identical), so behavior is unchanged and runs skip the JIT step. See [[reference-osn-gs-docs]] → `docs/worklogs/15_baseline_3dgs_comparison_setup.md`.

**Why / fairness caveats (apply before trusting any comparison):** OSN-GS defaults to half-resolution via `--low_vram` (see [[project-notebook-cli-parity]]) while the baseline trains near full resolution (auto-downscaled only if image width >1.6K). They also use different image losses — baseline L1 + D-SSIM (`lambda_dssim=0.2`) vs OSN-GS L1 + MSE with no SSIM (the #1 suspected quality-gap cause in `TODO.md`). To compare like-for-like, match resolution (run OSN-GS with `--no-low_vram`, or pass `-r` to the baseline) and remember the loss difference.
