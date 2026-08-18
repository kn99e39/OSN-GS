# Vendored `diff-surfel-rasterization` provenance

## Source

| item | value |
|---|---|
| upstream repository | `https://github.com/hbb1/diff-surfel-rasterization` |
| reached through | `hbb1/2d-gaussian-splatting` submodule `submodules/diff-surfel-rasterization` |
| parent repo revision inspected | `335ad612f2e783a4e57b9cbc4d1e167bd599fc98` (2025-11-24, "Add Diff-Surfel-Tracing resource to README") |
| **rasterizer revision vendored** | **`e0ed0207b3e0669960cfad70852200a4a5847f61`** (2024-11-10, "fix near cull") |
| glm submodule pinned upstream | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` (glm 0.9.9.9) |
| paper | Huang, Yu, Chen, Geiger, Gao. *2D Gaussian Splatting for Geometrically Accurate Radiance Fields.* SIGGRAPH 2024 / ACM TOG. arXiv:2403.17888**v3** (2025-02-22) |

## What was changed relative to upstream

Every compiled source file is **byte-identical** to upstream
`e0ed020`:

- `cuda_rasterizer/{auxiliary.h,backward.cu,backward.h,config.h,forward.cu,forward.h,rasterizer.h,rasterizer_impl.cu,rasterizer_impl.h}`
- `rasterize_points.cu`, `rasterize_points.h`, `ext.cpp`
- `diff_surfel_rasterization/__init__.py`

Exactly two packaging-only changes were made, neither touching compiled code:

1. `third_party/glm/` was removed and the `-I` include path in `setup.py`
   redirected to the glm tree OSN-GS already vendors for the 3DGS rasterizer
   (`osn_gs/render/vendor/diff_gaussian_rasterization/third_party/glm`). The
   two `glm/` header trees are byte-identical (`diff -r`; only glm's own
   docs/tests/CI files, which are never compiled, are absent). See
   `third_party/GLM_SHARED_WITH_3DGS.md`.
2. This provenance file was added.

`osn_gs/render/_diff_surfel_wrapper.py` is a copy of
`diff_surfel_rasterization/__init__.py` with `from . import _C` replaced by an
injected extension module, so the JIT (`torch.utils.cpp_extension.load`) build
path can reuse the identical autograd `Function`. It mirrors the existing
`osn_gs/render/_diff_gaussian_wrapper.py` arrangement for the 3DGS rasterizer.

## Fidelity classification

`OFFICIAL_CODE_FAITHFUL` — the perspective-correct ray-splat intersection
(`forward.cu::renderCUDA`, eqs. 8-10), the object-space low-pass filter
(`rho = min(rho3d, rho2d)`, `FilterInvSquare = 2.0` i.e. sigma = sqrt(2)/2,
eq. 11), the `DUAL_VISIABLE` camera-facing normal flip, the depth-distortion
accumulation, and both backward passes are the upstream CUDA kernels, unmodified.
