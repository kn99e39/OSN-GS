# OSN-GS Worklog 107 diagnostic-only package. Intentionally minimal -- this
# package exists only so `diff_surfel_rasterization_diag._C` (the compiled
# CUDA extension) is importable; `osn_gs/render/torch_surfel_representative_
# diagnostics.py` calls `_C.rasterize_gaussians` directly and does not need
# the autograd-Function wrapper the canonical package provides.
from . import _C  # noqa: F401
