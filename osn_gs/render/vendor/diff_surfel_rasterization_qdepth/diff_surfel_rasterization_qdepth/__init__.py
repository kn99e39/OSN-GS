# OSN-GS Worklog 120 diagnostic-only package (Candidate D -- renderer
# reachability). A SECOND sibling of the canonical vendored
# `diff_surfel_rasterization`, kept physically separate from worklog 107's
# `diff_surfel_rasterization_diag` so that every earlier worklog's replay
# (107/109/110/112-119) keeps calling a bit-identical, untouched build.
# Intentionally minimal -- this package exists only so
# `diff_surfel_rasterization_qdepth._C` is importable;
# `osn_gs/render/torch_surfel_query_depth_diagnostics.py` calls
# `_C.rasterize_gaussians` directly and never needs the autograd-Function
# wrapper the canonical package provides.
from . import _C  # noqa: F401
