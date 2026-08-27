---
name: project_worklog122_frontier_code_layout
description: Where WL122's candidate B frontier validation code lives; qdepth now has 10 post-median categories
metadata:
  type: project
---

Worklog 122 added a candidate-B-only validation layer. `candidate_b_median_depth.py`
and `shared.aggregate_global` were NOT modified.

- `scripts/devtools/observed_occluded/frontier_validation.py` —
  `evaluate_frontier_closure_for_view` (exhaustive per-view self-closure: G2
  reconstruct → reproject into the SAME camera → frozen candidate B), the exact
  `float32_ulp_distance`, `ClosureAccumulator` (streaming aggregates + cause
  classification + the exact-identity check), `PostMedianAccumulator`,
  `region_table`. Cause codes classify MEASURED facts (pixel changed? how many
  ULPs?) — never a tolerance.
- `scripts/devtools/observed_occluded/frontier_synthetic_contracts.py` — S1..S5
  plus an explicitly OUT-OF-SCOPE translucent control.
- `scripts/devtools/observed_occluded_median_frontier_validation.py` — driver.
- `tests/test_observed_occluded_median_frontier_validation.py` — 34 tests.

**CUDA**: the `_qdepth` sibling now also carries worklog 122's exhaustive
post-median accounting — optional per-primitive inputs `primitive_component` and
`primitive_representative_class`, and outputs `post_median_counts` /
`post_median_weights` (H, W, **10**), `total_accepted_weight` (H, W),
`post_median_depth_stats` (H, W, 3). Categories 8/9 split post-median
contributors by whether their per-pixel depth is in front of or behind the
median — traversal-order post-median does NOT imply physically behind.
The post-median test is worklog 110's own `T <= 0.5` at acceptance; no new
definition was introduced.

**Two build/coding traps, both hit once:**
1. Changing the output arity makes ninja link a STALE `ext.o` (LNK2019). Always
   `rmdir /s /q %TEMP%\osn_gs_diff_surfel_rasterization_qdepth` before
   `scripts\build_surfel_extension_qdepth.bat 12.0`.
2. Never hardcode the category width on the Python side. The synthetic module
   read the (H, W, 10) aggregate as `reshape(-1, 8)` and silently misaligned every
   category; use `len(POST_MEDIAN_CATEGORIES)`. Guarded by
   `test_s4_category_widths_match_the_cuda_layout`.

`out_others` channel offsets (vendored auxiliary.h): DEPTH 0, **ALPHA 1**,
NORMAL 2, **MIDDEPTH 5**, DISTORTION 6.

Full 161-view run: ~149 s. Results in [[project_median_surface_frontier_validation]].
