---
name: project_benchmark_surface_aligned_covariance
description: "Worklog 29 (orig. 134) — added idealized surface-aligned covariance variant to osn-gs benchmark's synthetic box/cylinder/sphere scenes, selectable via -surf/--surf"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T06:32:27.098Z
---

Worklog 29 (`docs/worklogs/29_synthetic_benchmark_surface_aligned_covariance.md` — originally written as worklog 134, renumbered same-day by a concurrent worklogs-directory renumbering/pruning pass, 2026-07-31) added a second covariance-generation mode to `nurbs_constructor_benchmark/scenes.py`'s synthetic box/cylinder/sphere scenes (the same dataset covered by [[project_osn_gs_benchmark_volumetric_dataset]]). The existing `_baseline_like_surface_covariance` is already tangent-frame aligned (local z axis = exact analytic surface normal) but mixes in log-normal anisotropy-ratio noise to mimic real baseline 3DGS statistics (median ratio 5.44, range 1.5-32x). The new `_surface_aligned_covariance` removes that noise: every Gaussian gets the same fixed flat ratio (`_SURFACE_ALIGNED_RATIO = 12.0`), a uniformly idealized flat tangent-plane disk — confirmed with the user via AskUserQuestion (they picked "idealized flat disk" over "rotation-only" or "reproduce real-training degenerate covariance").

`make_scene(...)` gained a `covariance_mode: "baseline_noisy" | "surface_aligned"` param (default `"baseline_noisy"`, fully backward compatible — added as the last positional param, all 18 existing call sites use ≤4 positional/keyword args). `osn-gs benchmark` gained a `-surf`/`--surf` flag (both dash forms, per explicit user request) that switches `covariance_mode` to `"surface_aligned"` for both the `canonical` and `boundary_first` constructor paths.

**Why:** User wanted a benchmark axis contrasting realistic-noisy vs. idealized covariance shape, independent from the real-DATASET NaN/reliable_count investigation ([[project_full_cloud_continuation_boundary_recovery]]) happening in the same session — these are separate, unrelated threads.

**How to apply:** `--constructor boundary_first` does NOT consume covariance at all (raw point/normal only) — `-surf` produces identical output there by design, not a bug. `--constructor canonical` DOES consume `covariance_scales`/`covariance_rotations`, but already hard-fails with `no_admissible_region` on box/cylinder/sphere regardless of `-surf` — this is the pre-existing disclosed gap from [[project_osn_gs_benchmark_volumetric_dataset]] (closed multi-face topology), not something this change introduced or fixed.
