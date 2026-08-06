---
name: project_baseline_compatible_deferred_schedule_mismatch
description: known issue - gaussian_initialization_mode=baseline_compatible is silently ignored under the adc_post_commit/disabled visible_nurbs_update_schedule
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-06T03:28:55.444Z
---

Known issue, recorded per explicit user request during the worklog 64 3k validation round, NOT fixed: [[project_gaussian_initialization_parity]]'s `gaussian_initialization_mode` flag only reaches `_initialize_canonical` (the production default "initialize" schedule). `initialize_deferred` (used by `visible_nurbs_update_schedule="adc_post_commit"`/`"disabled"`) intentionally ignores the flag and always uses covariance-KNN planar-surfel init, because that path's first post-ADC surface reconstruction (`reconstruct_visible_after_adc`) reuses the model's own scale/rotation as its only orientation evidence at that point.

**Why:** so setting `gaussian_initialization_mode=baseline_compatible` while also using a deferred schedule silently does NOT give isotropic init — a real semantic mismatch between the two config knobs.

**How to apply:** doesn't affect the production default schedule ("initialize"), so no action needed unless a future task actually trains with `adc_post_commit`/`disabled`. If that happens, resolve the mismatch before trusting `gaussian_initialization_mode` there.
