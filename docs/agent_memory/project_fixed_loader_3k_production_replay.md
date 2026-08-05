---
name: project_fixed_loader_3k_production_replay
description: worklog 63 - loader fix (worklog 62) did NOT recover real 3k anisotropy/screen-prune; superseded root cause found in worklog 64
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-05T08:10:00.045Z
---

worklog 63: applied worklog 62's FoV/BICUBIC loader fix to a real 3k production OSN-GS run (2900/3000/3100) and compared against pre-fix OSN-GS and Graphdeco baseline. Result: anisotropy median unchanged (35.06→35.68 vs baseline 5.46), screen-size prune unchanged/slightly worse (215,089→224,114) — completion criteria NOT met. Per the task's own fallback instruction, concluded the loader defect is only one of the first-divergence causes, not the dominant real-scale cause. This directly motivated [[project_gaussian_initialization_parity]] (worklog 64), which found and fixed the real dominant cause.

**Why:** lockstep parity (worklog 62) transplanted baseline's own tensors, so it never exercised OSN-GS's own Gaussian initialization pipeline — the loader fix it validated was real but insufficient at production scale.

**How to apply:** when reporting real-checkpoint anisotropy/prune numbers from before worklog 64, use worklog 63's fixed-loader values, not the original pre-loader-fix ones — but note worklog 64's baseline_compatible init is now the production default and supersedes both.
