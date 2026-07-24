---
name: project-branch-rule
description: All OSN-GS work must happen on the voxel-surface-regions branch only
metadata: 
  node_type: memory
  type: project
  originSessionId: ce2402f7-496a-46cf-8bcd-e1cd8bc965d3
---

The user stated (2026-07-16): always work ONLY on the `voxel-surface-regions` branch in the OSN-GS repo. Never commit to `main` or create other branches unless the user explicitly says so.

**Why:** the voxel-driven NURBS migration (see `OSN_GS_Voxel_Driven_NURBS_Migration_Plan.md` at repo root) is being developed in isolation on this branch.

**How to apply:** before any commit, verify `git branch --show-current` returns `voxel-surface-regions`; if not, stop and switch/ask rather than committing.
