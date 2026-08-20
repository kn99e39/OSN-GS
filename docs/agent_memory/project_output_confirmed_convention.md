---
name: project-output-confirmed-convention
description: "output/confirmed/ holds intermediate outputs the user has visually reviewed -- 'confirmed' means simply checked, NOT an architecture/Gate approval"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-20T06:12:44.071Z
---

The user stated (2026-08-20): going forward, intermediate output artifacts they have reviewed get moved into `output/confirmed/` (preserving the original subfolder structure, e.g. `output/confirmed/arch_2dgs_coverage_first_surface/...`). They already did this themselves for several Worklog 105-107 export directories (`extent_ab`, `arch_2dgs_coverage_first_surface`, `osn_gs_2dgs_coverage_first_subset_partition`, `osn_gs_coverage_first_subset_partition[_v2]`, `osn_gs_scene_latent_coverage_audit[_subdivided]`).

**Critical semantic the user explicitly called out**: "confirmed" here means the user has simply LOOKED AT / CHECKED the output -- it is NOT an architecture decision, NOT a Gate approval, NOT "this direction is correct." Do not conflate a file living under `output/confirmed/` with any worklog's "Architecture 판단 없음" contract being satisfied, and do not describe something as "approved" in a worklog or report merely because its output folder was moved there.

**How to apply:**
- When looking for a checkpoint/export path that used to be directly under `output/<name>/`, also check `output/confirmed/<name>/` -- the user may have relocated it there after review.
- Never claim or imply architecture approval, Gate approval, or "the user approved this direction" based on an `output/confirmed/` location alone.
- If asked to move something into `output/confirmed/` myself, that action means "the user is done looking at this / it's checked off their review list," not "this design is accepted."
- Keep referring to worklogs' actual "결론 없음" / no-architecture-decision language as the source of truth for whether a direction has been decided, independent of this filesystem convention.
